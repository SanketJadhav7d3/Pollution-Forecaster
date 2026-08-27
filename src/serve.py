"""FastAPI inference service for the next-day AQI forecaster (specs.md §9.5).

    uv run uvicorn src.serve:app --reload --port 8000

Endpoints:
    POST /predict          stateless - caller supplies the history
    GET  /predict/{city}   convenience - fetches recent readings from OpenAQ
    GET  /health           liveness/readiness, reports the loaded model
    GET  /model            what is loaded and how it was selected

Every response carries the persistence prediction alongside the model's. On this
data persistence is more accurate overall; the model earns its place by calling
deteriorations persistence structurally cannot. Returning one number without the
other would hide that trade from whoever consumes this.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import UTC, date, datetime, timedelta
from functools import lru_cache
from pathlib import Path

import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

from src.aqi import BAD_AIR_RANK, CATEGORIES, RANK, pm25_to_category
from src.features import CORE_FEATURES, build_x
from src.features_live import MIN_HISTORY_DAYS, HistoryError, build_feature_row

log = logging.getLogger("serve")

# A baked artifact directory wins when present (that is how the container
# ships); otherwise fall back to the local registry for development.
_BAKED = Path(__file__).resolve().parent.parent / "model_artifact"
MODEL_URI = os.getenv("MODEL_URI") or (
    str(_BAKED) if _BAKED.exists() else "models:/aqi-next-day-forecaster/2"
)
KNOWN_CITIES = ("delhi", "mumbai")
# The city feature was ordinal-encoded from a sorted category list at training
# time; serving must reproduce that mapping exactly, not re-derive it from
# whatever cities happen to appear in one request.
CITY_CODES = {name: i for i, name in enumerate(sorted(KNOWN_CITIES))}

app = FastAPI(
    title="Delhi/Mumbai next-day AQI forecaster",
    version="1.0.0",
    description="Forecasts tomorrow's CPCB AQI category from recent PM2.5 readings.",
)


class Reading(BaseModel):
    date: date
    pm25: float = Field(ge=0, le=2000, description="24h mean PM2.5 in ug/m3")


class PredictRequest(BaseModel):
    city: str
    history: list[Reading] = Field(min_length=MIN_HISTORY_DAYS)

    @field_validator("city")
    @classmethod
    def known_city(cls, v: str) -> str:
        v = v.strip().lower()
        if v not in KNOWN_CITIES:
            raise ValueError(f"unknown city '{v}'; expected one of {', '.join(KNOWN_CITIES)}")
        return v


class PredictResponse(BaseModel):
    city: str
    forecast_date: date
    predicted_category: str
    probabilities: dict[str, float]
    persistence_category: str
    agrees_with_persistence: bool
    bad_air_warning: bool
    latest_pm25: float
    model_version: str


@lru_cache(maxsize=1)
def model_version() -> str:
    """The registered version this artifact came from.

    A baked directory has no version in its path, so the exporter records it
    alongside the model; reporting the directory name instead would tell a
    caller nothing about which model answered them.
    """
    info = Path(MODEL_URI) / "export_info.json"
    if info.exists():
        try:
            return str(json.loads(info.read_text())["model_version"])
        except (OSError, ValueError, KeyError):
            pass
    return MODEL_URI.rsplit("/", 1)[-1]


@lru_cache(maxsize=1)
def get_model():
    """Load once per process. Cached so requests do not re-read the artifact."""
    log.info("loading model from %s", MODEL_URI)
    return mlflow.pyfunc.load_model(MODEL_URI)


_SCHEMA_DTYPES = {
    "double": "float64",
    "float": "float32",
    "long": "int64",
    "integer": "int32",
    "boolean": "bool",
}


def _coerce_to_signature(frame: pd.DataFrame, model) -> pd.DataFrame:
    """Cast columns to the dtypes the model declared when it was logged.

    Reading the signature rather than hard-coding dtypes means a retrained model
    with a different schema still serves correctly instead of failing on a
    silent int64/int32 mismatch.
    """
    schema = model.metadata.get_input_schema()
    if schema is None:
        return frame
    out = frame.copy()
    for spec in schema.inputs:
        if spec.name not in out.columns:
            continue
        dtype = _SCHEMA_DTYPES.get(str(spec.type).split(".")[-1])
        if dtype and str(out[spec.name].dtype) != dtype:
            out[spec.name] = out[spec.name].astype(dtype)
    return out[[s.name for s in schema.inputs]]


def _predict_from_history(city: str, history: list[dict]) -> PredictResponse:
    try:
        row, forecast_date = build_feature_row(city, history)
    except HistoryError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    features = build_x(row, CORE_FEATURES)
    # build_x derives city_code from the request's own data, which would encode
    # a single-city request as 0 regardless of which city it is. Pin it.
    features["city_code"] = CITY_CODES[city]

    model = get_model()
    features = _coerce_to_signature(features, model)

    # The registered model carries its own class labels, so the category name
    # and the per-class probabilities arrive already correctly paired.
    result = model.predict(features)
    predicted = str(result["predicted_category"].iloc[0])
    probabilities = {
        col: round(float(result[col].iloc[0]), 4)
        for col in result.columns
        if col != "predicted_category"
    }
    # Present in CPCB severity order rather than the encoder's alphabetical one.
    probabilities = {c: probabilities[c] for c in CATEGORIES if c in probabilities}

    latest = float(row["pm25"].iloc[0])
    persistence = str(pm25_to_category(pd.Series([latest])).iloc[0])

    return PredictResponse(
        city=city,
        forecast_date=forecast_date,
        predicted_category=predicted,
        probabilities=probabilities,
        persistence_category=persistence,
        agrees_with_persistence=predicted == persistence,
        bad_air_warning=RANK.get(predicted, 0) >= BAD_AIR_RANK,
        latest_pm25=round(latest, 1),
        model_version=model_version(),
    )


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    """Forecast the day after the last reading supplied."""
    history = [{"date": r.date.isoformat(), "pm25": r.pm25} for r in req.history]
    return _predict_from_history(req.city, history)


@app.get("/predict/{city}", response_model=PredictResponse)
def predict_live(city: str, days: int = 10) -> PredictResponse:
    """Fetch recent readings from OpenAQ, then forecast.

    Convenience only. It adds a network dependency and an API key to the request
    path, so the stateless POST above stays the one used by scheduled jobs.
    """
    city = city.strip().lower()
    if city not in KNOWN_CITIES:
        raise HTTPException(
            status_code=422,
            detail=f"unknown city '{city}'; expected one of {', '.join(KNOWN_CITIES)}",
        )
    from src.live_history import LiveHistoryError, fetch_recent_history

    try:
        history = fetch_recent_history(city, days=days)
    except LiveHistoryError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return _predict_from_history(city, history)


@app.get("/health")
def health() -> dict:
    """Readiness probe: reports whether the model actually loaded."""
    try:
        get_model()
    except Exception as exc:  # noqa: BLE001 - the probe must report, not crash
        return {"status": "unhealthy", "model_uri": MODEL_URI, "error": str(exc)[:200]}
    return {
        "status": "ok",
        "model_uri": MODEL_URI,
        "model_version": model_version(),
        "cities": list(KNOWN_CITIES),
        "min_history_days": MIN_HISTORY_DAYS,
        "time": datetime.now(UTC).isoformat(),
    }


@app.get("/model")
def model_info() -> dict:
    """What is loaded, and the honest note on how it was chosen."""
    return {
        "model_uri": MODEL_URI,
        "features": [*CORE_FEATURES, "city_code"],
        "categories": CATEGORIES,
        "selection_basis": "deterioration recall at usable precision",
        "beats_persistence_on_accuracy": False,
        "note": (
            "Persistence ('tomorrow looks like today') is more accurate overall on this "
            "data. This model is the early-warning layer: it calls ~43% of deteriorations, "
            "which persistence never does. Both predictions are returned so callers can see "
            "when they disagree."
        ),
    }


def _default_window(days: int) -> tuple[date, date]:
    today = datetime.now(UTC).date()
    return today - timedelta(days=days), today
