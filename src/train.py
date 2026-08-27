"""Baseline training with MLflow tracking (specs.md §9.3).

    uv run python -m src.train

Trains three things on one identical chronological split:
  * persistence baseline ("tomorrow looks like today") - the bar to clear
  * a pooled model with `city` as a feature (§5)
  * one model per city

The pooled-vs-per-city comparison is deliberately empirical: §5 argues for
pooling, but a city doing systematically worse is a finding to measure, not an
assumption to design around.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path

import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from src.aqi import CATEGORIES, pm25_to_category

log = logging.getLogger("train")

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "processed" / "city_daily.parquet"
EXPERIMENT = "aqi-next-day-baseline"
TEST_FRACTION = 0.2

from src.features import (
    CORE_FEATURES,
    FEATURES,
    TARGET,
    build_x,
)


def load_dataset(path: Path | str) -> tuple[pd.DataFrame, dict]:
    """Load and drop rows without a target or a full feature warm-up."""
    path = Path(path)
    df = pd.read_parquet(path)
    # Only the core is required; weather NaNs are handled by the model.
    present = [c for c in CORE_FEATURES if c in df.columns]
    usable = df.dropna(subset=[TARGET, *present]).copy()
    snapshot = {
        "data_snapshot": str(path),
        "data_rows": len(usable),
        # Hash the exact bytes trained on, so a run stays reproducible (§8).
        "data_sha256": hashlib.sha256(path.read_bytes()).hexdigest()[:16],
    }
    return usable, snapshot


def chronological_split(df: pd.DataFrame, test_fraction: float = TEST_FRACTION):
    """Split on a single global date cutoff - never shuffled (§4).

    One shared cutoff (rather than per-city quantiles) keeps the test period
    identical across cities, so per-city scores stay comparable and no city's
    future can leak in via another's past.
    """
    dates = df["date"].sort_values().unique()
    cutoff = dates[int(len(dates) * (1 - test_fraction))]
    train = df[df["date"] < cutoff]
    test = df[df["date"] >= cutoff]
    return train, test, pd.Timestamp(cutoff)


def encode(df: pd.DataFrame) -> pd.DataFrame:
    """Trees handle ordinal codes fine; no one-hot needed (§5)."""
    return build_x(df, FEATURES)


def score(y_true, y_pred, cities: pd.Series | None = None) -> dict:
    m = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }
    if cities is not None:
        for city in sorted(cities.unique()):
            mask = (cities == city).to_numpy()
            m[f"accuracy_{city}"] = accuracy_score(y_true[mask], y_pred[mask])
            m[f"f1_macro_{city}"] = f1_score(
                y_true[mask], y_pred[mask], average="macro", zero_division=0
            )
    return m


def persistence_prediction(df: pd.DataFrame) -> pd.Series:
    """Baseline: tomorrow's category equals today's.

    Uses `pm25` (day D) - the same information set the models get, so the
    comparison is like-for-like.
    """
    return pm25_to_category(df["pm25"]).astype(str)


def log_artifacts(y_true, y_pred, tag: str) -> None:
    report = classification_report(y_true, y_pred, zero_division=0, output_dict=True)
    mlflow.log_dict(report, f"classification_report_{tag}.json")
    labels = [c for c in CATEGORIES if c in set(y_true) | set(y_pred)]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    mlflow.log_dict(
        {"labels": labels, "matrix": cm.tolist()}, f"confusion_matrix_{tag}.json"
    )


def run(data_path: Path = DATA_PATH) -> dict:
    df, snapshot = load_dataset(data_path)
    train, test, cutoff = chronological_split(df)
    log.info(
        "split at %s -> train=%d test=%d", cutoff.date(), len(train), len(test)
    )
    mlflow.set_experiment(EXPERIMENT)
    results: dict[str, dict] = {}

    common = {**snapshot, "split_cutoff": str(cutoff.date()),
              "train_rows": len(train), "test_rows": len(test)}

    # --- baseline -------------------------------------------------------
    with mlflow.start_run(run_name="persistence-baseline"):
        mlflow.log_params({**common, "model": "persistence"})
        y_true = test[TARGET].astype(str)
        y_pred = persistence_prediction(test)
        m = score(y_true, y_pred, test["city"])
        mlflow.log_metrics(m)
        log_artifacts(y_true, y_pred, "persistence")
        results["persistence"] = m

    # --- pooled ---------------------------------------------------------
    with mlflow.start_run(run_name="pooled-rf"):
        mlflow.log_params({**common, "model": "RandomForestClassifier", "scope": "pooled"})
        clf = RandomForestClassifier(
            n_estimators=400, min_samples_leaf=2, class_weight="balanced",
            random_state=42, n_jobs=-1,
        )
        clf.fit(encode(train), train[TARGET].astype(str))
        y_true = test[TARGET].astype(str)
        y_pred = clf.predict(encode(test))
        m = score(y_true, y_pred, test["city"])
        mlflow.log_metrics(m)
        log_artifacts(y_true, y_pred, "pooled")
        imp = dict(zip(encode(train).columns, clf.feature_importances_.round(4).tolist(), strict=True))
        mlflow.log_dict(imp, "feature_importances.json")
        mlflow.sklearn.log_model(clf, name="model")
        results["pooled"] = m

    # --- per city -------------------------------------------------------
    per_city: dict[str, float] = {}
    for city in sorted(df["city"].unique()):
        tr, te = train[train.city == city], test[test.city == city]
        with mlflow.start_run(run_name=f"percity-rf-{city}"):
            mlflow.log_params({**common, "model": "RandomForestClassifier",
                               "scope": f"per-city:{city}", "train_rows": len(tr)})
            clf = RandomForestClassifier(
                n_estimators=400, min_samples_leaf=2, class_weight="balanced",
                random_state=42, n_jobs=-1,
            )
            clf.fit(tr[FEATURES], tr[TARGET].astype(str))
            y_true = te[TARGET].astype(str)
            y_pred = clf.predict(te[FEATURES])
            m = score(y_true, y_pred)
            mlflow.log_metrics(m)
            log_artifacts(y_true, y_pred, f"percity_{city}")
            per_city[f"accuracy_{city}"] = m["accuracy"]
            per_city[f"f1_macro_{city}"] = m["f1_macro"]
    results["per_city"] = per_city
    return results


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Train baseline AQI forecaster")
    p.add_argument("--data", default=str(DATA_PATH))
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
    res = run(Path(args.data))
    print(json.dumps(res, indent=2, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
