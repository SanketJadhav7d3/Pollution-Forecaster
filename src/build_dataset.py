"""Build the city-day training table from raw OpenAQ files (specs.md §9.2).

    uv run python -m src.build_dataset

Pipeline: per-sensor JSON -> pooled city-day -> continuous date index ->
lag/calendar features -> next-day target.

Two invariants this module exists to protect:
  1. Lags are computed on a *continuous date index*, never by row position, so
     a missing day can never masquerade as "yesterday".
  2. Every feature is drawn from day D or earlier while the target is day D+1
     (specs.md §4). Any leak here produces a great-looking, useless model.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.aqi import pm25_to_category

log = logging.getLogger("build_dataset")

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw" / "openaq"
OUT_DIR = ROOT / "data" / "processed"

STUBBLE_BURNING_MONTHS = (10, 11)  # Oct-Nov paddy burning drives Delhi's worst air


def load_sensor_rows() -> pd.DataFrame:
    """All sensor-day rows, with `city` from the directory layout (§2).

    `parameter` is carried explicitly: the raw directory now holds meteorology
    alongside pm25, and reading `value` without it would silently average
    temperature into the pollutant series.
    """
    records = []
    for city_dir in sorted(p for p in RAW_DIR.iterdir() if p.is_dir()):
        city = city_dir.name
        for path in sorted(city_dir.glob("sensor_*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            parameter = payload.get("parameter")
            for row in payload["rows"]:
                if row.get("value") is None:
                    continue
                records.append(
                    {
                        "city": city,
                        "sensor_id": payload["sensor_id"],
                        "parameter": parameter,
                        "date": row["date"],
                        "value": row["value"],
                    }
                )
    if not records:
        raise RuntimeError(f"no rows found under {RAW_DIR} - run the backfill first")
    df = pd.DataFrame.from_records(records)
    df["date"] = pd.to_datetime(df["date"])
    return df


# Wind direction is circular: averaging 350 and 10 degrees naively gives 180,
# the exact opposite of the truth. It is resolved into unit vectors instead and
# the components are averaged, which is also the form a tree can actually use.
CIRCULAR = "wind_direction"


def pool_to_city_day(sensor_rows: pd.DataFrame) -> pd.DataFrame:
    """Average sensors within a city-day, one column per parameter.

    Individual sensors are retired and re-registered under new ids, so no single
    sensor spans the window; the city-day mean is the stable unit. `n_sensors`
    counts pm25 sensors specifically, because that count runs 2 -> 60 over the
    window and a level shift caused by sampling must stay distinguishable from
    a real trend.
    """
    linear = sensor_rows[sensor_rows["parameter"] != CIRCULAR]
    wide = (
        linear.pivot_table(
            index=["city", "date"], columns="parameter", values="value", aggfunc="mean"
        )
        .rename(columns={"relativehumidity": "humidity"})
        .reset_index()
    )
    wide.columns.name = None

    circ = sensor_rows[sensor_rows["parameter"] == CIRCULAR]
    if not circ.empty:
        rad = np.deg2rad(circ["value"])
        comp = circ.assign(_sin=np.sin(rad), _cos=np.cos(rad))
        agg = comp.groupby(["city", "date"])[["_sin", "_cos"]].mean().reset_index()
        agg = agg.rename(columns={"_sin": "wind_dir_sin", "_cos": "wind_dir_cos"})
        wide = wide.merge(agg, on=["city", "date"], how="left")

    counts = (
        sensor_rows[sensor_rows["parameter"] == "pm25"]
        .groupby(["city", "date"])["sensor_id"]
        .nunique()
        .rename("n_sensors")
        .reset_index()
    )
    return wide.merge(counts, on=["city", "date"], how="left")


def reindex_continuous(city_day: pd.DataFrame) -> pd.DataFrame:
    """Give each city a gap-free daily index; missing days become explicit NaN."""
    frames = []
    for city, grp in city_day.groupby("city"):
        full = pd.date_range(grp["date"].min(), grp["date"].max(), freq="D")
        g = grp.set_index("date").reindex(full)
        g["city"] = city
        g["n_sensors"] = g["n_sensors"].fillna(0).astype(int)
        g.index.name = "date"
        frames.append(g.reset_index())
    return pd.concat(frames, ignore_index=True)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Lag, rolling, trend and calendar features (§3).

    The information set for a row dated D is "everything through D": forecasting
    D+1 on the evening of D, today's completed 24h mean is available and is by
    far the strongest predictor. The genuine leak would be using D+1's reading,
    which lives only in the target (see add_target).
    """
    out = []
    for city, grp in df.groupby("city"):
        g = grp.sort_values("date").copy()
        pm = g["pm25"]

        for lag in (1, 2, 3):
            g[f"pm25_lag_{lag}"] = pm.shift(lag)

        # Windows span D..D-6 inclusive - today included, tomorrow never.
        g["pm25_roll_3"] = pm.rolling(3, min_periods=2).mean()
        g["pm25_roll_7"] = pm.rolling(7, min_periods=4).mean()
        g["pm25_roll_7_std"] = pm.rolling(7, min_periods=4).std()
        g["pm25_delta_1"] = pm - pm.shift(1)
        g["pm25_delta_3"] = pm - pm.shift(3)
        g["pm25_vs_roll_7"] = pm - g["pm25_roll_7"]

        # Meteorology as of today - available when forecasting tomorrow. No NWP
        # forecast is used, so these describe conditions now, not conditions
        # during the day being predicted; expect a modest gain, not a step change.
        for col in ("temperature", "humidity", "wind_speed"):
            if col in g:
                g[f"{col}_lag_1"] = g[col].shift(1)
                g[f"{col}_roll_3"] = g[col].rolling(3, min_periods=2).mean()
                g[f"{col}_delta_1"] = g[col] - g[col].shift(1)
        if "wind_speed" in g:
            # Low wind after a calm spell is when pollutants accumulate.
            g["wind_speed_roll_3_min"] = g["wind_speed"].rolling(3, min_periods=2).min()

        d = g["date"].dt
        g["month"] = d.month
        g["day_of_week"] = d.dayofweek
        g["day_of_year"] = d.dayofyear
        g["is_stubble_season"] = d.month.isin(STUBBLE_BURNING_MONTHS).astype(int)
        g["city"] = city
        out.append(g)
    return pd.concat(out, ignore_index=True)


def add_target(df: pd.DataFrame) -> pd.DataFrame:
    """Target = tomorrow's AQI category (§4: next-day forecast, t+1)."""
    out = []
    for _, grp in df.groupby("city"):
        g = grp.sort_values("date").copy()
        g["pm25_next_day"] = g["pm25"].shift(-1)
        g["target_aqi_category"] = pm25_to_category(g["pm25_next_day"])
        out.append(g)
    return pd.concat(out, ignore_index=True)


def build() -> pd.DataFrame:
    sensor_rows = load_sensor_rows()
    log.info("loaded %d sensor-day rows", len(sensor_rows))
    city_day = pool_to_city_day(sensor_rows)
    city_day = reindex_continuous(city_day)
    log.info("pooled to %d city-day rows", len(city_day))
    df = add_features(city_day)
    df = add_target(df)
    return df.sort_values(["city", "date"]).reset_index(drop=True)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Build the city-day training table")
    p.add_argument("--out", default=str(OUT_DIR / "city_daily.parquet"))
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")

    df = build()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)
    log.info("wrote %s (%d rows, %d cols)", args.out, len(df), df.shape[1])

    # Parquet is what training reads (types survive the round-trip); the CSV is
    # written alongside it purely for eyeballing in a spreadsheet.
    csv_path = Path(args.out).with_suffix(".csv")
    readable = df.copy()
    readable["date"] = readable["date"].dt.strftime("%Y-%m-%d")
    for col in readable.select_dtypes("float").columns:
        readable[col] = readable[col].round(2)
    readable.to_csv(csv_path, index=False)
    log.info("wrote %s", csv_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
