"""Persist per-day walk-forward predictions for visualisation.

    uv run python -m src.backtest_predictions

backtest.py logs aggregate metrics per fold, which is enough to rank
configurations but not to plot a timeline. This re-runs the same folds with the
registered configuration and writes every individual prediction, so a chart can
show *where* the model agreed with, beat, or lost to persistence.

Each row's prediction comes from a model trained only on data before that fold -
the same out-of-sample guarantee the backtest makes.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import LabelEncoder

from src.aqi import CATEGORIES, pm25_to_category
from src.backtest import FOLD_MONTHS, MIN_TRAIN_MONTHS, make_folds
from src.experiments import build_x, model_registry
from src.metrics import RANK, score_all
from src.register_model import CHOSEN_MODEL
from src.train import CORE_FEATURES, DATA_PATH, TARGET, load_dataset

log = logging.getLogger("backtest_predictions")
OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"


def run(data_path=DATA_PATH) -> pd.DataFrame:
    df, snapshot = load_dataset(data_path)
    df = df[df["pm25_next_day"].notna()].sort_values("date").copy()
    df["today_category"] = pm25_to_category(df["pm25"]).astype(str)

    factory = model_registry()[CHOSEN_MODEL]
    encoder = LabelEncoder().fit(df[TARGET].astype(str))
    frames = []

    for train_end, test_end in make_folds(df["date"], MIN_TRAIN_MONTHS, FOLD_MONTHS):
        train = df[df["date"] < train_end]
        test = df[(df["date"] >= train_end) & (df["date"] < test_end)]
        if train.empty or test.empty:
            continue
        model = factory()
        model.fit(build_x(train, CORE_FEATURES), encoder.transform(train[TARGET].astype(str)))
        preds = encoder.inverse_transform(model.predict(build_x(test, CORE_FEATURES)))

        out = test[["date", "city", "pm25", "pm25_next_day", "n_sensors"]].copy()
        out["fold"] = str(train_end.date())
        out["actual"] = test[TARGET].astype(str).to_numpy()
        out["predicted"] = preds
        out["persistence"] = test["today_category"].to_numpy()
        frames.append(out)

    res = pd.concat(frames, ignore_index=True)
    res["model_correct"] = res["predicted"] == res["actual"]
    res["persistence_correct"] = res["persistence"] == res["actual"]
    res["changed"] = res["actual"] != res["persistence"]
    res["deteriorated"] = res["actual"].map(RANK) > res["persistence"].map(RANK)
    res["model_caught_deterioration"] = res["deteriorated"] & (
        res["predicted"].map(RANK) > res["persistence"].map(RANK)
    )
    log.info("%d predictions across %d folds", len(res), res["fold"].nunique())
    return res, snapshot


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Persist per-day backtest predictions")
    p.add_argument("--data", default=str(DATA_PATH))
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")

    res, snapshot = run(args.data)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    res.to_parquet(OUT_DIR / "backtest_predictions.parquet", index=False)

    summary = {"snapshot": snapshot, "categories": CATEGORIES, "folds": {}}
    for fold, grp in res.groupby("fold"):
        summary["folds"][fold] = score_all(
            grp["actual"], grp["predicted"], grp["city"], grp["persistence"]
        )
    (OUT_DIR / "backtest_summary.json").write_text(json.dumps(summary, indent=2, default=float))
    log.info("wrote backtest_predictions.parquet and backtest_summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
