"""Model/feature comparison grid, walk-forward, every run logged (specs.md §6, §8).

    uv run python -m src.experiments

Exists because the alternative - ad-hoc scripts printing numbers to a terminal -
produced conclusions that could not be reproduced, compared, or traced back to
the dataset they were computed on. Every configuration goes through one code
path, so logging cannot be skipped and every run records the exact data it saw.
"""

from __future__ import annotations

import argparse
import logging
import warnings

import mlflow
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

from src.aqi import pm25_to_category
from src.backtest import FOLD_MONTHS, MIN_TRAIN_MONTHS, make_folds
from src.metrics import score_all, summarise_folds
from src.train import CORE_FEATURES, DATA_PATH, TARGET, WEATHER_FEATURES, load_dataset

warnings.filterwarnings("ignore")
log = logging.getLogger("experiments")

EXPERIMENT = "aqi-model-comparison"

FEATURE_SETS = {
    "core": CORE_FEATURES,
    "core_weather": CORE_FEATURES + WEATHER_FEATURES,
}


def model_registry() -> dict:
    """Fresh instances per fold - a fitted model must never leak across folds."""
    import lightgbm as lgb
    import xgboost as xgb

    return {
        "random_forest": lambda: RandomForestClassifier(
            n_estimators=400, min_samples_leaf=2, class_weight="balanced",
            random_state=42, n_jobs=-1,
        ),
        "hist_gradient_boost": lambda: HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.05, class_weight="balanced", random_state=42,
        ),
        "lightgbm": lambda: lgb.LGBMClassifier(
            n_estimators=400, learning_rate=0.05, num_leaves=15,
            class_weight="balanced", random_state=42, verbose=-1,
        ),
        "xgboost": lambda: xgb.XGBClassifier(
            n_estimators=400, learning_rate=0.05, max_depth=4,
            random_state=42, verbosity=0,
        ),
    }


def build_x(frame: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    cols = [c for c in features if c in frame.columns]
    return frame[cols].assign(city_code=frame["city"].astype("category").cat.codes)


def fit_predict(name: str, factory, train: pd.DataFrame, test: pd.DataFrame,
                features: list[str]) -> pd.Series:
    model = factory()
    y_train = train[TARGET].astype(str)
    x_train, x_test = build_x(train, features), build_x(test, features)

    if name == "xgboost":
        # xgboost needs integer classes; map back so all models return labels.
        le = LabelEncoder().fit(pd.concat([y_train, test[TARGET].astype(str)]))
        model.fit(x_train, le.transform(y_train))
        return pd.Series(le.inverse_transform(model.predict(x_test)), index=test.index)
    model.fit(x_train, y_train)
    return pd.Series(model.predict(x_test), index=test.index)


def run_config(name: str, features_name: str, predict_fn, df: pd.DataFrame,
               snapshot: dict) -> dict:
    """Walk-forward one configuration, logging a parent run plus a run per fold."""
    fold_rows: list[dict] = []

    with mlflow.start_run(run_name=f"{name}__{features_name}"):
        mlflow.log_params({
            **snapshot, "model": name, "feature_set": features_name,
            "min_train_months": MIN_TRAIN_MONTHS, "fold_months": FOLD_MONTHS,
            "n_features": len(FEATURE_SETS.get(features_name, [])),
        })

        for train_end, test_end in make_folds(df["date"], MIN_TRAIN_MONTHS, FOLD_MONTHS):
            train = df[df["date"] < train_end]
            test = df[(df["date"] >= train_end) & (df["date"] < test_end)]
            if train.empty or test.empty:
                continue

            y_pred = predict_fn(train, test)
            m = score_all(
                test[TARGET].astype(str), y_pred, test["city"], test["today_category"]
            )
            with mlflow.start_run(run_name=f"fold-{train_end.date()}", nested=True):
                mlflow.log_params({"train_end": str(train_end.date()),
                                   "test_end": str(test_end.date()),
                                   "train_rows": len(train), "test_rows": len(test),
                                   "model": name, "feature_set": features_name})
                mlflow.log_metrics(m)
            fold_rows.append({"fold": str(train_end.date()), **m})

        keys = sorted({k for r in fold_rows for k in r if k != "fold"})
        summary = summarise_folds(fold_rows, keys)
        mlflow.log_metrics(summary)
        mlflow.log_dict(fold_rows, "fold_metrics.json")
        return {"model": name, "features": features_name, **summary}


def run(data_path=DATA_PATH) -> pd.DataFrame:
    df, snapshot = load_dataset(data_path)
    df = df.sort_values("date")
    df = df[df["pm25_next_day"].notna()].copy()
    # Needed to separate stable days from real transitions.
    df["today_category"] = pm25_to_category(df["pm25"]).astype(str)

    mlflow.set_experiment(EXPERIMENT)
    results = []

    # Baseline first, so every comparison has the bar in the same experiment.
    results.append(run_config(
        "persistence", "none",
        lambda tr, te: pd.Series(te["today_category"].to_numpy(), index=te.index),
        df, snapshot,
    ))

    for features_name, features in FEATURE_SETS.items():
        for model_name, factory in model_registry().items():
            log.info("running %s / %s", model_name, features_name)
            results.append(run_config(
                model_name, features_name,
                lambda tr, te, n=model_name, f=factory, ft=features: fit_predict(n, f, tr, te, ft),
                df, snapshot,
            ))
    return pd.DataFrame(results)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Model/feature comparison grid")
    p.add_argument("--data", default=str(DATA_PATH))
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")

    res = run(args.data)
    cols = ["model", "features", "mean_skill_score", "mean_recall_deterioration",
            "mean_accuracy_change_days", "mean_bad_air_recall",
            "mean_bad_air_precision", "mean_accuracy"]
    out = res[[c for c in cols if c in res.columns]].sort_values(
        "mean_skill_score", ascending=False
    )
    print(out.round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
