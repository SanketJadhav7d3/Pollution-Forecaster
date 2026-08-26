"""Train the chosen configuration on all history and register it (specs.md §9.3).

    uv run python -m src.register_model

Registers xgboost on the core (pm25 + calendar) feature set.

Why this one, given it does NOT beat persistence on accuracy: on this data no
model does. Copying today's category scores ~0.73 because ~69% of days do not
change, and that rule is structurally incapable of predicting the ones that do
(0% recall on changes, every fold). The registered model trades ~7 points of
accuracy for the ability to warn at all - it catches ~44% of deteriorations at
~0.76 precision. See the aqi-model-comparison experiment for the full grid.

Anyone deploying this should treat persistence as the accuracy benchmark and
this model as the early-warning layer, not as a strict improvement.
"""

from __future__ import annotations

import argparse
import logging

import mlflow
import pandas as pd
from mlflow.models import infer_signature
from sklearn.preprocessing import LabelEncoder

from src.aqi import pm25_to_category
from src.experiments import EXPERIMENT, build_x, model_registry
from src.metrics import score_all
from src.train import CORE_FEATURES, DATA_PATH, TARGET, chronological_split, load_dataset

log = logging.getLogger("register_model")

MODEL_NAME = "aqi-next-day-forecaster"
CHOSEN_MODEL = "xgboost"
CHOSEN_FEATURES = "core"
HOLDOUT_FRACTION = 0.2


def run(data_path=DATA_PATH, register: bool = True) -> str:
    df, snapshot = load_dataset(data_path)
    df = df[df["pm25_next_day"].notna()].sort_values("date").copy()
    df["today_category"] = pm25_to_category(df["pm25"]).astype(str)

    # Score on a held-out tail so the registered version carries honest numbers,
    # then refit on everything - the deployed model should see all history.
    train, test, cutoff = chronological_split(df, HOLDOUT_FRACTION)
    factory = model_registry()[CHOSEN_MODEL]

    y_train = train[TARGET].astype(str)
    encoder = LabelEncoder().fit(df[TARGET].astype(str))
    holdout = factory()
    holdout.fit(build_x(train, CORE_FEATURES), encoder.transform(y_train))
    preds = encoder.inverse_transform(holdout.predict(build_x(test, CORE_FEATURES)))
    metrics = score_all(
        test[TARGET].astype(str), pd.Series(preds, index=test.index),
        test["city"], test["today_category"],
    )

    x_all = build_x(df, CORE_FEATURES)
    final = factory()
    final.fit(x_all, encoder.transform(df[TARGET].astype(str)))

    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name=f"registered-{CHOSEN_MODEL}-{CHOSEN_FEATURES}") as run_ctx:
        mlflow.log_params({
            **snapshot,
            "model": CHOSEN_MODEL,
            "feature_set": CHOSEN_FEATURES,
            "n_features": x_all.shape[1],
            "holdout_cutoff": str(cutoff.date()),
            "trained_rows": len(df),
            "date_min": str(df["date"].min().date()),
            "date_max": str(df["date"].max().date()),
            "selection_basis": "deterioration recall at usable precision",
            "beats_persistence_on_accuracy": False,
        })
        mlflow.log_metrics(metrics)
        mlflow.log_dict(
            {"classes": encoder.classes_.tolist(), "features": list(x_all.columns)},
            "model_contract.json",
        )
        signature = infer_signature(x_all, encoder.inverse_transform(final.predict(x_all)))
        # The native xgboost flavor, not mlflow.sklearn: the sklearn flavor
        # serialises via skops, which refuses xgboost's Booster as an untrusted
        # type. This also keeps the artifact loadable without sklearn present.
        info = mlflow.xgboost.log_model(
            final, name="model", signature=signature,
            input_example=x_all.head(3),
            registered_model_name=MODEL_NAME if register else None,
        )
        log.info("logged model: %s", info.model_uri)
        log.info("holdout accuracy=%.3f deterioration_recall=%.3f bad_air_precision=%.3f",
                 metrics.get("accuracy", float("nan")),
                 metrics.get("recall_deterioration", float("nan")),
                 metrics.get("bad_air_precision", float("nan")))
        return run_ctx.info.run_id


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Register the chosen model")
    p.add_argument("--data", default=str(DATA_PATH))
    p.add_argument("--no-register", action="store_true")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
    run(args.data, register=not args.no_register)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
