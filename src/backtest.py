"""Expanding-window walk-forward backtesting (specs.md §9.4, §6).

    uv run python -m src.backtest

Answers "would this have worked, at each point in time, without seeing the
future" - as opposed to a single chronological split, which on two years of
data lands entirely inside one season and cannot separate a weak model from an
easy test window.

Each fold trains on everything before the fold start and tests on the next
quarter. Persistence is scored on the identical fold so the comparison is
like-for-like, and every fold is its own MLflow run (§6).
"""

from __future__ import annotations

import argparse
import io
import json
import logging

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from src.train import (
    DATA_PATH,
    TARGET,
    encode,
    load_dataset,
    log_artifacts,
    persistence_prediction,
    score,
)

log = logging.getLogger("backtest")

EXPERIMENT = "aqi-next-day-backtest"
MIN_TRAIN_MONTHS = 12  # no fold is scored until a full season sits behind it (§6)
FOLD_MONTHS = 3  # quarterly test windows; daily folds are too small to score


def make_folds(dates: pd.Series, min_train_months: int, fold_months: int):
    """Yield (train_end, test_end) boundaries for expanding-window folds."""
    start = dates.min()
    first_fold = start + pd.DateOffset(months=min_train_months)
    cursor = pd.Timestamp(first_fold).normalize()
    end = dates.max()
    while cursor < end:
        test_end = cursor + pd.DateOffset(months=fold_months)
        if (dates >= cursor).sum() == 0:
            break
        yield cursor, min(pd.Timestamp(test_end), end + pd.Timedelta(days=1))
        cursor = pd.Timestamp(test_end)


def new_model() -> RandomForestClassifier:
    # class_weight left at None: on this data it cost ~2pp accuracy by
    # upweighting severe classes that are absent from clean-season folds.
    return RandomForestClassifier(
        n_estimators=400, min_samples_leaf=2, random_state=42, n_jobs=-1
    )


def plot_folds(rows: list[dict]) -> bytes:
    df = pd.DataFrame(rows)
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    x = range(len(df))
    for ax, metric, title in (
        (axes[0], "accuracy", "Accuracy per fold"),
        (axes[1], "f1_macro", "Macro-F1 per fold"),
    ):
        ax.plot(x, df[f"model_{metric}"], "o-", label="RandomForest")
        ax.plot(x, df[f"persistence_{metric}"], "s--", label="Persistence")
        ax.set_title(title)
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)
        ax.legend()
    axes[1].set_xticks(list(x))
    axes[1].set_xticklabels(df["fold"], rotation=30, ha="right")
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120)
    plt.close(fig)
    return buf.getvalue()


def run(data_path=DATA_PATH) -> list[dict]:
    df, snapshot = load_dataset(data_path)
    df = df.sort_values("date")
    mlflow.set_experiment(EXPERIMENT)
    rows: list[dict] = []

    with mlflow.start_run(run_name="backtest-parent"):
        mlflow.log_params({**snapshot, "min_train_months": MIN_TRAIN_MONTHS,
                           "fold_months": FOLD_MONTHS, "model": "RandomForestClassifier"})

        for train_end, test_end in make_folds(df["date"], MIN_TRAIN_MONTHS, FOLD_MONTHS):
            train = df[df["date"] < train_end]
            test = df[(df["date"] >= train_end) & (df["date"] < test_end)]
            if test.empty or train.empty:
                continue
            label = f"{train_end.date()}..{test_end.date()}"

            with mlflow.start_run(run_name=f"fold-{train_end.date()}", nested=True):
                clf = new_model()
                clf.fit(encode(train), train[TARGET].astype(str))
                y_true = test[TARGET].astype(str)
                y_model = clf.predict(encode(test))
                y_persist = persistence_prediction(test)

                m = score(y_true, y_model, test["city"])
                b = score(y_true, y_persist, test["city"])
                mlflow.log_params({"train_end": str(train_end.date()),
                                   "test_end": str(test_end.date()),
                                   "train_rows": len(train), "test_rows": len(test)})
                mlflow.log_metrics({**{f"model_{k}": v for k, v in m.items()},
                                    **{f"persistence_{k}": v for k, v in b.items()}})
                # Class mix is logged so a bad fold can be read against whether
                # that quarter contained any severe days at all.
                mlflow.log_dict(y_true.value_counts().to_dict(), "test_class_counts.json")
                log_artifacts(y_true, y_model, "model")

                rows.append({
                    "fold": label,
                    "train_rows": len(train), "test_rows": len(test),
                    "model_accuracy": m["accuracy"], "model_f1_macro": m["f1_macro"],
                    "persistence_accuracy": b["accuracy"],
                    "persistence_f1_macro": b["f1_macro"],
                })
                log.info("%s model_acc=%.3f persist_acc=%.3f (n=%d)",
                         label, m["accuracy"], b["accuracy"], len(test))

        if rows:
            summary = pd.DataFrame(rows)
            mlflow.log_dict(summary.to_dict("records"), "fold_summary.json")
            mlflow.log_image(
                plt.imread(io.BytesIO(plot_folds(rows)), format="png"), "folds_over_time.png"
            )
            mlflow.log_metrics({
                "mean_model_accuracy": summary["model_accuracy"].mean(),
                "mean_persistence_accuracy": summary["persistence_accuracy"].mean(),
                "folds_model_wins": int((summary["model_accuracy"] > summary["persistence_accuracy"]).sum()),
                "n_folds": len(summary),
            })
    return rows


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Walk-forward backtest")
    p.add_argument("--data", default=str(DATA_PATH))
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
    rows = run(args.data)
    print(json.dumps(rows, indent=2, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
