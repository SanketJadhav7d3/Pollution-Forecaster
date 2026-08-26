"""One scoring function for every experiment.

Six-way accuracy is a misleading headline for this problem: roughly 69% of days
carry the same category as the day before, so a rule that copies today scores
~73% while being structurally unable to predict a single change. These metrics
therefore report the operational question - "will tomorrow be bad" - alongside
the aggregate, and split performance by whether the category actually moved.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from src.aqi import CATEGORIES

RANK = {c: i for i, c in enumerate(CATEGORIES)}

# "Poor" or worse: the point at which health advice changes for the general
# public. "Very Poor" or worse is the emergency band.
BAD_AIR_RANK = RANK["Poor"]
SEVERE_RANK = RANK["Very Poor"]


def _warning_scores(y_true: pd.Series, y_pred: pd.Series, threshold: int, tag: str) -> dict:
    """Recall/precision/F1 for the binary "is tomorrow at or past this band" call."""
    actual = y_true.map(RANK) >= threshold
    predicted = y_pred.map(RANK) >= threshold
    tp = int((actual & predicted).sum())
    fn = int((actual & ~predicted).sum())
    fp = int((~actual & predicted).sum())

    out = {f"{tag}_n_actual": int(actual.sum())}
    if tp + fn:
        out[f"{tag}_recall"] = tp / (tp + fn)
    if tp + fp:
        out[f"{tag}_precision"] = tp / (tp + fp)
    if out.get(f"{tag}_recall") and out.get(f"{tag}_precision"):
        r, p = out[f"{tag}_recall"], out[f"{tag}_precision"]
        out[f"{tag}_f1"] = 2 * r * p / (r + p) if (r + p) else 0.0
    return out


def score_all(
    y_true: pd.Series,
    y_pred: pd.Series,
    cities: pd.Series | None = None,
    today_category: pd.Series | None = None,
) -> dict:
    """Every metric an experiment reports. Keys are MLflow-safe."""
    y_true = pd.Series(y_true).astype(str)
    y_pred = pd.Series(y_pred, index=y_true.index).astype(str)

    m: dict[str, float] = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }

    m.update(_warning_scores(y_true, y_pred, BAD_AIR_RANK, "bad_air"))
    m.update(_warning_scores(y_true, y_pred, SEVERE_RANK, "severe"))

    # Per-category recall: an aggregate hides a class the model never predicts.
    for cat in CATEGORIES:
        mask = y_true == cat
        if mask.any():
            m[f"recall_{cat.replace(' ', '_').lower()}"] = float((y_pred[mask] == cat).mean())

    if cities is not None:
        cities = pd.Series(cities, index=y_true.index)
        for city in sorted(cities.unique()):
            k = cities == city
            m[f"accuracy_{city}"] = accuracy_score(y_true[k], y_pred[k])
            m[f"f1_macro_{city}"] = f1_score(
                y_true[k], y_pred[k], average="macro", zero_division=0
            )
            sub = _warning_scores(y_true[k], y_pred[k], BAD_AIR_RANK, f"bad_air_{city}")
            m.update(sub)

    if today_category is not None:
        # Skill relative to persistence (the standard reference forecast in
        # meteorology). Raw accuracy is dominated by the ~69% of days that do
        # not change, where persistence is right for free; skill asks instead
        # what fraction of persistence's errors the model actually fixed.
        #   >0 beats doing nothing, 0 ties it, <0 is worse than doing nothing.
        # The decisive split: persistence scores 1.0 on stable days by
        # construction and 0.0 on changes. A model earns its place here.
        today = pd.Series(today_category, index=y_true.index).astype(str)
        changed = today != y_true
        m["frac_days_changed"] = float(changed.mean())

        acc_persistence = accuracy_score(y_true, today)
        m["accuracy_persistence_ref"] = acc_persistence
        if acc_persistence < 1.0:
            m["skill_score"] = (m["accuracy"] - acc_persistence) / (1.0 - acc_persistence)
        if (~changed).any():
            m["accuracy_stable_days"] = accuracy_score(y_true[~changed], y_pred[~changed])
        if changed.any():
            m["accuracy_change_days"] = accuracy_score(y_true[changed], y_pred[changed])
        # Did it catch deteriorations specifically (air getting worse)?
        worse = y_true.map(RANK) > today.map(RANK)
        if worse.any():
            caught = (y_pred[worse].map(RANK) > today[worse].map(RANK)).mean()
            m["recall_deterioration"] = float(caught)
            m["n_deteriorations"] = int(worse.sum())
    return m


def summarise_folds(rows: list[dict], keys: list[str]) -> dict:
    """Mean of each metric across folds, skipping folds where it was undefined."""
    df = pd.DataFrame(rows)
    out = {}
    for k in keys:
        if k in df.columns:
            vals = df[k].dropna()
            if len(vals):
                out[f"mean_{k}"] = float(np.mean(vals))
    return out
