"""Turn a short run of daily readings into one model-ready feature row.

The functions here deliberately delegate to build_dataset rather than
re-implementing the feature maths. Training/serving skew is the classic way a
served model quietly degrades: if `pm25_roll_7` is computed even slightly
differently here, the model receives inputs unlike anything it trained on and
gets worse with no error anywhere. Reusing the training code makes divergence
impossible rather than merely unlikely.
"""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

from src.build_dataset import add_features, reindex_continuous

# roll_7 needs min_periods=4 and lag_3 needs three prior days, so a shorter run
# yields NaN features and a prediction the model cannot honestly make.
MIN_HISTORY_DAYS = 7


class HistoryError(ValueError):
    """The supplied history cannot produce a valid feature row."""


def _to_frame(city: str, history: list[dict]) -> pd.DataFrame:
    rows = []
    for item in history:
        rows.append(
            {
                "city": city,
                "date": pd.to_datetime(item["date"]),
                "pm25": float(item["pm25"]),
                "n_sensors": int(item.get("n_sensors") or 1),
            }
        )
    frame = pd.DataFrame(rows)
    if frame["date"].duplicated().any():
        dupes = sorted(frame.loc[frame["date"].duplicated(), "date"].dt.strftime("%Y-%m-%d"))
        raise HistoryError(f"duplicate dates in history: {', '.join(dupes)}")
    return frame.sort_values("date").reset_index(drop=True)


def build_feature_row(city: str, history: list[dict]) -> tuple[pd.DataFrame, date]:
    """Build the single feature row used to forecast the day after `history`.

    Returns (one-row frame, forecast_date). Raises HistoryError when the run is
    too short or too gappy to support the lag features.
    """
    if len(history) < MIN_HISTORY_DAYS:
        raise HistoryError(
            f"need at least {MIN_HISTORY_DAYS} days of history, got {len(history)}"
        )

    frame = _to_frame(city, history)
    last_day = frame["date"].iloc[-1].date()

    # Gaps must become explicit NaN rows, exactly as in training - otherwise a
    # missing day silently shifts what "yesterday" means.
    filled = reindex_continuous(frame)
    recent = filled[filled["date"] > pd.Timestamp(last_day - timedelta(days=MIN_HISTORY_DAYS))]
    if recent["pm25"].isna().any():
        missing = sorted(
            recent.loc[recent["pm25"].isna(), "date"].dt.strftime("%Y-%m-%d").tolist()
        )
        raise HistoryError(
            f"history has gaps in the {MIN_HISTORY_DAYS} days before {last_day}: "
            f"{', '.join(missing)}"
        )

    featured = add_features(filled)
    row = featured[featured["date"] == pd.Timestamp(last_day)]
    if row.empty:
        raise HistoryError(f"no row produced for {last_day}")
    return row.reset_index(drop=True), last_day + timedelta(days=1)
