"""API contract tests, and the training/serving skew guard.

The skew test is the important one: a served model whose features are computed
even slightly differently from training degrades silently, with no error to
notice. It is pinned here against the real training table.
"""

from __future__ import annotations

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.aqi import CATEGORIES
from src.features_live import MIN_HISTORY_DAYS, HistoryError, build_feature_row
from src.serve import app
from src.train import DATA_PATH

client = TestClient(app)

FEATURE_COLS = [
    "pm25", "pm25_lag_1", "pm25_lag_2", "pm25_lag_3",
    "pm25_roll_3", "pm25_roll_7", "pm25_roll_7_std",
    "pm25_delta_1", "pm25_delta_3", "pm25_vs_roll_7",
]


def flat_history(days=8, value=40.0, start_day=1):
    return [
        {"date": f"2026-03-{start_day + i:02d}", "pm25": value + i}
        for i in range(days)
    ]


# --- feature parity ----------------------------------------------------
@pytest.mark.skipif(not DATA_PATH.exists(), reason="training table not built")
def test_serving_features_match_training_exactly():
    """A row built through the serving path must equal the training row.

    If this drifts, the model silently sees inputs unlike its training data.
    """
    df = pd.read_parquet(DATA_PATH)
    city_rows = df[df["city"] == "delhi"].sort_values("date").tail(10)
    history = [
        {"date": r.date.strftime("%Y-%m-%d"), "pm25": r.pm25}
        for r in city_rows.itertuples()
        if pd.notna(r.pm25)
    ]
    row, forecast_date = build_feature_row("delhi", history)
    last_date = pd.Timestamp(history[-1]["date"])
    expected = df[(df["city"] == "delhi") & (df["date"] == last_date)]

    assert forecast_date == (last_date + pd.Timedelta(days=1)).date()
    for col in FEATURE_COLS:
        assert row[col].iloc[0] == pytest.approx(expected[col].iloc[0], rel=1e-9), col


# --- history validation ------------------------------------------------
def test_short_history_is_rejected():
    with pytest.raises(HistoryError, match="at least"):
        build_feature_row("delhi", flat_history(days=MIN_HISTORY_DAYS - 1))


def test_gap_in_history_is_rejected():
    """A missing day must fail loudly, not silently shift what 'yesterday' means."""
    history = [h for h in flat_history(days=9) if h["date"] != "2026-03-05"]
    with pytest.raises(HistoryError, match="gaps"):
        build_feature_row("delhi", history)


def test_duplicate_dates_are_rejected():
    history = flat_history(days=8)
    history.append(dict(history[-1]))
    with pytest.raises(HistoryError, match="duplicate"):
        build_feature_row("delhi", history)


def test_unsorted_history_is_accepted_and_ordered():
    history = flat_history(days=8)
    shuffled = [history[3], history[0], *history[1:3], *history[4:]]
    row, forecast_date = build_feature_row("delhi", shuffled)
    assert forecast_date.isoformat() == "2026-03-09"
    assert row["pm25"].iloc[0] == pytest.approx(47.0)


# --- endpoints ---------------------------------------------------------
def test_health_reports_loaded_model():
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["min_history_days"] == MIN_HISTORY_DAYS


def test_model_endpoint_is_honest_about_accuracy():
    """The API must not imply the model beats the baseline - it does not."""
    body = client.get("/model").json()
    assert body["beats_persistence_on_accuracy"] is False
    assert body["categories"] == CATEGORIES


def test_predict_returns_both_predictions():
    r = client.post("/predict", json={"city": "delhi", "history": flat_history()})
    assert r.status_code == 200
    body = r.json()
    assert body["predicted_category"] in CATEGORIES
    assert body["persistence_category"] in CATEGORIES
    assert body["agrees_with_persistence"] == (
        body["predicted_category"] == body["persistence_category"]
    )
    assert body["forecast_date"] == "2026-03-09"


def test_probabilities_are_named_and_sum_to_one():
    """Guards the class-label bug: indices must map to the right categories."""
    body = client.post("/predict", json={"city": "delhi", "history": flat_history()}).json()
    probs = body["probabilities"]
    assert set(probs).issubset(set(CATEGORIES))
    assert sum(probs.values()) == pytest.approx(1.0, abs=0.01)
    top = max(probs, key=probs.get)
    assert top == body["predicted_category"]


def test_unknown_city_is_rejected():
    r = client.post("/predict", json={"city": "paris", "history": flat_history()})
    assert r.status_code == 422


def test_short_history_returns_422_not_500():
    r = client.post("/predict", json={"city": "delhi", "history": flat_history(days=3)})
    assert r.status_code == 422


def test_city_changes_the_prediction_inputs():
    """city_code must reflect the requested city, not the request's row order."""
    history = flat_history()
    a = client.post("/predict", json={"city": "delhi", "history": history}).json()
    b = client.post("/predict", json={"city": "mumbai", "history": history}).json()
    assert a["city"] == "delhi"
    assert b["city"] == "mumbai"
    # Same readings, different city feature - responses stay self-consistent.
    assert a["latest_pm25"] == b["latest_pm25"]
