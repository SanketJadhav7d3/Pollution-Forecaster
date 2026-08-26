"""Feature/target correctness — above all, no lookahead leakage (specs.md §4)."""

import numpy as np
import pandas as pd
import pytest

from src.build_dataset import add_features, add_target, pool_to_city_day, reindex_continuous

FEATURE_COLS = [
    "pm25_lag_1", "pm25_lag_2", "pm25_lag_3",
    "pm25_roll_3", "pm25_roll_7", "pm25_roll_7_std",
    "pm25_delta_1", "pm25_delta_3", "pm25_vs_roll_7",
]


def frame(values, city="delhi", start="2025-01-01"):
    return pd.DataFrame(
        {
            "city": city,
            "date": pd.date_range(start, periods=len(values), freq="D"),
            "pm25": values,
            "n_sensors": 1,
        }
    )


def test_lag_1_is_yesterdays_value():
    df = add_features(frame([10.0, 20.0, 30.0, 40.0]))
    assert np.isnan(df["pm25_lag_1"].iloc[0])
    assert df["pm25_lag_1"].tolist()[1:] == [10.0, 20.0, 30.0]


def test_no_feature_contains_tomorrows_reading():
    """The real leak: a row's features must not see the day it is predicting.

    Today's own reading IS allowed - forecasting D+1 on the evening of D, day
    D's completed mean is available and is the strongest single predictor.
    """
    base = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]
    a = add_features(frame(base))
    changed = base.copy()
    changed[-1] = 9999.0  # perturb the FUTURE day relative to row -2
    b = add_features(frame(changed))
    pd.testing.assert_series_equal(
        a[FEATURE_COLS].iloc[-2], b[FEATURE_COLS].iloc[-2], check_names=False
    )


def test_todays_reading_is_available_as_a_feature():
    base = [10.0, 20.0, 30.0, 40.0, 50.0]
    a = add_features(frame(base))
    changed = base.copy()
    changed[-1] = 9999.0
    b = add_features(frame(changed))
    # Perturbing today MUST move today's own rolling features.
    assert a["pm25_roll_3"].iloc[-1] != b["pm25_roll_3"].iloc[-1]


def test_target_is_tomorrows_category_not_todays():
    df = add_target(frame([10.0, 400.0, 10.0]))
    # Day 0's target must describe day 1 (400 -> Severe), not day 0 (10 -> Good).
    assert df["target_aqi_category"].iloc[0] == "Severe"
    assert df["pm25_next_day"].iloc[0] == 400.0


def test_last_row_has_no_target():
    df = add_target(frame([10.0, 20.0, 30.0]))
    assert pd.isna(df["target_aqi_category"].iloc[-1])


def test_rolling_mean_spans_today_and_prior_days():
    df = add_features(frame([10.0, 20.0, 30.0, 40.0]))
    # Row 2: roll_3 = mean(10, 20, 30) - today inclusive, tomorrow excluded.
    assert df["pm25_roll_3"].iloc[2] == pytest.approx(20.0)
    assert df["pm25_roll_3"].iloc[3] == pytest.approx(30.0)


def test_gap_days_are_reindexed_not_dropped():
    """A missing day must become explicit NaN, else lag_1 silently spans the gap."""
    raw = pd.DataFrame(
        {
            "city": ["delhi"] * 3,
            "date": pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-05"]),
            "pm25": [10.0, 20.0, 50.0],
            "n_sensors": [1, 1, 1],
        }
    )
    out = reindex_continuous(raw)
    assert len(out) == 5
    assert out["pm25"].isna().sum() == 2
    assert out.loc[out["date"] == "2025-01-04", "n_sensors"].iloc[0] == 0


def test_lag_respects_gaps_after_reindex():
    raw = pd.DataFrame(
        {
            "city": ["delhi"] * 2,
            "date": pd.to_datetime(["2025-01-01", "2025-01-05"]),
            "pm25": [10.0, 50.0],
            "n_sensors": [1, 1],
        }
    )
    out = add_features(reindex_continuous(raw))
    jan5 = out[out["date"] == "2025-01-05"].iloc[0]
    # Jan 4 was missing, so "yesterday" is unknown - NOT Jan 1's value.
    assert pd.isna(jan5["pm25_lag_1"])


def test_cities_do_not_bleed_into_each_other():
    df = pd.concat([frame([10.0, 20.0], "delhi"), frame([500.0, 600.0], "mumbai")])
    out = add_features(df)
    mumbai_first = out[(out["city"] == "mumbai")].sort_values("date").iloc[0]
    assert pd.isna(mumbai_first["pm25_lag_1"])


def test_pooling_averages_sensors_and_counts_them():
    rows = pd.DataFrame(
        {
            "city": ["delhi"] * 3,
            "sensor_id": [1, 2, 1],
            "parameter": ["pm25"] * 3,
            "date": pd.to_datetime(["2025-01-01", "2025-01-01", "2025-01-02"]),
            "value": [10.0, 30.0, 50.0],
        }
    )
    out = pool_to_city_day(rows).sort_values("date")
    assert out["pm25"].tolist() == [20.0, 50.0]
    assert out["n_sensors"].tolist() == [2, 1]
