"""Multi-parameter pooling: parameter separation, circular wind, sensor counts."""

import numpy as np
import pandas as pd
import pytest

from src.build_dataset import pool_to_city_day


def rows(records):
    df = pd.DataFrame.from_records(
        records, columns=["city", "sensor_id", "parameter", "date", "value"]
    )
    df["date"] = pd.to_datetime(df["date"])
    return df


def test_parameters_become_separate_columns():
    """Temperature must never be averaged into the pollutant series."""
    out = pool_to_city_day(
        rows([
            ("delhi", 1, "pm25", "2025-01-01", 100.0),
            ("delhi", 2, "temperature", "2025-01-01", 20.0),
            ("delhi", 3, "relativehumidity", "2025-01-01", 60.0),
        ])
    )
    assert out["pm25"].iloc[0] == 100.0
    assert out["temperature"].iloc[0] == 20.0
    assert out["humidity"].iloc[0] == 60.0  # renamed from relativehumidity


def test_n_sensors_counts_only_pm25():
    """The sampling-shift signal is about pollutant sensors, not weather ones."""
    out = pool_to_city_day(
        rows([
            ("delhi", 1, "pm25", "2025-01-01", 100.0),
            ("delhi", 2, "pm25", "2025-01-01", 200.0),
            ("delhi", 3, "temperature", "2025-01-01", 20.0),
            ("delhi", 4, "temperature", "2025-01-01", 22.0),
        ])
    )
    assert out["n_sensors"].iloc[0] == 2
    assert out["pm25"].iloc[0] == 150.0


def test_wind_direction_uses_circular_mean():
    """350 deg and 10 deg average to ~0 deg, not 180 - the naive mean is backwards."""
    out = pool_to_city_day(
        rows([
            ("delhi", 1, "wind_direction", "2025-01-01", 350.0),
            ("delhi", 2, "wind_direction", "2025-01-01", 10.0),
            ("delhi", 3, "pm25", "2025-01-01", 100.0),
        ])
    )
    angle = np.rad2deg(np.arctan2(out["wind_dir_sin"].iloc[0], out["wind_dir_cos"].iloc[0]))
    assert angle == pytest.approx(0.0, abs=1e-6)
    # A naive arithmetic mean would have produced 180.
    assert out["wind_dir_cos"].iloc[0] > 0.9


def test_wind_direction_opposing_winds_cancel():
    """Genuinely opposed winds give a near-zero resultant - correct, not a bug."""
    out = pool_to_city_day(
        rows([
            ("delhi", 1, "wind_direction", "2025-01-01", 0.0),
            ("delhi", 2, "wind_direction", "2025-01-01", 180.0),
            ("delhi", 3, "pm25", "2025-01-01", 100.0),
        ])
    )
    mag = np.hypot(out["wind_dir_sin"].iloc[0], out["wind_dir_cos"].iloc[0])
    assert mag == pytest.approx(0.0, abs=1e-6)


def test_missing_weather_does_not_drop_pm25_rows():
    """A day with pm25 but no weather must survive with NaN weather."""
    out = pool_to_city_day(
        rows([
            ("delhi", 1, "pm25", "2025-01-01", 100.0),
            ("delhi", 1, "pm25", "2025-01-02", 110.0),
            ("delhi", 2, "temperature", "2025-01-02", 20.0),
        ])
    )
    assert len(out) == 2
    assert pd.isna(out.sort_values("date")["temperature"].iloc[0])
