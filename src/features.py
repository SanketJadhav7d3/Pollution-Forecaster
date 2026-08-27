"""Feature contract shared by training and serving.

Kept separate from train.py and experiments.py on purpose: the inference path
must not import the training stack. It previously reached `build_x` through
experiments -> backtest -> matplotlib, which dragged plotting and modelling
libraries into the serving image and broke the container the moment they were
stripped out.
"""

from __future__ import annotations

import pandas as pd

# Required: a row without these is dropped, since the pollutant history is the
# irreducible core of the forecast.
CORE_FEATURES = [
    "pm25",  # today's completed 24h mean - available when forecasting tomorrow
    "pm25_lag_1", "pm25_lag_2", "pm25_lag_3",
    "pm25_roll_3", "pm25_roll_7", "pm25_roll_7_std",
    "pm25_delta_1", "pm25_delta_3", "pm25_vs_roll_7",
    "n_sensors", "month", "day_of_week", "day_of_year", "is_stubble_season",
]

# Optional: meteorology only starts 2025-04, and requiring it would discard the
# first eight months of pollutant history. sklearn >=1.4 and xgboost route NaN
# down a learned default branch, so these stay usable where present and simply
# carry no information where absent.
WEATHER_FEATURES = [
    "temperature", "temperature_lag_1", "temperature_roll_3", "temperature_delta_1",
    "humidity", "humidity_lag_1", "humidity_roll_3", "humidity_delta_1",
    "wind_speed", "wind_speed_lag_1", "wind_speed_roll_3", "wind_speed_delta_1",
    "wind_speed_roll_3_min", "wind_dir_sin", "wind_dir_cos",
]

FEATURES = CORE_FEATURES + WEATHER_FEATURES
TARGET = "target_aqi_category"


def build_x(frame: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Select available features and append the ordinal city code.

    Trees handle ordinal codes without one-hot encoding (specs.md §5).
    """
    cols = [c for c in features if c in frame.columns]
    return frame[cols].assign(city_code=frame["city"].astype("category").cat.codes)
