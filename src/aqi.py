"""CPCB AQI categories for PM2.5 (24h average), India's national standard.

Breakpoints from the CPCB National Air Quality Index (2014). PM2.5 alone is
used because it drives the reported category on most days in Delhi/Mumbai and
is the only pollutant backfilled (config/cities.yaml).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# (upper_bound_inclusive, label). Ordered best -> worst.
PM25_BREAKPOINTS: list[tuple[float, str]] = [
    (30.0, "Good"),
    (60.0, "Satisfactory"),
    (90.0, "Moderate"),
    (120.0, "Poor"),
    (250.0, "Very Poor"),
    (float("inf"), "Severe"),
]

CATEGORIES: list[str] = [label for _, label in PM25_BREAKPOINTS]


def pm25_to_category(values: pd.Series) -> pd.Series:
    """Map 24h-mean PM2.5 (ug/m3) to its CPCB category.

    NaN in -> NaN out: a missing reading must not silently become "Good".
    """
    bins = [-np.inf] + [b for b, _ in PM25_BREAKPOINTS]
    cut = pd.cut(values, bins=bins, labels=CATEGORIES, right=True)
    return pd.Series(
        pd.Categorical(cut, categories=CATEGORIES, ordered=True),
        index=values.index,
        name="aqi_category",
    )
