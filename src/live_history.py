"""Fetch a city's recent daily PM2.5 straight from OpenAQ, for live serving.

Pools across the city's sensors exactly as build_dataset does, because a single
sensor cannot be relied on: stations are retired and re-registered under new ids,
and the network drops out for days at a time. Asking one sensor for "the last
week" regularly returns nothing.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta

from src.fetch_openaq import city_dir, load_config, read_json
from src.openaq_client import OpenAQClient, OpenAQError

log = logging.getLogger("live_history")

# Ask for more days than the model needs: the network is patchy, and the last
# day or two are often still filling in.
FETCH_MARGIN_DAYS = 6


class LiveHistoryError(RuntimeError):
    """Recent history could not be assembled from OpenAQ."""


def _cached_sensor_ids(city: str, parameter: str = "pm25") -> list[int]:
    """Sensor ids from the discovery cache written by `fetch_openaq discover`."""
    cache = city_dir(city) / "_sensors.json"
    sensors = read_json(cache)
    if not sensors:
        raise LiveHistoryError(
            f"no sensor cache for '{city}' - run: python -m src.fetch_openaq discover"
        )
    return [s["sensor_id"] for s in sensors if s.get("parameter") == parameter]


def fetch_recent_history(city: str, days: int = 10, client: OpenAQClient | None = None) -> list[dict]:
    """Return [{date, pm25, n_sensors}, ...] ascending, pooled across sensors."""
    cfg = load_config()
    known = {c["name"] for c in cfg["cities"]}
    if city not in known:
        raise LiveHistoryError(f"unknown city '{city}'")

    sensor_ids = _cached_sensor_ids(city)
    if not sensor_ids:
        raise LiveHistoryError(f"no pm25 sensors cached for '{city}'")

    client = client or OpenAQClient()
    today = datetime.now(UTC).date()
    date_from = today - timedelta(days=days + FETCH_MARGIN_DAYS)

    totals: dict[str, list[float]] = {}
    failures = 0
    for sensor_id in sensor_ids:
        try:
            rows = client.daily_measurements(sensor_id, date_from, today)
        except OpenAQError:
            failures += 1
            continue
        for row in rows:
            if row.get("value") is not None:
                totals.setdefault(row["date"], []).append(float(row["value"]))

    if not totals:
        raise LiveHistoryError(
            f"OpenAQ returned no readings for '{city}' since {date_from} "
            f"({failures} of {len(sensor_ids)} sensors errored)"
        )

    history = [
        {"date": day, "pm25": sum(vals) / len(vals), "n_sensors": len(vals)}
        for day, vals in sorted(totals.items())
    ]
    log.info("%s: %d days from %d sensors (%d errored)", city, len(history), len(sensor_ids), failures)
    return history[-days:]
