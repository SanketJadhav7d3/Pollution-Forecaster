"""Fetch recent readings, archive them, and publish what the API reads.

    uv run python -m src.publish --raw gs://<p>-raw --processed gs://<p>-processed

Runs on a schedule. Three things happen, in order:

  1. fetch the last N days for every sensor in each city (the slow part)
  2. write the untouched response to the raw bucket, append-only, never
     overwritten - this is the archive retraining and audits depend on
  3. pool sensors to one value per day and publish a small per-city file

Step 3 exists so the API never has to do step 1. Pooling ~100 sensors takes
about a minute per city against OpenAQ's rate limit, which is longer than a
Cloud Run request is allowed to live. Doing it once per day on write turns a
request that cannot finish into a single small read.
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path

from dotenv import load_dotenv

from src import gcs
from src.fetch_openaq import load_config
from src.openaq_client import OpenAQClient, OpenAQError

log = logging.getLogger("publish")

# Published history length. The model needs 7 days; the surplus absorbs the
# gaps and late-arriving readings that are normal for this network.
PUBLISH_DAYS = 30
FETCH_DAYS = PUBLISH_DAYS + 10


# Which sensors to query, shipped with the code. It previously came from the
# local data directory, which does not exist inside the container - the same
# assumption that broke the first deployed endpoint. Refresh it with
# `fetch_openaq discover` and commit the result.
SENSORS_PATH = Path(__file__).resolve().parent.parent / "config" / "sensors.json"


def latest_key(city: str) -> str:
    return f"latest/{city}.json"


def load_sensor_ids(cities: list[str]) -> dict[str, list[int]]:
    if not SENSORS_PATH.exists():
        raise RuntimeError(
            f"{SENSORS_PATH} is missing - regenerate it from the discovery cache"
        )
    known = json.loads(SENSORS_PATH.read_text(encoding="utf-8"))
    return {c: list(known.get(c, [])) for c in cities}


def fetch_city(client: OpenAQClient, city: str, sensor_ids: list[int],
               days: int = FETCH_DAYS) -> tuple[list[dict], dict]:
    """Return (pooled daily rows, raw per-sensor payload)."""
    today = datetime.now(UTC).date()
    date_from = today - timedelta(days=days)

    per_day: dict[str, list[float]] = {}
    raw: dict[str, list[dict]] = {}
    failures = 0

    for sensor_id in sensor_ids:
        try:
            rows = client.daily_measurements(sensor_id, date_from, today)
        except OpenAQError as exc:
            failures += 1
            log.warning("sensor %s failed: %s", sensor_id, exc)
            continue
        raw[str(sensor_id)] = rows
        for row in rows:
            if row.get("value") is not None:
                per_day.setdefault(row["date"], []).append(float(row["value"]))

    pooled = [
        {"date": day, "pm25": round(sum(v) / len(v), 3), "n_sensors": len(v)}
        for day, v in sorted(per_day.items())
    ]
    meta = {
        "city": city,
        "fetched_at": datetime.now(UTC).isoformat(),
        "date_from": date_from.isoformat(),
        "date_to": today.isoformat(),
        "sensors_queried": len(sensor_ids),
        "sensors_failed": failures,
        "days_returned": len(pooled),
    }
    return pooled, {**meta, "sensors": raw}


def publish_city(client: OpenAQClient, city: str, sensor_ids: list[int],
                 raw_root: str | None, processed_root: str | None) -> dict:
    pooled, raw_payload = fetch_city(client, city, sensor_ids)
    if not pooled:
        raise RuntimeError(f"no readings returned for {city}")

    if raw_root:
        # Append-only: a re-run writes a new object rather than replacing the
        # previous answer, so the archive stays auditable.
        gcs.write_json(f"{raw_root.rstrip('/')}/{gcs.raw_key(city)}", raw_payload)

    recent = pooled[-PUBLISH_DAYS:]
    published = {
        "city": city,
        "published_at": datetime.now(UTC).isoformat(),
        "latest_date": recent[-1]["date"],
        "days": len(recent),
        "history": recent,
    }
    if processed_root:
        gcs.write_json(f"{processed_root.rstrip('/')}/{latest_key(city)}", published)

    log.info("%s: %d days, latest %s (%d sensors queried, %d failed)",
             city, len(recent), recent[-1]["date"],
             raw_payload["sensors_queried"], raw_payload["sensors_failed"])
    return published


def run(raw_root: str | None, processed_root: str | None,
        sensors_by_city: dict[str, list[int]] | None = None) -> dict:
    cfg = load_config()
    # Check the destinations before the slow fetch, not after it.
    for root in (raw_root, processed_root):
        if root:
            gcs.check_writable(f"{root.rstrip('/')}/_preflight")
    client = OpenAQClient()

    if sensors_by_city is None:
        sensors_by_city = load_sensor_ids([c["name"] for c in cfg["cities"]])

    out = {}
    for city, sensor_ids in sensors_by_city.items():
        if not sensor_ids:
            log.warning("%s: no sensors known, skipping", city)
            continue
        out[city] = publish_city(client, city, sensor_ids, raw_root, processed_root)
    log.info("downloaded %.1f MB over %d requests",
             client.bytes_downloaded / 1_048_576, client.requests_made)
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Fetch, archive and publish recent readings")
    p.add_argument("--raw", default=gcs.default_bucket("raw"),
                   help="raw archive root, e.g. gs://<project>-raw")
    p.add_argument("--processed", default=gcs.default_bucket("processed"),
                   help="published root the API reads, e.g. gs://<project>-processed")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
    load_dotenv()

    if not args.raw and not args.processed:
        p.error("give at least one of --raw / --processed")
    run(args.raw, args.processed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
