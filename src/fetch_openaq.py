"""OpenAQ fetcher — backfill (once) and daily (scheduled) modes (specs.md §2).

    uv run python -m src.fetch_openaq discover
    uv run python -m src.fetch_openaq backfill --years 3
    uv run python -m src.fetch_openaq daily

Raw data is immutable and append-only: files are merged by date, never
truncated, and written via a temp file + atomic rename so an interrupted run
cannot corrupt existing history.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import yaml
from dotenv import load_dotenv

from src.openaq_client import DateFilterIgnored, OpenAQClient

log = logging.getLogger("fetch_openaq")

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config" / "cities.yaml"
RAW_DIR = ROOT / "data" / "raw" / "openaq"

# Daily mode re-pulls a short trailing window, not just yesterday: OpenAQ
# backfills late-arriving readings, so a strict 1-day fetch permanently misses
# values that land after we looked.
DAILY_LOOKBACK_DAYS = 2


def load_config() -> dict:
    with CONFIG_PATH.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def city_dir(city: str) -> Path:
    d = RAW_DIR / city
    d.mkdir(parents=True, exist_ok=True)
    return d


def write_json_atomic(path: Path, payload) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    tmp.replace(path)


def read_json(path: Path):
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


# --- discovery ---------------------------------------------------------
def discover(client: OpenAQClient, cfg: dict, refresh: bool = False) -> dict[str, list[dict]]:
    """Find live sensors per city, caching to _sensors.json."""
    out: dict[str, list[dict]] = {}
    for city in cfg["cities"]:
        name = city["name"]
        cache = city_dir(name) / "_sensors.json"
        if cache.exists() and not refresh:
            out[name] = read_json(cache)
            log.info("%-7s %3d sensors (cached)", name, len(out[name]))
            continue
        sensors = client.find_sensors(
            latitude=city["latitude"],
            longitude=city["longitude"],
            radius_meters=city["radius_meters"],
            parameters=cfg["parameters"],
            stale_after_days=cfg.get("stale_after_days", 120),
        )
        write_json_atomic(cache, sensors)
        out[name] = sensors
        log.info("%-7s %3d sensors (discovered)", name, len(sensors))
    return out


# --- merge -------------------------------------------------------------
def merge_rows(existing: dict | None, new_rows: list[dict], meta: dict) -> tuple[dict, int]:
    """Merge by date key. Returns (file_payload, count_of_new_dates)."""
    rows = {r["date"]: r for r in (existing or {}).get("rows", [])}
    before = len(rows)
    for r in new_rows:
        rows[r["date"]] = r
    merged = {
        **meta,
        "fetched_at": datetime.now(UTC).isoformat(),
        "rows": [rows[k] for k in sorted(rows)],
    }
    return merged, len(rows) - before


def sensor_path(city: str, sensor: dict) -> Path:
    return city_dir(city) / f"sensor_{sensor['sensor_id']}.json"


def fetch_sensor(
    client: OpenAQClient,
    city: str,
    sensor: dict,
    date_from: date,
    date_to: date,
    skip_complete: bool,
) -> tuple[int, bool]:
    """Fetch one sensor's range. Returns (new_row_count, skipped)."""
    path = sensor_path(city, sensor)
    existing = read_json(path)

    if skip_complete and existing and existing.get("rows"):
        covered = {r["date"] for r in existing["rows"]}
        # Already-covered ranges are skipped so an interrupted backfill resumes
        # instead of re-downloading — the run's bandwidth is the scarce resource.
        if existing.get("range_from", "9999") <= date_from.isoformat() and any(
            d >= (date_to - timedelta(days=7)).isoformat() for d in covered
        ):
            return 0, True

    rows = client.daily_measurements(sensor["sensor_id"], date_from, date_to)
    meta = {
        "city": city,
        **{k: sensor[k] for k in ("sensor_id", "parameter", "units", "location_id", "location_name", "latitude", "longitude")},
        "range_from": min(
            date_from.isoformat(), (existing or {}).get("range_from", date_from.isoformat())
        ),
    }
    merged, added = merge_rows(existing, rows, meta)
    write_json_atomic(path, merged)
    return added, False


def run_fetch(client: OpenAQClient, cfg: dict, date_from: date, date_to: date, skip_complete: bool, limit_sensors: int | None = None) -> None:
    sensors_by_city = discover(client, cfg)
    if limit_sensors:
        trimmed, budget = {}, limit_sensors
        for city, sensors in sensors_by_city.items():
            trimmed[city] = sensors[:budget]
            budget -= len(trimmed[city])
            if budget <= 0:
                break
        sensors_by_city = trimmed
    total = sum(len(v) for v in sensors_by_city.values())
    log.info("fetching %s → %s across %d sensors", date_from, date_to, total)

    done = added_total = skipped = failed = 0
    for city, sensors in sensors_by_city.items():
        for s in sensors:
            done += 1
            try:
                added, was_skipped = fetch_sensor(client, city, s, date_from, date_to, skip_complete)
            except DateFilterIgnored:
                # Never swallow this: it means every sensor is pulling full
                # history instead of the requested window. Abort the run.
                raise
            except Exception as exc:  # noqa: BLE001 - one dead sensor must not end the run
                failed += 1
                log.warning("[%d/%d] sensor %s failed: %s", done, total, s["sensor_id"], exc)
                continue
            if was_skipped:
                skipped += 1
            else:
                added_total += added
            if done % 20 == 0 or done == total:
                log.info("[%d/%d] +%d rows, %d skipped, %d failed", done, total, added_total, skipped, failed)

    mb = client.bytes_downloaded / 1_048_576
    log.info("DONE: %d new rows, %d skipped, %d failed", added_total, skipped, failed)
    log.info("downloaded %.2f MB over %d requests", mb, client.requests_made)
    fetched = total - skipped
    if fetched:
        log.info("~%.2f MB per sensor -> ~%.0f MB for all 151 sensors", mb / fetched, mb / fetched * 151)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Fetch OpenAQ daily data for configured cities")
    sub = p.add_subparsers(dest="mode", required=True)
    sub.add_parser("discover").add_argument("--refresh", action="store_true")
    b = sub.add_parser("backfill")
    b.add_argument("--years", type=float, default=2.0)
    b.add_argument("--force", action="store_true", help="re-fetch even if already covered")
    b.add_argument("--limit-sensors", type=int, help="stop after N sensors (dry run)")
    d = sub.add_parser("daily")
    d.add_argument("--days", type=int, default=DAILY_LOOKBACK_DAYS)
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
    load_dotenv()
    cfg = load_config()
    client = OpenAQClient(os.getenv("OPENAQ_API_KEY"))

    if args.mode == "discover":
        discover(client, cfg, refresh=args.refresh)
        return 0

    today = datetime.now(UTC).date()
    if args.mode == "backfill":
        date_from = today - timedelta(days=int(args.years * 365))
        run_fetch(client, cfg, date_from, today, skip_complete=not args.force, limit_sensors=args.limit_sensors)
    else:
        run_fetch(client, cfg, today - timedelta(days=args.days), today, skip_complete=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
