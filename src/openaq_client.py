"""Thin OpenAQ v3 client: auth, rate limiting, retries, pagination.

Deliberately has no knowledge of cities or files — that lives in
fetch_openaq.py, so this stays reusable inside a Cloud Function (specs.md §2).
"""

from __future__ import annotations

import logging
import os
import time
from collections import deque
from datetime import UTC, date, datetime, timedelta

import requests

log = logging.getLogger(__name__)

BASE_URL = "https://api.openaq.org/v3"

# Free tier is 60 req/min. Stay under it — a 429 mid-backfill wastes the
# bandwidth already spent on the in-flight page.
MAX_REQUESTS_PER_MINUTE = 45
MAX_RETRIES = 5
PAGE_LIMIT = 1000


class OpenAQError(RuntimeError):
    pass


class DateFilterIgnored(OpenAQError):
    """The API returned data outside the requested window.

    v3 aggregate endpoints silently ignore unknown date params: passing
    `datetime_from` instead of `date_from` returns HTTP 200 with the sensor's
    *entire* history. In daily mode that turns a ~2-row fetch into a 3-year
    one per sensor, while still looking like it worked. Fail loudly instead.
    """


class OpenAQClient:
    def __init__(self, api_key: str | None = None, session: requests.Session | None = None):
        key = api_key or os.getenv("OPENAQ_API_KEY")
        if not key:
            raise OpenAQError("OPENAQ_API_KEY is not set (copy .env.example to .env)")
        self.session = session or requests.Session()
        self.session.headers.update({"X-API-Key": key})
        self._calls: deque[float] = deque()
        self.bytes_downloaded = 0
        self.requests_made = 0

    # --- rate limiting -------------------------------------------------
    def _throttle(self) -> None:
        now = time.monotonic()
        while self._calls and now - self._calls[0] > 60:
            self._calls.popleft()
        if len(self._calls) >= MAX_REQUESTS_PER_MINUTE:
            sleep_for = 60 - (now - self._calls[0]) + 0.1
            log.debug("rate limit: sleeping %.1fs", sleep_for)
            time.sleep(sleep_for)
        self._calls.append(time.monotonic())

    def _get(self, path: str, params: dict) -> dict:
        url = f"{BASE_URL}/{path.lstrip('/')}"
        for attempt in range(MAX_RETRIES):
            self._throttle()
            try:
                r = self.session.get(url, params=params, timeout=60)
            except requests.RequestException as exc:
                wait = 2**attempt
                log.warning("%s failed (%s); retry in %ss", path, exc, wait)
                time.sleep(wait)
                continue

            if r.status_code == 200:
                self.bytes_downloaded += len(r.content)
                self.requests_made += 1
                return r.json()
            if r.status_code == 429 or r.status_code >= 500:
                wait = int(r.headers.get("retry-after", 2**attempt))
                log.warning("HTTP %s on %s; retry in %ss", r.status_code, path, wait)
                time.sleep(wait)
                continue
            raise OpenAQError(f"HTTP {r.status_code} on {path}: {r.text[:200]}")
        raise OpenAQError(f"{path} failed after {MAX_RETRIES} attempts")

    def _paginate(self, path: str, params: dict):
        page = 1
        while True:
            body = self._get(path, {**params, "limit": PAGE_LIMIT, "page": page})
            results = body.get("results", [])
            yield from results
            if len(results) < PAGE_LIMIT:
                return
            page += 1

    # --- discovery -----------------------------------------------------
    def find_sensors(
        self,
        latitude: float,
        longitude: float,
        radius_meters: int,
        parameters: list[str],
        stale_after_days: int,
    ) -> list[dict]:
        """Live sensors matching `parameters` within the radius.

        Skips locations gone silent past `stale_after_days` — a large share of
        CPCB stations are long dead and would burn quota returning nothing.
        """
        cutoff = datetime.now(UTC) - timedelta(days=stale_after_days)
        wanted = set(parameters)
        found: list[dict] = []

        for loc in self._paginate(
            "locations",
            {"coordinates": f"{latitude},{longitude}", "radius": radius_meters},
        ):
            last = (loc.get("datetimeLast") or {}).get("utc")
            if not last:
                continue
            if datetime.fromisoformat(last) < cutoff:
                continue
            coords = loc.get("coordinates") or {}
            for s in loc.get("sensors", []):
                if s["parameter"]["name"] not in wanted:
                    continue
                found.append(
                    {
                        "sensor_id": s["id"],
                        "parameter": s["parameter"]["name"],
                        "units": s["parameter"].get("units"),
                        "location_id": loc["id"],
                        "location_name": loc.get("name"),
                        "latitude": coords.get("latitude"),
                        "longitude": coords.get("longitude"),
                    }
                )
        return found

    # --- measurements --------------------------------------------------
    def daily_measurements(self, sensor_id: int, date_from: date, date_to: date) -> list[dict]:
        """Daily-aggregated rows for one sensor, normalised and date-checked."""
        rows: list[dict] = []
        for rec in self._paginate(
            f"sensors/{sensor_id}/days",
            {"date_from": date_from.isoformat(), "date_to": date_to.isoformat()},
        ):
            period = (rec.get("period") or {}).get("datetimeFrom", {})
            stamp = period.get("utc")
            if not stamp:
                continue
            day = stamp[:10]
            rows.append(
                {
                    "date": day,
                    "value": rec.get("value"),
                    "min": (rec.get("summary") or {}).get("min"),
                    "max": (rec.get("summary") or {}).get("max"),
                    "count": (rec.get("coverage") or {}).get("observedCount"),
                }
            )

        if rows:
            earliest = min(r["date"] for r in rows)
            # Tolerate one day of timezone slack at the boundary.
            if earliest < (date_from - timedelta(days=1)).isoformat():
                raise DateFilterIgnored(
                    f"sensor {sensor_id}: asked from {date_from}, got {earliest}. "
                    "The date filter was ignored — check the param names."
                )
        return rows
