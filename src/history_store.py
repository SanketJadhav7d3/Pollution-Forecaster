"""Read the published recent history the API predicts from.

Replaces the live OpenAQ query in the request path. That query needed one HTTP
call per sensor - about 65 seconds for Mumbai's 49 sensors against the rate
limit, and longer for Delhi - which is more than a Cloud Run request is allowed
to take. The scheduled publish job does that work once a day instead, so serving
becomes a single small read.

The result is cached briefly: the published file changes once a day, so
re-downloading it per request would add latency and cost for nothing.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

from src import gcs
from src.publish import latest_key

log = logging.getLogger("history_store")

# The published file is rewritten once a day. A few minutes of staleness is
# invisible to a next-day forecast and removes a network hop from most requests.
CACHE_SECONDS = 300

_cache: dict[str, tuple[float, dict]] = {}


class HistoryUnavailable(RuntimeError):
    """No published history could be read for this city."""


def processed_root() -> str | None:
    return os.getenv("PROCESSED_BUCKET")


def _fetch(city: str) -> dict:
    root = processed_root()
    if not root:
        raise HistoryUnavailable(
            "PROCESSED_BUCKET is not set, so there is nowhere to read history from"
        )
    uri = f"{root.rstrip('/')}/{latest_key(city)}"
    try:
        payload = gcs.read_json(uri)
    except Exception as exc:
        raise HistoryUnavailable(f"could not read {uri}: {exc}") from exc
    if not payload.get("history"):
        raise HistoryUnavailable(f"{uri} contains no history")
    return payload


def get_history(city: str, force: bool = False) -> dict[str, Any]:
    """Published history for a city, cached for CACHE_SECONDS."""
    now = time.monotonic()
    hit = _cache.get(city)
    if hit and not force and (now - hit[0]) < CACHE_SECONDS:
        return hit[1]
    payload = _fetch(city)
    _cache[city] = (now, payload)
    return payload


def clear_cache() -> None:
    _cache.clear()
