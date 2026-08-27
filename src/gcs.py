"""Thin object-storage helper that also works against the local filesystem.

Paths are either `gs://bucket/key` or an ordinary local path. Keeping both
behind one interface means the fetch and publish jobs are testable without a
cloud account, and run unchanged locally or on GCP.

Object storage is not a filesystem: there are no real directories (the slashes
are part of the key), and writing to an existing key replaces the whole object
rather than appending to it. Both facts shape how the raw layer is laid out.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import UTC, datetime
from pathlib import Path

log = logging.getLogger("gcs")

GCS_PREFIX = "gs://"


def is_gcs(path: str) -> bool:
    return str(path).startswith(GCS_PREFIX)


def split_uri(uri: str) -> tuple[str, str]:
    """gs://bucket/some/key -> ("bucket", "some/key")"""
    rest = uri[len(GCS_PREFIX):]
    bucket, _, key = rest.partition("/")
    return bucket, key


def _client():
    from google.cloud import storage  # imported lazily so local runs need no GCP

    return storage.Client()


def check_writable(uri: str) -> None:
    """Fail fast if the destination cannot be written.

    The publish job spends minutes fetching before it uploads anything, so an
    unusable destination must surface in seconds rather than after the
    expensive part has already run and been discarded.
    """
    if not is_gcs(uri):
        Path(uri).parent.mkdir(parents=True, exist_ok=True)
        return
    bucket, _ = split_uri(uri)
    try:
        _client().bucket(bucket).exists()
    except Exception as exc:
        raise RuntimeError(
            f"cannot reach {uri}: {exc}. "
            "If running locally, credentials for code are separate from the CLI's: "
            "run `gcloud auth application-default login`."
        ) from exc


def write_text(uri: str, text: str, content_type: str = "application/json") -> str:
    if is_gcs(uri):
        bucket, key = split_uri(uri)
        blob = _client().bucket(bucket).blob(key)
        blob.upload_from_string(text, content_type=content_type)
        log.info("wrote %s (%d bytes)", uri, len(text))
        return uri
    path = Path(uri)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return str(path)


def read_text(uri: str) -> str:
    if is_gcs(uri):
        bucket, key = split_uri(uri)
        return _client().bucket(bucket).blob(key).download_as_text()
    return Path(uri).read_text(encoding="utf-8")


def exists(uri: str) -> bool:
    if is_gcs(uri):
        bucket, key = split_uri(uri)
        return _client().bucket(bucket).blob(key).exists()
    return Path(uri).exists()


def write_json(uri: str, payload) -> str:
    return write_text(uri, json.dumps(payload, indent=2, sort_keys=True, default=str))


def read_json(uri: str):
    return json.loads(read_text(uri))


def upload_file(local_path: Path, uri: str) -> str:
    if is_gcs(uri):
        bucket, key = split_uri(uri)
        _client().bucket(bucket).blob(key).upload_from_filename(str(local_path))
        return uri
    dest = Path(uri)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(local_path.read_bytes())
    return str(dest)


def raw_key(city: str, fetched_at: datetime | None = None) -> str:
    """Append-only key for one fetch (specs.md §8).

    Partitioned `city=`/`date=` so query engines read those segments as columns,
    and stamped with the fetch time so a re-run writes a new object instead of
    overwriting the previous answer - which is what makes the raw layer
    auditable months later.
    """
    stamp = (fetched_at or datetime.now(UTC)).strftime("%Y%m%dT%H%M%SZ")
    day = (fetched_at or datetime.now(UTC)).strftime("%Y-%m-%d")
    return f"openaq/city={city}/date={day}/fetch_{stamp}.json"


def default_bucket(kind: str) -> str | None:
    """Bucket URI from the environment, e.g. RAW_BUCKET=gs://project-raw."""
    return os.getenv(f"{kind.upper()}_BUCKET")
