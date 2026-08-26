"""Merge/append-only guarantees (specs.md §2) and the date-filter guard."""

import datetime as dt

import pytest

from src.fetch_openaq import merge_rows
from src.openaq_client import DateFilterIgnored, OpenAQClient

META = {"city": "delhi", "sensor_id": 1}


def row(date, value):
    return {"date": date, "value": value, "min": None, "max": None, "count": 1}


def test_merge_into_empty():
    merged, added = merge_rows(None, [row("2025-01-01", 10)], META)
    assert added == 1
    assert merged["rows"] == [row("2025-01-01", 10)]


def test_merge_appends_without_dropping_history():
    existing = {"rows": [row("2025-01-01", 10)]}
    merged, added = merge_rows(existing, [row("2025-01-02", 20)], META)
    assert added == 1
    assert [r["date"] for r in merged["rows"]] == ["2025-01-01", "2025-01-02"]


def test_merge_is_idempotent():
    existing = {"rows": [row("2025-01-01", 10)]}
    merged, added = merge_rows(existing, [row("2025-01-01", 10)], META)
    assert added == 0
    assert len(merged["rows"]) == 1


def test_merge_updates_value_for_same_date():
    """Late-arriving readings revise a day in place rather than duplicating it."""
    existing = {"rows": [row("2025-01-01", 10)]}
    merged, _ = merge_rows(existing, [row("2025-01-01", 99)], META)
    assert len(merged["rows"]) == 1
    assert merged["rows"][0]["value"] == 99


def test_merge_output_is_date_sorted():
    existing = {"rows": [row("2025-01-05", 1)]}
    merged, _ = merge_rows(existing, [row("2025-01-02", 2), row("2025-01-09", 3)], META)
    assert [r["date"] for r in merged["rows"]] == ["2025-01-02", "2025-01-05", "2025-01-09"]


def test_empty_fetch_never_erases_existing_rows():
    """A dead sensor returning nothing must not blank out its history."""
    existing = {"rows": [row("2025-01-01", 10)]}
    merged, added = merge_rows(existing, [], META)
    assert added == 0
    assert merged["rows"] == [row("2025-01-01", 10)]


def test_date_filter_guard_fires_on_out_of_range_data(monkeypatch):
    """A silently-ignored date param must abort, not quietly pull full history."""
    client = OpenAQClient(api_key="x")
    monkeypatch.setattr(
        OpenAQClient,
        "_paginate",
        lambda self, path, params: iter(
            [{"period": {"datetimeFrom": {"utc": "2016-11-01T00:00:00Z"}}, "value": 1}]
        ),
    )
    with pytest.raises(DateFilterIgnored):
        client.daily_measurements(1, dt.date(2026, 8, 1), dt.date(2026, 8, 26))


def test_date_filter_guard_allows_one_day_timezone_slack(monkeypatch):
    client = OpenAQClient(api_key="x")
    monkeypatch.setattr(
        OpenAQClient,
        "_paginate",
        lambda self, path, params: iter(
            [{"period": {"datetimeFrom": {"utc": "2026-07-31T00:00:00Z"}}, "value": 1}]
        ),
    )
    rows = client.daily_measurements(1, dt.date(2026, 8, 1), dt.date(2026, 8, 26))
    assert rows[0]["date"] == "2026-07-31"
