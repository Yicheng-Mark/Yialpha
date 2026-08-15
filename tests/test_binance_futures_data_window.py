"""Binance /futures/data/* historical windows (workflow B11).

``openInterestHist`` and the *Ratio/basis endpoints DO accept
``startTime``/``endTime``; the previous signature exposed only
``look_back_days`` (most-recent-N live form, documented as backtest-unsafe).
These tests pin the new window plumbing:

* explicit dates pass through as startTime/endTime with ``end_date`` clamped
  to the run's pinned analysis date (PIT semantics, same guard as klines);
* the endpoint's 30-day retention horizon is enforced fail-closed (a window
  ending further back raises NoMarketDataError rather than silently serving
  the newest — future-for-a-backtest — rows);
* the live/"latest" snapshot row is appended only when the window reaches the
  present, and vendor rows beyond the requested end are trimmed client-side.

Hermetic: no network — ``_http_get`` is monkeypatched and the asserted params
are the ones actually requested.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from yiagents.dataflows import binance as bn
from yiagents.dataflows.errors import NoMarketDataError
from yiagents.dataflows.utils import set_analysis_date


def _days_ago(n: int) -> str:
    return (datetime.now(timezone.utc) - timedelta(days=n)).strftime("%Y-%m-%d")


def _ts(days_ago: int) -> int:
    """Epoch milliseconds for noon UTC, ``days_ago`` days back."""
    dt = datetime.now(timezone.utc) - timedelta(days=days_ago)
    dt = dt.replace(hour=12, minute=0, second=0, microsecond=0)
    return int(dt.timestamp() * 1000)


def _oi_rows(*days_ago: int) -> list[dict]:
    return [
        {"timestamp": _ts(d), "sumOpenInterest": "1000", "sumOpenInterestValue": "2000"}
        for d in days_ago
    ]


class _Capture:
    """Fake _http_get recording every (path, params) and answering OI rows."""

    def __init__(self, rows_by_days_ago):
        self.calls: list[tuple[str, dict]] = []
        self._rows = rows_by_days_ago

    def __call__(self, path, params, symbol, canonical):
        self.calls.append((path, dict(params)))
        if path == "/fapi/v1/openInterest":
            return {"openInterest": "1234"}
        return [
            {"timestamp": _ts(d), "sumOpenInterest": "1000",
             "sumOpenInterestValue": "2000"}
            for d in self._rows
        ]


@pytest.mark.unit
def test_unwindowed_call_keeps_live_limit_form(monkeypatch):
    """No explicit dates: unchanged most-recent-``limit`` behaviour — params
    carry ``limit`` and NO startTime/endTime."""
    cap = _Capture([0, 1])
    monkeypatch.setattr(bn, "_http_get", cap)
    out = bn.get_binance_open_interest("BTCUSDT", 5)
    path, params = cap.calls[0]
    assert path == "/futures/data/openInterestHist"
    assert params["limit"] == 5
    assert "startTime" not in params and "endTime" not in params
    assert "latest" in out  # live snapshot row still appended


@pytest.mark.unit
def test_explicit_window_passes_start_end_and_notes_retention(monkeypatch):
    cap = _Capture([2, 3])
    monkeypatch.setattr(bn, "_http_get", cap)
    out = bn.get_binance_open_interest(
        "BTCUSDT", 7, start_date=_days_ago(5), end_date=_days_ago(2),
    )
    _, params = cap.calls[0]
    assert "limit" not in params or params["limit"] <= bn._FUTURES_DATA_LIMIT_CAP
    assert params["startTime"] < params["endTime"]
    # The output must disclose the window AND the 30-day retention limit.
    assert f"window through {_days_ago(2)}" in out
    assert "30 days only" in out


@pytest.mark.unit
def test_ratio_endpoints_pass_the_window_through(monkeypatch):
    """The LSR series endpoints share the same window plumbing."""
    cap = _Capture([1, 2])
    monkeypatch.setattr(bn, "_http_get", cap)
    bn.get_binance_long_short_ratio(
        "BTCUSDT", 7, start_date=_days_ago(4), end_date=_days_ago(1),
    )
    for _path, params in cap.calls:
        assert params["startTime"] < params["endTime"]
        assert "limit" in params


@pytest.mark.unit
def test_end_date_clamped_to_pinned_analysis_date(monkeypatch):
    """PIT: with a pinned analysis date, an LLM-supplied end_date beyond it is
    clamped before the request (same guard as the klines endpoints)."""
    cap = _Capture([1, 2])
    monkeypatch.setattr(bn, "_http_get", cap)
    set_analysis_date(_days_ago(1))
    try:
        bn.get_binance_open_interest(
            "BTCUSDT", 7, start_date=_days_ago(5), end_date="2030-01-01",
        )
    finally:
        set_analysis_date(None)
    _, params = cap.calls[0]
    clamped_end_ms = int(
        datetime.strptime(_days_ago(1), "%Y-%m-%d")
        .replace(tzinfo=timezone.utc)
        .timestamp() * 1000
    ) + 86399 * 1000
    assert params["endTime"] <= clamped_end_ms


@pytest.mark.unit
def test_window_older_than_retention_raises(monkeypatch):
    """/futures/data/* retains only the last 30 days: a window ending further
    back must raise (fail-closed) — falling back to the newest rows would hand
    a backtest future positioning data."""
    cap = _Capture([0])
    monkeypatch.setattr(bn, "_http_get", cap)
    with pytest.raises(NoMarketDataError, match="retain"):
        bn.get_binance_open_interest(
            "BTCUSDT", 7, start_date=_days_ago(45), end_date=_days_ago(40),
        )
    assert cap.calls == []  # refused BEFORE any request


@pytest.mark.unit
def test_past_window_neither_appends_live_row_nor_leaks_future_rows(monkeypatch):
    """A window ending in the past (but within retention): (a) no 'latest'
    live row — it would be future data for that decision point; (b) rows a
    non-compliant vendor returns past the requested end are trimmed."""
    end = _days_ago(1)
    cap = _Capture([0, 1, 2])  # includes a row NEWER than the window end
    monkeypatch.setattr(bn, "_http_get", cap)
    out = bn.get_binance_open_interest(
        "BTCUSDT", 7, start_date=_days_ago(4), end_date=end,
    )
    assert "latest" not in out
    # Rows on/before the window end survive; the day-0 row (after end) was
    # trimmed client-side even though the fake vendor returned it.
    assert f"{_days_ago(1)},1000" in out
    assert f"{_days_ago(2)},1000" in out
    today_row = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    assert f"{today_row},1000" not in out
