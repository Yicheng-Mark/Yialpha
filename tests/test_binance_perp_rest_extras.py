"""REST perp extras: ``period`` passthrough, index-price klines, depth snapshot.

Extends the window contract pinned in test_binance_futures_data_window.py:

* ``period`` (5m..12h) flows into the ``/futures/data/*`` request params and
  scales the row ``limit`` with rows-per-day (an intraday window must not be
  silently truncated to a daily-sized limit);
* intraday rows are timestamped ``YYYY-MM-DD HH:MM`` and the belt-and-
  suspenders PIT trim compares TIMESTAMP MS — a string compare would drop
  every row of the end day ("2026-08-15 13:00" > "2026-08-15");
* ``price_type="index"`` routes to ``/fapi/v1/indexPriceKlines``;
* ``/fapi/v1/depth`` parses into a ladder + mid/spread/imbalance header.

Hermetic: ``_http_get`` is monkeypatched.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from yiagents.dataflows import binance as bn
from yiagents.dataflows.errors import NoMarketDataError


def _days_ago(n: int) -> str:
    return (datetime.now(UTC) - timedelta(days=n)).strftime("%Y-%m-%d")


def _ts_at(days_ago: int, hour: int, minute: int = 0) -> int:
    dt = datetime.now(UTC) - timedelta(days=days_ago)
    dt = dt.replace(hour=hour, minute=minute, second=0, microsecond=0)
    return int(dt.timestamp() * 1000)


class _Capture:
    """Fake _http_get recording every (path, params); answers per-path rows."""

    def __init__(self, rows):
        self.calls: list[tuple[str, dict]] = []
        self._rows = rows

    def __call__(self, path, params, symbol, canonical, **kwargs):
        self.calls.append((path, dict(params)))
        if path == "/fapi/v1/openInterest":
            return {"openInterest": "1234"}
        rows = self._rows.get(path)
        if rows is None:
            return []
        return [
            {
                "timestamp": _ts_at(d, h),
                "sumOpenInterest": "1000",
                "sumOpenInterestValue": "2000",
                "longAccount": "0.6",
                "longShortRatio": "1.5",
                "shortAccount": "0.4",
                "buySellRatio": "1.2",
                "buyVol": "10",
                "sellVol": "8",
                "basis": "5",
                "futuresPrice": "100",
                "indexPrice": "95",
                "basisRate": "0.05",
            }
            for d, h in rows
        ]


# ---- period passthrough -------------------------------------------------------


@pytest.mark.unit
def test_period_flows_into_params_and_scales_limit(monkeypatch):
    cap = _Capture({"/futures/data/openInterestHist": []})
    monkeypatch.setattr(bn, "_http_get", cap)
    with pytest.raises(NoMarketDataError):
        # Empty rows -> no-data raise, but the REQUEST params are what matters.
        bn.get_binance_open_interest("BTCUSDT", 2, period="1h")
    _path, params = cap.calls[0]
    assert params["period"] == "1h"
    assert params["limit"] == 2 * 24  # 2 days of hourly rows


@pytest.mark.unit
def test_lsr_intraday_limit_scales(monkeypatch):
    cap = _Capture({"/futures/data/topLongShortAccountRatio": []})
    monkeypatch.setattr(bn, "_http_get", cap)
    with pytest.raises(NoMarketDataError):
        bn.get_binance_long_short_ratio("BTCUSDT", 1, period="15m")
    _path, params = cap.calls[0]
    assert params["period"] == "15m"
    assert params["limit"] == 96  # 1 day of 15m rows


@pytest.mark.unit
def test_daily_default_remains_byte_equivalent(monkeypatch):
    cap = _Capture({"/futures/data/openInterestHist": [(1, 12)]})
    monkeypatch.setattr(bn, "_http_get", cap)
    out = bn.get_binance_open_interest("BTCUSDT", 5)
    _path, params = cap.calls[0]
    assert params["period"] == "1d"
    assert params["limit"] == 5
    assert "latest" in out  # live snapshot still appended for daily windows


@pytest.mark.unit
def test_windowed_intraday_limit_capped_at_endpoint_ceiling(monkeypatch):
    cap = _Capture({"/futures/data/openInterestHist": []})
    monkeypatch.setattr(bn, "_http_get", cap)
    with pytest.raises(NoMarketDataError):
        bn.get_binance_open_interest(
            "BTCUSDT", 30,
            start_date=_days_ago(10), end_date=_days_ago(1), period="5m",
        )
    _path, params = cap.calls[0]
    assert params["period"] == "5m"
    assert params["limit"] == bn._FUTURES_DATA_LIMIT_CAP  # 10 days x 288 capped
    # The request must be END-anchored: these endpoints serve rows ascending
    # from startTime, so the window's own start would return the OLDEST 500
    # rows and silently drop the decision-critical tail. Anchor = one full
    # 500-row page ending at the last in-window row boundary.
    period_ms = 86_400_000 // 288
    last_row_ms = (params["endTime"] // period_ms) * period_ms
    expected_start = last_row_ms - (bn._FUTURES_DATA_LIMIT_CAP - 1) * period_ms
    assert params["startTime"] == expected_start
    window_start_ms = int(
        datetime.strptime(_days_ago(10), "%Y-%m-%d")
        .replace(tzinfo=UTC)
        .timestamp() * 1000
    )
    assert params["startTime"] > window_start_ms  # head genuinely dropped


@pytest.mark.unit
def test_windowed_intraday_truncation_disclosed_in_header(monkeypatch):
    """An over-cap window must not lie by omission: the header discloses the
    truncation (kept tail, dropped head, archive-tool pointer)."""
    cap = _Capture({"/futures/data/openInterestHist": [(1, 12), (1, 13)]})
    monkeypatch.setattr(bn, "_http_get", cap)
    out = bn.get_binance_open_interest(
        "BTCUSDT", 30,
        start_date=_days_ago(10), end_date=_days_ago(1), period="5m",
    )
    assert "truncated to the last 500 rows" in out
    assert "day(s) dropped from the head" in out
    assert "get_binance_vision_metrics" in out
    # The end-day rows survive: the kept coverage always includes the tail
    # (intraday rows are timestamped "YYYY-MM-DD HH:MM").
    assert f"{_days_ago(1)} 12:00,1000" in out


@pytest.mark.unit
def test_invalid_period_rejected_before_any_request(monkeypatch):
    cap = _Capture({})
    monkeypatch.setattr(bn, "_http_get", cap)
    for fn in (
        bn.get_binance_open_interest,
        bn.get_binance_long_short_ratio,
        bn.get_binance_taker_buy_sell,
        bn.get_binance_basis,
    ):
        with pytest.raises(ValueError, match="period"):
            fn("BTCUSDT", 7, period="3h")
    assert cap.calls == []


@pytest.mark.unit
def test_intraday_rows_keep_end_day_rows_and_trim_future_ms(monkeypatch):
    """The PIT trim must compare timestamps, not date strings: rows at 08:00
    and 09:00 of the END day stay; a row beyond the end-of-day is dropped."""
    cap = _Capture({
        "/futures/data/openInterestHist": [
            (1, 8), (1, 9), (0, 9),  # end day rows + a beyond-end row (today)
        ],
    })
    monkeypatch.setattr(bn, "_http_get", cap)
    out = bn.get_binance_open_interest(
        "BTCUSDT", 7, start_date=_days_ago(5), end_date=_days_ago(1), period="1h",
    )
    lines = [ln for ln in out.splitlines() if ln and not ln.startswith("#")]
    header_row = lines[0]
    times = [ln.split(",")[0] for ln in lines[1:]]
    assert header_row.startswith("time")
    assert all(":" in t for t in times)  # HH:MM granularity
    assert len(times) == 2  # the (0, 9) today-row was trimmed, end-day rows kept
    assert all(t.startswith(_days_ago(1)) for t in times)
    assert "latest" not in times  # intraday windows append no live row


# ---- index-price klines -------------------------------------------------------


def _kline_rows(days_ago_list):
    out = []
    for d in days_ago_list:
        dt = datetime.now(UTC) - timedelta(days=d)
        dt = dt.replace(hour=0, minute=0, second=0, microsecond=0)
        ms = int(dt.timestamp() * 1000)
        out.append([ms, "100", "101", "99", "100.5", "0", ms, "0", 0, "0", "0", "0"])
    return out


class _PathRecorder:
    def __init__(self, rows):
        self.rows = rows
        self.paths: list[str] = []

    def __call__(self, path, params, symbol, canonical, **kwargs):
        self.paths.append(path)
        return [r[:] for r in self.rows]


@pytest.mark.unit
def test_index_price_routes_to_index_endpoint(monkeypatch):
    rec = _PathRecorder(_kline_rows(range(1, 6)))
    monkeypatch.setattr(bn, "_http_get", rec)
    bn.binance_klines_frame(
        "BTCUSDT", _days_ago(6), _days_ago(1), price_type="index",
    )
    assert rec.paths and all(p == "/fapi/v1/indexPriceKlines" for p in rec.paths)


@pytest.mark.unit
def test_spot_index_combination_rejected(monkeypatch):
    rec = _PathRecorder([])
    monkeypatch.setattr(bn, "_http_get", rec)
    with pytest.raises(ValueError, match="perp-only"):
        bn.binance_klines_frame(
            "BTCUSDT", _days_ago(6), _days_ago(1),
            venue="binance_spot", price_type="index",
        )
    assert rec.paths == []


@pytest.mark.unit
def test_invalid_price_type_rejected(monkeypatch):
    monkeypatch.setattr(bn, "_http_get", _PathRecorder([]))
    with pytest.raises(ValueError, match="price_type"):
        bn.binance_klines_frame(
            "BTCUSDT", _days_ago(6), _days_ago(1), price_type="oracle",
        )


# ---- depth snapshot -----------------------------------------------------------


def _depth_payload():
    return {
        "bids": [["100.0", "2.0"], ["99.5", "3.0"]],
        "asks": [["100.2", "1.0"], ["100.3", "4.0"]],
    }


@pytest.mark.unit
def test_depth_snapshot_ladder_and_header(monkeypatch):
    monkeypatch.setattr(
        bn, "_http_get", lambda *a, **k: _depth_payload(),
    )
    out = bn.get_binance_depth_snapshot("BTCUSDT", 5)
    lines = [ln for ln in out.splitlines() if ln and not ln.startswith("#")]
    assert lines[0].startswith("level,bid_price")
    assert lines[1].startswith("1,100.0,2.0,100.2,1.0")
    assert "mid=100.1" in out
    assert "imbalance=" in out


@pytest.mark.unit
def test_depth_snapshot_invalid_limit_rejected(monkeypatch):
    called = []
    monkeypatch.setattr(bn, "_http_get", lambda *a, **k: called.append(1))
    with pytest.raises(ValueError, match="limit"):
        bn.get_binance_depth_snapshot("BTCUSDT", 33)
    assert called == []


@pytest.mark.unit
def test_depth_snapshot_empty_book_raises(monkeypatch):
    monkeypatch.setattr(bn, "_http_get", lambda *a, **k: {"bids": [], "asks": []})
    with pytest.raises(NoMarketDataError, match="no levels"):
        bn.get_binance_depth_snapshot("BTCUSDT", 5)
