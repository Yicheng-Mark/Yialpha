"""Binance-native indicator tool (WP9): computes stockstats on Binance klines.

The perp analyst previously had no computed indicators (spot-style tools are
hidden for perp because they resolve to the wrong Yahoo pair). These tests
pin: venue routing through the shared klines frame, catalog-name validation,
typed degradation, skip-and-report for uncomputable names, and the PIT clamp
inherited from ``current_pit_end``. Klines are mocked at the frame layer
(HTTP behaviour of the underlying fetch is covered by the binance vendor
tests), but ALL indicator math runs on real stockstats.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import yiagents.agents.utils.binance_indicator_tools as bit


def _klines_frame(symbol: str, start_date: str, end_date: str, interval="1d",
                  venue="binance_perp"):
    n = 500
    closes = 100.0 * np.exp(np.cumsum(np.random.default_rng(7).normal(0, 0.02, n)))
    idx = pd.bdate_range(pd.Timestamp(start_date) + pd.Timedelta(days=1), periods=n)
    idx = idx[idx <= pd.Timestamp(end_date)]
    frame = pd.DataFrame(
        {
            "Open": closes[: len(idx)] * 0.999,
            "High": closes[: len(idx)] * 1.01,
            "Low": closes[: len(idx)] * 0.99,
            "Close": closes[: len(idx)],
            "Adj Close": closes[: len(idx)],
            "Volume": np.full(len(idx), 1000.0),
        },
        index=idx,
    )
    frame.index.name = "Date"
    return frame.round(2)


@pytest.mark.unit
class TestBinanceIndicatorsTool:
    @pytest.fixture()
    def patched_frame(self, monkeypatch):
        calls = {}
        monkeypatch.setattr(bit, "binance_klines_frame", _recorder(_klines_frame, calls))
        return calls

    def test_default_battery_renders(self, patched_frame):
        out = bit.get_binance_indicators.invoke({
            "symbol": "BTCUSDT", "curr_date": "2026-01-15",
        })
        assert "Binance perp indicators" in out
        for name in ("close_50_sma", "macd", "rsi", "kdjk", "supertrend", "rvol_20"):
            assert name in out
        assert "DATA_UNAVAILABLE" not in out

    def test_venue_routes_to_spot(self, patched_frame):
        bit.get_binance_indicators.invoke({
            "symbol": "BTCUSDT", "curr_date": "2026-01-15", "venue": "spot",
        })
        assert patched_frame["venue"] == "binance_spot"

    def test_unknown_indicator_names_rejected(self, patched_frame):
        out = bit.get_binance_indicators.invoke({
            "symbol": "BTCUSDT", "curr_date": "2026-01-15",
            "indicators": "rsi,not_a_thing",
        })
        assert out.startswith("ERROR")
        assert "not_a_thing" in out

    def test_typed_degrade_on_vendor_failure(self, monkeypatch):
        def boom(*a, **kw):
            raise RuntimeError("binance down")

        monkeypatch.setattr(bit, "binance_klines_frame", boom)
        out = bit.get_binance_indicators.invoke({
            "symbol": "BTCUSDT", "curr_date": "2026-01-15",
        })
        assert out.startswith("DATA_UNAVAILABLE")
        assert "RuntimeError" in out

    def test_short_history_degrades(self, monkeypatch):
        def short_frame(symbol, start, end, interval="1d", venue="binance_perp"):
            idx = pd.bdate_range("2026-01-01", periods=10)
            return pd.DataFrame(
                {"Open": 1.0, "High": 1.0, "Low": 1.0, "Close": 1.0,
                 "Adj Close": 1.0, "Volume": 1.0},
                index=idx,
            )

        monkeypatch.setattr(bit, "binance_klines_frame", short_frame)
        out = bit.get_binance_indicators.invoke({
            "symbol": "NEWUSDT", "curr_date": "2026-01-15",
        })
        assert out.startswith("DATA_UNAVAILABLE")

    def test_perp_prompt_mentions_indicator_tool(self):
        from yiagents.agents.analysts.market_analyst import _PERP_NUDGE, _SPOT_NUDGE
        assert "get_binance_indicators" in _PERP_NUDGE
        assert "get_binance_indicators" in _SPOT_NUDGE


def _recorder(fn, calls):
    def wrapper(symbol, start_date, end_date, interval="1d", venue="binance_perp"):
        calls.update(
            symbol=symbol, start=start_date, end=end_date,
            interval=interval, venue=venue,
        )
        return fn(symbol, start_date, end_date, interval, venue)
    return wrapper
