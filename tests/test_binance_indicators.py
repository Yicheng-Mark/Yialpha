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

import contextlib

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
        # 2026-08-16: the spot run binds the spot-default variant so an
        # omitted venue arg can never price the perpetual.
        assert "get_binance_spot_indicators" in _SPOT_NUDGE

    def test_spot_tool_defaults_to_spot_venue(self, patched_frame):
        bit.get_binance_spot_indicators.invoke({
            "symbol": "BTCUSDT", "curr_date": "2026-01-15",
        })
        assert patched_frame["venue"] == "binance_spot"

    def test_low_price_contract_not_zeroed(self, monkeypatch):
        """PEPE-class closes (~1e-5) must survive the table, not render 0.00.

        The data layer no longer rounds OHLC (P0 2026-08-16); this pins the
        display twin — the markdown table's adaptive decimals — so a low-price
        contract's close and indicator values stay legible.
        """
        n = 400
        closes = 1.23e-5 * np.exp(np.cumsum(np.random.default_rng(3).normal(0, 0.02, n)))
        idx = pd.bdate_range("2024-06-01", periods=n)
        frame = pd.DataFrame(
            {
                "Open": closes * 0.999, "High": closes * 1.01, "Low": closes * 0.99,
                "Close": closes, "Adj Close": closes,
                "Volume": np.full(n, 1e11),
            },
            index=idx,
        )
        frame.index.name = "Date"
        monkeypatch.setattr(bit, "binance_klines_frame", lambda *a, **k: frame)
        out = bit.get_binance_indicators.invoke({
            "symbol": "1000PEPEUSDT", "curr_date": "2025-08-05",
            "indicators": "close_50_sma,rsi",
        })
        assert "DATA_UNAVAILABLE" not in out and not out.startswith("ERROR")
        # Every Close cell in the table must carry the 1e-5-scale value, not
        # a zeroed "0.00" (substring checks don't work — "0.0000123" starts
        # with "0.00" — so parse the cells).
        close_cells = []
        for line in out.splitlines():
            if not line.startswith("|"):
                continue
            cells = [c.strip() for c in line.strip("|").split("|")]
            if len(cells) >= 3 and cells[0][:2].isdigit() and cells[1]:
                with contextlib.suppress(ValueError):
                    close_cells.append(float(cells[1].replace(",", "")))
        assert close_cells, "no data rows found in table"
        assert all(v > 1e-7 for v in close_cells), close_cells[:5]

    def test_rvol_annualizes_with_365_on_crypto_candles(self, monkeypatch):
        """Crypto candles are a 24/7 daily series: rvol_20 must annualize
        with 365, not the 252 equity default (P1 fix, 2026-08-16).

        The rendered Latest rvol_20 must equal the 365-annualized value and
        differ from the 252 one by exactly sqrt(365/252) ≈ 1.204.
        """
        import re

        from yiagents.dataflows.vol_estimators import (
            CRYPTO_TRADING_DAYS_PER_YEAR,
            TRADING_DAYS_PER_YEAR,
            close_to_close_vol,
        )

        n = 400
        rng = np.random.default_rng(42)
        closes = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.02, n)))
        idx = pd.bdate_range("2024-06-01", periods=n)
        frame = pd.DataFrame(
            {
                "Open": closes * 0.999, "High": closes * 1.01, "Low": closes * 0.99,
                "Close": closes, "Adj Close": closes,
                "Volume": np.full(n, 1000.0),
            },
            index=idx,
        )
        frame.index.name = "Date"
        monkeypatch.setattr(bit, "binance_klines_frame", lambda *a, **k: frame)

        out = bit.get_binance_indicators.invoke({
            "symbol": "BTCUSDT", "curr_date": "2025-08-05",
            "indicators": "rvol_20",
        })
        assert "DATA_UNAVAILABLE" not in out and not out.startswith("ERROR")

        m = re.search(r"Latest \([^)]*\): rvol_20=([0-9]*\.?[0-9]+)", out)
        assert m, out[-300:]
        rendered = float(m.group(1))

        reset = frame.reset_index()
        expected_365 = float(
            close_to_close_vol(
                reset, window=20, periods_per_year=CRYPTO_TRADING_DAYS_PER_YEAR
            ).iloc[-1]
        )
        expected_252 = float(
            close_to_close_vol(
                reset, window=20, periods_per_year=TRADING_DAYS_PER_YEAR
            ).iloc[-1]
        )
        assert rendered == pytest.approx(expected_365, rel=5e-3)
        assert rendered != pytest.approx(expected_252, rel=5e-3)
        assert rendered / expected_252 == pytest.approx(
            (365.0 / 252.0) ** 0.5, rel=5e-3
        )


def _recorder(fn, calls):
    def wrapper(symbol, start_date, end_date, interval="1d", venue="binance_perp"):
        calls.update(
            symbol=symbol, start=start_date, end=end_date,
            interval=interval, venue=venue,
        )
        return fn(symbol, start_date, end_date, interval, venue)
    return wrapper
