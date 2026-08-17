"""Point-in-time (PIT) clamp for the OHLCV tool execution layer.

The stock/crypto klines tools are called by the LLM with
``(symbol, start_date, end_date)`` and carry no analysis-date parameter — the
LLM picks end_date from its prompt context. The real clock may be ahead of the
analysis date (a backtest for 2020 run today), so a vendor that honors whatever
window the LLM supplies would return rows after the analysis date, leaking the
future into the backtest.

``clamp_end_date`` + the run-pinned analysis-date ``ContextVar`` are the third
PIT guard (after ``is_filing_public`` and ``overview_would_leak_future``).
These tests pin the pure-function semantics and the vendor-level clamping
without any network.

Hermetic: no network. The clamp fires before the HTTP call, and the vendor
tests mock the fetch so the asserted window is the one actually requested.
"""

from __future__ import annotations

from datetime import UTC

from yiagents.dataflows.utils import (
    clamp_end_date,
    current_pit_end,
    get_analysis_date,
    set_analysis_date,
)


class TestClampEndDate:
    """Pure-function semantics of ``clamp_end_date``."""

    def test_clamps_future_end_to_analysis_date(self):
        assert clamp_end_date("2030-06-15", "2020-01-02") == "2020-01-02"

    def test_no_clamp_when_end_already_on_or_before(self):
        # Exactly the analysis date — inclusive, not clamped.
        assert clamp_end_date("2020-01-02", "2020-01-02") == "2020-01-02"
        # Before — not clamped.
        assert clamp_end_date("2019-12-31", "2020-01-02") == "2019-12-31"

    def test_live_mode_no_clamp_when_curr_date_none(self):
        # Live mode (no as-of constraint) must pass through unchanged.
        assert clamp_end_date("2030-01-01", None) == "2030-01-01"
        assert clamp_end_date("2030-01-01", "") == "2030-01-01"

    def test_no_clamp_when_end_date_none(self):
        assert clamp_end_date(None, "2020-01-02") is None

    def test_unparseable_analysis_date_passes_through(self):
        # Cannot prove a clamp is needed → fail open (only live is truly safe,
        # but a malformed analysis date is a caller bug, not a leak to hide).
        assert clamp_end_date("2030-01-01", "not-a-date") == "2030-01-01"

    def test_unparseable_end_date_passes_through(self):
        assert clamp_end_date("garbage", "2020-01-02") == "garbage"


class TestCurrentPitEnd:
    """The ContextVar-backed convenience used by vendors."""

    def test_live_when_no_analysis_date_pinned(self):
        set_analysis_date(None)
        assert get_analysis_date() is None
        assert current_pit_end("2030-01-01") == "2030-01-01"

    def test_clamps_to_pinned_analysis_date(self):
        set_analysis_date("2020-01-02")
        try:
            assert current_pit_end("2030-01-01") == "2020-01-02"
            assert current_pit_end("2019-12-31") == "2019-12-31"
        finally:
            set_analysis_date(None)

    def test_empty_string_clears(self):
        set_analysis_date("2020-01-02")
        set_analysis_date("")  # empty = live
        try:
            assert get_analysis_date() is None
            assert current_pit_end("2030-01-01") == "2030-01-01"
        finally:
            set_analysis_date(None)


class TestVendorClamp:
    """The vendor fetch window is clamped before the HTTP call."""

    def test_yfinance_get_YFin_data_online_clamps_end(self, monkeypatch):
        """get_YFin_data_online must clamp end_date to the pinned analysis date."""
        set_analysis_date("2020-01-02")
        captured = {}

        def fake_history(self, start, end):
            captured["start"] = start
            captured["end"] = end
            import pandas as pd
            # Return one row so the stale-check + formatting path runs.
            return pd.DataFrame(
                {"Open": [300], "High": [310], "Low": [295], "Close": [305],
                 "Adj Close": [305], "Volume": [1000]},
                index=pd.Index([pd.Timestamp("2020-01-02")], name="Date"),
            )

        import yfinance as yf
        monkeypatch.setattr(yf.Ticker, "history", fake_history, raising=False)

        try:
            from yiagents.dataflows.y_finance import get_YFin_data_online
            # LLM asks for 2030 — must be clamped to 2020-01-02.
            result = get_YFin_data_online("AAPL", "2019-12-01", "2030-06-15")
            # get_YFin_data_online adds +1 day for yfinance's exclusive end, so
            # the inclusive end sent is 2020-01-03 (the day after the clamped
            # 2020-01-02). The key assertion: NO 2030 leaked into the request.
            assert captured["end"] == "2020-01-03", captured["end"]
            assert "2030" not in captured["end"]
            assert "AAPL" in result
        finally:
            set_analysis_date(None)

    def test_binance_klines_clamps_end(self, monkeypatch):
        """get_binance_klines must clamp end_date to the pinned analysis date."""
        set_analysis_date("2020-01-02")
        captured = {}

        import yiagents.dataflows.binance as binance_mod

        def fake_paginate(endpoint, params, limit, key_fn, start_ms, end_ms, *a, **kw):
            captured["start_ms"] = start_ms
            captured["end_ms"] = end_ms
            # Return one kline so the no-data check passes.
            return [[1577923200000, "7000", "7100", "6900", "7050", "100"]]

        monkeypatch.setattr(binance_mod, "_paginate_history", fake_paginate)

        try:
            # LLM asks for 2030 — must be clamped to 2020-01-02.
            result = binance_mod.get_binance_klines("BTCUSDT", "2019-12-01", "2030-06-15")
            # end_ms should be end-of-day 2020-01-02, NOT 2030.
            from datetime import datetime
            clamped_end = datetime.strptime("2020-01-02", "%Y-%m-%d").replace(
                tzinfo=UTC
            )
            expected_end_ms = int((clamped_end.timestamp() + 86399) * 1000)
            assert captured["end_ms"] == expected_end_ms
            assert "BTCUSDT" in result or "btcusdt" in result.lower()
        finally:
            set_analysis_date(None)
