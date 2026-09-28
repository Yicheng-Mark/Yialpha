"""Round-8 regression pins — S2 shard (dataflows vendors).

Each pin locks one defect found by the round-8 multi-shard adversarial sweep
so it cannot silently regress. All pins are offline: crafted payloads and
monkeypatched fetchers only, zero network.
"""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from yialpha.dataflows import sec_edgar, sec_ownership
from yialpha.dataflows.errors import NoMarketDataError

# --- _as_of_date: malformed curr_date fails closed (PIT upper bound) -------- #

def test_as_of_date_live_and_parseable():
    assert sec_ownership._as_of_date(None) == date.today()
    assert sec_ownership._as_of_date("") == date.today()
    assert sec_ownership._as_of_date("2024-06-01") == date(2024, 6, 1)


def test_as_of_date_malformed_fails_closed():
    # Pre-round-8 a non-empty unparseable curr_date degraded to date.today(),
    # widening a backtest run's PIT upper bound to today (post-curr_date
    # filings / FTD rows / 13F holdings passed the gates = lookahead).
    with pytest.raises(NoMarketDataError):
        sec_ownership._as_of_date("2024/6/1")
    with pytest.raises(NoMarketDataError):
        sec_ownership._as_of_date("garbage")


# --- get_ftd_data: look_back_days hard cap ---------------------------------- #

def test_ftd_lookback_capped(monkeypatch):
    seen = {}

    def fake_enum(start, end):
        seen["start"] = start
        return []

    monkeypatch.setattr(sec_ownership, "_enumerate_ftd_files", fake_enum)
    out = sec_ownership.get_ftd_data("GME", "2024-06-01", look_back_days=3000)
    # Header shows the effective window; the enumerate start proves the cap
    # applied before the window math (730d = 48 half-month ZIPs max).
    assert "last 730 days" in out
    assert seen["start"] == date(2024, 6, 1) - timedelta(days=730)


# --- get_fundamentals._latest: duration-aware pick --------------------------- #

def _facts_qtd_ytd_same_end() -> dict:
    # One 10-Q reports BOTH the 3-month quarter (300) and the 9-month YTD
    # cumulative (900) under the same period end; YTD listed FIRST so the
    # pre-fix array-order pick would have taken it.
    recs = [
        {"start": "2024-01-01", "end": "2024-09-30", "val": 900,
         "fy": 2024, "fp": "Q3", "form": "10-Q", "filed": "2024-11-01"},
        {"start": "2024-07-01", "end": "2024-09-30", "val": 300,
         "fy": 2024, "fp": "Q3", "form": "10-Q", "filed": "2024-11-01"},
    ]
    return {
        "entityName": "Test Inc",
        "facts": {"us-gaap": {"NetIncomeLoss": {"units": {"USD": recs}}}},
    }


def test_get_fundamentals_latest_prefers_quarter_over_ytd(monkeypatch):
    monkeypatch.setattr(sec_edgar, "_cik_for_ticker", lambda t: "1")
    monkeypatch.setattr(
        sec_edgar, "_fetch_company_facts", lambda cik: _facts_qtd_ytd_same_end()
    )
    out = sec_edgar.get_fundamentals("TST", "2024-12-01")
    assert "Net Income (latest): 300" in out
    assert "Net Income (latest): 900" not in out


# --- get_form4: recent-segment truncation honesty note ----------------------- #

def test_form4_flags_recent_segment_cap(monkeypatch):
    n = 1000
    subs = {"filings": {"recent": {
        "form": ["4"] * n,
        "accessionNumber": [f"acc{i}" for i in range(n)],
        "filingDate": ["2030-01-01"] * n,  # all beyond upper_d -> zero fetched
        "primaryDocument": ["x.xml"] * n,
    }}}
    monkeypatch.setattr(sec_ownership, "_cik_for_ticker", lambda t: "1")
    monkeypatch.setattr(sec_ownership, "_fetch_submissions", lambda cik: subs)
    out = sec_ownership.get_form4_insider_trading("TST", "2024-06-01")
    assert "No Form 4 transactions" in out
    assert "'recent' filings segment is at its ~1000 cap" in out


# =============================================================================
# S1 shard (dataflows core)
# =============================================================================

# --- fundamentals PIT clamp: echoed-today curr_date -------------------------- #

def test_fundamentals_tool_clamps_echoed_today_date():
    """Round-7 clamped indicator/weekly/validation tools; fundamentals was
    the gap — an LLM echoing a cached "Data retrieved on" header date (today)
    must be clamped to the pinned analysis date before any PIT filter."""
    from yialpha.agents.utils.fundamental_data_tools import _pit_curr_date
    from yialpha.dataflows.utils import set_analysis_date

    set_analysis_date("2024-06-03")
    try:
        assert _pit_curr_date("2026-09-28") == "2024-06-03"
        assert _pit_curr_date("2024-01-15") == "2024-01-15"
        # Review P2-1 (round-8 known-issues batch): an EMPTY/omitted date in
        # a PINNED run now falls back to the pinned date — "or empty for
        # live mode" must not disarm a historical gate. Live passthrough
        # (None in, None out) holds only without a pinned date, asserted
        # below.
        assert _pit_curr_date(None) == "2024-06-03"
        assert _pit_curr_date("") == "2024-06-03"
    finally:
        set_analysis_date(None)
    assert _pit_curr_date(None) is None  # live mode: no pinned date


def test_balance_sheet_vendor_clamps_curr_date(monkeypatch):
    from yialpha.dataflows import y_finance
    from yialpha.dataflows.utils import set_analysis_date

    seen = {}

    def fake_filter(frame, curr_date):
        seen["curr_date"] = curr_date
        return pd.DataFrame()  # empty -> the function's own NoMarketDataError

    monkeypatch.setattr(y_finance, "filter_financials_by_date", fake_filter)
    monkeypatch.setattr(
        y_finance, "_cached_yf_frame",
        lambda *a, **k: pd.DataFrame({"x": [1]}),
    )
    set_analysis_date("2024-06-03")
    try:
        with pytest.raises(NoMarketDataError):
            y_finance.get_balance_sheet("AAPL", "quarterly", "2026-09-28")
    finally:
        set_analysis_date(None)
    assert seen["curr_date"] == "2024-06-03"


def test_insider_transactions_refuses_historical_date():
    """The insider feed is a today-snapshot; a historical curr_date must
    refuse like the overview snapshot (dormant tool, gate-first so a future
    wiring cannot leak)."""
    from yialpha.dataflows import y_finance

    with pytest.raises(NoMarketDataError):
        y_finance.get_insider_transactions("AAPL", "2019-01-01")


# --- load_ohlcv: empty post-PIT-filter frame fails closed -------------------- #

def test_load_ohlcv_empty_after_pit_filter_raises(monkeypatch, tmp_path):
    """Pre-round-8 this silently returned an empty frame, producing a
    "successful" all-N/A indicator wall + a core success in the quality
    ledger (analysis date before the symbol's first cached bar)."""
    from yialpha.dataflows import stockstats_utils as ss

    f = tmp_path / "ohlc.csv"
    pd.DataFrame({
        "Date": ["2024-01-02", "2024-01-03"],
        "Open": [10.0, 10.5], "High": [11.0, 11.5], "Low": [9.5, 10.0],
        "Close": [10.5, 11.0], "Volume": [1000, 1100],
    }).to_csv(f, index=False)
    monkeypatch.setattr(ss, "_ohlcv_cache_path", lambda cfg, sym: str(f))
    monkeypatch.setattr(ss, "_needs_same_day_refresh", lambda *a: False)
    monkeypatch.setattr(ss, "_ohlcv_memo_get", lambda *a: None)
    with pytest.raises(NoMarketDataError, match="no OHLCV rows on or before"):
        ss.load_ohlcv("TST", "2024-01-01")


# --- resample_weekly: completed Friday bin survives a weekend live run ------- #

def _daily_frame(dates, close_base=100.0):
    n = len(dates)
    closes = close_base + np.arange(n, dtype=float)
    return pd.DataFrame({
        "Date": pd.to_datetime(dates),
        "Open": closes - 0.5, "High": closes + 1.0, "Low": closes - 1.0,
        "Close": closes, "Volume": [1000] * n,
    })


def test_resample_weekly_keeps_completed_week_on_weekend_live(monkeypatch):
    """Equities never print a Saturday row: the old ``label >= last_daily``
    heuristic deleted every completed Friday bin on weekend/Monday-premarket
    live runs. Mid-week forming bins must still drop."""
    from yialpha.dataflows import ohlcv_resample, utils as df_utils

    monkeypatch.setattr(df_utils, "is_historical_date", lambda d: False)
    daily = _daily_frame(["2026-09-21", "2026-09-22", "2026-09-25"])
    wk = ohlcv_resample.resample_weekly(daily, "2026-09-26")  # Saturday live
    assert len(wk) == 1 and str(wk["Date"].iloc[0].date()) == "2026-09-25"
    # Wednesday live: label (Fri) beyond last daily row -> forming -> dropped
    wk2 = ohlcv_resample.resample_weekly(daily, "2026-09-23")
    assert len(wk2) == 0


# --- atr_stop: forming-bar guard must read the Date COLUMN ------------------- #

def test_atr_forming_guard_reads_date_column():
    """load_ohlcv frames carry Date as a column with a RangeIndex;
    pd.Timestamp(<row number>) parsed as 1970-01-01, so the guard never fired
    for equity frames and today's intraday bar leaked into the ATR window."""
    from yialpha.risk.atr_stop import latest_atr_from_frame

    today = pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()
    dates = [today - pd.Timedelta(days=i) for i in range(19, -1, -1)]
    closes = 100.0 + np.arange(20, dtype=float)

    def frame(n):
        return pd.DataFrame({
            "Date": dates[:n],
            "Open": closes[:n] - 0.5, "High": closes[:n] + 1.0,
            "Low": closes[:n] - 1.0, "Close": closes[:n],
            "Volume": [1000] * n,
        })

    atr_full = latest_atr_from_frame(frame(20))[1]     # last row dated today
    atr_trimmed = latest_atr_from_frame(frame(19))[1]  # forming row removed
    assert atr_full == pytest.approx(atr_trimmed, rel=1e-9)
    # The close is still read from the FULL frame's tail (design: intraday
    # close is the live price), proving the guard dropped only the ATR row.
    assert latest_atr_from_frame(frame(20))[0] == pytest.approx(119.0)


# --- tool-layer PIT clamp coverage: a_share / eastmoney / macro / sec_ownership #

def _invoke_routed(monkeypatch, module, tool_name, payload):
    captured = {}

    def fake_route(method, *args):
        captured["method"] = method
        captured["args"] = args
        return "SENTINEL"

    monkeypatch.setattr(module, "route_to_vendor", fake_route)
    getattr(module, tool_name).invoke(payload)
    return captured


def test_thin_tool_layer_clamps_echoed_today_date(monkeypatch):
    """a_share_native / eastmoney / macro / sec_ownership tools were the last
    thin pass-throughs whose vendors PIT-filter by the passed curr_date
    verbatim — an LLM echoing a today-date would admit post-decision rows
    into a backtest. Each must clamp to the pinned analysis date, and stay
    untouched in live mode."""
    import yialpha.agents.utils.a_share_native_tools as a_share
    import yialpha.agents.utils.eastmoney_tools as em
    import yialpha.agents.utils.macro_data_tools as macro
    import yialpha.agents.utils.sec_ownership_tools as so
    from yialpha.dataflows.utils import set_analysis_date

    cases = [
        (a_share, "get_a_share_fundamentals_native",
         {"ticker": "600519.SS", "curr_date": "2026-09-28"}),
        (em, "get_margin_trading",
         {"ticker": "600519.SS", "curr_date": "2026-09-28"}),
        (macro, "get_macro_indicators",
         {"indicator": "cpi", "curr_date": "2026-09-28"}),
        (so, "get_form4_insider_trading",
         {"ticker": "AAPL", "curr_date": "2026-09-28"}),
    ]
    set_analysis_date("2024-06-03")
    try:
        for module, tool_name, payload in cases:
            cap = _invoke_routed(monkeypatch, module, tool_name, payload)
            assert cap["args"][1] == "2024-06-03", f"{tool_name} not clamped"
    finally:
        set_analysis_date(None)
    # Live mode (no pinned date): passthrough, byte-for-byte.
    module, tool_name, payload = cases[0]
    cap = _invoke_routed(monkeypatch, module, tool_name, payload)
    assert cap["args"][1] == "2026-09-28"
