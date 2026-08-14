"""Unit tests for ``yiagents.dataflows.y_finance``.

Covers: get_YFin_data_online error/tz paths, fundamentals (PIT + stub-info),
the three financial statements (balance_sheet / cashflow / income_statement),
insider transactions (empty = normal), and the indicator-window functions.
Focus on the fail-open contract: errors must propagate (raise), not be
swallowed into prose.

All yfinance access is mocked via ``yf.Ticker`` so no network is needed.
"""

import pandas as pd
import pytest

import yiagents.dataflows.y_finance as y_finance
from yiagents.dataflows.symbol_utils import NoMarketDataError

# ---------------------------------------------------------------------------
# Test fixtures / helpers
# ---------------------------------------------------------------------------

def _ohlcv_frame(date="2026-06-10", tz=None):
    """Build a 1-row OHLCV DataFrame with a DatetimeIndex."""
    idx = pd.DatetimeIndex([pd.Timestamp(date)], name="Date")
    if tz is not None:
        idx = idx.tz_localize(tz)
    return pd.DataFrame(
        {"Open": [330.0], "High": [332.0], "Low": [328.0],
         "Close": [330.58], "Adj Close": [330.58], "Volume": [1_000_000]},
        index=idx,
    )


class _DummyTicker:
    """Minimal yf.Ticker mock: set instance attrs to control .history / .info etc.

    ``info`` is a property delegating to ``self._info`` so subclasses can override
    it without clashing with the __init__ attribute assignment.
    """

    def __init__(self, symbol):
        self.symbol = symbol
        self.history_return = _ohlcv_frame("2026-06-10")
        self._info = {}
        self.quarterly_balance_sheet = pd.DataFrame()
        self.balance_sheet = pd.DataFrame()
        self.quarterly_cashflow = pd.DataFrame()
        self.cashflow = pd.DataFrame()
        self.quarterly_income_stmt = pd.DataFrame()
        self.income_stmt = pd.DataFrame()
        self.insider_transactions = pd.DataFrame()

    @property
    def info(self):
        return self._info

    @info.setter
    def info(self, value):
        self._info = value

    def history(self, start, end):
        return self.history_return


# ---------------------------------------------------------------------------
# get_YFin_data_online
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_empty_history_raises_no_market_data(monkeypatch):
    """An empty history frame raises NoMarketDataError (not prose)."""
    dummy = _DummyTicker("AAPL")
    dummy.history_return = pd.DataFrame(columns=["Open", "Close"])
    monkeypatch.setattr(y_finance.yf, "Ticker", lambda s: dummy)
    with pytest.raises(NoMarketDataError):
        y_finance.get_YFin_data_online("AAPL", "2026-06-01", "2026-06-11")


@pytest.mark.unit
def test_tz_aware_index_stripped(monkeypatch):
    """A timezone-aware DatetimeIndex gets tz_localize(None)."""
    dummy = _DummyTicker("AAPL")
    dummy.history_return = _ohlcv_frame("2026-06-10", tz="UTC")
    monkeypatch.setattr(y_finance.yf, "Ticker", lambda s: dummy)
    result = y_finance.get_YFin_data_online("AAPL", "2026-06-01", "2026-06-11")
    assert "Stock data for" in result
    # The CSV portion should contain the date without tz suffix
    assert "2026-06-10" in result


@pytest.mark.unit
def test_fresh_ohlcv_cache_serves_without_yahoo_call(monkeypatch):
    """The indicator pipeline's cache serves the window -> no Ticker call.

    One run previously hit Yahoo twice per symbol (indicator pipeline +
    stock-data tool); the opportunistic cache read collapses that.
    """
    cached = pd.DataFrame({
        # First row predates the requested start so the cache COVERS the window.
        "Date": pd.to_datetime(["2026-05-30", "2026-06-09", "2026-06-10"]),
        "Open": [329.0, 330.0, 331.0], "High": [330.0, 332.0, 333.0],
        "Low": [327.0, 328.0, 329.0], "Close": [329.9, 330.58, 331.2],
        "Volume": [900_000, 1_000_000, 1_100_000],
    })

    def boom(s):
        raise AssertionError("yf.Ticker must not be reached on a cache hit")

    monkeypatch.setattr(y_finance.yf, "Ticker", boom)
    monkeypatch.setattr(
        y_finance, "read_cached_ohlcv", lambda sym, curr: cached.copy()
    )
    out = y_finance.get_YFin_data_online("AAPL", "2026-06-01", "2026-06-10")
    assert "Stock data for AAPL" in out
    assert "2026-06-10" in out


@pytest.mark.unit
def test_cache_not_covering_window_falls_back_online(monkeypatch):
    """start_date older than the cache's first row -> the online path runs."""
    dummy = _DummyTicker("AAPL")
    monkeypatch.setattr(y_finance.yf, "Ticker", lambda s: dummy)
    cached = pd.DataFrame({
        "Date": pd.to_datetime(["2026-06-09", "2026-06-10"]),
        "Open": [330.0, 331.0], "High": [332.0, 333.0],
        "Low": [328.0, 329.0], "Close": [330.58, 331.2],
        "Volume": [1_000_000, 1_100_000],
    })
    monkeypatch.setattr(
        y_finance, "read_cached_ohlcv", lambda sym, curr: cached.copy()
    )
    # Request a window starting BEFORE the cache's coverage.
    out = y_finance.get_YFin_data_online("AAPL", "2026-01-01", "2026-06-10")
    assert "Stock data for" in out
    assert "2026-06-10" in out


@pytest.mark.unit
def test_cache_miss_falls_back_online(monkeypatch):
    """No cache at all -> the original online path runs unchanged."""
    dummy = _DummyTicker("AAPL")
    monkeypatch.setattr(y_finance.yf, "Ticker", lambda s: dummy)
    monkeypatch.setattr(y_finance, "read_cached_ohlcv", lambda sym, curr: None)
    out = y_finance.get_YFin_data_online("AAPL", "2026-06-01", "2026-06-11")
    assert "Stock data for" in out


# ---------------------------------------------------------------------------
# get_fundamentals
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_fundamentals_happy_path(monkeypatch):
    """Live mode with populated info returns formatted fields."""
    dummy = _DummyTicker("AAPL")
    dummy._info = {"longName": "Apple Inc.", "trailingPE": 28.5, "marketCap": 3000000000000}
    monkeypatch.setattr(y_finance.yf, "Ticker", lambda s: dummy)
    # monkeypatch yf_retry to bypass retry/network for .info
    monkeypatch.setattr(y_finance, "yf_retry", lambda fn, **kw: fn())
    result = y_finance.get_fundamentals("AAPL")
    assert "Company Fundamentals" in result
    assert "Name: Apple Inc." in result
    assert "PE Ratio (TTM): 28.5" in result


@pytest.mark.unit
def test_fundamentals_skips_none_fields(monkeypatch):
    """Fields with value=None are omitted from the output."""
    dummy = _DummyTicker("AAPL")
    dummy._info = {"longName": "Apple Inc.", "trailingPE": None}
    monkeypatch.setattr(y_finance.yf, "Ticker", lambda s: dummy)
    monkeypatch.setattr(y_finance, "yf_retry", lambda fn, **kw: fn())
    result = y_finance.get_fundamentals("AAPL")
    assert "Name: Apple Inc." in result
    assert "PE Ratio" not in result  # None field skipped


@pytest.mark.unit
def test_fundamentals_all_none_stub_raises(monkeypatch):
    """yfinance stub info (all None) raises NoMarketDataError."""
    dummy = _DummyTicker("UNK")
    dummy._info = {"trailingPegRatio": None}
    monkeypatch.setattr(y_finance.yf, "Ticker", lambda s: dummy)
    monkeypatch.setattr(y_finance, "yf_retry", lambda fn, **kw: fn())
    with pytest.raises(NoMarketDataError, match="no fundamental fields"):
        y_finance.get_fundamentals("UNK")


@pytest.mark.unit
def test_fundamentals_empty_info_raises(monkeypatch):
    """An empty info dict raises NoMarketDataError."""
    dummy = _DummyTicker("UNK")
    dummy._info = {}
    monkeypatch.setattr(y_finance.yf, "Ticker", lambda s: dummy)
    monkeypatch.setattr(y_finance, "yf_retry", lambda fn, **kw: fn())
    with pytest.raises(NoMarketDataError, match="no fundamentals returned"):
        y_finance.get_fundamentals("UNK")


@pytest.mark.unit
def test_fundamentals_generic_exception_propagates(monkeypatch):
    """A RuntimeError during info access propagates (not swallowed)."""
    class _ExplodingTicker:
        def __init__(self, symbol):
            pass

        @property
        def info(self):
            raise RuntimeError("yfinance exploded")

    monkeypatch.setattr(y_finance.yf, "Ticker", lambda s: _ExplodingTicker(s))
    monkeypatch.setattr(y_finance, "yf_retry", lambda fn, **kw: fn())
    with pytest.raises(RuntimeError, match="yfinance exploded"):
        y_finance.get_fundamentals("AAPL")


# ---------------------------------------------------------------------------
# get_balance_sheet / get_cashflow / get_income_statement
# ---------------------------------------------------------------------------

def _stmt_frame():
    return pd.DataFrame(
        {"TotalAssets": [100, 90]},
        index=pd.Index(["2025-12-31", "2025-09-30"], name="Period"),
    )


@pytest.mark.unit
def test_balance_sheet_quarterly_happy_path(monkeypatch):
    dummy = _DummyTicker("AAPL")
    dummy.quarterly_balance_sheet = _stmt_frame()
    monkeypatch.setattr(y_finance.yf, "Ticker", lambda s: dummy)
    monkeypatch.setattr(y_finance, "yf_retry", lambda fn, **kw: fn())
    result = y_finance.get_balance_sheet("AAPL", freq="quarterly")
    assert "Balance Sheet data for AAPL (quarterly)" in result
    assert "TotalAssets" in result


@pytest.mark.unit
def test_balance_sheet_annual_uses_balance_sheet_attr(monkeypatch):
    dummy = _DummyTicker("AAPL")
    dummy.balance_sheet = _stmt_frame()
    monkeypatch.setattr(y_finance.yf, "Ticker", lambda s: dummy)
    monkeypatch.setattr(y_finance, "yf_retry", lambda fn, **kw: fn())
    result = y_finance.get_balance_sheet("AAPL", freq="annual")
    assert "Balance Sheet data for AAPL (annual)" in result


@pytest.mark.unit
def test_balance_sheet_empty_raises(monkeypatch):
    dummy = _DummyTicker("UNK")
    # quarterly_balance_sheet is empty by default
    monkeypatch.setattr(y_finance.yf, "Ticker", lambda s: dummy)
    monkeypatch.setattr(y_finance, "yf_retry", lambda fn, **kw: fn())
    with pytest.raises(NoMarketDataError, match="no balance sheet"):
        y_finance.get_balance_sheet("UNK")


@pytest.mark.unit
def test_balance_sheet_exception_propagates(monkeypatch):
    class _ExplodingTicker:
        def __init__(self, symbol):
            pass

        @property
        def quarterly_balance_sheet(self):
            raise RuntimeError("boom")

    monkeypatch.setattr(y_finance.yf, "Ticker", lambda s: _ExplodingTicker(s))
    monkeypatch.setattr(y_finance, "yf_retry", lambda fn, **kw: fn())
    with pytest.raises(RuntimeError, match="boom"):
        y_finance.get_balance_sheet("AAPL")


@pytest.mark.unit
def test_cashflow_quarterly_happy_path(monkeypatch):
    dummy = _DummyTicker("AAPL")
    dummy.quarterly_cashflow = _stmt_frame()
    monkeypatch.setattr(y_finance.yf, "Ticker", lambda s: dummy)
    monkeypatch.setattr(y_finance, "yf_retry", lambda fn, **kw: fn())
    result = y_finance.get_cashflow("AAPL", freq="quarterly")
    assert "Cash Flow data for AAPL" in result


@pytest.mark.unit
def test_cashflow_empty_raises(monkeypatch):
    dummy = _DummyTicker("UNK")
    monkeypatch.setattr(y_finance.yf, "Ticker", lambda s: dummy)
    monkeypatch.setattr(y_finance, "yf_retry", lambda fn, **kw: fn())
    with pytest.raises(NoMarketDataError, match="no cash flow"):
        y_finance.get_cashflow("UNK")


@pytest.mark.unit
def test_income_statement_annual_happy_path(monkeypatch):
    dummy = _DummyTicker("AAPL")
    dummy.income_stmt = _stmt_frame()
    monkeypatch.setattr(y_finance.yf, "Ticker", lambda s: dummy)
    monkeypatch.setattr(y_finance, "yf_retry", lambda fn, **kw: fn())
    result = y_finance.get_income_statement("AAPL", freq="annual")
    assert "Income Statement data for AAPL" in result


@pytest.mark.unit
def test_income_statement_empty_raises(monkeypatch):
    dummy = _DummyTicker("UNK")
    monkeypatch.setattr(y_finance.yf, "Ticker", lambda s: dummy)
    monkeypatch.setattr(y_finance, "yf_retry", lambda fn, **kw: fn())
    with pytest.raises(NoMarketDataError, match="no income statement"):
        y_finance.get_income_statement("UNK")


# ---------------------------------------------------------------------------
# get_insider_transactions  (NOTE: empty is NORMAL — returns prose, not raise)
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_insider_transactions_empty_returns_prose(monkeypatch):
    """Empty insider data is normal — returns prose, NOT an exception."""
    dummy = _DummyTicker("AAPL")
    monkeypatch.setattr(y_finance.yf, "Ticker", lambda s: dummy)
    monkeypatch.setattr(y_finance, "yf_retry", lambda fn, **kw: fn())
    result = y_finance.get_insider_transactions("AAPL")
    assert "No insider transactions reported" in result


@pytest.mark.unit
def test_insider_transactions_none_returns_prose(monkeypatch):
    """None insider_transactions also returns prose (not raise)."""
    dummy = _DummyTicker("AAPL")
    dummy.insider_transactions = None
    monkeypatch.setattr(y_finance.yf, "Ticker", lambda s: dummy)
    monkeypatch.setattr(y_finance, "yf_retry", lambda fn, **kw: fn())
    result = y_finance.get_insider_transactions("AAPL")
    assert "No insider transactions reported" in result


@pytest.mark.unit
def test_insider_transactions_with_data_returns_csv(monkeypatch):
    dummy = _DummyTicker("AAPL")
    dummy.insider_transactions = pd.DataFrame(
        {"Transaction": ["BUY"]}, index=pd.Index(["2025-01-01"], name="Date"),
    )
    monkeypatch.setattr(y_finance.yf, "Ticker", lambda s: dummy)
    monkeypatch.setattr(y_finance, "yf_retry", lambda fn, **kw: fn())
    result = y_finance.get_insider_transactions("AAPL")
    assert "Insider Transactions data for AAPL" in result
    assert "BUY" in result


@pytest.mark.unit
def test_insider_transactions_exception_propagates(monkeypatch):
    class _ExplodingTicker:
        def __init__(self, symbol):
            pass

        @property
        def insider_transactions(self):
            raise RuntimeError("boom")

    monkeypatch.setattr(y_finance.yf, "Ticker", lambda s: _ExplodingTicker(s))
    monkeypatch.setattr(y_finance, "yf_retry", lambda fn, **kw: fn())
    with pytest.raises(RuntimeError, match="boom"):
        y_finance.get_insider_transactions("AAPL")


# ---------------------------------------------------------------------------
# get_stock_stats_indicators_window / get_stockstats_indicator
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_indicators_window_unsupported_raises(monkeypatch):
    with pytest.raises(ValueError, match="not supported"):
        y_finance.get_stock_stats_indicators_window("AAPL", "bogus", "2026-06-12", 3)


@pytest.mark.unit
def test_indicators_window_builds_window_with_na(monkeypatch):
    """Non-trading days are filled with 'N/A: Not a trading day'."""
    # _get_stock_stats_bulk returns data for only one date in the window
    monkeypatch.setattr(y_finance, "_get_stock_stats_bulk",
                        lambda s, i, c: {"2026-06-10": "70.5"})
    result = y_finance.get_stock_stats_indicators_window("AAPL", "rsi", "2026-06-12", 3)
    # Window covers 06-10, 06-11, 06-12. Only 06-10 has data.
    assert "2026-06-12: N/A: Not a trading day" in result
    assert "2026-06-11: N/A: Not a trading day" in result
    assert "2026-06-10: 70.5" in result
    assert "RSI:" in result  # description trailer


@pytest.mark.unit
def test_indicators_window_no_market_data_propagates(monkeypatch):
    def _raise(*a, **kw):
        raise NoMarketDataError("AAPL", "AAPL", "no data")
    monkeypatch.setattr(y_finance, "_get_stock_stats_bulk", _raise)
    with pytest.raises(NoMarketDataError):
        y_finance.get_stock_stats_indicators_window("AAPL", "rsi", "2026-06-12", 3)


@pytest.mark.unit
def test_indicators_window_generic_exception_propagates(monkeypatch, caplog):
    def _raise(*a, **kw):
        raise RuntimeError("stockstats boom")
    monkeypatch.setattr(y_finance, "_get_stock_stats_bulk", _raise)
    with pytest.raises(RuntimeError):
        y_finance.get_stock_stats_indicators_window("AAPL", "rsi", "2026-06-12", 3)
    assert any("bulk stockstats calc failed" in r.message for r in caplog.records)


@pytest.mark.unit
def test_stockstats_indicator_happy_path(monkeypatch):
    monkeypatch.setattr(y_finance.StockstatsUtils, "get_stock_stats",
                        staticmethod(lambda s, i, c: 42.5))
    result = y_finance.get_stockstats_indicator("AAPL", "rsi", "2026-06-10")
    assert result == "42.5"


@pytest.mark.unit
def test_stockstats_indicator_no_market_data_propagates(monkeypatch):
    def _raise(*a, **kw):
        raise NoMarketDataError("AAPL", "AAPL", "no data")
    monkeypatch.setattr(y_finance.StockstatsUtils, "get_stock_stats", staticmethod(_raise))
    with pytest.raises(NoMarketDataError):
        y_finance.get_stockstats_indicator("AAPL", "rsi", "2026-06-10")


@pytest.mark.unit
def test_stockstats_indicator_generic_exception_propagates(monkeypatch):
    def _raise(*a, **kw):
        raise RuntimeError("boom")
    monkeypatch.setattr(y_finance.StockstatsUtils, "get_stock_stats", staticmethod(_raise))
    with pytest.raises(RuntimeError):
        y_finance.get_stockstats_indicator("AAPL", "rsi", "2026-06-10")
