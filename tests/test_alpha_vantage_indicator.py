"""Unit tests for ``alpha_vantage_indicator.get_indicator``.

Covers indicator routing, CSV parsing, date-range filtering, and the fail-open
contract (errors must propagate, NOT be swallowed into prose — the project's #1
risk). The shared ``_make_api_request`` is mocked so no network/API key is needed.
"""
import pytest

import yialpha.dataflows.alpha_vantage_indicator as avi
import yialpha.dataflows.indicator_catalog as ic
from yialpha.dataflows.alpha_vantage_common import AlphaVantageNotConfiguredError
from yialpha.dataflows.errors import NoMarketDataError

# A representative CSV with rows before / inside / after the window.
# Default test window: curr_date=2025-03-01, look_back_days=30  => [2025-01-30, 2025-03-01]
_RSI_CSV = "\n".join([
    "time,RSI",
    "2025-01-15,55.0",   # before window
    "2025-02-01,60.0",   # inside window (start)
    "2025-02-15,65.0",   # inside window
    "2025-03-01,70.0",   # inside window (end boundary)
    "2025-03-10,75.0",   # after window
])


def _mock_api_request(return_value="", capture=None):
    """Build a mock _make_api_request that returns ``return_value``.

    If ``capture`` is a dict, the ``function_name`` + ``params`` are stored there.
    """
    def _mock(function_name, params):
        if capture is not None:
            capture["function_name"] = function_name
            capture["params"] = params
        return return_value
    return _mock


# ---------------------------------------------------------------------------
# A. Pure logic / no-network paths
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_unsupported_indicator_raises_valueerror(monkeypatch):
    """An unsupported indicator name raises ValueError listing valid options."""
    monkeypatch.setattr(avi, "_make_api_request", _mock_api_request(""))
    with pytest.raises(ValueError, match="not supported"):
        avi.get_indicator("AAPL", "bogus", "2025-03-01", 30)


@pytest.mark.unit
@pytest.mark.parametrize("indicator", ["vwma", "kdjk", "adx", "supertrend", "cci", "vr"])
def test_yfinance_only_indicators_raise_typed_no_data(monkeypatch, indicator):
    """yfinance-only indicators raise NoMarketDataError so the router falls
    through to the yfinance/stockstats vendor.

    AV has no endpoint for the catalog's YFINANCE_ONLY set (vwma plus the
    2026-08-15 expansion: KDJ family, adx, supertrend, cci, wr, stochrsi,
    roc, cmo, trix, vr). Returning prose here made the router treat the
    message as a successful result, so it NEVER tried the yfinance indicator
    vendor — the one that can compute these from OHLCV. The typed error takes
    the router's try-next-vendor path instead.
    """
    assert indicator in ic.YFINANCE_ONLY  # keeps the parametrize list honest
    called = []
    monkeypatch.setattr(avi, "_make_api_request", lambda *a, **kw: called.append(1))
    with pytest.raises(NoMarketDataError, match=indicator):
        avi.get_indicator("AAPL", indicator, "2025-03-01", 30)
    assert called == []  # no network call was made


# ---------------------------------------------------------------------------
# B. CSV parsing — routing / params
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_close_50_sma_routes_to_sma_time_period_50(monkeypatch):
    capture = {}
    monkeypatch.setattr(avi, "_make_api_request", _mock_api_request(
        "time,SMA\n2025-02-01,100.0\n", capture,
    ))
    avi.get_indicator("AAPL", "close_50_sma", "2025-03-01", 30)
    assert capture["function_name"] == "SMA"
    assert capture["params"]["time_period"] == "50"
    assert capture["params"]["series_type"] == "close"
    assert capture["params"]["datatype"] == "csv"


@pytest.mark.unit
def test_close_200_sma_routes_to_sma_time_period_200(monkeypatch):
    capture = {}
    monkeypatch.setattr(avi, "_make_api_request", _mock_api_request(
        "time,SMA\n2025-02-01,100.0\n", capture,
    ))
    avi.get_indicator("AAPL", "close_200_sma", "2025-03-01", 30)
    assert capture["function_name"] == "SMA"
    assert capture["params"]["time_period"] == "200"


@pytest.mark.unit
def test_close_10_ema_routes_to_ema_time_period_10(monkeypatch):
    capture = {}
    monkeypatch.setattr(avi, "_make_api_request", _mock_api_request(
        "time,EMA\n2025-02-01,100.0\n", capture,
    ))
    avi.get_indicator("AAPL", "close_10_ema", "2025-03-01", 30)
    assert capture["function_name"] == "EMA"
    assert capture["params"]["time_period"] == "10"


@pytest.mark.unit
@pytest.mark.parametrize("indicator,column", [
    ("macd", "MACD"),
    ("macds", "MACD_Signal"),
    ("macdh", "MACD_Hist"),
])
def test_macd_family_routes_to_macd_function(monkeypatch, indicator, column):
    csv_data = f"time,{column}\n2025-02-01,1.5\n"
    capture = {}
    monkeypatch.setattr(avi, "_make_api_request", _mock_api_request(csv_data, capture))
    result = avi.get_indicator("AAPL", indicator, "2025-03-01", 30)
    assert capture["function_name"] == "MACD"
    assert "1.5" in result  # the value from the mapped column was extracted


@pytest.mark.unit
def test_rsi_routes_to_rsi_function(monkeypatch):
    capture = {}
    monkeypatch.setattr(avi, "_make_api_request", _mock_api_request(_RSI_CSV, capture))
    avi.get_indicator("AAPL", "rsi", "2025-03-01", 30)
    assert capture["function_name"] == "RSI"
    assert capture["params"]["time_period"] == "14"  # default


@pytest.mark.unit
@pytest.mark.parametrize("indicator,column", [
    ("boll", "Real Middle Band"),
    ("boll_ub", "Real Upper Band"),
    ("boll_lb", "Real Lower Band"),
])
def test_bollinger_family_routes_to_bbands(monkeypatch, indicator, column):
    csv_data = f"time,{column}\n2025-02-01,100.0\n"
    capture = {}
    monkeypatch.setattr(avi, "_make_api_request", _mock_api_request(csv_data, capture))
    result = avi.get_indicator("AAPL", indicator, "2025-03-01", 30)
    assert capture["function_name"] == "BBANDS"
    assert capture["params"]["time_period"] == "20"
    assert "100.0" in result  # the value from the mapped column was extracted


@pytest.mark.unit
def test_atr_routes_to_atr_and_omits_series_type(monkeypatch):
    """ATR doesn't use series_type; the required_series_type is None so it's left unchanged."""
    capture = {}
    monkeypatch.setattr(avi, "_make_api_request", _mock_api_request(
        "time,ATR\n2025-02-01,2.5\n", capture,
    ))
    avi.get_indicator("AAPL", "atr", "2025-03-01", 30)
    assert capture["function_name"] == "ATR"
    assert "series_type" not in capture["params"]


@pytest.mark.unit
def test_mfi_routes_to_mfi_and_omits_series_type(monkeypatch):
    """mfi is in the unified catalog: AV routes it to the MFI function,
    volume-based like ATR (no series_type), and parses its column."""
    capture = {}
    monkeypatch.setattr(avi, "_make_api_request", _mock_api_request(
        "time,MFI\n2025-02-01,62.5\n", capture,
    ))
    result = avi.get_indicator("AAPL", "mfi", "2025-03-01", 30)
    assert capture["function_name"] == "MFI"
    assert "series_type" not in capture["params"]
    assert capture["params"]["time_period"] == "14"  # default
    assert "62.5" in result
    assert "MFI:" in result  # catalog description trailer


# ---------------------------------------------------------------------------
# C. CSV parsing — date filtering & output formatting
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_rsi_filters_by_date_range_and_sorts(monkeypatch):
    """Only in-window rows appear, sorted ascending; out-of-window excluded."""
    monkeypatch.setattr(avi, "_make_api_request", _mock_api_request(_RSI_CSV))
    result = avi.get_indicator("AAPL", "rsi", "2025-03-01", 30)
    # Window is [2025-01-30, 2025-03-01]; rows 02-01, 02-15, 03-01 inside.
    assert "2025-02-01: 60.0" in result
    assert "2025-02-15: 65.0" in result
    assert "2025-03-01: 70.0" in result
    # Out-of-window excluded
    assert "55.0" not in result   # 2025-01-15 before
    assert "75.0" not in result   # 2025-03-10 after
    # Description appended
    assert "RSI:" in result


# ---------------------------------------------------------------------------
# D. Error paths (fail-closed contract — errors must surface as typed errors,
#    never be returned as "Error: ..." strings that the router reads as data)
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_non_csv_data_raises_no_market_data(monkeypatch):
    """A non-str (dict) response raises NoMarketDataError (not prose)."""
    monkeypatch.setattr(avi, "_make_api_request", _mock_api_request({"not": "csv"}))
    with pytest.raises(NoMarketDataError, match="no CSV data"):
        avi.get_indicator("AAPL", "rsi", "2025-03-01", 30)


@pytest.mark.unit
def test_single_line_csv_raises_no_market_data(monkeypatch):
    """Header-only CSV (no data rows) raises NoMarketDataError."""
    monkeypatch.setattr(avi, "_make_api_request", _mock_api_request("time,RSI\n"))
    with pytest.raises(NoMarketDataError, match="no data rows"):
        avi.get_indicator("AAPL", "rsi", "2025-03-01", 30)


@pytest.mark.unit
def test_missing_time_column_raises_no_market_data(monkeypatch):
    """CSV without a 'time' column raises with the available columns listed."""
    monkeypatch.setattr(avi, "_make_api_request", _mock_api_request(
        "date,RSI\n2025-02-01,60.0\n",
    ))
    with pytest.raises(NoMarketDataError, match="'time' column not found"):
        avi.get_indicator("AAPL", "rsi", "2025-03-01", 30)


@pytest.mark.unit
def test_missing_value_column_raises_no_market_data(monkeypatch):
    """CSV with 'time' but missing the expected value column raises."""
    monkeypatch.setattr(avi, "_make_api_request", _mock_api_request(
        "time,SMA\n2025-02-01,100.0\n",  # request macd but only SMA column present
    ))
    with pytest.raises(NoMarketDataError, match="not found"):
        avi.get_indicator("AAPL", "macd", "2025-03-01", 30)


@pytest.mark.unit
def test_empty_date_window_returns_no_data_message(monkeypatch):
    """Valid CSV but all rows outside the window → 'No data available' message."""
    monkeypatch.setattr(avi, "_make_api_request", _mock_api_request(
        "time,RSI\n2024-01-01,50.0\n",  # far before the window
    ))
    result = avi.get_indicator("AAPL", "rsi", "2025-03-01", 30)
    assert "No data available for the specified date range" in result


@pytest.mark.unit
def test_alpha_vantage_not_configured_propagates(monkeypatch):
    """AlphaVantageNotConfiguredError must propagate (not be swallowed into prose)."""
    def _raise(*a, **kw):
        raise AlphaVantageNotConfiguredError("no key")
    monkeypatch.setattr(avi, "_make_api_request", _raise)
    with pytest.raises(AlphaVantageNotConfiguredError):
        avi.get_indicator("AAPL", "rsi", "2025-03-01", 30)


@pytest.mark.unit
def test_generic_exception_propagates_and_logs(monkeypatch, caplog):
    """A generic exception propagates AND is logged (surface, don't swallow)."""
    def _raise(*a, **kw):
        raise RuntimeError("network down")
    monkeypatch.setattr(avi, "_make_api_request", _raise)
    with pytest.raises(RuntimeError, match="network down"):
        avi.get_indicator("AAPL", "rsi", "2025-03-01", 30)
    # The exception must have been logged (not silently swallowed)
    assert any(
        "failed" in record.message and "rsi" in record.message
        for record in caplog.records
    )


# ---------------------------------------------------------------------------
# E. Router integration: the typed vwma error must FALL THROUGH to yfinance
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_router_falls_back_to_yfinance_on_vwma(monkeypatch):
    """AV's vwma NoMarketDataError routes to the yfinance indicator vendor.

    With the old returned-prose behaviour the chain stopped at Alpha Vantage
    with a message the agent read as data; the typed error keeps the chain
    moving to the vendor that can actually compute vwma. The AV side is the
    REAL implementation (vwma raises before any API call — no network); the
    yfinance side is stubbed at the router table.
    """
    from unittest import mock

    from yialpha.dataflows import config as cfgmod, interface

    orig = cfgmod.get_config()
    try:
        cfgmod.set_config({
            **orig,
            "data_vendors": {"technical_indicators": "alpha_vantage,yfinance"},
        })
        with mock.patch.dict(
            interface.VENDOR_METHODS,
            {"get_indicators": {"alpha_vantage": avi.get_indicator,
                                "yfinance": lambda *a, **k: "YFINANCE_VWMA_OK"}},
            clear=False,
        ):
            out = interface.route_to_vendor(
                "get_indicators", "AAPL", "vwma", "2025-03-01", 30,
            )
    finally:
        cfgmod.set_config(orig)
    assert "YFINANCE_VWMA_OK" in out
