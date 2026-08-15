"""Alpha Vantage request hardening.

Regressions for #990 (no request timeout -> can hang) and #991 (invalid-key
responses mislabeled as rate limits and silently treated as transient).
"""
import pytest

import yiagents.dataflows.alpha_vantage_common as av


@pytest.fixture(autouse=True)
def _isolated_av_cache(tmp_path, monkeypatch):
    """Route the AV disk cache into a per-test tmp dir.

    ``_make_api_request`` caches responses on disk; without isolation the
    first test's fake body would be served to every later test (and pollute
    the user's real cache directory).
    """
    def _cache_dir(name):
        d = tmp_path / name
        d.mkdir(parents=True, exist_ok=True)
        return str(d)

    monkeypatch.setattr(av, "vendor_cache_dir", _cache_dir)


class _FakeResponse:
    def __init__(self, text):
        self.text = text

    def raise_for_status(self):
        pass


def _patched_get(body, capture=None):
    def fake_get(url, params=None, **kwargs):
        if capture is not None:
            capture.update(kwargs)
        return _FakeResponse(body)
    return fake_get


@pytest.mark.unit
def test_request_passes_timeout(monkeypatch):
    captured = {}
    monkeypatch.setattr(av.requests, "get", _patched_get("Date,Close\n2025-01-02,1.0", captured))
    av._make_api_request("TIME_SERIES_DAILY", {"symbol": "AAPL"})
    assert captured.get("timeout") == av.REQUEST_TIMEOUT  # #990


@pytest.mark.unit
def test_rate_limit_detected(monkeypatch):
    body = '{"Information": "Our standard API rate limit is 25 requests per day. ... your API key ..."}'
    monkeypatch.setattr(av.requests, "get", _patched_get(body))
    with pytest.raises(av.AlphaVantageRateLimitError):
        av._make_api_request("TIME_SERIES_DAILY", {"symbol": "AAPL"})


@pytest.mark.unit
def test_invalid_key_not_mislabeled_as_rate_limit(monkeypatch):
    # AV's invalid-key notice mentions "API key"; it must NOT be treated as a
    # (transient) rate limit, but surface as a real configuration error (#991).
    body = ('{"Information": "the parameter apikey is invalid or missing. '
            'Please claim your free API key on (https://www.alphavantage.co/support/#api-key)."}')
    monkeypatch.setattr(av.requests, "get", _patched_get(body))
    with pytest.raises(av.AlphaVantageNotConfiguredError):
        av._make_api_request("TIME_SERIES_DAILY", {"symbol": "AAPL"})
    with pytest.raises(av.AlphaVantageRateLimitError):  # sanity: rate-limit path still distinct
        monkeypatch.setattr(av.requests, "get", _patched_get('{"Note": "API call frequency is 5 calls per minute."}'))
        av._make_api_request("TIME_SERIES_DAILY", {"symbol": "AAPL"})


@pytest.mark.unit
def test_error_message_json_raises_no_market_data(monkeypatch):
    """AV's hard-failure body ({"Error Message": ...}, e.g. an invalid symbol)
    is classified as NoMarketDataError — not returned as text for callers to
    read as a successful payload, and not mislabeled as a rate limit."""
    from yiagents.dataflows.errors import NoMarketDataError, VendorRateLimitError

    body = ('{"Error Message": "Invalid API call. Please retry or visit the '
            'documentation."}')
    monkeypatch.setattr(av.requests, "get", _patched_get(body))
    with pytest.raises(NoMarketDataError) as ei:
        av._make_api_request("TIME_SERIES_DAILY", {"symbol": "NOSUCH"})
    assert not isinstance(ei.value, VendorRateLimitError)
    assert "Invalid API call" in ei.value.detail
    assert ei.value.symbol == "NOSUCH"  # the queried symbol is carried for the router


@pytest.mark.unit
def test_error_message_uses_function_name_when_no_symbol(monkeypatch):
    """Endpoints without a ``symbol`` param (e.g. NEWS_SENTIMENT's ``tickers``)
    still classify; the symbol field falls back to the function name."""
    from yiagents.dataflows.errors import NoMarketDataError

    monkeypatch.setattr(
        av.requests, "get", _patched_get('{"Error Message": "Invalid API call."}')
    )
    with pytest.raises(NoMarketDataError) as ei:
        av._make_api_request("NEWS_SENTIMENT", {"tickers": "AAPL"})
    assert ei.value.symbol == "AAPL"  # tickers is also considered


@pytest.mark.unit
def test_error_message_is_not_cached_as_a_success(monkeypatch, tmp_path):
    """A raised classification must not poison the cache: the next call
    re-fetches instead of being served the stored error body."""
    calls = {"n": 0}

    def counting_get(url, params=None, **kwargs):
        calls["n"] += 1
        return _FakeResponse('{"Error Message": "Invalid API call."}')

    monkeypatch.setattr(av.requests, "get", counting_get)
    from yiagents.dataflows.errors import NoMarketDataError

    with pytest.raises(NoMarketDataError):
        av._make_api_request("TIME_SERIES_DAILY", {"symbol": "AAPL"})
    with pytest.raises(NoMarketDataError):
        av._make_api_request("TIME_SERIES_DAILY", {"symbol": "AAPL"})
    assert calls["n"] == 2  # nothing was cached; both calls hit the transport


@pytest.mark.unit
def test_second_identical_request_served_from_cache(monkeypatch):
    """The disk cache collapses N identical batch-run calls to one HTTP hit."""
    calls = {"n": 0}

    def counting_get(url, params=None, **kwargs):
        calls["n"] += 1
        return _FakeResponse("Date,Close\n2025-01-02,1.0")

    monkeypatch.setattr(av.requests, "get", counting_get)
    first = av._make_api_request("TIME_SERIES_DAILY", {"symbol": "AAPL"})
    second = av._make_api_request("TIME_SERIES_DAILY", {"symbol": "AAPL"})
    assert calls["n"] == 1
    assert first == second


@pytest.mark.unit
def test_transient_connection_error_retried_once(monkeypatch):
    monkeypatch.setattr(
        "yiagents.dataflows.netretry.time.sleep", lambda s: None
    )
    calls = {"n": 0}

    def flaky_get(url, params=None, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise av.requests.exceptions.ConnectionError("reset")
        return _FakeResponse("Date,Close\n2025-01-02,1.0")

    monkeypatch.setattr(av.requests, "get", flaky_get)
    out = av._make_api_request("TIME_SERIES_DAILY", {"symbol": "MSFT"})
    assert "2025-01-02" in out
    assert calls["n"] == 2


# ---------------------------------------------------------------------------
# _filter_csv_by_date_range — PIT date filter must never leak future rows.
# ---------------------------------------------------------------------------

_VALID_CSV = (
    "timestamp,open,high,low,close,volume\n"
    "2025-01-01,100,101,99,100.5,1000\n"
    "2025-01-02,100.5,102,100,101.0,1100\n"
    "2025-01-03,101,103,100.5,102.5,1200\n"
    "2025-01-04,102.5,104,102,103.0,1300\n"
    "2025-01-05,103,105,102.5,104.5,1400\n"
)


@pytest.mark.unit
def test_filter_success_returns_only_rows_in_range():
    """Normal CSV + date range → only rows within [start, end] are returned."""
    result = av._filter_csv_by_date_range(_VALID_CSV, "2025-01-02", "2025-01-04")
    assert "2025-01-01" not in result  # before start
    assert "2025-01-05" not in result  # after end
    assert "2025-01-02" in result
    assert "2025-01-03" in result
    assert "2025-01-04" in result


@pytest.mark.unit
def test_filter_failure_raises_not_returns_unfiltered():
    """A malformed CSV (no parseable date column) must RAISE, not return the
    unfiltered data — otherwise a backtest silently sees future rows (lookahead
    leak) and a live run is over-fed.  This mirrors the y_finance.py contract.
    """
    bad_csv = "garbage_col,foo\nnot_a_date,42\nalso_bad,99\n"
    with pytest.raises((ValueError, TypeError)):
        av._filter_csv_by_date_range(bad_csv, "2025-01-01", "2025-01-03")


@pytest.mark.unit
def test_filter_empty_csv_returns_empty():
    """Empty / whitespace-only CSV is a no-op (no rows to filter, no leak)."""
    assert av._filter_csv_by_date_range("", "2025-01-01", "2025-01-03") == ""
    assert av._filter_csv_by_date_range("   \n  ", "2025-01-01", "2025-01-03") == "   \n  "
