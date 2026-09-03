"""Alpha Vantage request hardening.

Regressions for #990 (no request timeout -> can hang), #991 (invalid-key
responses mislabeled as rate limits and silently treated as transient), and
the R6 credential-redaction batch (vendor notices echo the request's apikey
back; raised exceptions and downstream logs must never carry it).
"""
import io
import json
import logging
from types import SimpleNamespace

import pytest
from rich.console import Console

import yialpha.dataflows.alpha_vantage_common as av
from yialpha.logging_config import setup_logging


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
    from yialpha.dataflows.errors import NoMarketDataError, VendorRateLimitError

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
    from yialpha.dataflows.errors import NoMarketDataError

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
    from yialpha.dataflows.errors import NoMarketDataError

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
        "yialpha.dataflows.netretry.time.sleep", lambda s: None
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


# ---------------------------------------------------------------------------
# Credential redaction (R6). Alpha Vantage echoes the request's apikey back
# inside "Information" / "Note" / "Error Message" bodies; those texts used to
# be embedded verbatim in raised exceptions, which callers then log. Every
# synthetic key below is fake; HTTP is mocked at av.requests.get.
# ---------------------------------------------------------------------------

_FAKE_KEY = "SYNTHETIC_AVCHECK_KEY42"


@pytest.mark.unit
def test_rate_limit_notice_redacts_api_key(monkeypatch):
    """A rate-limit notice echoing the apikey raises the typed error with the
    notice's semantics intact but the key scrubbed (#R6)."""
    body = json.dumps({"Information": f"Your API key {_FAKE_KEY} is limited to 25 requests per day."})
    monkeypatch.setattr(av, "get_api_key", lambda: _FAKE_KEY)
    monkeypatch.setattr(av.requests, "get", _patched_get(body))
    with pytest.raises(av.AlphaVantageRateLimitError) as ei:
        av._raw_api_request("NEWS_SENTIMENT", {"tickers": "MU"})
    message = str(ei.value)
    assert _FAKE_KEY not in message
    assert "[REDACTED]" in message
    assert "rate limit exceeded" in message  # error semantics preserved
    assert "limited to 25 requests per day" in message


@pytest.mark.unit
def test_invalid_key_notice_redacts_api_key(monkeypatch):
    """The invalid-key path stays correctly classified (#991) AND scrubbed
    (#R6): redaction must not flip the bad-key error into a rate limit."""
    body = json.dumps({"Information": f"the parameter apikey={_FAKE_KEY} is invalid or missing."})
    monkeypatch.setattr(av, "get_api_key", lambda: _FAKE_KEY)
    monkeypatch.setattr(av.requests, "get", _patched_get(body))
    with pytest.raises(av.AlphaVantageNotConfiguredError) as ei:
        av._raw_api_request("TIME_SERIES_DAILY", {"symbol": "AAPL"})
    message = str(ei.value)
    assert _FAKE_KEY not in message
    assert "API key invalid or missing" in message


@pytest.mark.unit
def test_error_message_detail_redacts_api_key(monkeypatch):
    """The hard-failure "Error Message" body is scrubbed inside the
    NoMarketDataError detail too (#R6)."""
    from yialpha.dataflows.errors import NoMarketDataError

    body = json.dumps({"Error Message": f"Invalid API call for key {_FAKE_KEY}."})
    monkeypatch.setattr(av, "get_api_key", lambda: _FAKE_KEY)
    monkeypatch.setattr(av.requests, "get", _patched_get(body))
    with pytest.raises(NoMarketDataError) as ei:
        av._raw_api_request("TIME_SERIES_DAILY", {"symbol": "NOSUCH"})
    assert _FAKE_KEY not in ei.value.detail
    assert "Invalid API call" in ei.value.detail


@pytest.mark.unit
def test_notice_redacts_apikey_query_parameter(monkeypatch):
    """``apikey=<token>`` forms are scrubbed by shape alone — the token here
    is deliberately short and key-unlike, so only the query-parameter rule
    can catch it."""
    body = ('{"Information": "the parameter apikey=demo123 is invalid or missing. '
            'Please claim your free API key on (https://www.alphavantage.co/support/#api-key)."}')
    monkeypatch.setattr(av, "get_api_key", lambda: _FAKE_KEY)
    monkeypatch.setattr(av.requests, "get", _patched_get(body))
    with pytest.raises(av.AlphaVantageNotConfiguredError) as ei:
        av._raw_api_request("TIME_SERIES_DAILY", {"symbol": "AAPL"})
    message = str(ei.value)
    assert "demo123" not in message
    assert "apikey=[REDACTED]" in message


@pytest.mark.unit
def test_exception_and_log_output_never_echo_synthetic_key(monkeypatch):
    """End-to-end R6 probe (mirrors the offline review's synthetic-key
    redaction check): a rate-limit notice echoing the request's apikey must
    not leak into the exception message NOR into the production log sink —
    and the log-side handler filter must additionally scrub a key pasted
    straight into a future log call."""
    fake_key = "SYNTHETIC_REVIEW_ONLY_KEY"
    body = json.dumps({"Information": f"Your API key {fake_key} is limited to 25 requests per day."})
    # Production posture: the request key is the configured env key.
    monkeypatch.setenv("ALPHA_VANTAGE_API_KEY", fake_key)
    monkeypatch.setattr(av, "get_api_key", lambda: fake_key)
    monkeypatch.setattr(av.requests, "get", _patched_get(body))
    with pytest.raises(av.AlphaVantageRateLimitError) as ei:
        av._raw_api_request("NEWS_SENTIMENT", {"tickers": "MU"})
    assert fake_key not in str(ei.value)

    stream = io.StringIO()
    root = logging.getLogger()
    prev_handlers, prev_level = list(root.handlers), root.level
    try:
        setup_logging("INFO")
        for handler in root.handlers:
            if hasattr(handler, "console"):
                handler.console = Console(file=stream, width=300, no_color=True)
        log = logging.getLogger("yialpha.dataflows.interface")
        log.warning("Returning NO_DATA, but a vendor errored earlier: %s", ei.value)
        log.warning("direct leak probe: %s", fake_key)  # fallback-layer probe
    finally:
        root.handlers = prev_handlers
        root.setLevel(prev_level)
    output = stream.getvalue()
    assert fake_key not in output
    assert "vendor errored earlier" in output
    assert "rate limit exceeded" in output
    assert "[REDACTED]" in output


@pytest.mark.unit
def test_synthetic_response_namespace_path_redacts(monkeypatch):
    """The review probe drove _raw_api_request with a SimpleNamespace response
    (raise_for_status as a lambda attribute); keep that shape covered too."""
    fake_key = "SYNTHETIC_REVIEW_ONLY_KEY"
    response = SimpleNamespace(
        text=json.dumps({"Information": f"Your API key {fake_key} is limited to 25 requests per day."}),
        raise_for_status=lambda: None,
    )
    monkeypatch.setattr(av, "get_api_key", lambda: fake_key)
    monkeypatch.setattr(av.requests, "get", lambda url, params=None, **kwargs: response)
    with pytest.raises(av.AlphaVantageRateLimitError) as ei:
        av._raw_api_request("NEWS_SENTIMENT", {"tickers": "MU"})
    assert fake_key not in str(ei.value)
