"""Binance transport resilience — keepalive / transient retry / Retry-After.

Zero-network, zero-LLM. Mirrors the mock-requests + sleep-intercept shape of
test_binance_rate_limit.py. The headline guard is byte-equivalence: with the
three toggles at their defaults (binance_http_keepalive=False,
binance_http_retries=0, binance_honor_retry_after=False), ``_http_get`` behaves
identically to today — module ``requests.get`` (not a shared session), raw
transport-exception propagation (not converted to NoMarketDataError), and an
immediate 429 raise (no Retry-After sleep).
"""
import pytest
import requests

from yiagents.dataflows import binance
from yiagents.dataflows import binance_http
from yiagents.dataflows.config import set_config
from yiagents.dataflows.errors import NoMarketDataError, VendorRateLimitError


# --- fakes -------------------------------------------------------------------

class _FakeResp:
    """Minimal stand-in for requests.Response: status/text/json/headers."""

    def __init__(self, status_code=200, text="[]", json_data=None, headers=None):
        self.status_code = status_code
        self.text = text
        self._json = json_data
        self.headers = headers if headers is not None else {}

    def json(self):
        if self._json is None:
            raise ValueError("no json configured")
        return self._json


def _get_returning(resp):
    """fake requests.get that always returns ``resp``."""
    def fake_get(url, params=None, **kwargs):
        return resp
    return fake_get


class _ScriptedGet:
    """Fake ``requests.get`` that plays back a scripted sequence, replaying the
    last entry once the sequence is exhausted (so a single-element list means
    "always this"). Entries are either a _FakeResp (returned) or an Exception
    instance (raised)."""

    def __init__(self, results):
        self._results = list(results)
        self.calls = 0

    def __call__(self, url, params=None, **kwargs):
        self.calls += 1
        idx = min(self.calls - 1, len(self._results) - 1)
        result = self._results[idx]
        if isinstance(result, BaseException):
            raise result
        return result


class _FakeSession:
    """Stand-in for the shared requests.Session used when keepalive is on."""

    def __init__(self, fake_get):
        self._fake_get = fake_get
        self.get_calls = 0

    def get(self, url, params=None, **kwargs):
        self.get_calls += 1
        return self._fake_get(url, params=params, **kwargs)

    def close(self):
        pass


@pytest.fixture(autouse=True)
def _reset_resilience_state():
    """Fresh shared-session singleton per test (config is reset by conftest)."""
    binance_http.reset_for_test()
    yield
    binance_http.reset_for_test()


# --- byte-equivalence guard (the headline) -----------------------------------

@pytest.mark.unit
def test_defaults_success_uses_requests_get_not_session(monkeypatch):
    # All three toggles at default: _http_get must use the module-level
    # requests.get and must NOT create the shared keepalive session, so the
    # success path is byte-identical to today.
    monkeypatch.setattr(
        binance.requests, "get",
        _get_returning(_FakeResp(status_code=200, text="[[1]]", json_data=[[1]])),
    )
    assert binance_http.has_shared_session() is False

    out = binance._http_get("/fapi/v1/klines", {"symbol": "BTCUSDT"}, "BTCUSDT", "BTCUSDT")

    assert out == [[1]]
    assert binance_http.has_shared_session() is False


@pytest.mark.unit
def test_retries_off_propagates_raw_transport_error(monkeypatch):
    # binance_http_retries=0 (default): a transport exception propagates RAW,
    # not converted to NoMarketDataError — byte-equivalent to today.
    sleeps = []
    monkeypatch.setattr(binance.time, "sleep", lambda s: sleeps.append(s))
    monkeypatch.setattr(
        binance.requests, "get",
        _ScriptedGet([requests.ConnectionError("dns boom")]),
    )

    with pytest.raises(requests.ConnectionError):
        binance._http_get("/fapi/v1/klines", {"symbol": "BTCUSDT"}, "BTCUSDT", "BTCUSDT")
    assert sleeps == []  # no retry sleep with retries=0


@pytest.mark.unit
def test_retryafter_off_raises_immediately(monkeypatch):
    # honor_retry_after=False (default): 429 raises without sleeping, even when
    # a Retry-After header is present — byte-equivalent to today.
    sleeps = []
    monkeypatch.setattr(binance.time, "sleep", lambda s: sleeps.append(s))
    monkeypatch.setattr(
        binance.requests, "get",
        _get_returning(_FakeResp(
            status_code=429, text="{}", headers={"Retry-After": "5"},
        )),
    )

    with pytest.raises(VendorRateLimitError):
        binance._http_get("/fapi/v1/klines", {"symbol": "BTCUSDT"}, "BTCUSDT", "BTCUSDT")
    assert sleeps == []


# --- retry layer (transient transport errors) --------------------------------

@pytest.mark.unit
def test_retry_succeeds_on_transient(monkeypatch):
    set_config({"binance_http_retries": 2})
    sleeps = []
    monkeypatch.setattr(binance.time, "sleep", lambda s: sleeps.append(s))
    scripted = _ScriptedGet([
        requests.ConnectionError("first attempt fails"),
        _FakeResp(status_code=200, text="[[1]]", json_data=[[1]]),
    ])
    monkeypatch.setattr(binance.requests, "get", scripted)

    out = binance._http_get("/fapi/v1/klines", {"symbol": "BTCUSDT"}, "BTCUSDT", "BTCUSDT")

    assert out == [[1]]
    assert scripted.calls == 2          # 1 fail + 1 success
    assert sleeps == [2.0]              # _RETRY_BASE_DELAY * 2**0


@pytest.mark.unit
def test_retry_exhausted_raises_nomarketdata(monkeypatch):
    set_config({"binance_http_retries": 2})
    monkeypatch.setattr(binance.time, "sleep", lambda s: None)
    scripted = _ScriptedGet([requests.ConnectionError("always fails")])
    monkeypatch.setattr(binance.requests, "get", scripted)

    with pytest.raises(NoMarketDataError):
        binance._http_get("/fapi/v1/klines", {"symbol": "BTCUSDT"}, "BTCUSDT", "BTCUSDT")
    assert scripted.calls == 3          # 1 + 2 retries


@pytest.mark.unit
def test_retry_backoff_is_exponential(monkeypatch):
    # With 3 retries all failing, the delays are 2, 4, 8 (base * 2**attempt).
    set_config({"binance_http_retries": 3})
    sleeps = []
    monkeypatch.setattr(binance.time, "sleep", lambda s: sleeps.append(s))
    monkeypatch.setattr(
        binance.requests, "get",
        _ScriptedGet([requests.ConnectionError("always fails")]),
    )

    with pytest.raises(NoMarketDataError):
        binance._http_get("/fapi/v1/klines", {"symbol": "BTCUSDT"}, "BTCUSDT", "BTCUSDT")
    assert sleeps == [2.0, 4.0, 8.0]


# --- retry layer (5xx) -------------------------------------------------------

@pytest.mark.unit
def test_retry_5xx_then_succeeds(monkeypatch):
    set_config({"binance_http_retries": 2})
    monkeypatch.setattr(binance.time, "sleep", lambda s: None)
    scripted = _ScriptedGet([
        _FakeResp(status_code=503, text="{}"),
        _FakeResp(status_code=200, text="[[1]]", json_data=[[1]]),
    ])
    monkeypatch.setattr(binance.requests, "get", scripted)

    out = binance._http_get("/fapi/v1/klines", {"symbol": "BTCUSDT"}, "BTCUSDT", "BTCUSDT")
    assert out == [[1]]
    assert scripted.calls == 2


@pytest.mark.unit
def test_retry_5xx_exhausted_raises_nomarketdata(monkeypatch):
    # 5xx that exhausts retries is returned to _http_get, whose existing
    # non-200 → NoMarketDataError path fires (today's behaviour).
    set_config({"binance_http_retries": 1})
    monkeypatch.setattr(binance.time, "sleep", lambda s: None)
    scripted = _ScriptedGet([_FakeResp(status_code=502, text="{}")])
    monkeypatch.setattr(binance.requests, "get", scripted)

    with pytest.raises(NoMarketDataError):
        binance._http_get("/fapi/v1/klines", {"symbol": "BTCUSDT"}, "BTCUSDT", "BTCUSDT")
    assert scripted.calls == 2          # 1 + 1 retry


@pytest.mark.unit
def test_429_not_retried_by_retry_layer(monkeypatch):
    # 429 is NOT in _RETRIABLE_STATUS: the retry layer returns it immediately and
    # the reactive VendorRateLimitError path handles it. No retries consumed.
    set_config({"binance_http_retries": 3})
    monkeypatch.setattr(binance.time, "sleep", lambda s: None)
    scripted = _ScriptedGet([_FakeResp(status_code=429, text="{}", headers={})])
    monkeypatch.setattr(binance.requests, "get", scripted)

    with pytest.raises(VendorRateLimitError):
        binance._http_get("/fapi/v1/klines", {"symbol": "BTCUSDT"}, "BTCUSDT", "BTCUSDT")
    assert scripted.calls == 1


# --- keepalive (shared session) ----------------------------------------------

@pytest.mark.unit
def test_keepalive_on_uses_shared_session(monkeypatch):
    set_config({"binance_http_keepalive": True})
    monkeypatch.setattr(binance.time, "sleep", lambda s: None)
    fake_session = _FakeSession(_get_returning(
        _FakeResp(status_code=200, text="[[1]]", json_data=[[1]]),
    ))
    monkeypatch.setattr(binance, "get_shared_binance_session", lambda: fake_session)

    def boom(url, params=None, **kwargs):
        raise AssertionError("requests.get must not be called when keepalive is on")
    monkeypatch.setattr(binance.requests, "get", boom)

    out = binance._http_get("/fapi/v1/klines", {"symbol": "BTCUSDT"}, "BTCUSDT", "BTCUSDT")

    assert out == [[1]]
    assert fake_session.get_calls == 1


@pytest.mark.unit
def test_keepalive_session_is_singleton(monkeypatch):
    # Two calls reuse the SAME session instance (process-wide pool). Patch the
    # .get on the real shared session so no network call is made (keepalive path
    # ignores binance.requests.get, so that monkeypatch would be a no-op here).
    set_config({"binance_http_keepalive": True})
    monkeypatch.setattr(binance.time, "sleep", lambda s: None)
    sess = binance_http.get_shared_binance_session()
    monkeypatch.setattr(
        sess, "get",
        _get_returning(_FakeResp(status_code=200, text="[[1]]", json_data=[[1]])),
    )

    out1 = binance._http_get("/fapi/v1/klines", {"symbol": "BTCUSDT"}, "BTCUSDT", "BTCUSDT")
    out2 = binance._http_get("/fapi/v1/klines", {"symbol": "BTCUSDT"}, "BTCUSDT", "BTCUSDT")

    assert out1 == [[1]] and out2 == [[1]]
    assert binance_http.get_shared_binance_session() is sess  # same instance reused


# --- Retry-After honoring ----------------------------------------------------

@pytest.mark.unit
def test_parse_retry_after_seconds():
    assert binance._parse_retry_after({"Retry-After": "7"}) == 7
    assert binance._parse_retry_after({"retry-after": "7"}) == 7  # case-insensitive
    assert binance._parse_retry_after({}) is None
    assert binance._parse_retry_after(None) is None
    # HTTP-date form unsupported (Binance sends seconds) → None.
    assert binance._parse_retry_after(
        {"Retry-After": "Wed, 21 Oct 2015 07:28:00 GMT"}
    ) is None


@pytest.mark.unit
def test_retryafter_on_sleeps_then_raises(monkeypatch):
    set_config({"binance_honor_retry_after": True})
    sleeps = []
    monkeypatch.setattr(binance.time, "sleep", lambda s: sleeps.append(s))
    monkeypatch.setattr(
        binance.requests, "get",
        _get_returning(_FakeResp(
            status_code=429, text="{}", headers={"Retry-After": "5"},
        )),
    )

    with pytest.raises(VendorRateLimitError):
        binance._http_get("/fapi/v1/klines", {"symbol": "BTCUSDT"}, "BTCUSDT", "BTCUSDT")
    assert sleeps == [5]


@pytest.mark.unit
def test_retryafter_over_cap_no_sleep(monkeypatch):
    # A multi-minute/day ban (> _RETRY_AFTER_CAP_S=60) is deferred to run_robust
    # rather than slept out inline — do not hang the ticker.
    set_config({"binance_honor_retry_after": True})
    sleeps = []
    monkeypatch.setattr(binance.time, "sleep", lambda s: sleeps.append(s))
    monkeypatch.setattr(
        binance.requests, "get",
        _get_returning(_FakeResp(
            status_code=418, text="{}", headers={"Retry-After": "9999"},
        )),
    )

    with pytest.raises(VendorRateLimitError):
        binance._http_get("/fapi/v1/klines", {"symbol": "BTCUSDT"}, "BTCUSDT", "BTCUSDT")
    assert sleeps == []


@pytest.mark.unit
def test_retryafter_missing_header_no_sleep(monkeypatch):
    set_config({"binance_honor_retry_after": True})
    sleeps = []
    monkeypatch.setattr(binance.time, "sleep", lambda s: sleeps.append(s))
    monkeypatch.setattr(
        binance.requests, "get",
        _get_returning(_FakeResp(status_code=429, text="{}", headers={})),
    )

    with pytest.raises(VendorRateLimitError):
        binance._http_get("/fapi/v1/klines", {"symbol": "BTCUSDT"}, "BTCUSDT", "BTCUSDT")
    assert sleeps == []


# --- toggle combinations -----------------------------------------------------

@pytest.mark.unit
def test_keepalive_and_retry_compose(monkeypatch):
    # keepalive on + retries on: a transient failure on the shared session is
    # retried and the second attempt (also via the session) succeeds.
    set_config({"binance_http_keepalive": True, "binance_http_retries": 2})
    monkeypatch.setattr(binance.time, "sleep", lambda s: None)
    scripted = _ScriptedGet([
        requests.ConnectionError("first fails"),
        _FakeResp(status_code=200, text="[[1]]", json_data=[[1]]),
    ])
    fake_session = _FakeSession(scripted)
    monkeypatch.setattr(binance, "get_shared_binance_session", lambda: fake_session)

    out = binance._http_get("/fapi/v1/klines", {"symbol": "BTCUSDT"}, "BTCUSDT", "BTCUSDT")

    assert out == [[1]]
    assert fake_session.get_calls == 2  # both attempts went through the session
