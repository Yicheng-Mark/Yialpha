"""Binance proactive rate-limit backoff (header-driven).

Zero-network, zero-LLM. Mirrors the mock-requests shape of
test_alpha_vantage_hardening.py and the sleep-intercept shape of
test_reddit_fallback.py. Guards both the backoff behaviour AND the
byte-equivalence contract: with ``binance_proactive_backoff`` off (default),
``_http_get`` neither reads the weight header nor sleeps, identical to today.
"""
import threading
import time

import pytest

from yiagents.dataflows import binance
from yiagents.dataflows import binance_rate_limiter as rl
from yiagents.dataflows.config import set_config
from yiagents.dataflows.errors import VendorRateLimitError


class _FakeBinanceResp:
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


@pytest.fixture(autouse=True)
def _reset_limiter():
    """Fresh limiter registry per test (config is already reset by conftest)."""
    rl.reset_for_test()
    yield
    rl.reset_for_test()


def _limiter():
    """The process-wide fapi limiter, freshly created after reset_for_test."""
    return rl.get_binance_weight_limiter("fapi")


# --- byte-equivalence guard -------------------------------------------------

@pytest.mark.unit
def test_backoff_off_does_not_read_header_or_sleep(monkeypatch):
    # Default config: binance_proactive_backoff=False. _http_get must not touch
    # the limiter — neither acquire (sleep) nor observe (header read).
    sleeps = []
    monkeypatch.setattr(rl.time, "sleep", lambda s: sleeps.append(s))
    monkeypatch.setattr(
        binance.requests, "get",
        _get_returning(_FakeBinanceResp(
            status_code=200, text="[[1]]", json_data=[[1]],
            headers={"X-MBX-USED-WEIGHT-1M": "999"},  # would trip if read
        )),
    )

    binance._http_get("/fapi/v1/klines", {"symbol": "BTCUSDT"}, "BTCUSDT", "BTCUSDT")

    assert _limiter().used == 0          # observe was skipped
    assert sleeps == []                   # acquire was skipped


# --- observe path -----------------------------------------------------------

@pytest.mark.unit
def test_observe_feeds_weight_from_header(monkeypatch):
    set_config({"binance_proactive_backoff": True})
    monkeypatch.setattr(rl.time, "sleep", lambda s: None)
    monkeypatch.setattr(
        binance.requests, "get",
        _get_returning(_FakeBinanceResp(
            status_code=200, text="[[1]]", json_data=[[1]],
            headers={"X-MBX-USED-WEIGHT-1M": "50"},
        )),
    )

    binance._http_get("/fapi/v1/klines", {"symbol": "BTCUSDT"}, "BTCUSDT", "BTCUSDT")

    assert _limiter().used == 50


@pytest.mark.unit
def test_observe_weight_missing_header_is_noop():
    set_config({"binance_proactive_backoff": True})
    resp_no_header = _FakeBinanceResp(
        status_code=200, text="[[1]]", json_data=[[1]], headers={}
    )
    binance._observe_weight(resp_no_header)
    assert _limiter().used == 0


@pytest.mark.unit
def test_observe_weight_unparseable_header_is_noop():
    set_config({"binance_proactive_backoff": True})
    resp_bad = _FakeBinanceResp(
        status_code=200, text="[[1]]", json_data=[[1]],
        headers={"X-MBX-USED-WEIGHT-1M": "not-a-number"},
    )
    binance._observe_weight(resp_bad)  # must not raise
    assert _limiter().used == 0


# --- acquire path -----------------------------------------------------------

@pytest.mark.unit
def test_acquire_sleeps_when_over_threshold(monkeypatch):
    # Default limit 2400 -> trip at 1920. observe(2300) is hot and fresh.
    sleeps = []
    monkeypatch.setattr(rl.time, "sleep", lambda s: sleeps.append(s))
    lim = _limiter()
    lim.observe(2300)

    lim.acquire()

    assert len(sleeps) == 1
    # age is ~0, so wait is ~60s (the rest of the rolling window).
    assert 59.0 < sleeps[0] <= 60.0


@pytest.mark.unit
def test_acquire_no_sleep_when_observation_stale(monkeypatch):
    sleeps = []
    monkeypatch.setattr(rl.time, "sleep", lambda s: sleeps.append(s))
    lim = _limiter()
    lim.observe(2300)
    lim._used_ts = time.monotonic() - 70.0  # aged out of the 60s window

    lim.acquire()

    assert sleeps == []


@pytest.mark.unit
def test_acquire_no_sleep_when_under_threshold(monkeypatch):
    sleeps = []
    monkeypatch.setattr(rl.time, "sleep", lambda s: sleeps.append(s))
    lim = _limiter()
    lim.observe(100)  # well under the 1920 trip point

    lim.acquire()

    assert sleeps == []


# --- reactive path intact ---------------------------------------------------

@pytest.mark.unit
def test_429_still_raises_rate_limit_error(monkeypatch):
    # The reactive 429/418 handling must stay regardless of the proactive flag.
    set_config({"binance_proactive_backoff": True})
    monkeypatch.setattr(rl.time, "sleep", lambda s: None)
    monkeypatch.setattr(
        binance.requests, "get",
        _get_returning(_FakeBinanceResp(
            status_code=429, text="{}", json_data={}, headers={},
        )),
    )

    with pytest.raises(VendorRateLimitError):
        binance._http_get(
            "/fapi/v1/klines", {"symbol": "BTCUSDT"}, "BTCUSDT", "BTCUSDT"
        )


# --- config override --------------------------------------------------------

@pytest.mark.unit
def test_threshold_from_config_override():
    set_config({"binance_weight_threshold": 1000})
    rl.reset_for_test()  # force re-creation against the new config

    assert _limiter().weight_limit == 1000


# --- thread safety smoke ----------------------------------------------------

@pytest.mark.unit
def test_limiter_thread_safe_under_concurrent_observe_acquire(monkeypatch):
    # batch_concurrency=true runs K market analysts in a thread pool; the
    # weight state must not corrupt or deadlock. sleep is a no-op so threads
    # don't actually wait.
    monkeypatch.setattr(rl.time, "sleep", lambda s: None)
    lim = _limiter()

    errors = []

    def worker():
        try:
            for i in range(200):
                lim.observe(2000 if i % 3 == 0 else 10)  # toggles hot/cold
                lim.acquire()
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)

    assert all(t.is_alive() is False for t in threads)  # no deadlock
    assert errors == []
    assert lim.used >= 0  # state remained a sane int throughout
