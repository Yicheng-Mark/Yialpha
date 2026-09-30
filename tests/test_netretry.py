"""Tests for the shared transient-network retry helper (dataflows/netretry)."""

from __future__ import annotations

import pytest
import requests

from yialpha.dataflows.netretry import with_transient_retry


@pytest.mark.unit
def test_success_first_try_no_retry():
    calls = {"n": 0}

    def fetch() -> str:
        calls["n"] += 1
        return "ok"

    assert with_transient_retry(fetch, vendor="t", retry_on=(ConnectionError,)) == "ok"
    assert calls["n"] == 1


@pytest.mark.unit
def test_retries_once_then_succeeds(monkeypatch):
    monkeypatch.setattr("yialpha.dataflows.netretry.time.sleep", lambda s: None)
    calls = {"n": 0}

    def fetch() -> str:
        calls["n"] += 1
        if calls["n"] == 1:
            raise ConnectionError("reset by peer")
        return "ok"

    assert with_transient_retry(fetch, vendor="t", retry_on=(ConnectionError,)) == "ok"
    assert calls["n"] == 2


@pytest.mark.unit
def test_exhausted_retries_reraises_last(monkeypatch):
    monkeypatch.setattr("yialpha.dataflows.netretry.time.sleep", lambda s: None)
    calls = {"n": 0}

    def fetch():
        calls["n"] += 1
        raise TimeoutError(f"attempt {calls['n']}")

    with pytest.raises(TimeoutError, match="attempt 2"):
        with_transient_retry(fetch, vendor="t", retry_on=(TimeoutError,))
    assert calls["n"] == 2  # 1 try + 1 retry


@pytest.mark.unit
def test_non_retryable_exception_propagates_immediately():
    calls = {"n": 0}

    def fetch():
        calls["n"] += 1
        raise ValueError("application error")

    with pytest.raises(ValueError):
        with_transient_retry(fetch, vendor="t", retry_on=(ConnectionError,))
    assert calls["n"] == 1  # never retried


# --------------------------------------------------------------------------- #
# Round-11 — HTTPError is conditionally transient: only a 5xx retries
# --------------------------------------------------------------------------- #
class _Resp:
    """Minimal requests.Response stand-in carrying a status code."""

    def __init__(self, status: int):
        self.status_code = status


def _http_error(status: int) -> requests.exceptions.HTTPError:
    exc = requests.exceptions.HTTPError(f"{status} error")
    exc.response = _Resp(status)
    return exc


@pytest.mark.unit
def test_5xx_http_error_retried_then_succeeds(monkeypatch):
    """A server-side 5xx (raise_for_status on a 503 gateway blip) heals on an
    immediate retry — previously no caller's retry_on matched HTTPError at
    all, so every 502/503 propagated on the first attempt."""
    monkeypatch.setattr("yialpha.dataflows.netretry.time.sleep", lambda s: None)
    calls = {"n": 0}

    def fetch() -> str:
        calls["n"] += 1
        if calls["n"] == 1:
            raise _http_error(503)
        return "ok"

    assert with_transient_retry(
        fetch, vendor="t", retry_on=(requests.exceptions.HTTPError,)
    ) == "ok"
    assert calls["n"] == 2


@pytest.mark.unit
def test_4xx_http_error_never_retried():
    """429/418/4xx are deterministic for a plain re-request: they re-raise
    immediately (rate limits belong to richer per-vendor contracts)."""
    calls = {"n": 0}

    def fetch():
        calls["n"] += 1
        raise _http_error(429)

    with pytest.raises(requests.exceptions.HTTPError):
        with_transient_retry(
            fetch, vendor="t",
            retry_on=(requests.exceptions.ConnectionError,
                      requests.exceptions.HTTPError),
        )
    assert calls["n"] == 1  # no retry, no backoff sleep


@pytest.mark.unit
def test_http_error_without_response_not_retried():
    calls = {"n": 0}

    def fetch():
        calls["n"] += 1
        raise requests.exceptions.HTTPError("no response attached")

    with pytest.raises(requests.exceptions.HTTPError):
        with_transient_retry(
            fetch, vendor="t", retry_on=(requests.exceptions.HTTPError,)
        )
    assert calls["n"] == 1  # conservatively non-transient


@pytest.mark.unit
def test_http_error_not_listed_in_retry_on_untouched():
    """A caller that deliberately omits HTTPError (binance_square's anti-bot
    403 doubling concern; tavily's key rotation) keeps the old behaviour: the
    exception is not caught by this loop at all."""
    calls = {"n": 0}

    def fetch():
        calls["n"] += 1
        raise _http_error(503)

    with pytest.raises(requests.exceptions.HTTPError):
        with_transient_retry(
            fetch, vendor="t", retry_on=(ConnectionError,)
        )
    assert calls["n"] == 1
