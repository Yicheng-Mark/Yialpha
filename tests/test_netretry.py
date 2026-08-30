"""Tests for the shared transient-network retry helper (dataflows/netretry)."""

from __future__ import annotations

import pytest

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
