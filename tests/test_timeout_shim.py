"""Tests for the requests default-timeout shim (dataflows/timeout_shim).

Verifies: calls without a timeout get the default inside the window; explicit
timeouts are untouched; the patch is restored on normal exit AND on exception;
nesting is reference-counted (outermost value wins, original restored only at
depth 0); non-positive seconds is rejected.
"""

from __future__ import annotations

import pytest
import requests

from yiagents.dataflows import timeout_shim as ts


def _session_request_original():
    return requests.sessions.Session.request


@pytest.mark.unit
def test_timeout_filled_inside_window():
    seen = {}
    original = _session_request_original()

    def fake(self, method, url, **kwargs):
        seen.update(kwargs)
        return None

    requests.sessions.Session.request = fake
    try:
        with ts.default_request_timeout(20):
            s = requests.Session()
            s.get("http://example.invalid/x")
    finally:
        requests.sessions.Session.request = original
    assert seen.get("timeout") == 20


@pytest.mark.unit
def test_explicit_timeout_untouched():
    seen = {}
    original = _session_request_original()

    def fake(self, method, url, **kwargs):
        seen.update(kwargs)
        return None

    requests.sessions.Session.request = fake
    try:
        with ts.default_request_timeout(20):
            s = requests.Session()
            s.get("http://example.invalid/x", timeout=5)
    finally:
        requests.sessions.Session.request = original
    assert seen.get("timeout") == 5


@pytest.mark.unit
def test_restored_after_window():
    original = _session_request_original()
    with ts.default_request_timeout(20):
        pass
    assert requests.sessions.Session.request is original
    # And a post-window call without timeout gets no timeout injected.
    seen = {}

    def fake(self, method, url, **kwargs):
        seen.update(kwargs)
        return None

    requests.sessions.Session.request = fake
    try:
        s = requests.Session()
        s.get("http://example.invalid/x")
    finally:
        requests.sessions.Session.request = original
    assert "timeout" not in seen or seen["timeout"] is None


@pytest.mark.unit
def test_restored_after_exception_inside_window():
    original = _session_request_original()
    with pytest.raises(RuntimeError, match="boom"), ts.default_request_timeout(20):
        raise RuntimeError("boom")
    assert requests.sessions.Session.request is original
    assert ts._depth == 0
    assert ts._original_request is None


@pytest.mark.unit
def test_nested_windows_outermost_value_and_single_restore():
    original = _session_request_original()
    seen = {}

    def fake(self, method, url, **kwargs):
        seen.update(kwargs)
        return None

    # Install the observer BEFORE the window so the shim wraps it (installing
    # inside would replace the wrapper itself and see no injection).
    requests.sessions.Session.request = fake
    try:
        with ts.default_request_timeout(20), ts.default_request_timeout(99):
            s = requests.Session()
            s.get("http://example.invalid/x")
    finally:
        requests.sessions.Session.request = original
    assert seen.get("timeout") == 20  # outermost window's value applies
    assert requests.sessions.Session.request is original
    assert ts._depth == 0


@pytest.mark.unit
def test_non_positive_timeout_rejected():
    with pytest.raises(ValueError), ts.default_request_timeout(0):
        pass


@pytest.mark.unit
def test_akshare_direct_connect_applies_timeout(monkeypatch):
    """The akshare vendor's _direct_connect window fills missing timeouts."""
    import os

    from yiagents.dataflows import akshare_vendor as akv

    monkeypatch.setenv("HTTP_PROXY", "socks5h://127.0.0.1:1080")
    original = _session_request_original()
    seen = {}

    def fake(self, method, url, **kwargs):
        seen.update(kwargs)
        return None

    requests.sessions.Session.request = fake  # before the window: shim wraps it
    try:
        with akv._direct_connect():
            assert "HTTP_PROXY" not in os.environ  # popped inside the window
            s = requests.Session()
            s.get("http://example.invalid/x")
    finally:
        requests.sessions.Session.request = original
    assert os.environ.get("HTTP_PROXY") == "socks5h://127.0.0.1:1080"  # restored
    assert requests.sessions.Session.request is original
    assert seen.get("timeout") == akv._TIMEOUT_S
