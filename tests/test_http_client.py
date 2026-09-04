"""Process-global shared httpx client: singleton, trust_env, reset semantics.

Covers yialpha/llm_clients/http_client.py — ``get_shared_http_client()`` /
``reset_for_test()``. Transport-configuration only: nothing here dials the
network. Proxy handling is asserted through httpx's in-memory proxy mounts
(what ``trust_env=True`` populates from HTTP(S)_PROXY / NO_PROXY at client
construction), and the concurrency test counts constructor calls on a fake
client class instead of touching real sockets.
"""

from __future__ import annotations

import threading

import httpx
import pytest

import yialpha.llm_clients.http_client as http_client_module
from yialpha.llm_clients.http_client import (
    get_shared_http_client,
    reset_for_test,
)

# Both cases: dev machines commonly export upper- or lower-case proxy vars.
_PROXY_VARS = (
    "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY",
    "http_proxy", "https_proxy", "all_proxy", "no_proxy",
)


@pytest.fixture(autouse=True)
def _clean_proxy_env_and_singleton(monkeypatch):
    """Scrub ambient proxy vars (this repo's dev shells export some) and
    reset the process-wide client around every test — the module caches a
    global ``_client``, so tests must not leak instances or env into each
    other (same discipline as tests/test_glm_key_pool.py's pool reset)."""
    for var in _PROXY_VARS:
        monkeypatch.delenv(var, raising=False)
    reset_for_test()
    yield
    reset_for_test()


def _proxy_targets(client):
    """``(mount_pattern, proxy_host_or_None)`` for every mount httpx built."""
    targets = []
    for pattern, transport in client._mounts.items():
        host = None
        if transport is not None:
            pool = getattr(transport, "_pool", None)
            proxy_url = getattr(pool, "_proxy_url", None)
            if proxy_url is not None:
                raw = proxy_url.host
                host = raw.decode() if isinstance(raw, bytes) else raw
        targets.append((pattern.pattern, host))
    return targets


# --------------------------------------------------------------------------- #
# Singleton semantics
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_shared_client_is_singleton():
    first = get_shared_http_client()
    second = get_shared_http_client()
    assert first is second
    assert isinstance(first, httpx.Client)


@pytest.mark.unit
def test_client_is_built_with_trust_env_enabled():
    """The module contract: env-derived proxy routing must stay ON (it is how
    the SOCKS5 proxy from .env reaches the client, matching the SDKs)."""
    assert get_shared_http_client().trust_env is True


@pytest.mark.unit
def test_missing_httpx_falls_back_to_none(monkeypatch):
    monkeypatch.setattr(http_client_module, "_HAS_HTTPX", False)
    assert get_shared_http_client() is None


# --------------------------------------------------------------------------- #
# trust_env: proxy env vars shape the client's mounts
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_proxy_env_vars_are_picked_up(monkeypatch):
    monkeypatch.setenv("HTTP_PROXY", "http://proxy.example:9999")
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.example:9999")
    client = get_shared_http_client()

    targets = _proxy_targets(client)
    assert targets, "proxy env set but httpx built no proxy mounts"
    proxied_hosts = {host for _pattern, host in targets if host is not None}
    assert proxied_hosts == {"proxy.example"}


@pytest.mark.unit
def test_no_proxy_source_yields_no_proxy_mounts(monkeypatch):
    """With trust_env on but NO proxy source at all, httpx builds a plain
    direct client (no mounts). We stub urllib's ``getproxies`` (which httpx
    calls) rather than relying on a scrubbed environment alone: on Windows,
    ``getproxies()`` falls back to the system registry's proxy settings even
    with every *_proxy env var unset, so the naive assertion would not be
    hermetic on this repo's dev machines."""
    monkeypatch.setattr("httpx._utils.getproxies", lambda: {})
    client = get_shared_http_client()
    assert client.trust_env is True
    assert client._mounts == {}


@pytest.mark.unit
def test_no_proxy_excludes_host_from_proxied_routes(monkeypatch):
    monkeypatch.setenv("HTTP_PROXY", "http://proxy.example:9999")
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.example:9999")
    monkeypatch.setenv("NO_PROXY", "internal.host")
    client = get_shared_http_client()

    targets = _proxy_targets(client)
    direct = [pattern for pattern, host in targets if host is None]
    proxied = {host for _pattern, host in targets if host is not None}
    assert any("internal.host" in pattern for pattern in direct)
    assert proxied == {"proxy.example"}


# --------------------------------------------------------------------------- #
# reset_for_test: close the old, rebuild fresh
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_reset_closes_old_client_and_rebuilds_new():
    old = get_shared_http_client()
    assert old.is_closed is False

    reset_for_test()
    assert old.is_closed is True  # sockets of the stale client are released

    new = get_shared_http_client()
    assert new is not old
    assert new.is_closed is False
    # And the new instance is itself the cached singleton again.
    assert get_shared_http_client() is new


@pytest.mark.unit
def test_reset_without_a_client_is_a_noop():
    reset_for_test()  # nothing cached yet: must not raise
    client = get_shared_http_client()
    assert client is not None
    reset_for_test()
    assert client.is_closed is True


# --------------------------------------------------------------------------- #
# Concurrency: first concurrent get() creates exactly one client
# --------------------------------------------------------------------------- #


class _CountingClient:
    """httpx.Client stand-in that records every construction."""

    instances = []
    lock = threading.Lock()

    def __init__(self, trust_env=False):
        self.trust_env = trust_env
        self.closed = False
        with _CountingClient.lock:
            _CountingClient.instances.append(self)

    def close(self):
        self.closed = True


@pytest.mark.unit
def test_concurrent_first_get_creates_exactly_one_client(monkeypatch):
    monkeypatch.setattr(http_client_module.httpx, "Client", _CountingClient)
    _CountingClient.instances = []

    workers_n = 16
    barrier = threading.Barrier(workers_n)
    results = [None] * workers_n

    def worker(slot: int) -> None:
        barrier.wait()  # everyone releases at once: maximal contention
        results[slot] = get_shared_http_client()

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(workers_n)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    # Idempotent under the race: one construction, one shared instance.
    assert len(_CountingClient.instances) == 1
    assert all(result is _CountingClient.instances[0] for result in results)
    assert get_shared_http_client() is _CountingClient.instances[0]

    # The module-level reset still tears the winner down.
    reset_for_test()
    assert _CountingClient.instances[0].closed is True
