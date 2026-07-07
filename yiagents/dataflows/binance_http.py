"""Process-global shared requests.Session for Binance market-data calls.

When ``YIAGENTS_BINANCE_HTTP_KEEPALIVE=true``, every Binance GET in the process
shares ONE ``requests.Session`` so the TLS / SOCKS5-proxy connection is kept
alive and reused across calls (and across K worker graphs sharing one IP),
instead of each call opening a fresh ``requests.get`` with a full TLS+proxy
handshake. Mirrors :mod:`yiagents.llm_clients.http_client` (the LLM-side
shared ``httpx.Client``) for the data-vendor side.

Transport-only: it changes nothing about what is sent to Binance — same URL,
params, headers, proxies, timeout → the same response bytes. Off by default;
the per-call ``requests.get`` (today's behaviour) is used otherwise.

Thread-safety: a ``requests.Session`` is safe for concurrent read-only GETs in
our usage — the underlying urllib3 ``PoolManager`` is thread-safe, and the
public market-data endpoints carry no auth/cookies so there is no session-level
state to race on. This matches the already-trusted LLM keepalive pattern.
``requests.Session`` defaults to ``trust_env=True``, so it reads
``HTTP_PROXY``/``HTTPS_PROXY``/``NO_PROXY`` from the environment exactly as
:func:`yiagents.dataflows.binance._proxies` already does — the SOCKS5 proxy in
``.env`` is honoured with no new proxy configuration here.
"""
from __future__ import annotations

import threading

import requests

_session = None
_guard = threading.Lock()


def get_shared_binance_session() -> requests.Session:
    """Return the process-wide shared ``requests.Session``, creating it once.

    Thread-safe and idempotent. The session is reused for every later Binance
    call in the process so the connection pool (and the SOCKS5 proxy connection)
    is shared, not re-opened per call.
    """
    global _session
    with _guard:
        if _session is None:
            # trust_env=True (the default) makes requests read
            # HTTP_PROXY/HTTPS_PROXY/NO_PROXY from the environment, matching how
            # binance._proxies() routes through the SOCKS5 proxy.
            _session = requests.Session()
        return _session


def has_shared_session() -> bool:
    """True iff the shared session has been lazily created (test introspection)."""
    with _guard:
        return _session is not None


def reset_for_test() -> None:
    """Drop and close the cached session (tests only — fresh process state)."""
    global _session
    with _guard:
        if _session is not None:
            _session.close()
            _session = None
