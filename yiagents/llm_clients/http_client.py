"""Process-global shared httpx client for LLM calls (P1a).

When ``YIAGENTS_HTTP_KEEPALIVE=true``, every LLM client in the process shares
ONE ``httpx.Client`` so the connection to the provider (and the SOCKS5 proxy) is
kept alive and reused across all K worker graphs, instead of each call opening a
fresh TLS+proxy handshake. This is the transport change that lets ``--workers``
scale past ~3 without saturating the proxy's connection table.

Transport-only: it changes nothing about what is sent to the model — no prompt,
reasoning, or temperature effect. Off by default; the provider SDK's own
per-client httpx pool is used otherwise (today's behaviour).

``httpx.Client`` (sync) is thread-safe, so the worker graphs may share one
instance. It reads proxy settings from the environment (``trust_env=True``), so
the SOCKS5 proxy in ``.env`` is honored exactly as the provider SDK already does
it — no new proxy configuration here.
"""
from __future__ import annotations

import threading

try:
    import httpx

    _HAS_HTTPX = True
except ImportError:  # pragma: no cover - httpx ships with langchain-openai
    _HAS_HTTPX = False

_client = None
_guard = threading.Lock()


def get_shared_http_client():
    """Return the process-wide shared httpx.Client, creating it once.

    Returns ``None`` if httpx is unavailable, so the caller can fall back to the
    provider SDK's default client (the kwargs simply won't include ``http_client``).
    """
    if not _HAS_HTTPX:
        return None
    global _client
    with _guard:
        if _client is None:
            # trust_env=True (the default) makes httpx read HTTP_PROXY/HTTPS_PROXY
            # from the environment, matching how the provider SDK already routes.
            _client = httpx.Client(trust_env=True)
        return _client


def reset_for_test() -> None:
    """Drop and close the cached client (tests only)."""
    global _client
    with _guard:
        if _client is not None:
            _client.close()
            _client = None
