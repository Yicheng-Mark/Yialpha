"""Rotating API-key pool for OpenAI-compatible LLM providers (GLM Coding Plan).

Subscription-plan keys (Zhipu's GLM Coding Plan in particular) are quota-capped
per key — concurrency limits and per-5h prompt windows — while the trading
graph issues a few dozen LLM calls per ticker across three model channels. One
key means one shared cap; N keys multiply it. This module pools them:

* ``ZHIPU_API_KEYS`` / ``ZHIPU_CN_API_KEYS`` hold a comma-separated pool; the
  single-key ``ZHIPU_API_KEY`` / ``ZHIPU_CN_API_KEY`` var (api_key_env's map)
  is merged in, deduped, order-stable — the same contract as the Tavily pool
  (``dataflows/tavily.py``).
* ``get_llm`` attaches one shared ``httpx.Client`` per pooled provider whose
  ``RotatingKeyAuth`` picks the NEXT live key for EVERY request, so load
  spreads evenly across the quick/deep/debate channels instead of pinning one
  key per channel. (langchain's ChatOpenAI fixes ``api_key`` at construction,
  which is why the rotation lives in the transport, not the client.)
* A key answering 401/403 is removed for the rest of the process (auth does
  not heal mid-run) and the SAME request is re-sent on the next key inside the
  auth flow. 429 (quota window / concurrency cap) cools a key down for
  ``YIALPHA_LLM_KEY_COOLDOWN_S`` seconds (default 120) — burst limits recover
  in seconds, exhausted 5h windows keep failing out of the pool cheaply. When
  every key is cooling down, the soonest-to-revive key is served anyway: the
  request still goes out and the SDK/langchain error path decides, rather than
  the auth layer manufacturing a failure.

Transport-only: no prompt, model, or reasoning parameter is touched. The pool
machinery engages ONLY when the provider's ``*_API_KEYS`` env var is set and
non-empty; single-key setups are byte-equivalent to today.
"""
from __future__ import annotations

import logging
import os
import threading
import time

logger = logging.getLogger(__name__)

try:
    import httpx

    _HAS_HTTPX = True
except ImportError:  # pragma: no cover - httpx ships with langchain-openai
    _HAS_HTTPX = False

#: provider -> comma-separated pool env var. The single-key companion var
#: stays in api_key_env.PROVIDER_API_KEY_ENV (the single source the CLI
#: prompt consults); only the *pool* var lives here.
PROVIDER_API_KEYS_ENV: dict[str, str] = {
    # GLM Coding Plan subscriptions are sold per key on BigModel (China) and
    # Z.AI (international); accounts cannot share credentials across regions.
    "glm":    "ZHIPU_API_KEYS",
    "glm-cn": "ZHIPU_CN_API_KEYS",
}

#: How long a 429'd key sits out before it is retried; override with
#: YIALPHA_LLM_KEY_COOLDOWN_S. Long enough to route traffic at the healthy
#: keys during a burst, short enough that a recovering key returns quickly.
DEFAULT_COOLDOWN_S = 120.0

#: Statuses that mean "this KEY is the problem" (auth/quota), not "the request
#: is bad". Everything else propagates untouched to the SDK's own handling.
_KEY_FAILURE_STATUSES = (401, 403, 429)


class _PoolState:
    """Live rotation state for one provider's pool (process-wide).

    Not a ContextVar: unlike the Tavily run-budget counters, the pool outlives
    one graph run (batch workers construct many graphs per process) and the
    "dead key" knowledge is exactly what should NOT reset between runs of a
    long-lived process. Thread-safe via the module lock.
    """

    def __init__(self, size: int, cooldown_s: float):
        self.size = size
        self.cooldown_s = cooldown_s
        self.cursor = 0
        #: idx -> monotonic time after which a 429'd key is live again.
        self.dead_until: dict[int, float] = {}
        #: 401/403'd keys: out for the rest of the process.
        self.permanent: set[int] = set()

    def is_live(self, idx: int) -> bool:
        if idx in self.permanent:
            return False
        return self.dead_until.get(idx, 0.0) <= time.monotonic()

    def pick(self) -> int | None:
        """Next live index round-robin, or None when every key is out."""
        for _ in range(self.size):
            idx = self.cursor % self.size
            self.cursor += 1
            if self.is_live(idx):
                return idx
        return None

    def force_pick(self) -> int | None:
        """When nothing is live: the key that revives soonest.

        Prefer cooling-down keys (they recover); fall back to the first
        permanently-dead one so the request still reaches the API and fails
        with the provider's real status code instead of a synthetic error.
        """
        cooling = sorted(
            (until, idx) for idx, until in self.dead_until.items() if idx not in self.permanent
        )
        if cooling:
            return cooling[0][1]
        if self.permanent:
            return min(self.permanent)
        return None

    def mark(self, idx: int, status: int) -> str:
        """Record a key failure; returns a human-readable consequence."""
        if status in (401, 403):
            self.permanent.add(idx)
            return "removed for the rest of the process (auth failure)"
        self.dead_until[idx] = time.monotonic() + self.cooldown_s
        return f"cooling down for {self.cooldown_s:.0f}s (quota/concurrency)"


_states: dict[str, _PoolState] = {}
_clients: dict[str, httpx.Client] = {}
_lock = threading.Lock()


def api_key_pool(provider: str) -> list[str]:
    """Active key pool: ``*_API_KEYS`` entries + the single-key env var.

    Order-stable, exact-duplicate-removed. Public so ``config-check`` can
    show the pool size (never the values) and tests can pin the parse.
    """
    provider = provider.lower()
    pool_env = PROVIDER_API_KEYS_ENV.get(provider)
    if pool_env is None:
        return []
    keys: list[str] = []
    raw = os.environ.get(pool_env)
    if raw:
        for entry in raw.split(","):
            key = entry.strip()
            if key and key not in keys:  # a pasted duplicate must not double its rotation weight
                keys.append(key)
    single = (os.environ.get(_single_key_env(provider)) or "").strip()
    if single and single not in keys:
        keys.append(single)
    return keys


def _single_key_env(provider: str) -> str:
    from .api_key_env import get_api_key_env

    return get_api_key_env(provider) or ""


def has_pool(provider: str) -> bool:
    """Whether the provider's pool env var is set to something non-empty.

    This is the gate for engaging the rotating transport: a provider with only
    the single-key var set behaves exactly as before (SDK default client).
    """
    pool_env = PROVIDER_API_KEYS_ENV.get(provider.lower())
    return bool(pool_env and os.environ.get(pool_env, "").strip())


def cooldown_seconds() -> float:
    """Effective 429 cooldown; YIALPHA_LLM_KEY_COOLDOWN_S overrides the default."""
    raw = os.getenv("YIALPHA_LLM_KEY_COOLDOWN_S")
    if raw is None:
        return DEFAULT_COOLDOWN_S
    try:
        return max(1.0, float(raw))
    except ValueError:
        logger.warning(
            "Ignoring non-numeric YIALPHA_LLM_KEY_COOLDOWN_S=%r; using %.0fs.",
            raw, DEFAULT_COOLDOWN_S,
        )
        return DEFAULT_COOLDOWN_S


class RotatingKeyAuth(httpx.Auth if _HAS_HTTPX else object):  # type: ignore[misc,valid-type]
    """httpx auth that picks a pooled key per request and rotates on failure.

    The provider SDK already stamped an ``Authorization`` header from the
    construction-time key; every request re-stamps it here, so the header is
    decided per call, not per client. On 401/403/429 the failed body is
    drained and the SAME request is re-sent on the next live key — one
    rotation cycle per request at most (len(keys) attempts), after which the
    last response propagates to the SDK untouched.
    """

    # auth_flow reads failed response bodies before re-sending; tell httpx so
    # streaming responses still expose a readable body to the auth flow.
    requires_response_body = True

    def __init__(self, keys: list[str], state: _PoolState):
        self.keys = keys
        self.state = state

    def auth_flow(self, request):  # type: ignore[no-untyped-def]
        for _ in range(len(self.keys)):
            with _lock:
                idx = self.state.pick()
                forced = False
                if idx is None:
                    idx = self.state.force_pick()
                    forced = idx is not None
                    if idx is None:
                        return  # empty pool guard; SDK surfaces the 401
                request.headers["Authorization"] = f"Bearer {self.keys[idx]}"

            response = yield request
            if response.status_code not in _KEY_FAILURE_STATUSES:
                return

            # Drain the failed body before re-using the connection.
            response.read()
            with _lock:
                detail = self.state.mark(idx, response.status_code)
            logger.warning(
                "key_pool: key %d/%d for this provider got HTTP %d — %s; %s.",
                idx + 1, len(self.keys), response.status_code, detail,
                "retrying on the next key"
                if not forced
                else "no live key left, serving it anyway next time",
            )
            if forced:
                # Even the force-picked key refused: rotating further cannot
                # help — let this response be the final one.
                return
        # Every key refused this request once; the last response stands.


def get_pool_http_client(provider: str):
    """Shared ``httpx.Client`` whose auth rotates the provider's key pool.

    Returns ``None`` when httpx is unavailable or the provider has no pool env
    set, so the caller keeps the SDK's default transport (today's behaviour).
    One client per provider, process-wide: like the P1a keepalive client it
    pools TLS connections, and every LLM channel in the process shares both
    the pool and the connections.
    """
    if not _HAS_HTTPX or not has_pool(provider):
        return None
    keys = api_key_pool(provider)
    if not keys:
        return None
    provider = provider.lower()
    with _lock:
        if provider not in _clients:
            state = _states.get(provider)
            if state is None or state.size != len(keys):
                state = _PoolState(len(keys), cooldown_seconds())
                _states[provider] = state
            _clients[provider] = httpx.Client(
                auth=RotatingKeyAuth(keys, state), trust_env=True
            )
        return _clients[provider]


def reset_for_test() -> None:
    """Drop pooled clients + rotation state (tests only)."""
    with _lock:
        for client in _clients.values():
            client.close()
        _clients.clear()
        _states.clear()
