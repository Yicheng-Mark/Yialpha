"""Process-global LLM rate limiter (Phase C).

When ``YIAGENTS_LLM_RATE_LIMITER=true``, every LLM client in the process shares
ONE limiter per ``(provider, rpm)``. Sharing across all worker graphs is the
whole point: a per-graph limiter would let K batch workers each fire at the full
rate and still trip the provider's 429 ceiling. The limiter throttles REQUEST
RATE only — it never touches prompts, reasoning params, model, or temperature,
so the per-ticker analysis stays byte-equivalent to an unthrottled run (just
slower). Off by default; enable only after measuring the ceiling.
"""
from __future__ import annotations

import threading

from langchain_core.rate_limiters import InMemoryRateLimiter

_limiters: dict[tuple[str, int], InMemoryRateLimiter] = {}
_guard = threading.Lock()


def get_shared_rate_limiter(provider: str, rpm: int) -> InMemoryRateLimiter:
    """Return the process-wide limiter for ``(provider, rpm)``, creating it once.

    Thread-safe and idempotent: the first caller builds the limiter, every later
    caller (including other worker graphs in the same batch) reuses the same
    instance so the rate budget is shared, not multiplied.
    """
    rpm = max(1, int(rpm))
    key = (provider.lower(), rpm)
    with _guard:
        lim = _limiters.get(key)
        if lim is None:
            # Steady-state rate from the per-minute budget. The token bucket
            # (max_bucket_size ~ 2s of budget) allows brief bursts so a short
            # stall doesn't strand capacity, but sustained demand is capped.
            rps = rpm / 60.0
            lim = InMemoryRateLimiter(
                requests_per_second=rps,
                check_every_n_seconds=0.1,
                max_bucket_size=max(1, int(rps * 2)),
            )
            _limiters[key] = lim
        return lim


def reset_for_test() -> None:
    """Drop cached limiters (tests only — fresh process state per case)."""
    with _guard:
        _limiters.clear()
