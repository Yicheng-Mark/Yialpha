"""Tiny shared retry helper for transient network failures.

One retry with linear backoff — deliberately minimal. This is NOT a
resilience framework: the router already degrades typed vendor errors to
sentinels, and run_robust's per-ticker rerun is the outer safety net. The
goal is only to absorb single-packet hiccups (reset connections, DNS blips)
so a batch run does not degrade a whole category over one dropped SYN.

Vendors with richer contracts keep their own loops: binance honours
Retry-After + weight headers, yfinance has ``yf_retry``, eastmoney maps its
transport errors mid-loop. New simple HTTPS vendors should use this.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


def with_transient_retry(
    fetch: Callable[[], T],
    *,
    vendor: str,
    retry_on: tuple[type[BaseException], ...],
    retries: int = 1,
    backoff_s: float = 2.0,
) -> T:
    """Call ``fetch``, retrying once (by default) on the given exception types.

    ``retry_on`` must list transport-level exceptions only (e.g.
    ``requests.exceptions.ConnectionError``/``Timeout``) — never application
    errors, whose retry would just duplicate a deterministic failure.
    """
    last: BaseException | None = None
    for attempt in range(retries + 1):
        try:
            return fetch()
        except retry_on as exc:
            last = exc
            if attempt < retries:
                logger.warning(
                    "%s: transient transport failure (%r); retrying (%d/%d)",
                    vendor, exc, attempt + 1, retries,
                )
                time.sleep(backoff_s * (attempt + 1))
    assert last is not None  # only reachable when fetch raised at least once
    raise last
