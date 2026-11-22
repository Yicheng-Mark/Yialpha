"""Tiny shared retry helper for transient network failures.

One retry with linear backoff — deliberately minimal. This is NOT a
resilience framework: the router already degrades typed vendor errors to
sentinels, and run_robust's per-ticker rerun is the outer safety net. The
goal is only to absorb single-packet hiccups (reset connections, DNS blips)
so a batch run does not degrade a whole category over one dropped SYN.

Vendors with richer contracts keep their own loops: binance honours
Retry-After + weight headers, yfinance has ``yf_retry``. New simple HTTPS
vendors should use this.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import TypeVar

import requests

logger = logging.getLogger(__name__)

T = TypeVar("T")


def _is_transient_http_error(exc: BaseException) -> bool:
    """Whether an ``HTTPError`` is transient: a server-side 5xx only.

    A 5xx (502/503/504 gateway blip) often heals on an immediate retry; a 4xx
    (429/418 rate limits, 403 blocks, 404s) is deterministic for a plain
    re-request — retrying it would duplicate a deterministic failure while
    richer per-vendor contracts (key rotation, ``Retry-After``) stay in
    charge. An ``HTTPError`` without a response attached is conservatively
    treated as non-transient.
    """
    return (
        isinstance(exc, requests.exceptions.HTTPError)
        and exc.response is not None
        and exc.response.status_code >= 500
    )


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
    ``requests.exceptions.HTTPError`` is the one CONDITIONALLY transient type
    (round-11): when a caller lists it in ``retry_on``, only a 5xx response is
    retried (:func:`_is_transient_http_error`); 4xx re-raises immediately.
    """
    last: BaseException | None = None
    for attempt in range(retries + 1):
        try:
            return fetch()
        except retry_on as exc:
            if isinstance(exc, requests.exceptions.HTTPError) and not (
                _is_transient_http_error(exc)
            ):
                # 4xx / response-less HTTPError: deterministic for an
                # immediate re-request — not this loop's business.
                raise
            last = exc
            if attempt < retries:
                logger.warning(
                    "%s: transient transport failure (%r); retrying (%d/%d)",
                    vendor, exc, attempt + 1, retries,
                )
                time.sleep(backoff_s * (attempt + 1))
    assert last is not None  # only reachable when fetch raised at least once
    raise last
