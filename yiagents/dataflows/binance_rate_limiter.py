"""Process-global Binance IP-weight rate limiter (header-driven backoff).

Binance throttles by *request weight*, not request count, and the budget is
per-IP per-product-line in a rolling 1-minute window. The server reports the
current cumulative consumption in the ``X-MBX-USED-WEIGHT-1M`` response header
on every reply, so the client can track the budget *as the server sees it* and
back off before tripping a 429 — instead of only reacting to one. This module
owns that tracking; :mod:`yiagents.dataflows.binance` consults it.

One limiter per product line (``"fapi"`` for USDT-M perpetuals today; spot will
be a separate budget when that vendor lands — Binance counts spot and futures
weight independently against the same IP). The limiter is shared across every
worker graph in the process, so K batch workers drawing on one IP count against
ONE budget rather than K. It is thread-safe (``batch_concurrency=true`` runs K
market analysts in a thread pool) but NOT cross-process — ``run_robust``'s
per-ticker subprocesses share an IP but no budget today, and the reactive
429/418 path in ``binance._http_get`` remains the floor either way.

The limiter only changes *when* a request is issued, never what data comes back,
so a per-ticker analysis stays byte-equivalent to an unthrottled run (possibly
slower). It is consulted only when ``binance_proactive_backoff`` is on (off by
default = byte-equivalent to today).
"""
from __future__ import annotations

import logging
import threading
import time

from .config import get_config

logger = logging.getLogger(__name__)

# Rolling window Binance enforces server-side (seconds). An observed weight
# value only counts against the budget for this long; once it ages out, the same
# observation no longer justifies backing off.
_WINDOW_S = 60.0

# Trip the backoff at this fraction of the limit. Binance's 429 trips at the
# full limit; backing off a little early leaves headroom for the in-flight
# request's own weight (which the server has not counted yet) and for other
# traffic sharing the IP (a second batch worker, or the VPN exit in use by
# something else). Only the limit itself is user-configurable
# (YIAGENTS_BINANCE_WEIGHT_THRESHOLD); the trip fraction is a conservative
# internal constant.
_HIGH_WATER_FRACTION = 0.8


class BinanceWeightLimiter:
    """Header-driven, thread-safe tracker for Binance's rolling IP-weight budget.

    :meth:`observe` records the server-reported used weight after each GET;
    :meth:`acquire` blocks before the next GET if the last in-window observation
    was at/above the high-water mark. The server is the single source of truth
    for consumed weight, so the client cannot over- or under-count — it reacts
    to what each response header reports.
    """

    def __init__(self, weight_limit: int, high_water_fraction: float = _HIGH_WATER_FRACTION):
        self._limit = max(1, int(weight_limit))
        self._trip = self._limit * float(high_water_fraction)
        self._lock = threading.Lock()
        # Last observed (used_weight, monotonic_ts). A ts of 0.0 means "never
        # observed"; acquire() treats a cold budget as not-hot so the very first
        # request of the process is never delayed on a stale guess.
        self._used = 0
        self._used_ts = 0.0

    @property
    def weight_limit(self) -> int:
        """The per-minute weight ceiling this limiter backs off against."""
        return self._limit

    @property
    def used(self) -> int:
        """Last server-reported used weight (0 if none observed yet)."""
        with self._lock:
            return self._used

    def observe(self, used_weight: int | None) -> None:
        """Record the latest ``X-MBX-USED-WEIGHT-1M`` value (call after each GET).

        ``None``/unparseable values are ignored — the next response carries a
        fresh value, so a single missing header is harmless.
        """
        if used_weight is None:
            return
        try:
            w = int(used_weight)
        except (TypeError, ValueError):
            return
        with self._lock:
            self._used = w
            self._used_ts = time.monotonic()

    def acquire(self) -> None:
        """Block before the next GET if the budget is hot.

        If the last observation was at/above the high-water mark AND it is still
        inside the rolling window, sleep until that observation has aged out (so
        the server's window can only have rolled down), then return to let the
        caller issue the request and re-observe. A one-shot wait — no busy-loop;
        the next ``acquire()`` re-evaluates against a fresh header.

        ``time.sleep`` is resolved at call time (not bound as a default arg) so
        tests can patch ``time.sleep`` on the module to intercept the wait.
        """
        with self._lock:
            if self._used < self._trip or self._used_ts == 0.0:
                return
            age = time.monotonic() - self._used_ts
            if age >= _WINDOW_S:
                return  # observation already rolled out of the window
            wait = _WINDOW_S - age
            trip_at = self._trip
            used = self._used
            limit = self._limit
        # Sleep OUTSIDE the lock so concurrent threads are not serialized behind
        # one sleeper; each that trips independently waits out its own estimate.
        logger.info(
            "Binance IP weight %d >= %.0f (%.0f%% of limit %d); backing off %.1fs "
            "for the rolling window to recede",
            used, trip_at, _HIGH_WATER_FRACTION * 100, limit, wait,
        )
        time.sleep(wait)


# Process-wide registry: one limiter per product line. Binance's IP-weight
# budget is per-IP, so separate limiters per worker graph would each draw on the
# same budget and still trip 429 together — hence the process-global shared
# instance, mirroring yiagents/llm_clients/rate_limiter.py.
_limiters: dict[str, BinanceWeightLimiter] = {}
_guard = threading.Lock()


def get_binance_weight_limiter(product: str = "fapi") -> BinanceWeightLimiter:
    """Return the process-wide limiter for ``product``, creating it once.

    Thread-safe and idempotent. Reads ``binance_weight_threshold`` from config at
    creation time (lazy — first request, not import — so this module stays
    import-side-effect-free and a config change is honoured for a fresh product
    key without re-importing). The created instance is reused for every later
    caller in the process so the weight budget is shared, not multiplied.
    """
    with _guard:
        lim = _limiters.get(product)
        if lim is None:
            limit = int(get_config().get("binance_weight_threshold", 2400))
            lim = BinanceWeightLimiter(limit)
            _limiters[product] = lim
        return lim


def reset_for_test() -> None:
    """Drop cached limiters (tests only — fresh process state per case)."""
    with _guard:
        _limiters.clear()
