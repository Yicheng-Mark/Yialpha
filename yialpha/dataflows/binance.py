"""Binance USDT-M perpetual market-data vendor (Track A, analysis-only).

Read-only public GET endpoints (no API key, no trading):

  - /fapi/v1/klines            daily OHLCV for the market analyst
  - /fapi/v1/markPriceKlines   mark-price OHLCV (liquidation anchor)
  - /fapi/v1/fundingRate        funding-rate history (perp cost-of-carry)
  - /fapi/v1/openInterest       live open interest
  - /fapi/v1/premiumIndex       mark/index price + current funding snapshot
  - /futures/data/openInterestHist  daily open-interest history

Returns CSV-shaped ``str`` (header + ``df.to_csv()``) so they slot into the
existing ``route_to_vendor`` plumbing exactly like the yfinance vendors. On
any failure they raise the typed errors from :mod:`yialpha.dataflows.errors`
so the router can degrade a flaky/unsupported symbol to a sentinel rather than
aborting the run.

Design constraints:
  - Module import is side-effect free: no top-level ``requests.Session``, no
    env read, no network. The functions read ``HTTP_PROXY``/``HTTPS_PROXY`` at
    call time so they pick up the proxy injected by the run's ``.env``.
  - ``timeout=(connect, read) = (5, 30)`` gives full control over the read
    phase (this repo has a mid-response ``ssl.read`` hang precedent), with the
    outer ``run_robust`` watchdog as the OS-level backstop.
  - No new dependencies: plain ``requests`` + the already-present PySocks for
    the SOCKS5 proxy. Track B (execution) will bring the official SDK.
"""

from __future__ import annotations

import ipaddress
import logging
import math
import threading
import time
from datetime import UTC, datetime, timedelta
from typing import Any
from urllib.parse import urlsplit

import pandas as pd
import requests

from .binance_http import get_shared_binance_session
from .binance_rate_limiter import get_binance_weight_limiter
from .config import get_config
from .errors import NoMarketDataError, VendorRateLimitError
from .stockstats_utils import MAX_OHLCV_STALE_DAYS, _assert_ohlcv_not_stale
from .symbol_utils import (
    _EQUITY_PERP_SEED_BASES,
    normalize_symbol_for_venue,
    tokenized_stock_perp_underlying,
)
from .utils import current_pit_end, proxy_map

logger = logging.getLogger(__name__)

_FAPI_BASE = "https://fapi.binance.com"
# connect, read — short connect, generous-but-bounded read so a stalled
# mid-response cannot hang the ticker (mirrors the openai_client read-timeout
# safety net; run_robust's watchdog is the OS-level floor).
_TIMEOUT = (5, 30)

# fapi per-request caps. klines tops out at 1500, fundingRate at 1000; without
# paging, a long range returns only the OLDEST page and silently drops the
# recent, decision-critical rows. indexPriceKlines is the exception in the
# kline family: its documented ceiling is 1000 rows (mark/last stay 1500).
_FAPI_KLINES_LIMIT = 1500
_FAPI_INDEX_KLINES_LIMIT = 1000
_FAPI_FUNDING_LIMIT = 1000
# Runaway guard: ~50000 daily bars ≈ 137 years. Purely a safety ceiling so a
# mis-sized range can never spin the paginator unboundedly.
_FAPI_PAGINATION_SAFETY_CAP = 50000

# Closed-window history memo (see _paginate_history): FIFO like the yfinance
# _OHLCV_MEMO, generous TTL — an immutable past window does not go stale, the
# TTL only bounds memory across a very long-lived process.
_HISTORY_MEMO: dict[tuple, tuple[float, list]] = {}
_HISTORY_MEMO_LOCK = threading.Lock()
_HISTORY_MEMO_MAX = 64
_HISTORY_MEMO_TTL_S = 6 * 3_600.0


def reset_history_memo_for_test() -> None:
    """Drop the closed-window history memo (tests only — fresh state per case)."""
    with _HISTORY_MEMO_LOCK:
        _HISTORY_MEMO.clear()


def _now_ms() -> int:
    """Current UTC epoch milliseconds, as a module-level indirection.

    Not an inline ``time.time()``: the PIT window anchors that read it must be
    injectable, because the failure mode they guard against — a LOCAL date
    ahead of the UTC date, so the requested window ends in the future — only
    reproduces on a real clock once a day and cannot be pinned otherwise.
    """
    return int(time.time() * 1000)


# The /futures/data/* family (openInterestHist, the *Ratio endpoints, taker
# volume, basis) accepts startTime/endTime, but Binance retains ONLY the most
# recent 30 days for these series — a window that ends before that horizon
# returns nothing no matter which parameters are sent. The retention constant
# mirrors that documented horizon; older windows degrade with a typed error
# rather than silently receiving the newest (future-for-a-backtest) rows.
_FUTURES_DATA_RETENTION_DAYS = 30
# Documented per-request ``limit`` ceiling for /futures/data/* endpoints.
_FUTURES_DATA_LIMIT_CAP = 500
# The same family also accepts a ``period`` granularity parameter (5m..1d).
# Rows-per-day per period — used to translate a day-based window/look-back
# into a row ``limit`` so an intraday request is not silently truncated to
# the first 500 (which would drop the most recent rows, the decision-critical
# ones — the display twin of the pagination gap).
_FUTURES_DATA_PERIODS: dict[str, int] = {
    "5m": 288, "15m": 96, "30m": 48, "1h": 24,
    "2h": 12, "4h": 6, "6h": 4, "12h": 2, "1d": 1,
}


def _futures_period_per_day(period: str) -> int:
    """Validate a ``period`` granularity and return its rows-per-day count."""
    per_day = _FUTURES_DATA_PERIODS.get(period)
    if per_day is None:
        raise ValueError(
            f"period must be one of {sorted(_FUTURES_DATA_PERIODS)}, got {period!r}"
        )
    return per_day


def _fmt_futures_data_ts(ms: int, period: str) -> str:
    """Format a /futures/data/* row timestamp for its period's granularity."""
    dt = datetime.fromtimestamp(ms / 1000, tz=UTC)
    return dt.strftime("%Y-%m-%d" if period == "1d" else "%Y-%m-%d %H:%M")

# ---- Binance SPOT (crypto_spot asset type) ---------------------------------
# Spot public market data lives under /api/v3/*. Two hosts: the canonical
# api.binance.com (same family as fapi.binance.com, proven through the SOCKS5
# proxy) and the key-free market-data mirror data-api.binance.vision (Binance's
# recommended host for read-only consumers; same data, same 6000/min IP weight).
# Default to the canonical host; flip binance_spot_mirror on for the mirror.
_SPOT_BASE = "https://api.binance.com"
_SPOT_MIRROR_BASE = "https://data-api.binance.vision"
# Spot /api/v3/klines caps at 1000 rows/page (fapi is 1500).
_SPOT_KLINES_LIMIT = 1000

# ---- Transport resilience (env: YIALPHA_BINANCE_HTTP_*) --------------------
# All default-off / zero below; binance_http_retries=0 and binance_http_keepalive
# =False leave _http_get byte-equivalent to today (single requests.get, raw
# exception propagation). Flipped on, they only change WHEN/WHETHER a request is
# issued or HOW a transient failure recovers — never the successful-response
# bytes. See _request_with_retry / _parse_retry_after.
# Exponential backoff base (seconds); mirrors yf_retry's base_delay.
_RETRY_BASE_DELAY = 2.0
# HTTP statuses that are transient + idempotent-safe to retry on a read-only GET
# (server-side UNKNOWN per Binance docs). 429/418 are NOT here — those stay on
# the reactive VendorRateLimitError path (optionally with Retry-After).
_RETRIABLE_STATUS = frozenset({500, 502, 503, 504})
# Cap on a Retry-After sleep (seconds). Binance IP bans scale 2min..3days; a long
# ban must not hang the ticker — defer it to run_robust's per-ticker rerun.
_RETRY_AFTER_CAP_S = 60


def _paginate_history(
    path: str,
    base_params: dict,
    page_limit: int,
    cursor_of,
    start_ms: int,
    end_ms: int,
    symbol_for_error: str,
    canonical: str,
    base: str = _FAPI_BASE,
    weight_key: str = "fapi",
) -> list:
    """Page through a Binance history endpoint over ``[start_ms, end_ms]``.

    Binance caps each request at ``page_limit`` rows and returns them oldest-
    first, so a range longer than the cap would silently drop the most recent
    rows (the ones that matter most for a decision) without paging. Cursor by
    each page's last row (``cursor_of(item) -> ms``) +1ms until a page is short,
    the cursor passes ``end_ms``, or the safety cap is hit. Each page is its own
    ``_http_get`` so the proactive-backoff / reactive-429 handling still applies
    per request. Data is unchanged — only missing rows are filled in.

    ``base`` and ``weight_key`` default to the fapi perp host/budget so the two
    perp callers (klines, fundingRate) are byte-identical to the pre-spot form;
    spot callers pass ``base=_spot_host(), weight_key="spot"``.

    Closed-window memoization (2026-08-16): a window ending more than 12h in
    the past is immutable on the exchange side, but ONE analysis run fetches
    the same window 2-3x (raw kline tool + indicator battery + basis + the
    backtest's per-signal propagation), burning the shared per-IP weight
    budget that then throttles the whole batch. Fully-closed results are
    memoized keyed by ``(path, host, params, start_ms, end_ms)`` — the PIT end
    is part of the key, so a clamped window can never be served from a wider
    cached one, and a window touching the present is never cached at all.
    """
    cache_key = (
        path, base, tuple(sorted((base_params or {}).items())), start_ms, end_ms,
    )
    # 12h margin: the current UTC day's kline is still forming and an 8h
    # funding cadence may have a settlement pending — never memoize those.
    cacheable = end_ms <= int(time.time() * 1000) - 12 * 3_600_000
    if cacheable:
        with _HISTORY_MEMO_LOCK:
            hit = _HISTORY_MEMO.get(cache_key)
            if hit is not None and time.monotonic() - hit[0] < _HISTORY_MEMO_TTL_S:
                logger.debug("Binance history cache hit for %s %s", path, canonical)
                return list(hit[1])

    cursor = start_ms
    out: list = []
    while cursor <= end_ms and len(out) < _FAPI_PAGINATION_SAFETY_CAP:
        params = dict(base_params)
        params["startTime"] = cursor
        params["endTime"] = end_ms
        params["limit"] = page_limit
        page = _http_get(
            path, params, symbol_for_error, canonical,
            base=base, weight_key=weight_key,
        )
        if not isinstance(page, list) or not page:
            break
        out.extend(page)
        try:
            last_ms = cursor_of(page[-1])
        except (TypeError, KeyError, IndexError):
            break
        nxt = int(last_ms) + 1
        if nxt <= cursor:  # no forward progress — avoid an infinite loop
            break
        cursor = nxt
        if len(page) < page_limit:  # final partial page reached
            break
    if len(out) >= _FAPI_PAGINATION_SAFETY_CAP:
        # NOTE: hit pagination safety cap. Older rows may be truncated; the
        # recent rows (which drive the decision) are still complete.
        logger.warning(
            "Binance %s pagination hit safety cap (%d rows) for %s; "
            "older rows may be truncated",
            path, _FAPI_PAGINATION_SAFETY_CAP, symbol_for_error,
        )
    if cacheable and out:
        with _HISTORY_MEMO_LOCK:
            if len(_HISTORY_MEMO) >= _HISTORY_MEMO_MAX and cache_key not in _HISTORY_MEMO:
                _HISTORY_MEMO.pop(next(iter(_HISTORY_MEMO)))  # FIFO eviction
            _HISTORY_MEMO[cache_key] = (time.monotonic(), list(out))
    return out


def _observe_weight(resp, weight_key: str = "fapi") -> None:
    """Feed the server-reported IP weight to the process-wide limiter.

    Reads ``X-MBX-USED-WEIGHT-1M`` (Binance's per-IP rolling 1-min weight
    counter; requests' headers are case-insensitive) and feeds it to the
    ``weight_key`` product-line limiter. The header is optional — a missing or
    unparseable value is logged at debug and skipped, since the next response
    carries a fresh value and a single miss is harmless.
    """
    headers = getattr(resp, "headers", None)
    raw = headers.get("X-MBX-USED-WEIGHT-1M") if headers else None
    if raw is None:
        return
    try:
        used = int(raw)
    except (TypeError, ValueError):
        logger.debug("Binance X-MBX-USED-WEIGHT-1M unparseable: %r", raw)
        return
    get_binance_weight_limiter(weight_key).observe(used)


def _validate_outbound_url(url: str) -> None:
    """Refuse non-HTTP(S) schemes and non-public hosts before any request.

    Defense-in-depth for this module's fixed fapi/spot bases (and any future
    configurable host): the scheme must be http/https and a literal IP host
    must not be loopback/private/reserved/link-local/unspecified, and the name
    must not be ``localhost``/``*.localhost``/``*.internal``. A mis-set base
    can then never turn a market-data fetch into a probe of the internal
    network. Names that resolve to private IPs are out of scope for this cheap
    literal check (no DNS resolution here by design).
    """
    try:
        parts = urlsplit(url)
    except ValueError as exc:
        raise ValueError(f"unparseable outbound URL: {url!r}") from exc
    if parts.scheme not in ("http", "https"):
        raise ValueError(f"non-HTTP(S) outbound URL refused: {url!r}")
    host = (parts.hostname or "").strip("[]").lower()
    if not host:
        raise ValueError(f"outbound URL without a host refused: {url!r}")
    if host == "localhost" or host.endswith(".localhost") or host.endswith(".internal"):
        raise ValueError(f"non-public outbound host refused: {host!r}")
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return  # ordinary DNS name — allowed
    if (
        ip.is_private or ip.is_loopback or ip.is_reserved
        or ip.is_link_local or ip.is_unspecified
    ):
        raise ValueError(f"non-public outbound host refused: {host!r}")


def _do_request(url: str, params: dict, use_session: bool, headers: dict | None = None):
    """Fire ONE transport attempt and return the raw ``requests.Response``.

    When ``use_session`` is true, reuse the process-wide shared
    ``requests.Session`` (keepalive) so the TLS / SOCKS5-proxy connection is
    pooled across calls; otherwise a one-shot ``requests.get`` (today's form).
    Both paths take identical ``proxies`` / ``timeout`` so the response bytes are
    the same either way — the session only adds connection reuse. ``headers``
    (signed-endpoint API keys) defaults to None: absent = byte-identical to
    the public path.
    """
    _validate_outbound_url(url)
    if use_session:
        return get_shared_binance_session().get(
            url, params=params, proxies=proxy_map(), timeout=_TIMEOUT,
            headers=headers,
        )
    return requests.get(
        url, params=params, proxies=proxy_map(), timeout=_TIMEOUT, headers=headers,
    )


def _request_with_retry(do_request, max_retries: int, symbol_for_error: str, canonical: str):
    """Call ``do_request()`` with exponential backoff on transient failures.

    Mirrors :func:`yialpha.dataflows.stockstats_utils.yf_retry`. Retries on
    ``requests.RequestException`` (DNS / connection / TLS / timeout — the
    transport stack) AND on HTTP 5xx (server-side UNKNOWN, idempotent-safe for a
    read-only GET). 429/418 are NOT retried here — they stay on the reactive
    ``VendorRateLimitError`` path so the router/vendor-chain handles them.

    Success path: a non-retriable status is returned on the first attempt, so
    output is byte-identical regardless of ``max_retries``. Failure path:
    ``max_retries == 0`` re-raises the raw transport exception (byte-equivalent
    to today); ``max_retries > 0`` exhausted raises
    :class:`NoMarketDataError` so the routing layer degrades instead of crashing
    the node (matches ``yf_retry``). A 5xx that exhausts retries is returned so
    ``_http_get``'s existing non-200 → ``NoMarketDataError`` path fires (today's
    behaviour).
    """
    last_exc: BaseException | None = None
    for attempt in range(max_retries + 1):
        try:
            resp = do_request()
        except requests.RequestException as exc:
            last_exc = exc
            resp = None
        if resp is not None and resp.status_code not in _RETRIABLE_STATUS:
            return resp  # success, or a non-retriable status for _http_get to handle
        # Transient (transport error or 5xx) — retry if budget remains.
        if attempt < max_retries:
            reason = (
                type(last_exc).__name__ if last_exc is not None
                else f"HTTP {resp.status_code}"
            )
            delay = _RETRY_BASE_DELAY * (2 ** attempt)
            logger.warning(
                "Binance transient (%s) for %s; retrying in %.0fs (attempt %d/%d)",
                reason, symbol_for_error, delay, attempt + 1, max_retries,
            )
            time.sleep(delay)
            continue
        # Budget exhausted.
        if resp is not None:
            return resp  # 5xx exhausted → let _http_get's non-200 path run (today's behaviour)
        if max_retries > 0:
            raise NoMarketDataError(
                symbol_for_error, canonical,
                f"Binance unreachable after {max_retries} retries "
                f"({type(last_exc).__name__})",
            ) from last_exc
        assert last_exc is not None  # resp is None only when do_request() raised
        raise last_exc  # max_retries == 0: propagate raw, byte-equivalent to today


def _parse_retry_after(headers) -> int | None:
    """Parse the ``Retry-After`` response header as whole seconds (or ``None``).

    Binance sends the delay as an integer number of seconds. The HTTP-date form
    is not supported (and not used by Binance); a missing or unparseable header
    returns ``None`` so the caller falls back to today's immediate-raise path.
    """
    if not headers:
        return None
    raw = headers.get("Retry-After") or headers.get("retry-after")
    if not raw:
        return None
    try:
        return max(0, int(raw))
    except (TypeError, ValueError):
        return None


def _http_get(
    path: str,
    params: dict,
    symbol_for_error: str,
    canonical: str,
    base: str = _FAPI_BASE,
    weight_key: str = "fapi",
    headers: dict | None = None,
) -> object:
    """GET a Binance public endpoint and return parsed JSON.

    Raises :class:`VendorRateLimitError` on 429/418 (IP throttle), and
    :class:`NoMarketDataError` for any non-200, a Binance error body
    (``{"code": ..., "msg": ...}``), or an empty response. Other
    ``requests`` exceptions propagate so the router can treat them as a
    transient vendor failure (and degrade when the category is optional).

    ``base`` selects the host (fapi perp vs spot) and ``weight_key`` selects
    which product-line limiter budget the call counts against; both default to
    the fapi perp values so the existing perp path is byte-identical.

    When ``binance_proactive_backoff`` is on, the call first consults the
    process-wide weight limiter (:func:`get_binance_weight_limiter`) and blocks
    if the last server-reported IP weight was near the ceiling, then feeds the
    fresh ``X-MBX-USED-WEIGHT-1M`` header back to the limiter. This only changes
    *when* the request fires, never the data; off by default = byte-equivalent.

    Transport resilience (all default-off = byte-equivalent to today):
    ``binance_http_keepalive`` reuses a process-wide ``requests.Session`` so the
    TLS/SOCKS5 connection is pooled across calls; ``binance_http_retries`` adds
    exponential backoff on transient transport errors (DNS/timeout/TLS) and 5xx
    (mirrors ``yf_retry``; exhausted → ``NoMarketDataError``);
    ``binance_honor_retry_after`` sleeps the server's ``Retry-After`` window
    before raising on 429/418. None change the successful-response bytes.
    """
    cfg = get_config()
    proactive = cfg.get("binance_proactive_backoff", False)
    use_session = cfg.get("binance_http_keepalive", False)
    max_retries = int(cfg.get("binance_http_retries", 0))
    honor_retry_after = cfg.get("binance_honor_retry_after", False)
    if proactive:
        # Back off BEFORE the request if the budget is hot, so we avoid tripping
        # a 429 rather than only reacting to one. The reactive 429/418 handling
        # below stays as the floor either way.
        get_binance_weight_limiter(weight_key).acquire()

    url = f"{base}{path}"
    resp = _request_with_retry(
        lambda: _do_request(url, params, use_session, headers),
        max_retries, symbol_for_error, canonical,
    )

    if proactive:
        _observe_weight(resp, weight_key)

    if resp.status_code in (429, 418):
        # Honor the server's Retry-After (IP ban window) before raising, so the
        # next call (pagination / vendor chain / run_robust rerun) does not
        # immediately re-trip the ban. Off by default = raise immediately
        # (byte-equivalent to today); a ban longer than _RETRY_AFTER_CAP_S is
        # deferred to run_robust rather than slept out inline.
        if honor_retry_after:
            wait = _parse_retry_after(resp.headers)
            if wait is not None and 0 < wait <= _RETRY_AFTER_CAP_S:
                logger.info(
                    "Binance HTTP %d for %s; honoring Retry-After %ds before raising",
                    resp.status_code, symbol_for_error, wait,
                )
                time.sleep(wait)
            elif wait is not None and wait > _RETRY_AFTER_CAP_S:
                logger.info(
                    "Binance HTTP %d ban Retry-After %ds > cap %ds for %s; "
                    "deferring to run_robust rerun",
                    resp.status_code, wait, _RETRY_AFTER_CAP_S, symbol_for_error,
                )
        raise VendorRateLimitError(
            f"Binance rate-limited {symbol_for_error} (HTTP {resp.status_code})"
        )

    if resp.status_code != 200:
        # Non-throttle error: treat as no-data for this symbol/params so the
        # router emits a clear unavailable signal rather than crashing.
        snippet = (resp.text or "").strip()[:200]
        raise NoMarketDataError(
            symbol_for_error,
            canonical,
            f"Binance HTTP {resp.status_code}: {snippet}",
        )

    body = (resp.text or "").strip()
    if not body:
        raise NoMarketDataError(symbol_for_error, canonical, "empty response body")

    try:
        parsed = resp.json()
    except ValueError:
        # 200 but not JSON — unexpected for these endpoints; surface as no-data.
        raise NoMarketDataError(
            symbol_for_error, canonical, f"non-JSON response: {body[:200]}"
        ) from None

    # Binance signals errors inside a 200 body as {"code": <non-zero>, "msg": ...}.
    if isinstance(parsed, dict) and parsed.get("code") and parsed.get("code") != 200:
        err = NoMarketDataError(
            symbol_for_error,
            canonical,
            f"Binance code {parsed.get('code')}: {parsed.get('msg')}",
        )
        err.vendor_code = parsed.get("code")  # API-level rejection marker
        raise err

    return parsed


#: Close-column name of the klines frame schema. Single source for producers
#: AND consumers: a consumer keying on lowercase ``"close"`` raises KeyError,
#: which the fail-open stress path swallowed into basis=None for the entire
#: life of that component (caught only once the mock matched the real schema).
KLINE_CLOSE_COLUMN = "Close"


def binance_klines_frame(
    symbol: str,
    start_date: str,
    end_date: str,
    interval: str = "1d",
    venue: str = "binance_perp",
    price_type: str = "last",
    *,
    closed_as_of: int | None = None,
) -> pd.DataFrame:
    """OHLCV DataFrame for a Binance pair (perp or spot), PIT-clamped.

    Shared data layer for the klines CSV tools and the indicator tool
    (``get_binance_indicators``) so classic stockstats indicators can be
    computed on the SAME candles the analyst reads. Date-indexed, sorted,
    at the exchange's own precision (no rounding — Binance prices carry
    sub-cent low-price contracts, e.g. PEPE ≈ 1e-5, and a display-style
    round destroys them). Raises NoMarketDataError on empty windows, and
    ``current_pit_end`` clamps the end so a backtest never sees klines after
    its analysis date.

    ``closed_as_of`` (epoch ms, keyword-only) additionally drops rows whose
    closeTime (kline element 6) is strictly later, so a caller pinned to a
    wall-clock instant never reads a candle that had not closed by then —
    e.g. the current day's still-forming bar, which the window clamp above
    cannot remove. The filter deliberately runs after ``_paginate_history``:
    the closed-window memo caches raw rows, and every call filters by its
    own ``closed_as_of``. ``None`` (the default) keeps every row, forming
    bar included — the behavior existing callers are built on.

    ``price_type="mark"`` (perp only) switches the endpoint to
    ``/fapi/v1/markPriceKlines`` — the mark price Binance liquidates against —
    for liquidation-sensitive research; ``price_type="index"`` (perp only)
    serves ``/fapi/v1/indexPriceKlines`` — the settlement/index-price series
    (volume is 0 on index klines) — so mark-vs-index displacement can be
    studied as two aligned series rather than one premium snapshot. The
    default ``"last"`` is the ordinary last-traded-price kline. Note the
    index endpoint's identifier parameter is ``pair=`` (the last/mark
    endpoints use ``symbol=``) and it pages at 1000 rows.
    """
    if price_type not in ("last", "mark", "index"):
        raise ValueError(
            f"price_type must be 'last', 'mark' or 'index', got {price_type!r}"
        )
    if venue == "binance_spot":
        if price_type != "last":
            raise ValueError("mark/index-price klines are perp-only endpoints")
        path, limit, base, weight_key = (
            "/api/v3/klines", _SPOT_KLINES_LIMIT, _spot_host(), "spot",
        )
    else:
        path = {
            "mark": "/fapi/v1/markPriceKlines",
            "index": "/fapi/v1/indexPriceKlines",
        }.get(price_type, "/fapi/v1/klines")
        # indexPriceKlines deviates from its last/mark siblings twice: its
        # required identifier parameter is ``pair`` (NOT ``symbol``) and its
        # per-request ceiling is 1000 rows (not 1500). Sending symbol= to the
        # current API returns an error body, which the fail-open consumers
        # (derivatives stress) silently degraded to basis=None.
        limit = (
            _FAPI_INDEX_KLINES_LIMIT if price_type == "index" else _FAPI_KLINES_LIMIT
        )
        path, base, weight_key = path, None, None
    canonical = normalize_symbol_for_venue(symbol, venue)

    start_ms = int(
        datetime.strptime(start_date, "%Y-%m-%d")
        .replace(tzinfo=UTC)
        .timestamp()
        * 1000
    )
    end_date = current_pit_end(end_date) or end_date
    end_dt = datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=UTC)
    end_ms = int((end_dt.timestamp() + 86399) * 1000)  # end-of-day inclusive

    kwargs: dict = {"base": base, "weight_key": weight_key}
    if base is None:
        kwargs = {}
    # indexPriceKlines identifies the contract with ``pair`` (see docstring).
    identifier_key = "pair" if price_type == "index" and venue != "binance_spot" else "symbol"
    rows = _paginate_history(
        path,
        {identifier_key: canonical, "interval": interval},
        limit,
        lambda k: k[0],  # kline open_time (ms) is element 0
        start_ms,
        end_ms,
        symbol,
        canonical,
        **kwargs,
    )

    if not isinstance(rows, list) or not rows:
        raise NoMarketDataError(
            symbol, canonical, f"no klines between {start_date} and {end_date}"
        )

    # Binance kline array indices: [1]Open [2]High [3]Low [4]Close [5]Volume.
    close_col = KLINE_CLOSE_COLUMN
    records = []
    for k in rows:
        if not isinstance(k, list) or len(k) < 6:
            continue
        # D5a: k[6] is the bar's closeTime — drop candles that had not yet
        # closed by closed_as_of (e.g. the run's pinned wall-clock), such as
        # the current day's still-forming bar. Arrays with no element 6 pass
        # through unfiltered, exactly as the len>=6 guard above allows.
        if (
            closed_as_of is not None
            and len(k) >= 7
            and int(k[6]) > closed_as_of
        ):
            continue
        open_ms = int(k[0])
        records.append(
            {
                "Date": datetime.fromtimestamp(open_ms / 1000, tz=UTC)
                .strftime("%Y-%m-%d" if interval == "1d" else "%Y-%m-%d %H:%M:%S"),
                "Open": float(k[1]),
                "High": float(k[2]),
                "Low": float(k[3]),
                close_col: float(k[4]),
                "Adj Close": float(k[4]),
                "Volume": float(k[5]),
            }
        )

    if not records:
        raise NoMarketDataError(
            symbol, canonical, f"no parseable klines between {start_date} and {end_date}"
        )

    df = pd.DataFrame.from_records(records)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date").sort_index()
    # NO rounding: unlike equity quotes, Binance lists sub-cent contracts
    # (PEPE ≈ 1e-5, 1000PEPE ≈ 1e-2) where a yfinance-style round(2) zeroes
    # or badly distorts every price — and the indicator tool computes on this
    # same frame. Binance strings already carry the exchange's own precision.
    # Live-window staleness guard (same contract as the yfinance path, #1021):
    # when the caller asked for data up to (near) the present, a frame whose
    # last candle is far older — a delisted/renamed contract — must raise
    # rather than silently feed months-old "current" prices to the analyst.
    # Historical windows are exempt: an early-ending series is a legitimate
    # backtest input, not a freshness lie.
    if (datetime.now(UTC) - end_dt).days <= MAX_OHLCV_STALE_DAYS:
        _assert_ohlcv_not_stale(df, end_date, symbol, canonical)
    return df


#: Human disclosure rendered into the klines CSV header per price basis, so
#: a row's basis is visible in the artifact itself (not only the tool's
#: docstring): every downstream "price is X" claim can be checked against the
#: basis it was actually served on.
_PRICE_BASIS_NOTES = {
    "last": "last traded price; the default trend/entry series",
    "mark": "mark price — the price Binance liquidates against",
    "index": "index price — settlement fair-value anchor; volume is 0",
}


def get_binance_klines(
    symbol: str,
    start_date: str,
    end_date: str,
    interval: str = "1d",
    price_type: str = "last",
) -> str:
    """Daily OHLCV for a Binance USDT-M perpetual pair.

    Returns a ``str`` shaped like yfinance's ``get_YFin_data_online`` output —
    header block + CSV with columns ``Open, High, Low, Close, Adj Close,
    Volume`` (``Adj Close`` mirrors ``Close`` since perps have no splits) — so
    the downstream stockstats indicator path is reusable. ``interval`` defaults
    to ``"1d"``; the analyst passes it through for intraday if ever needed.
    ``price_type="mark"`` serves mark-price klines (the price Binance
    liquidates against) for liquidation-sensitive research;
    ``price_type="index"`` serves index-price klines (the settlement
    reference; volume column is 0) for mark-vs-index displacement work.
    """
    df = binance_klines_frame(
        symbol, start_date, end_date, interval=interval, venue="binance_perp",
        price_type=price_type,
    )
    canonical = normalize_symbol_for_venue(symbol, "binance_perp")
    end_date = current_pit_end(end_date) or end_date
    label = canonical if canonical == symbol.upper() else f"{canonical} (from {symbol})"
    header = f"# Perp USDT-M klines for {label} from {start_date} to {end_date}\n"
    header += f"# Price basis: {price_type} ({_PRICE_BASIS_NOTES[price_type]})\n"
    header += f"# Total records: {len(df)}\n"
    header += f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    return header + df.to_csv()


def _funding_cadence_hours(rows: list) -> int | None:
    """Modal spacing (whole hours) between adjacent funding settlements.

    Binance settles most USDT-M perps every 8h, but many newer contracts run
    4h or 1h — annualising carry with the wrong cadence misstates it 2-24x.
    The mode (not the min/median) is robust to a single missing settlement in
    an otherwise regular series. Returns ``None`` when there are fewer than
    two settlements to diff.
    """
    times = sorted(
        int(r["fundingTime"])
        for r in rows
        if isinstance(r, dict) and r.get("fundingTime") is not None
    )
    diffs = [b - a for a, b in zip(times, times[1:], strict=False) if b > a]
    if not diffs:
        return None
    mode_ms = max(set(diffs), key=diffs.count)
    hours = round(mode_ms / 3_600_000)
    return hours if hours > 0 else None


def _funding_interval_hours_authoritative(canonical: str) -> int | None:
    """Authoritative funding interval for ``canonical`` from ``/fapi/v1/fundingInfo``.

    The public endpoint lists the symbols whose funding interval differs from
    the 8h default and/or that carry funding caps (1h/4h contracts — often the
    newer/tokenized-stock perps). A symbol absent from the list uses the 8h
    default, so this returns ``None`` both for unlisted symbols and on any
    transport failure — the settlement-spacing inference stays the fallback.
    """
    try:
        data = _http_get("/fapi/v1/fundingInfo", {}, canonical, canonical)
    except Exception:  # noqa: BLE001 — advisory metadata; spacing inference remains
        return None
    for item in data if isinstance(data, list) else []:
        if isinstance(item, dict) and item.get("symbol") == canonical:
            try:
                hours = item.get("fundingIntervalHours")
                return int(hours) if hours is not None else None
            except (TypeError, ValueError):
                return None
    return None


def get_binance_funding_rate(
    symbol: str,
    start_date: str,
    end_date: str,
) -> str:
    """Funding-rate history for a Binance USDT-M perpetual pair.

    Returns header + CSV with ``fundingTime, fundingRate, symbol``. Funding
    settles on the contract's own cadence — 8h on most contracts, 4h/1h on
    many newer ones; the header states the cadence inferred from the
    settlement spacing. Persistently positive funding = longs pay shorts =
    crowding / cost-of-carry.
    """
    canonical = normalize_symbol_for_venue(symbol, "binance_perp")

    start_ms = int(
        datetime.strptime(start_date, "%Y-%m-%d")
        .replace(tzinfo=UTC)
        .timestamp()
        * 1000
    )
    # PIT guard: clamp to the analysis date (see get_binance_klines).
    end_date = current_pit_end(end_date) or end_date
    end_dt = datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=UTC)
    end_ms = int((end_dt.timestamp() + 86399) * 1000)

    rows = _paginate_history(
        "/fapi/v1/fundingRate",
        {"symbol": canonical},
        _FAPI_FUNDING_LIMIT,
        lambda r: r["fundingTime"],  # ms timestamp on each funding row
        start_ms,
        end_ms,
        symbol,
        canonical,
    )

    if not isinstance(rows, list) or not rows:
        raise NoMarketDataError(
            symbol, canonical, f"no funding rates between {start_date} and {end_date}"
        )

    # Authoritative cadence first: /fapi/v1/fundingInfo states the contract's
    # own interval for non-default (1h/4h) settlers; spacing inference is the
    # fallback for unlisted symbols / transport failures.
    cadence_h = _funding_interval_hours_authoritative(canonical)
    cadence_src = "fundingInfo endpoint"
    if cadence_h is None:
        cadence_h = _funding_cadence_hours(rows)
        cadence_src = "inferred from settlement spacing"

    records = [
        {
            "fundingTime": datetime.fromtimestamp(
                int(r["fundingTime"]) / 1000, tz=UTC
            ).strftime("%Y-%m-%d %H:%M:%S"),
            "fundingRate": r.get("fundingRate"),
            "symbol": r.get("symbol", canonical),
        }
        for r in rows
        if isinstance(r, dict) and r.get("fundingTime") is not None
    ]

    if not records:
        raise NoMarketDataError(
            symbol, canonical, f"no parseable funding rates between {start_date} and {end_date}"
        )

    df = pd.DataFrame.from_records(records)

    label = canonical if canonical == symbol.upper() else f"{canonical} (from {symbol})"
    header = f"# Perp USDT-M funding rate for {label} from {start_date} to {end_date}\n"
    if cadence_h is not None:
        header += (
            f"# funding settles every ~{cadence_h}h on this contract "
            f"({cadence_src}; annualised carry = mean rate x "
            f"{24 / cadence_h:.1f} x 365)\n"
        )
    header += f"# Total records: {len(df)}\n"
    header += f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    return header + df.to_csv(index=False)


def get_binance_open_interest(
    symbol: str, look_back_days: int = 7,
    start_date: str | None = None, end_date: str | None = None,
    period: str = "1d",
) -> str:
    """Open-interest snapshot + history for a Binance USDT-M perp.

    Combines the live ``/fapi/v1/openInterest`` snapshot with the
    ``/futures/data/openInterestHist`` series into a single ``time,
    openInterest, openInterestValue`` table. Rising OI + rising price confirms
    a trend; rising OI + falling price signals crowded shorts (or longs
    unwinding). ``look_back_days`` rows are requested by default.

    ``period`` selects the series granularity (``"1d"`` default; ``5m``/``15m``
    /``1h``/``2h``/``4h``/``6h``/``12h`` for intraday OI). Sub-daily rows are
    timestamped ``YYYY-MM-DD HH:MM`` and the live "latest" row is appended only
    for period ``1d`` windows reaching the present.

    Historical windows: ``start_date``/``end_date`` (``YYYY-MM-DD``, end
    inclusive) are passed through as the endpoint's ``startTime``/``endTime``,
    and ``end_date`` is clamped to the run's analysis date
    (:func:`yialpha.dataflows.utils.current_pit_end`) so a backtest never
    receives rows after it; the live snapshot row is appended only when the
    window reaches the present. The endpoint retains only the LAST 30 DAYS —
    a window ending further back raises :class:`NoMarketDataError` (the
    router degrades the optional category to a sentinel) instead of returning
    today's rows as if they were the past; the
    :func:`get_binance_vision_metrics` archive tool covers deeper history.
    Without explicit dates the call remains the most-recent-``limit`` live
    form.
    """
    canonical = normalize_symbol_for_venue(symbol, "binance_perp")
    per_day = _futures_period_per_day(period)
    limit = max(1, min(int(look_back_days) * per_day, _FUTURES_DATA_LIMIT_CAP))

    # Values mix str (symbol/period) and int (limit/startTime/endTime ms).
    params: dict[str, int | str] = {"symbol": canonical, "period": period}
    extra, end_iso, reaches_now, end_ms, coverage_note = _futures_data_window(
        symbol, canonical, look_back_days, start_date, end_date, period=period,
    )
    params.update(extra)
    if not extra:
        params["limit"] = limit

    hist = _http_get(
        "/futures/data/openInterestHist",
        params,
        symbol,
        canonical,
    )

    records: list[dict] = []
    if isinstance(hist, list):
        for r in hist:
            if not isinstance(r, dict) or r.get("timestamp") is None:
                continue
            records.append(
                {
                    "_ts": int(r["timestamp"]),
                    "time": _fmt_futures_data_ts(int(r["timestamp"]), period),
                    "openInterest": r.get("sumOpenInterest"),
                    "openInterestValue": r.get("sumOpenInterestValue"),
                }
            )
    # Belt-and-suspenders PIT trim: a vendor (or mock) that ignores the
    # endTime must never leak rows past the requested window end. Compares
    # TIMESTAMP ms (not the formatted string — see _futures_data_window).
    if end_ms is not None:
        records = [r for r in records if r["_ts"] <= end_ms]
    for r in records:
        r.pop("_ts", None)

    # Append the live snapshot so the analyst sees the most current OI too —
    # but only for daily windows reaching the present; for a past end_date or
    # an intraday period the "latest" row is future/other-grain data for that
    # decision point.
    live_unavailable = False
    if reaches_now and period == "1d":
        try:
            live = _http_get(
                "/fapi/v1/openInterest",
                {"symbol": canonical},
                symbol,
                canonical,
            )
            if isinstance(live, dict) and live.get("openInterest") is not None:
                records.append(
                    {
                        "time": "latest",
                        "openInterest": live.get("openInterest"),
                        "openInterestValue": None,
                    }
                )
        except (NoMarketDataError, VendorRateLimitError) as exc:
            # History is the analytically useful part; a missing live snapshot is
            # logged but does not fail the call (the series still carries value).
            # The header notes it so a missing "latest" row reads as "fetch
            # failed", not "no current open interest".
            live_unavailable = True
            logger.info("Binance live openInterest unavailable for %s: %s", canonical, exc)

    if not records:
        raise NoMarketDataError(
            symbol, canonical, f"no open-interest history (look_back_days={look_back_days})"
        )

    df = pd.DataFrame.from_records(records)

    label = canonical if canonical == symbol.upper() else f"{canonical} (from {symbol})"
    scope = "days + live" if period == "1d" else f"{period} rows"
    header = (
        f"# Perp USDT-M open interest for {label} (last {limit} {scope})\n"
    )
    if period != "1d":
        header += f"# period: {period}\n"
    if end_iso is not None:
        header += (
            f"# window through {end_iso} (endpoint retains the last "
            f"{_FUTURES_DATA_RETENTION_DAYS} days only; deeper history: "
            f"get_binance_vision_metrics)\n"
        )
    if coverage_note:
        header += coverage_note
    if live_unavailable:
        header += "# ⚠ live openInterest snapshot unavailable (fetch failed) — history only.\n"
    header += f"# Total records: {len(df)}\n"
    header += f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    return header + df.to_csv(index=False)


# ---- Perp positioning / order-flow / basis (the perp "sentiment" pillar) ----
# These /futures/data/* endpoints are public (no key) and share the same IP
# weight counter as openInterestHist, so they reuse _http_get verbatim. Each
# degrades independently to a sentinel on 429/unsupported-symbol rather than
# aborting the run — same optional-category contract as funding/OI.

_LSR_LIMIT_CAP = 30  # daily snapshots; >30 rarely adds analytical value and
                     # these series are short for newer TRADIFI perps anyway.


def _futures_data_window(
    symbol: str,
    canonical: str,
    look_back_days: int,
    start_date: str | None,
    end_date: str | None,
    period: str = "1d",
) -> tuple[dict, str | None, bool, int | None, str]:
    """Resolve an explicit historical window for a ``/futures/data/*`` call.

    Returns ``(extra_params, end_iso, reaches_now, end_ms, coverage_note)``:

      * ``extra_params`` — ``{"startTime", "endTime", "limit"}`` for the
        endpoint when an explicit window was requested, else ``{}`` (the
        caller keeps its plain most-recent-``limit`` behaviour);
      * ``end_iso`` — the resolved inclusive end date (``YYYY-MM-DD``) for
        the header, ``None`` when unwindowed;
      * ``reaches_now`` — whether the window extends to today (gates appending
        any "latest/live" row, which would be future data for a past end);
      * ``end_ms`` — end-of-day ms of the resolved end (``None`` unwindowed)
        for the belt-and-suspenders client-side trim. The trim MUST compare
        timestamps, not date strings: an intraday row "2026-08-15 13:00" is
        lexicographically > "2026-08-15" and a string compare would wrongly
        drop every row of the end day;
      * ``coverage_note`` — ``""`` or a header line disclosing that the window
        exceeded the 500-row endpoint cap and was end-anchored (see below).

    ``period`` scales the row ``limit`` with the granularity (a 7-day 5m
    window needs 2016 rows, not 7). When the window needs more rows than the
    500-row endpoint cap, the request is END-anchored — ``startTime`` moves to
    ``end - 499*period`` — because these endpoints serve rows ascending from
    ``startTime``: a head-anchored request would return the OLDEST rows and
    silently drop the tail nearest ``end_date``, the decision-critical part.
    The truncation is never silent: ``coverage_note`` discloses the dropped
    head and points at the archive tool for the full window.

    PIT + retention rules:

      * ``end_date`` is clamped by :func:`current_pit_end` to the run's pinned
        analysis date, so a backtest window never asks for (or receives) rows
        after it — same guard as the klines endpoints.
      * These endpoints retain only the last ``_FUTURES_DATA_RETENTION_DAYS``
        days. A window ending before that horizon CANNOT be served; raising
        :class:`NoMarketDataError` (the router degrades the optional category
        to a sentinel) is the honest result — falling back to the newest rows
        would hand a backtest future positioning data.
      * ``start_date`` alone defaults the end to now, clamped to the pinned
        analysis date (PIT) exactly like an explicit ``end_date``;
        ``end_date`` alone defaults the start to ``end - look_back_days``.
    """
    if start_date is None and end_date is None:
        return {}, None, True, None, ""

    if end_date:
        end_date = current_pit_end(end_date) or end_date
        end_dt = datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=UTC)
    else:
        # start_date-only: the nominal end is "now", but a pinned analysis
        # date must clamp it exactly like an explicit end_date — otherwise a
        # backtest would receive positioning rows past its decision point.
        now_iso = datetime.now(UTC).strftime("%Y-%m-%d")
        end_dt = datetime.strptime(
            current_pit_end(now_iso) or now_iso, "%Y-%m-%d"
        ).replace(tzinfo=UTC)
    if start_date:
        start_dt = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=UTC)
    else:
        start_dt = end_dt - timedelta(days=max(1, int(look_back_days)))

    horizon = datetime.now(UTC) - timedelta(days=_FUTURES_DATA_RETENTION_DAYS)
    if end_dt < horizon:
        raise NoMarketDataError(
            symbol, canonical,
            f"/futures/data endpoints retain only the last "
            f"{_FUTURES_DATA_RETENTION_DAYS} days; window ends "
            f"{end_dt.date()}, before the retention horizon",
        )

    start_ms = int(start_dt.timestamp() * 1000)
    end_ms = int((end_dt.timestamp() + 86399) * 1000)  # end-of-day inclusive
    days = max(1, (end_dt - start_dt).days + 1)
    per_day = _futures_period_per_day(period)
    rows = days * per_day
    coverage_note = ""
    if rows > _FUTURES_DATA_LIMIT_CAP:
        # End-anchor the request: rows are served ascending from startTime, so
        # keeping the window's own start would hand back the OLDEST 500 rows
        # and silently drop the tail nearest end_date. Anchor one full page
        # before the last in-window row boundary instead.
        period_ms = 86_400_000 // per_day
        last_row_ms = (end_ms // period_ms) * period_ms
        tail_start_ms = last_row_ms - (_FUTURES_DATA_LIMIT_CAP - 1) * period_ms
        if tail_start_ms > start_ms:
            coverage_start = datetime.fromtimestamp(tail_start_ms / 1000, tz=UTC)
            dropped_days = max(0, (coverage_start.date() - start_dt.date()).days)
            start_ms = tail_start_ms
            coverage_note = (
                f"# ⚠ window truncated to the last {_FUTURES_DATA_LIMIT_CAP} rows "
                f"(endpoint ceiling): coverage starts "
                f"{_fmt_futures_data_ts(tail_start_ms, period)}, {dropped_days} "
                f"day(s) dropped from the head; get_binance_vision_metrics "
                f"serves the full window\n"
            )
    extra = {
        "startTime": start_ms,
        "endTime": end_ms,
        "limit": min(rows, _FUTURES_DATA_LIMIT_CAP),
    }
    reaches_now = end_dt.date() >= datetime.now(UTC).date()
    return extra, end_dt.strftime("%Y-%m-%d"), reaches_now, end_ms, coverage_note


def get_binance_long_short_ratio(
    symbol: str, look_back_days: int = 7,
    start_date: str | None = None, end_date: str | None = None,
    period: str = "1d",
) -> str:
    """Trader long/short positioning for a Binance USDT-M perp.

    The perp-native counterpart to "social sentiment": it reports how the crowd
    is actually positioned in leverage, not what people are saying. Combines
    three Binance series into one table (``series, time, longAccount,
    longShortRatio, shortAccount``):

      - ``top_account``    — top-trader *account* long/short ratio
                             (``/futures/data/topLongShortAccountRatio``)
      - ``top_position``   — top-trader *position* long/short ratio
                             (``/futures/data/topLongShortPositionRatio``)
      - ``global_account`` — all-trader account long/short ratio
                             (``/futures/data/globalLongShortAccountRatio``)

    ``longAccount`` is the long share (0-1); ``longShortRatio`` > 1 means longs
    outnumber shorts. A top-trader ratio markedly below the global ratio (tops
    less long than the crowd) is a classic contrary signal. Each series is
    fetched independently and a 429/unsupported one is skipped, so a partial
    block still returns the surviving series; only an outright failure of all
    three raises ``NoMarketDataError`` (router then emits a sentinel).

    ``period`` selects the granularity (``"1d"`` default; intraday values
    ``5m``..``12h`` are timestamped ``YYYY-MM-DD HH:MM``). For ``1d`` the
    look-back is capped at ``_LSR_LIMIT_CAP`` daily rows; an explicit intraday
    period scales with its rows-per-day up to the 500-row endpoint cap.

    Historical windows: ``start_date``/``end_date`` (end inclusive) are passed
    through as ``startTime``/``endTime`` with ``end_date`` clamped to the run's
    analysis date (PIT; see :func:`_futures_data_window`). The endpoints retain
    only the LAST 30 DAYS — a window ending further back raises
    ``NoMarketDataError`` instead of returning today's positioning as if it
    were the past; :func:`get_binance_vision_metrics` covers deeper history.
    Without explicit dates the call remains the most-recent ``limit`` live
    form.
    """
    canonical = normalize_symbol_for_venue(symbol, "binance_perp")
    per_day = _futures_period_per_day(period)
    if period == "1d":
        # Cap so a runaway look_back_days can't request more than the
        # decision-useful tail (daily snapshots; older rows add noise).
        limit = max(1, min(int(look_back_days), _LSR_LIMIT_CAP))
    else:
        limit = max(1, min(int(look_back_days) * per_day, _FUTURES_DATA_LIMIT_CAP))

    extra, end_iso, _reaches_now, end_ms, coverage_note = _futures_data_window(
        symbol, canonical, look_back_days, start_date, end_date, period=period,
    )

    series_defs = [
        ("top_account", "/futures/data/topLongShortAccountRatio"),
        ("top_position", "/futures/data/topLongShortPositionRatio"),
        ("global_account", "/futures/data/globalLongShortAccountRatio"),
    ]
    records: list[dict] = []
    unavailable: list[str] = []
    for slabel, path in series_defs:
        try:
            params: dict[str, int | str] = {"symbol": canonical, "period": period}
            params.update(extra)
            if not extra:
                params["limit"] = limit
            rows = _http_get(path, params, symbol, canonical)
        except (NoMarketDataError, VendorRateLimitError) as exc:
            # One series 429'd or is unsupported for this contract — log and keep
            # the others rather than failing the whole call. The header notes
            # which are missing so absence reads as "fetch failed", not "no
            # positioning data".
            unavailable.append(slabel)
            logger.info("Binance %s L/S ratio unavailable for %s: %s",
                        slabel, canonical, exc)
            continue
        if not isinstance(rows, list):
            continue
        for r in rows:
            if not isinstance(r, dict) or r.get("timestamp") is None:
                continue
            records.append({
                "_ts": int(r["timestamp"]),
                "series": slabel,
                "time": _fmt_futures_data_ts(int(r["timestamp"]), period),
                "longAccount": r.get("longAccount"),
                "longShortRatio": r.get("longShortRatio"),
                "shortAccount": r.get("shortAccount"),
            })
    # Belt-and-suspenders PIT trim by timestamp ms (see _futures_data_window).
    if end_ms is not None:
        records = [r for r in records if r["_ts"] <= end_ms]
    for r in records:
        r.pop("_ts", None)

    if not records:
        raise NoMarketDataError(
            symbol, canonical,
            f"no long/short ratios (look_back_days={look_back_days})",
        )

    df = pd.DataFrame.from_records(records)
    vlabel = canonical if canonical == symbol.upper() else f"{canonical} (from {symbol})"
    header = f"# Perp USDT-M long/short ratio for {vlabel} (last {limit} {period} rows)\n"
    if end_iso is not None:
        header += (
            f"# window through {end_iso} (endpoints retain the last "
            f"{_FUTURES_DATA_RETENTION_DAYS} days only; deeper history: "
            f"get_binance_vision_metrics)\n"
        )
    if coverage_note:
        header += coverage_note
    if unavailable:
        header += (f"# ⚠ unavailable series: {', '.join(unavailable)} "
                   "(fetch failed/unsupported) — absence ≠ no positioning.\n")
    header += f"# Total records: {len(df)}\n"
    header += ("# series: top_account / top_position = 大户 (top traders), "
               "global_account = 全体; longShortRatio>1 = longs dominate; "
               "top < global = top traders less long than crowd (contrary).\n")
    header += f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    return header + df.to_csv(index=False)


def get_binance_taker_buy_sell(
    symbol: str, look_back_days: int = 7,
    start_date: str | None = None, end_date: str | None = None,
    period: str = "1d",
) -> str:
    """Taker buy/sell volume for a Binance USDT-M perp (order-flow aggression).

    ``/futures/data/takerlongshortRatio`` — the net of aggressive market buys vs
    sells. ``buySellRatio`` > 1 = takers buying more than selling (urgent long
    pressure); < 1 = selling pressure. ``buyVol`` / ``sellVol`` are the absolute
    taker volumes. A rally on buySellRatio < 1 (sellers dominant) is a low-
    conviction move; a dump on buySellRatio > 1 is often a capitulation wash.
    Returns ``time, buySellRatio, buyVol, sellVol``.

    ``period`` selects the granularity (``"1d"`` default; intraday values
    ``5m``..``12h`` are timestamped ``YYYY-MM-DD HH:MM`` and scale the row
    limit with rows-per-day, capped at the 500-row endpoint ceiling).

    Historical windows: ``start_date``/``end_date`` (end inclusive) are passed
    through as ``startTime``/``endTime`` with ``end_date`` clamped to the run's
    analysis date (PIT; see :func:`_futures_data_window`). The endpoint retains
    only the LAST 30 DAYS — a window ending further back raises
    ``NoMarketDataError`` instead of returning today's order flow as if it were
    the past; :func:`get_binance_vision_metrics` covers deeper history.
    Without explicit dates the call remains the most-recent ``limit`` live
    form.
    """
    canonical = normalize_symbol_for_venue(symbol, "binance_perp")
    per_day = _futures_period_per_day(period)
    if period == "1d":
        limit = max(1, min(int(look_back_days), _LSR_LIMIT_CAP))
    else:
        limit = max(1, min(int(look_back_days) * per_day, _FUTURES_DATA_LIMIT_CAP))

    extra, end_iso, _reaches_now, end_ms, coverage_note = _futures_data_window(
        symbol, canonical, look_back_days, start_date, end_date, period=period,
    )
    params: dict[str, int | str] = {"symbol": canonical, "period": period}
    params.update(extra)
    if not extra:
        params["limit"] = limit

    rows = _http_get("/futures/data/takerlongshortRatio", params, symbol, canonical)
    records: list[dict] = []
    if isinstance(rows, list):
        for r in rows:
            if not isinstance(r, dict) or r.get("timestamp") is None:
                continue
            records.append({
                "_ts": int(r["timestamp"]),
                "time": _fmt_futures_data_ts(int(r["timestamp"]), period),
                "buySellRatio": r.get("buySellRatio"),
                "buyVol": r.get("buyVol"),
                "sellVol": r.get("sellVol"),
            })
    # Belt-and-suspenders PIT trim by timestamp ms (see _futures_data_window).
    if end_ms is not None:
        records = [r for r in records if r["_ts"] <= end_ms]
    for r in records:
        r.pop("_ts", None)

    if not records:
        raise NoMarketDataError(
            symbol, canonical,
            f"no taker buy/sell volume (look_back_days={look_back_days})",
        )

    df = pd.DataFrame.from_records(records)
    vlabel = canonical if canonical == symbol.upper() else f"{canonical} (from {symbol})"
    header = f"# Perp USDT-M taker buy/sell for {vlabel} (last {limit} {period} rows)\n"
    if end_iso is not None:
        header += (
            f"# window through {end_iso} (endpoint retains the last "
            f"{_FUTURES_DATA_RETENTION_DAYS} days only; deeper history: "
            f"get_binance_vision_metrics)\n"
        )
    if coverage_note:
        header += coverage_note
    header += f"# Total records: {len(df)}\n"
    header += "# buySellRatio > 1 = takers buying > selling (long pressure); < 1 = selling pressure.\n"
    header += f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    return header + df.to_csv(index=False)


def get_binance_basis(
    symbol: str, look_back_days: int = 7,
    start_date: str | None = None, end_date: str | None = None,
    period: str = "1d",
) -> str:
    """Perp-vs-index basis for a Binance USDT-M perp.

    ``/futures/data/basis`` — premium/discount of the perpetual vs its
    underlying index price. Positive ``basis``/``basisRate`` = perp trades rich
    (long demand willing to pay up); negative = discount (short pressure /
    flight to the underlying). Returns ``time, basis, futuresPrice, indexPrice,
    basisRate``.

    Note: newer TRADIFI perps (e.g. stock-perps like AAPLUSDT/MUUSDT) are
    unsupported by this endpoint — Binance answers with an in-200 error body
    (e.g. code -4104), re-raised here as ``NoMarketDataError`` leading with
    the structural fact ("no basis data for this symbol") instead of the bare
    vendor code, so the router degrades to a sentinel and the analyst notes
    "basis unavailable" rather than seeing a cryptic API error. Major crypto
    perps (BTCUSDT, ETHUSDT, …) return real data.

    ``period`` selects the granularity (``"1d"`` default; intraday values are
    timestamped ``YYYY-MM-DD HH:MM`` and scale the row limit, capped at the
    500-row endpoint ceiling).

    Historical windows: ``start_date``/``end_date`` (end inclusive) are passed
    through as ``startTime``/``endTime`` with ``end_date`` clamped to the run's
    analysis date (PIT; see :func:`_futures_data_window`). The endpoint retains
    only the LAST 30 DAYS — a window ending further back raises
    ``NoMarketDataError`` instead of returning today's basis as if it were the
    past. Without explicit dates the call remains the most-recent ``limit``
    live form.
    """
    canonical = normalize_symbol_for_venue(symbol, "binance_perp")
    per_day = _futures_period_per_day(period)
    if period == "1d":
        limit = max(1, min(int(look_back_days), _LSR_LIMIT_CAP))
    else:
        limit = max(1, min(int(look_back_days) * per_day, _FUTURES_DATA_LIMIT_CAP))

    extra, end_iso, _reaches_now, end_ms, coverage_note = _futures_data_window(
        symbol, canonical, look_back_days, start_date, end_date, period=period,
    )
    params: dict[str, int | str] = {
        "pair": canonical, "contractType": "PERPETUAL", "period": period,
    }
    params.update(extra)
    if not extra:
        params["limit"] = limit

    try:
        rows = _http_get("/futures/data/basis", params, symbol, canonical)
    except NoMarketDataError as exc:
        if exc.vendor_code is None:
            raise  # transport/HTTP failure — keep the vendor's own message
        # API-level rejection: the endpoint does not serve basis rows for
        # this pair (TRADIFI/stock perps like MUUSDT have no
        # /futures/data/basis coverage). Lead with the structural fact;
        # the vendor's code/msg stays in the detail for debuggability.
        raise NoMarketDataError(
            symbol, canonical,
            f"no basis data for this symbol (the /futures/data/basis "
            f"endpoint does not cover it): {exc.detail}",
        ) from exc
    records: list[dict] = []
    if isinstance(rows, list):
        for r in rows:
            if not isinstance(r, dict) or r.get("timestamp") is None:
                continue
            records.append({
                "_ts": int(r["timestamp"]),
                "time": _fmt_futures_data_ts(int(r["timestamp"]), period),
                "basis": r.get("basis"),
                "futuresPrice": r.get("futuresPrice"),
                "indexPrice": r.get("indexPrice"),
                "basisRate": r.get("basisRate"),
            })
    # Belt-and-suspenders PIT trim by timestamp ms (see _futures_data_window).
    if end_ms is not None:
        records = [r for r in records if r["_ts"] <= end_ms]
    for r in records:
        r.pop("_ts", None)

    if not records:
        raise NoMarketDataError(
            symbol, canonical,
            f"no basis (look_back_days={look_back_days})",
        )

    df = pd.DataFrame.from_records(records)
    vlabel = canonical if canonical == symbol.upper() else f"{canonical} (from {symbol})"
    header = f"# Perp USDT-M basis for {vlabel} (last {limit} {period} rows)\n"
    if end_iso is not None:
        header += (
            f"# window through {end_iso} (endpoint retains the last "
            f"{_FUTURES_DATA_RETENTION_DAYS} days only)\n"
        )
    if coverage_note:
        header += coverage_note
    header += f"# Total records: {len(df)}\n"
    header += ("# basis = futuresPrice - indexPrice; positive = perp rich (long demand), "
               "negative = discount (short pressure).\n")
    header += f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    return header + df.to_csv(index=False)


def get_binance_premium_index(symbol: str) -> str:
    """Mark-price snapshot for a Binance USDT-M perp (``/fapi/v1/premiumIndex``).

    One weight-1 call returning ``markPrice``, ``indexPrice``,
    ``lastFundingRate`` (the rate currently in effect, payable at the next
    settlement) and ``nextFundingTime``. Binance liquidates USDT-M positions
    against the MARK price, not the last traded price — liquidation-distance
    claims must anchor to ``markPrice`` (vs ``indexPrice`` for the premium
    displacement), while the funding-rate history tool carries the settlement
    history. Returns ``symbol, markPrice, indexPrice, markVsIndexPct,
    lastFundingRate, nextFundingTime, time``.

    Live-only snapshot: bound to the analyst only for non-historical runs
    (same gate as the OI live snapshot); raising
    :class:`NoMarketDataError` / :class:`VendorRateLimitError` degrades to
    the optional-category sentinel.
    """
    canonical = normalize_symbol_for_venue(symbol, "binance_perp")

    data = _http_get(
        "/fapi/v1/premiumIndex", {"symbol": canonical}, symbol, canonical,
    )

    if not isinstance(data, dict) or data.get("markPrice") is None:
        raise NoMarketDataError(
            symbol, canonical, "premiumIndex returned no markPrice"
        )

    def _f(key: str) -> float | None:
        raw = data.get(key)
        if raw is None:
            return None
        try:
            return float(raw)
        except (TypeError, ValueError):
            return None

    mark, index = _f("markPrice"), _f("indexPrice")
    mark_vs_index_pct = (
        round((mark / index - 1.0) * 100.0, 4)
        if mark is not None and index not in (None, 0)
        else None
    )
    nft = data.get("nextFundingTime")
    ts = data.get("time")
    records = [
        {
            "symbol": data.get("symbol", canonical),
            "markPrice": data.get("markPrice"),
            "indexPrice": data.get("indexPrice"),
            "markVsIndexPct": mark_vs_index_pct,
            "lastFundingRate": data.get("lastFundingRate"),
            "nextFundingTime": (
                datetime.fromtimestamp(int(nft) / 1000, tz=UTC)
                .strftime("%Y-%m-%d %H:%M:%S UTC")
                if isinstance(nft, (int, float)) and nft > 0 else None
            ),
            "time": (
                datetime.fromtimestamp(int(ts) / 1000, tz=UTC)
                .strftime("%Y-%m-%d %H:%M:%S UTC")
                if isinstance(ts, (int, float)) and ts > 0 else None
            ),
        }
    ]

    df = pd.DataFrame.from_records(records)
    vlabel = canonical if canonical == symbol.upper() else f"{canonical} (from {symbol})"
    header = f"# Perp USDT-M mark-price snapshot for {vlabel}\n"
    header += (
        "# liquidations trigger on markPrice (not last price); lastFundingRate "
        "is the rate in effect for the NEXT settlement; markVsIndexPct = mark "
        "premium vs index (%).\n"
    )
    header += f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    return header + df.to_csv(index=False)


# Documented /fapi/v1/depth limit choices (the endpoint rejects anything else).
_FAPI_DEPTH_LIMIT_CHOICES = (5, 10, 20, 50, 100, 500, 1000)


def get_binance_depth_snapshot(symbol: str, limit: int = 100) -> str:
    """Live order-book depth snapshot for a Binance USDT-M perp.

    ``/fapi/v1/depth`` — the current resting book. Returns a level-by-level
    ``level, bid_price, bid_qty, ask_price, ask_qty`` ladder plus a header with
    ``mid``, ``spread_bps`` and the top-``limit`` quantity imbalance
    ((bid_qty - ask_qty) / total — positive = heavier bid side). This is the
    execution-reality tool: stop/target fills happen INTO this book, and a
    thin side into the price's direction is slippage (or, at leverage, the
    liquidation-cascade accelerant). For how depth PERSISTED through past
    moves, see :func:`yialpha.dataflows.binance_vision.get_binance_vision_book_depth`.

    Live-only snapshot: bound to the analyst only for non-historical runs
    (same gate as the premium-index snapshot); ``limit`` must be one of the
    documented choices (5/10/20/50/100/500/1000). Raising
    :class:`NoMarketDataError` / :class:`VendorRateLimitError` degrades to the
    optional-category sentinel.
    """
    canonical = normalize_symbol_for_venue(symbol, "binance_perp")
    if int(limit) not in _FAPI_DEPTH_LIMIT_CHOICES:
        raise ValueError(
            f"limit must be one of {_FAPI_DEPTH_LIMIT_CHOICES}, got {limit}"
        )

    data = _http_get(
        "/fapi/v1/depth",
        {"symbol": canonical, "limit": int(limit)},
        symbol,
        canonical,
    )

    bids = data.get("bids") if isinstance(data, dict) else None
    asks = data.get("asks") if isinstance(data, dict) else None
    if not isinstance(bids, list) or not isinstance(asks, list) or not bids or not asks:
        raise NoMarketDataError(symbol, canonical, "depth returned no levels")

    def _levels(rows: list, price_i: int = 0, qty_i: int = 1) -> list[tuple[float, float]]:
        out: list[tuple[float, float]] = []
        for lv in rows:
            if not isinstance(lv, list) or len(lv) < 2:
                continue
            try:
                out.append((float(lv[price_i]), float(lv[qty_i])))
            except (TypeError, ValueError):
                continue
        return out

    bid_levels = _levels(bids)
    ask_levels = _levels(asks)
    if not bid_levels or not ask_levels:
        raise NoMarketDataError(symbol, canonical, "depth levels unparseable")

    best_bid, best_ask = bid_levels[0][0], ask_levels[0][0]
    mid = (best_bid + best_ask) / 2.0
    spread_bps = (best_ask - best_bid) / mid * 1e4 if mid else None
    bid_qty = sum(q for _p, q in bid_levels)
    ask_qty = sum(q for _p, q in ask_levels)
    imbalance = (
        (bid_qty - ask_qty) / (bid_qty + ask_qty)
        if (bid_qty + ask_qty) > 0 else None
    )

    records = []
    for i in range(max(len(bid_levels), len(ask_levels))):
        records.append(
            {
                "level": i + 1,
                "bid_price": bid_levels[i][0] if i < len(bid_levels) else None,
                "bid_qty": bid_levels[i][1] if i < len(bid_levels) else None,
                "ask_price": ask_levels[i][0] if i < len(ask_levels) else None,
                "ask_qty": ask_levels[i][1] if i < len(ask_levels) else None,
            }
        )

    df = pd.DataFrame.from_records(records)
    vlabel = canonical if canonical == symbol.upper() else f"{canonical} (from {symbol})"
    header = f"# Perp USDT-M order-book depth snapshot for {vlabel} (top {int(limit)})\n"
    header += (
        f"# mid={mid} spread={spread_bps:.1f}bps; top-{int(limit)} qty imbalance="
        f"{imbalance:.3f}" if imbalance is not None and spread_bps is not None
        else "# mid/spread/imbalance unavailable"
    )
    header += (
        " (positive imbalance = heavier bid side).\n"
        "# prices are per-level; fills walk the ladder — a thin side into the "
        "price's direction is slippage / liquidation-cascade risk.\n"
    )
    header += f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    return header + df.to_csv(index=False)


# ---- Binance SPOT (crypto_spot asset type) ---------------------------------
# Spot public market data (/api/v3/*) mirrors the perp shape: same 12-tuple
# kline array, same X-MBX-USED-WEIGHT-1M header, same 429/418 throttle — only
# the host (api.binance.com vs fapi.binance.com) and the weight budget differ.
# Spot has NO funding / open-interest / long-short / taker / basis endpoints;
# it contributes OHLCV + 24h ticker, and the cross-venue spot-perp basis below.
# All spot calls reuse proxy_map() (SOCKS5) and the reactive 429/418 floor via
# _http_get; they count against the "spot" product-line limiter budget, which
# Binance tallies independently from fapi on the same IP.

def _spot_host() -> str:
    """Return the Binance spot REST host.

    Default is ``api.binance.com`` (same family as fapi.binance.com, proven
    through the SOCKS5 proxy). Flip ``binance_spot_mirror``
    (``YIALPHA_BINANCE_SPOT_MIRROR``) on to use the key-free market-data mirror
    ``data-api.binance.vision`` instead — same data, no API key, Binance's
    recommended host for read-only consumers. Default off = conservative.
    """
    if get_config().get("binance_spot_mirror", False):
        return _SPOT_MIRROR_BASE
    return _SPOT_BASE


def get_binance_spot_klines(
    symbol: str,
    start_date: str,
    end_date: str,
    interval: str = "1d",
) -> str:
    """Daily OHLCV for a Binance SPOT pair.

    Mirrors :func:`get_binance_klines`'s output shape exactly — header block +
    CSV with ``Open, High, Low, Close, Adj Close, Volume`` (``Adj Close`` ==
    ``Close``; spot has no splits) — so the downstream stockstats indicator path
    is reusable for spot runs. Hits ``/api/v3/klines`` on the spot host.
    """
    df = binance_klines_frame(
        symbol, start_date, end_date, interval=interval, venue="binance_spot"
    )
    canonical = normalize_symbol_for_venue(symbol, "binance_spot")
    end_date = current_pit_end(end_date) or end_date
    label = canonical if canonical == symbol.upper() else f"{canonical} (from {symbol})"
    header = f"# Spot USDT klines for {label} from {start_date} to {end_date}\n"
    header += f"# Total records: {len(df)}\n"
    header += f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    return header + df.to_csv()


def get_binance_spot_ticker24(symbol: str) -> str:
    """24h rolling stats for a Binance SPOT pair.

    ``/api/v3/ticker/24hr`` — the spot counterpart to the perp-native signals:
    latest price, 24h change %, high/low, base + quote volume. Spot has no
    funding/OI, so this (with the OHLCV klines) is the spot market snapshot.
    Returns a single-row ``lastPrice, priceChangePercent, highPrice, lowPrice,
    volume, quoteVolume`` CSV.
    """
    canonical = normalize_symbol_for_venue(symbol, "binance_spot")

    data = _http_get(
        "/api/v3/ticker/24hr",
        {"symbol": canonical},
        symbol,
        canonical,
        base=_spot_host(),
        weight_key="spot",
    )

    if not isinstance(data, dict):
        raise NoMarketDataError(symbol, canonical, "24h ticker returned non-object body")

    records = [
        {
            "lastPrice": data.get("lastPrice"),
            "priceChangePercent": data.get("priceChangePercent"),
            "highPrice": data.get("highPrice"),
            "lowPrice": data.get("lowPrice"),
            "volume": data.get("volume"),
            "quoteVolume": data.get("quoteVolume"),
        }
    ]

    df = pd.DataFrame.from_records(records)
    label = canonical if canonical == symbol.upper() else f"{canonical} (from {symbol})"
    header = f"# Spot USDT 24h ticker for {label}\n"
    header += f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    return header + df.to_csv(index=False)


def _sig_digits(x: float, digits: int = 6) -> float:
    """Round to ``digits`` SIGNIFICANT digits, not decimal places.

    ``round(x, 6)`` collapses sub-cent pairs: a 1.234567e-05 perp close loses
    five of its six significant digits, and a 2.4e-07 basis rounds to exactly
    0.0 while its basisRate says 2% — the display twin of the round(2) P0
    (round-5 audit, 2026-08-16). Significant-digit rounding keeps the same
    six-digit precision at every price magnitude.
    """
    if not math.isfinite(x) or x == 0.0:
        return x
    return float(f"%.{digits}g" % x)


def get_binance_spot_perp_basis(
    symbol: str, look_back_days: int = 7,
    start_date: str | None = None, end_date: str | None = None,
) -> str:
    """Cross-venue basis: Binance USDT-M perp vs Binance SPOT (crypto_spot).

    The genuinely new signal spot unlocks: pull daily closes from BOTH
    ``/fapi/v1/klines`` (perp) and ``/api/v3/klines`` (spot) for the same
    ``<BASE>USDT`` symbol, align by date, and compute:

      - ``basis``     = perpClose - spotClose
      - ``basisRate`` = basis / spotClose

    Positive basis = the perpetual trades rich vs spot (long demand willing to
    pay a premium to avoid settling); negative = discount (short pressure /
    flight to spot). This is the "real" basis traders watch — distinct from the
    perp-native :func:`get_binance_basis`, which is perp-vs-Binance-index.

    Backtest-safe windows: ``end_date`` (end inclusive) is clamped to the run's
    pinned analysis date via :func:`current_pit_end`, and the default window's
    end ("now" in live mode) is clamped the same way — so a pinned backtest can
    no longer see future closes. An intraday run's last row compares both
    venues' forming daily candles (same convention as the klines endpoints).

    Either side raising :class:`NoMarketDataError` / :class:`VendorRateLimitError`
    propagates so the router degrades this optional tool to a sentinel (the run
    continues without the basis column). Major USDT pairs (BTC/ETH/…) have both
    a deep perp and spot book; newer TRADIFI perps without a spot listing will
    cleanly degrade.
    """
    canonical = normalize_symbol_for_venue(symbol, "binance_spot")
    # Cap the window like the other perp daily series; older rows add noise.
    limit = max(1, min(int(look_back_days), _LSR_LIMIT_CAP))

    # Window: explicit dates (PIT-clamped) or the last `limit` days ending at
    # the PIT-clamped "now". Both venues queried over the same [start_ms,
    # end_ms] so the closes align by date.
    if end_date:
        end_date = current_pit_end(end_date) or end_date
        end_dt = datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=UTC)
    else:
        now_iso = datetime.now(UTC).strftime("%Y-%m-%d")
        end_date = current_pit_end(now_iso) or now_iso
        end_dt = datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=UTC)
    if start_date:
        start_dt = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=UTC)
    else:
        start_dt = end_dt - timedelta(days=limit)
    start_ms = int(start_dt.timestamp() * 1000)
    end_ms = int((end_dt.timestamp() + 86399) * 1000)  # end-of-day inclusive

    # Perp leg — fapi host/budget (defaults).
    perp_rows = _paginate_history(
        "/fapi/v1/klines",
        {"symbol": canonical, "interval": "1d"},
        _FAPI_KLINES_LIMIT,
        lambda k: k[0],
        start_ms,
        end_ms,
        symbol,
        canonical,
    )
    # Spot leg — spot host/budget.
    spot_rows = _paginate_history(
        "/api/v3/klines",
        {"symbol": canonical, "interval": "1d"},
        _SPOT_KLINES_LIMIT,
        lambda k: k[0],
        start_ms,
        end_ms,
        symbol,
        canonical,
        base=_spot_host(),
        weight_key="spot",
    )

    def _close_by_date(rows: list) -> dict:
        # kline[0] = open_time (ms), kline[4] = close. Key by UTC date string.
        out: dict[str, float] = {}
        for k in rows:
            if not isinstance(k, list) or len(k) < 6:
                continue
            d = datetime.fromtimestamp(int(k[0]) / 1000, tz=UTC).strftime("%Y-%m-%d")
            try:
                out[d] = float(k[4])
            except (TypeError, ValueError):
                continue
        return out

    perp_map = _close_by_date(perp_rows if isinstance(perp_rows, list) else [])
    spot_map = _close_by_date(spot_rows if isinstance(spot_rows, list) else [])

    # Inner join on date so each row compares like-for-like closes.
    dates = sorted(set(perp_map) & set(spot_map))
    if not dates:
        raise NoMarketDataError(
            symbol, canonical,
            f"no overlapping perp/spot daily closes (look_back_days={look_back_days})",
        )

    records = []
    for d in dates:
        pc, sc = perp_map[d], spot_map[d]
        basis = pc - sc
        records.append(
            {
                "date": d,
                "perpClose": _sig_digits(pc),
                "spotClose": _sig_digits(sc),
                "basis": _sig_digits(basis),
                "basisRate": (_sig_digits(basis / sc) if sc else None),
            }
        )

    df = pd.DataFrame.from_records(records)
    vlabel = canonical if canonical == symbol.upper() else f"{canonical} (from {symbol})"
    header = f"# Spot-perp basis for {vlabel} (last {limit} days)\n"
    header += f"# window through {end_date} (end inclusive)\n"
    header += f"# Total records: {len(df)}\n"
    header += ("# basis = perpClose - spotClose; positive = perp rich vs spot "
               "(long premium), negative = discount (short pressure).\n")
    header += f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    return header + df.to_csv(index=False)


# --- Tokenized US-equity perp resolution -------------------------------------
#
# Binance exchangeInfo types every USDT-M contract with an ``underlyingType``
# ("COIN" / "EQUITY" / "HK_EQUITY" / "KR_EQUITY" / "CN_EQUITY" / "COMMODITY" /
# "PREMARKET" / "INDEX"), which is the authoritative answer to "is this perp a
# tokenized stock?". The set is fetched ONCE per run by an explicit warm call
# (CLI selection / graph propagate) and then read cache-only: mid-graph and
# test-path resolvers are network-free and deterministic (seed snapshot when
# unwamed). Only US EQUITY is resolved — HK/KR/CN equity bases have no
# rule-based mapping to their Yahoo listings, and commodities / pre-IPO
# contracts have no company fundamentals at all.

_EQUITY_PERP_BASES_LOCK = threading.Lock()
_EQUITY_PERP_BASES_CACHE: frozenset[str] | None = None
#: Per-base listing facts from the LAST SUCCESSFUL exchangeInfo warm:
#: ``{base: {"onboard_date": iso|None, "status": "TRADING"}}``. Cache-only like
#: the base set — never fetched on read. Empty when the process only ever saw
#: the static seed (the seed carries no onboard dates).
_EQUITY_PERP_LISTING_CACHE: dict[str, dict] | None = None
#: When the last successful warm happened (``time.monotonic()`` seconds).
#: A SUCCESSFUL warm used to be cached for the whole process lifetime, so a
#: long-lived web subprocess never noticed newly listed/delisted equity perps
#: after its first warm — the TTL re-fetches at perp-run start when stale.
_EQUITY_PERP_WARMED_AT_MONO: float | None = None
_EQUITY_PERP_WARM_TTL_S = 6 * 3600.0


def refresh_equity_perp_bases() -> None:
    """Drop the equity-perp base/listing caches (tests / forced re-warm): back to seed."""
    global _EQUITY_PERP_BASES_CACHE, _EQUITY_PERP_LISTING_CACHE
    global _EQUITY_PERP_WARMED_AT_MONO
    with _EQUITY_PERP_BASES_LOCK:
        _EQUITY_PERP_BASES_CACHE = None
        _EQUITY_PERP_LISTING_CACHE = None
        _EQUITY_PERP_WARMED_AT_MONO = None


def equity_perp_bases() -> frozenset[str]:
    """Current US-equity perp base set — cache-only, NEVER fetches.

    Returns the warmed exchangeInfo snapshot when available, else the static
    seed snapshot from :mod:`yialpha.dataflows.symbol_utils`. Keeping this
    fetch-free means every resolver call (instrument context, analyst filter,
    tool remap) is network-free and deterministic — freshness comes from the
    ONE explicit :func:`warm_equity_perp_bases` call at perp-run start.
    """
    with _EQUITY_PERP_BASES_LOCK:
        if _EQUITY_PERP_BASES_CACHE is not None:
            return _EQUITY_PERP_BASES_CACHE
    return _EQUITY_PERP_SEED_BASES


def equity_perp_listing_info() -> dict[str, dict]:
    """Per-base listing facts from the last successful warm (cache-only).

    ``{base: {"onboard_date": "YYYY-MM-DD"|None, "status": "TRADING"}}``.
    Empty when no exchangeInfo snapshot has been warmed yet (seed mode) —
    callers must treat an absent entry as "classification evidence missing",
    never as "not listed". Fetch-free by the same contract as
    :func:`equity_perp_bases`.
    """
    with _EQUITY_PERP_BASES_LOCK:
        if _EQUITY_PERP_LISTING_CACHE is None:
            return {}
        return dict(_EQUITY_PERP_LISTING_CACHE)


def _persist_instrument_registry(symbol_rows: list) -> None:
    """Append the fresh exchangeInfo payload to the PIT instrument registry.

    Same payload the warm itself just fetched — ZERO extra HTTP. Gated on
    the ``instrument_registry`` config flag (V2.1 record stage; off = the
    warm path is byte-identical to pre-registry). Fail-soft: any failure
    downgrades to a WARNING and never aborts the warm — the in-memory
    classification caches remain authoritative for this run either way.
    """
    try:
        if not get_config().get("instrument_registry"):
            return
        from yialpha.instruments.registry import snapshot_instruments
        from yialpha.ledger.sqlite import utc_now_iso

        snapshot_instruments(
            [row for row in symbol_rows if isinstance(row, dict)],
            available_at=utc_now_iso(),
        )
    except Exception as exc:  # noqa: BLE001 — registry persistence is fail-soft
        logger.warning("instrument registry snapshot skipped: %s", exc)


def warm_equity_perp_bases() -> frozenset[str]:
    """Fetch the live EQUITY perp listing once per TTL window (perp-run start).

    exchangeInfo -> underlyingType == EQUITY, status TRADING. On ANY failure
    (network, rate limit, parse, empty universe) falls back to the static seed
    with a WARNING — fundamentals-analyst eligibility then rides on seed
    freshness, which is exactly the degraded mode the seed exists for. The
    listing snapshot is current-state (no as-of date): it gates analyst
    ELIGIBILITY only, never data content — the fundamentals vendors
    themselves remain PIT-correct by date.

    With the ``instrument_registry`` flag on, the same payload is also
    appended to the persistent instrument registry (fail-soft WARNING; the
    registry is the record-stage PIT store that survives restarts).

    A failed fetch is NOT cached: the seed is returned for THIS call only, so
    the next perp-run start re-attempts the live listing instead of serving a
    transient outage's seed for the whole process lifetime. A SUCCESSFUL warm
    is honoured for ``_EQUITY_PERP_WARM_TTL_S`` (6h) and then re-fetched at
    the next perp-run start — a long-lived web subprocess must pick up newly
    listed equity perps without a restart.
    """
    global _EQUITY_PERP_BASES_CACHE, _EQUITY_PERP_LISTING_CACHE
    global _EQUITY_PERP_WARMED_AT_MONO
    with _EQUITY_PERP_BASES_LOCK:
        if (
            _EQUITY_PERP_BASES_CACHE is not None
            and _EQUITY_PERP_WARMED_AT_MONO is not None
            and (time.monotonic() - _EQUITY_PERP_WARMED_AT_MONO)
            < _EQUITY_PERP_WARM_TTL_S
        ):
            return _EQUITY_PERP_BASES_CACHE
        try:
            payload = _http_get(
                "/fapi/v1/exchangeInfo",
                {},
                symbol_for_error="EQUITY_PERP_BASES",
                canonical="EQUITY_PERP_BASES",
            )
            symbols = (
                payload.get("symbols", []) if isinstance(payload, dict) else []
            )
            equity_rows = [
                s for s in symbols
                if isinstance(s, dict)
                and s.get("underlyingType") == "EQUITY"
                and s.get("status") == "TRADING"
                and s.get("quoteAsset") in ("USDT", "USDC")
            ]
            bases = frozenset(
                s["symbol"][: -len(s["quoteAsset"])] for s in equity_rows
            )
            if not bases:
                # A 200 with zero EQUITY rows would silently disable the
                # fundamentals analyst for every stock perp — treat it as a
                # failed fetch rather than an empty universe.
                raise NoMarketDataError(
                    "EQUITY_PERP_BASES", "EQUITY_PERP_BASES",
                    "exchangeInfo returned no TRADING EQUITY symbols",
                )
        except Exception as exc:  # noqa: BLE001 — fail-open floor, any failure
            if _EQUITY_PERP_BASES_CACHE is not None:
                # A previous successful warm exists but aged past the TTL: a
                # stale-but-real exchangeInfo snapshot beats the static seed —
                # serve it for this call and re-attempt at the next perp-run
                # start.
                logger.warning(
                    "warm_equity_perp_bases: exchangeInfo re-fetch failed "
                    "(%s); serving the previous warmed snapshot (%d bases) "
                    "for this call only",
                    exc, len(_EQUITY_PERP_BASES_CACHE),
                )
                return _EQUITY_PERP_BASES_CACHE
            logger.warning(
                "warm_equity_perp_bases: exchangeInfo fetch failed (%s); "
                "serving the static seed snapshot for this call only "
                "(%d bases; not cached — next perp-run start retries)",
                exc, len(_EQUITY_PERP_SEED_BASES),
            )
            return _EQUITY_PERP_SEED_BASES
        listing: dict[str, dict] = {}
        for row in equity_rows:
            base = row["symbol"][: -len(row["quoteAsset"])]
            if base in listing:
                continue  # first listing row per base wins (USDT vs USDC twin)
            onboard_ms = row.get("onboardDate")
            try:
                onboard_date = (
                    datetime.fromtimestamp(int(onboard_ms) / 1000, tz=UTC)
                    .strftime("%Y-%m-%d")
                    if isinstance(onboard_ms, (int, float)) and onboard_ms > 0
                    else None
                )
            except (OverflowError, OSError, ValueError):
                onboard_date = None
            listing[base] = {
                "onboard_date": onboard_date,
                "status": str(row.get("status") or "TRADING"),
            }
        # Record stage (V2.1): persist the WHOLE payload (all symbol rows,
        # not just EQUITY) into the PIT instrument registry — same response,
        # no second request. Fail-soft inside; flag-gated.
        _persist_instrument_registry(symbols)
        _EQUITY_PERP_BASES_CACHE = bases
        _EQUITY_PERP_LISTING_CACHE = listing
        _EQUITY_PERP_WARMED_AT_MONO = time.monotonic()
        return _EQUITY_PERP_BASES_CACHE


def stock_perp_underlying(ticker: str) -> str | None:
    """Tokenized-stock-perp resolver: Yahoo-ready underlying or None.

    Thin wrapper over the pure matcher in symbol_utils against the current
    base set (warmed snapshot or seed — never a fetch). The cheap syntactic
    pre-check short-circuits anything without a USDT/USDC quote before the
    set is even consulted.
    """
    if not isinstance(ticker, str):
        return None
    compact = ticker.strip().upper().replace("-", "")
    if not compact.endswith(("USDT", "USDC")):
        return None
    return tokenized_stock_perp_underlying(ticker, equity_perp_bases())


# ---------------------------------------------------------------------------
# V2.0 P0.4 — derivatives-stress input series (fail-open fetcher)
# ---------------------------------------------------------------------------

def _stress_series_from_records(
    rows: object, ts_key: str, value_key: str,
) -> pd.Series:  # type: ignore[type-arg]
    """Datetime-indexed float Series from one /futures/data|fapi record list.

    Rows with a missing/non-numeric value are dropped (not coerced to 0);
    sorted by timestamp ascending. An empty result is a valid Series the
    caller's window check will reject.
    """
    records: list[dict] = [r for r in rows if isinstance(r, dict)] if isinstance(
        rows, list
    ) else []
    data: dict[datetime, float] = {}
    for r in records:
        ts = r.get(ts_key)
        raw = r.get(value_key)
        if ts is None or raw is None:
            continue
        try:
            value = float(raw)
        except (TypeError, ValueError):
            continue
        data[datetime.fromtimestamp(int(ts) / 1000.0, tz=UTC)] = value
    return pd.Series(data).sort_index()


def derivatives_stress_series(
    symbol: str, end_date: str, window_days: int = 90,
) -> dict[str, Any]:
    """Trailing positioning series for the Derivatives Stress Score.

    Returns ``{funding, open_interest, global_lsr, basis, taker_ratio}`` —
    the first four as datetime-indexed Series, ``taker_ratio`` as the latest
    float. EVERY component is independently fail-open: a component that
    cannot be fetched is ``None`` (the pure ``compute_stress`` then drops it
    and lists it as missing) instead of failing the whole report. The window
    is PIT-clamped to ``end_date`` via :func:`current_pit_end`.

    Live-run practicality (why the REST paths, not the vision archive):
    ``/futures/data/*`` retains only the last 30 days AND rejects a request
    whose ``startTime``/``endTime`` span exceeds that horizon with HTTP 400
    ``-1130`` ("parameter 'startTime' is invalid") — the server does NOT
    silently truncate — so the OI/LSR window is clamped client-side before the
    request: ``endTime`` to ``min(end_ms, now)``, ``startTime`` to
    ``_FUTURES_DATA_RETENTION_DAYS`` back from that anchor. The ``min`` is
    deliberate, not a typo — ``end_ms`` is the LOCAL end-of-day, so a run after
    local midnight would otherwise measure retention back from an instant still
    in the future and lose a daily row that already exists (29 points, one
    short of ``MIN_WINDOW``); a replay's ``end_ms`` is already below now and is
    left exactly where PIT put it. The two typically arrive as ~30 points,
    enough for the score's ``MIN_WINDOW`` while the report carries the
    ``thin_history`` flag; funding (full-history ``/fapi/v1/fundingRate``) and
    the klines-based basis cover the full requested window. Downloading ~90
    per-day archive zips on a live decision path would cost far more than the
    extra context is worth; deep-history stress belongs to offline IC work.
    """
    canonical = normalize_symbol_for_venue(symbol, "binance_perp")
    end_clamped = current_pit_end(end_date) or end_date
    end_dt = datetime.strptime(end_clamped, "%Y-%m-%d").replace(tzinfo=UTC)
    start_dt = end_dt - timedelta(days=int(window_days))
    start_ms = int(start_dt.timestamp() * 1000)
    end_ms = int((end_dt.timestamp() + 86399) * 1000)  # end-of-day inclusive

    out: dict[str, Any] = {}

    try:
        rows = _http_get(
            "/fapi/v1/fundingRate",
            {
                "symbol": canonical,
                "startTime": start_ms,
                "endTime": end_ms,
                "limit": 1000,
            },
            symbol,
            canonical,
        )
        out["funding"] = _stress_series_from_records(rows, "fundingTime", "fundingRate")
    except Exception as exc:  # noqa: BLE001 — component fail-open, never abort
        logger.info("stress series funding unavailable for %s: %s", canonical, exc)
        out["funding"] = None

    # /futures/data/* retains only the last _FUTURES_DATA_RETENTION_DAYS days
    # and rejects a wider startTime/endTime span with HTTP 400 -1130 ("parameter
    # 'startTime' is invalid") — no silent truncation — so clamp client-side;
    # funding (/fapi full history) and basis (klines) keep the full start_ms.
    # Anchor on min(end_ms, now) because end_ms is the LOCAL end-of-day: past
    # local midnight it lies in the FUTURE, and measuring retention back from a
    # future instant drops a daily row that already exists (29 < MIN_WINDOW 30).
    data_end_ms = min(end_ms, _now_ms())
    data_start_ms = max(
        start_ms, data_end_ms - _FUTURES_DATA_RETENTION_DAYS * 86_400_000
    )

    for name, path, value_key in (
        ("open_interest", "/futures/data/openInterestHist", "sumOpenInterest"),
        ("global_lsr", "/futures/data/globalLongShortAccountRatio", "longShortRatio"),
    ):
        try:
            rows = _http_get(
                path,
                {
                    "symbol": canonical,
                    "period": "1d",
                    "startTime": data_start_ms,
                    "endTime": data_end_ms,
                    "limit": 30,
                },
                symbol,
                canonical,
            )
            series = _stress_series_from_records(rows, "timestamp", value_key)
            out[name] = series if len(series) > 0 else None
        except Exception as exc:  # noqa: BLE001 — component fail-open
            logger.info("stress series %s unavailable for %s: %s", name, canonical, exc)
            out[name] = None

    # Basis = perp close / index close − 1 over the window, from the two
    # aligned klines frames (both PIT-clamped by binance_klines_frame itself).
    try:
        perp = binance_klines_frame(
            symbol, start_dt.strftime("%Y-%m-%d"), end_clamped, "1d", "binance_perp",
            "last",
        )
        index = binance_klines_frame(
            symbol, start_dt.strftime("%Y-%m-%d"), end_clamped, "1d", "binance_perp",
            "index",
        )
        joined = pd.DataFrame(
            {
                "perp": perp[KLINE_CLOSE_COLUMN],
                "index": index[KLINE_CLOSE_COLUMN],
            }
        ).dropna()
        out["basis"] = (
            (joined["perp"] / joined["index"] - 1.0).dropna()
            if not joined.empty
            else None
        )
    except Exception as exc:  # noqa: BLE001 — component fail-open
        logger.info("stress series basis unavailable for %s: %s", canonical, exc)
        out["basis"] = None

    # Taker aggression: latest daily buySellRatio (float, not a series).
    try:
        rows = _http_get(
            "/futures/data/takerlongshortRatio",
            {
                "symbol": canonical,
                "period": "1d",
                "startTime": end_ms - 2 * 86_400_000,
                "endTime": end_ms,
                "limit": 2,
            },
            symbol,
            canonical,
        )
        series = _stress_series_from_records(rows, "timestamp", "buySellRatio")
        out["taker_ratio"] = float(series.iloc[-1]) if len(series) > 0 else None
    except Exception as exc:  # noqa: BLE001 — component fail-open
        logger.info("stress taker_ratio unavailable for %s: %s", canonical, exc)
        out["taker_ratio"] = None

    return out
