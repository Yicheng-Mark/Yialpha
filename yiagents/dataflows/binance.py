"""Binance USDT-M perpetual market-data vendor (Track A, analysis-only).

Three read-only public GET endpoints (no API key, no trading):

  - /fapi/v1/klines            daily OHLCV for the market analyst
  - /fapi/v1/fundingRate        funding-rate history (perp cost-of-carry)
  - /fapi/v1/openInterest       live open interest
  - /futures/data/openInterestHist  daily open-interest history

Returns CSV-shaped ``str`` (header + ``df.to_csv()``) so they slot into the
existing ``route_to_vendor`` plumbing exactly like the yfinance vendors. On
any failure they raise the typed errors from :mod:`yiagents.dataflows.errors`
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

import logging
import os
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests

from .binance_rate_limiter import get_binance_weight_limiter
from .config import get_config
from .errors import NoMarketDataError, VendorRateLimitError
from .symbol_utils import normalize_symbol_for_venue

logger = logging.getLogger(__name__)

_FAPI_BASE = "https://fapi.binance.com"
# connect, read — short connect, generous-but-bounded read so a stalled
# mid-response cannot hang the ticker (mirrors the openai_client read-timeout
# safety net; run_robust's watchdog is the OS-level floor).
_TIMEOUT = (5, 30)

# fapi per-request caps. klines tops out at 1500, fundingRate at 1000; without
# paging, a long range returns only the OLDEST page and silently drops the
# recent, decision-critical rows.
_FAPI_KLINES_LIMIT = 1500
_FAPI_FUNDING_LIMIT = 1000
# Runaway guard: ~50000 daily bars ≈ 137 years. Purely a safety ceiling so a
# mis-sized range can never spin the paginator unboundedly.
_FAPI_PAGINATION_SAFETY_CAP = 50000

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
    """
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
    return out


def _proxies() -> dict[str, str | None]:
    """Build a proxies dict from the run environment (read at call time).

    Returns ``{"http": ..., "https": ...}`` so requests routes through the
    SOCKS5 proxy the rest of the data layer uses (socks5h://127.0.0.1:1080).
    Missing env values fall back to ``None`` (requests' default behavior).
    """
    return {
        "http": os.environ.get("HTTP_PROXY"),
        "https": os.environ.get("HTTPS_PROXY"),
    }


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


def _http_get(
    path: str,
    params: dict,
    symbol_for_error: str,
    canonical: str,
    base: str = _FAPI_BASE,
    weight_key: str = "fapi",
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
    """
    proactive = get_config().get("binance_proactive_backoff", False)
    if proactive:
        # Back off BEFORE the request if the budget is hot, so we avoid tripping
        # a 429 rather than only reacting to one. The reactive 429/418 handling
        # below stays as the floor either way.
        get_binance_weight_limiter(weight_key).acquire()

    url = f"{base}{path}"
    resp = requests.get(url, params=params, proxies=_proxies(), timeout=_TIMEOUT)

    if proactive:
        _observe_weight(resp, weight_key)

    if resp.status_code in (429, 418):
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
        raise NoMarketDataError(
            symbol_for_error,
            canonical,
            f"Binance code {parsed.get('code')}: {parsed.get('msg')}",
        )

    return parsed


def get_binance_klines(
    symbol: str,
    start_date: str,
    end_date: str,
    interval: str = "1d",
) -> str:
    """Daily OHLCV for a Binance USDT-M perpetual pair.

    Returns a ``str`` shaped like yfinance's ``get_YFin_data_online`` output —
    header block + CSV with columns ``Open, High, Low, Close, Adj Close,
    Volume`` (``Adj Close`` mirrors ``Close`` since perps have no splits) — so
    the downstream stockstats indicator path is reusable. ``interval`` defaults
    to ``"1d"``; the analyst passes it through for intraday if ever needed.
    """
    canonical = normalize_symbol_for_venue(symbol, "binance_perp")

    start_ms = int(
        datetime.strptime(start_date, "%Y-%m-%d")
        .replace(tzinfo=timezone.utc)
        .timestamp()
        * 1000
    )
    end_dt = datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    # End-of-day so the requested end_date row is included.
    end_ms = int((end_dt.timestamp() + 86399) * 1000)

    rows = _paginate_history(
        "/fapi/v1/klines",
        {"symbol": canonical, "interval": interval},
        _FAPI_KLINES_LIMIT,
        lambda k: k[0],  # kline open_time (ms) is element 0
        start_ms,
        end_ms,
        symbol,
        canonical,
    )

    if not isinstance(rows, list) or not rows:
        raise NoMarketDataError(
            symbol, canonical, f"no klines between {start_date} and {end_date}"
        )

    # Binance kline array indices: [1]Open [2]High [3]Low [4]Close [5]Volume.
    # Index 0 is the open time (ms, UTC); cast numerics via pandas.
    records = []
    for k in rows:
        if not isinstance(k, list) or len(k) < 6:
            continue
        open_ms = int(k[0])
        records.append(
            {
                "Date": datetime.fromtimestamp(open_ms / 1000, tz=timezone.utc)
                .strftime("%Y-%m-%d"),
                "Open": float(k[1]),
                "High": float(k[2]),
                "Low": float(k[3]),
                "Close": float(k[4]),
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
    # Mirror yfinance: round numerics for cleaner display.
    for col in ("Open", "High", "Low", "Close", "Adj Close"):
        df[col] = df[col].round(2)

    label = canonical if canonical == symbol.upper() else f"{canonical} (from {symbol})"
    header = f"# Perp USDT-M klines for {label} from {start_date} to {end_date}\n"
    header += f"# Total records: {len(df)}\n"
    header += f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    return header + df.to_csv()


def get_binance_funding_rate(
    symbol: str,
    start_date: str,
    end_date: str,
) -> str:
    """Funding-rate history for a Binance USDT-M perpetual pair.

    Returns header + CSV with ``fundingTime, fundingRate, symbol``. Funding is
    charged every 8h and is the primary cost-of-carry / sentiment signal for a
    perp (persistently positive = longs pay shorts = crowding).
    """
    canonical = normalize_symbol_for_venue(symbol, "binance_perp")

    start_ms = int(
        datetime.strptime(start_date, "%Y-%m-%d")
        .replace(tzinfo=timezone.utc)
        .timestamp()
        * 1000
    )
    end_dt = datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
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

    records = [
        {
            "fundingTime": datetime.fromtimestamp(
                int(r["fundingTime"]) / 1000, tz=timezone.utc
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
    header += f"# Total records: {len(df)}\n"
    header += f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    return header + df.to_csv(index=False)


def get_binance_open_interest(symbol: str, look_back_days: int = 7) -> str:
    """Open-interest snapshot + daily history for a Binance USDT-M perp.

    Combines the live ``/fapi/v1/openInterest`` snapshot with the daily
    ``/futures/data/openInterestHist`` series (``look_back_days`` rows) into a
    single ``time, openInterest, openInterestValue`` table. Rising OI + rising
    price confirms a trend; rising OI + falling price signals crowded shorts
    (or longs unwinding). The live row is appended last with ``openInterestValue``
    blank (the snapshot endpoint exposes only the raw OI).
    """
    canonical = normalize_symbol_for_venue(symbol, "binance_perp")
    limit = max(1, int(look_back_days))

    hist = _http_get(
        "/futures/data/openInterestHist",
        {"symbol": canonical, "period": "1d", "limit": limit},
        symbol,
        canonical,
    )

    records = []
    if isinstance(hist, list):
        for r in hist:
            if not isinstance(r, dict) or r.get("timestamp") is None:
                continue
            records.append(
                {
                    "time": datetime.fromtimestamp(
                        int(r["timestamp"]) / 1000, tz=timezone.utc
                    ).strftime("%Y-%m-%d"),
                    "openInterest": r.get("sumOpenInterest"),
                    "openInterestValue": r.get("sumOpenInterestValue"),
                }
            )

    # Append the live snapshot so the analyst sees the most current OI too.
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
        logger.info("Binance live openInterest unavailable for %s: %s", canonical, exc)

    if not records:
        raise NoMarketDataError(
            symbol, canonical, f"no open-interest history (look_back_days={look_back_days})"
        )

    df = pd.DataFrame.from_records(records)

    label = canonical if canonical == symbol.upper() else f"{canonical} (from {symbol})"
    header = (
        f"# Perp USDT-M open interest for {label} (last {limit} days + live)\n"
    )
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


def get_binance_long_short_ratio(symbol: str, look_back_days: int = 7) -> str:
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
    """
    canonical = normalize_symbol_for_venue(symbol, "binance_perp")
    # Cap so a runaway look_back_days can't request more than the decision-useful
    # tail (these are daily snapshots; older rows add noise, not signal).
    limit = max(1, min(int(look_back_days), _LSR_LIMIT_CAP))

    series_defs = [
        ("top_account", "/futures/data/topLongShortAccountRatio"),
        ("top_position", "/futures/data/topLongShortPositionRatio"),
        ("global_account", "/futures/data/globalLongShortAccountRatio"),
    ]
    records: list[dict] = []
    for slabel, path in series_defs:
        try:
            rows = _http_get(
                path,
                {"symbol": canonical, "period": "1d", "limit": limit},
                symbol,
                canonical,
            )
        except (NoMarketDataError, VendorRateLimitError) as exc:
            # One series 429'd or is unsupported for this contract — log and keep
            # the others rather than failing the whole call.
            logger.info("Binance %s L/S ratio unavailable for %s: %s",
                        slabel, canonical, exc)
            continue
        if not isinstance(rows, list):
            continue
        for r in rows:
            if not isinstance(r, dict) or r.get("timestamp") is None:
                continue
            records.append({
                "series": slabel,
                "time": datetime.fromtimestamp(
                    int(r["timestamp"]) / 1000, tz=timezone.utc
                ).strftime("%Y-%m-%d"),
                "longAccount": r.get("longAccount"),
                "longShortRatio": r.get("longShortRatio"),
                "shortAccount": r.get("shortAccount"),
            })

    if not records:
        raise NoMarketDataError(
            symbol, canonical,
            f"no long/short ratios (look_back_days={look_back_days})",
        )

    df = pd.DataFrame.from_records(records)
    vlabel = canonical if canonical == symbol.upper() else f"{canonical} (from {symbol})"
    header = f"# Perp USDT-M long/short ratio for {vlabel} (last {limit} days)\n"
    header += f"# Total records: {len(df)}\n"
    header += ("# series: top_account / top_position = 大户 (top traders), "
               "global_account = 全体; longShortRatio>1 = longs dominate; "
               "top < global = top traders less long than crowd (contrary).\n")
    header += f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    return header + df.to_csv(index=False)


def get_binance_taker_buy_sell(symbol: str, look_back_days: int = 7) -> str:
    """Taker buy/sell volume for a Binance USDT-M perp (order-flow aggression).

    ``/futures/data/takerlongshortRatio`` — the net of aggressive market buys vs
    sells. ``buySellRatio`` > 1 = takers buying more than selling (urgent long
    pressure); < 1 = selling pressure. ``buyVol`` / ``sellVol`` are the absolute
    taker volumes. A rally on buySellRatio < 1 (sellers dominant) is a low-
    conviction move; a dump on buySellRatio > 1 is often a capitulation wash.
    Returns ``time, buySellRatio, buyVol, sellVol``.
    """
    canonical = normalize_symbol_for_venue(symbol, "binance_perp")
    limit = max(1, min(int(look_back_days), _LSR_LIMIT_CAP))

    rows = _http_get(
        "/futures/data/takerlongshortRatio",
        {"symbol": canonical, "period": "1d", "limit": limit},
        symbol,
        canonical,
    )
    records: list[dict] = []
    if isinstance(rows, list):
        for r in rows:
            if not isinstance(r, dict) or r.get("timestamp") is None:
                continue
            records.append({
                "time": datetime.fromtimestamp(
                    int(r["timestamp"]) / 1000, tz=timezone.utc
                ).strftime("%Y-%m-%d"),
                "buySellRatio": r.get("buySellRatio"),
                "buyVol": r.get("buyVol"),
                "sellVol": r.get("sellVol"),
            })

    if not records:
        raise NoMarketDataError(
            symbol, canonical,
            f"no taker buy/sell volume (look_back_days={look_back_days})",
        )

    df = pd.DataFrame.from_records(records)
    vlabel = canonical if canonical == symbol.upper() else f"{canonical} (from {symbol})"
    header = f"# Perp USDT-M taker buy/sell for {vlabel} (last {limit} days)\n"
    header += f"# Total records: {len(df)}\n"
    header += "# buySellRatio > 1 = takers buying > selling (long pressure); < 1 = selling pressure.\n"
    header += f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    return header + df.to_csv(index=False)


def get_binance_basis(symbol: str, look_back_days: int = 7) -> str:
    """Perp-vs-index basis for a Binance USDT-M perp.

    ``/futures/data/basis`` — premium/discount of the perpetual vs its
    underlying index price. Positive ``basis``/``basisRate`` = perp trades rich
    (long demand willing to pay up); negative = discount (short pressure /
    flight to the underlying). Returns ``time, basis, futuresPrice, indexPrice,
    basisRate``.

    Note: newer TRADIFI perps (e.g. stock-perps like AAPLUSDT/MUUSDT) are
    unsupported by this endpoint and return a Binance error body, which
    ``_http_get`` surfaces as ``NoMarketDataError`` — the router then degrades
    to a sentinel so the analyst notes "basis unavailable" rather than crashing.
    Major crypto perps (BTCUSDT, ETHUSDT, …) return real data.
    """
    canonical = normalize_symbol_for_venue(symbol, "binance_perp")
    limit = max(1, min(int(look_back_days), _LSR_LIMIT_CAP))

    rows = _http_get(
        "/futures/data/basis",
        {"pair": canonical, "contractType": "PERPETUAL",
         "period": "1d", "limit": limit},
        symbol,
        canonical,
    )
    records: list[dict] = []
    if isinstance(rows, list):
        for r in rows:
            if not isinstance(r, dict) or r.get("timestamp") is None:
                continue
            records.append({
                "time": datetime.fromtimestamp(
                    int(r["timestamp"]) / 1000, tz=timezone.utc
                ).strftime("%Y-%m-%d"),
                "basis": r.get("basis"),
                "futuresPrice": r.get("futuresPrice"),
                "indexPrice": r.get("indexPrice"),
                "basisRate": r.get("basisRate"),
            })

    if not records:
        raise NoMarketDataError(
            symbol, canonical,
            f"no basis (look_back_days={look_back_days})",
        )

    df = pd.DataFrame.from_records(records)
    vlabel = canonical if canonical == symbol.upper() else f"{canonical} (from {symbol})"
    header = f"# Perp USDT-M basis for {vlabel} (last {limit} days)\n"
    header += f"# Total records: {len(df)}\n"
    header += ("# basis = futuresPrice - indexPrice; positive = perp rich (long demand), "
               "negative = discount (short pressure).\n")
    header += f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    return header + df.to_csv(index=False)


# ---- Binance SPOT (crypto_spot asset type) ---------------------------------
# Spot public market data (/api/v3/*) mirrors the perp shape: same 12-tuple
# kline array, same X-MBX-USED-WEIGHT-1M header, same 429/418 throttle — only
# the host (api.binance.com vs fapi.binance.com) and the weight budget differ.
# Spot has NO funding / open-interest / long-short / taker / basis endpoints;
# it contributes OHLCV + 24h ticker, and the cross-venue spot-perp basis below.
# All spot calls reuse _proxies() (SOCKS5) and the reactive 429/418 floor via
# _http_get; they count against the "spot" product-line limiter budget, which
# Binance tallies independently from fapi on the same IP.

def _spot_host() -> str:
    """Return the Binance spot REST host.

    Default is ``api.binance.com`` (same family as fapi.binance.com, proven
    through the SOCKS5 proxy). Flip ``binance_spot_mirror``
    (``YIAGENTS_BINANCE_SPOT_MIRROR``) on to use the key-free market-data mirror
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
    canonical = normalize_symbol_for_venue(symbol, "binance_spot")

    start_ms = int(
        datetime.strptime(start_date, "%Y-%m-%d")
        .replace(tzinfo=timezone.utc)
        .timestamp()
        * 1000
    )
    end_dt = datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end_ms = int((end_dt.timestamp() + 86399) * 1000)  # end-of-day inclusive

    rows = _paginate_history(
        "/api/v3/klines",
        {"symbol": canonical, "interval": interval},
        _SPOT_KLINES_LIMIT,
        lambda k: k[0],  # kline open_time (ms) is element 0
        start_ms,
        end_ms,
        symbol,
        canonical,
        base=_spot_host(),
        weight_key="spot",
    )

    if not isinstance(rows, list) or not rows:
        raise NoMarketDataError(
            symbol, canonical, f"no spot klines between {start_date} and {end_date}"
        )

    # Same array indices as fapi klines: [1]Open [2]High [3]Low [4]Close [5]Volume.
    records = []
    for k in rows:
        if not isinstance(k, list) or len(k) < 6:
            continue
        open_ms = int(k[0])
        records.append(
            {
                "Date": datetime.fromtimestamp(open_ms / 1000, tz=timezone.utc)
                .strftime("%Y-%m-%d"),
                "Open": float(k[1]),
                "High": float(k[2]),
                "Low": float(k[3]),
                "Close": float(k[4]),
                "Adj Close": float(k[4]),
                "Volume": float(k[5]),
            }
        )

    if not records:
        raise NoMarketDataError(
            symbol, canonical, f"no parseable spot klines between {start_date} and {end_date}"
        )

    df = pd.DataFrame.from_records(records)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date").sort_index()
    for col in ("Open", "High", "Low", "Close", "Adj Close"):
        df[col] = df[col].round(2)

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


def get_binance_spot_perp_basis(symbol: str, look_back_days: int = 7) -> str:
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

    Either side raising :class:`NoMarketDataError` / :class:`VendorRateLimitError`
    propagates so the router degrades this optional tool to a sentinel (the run
    continues without the basis column). Major USDT pairs (BTC/ETH/…) have both
    a deep perp and spot book; newer TRADIFI perps without a spot listing will
    cleanly degrade.
    """
    canonical = normalize_symbol_for_venue(symbol, "binance_spot")
    # Cap the window like the other perp daily series; older rows add noise.
    limit = max(1, min(int(look_back_days), _LSR_LIMIT_CAP))

    # Window: the last `limit` days ending now (UTC). Both venues queried over
    # the same [start_ms, end_ms] so the closes align by date.
    end_dt = datetime.now(timezone.utc)
    start_dt = end_dt - timedelta(days=limit)
    start_ms = int(start_dt.timestamp() * 1000)
    end_ms = int(end_dt.timestamp() * 1000)

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
            d = datetime.fromtimestamp(int(k[0]) / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
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
                "perpClose": round(pc, 6),
                "spotClose": round(sc, 6),
                "basis": round(basis, 6),
                "basisRate": (round(basis / sc, 8) if sc else None),
            }
        )

    df = pd.DataFrame.from_records(records)
    vlabel = canonical if canonical == symbol.upper() else f"{canonical} (from {symbol})"
    header = f"# Spot-perp basis for {vlabel} (last {limit} days)\n"
    header += f"# Total records: {len(df)}\n"
    header += ("# basis = perpClose - spotClose; positive = perp rich vs spot "
               "(long premium), negative = discount (short pressure).\n")
    header += f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    return header + df.to_csv(index=False)
