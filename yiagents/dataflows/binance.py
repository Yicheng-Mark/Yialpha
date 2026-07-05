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
from datetime import datetime, timezone

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


def _observe_weight(resp) -> None:
    """Feed the server-reported IP weight to the process-wide limiter.

    Reads ``X-MBX-USED-WEIGHT-1M`` (Binance's per-IP rolling 1-min weight
    counter; requests' headers are case-insensitive). The header is optional —
    a missing or unparseable value is logged at debug and skipped, since the
    next response carries a fresh value and a single miss is harmless.
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
    get_binance_weight_limiter("fapi").observe(used)


def _http_get(path: str, params: dict, symbol_for_error: str, canonical: str) -> object:
    """GET a Binance public endpoint and return parsed JSON.

    Raises :class:`VendorRateLimitError` on 429/418 (IP throttle), and
    :class:`NoMarketDataError` for any non-200, a Binance error body
    (``{"code": ..., "msg": ...}``), or an empty response. Other
    ``requests`` exceptions propagate so the router can treat them as a
    transient vendor failure (and degrade when the category is optional).

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
        get_binance_weight_limiter("fapi").acquire()

    url = f"{_FAPI_BASE}{path}"
    resp = requests.get(url, params=params, proxies=_proxies(), timeout=_TIMEOUT)

    if proactive:
        _observe_weight(resp)

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

    rows = _http_get(
        "/fapi/v1/klines",
        {
            "symbol": canonical,
            "interval": interval,
            "startTime": start_ms,
            "endTime": end_ms,
        },
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
