"""CoinGecko keyless vendor for the optional ``crypto_native`` category.

Supplies token-level fundamentals (market cap, supply schedule, FDV, ATH/ATL,
developer & community footprint) for PURE-crypto instruments — the gap the
stock fundamentals analyst cannot fill for BTCUSDT-style tickers. Same
optional-category contract shape as ``sec_ownership``:

* **Free / keyless** — the public API tier (no DEMO key). The edge rejects
  default/anonymous Python user agents with 403, so every request carries a
  browser-style UA (verified live 2026-09-29).
* **Foreign host → rides the SOCKS5 proxy** via ``proxy_map()`` exactly like
  ``sec_edgar``; no env popping (that idiom exists for domestic-direct hosts).
* **Politeness** — ``MinIntervalThrottle(4.0)`` ≈ 15 req/min serial spacing,
  comfortably under the public tier's burst cap; a 429 maps to
  ``VendorRateLimitError`` so the router degrades to a sentinel.
* **PIT semantics** — the free tier only allows ``market_chart/range`` queries
  within the trailing ~365 days (HTTP 401 error 10012 beyond that, verified
  live). Live runs therefore get a current-only snapshot from ``/coins/{id}``
  (the same honest "today snapshot" shape as the yfinance overview), while
  historical runs get a date-bounded daily series (rows ``<= curr_date``);
  dates older than the 365-day window honestly degrade with an explicit
  message instead of fabricating. No current-only field is ever served into a
  historical run.
* **Fail-soft via typed errors** — transport failures and 4xx/5xx raise
  ``NoMarketDataError(detail=...)`` so the optional category degrades to the
  ``NO_DATA_AVAILABLE`` / ``DATA_UNAVAILABLE`` sentinel instead of aborting.
* **Disk-cached** through the shared ``cached_or_fetch`` (stale-serve +
  ``KIND_STALE_CACHE`` ledger contract inherited). One full trailing-window
  ``market_chart`` fetch per coin is cached per day and PIT-filtered per call,
  so every ``(coin, curr_date)`` view is deterministic within a day.

Symbol resolution: Binance perp/spot bases map to canonical CoinGecko ids via
``/search`` (exact symbol match, lowest ``market_cap_rank`` wins — the naive
``/coins/list`` symbol lookup has 12 junk "btc" collisions). Wrapper-multiplier
contracts (1000PEPE / 1000CAT / 1000000MOG) probe the un-prefixed token as a
fallback, and legacy dashed Yahoo symbols (BTC-USD, the auto-detected crypto
entrance) reduce to their base segment.
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import UTC, date, datetime, timedelta
from typing import Any

from .disk_cache import MinIntervalThrottle, cached_or_fetch, vendor_cache_dir
from .errors import NoMarketDataError, VendorRateLimitError
from .utils import is_historical_date, proxy_map

logger = logging.getLogger(__name__)

_BASE_URL = "https://api.coingecko.com/api/v3"
_TIMEOUT = 30
#: Public-tier politeness: ~15 req/min serial spacing (keyless bursts trip 429).
_throttler = MinIntervalThrottle(4.0)
#: Browser-style UA — the edge 403s default Python UAs (verified 2026-09-29).
_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)
#: The public tier refuses range queries starting before now - 365d (HTTP 401
#: error 10012). One safety day of margin keeps the single cached full-window
#: fetch on the allowed side of the moving boundary.
_FREE_TIER_RANGE_DAYS = 364


def _cache_dir() -> str:
    return vendor_cache_dir("coingecko")


def _throttle() -> None:
    _throttler.wait()


def _http_get_json(url: str, params: dict[str, Any] | None = None) -> Any:
    """GET a CoinGecko endpoint (throttle + UA + proxy) and parse JSON.

    Status mapping follows the ``sec_edgar`` idiom: 429 → VendorRateLimitError
    (router tries the next vendor / degrades), every other failure →
    NoMarketDataError with the status and a body excerpt in the detail so the
    sentinel the analyst sees carries the real reason.
    """
    import requests

    _throttle()
    try:
        resp = requests.get(
            url,
            headers={"User-Agent": _UA, "Accept": "application/json"},
            params=params,
            proxies=proxy_map(),
            timeout=_TIMEOUT,
        )
    except Exception as exc:  # noqa: BLE001 -- network failure -> typed error
        raise NoMarketDataError(url, detail=f"CoinGecko request failed: {exc!r}") from exc
    if resp.status_code == 429:
        raise VendorRateLimitError(f"CoinGecko returned 429 for {url}")
    if resp.status_code >= 400:
        body = resp.text[:160].replace("\n", " ")
        raise NoMarketDataError(
            url, detail=f"CoinGecko HTTP {resp.status_code}: {body}"
        )
    try:
        return resp.json()
    except ValueError as exc:
        raise NoMarketDataError(url, detail=f"CoinGecko returned non-JSON: {exc}") from exc


def _cached_json(filename: str, fetch, ttl_days: float) -> Any:
    """cached_or_fetch adapter returning parsed JSON (or raising on None).

    Parsed payloads are memoized per (cache dir, filename) keyed by the
    cache file's mtime: the analyst LLM tool loop can re-call these tools
    several times per run, and re-parsing a ~1MB JSON artifact per round is
    pure waste when the bytes are unchanged within the TTL (mtime moves on
    every cache refresh, so the memo cannot outlive the disk TTL).
    """
    raw = cached_or_fetch(
        _cache_dir(), filename, lambda: json.dumps(fetch()).encode("utf-8"),
        ttl_days=ttl_days, vendor="coingecko",
    )
    if raw is None:  # unreachable: fail_open is never set for this vendor
        raise NoMarketDataError(filename, detail="cache helper returned None")
    key = (_cache_dir(), filename)
    try:
        mtime = os.path.getmtime(os.path.join(key[0], filename))
    except OSError:
        mtime = -1.0
    hit = _PARSE_MEMO.get(key)
    # Round-9 S1: -1.0 is the no-file sentinel (cache write failing — AV
    # lock / read-only mount). Memo-hitting on it would freeze the first
    # successful parse for the process lifetime even as fresh bytes arrive
    # on every call; only a real mtime may serve from the memo.
    if mtime != -1.0 and hit is not None and hit[0] == mtime:
        return hit[1]
    parsed = json.loads(raw)
    # Round-13: bounded FIFO, same shape as binance's _HISTORY_MEMO_MAX —
    # the key carries the cache dir, so a process whose config rotates
    # data_cache_dir (or an unbounded artifact fan-out) would otherwise pin
    # every parse it ever did for its whole lifetime. The OLDEST INSERTED
    # key is evicted at the cap; a key already present is refreshed in
    # place (no new slot), so the mtime-hit semantics above are untouched.
    if len(_PARSE_MEMO) >= _PARSE_MEMO_MAX and key not in _PARSE_MEMO:
        _PARSE_MEMO.pop(next(iter(_PARSE_MEMO)))
    _PARSE_MEMO[key] = (mtime, parsed)
    return parsed


#: (cache dir, filename) -> (mtime, parsed payload); see _cached_json.
_PARSE_MEMO: dict[tuple[str, str], tuple[float, Any]] = {}
#: Round-13: FIFO cap for _PARSE_MEMO (binance._HISTORY_MEMO_MAX precedent).
_PARSE_MEMO_MAX = 64


def perp_base_symbol(ticker: str) -> str:
    """Reduce a crypto instrument symbol to its base token symbol.

    Handles the three shapes this category meets: compact Binance contracts
    (``BTCUSDT`` -> ``BTC``), dashed/separated pairs (legacy Yahoo-style
    auto-detected crypto like ``BTC-USD`` / ``ETH-BTC`` -> ``BTC`` / ``ETH``;
    ``BTC-USDT`` -> ``BTC``), and slash pairs. Shared by the DefiLlama
    vendor (single source for the contract -> base-symbol rule;
    ``routing._etf_base`` does the same strip but lives behind the routing
    layer's import graph).
    """
    raw = str(ticker).upper().strip()
    for sep in ("-", "/"):
        if sep in raw and raw.split(sep)[0]:
            return raw.split(sep)[0]
    for suffix in ("USDT", "USDC"):
        if raw.endswith(suffix) and len(raw) > len(suffix):
            return raw[: -len(suffix)]
    return raw


#: Binance wrapper-multiplier prefixes, longest first (1000000MOG tracks
#: MOG at 1,000,000x — a single "1000" strip would mangle it to 000MOG).
_MULTIPLIER_PREFIXES = ("1000000", "1000")


def _base_candidates(ticker: str) -> list[str]:
    """Resolution candidates: the bare base, then the multiplier-prefix-
    stripped form (Binance 1000PEPE / 1000CAT / 1000000MOG-style contracts
    track the un-prefixed token; CoinGecko lists the token). The remainder
    must be >= 3 chars so a genuinely short symbol like 1000X is not
    unwrapped."""
    base = perp_base_symbol(ticker)
    candidates = [base]
    for prefix in _MULTIPLIER_PREFIXES:
        if base.startswith(prefix) and len(base) - len(prefix) >= 3:
            candidates.append(base[len(prefix):])
            break
    return candidates


def _resolve_coin_id(ticker: str) -> tuple[str, str] | None:
    """Canonical CoinGecko id for the contract's base symbol, or None.

    ``/search`` returns ranked matches (with ``market_cap_rank``); the winner
    is the exact-symbol match with the lowest rank (unranked junk sorts last).
    Cached per base for 7 days — listings move slowly and each miss would
    otherwise burn a throttled request on every run.
    """
    for base in _base_candidates(ticker):
        payload = _cached_json(
            f"search_{base.lower()}.json",
            lambda b=base: _http_get_json(
                f"{_BASE_URL}/search", params={"query": b}
            ),
            ttl_days=7.0,
        )
        coins = payload.get("coins", []) if isinstance(payload, dict) else []
        matches = [
            c for c in coins
            if isinstance(c, dict)
            and str(c.get("symbol", "")).upper() == base.upper()
            and c.get("id")
        ]
        if matches:
            matches.sort(key=lambda c: (c.get("market_cap_rank") is None,
                                        c.get("market_cap_rank") or 10**9))
            best = matches[0]
            return str(best["id"]), base
    return None


def _end_of_day_ts(curr_date: str) -> int:
    """Epoch seconds at END of the as-of UTC day (inclusive row filter)."""
    d = date.fromisoformat(str(curr_date)[:10])
    return int(datetime(d.year, d.month, d.day, tzinfo=UTC)
               .timestamp()) + 86_400 - 1


def _fmt_usd(value: Any) -> str:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if v >= 1e12:
        return f"${v / 1e12:.2f}T"
    if v >= 1e9:
        return f"${v / 1e9:.2f}B"
    if v >= 1e6:
        return f"${v / 1e6:.2f}M"
    if v >= 1e3:
        return f"${v / 1e3:.2f}K"
    if 0 < v < 0.01:
        # sub-cent values (meme coins): a :.4f format collapses to $0.0000
        s = f"${v:.10f}".rstrip("0")
        return s + "0" if s.endswith(".") else s
    return f"${v:.4f}"


def _fmt_price(value: Any) -> str:
    """Price renderer with sub-cent support — meme-coin prices live at
    1e-5..1e-8 and a plain :.2f collapses them to a misleading 0.00."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if v >= 1:
        return f"{v:,.2f}"
    if v >= 0.01:
        return f"{v:.4f}"
    if v > 0:
        s = f"{v:.10f}".rstrip("0")
        return s + "0" if s.endswith(".") else s
    return f"{v}"


def _fmt_num(value: Any) -> str:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "n/a"
    for unit, div in (("T", 1e12), ("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if abs(v) >= div:
            return f"{v / div:.2f}{unit}"
    return f"{v:.2f}"


def _market_chart_series(coin_id: str) -> dict[str, list[tuple[str, float]]]:
    """Trailing-window daily series for a coin, cached once per day.

    One fetch of the FULL free-tier window (now-364d .. now) per coin per day;
    every ``(curr_date, look_back_days)`` view is then an in-memory filter of
    the cached series, so identical questions within a day give identical
    answers (backtest repeatability) and no per-date refetch storms.
    """
    to_ts = int(time.time())
    from_ts = to_ts - _FREE_TIER_RANGE_DAYS * 86_400
    payload = _cached_json(
        f"mchart_{coin_id}.json",
        lambda: _http_get_json(
            f"{_BASE_URL}/coins/{coin_id}/market_chart/range",
            params={"vs_currency": "usd", "from": from_ts, "to": to_ts},
        ),
        ttl_days=1.0,
    )

    def _daily(points: Any) -> list[tuple[str, float]]:
        """Last observation per UTC day (the API returns hourly points for
        short ranges — resample to daily so tables stay stable)."""
        if not isinstance(points, list):
            return []
        per_day: dict[int, float] = {}
        for p in points:
            if not (isinstance(p, (list, tuple)) and len(p) == 2):
                continue
            try:
                ts, val = int(p[0]), float(p[1])
            except (TypeError, ValueError):
                continue
            if ts > 10**11:  # market_chart timestamps are milliseconds
                ts //= 1000
            per_day[ts // 86_400] = val
        return [
            (datetime.fromtimestamp(day * 86_400, tz=UTC).strftime("%Y-%m-%d"), v)
            for day, v in sorted(per_day.items())
        ]

    return {
        "prices": _daily(payload.get("prices")),
        "market_caps": _daily(payload.get("market_caps")),
        "total_volumes": _daily(payload.get("total_volumes")),
    }


def _render_historical(
    coin_id: str, base: str, curr_date: str, look_back_days: int,
    series: dict[str, list[tuple[str, float]]],
) -> str:
    as_of = str(curr_date)[:10]
    lower = date.fromisoformat(as_of) - timedelta(days=look_back_days)

    def _window(points: list[tuple[str, float]]) -> list[tuple[str, float]]:
        return [(d, v) for d, v in points if lower.isoformat() <= d <= as_of]

    prices = _window(series["prices"])
    if not prices:
        return (
            f"Crypto token fundamentals for {base} (CoinGecko id '{coin_id}') as of "
            f"{as_of}: historical market data is NOT available — the free public "
            "CoinGecko tier only serves market_chart history within the trailing "
            f"~365 days, and {as_of} predates that window (or no rows fall inside "
            f"the {look_back_days}-day look-back). Report 'data not available' for "
            "this period; do not estimate price, market cap or supply."
        )

    caps = dict(_window(series["market_caps"]))
    vols = dict(_window(series["total_volumes"]))
    rows: list[str] = []
    for d, price in prices[-30:]:
        cap = caps.get(d)
        vol = vols.get(d)
        supply = (cap / price) if (cap is not None and price) else None
        rows.append(
            f"| {d} | {_fmt_price(price)} | {_fmt_usd(cap)} | {_fmt_usd(vol)} | {_fmt_num(supply)} |"
        )
    first_p, last_p = prices[0][1], prices[-1][1]
    first_c = caps.get(prices[0][0])
    last_c = caps.get(prices[-1][0])
    chg = (last_p / first_p - 1) * 100 if first_p else None
    cap_chg = ((last_c / first_c - 1) * 100) if (first_c and last_c) else None
    header = (
        f"Crypto token fundamentals — {base} (CoinGecko id '{coin_id}'), "
        f"point-in-time series as of {as_of} (rows filtered to date <= {as_of}; "
        f"window start {lower.isoformat()}). Circulating supply is derived as "
        f"market_cap / price. Free-tier window: trailing ~365 days only.\n\n"
    )
    summary = (
        f"\nWindow summary: price {_fmt_price(first_p)} -> {_fmt_price(last_p)} "
        f"({chg:+.1f}% over {len(prices)} days)" if chg is not None else
        f"\nWindow summary: {len(prices)} daily rows."
    )
    if cap_chg is not None:
        summary += (
            f"; market cap {_fmt_usd(first_c)} -> {_fmt_usd(last_c)} ({cap_chg:+.1f}%)"
        )
    supply_note = (
        "\nNOTE: ATH/ATL dates, developer and community stats are current-only "
        "snapshots on this tier and are deliberately EXCLUDED from this "
        "historical view (serving them would leak future information)."
    )
    return (
        header
        + "| date | close (USD) | market cap | 24h volume | circ. supply (derived) |\n"
        + "|---|---|---|---|---|\n"
        + "\n".join(rows)
        + summary
        + supply_note
    )


def _usd_field(value: Any) -> Any:
    """ath/atl and their dates come back as {currency: value} dicts — take USD."""
    if isinstance(value, dict):
        return value.get("usd")
    return value


def _render_live(coin_id: str, base: str, payload: dict[str, Any]) -> str:
    md = payload.get("market_data") or {}
    price = (md.get("current_price") or {}).get("usd")
    mcap = (md.get("market_cap") or {}).get("usd")
    vol = (md.get("total_volume") or {}).get("usd")
    circ = md.get("circulating_supply")
    total = md.get("total_supply")
    max_sup = md.get("max_supply")
    ath = _usd_field(md.get("ath"))
    ath_date = _usd_field(md.get("ath_date"))
    atl = _usd_field(md.get("atl"))
    atl_date = _usd_field(md.get("atl_date"))
    fdv = None
    for supply in (max_sup, total, circ):
        if supply and price:
            fdv = supply * price
            break
    chg = {
        k: md.get(k)
        for k in (
            "price_change_percentage_24h", "price_change_percentage_7d",
            "price_change_percentage_30d", "price_change_percentage_1y",
        )
    }
    categories = (payload.get("categories") or [])[:3]
    lines = [
        f"Crypto token fundamentals — {base} (CoinGecko id '{coin_id}')",
        "CURRENT SNAPSHOT (CoinGecko free tier; as-of = retrieval time, no "
        "historical dimension). All figures USD.",
        "",
        f"- Market cap rank: #{payload['market_cap_rank']}"
        if payload.get("market_cap_rank")
        else "- Market cap rank: n/a",
        f"- Price: {_fmt_price(price)}" if price else "- Price: n/a",
        f"- Market cap: {_fmt_usd(mcap)}",
        f"- 24h volume: {_fmt_usd(vol)}",
        f"- FDV (derived price x max/total supply): {_fmt_usd(fdv)}",
        f"- Circulating supply: {_fmt_num(circ)}",
        f"- Total supply: {_fmt_num(total)} | Max supply: {_fmt_num(max_sup)}",
        f"- ATH: {_fmt_usd(ath)} on {str(ath_date or '?')[:10]} | "
        f"ATL: {_fmt_usd(atl)} on {str(atl_date or '?')[:10]}",
        # join-then-branch: `"prefix" + join(...) or fallback` binds the `or`
        # to the whole truthy concatenation, making the fallback unreachable
        f"- Price change: {joined}" if (joined := " | ".join(
            f"{k.rsplit('_', 1)[-1]}: {v:+.1f}%"
            for k, v in chg.items() if isinstance(v, (int, float))
        )) else "- Price change: n/a",
        f"- Categories: {', '.join(categories)}" if categories else "- Categories: n/a",
        f"- Watchlist users (CoinGecko): {_fmt_num(payload.get('watchlist_portfolio_users'))}",
        "- Developer & community stats: not served on the free public tier "
        "(report 'data not available' if needed — do not estimate)",
        f"- Genesis date: {payload.get('genesis_date') or 'n/a'}",
        "",
        "Interpretation: supply schedule vs circulating supply indicates unlock "
        "dilution risk (FDV >> market cap = large unvested overhang); volume / "
        "market cap is a liquidity proxy; developer + community traction are "
        "leading indicators of network maintenance. If a field reads n/a, report "
        "'data not available' — do not estimate it.",
    ]
    return "\n".join(lines)


def get_crypto_token_fundamentals(
    ticker: str, curr_date: str | None = None, look_back_days: int = 180,
) -> str:
    """Token-level fundamentals for a pure-crypto instrument (CoinGecko).

    Live runs (curr_date empty or today) return a current snapshot: price,
    market cap + rank, FDV, circulating/total/max supply, ATH/ATL with dates,
    multi-horizon price change, developer & community footprint. Historical
    runs return a point-in-time daily series (price / market cap / volume /
    derived supply, rows <= curr_date) from the free tier's trailing ~365-day
    window; older as-of dates honestly report 'data not available'.

    Args:
        ticker: Contract symbol, e.g. BTCUSDT / ETHUSDT / 1000PEPEUSDT.
        curr_date: As-of date yyyy-mm-dd (the trade date being analyzed).
        look_back_days: Historical table window in days (default 180; the
            table itself samples the most recent 30 rows).

    Returns:
        Markdown snapshot/table, or an explicit not-available message. Raises
        NoMarketDataError when no canonical CoinGecko listing resolves (the
        router then degrades this optional category to a sentinel).
    """
    resolved = _resolve_coin_id(ticker)
    if resolved is None:
        base = perp_base_symbol(ticker)
        raise NoMarketDataError(
            ticker,
            detail=(
                f"no canonical CoinGecko listing found for base symbol '{base}' "
                "(unlisted or ambiguous symbol)"
            ),
        )
    coin_id, base = resolved

    if is_historical_date(curr_date):
        # is_historical_date deliberately routes malformed non-empty dates
        # (e.g. an LLM-echoed "2026/08/01") into the historical branch
        # (fail-closed); parsing here must degrade honestly rather than
        # crash on date.fromisoformat downstream.
        try:
            as_of = str(curr_date)[:10]
            date.fromisoformat(as_of)
        except ValueError:
            return (
                f"Crypto token fundamentals for {base}: the passed curr_date "
                f"'{curr_date}' is not a parseable yyyy-mm-dd date, so no "
                "point-in-time window can be computed. Report 'data not "
                "available' for this item; do not estimate."
            )
        series = _market_chart_series(coin_id)
        # Pass the validated slice: is_historical_date's bool return cannot
        # narrow str | None for the type checker, and str(curr_date)[:10] is
        # exactly — and solely — what _render_historical consumes.
        return _render_historical(coin_id, base, as_of, look_back_days, series)

    # community_data / developer_data are deliberately NOT requested: the
    # free public tier returns them as null (verified live), so requesting
    # them only inflates the daily-cached payload.
    payload = _cached_json(
        f"coin_{coin_id}.json",
        lambda: _http_get_json(
            f"{_BASE_URL}/coins/{coin_id}",
            params={
                "localization": "false", "tickers": "false",
                "market_data": "true", "community_data": "false",
                "developer_data": "false", "sparkline": "false",
            },
        ),
        ttl_days=1.0,
    )
    if not isinstance(payload, dict) or not payload.get("id"):
        raise NoMarketDataError(
            ticker, detail=f"CoinGecko returned an unusable payload for '{coin_id}'"
        )
    return _render_live(coin_id, base, payload)
