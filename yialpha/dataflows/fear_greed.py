"""Crypto Fear & Greed Index (alternative.me) — advisory crypto-sentiment cue.

The crypto-family runs had an outside-view sentiment gap: the market analyst
reads exchange-native positioning (long/short ratio, taker flow, funding) but
nothing that aggregates survey + volatility + momentum + dominance into one
market-wide mood gauge. This module supplies that one line from Alternative.me's
keyless free API (``https://api.alternative.me/fng/``), wired into the
conservative risk debater behind ``crypto_fear_greed`` (env
``YIALPHA_CRYPTO_FEAR_GREED``, default off = byte-equivalent) — extreme-greed
zones flag crowded longs, which is first a *risk* cue, so the risk debate is
its highest-leverage injection point (same shape as ``market_regime``).

Contract:

* **Keyless / free** — no auth; the default ``requests`` UA is accepted
  (verified live 2026-09-29, unlike CoinGecko's browser-UA requirement).
* **Foreign host → rides the SOCKS5 proxy** via ``proxy_map()`` like every
  other keyless foreign vendor.
* **PIT semantics, client-side** — the API's ``start``/``end`` params proved
  unreliable in live probing (requested an August 2026 window, got today's
  entry), so the module fetches one ``limit=1500`` series (verified live
  2026-09-29 to serve 1500 daily entries reaching back to 2022-08-21) and
  PIT-selects locally: the latest entry timestamped at or before the end of
  the analysis date. Analysis dates older than the trailing ~1500-day window
  honestly return ``None`` ("unavailable" in the debater), never a fabricated
  reading; the rendered line always discloses its own as-of date. Same-day
  visibility note: the day-D entry is published at 00:00 UTC of D computed
  from prior-day data, so treating it as visible for date D matches the
  framework's daily-grain convention (band gauge, disclosed date).
* **Disk-cached 1 day** through the shared ``cached_or_fetch`` (stale-serve
  contract inherited): the index updates ~daily, the debater can invoke it
  once per risk-debate round, and per-date views are deterministic within a
  cached day by the client-side filter (same shape as the CoinGecko
  full-window cache).
* **Advisory + fail-soft** — any fetch/parse/coverage failure returns
  ``None`` (and logs); the caller renders an explicit "unavailable" line for
  applicable runs so "configured but broken" stays distinguishable from
  "configured and calm" (the fail-visible rule the market-regime line set).

Applicability (see :func:`fear_greed_applicable`): crypto-family runs only —
``crypto_perp`` (pure, not tokenized-stock), ``crypto_spot``, and legacy
auto-detected ``crypto``. A tokenized-stock perp keeps the equity-market risk
narrative (the same rule ``resolve_market_benchmark`` applies), and stock runs
are simply out of scope for a crypto gauge — both never fetch and never show
a line.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone

from .disk_cache import MinIntervalThrottle, cached_or_fetch, vendor_cache_dir
from .errors import NoMarketDataError, VendorRateLimitError
from .utils import proxy_map

logger = logging.getLogger(__name__)

_BASE_URL = "https://api.alternative.me/fng/"
_TIMEOUT = 30
#: Politeness: one serial request per second — real traffic is one fetch per
#: cache TTL (the index updates ~daily), this only guards a cold burst.
_throttler = MinIntervalThrottle(1.0)
#: Series depth from the edge. Verified live 2026-09-29: the free endpoint
#: serves 1500 daily entries (back to 2022-08-21); deeper history was NOT
#: verified, so 1500 is the honest requested depth — older analysis dates
#: degrade rather than guess.
_LIMIT = 1500
#: One series fetch per day; per-date views derive deterministically from the
#: cached bytes via the client-side PIT filter.
_TTL_DAYS = 1.0
#: Crypto-family asset types the gauge applies to (legacy auto-detected crypto
#: included — it is a crypto run even though its OHLCV stays on Yahoo).
_CRYPTO_ASSET_TYPES = frozenset({"crypto_perp", "crypto_spot", "crypto"})


def _cache_dir() -> str:
    return vendor_cache_dir("fear_greed")


def _fetch_series_bytes() -> bytes:
    """GET the F&G series (throttle + proxy) and return the raw body.

    Status mapping follows the ``sec_edgar`` / ``coingecko_vendor`` idiom:
    429 → ``VendorRateLimitError`` (a later round may still serve the stale
    cache), every other failure → ``NoMarketDataError`` carrying the status
    and a body excerpt so the unavailable line can show the real reason.
    """
    import requests

    _throttler.wait()
    try:
        resp = requests.get(
            _BASE_URL,
            params={"limit": _LIMIT},
            proxies=proxy_map(),
            timeout=_TIMEOUT,
        )
    except Exception as exc:  # noqa: BLE001 -- network failure -> typed error
        raise NoMarketDataError(
            _BASE_URL, detail=f"Fear&Greed request failed: {exc!r}"
        ) from exc
    if resp.status_code == 429:
        raise VendorRateLimitError(f"Fear&Greed returned 429 for {_BASE_URL}")
    if resp.status_code >= 400:
        body = resp.text[:160].replace("\n", " ")
        raise NoMarketDataError(
            _BASE_URL, detail=f"Fear&Greed HTTP {resp.status_code}: {body}"
        )
    return resp.content


def _parse_series(raw: bytes) -> list[dict]:
    """Parse the payload into ``{"ts": int, "value": int, "label": str}``
    entries, newest-first as served. Malformed entries are skipped — never
    zero-filled, never fabricated. (API fields are strings; ``int(float(...))``
    coerces defensively without rejecting a future numeric body.)"""
    try:
        data = json.loads(raw).get("data") or []
    except ValueError:
        return []
    out: list[dict] = []
    for item in data:
        try:
            ts = int(float(item["timestamp"]))
            value = int(float(item["value"]))
            label = str(item["value_classification"]).strip()
        except (KeyError, TypeError, ValueError):
            continue
        if not label:
            continue
        out.append({"ts": ts, "value": value, "label": label})
    return out


def _select_pit(entries: list[dict], curr_date: str) -> tuple[dict, dict | None] | None:
    """Latest entry visible at ``curr_date`` plus the prior-day entry.

    Visible = timestamped at or before 23:59:59 UTC of the analysis date (the
    day-D entry is published at 00:00 UTC of D from prior-day data — see the
    module docstring's same-day visibility note). ``None`` when nothing in
    the series is that old (analysis date precedes index coverage).
    """
    cutoff = (
        datetime.strptime(curr_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        + timedelta(days=1)
        - timedelta(seconds=1)
    )
    cutoff_ts = int(cutoff.timestamp())
    visible = [e for e in entries if e["ts"] <= cutoff_ts]
    if not visible:
        return None
    return visible[0], (visible[1] if len(visible) > 1 else None)


def fear_greed_applicable(asset_type: str | None, ticker: str) -> bool:
    """Whether the F&G line applies to this run: crypto-family only.

    ``crypto_perp`` excludes tokenized-stock perps — their risk narrative is
    the underlying equity's market (the same rule
    ``resolve_market_benchmark`` applies for the benchmark), and a BTC-market
    mood gauge would be noise there. Stock runs and unknown types are out of
    scope; the flag never fetches for them.
    """
    if asset_type not in _CRYPTO_ASSET_TYPES:
        return False
    if asset_type == "crypto_perp":
        from .binance import stock_perp_underlying

        return not stock_perp_underlying(ticker)
    return True


def format_fear_greed_line(curr_date: str) -> str | None:
    """Render the PIT F&G reading for ``curr_date``, or ``None``.

    ``None`` covers: fetch failure (network / HTTP error with no fresh or
    stale cache), malformed body, and analysis dates older than the fetched
    series' coverage — the caller decides how to surface that; the debater
    wiring shows an explicit "unavailable" line for applicable runs.
    """
    try:
        raw = cached_or_fetch(
            _cache_dir(), "fng_series.json", _fetch_series_bytes,
            ttl_days=_TTL_DAYS, vendor="fear_greed",
        )
    except Exception as exc:  # noqa: BLE001 — advisory signal must be fail-soft
        logger.warning("fear_greed: series fetch failed: %s", exc)
        return None
    if not raw:
        return None
    try:
        selected = _select_pit(_parse_series(raw), curr_date)
    except ValueError:
        return None
    if selected is None:
        return None
    cur, prev = selected
    asof = datetime.fromtimestamp(cur["ts"], tz=timezone.utc).date().isoformat()
    line = f"alternative.me crypto Fear & Greed, as of {asof}: {cur['value']} ({cur['label']}"
    if prev is not None:
        line += f"; prior day {prev['value']} ({prev['label']})"
    return line + ")"
