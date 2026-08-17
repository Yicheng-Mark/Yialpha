"""Polymarket prediction-market vendor.

Surfaces live, market-implied probabilities for forward-looking events (Fed
decisions, recession, elections, geopolitics, crypto) to the news analyst, as a
complement to news (what happened) and FRED macro data (where things stand):
what the crowd actually prices to happen next.

Uses Polymarket's public Gamma API (https://gamma-api.polymarket.com) — no key,
no auth. Each market's ``outcomePrices`` are the implied probabilities of its
outcomes (a "Yes" at 0.76 means the market prices a 76% chance).
"""
import hashlib
import json
import logging
from datetime import UTC, datetime

import requests

from .disk_cache import cached_or_fetch, safe_cache_component, vendor_cache_dir

logger = logging.getLogger(__name__)

GAMMA_BASE = "https://gamma-api.polymarket.com"

# Network timeout (seconds), consistent with the other vendors.
REQUEST_TIMEOUT = 30

# Default number of markets to return, ranked by traded volume.
DEFAULT_LIMIT = 6

# Odds move continuously but not fast enough to matter for an analyst prompt:
# a 10-minute disk TTL collapses burst re-asks (batch runs, retries) to one
# network hit while probabilities stay effectively live.
_CACHE_TTL_DAYS = 10.0 / 1440.0


def _fetch_raw(path: str, params: dict) -> bytes:
    response = requests.get(
        f"{GAMMA_BASE}/{path}", params=params, timeout=REQUEST_TIMEOUT
    )
    response.raise_for_status()
    return response.content


def _request(path: str, params: dict) -> dict:
    """One Gamma API call through the shared on-disk cache.

    Identical (path, params) requests within the TTL are served from disk;
    on a fetch failure a recent cache is served stale (with the data-quality
    sentinel recorded by the cache helper), and only a cache-less failure
    reaches the caller as ``requests.RequestException``. A corrupt cached
    payload raises ``json.JSONDecodeError`` rather than degrading silently.
    """
    key_blob = json.dumps({"path": path, "params": params}, sort_keys=True)
    digest = hashlib.sha256(key_blob.encode("utf-8")).hexdigest()[:12]
    filename = f"search_{safe_cache_component(path)}_{digest}.json"

    def _fetch() -> bytes:
        return _fetch_raw(path, params)

    raw = cached_or_fetch(
        vendor_cache_dir("polymarket"), filename, _fetch,
        ttl_days=_CACHE_TTL_DAYS, vendor="polymarket",
    )
    assert raw is not None  # fail_open never set: fetch errors re-raise
    return json.loads(raw)


def _parse_json_list(value) -> list:
    """Gamma encodes ``outcomes``/``outcomePrices`` as JSON-string arrays."""
    if isinstance(value, list):
        return value
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return []


def _is_forward_looking(market: dict, now: datetime) -> bool:
    """Keep only open markets that resolve in the future.

    ``closed`` is the reliable resolved flag (``active`` stays True even for
    settled markets), and a past ``endDate`` means the event already resolved —
    either way it is not a forward-looking signal.
    """
    if market.get("closed"):
        return False
    end_date = market.get("endDate")
    if end_date:
        try:
            if datetime.fromisoformat(end_date.replace("Z", "+00:00")) < now:
                return False
        except ValueError:
            # Kept as forward-looking (docstring contract), but observable: a
            # malformed endDate may hide an already-settled market.
            logger.debug("polymarket: unparseable endDate %r treated as forward-looking",
                         end_date)
    return bool(_parse_json_list(market.get("outcomePrices"))) and bool(
        _parse_json_list(market.get("outcomes"))
    )


def get_prediction_markets(topic: str, limit: int | None = None) -> str:
    """Return live prediction-market probabilities for an event topic.

    Args:
        topic: Event keyword(s), e.g. "Fed rate cut", "recession 2026",
            "US election", or a sector/company event.
        limit: Max markets to return (ranked by traded volume); ``None`` uses
            DEFAULT_LIMIT.

    Returns:
        A markdown report of the most-traded open markets matching the topic,
        each with its implied probability, traded volume, resolution date, and
        recent (1-week) move.

    Raises:
        requests.RequestException: The Gamma API (or the fresh-cache fetch)
            failed and no cache existed to serve. Transport errors must reach
            the router so the optional ``prediction_markets`` category records
            its KIND_OPTIONAL_UNAVAILABLE sentinel and returns the
            DATA_UNAVAILABLE message — swallowing them here returned a prose
            "unavailable" string as if it were data, dropping the run's
            data-quality evidence for this degradation. (A recent cache is
            still served stale by ``cached_or_fetch`` with a stale-cache
            sentinel before this ever propagates.)
    """
    if limit is None:
        limit = DEFAULT_LIMIT

    data = _request("public-search", {"q": topic, "limit_per_type": 20})

    now = datetime.now(UTC)
    candidates = [
        m
        for event in data.get("events", [])
        for m in event.get("markets", [])
        if _is_forward_looking(m, now)
    ]
    candidates.sort(key=lambda m: m.get("volumeNum") or 0, reverse=True)

    header = (
        f'## Polymarket prediction markets: "{topic}"\n'
        f"Live, market-implied probabilities (higher traded volume = deeper, "
        f"more reliable). A probability is the crowd's priced odds of the event, "
        f"not a forecast you should take as certain.\n\n"
    )

    if not candidates:
        return header + (
            f"No open prediction markets matched '{topic}'. Polymarket coverage "
            f"is concentrated in macro, political, geopolitical, and crypto "
            f"events; a specific equity may have none."
        )

    lines = []
    for m in candidates[:limit]:
        prices = _parse_json_list(m.get("outcomePrices"))
        outcomes = _parse_json_list(m.get("outcomes"))
        try:
            prob = float(prices[0])
        except (ValueError, IndexError):
            continue
        label = outcomes[0] if outcomes else "Yes"
        volume = m.get("volumeNum") or 0
        end_date = (m.get("endDate") or "")[:10]
        wk = m.get("oneWeekPriceChange")
        wk_str = (
            f", 1-week {wk * 100:+.1f}pp"
            if isinstance(wk, (int, float)) and wk
            else ""
        )
        lines.append(
            f"- **{m.get('question')}** — {label} {prob:.0%} "
            f"(${volume:,.0f} volume, resolves {end_date}{wk_str})"
        )

    return header + "\n".join(lines) + "\n"
