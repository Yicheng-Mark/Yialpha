"""Binance USDT-M leverage brackets: maintenance-margin tiers for liquidation.

The perp backtest engine needs each symbol's maintenance-margin-ratio (MMR)
ladder to model isolated-margin liquidation honestly. The authoritative
source is ``GET /fapi/v1/leverageBracket`` — a SIGNED endpoint, so it is only
reachable when the operator configured ``BINANCE_API_KEY``/``BINANCE_API_SECRET``
(the same env pair the execution gateway reads). Without keys this module
raises :class:`NoMarketDataError` and the caller falls back to
:data:`DEFAULT_USDT_M_BRACKETS`, a documented approximation of Binance's
standard ladder (per-symbol ladders differ; the engine's ``config_summary``
always states which source was used).

Transport reuses :func:`yialpha.dataflows.binance._http_get` (SOCKS5 proxy,
weight limiter, typed errors) plus a minimal HMAC-SHA256 signature — no new
network stack. Results are TTL-cached like the exchangeInfo filters: the
ladder is static for days.
"""
from __future__ import annotations

import hashlib
import hmac
import logging
import os
import threading
import time
import urllib.parse
from dataclasses import dataclass
from typing import Any

from .binance import _http_get
from .errors import NoMarketDataError
from .symbol_utils import normalize_symbol_for_venue

logger = logging.getLogger(__name__)

_BRACKETS_TTL_S = 600.0

_brackets_cache: dict[str, tuple[float, list[Bracket]]] = {}
_brackets_lock = threading.Lock()


@dataclass(frozen=True)
class Bracket:
    """One maintenance-margin tier: MMR applies from ``notional_floor`` up."""

    notional_floor: float
    mmr: float


# Binance's standard USDT-M ladder for the common (<=125x max-leverage) tier
# family, in (notional floor USDT, MMR). Per-symbol ladders differ (BTC has
# its own); this approximation is only used when the authenticated
# leverageBracket endpoint is unreachable, and the backtest config_summary
# says so ("mmr_source: default-ladder").
DEFAULT_USDT_M_BRACKETS: tuple[Bracket, ...] = tuple(
    Bracket(floor, mmr) for floor, mmr in (
        (0.0, 0.004),
        (50_000.0, 0.005),
        (150_000.0, 0.006),
        (250_000.0, 0.007),
        (500_000.0, 0.0075),
        (750_000.0, 0.008),
        (1_000_000.0, 0.0125),
        (5_000_000.0, 0.05),
        (20_000_000.0, 0.1),
        (50_000_000.0, 0.125),
        (100_000_000.0, 0.5),
    )
)


def mmr_for_notional(brackets, notional: float) -> float:
    """The maintenance-margin ratio applicable to ``notional`` (USDT).

    ``brackets`` is any ascending-by-floor iterable of :class:`Bracket` (or
    ``(floor, mmr)`` pairs). Notional below the first floor gets the first
    tier; above the last, the last tier.
    """
    mmr = 0.0
    for item in brackets:
        floor, ratio = (item.notional_floor, item.mmr) if isinstance(item, Bracket) else item
        if notional >= floor:
            mmr = ratio
    return mmr


def signed_fapi_get(
    path: str, params: dict[str, Any], symbol: str, canonical: str,
) -> Any:
    """Signed USER-DATA GET on the /fapi futures API (needs operator keys).

    Adds ``timestamp``/``recvWindow``, HMAC-SHA256-signs the query and sends
    the ``X-MBX-APIKEY`` header — the same minimal signing the execution
    gateway contract uses. Raises :class:`NoMarketDataError` with the
    actionable reason when ``BINANCE_API_KEY``/``BINANCE_API_SECRET`` are
    not configured, so callers (leverage brackets, perp-bundle ADL) degrade
    instead of firing a doomed unsigned request at a signed endpoint.
    """
    api_key = os.environ.get("BINANCE_API_KEY", "")
    api_secret = os.environ.get("BINANCE_API_SECRET", "")
    if not (api_key and api_secret):
        raise NoMarketDataError(
            symbol, canonical,
            f"{path} is a signed endpoint and no BINANCE_API_KEY/"
            "BINANCE_API_SECRET is configured",
        )
    full = {
        **params,
        "timestamp": int(time.time() * 1000),
        "recvWindow": "5000",
    }
    query = urllib.parse.urlencode(full)
    signature = hmac.new(
        api_secret.encode(), query.encode(), hashlib.sha256
    ).hexdigest()
    return _http_get(
        path, {**full, "signature": signature},
        symbol, canonical, headers={"X-MBX-APIKEY": api_key},
    )


def _signed_leverage_brackets(symbol: str, canonical: str) -> list[Bracket]:
    """Signed ``/fapi/v1/leverageBracket`` fetch (needs operator API keys)."""
    data = signed_fapi_get(
        "/fapi/v1/leverageBracket", {"symbol": canonical}, symbol, canonical,
    )
    # Response: [{"symbol": ..., "brackets": [{bracket, initialLeverage,
    # notionalFloor, maintMarginRatio, cum}, ...]}]
    entry = data[0] if isinstance(data, list) and data else {}
    raw = entry.get("brackets") if isinstance(entry, dict) else None
    out: list[Bracket] = []
    for b in raw if isinstance(raw, list) else []:
        try:
            out.append(Bracket(float(b["notionalFloor"]), float(b["maintMarginRatio"])))
        except (KeyError, TypeError, ValueError):
            continue
    if not out:
        raise NoMarketDataError(
            symbol, canonical, "leverageBracket response carried no parseable brackets"
        )
    out.sort(key=lambda br: br.notional_floor)
    return out


def get_leverage_brackets(symbol: str, *, force_refresh: bool = False) -> list[Bracket]:
    """TTL-cached MMR ladder for a USDT-M perp ``symbol``.

    Raises :class:`NoMarketDataError` when the signed endpoint is unreachable
    (no keys configured, transport failure) — callers decide whether to fall
    back to :data:`DEFAULT_USDT_M_BRACKETS` and must say so in their output.
    """
    canonical = normalize_symbol_for_venue(symbol, "binance_perp")
    now = time.monotonic()
    with _brackets_lock:
        hit = _brackets_cache.get(canonical)
        if hit is not None and not force_refresh and now - hit[0] < _BRACKETS_TTL_S:
            return hit[1]
    brackets = _signed_leverage_brackets(symbol, canonical)
    with _brackets_lock:
        _brackets_cache[canonical] = (time.monotonic(), brackets)
    return brackets


def default_brackets() -> list[Bracket]:
    """The approximation ladder as a mutable list (engine-consumable)."""
    return list(DEFAULT_USDT_M_BRACKETS)


def reset_for_test() -> None:
    """Drop the bracket cache (tests only — fresh process state per case)."""
    with _brackets_lock:
        _brackets_cache.clear()
