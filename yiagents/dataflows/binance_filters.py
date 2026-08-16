"""Binance exchangeInfo filters: symbol precision rules and order quantization.

Binance rejects any order whose price is not a multiple of the symbol's
``tickSize`` (PRICE_FILTER), whose quantity is not a multiple of ``stepSize``
(LOT_SIZE), or whose notional is under ``MIN_NOTIONAL``/``NOTIONAL`` — with
``-1111 Precision is over the maximum`` / ``-4014`` errors that the execution
gateway would otherwise surface as a hard reject on EVERY LLM-sized order
(weights land on arbitrary floats like 0.30718...).

This module fetches a symbol's filter block from ``exchangeInfo`` (public,
weight 1 on fapi / 20 on spot, cacheable — the block is static for days) and
provides Decimal-exact quantization:

  - :func:`get_symbol_filters` — TTL-cached ``SymbolFilters`` lookup
  - :func:`quantize_order` — floor quantity to stepSize, round price to
    tickSize, and report MIN_NOTIONAL compliance

Consumers: the Track B execution gateway (pre-submit quantization) and any
Track A display that wants exchange-native price precision. Track A fetches
reuse :func:`yiagents.dataflows.binance._http_get` (SOCKS5 proxy, weight
limiter, typed errors), so no new transport is introduced.
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any

from .binance import _http_get, _spot_host
from .errors import NoMarketDataError
from .symbol_utils import normalize_symbol_for_venue

logger = logging.getLogger(__name__)

# exchangeInfo filter blocks are static for days at a time (Binance changes
# them in maintenance windows). A 10-minute process TTL bounds staleness
# while collapsing a batch run's per-order fetches to ~zero.
_FILTERS_TTL_S = 600.0

_filters_cache: dict[tuple[str, str], tuple[float, SymbolFilters]] = {}
_filters_lock = threading.Lock()


@dataclass(frozen=True)
class SymbolFilters:
    """The order-shaping rules Binance enforces for one symbol.

    All Decimal fields come straight from the exchange's string values —
    floats must never be used for quantization math (0.1 is not representable
    and would produce off-step quantities Binance rejects).
    """

    symbol: str
    status: str
    tick_size: Decimal          # PRICE_FILTER: price multiple
    step_size: Decimal          # LOT_SIZE: quantity multiple
    min_qty: Decimal            # LOT_SIZE lower bound
    max_qty: Decimal            # LOT_SIZE upper bound
    min_notional: Decimal       # MIN_NOTIONAL / NOTIONAL: price x qty floor
    price_precision: int        # exchange-declared price decimals
    quantity_precision: int     # exchange-declared quantity decimals


def _dec(value: Any, default: str = "0") -> Decimal:
    """Parse an exchange string into Decimal, tolerating None/bad values."""
    try:
        return Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return Decimal(default)


def _extract_filters(info: dict, symbol: str) -> SymbolFilters:
    """Build a :class:`SymbolFilters` from the symbol's exchangeInfo entry."""
    filters = {f.get("filterType"): f for f in info.get("filters", [])}
    price_f = filters.get("PRICE_FILTER", {})
    lot_f = filters.get("LOT_SIZE", {})
    notional_f = filters.get("NOTIONAL") or filters.get("MIN_NOTIONAL") or {}
    return SymbolFilters(
        symbol=info.get("symbol", symbol),
        status=info.get("status", ""),
        tick_size=_dec(price_f.get("tickSize"), "1e-8"),
        step_size=_dec(lot_f.get("stepSize"), "1e-8"),
        min_qty=_dec(lot_f.get("minQty"), "0"),
        max_qty=_dec(lot_f.get("maxQty"), "0"),
        min_notional=_dec(notional_f.get("notional") or notional_f.get("minNotional"), "0"),
        price_precision=int(info.get("pricePrecision", 8)),
        quantity_precision=int(info.get("quantityPrecision", 8)),
    )


def get_symbol_filters(
    symbol: str,
    venue: str = "binance_perp",
    *,
    force_refresh: bool = False,
) -> SymbolFilters:
    """Return the TTL-cached exchangeInfo filter block for ``symbol``.

    Fetches ``/fapi/v1/exchangeInfo?symbol=`` (perp) or ``/api/v3/exchangeInfo?
    symbol=`` (spot) through the shared Track A transport on a cache miss.
    A symbol missing from exchangeInfo (delisted, typo) raises
    :class:`NoMarketDataError` so callers degrade rather than order blindly.
    """
    canonical = normalize_symbol_for_venue(symbol, venue)
    key = (venue, canonical)
    now = time.monotonic()
    with _filters_lock:
        hit = _filters_cache.get(key)
        if hit is not None and not force_refresh and now - hit[0] < _FILTERS_TTL_S:
            return hit[1]

    if venue == "binance_spot":
        data = _http_get(
            "/api/v3/exchangeInfo", {"symbol": canonical},
            symbol, canonical, base=_spot_host(), weight_key="spot",
        )
    else:
        data = _http_get(
            "/fapi/v1/exchangeInfo", {"symbol": canonical},
            symbol, canonical,
        )

    symbols = data.get("symbols") if isinstance(data, dict) else None
    if not isinstance(symbols, list) or not symbols:
        raise NoMarketDataError(
            symbol, canonical,
            f"exchangeInfo has no entry for {canonical} on {venue} "
            "(delisted or unknown symbol)",
        )
    filters = _extract_filters(symbols[0], canonical)
    with _filters_lock:
        _filters_cache[key] = (time.monotonic(), filters)
    return filters


def quantize_order(
    price: float | None,
    quantity: float,
    filters: SymbolFilters,
    ref_price: float | None = None,
) -> dict[str, float | bool | None]:
    """Shape an order to Binance's precision rules; report compliance.

    Quantity floors to a multiple of ``stepSize`` (floor, never round up —
    rounding up could exceed the intended exposure and the margin behind it).
    Price, when supplied (limit orders), rounds HALF_EVEN to ``tickSize`` —
    the neutral choice, since rounding direction changes fill priority, not
    validity. Returns a dict with the quantized ``price``/``quantity`` (floats
    ready for the SDK), plus compliance flags::

        {"price": ..., "quantity": ..., "below_min_qty": bool,
         "below_min_notional": bool, "over_max_qty": bool}

    Callers are expected to refuse the order when any flag is true (the LLM's
    sizing was outside the symbol's rules) rather than silently clamping.

    ``ref_price``: an indicative price for MIN_NOTIONAL pre-validation of
    MARKET orders, which carry no ``price``. Binance rejects a market order
    whose notional is under the floor AT SUBMIT (-4014), not at fill — the
    old "the exchange checks it at fill" assumption was wrong — so live
    callers should pass their best reference (mark price / latest close).
    Without any price the check stays undecided (flag False) and the reject
    is left to the exchange, as before.
    """
    step = filters.step_size if filters.step_size > 0 else Decimal(1)
    tick = filters.tick_size if filters.tick_size > 0 else Decimal(1)
    qty_dec = Decimal(str(quantity))
    quantized_qty = (qty_dec // step) * step

    price_out: float | None = None
    if price is not None:
        price_dec = Decimal(str(price))
        # HALF_EVEN to the tick multiple (Decimal.quantize default context);
        # direction changes fill priority, not validity.
        price_out = float((price_dec / tick).quantize(Decimal("1")) * tick)

    # Notional floor check, Decimal-exact so an order landing exactly on the
    # minimum is not float-shaved into a false violation.
    below_min_notional = False
    notional_ref = price if price is not None else ref_price
    if notional_ref is not None and filters.min_notional > 0:
        ref_dec = (
            Decimal(str(price_out)) if price is not None else Decimal(str(notional_ref))
        )
        below_min_notional = ref_dec * quantized_qty < filters.min_notional

    return {
        "price": price_out,
        "quantity": float(quantized_qty),
        "below_min_qty": quantized_qty < filters.min_qty,
        "below_min_notional": below_min_notional,
        "over_max_qty": filters.max_qty > 0 and quantized_qty > filters.max_qty,
    }


def reset_for_test() -> None:
    """Drop the filter cache (tests only — fresh process state per case)."""
    with _filters_lock:
        _filters_cache.clear()
