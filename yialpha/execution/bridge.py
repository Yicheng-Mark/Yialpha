"""Translate an LLM decision into vnpy ``OrderRequest`` objects.

Pure functions, no side effects, no LLM calls, no network. This is the seam
between the analysis graph (which emits a ``PortfolioDecision`` +
``TraderProposal``) and the Track B execution layer (which speaks
``OrderRequest``). It is deliberately **not** wired into the LangGraph graph:
until a concrete gateway lands, nothing calls it, so importing this module
changes no agent input and no graph topology (byte-equivalent).

Direction resolution keeps the PM-rating priority used by
``scripts/trade_ticket.py:decide_direction``, but the executable bridge adds a
fail-closed short-sale policy: a bearish signal defaults to no order, an
explicit close may reduce a long, and opening a new short requires a separate
flag plus ``Offset.OPEN``. A ``Hold`` rating yields an empty list.

This does **not** replace ``scripts/trade_ticket.py``. That script renders a
rich, human-readable ticket (leverage / take-profit / stop-loss / margin /
liquidation) from the markdown report for manual execution; this module emits
bare ``OrderRequest`` structs for future automated submission. The two coexist.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from yialpha.agents.schemas import (
    PortfolioDecision,
    PortfolioRating,
    TraderAction,
    TraderProposal,
)

from .domain import Direction, Exchange, Offset, OrderRequest, OrderType

if TYPE_CHECKING:
    # Avoid a runtime import cycle: risk/manager.py does not import execution/,
    # but keeping the reference under TYPE_CHECKING means this module never
    # pulls the risk layer at runtime — bridge stays pure-struct + pure-function.
    from yialpha.risk.manager import RiskDecision

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RiskAudit:
    """Audit trail for :func:`pre_trade_risk_check`.

    Records which fail-closed gate accepted, clipped, or rejected orders.
    Missing quantitative context never releases opening exposure; an explicit
    close can remain available as a risk-reducing emergency action.

    Attributes:
        risk_checked: ``True`` when a real ``RiskDecision`` was applied.
            ``False`` when it was missing; in that case only explicit close
            offsets can survive.
        gate: Human-readable label of the gate that decided the outcome
            (for example ``"missing_risk_decision"``,
            ``"breaker_close_only"``, ``"invalid_risk_context"``, or
            ``"weight_cap"``).
        n_in / n_out: Order count before and after the check.
    """

    risk_checked: bool
    gate: str
    n_in: int
    n_out: int

_BULLISH_RATINGS = (PortfolioRating.BUY, PortfolioRating.OVERWEIGHT)
_BEARISH_RATINGS = (PortfolioRating.SELL, PortfolioRating.UNDERWEIGHT)
_CLOSE_OFFSETS = frozenset(
    {Offset.CLOSE, Offset.CLOSETODAY, Offset.CLOSEYESTERDAY}
)


def _resolve_direction(
    decision: PortfolioDecision | None,
    trader_proposal: TraderProposal | None,
    *,
    offset: Offset = Offset.NONE,
    allow_short_open: bool = False,
) -> Direction | None:
    """Map (decision.rating, trader_proposal.action) to a Direction or None.

    Uses the manual ticket's rating priority with stricter execution semantics:

    * rating Buy / Overweight      -> LONG
    * rating Sell / Underweight    -> SHORT only for an explicit close, or for
      ``Offset.OPEN`` plus ``allow_short_open=True``
    * rating Hold                  -> None  (no fallback to action)
    * rating missing               -> Trader Buy -> LONG; Trader Sell follows
      the same close/explicit-short policy
    * nothing committed            -> None
    """
    rating = getattr(decision, "rating", None)
    if rating is not None:
        if rating in _BULLISH_RATINGS:
            return Direction.LONG
        if rating in _BEARISH_RATINGS:
            if offset in _CLOSE_OFFSETS:
                return Direction.SHORT
            if offset == Offset.OPEN and allow_short_open is True:
                return Direction.SHORT
            return None
        if rating == PortfolioRating.HOLD:
            return None
    # Rating missing entirely — fall back to the Trader's action.
    action = getattr(trader_proposal, "action", None)
    if action == TraderAction.BUY:
        return Direction.LONG
    if action == TraderAction.SELL:
        if offset in _CLOSE_OFFSETS:
            return Direction.SHORT
        if offset == Offset.OPEN and allow_short_open is True:
            return Direction.SHORT
        return None
    return None


def decision_to_order_requests(
    decision: PortfolioDecision | None,
    trader_proposal: TraderProposal | None,
    *,
    symbol: str,
    exchange: Exchange,
    volume: float,
    order_type: OrderType = OrderType.MARKET,
    price: float = 0.0,
    offset: Offset = Offset.NONE,
    reference: str = "",
    allow_short_open: bool = False,
) -> list[OrderRequest]:
    """Turn a PM decision + Trader proposal into a list of ``OrderRequest``.

    Returns an empty list for Hold / no-direction / non-positive volume — the
    caller treats "no requests" as "no trade". Otherwise returns a single
    market (or configured-type) order in the resolved direction.

    Args:
        decision: Portfolio Manager structured decision (rating-driven), or
            ``None`` if only the Trader proposal is available.
        trader_proposal: Trader structured proposal (action fallback), or None.
        symbol: Resolved ticker the gateway understands.
        exchange: Exchange enum value for ``symbol``.
        volume: Order size in the instrument's units; must be positive.
        order_type: Defaults to market (immediate fill assumption).
        price: Limit price (only meaningful with ``OrderType.LIMIT``).
        offset: Futures open/close flag (``Offset.NONE`` for spot / equity).
        reference: Free-text tag propagated onto the resulting ``OrderData``.
        allow_short_open: Defaults to ``False``. A bearish signal may open a
            new short only when this is the literal ``True`` *and* ``offset``
            is explicitly ``Offset.OPEN``. An explicit close remains allowed.

    Returns:
        A list of zero or one ``OrderRequest`` objects.
    """
    if volume <= 0:
        return []

    direction = _resolve_direction(
        decision,
        trader_proposal,
        offset=offset,
        allow_short_open=allow_short_open,
    )
    if direction is None:
        return []

    return [
        OrderRequest(
            symbol=symbol,
            exchange=exchange,
            direction=direction,
            type=order_type,
            volume=volume,
            price=price,
            offset=offset,
            reference=reference,
        )
    ]


def pre_trade_risk_check(
    order_requests: list[OrderRequest],
    risk_decision: RiskDecision | None,
    *,
    equity: float = 0.0,
    max_weight: float = 0.20,
    reference_price: float | None = None,
    reference_prices: Mapping[str, float] | None = None,
) -> list[OrderRequest]:
    """Gate ``decision_to_order_requests`` output through the risk overlay.

    **This is the mandatory pre-``send_order`` guard.** ``BinanceGateway`` also
    invokes it at the final network edge so a caller cannot accidentally bypass
    quantitative risk checks; other gateways should do the same.

    The function is pure (no side effects, no network). It applies these gates:

    1. **Drawdown hard-stop / breaker block**: new exposure is dropped while
       explicit close offsets remain available for risk reduction.
    2. **Weight cap**: each order's value (``volume * price`` for limit,
       ``volume`` treated as a share/contract count requiring a price for market
       orders) is clipped so it never exceeds ``max_weight * equity``. Orders
       that would be clipped to zero volume are dropped.
    3. **Required context**: no ``risk_decision`` or no valid equity/reference
       price rejects opening orders. Market orders use ``reference_price`` or
       the per-symbol ``reference_prices`` mapping for value sizing.

    Args:
        order_requests: The output of :func:`decision_to_order_requests`.
        risk_decision: The :class:`~yialpha.risk.manager.RiskDecision` from the
            risk overlay, or ``None`` if the overlay did not run (opens reject).
        equity: Current account equity, used to translate ``max_weight`` into
            an absolute value cap. A non-positive/missing value rejects opens.
        max_weight: Maximum fraction of equity a single order may represent.
            Default 0.20 matches ``DrawdownBreaker.max_single_position``.

    Returns:
        A (possibly empty, possibly clipped) list of ``OrderRequest`` safe to
        submit to ``gateway.send_order``.
    """
    orders, audit = pre_trade_risk_check_with_audit(
        order_requests,
        risk_decision,
        equity=equity,
        max_weight=max_weight,
        reference_price=reference_price,
        reference_prices=reference_prices,
    )
    return orders


def pre_trade_risk_check_with_audit(
    order_requests: list[OrderRequest],
    risk_decision: RiskDecision | None,
    *,
    equity: float = 0.0,
    max_weight: float = 0.20,
    reference_price: float | None = None,
    reference_prices: Mapping[str, float] | None = None,
) -> tuple[list[OrderRequest], RiskAudit]:
    """Same gates as :func:`pre_trade_risk_check`, but returns an audit trail.

    The :class:`RiskAudit` makes the exact rejection/close-only/cap gate
    observable. This is the recommended entry point for any submission path.

    Returns:
        A ``(orders, audit)`` tuple where ``orders`` is the (possibly empty,
        possibly clipped) list and ``audit`` records which gate fired.
    """
    n_in = len(order_requests)
    if n_in == 0:
        return [], RiskAudit(
            risk_checked=risk_decision is not None,
            gate="empty",
            n_in=0,
            n_out=0,
        )

    def _is_close(req: OrderRequest) -> bool:
        return req.offset in _CLOSE_OFFSETS

    def _positive_finite(value: Any) -> float | None:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        return number if number > 0.0 and math.isfinite(number) else None

    def _valid_close(req: OrderRequest) -> bool:
        # A close is risk-reducing only when it has a concrete positive size.
        # BinanceGateway additionally maps it to reduce_only=true.
        return _positive_finite(req.volume) is not None

    close_orders = [
        req for req in order_requests if _is_close(req) and _valid_close(req)
    ]

    # Missing risk context is never permission to open/increase exposure.  A
    # structurally explicit close remains available as an emergency risk exit.
    if risk_decision is None:
        logger.warning(
            "pre_trade_risk_check: rejected %d unchecked opening order(s); "
            "only explicit CLOSE orders may pass without risk_decision.",
            n_in - len(close_orders),
        )
        return close_orders, RiskAudit(
            risk_checked=False,
            gate="missing_risk_decision",
            n_in=n_in,
            n_out=len(close_orders),
        )

    breaker = getattr(risk_decision, "breaker", None)
    action = getattr(risk_decision, "action", None)
    breaker_blocks = (
        action not in {"enter", "add"}
        or breaker is None
        or getattr(breaker, "regime", "") == "hard_stop"
        or getattr(breaker, "can_open_new", False) is not True
        or getattr(risk_decision, "exposure_ok", False) is not True
    )
    if breaker_blocks:
        # A hard stop must flatten risk, not delete the reduce-only orders that
        # perform the flattening. New/open exposure is removed.
        return close_orders, RiskAudit(
            risk_checked=True,
            gate="breaker_close_only",
            n_in=n_in,
            n_out=len(close_orders),
        )

    equity_value = _positive_finite(equity)
    weight_value = _positive_finite(max_weight)
    if equity_value is None or weight_value is None or weight_value > 1.0:
        logger.warning(
            "pre_trade_risk_check: missing/invalid equity or max_weight; "
            "rejecting all opening orders (CLOSE remains available)."
        )
        return close_orders, RiskAudit(
            risk_checked=True,
            gate="invalid_risk_context",
            n_in=n_in,
            n_out=len(close_orders),
        )

    value_cap = equity_value * weight_value

    def _valuation_price(req: OrderRequest) -> float | None:
        order_price = _positive_finite(req.price)
        if req.type != OrderType.MARKET:
            return order_price
        if order_price is not None:
            return order_price
        if reference_prices is not None:
            try:
                mapped_value = reference_prices.get(req.symbol)
            except (AttributeError, TypeError):
                mapped_value = None
            mapped = _positive_finite(mapped_value)
            if mapped is not None:
                return mapped
        return _positive_finite(reference_price)

    checked: list[OrderRequest] = []
    for req in order_requests:
        if _is_close(req):
            if _valid_close(req):
                checked.append(req)
            continue

        volume = _positive_finite(req.volume)
        valuation_price = _valuation_price(req)
        if volume is None or valuation_price is None:
            logger.warning(
                "pre_trade_risk_check: rejecting %s %s; positive volume and "
                "limit/reference price are required for sizing.",
                req.symbol,
                req.type.value,
            )
            continue

        max_volume = value_cap / valuation_price
        if not (max_volume > 0.0 and math.isfinite(max_volume)):
            continue
        safe_volume = min(volume, max_volume)
        if safe_volume != volume:
            req = OrderRequest(
                symbol=req.symbol,
                exchange=req.exchange,
                direction=req.direction,
                type=req.type,
                volume=safe_volume,
                price=req.price,
                offset=req.offset,
                reference=req.reference,
            )
        checked.append(req)

    return checked, RiskAudit(
        risk_checked=True,
        gate="weight_cap",
        n_in=n_in,
        n_out=len(checked),
    )
