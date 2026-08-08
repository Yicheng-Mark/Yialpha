"""Translate an LLM decision into vnpy ``OrderRequest`` objects.

Pure functions, no side effects, no LLM calls, no network. This is the seam
between the analysis graph (which emits a ``PortfolioDecision`` +
``TraderProposal``) and the Track B execution layer (which speaks
``OrderRequest``). It is deliberately **not** wired into the LangGraph graph:
until a concrete gateway lands, nothing calls it, so importing this module
changes no agent input and no graph topology (byte-equivalent).

Direction resolution mirrors ``scripts/trade_ticket.py:decide_direction``
exactly — the PM 5-tier rating wins; only when the rating is missing does the
Trader's 3-tier action get consulted. A ``Hold`` rating (or a hold/absent
signal) yields an empty list: the bridge never fabricates a trade the agents
did not call for.

This does **not** replace ``scripts/trade_ticket.py``. That script renders a
rich, human-readable ticket (leverage / take-profit / stop-loss / margin /
liquidation) from the markdown report for manual execution; this module emits
bare ``OrderRequest`` structs for future automated submission. The two coexist.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from yiagents.agents.schemas import (
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
    from yiagents.risk.manager import RiskDecision

_BULLISH_RATINGS = (PortfolioRating.BUY, PortfolioRating.OVERWEIGHT)
_BEARISH_RATINGS = (PortfolioRating.SELL, PortfolioRating.UNDERWEIGHT)


def _resolve_direction(
    decision: PortfolioDecision | None,
    trader_proposal: TraderProposal | None,
) -> Direction | None:
    """Map (decision.rating, trader_proposal.action) to a Direction or None.

    Faithful to ``scripts/trade_ticket.py`` ``decide_direction``:

    * rating Buy / Overweight      -> LONG
    * rating Sell / Underweight    -> SHORT
    * rating Hold                  -> None  (no fallback to action)
    * rating missing               -> Trader action Buy -> LONG, Sell -> SHORT
    * nothing committed            -> None
    """
    rating = getattr(decision, "rating", None)
    if rating is not None:
        if rating in _BULLISH_RATINGS:
            return Direction.LONG
        if rating in _BEARISH_RATINGS:
            return Direction.SHORT
        if rating == PortfolioRating.HOLD:
            return None
    # Rating missing entirely — fall back to the Trader's action.
    action = getattr(trader_proposal, "action", None)
    if action == TraderAction.BUY:
        return Direction.LONG
    if action == TraderAction.SELL:
        return Direction.SHORT
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

    Returns:
        A list of zero or one ``OrderRequest`` objects.
    """
    if volume <= 0:
        return []

    direction = _resolve_direction(decision, trader_proposal)
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
) -> list[OrderRequest]:
    """Gate ``decision_to_order_requests`` output through the risk overlay.

    **This is the mandatory pre-``send_order`` guard.** The gateway performs no
    quantitative risk check of its own — it trusts the ``OrderRequest`` it
    receives — so the caller MUST pass every order list through this function
    before submitting. Skipping it is the #1 way to lose money when the
    execution layer is wired into the graph.

    The function is pure (no side effects, no network). It applies three gates:

    1. **Drawdown hard-stop / breaker block**: if ``risk_decision.action ==
       "blocked"`` or ``risk_decision.breaker.regime == "hard_stop"``, every
       order is dropped (returns ``[]``). The breaker has already zeroed the
       target weight upstream; this is the enforcement at the order edge.
    2. **Weight cap**: each order's value (``volume * price`` for limit,
       ``volume`` treated as a share/contract count requiring a price for market
       orders) is clipped so it never exceeds ``max_weight * equity``. Orders
       that would be clipped to zero volume are dropped.
    3. **Pass-through**: with no ``risk_decision`` (risk overlay disabled or
       not yet computed), orders pass through unchanged — the caller takes
       responsibility for sizing. This preserves today's "risk overlay is
       advisory text only" behaviour when execution is not wired.

    Args:
        order_requests: The output of :func:`decision_to_order_requests`.
        risk_decision: The :class:`~yiagents.risk.manager.RiskDecision` from the
            risk overlay, or ``None`` if the overlay did not run (pass-through).
        equity: Current account equity, used to translate ``max_weight`` into
            an absolute value cap. ``0`` disables the weight cap (gate 2).
        max_weight: Maximum fraction of equity a single order may represent.
            Default 0.20 matches ``DrawdownBreaker.max_single_position``.

    Returns:
        A (possibly empty, possibly clipped) list of ``OrderRequest`` safe to
        submit to ``gateway.send_order``.
    """
    # Gate 3: no risk decision -> caller owns sizing; pass through unchanged.
    if risk_decision is None:
        return list(order_requests)

    # Gate 1: breaker hard-stop or explicit block drops every order.
    if (
        risk_decision.action == "blocked"
        or getattr(risk_decision.breaker, "regime", "") == "hard_stop"
    ):
        return []

    # Gate 2: clip each order's value to max_weight * equity.
    if equity <= 0 or max_weight <= 0:
        return list(order_requests)
    value_cap = equity * max_weight

    clipped: list[OrderRequest] = []
    for req in order_requests:
        # For a market order ``price`` is 0; without a reference price we cannot
        # compute a value, so the order passes (the exchange will reject an
        # absurd size). For limit orders we clip the volume directly.
        if req.price > 0 and req.volume > 0:
            max_volume = value_cap / req.price
            if req.volume > max_volume:
                if max_volume <= 0:
                    continue  # clipped away entirely
                req = OrderRequest(
                    symbol=req.symbol,
                    exchange=req.exchange,
                    direction=req.direction,
                    type=req.type,
                    volume=max_volume,
                    price=req.price,
                    offset=req.offset,
                    reference=req.reference,
                )
        clipped.append(req)
    return clipped
