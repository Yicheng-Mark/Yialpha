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

from yiagents.agents.schemas import (
    PortfolioDecision,
    PortfolioRating,
    TraderAction,
    TraderProposal,
)

from .domain import Direction, Exchange, Offset, OrderRequest, OrderType

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
