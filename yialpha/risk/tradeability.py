"""Directional edge + the NO_TRADE gate (V2.0 P0.2 / P0.3).

The tradeability layer answers ONE question deterministically: is this trade
worth executing at all? It never decides direction (that is the PM's opinion)
and never decides size (that is the risk overlay's constraint) — it prices
the PM's own target against the reference price and refuses trades whose net
edge does not clear costs.

The edge is DIRECTIONAL, never absolute: a BUY whose target sits BELOW the
reference price is a negative-edge mistake, not a "10% edge" (the abs()
formulation of the freeze review's first contract point). With ``side``:

    LONG :  gross_edge = target / reference - 1
    SHORT:  gross_edge = 1 - target / reference

``net_edge = gross_edge - round_trip_cost - risk_buffer`` and the gate is
``net_edge <= 0 -> NO_TRADE`` — equality means zero expected value, which has
no trade value either.

Data quality participates through the tier produced by
:func:`yialpha.dataflows.quality.classify_quality`: only a CRITICAL lack
(price/ATR/mark-index/core fundamentals) or an INVALID run vetoes here;
auxiliary degradation (Reddit down, Binance Square down) penalizes confidence
and is disclosed, but must not kill an otherwise healthy BTC perp trade.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel

from yialpha.dataflows.quality import (
    TIER_DEGRADED_CRITICAL,
    TIER_GOOD,
    TIER_INVALID,
)
from yialpha.risk.cost_model import CostEstimate

#: Conservative friction buffer beyond the explicitly priced legs. Absorbs
#: model error in the cost estimate itself (gap risk, partial fills) so the
#: edge bar sits a little above break-even rather than exactly on it.
RISK_BUFFER_BPS = 10.0

#: Sentinel reasons the verdict can carry (``tradeability_reason`` on the
#: ExecutionTicket). Stable strings — dashboards grep them.
REASON_HOLD = "hold_rating_no_directional_trade"
REASON_DATA_INVALID = "data_invalid_run_refused"
REASON_CRITICAL_MISSING = "critical_data_missing"
REASON_NO_TARGET = "no_target_price"
REASON_NO_REFERENCE = "no_reference_price"
REASON_EDGE = "net_edge_at_or_below_zero"
REASON_OK = "net_edge_above_costs"


class Tradeability(StrEnum):
    """Machine-readable tradeability of a candidate trade.

    ``UNEVALUATED`` is not a soft NO — it means the gate could not form an
    opinion (Hold rating, or missing target/reference price) and says so
    honestly instead of guessing.
    """

    TRADEABLE = "TRADEABLE"
    NO_TRADE = "NO_TRADE"
    UNEVALUATED = "UNEVALUATED"


class TicketSide(StrEnum):
    """Direction of the candidate trade implied by the PM's rating."""

    LONG = "LONG"
    SHORT = "SHORT"
    FLAT = "FLAT"


class TradeabilityVerdict(BaseModel):
    """The gate's full decision record; lands on the ExecutionTicket."""

    tradeability: Tradeability
    tradeability_reason: str
    side: TicketSide
    gross_edge_bps: float | None = None
    estimated_cost_bps: float | None = None
    net_edge_bps: float | None = None


def side_from_rating(rating: str) -> TicketSide:
    """Map the PM's 5-tier rating to a trade side (Hold/unknown -> FLAT)."""
    r = str(rating or "").strip().lower()
    if r in ("buy", "overweight"):
        return TicketSide.LONG
    if r in ("sell", "underweight"):
        return TicketSide.SHORT
    return TicketSide.FLAT


def gross_edge(
    side: TicketSide, target_price: float | None, reference_price: float | None
) -> float | None:
    """Directional edge as a fraction of the reference price, or None.

    None when the inputs cannot support the computation (missing prices,
    non-positive reference, or a FLAT side). A target on the WRONG side of
    the reference returns a negative fraction — that is the point.
    """
    if target_price is None or reference_price is None or reference_price <= 0.0:
        return None
    if side == TicketSide.LONG:
        return target_price / reference_price - 1.0
    if side == TicketSide.SHORT:
        return 1.0 - target_price / reference_price
    return None


def evaluate_tradeability(
    *,
    rating: str,
    target_price: float | None,
    reference_price: float | None,
    cost: CostEstimate,
    quality_tier: str = TIER_GOOD,
    risk_buffer_bps: float = RISK_BUFFER_BPS,
) -> TradeabilityVerdict:
    """Run the full gate. Rule order matters and is part of the contract:

    1. FLAT side (Hold / unparseable rating) -> UNEVALUATED. Nothing to gate.
    2. ``INVALID`` tier -> NO_TRADE (the vacuum gate should have refused the
       run already; under ``warn`` policy this is the belt-and-suspenders).
    3. ``DEGRADED_CRITICAL`` tier -> NO_TRADE: the decision was made without
       price/ATR/mark-index/core-fundamental data.
    4. Missing target or reference price -> UNEVALUATED with the specific
       missing input as the reason (the PM is allowed to rate without a
       numeric target; the gate then declines to invent one).
    5. ``net_edge_bps = gross_edge_bps - cost.total_bps - risk_buffer_bps``;
       ``<= 0`` -> NO_TRADE (zero expected value is not worth executing).
    6. Otherwise TRADEABLE.
    """
    side = side_from_rating(rating)
    if side == TicketSide.FLAT:
        return TradeabilityVerdict(
            tradeability=Tradeability.UNEVALUATED,
            tradeability_reason=REASON_HOLD,
            side=side,
        )

    if quality_tier == TIER_INVALID:
        return TradeabilityVerdict(
            tradeability=Tradeability.NO_TRADE,
            tradeability_reason=REASON_DATA_INVALID,
            side=side,
        )
    if quality_tier == TIER_DEGRADED_CRITICAL:
        return TradeabilityVerdict(
            tradeability=Tradeability.NO_TRADE,
            tradeability_reason=REASON_CRITICAL_MISSING,
            side=side,
        )

    if target_price is None:
        return TradeabilityVerdict(
            tradeability=Tradeability.UNEVALUATED,
            tradeability_reason=REASON_NO_TARGET,
            side=side,
        )
    if reference_price is None or reference_price <= 0.0:
        return TradeabilityVerdict(
            tradeability=Tradeability.UNEVALUATED,
            tradeability_reason=REASON_NO_REFERENCE,
            side=side,
        )

    edge = gross_edge(side, target_price, reference_price)
    assert edge is not None  # both prices validated above; side is LONG/SHORT
    gross_bps = edge * 1e4
    net_bps = gross_bps - cost.total_bps - risk_buffer_bps
    # Epsilon, not raw `<= 0`: a mathematically-exact edge==cost comparison
    # can land a few 1e-13 bps ABOVE zero after the round trip through
    # fractions, which would flip the verdict to TRADEABLE. The freeze
    # contract is "zero expected value is not worth executing".
    if net_bps <= 1e-9:
        return TradeabilityVerdict(
            tradeability=Tradeability.NO_TRADE,
            tradeability_reason=REASON_EDGE,
            side=side,
            gross_edge_bps=gross_bps,
            estimated_cost_bps=cost.total_bps,
            net_edge_bps=net_bps,
        )
    return TradeabilityVerdict(
        tradeability=Tradeability.TRADEABLE,
        tradeability_reason=REASON_OK,
        side=side,
        gross_edge_bps=gross_bps,
        estimated_cost_bps=cost.total_bps,
        net_edge_bps=net_bps,
    )
