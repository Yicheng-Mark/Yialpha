"""ExecutionTicket — the single cross-stage trading object (V2 freeze, I1).

Every trading action flows through ``Decision -> Candidate Ticket ->
RiskDecision -> Ticket Resolver -> Final Ticket -> Position``. The ticket is
the ONLY object that crosses those stages: the PM's opinion never touches the
portfolio directly and the risk layer never mutates the PM's decision — it
produces constraints (V2.4) that a resolver applies to a COPY of the ticket.

V2.0 scope: the overlay builds a CANDIDATE ticket (status ``CANDIDATE``) per
run and stores it on the graph state (lands in ``full_states_log`` alongside
the decision). ``final_size`` stays None until the V2.4 resolver fills it;
``prediction_ids`` / ``regime_id`` are the V2.1/V2.2 linkage keys the
attribution loop will populate — present in the schema from day one so the
data contract never migrates.
"""

from __future__ import annotations

import uuid
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

from yialpha.risk.cost_model import estimate_round_trip_cost
from yialpha.risk.perp_ticket import perp_ticket_numbers
from yialpha.risk.tradeability import (
    TicketSide,
    Tradeability,
    evaluate_tradeability,
    side_from_rating,
)
from yialpha.versions import COST_MODEL_VERSION, TICKET_VERSION

#: Holding horizon the cost model accrues carry legs over when the PM gave no
#: usable one; mirrors the accuracy loop's default holding horizon.
DEFAULT_HORIZON_DAYS = 5.0


class TicketStatus(StrEnum):
    """Lifecycle of a ticket; aligns 1:1 with the V2.4 ledger event stream."""

    CANDIDATE = "CANDIDATE"          # produced by the overlay, not yet risk-checked
    PENDING_RISK = "PENDING_RISK"    # handed to the portfolio preview (V2.4)
    APPROVED = "APPROVED"            # risk constraints passed untouched
    RESIZED = "RESIZED"              # risk shrank the size (multiplier < 1)
    VETOED = "VETOED"                # risk refused the trade
    OPENED = "OPENED"                # ledger position exists
    CLOSED = "CLOSED"                # position exited (stop/TP/manual)
    EXPIRED = "EXPIRED"              # never opened and the horizon passed


def new_ticket_id() -> str:
    """Fresh unique ticket id (``T`` + 12 hex chars)."""
    return "T" + uuid.uuid4().hex[:12]


def new_decision_id() -> str:
    """Fresh decision id for the PM decision this ticket executes."""
    return "D" + uuid.uuid4().hex[:12]


class ExecutionTicket(BaseModel):
    """The frozen V2 ticket schema (see docs/V2_BASELINE.md).

    Edge/cost fields are FRACTIONS of the reference price (0.047 = 4.7%),
    matching the freeze examples; the cost model's bps decomposition is
    recoverable from the inputs plus ``cost_model_version``. ``final_size``
    is owned by the V2.4 Ticket Resolver — nothing in V2.0 writes it.
    """

    ticket_id: str = Field(default_factory=new_ticket_id)
    status: TicketStatus = TicketStatus.CANDIDATE
    decision_id: str | None = None
    run_id: str | None = None
    symbol: str
    asset_type: str = "stock"
    side: TicketSide = TicketSide.FLAT
    tradeability: Tradeability = Tradeability.UNEVALUATED
    tradeability_reason: str = ""
    reference_price: float | None = None
    entry_price: float | None = None
    target_price: float | None = None
    stop_price: float | None = None
    gross_edge: float | None = None
    estimated_cost: float | None = None
    net_edge: float | None = None
    proposed_size: float | None = None
    final_size: float | None = None
    leverage: float | None = None
    confidence: float | None = None
    prediction_ids: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    regime_id: str | None = None
    cost_model_version: str = COST_MODEL_VERSION
    veto_reasons: list[str] = Field(default_factory=list)
    resize_reasons: list[str] = Field(default_factory=list)
    analysis_as_of: str | None = None
    ticket_version: str = TICKET_VERSION


def build_candidate_ticket(
    *,
    symbol: str,
    asset_type: str,
    rating: str,
    target_weight: float,
    entry_price: float | None,
    stop_loss: float | None,
    reference_price: float | None,
    quality_events: list[dict[str, Any]] | None = None,
    core_successes: set[str] | None = None,
    pm_fields: dict[str, Any] | None = None,
    atr: float | None = None,
    funding_rate_annualized: float | None = None,
    trade_date: str | None = None,
    horizon_days: float = DEFAULT_HORIZON_DAYS,
) -> ExecutionTicket:
    """Compose the per-run Candidate Ticket deterministically.

    Inputs come from the two layers the freeze separates: ``rating`` /
    ``pm_fields`` from the PM's PortfolioDecision (opinion), everything else
    from the risk overlay's own computation (constraint facts). The
    tradeability gate runs HERE — quality classification, directional edge,
    round-trip cost — so the verdict on the ticket is reproducible from the
    logged state. No LLM is involved at any point.

    ``pm_fields`` is the ``pm_decision_fields`` state dict (may be empty on a
    free-text PM fallback): only ``price_target`` and ``confidence`` are read
    here. Leverage is perp-only (shared ``perp_ticket_numbers`` math, so the
    ticket and the markdown advisory cannot drift).
    """
    from yialpha.dataflows.quality import classify_quality

    pm_fields = pm_fields or {}
    target_price = _safe_float(pm_fields.get("price_target"))
    confidence = _safe_float(pm_fields.get("confidence"))

    classification = classify_quality(quality_events, core_successes)
    side = side_from_rating(rating)

    cost = estimate_round_trip_cost(
        asset_type, side.value.lower(), horizon_days, funding_rate_annualized
    )
    verdict = evaluate_tradeability(
        rating=rating,
        target_price=target_price,
        reference_price=reference_price,
        cost=cost,
        quality_tier=classification["tier"],
    )

    leverage: float | None = None
    if asset_type == "crypto_perp" and reference_price is not None:
        entry_for_lev = entry_price if entry_price else reference_price
        numbers = perp_ticket_numbers(
            entry_for_lev, atr, rating, stop_loss, target_weight
        )
        if numbers is not None:
            leverage = numbers[0]

    return ExecutionTicket(
        decision_id=new_decision_id(),
        symbol=symbol,
        asset_type=asset_type,
        side=side,
        tradeability=verdict.tradeability,
        tradeability_reason=verdict.tradeability_reason,
        reference_price=reference_price,
        entry_price=entry_price,
        target_price=target_price,
        stop_price=stop_loss,
        gross_edge=(
            verdict.gross_edge_bps / 1e4 if verdict.gross_edge_bps is not None else None
        ),
        estimated_cost=cost.total_bps / 1e4,
        net_edge=(
            verdict.net_edge_bps / 1e4 if verdict.net_edge_bps is not None else None
        ),
        proposed_size=target_weight,
        leverage=leverage,
        confidence=confidence,
        analysis_as_of=trade_date,
    )


def _safe_float(value: Any) -> float | None:
    """Coerce to float or None; guards against schema-passthrough strings."""
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def render_ticket_lines(ticket: ExecutionTicket) -> str:
    """Markdown bullets for the risk overlay's ticket section.

    Compact, deterministic, and honest about every NOT-applicable field so a
    reader of the decision report can see what the gate actually decided and
    why — the same numbers the logged ticket carries.
    """
    lines = [f"- **Tradeability**: {ticket.tradeability.value}"]
    if ticket.tradeability_reason:
        lines[-1] += f" ({ticket.tradeability_reason})"

    def _bps(frac: float | None) -> str | None:
        return None if frac is None else f"{frac * 1e4:.0f} bps"

    parts = []
    if ticket.gross_edge is not None:
        parts.append(f"gross {_bps(ticket.gross_edge)}")
    if ticket.estimated_cost is not None:
        parts.append(f"cost {_bps(ticket.estimated_cost)}")
    if ticket.net_edge is not None:
        parts.append(f"net {_bps(ticket.net_edge)}")
    if parts:
        lines.append("- **Edge vs Cost**: " + " · ".join(parts))

    if ticket.side != TicketSide.FLAT:
        size = (
            f"{ticket.proposed_size:.1%}"
            if ticket.proposed_size is not None
            else "n/a"
        )
        line = (
            f"- **Ticket** {ticket.ticket_id} · side {ticket.side.value} · "
            f"proposed size {size}"
        )
        if ticket.asset_type == "crypto_perp":
            lev = f"{ticket.leverage:.1f}x" if ticket.leverage is not None else "n/a"
            line += f" · leverage ≤ {lev}"
        lines.append(line)
    return "".join(line + "\n" for line in lines)
