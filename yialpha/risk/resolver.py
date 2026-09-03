"""Ticket resolver — the min-multiplier composition of RiskDecisions (V2.4).

The architecture freeze (docs/V2_BASELINE.md) routes every trade through
``Decision -> Candidate Ticket -> RiskDecision -> Ticket Resolver -> Final
Ticket -> Position``. This module owns the resolver step and NOTHING else:
it composes the five hard-constraint verdicts into one final size, and it
is the ONLY layer allowed to shrink a ticket (risk can only constrain,
never re-decide — :mod:`yialpha.risk.manager` keeps the Phase-1 sizing).

Composition rules (frozen):

* ``final_multiplier = min(decision.multiplier)`` over all decisions — the
  TIGHTEST constraint binds (empty decision list -> 1.0: nothing objected).
* Any VETO -> action VETOED, final_size 0.0. A VETO dominates every PASS
  and every RESIZE.
* Else multiplier < 1 -> RESIZED with ``final_size = proposed_size x
  multiplier``; all 1.0 -> APPROVED with ``final_size = proposed_size``.
* ``final_size`` NEVER exceeds ``proposed_size``.
* ``side`` is echoed UNCHANGED — the resolver can never alter a direction;
* a negative or non-finite ``proposed_size`` is invalid input (sizes are
  non-negative absolutes; the sign lives in ``side``) -> VETOED with reason
  ``invalid_proposed_size``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from yialpha.risk.constraints import RiskDecision
from yialpha.risk.signed_math import Side

ResolverAction = Literal["APPROVED", "RESIZED", "VETOED"]


@dataclass(frozen=True)
class ResolverResult:
    """The resolver's final word on one candidate ticket.

    ``final_multiplier`` is the min across the contributing RiskDecisions
    (1.0 when nothing objected); ``final_size <= proposed_size`` always.
    ``side`` is the candidate's side echoed UNCHANGED. ``reasons`` collects
    the reasons of every decision that actually constrained the outcome
    (VETO, or multiplier < 1); ``risk_decision_rules`` lists those rules in
    frozen execution order for the report.
    """

    action: ResolverAction
    final_multiplier: float
    final_size: float
    side: Side
    reasons: list[str]
    risk_decision_rules: list[str]


def resolve_constraints(
    decisions: list[RiskDecision],
    proposed_size: float,
    side: Side,
) -> ResolverResult:
    """Compose the hard-constraint decisions into one final size.

    See the module docstring for the frozen composition rules. The decision
    list is consumed as-is (already validated by
    :class:`~yialpha.risk.constraints.RiskDecision`, so every multiplier is
    a finite float in [0.0, 1.0]); an empty list approves — no constraint
    was run, so nothing objected.
    """
    if not math.isfinite(proposed_size) or proposed_size < 0.0:
        return ResolverResult(
            action="VETOED",
            final_multiplier=0.0,
            final_size=0.0,
            side=side,
            reasons=["invalid_proposed_size"],
            risk_decision_rules=[],
        )

    final_multiplier = min((d.multiplier for d in decisions), default=1.0)

    action: ResolverAction
    if any(d.action == "VETO" for d in decisions):
        action = "VETOED"
        final_size = 0.0
    elif final_multiplier < 1.0:
        action = "RESIZED"
        # multipliers are validated into [0.0, 1.0], so this never exceeds
        # the proposal; the outer min() is belt-and-suspenders.
        final_size = min(proposed_size * final_multiplier, proposed_size)
    else:
        action = "APPROVED"
        final_size = proposed_size

    contributing = [d for d in decisions if d.action == "VETO" or d.multiplier < 1.0]
    reasons = [reason for d in contributing for reason in d.reasons]
    rules = [d.rule for d in contributing]
    return ResolverResult(
        action=action,
        final_multiplier=final_multiplier,
        final_size=final_size,
        side=side,
        reasons=reasons,
        risk_decision_rules=rules,
    )


def render_resolver_lines(result: ResolverResult) -> str:
    """Compact markdown bullets for the overlay / report.

    Deterministic text: one line for the verdict (action, final multiplier,
    final size), then the binding rules in frozen order, then veto reasons
    when the trade was refused. Numbers are formatted (never the stored
    values) so the rendered report and the logged object cannot disagree.
    """
    lines = [
        f"- **Resolver**: {result.action} · multiplier {result.final_multiplier:.2f}"
        f" · final size {result.final_size:.4f}"
    ]
    if result.risk_decision_rules:
        lines.append(
            "- **Binding constraints**: " + ", ".join(result.risk_decision_rules)
        )
    if result.action == "VETOED" and result.reasons:
        lines.append("- **Veto reasons**: " + "; ".join(result.reasons))
    return "".join(line + "\n" for line in lines)
