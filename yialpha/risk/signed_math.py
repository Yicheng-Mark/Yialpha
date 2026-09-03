"""Direction-decoupled perp math — the frozen V2.4 sign conventions.

V2.3 opened explicit SHORT intent (``desired_side``), so every downstream
calculation needs direction-aware primitives instead of the long-only forms
the Phase-1 modules grew up with (:mod:`yialpha.risk.atr_stop`,
``RiskManager.atr_stop_from_values``). This module is that shared
vocabulary: one place where the sign conventions are written down and pinned
by tests, so tickets, outcome pricing, and the V2.4 portfolio preview cannot
drift into three different sign languages.

FROZEN conventions (docs/V2_BASELINE.md, worklog/v2-perp-progress.md):

* **Funding** — ``-position_sign * notional * funding_rate``: a LONG PAYS
  positive funding, a SHORT RECEIVES it. This mirrors the outcome-compute
  convention (``funding = -sign x sum(rate)``, batch E) and the cost
  model's signed carry leg; the result is NEVER ``abs()``'d — a credit must
  stay a credit.
* **Stops trigger on the contract LAST trade price** (Binance conditional
  orders default to CONTRACT_PRICE) while **liquidations are judged on the
  MARK price** — the two can diverge, and
  :data:`yialpha.risk.perp_ticket.STOP_TRIGGER_BASIS` discloses the same
  split on every ticket.
* **Same-bar tie** — when ONE bar trips BOTH the stop and the liquidation,
  the resolution counts the LIQUIDATION first (conservative: the total loss
  dominates any stop salvage).

Every ``size`` argument is a NON-NEGATIVE absolute value; the sign lives in
``side`` (the same contract the V2.4 resolver enforces on
``proposed_size``). Pure functions, no I/O.
"""

from __future__ import annotations

import math
from typing import Literal

#: Signed position-intent vocabulary. Echoed UNCHANGED by the V2.4 resolver —
#: the resolver can shrink a size but never alter a direction.
Side = Literal["LONG", "SHORT", "FLAT"]


def position_sign(side: Side) -> int:
    """Return +1 for LONG, -1 for SHORT, 0 for FLAT.

    The one canonical mapping from the :data:`Side` vocabulary to arithmetic;
    every signed formula below (and the V2.4 constraint layer) routes through
    it so a typo cannot silently invert a PnL sign.
    """
    if side == "LONG":
        return 1
    if side == "SHORT":
        return -1
    return 0


def signed_position(side: Side, size: float) -> float:
    """Return ``sign(side) x size`` — the signed position from parts.

    ``size`` is a NON-NEGATIVE absolute value (contracts, coins, notional);
    the direction is carried by ``side`` alone. FLAT returns 0.0 regardless
    of ``size`` — a flat intent has no magnitude to inherit. Callers that
    pass a negative ``size`` on a non-FLAT side get the literal
    ``sign x size`` (i.e. an inverted position); the contract is enforced
    upstream (resolver rejects negative sizes), not silently abs()'d here.
    """
    if side == "FLAT":
        return 0.0
    return float(position_sign(side) * size)


def funding_pnl(position_sign: int, notional: float, funding_rate: float) -> float:
    """Return the funding PnL over one settlement: ``-sign * notional * rate``.

    EXACTLY ``-position_sign * notional * funding_rate`` — a long (+1 sign)
    PAYS positive funding (negative PnL), a short (-1 sign) RECEIVES it
    (positive PnL); a zero sign accrues nothing. Never ``abs()``: the frozen
    outcome-compute convention (batch E: ``funding = -sign x sum(rate)``)
    and the cost model's signed carry leg both rely on the sign surviving.
    ``notional`` is the ABSOLUTE (non-negative) position notional.
    """
    return -position_sign * notional * funding_rate


def atr_stop(side: Side, entry: float, atr: float, mult: float) -> float | None:
    """Return the direction-aware ATR stop, or None when it cannot be formed.

    LONG: ``entry - mult * atr``; SHORT: ``entry + mult * atr`` — the short
    form is the exact mirror of the long one. FLAT, a non-positive ``atr``
    or ``mult``, or any non-finite input yields None (no stop can be armed;
    the caller discloses rather than fabricate one). The stop is expressed
    against the contract LAST trade price per the frozen convention above.
    """
    if side == "FLAT":
        return None
    if not (math.isfinite(entry) and math.isfinite(atr) and math.isfinite(mult)):
        return None
    if atr <= 0.0 or mult <= 0.0:
        return None
    if side == "LONG":
        return entry - mult * atr
    return entry + mult * atr


def liquidation_breach(side: Side, mark: float, liquidation_price: float) -> bool:
    """Return whether the MARK price has reached the liquidation trigger.

    LONG: breached when ``mark <= liquidation_price``; SHORT: breached when
    ``mark >= liquidation_price`` (inclusive on both sides — at the trigger
    the position is gone). FLAT is never in breach. Judged on MARK price
    per the frozen convention (the exchange's liquidation engine marks on
    MARK, not last trade), so stop triggers and liquidation triggers can
    disagree inside one bar.
    """
    if side == "LONG":
        return mark <= liquidation_price
    if side == "SHORT":
        return mark >= liquidation_price
    return False


def same_bar_resolution(hit_stop: bool, hit_liquidation: bool) -> str | None:
    """Resolve a bar that may have tripped both the stop and the liquidation.

    Both tripped -> ``"liquidation"`` (CONSERVATIVE: when one bar triggers
    both, the liquidation is counted first — the total loss dominates any
    stop salvage, and assuming the stop filled would overstate equity).
    Stop only -> ``"stop"``. Neither -> None. The inputs are booleans the
    caller already evaluated against the two DIFFERENT price bases (stop on
    last trade, liquidation on mark); this function only owns the tie rule.
    """
    if hit_liquidation:
        return "liquidation"
    if hit_stop:
        return "stop"
    return None
