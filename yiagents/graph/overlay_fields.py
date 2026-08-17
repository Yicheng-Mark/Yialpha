"""Canonical field map for the quantitative risk overlay markdown section.

The overlay block is *rendered* by
``yiagents.graph.trading_graph.YiAgentsGraph._apply_risk_overlay``
(``## Quantitative Risk Overlay`` + bullet list appended to the Portfolio
Manager's final decision). This module is the authoritative machine-readable
view of that block: the section marker constant the renderer emits, and the
regex map every downstream consumer (web history view, window analyzer) uses
to extract the numbers — previously duplicated as two drifting "faithful
copies" because ``scripts/`` is not an importable package.

If ``trading_graph`` changes the overlay bullet format, this map must change
in the same commit (and ``tests/test_graph_overlay_fields.py`` fails loudly
when the two drift: it round-trips a rendered overlay through the parser).
"""

from __future__ import annotations

import re

#: Section marker the risk overlay appends to the final trade decision.
#: Deliberately the exact enabled-overlay string — the DISABLED banner
#: (``## ⚠️ Quantitative Risk Overlay DISABLED``) must not match it.
OVERLAY_MARKER = "## Quantitative Risk Overlay"

#: Per-field regexes over the overlay bullet block. ``position_value`` is the
#: parenthetical dollar amount after the target weight (absent when the
#: overlay uses list form without it); every other field is a plain bullet.
#: ``suggested_leverage`` / ``liquidation_price`` / ``funding_note`` are
#: perp-only bullets (crypto_perp runs; absent on stock/spot overlays).
OVERLAY_FIELDS: dict[str, str] = {
    "action": r"\*\*Action\*\*:\s*(.+)",
    "target_weight": r"\*\*Target Weight\*\*:\s*([0-9.]+%)",
    "position_value": r"\*\*Target Weight\*\*:.*?\(([-0-9,]+)\)",
    "stop_loss": r"\*\*Stop Loss\*\*:\s*([-0-9.]+)",
    "entry": r"\*\*Entry Reference\*\*:\s*([-0-9.]+)",
    "suggested_leverage": r"\*\*Suggested Leverage\*\*:\s*≤\s*([0-9.]+)x",
    "liquidation_price": r"\*\*Est\. Liquidation Price\*\*:\s*([-0-9.]+)",
    "funding_note": r"\*\*Funding \(7d\)\*\*:\s*(.+)",
    "regime": r"\*\*Drawdown Regime\*\*:\s*(\S+)",
    "rationale": r"\*\*Rationale\*\*:\s*(.+)",
}


def parse_overlay(decision_md: str | None) -> dict[str, str] | None:
    """Extract the risk-overlay numbers from a final trade decision.

    Returns ``None`` when the overlay section is absent (risk overlay off, or
    the decision is empty); otherwise a dict of the parsed fields — possibly
    partial, e.g. ``position_value`` is absent when the overlay renders
    without a parenthetical dollar amount.
    """
    if not decision_md or OVERLAY_MARKER not in decision_md:
        return None
    block = decision_md[decision_md.index(OVERLAY_MARKER):]
    out: dict[str, str] = {}
    for key, pat in OVERLAY_FIELDS.items():
        m = re.search(pat, block)
        if m:
            out[key] = m.group(1).strip()
    return out
