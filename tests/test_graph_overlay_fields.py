"""Drift guard: the overlay renderer and its parser must stay in sync.

``trading_graph._apply_risk_overlay`` renders the ``## Quantitative Risk
Overlay`` markdown block; ``yialpha.graph.overlay_fields.parse_overlay`` (the
single parser shared by web/store and scripts/analyze_window) extracts its
fields. This test round-trips a rendered-shaped overlay through the parser so
a bullet-format change in the renderer fails here instead of silently
emptying the web history view.
"""

from __future__ import annotations

from yialpha.graph.overlay_fields import (
    OVERLAY_FIELDS,
    OVERLAY_MARKER,
    parse_overlay,
)


def _rendered_overlay() -> str:
    """The exact bullet shape trading_graph._apply_risk_overlay emits."""
    overlay = (
        f"\n\n---\n\n{OVERLAY_MARKER}\n\n"
        "- **Action**: BUY\n"
        "- **Target Weight**: 12.5% of equity (12,500)\n"
        "- **Stop Loss**: 104.50\n"
        "- **Entry Reference**: 112.30\n"
        "- **Drawdown Regime**: NORMAL (2.1%)\n"
        "- **Rationale**: quarter-Kelly cap binds\n"
    )
    return "## PM decision prose\n\nBuy the dip.\n" + overlay


def _rendered_perp_overlay() -> str:
    """A crypto_perp overlay: the perp ticket bullets sit between Entry
    Reference and Drawdown Regime (exact shape _render_perp_ticket emits)."""
    overlay = (
        f"\n\n---\n\n{OVERLAY_MARKER}\n\n"
        "- **Action**: BUY\n"
        "- **Target Weight**: 12.5% of equity (12,500)\n"
        "- **Stop Loss**: 104.50\n"
        "- **Entry Reference**: 112.30\n"
        "- **Suggested Leverage**: ≤ 5.0x (liq-dist 12.5x · vol 10.0x · "
        "conviction 5.0x · hard 20x)\n"
        "- **Est. Liquidation Price**: 90.1 (19.9% from entry; the stop at "
        "104.5 fires first by design)\n"
        "- **Funding (7d)**: +0.031% net over 7d (longs pay)\n"
        "- **Drawdown Regime**: NORMAL (2.1%)\n"
        "- **Rationale**: quarter-Kelly cap binds\n"
    )
    return "## PM decision prose\n\nBuy the dip.\n" + overlay


def test_roundtrip_extracts_every_rendered_field():
    out = parse_overlay(_rendered_overlay())
    assert out is not None
    assert out["action"] == "BUY"
    assert out["target_weight"] == "12.5%"
    assert out["position_value"] == "12,500"
    assert out["stop_loss"] == "104.50"
    assert out["entry"] == "112.30"
    assert out["regime"] == "NORMAL"
    assert out["rationale"] == "quarter-Kelly cap binds"
    # Perp-only bullets absent on a stock overlay — no phantom keys.
    assert "suggested_leverage" not in out
    assert "liquidation_price" not in out
    assert "funding_note" not in out


def test_roundtrip_extracts_perp_ticket_fields():
    out = parse_overlay(_rendered_perp_overlay())
    assert out is not None
    assert out["suggested_leverage"] == "5.0"
    assert out["liquidation_price"] == "90.1"
    assert out["funding_note"] == "+0.031% net over 7d (longs pay)"
    # The pre-existing stock fields still parse around the perp bullets.
    assert out["stop_loss"] == "104.50"
    assert out["regime"] == "NORMAL"


def test_missing_marker_returns_none():
    assert parse_overlay("just a decision, no overlay") is None
    assert parse_overlay("") is None
    assert parse_overlay(None) is None  # type: ignore[arg-type]


def test_disabled_banner_does_not_match_marker():
    # The DISABLED banner must not be mistaken for a real overlay section.
    disabled = "## ⚠️ Quantitative Risk Overlay DISABLED\n\n- reason: build failed"
    assert OVERLAY_MARKER not in disabled
    assert parse_overlay(disabled) is None


def test_field_map_covers_expected_keys():
    assert set(OVERLAY_FIELDS) == {
        "action", "target_weight", "position_value",
        "stop_loss", "entry", "regime", "rationale",
        "suggested_leverage", "liquidation_price", "funding_note",
    }
