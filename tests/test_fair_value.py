"""V2.1-D — Fair Value Bridge math/disclosure + additive schema fields.

Pins the frozen contract in three parts:

* the bridge is a pure function with pinned math (USD target ÷ USD-per-USDT
  × (1 + expected basis)), full-precision targets, banded fx, and an
  explicit ``missing_inputs`` vocabulary when a leg is absent;
* the additive ExecutionTicket fields default None, round-trip, validate
  ``price_target_basis``, and stay unpopulated by the candidate builder;
* the additive PortfolioDecision fields normalize (usdt → USDT), reject
  anything outside the allowed sets, and never change the rendered markdown
  (byte-identity with the pre-V2.1 shape).
"""

from __future__ import annotations

import pytest

from yialpha.agents.schemas import PortfolioDecision, render_pm_decision
from yialpha.perp.fair_value import BRIDGE_HEADING, fair_value_bridge, render_bridge_block
from yialpha.tickets import ExecutionTicket, build_candidate_ticket

# ---------------------------------------------------------------------------
# Bridge math
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_pinned_math_convergence_default():
    result = fair_value_bridge(100.0, 0.9993, 0.0002)
    assert result.missing_inputs == ()
    # Default expected_basis = 0.0 (disclosed convergence assumption):
    # pure FX conversion, full precision on the stored target.
    assert result.contract_target_usdt == pytest.approx(100.0 / 0.9993)
    assert result.expected_basis == 0.0
    assert result.current_basis == 0.0002
    assert len(result.chain) == 4


@pytest.mark.unit
def test_missing_current_basis_is_diagnostic_only():
    result = fair_value_bridge(100.0, 0.9993, None)
    # Basis never enters the formula: the target is still computed and the
    # chain still emitted; the gap is only named + rendered as a warning.
    assert result.contract_target_usdt == pytest.approx(100.0 / 0.9993)
    assert result.missing_inputs == ("current_basis",)
    assert len(result.chain) == 4


@pytest.mark.unit
def test_expected_basis_scales_the_target():
    result = fair_value_bridge(100.0, 0.9993, 0.0002, expected_basis=0.001)
    assert result.contract_target_usdt == pytest.approx(100.0 / 0.9993 * 1.001)


@pytest.mark.unit
def test_each_missing_leg_names_itself():
    assert fair_value_bridge(None, 0.9993, 0.0).missing_inputs == (
        "underlying_target",
    )
    assert fair_value_bridge(100.0, None, 0.0).missing_inputs == ("fx",)
    both = fair_value_bridge(None, None, None)
    assert both.missing_inputs == ("underlying_target", "fx", "current_basis")
    assert both.contract_target_usdt is None


@pytest.mark.unit
def test_fx_out_of_band_refuses_conversion():
    for bad in (0.5, 1.5, 0.0, -1.0):
        result = fair_value_bridge(100.0, bad, 0.0)
        assert result.contract_target_usdt is None
        assert result.missing_inputs == ("fx_out_of_band",)
    # Inclusive band edges still convert.
    for edge in (0.9, 1.1):
        assert fair_value_bridge(
            100.0, edge, 0.0
        ).contract_target_usdt == pytest.approx(100.0 / edge)


@pytest.mark.unit
def test_chain_records_every_step_at_rendered_precision():
    result = fair_value_bridge(100.0, 0.9993, 0.0002, expected_basis=0.001)
    assert result.chain == (
        "underlying target 100.00 USD",
        "÷ USDT/USD 0.999300",
        "× (1 + expected basis 0.001000)",
        f"= contract target {100.0 / 0.9993 * 1.001:.6f} USDT",
    )
    # The STORED target is never rounded — only rendered strings are.
    assert result.contract_target_usdt != float(f"{result.contract_target_usdt:.6f}")


@pytest.mark.unit
def test_render_block_success_has_no_warnings():
    result = fair_value_bridge(100.0, 0.9993, 0.0002)
    block = render_bridge_block(result)
    assert block.splitlines()[0] == BRIDGE_HEADING
    assert BRIDGE_HEADING == "### Fair Value Bridge (USD → USDT contract target)"
    for step in result.chain:
        assert f"- {step}" in block
    assert "⚠" not in block


@pytest.mark.unit
def test_render_block_warns_per_missing_input():
    result = fair_value_bridge(None, None, None)
    block = render_bridge_block(result)
    assert block.splitlines()[0] == BRIDGE_HEADING
    for key in ("underlying_target", "fx", "current_basis"):
        assert f"- ⚠ missing input: {key}" in block
    assert "= contract target" not in block

    oob = render_bridge_block(fair_value_bridge(100.0, 0.5, 0.0))
    assert "- ⚠ missing input: fx_out_of_band" in oob


# ---------------------------------------------------------------------------
# ExecutionTicket additive fields
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_ticket_new_fields_default_none():
    t = ExecutionTicket(symbol="TSLAUSDT")
    assert t.underlying_target is None
    assert t.contract_target is None
    assert t.price_target_basis is None
    assert t.quote_fx is None
    assert t.basis_snapshot is None


@pytest.mark.unit
def test_ticket_model_dump_round_trips_new_fields():
    t = ExecutionTicket(
        symbol="TSLAUSDT",
        underlying_target=210.0,
        contract_target=210.14,
        price_target_basis="Mark",  # case-insensitive -> "mark"
        quote_fx={"rate": 0.9993, "source": "binance_spot:USDCUSDT_inverse"},
        basis_snapshot={"last_vs_index_bps": 2.1},
    )
    assert t.price_target_basis == "mark"
    rebuilt = ExecutionTicket(**t.model_dump())
    assert rebuilt == t


@pytest.mark.unit
def test_ticket_price_target_basis_validator():
    assert ExecutionTicket(symbol="X", price_target_basis=None).price_target_basis is None
    assert ExecutionTicket(symbol="X", price_target_basis="LAST").price_target_basis == "last"
    assert ExecutionTicket(symbol="X", price_target_basis="mark").price_target_basis == "mark"
    with pytest.raises(ValueError, match="price_target_basis"):
        ExecutionTicket(symbol="X", price_target_basis="close")


@pytest.mark.unit
def test_candidate_builder_never_populates_new_fields():
    t = build_candidate_ticket(
        symbol="AAPL", asset_type="stock", rating="Buy",
        target_weight=0.10, entry_price=190.0, stop_loss=180.0,
        reference_price=190.0, pm_fields={"price_target": 220.0},
        trade_date="2026-08-18",
    )
    assert t.underlying_target is None
    assert t.contract_target is None
    assert t.price_target_basis is None
    assert t.quote_fx is None
    assert t.basis_snapshot is None


# ---------------------------------------------------------------------------
# PortfolioDecision additive fields
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_price_target_currency_normalizes_and_rejects():
    d = PortfolioDecision(
        rating="Buy", executive_summary="s", investment_thesis="t",
        price_target_currency="usdt",
    )
    assert d.price_target_currency == "USDT"
    assert PortfolioDecision(
        rating="Buy", executive_summary="s", investment_thesis="t",
        price_target_currency="Usd",
    ).price_target_currency == "USD"
    with pytest.raises(ValueError, match="price_target_currency"):
        PortfolioDecision(
            rating="Buy", executive_summary="s", investment_thesis="t",
            price_target_currency="EUR",
        )


@pytest.mark.unit
def test_underlying_currency_is_usd_only():
    d = PortfolioDecision(
        rating="Buy", executive_summary="s", investment_thesis="t",
        underlying_price_target=210.0, underlying_target_currency="usd",
    )
    assert d.underlying_target_currency == "USD"
    assert d.underlying_price_target == 210.0
    with pytest.raises(ValueError, match="underlying_target_currency"):
        PortfolioDecision(
            rating="Buy", executive_summary="s", investment_thesis="t",
            underlying_target_currency="USDT",  # the bridge accepts USD only
        )


@pytest.mark.unit
def test_decision_basis_and_nullish_fields():
    d = PortfolioDecision(
        rating="Buy", executive_summary="s", investment_thesis="t",
        price_target_basis="LAST", underlying_price_target="N/A",
    )
    assert d.price_target_basis == "last"
    assert d.underlying_price_target is None  # same #1058 placeholder coercion
    with pytest.raises(ValueError, match="price_target_basis"):
        PortfolioDecision(
            rating="Buy", executive_summary="s", investment_thesis="t",
            price_target_basis="open",
        )


def _legacy_decision() -> PortfolioDecision:
    return PortfolioDecision(
        rating="Buy",
        executive_summary="Add on strength.",
        investment_thesis="Momentum with support.",
        price_target=210.0,
        time_horizon="3-6 months",
    )


@pytest.mark.unit
def test_render_byte_identity_with_new_fields():
    """The renderer ignores the new fields entirely — byte-identical output.

    Three decisions render to the exact same markdown: one built without
    the new fields, one with all of them explicitly None, and one with
    every new field filled. Legacy parsers and the pinned pre-V2.1 shape
    cannot tell them apart.
    """
    legacy = render_pm_decision(_legacy_decision())
    all_none = render_pm_decision(
        PortfolioDecision(
            rating="Buy",
            executive_summary="Add on strength.",
            investment_thesis="Momentum with support.",
            price_target=210.0,
            time_horizon="3-6 months",
            price_target_currency=None,
            price_target_basis=None,
            underlying_price_target=None,
            underlying_target_currency=None,
        )
    )
    filled = render_pm_decision(
        PortfolioDecision(
            rating="Buy",
            executive_summary="Add on strength.",
            investment_thesis="Momentum with support.",
            price_target=210.0,
            time_horizon="3-6 months",
            price_target_currency="USDT",
            price_target_basis="mark",
            underlying_price_target=210.0,
            underlying_target_currency="USD",
        )
    )
    assert all_none == legacy
    assert filled == legacy
    for leak in ("USDT", "mark", "Fair Value"):
        assert leak not in filled
