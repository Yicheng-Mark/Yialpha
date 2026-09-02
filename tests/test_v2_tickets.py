"""V2.0 P0.6 — ExecutionTicket schema + candidate builder.

The ticket is the single cross-stage trading object (I1): this file pins the
frozen field set, the lifecycle states, the version stamps, and the
deterministic candidate composition (rating -> side, edge/cost/net, perp
leverage via the shared perp_ticket_numbers).
"""

from __future__ import annotations

import pytest

from yialpha.risk.tradeability import TicketSide, Tradeability
from yialpha.tickets import (
    DEFAULT_HORIZON_DAYS,
    ExecutionTicket,
    TicketStatus,
    build_candidate_ticket,
    new_decision_id,
    new_ticket_id,
    render_ticket_lines,
)
from yialpha.versions import COST_MODEL_VERSION, TICKET_VERSION


@pytest.mark.unit
def test_frozen_field_set_is_complete():
    fields = set(ExecutionTicket.model_fields)
    expected = {
        "ticket_id", "status", "decision_id", "run_id", "symbol", "asset_type",
        "side", "tradeability", "tradeability_reason", "reference_price",
        "entry_price", "target_price", "stop_price", "gross_edge",
        "estimated_cost", "net_edge", "proposed_size", "final_size",
        "leverage", "confidence", "prediction_ids", "evidence_ids",
        "regime_id", "cost_model_version", "veto_reasons", "resize_reasons",
        # V2.1 fair-value linkage (additive optionals; TICKET_VERSION stays v1)
        "underlying_target", "contract_target", "price_target_basis",
        "quote_fx", "basis_snapshot",
        "analysis_as_of", "ticket_version",
    }
    assert fields == expected


@pytest.mark.unit
def test_lifecycle_statuses_and_defaults():
    assert {s.value for s in TicketStatus} == {
        "CANDIDATE", "PENDING_RISK", "APPROVED", "RESIZED", "VETOED",
        "OPENED", "CLOSED", "EXPIRED",
    }
    t = ExecutionTicket(symbol="BTCUSDT")
    assert t.status == TicketStatus.CANDIDATE
    assert t.side == TicketSide.FLAT
    assert t.tradeability == Tradeability.UNEVALUATED
    assert t.final_size is None            # owned by the V2.4 resolver
    assert t.prediction_ids == []          # V2.1 linkage, present from day one
    assert t.regime_id is None             # V2.2 linkage
    assert t.ticket_version == TICKET_VERSION
    assert t.cost_model_version == COST_MODEL_VERSION


@pytest.mark.unit
def test_ids_are_unique_and_prefixed():
    ids = {new_ticket_id() for _ in range(200)}
    assert len(ids) == 200
    assert all(i.startswith("T") and len(i) == 13 for i in ids)
    d = new_decision_id()
    assert d.startswith("D") and len(d) == 13


@pytest.mark.unit
def test_candidate_builder_tradeable_long():
    t = build_candidate_ticket(
        symbol="AAPL", asset_type="stock", rating="Buy",
        target_weight=0.10, entry_price=190.0, stop_loss=180.0,
        reference_price=190.0,
        pm_fields={"price_target": 220.0, "confidence": 0.7},
        trade_date="2026-08-18",
    )
    assert t.side == TicketSide.LONG
    assert t.tradeability == Tradeability.TRADEABLE
    assert t.gross_edge == pytest.approx(220.0 / 190.0 - 1.0)
    # stock: 2x5bps slippage; buffer applied inside the verdict.
    assert t.estimated_cost == pytest.approx(10.0 / 1e4)
    assert t.proposed_size == pytest.approx(0.10)
    assert t.confidence == pytest.approx(0.7)
    assert t.analysis_as_of == "2026-08-18"
    assert t.status == TicketStatus.CANDIDATE
    assert t.leverage is None              # stock: no leverage leg
    # net_edge = gross − cost(10bps) − buffer(10bps), bps-rounded in the
    # verdict before converting back to a fraction.
    assert t.net_edge == pytest.approx(
        t.gross_edge - t.estimated_cost - 10.0 / 1e4, abs=2e-6
    )


@pytest.mark.unit
def test_candidate_builder_reverse_target_is_no_trade():
    # BUY whose target sits BELOW the reference: the PM's directional
    # mistake must surface on the ticket, never as a positive edge.
    t = build_candidate_ticket(
        symbol="AAPL", asset_type="stock", rating="Buy",
        target_weight=0.10, entry_price=100.0, stop_loss=95.0,
        reference_price=100.0, pm_fields={"price_target": 90.0},
    )
    assert t.tradeability == Tradeability.NO_TRADE
    assert t.gross_edge < 0.0


@pytest.mark.unit
def test_candidate_builder_hold_is_unevaluated():
    t = build_candidate_ticket(
        symbol="AAPL", asset_type="stock", rating="Hold",
        target_weight=0.0, entry_price=None, stop_loss=None,
        reference_price=100.0, pm_fields={"price_target": 110.0},
    )
    assert t.side == TicketSide.FLAT
    assert t.tradeability == Tradeability.UNEVALUATED


@pytest.mark.unit
def test_candidate_builder_critical_data_gates_perp():
    events = [{"method": "get_binance_klines", "kind": "core_error", "detail": "x"}]
    t = build_candidate_ticket(
        symbol="BTCUSDT", asset_type="crypto_perp", rating="Buy",
        target_weight=0.05, entry_price=60000.0, stop_loss=58000.0,
        reference_price=60000.0, quality_events=events,
        # Other core categories succeeded -> degradation, not a vacuum.
        core_successes={"get_binance_funding_rate"},
        pm_fields={"price_target": 66000.0},
    )
    assert t.tradeability == Tradeability.NO_TRADE
    assert t.tradeability_reason == "critical_data_missing"
    # Veto reasons stay reserved for the V2.4 RiskDecision; the gate speaks
    # only through tradeability_reason.
    assert t.veto_reasons == []


@pytest.mark.unit
def test_candidate_builder_perp_leverage_from_shared_math():
    t = build_candidate_ticket(
        symbol="BTCUSDT", asset_type="crypto_perp", rating="Buy",
        target_weight=0.05, entry_price=60000.0, stop_loss=58000.0,
        reference_price=60000.0, atr=900.0,
        pm_fields={"price_target": 66000.0},
        funding_rate_annualized=0.10,
    )
    assert t.leverage is not None and t.leverage >= 1.0
    # Same numbers as the overlay's advisory (shared perp_ticket_numbers):
    from yialpha.risk.perp_ticket import perp_ticket_numbers

    result = perp_ticket_numbers(60000.0, 900.0, "Buy", 58000.0, 0.05)
    assert result is not None and t.leverage == result[0]
    # Carry priced signed for the long: +10%/yr over DEFAULT_HORIZON_DAYS.
    assert t.estimated_cost == pytest.approx(
        (2 * 5.0 + 2 * 2.0 + 0.10 * 1e4 * DEFAULT_HORIZON_DAYS / 365.0) / 1e4
    )


@pytest.mark.unit
def test_candidate_builder_missing_pm_fields_degrades_honestly():
    t = build_candidate_ticket(
        symbol="AAPL", asset_type="stock", rating="Buy",
        target_weight=0.1, entry_price=100.0, stop_loss=95.0,
        reference_price=100.0, pm_fields=None,
    )
    assert t.target_price is None
    assert t.tradeability == Tradeability.UNEVALUATED
    assert t.tradeability_reason == "no_target_price"
    assert t.confidence is None


@pytest.mark.unit
def test_candidate_builder_survives_garbage_pm_fields():
    t = build_candidate_ticket(
        symbol="AAPL", asset_type="stock", rating="Buy",
        target_weight=0.1, entry_price=100.0, stop_loss=95.0,
        reference_price=100.0,
        pm_fields={"price_target": "not-a-number", "confidence": "high"},
    )
    assert t.target_price is None
    assert t.confidence is None


@pytest.mark.unit
def test_render_lines_show_verdict_and_edges():
    t = build_candidate_ticket(
        symbol="AAPL", asset_type="stock", rating="Buy",
        target_weight=0.10, entry_price=190.0, stop_loss=180.0,
        reference_price=190.0, pm_fields={"price_target": 220.0},
    )
    md = render_ticket_lines(t)
    assert "**Tradeability**: TRADEABLE" in md
    assert "**Edge vs Cost**:" in md and "bps" in md
    assert t.ticket_id in md
    assert "LONG" in md and "10.0%" in md
    assert md.endswith("\n")


@pytest.mark.unit
def test_render_lines_flat_ticket_still_shows_tradeability():
    t = build_candidate_ticket(
        symbol="AAPL", asset_type="stock", rating="Hold",
        target_weight=0.0, entry_price=None, stop_loss=None,
        reference_price=100.0, pm_fields={},
    )
    md = render_ticket_lines(t)
    assert "**Tradeability**: UNEVALUATED (hold_rating_no_directional_trade)" in md
    assert "**Ticket**" not in md   # no FLAT ticket-id line
