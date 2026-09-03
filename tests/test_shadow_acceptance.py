"""Shadow-phase acceptance (2026-09-03) — the five verification items.

Baseline for comparison: commit 6cbfe1d (locked by the user). These tests
verify the shadow pipeline's RUN CORRECTNESS (not prediction quality —
that is judged separately, later, on accumulated samples):

1. Non-empty portfolio snapshots: previews run against the read-only
   positions input (ledger ∪ operator file), covering existing long /
   existing short / near-limit / over-limit books.
2. VETO ≠ close: refusing a new entry never touches the current book; only
   an APPROVED close-intent ticket closes the same-symbol position.
3. Complete reconciliation record: config, price basis, INPUT snapshot,
   candidate, every rule multiplier, shadow Final, veto reasons — persisted
   into the run log.
4. Precise write matrix: shadow writes NO portfolio rows (snapshots /
   positions / final ticket) while the prediction-ledger record stage stays
   fully independent and writing.
5. Findings disposition: the XML entity-expansion guard (real, fixed).
"""

from __future__ import annotations

import json

import pytest

import yialpha.risk.constraints as constraints_module
from yialpha.dataflows.utils import safe_xml_root
from yialpha.graph.trading_graph import YiAlphaGraph
from yialpha.ledger.evidence import register_run
from yialpha.ledger.portfolio import (
    commit_final_ticket,
    load_positions_input,
    new_snapshot_id,
    open_positions,
)
from yialpha.ledger.run_context import (
    reset_ledger_run_context,
    set_ledger_run_context,
)
from yialpha.ledger.sqlite import get_connection
from yialpha.tickets import TicketStatus

_TRADE_DATE = "2026-09-03"
_RUN_ID = "RSHADOWACC01"


def _make_graph(tmp_path, **config_over) -> YiAlphaGraph:
    g = YiAlphaGraph.__new__(YiAlphaGraph)
    g.config = {
        "risk_enabled": True,
        "kelly_fraction": 0.25,
        "max_single_position": 0.20,
        "max_single_sector": 0.30,
        "max_drawdown_hard_stop": 0.15,
        "atr_stop_mult": 2.0,
        "results_dir": str(tmp_path),
    }
    g.config.update(config_over)
    g._risk_overlay_degraded = False
    g.risk_manager = g._build_risk_manager()
    g.ticker = "BTCUSDT"
    return g


def _perp_state(**over) -> dict:
    base = {
        "final_trade_decision": "**Rating**: Buy\n\nThesis.",
        "pm_rating": "Buy",
        "pm_decision_fields": {"price_target": 70.0, "confidence": 0.7},
    }
    base.update(over)
    return base


def _stub_prices(g, monkeypatch) -> None:
    monkeypatch.setattr(
        g, "_latest_close_and_atr", lambda t, d, at="stock": (60.0, 1.5)
    )
    monkeypatch.setattr(g, "_trailing_funding_total", lambda t, d: None)
    monkeypatch.setattr(g, "_latest_mark_close", lambda t, d: None)


def _bind_run(instrument_class: str = "pure_crypto_perp") -> None:
    register_run(_RUN_ID, "BTCUSDT", "crypto_perp", instrument_class, _TRADE_DATE)
    set_ledger_run_context(
        _RUN_ID, "BTCUSDT", "crypto_perp", instrument_class, _TRADE_DATE
    )


def _write_positions_file(tmp_path, rows: list[dict]) -> str:
    path = tmp_path / "positions.json"
    path.write_text(json.dumps(rows), encoding="utf-8")
    return str(path)


def _seed_ledger_position(symbol: str, weight: float, side: str = "LONG") -> None:
    commit_final_ticket(
        ticket_payload={"ticket_id": f"T-{symbol}"},
        ticket_id=f"T-{symbol}",
        decision_id=None,
        run_id=None,
        ticket_version="v1",
        snapshot_payload={"positions": []},
        snapshot_id=new_snapshot_id(f"seed-{symbol}"),
        position_symbol=symbol,
        position_side=side,
        position_signed_weight=weight,
        open_position=True,
    )


@pytest.fixture(autouse=True)
def _clean_run_context():
    reset_ledger_run_context()
    yield
    reset_ledger_run_context()


# --------------------------------------------------------------------------- #
# Item 1 — non-empty portfolio snapshots from the read-only input
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_shadow_previews_against_positions_file_long_short_near_limit(
    tmp_path, monkeypatch
):
    from yialpha.dataflows.config import set_config

    _bind_run()
    path = _write_positions_file(
        tmp_path,
        [
            {"symbol": "ETHUSDT", "side": "LONG", "signed_weight": 0.5},
            {"symbol": "SOLUSDT", "side": "SHORT", "signed_weight": -0.2},
            # Near the single cap for a same-symbol second entry.
            {"symbol": "BTCUSDT", "side": "LONG", "signed_weight": 0.15},
        ],
    )
    set_config({"portfolio_control_mode": "shadow", "portfolio_positions_file": path})
    try:
        g = _make_graph(tmp_path, portfolio_control_mode="shadow")
        _stub_prices(g, monkeypatch)
        out = g._apply_risk_overlay(
            "BTCUSDT", _TRADE_DATE, _perp_state(), {"equity": 100_000},
            asset_type="crypto_perp",
        )
        shadow = out["portfolio_control_shadow"]
        # Non-empty input snapshot, auditable source label, both directions.
        assert len(shadow["snapshot_positions"]) == 3
        assert shadow["snapshot_source"].startswith("file:positions.json")
        sides = {p["symbol"]: p["side"] for p in shadow["snapshot_positions"]}
        assert sides["ETHUSDT"] == "LONG"
        assert sides["SOLUSDT"] == "SHORT"
        # The preview ran the real constraints against that book.
        assert len(shadow["decisions"]) == 5
        assert shadow["resolver"]["action"] in ("APPROVED", "RESIZED")
    finally:
        set_config({"portfolio_control_mode": "legacy", "portfolio_positions_file": ""})


@pytest.mark.unit
def test_shadow_over_limit_book_forces_visible_resize(tmp_path, monkeypatch):
    from yialpha.dataflows.config import set_config

    _bind_run()
    path = _write_positions_file(
        tmp_path,
        [
            {
                "symbol": "ETHUSDT", "signed_weight": 0.55,
                "instrument_class": "pure_crypto_perp",
            },
            {
                "symbol": "SOLUSDT", "signed_weight": 0.30,
                "instrument_class": "pure_crypto_perp",
            },
        ],
    )
    set_config({"portfolio_control_mode": "shadow", "portfolio_positions_file": path})
    try:
        g = _make_graph(tmp_path, portfolio_control_mode="shadow")
        _stub_prices(g, monkeypatch)
        out = g._apply_risk_overlay(
            "BTCUSDT", _TRADE_DATE, _perp_state(), {"equity": 100_000},
            asset_type="crypto_perp",
        )
        shadow = out["portfolio_control_shadow"]
        assert shadow["resolver"]["action"] == "RESIZED"
        by_rule = {d["rule"]: d for d in shadow["decisions"]}
        # With the file's classes aligned to the candidate's, the
        # post-candidate class gross (≈0.9) over the 0.6 cap is
        # deterministic; directional (0.9 > 0.8) clips too.
        assert by_rule["asset_class"]["action"] == "RESIZE"
        assert by_rule["directional_concentration"]["action"] == "RESIZE"
        assert shadow["resolver"]["final_size"] < shadow["proposed_size"]
    finally:
        set_config({"portfolio_control_mode": "legacy", "portfolio_positions_file": ""})


@pytest.mark.unit
def test_positions_file_overlays_replaces_same_symbol_ledger_row(tmp_path):
    from yialpha.dataflows.config import set_config

    _seed_ledger_position("BTCUSDT", 0.10)
    path = _write_positions_file(
        tmp_path, [{"symbol": "BTCUSDT", "signed_weight": 0.18}]
    )
    set_config({"portfolio_positions_file": path})
    try:
        rows, source = load_positions_input()
        assert source.startswith("ledger+file:positions.json")
        weights = {r["symbol"]: r["signed_weight"] for r in rows}
        assert weights["BTCUSDT"] == 0.18  # file REPLACES the ledger row
        assert len(rows) == 1
    finally:
        set_config({"portfolio_positions_file": ""})


@pytest.mark.unit
def test_empty_input_is_labeled_empty():
    rows, source = load_positions_input()
    assert rows == []
    assert source == "empty"


# --------------------------------------------------------------------------- #
# Item 2 — VETO ≠ close; APPROVED CLOSE is the only close path
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_veto_never_touches_the_existing_book(tmp_path, monkeypatch):
    _bind_run()
    _seed_ledger_position("BTCUSDT", 0.12)

    def _vetoing(candidate, snapshot, limits):  # noqa: ARG001
        return [
            constraints_module.RiskDecision(
                rule="global_gross", action="VETO", multiplier=0.0,
                reasons=["test veto"], metrics={},
            )
        ]

    monkeypatch.setattr(constraints_module, "evaluate_constraints", _vetoing)
    g = _make_graph(tmp_path, portfolio_control_mode="enforced")
    _stub_prices(g, monkeypatch)
    out = g._apply_risk_overlay(
        "BTCUSDT", _TRADE_DATE, _perp_state(), {"equity": 100_000},
        asset_type="crypto_perp",
    )
    assert out["execution_ticket"]["status"] == TicketStatus.VETOED.value
    # Refusing the NEW entry left the OLD position exactly as it was.
    (position,) = open_positions()
    assert position["symbol"] == "BTCUSDT"
    assert position["signed_weight"] == 0.12


@pytest.mark.unit
def test_approved_close_intent_closes_the_position(tmp_path, monkeypatch):
    _bind_run()
    _seed_ledger_position("BTCUSDT", 0.12)
    g = _make_graph(tmp_path, portfolio_control_mode="enforced")
    _stub_prices(g, monkeypatch)
    out = g._apply_risk_overlay(
        "BTCUSDT", _TRADE_DATE,
        _perp_state(
            final_trade_decision="**Rating**: Sell\n\nExit.",
            pm_rating="Sell",
            pm_decision_fields={},
        ),
        {"equity": 100_000},
        asset_type="crypto_perp",
    )
    control = out["portfolio_control"]
    assert control["action"] == "APPROVED"
    assert control["closed_existing_position"] is True
    assert open_positions() == []  # closed, nothing re-opened
    # The closed row remains as audit trail.
    total = get_connection(readonly=True).execute(
        "SELECT COUNT(*) AS n FROM positions WHERE symbol = ?", ("BTCUSDT",)
    ).fetchone()["n"]
    assert total == 1


@pytest.mark.unit
def test_reduce_intent_does_not_close(tmp_path, monkeypatch):
    _bind_run()
    _seed_ledger_position("BTCUSDT", 0.12)
    g = _make_graph(tmp_path, portfolio_control_mode="enforced")
    _stub_prices(g, monkeypatch)
    out = g._apply_risk_overlay(
        "BTCUSDT", _TRADE_DATE,
        _perp_state(
            final_trade_decision="**Rating**: Underweight\n\nTrim.",
            pm_rating="Underweight",
            pm_decision_fields={},
        ),
        {"equity": 100_000},
        asset_type="crypto_perp",
    )
    control = out["portfolio_control"]
    assert control["closed_existing_position"] is False
    # REDUCE maps to FLAT entry candidate: shrink-amount handling is the
    # position lifecycle's job; this ticket changes nothing.
    (position,) = open_positions()
    assert position["signed_weight"] == 0.12


# --------------------------------------------------------------------------- #
# Item 3 — complete, persisted reconciliation record
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_shadow_record_is_complete_and_self_contained(tmp_path, monkeypatch):
    _bind_run()
    g = _make_graph(tmp_path, portfolio_control_mode="shadow")
    _stub_prices(g, monkeypatch)
    out = g._apply_risk_overlay(
        "BTCUSDT", _TRADE_DATE, _perp_state(), {"equity": 100_000},
        asset_type="crypto_perp",
    )
    shadow = out["portfolio_control_shadow"]
    required = {
        "mode", "ticket_id", "tradeability", "reference_price", "side",
        "proposed_size", "resolver", "decisions", "eligibility_reasons",
        "candidate", "snapshot_positions", "snapshot_source", "limits",
        "config", "short_sizing", "snapshot_id",
    }
    assert required <= set(shadow)
    assert shadow["reference_price"] == 60.0
    assert shadow["limits"]["max_single"] == 0.20
    assert shadow["config"]["portfolio_control_mode"] == "shadow"
    # Zero un-explainable flips/enlargements: side echoed, final <= proposed,
    # every multiplier in [0, 1], and every decision carries its rule name.
    assert {d["rule"] for d in shadow["decisions"]} >= {
        "global_gross", "asset_class", "single_concentration",
        "directional_concentration", "correlation_cluster",
    }
    assert all(0.0 <= d["multiplier"] <= 1.0 for d in shadow["decisions"])
    assert shadow["resolver"]["side"] == shadow["side"]
    assert shadow["resolver"]["final_size"] <= shadow["proposed_size"]


@pytest.mark.unit
def test_shadow_record_persists_into_the_run_log(tmp_path, monkeypatch):
    _bind_run()
    g = _make_graph(tmp_path, portfolio_control_mode="shadow")
    _stub_prices(g, monkeypatch)
    out = g._apply_risk_overlay(
        "BTCUSDT", _TRADE_DATE, _perp_state(), {"equity": 100_000},
        asset_type="crypto_perp",
    )
    # Drive _log_state with the overlay's output plus the required state
    # keys (the production flow fills them from the graph's final state).
    state = dict(out)
    state.update(
        {
            "company_of_interest": "BTCUSDT",
            "trade_date": _TRADE_DATE,
            "market_report": "", "sentiment_report": "", "news_report": "",
            "fundamentals_report": "",
            "investment_debate_state": {
                "bull_history": "", "bear_history": "", "history": "",
                "current_response": "", "judge_decision": "",
            },
            "trader_investment_plan": "",
            "risk_debate_state": {
                "aggressive_history": "", "conservative_history": "",
                "neutral_history": "", "history": "", "judge_decision": "",
            },
            "investment_plan": "",
        }
    )
    g.ticker = "BTCUSDT"
    g.log_states_dict = {}
    quality_block = g._log_state(_TRADE_DATE, state)
    assert quality_block is not None  # the data-quality block rides on state
    entry = g.log_states_dict[_TRADE_DATE]
    assert "portfolio_control_shadow" in entry
    assert entry["portfolio_control_shadow"]["snapshot_id"].startswith("S")


# --------------------------------------------------------------------------- #
# Item 4 — precise write matrix under shadow mode
# --------------------------------------------------------------------------- #


def _table_counts() -> dict[str, int]:
    conn = get_connection(readonly=True)
    tickets = conn.execute("SELECT COUNT(*) AS n FROM tickets").fetchone()["n"]
    snapshots = conn.execute(
        "SELECT COUNT(*) AS n FROM portfolio_snapshots"
    ).fetchone()["n"]
    positions = conn.execute(
        "SELECT COUNT(*) AS n FROM positions"
    ).fetchone()["n"]
    return {
        "tickets": tickets,
        "portfolio_snapshots": snapshots,
        "positions": positions,
    }


@pytest.mark.unit
def test_shadow_write_matrix_and_prediction_ledger_independence(
    tmp_path, monkeypatch
):
    from yialpha.dataflows.config import set_config

    _bind_run()
    set_config({"portfolio_control_mode": "shadow"})
    try:
        g = _make_graph(tmp_path, portfolio_control_mode="shadow")
        _stub_prices(g, monkeypatch)
        out = g._apply_risk_overlay(
            "BTCUSDT", _TRADE_DATE, _perp_state(), {"equity": 100_000},
            asset_type="crypto_perp",
        )
        counts = _table_counts()
        # Shadow writes NO portfolio rows at all.
        assert counts["portfolio_snapshots"] == 0
        assert counts["positions"] == 0
        # The record-stage mirror (CANDIDATE payload) is written by the
        # prediction-ledger stage — INDEPENDENT of portfolio_control_mode.
        assert counts["tickets"] == 1
        assert out["execution_ticket"]["status"] == TicketStatus.CANDIDATE.value
        # And the prediction ledger itself accepts writes under shadow mode
        # (the scoreboard's sample entry point stays open).
        from yialpha.ledger.models import DIRECTION_UP, SCOPE_CONTRACT
        from yialpha.ledger.predictions import predictions_for_run, submit_predictions

        (prediction_id,) = submit_predictions(
            _RUN_ID, "market", "BTCUSDT", SCOPE_CONTRACT,
            [{"horizon_days": 1, "direction": DIRECTION_UP, "prob_up": 0.6}],
            _TRADE_DATE,
        )
        assert predictions_for_run(_RUN_ID)[0].prediction_id == prediction_id
    finally:
        set_config({"portfolio_control_mode": "legacy"})


# --------------------------------------------------------------------------- #
# Item 5 — XML entity-expansion guard (real finding, fixed)
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_safe_xml_root_refuses_dtd_and_entities():
    with pytest.raises(ValueError, match="DTD/ENTITY"):
        safe_xml_root(
            b'<?xml version="1.0"?><!DOCTYPE lolz [<!ENTITY lol "lol">]>'
            b"<r>&lol;</r>",
            source="test",
        )
    with pytest.raises(ValueError, match="DTD/ENTITY"):
        safe_xml_root(b"<r><!ENTITY x 'y'></r>", source="test")
    # Clean documents parse normally.
    root = safe_xml_root(b"<rss><item><title>t</title></item></rss>", source="test")
    assert root.tag == "rss"


@pytest.mark.unit
def test_sec_form4_guard_wraps_refusal_as_typed_miss():
    from yialpha.dataflows.interface import NoMarketDataError
    from yialpha.dataflows.sec_ownership import _parse_form4

    hostile = (
        b'<?xml version="1.0"?><!DOCTYPE r [<!ENTITY a "x">]><r>&a;</r>'
    )
    with pytest.raises(NoMarketDataError):
        _parse_form4(hostile)
