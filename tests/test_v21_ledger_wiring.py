"""V2.1 — record-stage ledger wiring at the graph seam (batch F).

Three wiring points, each flag-gated and byte-compat when off:

1. ``_run_graph`` binds the ledger run context (run row registered, one
   ``run_id`` on state, context dropped in the ``finally``);
2. the risk overlay fills the ticket's ledger linkage keys (``run_id`` /
   ``evidence_ids`` / ``prediction_ids``) and mirrors the ticket into the
   central ledger DB;
3. the stock-perp Fair Value Bridge lands on the ticket (underlying/contract
   targets, fx + basis snapshot) and the overlay markdown.

conftest isolates ``ledger_db_path`` per test and holds all three V2.1 flags
off; tests below opt in per case.
"""

from __future__ import annotations

from datetime import date, datetime
from types import SimpleNamespace

import pytest

from yialpha.graph.trading_graph import YiAlphaGraph
from yialpha.ledger.evidence import record_evidence, register_run
from yialpha.ledger.models import (
    DIRECTION_UP,
    HORIZON_LADDER_DAYS,
    REPLAYABILITY_LIVE_ONLY,
    SCOPE_CONTRACT,
)
from yialpha.ledger.predictions import predictions_for_run, submit_predictions
from yialpha.ledger.run_context import (
    current_ledger_run_context,
    reset_ledger_run_context,
    set_ledger_run_context,
)
from yialpha.ledger.sqlite import get_connection, ledger_exists
from yialpha.ledger.tickets_mirror import ticket_for_run
from yialpha.perp.quote_fx import QuoteFxResult

# Live-run anchoring (precise UTC instant) applies only when the trade date
# is NOT historical — and is_historical_date compares against the LOCAL
# today, so this test must use the same local "today". A hardcoded date
# rots: on 2026-09-04 the original "2026-09-03" turned historical and the
# run registered a date-only anchor, failing the "T"-in-anchor asserts.
_TRADE_DATE = date.today().isoformat()


def _make_graph(tmp_path, **config_over) -> YiAlphaGraph:
    """Graph shell with just enough state for the overlay + run (no LLMs)."""
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
    g.ticker = "MUUSDT"
    return g


def _perp_state(**over) -> dict:
    base = {
        "final_trade_decision": "**Rating**: Buy\n\nThesis.",
        "pm_rating": "Buy",
        "pm_decision_fields": {"price_target": 60.0, "confidence": 0.7},
    }
    base.update(over)
    return base


def _stub_pipeline(g) -> None:
    """Stub everything _run_graph touches below the seam under test."""
    g.debug = False
    g.perf_tracker = None
    g.memory_log = SimpleNamespace(
        get_past_context=lambda *a, **k: None,
        store_decision=lambda **k: None,
    )
    g.resolve_instrument_context = lambda *a, **k: None
    g.propagator = SimpleNamespace(
        create_initial_state=lambda *a, **k: {"final_trade_decision": "**Rating**: Buy"},
        get_graph_args=lambda: {},
    )
    g._invoke_or_stream = lambda state, args: dict(state)
    g._apply_risk_overlay = lambda company, date, fs, ps, asset_type="stock": fs
    g._log_state = lambda date, fs: {}
    g.process_signal = lambda md: "BUY"


# --------------------------------------------------------------------------- #
# 1. _run_graph: ledger run-context binding
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_run_graph_binds_run_id_and_registers_run(tmp_path):
    g = _make_graph(tmp_path, prediction_ledger=True)
    _stub_pipeline(g)
    final_state, _signal = g._run_graph(
        "MUUSDT", _TRADE_DATE, asset_type="crypto_perp"
    )
    run_id = final_state.get("run_id")
    assert isinstance(run_id, str) and run_id.startswith("R") and len(run_id) == 13
    assert ledger_exists()
    row = get_connection(readonly=True).execute(
        "SELECT ticker, asset_type, instrument_class, analysis_as_of FROM runs "
        "WHERE run_id = ?",
        (run_id,),
    ).fetchone()
    assert row is not None
    assert row["ticker"] == "MUUSDT"
    assert row["asset_type"] == "crypto_perp"
    # Live runs bind the precise UTC as-of (R4 time contract); only
    # historical replays keep the date-only form. The instant's calendar
    # date may differ from the local trade date around midnight (UTC+8),
    # so assert the precise-UTC form itself, not a date-prefix match.
    anchor = row["analysis_as_of"]
    assert "T" in anchor and anchor.endswith("+00:00")
    assert datetime.fromisoformat(anchor).tzinfo is not None
    assert row["instrument_class"] is not None
    # The finally-block drops the context — a later run in this context
    # cannot attribute anything to the finished run.
    assert current_ledger_run_context() is None


@pytest.mark.unit
def test_run_graph_flag_off_leaves_no_run_row_or_state_key(tmp_path):
    g = _make_graph(tmp_path)  # prediction_ledger absent -> off (conftest default)
    _stub_pipeline(g)
    final_state, _signal = g._run_graph(
        "MUUSDT", _TRADE_DATE, asset_type="crypto_perp"
    )
    # Key presence (not a null value) is the signal — flag-off state is
    # byte-identical to the pre-V2.1 shape.
    assert "run_id" not in final_state
    if ledger_exists():
        count = get_connection(readonly=True).execute(
            "SELECT COUNT(*) AS n FROM runs"
        ).fetchone()["n"]
        assert count == 0


# --------------------------------------------------------------------------- #
# 2. Overlay: ticket linkage + mirror
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_overlay_links_ticket_to_ledger_and_mirrors(tmp_path, monkeypatch):
    run_id = "RLINKTEST0001"
    register_run(run_id, "MUUSDT", "crypto_perp", "stock_perp", _TRADE_DATE)
    set_ledger_run_context(
        run_id, "MUUSDT", "crypto_perp", "stock_perp", _TRADE_DATE
    )
    try:
        g = _make_graph(tmp_path)
        monkeypatch.setattr(
            g, "_latest_close_and_atr", lambda t, d, at="stock": (50.0, 1.0)
        )
        monkeypatch.setattr(g, "_trailing_funding_total", lambda t, d: None)
        monkeypatch.setattr(g, "_latest_mark_close", lambda t, d: None)
        submit_predictions(
            run_id,
            "market",
            "MUUSDT",
            SCOPE_CONTRACT,
            [
                {
                    "horizon_days": horizon,
                    "direction": DIRECTION_UP,
                    "prob_up": 0.6,
                    "confidence": 0.5,
                }
                for horizon in HORIZON_LADDER_DAYS
            ],
            _TRADE_DATE,
        )
        record_evidence(
            run_id,
            "perp_market_bundle",
            "binance_perp",
            "MUUSDT",
            SCOPE_CONTRACT,
            "bundle payload text",
            replayability=REPLAYABILITY_LIVE_ONLY,
            available_at=f"{_TRADE_DATE}T00:00:00+00:00",
            analysis_as_of=_TRADE_DATE,
        )

        out = g._apply_risk_overlay(
            "MUUSDT", _TRADE_DATE, _perp_state(), {"equity": 100_000},
            asset_type="crypto_perp",
        )
        ticket = out["execution_ticket"]
        assert ticket["run_id"] == run_id
        expected_prediction_ids = {
            record.prediction_id for record in predictions_for_run(run_id)
        }
        assert set(ticket["prediction_ids"]) == expected_prediction_ids
        assert len(ticket["evidence_ids"]) == 1

        mirrored = ticket_for_run(run_id)
        assert mirrored is not None
        assert mirrored["ticket_id"] == ticket["ticket_id"]
        assert mirrored["run_id"] == run_id
    finally:
        reset_ledger_run_context()


@pytest.mark.unit
def test_overlay_without_context_keeps_linkage_empty(tmp_path, monkeypatch):
    reset_ledger_run_context()
    g = _make_graph(tmp_path)
    monkeypatch.setattr(
        g, "_latest_close_and_atr", lambda t, d, at="stock": (50.0, 1.0)
    )
    monkeypatch.setattr(g, "_trailing_funding_total", lambda t, d: None)
    out = g._apply_risk_overlay(
        "MUUSDT", _TRADE_DATE, _perp_state(), {"equity": 100_000},
        asset_type="crypto_perp",
    )
    ticket = out["execution_ticket"]
    assert ticket["run_id"] is None
    assert ticket["prediction_ids"] == []
    assert ticket["evidence_ids"] == []
    assert ticket_for_run("no-such-run") is None


# --------------------------------------------------------------------------- #
# 3. Fair Value Bridge on the ticket + overlay
# --------------------------------------------------------------------------- #


def _fx_ok(_as_of=None) -> QuoteFxResult:
    return QuoteFxResult(
        rate=0.9993,
        available_at=f"{_TRADE_DATE}T00:00:00+00:00",
        fetched_at=f"{_TRADE_DATE}T00:00:00+00:00",
    )


@pytest.mark.unit
def test_bridge_populates_stock_perp_ticket(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "yialpha.graph.routing.instrument_class", lambda a, t: "stock_perp"
    )
    monkeypatch.setattr("yialpha.perp.quote_fx.usdt_usd_as_of", _fx_ok)
    g = _make_graph(tmp_path, stock_perp_fair_value=True)
    monkeypatch.setattr(
        g, "_latest_close_and_atr", lambda t, d, at="stock": (0.5000, 0.01)
    )
    monkeypatch.setattr(g, "_trailing_funding_total", lambda t, d: None)
    monkeypatch.setattr(g, "_latest_mark_close", lambda t, d: 0.4999)

    out = g._apply_risk_overlay(
        "MUUSDT", _TRADE_DATE,
        _perp_state(
            pm_decision_fields={
                # PM quoted the target on the UNDERLYING in USD directly.
                "underlying_price_target": 100.0,
                "price_target": 100.05,
                "price_target_currency": "USDT",
                "confidence": 0.7,
            }
        ),
        {"equity": 100_000},
        asset_type="crypto_perp",
    )
    ticket = out["execution_ticket"]
    assert ticket["underlying_target"] == 100.0
    assert ticket["contract_target"] == pytest.approx(100.0 / 0.9993)
    assert ticket["price_target_basis"] == "last"
    assert ticket["quote_fx"]["rate"] == pytest.approx(0.9993)
    assert ticket["quote_fx"]["source"] == "binance_spot:USDCUSDT_inverse"
    assert ticket["basis_snapshot"]["last_vs_mark"] == pytest.approx(
        (0.5 - 0.4999) / 0.4999
    )
    md = out["final_trade_decision"]
    assert "Fair Value Bridge (USD → USDT contract target)" in md
    assert "USDT/USD fx: 0.999300" in md
    assert "shadow verdict" not in md


@pytest.mark.unit
def test_bridge_fields_land_in_db_mirror(tmp_path, monkeypatch):
    """Round-3e live regression: the CANDIDATE mirror must serialize the
    ticket AFTER the fair-value bridge fills its conversion fields. The
    original attach-before-bridge order wrote None for
    underlying_target / contract_target / quote_fx / basis_snapshot into
    the tickets table while the logged ticket carried the real values, so
    the live reconciliation's logged_ticket_equals_db_mirror check failed
    for the stock-perp sample."""
    monkeypatch.setattr(
        "yialpha.graph.routing.instrument_class", lambda a, t: "stock_perp"
    )
    monkeypatch.setattr("yialpha.perp.quote_fx.usdt_usd_as_of", _fx_ok)
    g = _make_graph(tmp_path, stock_perp_fair_value=True)
    monkeypatch.setattr(
        g, "_latest_close_and_atr", lambda t, d, at="stock": (0.5000, 0.01)
    )
    monkeypatch.setattr(g, "_trailing_funding_total", lambda t, d: None)
    monkeypatch.setattr(g, "_latest_mark_close", lambda t, d: 0.4999)

    run_id = "RBRIDGEMIRROR1"
    register_run(run_id, "MUUSDT", "crypto_perp", "stock_perp", _TRADE_DATE)
    set_ledger_run_context(
        run_id, "MUUSDT", "crypto_perp", "stock_perp", _TRADE_DATE
    )
    try:
        g._apply_risk_overlay(
            "MUUSDT", _TRADE_DATE,
            _perp_state(
                pm_decision_fields={
                    "underlying_price_target": 100.0,
                    "confidence": 0.7,
                }
            ),
            {"equity": 100_000},
            asset_type="crypto_perp",
        )
    finally:
        reset_ledger_run_context()

    mirrored = ticket_for_run(run_id)
    assert mirrored is not None
    assert mirrored["underlying_target"] == 100.0
    assert mirrored["contract_target"] == pytest.approx(100.0 / 0.9993)
    assert mirrored["price_target_basis"] == "last"
    assert mirrored["quote_fx"]["rate"] == pytest.approx(0.9993)
    assert mirrored["basis_snapshot"]["last_vs_mark"] == pytest.approx(
        (0.5 - 0.4999) / 0.4999
    )


@pytest.mark.unit
def test_bridge_usd_price_target_fallback(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "yialpha.graph.routing.instrument_class", lambda a, t: "stock_perp"
    )
    monkeypatch.setattr("yialpha.perp.quote_fx.usdt_usd_as_of", _fx_ok)
    g = _make_graph(tmp_path, stock_perp_fair_value=True)
    monkeypatch.setattr(
        g, "_latest_close_and_atr", lambda t, d, at="stock": (0.5, 0.01)
    )
    monkeypatch.setattr(g, "_trailing_funding_total", lambda t, d: None)
    monkeypatch.setattr(g, "_latest_mark_close", lambda t, d: None)

    out = g._apply_risk_overlay(
        "MUUSDT", _TRADE_DATE,
        _perp_state(
            pm_decision_fields={
                # No explicit underlying target, but the PM declared the
                # quoted target itself is USD (underlying-denominated).
                "price_target": 100.0,
                "price_target_currency": "USD",
            }
        ),
        {"equity": 100_000},
        asset_type="crypto_perp",
    )
    ticket = out["execution_ticket"]
    assert ticket["underlying_target"] == 100.0
    assert ticket["contract_target"] == pytest.approx(100.0 / 0.9993)
    # Mark missing -> basis snapshot records the gap without blocking (the
    # bridge treats the observed basis as diagnostic only).
    assert ticket["basis_snapshot"]["mark_close"] is None
    assert ticket["basis_snapshot"]["last_vs_mark"] is None


@pytest.mark.unit
def test_bridge_missing_fx_records_shadow_verdict(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "yialpha.graph.routing.instrument_class", lambda a, t: "stock_perp"
    )
    monkeypatch.setattr("yialpha.perp.quote_fx.usdt_usd_as_of", lambda as_of: None)
    g = _make_graph(tmp_path, stock_perp_fair_value=True)
    monkeypatch.setattr(
        g, "_latest_close_and_atr", lambda t, d, at="stock": (0.5, 0.01)
    )
    monkeypatch.setattr(g, "_trailing_funding_total", lambda t, d: None)

    out = g._apply_risk_overlay(
        "MUUSDT", _TRADE_DATE,
        _perp_state(
            pm_decision_fields={"underlying_price_target": 100.0}
        ),
        {"equity": 100_000},
        asset_type="crypto_perp",
    )
    ticket = out["execution_ticket"]
    assert ticket["underlying_target"] == 100.0
    assert ticket["contract_target"] is None
    assert ticket["quote_fx"] is None
    md = out["final_trade_decision"]
    assert "USDT/USD fx: UNAVAILABLE" in md
    assert "shadow verdict: DEGRADED_CRITICAL / NO_TRADE recorded" in md


@pytest.mark.unit
def test_bridge_skipped_for_pure_crypto_and_flag_off(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "yialpha.graph.routing.instrument_class", lambda a, t: "pure_crypto_perp"
    )
    monkeypatch.setattr("yialpha.perp.quote_fx.usdt_usd_as_of", _fx_ok)
    for config in ({}, {"stock_perp_fair_value": True}):
        g = _make_graph(tmp_path, **config)
        monkeypatch.setattr(
            g, "_latest_close_and_atr", lambda t, d, at="stock": (0.5, 0.01)
        )
        monkeypatch.setattr(g, "_trailing_funding_total", lambda t, d: None)
        out = g._apply_risk_overlay(
            "BTCUSDT", _TRADE_DATE,
            _perp_state(
                pm_decision_fields={"underlying_price_target": 100.0}
            ),
            {"equity": 100_000},
            asset_type="crypto_perp",
        )
        ticket = out["execution_ticket"]
        assert ticket["contract_target"] is None
        assert ticket["underlying_target"] is None
        assert "Fair Value Bridge" not in out["final_trade_decision"]


# --------------------------------------------------------------------------- #
# 4. Web store readers (read-only ledger surface)
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_web_store_ledger_readers():
    from web import store

    run_id = "RWEBTEST0001"
    # Unknown run -> None (the route maps that to 404).
    assert store.load_run_predictions(run_id) is None
    register_run(run_id, "MUUSDT", "crypto_perp", "stock_perp", _TRADE_DATE)
    submit_predictions(
        run_id,
        "market",
        "MUUSDT",
        SCOPE_CONTRACT,
        [
            {
                "horizon_days": horizon,
                "direction": DIRECTION_UP,
                "prob_up": 0.55,
                "confidence": 0.5,
            }
            for horizon in HORIZON_LADDER_DAYS
        ],
        _TRADE_DATE,
    )
    payload = store.load_run_predictions(run_id)
    assert payload is not None
    assert payload["run_id"] == run_id
    assert len(payload["predictions"]) == 3
    assert len(payload["prediction_ids"]) == 3

    outcomes = store.load_outcomes()
    assert outcomes["available"] is True
    assert isinstance(outcomes["outcomes"], list)

    calibration = store.load_calibration()
    assert calibration["available"] is True
    assert "overall" in calibration
    assert "by_analyst" in calibration
