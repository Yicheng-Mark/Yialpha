"""V2.2 Context — the versioned PIT ``RegimeState`` (shadow stage).

Covers the frozen state/id contract (determinism, version isolation, the
computed_at exclusion), per-class field assembly through monkeypatched
seams (live AND historical-PIT modes — the live legs must be skipped, not
leaked, for a past as-of), the all-missing -> ``None`` guard, migration v2
idempotency (including the duplicate-column recovery), the regime store
roundtrip, the regime_id thread through predictions/outcomes/tickets and
``_run_graph``, the market-analyst evidence injection (flag-on / flag-off
byte-identity), the scoreboard ``by_regime`` slice, and the vol
annualization fix (sessions-per-year by instrument class).

conftest holds ``prediction_ledger`` / ``regime_state`` OFF and points the
ledger DB at a per-test tmp file; tests opt in with ``set_config``.
"""

from __future__ import annotations

from dataclasses import asdict, replace
from datetime import date, timedelta

import pandas as pd
import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import Runnable

import yialpha.agents.analysts.market_analyst as ma
import yialpha.regime.compute as rc
from yialpha.dataflows import quality
from yialpha.dataflows.config import set_config
from yialpha.instruments.models import InstrumentRecord
from yialpha.ledger.evidence import evidence_for_run, register_run
from yialpha.ledger.models import SCOPE_CONTRACT
from yialpha.ledger.outcomes import write_outcome
from yialpha.ledger.predictions import prediction_by_id, submit_predictions
from yialpha.ledger.regime_store import regime_by_id, upsert_regime
from yialpha.ledger.run_context import (
    current_ledger_run_context,
    reset_ledger_run_context,
    set_ledger_run_context,
)
from yialpha.ledger.scoreboard import build_scoreboard, render_scoreboard_markdown
from yialpha.ledger.sqlite import _migrate, get_connection, ledger_exists
from yialpha.regime.state import (
    RegimeState,
    compute_regime_id,
    regime_state_from_mapping,
    render_regime_block,
)
from yialpha.versions import REGIME_VERSION

_TODAY = date.today().isoformat()
_PAST = (date.today() - timedelta(days=40)).isoformat()


@pytest.fixture(autouse=True)
def _clean_record_stage():
    from yialpha.dataflows.run_scope import reset_run_scope

    reset_ledger_run_context()
    quality.reset_quality()
    # Start cold: a prior test's shell run must never serve its prefetch
    # snapshot to this test (run-scope cache pollution in combined runs).
    reset_run_scope()
    yield
    reset_ledger_run_context()
    quality.reset_quality()
    reset_run_scope()


# --------------------------------------------------------------------------- #
# Helpers — kline frames and seam stubs
# --------------------------------------------------------------------------- #
def _frame(closes: list[float], *, opens: list[float] | None = None) -> pd.DataFrame:
    idx = pd.date_range("2025-01-01", periods=len(closes), freq="D")
    idx.name = "Date"  # binance_klines_frame ships a Date-named index
    return pd.DataFrame(
        {
            "Open": opens if opens is not None else [c * 0.999 for c in closes],
            "High": [c * 1.01 for c in closes],
            "Low": [c * 0.99 for c in closes],
            "Close": closes,
            "Volume": [1000.0] * len(closes),
        },
        index=idx,
    )


_UPTREND = [100.0 + 0.5 * i for i in range(260)]  # close > SMA50 > SMA200


def _klines_recorder(calls: list[tuple[str, str]]):
    """Stub for ``binance_klines_frame`` recording (venue, price_type)."""

    def fake_klines(
        symbol, start_date, end_date, interval="1d", venue="binance_perp",
        price_type="last", closed_as_of=None,
    ):
        calls.append((venue, price_type))
        if venue == "binance_spot":
            # ~+10 bps vs the perp close: present but below the 50 bps
            # stress trigger, so the composite stays deterministic here.
            return _frame([c * 0.999 for c in _UPTREND])
        if price_type == "index":
            return _frame([c * 0.999 for c in _UPTREND])
        return _frame(_UPTREND)

    return fake_klines


_FUNDING_CSV = (
    "fundingTime,fundingRate\n"
    "2026-08-26 00:00:00,0.00030\n"
    "2026-08-26 08:00:00,0.00020\n"
)
_DEPTH_OK = {
    "status": "ok",
    "spread_bps": 1.5,
    "bands": {
        "50": {"bid_notional": 1_000_000.0, "ask_notional": 1_000_000.0},
    },
}
_OI_OK = {"status": "ok", "latest": 100.0, "percentile": 42.0, "chg_1d": 0.01}
_LSR_OK = {"status": "ok", "global_account": {"status": "ok", "latest": 1.30}}
_TAKER_OK = {"status": "ok", "latest": 1.10}


def _mock_live_seams(monkeypatch, calls: list[tuple[str, str]] | None = None) -> None:
    monkeypatch.setattr(
        rc, "binance_klines_frame", _klines_recorder([] if calls is None else calls)
    )
    monkeypatch.setattr(rc, "get_binance_funding_rate", lambda s, a, b: _FUNDING_CSV)
    monkeypatch.setattr(rc, "_fetch_open_interest", lambda s, d: dict(_OI_OK))
    monkeypatch.setattr(rc, "_fetch_lsr", lambda s, d: dict(_LSR_OK))
    monkeypatch.setattr(rc, "_fetch_taker", lambda s, d: dict(_TAKER_OK))
    monkeypatch.setattr(rc, "_fetch_depth_bands", lambda s: dict(_DEPTH_OK))


# --------------------------------------------------------------------------- #
# 1. regime_id determinism + version isolation
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_regime_id_deterministic_and_input_sensitive():
    state = RegimeState(
        trend_regime="up", realized_vol_pct=3.2, funding_pct=0.0012,
        analysis_as_of="2026-09-03", missing_inputs=("depth_bands",),
    )
    same = RegimeState(
        trend_regime="up", realized_vol_pct=3.2, funding_pct=0.0012,
        analysis_as_of="2026-09-03", missing_inputs=("depth_bands",),
    )
    first = compute_regime_id(state)
    assert first.startswith("G") and len(first) == 13
    assert compute_regime_id(same) == first  # same inputs -> same id
    # one changed input -> a different id (per-field sensitivity)
    assert compute_regime_id(replace(same, funding_pct=0.0013)) != first
    # missing-input set participates: a partial regime is a DIFFERENT regime
    assert compute_regime_id(replace(same, missing_inputs=())) != first
    # computed_at is excluded from the preimage (idempotent recompute)
    later = replace(same, computed_at="2026-09-03T23:59:59+00:00")
    assert compute_regime_id(later) == first
    # the version prefix isolates definitions across a REGIME_VERSION bump
    # (v3 default vs the retired v2 definitions -> disjoint ids)
    assert compute_regime_id(same, "v2") != first
    assert compute_regime_id(same, "v4") != first
    # a plain mapping (stored payload) re-derives the same id
    assert compute_regime_id(asdict(same)) == first


@pytest.mark.unit
def test_regime_state_from_mapping_roundtrip_and_render():
    state = RegimeState(
        trend_regime="down", realized_vol_pct=4.1, funding_pct=-0.0007,
        analysis_as_of="2026-09-03", missing_inputs=("depth_bands", "lsr"),
        confidence_components={"price_history": 1.0, "depth": 0.0},
    )
    payload = asdict(state)
    payload["missing_inputs"] = sorted(payload["missing_inputs"], reverse=True)
    payload["ticker"] = "BTCUSDT"  # ledger-side column riding along
    payload["regime_id"] = compute_regime_id(state)
    restored = regime_state_from_mapping(payload)
    assert restored.trend_regime == "down"
    assert restored.missing_inputs == ("depth_bands", "lsr")  # canonical sorted
    assert restored.confidence_components == {"price_history": 1.0, "depth": 0.0}
    assert restored.regime_id == payload["regime_id"]

    block = render_regime_block(state)
    assert "### Regime State" in block
    assert "down" in block and "4.100" in block
    assert "**Missing inputs**: depth_bands, lsr" in block
    # None fields never render
    assert "Listing age" not in block
    assert "Session state" not in block


# --------------------------------------------------------------------------- #
# 2. Per-class assembly (monkeypatched seams)
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_pure_crypto_live_assembly(monkeypatch):
    calls: list[tuple[str, str]] = []
    _mock_live_seams(monkeypatch, calls)
    state = rc.compute_regime_state(
        "BTCUSDT", "crypto_perp", "pure_crypto_perp", _TODAY, end_date=_TODAY,
    )
    assert state is not None
    assert state.regime_version == REGIME_VERSION
    assert state.regime_id == compute_regime_id(state)
    assert state.trend_regime == "up"
    assert state.realized_vol_pct is not None and state.realized_vol_pct > 0
    assert state.funding_pct == pytest.approx(0.0005)
    assert state.oi_pct == pytest.approx(42.0)
    assert state.lsr_crowding == pytest.approx(1.30)
    assert state.taker_aggression == pytest.approx(1.10)
    assert state.spot_perp_basis_bps is not None and 0 < state.spot_perp_basis_bps < 50
    assert state.spread_depth_regime == "tight"
    assert state.contract_liquidity is None  # stock-perp field stays None
    assert state.liquidity_regime == "ample"
    assert state.market_stress is False  # no trigger fired
    assert state.liquidation_cascade_risk == "low"
    assert state.missing_inputs == ()
    assert state.confidence_components["positioning"] == pytest.approx(1.0)
    # the venue map proves the PIT legs came from date-bounded kline calls
    assert ("binance_spot", "last") in calls
    assert ("binance_perp", "last") in calls


@pytest.mark.unit
def test_historical_mode_skips_live_legs_no_leakage(monkeypatch):
    calls: list[tuple[str, str]] = []
    depth_calls: list[str] = []
    _mock_live_seams(monkeypatch, calls)
    monkeypatch.setattr(
        rc, "_fetch_depth_bands",
        lambda s: depth_calls.append(s) or dict(_DEPTH_OK),
    )
    state = rc.compute_regime_state(
        "BTCUSDT", "crypto_perp", "pure_crypto_perp", _PAST, end_date=_PAST,
    )
    assert state is not None
    # LIVE_ONLY leg never fetched (no leakage) and disclosed as missing
    assert depth_calls == []
    assert "depth_bands" in state.missing_inputs
    assert state.liquidity_regime is None
    assert state.spread_depth_regime is None
    assert state.confidence_components["depth"] == 0.0
    # PIT legs still computed from the date-bounded seams
    assert state.trend_regime == "up"
    assert state.funding_pct is not None


@pytest.mark.unit
def test_stock_perp_assembly_with_known_gaps(monkeypatch):
    calls: list[tuple[str, str]] = []
    _mock_live_seams(monkeypatch, calls)
    monkeypatch.setattr(
        rc, "classify_perp",
        lambda symbol, as_of=None: InstrumentRecord(
            symbol="MUUSDT", instrument_class="stock_perp",
            classification_source="binance_exchangeinfo",
            classification_confidence=1.0,
            underlying_symbol="MU", onboard_date="2024-06-01",
        ),
    )
    monkeypatch.setattr(
        rc, "get_YFin_history_cached",
        lambda symbol, start, end: _frame(_UPTREND),
    )
    state = rc.compute_regime_state(
        "MUUSDT", "crypto_perp", "stock_perp", _TODAY, end_date=_TODAY,
    )
    assert state is not None
    assert state.underlying_trend == "up"
    assert state.listing_age_days == (
        date.fromisoformat(_TODAY) - date(2024, 6, 1)
    ).days
    assert state.session_state == "continuous_24_7"  # v3: 24/7 bucket
    assert state.index_mark_basis_bps is not None
    assert state.overnight_gap_bps is not None
    assert state.contract_liquidity == "deep"
    # pure-crypto fields are structurally absent (None, NOT missing inputs)
    assert state.funding_pct is None
    assert state.oi_pct is None
    assert "funding_history" not in state.missing_inputs
    # honest gaps disclosed, never faked
    assert state.earnings_window is None and "earnings_calendar" in state.missing_inputs
    assert state.sector_index_trend is None and "sector_index" in state.missing_inputs
    # index leg came from the perp index klines (PIT)
    assert ("binance_perp", "index") in calls


@pytest.mark.unit
def test_all_inputs_missing_returns_none(monkeypatch):
    def raise_klines(*args, **kwargs):
        raise RuntimeError("no data")

    monkeypatch.setattr(rc, "binance_klines_frame", raise_klines)
    monkeypatch.setattr(rc, "_fetch_depth_bands", lambda s: {"status": "unavailable"})
    state = rc.compute_regime_state(
        "BTCUSDT", "crypto_perp", None, _PAST, end_date=_PAST,
    )
    assert state is None  # uncomputable -> callers disclose, never fake an id


@pytest.mark.unit
def test_non_perp_asset_returns_none():
    assert rc.compute_regime_state(
        "AAPL", "stock", "equity", _TODAY, end_date=_TODAY,
    ) is None


@pytest.mark.unit
def test_session_state_buckets():
    # v3 (klines-verified 2026-09-04): stock_perp trades 24/7, so EVERY
    # as-of form lands in the continuous bucket — weekends and US holidays
    # are never "closed", and the bucket no longer depends on the instant.
    from yialpha.instruments.sessions import SESSION_CONTINUOUS

    assert SESSION_CONTINUOUS == "continuous_24_7"
    assert rc._session_state("2026-09-03") == SESSION_CONTINUOUS  # Thursday
    assert rc._session_state("2026-09-05") == SESSION_CONTINUOUS  # Saturday
    assert rc._session_state("2026-07-03") == SESSION_CONTINUOUS  # US holiday
    assert rc._session_state("2026-09-03T14:30:00+00:00") == SESSION_CONTINUOUS
    assert rc._session_state("2026-09-03T02:00:00+00:00") == SESSION_CONTINUOUS
    assert rc._session_state("not-a-date") == SESSION_CONTINUOUS


@pytest.mark.unit
def test_dormant_et_session_state_machinery_retained():
    # The pre-v3 NYSE-equivalent ET buckets are dead code for stock_perp but
    # deliberately retained for future session-calendar classes: pin that the
    # machinery still reproduces the frozen v2 semantics byte-for-byte.
    # Date-only weekday/weekend approximation
    assert rc._et_session_state("2026-09-03") == "regular"  # Thursday
    assert rc._et_session_state("2026-09-05") == "closed"  # Saturday
    # Full instants in EDT (2026-09-03): 14:30 UTC = 10:30 ET (regular)
    assert rc._et_session_state("2026-09-03T14:30:00+00:00") == "regular"
    assert rc._et_session_state("2026-09-03T13:00:00+00:00") == "pre_market"
    assert rc._et_session_state("2026-09-03T21:00:00+00:00") == "post_market"
    assert rc._et_session_state("2026-09-03T02:00:00+00:00") == "closed"
    assert rc._et_session_state("garbage") == "closed"


# --------------------------------------------------------------------------- #
# 3. Migration v2 (idempotent; duplicate-column recovery)
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_migration_v2_idempotent_and_duplicate_column_safe():
    conn = get_connection()  # first open runs the migration
    assert ledger_exists()
    row = conn.execute(
        "SELECT value FROM schema_meta WHERE key = 'schema_version'"
    ).fetchone()
    # v4 (timing metadata) rides along; the v2 regimes/regime_id work
    # is verified by the table/column assertions below.
    assert row[0] == "4"
    # Re-running _migrate on the migrated DB must be a no-op, not an error.
    _migrate(conn)
    # The ALTER helpers must also swallow the duplicate-column error from a
    # raced migration (the loser sees the column already exists).
    from yialpha.ledger.sqlite import (
        _alter_outcomes_add_regime_id,
        _alter_predictions_add_regime_id,
    )

    _alter_predictions_add_regime_id(conn)
    _alter_outcomes_add_regime_id(conn)
    # The linkage columns really exist and read back.
    assert conn.execute("SELECT regime_id FROM predictions LIMIT 1").fetchone() is None
    assert conn.execute("SELECT regime_id FROM outcomes LIMIT 1").fetchone() is None
    assert conn.execute("SELECT regime_id FROM regimes LIMIT 1").fetchone() is None


@pytest.mark.unit
def test_migration_upgrades_a_v1_database():
    # Build a v1-only DB by re-running the full migration on a fresh
    # connection is already v2; instead verify a v1 ROW upgrades cleanly:
    # force the stored version back to 1 and re-migrate (idempotent blocks).
    conn = get_connection()
    conn.execute("BEGIN IMMEDIATE")
    conn.execute(
        "UPDATE schema_meta SET value = '1' WHERE key = 'schema_version'"
    )
    conn.execute("COMMIT")
    _migrate(conn)
    row = conn.execute(
        "SELECT value FROM schema_meta WHERE key = 'schema_version'"
    ).fetchone()
    assert row[0] == "4"


# --------------------------------------------------------------------------- #
# 4. Regime store roundtrip
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_upsert_regime_roundtrip_and_dedupe():
    state = RegimeState(
        trend_regime="up", realized_vol_pct=3.2, analysis_as_of="2026-09-03",
        regime_version=REGIME_VERSION, computed_at="2026-09-03T10:00:00+00:00",
    )
    state = replace(state, regime_id=compute_regime_id(state))
    payload = {
        **asdict(state), "ticker": "BTCUSDT", "instrument_class": "pure_crypto_perp",
    }
    upsert_regime(payload)
    upsert_regime(payload)  # INSERT OR IGNORE: same id dedupes
    stored = regime_by_id(state.regime_id)
    assert stored is not None
    assert stored["ticker"] == "BTCUSDT"
    assert stored["instrument_class"] == "pure_crypto_perp"
    assert stored["regime_version"] == REGIME_VERSION
    assert stored["payload"]["trend_regime"] == "up"
    assert regime_by_id("G" + "0" * 12) is None
    count = (
        get_connection(readonly=True)
        .execute("SELECT COUNT(*) AS n FROM regimes")
        .fetchone()["n"]
    )
    assert count == 1


@pytest.mark.unit
def test_upsert_regime_without_id_fails_soft(caplog):
    upsert_regime({"trend_regime": "up"})  # no regime_id -> WARNING, no raise
    assert not ledger_exists() or (
        get_connection(readonly=True)
        .execute("SELECT COUNT(*) AS n FROM regimes")
        .fetchone()["n"] == 0
    )


# --------------------------------------------------------------------------- #
# 5. regime_id through predictions / outcomes
# --------------------------------------------------------------------------- #
def _seed_run_prediction(regime_id: str | None) -> str:
    register_run("run-reg-1", "BTCUSDT", "crypto_perp", "pure_crypto_perp", _TODAY)
    (prediction_id,) = submit_predictions(
        "run-reg-1", "market", "BTCUSDT", SCOPE_CONTRACT,
        [{"horizon_days": 5, "direction": "up", "prob_up": 0.6}],
        _TODAY, regime_id=regime_id,
    )
    return prediction_id


@pytest.mark.unit
def test_predictions_carry_regime_id_and_conflict_on_change():
    prediction_id = _seed_run_prediction("G" + "a" * 12)
    record = prediction_by_id(prediction_id)
    assert record is not None and record.regime_id == "G" + "a" * 12
    # identical resubmission (same regime) stays idempotent
    ids = submit_predictions(
        "run-reg-1", "market", "BTCUSDT", SCOPE_CONTRACT,
        [{"horizon_days": 5, "direction": "up", "prob_up": 0.6}],
        _TODAY, regime_id="G" + "a" * 12,
    )
    assert ids == [prediction_id]
    # the same prediction id under a DIFFERENT regime is an immutability
    # conflict — a row names exactly the context it was decided under
    with pytest.raises(ValueError, match="is immutable"):
        submit_predictions(
            "run-reg-1", "market", "BTCUSDT", SCOPE_CONTRACT,
            [{"horizon_days": 5, "direction": "up", "prob_up": 0.6}],
            _TODAY, regime_id="G" + "b" * 12,
        )


@pytest.mark.unit
def test_outcome_row_carries_regime_id():
    # as_of far enough back that both horizons are due for the worklist
    register_run("run-reg-1", "BTCUSDT", "crypto_perp", "pure_crypto_perp", _PAST)
    prediction_ids = submit_predictions(
        "run-reg-1", "market", "BTCUSDT", SCOPE_CONTRACT,
        [
            {"horizon_days": 1, "direction": "up", "prob_up": 0.6},
            {"horizon_days": 5, "direction": "up", "prob_up": 0.6},
        ],
        _PAST, regime_id="G" + "a" * 12,
    )
    prediction_h1, prediction_h5 = prediction_ids
    outcome_id = write_outcome(
        prediction_h5, "run-reg-1", 5, status="complete",
        net_return=0.01, regime_id="G" + "a" * 12,
    )
    row = (
        get_connection(readonly=True)
        .execute("SELECT regime_id FROM outcomes WHERE outcome_id = ?", (outcome_id,))
        .fetchone()
    )
    assert row["regime_id"] == "G" + "a" * 12
    # the pending worklist dict exposes regime_id for the outcome writer
    from yialpha.ledger.outcomes import pending_predictions

    pending = pending_predictions(_TODAY)
    assert pending and all(item["regime_id"] == "G" + "a" * 12 for item in pending)
    write_outcome(prediction_h1, "run-reg-1", 1, status="incomplete",
                  legs_missing=["funding"], regime_id="G" + "a" * 12)
    assert pending_predictions(_TODAY) == []  # immutable incomplete is terminal
# --------------------------------------------------------------------------- #
# 6. Graph wiring: _run_graph binds regime_id; ticket linkage
# --------------------------------------------------------------------------- #
def _make_graph_shell(tmp_path, **config_over):
    from types import SimpleNamespace

    from yialpha.graph.trading_graph import YiAlphaGraph

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
    g._log_state = lambda d, fs: {}
    g.process_signal = lambda md: "BUY"
    return g


def _stub_pipeline(g) -> None:
    """Stub the LLM/overlay seam for _run_graph wiring tests."""
    g._apply_risk_overlay = lambda company, d, fs, ps, asset_type="stock": fs


def _stub_regime(monkeypatch, trend: str = "up") -> RegimeState:
    state = RegimeState(
        trend_regime=trend, realized_vol_pct=3.0, analysis_as_of=_TODAY,
        regime_version=REGIME_VERSION, computed_at="2026-09-03T00:00:00+00:00",
    )
    state = replace(state, regime_id=compute_regime_id(state))
    monkeypatch.setattr(
        "yialpha.regime.compute.compute_regime_state",
        lambda *a, **k: state,
    )
    return state


@pytest.mark.unit
def test_run_graph_threads_regime_id_flag_on(tmp_path, monkeypatch):
    set_config({"prediction_ledger": True, "regime_state": True})
    state = _stub_regime(monkeypatch)
    g = _make_graph_shell(tmp_path, prediction_ledger=True, regime_state=True)
    _stub_pipeline(g)
    final_state, _signal = g._run_graph("BTCUSDT", _TODAY, asset_type="crypto_perp")
    assert final_state["regime_id"] == state.regime_id
    stored = regime_by_id(state.regime_id)
    assert stored is not None
    assert stored["ticker"] == "BTCUSDT"
    # the run context was re-bound with the regime id before the nodes run
    # (dropped again in the finally block)
    assert current_ledger_run_context() is None


@pytest.mark.unit
def test_run_graph_live_regime_as_of_is_intraday(tmp_path, monkeypatch):
    # D7: a live same-day run hands compute_regime_state an INTRADAY as-of —
    # a bare date normalizes to midnight in registry._asof_bound and would
    # lexically exclude same-day snapshots — while end_date stays date-only
    # (compute.py parses "%Y-%m-%d"). "Today" follows is_historical_date's
    # own calendar (local date), derived at run time — never hardcoded.
    set_config({"prediction_ledger": True, "regime_state": True})
    stub = _stub_regime(monkeypatch)
    calls: list[tuple[tuple, dict]] = []

    def capture_compute(*args, **kwargs):
        calls.append((args, kwargs))
        return stub

    monkeypatch.setattr("yialpha.regime.compute.compute_regime_state", capture_compute)
    g = _make_graph_shell(tmp_path, prediction_ledger=True, regime_state=True)
    _stub_pipeline(g)
    today = date.today().isoformat()
    g._run_graph("BTCUSDT", today, asset_type="crypto_perp")
    assert calls  # flag on + crypto_perp -> the regime stage ran
    args, kwargs = calls[0]
    assert "T" in args[3]  # full ISO timestamp, not the bare date
    assert kwargs["end_date"] == today
    assert "T" not in kwargs["end_date"]


@pytest.mark.unit
def test_run_graph_regime_flag_off_no_state_key(tmp_path, monkeypatch):
    def explode(*args, **kwargs):
        raise AssertionError("regime compute must not run with the flag off")

    monkeypatch.setattr("yialpha.regime.compute.compute_regime_state", explode)
    set_config({"prediction_ledger": True})  # regime_state stays OFF (conftest)
    g = _make_graph_shell(tmp_path, prediction_ledger=True)
    _stub_pipeline(g)
    final_state, _signal = g._run_graph("BTCUSDT", _TODAY, asset_type="crypto_perp")
    assert "regime_id" not in final_state
    assert "run_id" in final_state


@pytest.mark.unit
def test_run_graph_uncomputable_regime_disclosed_not_faked(tmp_path, monkeypatch):
    monkeypatch.setattr("yialpha.regime.compute.compute_regime_state", lambda *a, **k: None)
    set_config({"prediction_ledger": True, "regime_state": True})
    g = _make_graph_shell(tmp_path, prediction_ledger=True, regime_state=True)
    _stub_pipeline(g)
    final_state, _signal = g._run_graph("BTCUSDT", _TODAY, asset_type="crypto_perp")
    assert "regime_id" not in final_state
    assert "run_id" in final_state


@pytest.mark.unit
def test_ticket_regime_id_populated_flag_on_and_none_off(tmp_path, monkeypatch):
    set_config({"prediction_ledger": True, "regime_state": True})
    state = _stub_regime(monkeypatch)
    g = _make_graph_shell(tmp_path, prediction_ledger=True, regime_state=True)
    monkeypatch.setattr(
        g, "_latest_close_and_atr", lambda t, d, at="stock": (50.0, 1.0)
    )
    monkeypatch.setattr(g, "_trailing_funding_total", lambda t, d: None)
    monkeypatch.setattr(g, "_latest_mark_close", lambda t, d: None)
    # bind the context exactly as _run_graph would have (regime re-bound)
    run_id = "RREGTICKET01"
    register_run(run_id, "BTCUSDT", "crypto_perp", "pure_crypto_perp", _TODAY)
    set_ledger_run_context(
        run_id, "BTCUSDT", "crypto_perp", "pure_crypto_perp", _TODAY,
        regime_id=state.regime_id,
    )
    try:
        out = g._apply_risk_overlay(
            "BTCUSDT", _TODAY,
            {"final_trade_decision": "**Rating**: Buy", "pm_rating": "Buy",
             "pm_decision_fields": {}},
            {"equity": 100_000},
            asset_type="crypto_perp",
        )
        assert out["execution_ticket"]["regime_id"] == state.regime_id
    finally:
        reset_ledger_run_context()
    # Flag off / no regime in context: ticket.regime_id stays None (pinned
    # pre-V2.2 value).
    set_ledger_run_context(
        run_id, "BTCUSDT", "crypto_perp", "pure_crypto_perp", _TODAY,
    )
    try:
        out = g._apply_risk_overlay(
            "BTCUSDT", _TODAY,
            {"final_trade_decision": "**Rating**: Buy", "pm_rating": "Buy",
             "pm_decision_fields": {}},
            {"equity": 100_000},
            asset_type="crypto_perp",
        )
        assert out["execution_ticket"]["regime_id"] is None
    finally:
        reset_ledger_run_context()


# --------------------------------------------------------------------------- #
# 7. Market analyst injection (flag-on present / flag-off byte-identical)
# --------------------------------------------------------------------------- #
class _CaptureLLM(Runnable):
    def __init__(self):
        super().__init__()
        self.prompt = None

    def invoke(self, inp, config=None, **kwargs):  # noqa: ARG002
        self.prompt = inp
        return AIMessage(content="FINAL REPORT", tool_calls=[])

    def bind_tools(self, tools, **kwargs):  # noqa: ARG002
        return self

    def human_contents(self) -> list[str]:
        """Human-message contents of the captured prompt (ChatPromptValue or
        plain message list)."""
        messages = (
            self.prompt.messages if hasattr(self.prompt, "messages") else self.prompt
        )
        return [
            str(m.content) for m in messages if isinstance(m, HumanMessage)
        ]


def _analyst_state(ticker: str = "BTCUSDT") -> dict[str, object]:
    return {
        "trade_date": _TODAY,
        "company_of_interest": ticker,
        "asset_type": "crypto_perp",
        "instrument_context": "CTX",
        "messages": [HumanMessage(content="analyze")],
    }


def _stored_regime_payload() -> dict:
    state = RegimeState(
        trend_regime="up", realized_vol_pct=3.2, analysis_as_of=_TODAY,
        regime_version=REGIME_VERSION, computed_at="2026-09-03T00:00:00+00:00",
        missing_inputs=("depth_bands",),
    )
    state = replace(state, regime_id=compute_regime_id(state))
    return {
        **asdict(state), "ticker": "BTCUSDT",
        "instrument_class": "pure_crypto_perp",
    }


@pytest.mark.unit
def test_market_analyst_injects_stored_regime_block(monkeypatch):
    set_config({"prediction_ledger": True, "regime_state": True})
    payload = _stored_regime_payload()
    upsert_regime(payload)
    run_id = "run-ma-reg-1"
    register_run(run_id, "BTCUSDT", "crypto_perp", "pure_crypto_perp", _TODAY)
    set_ledger_run_context(
        run_id, "BTCUSDT", "crypto_perp", "pure_crypto_perp", _TODAY,
        regime_id=payload["regime_id"],
    )
    try:
        llm = _CaptureLLM()
        ma.create_market_analyst(llm)(_analyst_state())
        injected = [
            content for content in llm.human_contents()
            if "<start_of_regime_state>" in content
        ]
        assert len(injected) == 1
        assert "<end_of_regime_state>" in injected[0]
        assert "[EXTERNAL EVIDENCE" in injected[0]
        assert "Missing inputs" in injected[0] and "depth_bands" in injected[0]
        rows = [r for r in evidence_for_run(run_id) if r.source == "regime_state"]
        assert len(rows) == 1
        assert rows[0].scope == SCOPE_CONTRACT
        assert rows[0].replayability == "LIVE_ONLY"  # live date -> live tag
    finally:
        reset_ledger_run_context()


@pytest.mark.unit
def test_market_analyst_regime_historical_is_pit_replayable(monkeypatch):
    set_config({"prediction_ledger": True, "regime_state": True})
    payload = _stored_regime_payload()
    upsert_regime(payload)
    run_id = "run-ma-reg-2"
    register_run(run_id, "BTCUSDT", "crypto_perp", "pure_crypto_perp", _TODAY)
    set_ledger_run_context(
        run_id, "BTCUSDT", "crypto_perp", "pure_crypto_perp", _TODAY,
        regime_id=payload["regime_id"],
    )
    monkeypatch.setattr(ma, "is_historical_date", lambda d: True)
    try:
        ma.create_market_analyst(_CaptureLLM())(_analyst_state())
        rows = [r for r in evidence_for_run(run_id) if r.source == "regime_state"]
        assert len(rows) == 1
        assert rows[0].replayability == "PIT_REPLAYABLE"
    finally:
        reset_ledger_run_context()


@pytest.mark.unit
def test_market_analyst_regime_flag_off_prompt_byte_identical(monkeypatch):
    set_config({"prediction_ledger": True})  # regime_state OFF (conftest)
    payload = _stored_regime_payload()
    upsert_regime(payload)
    run_id = "run-ma-reg-3"
    register_run(run_id, "BTCUSDT", "crypto_perp", "pure_crypto_perp", _TODAY)
    set_ledger_run_context(
        run_id, "BTCUSDT", "crypto_perp", "pure_crypto_perp", _TODAY,
        regime_id=payload["regime_id"],
    )
    try:
        llm = _CaptureLLM()
        ma.create_market_analyst(llm)(_analyst_state())
        # flag off: exactly the original user message, nothing injected
        assert llm.human_contents() == ["analyze"]
        assert [r for r in evidence_for_run(run_id) if r.source == "regime_state"] == []
    finally:
        reset_ledger_run_context()


@pytest.mark.unit
def test_market_analyst_no_regime_id_in_context_no_injection():
    set_config({"prediction_ledger": True, "regime_state": True})
    run_id = "run-ma-reg-4"
    register_run(run_id, "BTCUSDT", "crypto_perp", "pure_crypto_perp", _TODAY)
    set_ledger_run_context(
        run_id, "BTCUSDT", "crypto_perp", "pure_crypto_perp", _TODAY,
    )  # no regime_id bound (uncomputable regime)
    try:
        llm = _CaptureLLM()
        ma.create_market_analyst(llm)(_analyst_state())
        assert llm.human_contents() == ["analyze"]
        assert evidence_for_run(run_id) == []
    finally:
        reset_ledger_run_context()


# --------------------------------------------------------------------------- #
# 8. Scoreboard by_regime slice
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_scoreboard_by_regime_slice_and_no_regime_bucket():
    def seed(run_id: str, regime_id: str | None, net: float) -> None:
        register_run(run_id, "BTCUSDT", "crypto_perp", "pure_crypto_perp", _TODAY)
        (prediction_id,) = submit_predictions(
            run_id, "market", "BTCUSDT", SCOPE_CONTRACT,
            [{"horizon_days": 5, "direction": "up", "prob_up": 0.6}],
            _TODAY, regime_id=regime_id,
        )
        write_outcome(
            prediction_id, run_id, 5, status="complete",
            net_return=net, regime_id=regime_id,
        )

    g1 = "G" + "1" * 12
    g2 = "G" + "2" * 12
    seed("sr1", g1, 0.05)
    seed("sr2", g1, -0.01)
    seed("sr3", g2, 0.02)
    seed("sr4", None, 0.03)  # pre-V2.2 row -> no_regime bucket
    board = build_scoreboard()
    assert set(board["by_regime"]) == {g1, g2, "no_regime"}
    cell = board["by_regime"][g1]
    assert cell["n"] == 2
    assert cell["directional_accuracy"] == pytest.approx(0.5)  # one hit of two
    assert board["by_regime"]["no_regime"]["n"] == 1
    markdown = render_scoreboard_markdown(board)
    assert "## By regime" in markdown
    assert f"| {g1} | 2 |" in markdown
    assert "| no_regime | 1 |" in markdown


# --------------------------------------------------------------------------- #
# 9. Vol annualization fix (sessions/year by instrument class)
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_sessions_per_year_pinned_by_instrument_class():
    from yialpha.dataflows.vol_estimators import (
        CRYPTO_TRADING_DAYS_PER_YEAR,
        TRADING_DAYS_PER_YEAR,
        periods_per_year_for,
        sessions_per_year,
    )
    from yialpha.instruments.sessions import (
        SESSION_CONTINUOUS,
        trading_sessions_between,
    )

    # Derived from the session calendars over the fixed representative year.
    assert sessions_per_year("pure_crypto_perp") == CRYPTO_TRADING_DAYS_PER_YEAR
    assert sessions_per_year("pure_crypto_perp") == trading_sessions_between(
        SESSION_CONTINUOUS, "2025-01-01", "2026-01-01"
    )
    assert sessions_per_year("stock_perp") == trading_sessions_between(
        SESSION_CONTINUOUS, "2025-01-01", "2026-01-01"
    ) == 365.0  # klines-verified 24/7 (2026-09-04): same caliber as pure crypto
    # equity / unknown keep the 252 convention
    assert sessions_per_year("equity") == TRADING_DAYS_PER_YEAR
    assert sessions_per_year(None) == TRADING_DAYS_PER_YEAR
    # A known class wins over the asset-type rule: a tokenized-stock perp
    # annualizes at the klines-verified 24/7 factor 365 (the retired 261
    # weekday count understated vol — true/old ≈ sqrt(365/261) ≈ 1.18).
    assert periods_per_year_for("crypto_perp", "stock_perp") == pytest.approx(365.0)
    assert periods_per_year_for("crypto_perp", "pure_crypto_perp") == (
        CRYPTO_TRADING_DAYS_PER_YEAR
    )
    # No class verdict keeps the historical asset-type behaviour (pure
    # crypto / stocks byte-identical).
    assert periods_per_year_for("crypto_perp") == CRYPTO_TRADING_DAYS_PER_YEAR
    assert periods_per_year_for("stock") == TRADING_DAYS_PER_YEAR
    assert periods_per_year_for(None) == TRADING_DAYS_PER_YEAR


@pytest.mark.unit
def test_binance_indicator_tool_annualizes_stock_perp_on_sessions(monkeypatch):
    import yialpha.agents.utils.binance_indicator_tools as bit

    captured: dict[str, float] = {}

    def fake_derived(data, name, periods_per_year=252.0):  # noqa: ARG001
        if name == "rvol_20":
            captured["ppy"] = periods_per_year
        return pd.Series(0.5, index=data.index)

    monkeypatch.setattr(bit, "compute_derived", fake_derived)
    monkeypatch.setattr(
        bit, "binance_klines_frame", lambda *a, **k: _frame(_UPTREND)
    )
    monkeypatch.setattr(
        bit, "stock_perp_underlying", lambda s: "MU" if str(s).upper() == "MUUSDT" else None
    )
    out = bit._indicators_core("MUUSDT", _TODAY, 5, "perp", "rvol_20")
    assert captured["ppy"] == pytest.approx(365.0)  # stock perp: sessions factor
    assert "DATA_UNAVAILABLE" not in out
    captured.clear()
    bit._indicators_core("BTCUSDT", _TODAY, 5, "perp", "rvol_20")
    assert captured["ppy"] == pytest.approx(365.0)  # pure crypto unchanged


@pytest.mark.unit
def test_overnight_gap_survives_isolated_open_nan(monkeypatch):
    """An isolated NaN in the Open column must not misalign the legs.

    Dropping each column independently shifts the open list against the
    close list, so ``opens[-1] / closes[-2]`` can pair bars from
    non-adjacent days. The row-aligned drop removes the offending ROW from
    both lists, keeping the gap a true adjacent-day overnight reading.
    """
    closes = [100.0, 110.0, 120.0, 130.0, 140.0]
    opens = [99.0, 109.0, 119.0, 129.0, float("nan")]  # isolated NaN, last row
    frame = _frame(closes, opens=opens)

    def fake_klines(symbol, start_date, end_date, interval="1d",
                    venue="binance_perp", price_type="last", closed_as_of=None):
        return frame

    monkeypatch.setattr(rc, "binance_klines_frame", fake_klines)
    monkeypatch.setattr(rc, "get_binance_funding_rate", lambda s, a, b: _FUNDING_CSV)
    monkeypatch.setattr(rc, "_fetch_open_interest", lambda s, d: dict(_OI_OK))
    monkeypatch.setattr(rc, "_fetch_lsr", lambda s, d: dict(_LSR_OK))
    monkeypatch.setattr(rc, "_fetch_taker", lambda s, d: dict(_TAKER_OK))
    monkeypatch.setattr(rc, "_fetch_depth_bands", lambda s: dict(_DEPTH_OK))

    state = rc.compute_regime_state(
        "BTCUSDT", "crypto_perp", "pure_crypto_perp", _PAST,
        end_date=_PAST,
    )
    assert state is not None
    # Row-aligned: the NaN-open row drops whole, so the gap pairs the last
    # surviving open (129.0, day 4) with the prior day's close (120.0, day 3)
    # — NOT the misaligned 129.0/130.0 same-row pairing the independent
    # dropna produced.
    assert state.overnight_gap_bps == pytest.approx((129.0 / 120.0 - 1.0) * 1e4)
