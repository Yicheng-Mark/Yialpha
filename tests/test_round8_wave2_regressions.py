"""Round-8 wave-2 regression pins — S3/S4/S5/S6/S7 shards.

Each pin locks one defect found by the round-8 wave-2 multi-shard
adversarial sweep so it cannot silently regress. All pins are offline:
synthetic prices, scripted graphs, tmp_path caches only, zero network.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

from yialpha.backtest.cache import (
    DECISION_CACHE_SCHEMA_VERSION,
    CachedDecision,
    DecisionCache,
    _safe_component,
)
from yialpha.backtest.engine import (
    BacktestResult,
    _event_study_seed,
    _forward_buffer_days,
    run_backtest,
)
from yialpha.backtest.report import write_report
from yialpha.batch.locks import FileLock

# --- DecisionCache: enabled-but-dirless is a miss, never a crash (P3-1) ---- #

def test_cache_enabled_without_dir_misses_instead_of_raising():
    cache = DecisionCache(None, enabled=True)
    # Pre-fix this reached ``None / str`` on the legacy-path maths and raised
    # TypeError instead of behaving as "no cache directory = always miss".
    assert cache.get("AAPL", "2026-01-01") is None


# --- DecisionCache: was_degraded survives the round trip (P3-2) ------------ #

def test_cache_roundtrips_was_degraded(tmp_path):
    cache = DecisionCache(tmp_path)
    cache.remember(
        "AAPL", "2026-01-01", "Hold", "**Rating**: Hold",
        run_tag="t", was_degraded=True,
    )
    hit = cache.get("AAPL", "2026-01-01", run_tag="t")
    assert hit is not None and hit.was_degraded is True
    # Flag defaults to False for clean decisions.
    cache.remember("AAPL", "2026-01-02", "Buy", "md", run_tag="t")
    assert cache.get("AAPL", "2026-01-02", run_tag="t").was_degraded is False


def test_cache_schema_version_rejects_pre_v4_payloads(tmp_path):
    # v3 payloads lack the degraded flag; they must be treated as stale and
    # honestly recomputed, not replayed with an unknown degraded status.
    stale = CachedDecision(
        ticker="AAPL", date="2026-01-01", run_tag="t",
        rating="Hold", final_decision="md",
    ).to_dict()
    stale["schema_version"] = DECISION_CACHE_SCHEMA_VERSION - 1
    # Write it under the REAL key path so get() actually reads it.
    path = tmp_path / (
        f"{_safe_component('AAPL')}_{_safe_component('2026-01-01')}_"
        f"{_safe_component('t')}_{_safe_component('stock')}.json"
    )
    path.write_text(json.dumps(stale), encoding="utf-8")
    cache = DecisionCache(tmp_path)
    assert cache.get("AAPL", "2026-01-01", run_tag="t") is None
    assert not path.exists()  # corrupt/stale entry removed, not replayed


# --- locks: OS-lock failure must not strand the process lock (P3-8) ------- #

def test_lock_releases_process_lock_when_os_acquire_fails():
    lock = FileLock("some/resource.json")

    class _ExplodingOSLock:
        def acquire(self):
            raise OSError("read-only lock dir")

        def release(self):
            raise AssertionError("never acquired, never released")

    lock._oslock = _ExplodingOSLock()
    with pytest.raises(OSError), lock:
        pass
    # The stranded process lock would deadlock every later thread queuing on
    # this path; prove it was released by entering successfully without an
    # OS lock.
    lock._oslock = None
    with lock:
        pass


# --- report: mixed legs must not fold into one distribution (P2-1) -------- #

def _mini_result(ticker: str, run_tag: str) -> BacktestResult:
    return BacktestResult(
        ticker=ticker, initial_capital=100_000, holding_days=5,
        equity=[100_000, 101_000], equity_dates=["2024-01-01", "2024-01-02"],
        trades=[], benchmark_equity=[100_000, 101_000],
        benchmark_name=f"{ticker} buy-and-hold", metrics=None,
        config_summary={"run_tag": run_tag},
    )


def test_report_renders_distribution_for_distinct_run_tags(tmp_path):
    out = write_report(
        [_mini_result("AAPL", "base_r0"), _mini_result("AAPL", "base_r1")],
        path=tmp_path / "ok.md",
    )
    assert "Multi-run distribution" in out.read_text(encoding="utf-8")


def test_report_skips_distribution_for_mixed_legs(tmp_path):
    # run_baseline --full renders baseline + risk-improved legs under the
    # SAME pair tag (and several tickers): folding them into one mean/std
    # table blended two configurations per ticker.
    out = write_report(
        [
            _mini_result("AAPL", "pair0"),
            _mini_result("AAPL", "pair0"),  # same tag = A/B legs, not two draws
            _mini_result("NVDA", "pair0"),
        ],
        path=tmp_path / "mixed.md",
    )
    text = out.read_text(encoding="utf-8")
    assert "Multi-run distribution" not in text
    # Individual reports are still all written (two AAPL legs + one NVDA).
    assert text.count("# Backtest: AAPL") == 2
    assert text.count("# Backtest: NVDA") == 1


# --- forward-window buffer scales with trading days (P3-5) ---------------- #

def test_forward_buffer_days_converts_sessions_to_calendar():
    # Old formula ``holding_days + 10`` gave 25 sessions only 35 calendar
    # days (25 sessions ~ 35+ with holidays, exclusive fetch end): the last
    # decisions' forward windows silently truncated.
    assert _forward_buffer_days(5) == 17
    assert _forward_buffer_days(25) >= 45
    assert _forward_buffer_days(25) > 25 + 10


def test_buffer_widening_leaves_equity_metrics_byte_identical(monkeypatch):
    """Round-8 review P1 pin: the wider fetch exists only for the forward
    windows; equity/B&H marking stays capped at the legacy ``h + 10``
    boundary, so every equity-derived metric is IDENTICAL to a legacy-buffer
    run (pre-fix, the marking loop ran over the whole fetch window and the
    extra tail bars silently drifted Sharpe/alpha)."""
    import yialpha.backtest.engine as engine_mod

    dates = _decision_dates(8)
    ratings = dict.fromkeys(dates, "Buy")

    def _run() -> BacktestResult:
        return run_backtest(
            _FakeGraph(ratings), "AAPL", dates, holding_days=5,
            price_provider=_rising_prices,
        )

    widened = _run()
    monkeypatch.setattr(engine_mod, "_forward_buffer_days", lambda h: h + 10)
    legacy = _run()
    assert widened.equity == legacy.equity
    assert widened.equity_dates == legacy.equity_dates
    assert widened.benchmark_equity == legacy.benchmark_equity
    m_w, m_l = widened.metrics, legacy.metrics
    assert m_w.sharpe == m_l.sharpe
    assert m_w.alpha_vs_buyhold == m_l.alpha_vs_buyhold
    assert m_w.deflated_sharpe == m_l.deflated_sharpe
    assert m_w.total_return == m_l.total_return
    assert [t.raw_return for t in widened.trades] == [
        t.raw_return for t in legacy.trades
    ]


# --- event-study bootstrap seed is stable across replays (P3-4) ----------- #

def test_event_study_seed_stable_and_input_sensitive():
    a = _event_study_seed("AAPL", ["2026-01-02", "2026-01-09", "2026-01-16"])
    assert a == _event_study_seed(
        "AAPL", ["2026-01-02", "2026-01-09", "2026-01-16"]
    )
    assert a != _event_study_seed("NVDA", ["2026-01-02", "2026-01-09", "2026-01-16"])
    assert a != _event_study_seed("AAPL", ["2026-01-02", "2026-01-09"])
    assert isinstance(_event_study_seed("AAPL", []), int)  # empty tape OK


# --- long->short flip splits the episode (P3-6) ---------------------------- #

class _FakeGraph:
    """Stand-in for YiAlphaGraph: returns scripted ratings per date."""

    def __init__(self, ratings: dict[str, str]):
        self._ratings = dict(ratings)

    def propagate(self, company_name, trade_date, asset_type="stock"):
        rating = self._ratings.get(trade_date, "Hold")
        return {"final_trade_decision": f"**Rating**: {rating}"}, rating

    def _resolve_benchmark(self, ticker):
        return "SPY"


def _rising_prices(ticker, start, end):
    idx = pd.bdate_range(start, end)
    values = [100.0 * (1 + 0.001 * i) for i in range(len(idx))]
    return pd.Series(values, index=idx.strftime("%Y-%m-%d"), dtype=float)


def _decision_dates(n=6, start="2024-01-01"):
    idx = pd.bdate_range(start, periods=n * 6, freq="B")
    return [idx[i].strftime("%Y-%m-%d") for i in range(0, n * 6, 5)][:n]


def _zero_funding(ticker, start, end):
    """Deterministic zero funding — keeps this file's "zero network" promise.

    Pre-fix, ``test_flip_splits_position_episodes`` omitted ``funding_provider``,
    so the engine fell back to the REAL ``_binance_funding_provider`` and the
    pin's outcome depended on the ambient proxy reaching fapi.binance.com
    (the exact flake this module docstring claims cannot happen).
    """
    idx = pd.bdate_range(start, end)
    return pd.Series([0.0] * len(idx), index=idx.strftime("%Y-%m-%d"), dtype=float)


@pytest.mark.unit
def test_flip_splits_position_episodes():
    """Buy -> Sell under allow_short is TWO episodes, not one merged sample.

    Pre-fix, the +1 -> -1 flip satisfied neither the open branch nor the
    close branch, so win_rate / num_trades measured one sample spanning both
    legs of the trade.
    """
    dates = _decision_dates(6)
    ratings = {dates[0]: "Buy", dates[1]: "Sell"}  # later dates default Hold
    result = run_backtest(
        _FakeGraph(ratings), "BTCUSDT", dates, holding_days=5,
        price_provider=_rising_prices, funding_provider=_zero_funding,
        asset_type="crypto_perp",
        allow_short=True,
    )
    m = result.metrics
    assert m.num_trades == 2  # long leg + short leg, separately settled
    assert m.win_rate == pytest.approx(0.5)  # long won, short lost (rising tape)
    # The stamped per-trade returns keep the legs' identities distinct.
    long_leg = next(t for t in result.trades if t.rating == "Buy")
    short_leg = next(t for t in result.trades if t.rating == "Sell")
    assert long_leg.position_return > 0
    assert short_leg.position_return < 0


# ===========================================================================
# S3 shard — agents tree
# ===========================================================================

# --- fundamentals system message renders as a plain string (P2-2) --------- #

def test_fundamentals_system_message_is_plain_string():
    # Round-8: the system message was constructed as a 1-tuple, so the
    # template's {system_message} slot rendered the Python repr — the whole
    # instruction block (grounding rules + every nudge) wrapped in ("...",)
    # on EVERY fundamentals run.
    from langchain_core.messages import AIMessage, HumanMessage
    from langchain_core.runnables import Runnable

    from yialpha.agents.analysts.fundamentals_analyst import (
        create_fundamentals_analyst,
    )
    from yialpha.dataflows import config as cfgmod

    class _Capture(Runnable):
        def __init__(self):
            super().__init__()
            self.prompt = None

        def invoke(self, inp, config=None, **kwargs):  # noqa: ARG002
            self.prompt = inp
            return AIMessage(content="MOCK", tool_calls=[])

        def bind_tools(self, tools, **kwargs):  # noqa: ARG002
            return self

    orig = cfgmod.get_config()
    try:
        cfgmod.set_config({**orig, "web_search_enabled": False})
        llm = _Capture()
        node = create_fundamentals_analyst(llm)
        node({
            "trade_date": "2026-08-17", "company_of_interest": "AAPL",
            "asset_type": "stock", "instrument_context": "CTX",
            "messages": [HumanMessage(content="analyze")],
        })
        sm = str(llm.prompt)
        assert "You are a researcher tasked" in sm
        assert '("You are a researcher' not in sm  # no tuple-repr wrapper
    finally:
        cfgmod.set_config(orig)


# --- settle_prediction_capture flushes ONLY its own analyst (P2-1 half) --- #

def test_settle_scopes_flush_to_one_analyst(monkeypatch):
    from types import SimpleNamespace

    import yialpha.agents.utils.prediction_tools as pt

    ctx = SimpleNamespace(
        run_id="r-pin", analysis_as_of="2026-01-05", regime_id=None,
        instrument_class=None, asset_type="crypto_perp",
    )
    monkeypatch.setattr(pt, "current_ledger_run_context", lambda: ctx)
    monkeypatch.setattr(pt, "evidence_ids_for_run", lambda run_id: ())
    monkeypatch.setattr(pt, "submit_predictions", lambda **kw: [kw["analyst"]])

    pt.reset_prediction_capture_for_test()
    entries = [{"horizon_days": 5, "direction": "up", "prob_up": 0.6}]

    pt.begin_prediction_capture("market", "BTCUSDT", "CONTRACT")
    pt.submit_prediction.invoke({"predictions": entries})
    pt.begin_prediction_capture("news", "MU", "UNDERLYING")
    pt.submit_prediction.invoke({"predictions": entries})

    # Settling market must NOT mark news's in-flight capture as submitted
    # (the tool reply would then tell news "already been settled" and its
    # rows would be silently lost).
    pt.settle_prediction_capture("market", "BTCUSDT")

    scope = pt._CAPTURE_SCOPE.get()
    assert scope["market"].submitted is True
    assert scope["news"].submitted is False
    assert scope["news"].entries  # rows intact for news's own settlement
    pt.reset_prediction_capture_for_test()


# --- PortfolioDecision: direction case rescue + percent return (P3-1/2) --- #

def test_portfolio_decision_direction_case_rescue():
    from yialpha.agents.schemas import PortfolioDecision

    d = PortfolioDecision(
        rating="Buy", executive_summary="s", investment_thesis="t",
        underlying_direction="Bullish", contract_direction="NEUTRAL",
        desired_side="long",
    )
    assert d.underlying_direction == "bullish"
    assert d.contract_direction == "neutral"
    assert d.desired_side == "LONG"


def test_portfolio_decision_direction_junk_degrades_to_none():
    from yialpha.agents.schemas import PortfolioDecision

    d = PortfolioDecision(
        rating="Buy", executive_summary="s", investment_thesis="t",
        desired_side="sideways",
    )
    assert d.desired_side is None  # optional field degrades, never vetoes
    # Round-8 review: cross-vocabulary wording must degrade the FIELD to
    # None, not veto the decision — "Long" is desired_side vocab, not a
    # direction; "Bullish" is direction vocab, not a side.
    d2 = PortfolioDecision(
        rating="Buy", executive_summary="s", investment_thesis="t",
        underlying_direction="Long", desired_side="Bullish",
    )
    assert d2.underlying_direction is None
    assert d2.desired_side is None
    assert d2.rating == "Buy"


def test_portfolio_decision_expected_return_percent_rescue():
    from yialpha.agents.schemas import PortfolioDecision

    d = PortfolioDecision(
        rating="Buy", executive_summary="s", investment_thesis="t",
        expected_return="+4.7%",
    )
    assert d.expected_return == pytest.approx(0.047)
    # Plain fractions and negatives still pass through.
    d2 = PortfolioDecision(
        rating="Buy", executive_summary="s", investment_thesis="t",
        expected_return="-0.12",
    )
    assert d2.expected_return == pytest.approx(-0.12)


# ===========================================================================
# S4 shard — risk / perp / instruments / regime math
# ===========================================================================

# --- CVaR tail count: no float-dust extra observation (P2-1) -------------- #

def test_historical_cvar_40_obs_tail_is_exactly_two():
    from yialpha.risk.cvar import historical_cvar

    returns = [-0.10, -0.08, -0.06] + [0.01] * 37  # n=40 at the 95% level
    # 1-0.95 = 0.05000...044 in binary: ceil(40*that) used to be 3, pulling
    # the milder -0.06 into the tail (mean -0.08 instead of -0.09).
    assert historical_cvar(returns, confidence=0.95) == pytest.approx(-0.09)


# --- kelly_sizing exposes the pre-clip fraction (P3-1) -------------------- #

def test_kelly_sizing_return_raw_exposes_preclip():
    from yialpha.risk.kelly import kelly_sizing

    clipped, raw = kelly_sizing("Buy", return_raw=True)
    assert isinstance(clipped, float) and isinstance(raw, float)
    # The band CLIP can only pull the weight down...
    assert raw >= clipped
    # ...but the band FLOOR lifts it the other way: a 1-win/49-loss history
    # raw-Kellys to 0.0 while the Buy floor keeps a minimum applied weight.
    clipped_floor, raw_floor = kelly_sizing(
        "Buy", wins=1, losses=49, return_raw=True
    )
    assert raw_floor == 0.0
    assert clipped_floor > raw_floor


# --- breaker: NaN sector value no longer fails open (P3-3) ---------------- #

def test_breaker_rejects_nan_sector_value():
    from yialpha.risk.breaker import DrawdownBreaker

    ok, reason = DrawdownBreaker().check_exposure(
        100.0, 1000.0, sector_value=float("nan")
    )
    assert ok is False
    assert "finite" in reason


# --- tradeability: absurd edges refused (P3-7) ----------------------------- #

def test_gross_edge_refuses_near_total_short_edge():
    from yialpha.risk.tradeability import TicketSide, gross_edge

    # A mis-parsed 0.001 SHORT target used to fabricate +9990 bps of edge.
    assert gross_edge(TicketSide.SHORT, 0.001, 1.0) is None
    # A >20x LONG moonshot is the same garbage on the other side.
    assert gross_edge(TicketSide.LONG, 50.0, 1.0) is None
    # Sane targets in both directions still compute.
    assert gross_edge(TicketSide.SHORT, 0.95, 1.0) == pytest.approx(0.05)
    assert gross_edge(TicketSide.LONG, 1.10, 1.0) == pytest.approx(0.10)


# --- corporate actions: strict YYYY-MM-DD (P3-8) --------------------------- #

def test_corporate_action_rejects_compact_date_form():
    from yialpha.instruments.corporate_actions import (
        CorporateActionRecord,
        validate_corporate_action,
    )

    record = CorporateActionRecord(
        symbol="MU", action_type="dividend", event_date="20260301"
    )
    # Python 3.11+ date.fromisoformat accepts 'YYYYMMDD'; the contract is
    # strict YYYY-MM-DD (lexicographic date comparisons downstream).
    with pytest.raises(ValueError):
        validate_corporate_action(record)


# --- atr_stop: too-few-completed-bars degrades ATR, keeps the close (P3-5) -- #

def test_atr_stop_short_frame_returns_close_with_no_atr():
    from yialpha.risk.atr_stop import atr_stop, latest_atr_from_frame

    frame = pd.DataFrame({
        "Date": ["2026-01-02"], "Open": [10.0], "High": [11.0],
        "Low": [9.5], "Close": [10.5], "Volume": [100.0],
    })
    # A 1-row frame used to return that bar's High-Low as "the 14-period ATR"
    # (the SMMA seed value); the round-8 review then made it raise — which
    # also threw the valid close away. Correct radius: close returned, ATR
    # honestly None; the stop COMPUTING wrapper still refuses.
    close, atr = latest_atr_from_frame(frame)
    assert close == 10.5
    assert atr is None
    with pytest.raises(ValueError, match="completed bars"):
        atr_stop(frame)


# ===========================================================================
# S6 shard — graph + ledger
# ===========================================================================

# --- checkpoint signature covers input-affecting flags (P2-1) ------------- #

def test_signature_config_keys_cover_input_affecting_flags():
    from yialpha.graph.trading_graph import _SIGNATURE_CONFIG_KEYS

    # Round-8: web_search_enabled / regime_context (and formerly a_stock,
    # removed with the A-share layer) change node INPUTS (tool bindings,
    # prompt context); flipping one across a resumed checkpoint used to
    # replay the old configuration's analyst reports.
    assert {"web_search_enabled", "regime_context"} <= set(
        _SIGNATURE_CONFIG_KEYS
    )


# --- memory_resolution: corrupted pending row skipped, not fatal (P3-2) --- #

def test_memory_resolution_skips_entries_without_ticker(monkeypatch):
    from yialpha.graph import memory_resolution as mr

    class _FakeLog:
        def __init__(self, entries):
            self._entries = entries
            self.updates = None

        def get_pending_entries(self):
            return list(self._entries)

        def batch_update_with_outcomes(self, updates):
            self.updates = list(updates)

    good = {"ticker": "AAPL", "date": "2026-01-05", "rating": "Buy", "asset": "stock"}
    corrupted = {"date": "2026-01-06", "rating": "Buy"}  # hand-edited: no ticker
    log = _FakeLog([corrupted, good])

    class _Reflector:
        def reflect_on_final_decision(self, *a, **k):
            return "reflection"

    monkeypatch.setattr(
        "yialpha.accuracy.fetch_returns_yf", lambda *a, **k: (0.05, 0.02, 5)
    )
    resolved = mr.resolve_pending_entries(
        log, reflector=_Reflector(),
        ticker="AAPL", as_of_date="2026-02-01",
    )
    assert resolved == 1  # the good row resolved, the corrupted row skipped


# ===========================================================================
# S7 shard — accuracy / CLI / scripts
# ===========================================================================

# --- _resolve_asset_type: unknown tag degrades via inference (P3-2) ------- #

def test_resolve_asset_type_unknown_tag_degrades_via_inference():
    from yialpha.accuracy import _resolve_asset_type

    # A hand-edited garbage tag used to return ("stock", "logged") — silently
    # pricing a perp row on the Yahoo spot leg with no inference caveat.
    assert _resolve_asset_type("BTCUSDT", "perp") == ("crypto_perp", "inferred")
    assert _resolve_asset_type("BTCUSDT", "crypto-perp") == (
        "crypto_perp", "inferred"
    )
    assert _resolve_asset_type("AAPL", "garbage") == ("stock", "inferred")


# --- A/B gate: overlap rescue is small-sample only (P3-3) ----------------- #

_ab_spec = importlib.util.spec_from_file_location(
    "_run_analyst_parallel_ab_pin",
    Path(__file__).resolve().parents[1] / "scripts" / "run_analyst_parallel_ab.py",
)
ab = importlib.util.module_from_spec(_ab_spec)
_ab_spec.loader.exec_module(ab)


def _gate_metrics(**over):
    m = {
        "n_serial": 20, "n_parallel": 20,
        "rating_p": 1.0, "overlap_ratio": 1.0,
        "per_field_similarity": {},
        "risk_determinism_serial": True, "risk_determinism_parallel": True,
        "empty_reports_serial": 0, "empty_reports_parallel": 0,
        "exceptions_serial": 0, "exceptions_parallel": 0,
        "speedup": 3.0, "speedup_na": False, "dry_run": True,
    }
    m.update(over)
    return m


def test_gate_catches_real_drift_despite_overlap():
    # 18x Buy/2x Hold vs 10x Buy/10x Hold at n=20/leg: chi-square rejects
    # (p~0.006) but overlap 0.60 used to rescue it — a 0.9->0.5 Buy-rate
    # drift passing the distribution-equivalence gate.
    ok, reasons = ab.summarize_gate(
        _gate_metrics(rating_p=0.006, overlap_ratio=0.60)
    )
    assert ok is False
    assert any("chi-square" in r for r in reasons)


def test_gate_overlap_rescue_still_works_for_small_samples():
    ok, reasons = ab.summarize_gate(
        _gate_metrics(rating_p=0.01, overlap_ratio=0.60, n_serial=3, n_parallel=3)
    )
    assert ok is True, reasons
