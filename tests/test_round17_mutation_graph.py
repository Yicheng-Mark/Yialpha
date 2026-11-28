"""Round 17 — mutation_graph.

One-line mutants of ``yialpha/graph/*`` and the risk money path that the WHOLE
suite let through (8069 tests passed with the mutant in place), each closed by
the test that kills it. A killing test is named
``test_r17_mut_<module>_<what>_is_pinned`` and its docstring names its mutant:
``file:line``, old -> new, and what the mutant does. A ``test_r17_<topic>_<behavior>``
test is the regression test for a product fix found while choosing a mutant
(failing on the unmodified code first).

Mutants were applied only in isolated copies of the tree, by the round's
``mutate.py`` helper; nothing here mutates the modules under test at runtime.
"""

from __future__ import annotations

import hashlib
from datetime import date, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from yialpha import accuracy
from yialpha.agents.utils import prediction_tools
from yialpha.dataflows import binance_brackets as brackets_module
from yialpha.dataflows.binance_brackets import Bracket
from yialpha.dataflows.config import set_config
from yialpha.dataflows.utils import get_analysis_date
from yialpha.graph import (
    analyst_fanout,
    memory_resolution,
    perf_telemetry,
    routing,
    trading_graph as tg,
)
from yialpha.graph.analyst_execution import (
    AnalystWallTimeTracker,
    build_analyst_execution_plan,
)
from yialpha.graph.analyst_fanout import create_analyst_fanout_node
from yialpha.graph.checkpointer import (
    clear_all_checkpoints,
    clear_checkpoint,
    get_checkpointer,
    thread_id,
)
from yialpha.graph.conditional_logic import ConditionalLogic
from yialpha.graph.memory_resolution import _warn_if_stale_pending
from yialpha.graph.overlay_fields import OVERLAY_MARKER, parse_overlay
from yialpha.graph.perf_telemetry import (
    NodePerfTokenCallback,
    NodePerfTracker,
    dump_perf_report,
    wrap_node,
    wrap_runnable_node,
)
from yialpha.graph.propagation import Propagator
from yialpha.graph.reflection import Reflector
from yialpha.graph.signal_processing import SignalProcessor
from yialpha.graph.trading_graph import YiAlphaGraph
from yialpha.ledger import portfolio as portfolio_module
from yialpha.ledger.evidence import register_run
from yialpha.ledger.portfolio import (
    open_positions,
)
from yialpha.ledger.run_context import (
    reset_ledger_run_context,
    set_ledger_run_context,
)
from yialpha.perp.quote_fx import QuoteFxResult
from yialpha.risk import constraints, manager as risk_manager, perp_ticket
from yialpha.risk.constraints import (
    PortfolioLimits,
    PortfolioSnapshot,
    PositionView,
    advisory_metrics,
    correlation_cluster,
    directional_concentration,
    global_gross,
)
from yialpha.risk.cost_model import estimate_round_trip_cost
from yialpha.risk.manager import (
    PortfolioState,
    RiskManager,
    atr_stop_from_values,
    build_backtest_weight_fn,
)
from yialpha.risk.perp_ticket import compute_leverage
from yialpha.risk.resolver import ResolverResult, render_resolver_lines
from yialpha.tickets import DEFAULT_HORIZON_DAYS


def _pv(
    weight: float,
    symbol: str = "BTCUSDT",
    cls: str = "pure_crypto_perp",
    side: str | None = None,
) -> PositionView:
    if side is None:
        side = "LONG" if weight >= 0 else "SHORT"
    return PositionView(symbol=symbol, instrument_class=cls, side=side, weight=weight)  # type: ignore[arg-type]


def _snap(
    *positions: PositionView,
    equity: float = 100_000.0,
    returns: tuple[float, ...] = (),
) -> PortfolioSnapshot:
    return PortfolioSnapshot(equity=equity, positions=positions, returns_history=returns)


# --------------------------------------------------------------------------- #
# yialpha/risk/constraints.py — the five frozen hard constraints
# --------------------------------------------------------------------------- #


def test_r17_mut_constraints_unmapped_symbols_cluster_alone_is_pinned():
    """Kills constraints.py:279  ``cluster_of.get(symbol, symbol)`` -> ``cluster_of.get(symbol)``.

    Without the default, every symbol absent from ``cluster_of`` maps to ``None``
    and lands in ONE cluster. Two unmapped symbols (0.3 held + 0.2 proposed =
    0.5 gross over the 0.4 cluster cap) would then resize to x0.5, where the
    documented behaviour — "symbols absent from the map cluster alone" — passes
    at 1.0 with a 0.2 post-cluster. The existing test covers a single unmapped
    symbol next to a mapped one. Survived the full suite (8069 passed).
    """
    limits = PortfolioLimits(cluster_of={"BTCUSDT": "majors"})
    snap = _snap(_pv(0.3, symbol="NVDAUSDT", cls="stock_perp"))
    d = correlation_cluster(_pv(0.2, symbol="TSLAUSDT", cls="stock_perp"), snap, limits)
    assert (d.action, d.multiplier, d.reasons) == ("PASS", 1.0, ["within_limit"])
    assert d.metrics == {"post_cluster": 0.2, "max_cluster": 0.4}


def test_r17_mut_constraints_correlation_diagonal_needs_two_observations_is_pinned():
    """Kills constraints.py:600  ``len(series[a]) >= 2`` -> ``len(series[a]) >= 1``.

    The diagonal is 1.0 "for any series with >=2 finite observations": a series
    with ONE finite observation has nothing to correlate and its row is ``{}``.
    Under the mutant it would report a self-correlation of 1.0. Survived the
    full suite (8069 passed).
    """
    matrix = constraints._correlation_matrix(
        {"A": [0.01], "B": [0.01, 0.02], "C": [0.02, 0.03]}
    )
    assert set(matrix) == {"A", "B", "C"}
    assert matrix["A"] == {}
    assert matrix["B"] == {"B": pytest.approx(1.0), "C": pytest.approx(1.0)}
    assert matrix["C"] == {"B": pytest.approx(1.0), "C": pytest.approx(1.0)}
    # Same rule through the public entry point; non-finite values do not count.
    out = advisory_metrics(
        _pv(0.0, side="FLAT"),
        _snap(),
        PortfolioLimits(),
        position_returns={"A": [0.01], "D": [0.01, float("nan"), float("inf")]},
    )
    assert out["correlation_matrix"] == {"A": {}, "D": {}}


def test_r17_mut_constraints_book_within_tolerance_of_limit_is_already_over_is_pinned():
    """Kills constraints.py:253  ``others >= limit - 1e-12`` -> ``others > limit - 1e-12``.

    The at-limit tolerance is inclusive: an old book 1e-12 UNDER the cap counts
    as at the limit, so a candidate with nothing to add is RESIZE x0.0 with the
    reason ``book_already_over_limit``. The strict ``>`` misses that one float
    and falls through to the clip path (RESIZE x1e-11, ``exceeds_limit``) — a
    different reason in the reproducible log and a non-zero multiplier.
    Survived the full suite (8069 passed).
    """
    limits = PortfolioLimits(max_gross=0.5 + 1e-12)
    assert limits.max_gross - 1e-12 == 0.5          # the guard is exactly this float
    assert limits.max_gross > 0.5                   # ... and the book really is under it
    snap = _snap(_pv(0.5, symbol="SOLUSDT"))
    d = global_gross(_pv(0.1, symbol="ETHUSDT"), snap, limits)
    assert (d.action, d.multiplier, d.reasons) == ("RESIZE", 0.0, ["book_already_over_limit"])


def test_r17_mut_constraints_at_tolerance_edge_passes_is_pinned():
    """Kills constraints.py:263  ``limit - others >= contribution - 1e-12`` -> ``>``.

    A candidate that overshoots the free headroom by exactly the 1e-12
    tolerance (0.5 + 1e-12 into 0.75 - 0.25) lands the book on the limit and
    PASSes at multiplier 1.0. The strict ``>`` sends it down the clip path as a
    RESIZE of x0.999999999998 — the at-limit trade labelled "resized by a
    binding constraint" that the comment above the line guards against.
    Survived the full suite (8069 passed).
    """
    limits = PortfolioLimits(max_gross=0.75)
    assert limits.max_gross - 0.25 == 0.5           # exact headroom, no float dust
    snap = _snap(_pv(0.25, symbol="SOLUSDT"))
    d = global_gross(_pv(0.5 + 1e-12, symbol="ETHUSDT"), snap, limits)
    assert (d.action, d.multiplier, d.reasons) == ("PASS", 1.0, ["within_limit"])
    assert d.metrics["max_gross"] == 0.75
    assert d.metrics["post_gross"] == pytest.approx(0.75 + 1e-12)


def test_r17_mut_constraints_opposite_exposure_at_tolerance_edge_passes_is_pinned():
    """Kills constraints.py:420  ``opp_exposure <= limits.max_directional + 1e-12`` -> ``<``.

    A long 0.1 on a book netting -0.9 leaves -0.8 short — exactly
    ``cap + 1e-12`` for ``cap = 0.8 - 1e-12`` — which the inclusive check
    accepts (PASS at 1.0: the candidate REDUCES the short exposure). The strict
    ``<`` reads it as a breach and, the candidate pointing along its own side,
    refuses the trade with ``opposite_direction_exceeds_limit``. Survived the
    full suite (8069 passed).
    """
    limits = PortfolioLimits(max_directional=0.8 - 1e-12)
    assert limits.max_directional + 1e-12 == 0.8    # the guard is exactly this float
    snap = _snap(_pv(-0.9, symbol="SOLUSDT", side="SHORT"))
    d = directional_concentration(_pv(0.1, symbol="ETHUSDT"), snap, limits)
    assert (d.action, d.multiplier, d.reasons) == ("PASS", 1.0, ["within_limit"])
    assert d.metrics == {"post_directional": 0.0, "max_directional": limits.max_directional}


def test_r17_constraints_beta_is_none_when_bench_length_mismatches():
    """``advisory_metrics`` promises ``beta_vs_btc`` is None on misaligned input.

    Since round 7 the finite filter is pairwise by DATE (a day counts only when
    both sides are finite), so ``_beta`` never sees two same-length series over
    different dates produced by NaNs. Two series of DIFFERENT lengths had no
    such guard: ``zip(..., strict=False)`` truncated them from the start and
    the head of each was paired — a beta, and the compound
    ``btc_crash_x_earnings_gap``, read off a date-shifted overlap. That is the
    fabrication ``_beta``'s own docstring refuses ("unaligned daily series
    would fabricate a beta"). Before the fix this test got beta 0.5 (and
    -10800 for the crash leg) instead of None. No production caller passes a
    bench series yet (the graph calls ``advisory_metrics(candidate, snapshot,
    limits)``), so nothing displayed changed.
    """
    snap = _snap(returns=(0.01, 0.02, 0.03))
    shorter = advisory_metrics(
        _pv(0.1, symbol="AAPLUSDT", cls="stock_perp"), snap, PortfolioLimits(),
        btc_returns=(0.02, 0.04),
    )
    longer = advisory_metrics(
        _pv(0.1, symbol="AAPLUSDT", cls="stock_perp"), snap, PortfolioLimits(),
        btc_returns=(0.02, 0.04, 0.06, 0.08),
    )
    for out in (shorter, longer):
        assert out["beta_vs_btc"] is None
        assert out["btc_crash_x_earnings_gap"] is None
    # Equal lengths are paired day for day as before: cov / var = 0.5.
    aligned = advisory_metrics(
        _pv(0.1, symbol="AAPLUSDT", cls="stock_perp"), snap, PortfolioLimits(),
        btc_returns=(0.02, 0.04, 0.06),
    )
    assert aligned["beta_vs_btc"] == pytest.approx(0.5)
    assert aligned["btc_crash_x_earnings_gap"] == pytest.approx(
        100_000.0 * (-0.20 * 0.5 - 0.08 * 0.1)
    )


# --------------------------------------------------------------------------- #
# yialpha/risk/resolver.py — min-multiplier composition
# --------------------------------------------------------------------------- #


def test_r17_mut_resolver_lines_end_with_newline_is_pinned():
    """Kills resolver.py:126  ``"".join(line + "\\n" for line in lines)`` -> ``"\\n".join(lines)``.

    Every bullet is newline-terminated, the last one included. The block is
    concatenated into a markdown section (``render_resolver_lines(result) +
    "\\n" + rule_lines``), and the existing tests search it for substrings
    without looking at how it ends.
    """
    result = ResolverResult(
        action="VETOED", final_multiplier=0.0, final_size=0.0, side="LONG",
        reasons=["non_positive_equity"], risk_decision_rules=["global_gross"],
    )
    assert render_resolver_lines(result) == (
        "- **Resolver**: VETOED · multiplier 0.00 · final size 0.0000\n"
        "- **Binding constraints**: global_gross\n"
        "- **Veto reasons**: non_positive_equity\n"
    )


# --------------------------------------------------------------------------- #
# yialpha/risk/manager.py — Kelly x breaker x CVaR x caps
# --------------------------------------------------------------------------- #


def _flat() -> PortfolioState:
    return PortfolioState(cash=100_000, equity=100_000)


def _backtest_fn():
    """A backtest weight function with deterministic price / ATR lookups."""
    return build_backtest_weight_fn(
        RiskManager(), "AAPL",
        price_lookup=lambda _date: 100.0, atr_lookup=lambda _date: 2.0,
    )


def test_r17_mut_manager_default_kelly_fraction_is_pinned():
    """Kills manager.py:169  ``config.get("kelly_fraction", 0.25)`` -> ``0.50``.

    ``from_config`` promises that missing keys fall back to the constructor
    defaults. The overlay's ``_build_risk_manager`` hands it the whole run
    config, so the default IS the live sizing knob: a config without
    ``kelly_fraction`` must size on quarter-Kelly.
    """
    assert RiskManager.from_config({}).kelly_fraction == RiskManager().kelly_fraction == 0.25


def test_r17_mut_manager_default_atr_mult_is_pinned():
    """Kills manager.py:177  ``config.get("atr_stop_mult", 2.0)`` -> ``1.0``.

    The default ATR stop multiple is 2.0 (the same default
    ``perp_ticket.ATR_STOP_MULT`` documents); a config without
    ``atr_stop_mult`` must keep it.
    """
    assert RiskManager.from_config({}).atr_mult == RiskManager().atr_mult == 2.0


def test_r17_mut_manager_warn_ceiling_is_pinned():
    """Kills manager.py:174  ``min(hard_stop / 3.0, 0.05)`` -> ``min(hard_stop / 3.0, 0.10)``.

    The breaker ladder scales with ``max_drawdown_hard_stop`` only up to its
    ceilings: under a 30% hard stop the warn threshold stays at 5% (a third
    of 0.30 would be 0.10). The existing test covers the default 15%, where
    the ceiling does not bind.
    """
    rm = RiskManager.from_config({"max_drawdown_hard_stop": 0.30})
    assert (rm.breaker.hard_stop_drawdown, rm.breaker.warn_drawdown) == (0.30, 0.05)


def test_r17_mut_manager_no_new_ceiling_is_pinned():
    """Kills manager.py:175  ``min(hard_stop * 2.0 / 3.0, 0.10)`` -> ``min(hard_stop * 2.0 / 3.0, 0.20)``.

    Same ceiling rule for the no-new-entries threshold: under a 30% hard
    stop it stays at 10% (two thirds of 0.30 would be 0.20).
    """
    rm = RiskManager.from_config({"max_drawdown_hard_stop": 0.30})
    assert (rm.breaker.hard_stop_drawdown, rm.breaker.no_new_drawdown) == (0.30, 0.10)


def test_r17_mut_manager_empty_ledger_passes_none_not_zero_is_pinned(monkeypatch):
    """Kills manager.py:226  ``wins=wins if wins or losses else None`` -> ``wins=wins``
    and manager.py:227  ``losses=losses if wins or losses else None`` -> ``losses=losses``.

    A ledger with no decided trades (empty, or flat trades only) is an
    UNKNOWN record: both counts go to ``kelly_sizing`` as ``None`` so the
    rating's own conviction sizes the trade. Passing a literal 0-0 record is
    the "10-0 scores Kelly 0" class of bug the comment above the line
    describes. Each mutant alone is masked by ``kelly_sizing`` requiring both
    counts, so the exact call arguments are what pins the contract.
    """
    seen: dict[str, object] = {}
    real = risk_manager.kelly_sizing

    def spy(rating, **kwargs):
        seen.update(kwargs)
        return real(rating, **kwargs)

    monkeypatch.setattr(risk_manager, "kelly_sizing", spy)
    state = PortfolioState(cash=1.0, equity=1.0, trade_history=[{"ticker": "X", "return": 0.0}])
    risk_manager.RiskManager().decide("AAPL", "Buy", state, price=100.0, atr=2.0)
    assert (seen["wins"], seen["losses"]) == (None, None)


def _no_new_existing():
    """A held AAPL position re-judged after a 12% drawdown (regime no_new)."""
    rm = RiskManager()
    state = PortfolioState(cash=0, equity=100_000, positions={"AAPL": 10_000})
    at_peak = rm.decide("AAPL", "Buy", state, price=100.0, atr=2.0)
    assert at_peak.breaker.regime == "normal"
    assert at_peak.target_weight == pytest.approx(0.08)
    state.equity = 88_000
    return at_peak, rm.decide("AAPL", "Buy", state, price=100.0, atr=2.0)


def test_r17_mut_manager_breaker_multiplier_scales_size_is_pinned():
    """Kills manager.py:255  ``kelly_raw * breaker_state.position_multiplier * cvar_mult``
    -> ``kelly_raw * cvar_mult``.

    In a ``no_new`` regime the breaker trims an existing position to half
    size (regime -> multiplier 0.5). Without the multiplier the manager would
    carry the full Kelly weight through a drawdown that is supposed to cut
    it; the existing breaker test only checks that a NEW entry is blocked.
    """
    at_peak, trimmed = _no_new_existing()
    assert trimmed.breaker.regime == "no_new"
    assert trimmed.breaker.position_multiplier == 0.5
    assert trimmed.target_weight == pytest.approx(at_peak.target_weight * 0.5)


def test_r17_mut_manager_existing_position_not_blocked_is_pinned():
    """Kills manager.py:279  ``rating in _BULLISH and is_new and not can_open_new``
    -> ``rating in _BULLISH and not can_open_new``.

    The breaker blocks NEW entries only. A bullish re-size of a position the
    book already holds is never refused — it is trimmed by the regime
    multiplier and the decision stays admissible (``add``), so the overlay
    does not report a held position as ``blocked``.
    """
    _at_peak, trimmed = _no_new_existing()
    assert (trimmed.action, trimmed.exposure_ok, trimmed.exposure_reason) == ("add", True, "ok")


def test_r17_mut_manager_single_cap_binds_is_pinned():
    """Kills manager.py:256  ``min(target_weight, self.breaker.max_single_position)``
    -> ``min(target_weight, self.breaker.max_single_sector)``.

    The per-ticker cap is ``max_single_position`` (here 1%), not the looser
    sector cap (30%): a 8% Kelly weight is clipped to 1%.
    """
    d = RiskManager(max_single_position=0.01).decide(
        "AAPL", "Buy", _flat(), price=100.0, atr=2.0
    )
    assert d.kelly_raw == pytest.approx(0.08)
    assert d.target_weight == 0.01


def test_r17_mut_manager_funding_warn_threshold_inclusive_is_pinned():
    """Kills manager.py:267  ``funding_rate_annualized >= self.funding_warn_annual``
    -> ``... > self.funding_warn_annual``.

    The funding gate is inclusive: carry of exactly 30% a year against the
    position already halves the weight (``>= warn threshold halves it``).
    """
    d = RiskManager().decide(
        "AAPL", "Buy", _flat(), price=100.0, atr=2.0, funding_rate_annualized=0.30
    )
    assert d.target_weight == pytest.approx(d.kelly_raw * 0.5)


def test_r17_mut_manager_funding_hard_threshold_inclusive_is_pinned():
    """Kills manager.py:270  ``>= self.funding_hard_annual`` -> ``> self.funding_hard_annual``.

    Carry of exactly 60% a year quarters the weight; the hard threshold is
    inclusive like the warn one.
    """
    d = RiskManager().decide(
        "AAPL", "Buy", _flat(), price=100.0, atr=2.0, funding_rate_annualized=0.60
    )
    assert d.target_weight == pytest.approx(d.kelly_raw * 0.25)


def test_r17_mut_manager_infinite_equity_fails_sector_cap_is_pinned():
    """Kills manager.py:288  ``0.0 < state.equity < float("inf")`` -> ``0.0 < state.equity``.

    The sector cap refuses every invalid equity tick, infinite included
    (``0 < inf`` is true but ``sectors / inf`` is 0 — "no exposure"). Such a
    tick blocks like a full sector instead of letting the cap pass.
    """
    rm = RiskManager()
    state = PortfolioState(
        cash=0, equity=100_000, positions={"AAPL": 1_000}, sectors={"tech": 10_000}
    )
    rm.decide("AAPL", "Buy", state, price=100.0, atr=2.0)      # a valid regime first
    state.equity = float("inf")
    d = rm.decide("AAPL", "Buy", state, price=100.0, atr=2.0, sector="tech")
    assert (d.action, d.target_weight, d.exposure_ok) == ("blocked", 0.0, False)
    assert d.exposure_reason == "invalid equity (inf) for sector cap on tech"


def test_r17_mut_manager_sector_cap_exclusive_is_pinned():
    """Kills manager.py:311  ``sector_weight + target_weight > max_single_sector``
    -> ``sector_weight + target_weight >= max_single_sector``.

    A trade that lands the sector exactly ON its cap fits: it is not
    trimmed and the decision stays ``ok``. (Dyadic numbers — 0.1875 + 0.0625
    = 0.25 — so the sum is exact.) Under the mutant the same trade is
    relabelled ``trimmed to sector cap`` with zero room left.
    """
    rm = RiskManager(max_single_position=0.0625, max_single_sector=0.25)
    state = PortfolioState(
        cash=0, equity=100_000, positions={"AAPL": 0.0}, sectors={"tech": 18_750}
    )
    d = rm.decide("AAPL", "Buy", state, price=100.0, atr=2.0, sector="tech")
    assert (d.exposure_reason, d.target_weight) == ("ok", 0.0625)


def test_r17_mut_manager_negative_equity_zero_position_value_is_pinned():
    """Kills manager.py:342  ``... if state.equity > 0 else 0.0`` -> the guard dropped.

    With a negative equity tick no position VALUE can be derived: the weight
    is still reported (the breaker replays the last valid regime) but the
    dollar amount is 0.0, never negative.
    """
    rm = RiskManager()
    state = PortfolioState(cash=0, equity=100_000, positions={"AAPL": 1_000})
    rm.decide("AAPL", "Buy", state, price=100.0, atr=2.0)
    state.equity = -5.0
    d = rm.decide("AAPL", "Buy", state, price=100.0, atr=2.0)
    assert d.target_weight == pytest.approx(0.08)
    assert d.position_value == 0.0


def test_r17_mut_manager_zero_close_or_atr_rejected_is_pinned():
    """Kills manager.py:385  ``close <= 0 or atr <= 0`` -> ``close < 0 or atr < 0``.

    ``atr_stop_from_values`` "raises ValueError on non-positive inputs": a
    zero close or a zero ATR is non-positive. A zero ATR would otherwise
    arm a stop exactly AT the entry.
    """
    with pytest.raises(ValueError, match="close and atr must be positive"):
        atr_stop_from_values(0.0, 1.0, 2.0)
    with pytest.raises(ValueError, match="close and atr must be positive"):
        atr_stop_from_values(100.0, 0.0, 2.0)


def test_r17_mut_manager_backtest_uses_latest_equity_is_pinned():
    """Kills manager.py:414  ``float(equity_series[-1])`` -> ``float(equity_series[0])``.

    The backtest adapter marks the breaker with the LATEST equity of the
    series the engine hands it. Reading the oldest value instead would keep
    the breaker at the start-of-window equity: after 100k -> 88k -> 76k the
    true regime is hard_stop (24% down) and a new Buy is blocked, where the
    mutant still sees 100k and enters.
    """
    fn = _backtest_fn()
    fn("Buy", "2026-01-05", {"equity_history": [100_000], "current_position_value": 0.0})
    fn("Buy", "2026-01-06", {"equity_history": [100_000, 88_000], "current_position_value": 8_000})
    ctx = {"equity_history": [100_000, 88_000, 76_000], "current_position_value": 0.0}
    assert fn("Buy", "2026-01-07", ctx) == 0.0
    assert ctx["risk_decision"].action == "blocked"


def test_r17_mut_manager_realized_returns_folded_once_is_pinned():
    """Kills manager.py:419  ``realized[consumed_realized:]`` -> ``realized``.

    The engine passes the CUMULATIVE realized-return list; the adapter folds
    only the entries it has not seen yet. Re-folding the whole list turns a
    2-0 record into 3-0 (and grows on every rebalance), inflating the Kelly
    win-rate input: the pre-clip Kelly for 2-0 at +10% each is 0.0875, for
    3-0 it is 0.11.
    """
    fn = _backtest_fn()
    first = {"equity_history": [100_000], "current_position_value": 0.0, "realized_returns": [0.10]}
    fn("Buy", "2026-01-05", first)
    assert first["risk_decision"].kelly_preclip == pytest.approx(0.05)
    second = {"equity_history": [100_000], "current_position_value": 0.0,
              "realized_returns": [0.10, 0.10]}
    fn("Buy", "2026-01-06", second)
    assert second["risk_decision"].kelly_preclip == pytest.approx(0.0875)


def test_r17_mut_manager_hard_stop_flattens_hold_is_pinned():
    """Kills manager.py:452  ``rating == "Hold" and regime != "hard_stop"`` -> ``rating == "Hold"``.

    A literal Hold normally carries the current exposure (returns ``None``,
    no rebalance) — EXCEPT under a hard stop, which flattens: the adapter
    returns 0.0. Dropping the hard-stop clause would keep a position open
    through the breaker's flatten regime.
    """
    fn = _backtest_fn()
    fn("Buy", "2026-01-05", {"equity_history": [100_000], "current_position_value": 0.0})
    fn("Buy", "2026-01-06", {"equity_history": [100_000, 79_000], "current_position_value": 8_000})
    ctx = {"equity_history": [100_000, 79_000, 79_000], "current_position_value": 8_000}
    assert fn("Hold", "2026-01-07", ctx) == 0.0
    assert ctx["risk_decision"].breaker.regime == "hard_stop"


def test_r17_mut_manager_backtest_sees_current_position_is_pinned():
    """Kills manager.py:416  ``float(ctx.get("current_position_value") or 0.0)`` -> ``0.0``.

    The adapter sizes against the exposure the engine REPORTS. Ignoring it
    makes every decision a new entry: after a 12% drawdown a bullish re-size
    of a held position must still go through as ``add`` at half size (0.04),
    not be blocked as a new entry.
    """
    fn = _backtest_fn()
    assert fn("Hold", "2026-01-05", {"equity_history": [100_000], "current_position_value": 0.0}) is None
    ctx = {"equity_history": [100_000, 88_000], "current_position_value": 8_000}
    assert fn("Buy", "2026-01-06", ctx) == pytest.approx(0.04)
    assert ctx["risk_decision"].action == "add"


def test_r17_mut_manager_rationale_names_exposure_reason_is_pinned():
    """Kills manager.py:376  ``if exposure_reason != "ok":`` -> ``if False:``.

    A trimmed decision explains itself: the rendered rationale (it goes into
    the overlay's ``**Rationale**`` bullet) ends with the exposure reason.
    """
    state = PortfolioState(
        cash=0, equity=100_000, positions={"AAPL": 0.0}, sectors={"tech": 29_000}
    )
    d = RiskManager().decide("AAPL", "Buy", state, price=100.0, atr=2.0, sector="tech")
    assert d.rationale == (
        "Rating Buy -> action enter. Kelly raw 0.080 x breaker 1.00 (normal) x CVaR 1.00 "
        "= target weight 0.010. Exposure: trimmed to sector cap for tech."
    )


def test_r17_mut_manager_rationale_names_drawdown_regime_is_pinned():
    """Kills manager.py:378  ``if breaker.regime != "normal":`` -> ``if False:``.

    Outside the normal regime the rationale states the drawdown and the
    regime, so a half-sized position in the overlay is explained instead of
    looking like a sizing error.
    """
    _at_peak, trimmed = _no_new_existing()
    assert trimmed.rationale == (
        "Rating Buy -> action add. Kelly raw 0.080 x breaker 0.50 (no_new) x CVaR 1.00 "
        "= target weight 0.040. Drawdown -12.0%; regime no_new."
    )


# --------------------------------------------------------------------------- #
# yialpha/graph/propagation.py — initial state
# --------------------------------------------------------------------------- #


def test_r17_mut_propagation_seeds_ticker_message_is_pinned():
    """Kills propagation.py:57  ``"messages": [("human", company_name)]`` -> ``"messages": []``.

    The first analyst starts from the initial human message carrying the
    ticker (the analyst fan-out hands ``specs[0]`` the parent's messages
    verbatim for exactly this reason). The existing completeness tests only
    look at the scalar keys.
    """
    state = Propagator().create_initial_state("AAPL", "2026-01-15")
    assert state["messages"] == [("human", "AAPL")]


def test_r17_mut_propagation_seeds_instrument_context_is_pinned():
    """Kills propagation.py:60  ``"instrument_context": instrument_context`` -> ``""``.

    The resolved ticker identity reaches every agent through the initial
    state; an empty string is the "fall back to ticker-only context" signal,
    so dropping it would silently degrade every analyst to ticker-only.
    """
    context = "Ticker: MUUSDT. Company: Micron Technology."
    state = Propagator().create_initial_state("MUUSDT", "2026-01-15", instrument_context=context)
    assert state["instrument_context"] == context


def test_r17_mut_propagation_seeds_portfolio_state_is_pinned():
    """Kills propagation.py:63  ``"portfolio_state": portfolio_state`` -> ``None``.

    The live book snapshot the caller injects is carried on the state for the
    Portfolio Manager; ``None`` is the documented "no snapshot" value, so
    dropping the object would make every run size without the book.
    """
    book = PortfolioState(cash=1_000.0, equity=5_000.0, positions={"AAPL": 100.0})
    state = Propagator().create_initial_state("AAPL", "2026-01-15", portfolio_state=book)
    assert state["portfolio_state"] is book


def test_r17_mut_propagation_seeds_pm_decision_fields_dict_is_pinned():
    """Kills propagation.py:109  ``"pm_decision_fields": {}`` -> ``None``.

    "Empty until the PM node fills it; {} on free-text fallback": the key is
    seeded as an empty dict. Every reader does ``state.get("pm_decision_fields")
    or {}``, which tolerates None — so the shape is pinned here, where it is
    defined.
    """
    state = Propagator().create_initial_state("AAPL", "2026-01-15")
    assert state["pm_decision_fields"] == {}


def test_r17_mut_propagation_seeds_investment_judge_decision_is_pinned():
    """Kills propagation.py:70  ``"judge_decision": ""`` -> key removed.

    The investment debate sub-state is seeded with every key a node reads by
    direct indexing (the module's checkpoint-resume argument). The existing
    test reads ``count`` and ``current_response`` only.
    """
    debate = Propagator().create_initial_state("AAPL", "2026-01-15")["investment_debate_state"]
    assert dict(debate) == {
        "bull_history": "", "bear_history": "", "history": "", "current_response": "",
        "judge_decision": "", "count": 0,
    }


# --------------------------------------------------------------------------- #
# yialpha/graph/reflection.py — the deferred-reflection prompt
# --------------------------------------------------------------------------- #


class _RecordingLLM:
    """Duck-typed chat model that records the messages it is invoked with."""

    def __init__(self) -> None:
        self.calls: list[list[tuple[str, str]]] = []

    def invoke(self, messages):
        self.calls.append(messages)
        return SimpleNamespace(content="lesson")


def _reflection(**kwargs) -> list[tuple[str, str]]:
    llm = _RecordingLLM()
    out = Reflector(llm).reflect_on_final_decision(
        final_decision="**Rating**: Buy\n\nMomentum intact.", **kwargs
    )
    assert out == "lesson"
    assert len(llm.calls) == 1
    return llm.calls[0]


def test_r17_mut_reflection_system_prompt_is_sent_is_pinned():
    """Kills reflection.py:47  ``("system", self.log_reflection_prompt)`` -> ``("system", "")``.

    The reflection prompt is what keeps the entry to 2-4 sentences of plain
    prose (it is re-injected into future agent prompts). An empty system
    turn would hand the model a bare "here are some numbers" request.
    """
    messages = _reflection(raw_return=-0.08, alpha_return=-0.05)
    assert messages[0] == ("system", Reflector(_RecordingLLM()).log_reflection_prompt)
    assert messages[0][1].startswith(
        "You are a trading analyst reviewing your own past decision now that the "
        "outcome is known.\n"
    )
    assert "Write exactly 2-4 sentences of plain prose" in messages[0][1]


def test_r17_mut_reflection_positive_returns_keep_sign_is_pinned():
    """Kills reflection.py:51  ``{raw_return:+.1%}`` -> ``{raw_return:.1%}`` and
    reflection.py:52  ``{alpha_return:+.1%}`` -> ``{alpha_return:.1%}``.

    Returns are always signed: "+5.0%" / "+2.5%" for gains, "-8.0%" for
    losses. The existing test only uses negative figures, where both formats
    agree.
    """
    human = _reflection(raw_return=0.05, alpha_return=0.025)[1][1]
    assert human.startswith("Raw return: +5.0%\nAlpha vs SPY: +2.5%\n\nFinal Decision:\n")


# --------------------------------------------------------------------------- #
# yialpha/graph/overlay_fields.py — machine-readable view of the overlay
# --------------------------------------------------------------------------- #


def _overlay(*bullets: str) -> str:
    return OVERLAY_MARKER + "\n" + "".join(f"- {bullet}\n" for bullet in bullets)


def test_r17_mut_overlay_negative_target_weight_is_parsed_is_pinned():
    """Kills overlay_fields.py:41  ``(-?[0-9.]+%)`` -> ``([0-9.]+%)``.

    ``target_weight`` is a SIGNED percentage in the field map (the position
    value next to it is signed too). The risk manager clamps its own weight
    at >= 0 today, so the renderer cannot emit a negative weight yet — this
    pins the parser's documented pattern for the day it can.
    """
    out = parse_overlay(_overlay("**Target Weight**: -12.5% of equity (-1,250)"))
    assert (out["target_weight"], out["position_value"]) == ("-12.5%", "-1,250")


# --------------------------------------------------------------------------- #
# yialpha/graph/memory_resolution.py — pending reflection sweep
# --------------------------------------------------------------------------- #

_ENTRY = {"ticker": "AAPL", "date": "2026-01-05", "decision": "d1"}


class _PendingLog:
    """Minimal memory log: hands out pending entries, records the batch write."""

    def __init__(self, entries, write_error: Exception | None = None) -> None:
        self._entries = entries
        self._write_error = write_error
        self.updates = None

    def get_pending_entries(self):
        return list(self._entries)

    def batch_update_with_outcomes(self, updates):
        if self._write_error is not None:
            raise self._write_error
        self.updates = updates


class _Reflector:
    """Duck-typed reflector; can be told to fail on specific decisions."""

    def __init__(self, fail_on: frozenset[str] = frozenset(), error: Exception | None = None) -> None:
        self.fail_on = fail_on
        self.error = error if error is not None else RuntimeError("429 rate limited")
        self.calls: list[dict] = []

    def reflect_on_final_decision(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs["final_decision"] in self.fail_on:
            raise self.error
        return f"lesson for {kwargs['final_decision']}"


@pytest.fixture
def sweep(monkeypatch):
    """Run one pending-entry sweep with a recorded price fetch.

    ``fetch`` is the tuple ``fetch_returns_yf`` answers (or a callable). Returns
    ``(log, reflector, fetch_calls, resolved)``; each fetch call is
    ``(ticker, trade_date, kwargs)``.
    """

    def run(entries, fetch=(0.05, 0.01, 5), reflector=None, log=None, ticker="AAPL", **kwargs):
        calls: list[tuple[str, str, dict]] = []

        def fake_fetch(ticker_, trade_date, **kw):
            calls.append((ticker_, trade_date, kw))
            return fetch(ticker_, trade_date, kw) if callable(fetch) else fetch

        monkeypatch.setattr(accuracy, "fetch_returns_yf", fake_fetch)
        log = log if log is not None else _PendingLog(entries)
        reflector = reflector if reflector is not None else _Reflector()
        resolved = memory_resolution.resolve_pending_entries(log, reflector, ticker, **kwargs)
        return log, reflector, calls, resolved

    return run


def test_r17_mut_memory_stale_warning_after_seven_days_is_pinned(caplog):
    """Kills memory_resolution.py:48  ``age_days > STALE_PENDING_DAYS`` -> ``>=``.

    A pending entry is only worth a WARNING once it is OLDER than seven days:
    at exactly seven days it is still inside the normal price-lag window.
    (The existing test warns at 22 days, where both comparisons agree.)
    """
    cutoff = datetime(2026, 11, 23, 12, 0, 0)
    logger_name = "yialpha.graph.memory_resolution"
    with caplog.at_level("WARNING", logger=logger_name):
        _warn_if_stale_pending("AAPL", "2026-11-16", cutoff)       # exactly 7 days
    assert caplog.records == []
    with caplog.at_level("WARNING", logger=logger_name):
        _warn_if_stale_pending("AAPL", "2026-11-15", cutoff)       # 8 days
    assert [r.getMessage() for r in caplog.records] == [
        "Pending memory-log entry for AAPL @ 2026-11-15 is 8 days old and still "
        "unresolved (price data unavailable). It may be stuck due to delisting "
        "or a permanent data gap."
    ]


def test_r17_mut_memory_row_without_ticker_is_skipped_loudly_is_pinned(sweep, caplog):
    """Kills memory_resolution.py:96  ``if entry_ticker is None:`` -> ``if False:``.

    A hand-corrupted pending row with no ticker is skipped LOUDLY — the
    round-8 fix exists so the bad row is visible instead of silently
    dropped (and instead of the KeyError that used to kill every later run
    of the ticker). No price fetch and no reflection may happen for it.
    """
    with caplog.at_level("WARNING", logger="yialpha.graph.memory_resolution"):
        log, reflector, calls, resolved = sweep(
            [{"date": "2026-01-05", "decision": "orphan"}], as_of_date="2026-01-20"
        )
    assert (resolved, calls, reflector.calls, log.updates) == (0, [], [], None)
    warnings = [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]
    assert len(warnings) == 1
    assert "memory log: pending entry lacks a ticker; skipping" in warnings[0]
    assert "'orphan'" in warnings[0]


def test_r17_mut_memory_benchmark_is_threaded_is_pinned(sweep):
    """Kills memory_resolution.py:125  ``benchmark=benchmark`` -> ``benchmark="SPY"``.

    The alpha the reflection reads is measured against the caller's
    benchmark (the graph passes ``_resolve_benchmark(ticker)``: BTCUSDT for
    pure crypto, a regional index where configured).
    """
    _log, _reflector, calls, resolved = sweep([_ENTRY], as_of_date="2026-01-20", benchmark="^N225")
    assert resolved == 1
    assert calls[0][2]["benchmark"] == "^N225"


def test_r17_mut_memory_holding_period_is_threaded_is_pinned(sweep):
    """Kills memory_resolution.py:126  ``holding_days=holding_days`` -> ``holding_days=5``.

    The horizon the sweep prices is the caller's ``holding_days`` (the
    graph passes the run's own horizon), not a hard-coded five sessions.
    """
    _log, _reflector, calls, _resolved = sweep([_ENTRY], as_of_date="2026-01-20", holding_days=3)
    assert calls[0][2]["holding_days"] == 3


def test_r17_mut_memory_reflection_leg_fails_soft_on_any_error_is_pinned(sweep, caplog):
    """Kills memory_resolution.py:148  ``except Exception as exc:`` -> ``except RuntimeError as exc:``.

    The reflection leg must not be the one hard failure in the loop: ANY
    exception from the LLM call (a ``ValueError`` from a malformed response,
    an SDK error class that is not a ``RuntimeError``) leaves the entry
    pending instead of killing the whole analysis run before any node
    executed.
    """
    reflector = _Reflector(fail_on=frozenset({"d1"}), error=ValueError("malformed response"))
    with caplog.at_level("WARNING", logger="yialpha.graph.memory_resolution"):
        log, _reflector, _calls, resolved = sweep(
            [_ENTRY], reflector=reflector, as_of_date="2026-01-20"
        )
    assert (resolved, log.updates) == (0, None)
    assert any("reflection failed for AAPL 2026-01-05 (left pending)" in r.getMessage()
               for r in caplog.records)


def test_r17_mut_memory_update_carries_realised_holding_days_is_pinned(sweep):
    """Kills memory_resolution.py:159  ``"holding_days": days`` -> ``"holding_days": holding_days``.

    ``fetch_returns_yf`` reports the horizon it could actually price (fewer
    sessions when the window is cut short); that realised count — not the
    requested one — is written to the entry (its ``Nd`` label).
    """
    log, _reflector, _calls, _resolved = sweep(
        [_ENTRY], fetch=(0.05, 0.01, 4), as_of_date="2026-01-20", holding_days=5
    )
    assert log.updates[0]["holding_days"] == 4


def test_r17_mut_memory_update_keeps_entry_venue_is_pinned(sweep):
    """Kills memory_resolution.py:164  ``"asset_type": entry.get("asset")`` -> ``"asset_type": asset_type``.

    Same-day perp AND spot entries of one ticker are both legal; the outcome
    must attach to the venue entry it was priced for (the entry's own
    ``asset=`` tag), not to the venue of the run that happens to sweep it.
    A legacy untagged entry stays untagged.
    """
    entries = [
        {"ticker": "BTCUSDT", "date": "2026-01-05", "decision": "perp", "asset": "crypto_perp"},
        {"ticker": "BTCUSDT", "date": "2026-01-06", "decision": "legacy"},
    ]
    log, _reflector, _calls, _resolved = sweep(
        entries, ticker="BTCUSDT", as_of_date="2026-01-20", asset_type="crypto_spot"
    )
    assert [(u["trade_date"], u["asset_type"]) for u in log.updates] == [
        ("2026-01-05", "crypto_perp"), ("2026-01-06", None)
    ]


def test_r17_mut_memory_write_leg_fails_soft_on_any_error_is_pinned(sweep, caplog):
    """Kills memory_resolution.py:178  ``except Exception as exc:`` -> ``except OSError as exc:``.

    The final durable write is fail-soft for ANY failure (a lock helper's
    ``RuntimeError``, a serialisation ``TypeError``), not only ``OSError``:
    the entries stay pending and the analysis run proceeds.
    """
    log = _PendingLog([_ENTRY], write_error=RuntimeError("ledger busy"))
    with caplog.at_level("WARNING", logger="yialpha.graph.memory_resolution"):
        _log, _reflector, _calls, resolved = sweep([], log=log, as_of_date="2026-01-20")
    assert resolved == 0
    assert any("memory outcome write failed for AAPL (1 entries left pending" in r.getMessage()
               for r in caplog.records)


# --------------------------------------------------------------------------- #
# yialpha/risk/perp_ticket.py — leverage caps and the advisory bracket cap
# --------------------------------------------------------------------------- #


def test_r17_mut_perp_ticket_unknown_asset_ceiling_is_pinned():
    """Kills perp_ticket.py:119  ``HARD_CEILING.get(asset_type, 5.0)`` -> ``..., 20.0)``.

    An asset type outside the ceiling table gets the 5x default, and it is
    the BINDING cap here (liquidation distance 10x, volatility 20x,
    conviction 10x). (No caller passes an unlisted type today — the overlay
    and the ticket script only use the table's keys.)
    """
    lev, caps = compute_leverage(0.05, 0.01, "knockout", 2, profile="moderate")
    assert caps == {"L_liq": 10.0, "L_vol": 20.0, "L_conv": 10.0, "L_hard": 5.0}
    assert lev == 5.0


def test_r17_mut_perp_ticket_unknown_profile_cap_is_pinned():
    """Kills perp_ticket.py:123  ``CONV_CAP.get(abs(strength), {}).get(profile, 5.0)``
    -> ``..., 20.0)``.

    A risk profile that is not in the table (a config typo in
    ``risk_profile``) is sized under the conservative profile — the documented
    default (round-17 follow-up; it used to fall back to the flat 5x cap,
    which for a weak rating was LOOSER than conservative's 3x). For a strong
    rating that is the same 5x; with a 25x liquidation-distance and a 15x
    volatility cap, the conviction cap binds.
    """
    lev, caps = compute_leverage(0.02, 0.02, "crypto_perp", 2, profile="yolo")
    assert caps["L_conv"] == 5.0
    assert lev == 5.0


def test_r17_mut_perp_ticket_bracket_cap_uses_real_notional_is_pinned(monkeypatch):
    """Kills perp_ticket.py:302  ``ASSUMED_NOTIONAL_USDT if notional is None else notional``
    -> ``ASSUMED_NOTIONAL_USDT``.

    ``advisory_leverage_cap(symbol, notional)`` reads the exchange tier of the
    REAL position notional when the caller knows one. A 200k USDT position
    sits in the 25x tier; only a caller without a notional gets the assumed
    50k tier (75x).
    """
    ladder = [Bracket(0.0, 0.004, 75.0), Bracket(100_000.0, 0.005, 25.0)]
    monkeypatch.setattr(brackets_module, "get_leverage_brackets", lambda _symbol: ladder)
    assert perp_ticket.advisory_leverage_cap("BTCUSDT", notional=200_000.0) == 25.0
    assert perp_ticket.advisory_leverage_cap("BTCUSDT") == 75.0


# --------------------------------------------------------------------------- #
# yialpha/graph/checkpointer.py — per-ticker checkpoint databases
# --------------------------------------------------------------------------- #


def test_r17_mut_checkpointer_thread_id_is_sha256_of_upper_ticker_is_pinned():
    """Kills checkpointer.py:48  ``hashlib.sha256(base.encode())`` -> ``... .encode("utf-16")``.

    The thread id is "the first 32 hex chars of a SHA-256" of
    ``TICKER:date[:signature]`` with the ticker upper-cased, UTF-8. Existing
    checkpoint databases on disk are keyed by exactly this value: another
    encoding (or a lower-case ticker) would orphan every resumable run.
    """
    expected = hashlib.sha256(b"AAPL:2026-01-15").hexdigest()[:32]
    assert thread_id("AAPL", "2026-01-15") == expected
    assert thread_id("aapl", "2026-01-15") == expected
    signed = hashlib.sha256(b"AAPL:2026-01-15:analysts=market").hexdigest()[:32]
    assert thread_id("AAPL", "2026-01-15", "analysts=market") == signed


def _checkpoint_dir(tmp_path) -> Path:
    return tmp_path / "checkpoints"


def test_r17_mut_checkpointer_saver_connection_is_closed_is_pinned(tmp_path):
    """Kills checkpointer.py:61  ``conn.close()`` -> ``pass`` (``get_checkpointer``).

    The context manager closes its SQLite connection on exit. A live handle
    denies the unlink on Windows (the round-13 sweep failure), so a leaked
    connection shows up as a checkpoint DB the sweep cannot remove.
    """
    with get_checkpointer(tmp_path, "AAPL"):
        pass
    assert (_checkpoint_dir(tmp_path) / "AAPL.db").exists()
    assert clear_all_checkpoints(tmp_path) == 1
    assert list(_checkpoint_dir(tmp_path).iterdir()) == []


def test_r17_mut_checkpointer_clear_closes_its_connection_is_pinned(tmp_path):
    """Kills checkpointer.py:156  ``conn.close()`` -> ``pass`` (``clear_checkpoint``).

    The post-run clear opens its own connection; it must close it too, or the
    DB it just cleaned cannot be swept (Windows) and the handle leaks once
    per completed run.
    """
    with get_checkpointer(tmp_path, "AAPL"):
        pass
    clear_checkpoint(tmp_path, "AAPL", "2026-01-15")
    assert clear_all_checkpoints(tmp_path) == 1
    assert list(_checkpoint_dir(tmp_path).iterdir()) == []


def test_r17_mut_checkpointer_clearing_missing_checkpoint_leaves_no_db_is_pinned(tmp_path):
    """Kills checkpointer.py:134  ``if not db.exists(): return`` -> ``if False:``.

    Clearing the checkpoint of a ticker that never had one is a no-op: no
    database file is created as a side effect (every fresh ticker's first
    successful run would otherwise start by creating an empty DB).
    """
    clear_checkpoint(tmp_path, "AAPL", "2026-01-15")
    assert _checkpoint_dir(tmp_path).is_dir()
    assert list(_checkpoint_dir(tmp_path).iterdir()) == []


def test_r17_mut_checkpointer_sweep_removes_both_sidecars_is_pinned(tmp_path):
    """Kills checkpointer.py:116  ``("-wal", "-shm")`` -> ``("-wal",)``.

    WAL mode leaves ``-wal`` and ``-shm`` sidecars next to a database when
    another process held the connection at unlink time; the sweep deletes
    both alongside their DB so they do not accumulate as garbage.
    """
    cp_dir = _checkpoint_dir(tmp_path)
    cp_dir.mkdir()
    for name in ("AAPL.db", "AAPL.db-wal", "AAPL.db-shm"):
        (cp_dir / name).write_bytes(b"stale")
    assert clear_all_checkpoints(tmp_path) == 1
    assert list(cp_dir.iterdir()) == []


# --------------------------------------------------------------------------- #
# yialpha/graph/trading_graph.py — risk overlay, perp ticket, portfolio control
# --------------------------------------------------------------------------- #

_STOCK_DATE = "2024-01-15"
_PERP_DATE = "2026-09-03"
_RUN_ID = "R17MUTGRAPH0001"


def _shell(monkeypatch, tmp_path=None, close=(190.0, 3.0), **config_over) -> YiAlphaGraph:
    """Graph shell with just enough state for the overlay (no LLMs, no network)."""
    g = YiAlphaGraph.__new__(YiAlphaGraph)
    config = {
        "risk_enabled": True, "kelly_fraction": 0.25, "max_single_position": 0.20,
        "max_single_sector": 0.30, "max_drawdown_hard_stop": 0.15, "atr_stop_mult": 2.0,
    }
    if tmp_path is not None:
        config["results_dir"] = str(tmp_path)
    g.config = {**config, **config_over}
    g._risk_overlay_degraded = False
    g.risk_manager = g._build_risk_manager()
    g.signal_processor = SignalProcessor(None)
    g.ticker = "AAPL"
    monkeypatch.setattr(g, "_latest_close_and_atr", lambda t, d, at="stock": close)
    return g


def _pm_state(**over) -> dict:
    base = {
        "final_trade_decision": "**Rating**: Buy\n\nThesis.",
        "pm_rating": "Buy",
        "pm_decision_fields": {"price_target": 220.0, "confidence": 0.7},
    }
    base.update(over)
    return base


def _stub_perp(g, monkeypatch, funding=None, mark=None) -> None:
    monkeypatch.setattr(g, "_trailing_funding_total", lambda t, d: funding)
    monkeypatch.setattr(g, "_latest_mark_close", lambda t, d: mark)
    monkeypatch.setattr(g, "_render_stress_line", lambda t, d: "")


def test_r17_mut_graph_overlay_sees_existing_position_is_pinned(monkeypatch, tmp_path):
    """Kills trading_graph.py:770  ``positions=dict(state.get("positions", {}) or {})`` -> ``positions={}``.

    The book's open positions decide whether a bullish decision ENTERS a new
    ticker or ADDS to one already held (and whether the drawdown breaker may
    block it). With the positions dropped every bullish decision is a new
    entry.
    """
    g = _shell(monkeypatch, tmp_path)
    book = {"equity": 100_000, "positions": {"AAPL": 5_000}}
    out = g._apply_risk_overlay("AAPL", _STOCK_DATE, _pm_state(), book)
    assert "- **Action**: add" in out["final_trade_decision"]


def test_r17_mut_graph_overlay_sees_returns_history_is_pinned(monkeypatch, tmp_path):
    """Kills trading_graph.py:772  ``returns_history=list(state.get("returns_history", []) or [])``
    -> ``returns_history=[]``.

    The book's return series feeds the CVaR de-risking gate: a 5% tail loss
    (worst 2 of 40 days at -10%) halves the weight. Dropped, the overlay
    keeps sizing through a breached tail.
    """
    g = _shell(monkeypatch, tmp_path)
    book = {"equity": 100_000, "returns_history": [0.01] * 36 + [-0.10] * 4}
    out = g._apply_risk_overlay("AAPL", _STOCK_DATE, _pm_state(), book)
    assert "- **Target Weight**: 4.0%" in out["final_trade_decision"]


def test_r17_mut_graph_overlay_sees_trade_history_is_pinned(monkeypatch, tmp_path):
    """Kills trading_graph.py:773  ``trade_history=list(state.get("trade_history", []) or [])``
    -> ``trade_history=[]``.

    The closed-trade ledger feeds the Kelly win rate: a book that lost its
    last ten trades is sized at 5% (a 0-10 record blended into the win rate)
    instead of the rating's own 8%. Dropped, every book is sized on the
    rating's conviction alone.
    """
    g = _shell(monkeypatch, tmp_path)
    book = {"equity": 100_000, "trade_history": [{"ticker": "X", "return": -0.05}] * 10}
    out = g._apply_risk_overlay("AAPL", _STOCK_DATE, _pm_state(), book)
    assert (
        "Kelly raw 0.050 x breaker 1.00 (normal) x CVaR 1.00 = target weight 0.050."
        in out["final_trade_decision"]
    )


def test_r17_mut_graph_overlay_stop_keeps_significant_digits_is_pinned(monkeypatch, tmp_path):
    """Kills trading_graph.py:837  ``{decision.stop_loss:.6g}`` -> ``{decision.stop_loss:.2f}``.

    USDT-M carries ~1e-5 priced contracts: with two decimals the stop and
    the entry both render as "0.00" (the same class of bug the take-profit
    rounding carried). ``%g`` keeps the significant digits.
    """
    g = _shell(
        monkeypatch, tmp_path, close=(0.0000123, 0.0000011),
    )
    state = _pm_state(pm_decision_fields={"price_target": 0.0000150, "confidence": 0.7})
    out = g._apply_risk_overlay("PEPEUSDT", _STOCK_DATE, state, {"equity": 100_000})
    md = out["final_trade_decision"]
    assert "- **Stop Loss**: 1.01e-05" in md
    assert "- **Entry Reference**: 1.23e-05" in md


def test_r17_mut_graph_overlay_honours_configured_risk_profile_is_pinned(monkeypatch, tmp_path):
    """Kills trading_graph.py:872  ``self.config.get("risk_profile") or _PERP_DEFAULT_PROFILE``
    -> ``_PERP_DEFAULT_PROFILE``.

    The perp advisory bullet threads the run's ``risk_profile`` into the
    conviction leverage cap (aggressive: 15x for a strong rating). Ignoring
    it renders every run's advisory on the conservative 5x cap while the
    ExecutionTicket built a few lines later uses the configured one.
    """
    g = _shell(monkeypatch, tmp_path, close=(60.0, 1.5), risk_profile="aggressive")
    _stub_perp(g, monkeypatch)
    out = g._apply_risk_overlay(
        "BTCUSDT", _PERP_DATE, _pm_state(pm_decision_fields={}), {"equity": 100_000},
        asset_type="crypto_perp",
    )
    assert "conviction 15.0x" in out["final_trade_decision"]


def test_r17_mut_graph_portfolio_control_equity_floor_is_pinned(monkeypatch, tmp_path):
    """Kills trading_graph.py:934  ``equity=float(state.equity or 0.0) or 1.0`` -> ``equity=float(state.equity or 0.0)``.

    A run with no injected book has equity 0.0; the snapshot gets the honest
    floor 1.0 so the five constraints evaluate the candidate (they refuse a
    non-positive equity outright). Without the floor every constraint vetoes
    and the shadow record reads VETOED for a perfectly admissible trade.
    """
    g = _shell(monkeypatch, tmp_path, close=(60.0, 1.5), portfolio_control_mode="shadow")
    _stub_perp(g, monkeypatch)
    out = g._apply_risk_overlay(
        "BTCUSDT", _PERP_DATE, _pm_state(pm_decision_fields={"price_target": 70.0}),
        None, asset_type="crypto_perp",
    )
    shadow = out["portfolio_control_shadow"]
    assert shadow["equity"] == 1.0
    assert [d["reasons"] for d in shadow["decisions"]] == [["within_limit"]] * 5
    assert shadow["resolver"]["action"] == "APPROVED"


# --- _render_perp_ticket (deterministic advisory bullets) ------------------- #


def _perp_decision(weight: float, entry: float | None = None, stop: float | None = None):
    return SimpleNamespace(target_weight=weight, stop_loss=stop, entry_price=entry)


_PERP_ENTRY, _PERP_ATR = 61_000.0, 1_500.0


def _long_liquidation() -> float:
    """Liquidation level the renderer computes for the long ticket below."""
    stop_dist = 2.0 * _PERP_ATR / _PERP_ENTRY
    lev, _caps = compute_leverage(stop_dist, _PERP_ATR / _PERP_ENTRY, "crypto_perp", 2)
    return perp_ticket.liquidation_price(_PERP_ENTRY, lev, "long", "crypto_perp")


def test_r17_mut_graph_perp_ticket_uses_overlay_entry_reference_is_pinned():
    """Kills trading_graph.py:1849  ``decision.entry_price if decision.entry_price else close`` -> ``close``.

    The ticket is priced from the overlay's own entry reference (the close
    the risk manager sized and stopped on), with ``close`` only as the
    fallback. A different last price would move the leverage caps and the
    liquidation estimate off the stop that was actually placed.
    """
    rendered = YiAlphaGraph._render_perp_ticket(
        "BTCUSDT", _PERP_DATE, "Buy", _perp_decision(0.08, entry=_PERP_ENTRY, stop=58_000.0),
        60_000.0, _PERP_ATR,
    )
    assert f"- **Est. Liquidation Price**: {_long_liquidation():.6g}" in rendered


def test_r17_mut_graph_perp_ticket_mark_at_liquidation_is_a_breach_is_pinned():
    """Kills trading_graph.py:1886  ``mark_close <= liq`` -> ``mark_close < liq`` and
    trading_graph.py:1917  ``mark_safe = mark_close is None or (...)`` -> ``mark_safe = True or (...)``.

    For a long, a mark AT the estimated liquidation level is already
    breached (the level is where the liquidation triggers); the note must
    say so instead of printing a positive "0.0% to liq" distance.
    """
    liq = _long_liquidation()
    rendered = YiAlphaGraph._render_perp_ticket(
        "BTCUSDT", _PERP_DATE, "Buy", _perp_decision(0.08, entry=_PERP_ENTRY, stop=58_000.0),
        _PERP_ENTRY, _PERP_ATR, mark_close=liq,
    )
    assert "has ALREADY breached the estimated liquidation level" in rendered
    assert "MARK has crossed the estimated liquidation level" in rendered


def test_r17_mut_graph_perp_ticket_short_breach_check_is_mirrored_is_pinned():
    """Kills trading_graph.py:1888  ``mark_close >= liq`` -> ``mark_close <= liq``.

    For a short the liquidation level sits ABOVE the entry, so a mark BELOW
    it is safe: the ticket reports the distance to liquidation, not a
    breach. (The short branch is dormant in the overlay today — the risk
    manager emits long-only weights — so no rendered report exercises it.)
    """
    rendered = YiAlphaGraph._render_perp_ticket(
        "BTCUSDT", _PERP_DATE, "Sell", _perp_decision(-0.08, entry=_PERP_ENTRY, stop=64_000.0),
        _PERP_ENTRY, _PERP_ATR, mark_close=59_000.0,
    )
    assert "breached" not in rendered
    assert "from current mark 59000 to liq" in rendered


def test_r17_mut_graph_perp_ticket_liquidation_distance_is_pinned():
    """Kills trading_graph.py:1906  ``abs(liq / entry - 1.0)`` -> ``abs(entry / liq - 1.0)``.

    The "(x% from entry; ...)" figure is the liquidation distance measured
    from the entry (how far price falls), not the inverse ratio.
    """
    rendered = YiAlphaGraph._render_perp_ticket(
        "BTCUSDT", _PERP_DATE, "Buy", _perp_decision(0.08, entry=_PERP_ENTRY, stop=58_000.0),
        _PERP_ENTRY, _PERP_ATR,
    )
    distance = f"{abs(_long_liquidation() / _PERP_ENTRY - 1.0):.1%}"
    assert f"({distance} from entry;" in rendered
    assert f"({abs(_PERP_ENTRY / _long_liquidation() - 1.0):.1%} from entry;" not in rendered


# --- trailing funding -------------------------------------------------------- #


def _funding_provider_spy(monkeypatch, answer=(0.0001, 0.0002)):
    calls: list[tuple[str, str, str]] = []

    def provider(ticker, start, end):
        calls.append((ticker, start, end))
        return pd.Series(answer)

    monkeypatch.setattr("yialpha.backtest.engine._binance_funding_provider", provider)
    return calls


def test_r17_mut_graph_funding_is_live_within_three_days_is_pinned(monkeypatch):
    """Kills trading_graph.py:2013  ``date.today() - timedelta(days=3)`` -> ``timedelta(days=30)``.

    Only the last three days are "live": a trade date twenty days back is a
    historical run and must not pay a vendor call per decision (a backtest
    re-prices hundreds of dates).
    """
    calls = _funding_provider_spy(monkeypatch)
    historical = (date.today() - timedelta(days=20)).isoformat()
    assert YiAlphaGraph._trailing_funding_total("BTCUSDT", historical) is None
    assert calls == []


def test_r17_mut_graph_funding_note_names_the_absent_cause_is_pinned():
    """Kills trading_graph.py:2045  ``date.today() - timedelta(days=3)`` -> ``timedelta(days=30)``.

    A None funding sum explains WHY: a historical run did not fetch by
    policy, a live run's fetch failed. With a 30-day "live" window a
    three-week-old report would blame a fetch failure that never happened.
    """
    historical = (date.today() - timedelta(days=20)).isoformat()
    note = YiAlphaGraph._format_funding_note(None, historical)
    assert note == "n/a (historical run; not fetched)"
    assert YiAlphaGraph._format_funding_note(None, date.today().isoformat()) == "n/a (fetch failed)"


# --------------------------------------------------------------------------- #
# yialpha/graph/perf_telemetry.py — opt-in node telemetry
# --------------------------------------------------------------------------- #


@pytest.fixture
def perf_clock(monkeypatch):
    """Replace telemetry's clock with a hand-driven counter (returns the list)."""
    now = [1000.0]
    monkeypatch.setattr(perf_telemetry, "time", SimpleNamespace(monotonic=lambda: now[0]))
    return now


def test_r17_mut_perf_wrap_node_records_elapsed_wall_is_pinned(perf_clock):
    """Kills perf_telemetry.py:204  ``tracker.record(node_name, time.monotonic() - t0)``
    -> ``tracker.record(node_name, 0.0)``.

    The wrapper books the time the handler actually took. The existing tests
    assert ``wall_seconds >= 0.0``, which a hard-coded zero satisfies.
    """
    tracker = NodePerfTracker()

    def handler(state):
        perf_clock[0] += 2.5
        return state

    assert wrap_node(handler, "sizer", tracker)({"x": 1}) == {"x": 1}
    assert tracker.serialize()["nodes"]["sizer"] == {
        "wall_seconds": 2.5, "calls": 1, "tokens_in": 0, "tokens_out": 0, "tokens_reasoning": 0,
    }


def test_r17_mut_perf_wrap_runnable_records_elapsed_wall_is_pinned(perf_clock):
    """Kills perf_telemetry.py:261  ``tracker.record(node_name, time.monotonic() - t0)``
    -> ``tracker.record(node_name, 0.0)``.

    Same for Runnable (tool) nodes: their HTTP time is what the A/B speedup
    gate used to count as zero.
    """

    class Tool:
        def invoke(self, state, config=None, **kwargs):
            perf_clock[0] += 0.75
            return {"done": True}

    tracker = NodePerfTracker()
    assert wrap_runnable_node(Tool(), "tools_market", tracker)({}) == {"done": True}
    assert tracker.serialize()["nodes"]["tools_market"]["wall_seconds"] == 0.75


def test_r17_mut_perf_wrap_node_sets_active_node_is_pinned():
    """Kills perf_telemetry.py:197  ``tracker.set_active_node(node_name)`` -> ``tracker.set_active_node(None)``.

    While a wrapped node runs, the CURRENT THREAD's active node is that node:
    it is what attributes the LLM tokens the node spends to it instead of
    ``_unattributed_``. (The Runnable wrapper has this test; the plain one
    did not.)
    """
    tracker = NodePerfTracker()
    seen = []

    def handler(state):
        seen.append(tracker.get_active_node())
        return state

    wrap_node(handler, "Market Analyst", tracker)({})
    assert seen == ["Market Analyst"]
    assert tracker.get_active_node() is None


def test_r17_mut_perf_negative_duration_is_clamped_is_pinned():
    """Kills perf_telemetry.py:107  ``if dur < 0.0:`` -> ``if dur < -1.0:``.

    "Negative durations are clamped to 0": a clock oddity must never put a
    negative sample into a node's wall time.
    """
    tracker = NodePerfTracker()
    tracker.record("n", -0.5)
    assert tracker.serialize()["nodes"]["n"]["wall_seconds"] == 0.0


def test_r17_mut_perf_snapshot_is_independent_is_pinned():
    """Kills perf_telemetry.py:158  ``name: dict(slot) ...`` -> ``name: slot ...``.

    ``serialize()`` returns a SNAPSHOT: the per-node rows are copies, so a
    consumer mutating what it got (the dashboard, the JSON writer) cannot
    change the tracker's accumulators.
    """
    tracker = NodePerfTracker()
    tracker.record_tokens("n", input_tokens=5)
    snapshot = tracker.serialize()
    snapshot["nodes"]["n"]["tokens_in"] = 999
    snapshot["nodes"]["ghost"] = {}
    assert tracker.serialize()["nodes"] == {
        "n": {"wall_seconds": 0.0, "calls": 0, "tokens_in": 5, "tokens_out": 0,
              "tokens_reasoning": 0}
    }


def _llm_result(*generations, llm_output=None) -> LLMResult:
    return LLMResult(
        generations=[list(generations)], llm_output=llm_output or {}
    )


def _ai(input_tokens: int, output_tokens: int) -> AIMessage:
    return AIMessage(
        content="x",
        usage_metadata={"input_tokens": input_tokens, "output_tokens": output_tokens,
                        "total_tokens": input_tokens + output_tokens},
    )


def test_r17_mut_perf_reads_usage_only_from_ai_messages_is_pinned():
    """Kills perf_telemetry.py:307  ``if not isinstance(message, AIMessage): continue``
    -> ``if False:``.

    Usage is read from ``AIMessage`` generations only (the token callback's
    contract). A non-AI message in the response contributes nothing.
    """
    tracker = NodePerfTracker()
    cb = NodePerfTokenCallback(tracker)
    human = HumanMessage(
        content="echo",
        usage_metadata={"input_tokens": 10, "output_tokens": 4, "total_tokens": 14},
    )
    cb.on_llm_end(_llm_result(ChatGeneration(message=human)))
    assert tracker.serialize()["nodes"] == {}


def test_r17_mut_perf_sums_usage_over_generations_is_pinned():
    """Kills perf_telemetry.py:312  ``input_tokens += int(...)`` -> ``input_tokens = int(...)``.

    A response can carry several generations (n>1); their usage is summed,
    so the node is billed for every completion it paid for.
    """
    tracker = NodePerfTracker()
    cb = NodePerfTokenCallback(tracker)
    cb.on_llm_end(_llm_result(
        ChatGeneration(message=_ai(10, 4), generation_info={}),
        ChatGeneration(message=_ai(7, 3), generation_info={}),
    ))
    assert tracker.serialize()["nodes"]["_unattributed_"] == {
        "wall_seconds": 0.0, "calls": 0, "tokens_in": 17, "tokens_out": 7,
        "tokens_reasoning": 0,
    }


def test_r17_mut_perf_llm_output_fallback_does_not_add_is_pinned():
    """Kills perf_telemetry.py:335  ``if output_tokens == 0:`` -> ``if True:``.

    The ``llm_output`` fallback FILLS a field the generations left at zero;
    it never adds on top of a value they already supplied (double counting).
    """
    tracker = NodePerfTracker()
    cb = NodePerfTokenCallback(tracker)
    cb.on_llm_end(_llm_result(
        ChatGeneration(message=_ai(10, 4), generation_info={}),
        llm_output={"token_usage": {"input_tokens": 0, "output_tokens": 50}},
    ))
    assert tracker.serialize()["nodes"]["_unattributed_"]["tokens_out"] == 4


def test_r17_mut_perf_report_is_indented_json_is_pinned(tmp_path):
    """Kills perf_telemetry.py:436  ``json.dumps(payload, indent=2)`` -> ``json.dumps(payload)``.

    ``node_perf_<date>.json`` is written as indented JSON (the same bytes
    ``json.dump(..., indent=2)`` produced before the atomic write): people
    read this file next to the state log.
    """
    tracker = NodePerfTracker()
    tracker.record("sizer", 1.5)
    target = tmp_path / "perf" / "node_perf_2026-01-15.json"
    dump_perf_report(tracker, target)
    assert target.read_text(encoding="utf-8").splitlines()[:4] == [
        "{", '  "nodes": {', '    "sizer": {', '      "wall_seconds": 1.5,',
    ]


# --------------------------------------------------------------------------- #
# yialpha/graph/analyst_execution.py — plan order and wall-time tracking
# --------------------------------------------------------------------------- #


def test_r17_mut_execution_positioning_first_without_market_is_pinned():
    """Kills analyst_execution.py:86  ``... else 0`` -> ``... else len(keys)``.

    Without a market analyst "positioning" goes FIRST (or first when market
    is absent), not last.
    """
    plan = build_analyst_execution_plan(("social", "news", "positioning"))
    assert [spec.key for spec in plan.specs] == ["positioning", "social", "news"]


def test_r17_mut_execution_wall_times_are_a_copy_is_pinned():
    """Kills analyst_execution.py:133  ``return dict(self._wall_times)`` -> ``return self._wall_times``.

    ``get_wall_times`` hands out a copy: a consumer editing the mapping it
    received cannot rewrite the tracker's record.
    """
    tracker = AnalystWallTimeTracker(build_analyst_execution_plan(("market", "social")))
    tracker.mark_started("market", started_at=1.0)
    tracker.mark_completed("market", completed_at=3.0)
    times = tracker.get_wall_times()
    assert times == {"market": 2.0}
    times["market"] = 99.0
    times["social"] = 1.0
    assert tracker.get_wall_times() == {"market": 2.0}


def test_r17_mut_execution_zero_wall_time_is_listed_is_pinned():
    """Kills analyst_execution.py:139  ``if duration is not None:`` -> ``if duration:``.

    An analyst that completed (even in zero time on a frozen clock) HAS a
    wall time and is listed; only analysts that never completed are absent.
    """
    tracker = AnalystWallTimeTracker(build_analyst_execution_plan(("market", "social")))
    tracker.mark_started("market", started_at=5.0)
    tracker.mark_completed("market", completed_at=5.0)
    assert tracker.format_summary() == "Analyst wall time: Market 0.00s"


def test_r17_mut_execution_pending_summary_is_pinned():
    """Kills analyst_execution.py:143  ``return "Analyst wall time: pending"`` -> ``return ""``.

    Before any analyst completes the summary says so ("pending") instead of
    rendering an empty telemetry line.
    """
    tracker = AnalystWallTimeTracker(build_analyst_execution_plan(("market",)))
    assert tracker.format_summary() == "Analyst wall time: pending"


# --------------------------------------------------------------------------- #
# yialpha/graph/analyst_fanout.py — parallel analysts (iron law: serial-equivalent)
# --------------------------------------------------------------------------- #


def _fanout_state() -> dict:
    return {
        "messages": [HumanMessage(content="AAPL")],
        "company_of_interest": "AAPL",
        "asset_type": "stock",
        "trade_date": "2026-07-01",
        "market_report": "",
        "sentiment_report": "",
        "investment_debate_state": {"count": 0},
    }


def _fake_fanout(monkeypatch, keys, on_invoke=None, **kwargs):
    """A fan-out node whose compiled subgraphs are fakes.

    ``on_invoke(key, clone)`` runs inside the worker in place of the analyst;
    returns ``(node, seen)`` with every ``(key, clone, config)`` the subgraphs
    were invoked with.
    """
    plan = build_analyst_execution_plan(keys)
    seen: list[tuple[str, dict, dict]] = []

    class _FakeSubgraph:
        def __init__(self, spec) -> None:
            self.spec = spec

        def invoke(self, clone, config=None):
            seen.append((self.spec.key, clone, config))
            if on_invoke is not None:
                on_invoke(self.spec.key, clone)
            return {self.spec.report_key: f"R:{self.spec.key}"}

    monkeypatch.setattr(
        analyst_fanout, "build_analyst_subgraph",
        lambda spec, *_args, **_kw: _FakeSubgraph(spec),
    )
    node = create_analyst_fanout_node(
        plan, {key: (lambda: None) for key in keys}, dict.fromkeys(keys),
        ConditionalLogic(), **kwargs,
    )
    return node, seen


def test_r17_mut_fanout_passes_budget_to_each_subgraph_is_pinned(monkeypatch):
    """Kills analyst_fanout.py:376  ``invoke_config = {"recursion_limit": recursion_limit}`` -> ``{}``.

    The derived budget is APPLIED at invoke time: each subgraph is invoked
    with ``{"recursion_limit": <derived>}`` (``compile()`` cannot take it).
    Without it every subgraph runs on langgraph's default and the
    serial-vs-parallel parity the iron law relies on is gone.
    """
    set_config({"max_recur_limit": 50})
    node, seen = _fake_fanout(monkeypatch, ("market", "social"))
    assert node.resolved_recursion_limit == 25
    assert node(_fanout_state()) == {"market_report": "R:market", "sentiment_report": "R:social"}
    assert [config for _key, _clone, config in seen] == [
        {"recursion_limit": 25}, {"recursion_limit": 25},
    ]


def test_r17_mut_fanout_clones_are_independent_is_pinned(monkeypatch):
    """Kills analyst_fanout.py:339  ``copy.deepcopy(parent_state)`` -> ``dict(parent_state)``.

    Each analyst runs on a deep copy: whatever it writes into a NESTED field
    of its clone (a debate sub-state, the message list) never reaches the
    parent state or a sibling — "subgraph mutations never leak".
    """

    def on_invoke(key, clone):
        if key == "market":
            clone["investment_debate_state"]["count"] = 99
            clone["messages"].append(HumanMessage(content="private"))

    node, _seen = _fake_fanout(monkeypatch, ("market", "social"), on_invoke=on_invoke)
    state = _fanout_state()
    out = node(state)
    assert state["investment_debate_state"]["count"] == 0
    assert len(state["messages"]) == 1
    assert out == {"market_report": "R:market", "sentiment_report": "R:social"}


# --------------------------------------------------------------------------- #
# yialpha/graph/routing.py — instrument classification and descriptor
# --------------------------------------------------------------------------- #


def test_r17_mut_routing_bare_suffix_is_not_stripped_is_pinned():
    """Kills routing.py:143  ``base.endswith(suffix) and len(base) > len(suffix)``
    -> ``base.endswith(suffix)``.

    The suffix strip needs something LEFT: a ticker that is only the quote
    suffix ("USDT") is kept whole instead of collapsing to an empty base.
    """
    assert routing._etf_base("crypto_perp", "USDT") == "USDT"
    assert routing._etf_base("crypto_perp", "usdc") == "USDC"
    assert routing._etf_base("crypto_perp", "SPYUSDT") == "SPY"


def test_r17_mut_routing_descriptor_defaults_to_equity_is_pinned():
    """Kills routing.py:277  ``asset = str(asset_type or "stock")`` -> ``... or "crypto"``.

    A caller that passes no asset type gets the default EQUITY instrument:
    the descriptor reports ``asset_type="stock"`` and class ``equity``
    (``instrument_class`` already treats a missing type as stock).
    """
    described = routing.describe_instrument(None, "AAPL")
    assert (described.asset_type, described.instrument_class) == ("stock", "equity")


def test_r17_mut_routing_listed_on_as_of_day_counts_as_listed_is_pinned(monkeypatch):
    """Kills routing.py:341  ``onboard_date <= as_of`` -> ``onboard_date < as_of``.

    "Was this contract live on ``as_of``?": a contract that went live ON the
    as-of day was tradable at that date's close, so it counts as listed.
    """
    listing = {"MU": {"onboard_date": "2026-01-15", "status": "trading"}}
    monkeypatch.setattr(routing, "equity_perp_listing_info", lambda: listing)
    assert routing.describe_instrument("crypto_perp", "MUUSDT", as_of="2026-01-15").listed_asof is True
    assert routing.describe_instrument("crypto_perp", "MUUSDT", as_of="2026-01-14").listed_asof is False


# --- portfolio control (V2.4): the enforced / shadow pipeline -------------- #


@pytest.fixture
def enforced(monkeypatch, tmp_path):
    """ENFORCED portfolio control on a BTCUSDT perp run with an isolated ledger."""
    register_run(_RUN_ID, "BTCUSDT", "crypto_perp", "pure_crypto_perp", _PERP_DATE)
    set_ledger_run_context(_RUN_ID, "BTCUSDT", "crypto_perp", "pure_crypto_perp", _PERP_DATE)
    g = _shell(monkeypatch, tmp_path, close=(60.0, 1.5), portfolio_control_mode="enforced")
    _stub_perp(g, monkeypatch)
    yield g
    reset_ledger_run_context()


def _sell_state(**fields) -> dict:
    """A Sell decision; ``fields`` go into the structured PM fields."""
    return {
        "final_trade_decision": "**Rating**: Sell\n\nThesis.",
        "pm_rating": "Sell",
        "pm_decision_fields": {"confidence": 0.6, **fields},
    }


def _short_state() -> dict:
    """The only way a short may open: the explicit structured ``desired_side``."""
    return _sell_state(desired_side="SHORT", price_target=55.0)


def _control_run(g, state: dict) -> dict:
    return g._apply_risk_overlay(
        "BTCUSDT", _PERP_DATE, state, {"equity": 100_000}, asset_type="crypto_perp",
    )


def _see_book(monkeypatch, *positions: tuple[str, str, float]) -> None:
    """Make the portfolio-control stage read this book: (symbol, class, signed weight)."""
    rows = [
        {
            "symbol": symbol, "payload": {"instrument_class": klass},
            "side": "LONG" if weight >= 0 else "SHORT", "signed_weight": weight,
        }
        for symbol, klass, weight in positions
    ]
    monkeypatch.setattr(portfolio_module, "load_positions_input", lambda: (rows, "ledger"))


def test_r17_mut_graph_short_funding_gate_spares_longs_is_pinned(enforced, monkeypatch):
    """Kills trading_graph.py:1464  ``proposed > 0.0 and side == "SHORT" and funding_annualized is not None``
    -> ``proposed > 0.0 and funding_annualized is not None``.

    The sign-aware gate is the SHORT side's mirror of the long gate: a SHORT
    pays NEGATIVE funding, so adverse carry for a short is carry a long
    RECEIVES. A long with -52%/yr carry must not be damped by it.
    """
    _stub_perp(enforced, monkeypatch, funding=-0.01)
    control = _control_run(enforced, _pm_state(pm_decision_fields={"price_target": 70.0}))[
        "portfolio_control"
    ]
    assert (control["side"], control["proposed_size"]) == ("LONG", pytest.approx(0.08))


def test_r17_mut_graph_hard_short_threshold_is_inclusive_is_pinned(enforced, monkeypatch):
    """Kills trading_graph.py:1472  ``funding_annualized <= -hard_thr`` -> ``funding_annualized < -hard_thr``.

    Carry of exactly the hard threshold quarters the short, like the long
    gate's own hard threshold (inclusive at both ends).
    """
    total = -0.014
    _stub_perp(enforced, monkeypatch, funding=total)
    enforced.risk_manager.funding_hard_annual = -(total / 7.0 * 365.0)
    control = _control_run(enforced, _short_state())["portfolio_control"]
    assert control["proposed_size"] == pytest.approx(0.0125)


def test_r17_mut_graph_rule_multipliers_are_rendered_with_decimals_is_pinned(enforced, monkeypatch):
    """Kills trading_graph.py:1580  ``(x{d.multiplier:.3f})`` -> ``(x{d.multiplier:.0f})``.

    The rule lines render each constraint's multiplier with three decimals:
    a gross book of 0.95 leaves 0.05 of the 1.0 cap for a 0.08 candidate, so
    ``global_gross`` resizes by x0.625 — printed as "x1" it would read as a
    pass.
    """
    _see_book(
        monkeypatch,
        ("ETHUSDT", "pure_crypto_perp", 0.30), ("SOLUSDT", "pure_crypto_perp", 0.20),
        ("MUUSDT", "stock_perp", -0.25), ("NVDAUSDT", "stock_perp", -0.20),
    )
    md = _control_run(enforced, _pm_state(pm_decision_fields={"price_target": 70.0}))[
        "final_trade_decision"
    ]
    assert "- global_gross: RESIZE (x0.625) — exceeds_limit" in md
    assert "- single_concentration: PASS (x1.000)" in md


def test_r17_mut_graph_resized_to_zero_opens_no_position_is_pinned(enforced, monkeypatch):
    """Kills trading_graph.py:1713  ``... and result.final_size > 0.0 and side != "FLAT"``
    -> ``result.action != "VETOED" and side != "FLAT"``.

    A candidate resized all the way to zero (the book is already at its gross
    cap) opens NO position: the outcome is RESIZED at 0.0, not an open row.
    """
    _see_book(monkeypatch, ("ETHUSDT", "pure_crypto_perp", 1.0))
    control = _control_run(enforced, _pm_state(pm_decision_fields={"price_target": 70.0}))[
        "portfolio_control"
    ]
    assert (control["action"], control["final_size"]) == ("RESIZED", 0.0)
    assert open_positions() == []


# --- the synced Candidate ticket -------------------------------------------- #


def test_r17_mut_graph_synced_ticket_carries_its_cost_is_pinned(enforced):
    """Kills trading_graph.py:1278  ``ticket.estimated_cost = cost.total_bps / 1e4``
    -> ``ticket.estimated_cost = 0.0``.

    The ticket synced to the authoritative short carries the round-trip cost
    estimate of THAT side (as a fraction), the same input its re-derived
    tradeability verdict was judged against.
    """
    out = _control_run(enforced, _short_state())
    expected = estimate_round_trip_cost(
        "crypto_perp", "short", DEFAULT_HORIZON_DAYS, None,
    ).total_bps / 1e4
    assert expected > 0.0
    assert out["execution_ticket"]["estimated_cost"] == pytest.approx(expected)


# --- propagate / resolve_instrument_context / the run itself ---------------- #


def test_r17_mut_graph_pending_memory_resolves_as_of_the_trade_date_is_pinned():
    """Kills trading_graph.py:2218  ``as_of_date=str(trade_date)`` -> ``as_of_date=None``.

    Pending memory entries are resolved AS OF the run's trade date: on a
    historical run only outcomes that were known by that date may be folded
    into the reflection layer. Resolving "as of now" would leak later
    returns into a backtest's memory context.
    """
    seen: dict[str, object] = {}
    g = YiAlphaGraph.__new__(YiAlphaGraph)
    g.config = {}
    g._checkpointer_ctx = None
    g._resolve_pending_entries = lambda ticker, **kw: seen.update(ticker=ticker, **kw)
    g._run_graph = lambda *a, **kw: ("done", "Hold")
    g.propagate("AAPL", "2026-01-02 09:30:00")
    assert seen == {"ticker": "AAPL", "as_of_date": "2026-01-02", "asset_type": "stock"}


def test_r17_mut_graph_memory_sweep_uses_the_run_benchmark_is_pinned(monkeypatch):
    """Kills trading_graph.py:2104  ``benchmark=self._resolve_benchmark(ticker)`` -> ``benchmark="SPY"``.

    The reflection sweep prices alpha against the run's own benchmark
    (BTCUSDT for a pure-crypto run, the configured ``benchmark_ticker``), the
    same baseline ``verify-history`` scores the rating against.
    """
    seen: dict[str, object] = {}
    monkeypatch.setattr(
        memory_resolution, "resolve_pending_entries",
        lambda log, reflector, ticker, **kw: seen.update(ticker=ticker, **kw) or 0,
    )
    g = YiAlphaGraph.__new__(YiAlphaGraph)
    g.memory_log = object()
    g.reflector = object()
    g._resolve_benchmark = lambda ticker: "^N225"
    g._resolve_pending_entries("7203.T", as_of_date="2026-01-20", asset_type="stock")
    assert seen["benchmark"] == "^N225"


def test_r17_mut_graph_memory_sweep_passes_the_fallback_venue_is_pinned(monkeypatch):
    """Kills trading_graph.py:2106  ``asset_type=asset_type`` -> ``asset_type=None``.

    ``asset_type`` is the FALLBACK pricing venue for legacy untagged memory
    entries: the sweep passes the run's own venue through, so an untagged
    entry swept by a perp run prices on the perp's candles instead of
    falling to the equity default.
    """
    seen: dict[str, object] = {}
    monkeypatch.setattr(
        memory_resolution, "resolve_pending_entries",
        lambda log, reflector, ticker, **kw: seen.update(**kw) or 0,
    )
    g = YiAlphaGraph.__new__(YiAlphaGraph)
    g.memory_log = object()
    g.reflector = object()
    g._resolve_benchmark = lambda ticker: "BTCUSDT"
    g._resolve_pending_entries("BTCUSDT", as_of_date="2026-09-03", asset_type="crypto_perp")
    assert seen["asset_type"] == "crypto_perp"


def test_r17_mut_graph_identity_lookup_honours_the_pit_date_is_pinned(monkeypatch):
    """Kills trading_graph.py:2133  ``resolve_instrument_identity(identity_ticker, trade_date)``
    -> ``resolve_instrument_identity(identity_ticker, None)``.

    ``trade_date`` is the point-in-time guard of the identity lookup: on a
    historical date the live ``.info`` snapshot (company name, sector,
    exchange) is refused and the context degrades to ticker-only. Dropping
    the date would put today's company facts into a past run (look-ahead).
    """
    looked_up: list[tuple[str, str | None]] = []
    monkeypatch.setattr(
        tg, "resolve_instrument_identity",
        lambda ticker, trade_date=None: looked_up.append((ticker, trade_date)) or "identity",
    )
    monkeypatch.setattr(
        tg, "build_instrument_context", lambda ticker, asset_type, identity: identity
    )
    g = YiAlphaGraph.__new__(YiAlphaGraph)
    assert g.resolve_instrument_context("AAPL", "stock", trade_date="2024-01-15") == "identity"
    assert looked_up == [("AAPL", "2024-01-15")]


def _run_shell(monkeypatch, tmp_path, seen: dict, **config) -> YiAlphaGraph:
    """A graph whose every seam around ``_run_graph`` is a recording stub."""
    g = YiAlphaGraph.__new__(YiAlphaGraph)
    g.config = {"results_dir": str(tmp_path), **config}
    g.debug = False
    g.perf_tracker = None
    g.selected_analysts = ["market"]

    def invoke(state, args):
        seen["graph_input"] = state
        seen["analysis_date_during_run"] = get_analysis_date()
        return {"final_trade_decision": "**Rating**: Buy\n\nThesis."}

    def past_context(ticker, as_of_date=None, asset_type=None):
        seen["past"] = (as_of_date, asset_type)
        return ""

    g.memory_log = SimpleNamespace(
        get_past_context=past_context,
        store_decision=lambda **kw: seen.setdefault("stored", kw),
    )
    g.resolve_instrument_context = lambda *a, **kw: "instrument context"
    g.propagator = Propagator()
    g._bind_run_record_stage = lambda *a, **kw: (None, None)
    g._apply_risk_overlay = lambda cn, td, state, ps, asset_type="stock": state
    g._stamp_run_record_ids = lambda *a, **kw: None
    g._log_state = lambda td, state, at="stock": None
    g.signal_processor = SignalProcessor(None)
    g._invoke_or_stream = invoke
    return g


def test_r17_mut_graph_run_pins_the_pit_analysis_date_is_pinned(monkeypatch, tmp_path):
    """Kills trading_graph.py:2536  ``set_analysis_date(str(trade_date)) if str(trade_date) else ...``
    -> ``set_analysis_date(None)``.

    The analysis date is pinned for the whole run: every vendor clamp
    (``get_stock_data`` / ``get_binance_klines`` fetch windows) reads it, so a
    backtest never sees rows after the analysis date. Without the pin every
    historical run trades on today's data.
    """
    seen: dict[str, object] = {}
    g = _run_shell(monkeypatch, tmp_path, seen)
    g._run_graph("AAPL", "2024-01-15")
    assert seen["analysis_date_during_run"] == "2024-01-15"


def test_r17_mut_graph_past_memory_is_clamped_to_the_trade_date_is_pinned(monkeypatch, tmp_path):
    """Kills trading_graph.py:2580  ``as_of_date=str(trade_date)`` -> ``as_of_date=None``.

    The memory log's past decisions are read AS OF the trade date (and for
    the run's asset type): a historical run may only see decisions made
    before it, never a later entry's reflection.
    """
    seen: dict[str, object] = {}
    g = _run_shell(monkeypatch, tmp_path, seen)
    g._run_graph("BTCUSDT", "2026-09-03", asset_type="crypto_perp")
    assert seen["past"] == ("2026-09-03", "crypto_perp")


def test_r17_mut_graph_pending_entry_keeps_the_run_venue_is_pinned(monkeypatch, tmp_path):
    """Kills trading_graph.py:2723  ``asset_type=asset_type`` -> ``asset_type=None``.

    The decision stored for deferred reflection carries the run's asset type:
    it is the tag the later sweep prices the entry on (perp vs spot lessons
    of one ticker stay on their own venue).
    """
    seen: dict[str, object] = {}
    g = _run_shell(monkeypatch, tmp_path, seen)
    g._run_graph("BTCUSDT", "2026-09-03", asset_type="crypto_perp")
    assert seen["stored"]["asset_type"] == "crypto_perp"
    assert seen["stored"]["trade_date"] == "2026-09-03"


def test_r17_mut_graph_resumed_run_is_disclosed_on_the_state_is_pinned(monkeypatch, tmp_path):
    """Kills trading_graph.py:2648  ``final_state["checkpoint_resumed"] = True`` -> ``pass``.

    A resumed run says so on its state (presence-gated key, written into the
    log): its quality block covers only the RESUMED segment, so a zero
    sentinel count must not be read as "fully evidenced" (run_robust's
    DEGRADED verdict reads that count).
    """
    seen: dict[str, object] = {}
    g = _run_shell(
        monkeypatch, tmp_path, seen,
        checkpoint_enabled=True, data_cache_dir=str(tmp_path / "cache"),
        max_debate_rounds=1, max_risk_discuss_rounds=1,
    )
    final_state, _signal = g._run_graph("AAPL", "2024-01-15", _resuming=True)
    assert seen["graph_input"] is None
    assert final_state["checkpoint_resumed"] is True


def test_r17_mut_graph_zero_size_candidate_gets_no_stop_is_pinned(enforced):
    """Kills trading_graph.py:1299  ``side != "FLAT" and proposed > 0.0 and (...)``
    -> ``side != "FLAT" and (...)``.

    The synced ticket's ATR stop is armed "whenever the candidate actually
    enters with size". An explicit ``desired_side=LONG`` whose rating sizes
    it to zero proposes nothing: no stop is placed for it (a stop with no
    position behind it would read as protection the book does not have).
    """
    state = {
        "final_trade_decision": "**Rating**: Hold\n\nThesis.",
        "pm_rating": "Hold",
        "pm_decision_fields": {"desired_side": "LONG", "confidence": 0.5},
    }
    ticket = _control_run(enforced, state)["execution_ticket"]
    assert (ticket["side"], ticket["proposed_size"], ticket["stop_price"]) == (
        "LONG", 0.0, None,
    )


def test_r17_mut_graph_bridge_runs_only_when_enabled_is_pinned(monkeypatch, tmp_path):
    """Kills trading_graph.py:1015  ``asset_type == "crypto_perp" and self.config.get("stock_perp_fair_value")``
    -> ``asset_type == "crypto_perp"``.

    The Fair Value Bridge is flag-gated: with ``stock_perp_fair_value`` off
    (the default) a tokenized-stock perp run leaves the ticket unconverted —
    no USD target, no USDT contract target, no FX fetch, no bridge block.
    The existing flag-off test uses a PURE-CRYPTO ticker, where the bridge
    is skipped by the instrument class regardless of the flag.
    """
    monkeypatch.setattr("yialpha.graph.routing.instrument_class", lambda a, t: "stock_perp")
    fx_calls: list[str] = []

    def fx(as_of):
        fx_calls.append(as_of)
        return QuoteFxResult(
            rate=0.9993,
            available_at=f"{_PERP_DATE}T00:00:00+00:00",
            fetched_at=f"{_PERP_DATE}T00:00:00+00:00",
        )

    monkeypatch.setattr("yialpha.perp.quote_fx.usdt_usd_as_of", fx)
    state = _pm_state(
        pm_decision_fields={
            "underlying_price_target": 100.0, "price_target": 100.05,
            "price_target_currency": "USDT", "confidence": 0.7,
        }
    )

    def run(**config):
        g = _shell(monkeypatch, tmp_path, close=(0.5, 0.01), **config)
        _stub_perp(g, monkeypatch, mark=0.4999)
        return g._apply_risk_overlay(
            "MUUSDT", _PERP_DATE, dict(state), {"equity": 100_000}, asset_type="crypto_perp",
        )

    off = run()
    assert (off["execution_ticket"]["underlying_target"], off["execution_ticket"]["contract_target"]) == (
        None, None,
    )
    assert "Fair Value Bridge" not in off["final_trade_decision"]
    assert fx_calls == []

    on = run(stock_perp_fair_value=True)
    assert on["execution_ticket"]["contract_target"] == pytest.approx(100.0 / 0.9993)
    assert "Fair Value Bridge (USD → USDT contract target)" in on["final_trade_decision"]
    assert fx_calls == [_PERP_DATE]


def test_r17_mut_fanout_workers_get_their_own_capture_scope_is_pinned(monkeypatch):
    """Kills analyst_fanout.py:206  ``bind_isolated_prediction_capture_scope()`` -> ``pass``.

    Each analyst-subgraph worker binds a FRESH prediction-capture registry
    before invoking. The worker context is a copy of the parent's, so
    without its own registry every worker writes into the parent's shared
    dict: four parallel ``begin_prediction_capture`` calls race the one
    ``__active__`` slot and ``submit_prediction`` appends to whichever
    analyst went last. Here the parent's registry must stay empty after two
    analysts armed their captures in their workers.
    """
    set_ledger_run_context("R17MUTGRAPH0002", "AAPL", "stock", "equity", "2026-07-01")
    prediction_tools.ensure_prediction_capture_scope()      # the parent's registry, as a recorded run binds it
    try:

        def on_invoke(key, clone):
            prediction_tools.begin_prediction_capture(key, "AAPL", "r17")

        node, _seen = _fake_fanout(monkeypatch, ("market", "social"), on_invoke=on_invoke)
        node(_fanout_state())
        assert prediction_tools._capture_scope() == {}
    finally:
        prediction_tools.reset_prediction_capture_for_test()
        reset_ledger_run_context()
