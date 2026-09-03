"""V2.4 batch K — the five hard constraints, advisory metrics, and resolver.

The five-constraint freeze (docs/V2_BASELINE.md): global gross -> asset
class -> single concentration -> directional concentration -> correlation
cluster, composed by the min-multiplier resolver. VaR/CVaR/beta/correlation
are advisory-only (calculated, displayed, archived — never resizing).
"""

from __future__ import annotations

import math
import random
from typing import cast

import pytest

from yialpha.risk import constraints
from yialpha.risk.constraints import (
    FIVE_HARD_CONSTRAINTS,
    PortfolioLimits,
    PortfolioSnapshot,
    PositionView,
    RiskDecision,
    advisory_metrics,
    asset_class,
    directional_concentration,
    evaluate_constraints,
    global_gross,
    single_concentration,
)
from yialpha.risk.resolver import render_resolver_lines, resolve_constraints
from yialpha.risk.signed_math import Side


def _pv(
    weight: float,
    symbol: str = "BTCUSDT",
    cls: str = "pure_crypto_perp",
    side: Side | None = None,
) -> PositionView:
    if side is None:
        side = "LONG" if weight >= 0 else "SHORT"
    return PositionView(symbol=symbol, instrument_class=cls, side=side, weight=weight)


def _snap(
    positions: tuple[PositionView, ...] = (),
    equity: float = 100_000.0,
    returns: tuple[float, ...] = (),
) -> PortfolioSnapshot:
    return PortfolioSnapshot(
        equity=equity, positions=positions, returns_history=returns
    )


def _dec(rule: str, action: str, mult: float) -> RiskDecision:
    return RiskDecision(
        rule=rule, action=action, multiplier=mult,  # type: ignore[arg-type]
        reasons=[f"{rule}:{action}"], metrics={},
    )


# --- RiskDecision validation -------------------------------------------------


@pytest.mark.unit
def test_risk_decision_accepts_valid_verdicts():
    RiskDecision(rule="r", action="PASS", multiplier=1.0, reasons=[], metrics={})
    RiskDecision(rule="r", action="RESIZE", multiplier=0.5, reasons=["exceeds_limit"], metrics={})
    # A RESIZE all the way to zero is still a RESIZE (shrinking, not refusing).
    RiskDecision(rule="r", action="RESIZE", multiplier=0.0, reasons=["exceeds_limit"], metrics={})
    RiskDecision(rule="r", action="VETO", multiplier=0.0, reasons=["non_positive_equity"], metrics={})


@pytest.mark.unit
def test_risk_decision_rejects_bad_action():
    with pytest.raises(ValueError, match="PASS/RESIZE/VETO"):
        _dec("r", "HOLD", 1.0)


@pytest.mark.unit
def test_risk_decision_rejects_out_of_bounds_multiplier():
    for bad in (-0.1, 1.0001, float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError, match="multiplier"):
            RiskDecision(rule="r", action="RESIZE", multiplier=bad, reasons=[], metrics={})


@pytest.mark.unit
def test_veto_must_carry_zero_multiplier():
    with pytest.raises(ValueError, match="VETO"):
        RiskDecision(rule="r", action="VETO", multiplier=0.3, reasons=[], metrics={})


# --- constraint unit cases ----------------------------------------------------


@pytest.mark.unit
def test_global_gross_pass_under_limit():
    snap = _snap((_pv(0.3, symbol="ETHUSDT"),))
    d = global_gross(_pv(0.3), snap, PortfolioLimits())
    assert d.rule == "global_gross"
    assert d.action == "PASS"
    assert d.multiplier == 1.0
    assert d.reasons == ["within_limit"]
    assert d.metrics["post_gross"] == pytest.approx(0.6)
    assert d.metrics["max_gross"] == 1.0


@pytest.mark.unit
def test_global_gross_resize_ratio_pinned():
    # 0.6 existing + 0.65 candidate = 1.25 gross over max 1.0 -> 1.0/1.25 = 0.8.
    snap = _snap((_pv(0.6, symbol="ETHUSDT"),))
    d = global_gross(_pv(0.65), snap, PortfolioLimits())
    assert d.action == "RESIZE"
    assert d.multiplier == pytest.approx(0.8)
    assert d.reasons == ["exceeds_limit"]
    assert d.metrics["post_gross"] == pytest.approx(1.25)


@pytest.mark.unit
def test_candidate_replaces_same_symbol_position():
    # Same symbol: 0.5 REPLACED by 0.75 -> post gross 0.75 (PASS at 1.0).
    # An add would be 1.25 and resize — this pins replace-not-add.
    snap = _snap((_pv(0.5, symbol="BTCUSDT"),))
    d = global_gross(_pv(0.75, symbol="BTCUSDT"), snap, PortfolioLimits())
    assert d.action == "PASS"
    assert d.metrics["post_gross"] == pytest.approx(0.75)


@pytest.mark.unit
def test_asset_class_groups_by_instrument_class():
    # stock_perp sleeve 0.4 + 0.35 = 0.75 over 0.6 -> 0.8; the pure-crypto
    # 0.3 does not consume the stock budget.
    snap = _snap(
        (
            _pv(0.4, symbol="AAPLUSDT", cls="stock_perp"),
            _pv(0.3, symbol="ETHUSDT"),
        )
    )
    d = asset_class(_pv(0.35, symbol="TSLAUSDT", cls="stock_perp"), snap, PortfolioLimits())
    assert d.rule == "asset_class"
    assert d.action == "RESIZE"
    assert d.multiplier == pytest.approx(0.6 / 0.75)
    assert d.metrics["post_class_gross"] == pytest.approx(0.75)


@pytest.mark.unit
def test_asset_class_ignores_other_classes():
    snap = _snap((_pv(0.7, symbol="ETHUSDT"),))
    d = asset_class(_pv(0.5, symbol="TSLAUSDT", cls="stock_perp"), snap, PortfolioLimits())
    assert d.action == "PASS"
    assert d.metrics["post_class_gross"] == pytest.approx(0.5)


@pytest.mark.unit
def test_single_concentration_over_cap_alone_is_resize_not_veto():
    # 0.25 alone over max_single 0.2 -> RESIZE at 0.2/0.25 = 0.8. A lone
    # position can always be clipped; VETO is reserved for degenerate input.
    d = single_concentration(_pv(0.25), _snap(), PortfolioLimits())
    assert d.action == "RESIZE"
    assert d.multiplier == pytest.approx(0.2 / 0.25)
    assert d.metrics["post_single"] == pytest.approx(0.25)


@pytest.mark.unit
def test_single_concentration_at_limit_passes():
    d = single_concentration(_pv(0.2), _snap(), PortfolioLimits())
    assert d.action == "PASS"
    assert d.multiplier == 1.0


@pytest.mark.unit
def test_directional_concentration_nets_same_side_gross():
    # LONG 0.5 book, SHORT 0.7 hedge, LONG candidate 0.4: NET long = 0.2
    # (not the 0.9 same-side gross) -> PASS under max_directional 0.8.
    snap = _snap(
        (
            _pv(0.5, symbol="ETHUSDT"),
            _pv(-0.7, symbol="SOLUSDT", side="SHORT"),
        )
    )
    d = directional_concentration(_pv(0.4), snap, PortfolioLimits())
    assert d.rule == "directional_concentration"
    assert d.metrics["post_directional"] == pytest.approx(0.2)
    assert d.action == "PASS"


@pytest.mark.unit
def test_directional_concentration_resize_without_hedge():
    # Same book minus the short: net long 0.9 over 0.8 -> 0.8/0.9.
    snap = _snap((_pv(0.5, symbol="ETHUSDT"),))
    d = directional_concentration(_pv(0.4), snap, PortfolioLimits())
    assert d.action == "RESIZE"
    assert d.multiplier == pytest.approx(0.8 / 0.9)


@pytest.mark.unit
def test_directional_concentration_short_side_and_flat():
    snap = _snap((_pv(0.5, symbol="ETHUSDT"),))
    # Net 0.3 long: exposure in the SHORT direction is 0.0.
    d = directional_concentration(_pv(-0.2, side="SHORT"), snap, PortfolioLimits())
    assert d.metrics["post_directional"] == 0.0
    assert d.action == "PASS"
    flat = directional_concentration(_pv(0.0, side="FLAT"), snap, PortfolioLimits())
    assert flat.action == "PASS"
    assert flat.metrics["post_directional"] == 0.0


@pytest.mark.unit
def test_cluster_groups_via_cluster_of():
    # BTC + ETH share the "majors" cluster: 0.25 + 0.3 = 0.55 over 0.4.
    limits = PortfolioLimits(cluster_of={"BTCUSDT": "majors", "ETHUSDT": "majors"})
    snap = _snap((_pv(0.25, symbol="BTCUSDT"),))
    d = constraints.correlation_cluster(_pv(0.3, symbol="ETHUSDT"), snap, limits)
    assert d.rule == "correlation_cluster"
    assert d.action == "RESIZE"
    assert d.multiplier == pytest.approx(0.4 / 0.55)
    assert d.metrics["post_cluster"] == pytest.approx(0.55)


@pytest.mark.unit
def test_cluster_unmapped_symbol_clusters_alone():
    limits = PortfolioLimits(cluster_of={"BTCUSDT": "majors"})
    snap = _snap((_pv(0.3, symbol="BTCUSDT"),))
    d = constraints.correlation_cluster(_pv(0.35, symbol="DOGEUSDT"), snap, limits)
    assert d.metrics["post_cluster"] == pytest.approx(0.35)
    assert d.action == "PASS"


@pytest.mark.unit
def test_cluster_default_none_is_per_symbol_second_cap():
    # cluster_of=None: every symbol clusters alone -> the rule re-measures
    # abs(candidate.weight) against max_cluster (0.4), a second per-symbol
    # cap next to max_single (0.2) — the tighter cap binds. Here we pin the
    # CLUSTER cap: 0.5 over 0.4 -> 0.4/0.5 = 0.8.
    d = constraints.correlation_cluster(_pv(0.5), _snap(), PortfolioLimits())
    assert d.action == "RESIZE"
    assert d.multiplier == pytest.approx(0.4 / 0.5)


@pytest.mark.unit
def test_five_hard_constraints_frozen_order():
    assert tuple(fn.__name__ for fn in FIVE_HARD_CONSTRAINTS) == (
        "global_gross",
        "asset_class",
        "single_concentration",
        "directional_concentration",
        "correlation_cluster",
    )


# --- VETO is reserved for degenerate inputs -----------------------------------


@pytest.mark.unit
def test_veto_on_non_positive_or_non_finite_equity():
    for equity in (0.0, -5.0, float("nan")):
        for d in evaluate_constraints(_pv(0.1), _snap(equity=equity), PortfolioLimits()):
            assert d.action == "VETO"
            assert d.multiplier == 0.0
            assert d.reasons == ["non_positive_equity"]


@pytest.mark.unit
def test_veto_on_degenerate_candidate_weight():
    for bad in (float("nan"), float("inf"), float("-inf")):
        for d in evaluate_constraints(_pv(bad), _snap(), PortfolioLimits()):
            assert d.action == "VETO"
            assert d.reasons == ["non_finite_candidate_weight"]


@pytest.mark.unit
def test_veto_on_non_finite_existing_position_weight():
    snap = _snap((_pv(float("nan"), symbol="ETHUSDT"),))
    for d in evaluate_constraints(_pv(0.1), snap, PortfolioLimits()):
        assert d.action == "VETO"
        assert d.reasons == ["non_finite_position_weight"]


@pytest.mark.unit
def test_evaluate_constraints_runs_all_five_in_order():
    snap = _snap((_pv(0.3, symbol="ETHUSDT"),))
    decisions = evaluate_constraints(_pv(0.15), snap, PortfolioLimits())
    assert [d.rule for d in decisions] == [
        "global_gross",
        "asset_class",
        "single_concentration",
        "directional_concentration",
        "correlation_cluster",
    ]
    assert all(d.action == "PASS" for d in decisions)
    assert all(d.multiplier == 1.0 for d in decisions)


@pytest.mark.unit
def test_constraint_crash_fails_closed(monkeypatch):
    def _boom(candidate: PositionView, snapshot: PortfolioSnapshot, limits: PortfolioLimits):
        raise RuntimeError("boom")

    monkeypatch.setattr(
        constraints, "FIVE_HARD_CONSTRAINTS", (_boom,) + constraints.FIVE_HARD_CONSTRAINTS
    )
    decisions = evaluate_constraints(_pv(0.1), _snap(), PortfolioLimits())
    assert len(decisions) == 6
    assert decisions[0].rule == "_boom"
    assert decisions[0].action == "VETO"
    assert decisions[0].multiplier == 0.0
    assert decisions[0].reasons == ["constraint_error"]
    # The surviving constraints still ran and reported.
    assert decisions[1].rule == "global_gross"
    assert decisions[1].action == "PASS"


# --- advisory metrics (display-only) -------------------------------------------


@pytest.mark.unit
def test_advisory_var_cvar_pinned():
    # 40 returns: four -10% tail observations, then 0.0..3.5% drift. The
    # linear 5th percentile interpolates INSIDE the flat tail block and the
    # CVaR tail (ceil(40*5%) worst, 2 or 3 under float rounding) sits
    # entirely inside it too — both statistics pin at -0.10.
    returns = tuple([-0.10] * 4 + [i / 1000.0 for i in range(36)])
    snap = _snap(returns=returns)
    out = advisory_metrics(_pv(0.0, side="FLAT"), snap, PortfolioLimits())
    assert out["var_95"] == pytest.approx(-0.10)
    assert out["cvar_95"] == pytest.approx(-0.10)

    # On a graded series (-10%..+29%): var_95 interpolates to -0.0805 and
    # the tail mean is at least as bad as the percentile.
    graded = tuple(i / 100.0 for i in range(-10, 30))
    out2 = advisory_metrics(_pv(0.0, side="FLAT"), _snap(returns=graded), PortfolioLimits())
    assert out2["var_95"] == pytest.approx(-0.0805)
    assert out2["cvar_95"] <= out2["var_95"]


@pytest.mark.unit
def test_advisory_var_cvar_none_when_insufficient_history():
    snap = _snap(returns=(-0.01, 0.02))
    out = advisory_metrics(_pv(0.0, side="FLAT"), snap, PortfolioLimits())
    assert out["var_95"] is None
    assert out["cvar_95"] is None


@pytest.mark.unit
def test_advisory_beta_vs_btc():
    rng = random.Random(7)
    btc = [rng.gauss(0.0, 0.02) for _ in range(60)]
    port = [2.0 * b for b in btc]
    snap = _snap(returns=tuple(port))
    # No bench series supplied -> honestly None, never fabricated.
    out = advisory_metrics(_pv(0.0, side="FLAT"), snap, PortfolioLimits())
    assert out["beta_vs_btc"] is None
    out2 = advisory_metrics(_pv(0.0, side="FLAT"), snap, PortfolioLimits(), btc_returns=btc)
    assert out2["beta_vs_btc"] == pytest.approx(2.0)
    # A flat bench has no beta to estimate.
    out3 = advisory_metrics(
        _pv(0.0, side="FLAT"), snap, PortfolioLimits(), btc_returns=[0.0] * 60
    )
    assert out3["beta_vs_btc"] is None


@pytest.mark.unit
def test_advisory_correlation_matrix():
    out = advisory_metrics(_pv(0.0, side="FLAT"), _snap(), PortfolioLimits())
    assert out["correlation_matrix"] == {}
    xs = [i / 100.0 for i in range(5)]
    neg = [-x for x in xs]
    out2 = advisory_metrics(
        _pv(0.0, side="FLAT"),
        _snap(),
        PortfolioLimits(),
        position_returns={"AAA": xs, "BBB": list(xs), "CCC": neg},
    )
    matrix = out2["correlation_matrix"]
    assert isinstance(matrix, dict)
    assert matrix["AAA"]["AAA"] == pytest.approx(1.0)
    assert matrix["AAA"]["BBB"] == pytest.approx(1.0)
    assert matrix["AAA"]["CCC"] == pytest.approx(-1.0)
    assert matrix["CCC"]["BBB"] == pytest.approx(-1.0)


@pytest.mark.unit
def test_advisory_concentration_placeholders():
    # Every V2.4 position is a USDT-M perp on one venue: the concentration
    # is 1.0 whenever gross > 0 (0.0 on a flat book) and venue exposure is
    # the single-venue placeholder.
    out = advisory_metrics(_pv(0.3), _snap(), PortfolioLimits())
    assert out["usdt_collateral_concentration"] == 1.0
    assert out["venue_exposure"] == 1.0
    flat = advisory_metrics(_pv(0.0, side="FLAT"), _snap(), PortfolioLimits())
    assert flat["usdt_collateral_concentration"] == 0.0


@pytest.mark.unit
def test_advisory_stress_shocks_pinned():
    # equity 100k; post gross 0.5 (0.25 stock_perp + 0.25 crypto); returns
    # alternate +-1% -> stdev(ddof=1) = 0.01*sqrt(40/39); beta exactly 2.0.
    btc = [0.005 if i % 2 == 0 else -0.005 for i in range(40)]
    snap = _snap(
        (
            _pv(0.25, symbol="AAPLUSDT", cls="stock_perp"),
            _pv(0.25, symbol="ETHUSDT"),
        ),
        returns=tuple(2.0 * b for b in btc),
    )
    out = advisory_metrics(_pv(0.0, side="FLAT"), snap, PortfolioLimits(), btc_returns=btc)
    assert out["usdt_depeg_shock"] == pytest.approx(-0.10 * 100_000.0)
    expected_vol = 0.01 * math.sqrt(40.0 / 39.0)
    assert out["binance_outage_shock"] == pytest.approx(
        -100_000.0 * 0.5 * 2.0 * expected_vol
    )
    assert out["btc_crash_x_earnings_gap"] == pytest.approx(
        100_000.0 * (-0.20 * 2.0 - 0.08 * 0.25)
    )


@pytest.mark.unit
def test_advisory_btc_crash_none_without_beta():
    snap = _snap((_pv(0.2, symbol="AAPLUSDT", cls="stock_perp"),))
    out = advisory_metrics(_pv(0.1), snap, PortfolioLimits())
    assert out["beta_vs_btc"] is None
    assert out["btc_crash_x_earnings_gap"] is None
    # No returns history either -> no daily vol for the outage shock.
    assert out["binance_outage_shock"] is None


# --- resolver -------------------------------------------------------------------


@pytest.mark.unit
def test_resolver_min_multiplier():
    decisions = [
        _dec("a", "PASS", 1.0),
        _dec("b", "RESIZE", 0.8),
        _dec("c", "RESIZE", 0.9),
    ]
    r = resolve_constraints(decisions, 1.0, "LONG")
    assert r.action == "RESIZED"
    assert r.final_multiplier == pytest.approx(0.8)
    assert r.final_size == pytest.approx(0.8)
    assert r.risk_decision_rules == ["b", "c"]
    assert r.reasons == ["b:RESIZE", "c:RESIZE"]


@pytest.mark.unit
def test_resolver_veto_dominates():
    decisions = [_dec("a", "PASS", 1.0), _dec("b", "VETO", 0.0)]
    r = resolve_constraints(decisions, 0.5, "LONG")
    assert r.action == "VETOED"
    assert r.final_multiplier == 0.0
    assert r.final_size == 0.0
    assert r.risk_decision_rules == ["b"]
    assert r.reasons == ["b:VETO"]


@pytest.mark.unit
def test_resolver_final_never_exceeds_proposed():
    r_pass = resolve_constraints([_dec("a", "PASS", 1.0)], 0.7, "SHORT")
    assert r_pass.action == "APPROVED"
    assert r_pass.final_size == pytest.approx(0.7)
    r_shrunk = resolve_constraints([_dec("a", "RESIZE", 0.25)], 0.8, "SHORT")
    assert r_shrunk.final_size == pytest.approx(0.2)
    assert r_shrunk.final_size <= 0.8
    # A RESIZE decision carrying multiplier 1.0 approves at full size.
    r_full = resolve_constraints([_dec("a", "RESIZE", 1.0)], 0.7, "SHORT")
    assert r_full.action == "APPROVED"
    assert r_full.final_size == pytest.approx(0.7)


@pytest.mark.unit
def test_resolver_side_echoed_unchanged():
    for side in ("LONG", "SHORT", "FLAT"):
        r = resolve_constraints([_dec("a", "VETO", 0.0)], 0.1, side)
        assert r.side == side


@pytest.mark.unit
def test_resolver_negative_proposed_vetoed():
    r = resolve_constraints([_dec("a", "PASS", 1.0)], -0.1, "LONG")
    assert r.action == "VETOED"
    assert r.final_multiplier == 0.0
    assert r.final_size == 0.0
    assert r.reasons == ["invalid_proposed_size"]
    assert r.risk_decision_rules == []


@pytest.mark.unit
def test_resolver_non_finite_proposed_vetoed():
    for bad in (float("nan"), float("inf"), float("-inf")):
        r = resolve_constraints([], bad, "LONG")
        assert r.action == "VETOED"
        assert r.reasons == ["invalid_proposed_size"]


@pytest.mark.unit
def test_resolver_empty_decisions_approves():
    r = resolve_constraints([], 0.42, "LONG")
    assert r.action == "APPROVED"
    assert r.final_multiplier == 1.0
    assert r.final_size == pytest.approx(0.42)
    assert r.reasons == []
    assert r.risk_decision_rules == []


@pytest.mark.unit
def test_evaluate_then_resolve_end_to_end():
    snap = _snap((_pv(0.6, symbol="ETHUSDT"),))
    decisions = evaluate_constraints(_pv(0.65), snap, PortfolioLimits())
    r = resolve_constraints(decisions, 0.65, "LONG")
    # Post book: gross 1.25, class 1.25, single 0.65, directional 1.25,
    # cluster (default per-symbol) 0.65 -> the single cap binds at
    # 0.2/0.65, landing the final size exactly on max_single.
    assert r.action == "RESIZED"
    assert r.final_multiplier == pytest.approx(0.2 / 0.65)
    assert r.final_size == pytest.approx(0.2)
    assert r.risk_decision_rules == [
        "global_gross",
        "asset_class",
        "single_concentration",
        "directional_concentration",
        "correlation_cluster",
    ]
    assert r.side == "LONG"


@pytest.mark.unit
def test_render_resolver_lines():
    r = resolve_constraints([_dec("global_gross", "RESIZE", 0.8)], 1.0, "LONG")
    text = render_resolver_lines(r)
    assert "**Resolver**: RESIZED" in text
    assert "multiplier 0.80" in text
    assert "global_gross" in text
    v = resolve_constraints([_dec("a", "VETO", 0.0)], 1.0, "LONG")
    vtext = render_resolver_lines(v)
    assert "**Resolver**: VETOED" in vtext
    assert "**Veto reasons**: a:VETO" in vtext


# --- property loops ---------------------------------------------------------------


@pytest.mark.unit
def test_property_resolver_invariants():
    rng = random.Random(20260903)
    for _ in range(50):
        decisions = []
        for i in range(rng.randint(0, 4)):
            if rng.random() < 0.25:
                action, mult = "VETO", 0.0
            else:
                mult = 1.0 if rng.random() < 0.15 else rng.random()
                action = "PASS" if mult == 1.0 else "RESIZE"
            decisions.append(_dec(f"r{i}", action, mult))
        proposed = rng.random() * 2.0
        side = cast(Side, rng.choice(("LONG", "SHORT", "FLAT")))
        result = resolve_constraints(decisions, proposed, side)
        assert 0.0 <= result.final_multiplier <= 1.0
        assert result.final_size <= proposed
        assert result.side == side
        if any(d.action == "VETO" for d in decisions):
            assert result.action == "VETOED"
            assert result.final_size == 0.0
        elif result.final_multiplier < 1.0:
            assert result.action == "RESIZED"
        else:
            assert result.action == "APPROVED"


@pytest.mark.unit
def test_property_constraints_multipliers_bounded():
    rng = random.Random(42)
    for _ in range(50):
        n = rng.randint(0, 5)
        positions = tuple(
            _pv(
                rng.uniform(-0.6, 0.6),
                symbol=f"S{i}USDT",
                cls=rng.choice(("pure_crypto_perp", "stock_perp")),
            )
            for i in range(n)
        )
        snap = _snap(positions, equity=rng.uniform(1.0, 200_000.0))
        limits = PortfolioLimits(
            max_gross=rng.uniform(0.3, 1.5),
            max_asset_class=rng.uniform(0.2, 0.9),
            max_single=rng.uniform(0.1, 0.5),
            max_directional=rng.uniform(0.3, 1.2),
            max_cluster=rng.uniform(0.2, 0.8),
        )
        for d in evaluate_constraints(_pv(rng.uniform(-0.8, 0.8)), snap, limits):
            assert 0.0 <= d.multiplier <= 1.0
            assert d.action in ("PASS", "RESIZE", "VETO")
            assert d.rule in (
                "global_gross",
                "asset_class",
                "single_concentration",
                "directional_concentration",
                "correlation_cluster",
            )
