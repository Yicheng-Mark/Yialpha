"""Unit tests for the Phase-3 validation gate decision logic."""

from __future__ import annotations

import pytest

from yiagents.backtest.engine import BacktestResult, TradeRow
from yiagents.backtest.metrics import BacktestMetrics
from yiagents.backtest.validation_gate import evaluate_gate


def _result(total_return: float, dsr: float, sharpe: float = 1.0,
            bh_total: float = 0.05, mdd: float = -0.10) -> BacktestResult:
    """Build a minimal BacktestResult with a chosen metric profile."""
    metrics = BacktestMetrics(
        total_return=total_return, cagr=total_return, volatility=0.15,
        sharpe=sharpe, sortino=sharpe, max_drawdown=mdd, calmar=abs(total_return / mdd) if mdd else 0.0,
        deflated_sharpe=dsr, alpha_vs_buyhold=total_return - bh_total,
        num_periods=252, periods_per_year=252,
    )
    bh_end = 1.0 + bh_total
    return BacktestResult(
        ticker="X", initial_capital=100_000, holding_days=5,
        equity=[100_000, 100_000 * (1 + total_return)],
        equity_dates=["2024-01-01", "2024-12-31"],
        trades=[TradeRow(date="2024-01-01", rating="Buy", target_weight=1.0,
                         executed_weight=1.0, price=100.0)],
        benchmark_equity=[100_000, 100_000 * bh_end],
        benchmark_name="X buy-and-hold", metrics=metrics,
    )


@pytest.mark.unit
def test_gate_passes_when_dsr_positive_and_beats_bh():
    baseline = _result(total_return=0.02, dsr=0.1, bh_total=0.05)
    improved = [_result(total_return=0.12, dsr=0.6, bh_total=0.05)]
    v = evaluate_gate(baseline, improved)
    assert v.passes is True
    assert v.beats_buyhold is True
    assert v.clears_hurdle is True


@pytest.mark.unit
def test_gate_fails_when_does_not_beat_buyhold():
    baseline = _result(total_return=0.02, dsr=0.1, bh_total=0.20)
    improved = [_result(total_return=0.10, dsr=0.7, bh_total=0.20)]
    v = evaluate_gate(baseline, improved)
    assert v.passes is False
    assert v.beats_buyhold is False
    assert "No validated edge" in v.recommendation


@pytest.mark.unit
def test_gate_fails_when_dsr_zero_or_below():
    baseline = _result(total_return=0.02, dsr=0.0, bh_total=0.05)
    improved = [_result(total_return=0.12, dsr=0.0, bh_total=0.05)]
    v = evaluate_gate(baseline, improved)
    assert v.passes is False


@pytest.mark.unit
def test_gate_marginal_pass_flagged_when_below_hurdle():
    baseline = _result(total_return=0.02, dsr=0.1, bh_total=0.05)
    improved = [_result(total_return=0.12, dsr=0.2, bh_total=0.05)]  # dsr>0 but <0.5
    v = evaluate_gate(baseline, improved)
    assert v.passes is True
    assert v.clears_hurdle is False
    assert any("marginal" in n.lower() for n in v.notes)


@pytest.mark.unit
def test_gate_aggregates_multiple_runs():
    baseline = _result(total_return=0.02, dsr=0.1, bh_total=0.05)
    improved = [
        _result(total_return=0.12, dsr=0.6, bh_total=0.05),
        _result(total_return=0.04, dsr=0.55, bh_total=0.05),  # one run barely beats
        _result(total_return=0.15, dsr=0.65, bh_total=0.05),
    ]
    v = evaluate_gate(baseline, improved)
    assert v.n_runs == 3
    # Mean total return (0.12+0.04+0.15)/3 ~ 0.103 > 0.05 B&H.
    assert v.beats_buyhold is True
    assert 0.5 < v.mean_dsr < 0.7


@pytest.mark.unit
def test_margin_vs_baseline_computed():
    baseline = _result(total_return=0.02, dsr=0.1, sharpe=0.5, bh_total=0.05)
    improved = [_result(total_return=0.12, dsr=0.6, sharpe=1.5, bh_total=0.05)]
    v = evaluate_gate(baseline, improved)
    assert pytest.approx(v.margin_vs_baseline["total_return"], abs=1e-6) == 0.10
    assert pytest.approx(v.margin_vs_baseline["sharpe"], abs=1e-6) == 1.0


@pytest.mark.unit
def test_render_markdown_contains_verdict():
    baseline = _result(total_return=0.02, dsr=0.1, bh_total=0.05)
    improved = [_result(total_return=0.12, dsr=0.6, bh_total=0.05)]
    v = evaluate_gate(baseline, improved)
    md = v.render()
    assert "# Validation Gate: PASS" in md
    assert "Recommendation" in md


@pytest.mark.unit
def test_costs_not_applied_adds_note():
    baseline = _result(total_return=0.02, dsr=0.1, bh_total=0.05)
    improved = [_result(total_return=0.12, dsr=0.6, bh_total=0.05)]
    v = evaluate_gate(baseline, improved, cost_bps_already_applied=False)
    assert any("cost" in n.lower() for n in v.notes)


@pytest.mark.unit
def test_empty_improved_raises():
    baseline = _result(total_return=0.02, dsr=0.1, bh_total=0.05)
    with pytest.raises(ValueError):
        evaluate_gate(baseline, [])
