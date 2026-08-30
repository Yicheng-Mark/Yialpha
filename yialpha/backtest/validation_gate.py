"""Phase 3 boundary-validation gate: does the system have a real, post-cost edge?

The roadmap's Phase 3 is a *decision point*, not new code. But the decision must
be reproducible and recorded, so it lives here as a pure function over backtest
results. The gate asks two questions the plan states explicitly:

1. **Deflated Sharpe >= 0.5** (the configured multiple-testing hurdle)
   -- is the observed performance plausibly real rather than the best of many
   noise draws?
2. **Beats buy-and-hold after costs** -- does the improved variant's mean total
   return exceed simply holding the asset, net of transaction costs?

A green gate is evidence for further out-of-sample/paper analysis, never an
authorization to trade. A red gate means the measured sample does not establish
an edge.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from yialpha.backtest.engine import BacktestResult

_METRIC_KEYS = (
    "total_return", "cagr", "sharpe", "max_drawdown", "calmar", "deflated_sharpe",
)


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


@dataclass
class GateVerdict:
    """Outcome of the Phase-3 validation gate."""

    passes: bool                      # True only if both DSR and B&H conditions hold
    mean_dsr: float                   # mean Deflated Sharpe across improved runs
    clears_hurdle: bool               # mean_dsr >= configured hurdle_dsr
    beats_buyhold: bool               # improved mean total return > B&H total return
    buyhold_total_return: float
    improved_total_return: float
    margin_vs_baseline: dict[str, float]   # improved-mean minus baseline, per metric
    n_runs: int
    recommendation: str
    notes: list[str] = field(default_factory=list)
    hurdle_dsr: float = 0.5
    n_baseline_runs: int = 1

    def render(self) -> str:
        """One-page markdown summary of the gate decision."""
        flag = "PASS" if self.passes else "FAIL"
        lines = [
            f"# Validation Gate: {flag}",
            "",
            f"- Mean Deflated Sharpe: **{self.mean_dsr:.3f}**"
            f" ({'clears' if self.clears_hurdle else 'does not clear'} "
            f"the {self.hurdle_dsr:g} hurdle)",
            f"- Beats buy-and-hold (after cost): **{self.beats_buyhold}**"
            f" (improved {self.improved_total_return:.2%} vs B&H {self.buyhold_total_return:.2%})",
            f"- Paired runs aggregated: baseline {self.n_baseline_runs}, "
            f"improved {self.n_runs}",
            "",
            "## Margin vs baseline (improved mean - baseline)",
            "",
            "| Metric | Delta |",
            "|---|---:|",
        ]
        for k, v in self.margin_vs_baseline.items():
            lines.append(f"| {k} | {v:+.4f} |")
        lines += ["", "## Recommendation", "", self.recommendation]
        if self.notes:
            lines += ["", "## Notes", ""] + [f"- {n}" for n in self.notes]
        return "\n".join(lines)


def evaluate_gate(
    baseline: BacktestResult | Sequence[BacktestResult],
    improved: Sequence[BacktestResult],
    min_dsr: float = 0.0,
    hurdle_dsr: float = 0.5,
    cost_bps_already_applied: bool = True,
) -> GateVerdict:
    """Decide whether the improved variant clears the Phase-3 gate.

    Parameters
    ----------
    baseline:
        One or more paired backtests of the current system -- the reference to
        beat. Multiple runs are averaged so an A/B comparison can reuse the
        same realized LLM decision tape for each baseline/improved pair.
    improved:
        One or more backtests of the Phase-1/2 variant (multiple runs absorb LLM
        non-determinism).
    min_dsr:
        Optional additional strict floor. The configured ``hurdle_dsr`` remains
        mandatory regardless of this value.
    hurdle_dsr:
        Mandatory multiple-testing hurdle (default 0.5).
    cost_bps_already_applied:
        When True (default) the improved runs already net transaction costs, so
        the B&H comparison is apples-to-apples. False makes the gate fail closed
        and records that costs are not reflected.
    """
    baselines = [baseline] if isinstance(baseline, BacktestResult) else list(baseline)
    if not baselines:
        raise ValueError("evaluate_gate requires at least one baseline backtest run")
    improved = list(improved)
    if not improved:
        raise ValueError("evaluate_gate requires at least one improved backtest run")
    if not 0.0 <= hurdle_dsr <= 1.0:
        raise ValueError("hurdle_dsr must be between 0 and 1")

    dsr_values = [r.metrics.deflated_sharpe for r in improved if r.metrics]
    tr_values = [r.metrics.total_return for r in improved if r.metrics]
    mean_dsr = _mean(dsr_values)
    improved_total_return = _mean(tr_values)

    # Buy-and-hold is averaged across the same improved run windows. Normally
    # these are identical; averaging remains correct if paired windows differ.
    bh_returns = [
        r.benchmark_equity[-1] / r.benchmark_equity[0] - 1.0
        for r in improved
        if r.benchmark_equity and r.benchmark_equity[0]
    ]
    bh_total_return = _mean(bh_returns)

    beats = improved_total_return > bh_total_return

    # Per-metric margin vs the mean of paired baseline runs.
    base_means: dict[str, float] = {}
    for k in _METRIC_KEYS:
        base_vals = [
            float(r.metrics.__dict__[k])
            for r in baselines
            if r.metrics and isinstance(r.metrics.__dict__.get(k), (int, float))
        ]
        if base_vals:
            base_means[k] = _mean(base_vals)
    imp_means = {}
    for k in _METRIC_KEYS:
        raw = [r.metrics.__dict__.get(k) for r in improved if r.metrics]
        improved_vals: list[float] = [
            float(v) for v in raw if isinstance(v, (int, float))
        ]
        if improved_vals and k in base_means:
            imp_means[k] = _mean(improved_vals) - base_means[k]
    # max_drawdown: less-negative is better; keep raw delta but callers read sign.

    clears = mean_dsr >= hurdle_dsr
    passes = (
        clears
        and (mean_dsr > min_dsr)
        and beats
        and cost_bps_already_applied
    )

    notes: list[str] = []
    if not cost_bps_already_applied:
        notes.append("Transaction costs NOT applied to improved runs; the B&H "
                     "comparison likely flatters the strategy, so the gate "
                     "fails closed.")
    if not clears:
        notes.append(
            f"Mean DSR does not clear the mandatory {hurdle_dsr:g} "
            "multiple-testing hurdle."
        )
    if not beats:
        notes.append("Does not beat buy-and-hold after cost: the simplest "
                     "explanation is usually the right one.")

    if passes:
        recommendation = (
            "The research gate clears. Continue with a fresh out-of-sample or "
            "paper-analysis window. This result does not authorize live trading."
        )
    else:
        recommendation = (
            "No validated edge in this sample. Keep the result in analysis-only "
            "status, widen or redesign the out-of-sample test, and re-run the "
            "gate before drawing a strategy-effectiveness conclusion."
        )

    return GateVerdict(
        passes=passes,
        mean_dsr=mean_dsr,
        clears_hurdle=clears,
        beats_buyhold=beats,
        buyhold_total_return=bh_total_return,
        improved_total_return=improved_total_return,
        margin_vs_baseline=imp_means,
        n_runs=len(improved),
        recommendation=recommendation,
        notes=notes,
        hurdle_dsr=hurdle_dsr,
        n_baseline_runs=len(baselines),
    )
