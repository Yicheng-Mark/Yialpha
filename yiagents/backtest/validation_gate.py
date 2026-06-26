"""Phase 3 boundary-validation gate: does the system have a real, post-cost edge?

The roadmap's Phase 3 is a *decision point*, not new code. But the decision must
be reproducible and recorded, so it lives here as a pure function over backtest
results. The gate asks two questions the plan states explicitly:

1. **Deflated Sharpe > 0** (and ideally clears the 0.5 multiple-testing hurdle)
   -- is the observed performance plausibly real rather than the best of many
   noise draws?
2. **Beats buy-and-hold after costs** -- does the improved variant's mean total
   return exceed simply holding the asset, net of transaction costs?

A green gate sends the system to Phase 4 (live). A red gate stops the team from
burning money on a signal with no edge and recommends redirecting to pure-
quantitative execution of the browser-captured signals, or shipping the Phase-1
risk layer as a standalone product.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from yiagents.backtest.engine import BacktestResult


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
    clears_hurdle: bool               # mean_dsr >= 0.5 (the multiple-testing neutral)
    beats_buyhold: bool               # improved mean total return > B&H total return
    buyhold_total_return: float
    improved_total_return: float
    margin_vs_baseline: dict[str, float]   # improved-mean minus baseline, per metric
    n_runs: int
    recommendation: str
    notes: list[str] = field(default_factory=list)

    def render(self) -> str:
        """One-page markdown summary of the gate decision."""
        flag = "PASS" if self.passes else "FAIL"
        lines = [
            f"# Validation Gate: {flag}",
            "",
            f"- Mean Deflated Sharpe: **{self.mean_dsr:.3f}**"
            f" ({'clears' if self.clears_hurdle else 'does not clear'} the 0.5 hurdle)",
            f"- Beats buy-and-hold (after cost): **{self.beats_buyhold}**"
            f" (improved {self.improved_total_return:.2%} vs B&H {self.buyhold_total_return:.2%})",
            f"- Runs aggregated: {self.n_runs}",
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
    baseline: BacktestResult,
    improved: Sequence[BacktestResult],
    min_dsr: float = 0.0,
    hurdle_dsr: float = 0.5,
    cost_bps_already_applied: bool = True,
) -> GateVerdict:
    """Decide whether the improved variant clears the Phase-3 gate.

    Parameters
    ----------
    baseline:
        Backtest of the current (Phase-0) system -- the reference to beat.
    improved:
        One or more backtests of the Phase-1/2 variant (multiple runs absorb LLM
        non-determinism).
    min_dsr:
        Minimum mean Deflated Sharpe required to pass (roadmap: ``> 0``).
    hurdle_dsr:
        The multiple-testing neutral hurdle (0.5). ``clears_hurdle`` flags it
        separately so a pass at DSR>0 but <0.5 is honestly labelled marginal.
    cost_bps_already_applied:
        When True (default) the improved runs already net transaction costs, so
        the B&H comparison is apples-to-apples. When False the caller is warned
        via a note that costs are not reflected.
    """
    improved = list(improved)
    if not improved:
        raise ValueError("evaluate_gate requires at least one improved backtest run")

    dsr_values = [r.metrics.deflated_sharpe for r in improved if r.metrics]
    tr_values = [r.metrics.total_return for r in improved if r.metrics]
    mean_dsr = _mean(dsr_values)
    improved_total_return = _mean(tr_values)

    # Buy-and-hold total return from the (first) improved run's benchmark curve.
    bh = improved[0].benchmark_equity
    bh_total_return = (bh[-1] / bh[0] - 1.0) if bh and bh[0] else 0.0

    beats = improved_total_return > bh_total_return

    # Per-metric margin vs the single baseline run.
    base_m = baseline.metrics.__dict__ if baseline.metrics else {}
    imp_means = {}
    for k in _METRIC_KEYS:
        vals = [r.metrics.__dict__.get(k) for r in improved if r.metrics]
        vals = [v for v in vals if isinstance(v, (int, float))]
        if vals and isinstance(base_m.get(k), (int, float)):
            imp_means[k] = _mean(vals) - float(base_m[k])
    # max_drawdown: less-negative is better; keep raw delta but callers read sign.

    clears = mean_dsr >= hurdle_dsr
    passes = (mean_dsr > min_dsr) and beats

    notes: list[str] = []
    if not cost_bps_already_applied:
        notes.append("Transaction costs NOT applied to improved runs; the B&H "
                     "comparison likely flatters the strategy.")
    if passes and not clears:
        notes.append("Passes on DSR>0 but is below the 0.5 multiple-testing "
                     "hurdle -- treat as a marginal edge, escalate sample size "
                     "before live capital.")
    if not beats:
        notes.append("Does not beat buy-and-hold after cost: the simplest "
                     "explanation is usually the right one.")

    if passes:
        recommendation = (
            "Proceed to Phase 4 (live) with a small pilot: the strategy shows a "
            "real, post-cost edge over buy-and-hold. Scale position size gradually."
        )
    else:
        recommendation = (
            "No validated edge. Do NOT take this live. Options: (a) feed the "
            "browser-captured alternative signals into a pure-quantitative "
            "executor instead of the LLM debate; (b) ship the Phase-1 risk "
            "layer as a standalone product; (c) widen the backtest sample and "
            "re-test before revisiting."
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
    )
