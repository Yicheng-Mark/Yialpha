"""Baseline report rendering + multi-run distribution for Phase 0.

Turns :class:`~yiagents.backtest.engine.BacktestResult` objects into the
markdown baseline report the roadmap's Phase 0 ships: equity curve, the full
metric suite, a buy-and-hold comparison, and -- because LLM decisions are
non-deterministic -- an aggregate over N re-realized runs (mean +/- std of the
key statistics) so a single lucky draw can't masquerade as an edge.

Pure functions only: render / aggregate / write. No network, no LLM.
"""

from __future__ import annotations

import logging
import statistics
from collections.abc import Callable, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

from yiagents.backtest.engine import BacktestResult

logger = logging.getLogger(__name__)

# Metrics whose distribution we summarize across runs (annualized where relevant).
_DISTRIBUTION_KEYS: tuple[str, ...] = (
    "total_return", "cagr", "sharpe", "sortino", "max_drawdown", "calmar",
    "deflated_sharpe", "alpha_vs_buyhold",
)

_SPARK_BARS = "▁▂▃▄▅▆▇█"


def _sparkline(values: Sequence[float]) -> str:
    """Compact Unicode bar chart of a series, for inline equity visualization."""
    if not values:
        return ""
    lo = min(values)
    hi = max(values)
    if hi == lo:
        return _SPARK_BARS[-1] * len(values)
    span = hi - lo
    return "".join(
        _SPARK_BARS[min(len(_SPARK_BARS) - 1, int((v - lo) / span * (len(_SPARK_BARS) - 1)))]
        for v in values
    )


def _fmt_pct(x: float | None) -> str:
    if x is None:
        return "n/a"
    return f"{x * 100:.2f}%"


def _fmt_num(x: float | None) -> str:
    if x is None:
        return "n/a"
    return f"{x:.3f}"


def _metric_dict(result: BacktestResult) -> dict[str, Any]:
    return asdict(result.metrics) if result.metrics else {}


def render_backtest_report(result: BacktestResult) -> str:
    """Render a single backtest run to markdown."""
    m = _metric_dict(result)
    eq = result.equity
    bh = result.benchmark_equity
    bh_total = (bh[-1] / bh[0] - 1.0) if bh and bh[0] else None
    beats = "BEATS" if (m.get("total_return") is not None and bh_total is not None
                        and m["total_return"] > bh_total) else "TRAILS"

    lines: list[str] = []
    lines.append(f"# Backtest: {result.ticker}")
    lines.append("")
    lines.append(f"- Holding window: {result.holding_days} sessions  |  "
                 f"Run tag: `{result.config_summary.get('run_tag', 'default')}`  |  "
                 f"Index benchmark: `{result.config_summary.get('index_benchmark', 'n/a')}`")
    lines.append(f"- Rebalances: {len(result.trades)}  |  "
                 f"Initial capital: {result.initial_capital:,.0f}  |  "
                 f"Transaction cost: {result.config_summary.get('cost_bps', 0.0)} bps")
    lines.append(f"- Cache: {result.cached_hits} hits / {result.cached_misses} misses")
    lines.append("")
    lines.append("## Equity curve")
    lines.append("")
    lines.append(f"`{_sparkline(eq)}`  "
                 f"start {eq[0]:,.0f} -> end {eq[-1]:,.0f}")
    lines.append("")
    lines.append("## Metrics")
    lines.append("")
    lines.append("| Metric | Strategy | Buy & Hold |")
    lines.append("|---|---:|---:|")
    lines.append(f"| Total return | {_fmt_pct(m.get('total_return'))} | {_fmt_pct(bh_total)} |")
    lines.append(f"| CAGR | {_fmt_pct(m.get('cagr'))} | n/a |")
    lines.append(f"| Volatility (ann.) | {_fmt_pct(m.get('volatility'))} | n/a |")
    lines.append(f"| Sharpe | {_fmt_num(m.get('sharpe'))} | n/a |")
    lines.append(f"| Sortino | {_fmt_num(m.get('sortino'))} | n/a |")
    lines.append(f"| Max drawdown | {_fmt_pct(m.get('max_drawdown'))} | n/a |")
    lines.append(f"| Calmar | {_fmt_num(m.get('calmar'))} | n/a |")
    lines.append(f"| Deflated Sharpe | {_fmt_num(m.get('deflated_sharpe'))} | n/a |")
    lines.append(f"| Alpha vs B&H (ann.) | {_fmt_pct(m.get('alpha_vs_buyhold'))} | -- |")
    lines.append("")
    lines.append(f"**Verdict: {result.ticker} strategy {beats} buy-and-hold on total return.**")
    lines.append("")
    if result.trades:
        lines.append("## Trades")
        lines.append("")
        lines.append("| Date | Rating | Weight | Price | Realized ret | Alpha |")
        lines.append("|---|---|---:|---:|---:|---:|")
        for t in result.trades:
            price_str = f"{t.price:.2f}" if t.price is not None else "n/a"
            lines.append(
                f"| {t.date} | {t.rating} | {t.executed_weight:.2f} | "
                f"{price_str} | {_fmt_pct(t.raw_return)} | {_fmt_pct(t.alpha_vs_index)} |"
            )
        lines.append("")
    return "\n".join(lines)


def summarize_distribution(results: Sequence[BacktestResult]) -> dict[str, dict[str, float]]:
    """Aggregate key metrics across N runs to mean +/- std.

    Returns ``{metric: {"mean": ..., "std": ..., "min": ..., "max": ..., "n": int}}``.
    Metrics are only summarized when every run has a non-None value for them.
    """
    if not results:
        return {}
    per_key: dict[str, list[float]] = {k: [] for k in _DISTRIBUTION_KEYS}
    for r in results:
        m = _metric_dict(r)
        for k in _DISTRIBUTION_KEYS:
            v = m.get(k)
            if v is not None and isinstance(v, (int, float)):
                per_key[k].append(float(v))
    summary: dict[str, dict[str, float]] = {}
    for k, vs in per_key.items():
        if len(vs) < 1:
            continue
        entry = {
            "mean": statistics.fmean(vs),
            "min": min(vs),
            "max": max(vs),
            "n": len(vs),
        }
        entry["std"] = statistics.pstdev(vs) if len(vs) > 1 else 0.0
        summary[k] = entry
    return summary


def render_multi_run_report(results: Sequence[BacktestResult]) -> str:
    """Markdown table summarizing the metric distribution across N runs."""
    if not results:
        return "# Multi-run report\n\nNo runs.\n"
    ticker = results[0].ticker
    summary = summarize_distribution(results)

    lines = [
        f"# Multi-run distribution: {ticker}",
        "",
        f"- Runs aggregated: {len(results)}",
        "- LLM decisions are non-deterministic; report the distribution, not a single draw.",
        "",
        "| Metric | mean | std | min | max |",
        "|---|---:|---:|---:|---:|",
    ]
    label_map = {
        "total_return": "Total return", "cagr": "CAGR", "sharpe": "Sharpe",
        "sortino": "Sortino", "max_drawdown": "Max drawdown", "calmar": "Calmar",
        "deflated_sharpe": "Deflated Sharpe", "alpha_vs_buyhold": "Alpha vs B&H",
    }
    pct_keys = {"total_return", "cagr", "max_drawdown", "alpha_vs_buyhold"}
    for key in _DISTRIBUTION_KEYS:
        if key not in summary:
            continue
        e = summary[key]
        fmt = (lambda x: _fmt_pct(x)) if key in pct_keys else (lambda x: _fmt_num(x))
        lines.append(
            f"| {label_map[key]} | {fmt(e['mean'])} | {fmt(e['std'])} | "
            f"{fmt(e['min'])} | {fmt(e['max'])} |"
        )
    lines.append("")
    return "\n".join(lines)


def write_report(
    result_or_results: BacktestResult | Sequence[BacktestResult],
    path: str | Path | None = None,
    results_dir: str | Path | None = None,
) -> Path:
    """Write a single-run or multi-run baseline report to disk and return its path.

    When several results are passed, both the multi-run summary and each
    individual run are written into one file.
    """
    results: Sequence[BacktestResult]
    if isinstance(result_or_results, BacktestResult):
        results = [result_or_results]
    else:
        results = list(result_or_results)

    if path is None:
        base = Path(results_dir) if results_dir else Path(".")
        ticker = results[0].ticker if results else "backtest"
        safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in ticker)
        path = base / "backtest_reports" / f"{safe}_baseline.md"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    parts: list[str] = []
    if len(results) > 1:
        parts.append(render_multi_run_report(results))
        parts.append("\n\n---\n\n")
    for r in results:
        parts.append(render_backtest_report(r))
        parts.append("\n\n---\n\n")

    path.write_text("\n".join(parts), encoding="utf-8")
    logger.info("Wrote backtest report to %s", path)
    return path


def multi_run(
    run_once: Callable[[int], BacktestResult],
    n_runs: int,
) -> list[BacktestResult]:
    """Run a backtest factory ``n_runs`` times and return all results.

    The factory receives the run index so the caller can vary the
    ``run_tag`` / seed / cache to get independent LLM realizations.
    """
    if n_runs < 1:
        raise ValueError("n_runs must be >= 1")
    return [run_once(i) for i in range(n_runs)]
