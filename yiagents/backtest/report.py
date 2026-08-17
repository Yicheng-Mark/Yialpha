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
import math
import statistics
from collections.abc import Callable, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

from yiagents.backtest.engine import BacktestResult
from yiagents.fmt import fmt_num3, fmt_pct

logger = logging.getLogger(__name__)

# Metrics whose distribution we summarize across runs (annualized where relevant).
_DISTRIBUTION_KEYS: tuple[str, ...] = (
    "total_return", "cagr", "sharpe", "sortino", "max_drawdown", "calmar",
    "deflated_sharpe", "alpha_vs_buyhold", "factor_alpha",
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
    # Shared implementation (yiagents.fmt) — kept as a local alias so the
    # ~26 call sites read unchanged.
    return fmt_pct(x)


def _fmt_num(x: float | None) -> str:
    return fmt_num3(x)


def _fmt_profit_factor(x: float | None) -> str:
    """Profit factor with the inf case spelled out (wins, zero losses)."""
    if x is None:
        return "n/a"
    if math.isinf(x):
        return "inf (no losing trades)"
    return f"{x:.2f}"


def _fmt_price(x: float | None) -> str:
    """Price at adaptive precision — 2dp for equity-scale, sig-digits below.

    ``f"{x:.2f}"`` renders every sub-cent perpetual (PEPE/SHIB class, prices
    ~1e-5) as "0.00", making the trade table unusable exactly where perp
    simulation runs. Below 1.0 we keep 4 significant digits instead.
    """
    if x is None:
        return "n/a"
    if x >= 1.0:
        return f"{x:.2f}"
    return f"{x:.4g}"


def _metric_dict(result: BacktestResult) -> dict[str, Any]:
    return asdict(result.metrics) if result.metrics else {}


def _cost_summary(result: BacktestResult) -> str:
    """The cost model the engine actually charged, not the raw parameter.

    Perp runs charge (taker_bps or cost_bps), optionally x0.9 BNB, plus
    slippage — the engine carries the effective-fee string in config_summary
    (``perp_fees``). Showing the bare ``cost_bps`` parameter instead mis-states
    what ran (e.g. "0.0 bps" while taker_bps=8 charged every fill).
    """
    perp_fees = result.config_summary.get("perp_fees")
    if perp_fees:
        return str(perp_fees)
    return f"{result.config_summary.get('cost_bps', 0.0)} bps"


def _perp_section(result: BacktestResult) -> list[str]:
    """Perp-simulation observability block, or [] for non-perp runs.

    The engine models leverage, funding drag, isolated-margin liquidation,
    shorts and venue fill quantization — but none of it was rendered, so a
    run that liquidated produced a report structurally identical to one that
    never came close (round-5 audit, 2026-08-16). This surfaces the fields
    the engine already records.
    """
    cs = result.config_summary
    if "perp_leverage" not in cs:
        return []
    lines = ["", "## Perp simulation", ""]
    lines.append(
        f"- Leverage: {cs.get('perp_leverage', 1)}x  |  "
        f"Side: {'long + short' if cs.get('perp_short') else 'long-only'}  |  "
        f"Fill quantization: {cs.get('perp_fill_quantization', 'n/a')}"
    )
    liq_source = cs.get("perp_liq_price_source")
    if liq_source:
        lines.append(f"- Liquidation trigger source: {liq_source}")
    m = result.metrics
    funding = cs.get("perp_funding_paid_total")
    if funding is not None:
        # Signed: positive = net paid (long-biased drag), negative = net
        # received (short-biased runs collect when funding is positive).
        direction = "net paid (drag)" if funding >= 0 else "net received"
        drag = getattr(m, "funding_drag_annualized", None) if m else None
        drag_txt = f"  |  ~{drag:+.2%}/yr of initial capital" if drag else ""
        lines.append(
            f"- Funding over the window: {funding:+.2f} USDT ({direction})"
            f"{drag_txt}"
        )
    stops = cs.get("perp_stop_triggers") or []
    if stops:
        lines.append(f"- Stop-trigger exits: {len(stops)}")
    elif cs.get("perp_stop_simulation"):
        lines.append("- Stop-trigger exits: 0")
    liqs = cs.get("perp_liquidations") or []
    if liqs:
        lines.append(f"- ⚠️ Liquidation events: {len(liqs)}")
        lines.append("")
        lines.append("| Date | Side | Entry | Liquidation price | Fee |")
        lines.append("|---|---|---:|---:|---:|")
        for ev in liqs:
            lines.append(
                f"| {ev.get('date', 'n/a')} | {ev.get('side', 'n/a')} | "
                f"{_fmt_price(ev.get('entry'))} | "
                f"{_fmt_price(ev.get('liquidation_price'))} | "
                f"{_fmt_num(ev.get('fee'))} |"
            )
    else:
        lines.append("- Liquidation events: 0")
    note = cs.get("perp_model_note")
    if note:
        lines.append(f"- Model: {note}")
    return lines


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
    lines.append(
        f"- Signal-to-fill lag: {result.config_summary.get('execution_lag_bars', 1)} "
        "bar(s), next available close"
    )
    lines.append(f"- Decisions: {len(result.trades)}  |  "
                 f"Actual rebalances: {sum(t.is_rebalance for t in result.trades)}  |  "
                 f"Position episodes: {m.get('num_trades', 0)}  |  "
                 f"Initial capital: {result.initial_capital:,.0f}  |  "
                 f"Transaction cost: {_cost_summary(result)}  |  "
                 f"Annualization: {result.config_summary.get('periods_per_year', 'n/a')}/yr")
    lines.append(f"- Cache: {result.cached_hits} hits / {result.cached_misses} misses")
    if result.degraded_decision_count:
        lines.append(
            f"- ⚠️ Degraded decisions: {result.degraded_decision_count} "
            f"(propagate failed or rating unparseable — forced to Hold)"
        )
    if result.unexecuted_decision_count:
        lines.append(
            f"- WARNING: Unexecuted decisions: {result.unexecuted_decision_count} "
            "(no strictly-later price bar was available)"
        )
    for warning in result.config_summary.get("risk_warnings", []):
        lines.append(f"- WARNING: {warning}")
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
    lines.append(f"| Information ratio vs B&H | {_fmt_num(m.get('information_ratio'))} | -- |")
    lines.append(f"| Tracking error vs B&H (ann.) | {_fmt_pct(m.get('tracking_error'))} | -- |")
    lines.append(f"| Win rate (net position episodes) | {_fmt_pct(m.get('win_rate'))} | n/a |")
    lines.append(f"| Profit factor | {_fmt_profit_factor(m.get('profit_factor'))} | n/a |")
    lines.append(f"| Avg win / avg loss | {_fmt_pct(m.get('avg_win'))} / {_fmt_pct(m.get('avg_loss'))} | n/a |")
    lines.append(f"| Payoff ratio | {_fmt_num(m.get('payoff_ratio'))} | n/a |")
    lines.append(f"| Turnover (ann.) | {_fmt_pct(m.get('turnover_annual'))} | n/a |")
    _mdd_date = m.get("max_drawdown_date")
    lines.append(f"| Max drawdown date | {_mdd_date if _mdd_date else 'n/a'} | n/a |")
    lines.append("")
    # Fama-French attribution section, only when the run opted in (factor_model
    # set). Shows factor-adjusted alpha, R² and each factor's loading so a
    # reader can tell how much of the return is common-factor beta vs residual.
    if m.get("factor_model"):
        betas = m.get("factor_betas") or {}
        lines += [
            "## Factor attribution",
            "",
            (f"- Model: **{m['factor_model']}**"
             f"  |  R²: {_fmt_num(m.get('factor_r_squared'))}"
             f"  |  Factor alpha (ann.): {_fmt_pct(m.get('factor_alpha'))}"),
            "",
            "| Factor | Beta |",
            "|---|---:|",
        ]
        for k in sorted(betas):
            lines.append(f"| {k} | {float(betas[k]):+.3f} |")
        lines.append("")
    # Event-study section, only when the run opted in and produced decidable
    # events. A beta-controlled test of whether the asset moved abnormally in
    # the holding window after each analyst decision (vs the naive
    # raw-minus-index alpha shown per-trade above).
    if m.get("event_study_n"):
        _es_ci = m.get("event_study_ci")
        _es_ci_str = (f"[{_fmt_num(_es_ci[0])}, {_fmt_num(_es_ci[1])}]"
                      if _es_ci else "n/a")
        lines += [
            "## Event study (market-model abnormal returns)",
            "",
            (f"- Benchmark: `{m.get('event_study_benchmark') or 'benchmark'}`"
             f"  |  Events used: {m['event_study_n']}"),
            "",
            "| Statistic | Value |",
            "|---|---:|",
            f"| Mean CAR | {_fmt_num(m.get('event_study_mean_car'))} |",
            f"| t-statistic | {_fmt_num(m.get('event_study_t_stat'))} |",
            f"| p-value (two-sided) | {_fmt_num(m.get('event_study_p_value'))} |",
            f"| Bootstrap 95% CI (mean CAR) | {_es_ci_str} |",
            "",
        ]
    lines.append(f"**Verdict: {result.ticker} strategy {beats} buy-and-hold on total return.**")
    lines.append("")
    lines.extend(_perp_section(result))
    if result.trades:
        lines.append("## Trades")
        lines.append("")
        lines.append(
            "| Signal date | Execution date | Rating | Order | Weight | Price | "
            "Position P&L | Asset fwd ret | Alpha | Stop |"
        )
        lines.append("|---|---|---|---|---:|---:|---:|---:|---:|---:|")
        for t in result.trades:
            price_str = _fmt_price(t.price)
            stop_str = _fmt_price(t.stop_loss)
            lines.append(
                f"| {t.date} | {t.execution_date or 'n/a'} | {t.rating} | "
                f"{'rebalance' if t.is_rebalance else 'hold/no order'} | "
                f"{t.executed_weight:.2f} | {price_str} | "
                f"{_fmt_pct(t.position_return)} | {_fmt_pct(t.raw_return)} | "
                f"{_fmt_pct(t.alpha_vs_index)} | {stop_str} |"
            )
            if t.risk_warning:
                lines.append(f"|  |  | Risk warning | {t.risk_warning} |  |  |  |  |  |  |")
        lines.append("")
    return "\n".join(lines)


def summarize_distribution(results: Sequence[BacktestResult]) -> dict[str, dict[str, float]]:
    """Aggregate key metrics across N runs to mean +/- std.

    Returns ``{metric: {"mean": ..., "std": ..., "min": ..., "max": ..., "n": int}}``.
    A metric is summarized over the runs that produced a non-None value for
    it — optional evidence (e.g. factor attribution) may exist for only a
    subset of runs, and ``n`` records that subset size so a single-run
    "distribution" can't masquerade as a stable one.
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
        "| Metric | mean | std | min | max | n |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    label_map = {
        "total_return": "Total return", "cagr": "CAGR", "sharpe": "Sharpe",
        "sortino": "Sortino", "max_drawdown": "Max drawdown", "calmar": "Calmar",
        "deflated_sharpe": "Deflated Sharpe", "alpha_vs_buyhold": "Alpha vs B&H (ann.)",
        "factor_alpha": "Factor alpha (ann.)",
    }
    pct_keys = {"total_return", "cagr", "max_drawdown", "alpha_vs_buyhold", "factor_alpha"}
    for key in _DISTRIBUTION_KEYS:
        if key not in summary:
            continue
        e = summary[key]
        fmt = (lambda x: _fmt_pct(x)) if key in pct_keys else (lambda x: _fmt_num(x))
        lines.append(
            f"| {label_map[key]} | {fmt(e['mean'])} | {fmt(e['std'])} | "
            f"{fmt(e['min'])} | {fmt(e['max'])} | {int(e['n'])} |"
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
