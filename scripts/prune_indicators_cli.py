#!/usr/bin/env python
"""Offline CLI: prune low-IC indicators from the market analyst's battery.

This is the **wiring point** for the self-improvement loop described in the
optimization plan: the ``ic.prune_indicators`` function existed but was called
by *no* runtime code. This script makes it reachable as an offline analysis
tool that produces a keep/prune report for human review.

Design constraints (from the optimization plan):
- **Fail-closed**: this tool NEVER edits the live config. It only produces a
  report + an optional config-diff suggestion. A human must review and apply.
- **Evidence-backed**: every prune decision is backed by a rolling-IC series
  that stayed below the threshold for N consecutive days.
- **No network, no LLM**: pure math over a CSV the user supplies.

Usage
-----
Prepare a CSV with columns: ``date``, ``forward_return``, and one column per
indicator (e.g. ``close_50_sma``, ``atr``, ``rsi_14``, ...). The
``forward_return`` column is the realized return over the holding horizon the
analyst is trying to predict. Then run::

    python scripts/prune_indicators_cli.py data/ic_sample.csv --window 60 \
        --min-abs-ic 0.03 --min-consecutive 30

The script prints a markdown report to stdout and, with ``--output``, writes it
to a file. With ``--suggest-config``, it additionally prints a Python config
snippet showing the kept ``indicator_battery`` list (the real config key the
market analyst reads its catalog from) — for manual review only, never
auto-applied. With ``--json-out``, it also writes a machine-readable verdict
(the same keep/prune decision plus per-indicator IC statistics) so downstream
tooling can consume the suggestion without parsing markdown.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from yiagents.backtest.ic import (
    build_ic_report,
    consecutive_below_threshold,
    factor_turnover,
    ic_decay,
    prune_indicators,
    quantile_spread,
    rolling_ic,
)

#: Columns ``export_ic_dataset.py --extra-horizons`` emits (e.g. fwd_ret_10d).
_FWD_RET_RE = re.compile(r"^fwd_ret_(\d+)d$")


def _load_ic_data(
    csv_path: str,
) -> tuple[pd.Series, dict[str, pd.Series], dict[str, pd.Series]]:
    """Load indicator values + forward returns from a CSV.

    Returns ``(forward_returns, {indicator_name: values},
    {horizon_label: forward_returns})``. The horizon map captures the optional
    ``fwd_ret_<h>d`` columns the exporter's ``--extra-horizons`` emits; empty
    for single-horizon CSVs (the diagnostics sections then stay hidden and
    the report keeps its original shape).
    """
    df = pd.read_csv(csv_path, parse_dates=["date"])
    if "date" not in df.columns:
        raise ValueError("CSV must have a 'date' column")
    if "forward_return" not in df.columns:
        raise ValueError("CSV must have a 'forward_return' column")

    forward = df["forward_return"]
    horizon_returns: dict[str, pd.Series] = {}
    reserved = {"date", "forward_return"}
    indicators: dict[str, pd.Series] = {}
    for col in df.columns:
        match = _FWD_RET_RE.match(col)
        if match:
            horizon_returns[f"{match.group(1)}d"] = df[col]
            reserved.add(col)
        elif col not in reserved:
            indicators[col] = df[col]
    if not indicators:
        raise ValueError("CSV must have at least one indicator column")
    return forward, indicators, horizon_returns


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Prune low-IC indicators from the market analyst battery.",
    )
    parser.add_argument(
        "csv",
        help="Path to a CSV with columns: date, forward_return, <indicator>...",
    )
    parser.add_argument(
        "--window", type=int, default=60,
        help="Rolling IC window in rows (default: 60)",
    )
    parser.add_argument(
        "--min-abs-ic", type=float, default=0.03,
        help="Abs-IC threshold below which an indicator is 'useless' (default: 0.03)",
    )
    parser.add_argument(
        "--min-consecutive", type=int, default=30,
        help="Consecutive low-IC days required to prune (default: 30)",
    )
    parser.add_argument(
        "--min-observations", type=int, default=30,
        help="Minimum finite IC windows to consider pruning (default: 30)",
    )
    parser.add_argument(
        "--output", "-o", type=str, default=None,
        help="Write the markdown report to this file (default: stdout only)",
    )
    parser.add_argument(
        "--json-out", type=str, default=None,
        help="Write a machine-readable verdict (params + keep/prune + "
        "per-indicator IC stats) to this JSON file. This is the structured "
        "hand-off for downstream tooling — applying it to the live config "
        "remains a human decision (fail-closed).",
    )
    parser.add_argument(
        "--suggest-config", action="store_true",
        help="Also print a Python config-diff suggestion (an indicator_battery "
        "snippet for default_config.py; manual review only)",
    )
    args = parser.parse_args(argv)

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"Error: {csv_path} not found", file=sys.stderr)
        return 1

    try:
        forward, indicators, horizon_returns = _load_ic_data(str(csv_path))
    except (ValueError, KeyError) as exc:
        print(f"Error loading CSV: {exc}", file=sys.stderr)
        return 1

    # Compute rolling IC for each indicator.
    ic_by_indicator: dict[str, pd.Series] = {}
    for name, values in indicators.items():
        ic_by_indicator[name] = rolling_ic(
            values, forward, window=args.window
        )

    # Diagnostics (evidence beyond level+persistence). All optional — every
    # section is hidden when the CSV carries no extra-horizon columns, so a
    # single-horizon run keeps its original report shape.
    decay_by_indicator: dict[str, dict[str, float | None]] = {}
    if horizon_returns:
        for name, values in indicators.items():
            decay_by_indicator[name] = ic_decay(values, horizon_returns)
    spread_by_indicator: dict[str, dict] = {}
    turnover_by_indicator: dict[str, float] = {}
    for name, values in indicators.items():
        qs = quantile_spread(values, forward)
        if qs is not None:
            spread_by_indicator[name] = qs
        tvr = factor_turnover(values)
        if tvr is not None:
            turnover_by_indicator[name] = tvr

    # Apply the pruning rule.
    result = prune_indicators(
        ic_by_indicator,
        min_abs_ic=args.min_abs_ic,
        min_consecutive_days=args.min_consecutive,
        min_observation_windows=args.min_observations,
    )

    report = build_ic_report(result, ic_by_indicator)

    if decay_by_indicator:
        horizon_labels = sorted(
            horizon_returns, key=lambda lab: int(lab.rstrip("d"))
        )
        lines = ["", "## IC decay (full-sample IC by forward horizon)", ""]
        lines.append("| Indicator | " + " | ".join(horizon_labels) + " |")
        lines.append("|---|" + "---:|" * len(horizon_labels))
        for name in indicators:
            row = decay_by_indicator.get(name, {})
            cells = []
            for lab in horizon_labels:
                ic = row.get(lab)
                cells.append(f"{ic:.3f}" if ic is not None else "n/a")
            lines.append(f"| {name} | " + " | ".join(cells) + " |")
        lines.append(
            "\nIC collapsing toward 0 as the horizon lengthens marks a "
            "short-lived signal; a flat curve supports slower rebalancing."
        )
        report += "\n".join(lines) + "\n"

    if spread_by_indicator:
        n_q = max(qs["n_quantiles"] for qs in spread_by_indicator.values())
        lines = ["", "## Quantile spread & turnover (primary horizon)", ""]
        header = ["Indicator"] + [f"Q{i+1}" for i in range(n_q)]
        header += ["Q_hi−Q_lo", "Monotonic", "Turnover"]
        lines.append("| " + " | ".join(header) + " |")
        lines.append("|---|" + "---:|" * (len(header) - 2) + "---|---:|")
        for name, qs in spread_by_indicator.items():
            cells = [name]
            for i in range(n_q):
                if i < len(qs["quantile_means"]):
                    cells.append(f"{qs['quantile_means'][i]:.4f}")
                else:
                    cells.append("—")
            cells.append(f"{qs['spread']:.4f}")
            cells.append("yes" if qs["monotonic"] else "no")
            cells.append(
                f"{turnover_by_indicator.get(name, float('nan')):.3f}"
                if name in turnover_by_indicator else "n/a"
            )
            lines.append("| " + " | ".join(cells) + " |")
        lines.append(
            "\nMonotone quantile means make a factor usable as a ranked "
            "long-short signal; a spread concentrated only in the tail "
            "quantiles means tail-only usefulness. Turnover near 1.0 = the "
            "ranking reshuffles almost every row (expensive to capture)."
        )
        report += "\n".join(lines) + "\n"

    report += (
        f"\n---\n\n_Window={args.window}, min|IC|={args.min_abs_ic}, "
        f"min_consecutive={args.min_consecutive}, "
        f"min_observations={args.min_observations}_\n"
    )

    print(report)

    if args.output:
        Path(args.output).write_text(report, encoding="utf-8")
        print(f"\nReport written to {args.output}", file=sys.stderr)

    if args.json_out:
        # Machine-readable verdict: same decision, plus the per-indicator
        # evidence behind it. Consumed by tooling; applying the prune to the
        # live config stays a human step (fail-closed contract).
        per_indicator: dict[str, dict] = {}
        for name, series in ic_by_indicator.items():
            finite = series.dropna()
            finite_abs = finite.abs()
            entry: dict = {
                "verdict": "prune" if name in set(result["prune"]) else "keep",
                "mean_abs_ic": (
                    float(finite_abs.mean()) if not finite_abs.empty else None
                ),
                "finite_windows": int(len(finite)),
                "longest_low_run": int(
                    consecutive_below_threshold(
                        series, threshold=args.min_abs_ic,
                        min_consecutive=args.min_consecutive,
                    )
                ),
            }
            if name in decay_by_indicator:
                entry["ic_by_horizon"] = decay_by_indicator[name]
            if name in spread_by_indicator:
                entry["quantile_means"] = spread_by_indicator[name]["quantile_means"]
                entry["quantile_spread"] = spread_by_indicator[name]["spread"]
                entry["monotonic"] = spread_by_indicator[name]["monotonic"]
            if name in turnover_by_indicator:
                entry["turnover"] = turnover_by_indicator[name]
            per_indicator[name] = entry
        verdict = {
            "generated_at": datetime.now(UTC).isoformat(),
            "params": {
                "window": args.window,
                "min_abs_ic": args.min_abs_ic,
                "min_consecutive": args.min_consecutive,
                "min_observations": args.min_observations,
            },
            "keep": list(result["keep"]),
            "prune": list(result["prune"]),
            "per_indicator": per_indicator,
        }
        Path(args.json_out).write_text(
            json.dumps(verdict, indent=2), encoding="utf-8"
        )
        print(f"\nJSON verdict written to {args.json_out}", file=sys.stderr)

    if args.suggest_config and result["prune"]:
        # Suggest the kept names in the CSV's own column order — only the
        # indicators that were actually evaluated belong in the suggestion.
        kept = [name for name in indicators if name not in set(result["prune"])]
        print("\n## Config suggestion (REVIEW BEFORE APPLYING)\n")
        print("The following indicators are candidates for removal from the")
        print("market analyst's catalog. After review, set `indicator_battery`")
        print("(in default_config.py or via set_config) to the kept names:\n")
        print("```python")
        print("# default_config.py")
        print(f'"indicator_battery": {kept},')
        print("```")
        print(
            "\n⚠️ This is a suggestion only. Verify against out-of-sample data"
            " and apply manually — never auto-edit the live config."
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
