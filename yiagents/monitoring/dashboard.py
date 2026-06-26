"""Static-HTML monitoring dashboard generator (Phase 4).

Produces a single self-contained ``.html`` file: an inline-SVG equity curve
(no CDN / no network), the metric table, a drawdown alert when losses deepen,
an optional current-positions table, and a kill-switch banner. The file opens
in Edge or any browser; a Playwright/CLI loop can screenshot it on a schedule.

Input is :class:`~yiagents.backtest.engine.BacktestResult` objects (one or
many) plus an optional live portfolio snapshot. Pure string building -- no
templates engine, no runtime server.
"""

from __future__ import annotations

import html
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from yiagents.backtest.engine import BacktestResult

logger = logging.getLogger(__name__)

_DRAWDOWN_ALERT_THRESHOLD = 0.10  # flag red when max drawdown breaches 10%


def _fmt_pct(x: float | None) -> str:
    if x is None:
        return "n/a"
    return f"{x * 100:.2f}%"


def _fmt_num(x: float | None) -> str:
    if x is None:
        return "n/a"
    return f"{x:.3f}"


def _equity_svg(equity: Sequence[float], width: int = 760, height: int = 220) -> str:
    """Render an equity curve as an inline SVG polyline (self-contained)."""
    if not equity:
        return '<p class="muted">No equity data.</p>'
    lo = min(equity)
    hi = max(equity)
    span = (hi - lo) or 1.0
    n = len(equity)
    pad = 8
    def _x(i: int) -> float:
        return pad + (width - 2 * pad) * (i / (n - 1) if n > 1 else 0.5)
    def _y(v: float) -> float:
        return height - pad - (height - 2 * pad) * ((v - lo) / span)
    pts = " ".join(f"{_x(i):.1f},{_y(v):.1f}" for i, v in enumerate(equity))
    base_y = _y(equity[0])
    return (
        f'<svg width="{width}" height="{height}" role="img" '
        f'aria-label="equity curve" class="equity">'
        f'<rect width="100%" height="100%" fill="#0d1117"/>'
        f'<line x1="{pad}" y1="{base_y:.1f}" x2="{width - pad}" y2="{base_y:.1f}" '
        f'stroke="#30363d" stroke-dasharray="4 4"/>'
        f'<polyline points="{pts}" fill="none" stroke="#2ea043" '
        f'stroke-width="2"/>'
        f'</svg>'
    )


def _metrics_table(result: BacktestResult) -> str:
    m = result.metrics
    if m is None:
        return '<p class="muted">No metrics available.</p>'
    rows = [
        ("Total return", _fmt_pct(m.total_return)),
        ("CAGR", _fmt_pct(m.cagr)),
        ("Volatility (ann.)", _fmt_pct(m.volatility)),
        ("Sharpe", _fmt_num(m.sharpe)),
        ("Sortino", _fmt_num(m.sortino)),
        ("Max drawdown", _fmt_pct(m.max_drawdown)),
        ("Calmar", _fmt_num(m.calmar)),
        ("Deflated Sharpe", _fmt_num(m.deflated_sharpe)),
        ("Alpha vs B&H", _fmt_pct(m.alpha_vs_buyhold)),
    ]
    body = "".join(f"<tr><td>{html.escape(k)}</td><td>{html.escape(v)}</td></tr>" for k, v in rows)
    return (
        '<table class="metrics"><thead><tr><th>Metric</th><th>Value</th></tr></thead>'
        f'<tbody>{body}</tbody></table>'
    )


def _positions_table(portfolio_state: Any) -> str:
    if not portfolio_state:
        return ""
    positions = (
        portfolio_state.get("positions") if isinstance(portfolio_state, dict)
        else getattr(portfolio_state, "positions", None)
    )
    if not positions:
        return ""
    body = "".join(
        f"<tr><td>{html.escape(str(k))}</td><td>{float(v):,.0f}</td></tr>"
        for k, v in positions.items() if float(v) != 0
    )
    if not body:
        return ""
    return (
        '<h2>Current positions</h2><table class="pos">'
        "<thead><tr><th>Ticker</th><th>Market value</th></tr></thead>"
        f"<tbody>{body}</tbody></table>"
    )


def render_dashboard(
    results: BacktestResult | Sequence[BacktestResult],
    portfolio_state: Any = None,
    kill_switch: bool = False,
    title: str = "YiAgents Monitor",
) -> str:
    """Render the monitoring dashboard to an HTML string.

    ``kill_switch`` shows a halt banner when True. ``portfolio_state`` (a dict or
    object with a ``positions`` mapping) adds a current-positions table.
    """
    if isinstance(results, BacktestResult):
        results = [results]
    results = list(results)
    primary = results[0] if results else None

    kill_banner = (
        '<div class="halt">KILL SWITCH ENGAGED — new order submission is blocked.</div>'
        if kill_switch else ""
    )

    # Drawdown alert off the primary run's max drawdown.
    alert = ""
    if primary is not None and primary.metrics is not None:
        mdd = primary.metrics.max_drawdown
        if mdd is not None and mdd <= -_DRAWDOWN_ALERT_THRESHOLD:
            alert = (
                f'<div class="alert">Drawdown alert: max drawdown is '
                f'{_fmt_pct(mdd)} (beyond {_DRAWDOWN_ALERT_THRESHOLD:.0%}). '
                f'Review the risk breaker before adding exposure.</div>'
            )

    blocks: list[str] = []
    if primary is not None:
        blocks.append(f"<h2>{html.escape(primary.ticker)} equity curve</h2>")
        blocks.append(_equity_svg(primary.equity))
        blocks.append("<h2>Metrics</h2>")
        blocks.append(_metrics_table(primary))
    else:
        blocks.append('<p class="muted">No backtest result supplied.</p>')

    if portfolio_state is not None:
        blocks.append(_positions_table(portfolio_state))

    if len(results) > 1:
        rows = "".join(
            f"<tr><td>{html.escape(r.ticker)}</td><td>{_fmt_pct(r.metrics.total_return)}</td>"
            f"<td>{_fmt_pct(r.metrics.max_drawdown)}</td>"
            f"<td>{_fmt_num(r.metrics.deflated_sharpe)}</td></tr>"
            for r in results if r.metrics
        )
        blocks.append(
            "<h2>Runs</h2><table class='runs'>"
            "<thead><tr><th>Ticker</th><th>Total return</th><th>Max DD</th><th>DSR</th></tr></thead>"
            f"<tbody>{rows}</tbody></table>"
        )

    body_html = "\n".join(blocks)
    return _HTML_TEMPLATE.format(
        title=html.escape(title),
        kill_banner=kill_banner,
        alert=alert,
        body=body_html,
    )


def write_dashboard(
    results: BacktestResult | Sequence[BacktestResult],
    path: str | Path | None = None,
    results_dir: str | Path | None = None,
    portfolio_state: Any = None,
    kill_switch: bool = False,
) -> Path:
    """Render and write the dashboard to disk; return its path."""
    if path is None:
        base = Path(results_dir) if results_dir else Path(".")
        path = base / "monitoring" / "dashboard.html"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        render_dashboard(results, portfolio_state=portfolio_state, kill_switch=kill_switch),
        encoding="utf-8",
    )
    logger.info("Wrote monitoring dashboard to %s", path)
    return path


_HTML_TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>{title}</title>
<style>
  body {{ font-family: -apple-system, Segoe UI, sans-serif; background: #010409; color: #c9d1d9; margin: 24px; }}
  h1, h2 {{ color: #f0f6fc; }}
  table {{ border-collapse: collapse; margin: 8px 0 20px; }}
  th, td {{ border: 1px solid #30363d; padding: 6px 12px; text-align: left; }}
  th {{ background: #161b22; }}
  .metrics td:last-child, .pos td:last-child, .runs td:nth-child(n+2) {{ text-align: right; font-variant-numeric: tabular-nums; }}
  .equity {{ display: block; border: 1px solid #30363d; }}
  .muted {{ color: #8b949e; }}
  .halt {{ background: #da3633; color: #fff; padding: 12px; border-radius: 6px; font-weight: bold; margin-bottom: 16px; }}
  .alert {{ background: #bb8009; color: #211a00; padding: 12px; border-radius: 6px; margin-bottom: 16px; }}
</style></head>
<body>
<h1>{title}</h1>
{kill_banner}{alert}
{body}
</body></html>
"""
