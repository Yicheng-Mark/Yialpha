"""Monitoring dashboard for Phase 4.

Static HTML generation (no server process): an equity curve as inline SVG,
the metric suite, a drawdown alert, and an optional current-positions table.
Open the file in Edge, or have the CLI/Playwright screenshot it on a schedule.
"""

from tradingagents.monitoring.dashboard import (
    render_dashboard,
    write_dashboard,
)

__all__ = ["render_dashboard", "write_dashboard"]
