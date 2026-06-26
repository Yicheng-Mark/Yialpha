"""Backtesting harness for YiAgents (Phase 0).

A deterministic engine that runs the agent graph over historical dates,
simulates positions from the 5-tier rating, and produces an equity curve plus a
full metric suite (Sharpe / MDD / Calmar / Sortino / Deflated Sharpe /
vs buy-and-hold). Everything the later phases A/B test against is built here.
"""

from yiagents.backtest.cache import CachedDecision, DecisionCache
from yiagents.backtest.engine import (
    DEFAULT_RATING_TO_WEIGHT,
    BacktestResult,
    TradeRow,
    run_backtest,
)
from yiagents.backtest.metrics import BacktestMetrics, compute_metrics
from yiagents.backtest.report import (
    multi_run,
    render_backtest_report,
    summarize_distribution,
    write_report,
)

__all__ = [
    "DEFAULT_RATING_TO_WEIGHT",
    "BacktestResult",
    "TradeRow",
    "run_backtest",
    "BacktestMetrics",
    "compute_metrics",
    "DecisionCache",
    "CachedDecision",
    "render_backtest_report",
    "write_report",
    "summarize_distribution",
    "multi_run",
]
