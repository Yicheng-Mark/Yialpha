"""Quantitative risk-control layer (Phase 1).

The framework's decision nodes (Trader / Portfolio Manager) historically let
the LLM fill ``position_sizing`` and ``stop_loss`` as free text. This package
replaces that with deterministic math: Kelly fraction sizing scaled by the
rating's confidence, ATR-based stops, a portfolio drawdown breaker with
exposure caps, and an optional CVaR monitor. The decision nodes keep the LLM
for *direction* and let this layer own *size and risk*.

Public surface
--------------
:class:`RiskManager` orchestrates the four sub-modules into one overlay.
:func:`build_backtest_weight_fn` adapts it to the backtest engine so the same
rules drive both the A/B harness and the live decision flow.
"""

from tradingagents.risk.atr_stop import atr_stop, latest_atr, latest_atr_from_frame
from tradingagents.risk.breaker import BreakerState, DrawdownBreaker
from tradingagents.risk.cvar import cvar_position_multiplier, historical_cvar
from tradingagents.risk.kelly import (
    RATING_TO_BAND,
    RATING_TO_CONFIDENCE,
    bayesian_win_rate,
    kelly_fraction,
    kelly_sizing,
)
from tradingagents.risk.manager import (
    PortfolioState,
    RiskDecision,
    RiskManager,
    build_backtest_weight_fn,
)

__all__ = [
    # Kelly
    "RATING_TO_BAND",
    "RATING_TO_CONFIDENCE",
    "bayesian_win_rate",
    "kelly_fraction",
    "kelly_sizing",
    # ATR
    "atr_stop",
    "latest_atr",
    "latest_atr_from_frame",
    # Breaker
    "BreakerState",
    "DrawdownBreaker",
    # CVaR
    "cvar_position_multiplier",
    "historical_cvar",
    # Orchestrator
    "PortfolioState",
    "RiskDecision",
    "RiskManager",
    "build_backtest_weight_fn",
]
