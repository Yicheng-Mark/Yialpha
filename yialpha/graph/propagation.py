# yialpha/graph/propagation.py
"""Initial-state construction and graph-invocation arg assembly.

The class name ``Propagator`` is historical: this module does NOT perform
mid-graph state propagation. Field merging between nodes is handled by
LangGraph's ``Annotated`` reducers on ``AgentState`` (last-write-wins for
scalar fields; additive for ``messages``) — see
``yialpha/agents/utils/agent_states.py`` for the full ownership map. The two
debate sub-states are rebuilt wholesale by each speaker via
``build_investment_debate_update`` / ``build_risk_debate_update``, which are
the compensating layer for the absence of field-level reducers.

What this module actually owns:
  * :meth:`Propagator.create_initial_state` — seeds every key downstream nodes
    read via direct indexing (checkpoint-resume robustness).
  * :meth:`Propagator.get_graph_args` — assembles the ``stream_mode`` +
    ``recursion_limit`` config passed to ``graph.invoke``.
"""

from typing import Any

from yialpha.agents.utils.agent_states import (
    InvestDebateState,
    RiskDebateState,
)


class Propagator:
    """Handles state initialization and graph-invocation arg assembly."""

    def __init__(self, max_recur_limit=100):
        """Initialize with configuration parameters."""
        self.max_recur_limit = max_recur_limit

    def create_initial_state(
        self,
        company_name: str,
        trade_date: str,
        asset_type: str = "stock",
        past_context: str = "",
        instrument_context: str = "",
        portfolio_state=None,
    ) -> dict[str, Any]:
        """Create the initial state for the agent graph.

        ``instrument_context`` is the deterministic ticker-identity string
        resolved once at run start (see
        ``YiAlphaGraph.resolve_instrument_context``). When empty, agents
        fall back to ticker-only context via
        ``get_instrument_context_from_state``.

        ``portfolio_state`` is an optional live snapshot (Phase 1) the Portfolio
        Manager reads to size decisions against the actual book. ``None`` keeps
        baseline behaviour unchanged.
        """
        return {
            "messages": [("human", company_name)],
            "company_of_interest": company_name,
            "asset_type": asset_type,
            "instrument_context": instrument_context,
            "trade_date": str(trade_date),
            "past_context": past_context,
            "portfolio_state": portfolio_state,
            "investment_debate_state": InvestDebateState(
                {
                    "bull_history": "",
                    "bear_history": "",
                    "history": "",
                    "current_response": "",
                    "judge_decision": "",
                    "count": 0,
                }
            ),
            "risk_debate_state": RiskDebateState(
                {
                    "aggressive_history": "",
                    "conservative_history": "",
                    "neutral_history": "",
                    "history": "",
                    "latest_speaker": "",
                    "current_aggressive_response": "",
                    "current_conservative_response": "",
                    "current_neutral_response": "",
                    "judge_decision": "",
                    "count": 0,
                }
            ),
            "market_report": "",
            "fundamentals_report": "",
            "sentiment_report": "",
            "news_report": "",
            # Downstream nodes read these via direct ``state[key]`` indexing
            # (trader, portfolio_manager, risk debators). In the normal flow each
            # is written by its upstream node before it is read, so the absence
            # of a default is harmless. But under checkpoint resume — if a run
            # crashes before, say, the Research Manager writes
            # ``investment_plan`` — the resumed state would lack the key and the
            # Trader node would raise ``KeyError``, masking the original crash.
            # Initialising them to "" makes the resume path degrade gracefully.
            "investment_plan": "",
            "trader_investment_plan": "",
            "final_trade_decision": "",
            # pm_rating is "" when the PM hasn't run or fell back to free text;
            # the risk overlay then falls back to parse_rating on the markdown.
            "pm_rating": "",
            # V2.0: PM decision fields for the deterministic layers (ticket
            # builder reads price_target/confidence). Empty until the PM node
            # fills it; {} on free-text fallback, mirroring pm_rating.
            "pm_decision_fields": {},
        }

    def get_graph_args(self, callbacks: list | None = None) -> dict[str, Any]:
        """Get arguments for the graph invocation.

        Args:
            callbacks: Optional list of callback handlers for tool execution tracking.
                       Note: LLM callbacks are handled separately via LLM constructor.
        """
        config = {"recursion_limit": self.max_recur_limit}
        if callbacks:
            config["callbacks"] = callbacks
        return {
            "stream_mode": "values",
            "config": config,
        }
