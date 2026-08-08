# yiagents/graph/conditional_logic.py

from yiagents.agents.utils.agent_states import AgentState


class ConditionalLogic:
    """Handles conditional logic for determining graph flow."""

    def __init__(self, max_debate_rounds: int = 1, max_risk_discuss_rounds: int = 1):
        """Initialize with configuration parameters."""
        self.max_debate_rounds = max_debate_rounds
        self.max_risk_discuss_rounds = max_risk_discuss_rounds

    @staticmethod
    def _route_on_tool_calls(
        state: AgentState, tool_node: str, clear_node: str
    ) -> str:
        """Shared router for the four analyst ``should_continue_*`` methods.

        If the last message carries pending tool calls, route to the tool node;
        otherwise route to the message-clear node. Guards against an empty
        ``messages`` list (e.g. an abnormal state) by falling through to the
        clear node instead of raising ``IndexError`` and crashing the graph.
        """
        messages = state["messages"]
        if not messages:
            return clear_node
        last_message = messages[-1]
        if getattr(last_message, "tool_calls", None):
            return tool_node
        return clear_node

    def should_continue_market(self, state: AgentState) -> str:
        """Determine if market analysis should continue."""
        return self._route_on_tool_calls(state, "tools_market", "Msg Clear Market")

    def should_continue_social(self, state: AgentState) -> str:
        """Determine if sentiment-analyst tool round should continue.

        Method name keeps the legacy ``social`` suffix to match the
        ``AnalystType.SOCIAL = "social"`` wire value (saved-config
        back-compat); the returned ``clear_node`` label uses the v0.2.5
        rename so it matches the node registered by the execution plan.
        """
        return self._route_on_tool_calls(state, "tools_social", "Msg Clear Sentiment")

    def should_continue_news(self, state: AgentState) -> str:
        """Determine if news analysis should continue."""
        return self._route_on_tool_calls(state, "tools_news", "Msg Clear News")

    def should_continue_fundamentals(self, state: AgentState) -> str:
        """Determine if fundamentals analysis should continue."""
        return self._route_on_tool_calls(state, "tools_fundamentals", "Msg Clear Fundamentals")

    def should_continue_debate(self, state: AgentState) -> str:
        """Determine if debate should continue.

        Speaker alternation is driven by ``count`` parity, NOT by inspecting
        ``current_response.startswith("Bull")``. The two are equivalent under
        the normal flow (Bull speaks first at count=0, Bear at count=1, …) but
        parity is robust to LLM output that fails to carry the ``"Bull
        Analyst:"`` / ``"Bear Analyst:"`` prefix, which the text-match route
        relied on. This mirrors the risk debate's explicit ``latest_speaker``
        approach.
        """

        count = state["investment_debate_state"]["count"]
        if count >= 2 * self.max_debate_rounds:
            # Each round is one Bull + one Bear turn, so 2*max rounds total.
            return "Research Manager"
        # Even count -> Bull speaks next; odd -> Bear.
        return "Bear Researcher" if count % 2 == 1 else "Bull Researcher"

    def should_continue_risk_analysis(self, state: AgentState) -> str:
        """Determine if risk analysis should continue."""
        if (
            state["risk_debate_state"]["count"] >= 3 * self.max_risk_discuss_rounds
        ):  # 3 rounds of back-and-forth between 3 agents
            return "Portfolio Manager"
        if state["risk_debate_state"]["latest_speaker"].startswith("Aggressive"):
            return "Conservative Analyst"
        if state["risk_debate_state"]["latest_speaker"].startswith("Conservative"):
            return "Neutral Analyst"
        return "Aggressive Analyst"
