"""Portfolio Manager: synthesises the risk-analyst debate into the final decision.

Uses LangChain's ``with_structured_output`` so the LLM produces a typed
``PortfolioDecision`` directly, in a single call.  The result is rendered
back to markdown for storage in ``final_trade_decision`` so memory log,
CLI display, and saved reports continue to consume the same shape they do
today.  When a provider does not expose structured output, the agent falls
back gracefully to free-text generation.
"""

from __future__ import annotations

from yiagents.agents.schemas import PortfolioDecision, render_pm_decision
from yiagents.agents.utils.agent_utils import (
    get_instrument_context_from_state,
    get_language_instruction,
)
from yiagents.agents.utils.structured import (
    bind_structured,
    invoke_structured_or_freetext,
)


def _format_portfolio_state(portfolio_state) -> str:
    """Render an optional portfolio snapshot for the PM prompt.

    Accepts either a :class:`~yiagents.risk.manager.PortfolioState` or a
    plain dict with the same fields. Returns a single bulleted line, or "" when
    there is nothing useful to say (keeps the prompt unchanged for baseline
    runs that carry no portfolio context).
    """
    if portfolio_state is None:
        return ""

    def _get(key, default=None):
        if isinstance(portfolio_state, dict):
            return portfolio_state.get(key, default)
        return getattr(portfolio_state, key, default)

    equity = _get("equity")
    cash = _get("cash")
    positions = _get("positions") or {}
    if equity is None and cash is None and not positions:
        return ""

    parts = []
    if equity is not None:
        parts.append(f"Equity: {float(equity):,.0f}")
    if cash is not None:
        parts.append(f"Cash: {float(cash):,.0f}")
    if positions:
        # Guard each value: an external schema may pass non-numeric holdings
        # (e.g. {"NVDA": "100 shares"} or nested dicts). A ValueError/TypeError
        # here used to propagate and fail the whole PM node, so coerce to None
        # and drop the row instead. Numeric inputs render byte-identically.
        def _qty(k, v):
            q = _safe_float(v)
            return None if q is None or q == 0 else f"{k} {q:,.0f}"

        holdings = ", ".join(q for q in (_qty(k, v) for k, v in positions.items()) if q)
        if holdings:
            parts.append(f"Holdings: {holdings}")
    if not parts:
        return ""
    return "- Current portfolio: " + " | ".join(parts) + "\n"


def _safe_float(value) -> float | None:
    """Coerce ``value`` to float, returning ``None`` on any non-numeric input.

    Guards the portfolio-state renderer against schemas that pass strings
    (``"100 shares"``) or nested structures as holding quantities. Pure
    numeric inputs round-trip exactly, so well-formed portfolio_state renders
    byte-identically to the unguarded path.
    """
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def create_portfolio_manager(llm):
    structured_llm = bind_structured(llm, PortfolioDecision, "Portfolio Manager")

    def portfolio_manager_node(state) -> dict:
        instrument_context = get_instrument_context_from_state(state)

        history = state["risk_debate_state"]["history"]
        risk_debate_state = state["risk_debate_state"]
        research_plan = state["investment_plan"]
        trader_plan = state["trader_investment_plan"]

        past_context = state.get("past_context", "")
        lessons_line = (
            f"- Lessons from prior decisions and outcomes:\n{past_context}\n"
            if past_context
            else ""
        )

        # Optional live portfolio snapshot (Phase 1). Absent on baseline runs,
        # so the prompt -- and the model's behaviour -- is unchanged there.
        portfolio_line = _format_portfolio_state(state.get("portfolio_state"))

        prompt = f"""As the Portfolio Manager, synthesize the risk analysts' debate and deliver the final trading decision.

{instrument_context}

---

**Rating Scale** (use exactly one):
- **Buy**: Strong conviction to enter or add to position
- **Overweight**: Favorable outlook, gradually increase exposure
- **Hold**: Maintain current position, no action needed
- **Underweight**: Reduce exposure, take partial profits
- **Sell**: Exit position or avoid entry

**Context:**
- Research Manager's investment plan: **{research_plan}**
- Trader's transaction proposal: **{trader_plan}**
{lessons_line}{portfolio_line}
**Risk Analysts Debate History:**
{history}

---

Be decisive and ground every conclusion in specific evidence from the analysts.{get_language_instruction()}"""

        final_trade_decision = invoke_structured_or_freetext(
            structured_llm,
            llm,
            prompt,
            render_pm_decision,
            "Portfolio Manager",
        )

        new_risk_debate_state = {
            "judge_decision": final_trade_decision,
            "history": risk_debate_state["history"],
            "aggressive_history": risk_debate_state["aggressive_history"],
            "conservative_history": risk_debate_state["conservative_history"],
            "neutral_history": risk_debate_state["neutral_history"],
            "latest_speaker": "Judge",
            "current_aggressive_response": risk_debate_state["current_aggressive_response"],
            "current_conservative_response": risk_debate_state["current_conservative_response"],
            "current_neutral_response": risk_debate_state["current_neutral_response"],
            "count": risk_debate_state["count"],
        }

        return {
            "risk_debate_state": new_risk_debate_state,
            "final_trade_decision": final_trade_decision,
        }

    return portfolio_manager_node
