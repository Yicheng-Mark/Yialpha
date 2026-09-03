"""Portfolio Manager: synthesises the risk-analyst debate into the final decision.

Uses LangChain's ``with_structured_output`` so the LLM produces a typed
``PortfolioDecision`` directly, in a single call.  The result is rendered
back to markdown for storage in ``final_trade_decision`` so memory log,
CLI display, and saved reports continue to consume the same shape they do
today.  When a provider does not expose structured output, the agent falls
back gracefully to free-text generation.
"""

from __future__ import annotations

from yialpha.agents.schemas import PortfolioDecision, render_pm_decision
from yialpha.agents.utils.agent_utils import (
    get_instrument_context_from_state,
    get_language_instruction,
)
from yialpha.agents.utils.structured import (
    NO_EXTERNAL_TOOLS,
    STRUCTURED_FALLBACK,
    bind_structured,
    invoke_structured_or_freetext,
)
from yialpha.versions import SCHEMA_VERSION


def _format_portfolio_state(portfolio_state) -> str:
    """Render an optional portfolio snapshot for the PM prompt.

    Accepts either a :class:`~yialpha.risk.manager.PortfolioState` or a
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


def _decision_fields_dict(decision) -> dict:
    """Flatten the typed PortfolioDecision into the plain dict state carries.

    V2.0: the deterministic layers (ExecutionTicket builder, evidence log)
    read the PM's numeric/opinion fields from here — no markdown re-parsing.
    ``decision`` is the parsed Pydantic instance or None (free-text fallback /
    structured mode unavailable); None yields {} so consumers treat every
    field as absent rather than guessed. ``schema_version`` rides along per
    the V2 baseline's version-everything rule.
    """
    if decision is None:
        return {}
    probabilities = None
    if decision.probabilities is not None:
        probabilities = {
            "bull": decision.probabilities.bull,
            "neutral": decision.probabilities.neutral,
            "bear": decision.probabilities.bear,
        }
    fields = {
        "rating": decision.rating.value,
        "price_target": decision.price_target,
        "time_horizon": decision.time_horizon,
        "confidence": decision.confidence,
        "probabilities": probabilities,
        "expected_return": decision.expected_return,
        "invalidation": list(decision.invalidation) if decision.invalidation else None,
        "evidence_coverage": decision.evidence_coverage,
        "schema_version": SCHEMA_VERSION,
    }
    # V2.1 fair-value linkage + V2.3 dual-view fields (additive). Only
    # NON-None values ride along, so a PM that does not emit them keeps
    # pm_decision_fields — and therefore the logged state JSON —
    # byte-identical to the pre-V2.1 shape.
    for optional_key in (
        "price_target_currency",
        "price_target_basis",
        "underlying_price_target",
        "underlying_target_currency",
        "underlying_direction",
        "contract_direction",
        "basis_view",
        "desired_side",
    ):
        value = getattr(decision, optional_key, None)
        if value is not None:
            fields[optional_key] = value
    return fields


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

Be decisive and ground every conclusion in specific evidence from the analysts.{get_language_instruction()}""" + NO_EXTERNAL_TOOLS

        # The typed decision is stashed by the extract callback (below) so the
        # V2.0 field snapshot and the rating come from ONE parse — there is no
        # second extraction pass to drift out of sync.
        captured: dict = {}

        def _extract(decision) -> str:
            captured["decision"] = decision
            return decision.rating.value

        final_trade_decision, pm_rating = invoke_structured_or_freetext(
            structured_llm,
            llm,
            prompt,
            render_pm_decision,
            "Portfolio Manager",
            # Extract the structured rating directly so the risk overlay can read
            # it from state without re-parsing the rendered markdown. On a
            # free-text fallback pm_rating is STRUCTURED_FALLBACK -> the rating
            # is unknown and the overlay must fall back to parse_rating (see
            # trading_graph._apply_risk_overlay).
            extract=_extract,
        )

        # Detect the structured-output degradation path. When the PM fell back to
        # free text, pm_rating is STRUCTURED_FALLBACK (falsy, identity-distinct
        # from None). We flag this in the decision text so it is never silent:
        # a weak model that degrades to regex-parsed ratings is now observable.
        structured_degraded = pm_rating is STRUCTURED_FALLBACK
        if structured_degraded:
            final_trade_decision = (
                final_trade_decision
                + "\n\n---\n\n"
                + "> ⚠️ **Structured-output fallback**: the PM's typed "
                "``PortfolioDecision`` could not be parsed; the rating above "
                "was extracted from free text via regex. Treat the rating with "
                "caution."
            )

        # V2.0: snapshot the typed decision's fields for the ticket builder /
        # evidence log. {} when structured output did not happen — downstream
        # treats missing price_target as UNEVALUATED, never guesses.
        pm_decision_fields = _decision_fields_dict(captured.get("decision"))

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
            "pm_rating": pm_rating if pm_rating and not structured_degraded else "",
            "pm_decision_fields": pm_decision_fields,
        }

    return portfolio_manager_node
