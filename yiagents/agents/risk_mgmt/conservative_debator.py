from yiagents.agents.utils.agent_utils import (
    build_risk_debate_update,
    get_instrument_context_from_state,
    get_language_instruction,
)
from yiagents.dataflows.config import get_config
from yiagents.dataflows.market_regime import format_market_regime


def create_conservative_debator(llm):
    def conservative_node(state) -> dict:
        risk_debate_state = state["risk_debate_state"]
        history = risk_debate_state.get("history", "")

        current_aggressive_response = risk_debate_state.get("current_aggressive_response", "")
        current_neutral_response = risk_debate_state.get("current_neutral_response", "")

        market_research_report = state["market_report"]
        sentiment_report = state["sentiment_report"]
        news_report = state["news_report"]
        fundamentals_report = state["fundamentals_report"]
        instrument_context = get_instrument_context_from_state(state)

        # Opt-in market-stress reading (env: YIAGENTS_MARKET_REGIME, default
        # off). When off this stays "" and the prompt below is byte-identical
        # to the baseline. When on, a fail-soft benchmark turbulence line is
        # appended — a market-level ex-ante cue complementary to the
        # portfolio-level reactive DrawdownBreaker.
        market_regime_line = ""
        if get_config().get("market_regime"):
            trade_date = state.get("trade_date")
            if trade_date:
                reading = format_market_regime(
                    state["company_of_interest"], trade_date
                )
                if reading:
                    market_regime_line = f"Market Regime: {reading}\n"
                else:
                    # Configured but unavailable (fetch failed / too little
                    # history): say so explicitly instead of silently omitting
                    # the cue the operator opted into.
                    market_regime_line = (
                        "Market Regime: unavailable (fetch failed or "
                        "insufficient benchmark history)\n"
                    )

        trader_decision = state["trader_investment_plan"]

        prompt = f"""As the Conservative Risk Analyst, your primary objective is to protect assets, minimize volatility, and ensure steady, reliable growth. You prioritize stability, security, and risk mitigation, carefully assessing potential losses, economic downturns, and market volatility. When evaluating the trader's decision or plan, critically examine high-risk elements, pointing out where the decision may expose the firm to undue risk and where more cautious alternatives could secure long-term gains. Here is the trader's decision:

{trader_decision}

Your task is to actively counter the arguments of the Aggressive and Neutral Analysts, highlighting where their views may overlook potential threats or fail to prioritize sustainability. Respond directly to their points, drawing from the following data sources to build a convincing case for a low-risk approach adjustment to the trader's decision:

{instrument_context}
Market Research Report: {market_research_report}
Social Media Sentiment Report: {sentiment_report}
Latest World Affairs Report: {news_report}
Company Fundamentals Report: {fundamentals_report}
{market_regime_line}Here is the current conversation history: {history} Here is the last response from the aggressive analyst: {current_aggressive_response} Here is the last response from the neutral analyst: {current_neutral_response}. If there are no responses from the other viewpoints yet, present your own argument based on the available data.

Engage by questioning their optimism and emphasizing the potential downsides they may have overlooked. Address each of their counterpoints to showcase why a conservative stance is ultimately the safest path for the firm's assets. Focus on debating and critiquing their arguments to demonstrate the strength of a low-risk strategy over their approaches. Output conversationally as if you are speaking without any special formatting.""" + get_language_instruction()

        response = llm.invoke(prompt)

        argument = f"Conservative Analyst: {response.content}"

        new_risk_debate_state = build_risk_debate_update(
            risk_debate_state, "conservative", argument
        )

        return {"risk_debate_state": new_risk_debate_state}

    return conservative_node
