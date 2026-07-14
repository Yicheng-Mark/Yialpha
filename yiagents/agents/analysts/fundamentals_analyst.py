from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from yiagents.agents.utils.agent_utils import (
    get_balance_sheet,
    get_cashflow,
    get_form4_insider_trading,
    get_fundamentals,
    get_ftd_data,
    get_income_statement,
    get_institutional_holdings,
    get_instrument_context_from_state,
    get_language_instruction,
)
from yiagents.agents.utils.valuation_tools import get_valuation_metrics
from yiagents.dataflows.config import get_config


# Appended to the fundamentals system prompt only when YIAGENTS_SEC_OWNERSHIP is
# on. When off, the analyst's prompt (and tool list) are byte-for-byte unchanged.
_SEC_OWNERSHIP_NUDGE = (
    " Additional ownership & short-interest tools (US-listed only): "
    "`get_form4_insider_trading` (insider / officer / director >10% buys-sells "
    "from SEC Form 4, point-in-time by filing date), `get_ftd_data` (SEC "
    "fails-to-deliver balances, a naked-short / bearish-pressure proxy), and "
    "`get_institutional_holdings` (top institutional 13F holders reverse-"
    "aggregated from the SEC bulk Form 13F Data Sets by CUSIP, inherently "
    "~45 days stale). Use them to qualify the ownership and shorting picture "
    "when relevant. All three are US-listed only; if a tool returns 'data not "
    "available', 'no fails reported', or 'not yet published' for this symbol, "
    "report that honestly and do not estimate insider activity, short "
    "pressure, or institutional positioning."
)


def create_fundamentals_analyst(llm):
    def fundamentals_analyst_node(state):
        current_date = state["trade_date"]
        instrument_context = get_instrument_context_from_state(state)

        tools = [
            get_fundamentals,
            get_balance_sheet,
            get_cashflow,
            get_income_statement,
        ]
        # Deterministic intrinsic-value PoT tool (env: YIAGENTS_VALUATION_TOOLS,
        # off by default). When off, the tool list -- and therefore the tool
        # names injected into the prompt -- is byte-for-byte identical to the
        # prior behaviour, so the analyst's inputs/capabilities/depth are
        # unchanged. When on, the analyst delegates Graham number / NCAV / PEG /
        # owner-earnings / two-stage DCF / WACC / margin-of-safety arithmetic to
        # Python instead of confabulating it.
        if get_config().get("valuation_tools"):
            tools.append(get_valuation_metrics)
        # SEC ownership & short-interest (Track B2, env: YIAGENTS_SEC_OWNERSHIP,
        # off by default). Same byte-equivalence contract as valuation_tools:
        # when off, the tool list -- and therefore the tool names injected into
        # the prompt -- is byte-for-byte identical to the prior behaviour, so the
        # analyst's inputs/capabilities/depth are unchanged. When on, two
        # PIT-correct US-only tools are appended plus a short nudge.
        if get_config().get("sec_ownership"):
            tools.extend([get_form4_insider_trading, get_ftd_data, get_institutional_holdings])

        system_message = (
            "You are a researcher tasked with analyzing fundamental information over the past week about a company. Please write a comprehensive report of the company's fundamental information such as financial documents, company profile, basic company financials, and company financial history to gain a full view of the company's fundamental information to inform traders. Focus on the most decision-relevant figures rather than exhaustive detail, and tie every claim to a specific number and reporting period pulled from the tools. Provide specific, actionable insights with supporting evidence to help traders make informed decisions."
            + " Make sure to append a Markdown table at the end of the report to organize key points in the report, organized and easy to read."
            + " Use the available tools: `get_fundamentals` for comprehensive company analysis, `get_balance_sheet`, `get_cashflow`, and `get_income_statement` for specific financial statements."
            + " Grounding rules (anti-hallucination): (1) Every conclusion must cite a concrete data point with its date or reporting period (e.g. 'FY2025 Q3 revenue $X reported on YYYY-MM-DD'). (2) If two tools disagree (e.g. get_fundamentals vs get_income_statement), flag the discrepancy explicitly rather than inventing a reconciled number. (3) If a figure is missing, stale, or the tools return no data for the period, write 'data not available' for that item instead of estimating or extrapolating."
            + get_language_instruction(),
        )
        if get_config().get("sec_ownership"):
            # system_message is a 1-tuple by long-standing construction (the
            # trailing comma above); append the nudge to its string element,
            # preserving the tuple shape so prompt formatting is identical in
            # structure to the off-path (byte-equivalent when off).
            system_message = (system_message[0] + _SEC_OWNERSHIP_NUDGE,)

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful AI assistant, collaborating with other assistants."
                    " Use the provided tools to progress towards answering the question."
                    " If you are unable to fully answer, that's OK; another assistant with different tools"
                    " will help where you left off. Execute what you can to make progress."
                    " If you or any other assistant has the FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** or deliverable,"
                    " prefix your response with FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** so the team knows to stop."
                    " You have access to the following tools: {tool_names}."
                    " Today's date is {current_date}; treat it as 'now' for all analysis and tool-call date ranges. {instrument_context}\n"
                    "{system_message}",
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        prompt = prompt.partial(system_message=system_message)
        prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
        prompt = prompt.partial(current_date=current_date)
        prompt = prompt.partial(instrument_context=instrument_context)

        chain = prompt | llm.bind_tools(tools)

        result = chain.invoke(state["messages"])

        report = ""

        if len(result.tool_calls) == 0:
            report = result.content

        return {
            "messages": [result],
            "fundamentals_report": report,
        }

    return fundamentals_analyst_node
