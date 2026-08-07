from yiagents.agents.utils.agent_utils import (
    get_a_share_dragon_tiger_native,
    get_a_share_fundamentals_native,
    get_a_share_money_flow_native,
    get_a_share_ohlc_native,
    get_balance_sheet,
    get_cashflow,
    get_form4_insider_trading,
    get_fundamentals,
    get_ftd_data,
    get_income_statement,
    get_institutional_holdings,
    get_instrument_context_from_state,
    get_language_instruction,
    get_margin_trading,
)
from yiagents.agents.utils.prompt_builder import build_collaborator_prompt
from yiagents.agents.utils.valuation_tools import get_valuation_metrics
from yiagents.dataflows.config import get_config
from yiagents.dataflows.symbol_utils import is_a_stock


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


# Appended to the fundamentals system prompt only when YIAGENTS_A_STOCK is on
# AND the ticker is a China A-share (.SS/.SH/.SZ). When off (or non-A-share),
# the analyst's prompt (and tool list) are byte-for-byte unchanged.
_A_STOCK_NUDGE = (
    " Additional China A-share margin-trading tool (A-share only): "
    "`get_margin_trading` (融资融券 — margin balance 融资余额, short balance "
    "融券余额, margin buy 融资买入额, net margin buy 融资净买入额; exchange-"
    "disclosed daily with full history). Rising 融资余额 = bullish leverage "
    "build-up; rising 融券余额 = growing short interest. If the tool returns "
    "'data not available' for this symbol/date, report that honestly and do "
    "not estimate margin positions."
)


# Appended to the fundamentals system prompt only when YIAGENTS_A_SHARE_NATIVE
# is on AND the ticker is a China A-share (.SS/.SH/.SZ). When off (or non-
# A-share), the analyst's prompt (and tool list) are byte-for-byte unchanged.
_A_SHARE_NATIVE_NUDGE = (
    " Additional native China A-share data tools (A-share only): "
    "`get_a_share_fundamentals_native` (daily TTM valuation — PE-TTM / PB-MRQ / "
    "PS-TTM / PCF-TTM, server-computed so no single-period distortion, plus the "
    "ST flag) and `get_a_share_ohlc_native` (daily 前复权 OHLCV). These are more "
    "reliable for A-shares than the default Yahoo path; use them to cross-check "
    "multiples and prices. A negative PE = trailing loss. Also "
    "`get_a_share_money_flow_native` (daily 资金流 — 主力/超大单/大单/中单/小单 "
    "net inflow; a persistent positive 主力净流入 = institutional accumulation / "
    "bullish, persistent negative = distribution / bearish) and "
    "`get_a_share_dragon_tiger_native` (龙虎榜 appearances — net institutional "
    "buy-in vs sell-out, a smart-money signal; most stocks do not appear in a "
    "given window, which is normal, not bearish). If a tool returns 'data not "
    "available' or 'no coverage found' for this symbol/date, report that honestly "
    "and do not estimate multiples, prices, capital flow, or dragon-tiger activity."
)


def create_fundamentals_analyst(llm):
    def fundamentals_analyst_node(state):
        current_date = state["trade_date"]
        instrument_context = get_instrument_context_from_state(state)
        ticker = str(state["company_of_interest"])

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
        # China A-share margin trading (Track A, env: YIAGENTS_A_STOCK, off by
        # default). Same byte-equivalence contract as the flags above — gated
        # TWICE: the flag AND is_a_stock(ticker). When either fails, the tool
        # list / prompt / capabilities are byte-for-byte identical to the prior
        # behaviour, so US / crypto / HK tickers never enter this branch and the
        # analyst's inputs/capabilities/depth are unchanged. When both hold, one
        # PIT-correct A-share-only tool is appended plus a short nudge.
        if get_config().get("a_stock") and is_a_stock(ticker):
            tools.append(get_margin_trading)
        # Native A-share OHLC + TTM valuation (env: YIAGENTS_A_SHARE_NATIVE,
        # off by default). Same double-gate byte-equivalence contract as a_stock
        # above — flag AND is_a_stock(ticker). When either fails the tool list /
        # prompt / capabilities are byte-for-byte identical to the prior
        # behaviour, so US / crypto / HK tickers never enter this branch. When
        # both hold, two PIT-correct A-share-only tools are appended plus a nudge.
        if get_config().get("a_share_native") and is_a_stock(ticker):
            tools.extend([
                get_a_share_fundamentals_native,
                get_a_share_ohlc_native,
                get_a_share_money_flow_native,
                get_a_share_dragon_tiger_native,
            ])

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
        # A-share nudge uses the SAME double gate (flag AND is_a_stock) as the
        # tool extension above, so the prompt only changes when the tools do.
        if get_config().get("a_stock") and is_a_stock(ticker):
            system_message = (system_message[0] + _A_STOCK_NUDGE,)
        # Native A-share nudge uses the SAME double gate (flag AND is_a_stock) as
        # its tool extension above, so the prompt only changes when the tools do.
        if get_config().get("a_share_native") and is_a_stock(ticker):
            system_message = (system_message[0] + _A_SHARE_NATIVE_NUDGE,)

        prompt = build_collaborator_prompt(include_tools=True)

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
