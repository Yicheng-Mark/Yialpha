from yialpha.agents.utils.agent_utils import (
    final_analyst_report,
    get_a_share_balance_sheet_native,
    get_a_share_cashflow_statement_native,
    get_a_share_dragon_tiger_native,
    get_a_share_fundamentals_native,
    get_a_share_income_statement_native,
    get_a_share_money_flow_native,
    get_a_share_ohlc_native,
    get_balance_sheet,
    get_cashflow,
    get_form4_insider_trading,
    get_ftd_data,
    get_fundamentals,
    get_income_statement,
    get_institutional_holdings,
    get_instrument_context_from_state,
    get_language_instruction,
    get_margin_trading,
    web_search_fundamentals,
)
from yialpha.agents.utils.pot_tool import make_pot_compute_tool
from yialpha.agents.utils.prompt_builder import build_collaborator_prompt
from yialpha.agents.utils.valuation_tools import get_valuation_metrics
from yialpha.dataflows.binance import stock_perp_underlying
from yialpha.dataflows.config import get_config
from yialpha.dataflows.symbol_utils import is_a_stock
from yialpha.dataflows.utils import is_historical_date

# Appended to the fundamentals system prompt only when YIALPHA_SEC_OWNERSHIP is
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


# Appended to the fundamentals system prompt only when YIALPHA_A_STOCK is on
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


# Appended to the fundamentals system prompt only when YIALPHA_A_SHARE_NATIVE
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
    "given window, which is normal, not bearish). NEW: "
    "`get_a_share_income_statement_native` (quarterly 利润表 — revenue, net "
    "profit, ROE, EPS), `get_a_share_balance_sheet_native` (quarterly 资产负债表 "
    "— total assets, liabilities, debt ratio), and "
    "`get_a_share_cashflow_statement_native` (quarterly 现金流表 — operating / "
    "investing / financing cash flows). These quarterly statements are PIT-correct "
    "(pubDate <= curr_date) and provide the A-share financial-statement detail the "
    "default path covers thinly. If a tool returns 'data not available' or 'no "
    "coverage found' for this symbol/date, report that honestly and do not "
    "estimate multiples, prices, capital flow, dragon-tiger activity, or "
    "statement line items."
)


# Appended to the fundamentals system prompt when web_search is bound (config
# web_search_enabled, default ON, live dates only). The vendor degrades to a
# WEB_SEARCH_UNAVAILABLE sentinel + data_quality event on key-missing /
# budget-exhausted, so advertising it is always run-safe; the fundamentals
# instance charges its own Tavily budget scope and never competes with the
# news or market analysts' calls.
_WEB_SEARCH_NUDGE = (
    " Optionally use web_search(query) for open-web context the statement "
    "tools cannot surface (earnings-call color, guidance revisions, M&A and "
    "buyback announcements, industry supply/demand narrative). Web-search "
    "grounding rules: cite the source URL for every claim drawn from its "
    "results, and treat snippets as qualitative context ONLY — any prices or "
    "figures appearing in them are unverified text and must never be "
    "reported as data values (numbers come exclusively from the structured "
    "data tools)."
)


# Appended to the fundamentals system prompt only when the run is a Binance
# tokenized-stock USDT-M perp (asset_type == crypto_perp with an equity
# underlying, e.g. MUUSDT -> Micron). Pure-crypto perp / stock / crypto runs
# never enter this branch, so their prompt is byte-for-byte unchanged.
_STOCK_PERP_NUDGE = (
    " This instrument is a Binance tokenized-stock perpetual: analyze the "
    "UNDERLYING US-listed company/ETF, and pass its equity ticker (not the "
    "perp symbol) to the statement tools — the tool layer remaps a perp "
    "symbol to the underlying as a backstop, but name the equity ticker "
    "explicitly. The usual grounding rules apply unchanged: cite reporting "
    "periods and filing dates, honor the filing lag, and write 'data not "
    "available' rather than estimating. Keep the perp framing in mind when "
    "weighing the evidence: the contract trades 24/7 while filings and "
    "earnings land on the US session calendar, so fundamentals inform "
    "DIRECTION and earnings-gap risk, not entry timing; funding cost and "
    "leverage are assessed by other analysts."
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
        # Deterministic intrinsic-value PoT tool (env: YIALPHA_VALUATION_TOOLS,
        # off by default). When off, the tool list -- and therefore the tool
        # names injected into the prompt -- is byte-for-byte identical to the
        # prior behaviour, so the analyst's inputs/capabilities/depth are
        # unchanged. When on, the analyst delegates Graham number / NCAV / PEG /
        # owner-earnings / two-stage DCF / WACC / margin-of-safety arithmetic to
        # Python instead of confabulating it.
        if get_config().get("valuation_tools"):
            tools.append(get_valuation_metrics)
            # Ad-hoc PoT computation tool (same gate as valuation_tools). When
            # on, the analyst can delegate arbitrary numerical reasoning (ratio
            # percentile, implied growth, conversion) to Python code generated
            # by the LLM and run in the restricted sandbox — instead of doing
            # the arithmetic in its head. When off, byte-for-byte unchanged.
            tools.append(make_pot_compute_tool(llm))
        # SEC ownership & short-interest (Track B2, env: YIALPHA_SEC_OWNERSHIP,
        # off by default). Same byte-equivalence contract as valuation_tools:
        # when off, the tool list -- and therefore the tool names injected into
        # the prompt -- is byte-for-byte identical to the prior behaviour, so the
        # analyst's inputs/capabilities/depth are unchanged. When on, two
        # PIT-correct US-only tools are appended plus a short nudge.
        if get_config().get("sec_ownership"):
            tools.extend([get_form4_insider_trading, get_ftd_data, get_institutional_holdings])
        # China A-share margin trading (Track A, env: YIALPHA_A_STOCK, off by
        # default). Same byte-equivalence contract as the flags above — gated
        # TWICE: the flag AND is_a_stock(ticker). When either fails, the tool
        # list / prompt / capabilities are byte-for-byte identical to the prior
        # behaviour, so US / crypto / HK tickers never enter this branch and the
        # analyst's inputs/capabilities/depth are unchanged. When both hold, one
        # PIT-correct A-share-only tool is appended plus a short nudge.
        if get_config().get("a_stock") and is_a_stock(ticker):
            tools.append(get_margin_trading)
        # Native A-share OHLC + TTM valuation (env: YIALPHA_A_SHARE_NATIVE,
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
                get_a_share_income_statement_native,
                get_a_share_balance_sheet_native,
                get_a_share_cashflow_statement_native,
            ])
        # Open-web search (config: web_search_enabled, on by default). Live
        # dates only — Tavily has no as-of parameter, so binding it on a
        # historical replay date would leak future web context (same PIT
        # contract as the news analyst's prediction-markets gate).
        # Byte-equivalent when the flag is off or the date is historical: no
        # tool, no nudge.
        bind_web_search = (
            get_config().get("web_search_enabled", True)
            and not is_historical_date(current_date)
        )
        if bind_web_search:
            tools.append(web_search_fundamentals)

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
        # Web-search nudge uses the SAME gate (flag AND live date) as the tool
        # extension above, so the prompt only changes when the tools do.
        if bind_web_search:
            system_message = (system_message[0] + _WEB_SEARCH_NUDGE,)
        # Tokenized-stock perp nudge: gated on the run being a crypto_perp
        # whose symbol resolves to an equity underlying (the analyst only
        # exists on such a run because filter_analysts_for_asset_type kept
        # it). Prompt-only — the statement tools already remap the symbol.
        if state.get("asset_type") == "crypto_perp" and stock_perp_underlying(ticker):
            system_message = (system_message[0] + _STOCK_PERP_NUDGE,)

        prompt = build_collaborator_prompt(include_tools=True)

        prompt = prompt.partial(system_message=system_message)
        prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
        prompt = prompt.partial(current_date=current_date)
        prompt = prompt.partial(instrument_context=instrument_context)

        chain = prompt | llm.bind_tools(tools)

        result = chain.invoke(state["messages"])

        # Shared final-report extraction: "" while tool calls are pending,
        # content on the final answer, and a visible sentinel (plus WARNING)
        # when the final message carried malformed tool calls.
        report = final_analyst_report(
            result, agent_name="Fundamentals Analyst", ticker=ticker,
        )

        return {
            "messages": [result],
            "fundamentals_report": report,
        }

    return fundamentals_analyst_node
