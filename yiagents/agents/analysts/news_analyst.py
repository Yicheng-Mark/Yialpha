from yiagents.agents.utils.agent_utils import (
    final_analyst_report,
    get_a_share_news_native,
    get_global_news,
    get_instrument_context_from_state,
    get_language_instruction,
    get_macro_indicators,
    get_news,
    get_prediction_markets,
    web_search,
)
from yiagents.agents.utils.prompt_builder import build_collaborator_prompt
from yiagents.dataflows.config import get_config
from yiagents.dataflows.symbol_utils import is_a_stock
from yiagents.dataflows.utils import is_historical_date

# Appended to the news system prompt only when YIAGENTS_A_SHARE_NATIVE is on AND
# the ticker is a China A-share (.SS/.SH/.SZ). When off (or non-A-share), the
# analyst's prompt (and tool list) are byte-for-byte unchanged. Same double-gate
# contract as the fundamentals analyst's a_share_native nudge.
_A_SHARE_NATIVE_NUDGE = (
    " Additional native China A-share news tool (A-share only): "
    "`get_a_share_news_native` (东财 per-stock Chinese headlines via AKShare, "
    "point-in-time by publish date). Use it to surface A-share-specific news "
    "the default Reddit/StockTwits/yfinance path covers thinly. Headlines are "
    "Chinese-language. If the tool returns 'no coverage found' for this symbol/"
    "date, report that honestly and do not fabricate headlines."
)

# Appended to the news system prompt when config web_search_enabled is on
# (default). The tool itself degrades to a WEB_SEARCH_UNAVAILABLE sentinel +
# data_quality event when TAVILY_API_KEY is missing or the per-run budget is
# exhausted, so advertising it is always run-safe.
_WEB_SEARCH_INSTRUCTION = (
    " Optionally use web_search(query) for open-web context on recent "
    "developments the news vendors may cover thinly (regulatory actions, "
    "industry events, analyst commentary). Web-search grounding rules: cite "
    "the source URL for every claim drawn from its results, and treat "
    "snippets as qualitative context ONLY — any prices or figures appearing "
    "in them are unverified text and must never be reported as data values "
    "(numbers come exclusively from the structured data tools)."
)


def create_news_analyst(llm):
    def news_analyst_node(state):
        current_date = state["trade_date"]
        asset_type = state.get("asset_type", "stock")
        asset_label = "company" if asset_type == "stock" else "asset"
        instrument_context = get_instrument_context_from_state(state)
        ticker = str(state["company_of_interest"])

        tools = [get_news, get_global_news, get_macro_indicators]
        prediction_markets_instruction = ""
        if not is_historical_date(current_date):
            tools.append(get_prediction_markets)
            prediction_markets_instruction = (
                " Use get_prediction_markets(topic, limit) for live "
                "market-implied probabilities of forward-looking events "
                "(e.g. Fed decisions, geopolitics, or sector events)."
            )
        # Open-web search (config: web_search_enabled, on by default). The
        # vendor handles key-missing / budget-exhausted degradation itself
        # (sentinel + data_quality event), so the gate here only decides
        # whether the analyst sees the tool at all.
        web_search_instruction = ""
        if get_config().get("web_search_enabled", True):
            tools.append(web_search)
            web_search_instruction = _WEB_SEARCH_INSTRUCTION
        # Native A-share news (env: YIAGENTS_A_SHARE_NATIVE, off by default).
        # Double-gated byte-equivalence contract: flag AND is_a_stock(ticker).
        # When either fails the tool list / prompt are byte-for-byte identical to
        # the prior behaviour, so US / crypto / HK tickers never enter this branch.
        # When both hold, one PIT-correct A-share-only news tool is appended.
        if get_config().get("a_share_native") and is_a_stock(ticker):
            tools.append(get_a_share_news_native)

        system_message = (
            f"You are a news researcher tasked with analyzing recent news and trends over the past week. Please write a comprehensive report of the state of the world as of {current_date} that is relevant for trading and macroeconomics. Use the available tools: get_news(query, start_date, end_date) for {asset_label}-specific or targeted news searches, get_global_news(curr_date, look_back_days, limit) for broader macroeconomic news, and get_macro_indicators(indicator, curr_date, look_back_days) to ground macro commentary in actual data from FRED (e.g. 'cpi', 'core_pce', 'unemployment', 'fed_funds_rate', '10y_treasury', 'yield_curve')."
            + prediction_markets_instruction
            + web_search_instruction
            + " Provide specific, actionable insights with supporting evidence to help traders make informed decisions."
            + """ Make sure to append a Markdown table at the end of the report to organize key points in the report, organized and easy to read."""
            + " Grounding rules (anti-hallucination): (1) Every news item or macro claim must cite its source and date (e.g. 'per FRED, core_pce was X% on YYYY-MM-DD' or 'headline from get_news, YYYY-MM-DD'). (2) If two sources conflict, flag the discrepancy rather than inventing a reconciled narrative. (3) If a tool returns no results for the query/period, write 'no coverage found' for that angle instead of speculating or filling gaps from prior knowledge."
            + get_language_instruction()
        )
        # A-share news nudge uses the SAME double gate as the tool extension
        # above, so the prompt only changes when the tools do. (news keeps
        # system_message as a plain string, unlike fundamentals' 1-tuple.)
        if get_config().get("a_share_native") and is_a_stock(ticker):
            system_message = system_message + _A_SHARE_NATIVE_NUDGE

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
            result, agent_name="News Analyst", ticker=ticker,
        )

        return {
            "messages": [result],
            "news_report": report,
        }

    return news_analyst_node
