from datetime import datetime, timedelta

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
from yiagents.dataflows.binance import stock_perp_underlying
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
# (default) AND the run date is live. The tool itself degrades to a
# WEB_SEARCH_UNAVAILABLE sentinel + data_quality event when TAVILY_API_KEY is
# missing or the per-run budget is exhausted, so advertising it is always
# run-safe.
_WEB_SEARCH_INSTRUCTION = (
    " Optionally use web_search(query) for open-web context on recent "
    "developments the news vendors may cover thinly (regulatory actions, "
    "industry events, analyst commentary). Web-search grounding rules: cite "
    "the source URL for every claim drawn from its results, and treat "
    "snippets as qualitative context ONLY — any prices or figures appearing "
    "in them are unverified text and must never be reported as data values "
    "(numbers come exclusively from the structured data tools)."
)

#: Prefetch window for the tokenized-stock perp dual-angle blocks, matching
#: the sentiment analyst's 7-day lookback and the "past week" framing of the
#: news system message.
_STOCK_PERP_LOOKBACK_DAYS = 7


def _get_news_impl(ticker: str, start_date: str, end_date: str) -> str:
    """Call the underlying function behind the ``get_news`` LangChain tool.

    Same contract as the sentiment analyst's accessor (kept local here so the
    sentiment module's monkeypatched one stays untouched): ``get_news`` is a
    ``@tool``-decorated ``BaseTool``; its raw callable lives under ``.func``.
    mypy cannot see ``.func`` on the ``BaseTool`` type, so this helper
    centralizes the access with a safe ``getattr`` fallback. A vendor hard
    failure propagates (fail-closed), identical to the sentiment analyst's
    news prefetch.
    """
    fn = getattr(get_news, "func", None)
    if fn is not None:
        return fn(ticker, start_date, end_date)
    return get_news(ticker, start_date, end_date)  # type: ignore[operator]


def _lookback_start(trade_date: str) -> str:
    return (
        datetime.strptime(trade_date, "%Y-%m-%d") - timedelta(days=_STOCK_PERP_LOOKBACK_DAYS)
    ).strftime("%Y-%m-%d")


def _stock_perp_news_section(
    ticker: str,
    underlying: str,
    start_date: str,
    end_date: str,
    company_block: str,
    perp_block: str,
) -> str:
    """Render the deterministic dual-angle news blocks for an equity perp run.

    Both angles (underlying company + perp contract) are prefetched in code so
    coverage does not depend on the LLM choosing to query both; the blocks are
    labelled with the exact query each was fetched with so the report can cite
    them per angle.
    """
    return (
        f"\n### Pre-fetched news — tokenized-stock perp dual coverage "
        f"({start_date} to {end_date})\n"
        "This instrument is a Binance tokenized-stock perpetual; the two angles "
        "below have ALREADY been collected for you, each with the query shown "
        "on its tag. Your report MUST cover BOTH angles: the underlying "
        f"company/ETF (`{underlying}` — earnings, guidance, products, "
        "regulation) and the contract itself "
        f"(`{ticker}` — funding, basis, listing/delisting, crypto-market "
        "premium events). Ground each angle only in its own block, and if a "
        "block shows no coverage, state 'no coverage found' for that angle "
        "explicitly instead of generalizing from the other. Follow-up "
        "drill-down get_news queries are still allowed; do not re-fetch these "
        "two angles wholesale.\n\n"
        f'<start_of_company_news> (query: "{underlying}")\n'
        f"{company_block}\n"
        "<end_of_company_news>\n\n"
        f'<start_of_perp_news> (query: "{ticker}")\n'
        f"{perp_block}\n"
        "<end_of_perp_news>\n"
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
        # Open-web search (config: web_search_enabled, on by default). Live
        # dates only: Tavily returns today's web with no as-of parameter, so
        # advertising it on a historical replay date would leak future
        # information — same PIT contract as get_prediction_markets above.
        # The vendor handles key-missing / budget-exhausted degradation
        # itself (sentinel + data_quality event), so the gate here only
        # decides whether the analyst sees the tool at all.
        web_search_instruction = ""
        if (
            get_config().get("web_search_enabled", True)
            and not is_historical_date(current_date)
        ):
            tools.append(web_search)
            web_search_instruction = _WEB_SEARCH_INSTRUCTION
        # Native A-share news (env: YIAGENTS_A_SHARE_NATIVE, off by default).
        # Double-gated byte-equivalence contract: flag AND is_a_stock(ticker).
        # When either fails the tool list / prompt are byte-for-byte identical to
        # the prior behaviour, so US / crypto / HK tickers never enter this branch.
        # When both hold, one PIT-correct A-share-only news tool is appended.
        if get_config().get("a_share_native") and is_a_stock(ticker):
            tools.append(get_a_share_news_native)

        # Tokenized-stock perp dual coverage: deterministically prefetch BOTH
        # news angles (underlying company + perp contract) and inject them as
        # labelled blocks, mirroring the sentiment analyst's prefetch pattern.
        # Same gate as the fundamentals analyst's _STOCK_PERP_NUDGE — pure
        # crypto perps and non-perp runs append "" and stay byte-for-byte
        # unchanged. get_news carries explicit date bounds, so the blocks are
        # PIT-safe on historical replay dates too.
        stock_perp_section = ""
        underlying = stock_perp_underlying(ticker) if asset_type == "crypto_perp" else None
        if underlying:
            start_date = _lookback_start(current_date)
            stock_perp_section = _stock_perp_news_section(
                ticker=ticker,
                underlying=underlying,
                start_date=start_date,
                end_date=current_date,
                company_block=_get_news_impl(underlying, start_date, current_date),
                perp_block=_get_news_impl(ticker, start_date, current_date),
            )

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
        # Tokenized-stock perp section uses the SAME gate as the prefetch
        # above; appending "" keeps every non-equity-perp run byte-identical.
        system_message = system_message + stock_perp_section

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
