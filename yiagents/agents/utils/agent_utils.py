import functools
import logging
from collections.abc import Mapping
from typing import Any

import yfinance as yf
from langchain_core.messages import HumanMessage, RemoveMessage

from yiagents.agents.utils.a_share_native_tools import (
    get_a_share_balance_sheet_native,
    get_a_share_cashflow_statement_native,
    get_a_share_dragon_tiger_native,
    get_a_share_fundamentals_native,
    get_a_share_income_statement_native,
    get_a_share_market_breadth_native,
    get_a_share_money_flow_native,
    get_a_share_news_native,
    get_a_share_northbound_native,
    get_a_share_ohlc_native,
    get_a_share_realtime_quote_native,
    get_a_share_sector_flow_native,
)
from yiagents.agents.utils.binance_indicator_tools import (
    get_binance_indicators,
    get_binance_spot_indicators,
)

# Import tools from separate utility files
from yiagents.agents.utils.binance_perp_tools import (
    get_binance_basis,
    get_binance_depth_snapshot,
    get_binance_funding_rate,
    get_binance_klines,
    get_binance_long_short_ratio,
    get_binance_open_interest,
    get_binance_premium_index,
    get_binance_taker_buy_sell,
    get_binance_vision_book_depth,
    get_binance_vision_metrics,
)
from yiagents.agents.utils.binance_spot_tools import (
    get_binance_spot_klines,
    get_binance_spot_perp_basis,
    get_binance_spot_ticker24,
)
from yiagents.agents.utils.core_stock_tools import get_stock_data
from yiagents.agents.utils.eastmoney_tools import get_margin_trading
from yiagents.agents.utils.fundamental_data_tools import (
    get_balance_sheet,
    get_cashflow,
    get_fundamentals,
    get_income_statement,
)
from yiagents.agents.utils.macro_data_tools import get_macro_indicators
from yiagents.agents.utils.market_data_validation_tools import get_verified_market_snapshot
from yiagents.agents.utils.news_data_tools import (
    get_global_news,
    get_insider_transactions,
    get_news,
)
from yiagents.agents.utils.prediction_markets_tools import get_prediction_markets
from yiagents.agents.utils.price_structure_tools import (
    get_candlestick_patterns,
    get_relative_strength,
    get_support_resistance,
    get_volume_features,
)
from yiagents.agents.utils.sec_ownership_tools import (
    get_form4_insider_trading,
    get_ftd_data,
    get_institutional_holdings,
)
from yiagents.agents.utils.technical_indicators_tools import get_indicators
from yiagents.agents.utils.web_search_tools import (
    web_search,
    web_search_fundamentals,
    web_search_market,
)
from yiagents.agents.utils.weekly_indicators_tools import get_indicators_weekly
from yiagents.dataflows.binance import stock_perp_underlying

# Public surface: the data tools are imported here so agents and the graph
# import them from one place, plus the instrument/language helpers defined below.
__all__ = [
    "get_stock_data",
    "web_search",
    "web_search_market",
    "web_search_fundamentals",
    "get_indicators",
    "get_indicators_weekly",
    "get_support_resistance",
    "get_volume_features",
    "get_candlestick_patterns",
    "get_relative_strength",
    "get_binance_indicators",
    "get_binance_spot_indicators",
    "get_binance_klines",
    "get_binance_funding_rate",
    "get_binance_open_interest",
    "get_binance_long_short_ratio",
    "get_binance_taker_buy_sell",
    "get_binance_basis",
    "get_binance_premium_index",
    "get_binance_depth_snapshot",
    "get_binance_vision_metrics",
    "get_binance_vision_book_depth",
    "get_binance_spot_klines",
    "get_binance_spot_ticker24",
    "get_binance_spot_perp_basis",
    "get_fundamentals",
    "get_balance_sheet",
    "get_cashflow",
    "get_income_statement",
    "get_form4_insider_trading",
    "get_ftd_data",
    "get_institutional_holdings",
    "get_margin_trading",
    "get_a_share_fundamentals_native",
    "get_a_share_ohlc_native",
    "get_a_share_income_statement_native",
    "get_a_share_balance_sheet_native",
    "get_a_share_cashflow_statement_native",
    "get_a_share_news_native",
    "get_a_share_money_flow_native",
    "get_a_share_dragon_tiger_native",
    "get_a_share_northbound_native",
    "get_a_share_sector_flow_native",
    "get_a_share_realtime_quote_native",
    "get_a_share_market_breadth_native",
    "get_news",
    "get_global_news",
    "get_insider_transactions",
    "get_macro_indicators",
    "get_prediction_markets",
    "get_verified_market_snapshot",
    "build_instrument_context",
    "resolve_instrument_identity",
    "get_instrument_context_from_state",
    "get_language_instruction",
    "create_msg_delete",
    "build_clear_placeholder",
    "get_clear_placeholder_from_state",
    "build_risk_debate_update",
    "build_investment_debate_update",
]

logger = logging.getLogger(__name__)


def get_language_instruction() -> str:
    """Return a prompt instruction for the configured output language.

    Returns empty string when English (default), so no extra tokens are used.
    Applied to every agent whose output reaches the saved report —
    analysts, researchers, debaters, research manager, trader, and
    portfolio manager — so a non-English run produces a fully localized
    report rather than a mix of languages.
    """
    from yiagents.dataflows.config import get_config
    lang = get_config().get("output_language", "English")
    if lang.strip().lower() == "english":
        return ""
    return f" Write your entire response in {lang}."


def _clean_identity_value(value: Any) -> str | None:
    """Return a trimmed string, or None for empty / placeholder-ish values."""
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    if not cleaned or cleaned.lower() in {"none", "n/a", "nan", "null"}:
        return None
    return cleaned


@functools.lru_cache(maxsize=256)
def resolve_instrument_identity(ticker: str, curr_date: str | None = None) -> dict:
    """Resolve deterministic identity metadata (company name, sector, …) for a ticker.

    This exists to stop the pipeline from hallucinating a *different* company
    when a chart pattern suggests a different industry than the real one
    (#814): without a ground-truth name, the market analyst would pattern-match
    the price action to a narrative and invent an identity that then cascaded
    through every downstream agent.

    Best-effort by design: if yfinance is unavailable, rate-limited, or doesn't
    recognise the ticker, we return ``{}`` and the caller falls back to
    ticker-only context rather than failing before analysis starts. Cached so
    the lookup happens at most once per (ticker, curr_date) per process.

    The symbol is normalized first (e.g. ``XAUUSD`` -> ``GC=F``) so identity
    resolves for the same instrument the price path actually fetches (#983).

    Point-in-time guard: ``.info`` is a single current-point snapshot with no
    date dimension — the company name, sector, industry, and exchange are
    always *today's* values. On an explicit historical ``curr_date`` (a
    backtest decision date) surfacing them would leak the future identity,
    so we refuse and return ``{}`` exactly as the fundamentals overview does
    (``overview_would_leak_future``). Live mode (``curr_date`` empty/None or
    today) keeps the snapshot — it is legitimately current then.
    """
    from yiagents.dataflows.symbol_utils import normalize_symbol
    from yiagents.dataflows.utils import overview_would_leak_future

    # PIT: a past backtest date must not receive today's identity snapshot.
    # Falls back to ticker-only context (build_instrument_context handles an
    # empty identity), the same degradation as a yfinance failure.
    if overview_would_leak_future(curr_date):
        logger.debug(
            "Identity snapshot refused for %s on historical date %s (PIT guard)",
            ticker, curr_date,
        )
        return {}

    try:
        info = yf.Ticker(normalize_symbol(ticker)).info or {}
    except Exception as exc:  # noqa: BLE001 — fail open, never block the run
        # The anti-hallucination identity anchor disappears with this failure —
        # WARNING (not DEBUG) so the operator can see the guard went away.
        logger.warning(
            "Could not resolve instrument identity for %s: %s", ticker, exc
        )
        return {}

    identity: dict[str, str] = {}
    company_name = _clean_identity_value(info.get("longName")) or _clean_identity_value(
        info.get("shortName")
    )
    if company_name:
        identity["company_name"] = company_name
    for source_key, target_key in (
        ("sector", "sector"),
        ("industry", "industry"),
        ("exchange", "exchange"),
        ("quoteType", "quote_type"),
    ):
        value = _clean_identity_value(info.get(source_key))
        if value:
            identity[target_key] = value
    return identity


def clear_identity_cache() -> None:
    """Flush the :func:`resolve_instrument_identity` LRU cache.

    For a *historical* ``curr_date`` the cached identity is deterministic and
    never needs clearing. For *live* mode (``curr_date=None``) the identity is
    cached for the process lifetime, so a corporate action (rename, delisting,
    ticker change) would leave a long-running live process serving stale
    identity metadata. Call this after such an event to force a fresh lookup.
    """
    resolve_instrument_identity.cache_clear()


def build_instrument_context(
    ticker: str,
    asset_type: str = "stock",
    identity: Mapping[str, str] | None = None,
) -> str:
    """Describe the exact instrument so agents preserve identity and ticker.

    When ``identity`` is provided (resolved deterministically via
    :func:`resolve_instrument_identity`), the company name and business
    classification are injected so agents anchor to the real company rather
    than pattern-matching the price chart to a wrong one (#814).
    """
    is_crypto = asset_type == "crypto"
    is_perp = asset_type == "crypto_perp"
    is_spot = asset_type == "crypto_spot"
    instrument_label = "asset" if (is_crypto or is_perp or is_spot) else "instrument"
    context = (
        f"The {instrument_label} to analyze is `{ticker}`. "
        "Use this exact ticker in every tool call, report, and recommendation, "
        "preserving any exchange suffix (e.g. `.TO`, `.L`, `.HK`, `.T`, `-USD`)."
    )

    details = []
    if identity:
        name = identity.get("company_name") or identity.get("name")
        if name:
            details.append(f"{'Name' if is_crypto else 'Company'}: {name}")
        sector, industry = identity.get("sector"), identity.get("industry")
        if sector and industry:
            details.append(f"Business classification: {sector} / {industry}")
        elif sector:
            details.append(f"Sector: {sector}")
        elif industry:
            details.append(f"Industry: {industry}")
        if identity.get("exchange"):
            details.append(f"Exchange: {identity['exchange']}")

    if details:
        context += (
            f" Resolved identity: {'; '.join(details)}. "
            "Do not substitute a different company or ticker unless a tool "
            "result explicitly disproves this resolved identity."
        )

    if is_perp:
        underlying = stock_perp_underlying(ticker)
        if underlying:
            # Tokenized US-equity perp: the contract tracks a real listed
            # company/ETF, so fundamentals ARE in play — via the underlying
            # symbol, not the perp symbol (the tool layer remaps as backstop).
            context += (
                " This is a Binance USDT-M perpetual on a tokenized US "
                f"equity; the underlying company/ETF trades as `{underlying}`. "
                "Funding rate, open interest, and basis are first-class "
                "signals; mind leverage and funding-cost drag. Company "
                "fundamentals ARE available: pass the underlying ticker "
                f"`{underlying}` (not {ticker}) to the fundamentals tools, and "
                "weigh filings/valuation as you would for the listed stock."
            )
        else:
            context += (
                " This is a Binance USDT-M perpetual futures contract (crypto_perp). "
                "Funding rate, open interest, and basis are first-class signals; "
                "mind leverage and funding-cost drag. Do not assume company "
                "fundamentals are available."
            )
    if is_spot:
        context += (
            " This is a Binance SPOT pair (crypto_spot). No funding rate, open "
            "interest, leverage, or liquidations apply — it is the spot "
            "reference price the perpetual trades around. The spot-perp basis "
            "tool shows the perpetual's premium/discount vs this spot price. "
            "Do not assume company fundamentals are available."
        )
    if is_crypto:
        context += (
            " Treat it as a crypto asset rather than a company, and do not "
            "assume company fundamentals are available."
        )
    return context


def get_instrument_context_from_state(state: Mapping[str, Any]) -> str:
    """Return the instrument context for the current run.

    Prefers the identity-resolved context computed once at run start and
    stored on the state (see ``YiAgentsGraph.resolve_instrument_context``).
    Falls back to a ticker-only context — with no network lookup — when the
    state was constructed without it (bare programmatic states, tests), so a
    consumer is never forced to make a yfinance call mid-graph.
    """
    context = state.get("instrument_context")
    if isinstance(context, str) and context.strip():
        return context
    return build_instrument_context(
        str(state["company_of_interest"]),
        state.get("asset_type", "stock"),
    )


def build_clear_placeholder(instrument_context: str, trade_date: str) -> HumanMessage:
    """Build the context-anchored placeholder emitted after messages are cleared.

    The placeholder must not be a bare ``"Continue"``: some OpenAI-compatible
    providers interpret that literally as the user task and produce output
    about the word "continue" instead of analysing the instrument (#888).
    Anchoring it to the resolved instrument context and date keeps the next
    analyst on-task even if the provider treats the placeholder as a
    standalone request.

    This is the SINGLE source of truth for the placeholder text — both
    :func:`create_msg_delete` (serial path) and
    :func:`~yiagents.graph.analyst_fanout.create_analyst_fanout_node` (parallel
    path) MUST call this so serial and parallel produce byte-identical
    placeholders. Duplicating the string would risk the two paths silently
    diverging, violating the parallel iron law.
    """
    return HumanMessage(
        content=(
            f"Proceed with your assigned analysis for this workflow. "
            f"{instrument_context} The analysis date is {trade_date}."
        )
    )


def get_clear_placeholder_from_state(state: Mapping[str, Any]) -> HumanMessage:
    """Resolve the clear-placeholder for a run from its state.

    Thin wrapper over :func:`build_clear_placeholder` that pulls
    ``instrument_context`` and ``trade_date`` from ``state`` with the same
    fallbacks :func:`create_msg_delete` uses, so every caller builds the
    identical placeholder for the same state.
    """
    instrument_context = get_instrument_context_from_state(state)
    trade_date = state.get("trade_date", "the requested date")
    return build_clear_placeholder(instrument_context, trade_date)


def create_msg_delete():
    def delete_messages(state):
        """Clear messages and add a context-anchored placeholder.

        Delegates to :func:`get_clear_placeholder_from_state` so the serial
        clear-node and the parallel fan-out node share one placeholder builder.
        """
        messages = state["messages"]
        removal_operations = [RemoveMessage(id=m.id) for m in messages]
        placeholder = get_clear_placeholder_from_state(state)
        return {"messages": removal_operations + [placeholder]}

    return delete_messages


# Sentinel emitted when an analyst's final LLM response contains malformed tool
# calls (langchain puts them in ``invalid_tool_calls``) and no usable content.
# Same style as the engine's "[propagate error: ...]" marker: a visible, plain
# marker in the report slot so downstream nodes (and report readers) can see
# the analyst degraded, instead of silently receiving an empty string.
MALFORMED_TOOL_CALLS_SENTINEL = "[analyst produced no report: malformed tool calls]"


def final_analyst_report(result: Any, *, agent_name: str, ticker: str) -> Any:
    """Extract the final report from an analyst node's LLM ``result`` message.

    Reproduces the historical contract exactly — ``""`` while tool calls are
    pending, ``result.content`` when the model produced its final answer — and
    adds one guard: when the final message carries ``invalid_tool_calls`` (the
    model emitted a corrupt tool call; ``tool_calls`` is empty and ``content``
    is usually empty too), the analyst must not silently write an empty report
    into state. Instead:

    * a WARNING is logged with the agent / ticker context, and
    * if there is no usable content, the
      :data:`MALFORMED_TOOL_CALLS_SENTINEL` marker is returned so every
      downstream consumer (debate, trader, PM, reports) can see the analyst
      degraded rather than reading an empty report as "nothing to say".

    Non-empty content alongside invalid tool calls is kept (it is real model
    text) but still logged, because the response is suspect.
    """
    tool_calls = getattr(result, "tool_calls", None) or []
    if len(tool_calls) == 0:
        invalid = getattr(result, "invalid_tool_calls", None) or []
        content = getattr(result, "content", "")
        if invalid:
            logger.warning(
                "%s (%s): final LLM message contained %d malformed tool call(s) "
                "(invalid_tool_calls non-empty); report may be degraded",
                agent_name, ticker, len(invalid),
            )
            if isinstance(content, str) and not content.strip():
                return MALFORMED_TOOL_CALLS_SENTINEL
        return content
    return ""


def build_risk_debate_update(
    risk_debate_state: Mapping[str, Any], speaker: str, argument: str
) -> dict:
    """Assemble the next ``risk_debate_state`` after a risk debator speaks.

    ``speaker`` is one of ``"aggressive"`` / ``"conservative"`` / ``"neutral"``.
    The speaker's own ``<speaker>_history`` and ``current_<speaker>_response``
    receive the new ``argument``; the shared ``history`` log always appends it;
    every other field is carried over unchanged from ``risk_debate_state``;
    ``count`` advances by 1.

    Centralises the ~13-field state dict the three risk debators each rebuilt
    inline (differing only in which history/response field receives the argument
    and the speaker label), so the three cannot drift apart. Byte-equivalent to
    each debator's prior dict.
    """
    update: dict[str, Any] = {
        "history": risk_debate_state.get("history", "") + "\n" + argument,
        "aggressive_history": risk_debate_state.get("aggressive_history", ""),
        "conservative_history": risk_debate_state.get("conservative_history", ""),
        "neutral_history": risk_debate_state.get("neutral_history", ""),
        "latest_speaker": speaker.capitalize(),
        "current_aggressive_response": risk_debate_state.get(
            "current_aggressive_response", ""
        ),
        "current_conservative_response": risk_debate_state.get(
            "current_conservative_response", ""
        ),
        "current_neutral_response": risk_debate_state.get("current_neutral_response", ""),
        # Carry the judge's decision through a debator turn. The parent state
        # replaces this whole sub-dict (last-write-wins on the field), so a
        # helper that omitted the key would blank a judge_decision the
        # Portfolio Manager had already written (e.g. a checkpoint-resumed run
        # replaying a debator) until the PM ran again.
        "judge_decision": risk_debate_state.get("judge_decision", ""),
        "count": risk_debate_state["count"] + 1,
    }
    update[f"{speaker}_history"] = (
        risk_debate_state.get(f"{speaker}_history", "") + "\n" + argument
    )
    update[f"current_{speaker}_response"] = argument
    return update


def build_investment_debate_update(
    investment_debate_state: Mapping[str, Any], speaker: str, argument: str
) -> dict:
    """Assemble the next ``investment_debate_state`` after a researcher speaks.

    ``speaker`` is one of ``"bull"`` / ``"bear"``. The speaker's own
    ``<speaker>_history`` receives the new ``argument``; the shared ``history``
    log always appends it; ``current_response`` is set to ``argument``; the
    opposing side's history is carried over unchanged; ``count`` advances by 1.

    Centralises the 5-field state dict the bull/bear researchers each rebuilt
    inline, so the two cannot drift apart — the same role
    :func:`build_risk_debate_update` plays for the three risk debators.
    Byte-equivalent to each researcher's prior dict apart from the carried
    ``judge_decision`` (see below).

    ``judge_decision`` (present on :class:`InvestDebateState`) is carried
    through unchanged from the current sub-state (empty string when unset).
    The parent state replaces this whole sub-dict on a researcher turn, so a
    helper that omitted the key would blank a Research Manager decision that
    a resumed / replayed turn runs after. Neither researcher ever *writes* a
    judge decision.
    """
    opponent = "bear" if speaker == "bull" else "bull"
    return {
        "history": investment_debate_state.get("history", "") + "\n" + argument,
        f"{speaker}_history": (
            investment_debate_state.get(f"{speaker}_history", "") + "\n" + argument
        ),
        f"{opponent}_history": investment_debate_state.get(f"{opponent}_history", ""),
        "current_response": argument,
        "judge_decision": investment_debate_state.get("judge_decision", ""),
        "count": investment_debate_state["count"] + 1,
    }



