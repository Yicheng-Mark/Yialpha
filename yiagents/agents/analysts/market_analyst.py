import logging

from yiagents.agents.utils.agent_utils import (
    final_analyst_report,
    get_a_share_market_breadth_native,
    get_a_share_northbound_native,
    get_a_share_realtime_quote_native,
    get_a_share_sector_flow_native,
    get_binance_basis,
    get_binance_funding_rate,
    get_binance_indicators,
    get_binance_klines,
    get_binance_long_short_ratio,
    get_binance_open_interest,
    get_binance_premium_index,
    get_binance_spot_indicators,
    get_binance_spot_klines,
    get_binance_spot_perp_basis,
    get_binance_spot_ticker24,
    get_binance_taker_buy_sell,
    get_candlestick_patterns,
    get_indicators,
    get_indicators_weekly,
    get_instrument_context_from_state,
    get_language_instruction,
    get_relative_strength,
    get_stock_data,
    get_support_resistance,
    get_verified_market_snapshot,
    get_volume_features,
)
from yiagents.agents.utils.prompt_builder import build_collaborator_prompt, build_fincot_prompt
from yiagents.dataflows import indicator_catalog
from yiagents.dataflows.config import get_config
from yiagents.dataflows.market_regime import format_regime_context
from yiagents.dataflows.symbol_utils import is_a_stock
from yiagents.dataflows.utils import is_historical_date

logger = logging.getLogger(__name__)

# Appended to the system message ONLY for crypto_perp runs. Nudges the analyst
# to use the perp-native OHLCV and to treat funding/OI + positioning/order-flow
# as required signals. Non-perp runs append nothing, so the prompt is byte-
# identical to the baseline (stock/crypto unaffected).
_PERP_NUDGE = (
    " This is a Binance USDT-M PERPETUAL, not a spot pair. The spot-style tools "
    "(get_stock_data, get_indicators, get_verified_market_snapshot) are NOT "
    "available for this instrument — they would resolve the symbol to a "
    "different spot pair and return the wrong market. Use get_binance_klines "
    "for OHLCV and get_binance_indicators (venue='perp') for computed classic "
    "indicators (SMA/MACD/RSI/KDJ/ADX/SuperTrend/ATR/realized vol) on the "
    "actual perp candles — cite it for any exact indicator claim. You MUST "
    "call get_binance_funding_rate and get_binance_open_interest and discuss "
    "funding-rate direction, open-interest crowding, and liquidation risk in "
    "your report. You MUST ALSO call get_binance_long_short_ratio and discuss "
    "how the leveraged crowd is positioned (longShortRatio > 1 = longs "
    "dominate; top-trader vs global divergence is a contrary signal) — this is "
    "the perp-native sentiment signal and it frequently contradicts funding/OI "
    "inferences, so reconcile them explicitly. Call get_binance_taker_buy_sell "
    "(order-flow aggression) and, where available, get_binance_basis "
    "(perp-vs-index premium/discount) to round out the positioning picture. "
    "Call get_binance_premium_index for the mark-price snapshot and anchor ALL "
    "liquidation-distance claims to its markPrice (Binance liquidates on the "
    "mark price, not the last traded price) and use its lastFundingRate as the "
    "currently-effective rate. The generic instructions above to call "
    "get_stock_data / get_indicators / get_verified_market_snapshot and to "
    "cite get_indicators_weekly / get_support_resistance / "
    "get_volume_features / get_candlestick_patterns / get_relative_strength "
    "are WAIVED for this instrument — those tools price a different (Yahoo "
    "spot) market. Higher-timeframe and price-structure analysis must instead "
    "come from get_binance_klines / get_binance_indicators on the actual perp "
    "candles. If a tool returns a sentinel/unavailable marker, "
    "say so plainly rather than inventing values."
)

_PERP_HISTORICAL_NUDGE = (
    " This is a point-in-time historical analysis of a Binance USDT-M "
    "PERPETUAL. Use get_binance_klines and get_binance_funding_rate with date "
    "bounds ending on the stated analysis date. The generic instructions to "
    "call get_stock_data / get_indicators / get_verified_market_snapshot / "
    "get_indicators_weekly and the price-structure citation rules "
    "(get_support_resistance, get_volume_features, get_candlestick_patterns, "
    "get_relative_strength) are WAIVED — they price a different (Yahoo spot) "
    "market. Current open-interest snapshots, "
    "long/short positioning, taker order flow, and basis are intentionally not "
    "available because they cannot be reconstructed reliably as of that date. "
    "Do not infer or fabricate those omitted signals."
)

# Appended to the system message ONLY for crypto_spot runs. Spot shares the
# generic OHLCV/indicator/snapshot tools with the stock baseline (the symbol
# resolves correctly), so this nudge only directs the analyst to the spot-
# native OHLCV source and the cross-venue basis signal. Non-spot runs append
# nothing, so their prompts are byte-identical to the baseline.
_SPOT_NUDGE = (
    " This is a Binance SPOT pair (crypto_spot), not a perpetual and not a "
    "Yahoo pair. Use get_binance_spot_klines for the spot OHLCV (the actual "
    "Binance spot book) and get_binance_spot_ticker24 for the 24h snapshot. "
    "get_binance_spot_indicators computes the classic indicator "
    "battery on those same Binance candles — prefer it for exact indicator "
    "claims on this pair. There is no funding rate, open interest, leverage, "
    "or liquidation for a spot pair — do not discuss them. Call "
    "get_binance_spot_perp_basis to show where the USDT-M perpetual trades "
    "relative to this spot price (positive basis = perp rich / long premium; "
    "negative = discount / short pressure) and reconcile it with the spot "
    "trend. The get_indicators / get_verified_market_snapshot tools are "
    "available and resolve correctly for this symbol — use them as usual. If "
    "a tool returns a sentinel/unavailable marker, say so plainly rather than "
    "inventing values."
)

_SPOT_HISTORICAL_NUDGE = (
    " This is a point-in-time historical analysis of a Binance SPOT pair. Use "
    "get_binance_spot_klines with explicit date bounds. The rolling 24-hour "
    "ticker and current spot-perpetual basis are intentionally unavailable "
    "because they would expose present-day data; do not infer them."
)

# Appended to the system message ONLY for China A-share runs when
# YIAGENTS_A_SHARE_NATIVE is on AND the ticker is .SS/.SH/.SZ. Adds the
# northbound / sector-flow / realtime / breadth tools. When off (or non-A-
# share) the analyst's prompt and tool list are byte-for-byte unchanged.
_A_SHARE_MARKET_NUDGE = (
    " Additional China A-share market tools (A-share only): "
    "`get_a_share_northbound_native` (北向资金 / Stock Connect — daily foreign "
    "institutional holding; a persistent increase = foreign accumulation, a "
    "major bullish signal in A-shares; persistent decrease = foreign "
    "distribution), `get_a_share_sector_flow_native` (industry fund-flow "
    "ranking with this stock's sector highlighted — shows whether the sector "
    "is a capital magnet or in outflow), and `get_a_share_realtime_quote_native` "
    "(real-time spot quote — live mode only; for a historical date it returns "
    "a sentinel explaining real-time data is unavailable, so use "
    "get_stock_data / get_indicators for PIT-correct daily OHLCV). Also "
    "`get_a_share_market_breadth_native` (whole-market advance-decline breadth "
    "— A/D > 2 = broad advance, < 0.5 = broad decline; live mode only). If a "
    "tool returns 'data not available' or 'no coverage found', report that "
    "honestly and do not estimate northbound holdings, sector flows, or "
    "breadth."
)

# The indicator catalog the analyst selects from. Shared by both prompt forms so
# the available tool vocabulary never depends on which framing is active.
#
# Structured source of truth: (section header, [(indicator, description), ...]).
# _INDICATOR_SECTIONS below is rendered from the shared catalog module
# (yiagents/dataflows/indicator_catalog.py — one source for this prompt copy,
# the y_finance tool gate, and the Alpha Vantage descriptions), and the
# ``indicator_battery`` config key prunes the same rendering — that gives the
# IC-pruning loop (scripts/prune_indicators_cli.py) a real config landing
# point. With the battery unset (the default) the render is byte-identical to
# the original hand-written catalog literal (pinned by
# tests/test_market_analyst_prompts.py and tests/test_indicator_battery.py).
_INDICATOR_SECTIONS: list[tuple[str, list[tuple[str, str]]]] = (
    indicator_catalog.ANALYST_SECTIONS
)

# Known indicator names — the validation set for ``indicator_battery``.
INDICATOR_NAMES = indicator_catalog.ANALYST_NAMES


def _render_indicator_catalog(battery: "list[str] | None" = None) -> str:
    """Render the catalog prose; ``battery`` keeps only the named indicators.

    Sections whose indicators are all pruned disappear entirely. ``None``
    renders the full catalog (the default, byte-identical baseline).
    """
    keep = set(battery) if battery is not None else None
    blocks: list[str] = []
    for header, entries in _INDICATOR_SECTIONS:
        selected = [e for e in entries if keep is None or e[0] in keep]
        if not selected:
            continue
        lines = [f"{header}:"]
        lines.extend(f"- {name}: {desc}" for name, desc in selected)
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


INDICATOR_CATALOG = _render_indicator_catalog()


def _catalog_for_run() -> str:
    """Battery-aware catalog for prompt building (``indicator_battery`` config).

    Fail-visible, never tool-less: an empty/unknown-only battery warns and
    falls back to the full catalog rather than leaving the analyst with no
    indicator vocabulary.
    """
    battery = get_config().get("indicator_battery")
    if battery is None:
        return INDICATOR_CATALOG
    names = list(dict.fromkeys(battery))  # dedupe, keep order
    unknown = [n for n in names if n not in INDICATOR_NAMES]
    if unknown:
        logger.warning(
            "indicator_battery contains unknown indicators (ignored): %s — "
            "known names: %s", unknown, sorted(INDICATOR_NAMES),
        )
    keep = [n for n in names if n in INDICATOR_NAMES]
    if not keep:
        logger.warning(
            "indicator_battery has no known indicators; using the full catalog"
        )
        return INDICATOR_CATALOG
    return _render_indicator_catalog(keep)


def _legacy_system_message() -> str:
    """Original persona-style prompt (the Phase-0 baseline shape)."""
    return (
        """You are a trading assistant tasked with analyzing financial markets. Your role is to select the **most relevant indicators** for a given market condition or trading strategy from the following list. The goal is to choose up to **8 indicators** that provide complementary insights without redundancy. Categories and each category's indicators are:

"""
        + _catalog_for_run()
        + """

- Select indicators that provide diverse and complementary information. Avoid redundancy (e.g., do not select both rsi and stochrsi). Also briefly explain why they are suitable for the given market context. When you tool call, please use the exact name of the indicators provided above as they are defined parameters, otherwise your call will fail. Please make sure to call get_stock_data first to retrieve the CSV that is needed to generate indicators. Then use get_indicators with the specific indicator names.

Before writing the final report, call get_verified_market_snapshot for this ticker and the current date, and treat it as the source of truth for any exact OHLCV, price-level, or indicator-value claim. If another tool's output conflicts with the verified snapshot, flag the discrepancy rather than inventing a reconciled number. Do not claim historical validation or exact percentage moves unless they are directly supported by tool output with concrete dates and prices. Support/resistance and breakout-level claims must come from get_support_resistance output; volume-confirmation or divergence claims must come from get_volume_features; candlestick and chart-pattern claims must come from get_candlestick_patterns (cite the pattern name and its date); outperformance/underperformance vs the benchmark must come from get_relative_strength.

You also have get_indicators_weekly (weekly timeframe resampled from the same data, trailing incomplete week excluded): call it once to frame the higher-timeframe trend (weekly 10/30 SMA, weekly RSI/MACD) before concluding. When the weekly trend conflicts with the daily indicators, state the conflict explicitly and justify which timeframe the decision should follow.

Write a very detailed and nuanced report of the trends you observe. Provide specific, actionable insights with supporting evidence to help traders make informed decisions."""
        + """ Make sure to append a Markdown table at the end of the report to organize key points in the report, organized and easy to read."""
    )


def _fincot_system_message() -> str:
    """Phase-2b FinCoT prompt: de-persona, task -> reasoning steps -> constraints.

    Same indicator vocabulary and grounding rules as the legacy form, packed
    into a compact three-section structure (no "You are a ... Analyst" framing).
    """
    return build_fincot_prompt(
        context=(
            "Select up to 8 complementary technical indicators for the requested "
            "market condition, then write an evidence-grounded trend report. "
            "Available indicators:\n\n" + _catalog_for_run()
        ),
        task=(
            "Pick the most relevant, non-redundant indicators (use their exact "
            "parameter names in tool calls), call get_stock_data first, then "
            "get_indicators, then get_verified_market_snapshot, and produce a "
            "detailed, actionable trend report ending with a Markdown summary table."
        ),
        reasoning_steps=[
            "Call get_stock_data to load the OHLCV CSV for the ticker and date.",
            "Choose up to 8 complementary indicators; avoid redundancy.",
            "Call get_indicators with the exact indicator names.",
            "Call get_indicators_weekly once to frame the higher-timeframe trend.",
            "Call get_verified_market_snapshot and treat it as ground truth for OHLCV/levels.",
            "Flag any conflict between tools instead of inventing a number.",
            "State weekly-vs-daily timeframe conflicts explicitly.",
            "Write the trend report with specific, dated, price-backed evidence.",
            "Append a Markdown table summarizing the key points.",
        ],
        output_constraints=[
            "Use exact indicator parameter names in every tool call.",
            "Do not assert historical validation or exact percentage moves unless a tool result with concrete dates/prices supports it.",
            "Support/resistance and breakout-level claims must cite get_support_resistance; volume/divergence claims must cite get_volume_features; candlestick and chart-pattern claims must cite get_candlestick_patterns with the pattern's date; benchmark out/under-performance claims must cite get_relative_strength.",
            "If a tool conflicts with the verified snapshot, flag the discrepancy.",
            "Weekly-trend claims must come from get_indicators_weekly output.",
        ],
        include_workflow=True,
    )


def _system_message() -> str:
    """Pick the prompt form based on config; defaults to the legacy baseline."""
    from yiagents.dataflows.config import get_config
    try:
        if get_config().get("fin_cot_prompts"):
            return _fincot_system_message() + get_language_instruction()
    except Exception:  # noqa: BLE001 -- prompt selection must never block a run
        # The run continues on the legacy prompt, but the whole-run analysis
        # framing silently changes vs what the user configured — visible, not
        # silent.
        logger.warning(
            "fin_cot prompt selection failed; falling back to the legacy "
            "market-analyst prompt",
            exc_info=True,
        )
    return _legacy_system_message() + get_language_instruction()


def create_market_analyst(llm):

    def market_analyst_node(state):
        current_date = state["trade_date"]
        instrument_context = get_instrument_context_from_state(state)
        historical = is_historical_date(current_date)

        # Baseline stock tools: daily OHLCV + indicators, the anti-hallucination
        # snapshot, the higher-timeframe weekly context, the price-structure
        # evidence tools (S/R levels, volume confirmation/divergence,
        # candlestick patterns), and benchmark relative strength — the
        # 2026-08-15 technical-analysis expansion. Crypto branches replace
        # this list below.
        tools = [
            get_stock_data,
            get_indicators,
            get_verified_market_snapshot,
            get_indicators_weekly,
            get_support_resistance,
            get_volume_features,
            get_candlestick_patterns,
            get_relative_strength,
        ]

        if state.get("asset_type") == "crypto_perp":
            # Spot tools resolve a perp symbol to a *different* Yahoo spot pair
            # (normalize_symbol("BTCUSDT") -> "BTC-USD"), so get_stock_data /
            # get_verified_market_snapshot would silently return spot
            # OHLCV/indicators while the analyst's primary data is the Binance
            # USDT-M perp. The spot-vs-perp basis would corrupt the
            # "ground truth" snapshot the anti-hallucination layer relies on,
            # so they stay hidden for perp runs. Classic indicators are NOT
            # lost though: get_binance_indicators (2026-08-15) computes the
            # stockstats battery on the actual perp klines.
            tools = [
                get_binance_klines,
                get_binance_funding_rate,
                get_binance_indicators,
            ]
            if not historical:
                tools.extend(
                    [
                        get_binance_open_interest,
                        get_binance_long_short_ratio,
                        get_binance_taker_buy_sell,
                        get_binance_basis,
                        get_binance_premium_index,
                    ]
                )
        elif state.get("asset_type") == "crypto_spot":
            # A Binance SPOT pair. Unlike perp, the symbol resolves correctly
            # via Yahoo (BTCUSDT -> BTC-USD), so get_stock_data /
            # get_indicators / get_verified_market_snapshot AND the five
            # TA-expansion evidence tools (weekly resample, S/R, volume
            # features, candlestick patterns, benchmark relative strength)
            # are kept — they price the same spot market and the base prompt
            # mandates citations from all of them. The spot-native klines +
            # 24h ticker are bound to the actual Binance spot book,
            # get_binance_spot_indicators computes on those same Binance
            # candles (venue defaults to spot — the plain perp-default tool
            # would silently price the perpetual if the LLM omitted venue),
            # and the cross-venue spot-perp basis tool exposes the
            # perpetual's premium/discount vs this spot reference.
            tools = [
                get_stock_data,
                get_binance_spot_klines,
                get_binance_spot_indicators,
                get_indicators,
                get_verified_market_snapshot,
                get_indicators_weekly,
                get_support_resistance,
                get_volume_features,
                get_candlestick_patterns,
                get_relative_strength,
            ]
            if not historical:
                tools[1:1] = [
                    get_binance_spot_ticker24,
                    get_binance_spot_perp_basis,
                ]

        # China A-share market tools (env: YIAGENTS_A_SHARE_NATIVE, off by
        # default). Same double-gate byte-equivalence contract — flag AND
        # is_a_stock(ticker). When either fails the tool list / prompt are
        # byte-for-byte identical to the prior behaviour, so US / crypto / HK
        # tickers never enter this branch. When both hold, A-share-native
        # market signals (northbound capital, sector fund-flow, real-time
        # quote, market breadth) are appended plus a short nudge.
        ticker = str(state["company_of_interest"])
        if get_config().get("a_share_native") and is_a_stock(ticker):
            tools.extend([
                get_a_share_northbound_native,
                get_a_share_sector_flow_native,
                get_a_share_realtime_quote_native,
                get_a_share_market_breadth_native,
            ])

        system_message = _system_message()

        # Composite regime context (config: regime_context, default ON;
        # env YIAGENTS_REGIME_CONTEXT=false restores the bare prompt). The
        # line is deterministic and fail-soft — omitted entirely when neither
        # trend nor volatility state has enough history.
        if get_config().get("regime_context", True):
            try:
                regime_line = format_regime_context(
                    ticker, current_date, asset_type=state.get("asset_type"),
                )
            except Exception:  # noqa: BLE001 — advisory, never block the node
                regime_line = None
            if regime_line:
                system_message += (
                    f"\n\n{regime_line}\n"
                    "(Advisory regime context — deterministic computation. "
                    "Weigh it in your read and say so when your conclusion "
                    "disagrees with it.)"
                )

        # Perp/spot-only system-message append; other asset types leave
        # system_message unchanged (byte-identical to the baseline).
        if state.get("asset_type") == "crypto_perp":
            system_message = system_message + (
                _PERP_HISTORICAL_NUDGE if historical else _PERP_NUDGE
            )
        elif state.get("asset_type") == "crypto_spot":
            system_message = system_message + (
                _SPOT_HISTORICAL_NUDGE if historical else _SPOT_NUDGE
            )

        # A-share market nudge uses the SAME double gate (flag AND is_a_stock)
        # as the tool extension above, so the prompt only changes when the
        # tools do (byte-equivalent when off or non-A-share).
        if get_config().get("a_share_native") and is_a_stock(ticker):
            system_message = system_message + _A_SHARE_MARKET_NUDGE

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
            result, agent_name="Market Analyst", ticker=ticker,
        )

        return {
            "messages": [result],
            "market_report": report,
        }

    return market_analyst_node
