from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from yiagents.agents.utils.agent_utils import (
    get_binance_basis,
    get_binance_funding_rate,
    get_binance_klines,
    get_binance_long_short_ratio,
    get_binance_open_interest,
    get_binance_spot_klines,
    get_binance_spot_perp_basis,
    get_binance_spot_ticker24,
    get_binance_taker_buy_sell,
    get_indicators,
    get_instrument_context_from_state,
    get_language_instruction,
    get_stock_data,
    get_verified_market_snapshot,
)
from yiagents.agents.utils.prompt_builder import build_fincot_prompt

# Appended to the system message ONLY for crypto_perp runs. Nudges the analyst
# to use the perp-native OHLCV and to treat funding/OI + positioning/order-flow
# as required signals. Non-perp runs append nothing, so the prompt is byte-
# identical to the baseline (stock/crypto unaffected).
_PERP_NUDGE = (
    " This is a Binance USDT-M PERPETUAL, not a spot pair. The spot-style tools "
    "(get_stock_data, get_indicators, get_verified_market_snapshot) are NOT "
    "available for this instrument — they would resolve the symbol to a "
    "different spot pair and return the wrong market. Use get_binance_klines "
    "for OHLCV (you may compute trend/volatility observations directly from "
    "those candles). You MUST call get_binance_funding_rate and "
    "get_binance_open_interest and discuss funding-rate direction, open-interest "
    "crowding, and liquidation risk in your report. You MUST ALSO call "
    "get_binance_long_short_ratio and discuss how the leveraged crowd is "
    "positioned (longShortRatio > 1 = longs dominate; top-trader vs global "
    "divergence is a contrary signal) — this is the perp-native sentiment signal "
    "and it frequently contradicts funding/OI inferences, so reconcile them "
    "explicitly. Call get_binance_taker_buy_sell (order-flow aggression) and, "
    "where available, get_binance_basis (perp-vs-index premium/discount) to "
    "round out the positioning picture; if a tool returns a sentinel/unavailable "
    "marker, say so plainly rather than inventing values."
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
    "There is no funding rate, open interest, leverage, or liquidation for a "
    "spot pair — do not discuss them. Call get_binance_spot_perp_basis to show "
    "where the USDT-M perpetual trades relative to this spot price (positive "
    "basis = perp rich / long premium; negative = discount / short pressure) "
    "and reconcile it with the spot trend. The get_indicators / "
    "get_verified_market_snapshot tools are available and resolve correctly for "
    "this symbol — use them as usual. If a tool returns a sentinel/unavailable "
    "marker, say so plainly rather than inventing values."
)

# The indicator catalog the analyst selects from. Shared by both prompt forms so
# the available tool vocabulary never depends on which framing is active.
INDICATOR_CATALOG = """Moving Averages:
- close_50_sma: 50 SMA: A medium-term trend indicator. Usage: Identify trend direction and serve as dynamic support/resistance. Tips: It lags price; combine with faster indicators for timely signals.
- close_200_sma: 200 SMA: A long-term trend benchmark. Usage: Confirm overall market trend and identify golden/death cross setups. Tips: It reacts slowly; best for strategic trend confirmation rather than frequent trading entries.
- close_10_ema: 10 EMA: A responsive short-term average. Usage: Capture quick shifts in momentum and potential entry points. Tips: Prone to noise in choppy markets; use alongside longer averages for filtering false signals.

MACD Related:
- macd: MACD: Computes momentum via differences of EMAs. Usage: Look for crossovers and divergence as signals of trend changes. Tips: Confirm with other indicators in low-volatility or sideways markets.
- macds: MACD Signal: An EMA smoothing of the MACD line. Usage: Use crossovers with the MACD line to trigger trades. Tips: Should be part of a broader strategy to avoid false positives.
- macdh: MACD Histogram: Shows the gap between the MACD line and its signal. Usage: Visualize momentum strength and spot divergence early. Tips: Can be volatile; complement with additional filters in fast-moving markets.

Momentum Indicators:
- rsi: RSI: Measures momentum to flag overbought/oversold conditions. Usage: Apply 70/30 thresholds and watch for divergence to signal reversals. Tips: In strong trends, RSI may remain extreme; always cross-check with trend analysis.

Volatility Indicators:
- boll: Bollinger Middle: A 20 SMA serving as the basis for Bollinger Bands. Usage: Acts as a dynamic benchmark for price movement. Tips: Combine with the upper and lower bands to effectively spot breakouts or reversals.
- boll_ub: Bollinger Upper Band: Typically 2 standard deviations above the middle line. Usage: Signals potential overbought conditions and breakout zones. Tips: Confirm signals with other tools; prices may ride the band in strong trends.
- boll_lb: Bollinger Lower Band: Typically 2 standard deviations below the middle line. Usage: Indicates potential oversold conditions. Tips: Use additional analysis to avoid false reversal signals.
- atr: ATR: Averages true range to measure volatility. Usage: Set stop-loss levels and adjust position sizes based on current market volatility. Tips: It's a reactive measure, so use it as part of a broader risk management strategy.

Volume-Based Indicators:
- vwma: VWMA: A moving average weighted by volume. Usage: Confirm trends by integrating price action with volume data. Tips: Watch for skewed results from volume spikes; use in combination with other volume analyses."""


def _legacy_system_message() -> str:
    """Original persona-style prompt (the Phase-0 baseline shape)."""
    return (
        """You are a trading assistant tasked with analyzing financial markets. Your role is to select the **most relevant indicators** for a given market condition or trading strategy from the following list. The goal is to choose up to **8 indicators** that provide complementary insights without redundancy. Categories and each category's indicators are:

"""
        + INDICATOR_CATALOG
        + """

- Select indicators that provide diverse and complementary information. Avoid redundancy (e.g., do not select both rsi and stochrsi). Also briefly explain why they are suitable for the given market context. When you tool call, please use the exact name of the indicators provided above as they are defined parameters, otherwise your call will fail. Please make sure to call get_stock_data first to retrieve the CSV that is needed to generate indicators. Then use get_indicators with the specific indicator names.

Before writing the final report, call get_verified_market_snapshot for this ticker and the current date, and treat it as the source of truth for any exact OHLCV, price-level, or indicator-value claim. If another tool's output conflicts with the verified snapshot, flag the discrepancy rather than inventing a reconciled number. Do not claim historical validation, support/resistance bounces, or exact percentage moves unless they are directly supported by tool output with concrete dates and prices.

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
            "Available indicators:\n\n" + INDICATOR_CATALOG
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
            "Call get_verified_market_snapshot and treat it as ground truth for OHLCV/levels.",
            "Flag any conflict between tools instead of inventing a number.",
            "Write the trend report with specific, dated, price-backed evidence.",
            "Append a Markdown table summarizing the key points.",
        ],
        output_constraints=[
            "Use exact indicator parameter names in every tool call.",
            "Do not assert support/resistance bounces, historical validation, or exact percentage moves unless a tool result with concrete dates/prices supports it.",
            "If a tool conflicts with the verified snapshot, flag the discrepancy.",
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
        pass
    return _legacy_system_message() + get_language_instruction()


def create_market_analyst(llm):

    def market_analyst_node(state):
        current_date = state["trade_date"]
        instrument_context = get_instrument_context_from_state(state)

        # Baseline stock tools. Non-perp runs bind exactly this 3-element list
        # (same objects, same order) so the baseline path is byte-identical.
        tools = [
            get_stock_data,
            get_indicators,
            get_verified_market_snapshot,
        ]

        if state.get("asset_type") == "crypto_perp":
            # Spot tools resolve a perp symbol to a *different* Yahoo spot pair
            # (normalize_symbol("BTCUSDT") -> "BTC-USD"), so get_stock_data /
            # get_indicators / get_verified_market_snapshot would silently return
            # spot OHLCV/indicators while the analyst's primary data is the
            # Binance USDT-M perp. The spot-vs-perp basis would corrupt the
            # "ground truth" snapshot the anti-hallucination layer relies on, so
            # hide them for perp runs and bind only the perp-native tools. RSI/
            # MACD-style indicators are therefore not computed for perp (the
            # analyst reads the klines CSV directly) — an accepted trade-off.
            # The non-perp branch is untouched, so stock runs are unaffected.
            tools = [
                get_binance_klines,
                get_binance_funding_rate,
                get_binance_open_interest,
                get_binance_long_short_ratio,
                get_binance_taker_buy_sell,
                get_binance_basis,
            ]
        elif state.get("asset_type") == "crypto_spot":
            # A Binance SPOT pair. Unlike perp, the symbol resolves correctly
            # via Yahoo (BTCUSDT -> BTC-USD), so get_indicators /
            # get_verified_market_snapshot are kept — they price the same spot
            # market. The spot-native klines + 24h ticker are bound to the
            # actual Binance spot book, and the cross-venue spot-perp basis
            # tool exposes the perpetual's premium/discount vs this spot
            # reference. Stock/crypto/perp branches are untouched.
            tools = [
                get_binance_spot_klines,
                get_binance_spot_ticker24,
                get_binance_spot_perp_basis,
                get_indicators,
                get_verified_market_snapshot,
            ]

        system_message = _system_message()
        # Perp/spot-only system-message append; other asset types leave
        # system_message unchanged (byte-identical to the baseline).
        if state.get("asset_type") == "crypto_perp":
            system_message = system_message + _PERP_NUDGE
        elif state.get("asset_type") == "crypto_spot":
            system_message = system_message + _SPOT_NUDGE

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
            "market_report": report,
        }

    return market_analyst_node
