"""The single source of truth for the technical-indicator catalog.

The indicator name -> description mapping previously existed in three
drifted copies:

* ``y_finance.get_stock_stats_indicators_window``'s ``best_ind_params``
  (the stockstats tool gate + report trailer; includes ``mfi``);
* ``market_analyst._INDICATOR_SECTIONS`` (the analyst prompt catalog);
* ``alpha_vantage_indicator.get_indicator``'s ``indicator_descriptions``
  (already drifted: no ``mfi``).

This module is the one catalog all three sites render from:

* :data:`TOOL_DESCRIPTIONS` — the indicator-window tools' gate and report
  trailer (every entry, including ``mfi``).
* :data:`ANALYST_SECTIONS` — the market analyst's prompt sections. That
  render is pinned byte-for-byte by ``tests/test_indicator_battery.py``,
  so entries carry :attr:`IndicatorSpec.in_analyst_prompt`; ``mfi`` is
  supported by the indicator tools but, per the pinned baseline, is not
  advertised in the analyst prompt.
* :data:`AV_SUPPORTED` / :data:`AV_DESCRIPTIONS` / :data:`AV_COLUMNS` —
  the Alpha Vantage mapping (display label + required ``series_type``,
  description, CSV response column), which now covers ``mfi`` as well.

Entry order is significant: it is the display order of both the tool
gate's "choose from" list and the analyst prompt sections.
"""

from __future__ import annotations

from typing import NamedTuple


class IndicatorSpec(NamedTuple):
    """One indicator's catalog entry.

    Attributes:
        name: The stockstats parameter name used in tool calls
            (e.g. ``close_50_sma``).
        label: Short display label (e.g. ``"50 SMA"``).
        section: The analyst-prompt section header the entry belongs to.
        description: The full ``"<label>: ... Usage: ... Tips: ..."`` text.
        in_analyst_prompt: Whether the market analyst's prompt catalog
            advertises the entry. ``mfi`` is ``False`` — the prompt render
            is pinned byte-for-byte to the pre-unification baseline, which
            did not include it.
        av_function: Alpha Vantage technical-indicator function name, or
            ``None`` when AV cannot serve the indicator (``vwma``).
        av_series_type: The ``series_type`` AV requires (``"close"``), or
            ``None`` for volume-based functions that take none (ATR, MFI).
        av_column: The indicator's column in AV's CSV response.
    """

    name: str
    label: str
    section: str
    description: str
    in_analyst_prompt: bool = True
    av_function: str | None = None
    av_series_type: str | None = "close"
    av_column: str = ""


_INDICATOR_LIST: list[IndicatorSpec] = [
    # Moving Averages
    IndicatorSpec(
        name="close_50_sma",
        label="50 SMA",
        section="Moving Averages",
        description=(
            "50 SMA: A medium-term trend indicator. "
            "Usage: Identify trend direction and serve as dynamic support/resistance. "
            "Tips: It lags price; combine with faster indicators for timely signals."
        ),
        av_function="SMA",
        av_column="SMA",
    ),
    IndicatorSpec(
        name="close_200_sma",
        label="200 SMA",
        section="Moving Averages",
        description=(
            "200 SMA: A long-term trend benchmark. "
            "Usage: Confirm overall market trend and identify golden/death cross setups. "
            "Tips: It reacts slowly; best for strategic trend confirmation rather than frequent trading entries."
        ),
        av_function="SMA",
        av_column="SMA",
    ),
    IndicatorSpec(
        name="close_10_ema",
        label="10 EMA",
        section="Moving Averages",
        description=(
            "10 EMA: A responsive short-term average. "
            "Usage: Capture quick shifts in momentum and potential entry points. "
            "Tips: Prone to noise in choppy markets; use alongside longer averages for filtering false signals."
        ),
        av_function="EMA",
        av_column="EMA",
    ),
    # MACD Related
    IndicatorSpec(
        name="macd",
        label="MACD",
        section="MACD Related",
        description=(
            "MACD: Computes momentum via differences of EMAs. "
            "Usage: Look for crossovers and divergence as signals of trend changes. "
            "Tips: Confirm with other indicators in low-volatility or sideways markets."
        ),
        av_function="MACD",
        av_column="MACD",
    ),
    IndicatorSpec(
        name="macds",
        label="MACD Signal",
        section="MACD Related",
        description=(
            "MACD Signal: An EMA smoothing of the MACD line. "
            "Usage: Use crossovers with the MACD line to trigger trades. "
            "Tips: Should be part of a broader strategy to avoid false positives."
        ),
        av_function="MACD",
        av_column="MACD_Signal",
    ),
    IndicatorSpec(
        name="macdh",
        label="MACD Histogram",
        section="MACD Related",
        description=(
            "MACD Histogram: Shows the gap between the MACD line and its signal. "
            "Usage: Visualize momentum strength and spot divergence early. "
            "Tips: Can be volatile; complement with additional filters in fast-moving markets."
        ),
        av_function="MACD",
        av_column="MACD_Hist",
    ),
    # Momentum Indicators
    IndicatorSpec(
        name="rsi",
        label="RSI",
        section="Momentum Indicators",
        description=(
            "RSI: Measures momentum to flag overbought/oversold conditions. "
            "Usage: Apply 70/30 thresholds and watch for divergence to signal reversals. "
            "Tips: In strong trends, RSI may remain extreme; always cross-check with trend analysis."
        ),
        av_function="RSI",
        av_column="RSI",
    ),
    # Volatility Indicators
    IndicatorSpec(
        name="boll",
        label="Bollinger Middle",
        section="Volatility Indicators",
        description=(
            "Bollinger Middle: A 20 SMA serving as the basis for Bollinger Bands. "
            "Usage: Acts as a dynamic benchmark for price movement. "
            "Tips: Combine with the upper and lower bands to effectively spot breakouts or reversals."
        ),
        av_function="BBANDS",
        av_column="Real Middle Band",
    ),
    IndicatorSpec(
        name="boll_ub",
        label="Bollinger Upper Band",
        section="Volatility Indicators",
        description=(
            "Bollinger Upper Band: Typically 2 standard deviations above the middle line. "
            "Usage: Signals potential overbought conditions and breakout zones. "
            "Tips: Confirm signals with other tools; prices may ride the band in strong trends."
        ),
        av_function="BBANDS",
        av_column="Real Upper Band",
    ),
    IndicatorSpec(
        name="boll_lb",
        label="Bollinger Lower Band",
        section="Volatility Indicators",
        description=(
            "Bollinger Lower Band: Typically 2 standard deviations below the middle line. "
            "Usage: Indicates potential oversold conditions. "
            "Tips: Use additional analysis to avoid false reversal signals."
        ),
        av_function="BBANDS",
        av_column="Real Lower Band",
    ),
    IndicatorSpec(
        name="atr",
        label="ATR",
        section="Volatility Indicators",
        description=(
            "ATR: Averages true range to measure volatility. "
            "Usage: Set stop-loss levels and adjust position sizes based on current market volatility. "
            "Tips: It's a reactive measure, so use it as part of a broader risk management strategy."
        ),
        av_function="ATR",
        av_series_type=None,
        av_column="ATR",
    ),
    # Volume-Based Indicators
    IndicatorSpec(
        name="vwma",
        label="VWMA",
        section="Volume-Based Indicators",
        description=(
            "VWMA: A moving average weighted by volume. "
            "Usage: Confirm trends by integrating price action with volume data. "
            "Tips: Watch for skewed results from volume spikes; use in combination with other volume analyses."
        ),
        av_function=None,  # not directly available from Alpha Vantage
        av_column="",
    ),
    IndicatorSpec(
        name="mfi",
        label="MFI",
        section="Volume-Based Indicators",
        description=(
            "MFI: The Money Flow Index is a momentum indicator that uses both price and volume to measure buying and selling pressure. "
            "Usage: Identify overbought (>80) or oversold (<20) conditions and confirm the strength of trends or reversals. "
            "Tips: Use alongside RSI or MACD to confirm signals; divergence between price and MFI can indicate potential reversals."
        ),
        in_analyst_prompt=False,  # pinned prompt baseline does not advertise mfi
        av_function="MFI",
        av_series_type=None,
        av_column="MFI",
    ),
]

#: The catalog itself, in display order (tool gate list + prompt sections).
INDICATORS: dict[str, IndicatorSpec] = {spec.name: spec for spec in _INDICATOR_LIST}

#: y_finance's indicator-window gate + report trailer: every entry.
TOOL_DESCRIPTIONS: dict[str, str] = {
    spec.name: spec.description for spec in _INDICATOR_LIST
}

#: The market analyst's prompt sections: (header, [(name, description), ...]).
ANALYST_SECTIONS: list[tuple[str, list[tuple[str, str]]]] = [
    (
        header,
        [
            (spec.name, spec.description)
            for spec in _INDICATOR_LIST
            if spec.section == header and spec.in_analyst_prompt
        ],
    )
    for header in ("Moving Averages", "MACD Related", "Momentum Indicators",
                   "Volatility Indicators", "Volume-Based Indicators")
]

#: Known indicator names for the analyst prompt — the ``indicator_battery``
#: validation set (market_analyst.INDICATOR_NAMES renders from this).
ANALYST_NAMES = frozenset(name for _, entries in ANALYST_SECTIONS for name, _ in entries)

#: Alpha Vantage gate: name -> (display label, required series_type or None).
AV_SUPPORTED: dict[str, tuple[str, str | None]] = {
    spec.name: (spec.label, spec.av_series_type) for spec in _INDICATOR_LIST
}

#: Alpha Vantage descriptions (same set as the catalog — mfi included).
AV_DESCRIPTIONS: dict[str, str] = dict(TOOL_DESCRIPTIONS)

#: Alpha Vantage CSV response column per indicator (empty columns excluded).
AV_COLUMNS: dict[str, str] = {
    spec.name: spec.av_column for spec in _INDICATOR_LIST if spec.av_column
}
