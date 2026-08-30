"""Weekly-timeframe indicator tool for the market analyst.

Multi-timeframe confirmation: the analyst's daily indicators frame the entry
decision, this tool frames the higher-timeframe trend around it. It resamples
the same PIT-safe daily OHLCV to weekly bars (dropping the still-open week)
and computes a fixed small indicator set on them via stockstats, so weekly
trend claims have tool evidence exactly like daily ones — the anti-
hallucination contract extended to the higher timeframe.
"""

from __future__ import annotations

from typing import Annotated

import pandas as pd
from langchain_core.tools import tool
from stockstats import wrap

from yialpha.dataflows import quality
from yialpha.dataflows.ohlcv_resample import weekly_ohlcv
from yialpha.dataflows.stockstats_utils import compute_indicator

#: The fixed weekly battery: enough to state the weekly trend direction,
#: momentum, and trend-vs-momentum conflict without re-opening the whole
#: daily catalog on a second timeframe.
WEEKLY_INDICATORS: tuple[str, ...] = (
    "close_10_sma",  # ~2-month weekly trend
    "close_30_sma",  # ~6-month weekly trend
    "rsi",
    "macd",
)

_WEEKLY_DESCRIPTIONS = {
    "close_10_sma": "weekly close 10 SMA (~2-month trend)",
    "close_30_sma": "weekly close 30 SMA (~6-month trend)",
    "rsi": "weekly RSI (14)",
    "macd": "weekly MACD (12,26,9)",
}


@tool
def get_indicators_weekly(
    symbol: Annotated[str, "ticker symbol of the company"],
    curr_date: Annotated[str, "The current trading date you are trading on, YYYY-mm-dd"],
    look_back_weeks: Annotated[int, "how many weekly bars to look back"] = 26,
) -> str:
    """
    Retrieve the higher-timeframe (weekly) trend context for a ticker: weekly
    bars resampled from the same PIT-safe daily OHLCV, with a fixed indicator
    battery (weekly 10/30 SMA, weekly RSI, weekly MACD). Use once per analysis
    to frame the weekly trend; when it conflicts with the daily indicators,
    state the conflict explicitly.
    Args:
        symbol (str): Ticker symbol, e.g. AAPL, 600519.SS
        curr_date (str): The current trading date, YYYY-mm-dd
        look_back_weeks (int): How many weekly bars back (default 26 ≈ 6 months)
    Returns:
        str: Weekly indicator table (one row per week, columns = indicators)
        plus the latest-row summary.
    """
    try:
        weekly = weekly_ohlcv(symbol, curr_date)
    except Exception as exc:  # noqa: BLE001 — typed degrade, never crash the node
        quality.record_sentinel(
            "get_indicators_weekly",
            quality.KIND_OPTIONAL_UNAVAILABLE,
            f"{symbol}: {type(exc).__name__}: {exc}",
        )
        return (
            f"DATA_UNAVAILABLE: weekly OHLCV for {symbol!r} could not be "
            f"resampled as of {curr_date} ({type(exc).__name__}: {exc}). "
            "Report the weekly timeframe as unavailable; do not estimate it."
        )

    if weekly.empty or len(weekly) < 2:
        quality.record_sentinel(
            "get_indicators_weekly",
            quality.KIND_OPTIONAL_UNAVAILABLE,
            f"{symbol}: fewer than 2 complete weekly bars as of {curr_date}",
        )
        return (
            f"DATA_UNAVAILABLE: fewer than 2 complete weekly bars for "
            f"{symbol!r} as of {curr_date}; weekly timeframe not assessable."
        )

    sdf = wrap(weekly.copy())
    weekly["Date"] = pd.to_datetime(weekly["Date"]).dt.strftime("%Y-%m-%d")

    columns: dict[str, pd.Series] = {}
    for name in WEEKLY_INDICATORS:
        try:
            columns[name] = compute_indicator(sdf, name)
        except Exception as exc:  # noqa: BLE001 — one column must not sink the table
            columns[name] = pd.Series(dtype=float)
            columns[name].name = f"{name} (unavailable: {type(exc).__name__})"

    window = max(1, min(int(look_back_weeks), len(weekly)))
    frame = weekly.tail(window).reset_index(drop=True)
    for name, col in columns.items():
        frame[name] = col.tail(window).reset_index(drop=True)

    lines = [
        f"## Weekly ({frame['Date'].iloc[0]} .. {frame['Date'].iloc[-1]}, "
        f"{len(frame)} complete weeks, W-FRI) — higher-timeframe context",
        "",
        "| Week ending | Close | "
        + " | ".join(
            f"{n} ({_WEEKLY_DESCRIPTIONS[n]})" for n in columns
        )
        + " |",
        "|---|---:|" + "---:|" * (2 + len(columns)),
    ]
    for _, row in frame.iterrows():
        cells = [row["Date"], f"{row['Close']:.2f}"]
        for name in columns:
            value = row.get(name)
            cells.append("N/A" if pd.isna(value) else f"{value:.2f}")
        lines.append("| " + " | ".join(cells) + " |")

    last = frame.iloc[-1]
    summary = ", ".join(
        f"{_WEEKLY_DESCRIPTIONS[n]}={('N/A' if pd.isna(last.get(n)) else f'{last.get(n):.2f}')}"
        for n in columns
    )
    lines += [
        "",
        f"Latest complete week ({last['Date']}): {summary}.",
        "The trailing incomplete week is excluded (its close is not final).",
        "Treat this table as the source of truth for weekly-trend claims; "
        "when the weekly trend conflicts with your daily read, state the "
        "conflict explicitly rather than quietly following one side.",
    ]
    return "\n".join(lines)
