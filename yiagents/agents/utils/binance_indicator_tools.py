"""Classic stockstats indicators computed on Binance klines.

The perp market analyst previously had NO computed indicators — the spot-style
tools are hidden for perp runs (they resolve the symbol to the wrong Yahoo
pair), so trend/momentum/volatility readings were read by eye from raw
candles. This tool computes the same stockstats battery the stock path uses,
on the actual Binance candles (perp or spot venue), closing the gap. Derived
features (vol estimators) dispatch through the shared feature registry.
"""

from __future__ import annotations

from typing import Annotated

import pandas as pd
from langchain_core.tools import tool
from stockstats import wrap

from yiagents.dataflows.binance import binance_klines_frame
from yiagents.dataflows.feature_registry import DERIVED_FEATURES, compute_derived
from yiagents.dataflows.indicator_catalog import INDICATORS
from yiagents.dataflows.stockstats_utils import compute_indicator

#: Default battery for crypto: trend + momentum + volatility groups (the
#: perp-appropriate set; positioning/funding live in their own perp tools).
BINANCE_INDICATOR_DEFAULTS: tuple[str, ...] = (
    "close_50_sma", "close_200_sma", "macd", "rsi", "atr",
    "adx", "kdjk", "cci", "supertrend", "boll_ub", "boll_lb", "rvol_20",
)

#: Calendar days fetched per lookback row (crypto trades daily, weekends
#: included, but a margin keeps SMA-200 warm at modest lookbacks).
_FETCH_MARGIN = 1.3


@tool
def get_binance_indicators(
    symbol: Annotated[str, "Binance pair symbol, e.g. BTCUSDT"],
    curr_date: Annotated[str, "The current trading date you are trading on, YYYY-mm-dd"],
    look_back_days: Annotated[int, "how many daily bars of indicator history to show"] = 15,
    venue: Annotated[str, "'perp' for USDT-M perpetual, 'spot' for the spot pair"] = "perp",
    indicators: Annotated[str, "comma-separated indicator names (default: the crypto battery)"] = "",
) -> str:
    """
    Classic technical indicators (SMA/EMA, MACD, RSI, KDJ, CCI, ADX,
    SuperTrend, Bollinger, ATR, realized vol) computed on Binance klines for
    the actual pair — perp or spot venue. Use for any exact indicator claim;
    the funding/OI/positioning tools remain the perp-native signals.
    """
    names = (
        [n.strip() for n in indicators.split(",") if n.strip()]
        if indicators
        else list(BINANCE_INDICATOR_DEFAULTS)
    )
    unknown = [n for n in names if n not in INDICATORS]
    if unknown:
        return (
            f"ERROR: unknown indicator name(s) {unknown}. Valid names are the "
            f"catalog: {sorted(INDICATORS)}. Retry with valid names."
        )
    if venue not in ("perp", "spot"):
        return "ERROR: venue must be 'perp' or 'spot'."

    # Warm-up margin: SMA-200 needs 200 rows; ~260 calendar days of crypto
    # bars covers it, multiplied for safety on longer batteries.
    fetch_days = max(int(400 * _FETCH_MARGIN), int(look_back_days * 3))
    start = (
        pd.Timestamp(curr_date) - pd.Timedelta(days=fetch_days)
    ).strftime("%Y-%m-%d")

    try:
        frame = binance_klines_frame(
            symbol,
            start,
            curr_date,
            interval="1d",
            venue="binance_perp" if venue == "perp" else "binance_spot",
        )
    except Exception as exc:  # noqa: BLE001 — typed degrade, never crash the node
        return (
            f"DATA_UNAVAILABLE: Binance klines for {symbol!r} ({venue}) up to "
            f"{curr_date} could not be fetched ({type(exc).__name__}: {exc}). "
            "Report indicators as unavailable; do not estimate them."
        )

    if frame.empty or len(frame) < 30:
        return (
            f"DATA_UNAVAILABLE: fewer than 30 daily bars for {symbol!r} up to "
            f"{curr_date}; indicators not computable."
        )

    sdf = wrap(frame.copy().reset_index())
    columns: dict[str, pd.Series] = {}
    skipped: list[str] = []
    for name in names:
        try:
            if name in DERIVED_FEATURES:
                derived = compute_derived(frame.reset_index(), name)
                if derived is None:
                    raise RuntimeError("registry miss")
                columns[name] = derived
            else:
                columns[name] = compute_indicator(sdf, name)
        except Exception:  # noqa: BLE001 — skip-and-report, never zero-fill
            skipped.append(name)

    if not columns:
        return (
            f"DATA_UNAVAILABLE: none of {names} could be computed on Binance "
            f"klines for {symbol!r}."
        )

    rows_out = max(1, min(int(look_back_days), 30))
    dates = pd.to_datetime(frame.reset_index()["Date"]).dt.strftime("%Y-%m-%d")
    closes = frame["Close"].round(2)
    lines = [
        f"## Binance {venue} indicators for {symbol.upper()} "
        f"(daily klines, as of {curr_date})",
        "",
        "| Date | Close | " + " | ".join(columns) + " |",
        "|---|---:|" + "---:|" * (2 + len(columns)),
    ]
    for i in range(len(frame) - rows_out, len(frame)):
        cells = [str(dates.iloc[i]), f"{closes.iloc[i]:.2f}"]
        for _, col in columns.items():
            value = col.iloc[i] if i < len(col) else float("nan")
            cells.append("N/A" if pd.isna(value) else f"{value:.2f}")
        lines.append("| " + " | ".join(cells) + " |")

    last_i = len(frame) - 1
    summary = ", ".join(
        f"{n}={'N/A' if pd.isna(c.iloc[last_i]) else f'{c.iloc[last_i]:.2f}'}"
        for n, c in columns.items()
    )
    lines += ["", f"Latest ({dates.iloc[last_i]}): {summary}."]
    if skipped:
        lines.append(f"(Skipped — not computable on this window: {', '.join(skipped)}.)")
    lines.append(
        "Cite this tool for indicator claims on this pair; funding/OI/"
        "positioning claims cite their own perp tools."
    )
    return "\n".join(lines)
