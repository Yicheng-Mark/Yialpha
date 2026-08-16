"""Classic stockstats indicators computed on Binance klines.

The perp market analyst previously had NO computed indicators — the spot-style
tools are hidden for perp runs (they resolve the symbol to the wrong Yahoo
pair), so trend/momentum/volatility readings were read by eye from raw
candles. This tool computes the same stockstats battery the stock path uses,
on the actual Binance candles (perp or spot venue), closing the gap. Derived
features (vol estimators) dispatch through the shared feature registry.

Two bindings share one implementation: ``get_binance_indicators`` (venue
defaults to perp) and ``get_binance_spot_indicators`` (venue defaults to
spot). The default matters because the tool is stateless — it cannot see the
run's asset_type — and a spot run silently computing on perp candles (or vice
versa) is exactly the wrong-market failure the venue split exists to prevent.
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
from yiagents.dataflows.vol_estimators import CRYPTO_TRADING_DAYS_PER_YEAR

#: Default battery for crypto: trend + momentum + volatility groups (the
#: perp-appropriate set; positioning/funding live in their own perp tools).
BINANCE_INDICATOR_DEFAULTS: tuple[str, ...] = (
    "close_50_sma", "close_200_sma", "macd", "rsi", "atr",
    "adx", "kdjk", "cci", "supertrend", "boll_ub", "boll_lb", "rvol_20",
)

#: Calendar days fetched per lookback row (crypto trades daily, weekends
#: included, but a margin makes SMA-200 warm at modest lookbacks).
_FETCH_MARGIN = 1.3


def _fmt_val(value: object) -> str:
    """Format a price/indicator value with decimals adaptive to magnitude.

    Binance pairs span 1e-5 (PEPE) to 1e5 (BTC); a fixed ``.2f`` zeroes
    sub-cent prices — the display twin of the round(2) data bug — and hides
    real differences between indicator readings on low-price contracts.
    """
    try:
        v = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return "N/A"
    if pd.isna(v):
        return "N/A"
    a = abs(v)
    if a >= 100:
        return f"{v:,.2f}"
    if a >= 1:
        return f"{v:.4f}"
    if a >= 0.01:
        return f"{v:.6f}"
    if a >= 1e-6:
        return f"{v:.8f}"
    return f"{v:.10f}"


def _indicators_core(
    symbol: str,
    curr_date: str,
    look_back_days: int,
    venue: str,
    indicators: str,
) -> str:
    """Shared implementation behind the perp/spot indicator tools."""
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

    frame_reset = frame.reset_index()
    # The wrap() copy is load-bearing: stockstats converts/adds columns on the
    # frame it wraps, and frame_reset stays the read-only source below.
    sdf = wrap(frame_reset.copy())
    columns: dict[str, pd.Series] = {}
    skipped: list[str] = []
    for name in names:
        try:
            if name in DERIVED_FEATURES:
                # compute_derived never mutates its input (obv copies
                # internally; rvol/ewma_vol/rel_vol_20 are read-only) — the
                # old per-name full-frame copy was one DataFrame copy per
                # derived indicator, ~12 per tool call.
                # Crypto candles are a 24/7 daily series: annualize the vol
                # estimators with 365 — the 252 equity default understates
                # every vol reading by sqrt(252/365) ≈ 0.83 (P1, 2026-08-16).
                derived = compute_derived(
                    frame_reset, name, periods_per_year=CRYPTO_TRADING_DAYS_PER_YEAR,
                )
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
    dates = pd.to_datetime(frame_reset["Date"]).dt.strftime("%Y-%m-%d")
    lines = [
        f"## Binance {venue} indicators for {symbol.upper()} "
        f"(daily klines, as of {curr_date})",
        "",
        "| Date | Close | " + " | ".join(columns) + " |",
        "|---|---:|" + "---:|" * (2 + len(columns)),
    ]
    for i in range(len(frame) - rows_out, len(frame)):
        cells = [str(dates.iloc[i]), _fmt_val(frame["Close"].iloc[i])]
        for _, col in columns.items():
            value = col.iloc[i] if i < len(col) else float("nan")
            cells.append(_fmt_val(value))
        lines.append("| " + " | ".join(cells) + " |")

    last_i = len(frame) - 1
    summary = ", ".join(
        f"{n}={_fmt_val(c.iloc[last_i]) if last_i < len(c) else 'N/A'}"
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


@tool
def get_binance_indicators(
    symbol: Annotated[str, "Binance pair symbol, e.g. BTCUSDT"],
    curr_date: Annotated[str, "The current trading date you are trading on, YYYY-mm-dd"],
    look_back_days: Annotated[int, "how many daily bars of indicator history to show"] = 15,
    venue: Annotated[str, "'perp' (default, USDT-M perpetual) or 'spot' for the spot pair"] = "perp",
    indicators: Annotated[str, "comma-separated indicator names (default: the crypto battery)"] = "",
) -> str:
    """
    Classic technical indicators (SMA/EMA, MACD, RSI, KDJ, CCI, ADX,
    SuperTrend, Bollinger, ATR, realized vol) computed on Binance klines for
    the actual pair — venue defaults to the USDT-M PERPETUAL (pass
    venue='spot' only when you deliberately want the spot pair). Use for any
    exact indicator claim; the funding/OI/positioning tools remain the
    perp-native signals.
    """
    return _indicators_core(symbol, curr_date, look_back_days, venue, indicators)


@tool
def get_binance_spot_indicators(
    symbol: Annotated[str, "Binance pair symbol, e.g. BTCUSDT"],
    curr_date: Annotated[str, "The current trading date you are trading on, YYYY-mm-dd"],
    look_back_days: Annotated[int, "how many daily bars of indicator history to show"] = 15,
    venue: Annotated[str, "'spot' (default, the spot pair) or 'perp' for the USDT-M perpetual"] = "spot",
    indicators: Annotated[str, "comma-separated indicator names (default: the crypto battery)"] = "",
) -> str:
    """
    Classic technical indicators (SMA/EMA, MACD, RSI, KDJ, CCI, ADX,
    SuperTrend, Bollinger, ATR, realized vol) computed on Binance klines for
    the actual pair — venue defaults to the SPOT pair (pass venue='perp' only
    when you deliberately want the perpetual). Use for any exact indicator
    claim on a crypto_spot run.
    """
    return _indicators_core(symbol, curr_date, look_back_days, venue, indicators)
