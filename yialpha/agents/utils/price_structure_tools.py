"""Price-structure tools for the market analyst: S/R, volume, patterns.

These tools turn the three previously prompt-prose-only analysis families
into computed evidence on the same PIT-safe OHLCV the indicator tools use:

* ``get_support_resistance`` — pivots (daily/weekly), rolling highs/lows,
  volume profile (WP5);
* ``get_volume_features`` — OBV/relative-volume/vr readings plus the
  deterministic price-vs-OBV divergence check (WP7);
* ``get_candlestick_patterns`` — hand-rolled candlestick shapes and
  pivot-based double top/bottom (WP8).

Every tool degrades to a typed DATA_UNAVAILABLE sentinel instead of raising
into the node, and each render tells the analyst to CITE it for the matching
claim family — evidence-gated claims, not banned claims.
"""

from __future__ import annotations

from typing import Annotated

import pandas as pd
from langchain_core.tools import tool
from stockstats import wrap

from yialpha.dataflows import quality
from yialpha.dataflows.candlestick_patterns import (
    detect_double_top_bottom,
    scan_candlestick_patterns,
)
from yialpha.dataflows.market_regime import resolve_market_benchmark
from yialpha.dataflows.stockstats_utils import compute_indicator, load_ohlcv
from yialpha.dataflows.support_resistance import build_support_resistance
from yialpha.dataflows.volume_features import obv, relative_volume, volume_divergence


def _fmt(value: float | int | None, digits: int = 2) -> str:
    try:
        if value is None or pd.isna(value):
            return "N/A"
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "N/A"


def _fmt_price(value: float | int | None) -> str:
    """Magnitude-aware PRICE rendering (not for percentages/counts).

    These tools are bound on the crypto_spot track, where PEPEUSDT-class
    symbols resolve to ~1e-5 prices — ``.2f`` rendered every level as
    "0.00" and destroyed the mandated-citation evidence at the render seam
    while the math was correct. Three tiers mirror ``trade_ticket._money``:
    thousands with separators, normal range to 4 decimals (trailing zeros
    stripped), micro prices in ``%.4g`` significant digits.
    """
    import math

    try:
        if value is None or pd.isna(value):
            return "N/A"
        x = float(value)
        if not math.isfinite(x):
            return "N/A"
    except (TypeError, ValueError):
        return "N/A"
    a = abs(x)
    if a >= 1000:
        return f"{x:,.2f}"
    if a >= 1e-3:
        stripped = f"{x:.4f}".rstrip("0").rstrip(".")
        return stripped if stripped not in ("", "-") else "0"
    return f"{x:.4g}"


def _load(symbol: str, curr_date: str) -> pd.DataFrame | str:
    """Shared OHLCV loader for the price-structure tools.

    Degrades to a DATA_UNAVAILABLE sentinel (never raises into the node) and
    records the degrade in the run's data_quality ledger — these tools bypass
    the vendor router, so without this the evidence chain has a blind spot.
    """
    # PIT clamp: curr_date comes from the LLM; the pinned analysis date
    # (set_analysis_date ContextVar) is the authority a historical replay
    # must not see past — the same guard the news vendors carry
    # (yfinance_news / alpha_vantage_news). get_stock_data clamps inside
    # y_finance; these TA tools feed load_ohlcv directly, so clamp here.
    from yialpha.dataflows.utils import current_pit_end

    curr_date = str(current_pit_end(curr_date) or curr_date)
    try:
        data = load_ohlcv(symbol, curr_date)
    except Exception as exc:  # noqa: BLE001 — typed degrade, never crash the node
        quality.record_sentinel(
            "price_structure_ohlcv",
            quality.KIND_OPTIONAL_UNAVAILABLE,
            f"{symbol}: {type(exc).__name__}: {exc}",
        )
        return (
            f"DATA_UNAVAILABLE: OHLCV for {symbol!r} as of {curr_date} could "
            f"not be loaded ({type(exc).__name__}: {exc}). Report this "
            f"analysis family as unavailable; do not estimate it."
        )
    if data is None or len(data) < 2:
        quality.record_sentinel(
            "price_structure_ohlcv",
            quality.KIND_OPTIONAL_UNAVAILABLE,
            f"{symbol}: insufficient OHLCV history as of {curr_date}",
        )
        return (
            f"DATA_UNAVAILABLE: insufficient OHLCV history for {symbol!r} "
            f"as of {curr_date}."
        )
    return data


@tool
def get_support_resistance(
    symbol: Annotated[str, "ticker symbol of the company"],
    curr_date: Annotated[str, "The current trading date you are trading on, YYYY-mm-dd"],
) -> str:
    """
    Deterministic support/resistance levels: classic pivot points from the
    previous session AND previous completed week, rolling 20/55/252-day
    highs/lows with the latest close's distance to each, and a 120-day volume
    profile (point of control + 70% value area). Cite THIS tool for any
    support/resistance or breakout-level claim.
    """
    loaded = _load(symbol, curr_date)
    if isinstance(loaded, str):
        return loaded
    levels = build_support_resistance(loaded, curr_date)

    lines = [
        f"## Support/Resistance levels for {symbol.upper()} (as of {curr_date})",
        f"Latest close: {_fmt_price(levels['latest_close'])} on {levels['latest_date']}",
        "",
    ]

    for label, pivots in (
        ("Previous session pivots (classic)", levels["daily_pivots"]),
        ("Previous completed week pivots (classic)", levels["weekly_pivots"]),
    ):
        lines.append(f"### {label}")
        if pivots is None:
            lines.append("_Not enough history._")
        else:
            lines.append(
                f"R2 {_fmt_price(pivots['R2'])} | R1 {_fmt_price(pivots['R1'])} | "
                f"P {_fmt_price(pivots['P'])} | S1 {_fmt_price(pivots['S1'])} | "
                f"S2 {_fmt_price(pivots['S2'])}"
            )
        lines.append("")

    rolling = levels["rolling_levels"]
    lines.append("### Rolling lookback extremes")
    if rolling:
        lines.append("| Window | High | Dist from close | Low | Dist from close |")
        lines.append("|---|---:|---:|---:|---:|")
        for window, lv in sorted(rolling.items()):
            lines.append(
                f"| {window}d | {_fmt_price(lv['high'])} | {_fmt(lv['high_dist_pct'])}% | "
                f"{_fmt_price(lv['low'])} | {_fmt(lv['low_dist_pct'])}% |"
            )
    else:
        lines.append("_Not enough history._")
    lines.append("")

    vp = levels["volume_profile"]
    lines.append("### Volume profile (last 120 sessions, 50 bins)")
    if vp is None:
        lines.append("_Not enough history._")
    else:
        lines.append(
            f"Point of control (highest-volume price): {_fmt_price(vp['poc'])} | "
            f"70% value area: {_fmt_price(vp['value_area_low'])} – "
            f"{_fmt_price(vp['value_area_high'])}"
        )
    lines.append("")
    lines.append(
        "Cite these computed levels (with their derivation window) for any "
        "support/resistance, pivot, or breakout claim. A level 'holding' or "
        "'breaking' must be shown by price action in the OHLCV data, not "
        "asserted."
    )
    return "\n".join(lines)


@tool
def get_volume_features(
    symbol: Annotated[str, "ticker symbol of the company"],
    curr_date: Annotated[str, "The current trading date you are trading on, YYYY-mm-dd"],
) -> str:
    """
    Volume-price evidence: OBV trend vs price, relative volume (today vs the
    prior 20 sessions), stockstats vr, and a deterministic 20-session
    price-vs-OBV divergence check (bearish = new price high without OBV
    confirmation; bullish = new low with OBV holding). Cite THIS tool for
    volume-confirmation or divergence claims.
    """
    loaded = _load(symbol, curr_date)
    if isinstance(loaded, str):
        return loaded

    obv_series = obv(loaded)
    rel = relative_volume(loaded, window=20)
    try:
        sdf = wrap(loaded.copy())
        vr_value = compute_indicator(sdf, "vr").iloc[-1]
    except Exception:  # noqa: BLE001 — one reading missing must not sink the rest
        vr_value = None
    divergence = volume_divergence(loaded, window=20)

    obv_tail = obv_series.tail(10)
    obv_slope = float(obv_tail.iloc[-1] - obv_tail.iloc[0]) if len(obv_tail) == 10 else None
    price_series = pd.to_numeric(loaded["Close"], errors="coerce")
    price_slope = (
        float(price_series.iloc[-1] - price_series.iloc[-10])
        if len(price_series) >= 10 else None
    )

    lines = [
        f"## Volume-price features for {symbol.upper()} (as of {curr_date})",
        "",
        f"- OBV (latest): {_fmt(obv_series.iloc[-1] if len(obv_series) else None, 0)}",
        f"- OBV 10-session change: {_fmt(obv_slope, 0)} "
        f"({'rising' if obv_slope and obv_slope > 0 else 'falling' if obv_slope and obv_slope < 0 else 'flat' if obv_slope == 0 else 'n/a'})",
        f"- Price 10-session change: {_fmt_price(price_slope)}",
        f"- Relative volume (today vs prior 20 sessions): {_fmt(rel.iloc[-1] if len(rel) else None)}x",
        f"- Volume Ratio (vr, 26-session): {_fmt(vr_value)}",
        "",
    ]

    if divergence is None:
        lines.append("Divergence check: _not enough history (needs 20 sessions)._")
    else:
        bear = divergence["bearish_divergence"]
        bull = divergence["bullish_divergence"]
        verdict = (
            "BEARISH divergence — price printed a 20-session high WITHOUT OBV confirmation"
            if bear else
            "BULLISH divergence — price printed a 20-session low while OBV held above its low"
            if bull else
            "No divergence — volume confirms the latest price extreme (or price is inside its range)"
        )
        lines.append(f"Divergence check (20 sessions): **{verdict}**.")
        lines.append(
            f"Price window high/low: {_fmt_price(divergence['price_window_high'])} / "
            f"{_fmt_price(divergence['price_window_low'])}; OBV window high/low: "
            f"{_fmt(divergence['obv_window_high'], 0)} / {_fmt(divergence['obv_window_low'], 0)}."
        )
    lines.append("")
    lines.append(
        "Cite this tool for volume-confirmation or divergence claims; do not "
        "assert divergences it did not report."
    )
    return "\n".join(lines)


@tool
def get_candlestick_patterns(
    symbol: Annotated[str, "ticker symbol of the company"],
    curr_date: Annotated[str, "The current trading date you are trading on, YYYY-mm-dd"],
    look_back_days: Annotated[int, "how many daily bars to scan for candlestick patterns"] = 30,
) -> str:
    """
    Deterministic candlestick-pattern scan (doji, hammer, engulfing, morning/
    evening star, three soldiers/crows, piercing, dark cloud, harami, ...)
    over the last N daily bars, plus a pivot-based double-top/double-bottom
    check over ~120 bars (with neckline and confirmation status). Cite THIS
    tool for any candlestick or chart-pattern claim, with the pattern's date.
    """
    loaded = _load(symbol, curr_date)
    if isinstance(loaded, str):
        return loaded

    lookback = max(5, min(int(look_back_days), 120))
    hits = scan_candlestick_patterns(loaded, lookback=lookback)
    shapes = detect_double_top_bottom(loaded, lookback=120)

    lines = [
        f"## Candlestick & chart patterns for {symbol.upper()} "
        f"(last {lookback} bars, as of {curr_date})",
        "",
    ]
    if hits:
        lines.append("| Pattern | Date | Direction |")
        lines.append("|---|---|---|")
        for hit in hits[-15:]:  # newest 15 hits keep the table readable
            lines.append(f"| {hit['pattern']} | {hit['date']} | {hit['direction']} |")
    else:
        lines.append(f"No classic candlestick patterns in the last {lookback} bars.")
    lines.append("")

    if shapes:
        for shape in shapes:
            confirm = "CONFIRMED (neckline broken on a close)" if shape["confirmed"] \
                else "UNCONFIRMED (neckline not yet broken — a setup, not a signal)"
            lines.append(
                f"- **{shape['pattern']}** ({shape['direction']}): pivots "
                f"{shape['first_pivot_date']} & {shape['second_pivot_date']} near "
                f"{_fmt_price(shape['level'])}; neckline {_fmt_price(shape['neckline'])} — {confirm}."
            )
    else:
        lines.append("No double-top/bottom shape among recent swing pivots (120 bars).")
    lines.append("")
    lines.append(
        "Cite this tool (pattern name + date) for every candlestick or chart-"
        "pattern claim; do not describe patterns it did not report. Patterns "
        "are descriptive evidence, not standalone signals — weigh them with "
        "trend, momentum, and volume context."
    )
    return "\n".join(lines)


#: Lookback windows (trading rows) for the relative-strength comparison.
_RS_WINDOWS: tuple[int, ...] = (21, 63, 126, 252)  # ~1m / 3m / 6m / 12m


def _dated_close(df: pd.DataFrame | None) -> pd.Series:
    """Date-indexed numeric Close series (sorted, NaN-dropped)."""
    if df is None or df.empty:
        return pd.Series(dtype=float)
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(df["Date"], errors="coerce"),
            "close": pd.to_numeric(df["Close"], errors="coerce"),
        }
    ).dropna()
    return frame.sort_values("date").set_index("date")["close"]


def _window_return(close: pd.Series, rows: int) -> float | None:
    if len(close) <= rows:
        return None
    then = float(close.iloc[-1 - rows])
    if then <= 0:
        return None
    return float(close.iloc[-1]) / then - 1.0


def _bench_return_over(
    bench: pd.Series, start_date: pd.Timestamp, end_value: float | None
) -> float | None:
    """Benchmark return over the SAME calendar span as the ticker window.

    The ticker leg counts its own trading rows (its calendar defines the
    window); the benchmark leg anchors on dates — ``asof`` takes the last
    benchmark close at-or-before the window's start date. Row-count alignment
    would compare 21 crypto days against 21 SPY sessions ≈ 29 calendar days,
    stretching the benchmark's window and skewing the RS ratio.
    """
    if bench.empty or end_value is None or end_value <= 0:
        return None
    try:
        start = float(bench.asof(start_date))
    except (TypeError, ValueError):
        return None
    if not pd.notna(start) or start <= 0:
        return None
    return end_value / start - 1.0


@tool
def get_relative_strength(
    symbol: Annotated[str, "ticker symbol of the company"],
    curr_date: Annotated[str, "The current trading date you are trading on, YYYY-mm-dd"],
) -> str:
    """
    Relative strength vs the regional benchmark index: 1/3/6/12-month returns
    of the ticker and of its benchmark (SPY for US, CSI 300 for A-shares, and
    the regional index map elsewhere), the ticker/benchmark return ratio per
    window, and whether the 3-month RS trend is improving or fading. Cite
    THIS tool for any outperformance / underperformance claim.
    """
    # PIT clamp: curr_date comes from the LLM; the pinned analysis date
    # (set_analysis_date ContextVar) is the authority a historical replay
    # must not see past — the same guard the news vendors carry
    # (yfinance_news / alpha_vantage_news). get_stock_data clamps inside
    # y_finance; these TA tools feed load_ohlcv directly, so clamp here.
    from yialpha.dataflows.utils import current_pit_end

    curr_date = str(current_pit_end(curr_date) or curr_date)
    try:
        data = load_ohlcv(symbol, curr_date)
    except Exception as exc:  # noqa: BLE001
        quality.record_sentinel(
            "get_relative_strength",
            quality.KIND_OPTIONAL_UNAVAILABLE,
            f"{symbol}: {type(exc).__name__}: {exc}",
        )
        return (
            f"DATA_UNAVAILABLE: OHLCV for {symbol!r} as of {curr_date} could "
            f"not be loaded ({type(exc).__name__}: {exc})."
        )
    benchmark = resolve_market_benchmark(symbol)
    try:
        bench_data = load_ohlcv(benchmark, curr_date)
    except Exception:  # noqa: BLE001 — benchmark unavailable: RS not computable
        bench_data = None

    close = _dated_close(data)
    if close.empty:
        quality.record_sentinel(
            "get_relative_strength",
            quality.KIND_OPTIONAL_UNAVAILABLE,
            f"{symbol}: no OHLCV history as of {curr_date}",
        )
        return f"DATA_UNAVAILABLE: no OHLCV history for {symbol!r} as of {curr_date}."
    bench_close = _dated_close(bench_data)

    lines = [
        f"## Relative strength: {symbol.upper()} vs {benchmark} (as of {curr_date})",
        "",
        "| Window | Ticker | Benchmark | RS ratio |",
        "|---|---:|---:|---:|",
    ]
    ratios: list[float] = []
    bench_end = float(bench_close.iloc[-1]) if not bench_close.empty else None
    for rows, label in zip(_RS_WINDOWS, ("1m", "3m", "6m", "12m"), strict=False):
        r = _window_return(close, rows)
        then_date = close.index[-1 - rows] if len(close) > rows else None
        b = (
            _bench_return_over(bench_close, then_date, bench_end)
            if then_date is not None else None
        )
        # Wealth-ratio RS: (1+r)/(1+b). The plain r/b ratio flips sign and
        # magnitude whenever the benchmark return is negative — ticker +2% vs
        # benchmark -1% scored -2.0 ("underperforming") and ticker -1% vs
        # benchmark -4% scored 0.25 — while a wealth ratio stays >1 exactly
        # when the ticker outperformed over the window, in up and down
        # markets alike (round-5 audit, 2026-08-16).
        ratio = (
            ((1.0 + r) / (1.0 + b))
            if (r is not None and b is not None and (1.0 + b) != 0.0)
            else None
        )
        if ratio is not None:
            ratios.append(ratio)
        lines.append(
            f"| {label} | {_fmt_pct(r)} | {_fmt_pct(b)} | "
            f"{'n/a' if ratio is None else f'{ratio:.2f}'} |"
        )
    lines.append("")
    if len(ratios) >= 2:
        trend = (
            "improving (recent windows stronger)"
            if ratios[-1] > ratios[0]
            else "fading (recent windows weaker)"
        )
        lines.append(
            f"RS trend across windows: **{trend}** (ratio {ratios[0]:.2f} → "
            f"{ratios[-1]:.2f}). RS ratio > 1 = outperforming the benchmark "
            "over that window."
        )
    elif not bench_close.empty:
        lines.append("RS trend: not enough overlapping windows to judge the trend.")
    else:
        lines.append(
            f"Benchmark {benchmark} unavailable as of {curr_date} — absolute "
            "returns only; report relative strength as unavailable."
        )
    lines.append("")
    lines.append(
        "Cite this tool for outperformance/underperformance claims; do not "
        "assert them from memory."
    )
    return "\n".join(lines)


def _fmt_pct(x: float | None) -> str:
    try:
        if x is None or pd.isna(x):
            return "n/a"
        return f"{float(x) * 100.0:+.2f}%"
    except (TypeError, ValueError):
        return "n/a"
