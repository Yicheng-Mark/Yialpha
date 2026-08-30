"""Deterministic support/resistance levels from OHLCV.

Before this module, "support/resistance" existed only in prompt prose — the
analyst was told NOT to claim S/R bounces without dated tool evidence, which
meant the capability was banned rather than supported. These functions turn
the standard level families into computed evidence: classic pivot points
(daily and weekly), rolling lookback highs/lows, and a volume profile with
point-of-control and value area. All pure math on the PIT-filtered frame —
the same anti-hallucination contract as the verified snapshot, now
supplying the levels instead of forbidding the claim.
"""

from __future__ import annotations

from typing import TypedDict

import numpy as np
import pandas as pd

from .ohlcv_resample import resample_weekly


class SupportResistanceReport(TypedDict):
    """All level families for one symbol/date (see build_support_resistance)."""

    daily_pivots: dict[str, float] | None
    weekly_pivots: dict[str, float] | None
    rolling_levels: dict[int, dict[str, float]]
    volume_profile: dict[str, float | int] | None
    latest_close: float | None
    latest_date: str | None


def classic_pivots(high: float, low: float, close: float) -> dict[str, float]:
    """Classic floor-trader pivots from one completed session's H/L/C."""
    p = (high + low + close) / 3.0
    return {
        "P": p,
        "R1": 2.0 * p - low,
        "S1": 2.0 * p - high,
        "R2": p + (high - low),
        "S2": p - (high - low),
    }


def daily_pivots(df: pd.DataFrame) -> dict[str, float] | None:
    """Pivot levels derived from the PREVIOUS completed session.

    ``df`` is PIT-filtered through the analysis date, so its last row IS the
    current session; the levels a trader would reference today come from the
    session before it. ``None`` when fewer than 2 rows exist.
    """
    if df is None or len(df) < 2:
        return None
    prev = df.iloc[-2]
    try:
        return classic_pivots(
            float(prev["High"]), float(prev["Low"]), float(prev["Close"])
        )
    except (TypeError, ValueError):
        return None


def weekly_pivots(df: pd.DataFrame, curr_date: str) -> dict[str, float] | None:
    """Pivot levels derived from the previous COMPLETED weekly bar.

    ``resample_weekly`` drops the still-open week (its Friday label lies
    after ``curr_date``), so the LAST weekly row already IS the most recent
    completed week — read it directly. (The previous ``iloc[-2]`` reached one
    week further back than the label promised: a Wednesday analysis returned
    the week-before-last's levels. On a Friday ``curr_date`` the W-FRI label
    keeps the current week, which completes that day.) ``None`` when no
    complete week exists.
    """
    weekly = resample_weekly(df, curr_date)
    if weekly is None or weekly.empty:
        return None
    prev = weekly.iloc[-1]
    try:
        return classic_pivots(
            float(prev["High"]), float(prev["Low"]), float(prev["Close"])
        )
    except (TypeError, ValueError):
        return None


def rolling_levels(
    df: pd.DataFrame, windows: tuple[int, ...] = (20, 55, 252)
) -> dict[int, dict[str, float]]:
    """Lookback-window highs/lows plus the latest close's distance to them.

    The 20/55/252 trio maps to the ~1month/~1quarter/~1year horizons traders
    actually reference. Distance is percent from the latest close (positive =
    level above price, i.e. resistance room; negative = below, support room).
    """
    if df is None or df.empty:
        return {}
    close = float(df.iloc[-1]["Close"])
    out: dict[int, dict[str, float]] = {}
    for window in windows:
        if len(df) < window:
            continue
        tail = df.tail(window)
        hi = float(tail["High"].max())
        lo = float(tail["Low"].min())
        out[window] = {
            "high": hi,
            "high_dist_pct": (hi - close) / close * 100.0,
            "low": lo,
            "low_dist_pct": (lo - close) / close * 100.0,
        }
    return out


def volume_profile(
    df: pd.DataFrame, lookback: int = 120, bins: int = 50
) -> dict[str, float | int] | None:
    """Price-binned volume profile: POC and the 70% value area.

    Each bar's volume is credited to the bin of its typical price
    ``(H+L+C)/3`` (the standard approximation when only OHLCV is available).
    POC = the highest-volume bin's price. The value area expands outward
    from the POC (higher-volume neighbour first) until ≥70% of the lookback
    volume is covered. ``None`` when there is not enough data.
    """
    if df is None or len(df) < lookback // 2 or bins < 2:
        return None
    tail = df.tail(lookback)
    high = pd.to_numeric(tail["High"], errors="coerce")
    low = pd.to_numeric(tail["Low"], errors="coerce")
    close = pd.to_numeric(tail["Close"], errors="coerce")
    volume = pd.to_numeric(tail["Volume"], errors="coerce")
    usable = (
        high.notna() & low.notna() & close.notna() & volume.notna() & (volume > 0)
    )
    if int(usable.sum()) < bins // 2:
        return None

    typical = (high[usable] + low[usable] + close[usable]) / 3.0
    vol = volume[usable]
    lo, hi = float(typical.min()), float(typical.max())
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return None

    edges = np.linspace(lo, hi, bins + 1)
    idx = np.clip(np.digitize(typical.to_numpy(), edges[1:-1], right=True), 0, bins - 1)
    bucket_vol = np.zeros(bins)
    np.add.at(bucket_vol, idx, vol.to_numpy())

    total = bucket_vol.sum()
    if total <= 0:
        return None
    poc_bin = int(np.argmax(bucket_vol))
    bin_center = lambda i: float((edges[i] + edges[i + 1]) / 2.0)  # noqa: E731

    # Value area: expand from POC, always taking the heavier neighbour.
    covered = bucket_vol[poc_bin]
    lo_bin = hi_bin = poc_bin
    while covered < 0.70 * total and (lo_bin > 0 or hi_bin < bins - 1):
        down = bucket_vol[lo_bin - 1] if lo_bin > 0 else -1.0
        up = bucket_vol[hi_bin + 1] if hi_bin < bins - 1 else -1.0
        if up >= down:
            hi_bin += 1
            covered += bucket_vol[hi_bin]
        else:
            lo_bin -= 1
            covered += bucket_vol[lo_bin]

    return {
        "poc": bin_center(poc_bin),
        "value_area_low": float(edges[lo_bin]),
        "value_area_high": float(edges[hi_bin + 1]),
        "lookback_rows": int(usable.sum()),
        "bins": bins,
    }


def build_support_resistance(
    df: pd.DataFrame, curr_date: str
) -> SupportResistanceReport:
    """All level families for one symbol/date, as a structured dict.

    Consumers (the LLM tool) render this; every level carries its derivation
    window so a claim can cite it. Missing families are ``None`` — honest
    absence, never a fabricated level.
    """
    return SupportResistanceReport(
        daily_pivots=daily_pivots(df),
        weekly_pivots=weekly_pivots(df, curr_date),
        rolling_levels=rolling_levels(df),
        volume_profile=volume_profile(df),
        latest_close=float(df.iloc[-1]["Close"]) if df is not None and len(df) else None,
        latest_date=str(df.iloc[-1]["Date"]) if df is not None and len(df) else None,
    )
