"""Hand-rolled candlestick + pivot-shape pattern detection.

No TA-Lib (the project keeps a zero C-extension dependency footprint), so the
classic single-bar / two-bar / three-bar patterns and the pivot-based double
top/bottom shapes are implemented directly with the standard body/wick ratio
definitions. Every detector is a pure function of the OHLCV tail; the tool
layer reports only what these find, with dates — pattern claims stop being
LLM impressions and become citable tool output, the same contract as the
verified snapshot.
"""

from __future__ import annotations

from typing import TypedDict

import numpy as np
import pandas as pd

#: Body >= this fraction of the bar's total range counts as a "long" body;
#: below the doji threshold counts as (near-)bodied-out.
_LONG_BODY_RATIO = 0.6
_DOJI_BODY_RATIO = 0.1
#: Hammer family: the shadow on the "signal" side must be >= 2x the body and
#: the opposite shadow <= ~0.3 of the range.
_SHADOW_BODY_MULT = 2.0
_SMALL_SHADOW_RATIO = 0.3
#: Engulfing: the second body must strictly contain the first, both real
#: bodies (not just wicks), first body meaningfully sized.
_ENGULF_MIN_BODY_RATIO = 0.25
#: Star family: middle bar's body is small vs both neighbours' bodies.
_STAR_SMALL_RATIO = 0.35
#: Piercing / dark-cloud: close into the upper/lower half of the first body.
_PIERCE_MIN = 0.5
#: Two highs/lows within this fraction of their mean count as "equal".
_DOUBLE_EXTREME_TOL = 0.02
#: Pivot strength: bars required on each side of a swing point.
_PIVOT_WING = 3


class DoubleExtremeShape(TypedDict):
    """One detected double-top/bottom shape (see detect_double_top_bottom)."""

    pattern: str
    first_pivot_date: str
    second_pivot_date: str
    level: float
    neckline: float
    confirmed: bool
    direction: str


def _body(o: float, h: float, lo: float, c: float) -> tuple[float, float, float]:
    """(body size, range, body/range) with a floor to avoid /0 on flat bars."""
    body = abs(c - o)
    rng = max(h - lo, 1e-12)
    return body, rng, body / rng


def _detect_three_white_soldiers(rows: pd.DataFrame) -> bool:
    o, h, c = (rows[k].to_numpy(dtype=float) for k in ("Open", "High", "Close"))
    if len(o) != 3:
        return False
    bodies_ok = all(c[i] > o[i] for i in range(3))
    if not bodies_ok:
        return False
    seq_ok = c[0] < c[1] < c[2]
    within = o[1] >= o[0] and o[2] >= o[1]
    small_upper = all(
        (h[i] - max(o[i], c[i])) <= 0.3 * abs(c[i] - o[i]) + 1e-12 for i in range(3)
    )
    return bool(seq_ok and within and small_upper)


def _detect_three_black_crows(rows: pd.DataFrame) -> bool:
    """Textbook three black crows: the price-mirror of three white soldiers.

    Three red bars with strictly falling closes, each opening inside the prior
    bar's real body, and small lower shadows (selling that closes near the
    low). NOTE: the previous column-swap implementation was NOT a mirror —
    swapping Open/Close without negating prices required RISING opens and
    closes, so textbook crows never matched and ascending red bars were
    flagged as crows (P0, 2026-08-16).
    """
    o = rows["Open"].to_numpy(dtype=float)
    lo = rows["Low"].to_numpy(dtype=float)
    c = rows["Close"].to_numpy(dtype=float)
    if len(o) != 3:
        return False
    if not all(c[i] < o[i] for i in range(3)):
        return False
    if not (c[0] > c[1] > c[2]):
        return False
    # Each open within (or at the edge of) the prior bar's real body.
    within = (c[0] - 1e-12 <= o[1] <= o[0]) and (c[1] - 1e-12 <= o[2] <= o[1])
    small_lower = all(
        (min(o[i], c[i]) - lo[i]) <= 0.3 * abs(c[i] - o[i]) + 1e-12
        for i in range(3)
    )
    return bool(within and small_lower)


def _prior_trend(c, i: int, lookback: int = 3) -> str | None:
    """Classify the close run just before bar ``i``: "down", "up", or None.

    Trend-context patterns (hammer vs hanging man, shooting star vs inverted
    hammer) are named by the trend they reverse, so the scanner needs to know
    whether the bars leading into the shape were declining or advancing.
    Flat / too-short lead-ins return None and the caller keeps its fallback.
    """
    start = max(0, i - lookback)
    window = c[start:i]
    if len(window) < 2:
        return None
    if window[-1] < window[0]:
        return "down"
    if window[-1] > window[0]:
        return "up"
    return None


def scan_candlestick_patterns(
    df: pd.DataFrame, lookback: int = 30
) -> list[dict[str, object]]:
    """Scan the last ``lookback`` bars for classic patterns.

    Returns a list of ``{"pattern": str, "date": str, "direction": "bullish"
    | "bearish" | "neutral"}`` hits, newest-last. Single-bar patterns attach
    to their bar; multi-bar patterns attach to the LAST bar of the shape.
    """
    if df is None or len(df) < 3:
        return []
    tail = df.tail(lookback).reset_index(drop=True)
    o = tail["Open"].astype(float).to_numpy()
    h = tail["High"].astype(float).to_numpy()
    lo = tail["Low"].astype(float).to_numpy()
    c = tail["Close"].astype(float).to_numpy()
    dates = pd.to_datetime(tail["Date"]).dt.strftime("%Y-%m-%d").to_numpy()
    n = len(tail)
    hits: list[dict[str, object]] = []

    def add(pattern: str, i: int, direction: str) -> None:
        hits.append({"pattern": pattern, "date": str(dates[i]), "direction": direction})

    for i in range(n):
        body, rng, body_ratio = _body(o[i], h[i], lo[i], c[i])
        if rng <= 1e-9:
            continue
        upper = h[i] - max(o[i], c[i])
        lower = min(o[i], c[i]) - lo[i]

        if body_ratio <= _DOJI_BODY_RATIO:
            add("doji", i, "neutral")

        # Hammer-family / star-family shapes (only meaningful bodies —
        # near-dojis are already captured by the doji branch). Traditional
        # definitions split the PAIR by prior trend, not candle color: the
        # same long-lower-shadow body is a bullish hammer after a decline
        # and a bearish hanging man after an advance (mirror for shooting
        # star / inverted hammer). Color-based naming mislabeled the red
        # hammer in a downtrend as bearish (round-5 audit, 2026-08-16);
        # when the prior trend is flat the color heuristic is kept.
        if body_ratio >= 0.05:
            trend = _prior_trend(c, i)
            if lower >= _SHADOW_BODY_MULT * body and upper <= _SMALL_SHADOW_RATIO * rng:
                if trend == "up":
                    add("hanging_man", i, "bearish")
                elif trend == "down":
                    add("hammer", i, "bullish")
                else:
                    add("hammer" if c[i] >= o[i] else "hanging_man", i,
                        "bullish" if c[i] >= o[i] else "bearish")
            if upper >= _SHADOW_BODY_MULT * body and lower <= _SMALL_SHADOW_RATIO * rng:
                if trend == "up":
                    add("shooting_star", i, "bearish")
                elif trend == "down":
                    add("inverted_hammer", i, "bullish")
                else:
                    add("shooting_star" if c[i] < o[i] else "inverted_hammer", i,
                        "bearish" if c[i] < o[i] else "bullish")

        if i >= 1:
            pbody, prng, pbody_ratio = _body(o[i-1], h[i-1], lo[i-1], c[i-1])
            bull, bear = c[i] >= o[i], c[i] < o[i]
            pbull, pbear = c[i-1] >= o[i-1], c[i-1] < o[i-1]
            # Engulfing: today's body strictly contains yesterday's opposite body.
            # Bullish: open at/below the prior close, close at/above the prior
            # open (today's green body swallows yesterday's red body).
            # Bearish is the mirror image.
            if (
                pbody_ratio >= _ENGULF_MIN_BODY_RATIO
                and bear and pbull
                and o[i] >= c[i-1] and c[i] <= o[i-1] and body > pbody
            ):
                add("bearish_engulfing", i, "bearish")
            if (
                pbody_ratio >= _ENGULF_MIN_BODY_RATIO
                and bull and pbear
                and o[i] <= c[i-1] and c[i] >= o[i-1] and body > pbody
            ):
                add("bullish_engulfing", i, "bullish")
            # Harami: today's small body inside yesterday's long opposite body.
            if (
                pbody_ratio >= 0.5 and body_ratio <= 0.4
                and max(o[i], c[i]) <= max(o[i-1], c[i-1])
                and min(o[i], c[i]) >= min(o[i-1], c[i-1])
            ):
                add("harami", i, "neutral")
            # Piercing (bullish) / dark-cloud cover (bearish): gap-open beyond
            # the prior body, close back into its half.
            prior_mid = (o[i-1] + c[i-1]) / 2.0
            if pbear and bull and o[i] <= c[i-1] and c[i] > prior_mid and c[i] < o[i-1]:
                add("piercing", i, "bullish")
            if pbull and bear and o[i] >= c[i-1] and c[i] < prior_mid and c[i] > o[i-1]:
                add("dark_cloud_cover", i, "bearish")

        if i >= 2:
            trio = tail.iloc[i-2 : i+1]
            b0, _, r0 = _body(o[i-2], h[i-2], lo[i-2], c[i-2])
            b1, _, r1 = _body(o[i-1], h[i-1], lo[i-1], c[i-1])
            b2, _, r2 = _body(o[i], h[i], lo[i], c[i])
            down_then_up = c[i-2] < o[i-2] and c[i] > o[i]
            up_then_down = c[i-2] > o[i-2] and c[i] < o[i]
            small_middle = (
                b1 <= _STAR_SMALL_RATIO * max(b0, b2, 1e-12)
            )
            if down_then_up and small_middle and c[i] > (o[i-2] + c[i-2]) / 2.0:
                add("morning_star", i, "bullish")
            if up_then_down and small_middle and c[i] < (o[i-2] + c[i-2]) / 2.0:
                add("evening_star", i, "bearish")
            if _detect_three_white_soldiers(trio):
                add("three_white_soldiers", i, "bullish")
            if _detect_three_black_crows(trio):
                add("three_black_crows", i, "bearish")

    # Deduplicate: same pattern+date only once (e.g. doji on a hammer bar).
    seen: set[tuple[str, str]] = set()
    unique: list[dict[str, object]] = []
    for hit in hits:
        key = (str(hit["pattern"]), str(hit["date"]))
        if key not in seen:
            seen.add(key)
            unique.append(hit)
    return unique


def find_pivots(
    df: pd.DataFrame, wing: int = _PIVOT_WING
) -> tuple[list[int], list[int]]:
    """Swing highs/lows: indices stricter than ``wing`` bars on both sides."""
    if df is None or len(df) < 2 * wing + 1:
        return [], []
    h = df["High"].astype(float).to_numpy()
    lo = df["Low"].astype(float).to_numpy()
    highs: list[int] = []
    lows: list[int] = []
    for i in range(wing, len(df) - wing):
        window_h = h[i - wing : i + wing + 1]
        window_l = lo[i - wing : i + wing + 1]
        if h[i] == window_h.max() and (window_h >= h[i]).sum() == 1:
            highs.append(i)
        if lo[i] == window_l.min() and (window_l <= lo[i]).sum() == 1:
            lows.append(i)
    return highs, lows


def detect_double_top_bottom(
    df: pd.DataFrame, lookback: int = 120
) -> list[DoubleExtremeShape]:
    """Pivot-based double top/bottom over the last ``lookback`` bars.

    Two swing extremes within ``_DOUBLE_EXTREME_TOL`` (2% of their mean)
    with at least ``wing`` bars between them. The neckline is the intervening
    opposite extreme; confirmation requires a later close beyond it (a
    double top is only COMPLETE once price closes below the neckline low).
    Unconfirmed shapes are reported with ``confirmed: False``. When several
    pairs qualify, the MOST RECENT one (latest second pivot) is reported —
    the decision-useful shape is the one nearest the current bar, not the
    oldest coincidence in the window.
    """
    if df is None or len(df) < 3 * _PIVOT_WING + 2:
        return []
    tail = df.tail(lookback).reset_index(drop=True)
    highs, lows = find_pivots(tail)
    dates = pd.to_datetime(tail["Date"]).dt.strftime("%Y-%m-%d").to_numpy()
    out: list[DoubleExtremeShape] = []

    def scan(extremes: list[int], values: np.ndarray, is_top: bool) -> None:
        best: tuple[int, DoubleExtremeShape] | None = None
        for a_pos in range(len(extremes)):
            for b_pos in range(a_pos + 1, len(extremes)):
                i, j = extremes[a_pos], extremes[b_pos]
                v1, v2 = float(values[i]), float(values[j])
                mean_v = (v1 + v2) / 2.0
                if abs(v1 - v2) > _DOUBLE_EXTREME_TOL * mean_v:
                    continue
                # Neckline: the opposite extreme strictly between the two pivots.
                between = tail.iloc[i + 1 : j]
                if between.empty:
                    continue
                if is_top:
                    neck = float(between["Low"].astype(float).min())
                    later = tail.iloc[j + 1 :]
                    confirmed = bool(
                        (later["Close"].astype(float) < neck).any()
                    ) if not later.empty else False
                else:
                    neck = float(between["High"].astype(float).max())
                    later = tail.iloc[j + 1 :]
                    confirmed = bool(
                        (later["Close"].astype(float) > neck).any()
                    ) if not later.empty else False
                shape = DoubleExtremeShape(
                    pattern="double_top" if is_top else "double_bottom",
                    first_pivot_date=str(dates[i]),
                    second_pivot_date=str(dates[j]),
                    level=mean_v,
                    neckline=neck,
                    confirmed=confirmed,
                    direction="bearish" if is_top else "bullish",
                )
                # Keep the most recent qualifying pair (largest second pivot).
                if best is None or j > best[0]:
                    best = (j, shape)
        if best is not None:
            out.append(best[1])  # one shape per side is enough signal

    scan(highs, tail["High"].astype(float).to_numpy(), True)
    scan(lows, tail["Low"].astype(float).to_numpy(), False)
    return out


#: Registry used by the tool layer to report what it can detect.
PATTERN_FAMILIES: tuple[str, ...] = (
    "doji", "hammer", "hanging_man", "inverted_hammer", "shooting_star",
    "bullish_engulfing", "bearish_engulfing", "harami", "piercing",
    "dark_cloud_cover", "morning_star", "evening_star",
    "three_white_soldiers", "three_black_crows",
    "double_top", "double_bottom",
)

__all__ = [
    "PATTERN_FAMILIES",
    "detect_double_top_bottom",
    "find_pivots",
    "scan_candlestick_patterns",
]
