"""Volume-price features: OBV, relative volume, and divergence detection.

stockstats has no OBV column (verified against its source), so it is
hand-rolled here along with the relative-volume gauge and a deterministic
20-session divergence detector. "量在价先" — volume leads price — is the
classic A-share read; the detector turns it into a tool-backed claim the
analyst can cite instead of asserting a divergence from memory.
"""

from __future__ import annotations

from typing import TypedDict

import numpy as np
import pandas as pd


class VolumeDivergenceReading(TypedDict):
    """Latest-row price-vs-OBV divergence evidence (see volume_divergence)."""

    bearish_divergence: bool
    bullish_divergence: bool
    price_window_high: float
    price_window_low: float
    obv_window_high: float
    obv_window_low: float
    window: int
    as_of_row: int


def obv(df: pd.DataFrame) -> pd.Series:
    """On-Balance Volume: cumulative volume signed by daily close direction.

    The first row's direction is undefined → sign 0 (no contribution),
    matching the standard definition. Output is unbounded/cumulative — read
    its SLOPE against price, never its level.
    """
    if df is None or df.empty:
        return pd.Series(dtype=float)
    data = df.copy()
    data["Close"] = pd.to_numeric(data["Close"], errors="coerce")
    data["Volume"] = pd.to_numeric(data.get("Volume"), errors="coerce").fillna(0.0)
    data = data.dropna(subset=["Close"])
    direction = np.sign(data["Close"].diff()).fillna(0.0)
    return (direction * data["Volume"]).cumsum()


def relative_volume(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """Today's volume vs the mean of the PRIOR ``window`` sessions (today
    excluded, the conventional relative-volume read: is *today* unusual
    against the recent norm?). Warm-up rows are NaN. A zero prior mean
    (suspended/new listings with no prints) also yields NaN, not ``inf`` —
    "unmeasurable", not "infinitely unusual"."""
    if df is None or df.empty or window <= 1:
        return pd.Series(dtype=float)
    vol = pd.to_numeric(df["Volume"], errors="coerce")
    base = vol.shift(1).rolling(window, min_periods=window).mean()
    return vol / base.replace(0.0, np.nan)


def volume_divergence(
    df: pd.DataFrame, window: int = 20
) -> VolumeDivergenceReading | None:
    """Deterministic price-vs-OBV divergence check on the LATEST row.

    Bearish (top) divergence: close prints a ``window``-session high while
    OBV is below its own ``window``-session high — the advance lacks volume
    confirmation. Bullish (bottom) divergence: close prints a ``window``
    low while OBV holds above its ``window``-session low — selling pressure
    is exhausting.

    Returns a dict of evidence for the latest row (price/obv window
    extremes, both flags), or ``None`` when there is not enough data
    (fewer than ``window`` rows). Equal-to-extreme counts as AT the extreme
    for OBV (``>=``/``<=``), so only strictly-weaker OBV diverges.
    """
    if df is None or df.empty or window <= 1:
        return None
    data = df.copy()
    data["Close"] = pd.to_numeric(data["Close"], errors="coerce")
    data = data.dropna(subset=["Close"])
    if len(data) < window:
        return None

    obv_series = obv(data)
    close = data["Close"]
    price_hi = close.rolling(window, min_periods=window).max()
    price_lo = close.rolling(window, min_periods=window).min()
    obv_hi = obv_series.rolling(window, min_periods=window).max()
    obv_lo = obv_series.rolling(window, min_periods=window).min()

    price_at_high = bool(close.iloc[-1] >= price_hi.iloc[-1])
    price_at_low = bool(close.iloc[-1] <= price_lo.iloc[-1])
    obv_confirms_high = bool(obv_series.iloc[-1] >= obv_hi.iloc[-1])
    obv_confirms_low = bool(obv_series.iloc[-1] <= obv_lo.iloc[-1])

    return VolumeDivergenceReading(
        bearish_divergence=price_at_high and not obv_confirms_high,
        bullish_divergence=price_at_low and not obv_confirms_low,
        price_window_high=float(price_hi.iloc[-1]),
        price_window_low=float(price_lo.iloc[-1]),
        obv_window_high=float(obv_hi.iloc[-1]),
        obv_window_low=float(obv_lo.iloc[-1]),
        window=window,
        as_of_row=int(len(data) - 1),
    )
