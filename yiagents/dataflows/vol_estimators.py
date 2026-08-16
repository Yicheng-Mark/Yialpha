"""Range- and return-based volatility estimators on daily OHLCV.

Close-to-close realized vol only uses one number per day; the high/low range
carries far more information about intraday dispersion. These estimators are
the standard ladder (Parkinson 1980, Garman-Klass 1980, Yang-Zhang 2000,
plus RiskMetrics EWMA) computed as rolling series aligned to the input frame.
All outputs are ANNUALIZED volatilities — comparable across windows and
directly usable for vol-targeted sizing. The annualization factor defaults
to the equity convention (× sqrt(252)); crypto (spot and perp) trades 24/7
and MUST pass ``periods_per_year=CRYPTO_TRADING_DAYS_PER_YEAR`` (365), or
every reading is understated by sqrt(252/365) ≈ 0.83.

Pure pandas on the capitalized OHLCV columns ``load_ohlcv`` produces; no
network, no LLM. Rows with a non-finite Close are dropped first; estimators
that need the previous close (Yang-Zhang overnight term, return series)
leave the warm-up rows NaN rather than borrowing neighbours.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

#: Annualization factor for daily-frequency vol estimates (equity convention).
TRADING_DAYS_PER_YEAR = 252.0

#: Crypto trades every calendar day — a daily crypto series annualizes with
#: 365, not 252. Using 252 on Binance candles understates annualized vol by
#: sqrt(252/365) ≈ 0.83 (a ~17% error on every vol-based reading).
CRYPTO_TRADING_DAYS_PER_YEAR = 365.0


def periods_per_year_for(asset_type: str | None) -> float:
    """Annualization factor for a daily series of ``asset_type``.

    Crypto (spot and perp) trades every calendar day; equities/A-shares keep
    the 252-session convention. Unknown/None defaults to 252 (the historical
    behaviour), matching the backtest engine's ``periods_per_year`` rule.
    """
    if (asset_type or "").startswith("crypto"):
        return CRYPTO_TRADING_DAYS_PER_YEAR
    return TRADING_DAYS_PER_YEAR

#: RiskMetrics decay for the EWMA variance recursion (λ = 0.94, the standard
#: daily-frequency value from the 1996 RiskMetrics technical document).
EWMA_LAMBDA = 0.94


def _clean_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    for col in ("Open", "High", "Low", "Close"):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=["Open", "High", "Low", "Close"])
    return out if not out.empty else pd.DataFrame()


def close_to_close_vol(
    df: pd.DataFrame, window: int = 20,
    periods_per_year: float = TRADING_DAYS_PER_YEAR,
) -> pd.Series:
    """Annualized rolling std (ddof=1) of daily log returns.

    The textbook realized-vol baseline every other estimator here is judged
    against: one close per day, so it wastes the intraday range but is
    robust to data glitches in High/Low.
    """
    data = _clean_ohlcv(df)
    if data.empty or window <= 1:
        return pd.Series(dtype=float)
    log_ret = np.log(data["Close"] / data["Close"].shift(1))
    var = log_ret.rolling(window, min_periods=window).var(ddof=1)
    return np.sqrt(var * periods_per_year)


def parkinson_vol(
    df: pd.DataFrame, window: int = 20,
    periods_per_year: float = TRADING_DAYS_PER_YEAR,
) -> pd.Series:
    """Annualized Parkinson (1980) high-low-range volatility.

    σ² = mean(ln(H/L)²) / (4 ln 2). ~5× more efficient than close-to-close
    under a GBM assumption, but ignores overnight gaps and underestimates in
    trending markets.
    """
    data = _clean_ohlcv(df)
    if data.empty or window <= 1:
        return pd.Series(dtype=float)
    hl = np.log(data["High"] / data["Low"])
    var = hl.pow(2).rolling(window, min_periods=window).mean() / (4.0 * math.log(2.0))
    return np.sqrt(var * periods_per_year)


def garman_klass_vol(
    df: pd.DataFrame, window: int = 20,
    periods_per_year: float = TRADING_DAYS_PER_YEAR,
) -> pd.Series:
    """Annualized Garman-Klass (1980) OHLC volatility.

    σ² = mean( 0.5·ln(H/L)² − (2·ln2 − 1)·ln(C/O)² ). Adds the open-close
    body to Parkinson's range. The per-row term can be negative on odd bars;
    the rolling variance is clipped at zero before the square root so a
    single weird bar yields 0 rather than NaN.
    """
    data = _clean_ohlcv(df)
    if data.empty or window <= 1:
        return pd.Series(dtype=float)
    hl = np.log(data["High"] / data["Low"])
    oc = np.log(data["Close"] / data["Open"])
    term = 0.5 * hl.pow(2) - (2.0 * math.log(2.0) - 1.0) * oc.pow(2)
    var = term.rolling(window, min_periods=window).mean().clip(lower=0.0)
    return np.sqrt(var * periods_per_year)


def yang_zhang_vol(
    df: pd.DataFrame, window: int = 20,
    periods_per_year: float = TRADING_DAYS_PER_YEAR,
) -> pd.Series:
    """Annualized Yang-Zhang (2000) volatility — min-variance unbiased under
    both overnight gaps and drift.

    σ² = σ²_overnight + k·σ²_open_to_close + (1−k)·mean(RS), where the
    overnight term uses ln(O_t / C_{t−1}), the open-to-close term ln(C_t/O_t),
    RS is the Rogers-Satchell per-bar term, and k = 0.34/(1.34 + (n+1)/(n−1))
    for window n. The variance is clipped at zero (numerical guard) before
    the square root.
    """
    data = _clean_ohlcv(df)
    if data.empty or window <= 2:
        return pd.Series(dtype=float)
    open_, high, low, close = data["Open"], data["High"], data["Low"], data["Close"]
    overnight = np.log(open_ / close.shift(1))
    open_to_close = np.log(close / open_)
    rs = (
        np.log(high / open_) * np.log(high / close)
        + np.log(low / open_) * np.log(low / close)
    )
    k = 0.34 / (1.34 + (window + 1) / (window - 1))
    var_o = overnight.rolling(window, min_periods=window).var(ddof=1)
    var_c = open_to_close.rolling(window, min_periods=window).var(ddof=1)
    var_rs = rs.rolling(window, min_periods=window).mean()
    var = var_o + k * var_c + (1.0 - k) * var_rs
    return np.sqrt(var.clip(lower=0.0) * periods_per_year)


def ewma_vol(
    df: pd.DataFrame, lam: float = EWMA_LAMBDA,
    periods_per_year: float = TRADING_DAYS_PER_YEAR,
) -> pd.Series:
    """Annualized RiskMetrics EWMA volatility (λ = 0.94 by default).

    σ²_t = λ·σ²_{t−1} + (1−λ)·r²_t with the recursion seeded at the first
    squared return (``ewm(adjust=False)``); the estimate INCLUDES the latest
    close's return, so as a feature it is known at that close. Reacts far
    faster than a flat 20d window after a shock — the ewma_vol vs rvol_20
    gap is a cheap vol-regime transition marker.
    """
    data = _clean_ohlcv(df)
    if data.empty or not (0.0 < lam < 1.0):
        return pd.Series(dtype=float)
    log_ret = np.log(data["Close"] / data["Close"].shift(1))
    var = log_ret.pow(2).ewm(alpha=1.0 - lam, adjust=False, min_periods=1).mean()
    return np.sqrt(var * periods_per_year)
