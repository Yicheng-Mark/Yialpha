"""Derived-feature registry — the single source for non-stockstats features.

The indicator catalog mixes two kinds of names: stockstats columns
(computed lazily by ``wrap(df)[name]``) and derived features computed by our
own math on the raw OHLCV frame (vol estimators, OBV, relative volume).
This registry is the one place a derived name maps to its computation, so
the yfinance indicator window, the verified-snapshot validator, the IC
dataset exporter and the signature smoke test all dispatch through the same
code path — no per-caller drift (the failure mode the 46-item audit found
with vendor signatures).
"""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd

from .vol_estimators import TRADING_DAYS_PER_YEAR, close_to_close_vol, ewma_vol
from .volume_features import obv, relative_volume


def _rvol_20(data: pd.DataFrame, periods_per_year: float) -> pd.Series:
    return close_to_close_vol(data, window=20, periods_per_year=periods_per_year)


def _ewma_vol(data: pd.DataFrame, periods_per_year: float) -> pd.Series:
    return ewma_vol(data, periods_per_year=periods_per_year)


def _rel_vol_20(data: pd.DataFrame) -> pd.Series:
    return relative_volume(data, window=20)


#: Derived feature name -> computation on the raw capitalized OHLCV frame.
#: Keys are catalog entries (indicator_catalog.py) and IC-exportable columns.
#: Vol estimators take the annualization factor (252 equity / 365 crypto);
#: volume features are ratios and ignore it.
DERIVED_FEATURES: dict[str, Callable[..., pd.Series]] = {
    "rvol_20": _rvol_20,
    "ewma_vol": _ewma_vol,
    "obv": obv,
    "rel_vol_20": _rel_vol_20,
}


def compute_derived(
    data: pd.DataFrame, name: str,
    periods_per_year: float = TRADING_DAYS_PER_YEAR,
) -> pd.Series | None:
    """Compute derived feature ``name`` on a raw OHLCV frame.

    ``periods_per_year`` annualizes the vol estimators — pass 365 for a
    24/7 crypto daily series (:data:`CRYPTO_TRADING_DAYS_PER_YEAR`); the
    default 252 keeps the equity behaviour byte-identical. Volume features
    are unit-free and ignore it.

    Returns ``None`` when ``name`` is not a derived feature (the caller then
    falls back to the stockstats path). Computation errors propagate —
    fail-visible, never a zero-filled column.
    """
    fn = DERIVED_FEATURES.get(name)
    if fn is None:
        return None
    if name in ("rvol_20", "ewma_vol"):
        return fn(data, periods_per_year=periods_per_year)
    return fn(data)
