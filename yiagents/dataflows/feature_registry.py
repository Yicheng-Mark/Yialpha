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

from .vol_estimators import close_to_close_vol, ewma_vol
from .volume_features import obv, relative_volume


def _rvol_20(data: pd.DataFrame) -> pd.Series:
    return close_to_close_vol(data, window=20)


def _ewma_vol(data: pd.DataFrame) -> pd.Series:
    return ewma_vol(data)


def _rel_vol_20(data: pd.DataFrame) -> pd.Series:
    return relative_volume(data, window=20)


#: Derived feature name -> computation on the raw capitalized OHLCV frame.
#: Keys are catalog entries (indicator_catalog.py) and IC-exportable columns.
DERIVED_FEATURES: dict[str, Callable[[pd.DataFrame], pd.Series]] = {
    "rvol_20": _rvol_20,
    "ewma_vol": _ewma_vol,
    "obv": obv,
    "rel_vol_20": _rel_vol_20,
}


def compute_derived(data: pd.DataFrame, name: str) -> pd.Series | None:
    """Compute derived feature ``name`` on a raw OHLCV frame.

    Returns ``None`` when ``name`` is not a derived feature (the caller then
    falls back to the stockstats path). Computation errors propagate —
    fail-visible, never a zero-filled column.
    """
    fn = DERIVED_FEATURES.get(name)
    if fn is None:
        return None
    return fn(data)
