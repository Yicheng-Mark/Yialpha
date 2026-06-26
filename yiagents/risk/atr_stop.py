"""ATR-based stop-loss computation.

Reuses the existing ATR indicator produced by ``stockstats`` on top of the
project's :func:`~yiagents.dataflows.stockstats_utils.load_ohlcv`
loader rather than recomputing ATR from scratch. This keeps the data path
(identical caching, point-in-time truncation, stale-frame guard) in one
place.

Only the long stop is implemented here. Short positions are the caller's
responsibility (negate and mirror the offset).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from stockstats import wrap

from yiagents.dataflows.stockstats_utils import load_ohlcv

# A precomputed OHLCV frame OR a ticker symbol. Accepted by atr_stop.
SymbolOrFrame = str | pd.DataFrame


def latest_atr(symbol: str, curr_date: str) -> tuple[float, float]:
    """Return ``(last_close, last_atr)`` for ``symbol`` truncated to ``curr_date``.

    Point-in-time safe: ``load_ohlcv`` caches the full history but filters
    rows to ``<= curr_date`` so backtests never see future prices, and it
    rejects a present-but-stale frame. ATR is the stockstats 14-period ATR.

    Raises :class:`ValueError` when there is no data or the ATR cannot be
    computed (too few rows / all-NaN tail).
    """
    data = load_ohlcv(symbol, curr_date)
    return latest_atr_from_frame(data)


def latest_atr_from_frame(ohlcv: pd.DataFrame) -> tuple[float, float]:
    """Return ``(last_close, last_atr)`` from a precomputed OHLCV DataFrame.

    The project's ``load_ohlcv`` returns a frame with capitalised columns
    ``Date`` / ``Open`` / ``High`` / ``Low`` / ``Close`` / ``Volume``. This
    helper wraps it with ``stockstats`` to compute the 14-period ``atr``
    and reads the last row.

    Raises :class:`ValueError` on empty/NaN input.
    """
    if ohlcv is None:
        raise ValueError("atr_stop: OHLCV frame is None")
    try:
        rows = len(ohlcv)
    except TypeError as exc:
        raise ValueError("atr_stop: OHLCV frame has no length") from exc
    if rows == 0:
        raise ValueError("atr_stop: OHLCV frame is empty")

    if "Close" not in ohlcv.columns:
        raise ValueError("atr_stop: OHLCV frame missing a 'Close' column")

    # Work on a copy so stockstats' column additions never leak back.
    df = wrap(ohlcv.copy())
    close_series = df["close"]
    atr_series = df["atr"]  # access triggers stockstats computation

    # Walk back from the tail to the last finite (close, atr) pair.
    close_arr = np.asarray(close_series, dtype=float)
    atr_arr = np.asarray(atr_series, dtype=float)
    for i in range(len(close_arr) - 1, -1, -1):
        c = close_arr[i]
        a = atr_arr[i]
        if np.isfinite(c) and np.isfinite(a):
            if a <= 0.0:
                raise ValueError("atr_stop: computed ATR is non-positive")
            return float(c), float(a)

    raise ValueError("atr_stop: no finite (close, atr) row in the OHLCV frame")


def atr_stop(
    symbol_or_frame: SymbolOrFrame,
    curr_date: str | None = None,
    mult: float = 2.0,
    period: int = 14,
) -> float:
    """Long ATR stop ``= last_close - mult * last_atr``.

    For shorts the caller mirrors the offset (``last_close + mult*atr``).

    ``symbol_or_frame`` is either a ticker symbol string — in which case
    the OHLCV is fetched via :func:`latest_atr` using ``curr_date`` — or a
    precomputed OHLCV DataFrame, in which case :func:`latest_atr_from_frame`
    is used and ``curr_date`` is ignored.

    ``period`` is accepted for API symmetry; stockstats' built-in ``atr``
    is a fixed 14-period indicator, so only the default is honoured.
    """
    if isinstance(symbol_or_frame, str):
        if curr_date is None:
            raise ValueError("atr_stop: curr_date is required when a symbol is passed")
        last_close, last_atr = latest_atr(symbol_or_frame, curr_date)
    else:
        last_close, last_atr = latest_atr_from_frame(symbol_or_frame)

    m = float(mult)
    if not np.isfinite(m):
        raise ValueError("atr_stop: mult is not finite")
    if m < 0:
        raise ValueError("atr_stop: mult must be non-negative")

    return float(last_close - m * last_atr)
