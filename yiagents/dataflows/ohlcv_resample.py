"""Daily -> weekly OHLCV resampling for higher-timeframe analysis.

The indicator pipeline is daily-only: ``load_ohlcv`` fetches 5y of daily
bars and stockstats computes on them. Multi-timeframe confirmation (weekly
trend framing a daily decision) is standard technical-analysis practice, and
stockstats is resample-agnostic — it computes on any OHLCV frame — so the
only missing piece is a correct, PIT-safe resampler.

Point-in-time discipline: the caller's ``curr_date`` filter (applied by
``load_ohlcv`` BEFORE resampling) already removes look-ahead rows; the
resampler additionally drops the final week whose Friday label lies after
``curr_date`` — that bin can still gain rows in the real world, so treating
it as a completed weekly bar would quote a "weekly close" that is really a
mid-week snapshot.
"""

from __future__ import annotations

import logging

import pandas as pd

from .stockstats_utils import load_ohlcv

logger = logging.getLogger(__name__)

#: Weekly aggregation: first open, extremes for high/low, last close, summed
#: volume — the standard OHLCV rollup for a higher timeframe.
_WEEKLY_AGG = {
    "Open": "first",
    "High": "max",
    "Low": "min",
    "Close": "last",
    "Volume": "sum",
}


def resample_weekly(daily: pd.DataFrame, curr_date: str | None = None) -> pd.DataFrame:
    """Resample a daily OHLCV frame (capitalized columns) to W-FRI weekly bars.

    ``curr_date`` (YYYY-MM-DD) additionally drops the trailing week whose
    Friday label is after that date — an incomplete bin that could still gain
    daily rows. Weeks with no daily rows produce all-NaN aggregates and are
    dropped. The output keeps the capitalized column shape (``Date`` holds
    the week's Friday label) so stockstats can wrap it like any OHLCV frame.
    """
    if daily is None or daily.empty:
        return pd.DataFrame(columns=["Date", "Open", "High", "Low", "Close", "Volume"])

    df = daily.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date", "Close"]).sort_values("Date")
    if df.empty:
        return pd.DataFrame(columns=["Date", "Open", "High", "Low", "Close", "Volume"])

    indexed = df.set_index("Date")
    weekly = indexed.resample("W-FRI").agg(_WEEKLY_AGG)
    # Holidays-only weeks leave Open/Close NaN; a week without a close is not
    # a bar. Volume sums to 0 on data gaps — leave those alone, the close is
    # the gate.
    weekly = weekly.dropna(subset=["Close"])

    if curr_date is not None:
        cutoff = pd.to_datetime(curr_date, errors="coerce")
        if pd.notna(cutoff):
            # Drop the still-open week: its Friday label is in the future
            # relative to the analysis date, so the bin is incomplete.
            weekly = weekly[weekly.index <= cutoff]
        else:
            # Fail-visible: an unparseable curr_date must not silently keep
            # the trailing incomplete week (a look-ahead-style weekly close).
            # Upstream load_ohlcv would have raised on such a date already;
            # this branch guards direct callers of a PIT-safe public API.
            raise ValueError(
                f"curr_date {curr_date!r} is not a parseable YYYY-MM-DD date; "
                "refusing to resample with an unknown PIT cutoff"
            )

    return weekly.reset_index()


def weekly_ohlcv(symbol: str, curr_date: str) -> pd.DataFrame:
    """PIT-safe weekly OHLCV for ``symbol`` as of ``curr_date``.

    ``load_ohlcv`` filters daily rows to ``<= curr_date`` before resampling,
    and the trailing incomplete week is dropped by :func:`resample_weekly` —
    no weekly bar can therefore contain or imply a row after ``curr_date``.
    """
    daily = load_ohlcv(symbol, curr_date)
    return resample_weekly(daily, curr_date)
