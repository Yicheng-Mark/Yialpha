"""Completed-bars-only ATR with a live forming close (atr_stop).

A daily candle dated today (UTC) is still forming: its high/low expand with
the session and would distort the ATR feeding stops/leverage — so the
forming row is excluded from the ATR window. The returned close still comes
from the frame's last row: a forming candle's close IS the live price, and
excluding it would report yesterday's close as "current".
"""

from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd
import pytest

from yialpha.risk.atr_stop import latest_atr_from_frame


def _frame(closes, start="2024-01-01"):
    idx = pd.date_range(start, periods=len(closes), freq="D")
    return pd.DataFrame(
        {
            "Open": closes,
            "High": [c + 1.0 for c in closes],
            "Low": [c - 1.0 for c in closes],
            "Close": closes,
            "Adj Close": closes,
            "Volume": [10.0] * len(closes),
        },
        index=idx,
    )


@pytest.mark.unit
def test_atr_excludes_today_forming_bar_but_keeps_live_close():
    n = 60
    closes = [100.0 + i * 0.1 for i in range(n)]
    hist = _frame(closes)
    today = pd.Timestamp(datetime.now(UTC).date())
    # A wild forming candle: a 50-point range that would massively inflate
    # the ATR if the still-open bar were treated as a completed one.
    forming = pd.DataFrame(
        {
            "Open": [105.0], "High": [130.0], "Low": [80.0], "Close": [106.0],
            "Adj Close": [106.0], "Volume": [99.0],
        },
        index=[today],
    )
    full = pd.concat([hist, forming])

    live_close, atr_with_forming_present = latest_atr_from_frame(full)
    hist_close, atr_completed_only = latest_atr_from_frame(hist)

    # The live price is the forming candle's close, not yesterday's.
    assert live_close == pytest.approx(106.0)
    assert live_close != pytest.approx(hist_close)
    # The ATR is computed on completed bars only — the 50-point open bar
    # leaves no trace in the volatility reading.
    assert atr_with_forming_present == pytest.approx(atr_completed_only)


@pytest.mark.unit
def test_historical_frame_behaviour_unchanged():
    # Nothing dated today -> no row excluded -> historical semantics intact.
    closes = [100.0 + (i % 3) for i in range(50)]
    frame = _frame(closes, start="2023-05-01")
    close, atr = latest_atr_from_frame(frame)
    assert close == pytest.approx(closes[-1])
    assert atr > 0.0


@pytest.mark.unit
def test_tz_aware_index_forming_bar_still_excluded():
    """A tz-aware index must not defeat the forming-bar exclusion.

    The guard compared an aware stamp against the naive-UTC 'today' — that
    comparison raises TypeError, which the guard swallowed, silently keeping
    the forming bar in the ATR window. Both sides are now folded onto the
    file's naive-UTC convention before comparing.
    """
    n = 60
    closes = [100.0 + i * 0.1 for i in range(n)]
    hist = _frame(closes)
    hist.index = hist.index.tz_localize("UTC")  # same data, aware stamps
    today = pd.Timestamp.now(tz="UTC").normalize()
    forming = pd.DataFrame(
        {
            "Open": [105.0], "High": [130.0], "Low": [80.0], "Close": [106.0],
            "Adj Close": [106.0], "Volume": [99.0],
        },
        index=[today],
    )
    full = pd.concat([hist, forming])

    live_close, atr_with_forming = latest_atr_from_frame(full)
    hist_close, atr_completed = latest_atr_from_frame(hist)

    # The live price is still the forming candle's close...
    assert live_close == pytest.approx(106.0)
    # ...but the ATR is computed on completed bars only — the 50-point open
    # bar leaves no trace even on an aware index.
    assert atr_with_forming == pytest.approx(atr_completed)
