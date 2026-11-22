"""Round-13 adversarial-remediation regressions (2026-11-22).

Findings from the round-13 adversarial sweep whose subject code still exists,
each pinned here so its exact failure mode cannot silently return (the
A-share native findings were removed with the A-share data layer):

1. [P1] ``y_finance._get_stock_stats_bulk`` zipped the derived series against
   the full frame with ``strict=True``, but the vol estimators clean their
   input first (``vol_estimators._clean_ohlcv`` dropna's OHLC and drops
   non-positive rows), so a single Low=0 bad tick or a leading NaN High made
   the derived series SHORTER than the frame and the ValueError collapsed the
   WHOLE rvol_20/ewma_vol window into NO_DATA over otherwise-good rows.
2. [P2] ``fundamentals_overview`` and ``alpha_vantage_fundamentals`` judged
   the overview leak gate on an UNCLAMPED curr_date — an LLM echoing a cached
   header's "today" into a run pinned to a past analysis date passed as live
   mode and served today's snapshot into the historical run (the round-8/
   round-11 clamps existed everywhere else around these gates).

Hermetic: every vendor seam is monkeypatched; no network, no tokens.
"""

from __future__ import annotations

import unittest
from datetime import date

import pandas as pd
import pytest

import yialpha.dataflows.alpha_vantage_fundamentals as avf
import yialpha.dataflows.fundamentals_overview as fo
import yialpha.dataflows.y_finance as y_finance
from yialpha.dataflows.errors import NoMarketDataError
from yialpha.dataflows.utils import set_analysis_date

_TODAY = date.today().isoformat()


# --------------------------------------------------------------------------- #
# 1. [P1] y_finance — derived-series alignment after estimator row cleaning
# --------------------------------------------------------------------------- #
def _dirty_ohlcv(n: int = 30) -> pd.DataFrame:
    """A load_ohlcv-shaped frame carrying the two row classes that survive
    ``_clean_dataframe`` but are dropped by ``vol_estimators._clean_ohlcv``:
    a LEADING NaN High (forward-only gap fill cannot repair leading gaps) and
    a Low=0 bad tick (non-positive prices poison log-ratio windows)."""
    closes = [100.0 + (i % 5) for i in range(n)]
    frame = pd.DataFrame({
        "Date": pd.bdate_range("2026-07-01", periods=n),
        "Open": [c - 0.5 for c in closes],
        "High": [c + 1.0 for c in closes],
        "Low": [c - 1.0 for c in closes],
        "Close": closes,
        "Volume": [1000.0] * n,
    })
    frame.loc[0, "High"] = float("nan")  # leading NaN High (bdate row 2026-07-01)
    frame.loc[5, "Low"] = 0.0            # bad tick (bdate row 2026-07-08)
    return frame


def _window_lines(report: str) -> dict[str, str]:
    """Parse an indicator-window report into {date: rendered_value}."""
    return {
        ln.split(": ", 1)[0]: ln.split(": ", 1)[1]
        for ln in report.splitlines()
        if ln[:4].isdigit() and ": " in ln
    }


@pytest.mark.unit
@pytest.mark.parametrize("indicator", ["rvol_20", "ewma_vol"])
def test_indicators_window_survives_dirty_ohlcv_rows(monkeypatch, indicator):
    """A frame with cleanable bad rows must NOT crash the window into
    NO_DATA — the dropped days render 'N/A' and the rest keep their values
    (pre-fix: ValueError from the strict zip, whole window unavailable)."""
    frame = _dirty_ohlcv()
    monkeypatch.setattr(y_finance, "load_ohlcv", lambda s, d: frame.copy())
    out = y_finance.get_stock_stats_indicators_window(
        "AAPL", indicator, "2026-08-11", 45)
    assert f"## {indicator} values" in out
    lines = _window_lines(out)
    # The two dropped trading days render an honest 'N/A' VALUE — distinct
    # from a non-trading day's "N/A: Not a trading day" filler.
    assert lines["2026-07-01"] == "N/A"
    assert lines["2026-07-08"] == "N/A"
    # The last (good) trading day still carries a numeric reading.
    assert lines["2026-08-11"] != "N/A"


@pytest.mark.unit
def test_dirty_ohlcv_good_rows_match_clean_estimator(monkeypatch):
    """好行数值不变: the surviving rows must carry EXACTLY the values the
    estimator yields on the pre-cleaned frame — the alignment repair may not
    shift, refit, or otherwise alter good rows' numbers."""
    frame = _dirty_ohlcv()
    dropped = [0, 5]  # what vol_estimators._clean_ohlcv will remove
    clean = frame.drop(index=dropped)
    expected = y_finance.compute_derived(clean, "ewma_vol", periods_per_year=252.0)

    monkeypatch.setattr(y_finance, "load_ohlcv", lambda s, d: frame.copy())
    got = y_finance._get_stock_stats_bulk("AAPL", "ewma_vol", "2026-08-11")

    for idx in clean.index:
        date_str = frame.loc[idx, "Date"].strftime("%Y-%m-%d")
        exp = expected.loc[idx]
        assert got[date_str] == ("N/A" if pd.isna(exp) else str(float(exp))), date_str
    # The dropped days themselves are present-but-N/A, not absent.
    assert got["2026-07-01"] == "N/A"
    assert got["2026-07-08"] == "N/A"
    assert len(got) == len(frame)


@pytest.mark.unit
def test_clean_frame_path_is_byte_equivalent(monkeypatch):
    """A well-formed frame (no rows dropped) keeps the exact pre-round-13
    mapping — the reindex branch is a no-op when the derived index already
    equals the frame index."""
    frame = _dirty_ohlcv().drop(index=[0, 5]).reset_index(drop=True)
    expected = y_finance.compute_derived(frame, "ewma_vol")

    monkeypatch.setattr(y_finance, "load_ohlcv", lambda s, d: frame.copy())
    got = y_finance._get_stock_stats_bulk("AAPL", "ewma_vol", "2026-08-11")

    # The literal pre-fix construction (dates zipped against the series).
    pre_fix = {
        d.strftime("%Y-%m-%d"): ("N/A" if pd.isna(v) else str(float(v)))
        for d, v in zip(frame["Date"], expected, strict=True)
    }
    assert got == pre_fix


@pytest.mark.unit
def test_all_rows_dropped_renders_all_na(monkeypatch):
    """Pathological frame where EVERY row fails the cleaner: an honest
    all-N/A wall instead of a ValueError (the empty estimator series used to
    fail the strict zip like any other length mismatch)."""
    frame = _dirty_ohlcv()
    frame["Low"] = 0.0  # every row non-positive -> empty cleaned frame
    monkeypatch.setattr(y_finance, "load_ohlcv", lambda s, d: frame.copy())
    out = y_finance.get_stock_stats_indicators_window(
        "AAPL", "ewma_vol", "2026-08-11", 45)
    assert "## ewma_vol values" in out
    assert "2026-08-11: N/A\n" in out


# --------------------------------------------------------------------------- #
# 2. [P2] overview leak gates — clamp before overview_would_leak_future
# --------------------------------------------------------------------------- #
_SEC_OVERVIEW = (
    "# Company Fundamentals for MU\n"
    "# Source: SEC EDGAR XBRL\n"
    "Name: Micron Technology Inc.\n"
    "Total Assets: 130,000,000,000  (period ending 2026-06-05)\n"
)


@pytest.mark.unit
def test_aggregate_overview_inflated_curr_date_clamped_to_pinned(monkeypatch):
    """Round-13: an LLM-echoed 'today' curr_date in a run pinned to a past
    analysis date must be clamped BEFORE the Yahoo leak gate judges it — the
    aggregate degrades to SEC-only exactly like the direct past-date call
    (pre-fix the gate saw live mode and appended today's snapshot)."""
    monkeypatch.setattr(fo, "_sec_fundamentals", lambda t, c: _SEC_OVERVIEW)

    def no_yahoo(t, c):
        raise AssertionError("Yahoo snapshot fetched on a clamped-past run")

    monkeypatch.setattr(fo, "_cached_ticker_info", no_yahoo)
    set_analysis_date("2026-01-15")
    try:
        clamped = fo.get_fundamentals("MU", _TODAY)   # the inflated echo
        direct = fo.get_fundamentals("MU", "2026-01-15")
    finally:
        set_analysis_date(None)
    assert clamped == direct
    assert "point-in-time guard" in clamped   # Yahoo half gated, not served
    assert "Market Cap" not in clamped


@pytest.mark.unit
def test_av_overview_inflated_curr_date_refused_after_clamp(monkeypatch):
    """Same clamp for the Alpha Vantage OVERVIEW gate: the echoed 'today'
    must not reach the endpoint once the run is pinned to a past date."""
    def boom(_fn, _params):
        raise AssertionError("OVERVIEW endpoint reached on a clamped-past run")

    monkeypatch.setattr(avf, "_make_api_request", boom)
    set_analysis_date("2026-01-15")
    try:
        with pytest.raises(NoMarketDataError, match="point-in-time"):
            avf.get_fundamentals("AAPL", curr_date=_TODAY)
    finally:
        set_analysis_date(None)


@pytest.mark.unit
def test_av_overview_live_mode_still_serves(monkeypatch):
    """Live mode (no pinned analysis date): today's curr_date passes the gate
    unchanged — the round-13 clamp is a no-op there, byte-equivalent."""
    monkeypatch.setattr(
        avf, "_make_api_request", lambda _fn, _params: {"Symbol": "AAPL"})
    assert avf.get_fundamentals("AAPL", curr_date=_TODAY) == {"Symbol": "AAPL"}


if __name__ == "__main__":
    unittest.main()
