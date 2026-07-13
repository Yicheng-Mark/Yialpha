"""Point-in-time (PIT) guards for fundamental data.

Covers the two lookahead leaks fixed in the dataflows layer:

1. Financial-statement filters (yfinance ``filter_financials_by_date`` and
   Alpha Vantage ``_filter_reports_by_date``) now apply a filing lag -- a
   fiscal period ending Sep 30 is NOT public on Oct 1 (the 10-Q is filed weeks
   later), so it must not be visible to a backtest decision date in between.
2. Overview snapshots (yfinance ``.info`` / AV ``OVERVIEW``) are today-only
   current-point values; on a past backtest date they leak the future, so the
   vendors now refuse and the router emits its NO_DATA_AVAILABLE sentinel.

Hermetic: no network. The PIT guards fire before any HTTP call, so the
router-level test is deterministic without mocking yfinance/Alpha Vantage.
"""

from __future__ import annotations

import pandas as pd
import pytest

from yiagents.dataflows.alpha_vantage_fundamentals import (
    _filter_reports_by_date,
    get_fundamentals as get_av_fundamentals,
)
from yiagents.dataflows.errors import NoMarketDataError
from yiagents.dataflows.interface import route_to_vendor
from yiagents.dataflows.stockstats_utils import filter_financials_by_date
from yiagents.dataflows.utils import (
    FUNDAMENTALS_FILING_LAG_DAYS,
    is_filing_public,
    overview_would_leak_future,
)
from yiagents.dataflows.y_finance import get_fundamentals as get_yf_fundamentals


# ---------------------------------------------------------------------------
# is_filing_public
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_filing_public_when_period_plus_lag_on_or_before_curr_date():
    # 2024-06-30 + 45d = 2024-08-14 <= 2024-10-15 -> public.
    assert is_filing_public("2024-06-30", "2024-10-15") is True


@pytest.mark.unit
def test_filing_not_public_within_filing_window():
    # The headline leak: a Q3 period ending 2024-09-30 is NOT public on
    # 2024-10-01 even though fiscalDateEnding <= curr_date -- the 10-Q has not
    # been filed yet. Pre-fix this would have been visible to the backtest.
    assert is_filing_public("2024-09-30", "2024-10-01") is False
    # ...and only becomes visible once 45 days have elapsed.
    assert is_filing_public("2024-09-30", "2024-11-14") is True


@pytest.mark.unit
def test_filing_public_in_live_mode_no_curr_date():
    # No as-of date (live mode) -> keep everything; PIT only constrains backtests.
    assert is_filing_public("2099-12-31", "") is True
    assert is_filing_public("2099-12-31", None) is True


@pytest.mark.unit
def test_filing_conservative_on_unparseable_period():
    # Cannot prove a report was public -> drop it rather than risk lookahead.
    assert is_filing_public("not-a-date", "2024-10-15") is False


@pytest.mark.unit
def test_filing_lag_zero_is_the_old_semantics():
    # The env escape hatch (YIAGENTS_FUNDAMENTALS_FILING_LAG_DAYS=0) reverts to
    # the pre-fix ``fiscalDateEnding <= curr_date`` behaviour: with lag 0 the
    # Sep-30 period IS visible on Oct-1.
    assert is_filing_public("2024-09-30", "2024-10-01", lag_days=0) is True
    assert is_filing_public("2024-10-02", "2024-10-01", lag_days=0) is False


@pytest.mark.unit
def test_filing_lag_default_is_45():
    assert FUNDAMENTALS_FILING_LAG_DAYS == 45


# ---------------------------------------------------------------------------
# overview_would_leak_future
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_overview_leaks_on_past_date():
    assert overview_would_leak_future("2024-01-01") is True


@pytest.mark.unit
def test_overview_does_not_leak_today_or_live():
    from datetime import date

    assert overview_would_leak_future(date.today().isoformat()) is False
    assert overview_would_leak_future(None) is False
    assert overview_would_leak_future("") is False


@pytest.mark.unit
def test_overview_conservative_on_unparseable():
    assert overview_would_leak_future("not-a-date") is False


# ---------------------------------------------------------------------------
# filter_financials_by_date (yfinance statements)
# ---------------------------------------------------------------------------
def _statement_df():
    # Columns are fiscal period end dates, plus a trailing "ttm" aggregate that
    # is metadata-like (non-date) and must survive the filter.
    return pd.DataFrame(
        {"2024-06-30": [1], "2024-09-30": [2], "2025-03-31": [3], "ttm": [4]}
    )


@pytest.mark.unit
def test_yfinance_statement_filter_applies_filing_lag():
    # At curr_date 2024-10-15: 2024-06-30 (+45d=08-14) is public; 2024-09-30
    # (+45d=11-14) and 2025-03-31 are not. "ttm" is non-date metadata -> kept.
    out = filter_financials_by_date(_statement_df(), "2024-10-15")
    assert list(out.columns) == ["2024-06-30", "ttm"]


@pytest.mark.unit
def test_yfinance_statement_filter_keeps_all_when_filed():
    # Late enough date that every period + lag has elapsed -> all date columns
    # survive (plus ttm).
    out = filter_financials_by_date(_statement_df(), "2025-12-31")
    assert list(out.columns) == ["2024-06-30", "2024-09-30", "2025-03-31", "ttm"]


@pytest.mark.unit
def test_yfinance_statement_filter_live_mode_noop():
    # No curr_date -> live mode, nothing dropped (byte-equivalent to passing the
    # frame through).
    out = filter_financials_by_date(_statement_df(), "")
    assert list(out.columns) == ["2024-06-30", "2024-09-30", "2025-03-31", "ttm"]


@pytest.mark.unit
def test_yfinance_statement_filter_empty_frame_noop():
    assert filter_financials_by_date(pd.DataFrame(), "2024-10-15").empty


# ---------------------------------------------------------------------------
# _filter_reports_by_date (Alpha Vantage statements)
# ---------------------------------------------------------------------------
def _av_result():
    return {
        "symbol": "AAPL",
        "quarterlyReports": [
            {"fiscalDateEnding": "2024-06-30", "totalAssets": "1"},
            {"fiscalDateEnding": "2024-09-30", "totalAssets": "2"},
            {"fiscalDateEnding": "2025-03-31", "totalAssets": "3"},
        ],
    }


@pytest.mark.unit
def test_av_statement_filter_applies_filing_lag():
    out = _filter_reports_by_date(_av_result(), "2024-10-15")
    periods = [r["fiscalDateEnding"] for r in out["quarterlyReports"]]
    assert periods == ["2024-06-30"]


@pytest.mark.unit
def test_av_statement_filter_live_mode_noop():
    out = _filter_reports_by_date(_av_result(), "")
    assert len(out["quarterlyReports"]) == 3


# ---------------------------------------------------------------------------
# Overview refusal (M2) -- both vendors raise before any network call.
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_yfinance_get_fundamentals_refuses_past_date():
    with pytest.raises(NoMarketDataError):
        get_yf_fundamentals("AAPL", "2024-01-01")


@pytest.mark.unit
def test_av_get_fundamentals_refuses_past_date():
    with pytest.raises(NoMarketDataError):
        get_av_fundamentals("AAPL", "2024-01-01")


@pytest.mark.unit
def test_router_emits_no_data_sentinel_for_past_overview():
    # End-to-end: both vendors' overview is today-only, so on a past backtest
    # date both raise NoMarketDataError before any HTTP call and the router
    # returns its NO_DATA_AVAILABLE sentinel instead of leaking future data.
    out = route_to_vendor("get_fundamentals", "AAPL", "2024-01-01")
    assert isinstance(out, str)
    assert "NO_DATA_AVAILABLE" in out
