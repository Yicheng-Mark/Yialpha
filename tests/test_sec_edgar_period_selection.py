"""SEC statement rendering must pick facts by REPORTING DURATION (B6).

A 10-Q reports the standalone quarter AND the year-to-date cumulative under
the SAME period-end date; the units array order is arbitrary. The previous
last-write-wins made the "quarterly" revenue silently render the 9-month YTD
value whenever that fact happened to come last, drifting with filing order.
"""

from __future__ import annotations

import json

import pytest

from yialpha.dataflows import sec_edgar

TICKERS_JSON = json.dumps({
    "0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."},
}).encode("utf-8")


def _facts_with_revenues(records: list[dict]) -> bytes:
    """companyfacts payload whose us-gaap.Revenues carries ``records``."""
    return json.dumps({
        "cik": 320193,
        "entityName": "Apple Inc.",
        "facts": {"us-gaap": {
            "Revenues": {"units": {"USD": records}},
        }},
    }).encode("utf-8")


def _patch_fetch(monkeypatch, tmp_path, facts):
    monkeypatch.setattr(sec_edgar, "_cache_dir", lambda: str(tmp_path))

    def fake_cached_or_fetch(_path, url, ttl_days):
        if url.endswith("company_tickers.json"):
            return TICKERS_JSON
        if "companyfacts" in url:
            return facts
        raise AssertionError(f"unexpected url: {url}")

    monkeypatch.setattr(sec_edgar, "_cached_or_fetch", fake_cached_or_fetch)


# Q3: the standalone 3-month quarter and the 9-month YTD both end 2024-06-30.
_QUARTER = {"start": "2024-04-01", "end": "2024-06-30", "val": 100,
            "fy": 2024, "fp": "Q3", "form": "10-Q", "filed": "2024-08-01"}
_YTD = {"start": "2023-10-01", "end": "2024-06-30", "val": 300,
        "fy": 2024, "fp": "Q3", "form": "10-Q", "filed": "2024-08-01"}


@pytest.mark.unit
@pytest.mark.parametrize("order", [("quarter_first", [_QUARTER, _YTD]),
                                   ("ytd_first", [_YTD, _QUARTER])])
def test_quarterly_picks_standalone_quarter_regardless_of_order(
    monkeypatch, tmp_path, order,
):
    """The quarterly income statement must show the ~90-day figure whether the
    YTD fact comes first or last in the units array (was: last-write-wins)."""
    _name, records = order
    _patch_fetch(monkeypatch, tmp_path, _facts_with_revenues(records))
    out = sec_edgar.get_income_statement("AAPL", "quarterly", "2024-12-01")
    rev_line = next(line for line in out.splitlines() if line.startswith("Revenue,"))
    assert "100" in rev_line
    assert "300" not in rev_line


@pytest.mark.unit
def test_annual_picks_full_year_over_same_end_shorter_duration(
    monkeypatch, tmp_path,
):
    """Annual statements target ~365 days: a 12-month fact beats a shorter
    duration sharing the same end (e.g. a transition-period filing)."""
    records = [
        {"start": "2024-01-01", "end": "2024-12-31", "val": 700,
         "fy": 2024, "fp": "FY", "form": "10-K", "filed": "2025-02-01"},
        {"start": "2024-07-01", "end": "2024-12-31", "val": 400,
         "fy": 2024, "fp": "FY", "form": "10-K", "filed": "2025-02-01"},
    ]
    _patch_fetch(monkeypatch, tmp_path, _facts_with_revenues(records))
    out = sec_edgar.get_income_statement("AAPL", "annual", "2025-06-01")
    rev_line = next(line for line in out.splitlines() if line.startswith("Revenue,"))
    assert "700" in rev_line
    assert "400" not in rev_line


@pytest.mark.unit
def test_equal_distance_tie_breaks_to_latest_start(monkeypatch, tmp_path):
    """Two facts EQUIDISTANT from the quarterly target (89 and 91 days): the
    later ``start`` (the more recent reporting of the period) wins."""
    records = [
        {"start": "2024-03-31", "end": "2024-06-30", "val": 99,  # 91 days
         "fy": 2024, "fp": "Q3", "form": "10-Q", "filed": "2024-08-01"},
        {"start": "2024-04-02", "end": "2024-06-30", "val": 101,  # 89 days
         "fy": 2024, "fp": "Q3", "form": "10-Q/A", "filed": "2024-09-01"},
    ]
    assert abs(sec_edgar._duration_days(records[0]) - 90) == \
        abs(sec_edgar._duration_days(records[1]) - 90)  # sanity: equidistant
    _patch_fetch(monkeypatch, tmp_path, _facts_with_revenues(records))
    out = sec_edgar.get_income_statement("AAPL", "quarterly", "2024-12-01")
    rev_line = next(line for line in out.splitlines() if line.startswith("Revenue,"))
    assert "101" in rev_line


@pytest.mark.unit
def test_instant_facts_unaffected_by_duration_matching(monkeypatch, tmp_path):
    """Balance-sheet concepts are INSTANT facts (no ``start``): duration
    selection must not disturb them, and duplicate filings resolve cleanly."""
    facts = json.dumps({
        "cik": 320193,
        "entityName": "Apple Inc.",
        "facts": {"us-gaap": {
            "Assets": {"units": {"USD": [
                {"end": "2024-06-30", "val": 35300000000,
                 "fy": 2024, "fp": "Q3", "form": "10-Q", "filed": "2024-08-01"},
                {"end": "2024-06-30", "val": 35300000000,
                 "fy": 2024, "fp": "Q3", "form": "10-Q/A", "filed": "2024-09-01"},
            ]}},
        }},
    }).encode("utf-8")
    _patch_fetch(monkeypatch, tmp_path, facts)
    out = sec_edgar.get_balance_sheet("AAPL", "quarterly", "2024-12-01")
    assets_line = next(line for line in out.splitlines() if line.startswith("Total Assets,"))
    assert "35300000000" in assets_line


@pytest.mark.unit
def test_duration_days_helper():
    assert sec_edgar._duration_days(_QUARTER) == 90
    assert sec_edgar._duration_days(_YTD) == 273
    # Instant facts (no start) and malformed dates are not durations.
    assert sec_edgar._duration_days({"end": "2024-06-30"}) is None
    assert sec_edgar._duration_days({"start": "garbage", "end": "2024-06-30"}) is None
