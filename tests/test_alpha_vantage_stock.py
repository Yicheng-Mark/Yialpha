"""Direct coverage for ``yialpha/dataflows/alpha_vantage_stock.py``.

Gap context (2026-09-29 test-gap audit): the AV *price* path
(``get_stock`` / ``TIME_SERIES_DAILY_ADJUSTED``) had zero direct tests —
``test_alpha_vantage_pit.py`` covers only the statement getters and
``test_market_data_date_clamp.py`` only the yfinance/binance clamps, so the
AV CSV window semantics (PIT clamp, client-side date filter, outputsize
heuristic, non-str fail-soft) were never executed by the suite.

Hermetic: ``_make_api_request`` is monkeypatched at the module namespace
(``avs._make_api_request`` — the ``from .alpha_vantage_common import``
binding), no network, no API key.
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from yialpha.dataflows import alpha_vantage_stock as avs
from yialpha.dataflows.utils import set_analysis_date

# AV DAILY_ADJUSTED CSV: newest-first, header + one row per session.
CSV_TEMPLATE = (
    "timestamp,open,high,low,close,adjusted_close,volume,dividend_amount,"
    "split_coefficient\n"
    "{rows}"
)


def _csv(*dates: str) -> str:
    rows = "\n".join(f"{d},100,110,95,105,104,1e6,0,1" for d in dates)
    return CSV_TEMPLATE.format(rows=rows)


def _capturing(monkeypatch, response):
    captured = {}

    def fake_request(function_name, params):
        captured["function"] = function_name
        captured["params"] = dict(params)
        return response

    monkeypatch.setattr(avs, "_make_api_request", fake_request)
    return captured


def _dates(result: str) -> list[str]:
    lines = [ln for ln in result.strip().splitlines() if ln]
    return [ln.split(",")[0] for ln in lines[1:]]  # drop header


@pytest.mark.unit
def test_pit_clamp_drops_rows_after_analysis_date(monkeypatch):
    """Pinned analysis date 2020-01-02: rows after it must never survive,
    even though the caller asked for a window ending 2030."""
    set_analysis_date("2020-01-02")
    try:
        captured = _capturing(
            monkeypatch,
            _csv("2020-01-03", "2020-01-02", "2019-12-30", "2030-06-01"),
        )
        result = avs.get_stock("AAPL", "2019-12-01", "2030-06-01")
        assert _dates(result) == ["2020-01-02", "2019-12-30"]
        # The clamp is client-side: the request itself is unaffected.
        assert captured["function"] == "TIME_SERIES_DAILY_ADJUSTED"
    finally:
        set_analysis_date(None)


@pytest.mark.unit
def test_live_mode_keeps_requested_window(monkeypatch):
    """No pinned analysis date → no clamp; the requested window is served
    verbatim (future-dated rows relative to the real clock included)."""
    set_analysis_date(None)
    _capturing(
        monkeypatch, _csv("2030-01-04", "2030-01-02", "2030-01-01")
    )
    result = avs.get_stock("AAPL", "2030-01-02", "2030-01-03")
    # 2030-01-04 is beyond the requested window, 2030-01-01 before it; the
    # in-window row survives with no clamp even though it is in the future
    # relative to the real clock.
    assert _dates(result) == ["2030-01-02"]


@pytest.mark.unit
def test_window_filter_is_inclusive_and_ordered_by_input(monkeypatch):
    """Both bounds inclusive; rows are kept in the CSV's own order."""
    _capturing(monkeypatch, _csv("2020-01-05", "2020-01-01", "2020-01-03"))
    result = avs.get_stock("AAPL", "2020-01-01", "2020-01-03")
    assert _dates(result) == ["2020-01-01", "2020-01-03"]


class TestOutputsizeHeuristic:
    """compact iff the window starts within the latest 100 days."""

    def test_recent_start_requests_compact(self, monkeypatch):
        start = (date.today() - timedelta(days=10)).isoformat()
        captured = _capturing(monkeypatch, _csv(start))
        avs.get_stock("AAPL", start, start)
        assert captured["params"]["outputsize"] == "compact"

    def test_old_start_requests_full(self, monkeypatch):
        start = (date.today() - timedelta(days=200)).isoformat()
        captured = _capturing(monkeypatch, _csv(start))
        avs.get_stock("AAPL", start, start)
        assert captured["params"]["outputsize"] == "full"


@pytest.mark.unit
def test_request_params_passthrough(monkeypatch):
    captured = _capturing(monkeypatch, _csv("2020-01-02"))
    avs.get_stock("IBM", "2020-01-01", "2020-01-02")
    assert captured["params"]["symbol"] == "IBM"
    assert captured["params"]["datatype"] == "csv"
    assert set(captured["params"]) == {"symbol", "outputsize", "datatype"}


@pytest.mark.unit
@pytest.mark.parametrize("bad_response", [{}, None, ["not", "a", "str"], 42])
def test_non_str_response_returns_empty_without_filtering(
    monkeypatch, bad_response
):
    """A non-str payload (error dict / None) must fail soft to "" BEFORE the
    date filter — pandas would raise on it and the router sentinel contract
    expects an empty string, not an exception."""
    _capturing(monkeypatch, bad_response)
    called = {"filter": False}
    original = avs._filter_csv_by_date_range
    monkeypatch.setattr(
        avs,
        "_filter_csv_by_date_range",
        lambda *a, **k: called.update(filter=True) or original(*a, **k),
    )
    assert avs.get_stock("AAPL", "2020-01-01", "2020-01-02") == ""
    assert called["filter"] is False
