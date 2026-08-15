"""PIT regression tests for Alpha Vantage statement filtering (A5).

Root cause under test
---------------------
``_make_api_request`` ALWAYS returns the raw response *text* (``str`` —
``alpha_vantage_common.py`` does ``raw.decode()`` before returning). The old
``_filter_reports_by_date`` guarded with ``isinstance(result, dict)``, which is
always False for those strings, so the annual/quarterly report filters never
executed and backtests ingested reports that were not yet public on
``curr_date`` (lookahead bias).

The fix parses the string with ``json.loads`` when it is a JSON statement
payload, filters ``annualReports`` / ``quarterlyReports`` by
:func:`yiagents.dataflows.utils.is_filing_public` (period end + filing lag),
and returns the filtered ``dict``. A non-JSON string (CSV datasets, error
bodies) is returned unchanged — there is nothing statement-shaped in it.

Hermetic: ``_make_api_request`` is monkeypatched to return synthetic strings;
no network, no API key.
"""

from __future__ import annotations

import json

import pytest

from yiagents.dataflows import alpha_vantage_fundamentals as avf

#curr_date 2024-06-15. is_filing_public uses a 45-day default lag (env-tunable
# via YIAGENTS_FUNDAMENTALS_FILING_LAG_DAYS); the fixtures below stay
# deterministic for any lag in [0, 300]: the past report (2023-12-31) is
# public by 2024-06-15, the future one (2024-06-30) is not.
CURR = "2024-06-15"

STATEMENTS_JSON = {
    "symbol": "AAPL",
    "annualReports": [
        {"fiscalDateEnding": "2023-12-31", "totalAssets": "1000"},
        {"fiscalDateEnding": "2024-06-30", "totalAssets": "9999"},  # not public
    ],
    "quarterlyReports": [
        {"fiscalDateEnding": "2024-03-31", "totalAssets": "500"},
        {"fiscalDateEnding": "2024-06-30", "totalAssets": "8888"},  # not public
    ],
}


@pytest.mark.unit
@pytest.mark.parametrize("getter", [avf.get_balance_sheet, avf.get_cashflow,
                                    avf.get_income_statement])
def test_str_json_payload_is_parsed_and_future_reports_dropped(monkeypatch, getter):
    """The A5 regression: a str payload from _make_api_request must be parsed
    and PIT-filtered, not passed through untouched."""
    monkeypatch.setattr(
        avf, "_make_api_request",
        lambda _fn, _params: json.dumps(STATEMENTS_JSON))
    result = getter("AAPL", curr_date=CURR)
    assert isinstance(result, dict), (
        "the str payload must be json.loads-ed into a dict by the filter")
    assert [r["fiscalDateEnding"] for r in result["annualReports"]] == \
        ["2023-12-31"]
    assert [r["fiscalDateEnding"] for r in result["quarterlyReports"]] == \
        ["2024-03-31"]
    assert "9999" not in json.dumps(result)
    assert "8888" not in json.dumps(result)


@pytest.mark.unit
def test_non_json_str_returned_unchanged(monkeypatch):
    """A CSV / error body has nothing statement-shaped to filter — it must be
    returned as the original str (the caller classifies it), never crash."""
    csv_text = "timestamp,open,high,low,close\n2024-06-14,190,195,189,194"
    monkeypatch.setattr(avf, "_make_api_request",
                        lambda _fn, _params: csv_text)
    assert avf.get_balance_sheet("AAPL", curr_date=CURR) == csv_text
    assert avf.get_cashflow("AAPL", curr_date=CURR) == csv_text
    assert avf.get_income_statement("AAPL", curr_date=CURR) == csv_text


@pytest.mark.unit
def test_live_mode_passthrough_unchanged(monkeypatch):
    """curr_date empty (live mode) -> no as-of constraint; the payload is
    returned UNTOUCHED (still the raw str) so live-mode behaviour stays
    byte-identical to the pre-filter path."""
    payload = json.dumps(STATEMENTS_JSON)
    monkeypatch.setattr(avf, "_make_api_request",
                        lambda _fn, _params: payload)
    assert avf.get_balance_sheet("AAPL") == payload
    assert avf.get_cashflow("AAPL") == payload
    assert avf.get_income_statement("AAPL") == payload


@pytest.mark.unit
def test_dict_payload_still_filtered(monkeypatch):
    """Defence in depth: if _make_api_request ever returns a dict directly,
    the filter must still apply (the isinstance branch that used to be the
    only path)."""
    monkeypatch.setattr(
        avf, "_make_api_request", lambda _fn, _params: dict(STATEMENTS_JSON))
    result = avf.get_balance_sheet("AAPL", curr_date=CURR)
    assert isinstance(result, dict)
    assert len(result["annualReports"]) == 1
    assert result["annualReports"][0]["fiscalDateEnding"] == "2023-12-31"


@pytest.mark.unit
def test_filtered_dict_is_json_serializable(monkeypatch):
    """The filtered dict must survive the tool layer's JSON serialization of
    non-str tool outputs (how LangGraph's ToolNode renders it to the LLM)."""
    monkeypatch.setattr(
        avf, "_make_api_request",
        lambda _fn, _params: json.dumps(STATEMENTS_JSON))
    result = avf.get_income_statement("AAPL", curr_date=CURR)
    round_trip = json.loads(json.dumps(result))
    assert round_trip == result
