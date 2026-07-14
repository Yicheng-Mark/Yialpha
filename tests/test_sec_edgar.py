"""Unit tests for ``yiagents.dataflows.sec_edgar`` (XBRL fundamentals vendor).

Hermetic: no network. ``_sec_get`` status mapping is checked by patching
``requests.get``; the CIK/companyfacts paths are checked by patching
``_cached_or_fetch`` to serve synthetic JSON. PIT filtering, concept aliasing,
format, and router integration round out coverage.
"""

from __future__ import annotations

import json

import pytest

from yiagents.dataflows import sec_edgar
from yiagents.dataflows.errors import NoMarketDataError, VendorRateLimitError


# --------------------------------------------------------------------------- #
# Synthetic SEC payloads
# --------------------------------------------------------------------------- #
TICKERS_JSON = json.dumps({
    "0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."},
    "1": {"cik_str": 789019, "ticker": "MSFT", "title": "Microsoft Corp"},
}).encode("utf-8")

FACTS_JSON = json.dumps({
    "cik": 320193,
    "entityName": "Apple Inc.",
    "facts": {"us-gaap": {
        # Assets: one FY (filed late) + three quarters. Lets PIT tests drop the
        # not-yet-filed FY at a curr_date before its filing.
        "Assets": {"units": {"USD": [
            {"end": "2024-09-28", "val": 36498000000, "fy": 2024, "fp": "FY", "form": "10-K", "filed": "2024-11-01"},
            {"end": "2024-06-29", "val": 35300000000, "fy": 2024, "fp": "Q3", "form": "10-Q", "filed": "2024-08-01"},
            {"end": "2024-03-30", "val": 34700000000, "fy": 2024, "fp": "Q2", "form": "10-Q", "filed": "2024-05-02"},
            {"end": "2023-12-30", "val": 34100000000, "fy": 2023, "fp": "Q1", "form": "10-Q", "filed": "2024-02-01"},
        ]}},
        "StockholdersEquity": {"units": {"USD": [
            {"end": "2024-09-28", "val": 57400000000, "fy": 2024, "fp": "FY", "form": "10-K", "filed": "2024-11-01"},
            {"end": "2024-06-29", "val": 56300000000, "fy": 2024, "fp": "Q3", "form": "10-Q", "filed": "2024-08-01"},
        ]}},
        "Revenues": {"units": {"USD": [
            {"start": "2023-09-25", "end": "2024-09-28", "val": 391035000000, "fy": 2024, "fp": "FY", "form": "10-K", "filed": "2024-11-01"},
        ]}},
        "EarningsPerShareDiluted": {"units": {"USD/shares": [
            {"start": "2023-09-25", "end": "2024-09-28", "val": 6.11, "fy": 2024, "fp": "FY", "form": "10-K", "filed": "2024-11-01"},
        ]}},
    }},
}).encode("utf-8")


def _patch_fetch(monkeypatch, tmp_path, tickers=TICKERS_JSON, facts=FACTS_JSON):
    """Point the vendor at synthetic payloads + a tmp cache dir; no network."""
    monkeypatch.setattr(sec_edgar, "_cache_dir", lambda: str(tmp_path))

    def fake_cached_or_fetch(_path, url, ttl_days):
        if url.endswith("company_tickers.json"):
            return tickers
        if "companyfacts" in url:
            return facts
        raise AssertionError(f"unexpected url: {url}")

    monkeypatch.setattr(sec_edgar, "_cached_or_fetch", fake_cached_or_fetch)


# --------------------------------------------------------------------------- #
# PIT filtering + concept mapping + format
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_balance_sheet_pit_drops_not_yet_filed_period(monkeypatch, tmp_path):
    _patch_fetch(monkeypatch, tmp_path)
    out = sec_edgar.get_balance_sheet("AAPL", "quarterly", "2024-10-01")
    # FY 2024 was filed 2024-11-01 > 2024-10-01 -> must NOT appear as a column.
    assert "2024-09-28" not in out.split("\n")[3]  # header row (after 2 # + blank)
    # Q3 (filed 2024-08-01) and earlier are public -> present.
    assert "2024-06-29" in out
    assert "2023-12-30" in out
    # Total Assets row carries the Q3 value, integer-formatted.
    assets_line = next(l for l in out.splitlines() if l.startswith("Total Assets,"))
    assert "35300000000" in assets_line  # Q3 value


@pytest.mark.unit
def test_balance_sheet_columns_descending_and_capped(monkeypatch, tmp_path):
    _patch_fetch(monkeypatch, tmp_path)
    out = sec_edgar.get_balance_sheet("AAPL", "quarterly", "2025-01-01")
    header = out.splitlines()[3]  # "# Balance...", "# Source...", "", header
    dates = header.split(",")[1:]          # drop the empty corner cell
    assert dates == sorted(dates, reverse=True)   # most-recent-first
    assert len(dates) <= 8


@pytest.mark.unit
def test_annual_freq_only_shows_FY(monkeypatch, tmp_path):
    _patch_fetch(monkeypatch, tmp_path)
    out = sec_edgar.get_balance_sheet("AAPL", "annual", "2024-12-01")
    # Only the FY column survives (Q1-Q4 excluded for annual).
    header = out.splitlines()[3]
    dates = header.split(",")[1:]
    assert dates == ["2024-09-28"]


@pytest.mark.unit
def test_no_public_period_raises_no_market_data(monkeypatch, tmp_path):
    _patch_fetch(monkeypatch, tmp_path)
    # curr_date before every filing -> nothing public.
    with pytest.raises(NoMarketDataError):
        sec_edgar.get_balance_sheet("AAPL", "annual", "2020-01-01")


@pytest.mark.unit
def test_format_contract_header_and_source(monkeypatch, tmp_path):
    _patch_fetch(monkeypatch, tmp_path)
    out = sec_edgar.get_balance_sheet("AAPL", "quarterly", "2025-01-01")
    assert out.startswith("# Balance Sheet for AAPL (quarterly)")
    assert "SEC EDGAR XBRL" in out
    # Transposed CSV: first data column is the line-item label.
    body = out.split("\n\n", 1)[1]
    first_col = body.splitlines()[0].split(",")[0]
    assert first_col == ""  # header row's first cell is empty (corner)


@pytest.mark.unit
def test_get_fundamentals_snapshot_is_pit(monkeypatch, tmp_path):
    _patch_fetch(monkeypatch, tmp_path)
    # curr_date before the FY filing -> Revenue/EPS (only FY data exists) show n/a.
    out = sec_edgar.get_fundamentals("AAPL", "2024-10-01")
    assert out.startswith("# Company Fundamentals for AAPL")
    assert "CIK: 320193" in out
    assert "Apple Inc." in out
    # EPS only has FY (filed 2024-11-01) -> not public on 2024-10-01.
    assert "EPS Diluted (latest): n/a" in out
    # But Q3 balance-sheet-derived items are public.
    assert "Total Assets" in out


# --------------------------------------------------------------------------- #
# CIK resolution + non-US + caching
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_non_us_ticker_raises_no_market_data(monkeypatch, tmp_path):
    _patch_fetch(monkeypatch, tmp_path)
    with pytest.raises(NoMarketDataError, match="US-listed"):
        sec_edgar.get_balance_sheet("0700.HK", "quarterly", "2025-01-01")


@pytest.mark.unit
def test_cik_resolution_caches_and_avoids_refetch(monkeypatch, tmp_path):
    """The real _cached_or_fetch serves the tickers table from disk on the second
    call, so _sec_get (the network) is hit exactly once."""
    monkeypatch.setattr(sec_edgar, "_cache_dir", lambda: str(tmp_path))
    calls = {"n": 0}

    def counting_sec_get(url):
        calls["n"] += 1
        return TICKERS_JSON if url.endswith("company_tickers.json") else FACTS_JSON

    monkeypatch.setattr(sec_edgar, "_sec_get", counting_sec_get)
    # NOTE: _cached_or_fetch is the real implementation here (not patched).
    cik1 = sec_edgar._cik_for_ticker("AAPL")
    cik2 = sec_edgar._cik_for_ticker("AAPL")
    assert cik1 == 320193 == cik2
    assert calls["n"] == 1   # second call served from the on-disk cache


# --------------------------------------------------------------------------- #
# _sec_get status mapping
# --------------------------------------------------------------------------- #
class _FakeResp:
    def __init__(self, status):
        self.status_code = status
        self.content = b"{}"


@pytest.mark.unit
def test_sec_get_429_raises_rate_limit(monkeypatch):
    monkeypatch.setattr("requests.get", lambda *a, **k: _FakeResp(429))
    with pytest.raises(VendorRateLimitError):
        sec_edgar._sec_get("https://data.sec.gov/x")


@pytest.mark.unit
def test_sec_get_403_treated_as_rate_limit(monkeypatch):
    monkeypatch.setattr("requests.get", lambda *a, **k: _FakeResp(403))
    with pytest.raises(VendorRateLimitError):
        sec_edgar._sec_get("https://data.sec.gov/x")


@pytest.mark.unit
def test_sec_get_404_raises_no_market_data(monkeypatch):
    monkeypatch.setattr("requests.get", lambda *a, **k: _FakeResp(404))
    with pytest.raises(NoMarketDataError):
        sec_edgar._sec_get("https://data.sec.gov/x")


@pytest.mark.unit
def test_sec_get_network_error_raises_no_market_data(monkeypatch):
    def _boom(*a, **k):
        raise OSError("connection refused")
    monkeypatch.setattr("requests.get", _boom)
    with pytest.raises(NoMarketDataError):
        sec_edgar._sec_get("https://data.sec.gov/x")


# --------------------------------------------------------------------------- #
# Router integration
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_router_routes_to_sec_edgar_vendor(monkeypatch, tmp_path):
    _patch_fetch(monkeypatch, tmp_path)
    from yiagents.dataflows import config as cfgmod
    from yiagents.dataflows.interface import route_to_vendor

    orig = cfgmod.get_config()
    try:
        cfgmod.set_config({**orig, "tool_vendors": {"get_balance_sheet": "sec_edgar"}})
        out = route_to_vendor("get_balance_sheet", "AAPL", "quarterly", "2025-01-01")
    finally:
        cfgmod.set_config(orig)
    assert "SEC EDGAR XBRL" in out
    assert "Total Assets" in out
