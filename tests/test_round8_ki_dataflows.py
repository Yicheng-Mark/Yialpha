"""Round-8 wave-1 known-issue landing pins (dataflows).

Pins the landed vendor-layer fixes: SEC Form 4 files[] archive backfill, the
Alpha Vantage insider historical refusal, and the insider tool's
curr_date PIT pass-through. Hermetic: every vendor seam is mocked.
"""

from __future__ import annotations

import pytest

import yialpha.agents.utils.news_data_tools as ndt
import yialpha.dataflows.alpha_vantage_news as av_news
import yialpha.dataflows.sec_ownership as so
from yialpha.dataflows.errors import NoMarketDataError
from yialpha.dataflows.sec_edgar import SecNoFileError

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# KI-3: Form 4 files[] archive backfill
# ---------------------------------------------------------------------------


def _subs_at_cap(files=None):
    """submissions JSON with the recent segment at its ~1000 cap (no 4s)."""
    n = 1000
    return {"filings": {"recent": {
        "form": ["10-K"] * n,
        "accessionNumber": [f"acc-recent-{i}" for i in range(n)],
        "filingDate": ["2025-06-01"] * n,
        "primaryDocument": ["x.xml"] * n,
    }, "files": files}}


def _segment_with_form4(filing_date: str) -> dict:
    return {"filings": {"recent": {
        "form": ["4"],
        "accessionNumber": ["acc-archive-4"],
        "filingDate": [filing_date],
        "primaryDocument": ["form4.xml"],
    }}}


class TestForm4ArchiveBackfill:
    def _patch(self, monkeypatch, segment):
        monkeypatch.setattr(so, "_cik_for_ticker", lambda t: 1)
        monkeypatch.setattr(so, "_fetch_submissions", lambda cik: _subs_at_cap(
            files=[{"name": "CIK0000000001-submissions-001.json"}]))
        monkeypatch.setattr(
            so, "_fetch_submissions_segment",
            lambda cik, name: segment if segment is not None else (_ for _ in ()).throw(
                SecNoFileError("gone")))
        monkeypatch.setattr(so, "_fetch_form4_xml", lambda cik, acc, doc: b"<xml/>")
        monkeypatch.setattr(so, "_parse_form4", lambda raw: {
            "owner": "Archived Insider", "title": "CEO",
            "trades": [{"date": "2024-05-20", "code": "S", "action": "SELL",
                        "shares": "1000", "price": "50", "post_shares": "9000"}],
        })

    def test_archive_form4_recovered(self, monkeypatch):
        self._patch(monkeypatch, _segment_with_form4("2024-05-21"))
        out = so.get_form4_insider_trading("TST", "2024-06-01", 180)
        assert "Archived Insider" in out
        assert "2024-05-20" in out
        # Backfill completed: the at-cap coverage note must NOT fire.
        assert "not fully scanned" not in out

    def test_missing_segment_degrades_to_note(self, monkeypatch):
        self._patch(monkeypatch, None)  # segment 404s
        out = so.get_form4_insider_trading("TST", "2024-06-01", 180)
        assert "No Form 4 transactions" in out
        assert "'recent' filings segment is at its ~1000 cap" in out
        assert "not fully scanned" in out

    def test_no_files_listed_degrades_to_note(self, monkeypatch):
        monkeypatch.setattr(so, "_cik_for_ticker", lambda t: 1)
        monkeypatch.setattr(
            so, "_fetch_submissions", lambda cik: _subs_at_cap(files=None))
        out = so.get_form4_insider_trading("TST", "2024-06-01", 180)
        assert "'recent' filings segment is at its ~1000 cap" in out


# ---------------------------------------------------------------------------
# KI-4: AV insider historical refusal
# ---------------------------------------------------------------------------


class TestAVInsiderPit:
    def test_historical_date_refused(self, monkeypatch):
        def _must_not_call(*a, **kw):  # noqa: ARG002
            pytest.fail("AV insider must not be fetched for a historical date")

        monkeypatch.setattr(av_news, "_make_api_request", _must_not_call)
        with pytest.raises(NoMarketDataError):
            av_news.get_insider_transactions("IBM", "2024-06-01")

    def test_live_mode_still_calls(self, monkeypatch):
        seen = {}

        def fake_request(endpoint, params):
            seen["endpoint"] = endpoint
            seen["params"] = params
            return {"data": []}

        monkeypatch.setattr(av_news, "_make_api_request", fake_request)
        av_news.get_insider_transactions("IBM")
        assert seen["endpoint"] == "INSIDER_TRANSACTIONS"
        assert seen["params"] == {"symbol": "IBM"}


# ---------------------------------------------------------------------------
# KI-5: the insider tool passes the PIT-clamped curr_date through
# ---------------------------------------------------------------------------


class TestInsiderToolPit:
    def _spy(self, monkeypatch):
        seen = []
        monkeypatch.setattr(ndt, "_vendor_ticker", lambda t: t)
        monkeypatch.setattr(
            ndt, "route_to_vendor",
            lambda method, ticker, curr_date=None: seen.append(
                (method, ticker, curr_date)) or "ok")
        return seen

    def test_curr_date_reaches_the_router(self, monkeypatch):
        seen = self._spy(monkeypatch)
        out = ndt.get_insider_transactions.invoke(
            {"ticker": "MU", "curr_date": "2024-06-01"})
        assert out == "ok"
        assert seen == [("get_insider_transactions", "MU", "2024-06-01")]

    def test_omitted_curr_date_is_none_in_live_mode(self, monkeypatch):
        seen = self._spy(monkeypatch)
        ndt.get_insider_transactions.invoke({"ticker": "MU"})
        assert seen == [("get_insider_transactions", "MU", None)]

    def test_empty_curr_date_falls_back_to_pinned_date(self, monkeypatch):
        # Review P2-1: "or empty for live mode" must not be a bypass — in a
        # run with a PINNED analysis date, an omitted/empty curr_date falls
        # back to the pinned date so the vendor-side historical gate fires.
        from yialpha.dataflows.utils import pinned_analysis_date

        seen = self._spy(monkeypatch)
        with pinned_analysis_date("2024-06-01"):
            ndt.get_insider_transactions.invoke({"ticker": "MU", "curr_date": ""})
            ndt.get_insider_transactions.invoke({"ticker": "MU"})
        assert seen == [
            ("get_insider_transactions", "MU", "2024-06-01"),
            ("get_insider_transactions", "MU", "2024-06-01"),
        ]
