"""Round-8 wave-1 known-issue landing pins (dataflows).

Pins the five landed vendor-layer fixes: the A-share sector-flow live-only
gate, Eastmoney margin pagination, SEC Form 4 files[] archive backfill, the
Alpha Vantage insider historical refusal, and the insider tool's
curr_date PIT pass-through. Hermetic: every vendor seam is mocked.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pandas as pd
import pytest

import yialpha.agents.utils.news_data_tools as ndt
import yialpha.dataflows.akshare_vendor as akv
import yialpha.dataflows.alpha_vantage_news as av_news
import yialpha.dataflows.eastmoney as eastmoney
import yialpha.dataflows.sec_ownership as so
from yialpha.dataflows.errors import NoMarketDataError
from yialpha.dataflows.sec_edgar import SecNoFileError

pytestmark = pytest.mark.unit


def _patch_ak(monkeypatch, **fns):
    """Patch ``_require_akshare`` to a fake module exposing the given fns."""
    monkeypatch.setattr(akv, "_require_akshare", lambda: SimpleNamespace(**fns))


def _sector_flow_df():
    return pd.DataFrame([
        {"名称": "白酒", "今日主力净流入-净额": 5.0e8, "今日主力净流入-净占比": 3.5,
         "今日超大单净流入-净额": 3.0e8, "今日大单净流入-净额": 2.0e8},
        {"名称": "半导体", "今日主力净流入-净额": 3.0e8, "今日主力净流入-净占比": 2.0,
         "今日超大单净流入-净额": 2.0e8, "今日大单净流入-净额": 1.0e8},
    ])


# ---------------------------------------------------------------------------
# KI-1: sector fund-flow is live-only (a today snapshot has no as-of form)
# ---------------------------------------------------------------------------


class TestSectorFlowLiveOnly:
    def test_historical_date_refused_without_network(self, monkeypatch):
        def _must_not_fetch(**kw):  # noqa: ARG001
            pytest.fail("sector flow must not be fetched for a historical date")

        _patch_ak(monkeypatch, stock_sector_fund_flow_rank=_must_not_fetch)
        monkeypatch.setattr(akv, "_baostock_industry", lambda t: "白酒")
        out = akv.get_a_share_sector_flow_native("600519.SS", "2024-06-20", 1)
        assert "SECTOR_FLOW_UNAVAILABLE" in out
        assert "leak future ranking information" in out

    def test_live_mode_still_fetches(self, monkeypatch):
        _patch_ak(
            monkeypatch, stock_sector_fund_flow_rank=lambda **kw: _sector_flow_df())
        monkeypatch.setattr(akv, "_baostock_industry", lambda t: "白酒")
        out = akv.get_a_share_sector_flow_native("600519.SS", None, 1)
        assert "白酒" in out
        assert "SECTOR_FLOW_UNAVAILABLE" not in out

    def test_today_is_live_not_refused(self, monkeypatch):
        # An explicit TODAY date is live mode per is_historical_date — the
        # gate must not have turned the tool off for same-day runs.
        _patch_ak(
            monkeypatch, stock_sector_fund_flow_rank=lambda **kw: _sector_flow_df())
        monkeypatch.setattr(akv, "_baostock_industry", lambda t: None)
        today = pd.Timestamp.today().strftime("%Y-%m-%d")
        out = akv.get_a_share_sector_flow_native("600519.SS", today, 1)
        assert "SECTOR_FLOW_UNAVAILABLE" not in out


# ---------------------------------------------------------------------------
# KI-2: Eastmoney margin pagination
# ---------------------------------------------------------------------------


def _margin_page(dates: list[str]) -> bytes:
    rows = [
        {"DATE": f"{d} 00:00:00", "SCODE": "600519", "RZYE": 1.0e10,
         "RQYE": 2.0e9, "RZRQYE": 1.2e10, "RZMRE": 5.0e8, "RQMCL": 100000,
         "RZJME": 1.0e8, "RZYEZB": 0.5}
        for d in dates
    ]
    return json.dumps({"result": {"pages": 2, "data": rows},
                       "success": True}).encode("utf-8")


class TestMarginPagination:
    def test_deep_window_pulls_second_page(self, monkeypatch, tmp_path):
        monkeypatch.setattr(eastmoney, "_cache_dir", lambda: str(tmp_path))
        # 400 business days ending 2025-12-31 (newest first), then 3 older
        # rows on page 2 — the window (1500d back from 2026-01-01) reaches
        # past page 1's oldest row.
        p1_dates = [
            d.strftime("%Y-%m-%d")
            for d in pd.bdate_range(end="2025-12-31", periods=400)
        ][::-1]
        p2_dates = ["2024-04-15", "2024-04-10", "2024-04-05"]
        calls: list[str] = []

        def fake_fetch(_path, url, params=None, ttl_days=1.0):  # noqa: ARG001
            page = str(params.get("pageNumber"))
            calls.append(page)
            return _margin_page(p1_dates if page == "1" else p2_dates)

        monkeypatch.setattr(eastmoney, "_cached_or_fetch", fake_fetch)
        out = eastmoney.get_margin_trading("600519.SS", "2026-01-01", 1500)
        assert calls == ["1", "2"]              # page 2 actually pulled
        assert p1_dates[0] in out               # newest page-1 row served
        assert "2024-04-15" in out              # page-2 row merged in
        assert "2024-04-05" in out

    def test_short_first_page_stops_after_one(self, monkeypatch, tmp_path):
        monkeypatch.setattr(eastmoney, "_cache_dir", lambda: str(tmp_path))
        calls: list[str] = []

        def fake_fetch(_path, url, params=None, ttl_days=1.0):  # noqa: ARG001
            calls.append(str(params.get("pageNumber")))
            return _margin_page(["2025-12-30", "2025-12-29"])

        monkeypatch.setattr(eastmoney, "_cached_or_fetch", fake_fetch)
        eastmoney.get_margin_trading("600519.SS", "2026-01-01", 30)
        assert calls == ["1"]                   # short page = end of history

    def test_window_covered_by_page1_stops(self, monkeypatch, tmp_path):
        monkeypatch.setattr(eastmoney, "_cache_dir", lambda: str(tmp_path))
        # A FULL 400-row page whose oldest row already predates the window
        # lower bound: page 2 can only add out-of-window rows — no fetch.
        p1_dates = [
            d.strftime("%Y-%m-%d")
            for d in pd.bdate_range(end="2025-12-31", periods=400)
        ][::-1]
        calls: list[str] = []

        def fake_fetch(_path, url, params=None, ttl_days=1.0):  # noqa: ARG001
            calls.append(str(params.get("pageNumber")))
            return _margin_page(p1_dates)

        monkeypatch.setattr(eastmoney, "_cached_or_fetch", fake_fetch)
        eastmoney.get_margin_trading("600519.SS", "2026-01-01", 60)
        assert calls == ["1"]


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


class TestAshareToolPitFallback:
    def test_sector_flow_tool_empty_curr_date_uses_pinned_date(self, monkeypatch):
        import yialpha.agents.utils.a_share_native_tools as ask
        from yialpha.dataflows.utils import pinned_analysis_date

        seen = []
        monkeypatch.setattr(
            ask, "route_to_vendor",
            lambda method, ticker, curr_date, *a: seen.append(
                (method, curr_date)) or "ok")
        with pinned_analysis_date("2024-06-01"):
            ask.get_a_share_sector_flow_native.invoke(
                {"ticker": "600519.SS", "curr_date": ""})
        assert seen == [("get_a_share_sector_flow_native", "2024-06-01")]
