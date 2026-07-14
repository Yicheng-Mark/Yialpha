"""Unit tests for ``yiagents.dataflows.sec_ownership`` (Form 4 + FTD) and the
fundamentals-analyst wiring (Track B2).

Hermetic: no network. The vendor paths are exercised by patching
``sec_ownership._cached_or_fetch`` to serve synthetic JSON/XML/text payloads and
``sec_ownership._cik_for_ticker`` to a stub. PIT filtering (Form 4 filingDate;
FTD cutoff + publication lag), header-driven FTD parsing, the non-US/empty
contracts, and router integration round out the dataflow coverage.

The wiring tests pin the byte-equivalence contract (the project "iron rule"):
with ``sec_ownership`` off (the default) the fundamentals analyst binds exactly
its 4 baseline tools and an unchanged prompt; with it on the two new tools are
appended in order. Pure mock-LLM, zero network, zero LLM cost.
"""

from __future__ import annotations

import unittest

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import Runnable

from yiagents.agents.analysts.fundamentals_analyst import create_fundamentals_analyst
from yiagents.dataflows import sec_ownership
from yiagents.dataflows.errors import NoMarketDataError


# --------------------------------------------------------------------------- #
# Synthetic payloads
# --------------------------------------------------------------------------- #
def _form4_xml(owner: str, title: str, code: str, shares: str, price: str,
               tdate: str, post: str) -> bytes:
    return (
        "<ownershipDocument>"
        "<reportingOwner>"
        f"<reportingOwnerId><rptOwnerName>{owner}</rptOwnerName></reportingOwnerId>"
        f"<reportingOwnerRelationship><officerTitle>{title}</officerTitle></reportingOwnerRelationship>"  # noqa: E501
        "</reportingOwner>"
        "<nonDerivativeTable><nonDerivativeTransaction>"
        f"<transactionDate><value>{tdate}</value></transactionDate>"
        f"<transactionCoding><transactionCode>{code}</transactionCode></transactionCoding>"
        "<transactionAmounts>"
        f"<transactionShares><value>{shares}</value></transactionShares>"
        f"<transactionPricePerShare><value>{price}</value></transactionPricePerShare>"
        "<transactionAcquiredSoldCode><value>D</value></transactionAcquiredSoldCode>"
        "</transactionAmounts>"
        "<postTransactionAmounts>"
        f"<sharesOwnedFollowingTransaction><value>{post}</value></sharesOwnedFollowingTransaction>"
        "</postTransactionAmounts>"
        "</nonDerivativeTransaction></nonDerivativeTable>"
        "</ownershipDocument>"
    ).encode("utf-8")


# Three Form 4 filings: 2024-04-01 (BUY), 2024-06-10 (SELL), 2024-07-01 (SELL,
# not-yet-public at the PIT cutoff used below).
SUBMISSIONS_JSON = (
    '{"cik":320193,"name":"Apple Inc.","filings":{"recent":{'
    '"form":["4","4","4"],'
    '"accessionNumber":["000032019324000003","000032019324000002","000032019324000001"],'
    '"filingDate":["2024-07-01","2024-06-10","2024-04-01"],'
    '"primaryDocument":["f3.xml","f2.xml","f1.xml"]'
    "}}}"
).encode("utf-8")

XML_BY_DOC = {
    "f1.xml": _form4_xml("KATHERINE ADAMS", "GC", "P", "10000", "170.00",
                         "2024-03-28", "200000"),       # BUY (code P)
    "f2.xml": _form4_xml("LUCA MAESTRI", "CFO", "S", "50000", "195.20",
                         "2024-06-05", "100000"),       # SELL
    "f3.xml": _form4_xml("JEFF WILLIAMS", "COO", "S", "30000", "210.00",
                         "2024-06-28", "80000"),        # SELL (not-yet-public)
}

# FTD file content (pipe-delimited, with a parenthesized fails column to prove
# header-driven parsing tolerates the SEC's label variants).
FTD_PIPE = (
    "Date|CUSIP|Issuer Name|Symbol|Total Fails (To Deliver)|Price\n"
    "20240610|037833100|APPLE INC|AAPL|1234567|195.20\n"
    "20240612|037833100|APPLE INC|AAPL|2345678|196.10\n"
    "20240611|594918104|MICROSOFT CORP|MSFT|999999|420.00\n"
).encode("utf-8")

# Same data, tab-delimited — proves the delimiter auto-detection.
FTD_TAB = (
    "Date\tCUSIP\tIssuer Name\tSymbol\tTotal Fails\tPrice\n"
    "20240610\t037833100\tAPPLE INC\tAAPL\t1234567\t195.20\n"
).encode("utf-8")


def _patch_form4(monkeypatch, tmp_path):
    """Point sec_ownership at synthetic CIK + submissions + Form4 XMLs."""
    monkeypatch.setattr(sec_ownership, "_cache_dir", lambda: str(tmp_path))
    monkeypatch.setattr(sec_ownership, "_cik_for_ticker", lambda t: 320193)

    def fake_fetch(_path, url, ttl_days):
        if "submissions" in url:
            return SUBMISSIONS_JSON
        for doc, payload in XML_BY_DOC.items():
            if doc in url:
                return payload
        raise AssertionError(f"unexpected url: {url}")

    monkeypatch.setattr(sec_ownership, "_cached_or_fetch", fake_fetch)


# --------------------------------------------------------------------------- #
# Form 4
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_form4_pit_drops_not_yet_filed(monkeypatch, tmp_path):
    _patch_form4(monkeypatch, tmp_path)
    # curr_date 2024-06-15: the 2024-07-01 filing is not yet public -> dropped.
    out = sec_ownership.get_form4_insider_trading("AAPL", "2024-06-15", 180)
    assert "# Form 4 Insider Trading for AAPL" in out
    assert "KATHERINE ADAMS" in out     # 2024-04-01 BUY
    assert "LUCA MAESTRI" in out        # 2024-06-10 SELL
    assert "JEFF WILLIAMS" not in out   # 2024-07-01 not public yet


@pytest.mark.unit
def test_form4_action_codes_and_summary(monkeypatch, tmp_path):
    _patch_form4(monkeypatch, tmp_path)
    out = sec_ownership.get_form4_insider_trading("AAPL", "2024-07-15", 180)
    # All three filings public now.
    assert "BUY" in out                 # Adams P-code -> BUY
    assert "SELL" in out                # Maestri + Williams S-code -> SELL
    # Summary line present with a net direction.
    assert "Summary" in out
    assert "net seller" in out          # sells >> buys in dollar terms


@pytest.mark.unit
def test_form4_non_us_raises_no_market_data(monkeypatch, tmp_path):
    monkeypatch.setattr(sec_ownership, "_cik_for_ticker",
                        lambda t: (_ for _ in ()).throw(
                            NoMarketDataError(t, detail="US-listed only")))
    with pytest.raises(NoMarketDataError):
        sec_ownership.get_form4_insider_trading("0700.HK", "2024-06-15", 180)


@pytest.mark.unit
def test_form4_no_filings_in_window(monkeypatch, tmp_path):
    _patch_form4(monkeypatch, tmp_path)
    # curr_date well before all filings -> nothing public -> honest empty string.
    out = sec_ownership.get_form4_insider_trading("AAPL", "2020-01-01", 180)
    assert out.startswith("# Form 4 Insider Trading for AAPL")
    assert "No Form 4 transactions" in out


# --------------------------------------------------------------------------- #
# FTD
# --------------------------------------------------------------------------- #
def _patch_ftd(monkeypatch, tmp_path, content=FTD_PIPE):
    """Serve `content` only for the cnbs20240531 file; 404 everything else.

    Records requested URLs so a test can assert the PIT gate kept not-yet-public
    cutoff files from even being requested.
    """
    monkeypatch.setattr(sec_ownership, "_cache_dir", lambda: str(tmp_path))
    requested = []
    monkeypatch.setattr(sec_ownership, "_requested_ftd_urls", requested, raising=False)

    def fake_fetch(_path, url, ttl_days):
        requested.append(url)
        if "cnbs20240531" in url:
            return content
        raise NoMarketDataError(url, detail="404")

    monkeypatch.setattr(sec_ownership, "_cached_or_fetch", fake_fetch)
    return requested


@pytest.mark.unit
def test_ftd_pit_skips_not_yet_public_file(monkeypatch, tmp_path):
    requested = _patch_ftd(monkeypatch, tmp_path)
    # curr_date 2024-06-20, lag 10 -> visible cutoffs must satisfy
    # cutoff + 10 <= 2024-06-20 (cutoff <= 2024-06-10). The 2024-06-15 cutoff
    # file is therefore NOT public yet and must never be requested.
    out = sec_ownership.get_ftd_data("AAPL", "2024-06-20", 90)
    assert not any("cnbs20240615" in u for u in requested)
    # The public 2024-05-31 file was served with AAPL rows.
    assert any("cnbs20240531" in u for u in requested)
    assert "# Fails-to-Deliver for AAPL" in out
    assert "1234567" in out
    assert "2345678" in out
    assert "MSFT" not in out          # ticker filter excludes Microsoft rows


@pytest.mark.unit
def test_ftd_tab_delimiter_parsed(monkeypatch, tmp_path):
    _patch_ftd(monkeypatch, tmp_path, content=FTD_TAB)
    out = sec_ownership.get_ftd_data("AAPL", "2024-06-20", 90)
    assert "# Fails-to-Deliver for AAPL" in out
    assert "1234567" in out


@pytest.mark.unit
def test_ftd_no_rows_returns_honest_empty(monkeypatch, tmp_path):
    _patch_ftd(monkeypatch, tmp_path)   # content has AAPL/MSFT only
    out = sec_ownership.get_ftd_data("0700.HK", "2024-06-20", 90)
    assert out.startswith("# Fails-to-Deliver for 0700.HK")
    assert "No fails-to-deliver reported" in out


@pytest.mark.unit
def test_ftd_row_date_gate(monkeypatch, tmp_path):
    # curr_date 2024-06-11: the 2024-05-31 file is public (cutoff 5/31 + 10d =
    # 6/10 <= 6/11), but the 20240612 row postdates curr_date -> dropped, while
    # the 20240610 row is kept. Isolates the row-level Date <= curr_date gate
    # from the file-level publication-lag gate.
    _patch_ftd(monkeypatch, tmp_path)
    out = sec_ownership.get_ftd_data("AAPL", "2024-06-11", 90)
    assert "# Fails-to-Deliver for AAPL" in out
    assert "1234567" in out          # 20240610 <= 2024-06-11 -> kept
    assert "2345678" not in out      # 20240612 > 2024-06-11 -> dropped


# --------------------------------------------------------------------------- #
# Router integration
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_router_routes_form4_via_sec_edgar(monkeypatch, tmp_path):
    _patch_form4(monkeypatch, tmp_path)
    from yiagents.dataflows import config as cfgmod
    from yiagents.dataflows.interface import route_to_vendor

    orig = cfgmod.get_config()
    try:
        cfgmod.set_config({**orig, "data_vendors": {**orig.get("data_vendors", {}),
                                                    "sec_ownership": "sec_edgar"}})
        out = route_to_vendor("get_form4_insider_trading", "AAPL", "2024-06-15", 180)
    finally:
        cfgmod.set_config(orig)
    assert "Form 4 Insider Trading" in out


@pytest.mark.unit
def test_router_optional_category_degrades_to_sentinel(monkeypatch, tmp_path):
    """A non-US ticker -> NoMarketDataError -> the router's NO_DATA_AVAILABLE
    sentinel (the typed "report unavailable" path), not a crash. Crucially the
    optional category never re-raises: a missing-ownership signal can't abort
    the run."""
    monkeypatch.setattr(sec_ownership, "_cik_for_ticker",
                        lambda t: (_ for _ in ()).throw(
                            NoMarketDataError(t, detail="US-listed only")))
    from yiagents.dataflows import config as cfgmod
    from yiagents.dataflows.interface import route_to_vendor

    orig = cfgmod.get_config()
    try:
        cfgmod.set_config({**orig, "data_vendors": {**orig.get("data_vendors", {}),
                                                    "sec_ownership": "sec_edgar"}})
        out = route_to_vendor("get_form4_insider_trading", "0700.HK", "2024-06-15", 180)
    finally:
        cfgmod.set_config(orig)
    assert out.startswith("NO_DATA_AVAILABLE")


# --------------------------------------------------------------------------- #
# Wiring byte-equivalence (mock-LLM)
# --------------------------------------------------------------------------- #
class _BoundLLM(Runnable):
    def invoke(self, inp, config=None, **kwargs):  # noqa: D401, ARG002
        return AIMessage(content="MOCK REPORT", tool_calls=[])


class _RecordingLLM(Runnable):
    def __init__(self):
        super().__init__()
        self.bound_tools = None

    def invoke(self, inp, config=None, **kwargs):  # noqa: D401, ARG002
        return AIMessage(content="", tool_calls=[])

    def bind_tools(self, tools, **kwargs):  # noqa: ARG002
        self.bound_tools = list(tools)
        return _BoundLLM()


def _state():
    return {
        "trade_date": "2024-06-15",
        "company_of_interest": "AAPL",
        "asset_type": "stock",
        "instrument_context": "CTX",
        "messages": [HumanMessage(content="analyze")],
    }


class FundamentalsWiringTests(unittest.TestCase):
    """B2 — default-off byte-equivalence + on-appends-two-tools."""

    def _tool_names(self, config_overrides=None):
        from yiagents.dataflows import config as cfgmod
        orig = cfgmod.get_config()
        try:
            if config_overrides:
                cfgmod.set_config({**orig, **config_overrides})
            llm = _RecordingLLM()
            node = create_fundamentals_analyst(llm)
            node(_state())
            return [t.name for t in llm.bound_tools]
        finally:
            cfgmod.set_config(orig)

    def test_default_off_byte_equivalent_baseline(self):
        # sec_ownership unset/False -> exactly the 4 baseline fundamentals tools.
        names = self._tool_names({"sec_ownership": False})
        self.assertEqual(
            names,
            ["get_fundamentals", "get_balance_sheet", "get_cashflow",
             "get_income_statement"],
        )

    def test_on_appends_two_tools(self):
        names = self._tool_names({"sec_ownership": True})
        self.assertEqual(
            names,
            ["get_fundamentals", "get_balance_sheet", "get_cashflow",
             "get_income_statement", "get_form4_insider_trading", "get_ftd_data"],
        )

    def test_valuation_tools_independent(self):
        # The two flags compose independently; neither perturbs the other.
        names = self._tool_names({"sec_ownership": False, "valuation_tools": True})
        self.assertIn("get_valuation_metrics", names)
        self.assertNotIn("get_form4_insider_trading", names)


if __name__ == "__main__":
    unittest.main()
