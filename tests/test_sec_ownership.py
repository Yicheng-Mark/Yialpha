"""Unit tests for ``yiagents.dataflows.sec_ownership`` (Form 4 + FTD) and the
fundamentals-analyst wiring (Track B2).

Hermetic: no network. The vendor paths are exercised by patching
``sec_ownership._cached_or_fetch`` to serve synthetic JSON/XML/text payloads and
``sec_ownership._cik_for_ticker`` to a stub. PIT filtering (Form 4 filingDate;
FTD cutoff + publication lag), header-driven FTD parsing, the non-US/empty
contracts, and router integration round out the dataflow coverage.

The wiring tests pin the byte-equivalence contract (the project "iron rule"):
with ``sec_ownership`` off (the default) the fundamentals analyst binds exactly
its 4 baseline tools and an unchanged prompt; with it on the three new tools are
appended in order. Pure mock-LLM, zero network, zero LLM cost.
"""

from __future__ import annotations

import io
import unittest
import zipfile

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

    def test_on_appends_three_tools(self):
        names = self._tool_names({"sec_ownership": True})
        self.assertEqual(
            names,
            ["get_fundamentals", "get_balance_sheet", "get_cashflow",
             "get_income_statement", "get_form4_insider_trading", "get_ftd_data",
             "get_institutional_holdings"],
        )

    def test_valuation_tools_independent(self):
        # The two flags compose independently; neither perturbs the other.
        names = self._tool_names({"sec_ownership": False, "valuation_tools": True})
        self.assertIn("get_valuation_metrics", names)
        self.assertNotIn("get_form4_insider_trading", names)


# --------------------------------------------------------------------------- #
# 13F institutional holdings (Track B2.1)
# --------------------------------------------------------------------------- #
COVER_HEADER = (
    "ACCESSION_NUMBER\tFILER_CIK\tFILING_DATE\tFILING_MANAGER_NAME\t"
    "REPORT_CALENDAR_OR_QUARTER"
)
HOLDING_HEADER = (
    "ACCESSION_NUMBER\tNAME_OF_ISSUER\tCUSIP\tTITLE_OF_CLASS\tVALUE\t"
    "SSH_PRNAMT\tSSH_PRNAMT_TYPE\tPUT_CALL\tINVESTMENT_DISCRETION\t"
    "VOTING_AUTH_SOLE\tVOTING_AUTH_SHARED\tVOTING_AUTH_NONE"
)
COMPANYFACTS_AAPL = {
    "dei": {"EntityCusip": {"units": {"NONE": [
        {"end": "2023-12-31", "filed": "2024-02-02", "val": "037833100"},
        {"end": "2024-03-31", "filed": "2024-04-26", "val": "037833100"},
    ]}}}
}


def _13f_zip(cover_tsv: str, holding_tsv: str) -> bytes:
    """Build a synthetic bulk-13F ZIP from TSV strings (cover + holding)."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("filing_cover.tsv", cover_tsv.encode("utf-8"))
        zf.writestr("filing_holding.tsv", holding_tsv.encode("utf-8"))
    return buf.getvalue()


def _patch_13f(monkeypatch, tmp_path, zip_bytes, facts=None):
    """Point sec_ownership at a synthetic CIK + companyfacts + 13F ZIP.

    Records requested URLs so a PIT test can assert a not-yet-published dataset
    was never fetched."""
    monkeypatch.setattr(sec_ownership, "_cache_dir", lambda: str(tmp_path))
    monkeypatch.setattr(sec_ownership, "_cik_for_ticker", lambda t: 320193)
    monkeypatch.setattr(
        sec_ownership, "_fetch_company_facts",
        lambda cik: facts if facts is not None else COMPANYFACTS_AAPL)
    requested = []

    def fake_fetch(_path, url, ttl_days):
        requested.append(url)
        if "form13f.zip" in url:
            return zip_bytes
        raise NoMarketDataError(url, detail="404")

    monkeypatch.setattr(sec_ownership, "_cached_or_fetch", fake_fetch)
    return requested


@pytest.mark.unit
def test_normalize_cusip():
    assert sec_ownership._normalize_cusip("037-833-100") == "037833100"
    assert sec_ownership._normalize_cusip(" 037833100 ") == "037833100"
    assert sec_ownership._normalize_cusip("") == ""
    assert sec_ownership._normalize_cusip("037833100EXTRA") == "037833100"


@pytest.mark.unit
def test_13f_basic_aggregation(monkeypatch, tmp_path):
    cover = COVER_HEADER + "\n" + (
        "00001A\t0001067983\t2024-05-15\tBERKSHIRE HATHAWAY INC\t2024-03-31\n"
        "00001B\t0001357790\t2024-05-14\tVANGUARD GROUP INC\t2024-03-31"
    )
    holding = HOLDING_HEADER + "\n" + (
        "00001A\tAPPLE INC\t037833100\tCOM\t150000000\t780000000\tSH\t\tSOLE\t780000000\t0\t0\n"
        "00001B\tAPPLE INC\t037833100\tCOM\t90000000\t470000000\tSH\t\tSOLE\t470000000\t0\t0\n"
        "00001C\tMICROSOFT CORP\t594918104\tCOM\t9999\t10\tSH\t\tSOLE\t10\t0\t0"
    )
    requested = _patch_13f(monkeypatch, tmp_path, _13f_zip(cover, holding))
    out = sec_ownership.get_institutional_holdings("AAPL", "2024-06-15", 180)
    # Most-recent PIT-visible dataset is the Mar-May 2024 window (report Q1).
    assert "01mar2024-31may2024_form13f.zip" in requested[0]
    assert "2024 Q1" in out
    assert "BERKSHIRE HATHAWAY" in out
    assert "VANGUARD" in out
    assert "MICROSOFT" not in out          # wrong CUSIP filtered out
    assert out.index("BERKSHIRE") < out.index("VANGUARD")  # value desc


@pytest.mark.unit
def test_13f_pit_quarter_not_yet_published(monkeypatch, tmp_path):
    requested = _patch_13f(monkeypatch, tmp_path, _13f_zip(COVER_HEADER, HOLDING_HEADER))
    # curr_date 2024-03-01, look-back 10d -> visible_end 2024-02-24, lower
    # 2024-02-20. No Feb/May/Aug/Nov month-end lies in [2024-02-20, 2024-02-24]
    # (2024-02-29 is after it), so no dataset is PIT-visible.
    out = sec_ownership.get_institutional_holdings("AAPL", "2024-03-01", 10)
    assert "No bulk 13F data set was public" in out
    assert requested == []                 # the ZIP was never fetched


@pytest.mark.unit
def test_13f_cusip_missing_degrades(monkeypatch, tmp_path):
    _patch_13f(monkeypatch, tmp_path, _13f_zip(COVER_HEADER, HOLDING_HEADER),
               facts={"dei": {}})
    with pytest.raises(NoMarketDataError):
        sec_ownership.get_institutional_holdings("AAPL", "2024-06-15", 180)


@pytest.mark.unit
def test_13f_non_us_raises_no_market_data(monkeypatch, tmp_path):
    monkeypatch.setattr(sec_ownership, "_cik_for_ticker",
                        lambda t: (_ for _ in ()).throw(
                            NoMarketDataError(t, detail="US-listed only")))
    with pytest.raises(NoMarketDataError):
        sec_ownership.get_institutional_holdings("0700.HK", "2024-06-15", 180)


@pytest.mark.unit
def test_13f_denoise_skips_options_and_prn(monkeypatch, tmp_path):
    cover = COVER_HEADER + "\n" + "00002A\t0000914201\t2024-05-15\tBLACKROCK INC\t2024-03-31"
    holding = HOLDING_HEADER + "\n" + (
        # Common-stock holding: 1000 shares, value 100000 ($000s) = $100M.
        "00002A\tAPPLE INC\t037833100\tCOM\t100000\t1000\tSH\t\tSOLE\t1000\t0\t0\n"
        # Same holder, AAPL CALL option -> dropped (non-empty PUT_CALL).
        "00002A\tAPPLE INC\t037833100\tCOM\t50000\t500\tSH\tCALL\tSOLE\t500\t0\t0\n"
        # Same holder, principal/bond (PRN) -> dropped.
        "00002A\tAPPLE INC\t037833100\tCOM\t20000\t200\tPRN\t\tSOLE\t200\t0\t0"
    )
    _patch_13f(monkeypatch, tmp_path, _13f_zip(cover, holding))
    out = sec_ownership.get_institutional_holdings("AAPL", "2024-06-15", 180)
    assert "BLACKROCK" in out
    assert "1,000" in out       # only the SH row's shares survived
    assert "100.00" in out      # value $100M = 100000 * 1000 / 1e6


@pytest.mark.unit
def test_13f_row_pit_filing_date_gate(monkeypatch, tmp_path):
    cover = COVER_HEADER + "\n" + (
        "00003A\t0001\t2024-05-15\tEARLY CAPITAL\t2024-03-31\n"
        "00003B\t0002\t2024-08-15\tLATE CAPITAL\t2024-06-30"
    )
    holding = HOLDING_HEADER + "\n" + (
        "00003A\tAPPLE INC\t037833100\tCOM\t10000\t100\tSH\t\tSOLE\t100\t0\t0\n"
        "00003B\tAPPLE INC\t037833100\tCOM\t99999\t999\tSH\t\tSOLE\t999\t0\t0"
    )
    _patch_13f(monkeypatch, tmp_path, _13f_zip(cover, holding))
    out = sec_ownership.get_institutional_holdings("AAPL", "2024-06-15", 180)
    assert "EARLY CAPITAL" in out
    assert "LATE CAPITAL" not in out      # FILING_DATE 2024-08-15 > curr_date


@pytest.mark.unit
def test_13f_no_holders_honest_empty(monkeypatch, tmp_path):
    holding = HOLDING_HEADER + "\n" + (
        "00004A\tMICROSOFT CORP\t594918104\tCOM\t9999\t10\tSH\t\tSOLE\t10\t0\t0"
    )
    _patch_13f(monkeypatch, tmp_path, _13f_zip(COVER_HEADER, holding))
    out = sec_ownership.get_institutional_holdings("AAPL", "2024-06-15", 180)
    assert "No 13F institutional holders reported owning AAPL" in out


@pytest.mark.unit
def test_router_routes_13f_via_sec_edgar(monkeypatch, tmp_path):
    cover = COVER_HEADER + "\n" + "00005A\t0001\t2024-05-15\tTEST HOLDER\t2024-03-31"
    holding = HOLDING_HEADER + "\n" + (
        "00005A\tAPPLE INC\t037833100\tCOM\t1000\t10\tSH\t\tSOLE\t10\t0\t0"
    )
    _patch_13f(monkeypatch, tmp_path, _13f_zip(cover, holding))
    from yiagents.dataflows import config as cfgmod
    from yiagents.dataflows.interface import route_to_vendor

    orig = cfgmod.get_config()
    try:
        cfgmod.set_config({**orig, "data_vendors": {**orig.get("data_vendors", {}),
                                                    "sec_ownership": "sec_edgar"}})
        out = route_to_vendor("get_institutional_holdings", "AAPL", "2024-06-15", 180)
    finally:
        cfgmod.set_config(orig)
    assert "13F Institutional Holdings" in out
    assert "TEST HOLDER" in out


@pytest.mark.unit
def test_13f_router_optional_category_degrades_to_sentinel(monkeypatch, tmp_path):
    monkeypatch.setattr(sec_ownership, "_cik_for_ticker",
                        lambda t: (_ for _ in ()).throw(
                            NoMarketDataError(t, detail="US-listed only")))
    from yiagents.dataflows import config as cfgmod
    from yiagents.dataflows.interface import route_to_vendor

    orig = cfgmod.get_config()
    try:
        cfgmod.set_config({**orig, "data_vendors": {**orig.get("data_vendors", {}),
                                                    "sec_ownership": "sec_edgar"}})
        out = route_to_vendor("get_institutional_holdings", "0700.HK", "2024-06-15", 180)
    finally:
        cfgmod.set_config(orig)
    assert out.startswith("NO_DATA_AVAILABLE")


if __name__ == "__main__":
    unittest.main()
