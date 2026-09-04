"""Deterministic fundamentals bundle (yialpha.dataflows.fundamentals_bundle)
and the SEC+Yahoo aggregate overview (yialpha.dataflows.fundamentals_overview)
and the ETF fund-data branch (yialpha.dataflows.etf_fund_data).

Pins the contract from the 2026-09 fundamentals review:
* the overview MERGES SEC filing facts with Yahoo real-time valuation
  (either half may degrade; both failing raises NoMarketDataError);
* the bundle deterministically prefetches overview + three quarterly
  statements through the router (the LLM no longer decides WHETHER core
  fundamentals get fetched), REPLACES the statements with the ETF fund
  snapshot for high-confidence funds, skips the snapshot (live-only) on
  historical dates, skips EVERYTHING on pre-listing replays, and lands its
  quality evidence in the run ledger via submit_with_context;
* the fundamentals analyst injects the rendered block as marked untrusted
  evidence on the final USER message when the config is on and stays
  byte-clean when off.
Hermetic: every vendor seam is monkeypatched.
"""

from __future__ import annotations

from datetime import date

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import Runnable

import yialpha.dataflows.etf_fund_data as efd
import yialpha.dataflows.fundamentals_bundle as fb
import yialpha.dataflows.fundamentals_overview as fo
from yialpha.dataflows.errors import NoMarketDataError

_TODAY = date.today().isoformat()


@pytest.fixture(autouse=True)
def _isolated_listing_snapshot():
    """Drop the global equity-perp listing cache around every test.

    The as-of listing gate reads the process-global warmed exchangeInfo
    snapshot; other test files (e.g. test_cli_symbol_handling) warm it with
    a realistic snapshot whose onboard dates are mid-2026. Without this
    isolation, a bundle test replaying a January date would flip into
    skipped_not_listed depending on suite ordering.
    """
    import yialpha.dataflows.binance as bnb

    bnb.refresh_equity_perp_bases()
    yield
    bnb.refresh_equity_perp_bases()

_SEC_OVERVIEW = (
    "# Company Fundamentals for MU\n"
    "# Source: SEC EDGAR XBRL\n"
    "Name: Micron Technology Inc.\n"
    "Total Assets: 130,000,000,000  (period ending 2026-06-05)\n"
    "EPS Diluted (latest): 1.2000  (period ending 2026-06-05)\n"
)

_YAHOO_INFO = {
    "longName": "Micron Technology Inc.",
    "marketCap": 250_000_000_000,
    "trailingPE": 30.5,
    "beta": 1.25,
    "fiftyTwoWeekHigh": 250.0,
}


# ---- aggregate overview -------------------------------------------------------


@pytest.mark.unit
def test_overview_merges_sec_and_yahoo(monkeypatch):
    monkeypatch.setattr(fo, "_sec_fundamentals", lambda t, c: _SEC_OVERVIEW)
    monkeypatch.setattr(fo, "_cached_ticker_info", lambda t, c: dict(_YAHOO_INFO))
    out = fo.get_fundamentals("MU", None)
    assert "## SEC filing facts" in out
    assert "Total Assets: 130,000,000,000" in out
    assert "## Real-time valuation & snapshot (Yahoo Finance, live)" in out
    assert "Market Cap: 250000000000" in out
    assert "Beta: 1.25" in out
    # vendor headers are stripped — the aggregate owns the framing
    assert "# Source: SEC EDGAR" not in out


@pytest.mark.unit
def test_overview_sec_failure_degrades_to_yahoo(monkeypatch):
    def boom(t, c):
        raise NoMarketDataError(t, t, "no cik")

    monkeypatch.setattr(fo, "_sec_fundamentals", boom)
    monkeypatch.setattr(fo, "_cached_ticker_info", lambda t, c: dict(_YAHOO_INFO))
    out = fo.get_fundamentals("MU", None)
    assert "data not available" in out
    assert "Market Cap: 250000000000" in out


@pytest.mark.unit
def test_overview_yahoo_failure_degrades_to_sec(monkeypatch):
    monkeypatch.setattr(fo, "_sec_fundamentals", lambda t, c: _SEC_OVERVIEW)
    monkeypatch.setattr(
        fo, "_cached_ticker_info",
        lambda t, c: (_ for _ in ()).throw(RuntimeError("net down")),
    )
    out = fo.get_fundamentals("MU", None)
    assert "Total Assets" in out
    assert "real-time valuation unavailable" in out


@pytest.mark.unit
def test_overview_both_halves_fail_raises_no_data(monkeypatch):
    def boom(t, c):
        raise NoMarketDataError(t, t, "no cik")

    monkeypatch.setattr(fo, "_sec_fundamentals", boom)
    monkeypatch.setattr(
        fo, "_cached_ticker_info",
        lambda t, c: (_ for _ in ()).throw(RuntimeError("net down")),
    )
    with pytest.raises(NoMarketDataError):
        fo.get_fundamentals("MU", None)


@pytest.mark.unit
def test_overview_historical_date_is_sec_only(monkeypatch):
    monkeypatch.setattr(
        fo, "_sec_fundamentals", lambda t, c: _SEC_OVERVIEW,
    )
    # The Yahoo half must not even be attempted (PIT): the leak guard fires
    # before any info fetch.
    def no_fetch(t, c):
        raise AssertionError("Yahoo info fetched on a historical date")

    monkeypatch.setattr(fo, "_cached_ticker_info", no_fetch)
    out = fo.get_fundamentals("MU", "2026-01-15")
    assert "Total Assets" in out
    assert "point-in-time guard" in out
    assert "Market Cap" not in out


# ---- ETF fund data ------------------------------------------------------------


@pytest.mark.unit
def test_etf_fund_data_fields_and_holdings(monkeypatch):
    info = {
        "longName": "SPDR S&P 500 ETF Trust",
        "category": "Large Blend",
        "navPrice": 555.0,
        "totalAssets": 600_000_000_000,
        "annualReportExpenseRatio": 0.0009,
        "currentPrice": 556.1,
    }
    monkeypatch.setattr(efd, "_cached_ticker_info", lambda t, c: info)
    monkeypatch.setattr(efd, "_top_holdings", lambda t: "symbol,holding_name,weight_pct\nAAPL,Apple Inc.,6.5")
    out = efd.get_etf_fund_data("SPY", None)
    assert "NAV (per share): 555.0" in out
    assert "Total Assets (AUM): $600,000,000,000" in out
    assert "Annual Expense Ratio: 0.0900%" in out
    # premium/discount derives from price vs NAV: (556.1/555 - 1)*1e4 ≈ +19.8 bps
    assert "Premium/Discount vs NAV: +19.8 bps" in out
    assert "AAPL,Apple Inc.,6.5" in out


@pytest.mark.unit
def test_etf_fund_data_holdings_absent_disclosed(monkeypatch):
    monkeypatch.setattr(
        efd, "_cached_ticker_info",
        lambda t, c: {"navPrice": 555.0},
    )
    monkeypatch.setattr(efd, "_top_holdings", lambda t: None)
    out = efd.get_etf_fund_data("SPY", None)
    assert "Top holdings: unavailable" in out


@pytest.mark.unit
def test_etf_fund_data_historical_refused(monkeypatch):
    def no_fetch(t, c):
        raise AssertionError("fund info fetched on a historical date")

    monkeypatch.setattr(efd, "_cached_ticker_info", no_fetch)
    with pytest.raises(NoMarketDataError):
        efd.get_etf_fund_data("SPY", "2026-01-15")


@pytest.mark.unit
def test_etf_fund_data_empty_info_no_data(monkeypatch):
    monkeypatch.setattr(efd, "_cached_ticker_info", lambda t, c: {})
    with pytest.raises(NoMarketDataError):
        efd.get_etf_fund_data("SPY", None)


# ---- bundle -------------------------------------------------------------------


def _ok(payload):
    return {"status": fb.STATUS_OK, "payload": payload}


def _unavail(reason="down"):
    return {"status": fb.STATUS_UNAVAILABLE, "reason": reason}


@pytest.mark.unit
def test_bundle_routes_all_four_core_calls_for_underlying(monkeypatch):
    calls = []

    def fake_route(method, *args):
        calls.append((method, args))
        return f"# {method}\nrow1"

    monkeypatch.setattr(fb, "route_to_vendor", fake_route)
    bundle = fb.fetch_fundamentals_bundle("crypto_perp", "MUUSDT", _TODAY)
    methods = sorted(m for m, _ in calls)
    # Order-insensitive on purpose: the bundle dispatches these four through a
    # ThreadPoolExecutor, so arrival order is a thread race, not a contract
    # (flaked on CI py3.13). Sorted lists, not a set, so duplicates still fail.
    assert methods == sorted([
        "get_fundamentals", "get_income_statement",
        "get_balance_sheet", "get_cashflow",
    ])
    # every call addresses the UNDERLYING equity, not the perp symbol
    for _m, args in calls:
        assert args[0] == "MU"
    assert bundle["fundamentals_symbol"] == "MU"
    assert bundle["overview"]["status"] == fb.STATUS_OK
    assert bundle["income_statement"]["payload"] == "# get_income_statement\nrow1"


@pytest.mark.unit
def test_bundle_etf_replaces_statements_with_fund_snapshot(monkeypatch):
    # ETF REPLACE routing: a fund has no operating-company statements, so the
    # bundle must NOT fetch income/balance/cashflow at all — their expected
    # failures could otherwise wrongly grade a well-covered ETF run as
    # degraded. The fund snapshot + overview are the core evidence.
    calls = []

    def fake_route(method, *args):
        calls.append(method)
        return "ok"

    monkeypatch.setattr(fb, "route_to_vendor", fake_route)
    monkeypatch.setattr(
        fb, "_fetch_etf_fund_data",
        lambda t, c: _ok("NAV (per share): 555.0"),
    )
    bundle = fb.fetch_fundamentals_bundle("crypto_perp", "SPYUSDT", _TODAY)
    assert bundle["is_etf"] is True
    assert bundle["etf_fund_data"]["status"] == fb.STATUS_OK
    assert calls == ["get_fundamentals"]  # overview only — no statements
    block = fb.render_fundamentals_bundle_block(bundle)
    assert "ETF fund data" in block
    assert "Income statement" not in block
    assert (
        "not applicable (ETF — the fund snapshot above replaces "
        "income/balance/cashflow)" in block
    )


@pytest.mark.unit
def test_bundle_pre_listing_replay_skips_all_vendor_calls(monkeypatch):
    # as-of wiring: a replay dated BEFORE the contract's onboard date (with
    # listing evidence present) must make ZERO vendor calls — reading the
    # underlying's live fundamentals on a pre-listing date is a PIT violation
    # dressed as evidence. Every component carries the skip + the descriptor
    # disclosure.
    import yialpha.dataflows.binance as bnb
    import yialpha.graph.routing as routing

    monkeypatch.setattr(
        bnb, "equity_perp_listing_info",
        lambda: {"MU": {"onboard_date": "2026-03-10", "status": "TRADING"}},
    )
    monkeypatch.setattr(
        routing, "equity_perp_listing_info",
        lambda: {"MU": {"onboard_date": "2026-03-10", "status": "TRADING"}},
    )

    def no_route(*_a, **_k):  # noqa: ARG001
        raise AssertionError("vendor routed on a pre-listing replay")

    monkeypatch.setattr(fb, "route_to_vendor", no_route)
    bundle = fb.fetch_fundamentals_bundle("crypto_perp", "MUUSDT", "2026-01-15")
    for key in (
        "overview", "income_statement", "balance_sheet", "cashflow",
        "etf_fund_data",
    ):
        assert bundle[key]["status"] == fb.STATUS_SKIPPED_NOT_LISTED, key
    assert "NOT yet listed" in bundle["overview"]["reason"]
    block = fb.render_fundamentals_bundle_block(bundle)
    assert "NOT yet listed" in block
    assert "skipped_not_listed" in block


@pytest.mark.unit
def test_bundle_listing_disclosure_rides_post_listing_replay(monkeypatch):
    # Same evidence, replay AFTER onboard: the bundle fetches normally and
    # the rendered block carries the as-of listing verdict (verifiable, not
    # silent).
    import yialpha.dataflows.binance as bnb
    import yialpha.graph.routing as routing

    listing = {"MU": {"onboard_date": "2026-03-10", "status": "TRADING"}}
    monkeypatch.setattr(bnb, "equity_perp_listing_info", lambda: listing)
    monkeypatch.setattr(routing, "equity_perp_listing_info", lambda: listing)
    monkeypatch.setattr(fb, "route_to_vendor", lambda m, *a: "ok")
    bundle = fb.fetch_fundamentals_bundle("crypto_perp", "MUUSDT", "2026-08-01")
    assert bundle["overview"]["status"] == fb.STATUS_OK
    block = fb.render_fundamentals_bundle_block(bundle)
    assert "**Listing**: listing as of 2026-08-01: listed" in block


@pytest.mark.unit
def test_bundle_core_failures_reach_quality_ledger(monkeypatch):
    # The ContextVar contract: bundle fetches run inside submit_with_context
    # workers, so core failures/successes land in the RUN's ledger (a plain
    # pool.submit binds fresh worker-local lists and the evidence vanishes —
    # four failed components used to read as a clean quality state).
    from yialpha.dataflows import interface, quality
    from yialpha.dataflows.config import get_config, set_config

    methods = (
        "get_fundamentals", "get_income_statement",
        "get_balance_sheet", "get_cashflow",
    )

    def boom(*_a, **_k):
        raise RuntimeError("vendor down")

    for method in methods:
        monkeypatch.setitem(interface.VENDOR_METHODS, method, {"mockfail": boom})
    orig = get_config()
    try:
        set_config({
            **orig,
            "tool_vendors": {
                **orig.get("tool_vendors", {}),
                **dict.fromkeys(methods, "mockfail"),
            },
        })
        quality.ensure_run_context()
        bundle = fb.fetch_fundamentals_bundle("stock", "MU", _TODAY)
        events = quality.snapshot_quality()
        successes = quality.snapshot_core_successes()
    finally:
        set_config(orig)
        quality.reset_quality()

    assert bundle["overview"]["status"] == fb.STATUS_UNAVAILABLE
    core = [e for e in events if e["method"] in methods]
    assert len(core) == 4  # every failed core component left its sentinel
    assert all(e["kind"] == quality.KIND_CORE_ERROR for e in core)
    assert successes == set()  # nothing falsely recorded as a core success


@pytest.mark.unit
def test_bundle_etf_snapshot_skipped_on_historical(monkeypatch):
    monkeypatch.setattr(fb, "route_to_vendor", lambda m, *a: "ok")
    # No stub on _fetch_etf_fund_data: the REAL helper must apply the PIT
    # gate itself (past date -> skipped before any vendor call).
    bundle = fb.fetch_fundamentals_bundle("crypto_perp", "SPYUSDT", "2026-01-15")
    assert bundle["etf_fund_data"]["status"] == fb.STATUS_SKIPPED_LIVE_ONLY


@pytest.mark.unit
def test_bundle_sentinel_string_is_unavailable(monkeypatch):
    monkeypatch.setattr(
        fb, "route_to_vendor",
        lambda m, *a: "NO_DATA_AVAILABLE: no usable market data for 'ZZZZ'",
    )
    bundle = fb.fetch_fundamentals_bundle("stock", "ZZZZ", _TODAY)
    assert bundle["overview"]["status"] == fb.STATUS_UNAVAILABLE
    assert "NO_DATA_AVAILABLE" in bundle["overview"]["reason"]


@pytest.mark.unit
def test_bundle_renderer_sections_and_footer(monkeypatch):
    monkeypatch.setattr(fb, "route_to_vendor", lambda m, *a: "ok")
    bundle = {
        "symbol": "MUUSDT",
        "fundamentals_symbol": "MU",
        "as_of": _TODAY,
        "instrument_class": "stock_perp",
        "is_etf": False,
        "overview": _ok("Total Assets: 1"),
        "income_statement": _ok("revenue rows"),
        "balance_sheet": _unavail("no data"),
        "cashflow": _ok("cash rows"),
    }
    block = fb.render_fundamentals_bundle_block(bundle)
    assert "### Fundamentals Bundle — MUUSDT" in block
    assert "**Underlying equity**: MU" in block
    assert "revenue rows" in block
    assert "balance_sheet: unavailable (no data)" in block


@pytest.mark.unit
def test_bundle_renderer_truncates_long_payloads():
    block = fb.render_fundamentals_bundle_block({
        "symbol": "MU", "fundamentals_symbol": "MU", "as_of": _TODAY,
        "income_statement": _ok("x" * 10_000),
    })
    assert "truncated — call the bound tool" in block
    assert len(block) < 6_000


# ---- analyst injection --------------------------------------------------------


class _PromptCaptureLLM(Runnable):
    def __init__(self):
        super().__init__()
        self.prompt = None

    def invoke(self, inp, config=None, **kwargs):  # noqa: ARG002
        self.prompt = inp
        return AIMessage(content="MOCK REPORT", tool_calls=[])

    def bind_tools(self, tools, **kwargs):  # noqa: ARG002
        return self


def _fundamentals_state():
    return {
        "trade_date": _TODAY,
        "company_of_interest": "MU",
        "asset_type": "stock",
        "instrument_context": "CTX",
        "messages": [HumanMessage(content="analyze")],
    }


@pytest.mark.unit
def test_fundamentals_analyst_injects_bundle_when_enabled(monkeypatch):
    from yialpha.agents.analysts.fundamentals_analyst import (
        create_fundamentals_analyst,
    )
    from yialpha.dataflows.config import set_config

    set_config({"fundamentals_bundle": True})
    monkeypatch.setattr(
        fb, "fetch_fundamentals_bundle",
        lambda at, t, d: {"symbol": t, "fundamentals_symbol": t, "as_of": d},
    )
    monkeypatch.setattr(
        fb, "render_fundamentals_bundle_block", lambda b: "SENTINEL_BLOCK",
    )

    llm = _PromptCaptureLLM()
    create_fundamentals_analyst(llm)(_fundamentals_state())
    rendered = str(llm.prompt)
    assert "SENTINEL_BLOCK" in rendered
    assert "Advisory deterministic prefetch" in rendered
    # Evidence posture (news/sentiment parity): the vendor block rides the
    # final USER message as explicitly marked untrusted evidence — the
    # system message never carries raw vendor output.
    messages = llm.prompt.to_messages()
    assert messages[-1].type == "human"
    assert "[EXTERNAL EVIDENCE — untrusted third-party content]" in messages[-1].content
    assert "<start_of_fundamentals_bundle>" in messages[-1].content
    system = next(m for m in messages if m.type == "system")
    assert "SENTINEL_BLOCK" not in system.content


@pytest.mark.unit
def test_fundamentals_analyst_bundle_off_by_default():
    from yialpha.agents.analysts.fundamentals_analyst import (
        create_fundamentals_analyst,
    )

    # conftest's _runtime_prefetch_bundles_off fixture holds the config
    # off; the prompt must not carry a bundle section.
    llm = _PromptCaptureLLM()
    create_fundamentals_analyst(llm)(_fundamentals_state())
    assert "Fundamentals Bundle" not in str(llm.prompt)


@pytest.mark.unit
def test_fundamentals_analyst_bundle_fetch_failure_never_blocks(monkeypatch):
    from yialpha.agents.analysts.fundamentals_analyst import (
        create_fundamentals_analyst,
    )
    from yialpha.dataflows.config import set_config

    set_config({"fundamentals_bundle": True})

    def boom(at, t, d):
        raise RuntimeError("vendor outage")

    monkeypatch.setattr(fb, "fetch_fundamentals_bundle", boom)
    llm = _PromptCaptureLLM()
    out = create_fundamentals_analyst(llm)(_fundamentals_state())
    assert "fundamentals_report" in out  # node completed tools-only
    assert "Fundamentals Bundle" not in str(llm.prompt)


@pytest.mark.unit
def test_default_chain_dispatches_overview_to_the_aggregate(monkeypatch):
    """The default tool_vendors override must actually reach the merge
    implementation through the router (not just resolve as a string)."""
    import yialpha.dataflows.interface as iface

    seen = []

    def fake_aggregate(ticker, curr_date=None):
        seen.append((ticker, curr_date))
        return "AGGREGATE OVERVIEW"

    monkeypatch.setitem(
        iface.VENDOR_METHODS["get_fundamentals"],
        "fundamentals_overview", fake_aggregate,
    )
    out = iface.route_to_vendor("get_fundamentals", "MU", None)
    assert out == "AGGREGATE OVERVIEW"
    assert seen == [("MU", None)]
