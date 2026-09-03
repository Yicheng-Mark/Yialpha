"""Tokenized US-equity perp news: deterministic dual-angle prefetch.

Binance tokenized-stock perp runs (e.g. MUUSDT -> Micron) pre-fetch BOTH news
angles in code — the UNDERLYING equity via the news vendor chain (query
"MU", date-bounded) and the perp CONTRACT via an exact-symbol open-web
search (PR5: the stock-news vendors cannot answer a "MUUSDT" query, so the
old vendor-chain perp angle returned "no news" by construction). These
tests pin:

  * the gate: blocks fire ONLY on crypto_perp runs with an equity underlying
    (stock runs and pure-crypto perps append nothing, byte-unchanged),
  * the company query and the 7-day window anchored on trade_date,
  * per-angle fault isolation (a company-angle failure never kills the
    node or the contract angle),
  * the historical contract: the company angle is date-bounded and PIT-safe,
    while the contract angle degrades to an honest unavailable placeholder
    (today's web must not leak into a replay),
  * the evidence-message injection posture (blocks in the final USER
    message, never the system prompt),
  * the sentiment-side requirement shipped with this change: an equity perp
    run queries Binance Square with the PERP symbol (the existing crypto
    parametrize in test_sentiment_fetch.py covers pure-crypto perps),
  * the sentiment-side news-leg fix (D8): an equity perp run hands the news
    vendor chain the UNDERLYING equity ticker — the vendor chain cannot
    answer the contract symbol — on both the sequential and parallel paths,
  * the sentiment-side evidence-identity fix (D8 follow-up): the
    ``sentiment_news`` evidence row carries that same UNDERLYING identity
    (symbol=MU / scope=UNDERLYING), not the contract symbol — while the
    contract-side legs (Reddit, Binance Square) and a pure-crypto perp's
    news row keep the contract identity.

Everything runs with ZERO network: the symbol resolver is cache-or-seed by
design, and every fetch seam is mocked.
"""

import contextlib
import unittest
from unittest import mock

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import Runnable

import yialpha.agents.analysts.news_analyst as news
import yialpha.agents.analysts.sentiment_analyst as sent
import yialpha.agents.utils.news_data_tools as news_tools
from yialpha.agents.analysts.news_analyst import create_news_analyst
from yialpha.dataflows import config as cfgmod
from yialpha.ledger.models import SCOPE_CONTRACT, SCOPE_UNDERLYING


class _PromptCaptureLLM(Runnable):
    def __init__(self):
        super().__init__()
        self.prompt = None

    def invoke(self, inp, config=None, **kwargs):  # noqa: ARG002
        self.prompt = inp
        return AIMessage(content="MOCK REPORT", tool_calls=[])

    def bind_tools(self, tools, **kwargs):  # noqa: ARG002
        return self


def _state(ticker, asset_type="crypto_perp", trade_date="2026-08-17"):
    return {
        "trade_date": trade_date,
        "company_of_interest": ticker,
        "asset_type": asset_type,
        "instrument_context": "CTX",
        "messages": [HumanMessage(content="analyze")],
    }


def _run_node(state, company="COMPANY NEWS", perp="PERP NEWS", historical=False,
              patch_contract=True, web_search_enabled=True):
    """Invoke the news analyst node with both angle fetchers mocked.

    The company angle is mocked at ``_get_news_impl`` (the vendor-chain
    accessor); the contract angle at ``_fetch_perp_contract_news`` — PR5
    split the two angles onto different sources (vendor chain vs
    exact-symbol web search), so they mock at their own seams.
    ``patch_contract=False`` keeps the REAL contract fetcher, whose
    historical branch returns the unavailable placeholder without any
    network. Returns (company_impl_calls, rendered_prompt). Config is
    snapshotted around the run; the contract angle honours
    ``web_search_enabled`` exactly like the bound tool, so runs that want
    the mocked contract content keep the flag ON (the mock itself prevents
    any network), and gate tests turn it OFF explicitly.
    """
    calls = []

    def fake_impl(ticker, start_date, end_date):
        calls.append((ticker, start_date, end_date))
        return company

    orig = cfgmod.get_config()
    try:
        cfgmod.set_config({**orig, "web_search_enabled": web_search_enabled})
        with mock.patch.object(
            news, "is_historical_date", lambda _d: historical
        ), mock.patch.object(news, "_get_news_impl", side_effect=fake_impl):
            if patch_contract:
                with mock.patch.object(
                    news, "_fetch_perp_contract_news", lambda t, d: perp
                ):
                    llm = _PromptCaptureLLM()
                    node = create_news_analyst(llm)
                    node(state)
            else:
                llm = _PromptCaptureLLM()
                node = create_news_analyst(llm)
                node(state)
        return calls, str(llm.prompt)
    finally:
        cfgmod.set_config(orig)


def _descape(prompt_repr: str) -> str:
    """Un-escape quote chars that str(message-list) reprs escape.

    Which quotes get escaped depends on each string element's own quote mix
    (repr picks its delimiter per element), so assertions containing quotes
    run against a de-escaped copy.
    """
    return prompt_repr.replace('\\"', '"').replace("\\'", "'")


class NewsPrefetchGateTests(unittest.TestCase):
    """The dual-angle prefetch fires ONLY on an equity-perp run."""

    def test_equity_perp_prefetches_both_angles(self):
        # PR5: the company angle goes through the vendor-chain accessor; the
        # CONTRACT angle no longer does (it is an exact-symbol web search —
        # the stock-news vendors cannot answer a MUUSDT query).
        calls, prompt = _run_node(_state("MUUSDT"))
        self.assertEqual(calls, [("MU", "2026-08-10", "2026-08-17")])
        self.assertIn('<start_of_company_news> (query: "MU"', prompt)
        self.assertIn("COMPANY NEWS", prompt)
        self.assertIn('<start_of_perp_news> (query: "MUUSDT"', prompt)
        self.assertIn("PERP NEWS", prompt)
        self.assertIn("MUST cover BOTH angles", prompt)
        # Injection posture (PR4/PR5): the blocks ride an evidence message.
        self.assertIn("[EXTERNAL EVIDENCE", prompt)

    def test_pure_crypto_perp_appends_nothing(self):
        calls, prompt = _run_node(_state("BTCUSDT"))
        self.assertEqual(calls, [])
        self.assertNotIn("Pre-fetched news", prompt)
        self.assertNotIn("start_of_company_news", prompt)
        self.assertNotIn("start_of_perp_news", prompt)
        self.assertNotIn("[EXTERNAL EVIDENCE", prompt)

    def test_stock_run_with_perp_shaped_ticker_appends_nothing(self):
        calls, prompt = _run_node(_state("MUUSDT", asset_type="stock"))
        self.assertEqual(calls, [])
        self.assertNotIn("Pre-fetched news", prompt)

    def test_window_anchors_on_trade_date(self):
        calls, _ = _run_node(_state("NVDAUSDT", trade_date="2026-06-30"))
        self.assertEqual(calls, [("NVDA", "2026-06-23", "2026-06-30")])

    def test_web_search_disabled_disables_contract_prefetch(self):
        # web_search_enabled=False must gate the automatic contract-angle
        # prefetch exactly like the bound web_search tool — an explicit
        # unavailable placeholder, no budget charge, and the (mocked) fetcher
        # never invoked. The company angle is a vendor-chain call, NOT web
        # search, so it still prefetches.
        with mock.patch.object(
            news, "_fetch_perp_contract_news",
            mock.MagicMock(return_value="SHOULD NOT APPEAR"),
        ) as contract_fetch, mock.patch.object(
            news, "is_historical_date", lambda _d: False,
        ):
            llm = _PromptCaptureLLM()
            orig = cfgmod.get_config()
            try:
                cfgmod.set_config({**orig, "web_search_enabled": False})
                create_news_analyst(llm)(_state("MUUSDT"))
            finally:
                cfgmod.set_config(orig)
        prompt = str(llm.prompt)
        self.assertIn(
            "<contract-angle news unavailable: web search disabled", prompt,
        )
        self.assertNotIn("SHOULD NOT APPEAR", prompt)
        contract_fetch.assert_not_called()


class NewsPrefetchContentTests(unittest.TestCase):
    """Block contents land verbatim with honest no-coverage wording."""

    def test_empty_blocks_injected_verbatim_with_instruction(self):
        calls, prompt = _run_node(
            _state("MUUSDT"),
            company="no coverage found",
            perp="WEB_SEARCH_UNAVAILABLE: no key",
        )
        self.assertEqual(len(calls), 1)
        self.assertIn("no coverage found", prompt)
        self.assertIn("WEB_SEARCH_UNAVAILABLE", prompt)
        # The per-angle honesty rule is static instruction text.
        # (Assertion fragment avoids apostrophes: str(prompt) is a
        # message-list repr, which escapes embedded quotes.)
        self.assertIn("state that explicitly for that angle", prompt)

    def test_historical_company_angle_still_prefetched_contract_placeholder(self):
        # get_news carries explicit date bounds, so the COMPANY angle is
        # PIT-safe on replay dates and must NOT be gated off. The CONTRACT
        # angle has no as-of-capable source: the REAL historical branch
        # returns an unavailable placeholder instead of today's web.
        calls, prompt = _run_node(
            _state("MUUSDT"), historical=True, patch_contract=False
        )
        prompt = _descape(prompt)
        self.assertEqual(calls, [("MU", "2026-08-10", "2026-08-17")])
        self.assertIn(
            '<start_of_company_news> (query: "MU"; source: news vendor chain',
            prompt,
        )
        self.assertIn("no as-of boundary", prompt)
        self.assertIn("historical analysis", prompt)

    def test_company_angle_failure_is_isolated(self):
        # A hard company-angle failure degrades to a per-angle placeholder —
        # the node (and the contract angle) must survive it.
        state = _state("MUUSDT")
        orig = cfgmod.get_config()
        try:
            cfgmod.set_config({**orig, "web_search_enabled": True})
            with (
                mock.patch.object(news, "is_historical_date", lambda _d: False),
                mock.patch.object(
                    news, "_get_news_impl",
                    side_effect=RuntimeError("vendor chain down"),
                ),
                mock.patch.object(
                    news, "_fetch_perp_contract_news", lambda t, d: "PERP NEWS",
                ),
            ):
                llm = _PromptCaptureLLM()
                node = create_news_analyst(llm)
                out = node(state)  # must not raise
        finally:
            cfgmod.set_config(orig)
        prompt = _descape(str(llm.prompt))
        self.assertIn("<company-angle news unavailable: RuntimeError>", prompt)
        self.assertIn("PERP NEWS", prompt)
        self.assertIn("MOCK REPORT", out["news_report"])

    def test_contract_angle_failure_is_isolated(self):
        # Symmetric isolation: an unexpected exception on the CONTRACT angle's
        # vendor seam (tavily) degrades to a per-angle placeholder instead of
        # killing the node — the real fetcher wraps its own body, and the
        # placeholder (plus the company angle and the report) still land.
        import yialpha.dataflows.tavily as tavily_vendor

        state = _state("MUUSDT")
        orig = cfgmod.get_config()
        try:
            cfgmod.set_config({**orig, "web_search_enabled": True})
            with (
                mock.patch.object(news, "is_historical_date", lambda _d: False),
                mock.patch.object(
                    news, "_get_news_impl", lambda *a: "COMPANY NEWS",
                ),
                mock.patch.object(
                    tavily_vendor, "get_web_search",
                    mock.MagicMock(side_effect=RuntimeError("tavily impl bug")),
                ),
            ):
                llm = _PromptCaptureLLM()
                node = create_news_analyst(llm)
                out = node(state)  # must not raise
        finally:
            cfgmod.set_config(orig)
        prompt = _descape(str(llm.prompt))
        self.assertIn(
            "<contract-angle news unavailable: RuntimeError>", prompt,
        )
        self.assertIn("COMPANY NEWS", prompt)
        self.assertIn("MOCK REPORT", out["news_report"])


class EquityPerpSquareGuardTests(unittest.TestCase):
    """An equity perp run queries Binance Square with the PERP symbol."""

    def test_equity_perp_queries_square_with_perp_symbol(self):
        square_calls = []

        def fake_square(ticker, as_of=None):  # as_of: PR4 freshness contract
            square_calls.append((ticker, as_of))
            return "SQUARE"

        with (
            mock.patch.object(sent, "_get_news_impl", lambda *a: "NEWS"),
            mock.patch.object(
                sent, "fetch_stocktwits_messages", lambda *a, **k: "STOCKTWITS"
            ),
            mock.patch.object(sent, "fetch_reddit_posts", lambda *a: "REDDIT"),
            mock.patch.object(sent, "fetch_binance_square_block", side_effect=fake_square),
            mock.patch.object(sent, "is_historical_date", lambda _d: False),
            mock.patch.object(sent, "_SENTIMENT_PARALLEL_FETCH", False),
        ):
            out = sent._fetch_sentiment_sources(
                "MUUSDT", "2026-01-01", "2026-01-08", asset_type="crypto_perp"
            )
        self.assertEqual(out[3], "SQUARE")
        # Square is queried with the perp symbol; as_of anchors the recency
        # window to the run's trade date.
        self.assertEqual(square_calls, [("MUUSDT", "2026-01-08")])


class StockPerpNewsUnderlyingTickerTests(unittest.TestCase):
    """The news vendor chain receives the UNDERLYING equity ticker.

    D8: the sentiment news leg used to receive the CONTRACT symbol
    ("MUUSDT"), which the vendor chain cannot answer — route_to_vendor's
    symbol normalization rejects it, yfinance returns nothing and Alpha
    Vantage 404s. Both fetch paths (sequential and the opt-in parallel
    pool) must hand the vendor the underlying equity ticker instead. The
    gate keys on ``asset_type == "crypto_perp"`` — the CLI AssetType value
    equity-perp runs actually carry ("stock_perp" is the derived
    instrument_class label, which never rides state["asset_type"]).
    """

    # ``-m unit`` selects only marked tests, and this file's legacy cases
    # carry no marker -- the new test opts in explicitly so the mandated
    # unit-filtered run exercises it.
    @pytest.mark.unit
    def test_stock_perp_news_vendor_receives_underlying_ticker(self):
        news_calls = []

        def fake_impl(ticker, start_date, end_date):
            news_calls.append((ticker, start_date, end_date))
            return "NEWS"

        def run_fetch(parallel):
            with (
                mock.patch.object(sent, "_get_news_impl", side_effect=fake_impl),
                mock.patch.object(
                    sent, "fetch_stocktwits_messages", lambda *a, **k: "STOCKTWITS"
                ),
                mock.patch.object(sent, "fetch_reddit_posts", lambda *a: "REDDIT"),
                mock.patch.object(
                    sent, "fetch_binance_square_block", lambda *a, **k: "SQUARE"
                ),
                mock.patch.object(sent, "is_historical_date", lambda _d: False),
                mock.patch.object(sent, "_SENTIMENT_PARALLEL_FETCH", parallel),
            ):
                return sent._fetch_sentiment_sources(
                    "MUUSDT", "2026-01-01", "2026-01-08", asset_type="crypto_perp"
                )

        sequential = run_fetch(parallel=False)
        parallel = run_fetch(parallel=True)
        self.assertEqual(sequential[0], "NEWS")
        self.assertEqual(parallel[0], "NEWS")
        # Both paths query the vendor chain with the UNDERLYING equity
        # ticker (not the contract symbol), date bounds passed through.
        self.assertEqual(
            news_calls, [("MU", "2026-01-01", "2026-01-08")] * 2
        )


class SentimentNewsEvidenceIdentityTests(unittest.TestCase):
    """The sentiment_news evidence row carries the news query's identity.

    D8 follow-up: the vendor chain is queried with the UNDERLYING equity
    ticker on tokenized-stock perp runs, so the ``sentiment_news`` evidence
    row must record that SAME identity — symbol=MU / scope=UNDERLYING, not
    the contract symbol/scope (an evidence row is the ledger's record of
    WHAT the analyst saw; a "MUUSDT"-labeled company-news row misattributes
    the underlying's news flow to the contract). The patch shape mirrors the
    offline review probe (``_source_profile`` stubbed to the
    square_plus_underlying profile, the resolver stubbed at its source
    module for the fetch leg). Contract-side legs keep the contract
    identity, and a pure-crypto perp (no resolvable underlying) keeps it
    for news too.
    """

    def _capture(self, state, *, source_profile=None):
        """Run the full sentiment node with every seam mocked.

        Returns the recorded ``record_evidence_block`` rows as dicts. When
        ``source_profile`` is given, ``sent._source_profile`` is stubbed to
        it (probe-mirror); otherwise the real policy runs.
        """
        rows = []

        def fake_record(source, category, symbol, scope, *args, **kwargs):
            rows.append(
                {"source": source, "category": category,
                 "symbol": symbol, "scope": scope}
            )

        patches = [
            mock.patch.object(sent, "bind_structured", return_value=None),
            mock.patch.object(
                sent, "invoke_structured_or_freetext", return_value="MOCK REPORT"
            ),
            mock.patch.object(sent, "_get_news_impl", lambda *a: "NEWS"),
            mock.patch.object(
                sent, "fetch_stocktwits_messages", lambda *a, **k: "STOCKTWITS"
            ),
            mock.patch.object(sent, "fetch_reddit_posts", lambda *a: "REDDIT"),
            mock.patch.object(
                sent, "fetch_binance_square_block", lambda *a, **k: "SQUARE"
            ),
            mock.patch.object(sent, "is_historical_date", lambda _d: False),
            mock.patch.object(sent, "_SENTIMENT_PARALLEL_FETCH", False),
            mock.patch.object(
                sent, "record_evidence_block", side_effect=fake_record
            ),
            mock.patch(
                "yialpha.dataflows.binance.stock_perp_underlying",
                return_value="MU",
            ),
        ]
        if source_profile is not None:
            patches.append(
                mock.patch.object(
                    sent, "_source_profile", return_value=source_profile
                )
            )
        with contextlib.ExitStack() as stack:
            for patcher in patches:
                stack.enter_context(patcher)
            sent.create_sentiment_analyst(_PromptCaptureLLM())(state)
        return {row["source"]: row for row in rows}

    def test_stock_perp_news_evidence_carries_underlying_identity(self):
        evidence = self._capture(
            _state("MUUSDT"), source_profile=("square_plus_underlying", "MU"),
        )
        news_row = evidence["sentiment_news"]
        self.assertEqual(news_row["symbol"], "MU")
        self.assertEqual(news_row["scope"], SCOPE_UNDERLYING)
        # The StockTwits row was already underlying-identified; the
        # contract-side legs keep the contract identity.
        self.assertEqual(evidence["sentiment_stocktwits"]["symbol"], "MU")
        self.assertEqual(
            evidence["sentiment_stocktwits"]["scope"], SCOPE_UNDERLYING
        )
        self.assertEqual(evidence["sentiment_reddit"]["symbol"], "MUUSDT")
        self.assertEqual(evidence["sentiment_reddit"]["scope"], SCOPE_CONTRACT)
        self.assertEqual(evidence["binance_square"]["symbol"], "MUUSDT")
        self.assertEqual(evidence["binance_square"]["scope"], SCOPE_CONTRACT)

    def test_pure_crypto_perp_news_evidence_keeps_contract_identity(self):
        # No resolvable underlying: the news row stays contract-scoped
        # (real source policy runs — BTCUSDT resolves no equity underlying).
        evidence = self._capture(_state("BTCUSDT"))
        news_row = evidence["sentiment_news"]
        self.assertEqual(news_row["symbol"], "BTCUSDT")
        self.assertEqual(news_row["scope"], SCOPE_CONTRACT)


class GenericNewsToolRoutingTests(unittest.TestCase):
    """The LLM-facing generic tools map stock-perp tickers before vendors.

    Shadow round 2 finding: the deterministic prefetch legs already query
    the UNDERLYING, but the LLM can still call the generic ``get_news`` /
    ``get_insider_transactions`` tools with the contract ticker (the shared
    instrument context demands the exact contract symbol), and those routes
    end at equity-only vendors (Yahoo / Alpha Vantage) that cannot answer
    "MUUSDT". The fix resolves the Yahoo-ready underlying INSIDE the tool
    entry, so a prompt-compliant tool call still reaches real company data.
    These tests pin the routing matrix: stock-perp -> underlying, pure-crypto
    perp and plain equity -> unchanged. Zero network: the resolver is stubbed
    at its source module and the vendor router is an in-memory spy (probe
    mirrors the review's offline reproduction).
    """

    def _vendor_calls(self, tool_fn, kwargs, underlying):
        calls = []

        def spy(*args):
            calls.append(args)
            return "MOCK"

        with (
            mock.patch.object(news_tools, "route_to_vendor", side_effect=spy),
            mock.patch(
                "yialpha.dataflows.binance.stock_perp_underlying",
                return_value=underlying,
            ),
        ):
            tool_fn.invoke(kwargs)
        return calls

    def test_get_news_maps_stock_perp_to_underlying(self):
        calls = self._vendor_calls(
            news_tools.get_news,
            {"ticker": "MUUSDT", "start_date": "2026-08-27",
             "end_date": "2026-09-03"},
            underlying="MU",
        )
        self.assertEqual(
            calls, [("get_news", "MU", "2026-08-27", "2026-09-03")]
        )

    def test_get_news_keeps_pure_crypto_contract_ticker(self):
        calls = self._vendor_calls(
            news_tools.get_news,
            {"ticker": "BTCUSDT", "start_date": "2026-08-27",
             "end_date": "2026-09-03"},
            underlying=None,
        )
        self.assertEqual(
            calls, [("get_news", "BTCUSDT", "2026-08-27", "2026-09-03")]
        )

    def test_get_news_keeps_plain_equity_ticker(self):
        calls = self._vendor_calls(
            news_tools.get_news,
            {"ticker": "MU", "start_date": "2026-08-27",
             "end_date": "2026-09-03"},
            underlying=None,
        )
        self.assertEqual(calls, [("get_news", "MU", "2026-08-27", "2026-09-03")])

    def test_get_insider_transactions_maps_stock_perp_to_underlying(self):
        calls = self._vendor_calls(
            news_tools.get_insider_transactions,
            {"ticker": "MUUSDT"},
            underlying="MU",
        )
        self.assertEqual(calls, [("get_insider_transactions", "MU")])

    def test_get_insider_transactions_keeps_plain_equity_ticker(self):
        calls = self._vendor_calls(
            news_tools.get_insider_transactions,
            {"ticker": "MU"},
            underlying=None,
        )
        self.assertEqual(calls, [("get_insider_transactions", "MU")])


if __name__ == "__main__":
    unittest.main()
