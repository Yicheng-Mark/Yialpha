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
    parametrize in test_sentiment_fetch.py covers pure-crypto perps).

Everything runs with ZERO network: the symbol resolver is cache-or-seed by
design, and every fetch seam is mocked.
"""

import unittest
from unittest import mock

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import Runnable

import yialpha.agents.analysts.news_analyst as news
import yialpha.agents.analysts.sentiment_analyst as sent
from yialpha.agents.analysts.news_analyst import create_news_analyst
from yialpha.dataflows import config as cfgmod


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


if __name__ == "__main__":
    unittest.main()
