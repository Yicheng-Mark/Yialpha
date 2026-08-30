"""Tokenized US-equity perp news: deterministic dual-angle prefetch.

Binance tokenized-stock perp runs (e.g. MUUSDT -> Micron) pre-fetch BOTH news
angles in code — the UNDERLYING equity (query "MU") and the perp contract
(query "MUUSDT") — and inject them as labelled blocks, so dual coverage does
not depend on the LLM choosing to query both. These tests pin:

  * the gate: blocks fire ONLY on crypto_perp runs with an equity underlying
    (stock runs and pure-crypto perps append nothing, byte-unchanged),
  * the exact two queries and the 7-day window anchored on trade_date,
  * block contents injected verbatim with the honest per-angle no-coverage
    instruction present,
  * historical replay dates still prefetch (get_news is date-bounded, so the
    blocks are PIT-safe and need no live gate),
  * the sentiment-side requirement shipped with this change: an equity perp
    run queries Binance Square with the PERP symbol (the existing crypto
    parametrize in test_sentiment_fetch.py covers pure-crypto perps).

Everything runs with ZERO network: the symbol resolver is cache-or-seed by
design, and the prefetch accessor ``_get_news_impl`` is mocked throughout.
"""

import unittest
from unittest import mock

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import Runnable

import yiagents.agents.analysts.news_analyst as news
import yiagents.agents.analysts.sentiment_analyst as sent
from yiagents.agents.analysts.news_analyst import create_news_analyst
from yiagents.dataflows import config as cfgmod


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


def _run_node(state, blocks=("COMPANY NEWS", "PERP NEWS"), historical=False):
    """Invoke the news analyst node with the prefetch accessor mocked.

    Returns (recorded_calls, rendered_prompt). Config is snapshotted around
    the run (web_search off) so the prompt bytes are deterministic.
    """
    calls = []

    def fake_impl(ticker, start_date, end_date):
        calls.append((ticker, start_date, end_date))
        return blocks[min(len(calls) - 1, len(blocks) - 1)]

    orig = cfgmod.get_config()
    try:
        cfgmod.set_config({**orig, "web_search_enabled": False})
        with (
            mock.patch.object(news, "is_historical_date", lambda _d: historical),
            mock.patch.object(news, "_get_news_impl", side_effect=fake_impl),
        ):
            llm = _PromptCaptureLLM()
            node = create_news_analyst(llm)
            node(state)
        return calls, str(llm.prompt)
    finally:
        cfgmod.set_config(orig)


class NewsPrefetchGateTests(unittest.TestCase):
    """The dual-angle prefetch fires ONLY on an equity-perp run."""

    def test_equity_perp_prefetches_both_angles(self):
        calls, prompt = _run_node(_state("MUUSDT"))
        self.assertEqual(
            calls,
            [
                ("MU", "2026-08-10", "2026-08-17"),
                ("MUUSDT", "2026-08-10", "2026-08-17"),
            ],
        )
        self.assertIn('<start_of_company_news> (query: "MU")', prompt)
        self.assertIn("COMPANY NEWS", prompt)
        self.assertIn('<start_of_perp_news> (query: "MUUSDT")', prompt)
        self.assertIn("PERP NEWS", prompt)
        self.assertIn("MUST cover BOTH angles", prompt)

    def test_pure_crypto_perp_appends_nothing(self):
        calls, prompt = _run_node(_state("BTCUSDT"))
        self.assertEqual(calls, [])
        self.assertNotIn("Pre-fetched news", prompt)
        self.assertNotIn("start_of_company_news", prompt)
        self.assertNotIn("start_of_perp_news", prompt)

    def test_stock_run_with_perp_shaped_ticker_appends_nothing(self):
        calls, prompt = _run_node(_state("MUUSDT", asset_type="stock"))
        self.assertEqual(calls, [])
        self.assertNotIn("Pre-fetched news", prompt)

    def test_window_anchors_on_trade_date(self):
        calls, _ = _run_node(_state("NVDAUSDT", trade_date="2026-06-30"))
        self.assertEqual(
            calls,
            [
                ("NVDA", "2026-06-23", "2026-06-30"),
                ("NVDAUSDT", "2026-06-23", "2026-06-30"),
            ],
        )


class NewsPrefetchContentTests(unittest.TestCase):
    """Block contents land verbatim with honest no-coverage wording."""

    def test_empty_blocks_injected_verbatim_with_instruction(self):
        calls, prompt = _run_node(
            _state("MUUSDT"), blocks=("no coverage found", "no coverage found")
        )
        self.assertEqual(len(calls), 2)
        self.assertIn("no coverage found", prompt)
        # The per-angle honesty rule is static instruction text in the section.
        # (Assertion fragment avoids apostrophes: str(prompt) is a message-list
        # repr, which escapes embedded quotes.)
        self.assertIn("for that angle explicitly instead of generalizing", prompt)

    def test_historical_date_still_prefetches_date_bounded(self):
        # get_news carries explicit date bounds, so the blocks are PIT-safe on
        # replay dates and must NOT be gated off (unlike web_search/Square).
        calls, prompt = _run_node(_state("MUUSDT"), historical=True)
        self.assertEqual(len(calls), 2)
        self.assertIn('<start_of_company_news> (query: "MU")', prompt)


class EquityPerpSquareGuardTests(unittest.TestCase):
    """An equity perp run queries Binance Square with the PERP symbol."""

    def test_equity_perp_queries_square_with_perp_symbol(self):
        square_calls = []

        def fake_square(ticker):
            square_calls.append(ticker)
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
        self.assertEqual(square_calls, ["MUUSDT"])


if __name__ == "__main__":
    unittest.main()
