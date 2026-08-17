"""Tokenized US-equity perp fundamentals wiring (e.g. MUUSDT -> Micron).

Binance exchangeInfo types tokenized-stock perps as ``underlyingType ==
"EQUITY"``; those runs keep the Fundamentals Analyst, which reads the
UNDERLYING US equity through the stock vendors. These tests pin:

  * the pure symbol matcher (suffix strip + base set + Yahoo aliases),
  * the cache-or-seed resolver contract (warm fetch / refresh / fail-open),
  * the analyst-filter keep rule for equity perps (drop for pure-crypto),
  * the deterministic perp->underlying remap in the fundamentals tools,
  * the instrument-context and fundamentals-nudge dual branches,
  * the identity resolution anchoring on the underlying ticker,
  * the warm-call wiring at both perp-run entry points.

Everything runs with ZERO network: the resolver never fetches by design
(:func:`yiagents.dataflows.binance.equity_perp_bases` is cache-or-seed), and
the one fetch path (:func:`warm_equity_perp_bases`) is tested behind a mocked
``_http_get``.
"""

import unittest
from pathlib import Path
from unittest import mock

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import Runnable

from yiagents.agents.analysts.fundamentals_analyst import create_fundamentals_analyst
from yiagents.agents.utils.agent_utils import build_instrument_context
from yiagents.cli.models import AnalystType, AssetType
from yiagents.cli.utils import filter_analysts_for_asset_type
from yiagents.dataflows import binance as bn
from yiagents.dataflows.symbol_utils import (
    _EQUITY_PERP_SEED_BASES,
    tokenized_stock_perp_underlying,
)

ALL_ANALYSTS = [
    AnalystType.MARKET,
    AnalystType.SOCIAL,
    AnalystType.NEWS,
    AnalystType.FUNDAMENTALS,
]


class TokenizedStockPerpUnderlyingTests(unittest.TestCase):
    """Pure matcher: syntactic only, no network, deterministic."""

    SET = frozenset({"MU", "META", "BRKB"})

    def test_positive_forms(self):
        for raw, expected in [
            ("MUUSDT", "MU"),
            ("MU-USDT", "MU"),
            ("muusdt", "MU"),
            ("MUUSDC", "MU"),
            ("METAUSDT", "META"),
        ]:
            self.assertEqual(
                tokenized_stock_perp_underlying(raw, self.SET), expected, raw
            )

    def test_brkb_maps_to_yahoo_dashed_form(self):
        self.assertEqual(
            tokenized_stock_perp_underlying("BRKBUSDT", self.SET), "BRK-B"
        )

    def test_negative_forms(self):
        for raw in (
            "BTCUSDT",        # pure-crypto perp
            "PEPEUSDT",       # unlisted alt
            "1000PEPEUSDT",   # prefixed alt
            "TENCENTUSDT",    # HK_EQUITY — out of scope
            "OPENAIUSDT",     # PREMARKET — out of scope
            "MU",             # no stablecoin quote
            "600519.SS",      # A-share
            "",               # empty
        ):
            self.assertIsNone(
                tokenized_stock_perp_underlying(raw, self.SET), raw
            )

    def test_non_string_is_none(self):
        self.assertIsNone(tokenized_stock_perp_underlying(None, self.SET))  # type: ignore[arg-type]

    def test_empty_explicit_set_matches_nothing(self):
        self.assertIsNone(
            tokenized_stock_perp_underlying("MUUSDT", frozenset())
        )

    def test_default_set_is_the_seed_snapshot(self):
        # No explicit set -> static seed (2026-08-17 snapshot, >= 130 bases).
        self.assertGreaterEqual(len(_EQUITY_PERP_SEED_BASES), 130)
        self.assertEqual(tokenized_stock_perp_underlying("TSLAUSDT"), "TSLA")
        self.assertEqual(tokenized_stock_perp_underlying("AAPLUSDC"), "AAPL")
        self.assertIsNone(tokenized_stock_perp_underlying("TENCENTUSDT"))


class EquityPerpBasesCacheTests(unittest.TestCase):
    """Cache-or-seed resolver: warm parse, cache reuse, refresh, fail-open."""

    PAYLOAD = {
        "symbols": [
            {"symbol": "MUUSDT", "underlyingType": "EQUITY", "status": "TRADING", "quoteAsset": "USDT"},
            {"symbol": "NEWCOUSDT", "underlyingType": "EQUITY", "status": "TRADING", "quoteAsset": "USDT"},
            {"symbol": "BTCUSDT", "underlyingType": "COIN", "status": "TRADING", "quoteAsset": "USDT"},
            {"symbol": "OLDCOUSDT", "underlyingType": "EQUITY", "status": "BREAK", "quoteAsset": "USDT"},
        ]
    }

    def setUp(self):
        bn.refresh_equity_perp_bases()

    def tearDown(self):
        bn.refresh_equity_perp_bases()

    def test_unwarmed_returns_seed_without_network(self):
        with mock.patch.object(bn, "_http_get", side_effect=AssertionError("fetch!")):
            self.assertEqual(bn.equity_perp_bases(), _EQUITY_PERP_SEED_BASES)

    def test_warm_parses_equity_trading_only(self):
        with mock.patch.object(bn, "_http_get", return_value=self.PAYLOAD) as get:
            bases = bn.warm_equity_perp_bases()
        self.assertEqual(bases, frozenset({"MU", "NEWCO"}))
        self.assertEqual(get.call_count, 1)
        self.assertEqual(bn.equity_perp_bases(), bases)  # cached read reuses

    def test_refresh_drops_back_to_seed_then_rewarms(self):
        with mock.patch.object(bn, "_http_get", return_value=self.PAYLOAD):
            bn.warm_equity_perp_bases()
        bn.refresh_equity_perp_bases()
        self.assertEqual(bn.equity_perp_bases(), _EQUITY_PERP_SEED_BASES)
        with mock.patch.object(bn, "_http_get", return_value=self.PAYLOAD) as get:
            bn.warm_equity_perp_bases()
        self.assertEqual(get.call_count, 1)  # re-fetched after refresh

    def test_fetch_failure_falls_back_to_seed_with_warning(self):
        with (
            mock.patch.object(
                bn, "_http_get",
                side_effect=bn.NoMarketDataError("EQUITY_PERP_BASES", "EQUITY_PERP_BASES", "boom"),
            ),
            self.assertLogs("yiagents.dataflows.binance", level="WARNING"),
        ):
            bases = bn.warm_equity_perp_bases()
        self.assertEqual(bases, _EQUITY_PERP_SEED_BASES)
        self.assertEqual(bn.equity_perp_bases(), _EQUITY_PERP_SEED_BASES)

    def test_empty_universe_treated_as_failure(self):
        # A 200 with zero EQUITY rows must not silently disable the analyst.
        with (
            mock.patch.object(bn, "_http_get", return_value={"symbols": []}),
            self.assertLogs("yiagents.dataflows.binance", level="WARNING"),
        ):
            bases = bn.warm_equity_perp_bases()
        self.assertEqual(bases, _EQUITY_PERP_SEED_BASES)


class StockPerpUnderlyingWrapperTests(unittest.TestCase):
    """The live wrapper: network-free (cache-or-seed), suffix pre-check."""

    def setUp(self):
        bn.refresh_equity_perp_bases()

    def tearDown(self):
        bn.refresh_equity_perp_bases()

    def test_non_stablecoin_symbol_short_circuits_before_set_lookup(self):
        with mock.patch.object(
            bn, "equity_perp_bases", side_effect=AssertionError("consulted!")
        ):
            self.assertIsNone(bn.stock_perp_underlying("AAPL"))
            self.assertIsNone(bn.stock_perp_underlying("BTC-USD"))
            self.assertIsNone(bn.stock_perp_underlying(""))

    def test_unwarmed_resolves_against_seed(self):
        self.assertEqual(bn.stock_perp_underlying("MUUSDT"), "MU")
        self.assertIsNone(bn.stock_perp_underlying("BTCUSDT"))

    def test_warmed_set_wins_over_seed(self):
        payload = {"symbols": [
            {"symbol": "NEWCOUSDT", "underlyingType": "EQUITY",
             "status": "TRADING", "quoteAsset": "USDT"},
        ]}
        with mock.patch.object(bn, "_http_get", return_value=payload):
            bn.warm_equity_perp_bases()
        self.assertEqual(bn.stock_perp_underlying("NEWCOUSDT"), "NEWCO")
        # A seed base absent from the live listing no longer resolves.
        self.assertIsNone(bn.stock_perp_underlying("MUUSDT"))

    def test_non_string_is_none(self):
        self.assertIsNone(bn.stock_perp_underlying(None))  # type: ignore[arg-type]


class FilterAnalystsStockPerpTests(unittest.TestCase):
    """Equity perps KEEP Fundamentals; pure-crypto perps drop it."""

    def test_equity_perp_keeps_fundamentals(self):
        self.assertEqual(
            filter_analysts_for_asset_type(ALL_ANALYSTS, AssetType.CRYPTO_PERP, "MUUSDT"),
            ALL_ANALYSTS,
        )

    def test_pure_crypto_perp_drops_fundamentals(self):
        self.assertEqual(
            filter_analysts_for_asset_type(ALL_ANALYSTS, AssetType.CRYPTO_PERP, "BTCUSDT"),
            [AnalystType.MARKET, AnalystType.SOCIAL, AnalystType.NEWS],
        )

    def test_default_ticker_keeps_legacy_drop(self):
        # Back-compat: existing callers/tests that never pass a ticker.
        self.assertEqual(
            filter_analysts_for_asset_type(ALL_ANALYSTS, AssetType.CRYPTO_PERP),
            [AnalystType.MARKET, AnalystType.SOCIAL, AnalystType.NEWS],
        )

    def test_stock_path_unchanged_even_with_perp_shaped_ticker(self):
        self.assertEqual(
            filter_analysts_for_asset_type(ALL_ANALYSTS, AssetType.STOCK, "MUUSDT"),
            ALL_ANALYSTS,
        )

    def test_crypto_and_spot_drop(self):
        for at in (AssetType.CRYPTO, AssetType.CRYPTO_SPOT):
            self.assertEqual(
                filter_analysts_for_asset_type(ALL_ANALYSTS, at, "MUUSDT"),
                [AnalystType.MARKET, AnalystType.SOCIAL, AnalystType.NEWS],
                at,
            )


class FundamentalsToolRemapTests(unittest.TestCase):
    """Deterministic perp->underlying remap at the fundamentals tool layer."""

    def _record(self, module):
        calls = []
        def fake_route(name, *args):
            calls.append((name, args))
            return "ok"
        return calls, fake_route

    def test_core_four_remap_equity_perp_to_underlying(self):
        import yiagents.agents.utils.fundamental_data_tools as fdt

        calls, fake = self._record(fdt)
        with mock.patch.object(fdt, "route_to_vendor", side_effect=fake):
            fdt.get_fundamentals.invoke({"ticker": "MUUSDT", "curr_date": "2026-08-17"})
            fdt.get_balance_sheet.invoke(
                {"ticker": "BRKBUSDT", "freq": "quarterly", "curr_date": "2026-08-17"}
            )
            fdt.get_cashflow.invoke({"ticker": "METAUSDT"})
            fdt.get_income_statement.invoke({"ticker": "TSLAUSDT"})
        self.assertEqual(
            [c[1][0] for c in calls], ["MU", "BRK-B", "META", "TSLA"]
        )

    def test_core_four_pass_through_non_perp_symbols(self):
        import yiagents.agents.utils.fundamental_data_tools as fdt

        calls, fake = self._record(fdt)
        with mock.patch.object(fdt, "route_to_vendor", side_effect=fake):
            fdt.get_fundamentals.invoke({"ticker": "AAPL", "curr_date": "2026-08-17"})
            fdt.get_fundamentals.invoke({"ticker": "BTCUSDT", "curr_date": "2026-08-17"})
        self.assertEqual([c[1][0] for c in calls], ["AAPL", "BTCUSDT"])

    def test_sec_ownership_trio_remaps(self):
        import yiagents.agents.utils.sec_ownership_tools as sot

        calls, fake = self._record(sot)
        with mock.patch.object(sot, "route_to_vendor", side_effect=fake):
            sot.get_form4_insider_trading.invoke(
                {"ticker": "MUUSDT", "curr_date": "2026-08-17"}
            )
            sot.get_ftd_data.invoke({"ticker": "NVDAUSDT", "curr_date": "2026-08-17"})
            sot.get_institutional_holdings.invoke(
                {"ticker": "GOOGLUSDT", "curr_date": "2026-08-17"}
            )
        self.assertEqual(
            [c[1][0] for c in calls], ["MU", "NVDA", "GOOGL"]
        )


# Byte-stable baselines shared with test_crypto_perp_mode.py expectations.
_STOCK_CTX = (
    "The instrument to analyze is `AAPL`. Use this exact ticker in every "
    "tool call, report, and recommendation, preserving any exchange suffix "
    "(e.g. `.TO`, `.L`, `.HK`, `.T`, `-USD`)."
)
_PURE_CRYPTO_PERP_CTX = (
    "The asset to analyze is `BTCUSDT`. Use this exact ticker in every "
    "tool call, report, and recommendation, preserving any exchange suffix "
    "(e.g. `.TO`, `.L`, `.HK`, `.T`, `-USD`). This is a Binance USDT-M "
    "perpetual futures contract (crypto_perp). Funding rate, open interest, "
    "and basis are first-class signals; mind leverage and funding-cost "
    "drag. Do not assume company fundamentals are available."
)


class BuildInstrumentContextStockPerpTests(unittest.TestCase):
    """Dual perp branch: equity underlying vs pure crypto (byte-stable)."""

    def test_equity_perp_context_names_underlying(self):
        ctx = build_instrument_context("MUUSDT", "crypto_perp")
        self.assertIn("tokenized US", ctx)
        self.assertIn("trades as `MU`", ctx)
        self.assertIn("fundamentals ARE available", ctx)
        self.assertNotIn("Do not assume company fundamentals", ctx)

    def test_pure_crypto_perp_context_byte_equal_to_baseline(self):
        self.assertEqual(
            build_instrument_context("BTCUSDT", "crypto_perp"),
            _PURE_CRYPTO_PERP_CTX,
        )

    def test_stock_context_byte_equal_to_baseline(self):
        self.assertEqual(build_instrument_context("AAPL", "stock"), _STOCK_CTX)


class ResolveInstrumentContextStockPerpTests(unittest.TestCase):
    """Identity resolution anchors on the UNDERLYING ticker for equity perps."""

    def _resolve(self, ticker, asset_type):
        import yiagents.graph.trading_graph as tg

        identity_calls = []

        def fake_identity(sym, trade_date=None):
            identity_calls.append(sym)
            return {}

        with mock.patch.object(tg, "resolve_instrument_identity", side_effect=fake_identity):
            ctx = tg.YiAgentsGraph.resolve_instrument_context(
                None, ticker, asset_type, "2026-08-17"
            )
        return identity_calls, ctx

    def test_equity_perp_resolves_identity_of_underlying(self):
        calls, ctx = self._resolve("MUUSDT", "crypto_perp")
        self.assertEqual(calls, ["MU"])
        self.assertIn("trades as `MU`", ctx)

    def test_pure_crypto_perp_resolves_identity_of_perp_symbol(self):
        calls, _ = self._resolve("BTCUSDT", "crypto_perp")
        self.assertEqual(calls, ["BTCUSDT"])

    def test_stock_resolves_identity_of_ticker_itself(self):
        calls, _ = self._resolve("AAPL", "stock")
        self.assertEqual(calls, ["AAPL"])


class _PromptCaptureLLM(Runnable):
    def __init__(self):
        super().__init__()
        self.prompt = None

    def invoke(self, inp, config=None, **kwargs):  # noqa: ARG002
        self.prompt = inp
        return AIMessage(content="MOCK REPORT", tool_calls=[])

    def bind_tools(self, tools, **kwargs):  # noqa: ARG002
        return self


def _perp_state(ticker, asset_type="crypto_perp"):
    return {
        "trade_date": "2026-08-17",
        "company_of_interest": ticker,
        "asset_type": asset_type,
        "instrument_context": "CTX",
        "messages": [HumanMessage(content="analyze")],
    }


class FundamentalsStockPerpNudgeTests(unittest.TestCase):
    """The nudge fires ONLY on a crypto_perp run with an equity underlying."""

    def _system_message(self, state):
        from yiagents.dataflows import config as cfgmod

        orig = cfgmod.get_config()
        try:
            cfgmod.set_config({**orig, "web_search_enabled": False})
            llm = _PromptCaptureLLM()
            node = create_fundamentals_analyst(llm)
            node(state)
            return str(llm.prompt)
        finally:
            cfgmod.set_config(orig)

    def test_equity_perp_run_gets_nudge(self):
        sm = self._system_message(_perp_state("MUUSDT"))
        self.assertIn("tokenized-stock perpetual", sm)
        self.assertIn("UNDERLYING", sm)

    def test_pure_crypto_perp_run_gets_no_nudge(self):
        sm = self._system_message(_perp_state("BTCUSDT"))
        self.assertNotIn("tokenized-stock perpetual", sm)

    def test_stock_run_with_perp_shaped_ticker_gets_no_nudge(self):
        sm = self._system_message(_perp_state("MUUSDT", asset_type="stock"))
        self.assertNotIn("tokenized-stock perpetual", sm)


class WarmWiringGuardTests(unittest.TestCase):
    """Both perp-run entry points warm the listing (source-level guard)."""

    ROOT = Path(__file__).resolve().parents[1]

    def test_cli_selection_warms_before_filter(self):
        src = (self.ROOT / "yiagents" / "cli" / "main.py").read_text(encoding="utf-8")
        self.assertIn("warm_equity_perp_bases()", src)

    def test_graph_propagate_warms_for_perp_runs(self):
        src = (self.ROOT / "yiagents" / "graph" / "trading_graph.py").read_text(encoding="utf-8")
        self.assertIn("warm_equity_perp_bases()", src)


if __name__ == "__main__":
    unittest.main()
