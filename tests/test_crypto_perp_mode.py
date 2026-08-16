"""Track A — Binance USDT-M perpetual analysis mode (crypto_perp).

These tests pin the perp-specific behavior AND guard the byte-equivalence
contract (the project "iron rule"): stock and crypto-spot runs must stay
identical to the pre-change baseline. They are pure-logic / mock-LLM, so they
run with zero network and zero LLM cost.
"""

import unittest
from datetime import date

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import Runnable

from yiagents.agents.analysts.market_analyst import create_market_analyst
from yiagents.agents.utils.agent_utils import build_instrument_context
from yiagents.cli.models import AnalystType, AssetType
from yiagents.cli.utils import filter_analysts_for_asset_type
from yiagents.dataflows.symbol_utils import normalize_symbol, normalize_symbol_for_venue


class NormalizeSymbolForVenueTests(unittest.TestCase):
    """T15.1 — pure syntactic normalization, no network."""

    def test_dashed_usd_maps_to_usdt(self):
        self.assertEqual(normalize_symbol_for_venue("BTC-USD"), "BTCUSDT")

    def test_lower_dashed_usdt_preserved(self):
        self.assertEqual(normalize_symbol_for_venue("btc-usdt"), "BTCUSDT")

    def test_long_base_kept_intact(self):
        self.assertEqual(normalize_symbol_for_venue("1000PEPE-USDT"), "1000PEPEUSDT")

    def test_usdc_quote_collapsed_to_usdt(self):
        self.assertEqual(normalize_symbol_for_venue("BTC-USDC"), "BTCUSDT")

    def test_plain_usdt_pair_uppercased(self):
        self.assertEqual(normalize_symbol_for_venue("btcusdt"), "BTCUSDT")

    def test_bare_base_gets_usdt_appended(self):
        self.assertEqual(normalize_symbol_for_venue("BTC"), "BTCUSDT")

    def test_degenerate_usdt_only_rejected(self):
        with self.assertRaises(ValueError):
            normalize_symbol_for_venue("USDT")

    def test_degenerate_doubled_usdt_rejected(self):
        with self.assertRaises(ValueError):
            normalize_symbol_for_venue("USDTUSDT")

    def test_yahoo_venue_delegates_to_normalize_symbol(self):
        # venue="yahoo" must be the exact same function + output as before.
        for raw in ("BTC-USD", "AAPL", "XAUUSD", "EURUSD"):
            self.assertEqual(
                normalize_symbol_for_venue(raw, venue="yahoo"),
                normalize_symbol(raw),
            )

    def test_delivery_contract_refused_not_mangled(self):
        # 2026-08-16: a dated delivery suffix used to fall into the bare-base
        # branch and price a fabricated "BTCUSDT_260327USDT" — the perpetual
        # data layer must refuse instead.
        for raw in ("BTCUSDT_260327", "BTCUSDT260327", "BTC_260626", "ethusdt_270924"):
            with self.assertRaises(ValueError, msg=raw):
                normalize_symbol_for_venue(raw)

    def test_numeric_bases_do_not_trip_delivery_guard(self):
        # Real bases carry digits (prefix form), never a trailing date-shaped
        # 6-pack; these must keep normalizing normally.
        self.assertEqual(normalize_symbol_for_venue("1000PEPE"), "1000PEPEUSDT")
        self.assertEqual(normalize_symbol_for_venue("1MBABYDOGE-USDT"), "1MBABYDOGEUSDT")


class FilterAnalystsPerpTests(unittest.TestCase):
    """T15.2 — perp drops Fundamentals like crypto; stock/crypto unchanged."""

    ALL = [
        AnalystType.MARKET,
        AnalystType.SOCIAL,
        AnalystType.NEWS,
        AnalystType.FUNDAMENTALS,
    ]

    def test_perp_excludes_fundamentals(self):
        self.assertEqual(
            filter_analysts_for_asset_type(self.ALL, AssetType.CRYPTO_PERP),
            [AnalystType.MARKET, AnalystType.SOCIAL, AnalystType.NEWS],
        )

    def test_crypto_still_excludes_fundamentals(self):
        self.assertEqual(
            filter_analysts_for_asset_type(self.ALL, AssetType.CRYPTO),
            [AnalystType.MARKET, AnalystType.SOCIAL, AnalystType.NEWS],
        )

    def test_stock_keeps_all(self):
        self.assertEqual(
            filter_analysts_for_asset_type(self.ALL, AssetType.STOCK),
            self.ALL,
        )


class BuildInstrumentContextPerpTests(unittest.TestCase):
    """T15.3 — perp context has the perp instruction; stock/crypto byte-stable."""

    _STOCK_CTX = (
        "The instrument to analyze is `AAPL`. Use this exact ticker in every "
        "tool call, report, and recommendation, preserving any exchange suffix "
        "(e.g. `.TO`, `.L`, `.HK`, `.T`, `-USD`)."
    )
    _CRYPTO_CTX = (
        "The asset to analyze is `BTC-USD`. Use this exact ticker in every "
        "tool call, report, and recommendation, preserving any exchange suffix "
        "(e.g. `.TO`, `.L`, `.HK`, `.T`, `-USD`). Treat it as a crypto asset "
        "rather than a company, and do not assume company fundamentals are "
        "available."
    )

    def test_perp_context_carries_perp_instruction(self):
        ctx = build_instrument_context("BTCUSDT", "crypto_perp")
        self.assertIn("BTCUSDT", ctx)
        self.assertIn("Binance USDT-M perpetual futures contract", ctx)
        self.assertIn("Funding rate", ctx)
        self.assertIn("open interest", ctx)

    def test_stock_context_byte_equal_to_baseline(self):
        self.assertEqual(build_instrument_context("AAPL", "stock"), self._STOCK_CTX)

    def test_crypto_context_byte_equal_to_baseline(self):
        self.assertEqual(
            build_instrument_context("BTC-USD", "crypto"), self._CRYPTO_CTX
        )


class _BoundLLM(Runnable):
    """The object returned by RecordingLLM.bind_tools — emits a fixed reply."""

    def invoke(self, inp, config=None, **kwargs):  # noqa: D401, ARG002
        return AIMessage(content="MOCK REPORT", tool_calls=[])


class RecordingLLM(Runnable):
    """Fake LLM that records the tools handed to ``bind_tools``.

    Mirrors just enough of the langchain Runnable interface for
    ``prompt | llm.bind_tools(tools)`` to execute inside the analyst node.
    """

    def __init__(self):
        super().__init__()
        self.bound_tools = None

    def invoke(self, inp, config=None, **kwargs):  # noqa: D401, ARG002
        # Never reached — bind_tools returns the bound runnable that the chain
        # actually invokes — but Runnable requires a concrete invoke.
        return AIMessage(content="", tool_calls=[])

    def bind_tools(self, tools, **kwargs):  # noqa: ARG002
        self.bound_tools = list(tools)
        return _BoundLLM()


class MarketAnalystToolBindingTests(unittest.TestCase):
    """T15.4 — perp binds ONLY the 3 Binance tools (spot tools hidden, since
    they resolve a perp symbol to a different Yahoo spot pair and would corrupt
    the anti-hallucination "ground truth"); stock/crypto bind the baseline 3
    spot tools byte-for-byte."""

    @staticmethod
    def _state(asset_type, trade_date=None):
        return {
            "trade_date": trade_date or date.today().isoformat(),
            "company_of_interest": "BTCUSDT" if asset_type == "crypto_perp" else "AAPL",
            "asset_type": asset_type,
            "instrument_context": "CTX",
            "messages": [HumanMessage(content="analyze")],
        }

    def _tool_names(self, asset_type, trade_date=None):
        llm = RecordingLLM()
        node = create_market_analyst(llm)
        node(self._state(asset_type, trade_date))
        return [t.name for t in llm.bound_tools]

    def test_stock_binds_baseline_plus_expansion_tools(self):
        # 2026-08-15 technical-analysis expansion: the baseline trio gains the
        # weekly-timeframe context, the three price-structure evidence tools
        # (S/R levels, volume confirmation/divergence, candlestick patterns),
        # and benchmark relative strength.
        self.assertEqual(
            self._tool_names("stock"),
            [
                "get_stock_data",
                "get_indicators",
                "get_verified_market_snapshot",
                "get_indicators_weekly",
                "get_support_resistance",
                "get_volume_features",
                "get_candlestick_patterns",
                "get_relative_strength",
            ],
        )

    def test_crypto_binds_baseline_plus_expansion_tools(self):
        self.assertEqual(
            self._tool_names("crypto"),
            [
                "get_stock_data",
                "get_indicators",
                "get_verified_market_snapshot",
                "get_indicators_weekly",
                "get_support_resistance",
                "get_volume_features",
                "get_candlestick_patterns",
                "get_relative_strength",
            ],
        )

    def test_perp_hides_spot_tools_binds_only_binance(self):
        names = self._tool_names("crypto_perp")
        # Spot tools are hidden for perp runs — they would resolve the perp
        # symbol to a different Yahoo spot pair (BTCUSDT -> BTC-USD) and return
        # the wrong market as the "verified" ground truth.
        self.assertNotIn("get_stock_data", names)
        self.assertNotIn("get_indicators", names)
        self.assertNotIn("get_verified_market_snapshot", names)
        # The 8 perp-native Binance tools: OHLCV/funding/OI (the original
        # cost-of-carry & crowding trio) plus the positioning/order-flow/basis
        # trio, plus get_binance_indicators (2026-08-15) which computes the
        # classic stockstats battery on the actual perp klines — the indicator
        # capability perp runs previously lacked entirely — and
        # get_binance_premium_index (2026-08-16): the mark-price snapshot that
        # liquidation-distance claims must anchor to.
        self.assertEqual(len(names), 8)
        self.assertEqual(
            sorted(names),
            [
                "get_binance_basis",
                "get_binance_funding_rate",
                "get_binance_indicators",
                "get_binance_klines",
                "get_binance_long_short_ratio",
                "get_binance_open_interest",
                "get_binance_premium_index",
                "get_binance_taker_buy_sell",
            ],
        )

    def test_historical_perp_omits_current_positioning_and_order_flow(self):
        self.assertEqual(
            sorted(self._tool_names("crypto_perp", "2020-01-02")),
            [
                "get_binance_funding_rate",
                "get_binance_indicators",
                "get_binance_klines",
            ],
        )


if __name__ == "__main__":
    unittest.main()
