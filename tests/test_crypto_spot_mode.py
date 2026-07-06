"""Track A — Binance SPOT analysis mode (crypto_spot).

These tests pin the spot-specific behavior AND guard the byte-equivalence
contract (the project "iron rule"): stock and crypto-spot-baseline and perp
runs must stay identical to the pre-change baseline. They also unit-test the
new cross-venue spot-perp basis computation and the spot host-mirror switch.
Pure-logic / mock-LLM / mock-transport, so they run with zero network and zero
LLM cost.
"""

import unittest
from datetime import datetime, timezone
from unittest import mock

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import Runnable

from cli.models import AnalystType, AssetType
from cli.utils import filter_analysts_for_asset_type
from yiagents.agents.analysts.market_analyst import create_market_analyst
from yiagents.agents.utils.agent_utils import build_instrument_context
from yiagents.dataflows import binance as binance_vendor
from yiagents.dataflows.errors import NoMarketDataError
from yiagents.dataflows.symbol_utils import (
    normalize_symbol,
    normalize_symbol_for_venue,
)


class NormalizeSymbolForVenueSpotTests(unittest.TestCase):
    """Spot shares the perp USDT normalization — same output for both venues."""

    _CASES = [
        "BTCUSDT", "BTC-USD", "btc-usdt", "BTC-USDC", "BTC", "ETHUSDT",
        "1000PEPE-USDT", "SOL-USDT",
    ]

    def test_spot_equals_perp_for_all_cases(self):
        for raw in self._CASES:
            self.assertEqual(
                normalize_symbol_for_venue(raw, "binance_spot"),
                normalize_symbol_for_venue(raw, "binance_perp"),
                msg=f"spot/perp divergence for {raw!r}",
            )

    def test_spot_basic_shapes(self):
        self.assertEqual(normalize_symbol_for_venue("BTC-USD", "binance_spot"), "BTCUSDT")
        self.assertEqual(normalize_symbol_for_venue("btcusdt", "binance_spot"), "BTCUSDT")
        self.assertEqual(normalize_symbol_for_venue("BTC", "binance_spot"), "BTCUSDT")

    def test_yahoo_venue_still_delegates(self):
        # venue="yahoo" must be unchanged (regression guard).
        for raw in ("BTC-USD", "AAPL", "XAUUSD", "EURUSD"):
            self.assertEqual(
                normalize_symbol_for_venue(raw, venue="yahoo"),
                normalize_symbol(raw),
            )

    def test_perp_output_unchanged_regression(self):
        # The perp path was byte-identical before this change; pin a few outputs.
        self.assertEqual(normalize_symbol_for_venue("BTC-USDC"), "BTCUSDT")
        self.assertEqual(normalize_symbol_for_venue("btc-usdt"), "BTCUSDT")
        self.assertEqual(normalize_symbol_for_venue("1000PEPE-USDT"), "1000PEPEUSDT")


class FilterAnalystsSpotTests(unittest.TestCase):
    """crypto_spot drops Fundamentals like crypto/perp; stock unchanged."""

    ALL = [
        AnalystType.MARKET,
        AnalystType.SOCIAL,
        AnalystType.NEWS,
        AnalystType.FUNDAMENTALS,
    ]

    def test_spot_excludes_fundamentals(self):
        self.assertEqual(
            filter_analysts_for_asset_type(self.ALL, AssetType.CRYPTO_SPOT),
            [AnalystType.MARKET, AnalystType.SOCIAL, AnalystType.NEWS],
        )

    def test_stock_keeps_all(self):
        self.assertEqual(
            filter_analysts_for_asset_type(self.ALL, AssetType.STOCK),
            self.ALL,
        )


class BuildInstrumentContextSpotTests(unittest.TestCase):
    """Spot context carries the spot instruction; stock/crypto byte-stable."""

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

    def test_spot_context_carries_spot_instruction(self):
        ctx = build_instrument_context("BTCUSDT", "crypto_spot")
        self.assertIn("BTCUSDT", ctx)
        self.assertIn("Binance SPOT pair", ctx)
        self.assertIn("spot-perp basis", ctx)
        # Spot must NOT carry the perp instruction.
        self.assertNotIn("perpetual futures contract", ctx)

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
    """Fake LLM that records the tools handed to ``bind_tools``."""

    def __init__(self):
        super().__init__()
        self.bound_tools = None

    def invoke(self, inp, config=None, **kwargs):  # noqa: D401, ARG002
        return AIMessage(content="", tool_calls=[])

    def bind_tools(self, tools, **kwargs):  # noqa: ARG002
        self.bound_tools = list(tools)
        return _BoundLLM()


class MarketAnalystToolBindingSpotTests(unittest.TestCase):
    """crypto_spot binds the 3 spot tools + indicators + snapshot (5 total);
    stock/crypto still bind the baseline 3; perp still binds 6."""

    @staticmethod
    def _state(asset_type):
        ticker = {
            "crypto_perp": "BTCUSDT",
            "crypto_spot": "BTCUSDT",
            "crypto": "BTC-USD",
            "stock": "AAPL",
        }[asset_type]
        return {
            "trade_date": "2026-07-01",
            "company_of_interest": ticker,
            "asset_type": asset_type,
            "instrument_context": "CTX",
            "messages": [HumanMessage(content="analyze")],
        }

    def _tool_names(self, asset_type):
        llm = RecordingLLM()
        node = create_market_analyst(llm)
        node(self._state(asset_type))
        return [t.name for t in llm.bound_tools]

    def test_stock_binds_three_baseline_tools(self):
        self.assertEqual(
            self._tool_names("stock"),
            ["get_stock_data", "get_indicators", "get_verified_market_snapshot"],
        )

    def test_crypto_binds_three_baseline_tools(self):
        self.assertEqual(
            self._tool_names("crypto"),
            ["get_stock_data", "get_indicators", "get_verified_market_snapshot"],
        )

    def test_spot_binds_five_tools(self):
        names = self._tool_names("crypto_spot")
        # Spot keeps indicators + verified snapshot (symbol resolves correctly)
        # and adds the 3 spot-native tools.
        self.assertEqual(
            sorted(names),
            [
                "get_binance_spot_klines",
                "get_binance_spot_perp_basis",
                "get_binance_spot_ticker24",
                "get_indicators",
                "get_verified_market_snapshot",
            ],
        )
        # Spot must NOT bind the perp-native tools.
        self.assertNotIn("get_binance_klines", names)
        self.assertNotIn("get_binance_funding_rate", names)


def _kline(day_iso: str, close: float) -> list:
    """Build a minimal Binance kline 12-tuple for a UTC date + close price."""
    open_ms = int(
        datetime.strptime(day_iso, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp()
        * 1000
    )
    return [open_ms, close, close, close, close, 100.0, open_ms + 86_400_000,
            1.0, 1, 50.0, 50.0, "0"]


class SpotPerpBasisUnitTests(unittest.TestCase):
    """Unit-test the cross-venue basis computation with mocked pagination.

    ``get_binance_spot_perp_basis`` calls ``_paginate_history`` twice — once for
    the perp leg (``/fapi/v1/klines``) and once for the spot leg
    (``/api/v3/klines``). We stub it to return fixed samples keyed by path and
    assert the basis / basisRate math and date inner-join.
    """

    def _run_basis(self, perp_rows, spot_rows):
        def fake_paginate(path, *args, **kwargs):
            if path.startswith("/fapi"):
                return perp_rows
            if path.startswith("/api"):
                return spot_rows
            return []
        with mock.patch.object(binance_vendor, "_paginate_history", side_effect=fake_paginate):
            return binance_vendor.get_binance_spot_perp_basis("BTCUSDT", look_back_days=7)

    def test_basis_aligned_and_computed(self):
        perp = [_kline("2026-06-30", 100.0), _kline("2026-07-01", 102.0),
                _kline("2026-07-02", 101.0)]
        spot = [_kline("2026-06-30", 99.0), _kline("2026-07-01", 100.0),
                _kline("2026-07-02", 103.0)]
        out = self._run_basis(perp, spot)
        # Header + CSV; three data rows, ordered by date.
        self.assertIn("date,perpClose,spotClose,basis,basisRate", out)
        rows = [ln for ln in out.splitlines() if ln and not ln.startswith("#")
                and not ln.startswith("date,")]
        self.assertEqual(len(rows), 3)
        # Row 0: 2026-06-30, basis = 100 - 99 = 1, rate = 1/99.
        self.assertTrue(rows[0].startswith("2026-06-30,"))
        self.assertAlmostEqual(float(rows[0].split(",")[3]), 1.0)   # basis
        self.assertAlmostEqual(float(rows[0].split(",")[4]), 1.0 / 99.0)  # basisRate
        # Row 2: 2026-07-02, basis = 101 - 103 = -2.
        self.assertAlmostEqual(float(rows[2].split(",")[3]), -2.0)

    def test_inner_join_drops_non_overlapping_dates(self):
        # Perp has 06-30/07-01; spot has 07-01/07-02 → only 07-01 overlaps.
        perp = [_kline("2026-06-30", 100.0), _kline("2026-07-01", 102.0)]
        spot = [_kline("2026-07-01", 100.0), _kline("2026-07-02", 103.0)]
        out = self._run_basis(perp, spot)
        rows = [ln for ln in out.splitlines() if ln and not ln.startswith("#")
                and not ln.startswith("date,")]
        self.assertEqual(len(rows), 1)
        self.assertTrue(rows[0].startswith("2026-07-01,"))

    def test_no_overlap_raises_no_market_data(self):
        perp = [_kline("2026-06-30", 100.0)]
        spot = [_kline("2026-07-02", 103.0)]
        with self.assertRaises(NoMarketDataError):
            self._run_basis(perp, spot)

    def test_empty_leg_raises_no_market_data(self):
        # One leg empty (e.g. a perp with no spot listing) → no overlap → raise.
        perp = [_kline("2026-06-30", 100.0)]
        with self.assertRaises(NoMarketDataError):
            self._run_basis(perp, [])


class SpotMirrorHostTests(unittest.TestCase):
    """``_spot_host()`` honors the binance_spot_mirror config flag."""

    def test_default_is_canonical_host(self):
        with mock.patch.object(binance_vendor, "get_config",
                               return_value={"binance_spot_mirror": False}):
            self.assertEqual(binance_vendor._spot_host(),
                             binance_vendor._SPOT_BASE)

    def test_mirror_flag_returns_mirror_host(self):
        with mock.patch.object(binance_vendor, "get_config",
                               return_value={"binance_spot_mirror": True}):
            self.assertEqual(binance_vendor._spot_host(),
                             binance_vendor._SPOT_MIRROR_BASE)

    def test_missing_key_defaults_to_canonical(self):
        with mock.patch.object(binance_vendor, "get_config", return_value={}):
            self.assertEqual(binance_vendor._spot_host(),
                             binance_vendor._SPOT_BASE)


if __name__ == "__main__":
    unittest.main()
