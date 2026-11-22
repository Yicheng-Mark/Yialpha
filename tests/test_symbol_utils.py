"""Tests for symbol normalization and the no-data routing sentinel.

Commodity/forex/index-CDN alias mapping was removed with the US+crypto-only
policy — those inputs now pass through unchanged here and are rejected by the
market gate (see test_market_gate.py) with a dedicated reason string.
"""

import unittest

import pytest

from yialpha.dataflows.symbol_utils import (
    NoMarketDataError,
    is_yahoo_safe,
    normalize_symbol,
)


@pytest.mark.unit
class TestNormalizeSymbol(unittest.TestCase):
    def test_plain_equities_unchanged(self):
        for sym in ("AAPL", "MSFT", "TSM", "BRK.B", "^GSPC", "GC=F"):
            self.assertEqual(normalize_symbol(sym), sym)

    def test_lowercases_are_upper(self):
        self.assertEqual(normalize_symbol("aapl"), "AAPL")
        self.assertEqual(normalize_symbol("  msft  "), "MSFT")

    def test_removed_aliases_pass_through_unchanged(self):
        # The alias table is gone: former CFD/forex names reach the market
        # gate verbatim (which rejects them) instead of being silently
        # remapped to a futures/index symbol.
        for sym in ("XAUUSD", "GOLD", "USOIL", "SPX500", "GER40", "EURUSD"):
            self.assertEqual(normalize_symbol(sym), sym)

    def test_crypto_pairs_get_dash_usd(self):
        self.assertEqual(normalize_symbol("BTCUSD"), "BTC-USD")
        self.assertEqual(normalize_symbol("ETHUSD"), "ETH-USD")

    def test_six_letter_non_currency_left_alone(self):
        # GOOGLE-style 6-letter tickers that aren't two currency codes
        # must not be mangled into a fake forex pair.
        self.assertEqual(normalize_symbol("ABCDEF"), "ABCDEF")

    def test_empty_input_passthrough(self):
        self.assertEqual(normalize_symbol(""), "")


@pytest.mark.unit
class TestNoMarketDataError(unittest.TestCase):
    def test_message_includes_resolution(self):
        err = NoMarketDataError("XAUUSD+", "GC=F", "no rows")
        self.assertIn("XAUUSD+", str(err))
        self.assertIn("GC=F", str(err))
        self.assertEqual(err.symbol, "XAUUSD+")
        self.assertEqual(err.canonical, "GC=F")

    def test_canonical_defaults_to_symbol(self):
        err = NoMarketDataError("FOOBAR")
        self.assertEqual(err.canonical, "FOOBAR")


@pytest.mark.unit
class TestIsYahooSafe(unittest.TestCase):
    def test_accepts_structural_chars(self):
        for sym in ("AAPL", "GC=F", "^GSPC", "BRK.B", "BTC-USD"):
            self.assertTrue(is_yahoo_safe(sym))

    def test_rejects_slash_and_space(self):
        for sym in ("a/b", "AA PL", ""):
            self.assertFalse(is_yahoo_safe(sym))


if __name__ == "__main__":
    unittest.main()
