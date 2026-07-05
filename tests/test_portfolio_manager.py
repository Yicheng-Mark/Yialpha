"""Portfolio Manager portfolio-state renderer: guard against non-numeric holdings.

Covers the regression where an external schema passed ``{"NVDA": "100 shares"}``
(or a nested dict) as a holding and the unguarded ``float(v)`` raised
ValueError/TypeError, failing the whole PM node. Numeric inputs must render
byte-identically to the baseline.
"""
import unittest

from yiagents.agents.managers.portfolio_manager import (
    _format_portfolio_state,
    _safe_float,
)


class TestSafeFloat(unittest.TestCase):
    def test_numeric_round_trips(self):
        self.assertEqual(_safe_float(100), 100.0)
        self.assertEqual(_safe_float("42"), 42.0)
        self.assertEqual(_safe_float(3.5), 3.5)
        self.assertEqual(_safe_float(0), 0.0)

    def test_non_numeric_returns_none(self):
        self.assertIsNone(_safe_float("100 shares"))
        self.assertIsNone(_safe_float(None))
        self.assertIsNone(_safe_float({"nested": 1}))
        self.assertIsNone(_safe_float([1, 2]))


class TestFormatPortfolioState(unittest.TestCase):
    def test_none_returns_empty(self):
        self.assertEqual(_format_portfolio_state(None), "")

    def test_empty_portfolio_returns_empty(self):
        self.assertEqual(_format_portfolio_state({}), "")
        self.assertEqual(
            _format_portfolio_state({"equity": None, "cash": None, "positions": {}}),
            "",
        )

    def test_numeric_holdings_render_as_baseline(self):
        # Byte-identical to the pre-guard path for well-formed numeric input.
        out = _format_portfolio_state(
            {"equity": 100000, "cash": 25000, "positions": {"NVDA": 100, "AAPL": 50}}
        )
        self.assertEqual(
            out,
            "- Current portfolio: Equity: 100,000 | Cash: 25,000 | "
            "Holdings: NVDA 100, AAPL 50\n",
        )

    def test_zero_quantity_holdings_dropped(self):
        out = _format_portfolio_state({"positions": {"NVDA": 0, "AAPL": 50}})
        self.assertIn("AAPL 50", out)
        self.assertNotIn("NVDA", out)

    def test_non_numeric_holdings_do_not_crash(self):
        # The regression: strings / dicts as quantities used to raise.
        out = _format_portfolio_state(
            {"positions": {"NVDA": "100 shares", "AAPL": 50}}
        )
        self.assertIsInstance(out, str)
        # NVDA (non-numeric) is dropped; AAPL (numeric) still renders.
        self.assertIn("AAPL 50", out)
        self.assertNotIn("NVDA", out)
        self.assertNotIn("100 shares", out)

    def test_all_non_numeric_holdings_still_returns_equity(self):
        out = _format_portfolio_state(
            {"equity": 100000, "positions": {"NVDA": "100 shares", "TSLA": None}}
        )
        self.assertIn("Equity: 100,000", out)
        # No Holdings segment because no numeric quantity survived.
        self.assertNotIn("Holdings:", out)

    def test_float_holdings_formatted(self):
        out = _format_portfolio_state({"positions": {"BTC": 2.5}})
        self.assertIn("BTC 2", out)  # :,.0f rounds 2.5 -> 2


if __name__ == "__main__":
    unittest.main()
