"""Pin the shared display contract of ``yialpha/fmt.py``.

``fmt_pct`` / ``fmt_num3`` are the 2026-08-16 dedup of the EXACT duplicates
that lived in backtest/report.py, monitoring/dashboard.py and
agents/utils/valuation_tools.py. Three production consumers now share one
formatter, but nothing locked the contract ("n/a" lowercase, .2f / .3f) —
a stray rename or case change would silently shift every report surface.

Hermetic: pure functions, no imports beyond the module under test.
"""

from __future__ import annotations

import pytest

from yialpha.fmt import fmt_num3, fmt_pct


@pytest.mark.unit
class TestFmtPct:
    def test_fraction_to_percent(self):
        assert fmt_pct(0.0123) == "1.23%"

    def test_none_is_lowercase_na(self):
        assert fmt_pct(None) == "n/a"

    def test_negative_and_zero(self):
        assert fmt_pct(-0.05) == "-5.00%"
        assert fmt_pct(0.0) == "0.00%"

    def test_rounding_to_two_decimals(self):
        assert fmt_pct(0.123456) == "12.35%"


@pytest.mark.unit
class TestFmtNum3:
    def test_three_decimals(self):
        assert fmt_num3(1.23456) == "1.235"

    def test_none_is_lowercase_na(self):
        assert fmt_num3(None) == "n/a"

    def test_negative(self):
        assert fmt_num3(-2.5) == "-2.500"
