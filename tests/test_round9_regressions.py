"""Round 9 adversarial-sweep regression pins (2026-09-29).

Shards: S1 crypto_native/fear_greed second pass, S2 binance core REST,
S3 vision archive + perp bundle, S4 perp downstream (indicators / ticket /
accuracy), S5 coverage-gap map. One pin per landed fix; accepted findings
are recorded in the round-9 known-issues ledger (CHANGELOG), not here.
"""

from __future__ import annotations

import pytest

import yialpha.agents.utils.binance_indicator_tools as bit


@pytest.mark.unit
class TestBinanceIndicatorsCurrDateFallback:
    """S4-P2: this file was missed in the round-8 10-file clamp sweep —
    ``pd.Timestamp("")`` is NaT and its strftime raised a raw ValueError
    BEFORE the try, escaping the tool with no typed degrade and no ledger
    event."""

    def _spy_frame(self, monkeypatch):
        seen = {}

        def fake_frame(symbol, start, end, interval="1d", venue="binance_perp"):
            seen["start"] = start
            seen["end"] = end
            raise RuntimeError("spy: not fetched in this test")

        monkeypatch.setattr(bit, "binance_klines_frame", fake_frame)
        return seen

    def test_empty_curr_date_falls_back_to_pinned_date(self, monkeypatch):
        from yialpha.dataflows.utils import pinned_analysis_date

        seen = self._spy_frame(monkeypatch)
        with pinned_analysis_date("2026-06-01"):
            bit.get_binance_indicators.invoke(
                {"symbol": "BTCUSDT", "curr_date": ""})
        # The window END is the pinned analysis date, not NaT/"".
        assert seen["end"] == "2026-06-01"

    def test_live_mode_empty_curr_date_degrades_typed(self, monkeypatch):
        # No pinned date (live): the degenerate value still reaches
        # pd.Timestamp, but inside the try — the tool must return its
        # DATA_UNAVAILABLE sentinel, never raise raw.
        self._spy_frame(monkeypatch)
        out = bit.get_binance_indicators.invoke(
            {"symbol": "BTCUSDT", "curr_date": ""})
        assert out.startswith("DATA_UNAVAILABLE")

    def test_unparseable_curr_date_degrades_typed(self, monkeypatch):
        # "not-a-date": pd.Timestamp cannot parse it at all (a slash-form
        # date WOULD parse — the self-review caught that as a vacuous pin).
        self._spy_frame(monkeypatch)
        out = bit.get_binance_indicators.invoke(
            {"symbol": "BTCUSDT", "curr_date": "not-a-date"})
        assert out.startswith("DATA_UNAVAILABLE")
        assert "spy" not in out  # the fetch layer was never reached
