"""Unit tests for the market turbulence signal (yialpha.dataflows.market_regime)
and its opt-in wiring into the conservative risk debater.

Covers: benchmark resolution, the single-asset turbulence computation
(correctness vs the squared-z definition, plus the None short-history /
degenerate / fetch-failure paths), the formatter, and the byte-equivalence
contract of the conservative debater (off = prompt identical; on = exactly one
injected line). All ``@pytest.mark.unit``.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pandas as pd
import pytest

import yialpha.agents.risk_mgmt.conservative_debator as cd
import yialpha.dataflows.market_regime as mr
from yialpha.dataflows.market_regime import (
    compute_turbulence,
    resolve_market_benchmark,
)


def _ohlcv_frame(closes, start="2023-01-02"):
    dates = pd.date_range(start, periods=len(closes), freq="D")
    return pd.DataFrame({"Date": dates, "Close": closes})


def _closes_from_returns(returns, start=100.0):
    closes = [start]
    for r in returns:
        closes.append(closes[-1] * (1 + r))
    return closes


@pytest.mark.unit
class TestResolveBenchmark:
    def test_default_is_spy(self, monkeypatch):
        # No explicit override, unrecognised suffix -> empty-suffix entry (SPY).
        monkeypatch.setattr(mr, "get_config", lambda: {"benchmark_map": {"": "SPY"}})
        assert resolve_market_benchmark("AAPL") == "SPY"

    def test_suffix_map(self, monkeypatch):
        cfg = {"benchmark_map": {".HK": "^HSI", "": "SPY"}}
        monkeypatch.setattr(mr, "get_config", lambda: cfg)
        assert resolve_market_benchmark("0700.HK") == "^HSI"

    def test_explicit_override_wins(self, monkeypatch):
        cfg = {"benchmark_ticker": "BTCUSDT", "benchmark_map": {"": "SPY"}}
        monkeypatch.setattr(mr, "get_config", lambda: cfg)
        assert resolve_market_benchmark("AAPL") == "BTCUSDT"


@pytest.mark.unit
class TestComputeTurbulence:
    def test_matches_squared_z_definition(self, monkeypatch):
        # A deterministic, non-constant, non-degenerate return series.
        returns = [((i * 7) % 23 - 11) * 0.001 for i in range(300)]
        closes = _closes_from_returns(returns)
        monkeypatch.setattr(mr, "load_ohlcv", lambda sym, d: _ohlcv_frame(closes))

        turb = compute_turbulence("SYN", "2024-12-31", window=252, min_periods=60)

        # Independently compute the squared z-score the function claims to use.
        r = pd.Series(returns, dtype=float)
        hist = r.iloc[-253:-1]  # 252 returns before the last
        curr = r.iloc[-1]
        expected = (curr - hist.mean()) ** 2 / hist.var(ddof=1)

        assert turb is not None
        assert turb == pytest.approx(float(expected), rel=1e-9)
        assert turb >= 0

    def test_none_when_insufficient_history(self, monkeypatch):
        returns = [0.001] * 30  # only 30 returns (< min_periods+1 with min=60)
        closes = _closes_from_returns(returns)
        monkeypatch.setattr(mr, "load_ohlcv", lambda sym, d: _ohlcv_frame(closes))
        assert compute_turbulence("SYN", "2024-01-01", min_periods=60) is None

    def test_none_when_degenerate_variance(self, monkeypatch):
        # Constant prices -> zero returns -> zero variance -> None.
        closes = [100.0] * 300
        monkeypatch.setattr(mr, "load_ohlcv", lambda sym, d: _ohlcv_frame(closes))
        assert compute_turbulence("SYN", "2024-01-01") is None

    def test_none_when_load_fails(self, monkeypatch):
        def boom(sym, d):
            raise RuntimeError("network down")

        monkeypatch.setattr(mr, "load_ohlcv", boom)
        # Advisory signal must be fail-soft, not raise.
        assert compute_turbulence("SYN", "2024-01-01") is None

    def test_none_when_empty_frame(self, monkeypatch):
        monkeypatch.setattr(mr, "load_ohlcv", lambda sym, d: pd.DataFrame())
        assert compute_turbulence("SYN", "2024-01-01") is None


# NOTE: the pre-expansion turbulence-only renderer ``format_market_regime``
# was removed 2026-08-16 (format_regime_context is its strict superset; its
# turbulence-label semantics — elevated >= 2.5σ vs normal — are pinned by
# TestFormatRegimeContext in tests/test_price_structure_and_regime.py).


# ---------------------------------------------------------------------------
# Asset-aware venue routing: a crypto_perp/crypto_spot regime line reads
# Binance — the Yahoo load_ohlcv path must see ZERO calls (BTCUSDT would be
# silently remapped to the BTC-USD spot index, a different instrument at a
# different basis; a tokenized-stock perp has no Yahoo symbol at all).
# ---------------------------------------------------------------------------


def _perp_frame(n=260, start="2025-01-01"):
    closes = [100.0 + (i % 5) * 0.8 + i * 0.03 for i in range(n)]
    # Named "Date" index — the exact shape binance_klines_frame yields (the
    # regime loader reset_index()es it back into the Yahoo column shape).
    idx = pd.date_range(start, periods=n, freq="D", name="Date")
    return pd.DataFrame(
        {
            "Open": closes,
            "High": [c + 1.0 for c in closes],
            "Low": [c - 1.0 for c in closes],
            "Close": closes,
            "Adj Close": closes,
            "Volume": [10.0] * n,
        },
        index=idx,
    )


@pytest.mark.unit
class TestAssetAwareRegimeVenue:
    def test_crypto_perp_instrument_loads_from_binance_never_yahoo(self, monkeypatch):
        import yialpha.dataflows.binance as bn

        def boom(sym, d):
            raise AssertionError(
                f"Yahoo load_ohlcv must not serve a crypto_perp regime line "
                f"(called with symbol={sym})"
            )

        monkeypatch.setattr(mr, "load_ohlcv", boom)
        seen: dict = {}

        def fake_klines(symbol, start, end, interval="1d",
                        venue="binance_perp", price_type="last",
                        closed_as_of=None):
            seen["venue"] = venue
            return _perp_frame()

        monkeypatch.setattr(bn, "binance_klines_frame", fake_klines)
        monkeypatch.setattr(bn, "stock_perp_underlying", lambda t: None)

        line = mr.format_regime_context(
            "BTCUSDT", "2025-12-31", asset_type="crypto_perp",
        )
        assert seen.get("venue") == "binance_perp"
        assert line is not None and "trend=" in line and "vol=" in line

    def test_crypto_spot_uses_spot_venue(self, monkeypatch):
        import yialpha.dataflows.binance as bn

        monkeypatch.setattr(
            mr, "load_ohlcv",
            lambda sym, d: _ohlcv_frame(_closes_from_returns([0.001] * 300)),
        )
        seen: dict = {}

        def fake_klines(symbol, start, end, interval="1d",
                        venue="binance_spot", price_type="last",
                        closed_as_of=None):
            seen["venue"] = venue
            return _perp_frame()

        monkeypatch.setattr(bn, "binance_klines_frame", fake_klines)
        line = mr.format_regime_context(
            "BTCUSDT", "2025-12-31", asset_type="crypto_spot",
        )
        assert seen.get("venue") == "binance_spot"
        assert line is not None

    def test_legacy_crypto_mode_stays_on_yahoo(self, monkeypatch):
        # Byte-equivalence for the legacy auto-detected mode: the risk
        # overlay deliberately keeps crypto spot on the Yahoo index source,
        # and the regime line matches that venue choice.
        import yialpha.dataflows.binance as bn

        def must_not_fetch(*a, **k):
            raise AssertionError("legacy crypto mode must not hit Binance")

        monkeypatch.setattr(bn, "binance_klines_frame", must_not_fetch)
        monkeypatch.setattr(
            mr, "load_ohlcv", lambda sym, d: _perp_frame().reset_index(),
        )
        line = mr.format_regime_context(
            "BTC-USD", "2025-12-31", asset_type="crypto",
        )
        assert line is not None and "trend=" in line

    def test_pure_perp_benchmark_defaults_to_btcusdt(self, monkeypatch):
        import yialpha.dataflows.binance as bn

        monkeypatch.setattr(mr, "get_config", lambda: {"benchmark_map": {"": "SPY"}})
        monkeypatch.setattr(bn, "stock_perp_underlying", lambda t: None)
        assert mr.resolve_market_benchmark("ETHUSDT", "crypto_perp") == "BTCUSDT"
        assert mr.resolve_market_benchmark("BTCUSDT", "crypto_spot") == "BTCUSDT"

    def test_stock_perp_benchmark_stays_spy(self, monkeypatch):
        # A tokenized-stock perp's market narrative is the underlying
        # equity's — SPY remains its context benchmark.
        import yialpha.dataflows.binance as bn

        monkeypatch.setattr(mr, "get_config", lambda: {"benchmark_map": {"": "SPY"}})
        monkeypatch.setattr(bn, "stock_perp_underlying", lambda t: "MU")
        assert mr.resolve_market_benchmark("MUUSDT", "crypto_perp") == "SPY"

    def test_stock_benchmark_resolution_unchanged(self, monkeypatch):
        monkeypatch.setattr(mr, "get_config", lambda: {"benchmark_map": {"": "SPY"}})
        assert mr.resolve_market_benchmark("AAPL", "stock") == "SPY"
        assert mr.resolve_market_benchmark("AAPL") == "SPY"  # default arg


# ---------------------------------------------------------------------------
# Byte-equivalence of the conservative debater wiring
# ---------------------------------------------------------------------------


def _state():
    return {
        "company_of_interest": "AAPL",
        "trade_date": "2024-01-15",
        "market_report": "market-report-body",
        "sentiment_report": "sentiment-report-body",
        "news_report": "news-report-body",
        "fundamentals_report": "fundamentals-report-body",
        "instrument_context": "AAPL instrument context",
        "trader_investment_plan": "trader-plan-body",
        "risk_debate_state": {
            "history": "hist",
            "current_aggressive_response": "agg",
            "current_neutral_response": "neu",
            "count": 0,
        },
    }


class _RecordingLLM:
    def __init__(self):
        self.prompts = []

    def invoke(self, prompt):
        self.prompts.append(prompt)
        return MagicMock(content="conservative argument")


@pytest.mark.unit
class TestConservativeDebaterWiring:
    def test_off_is_byte_equivalent_on_injects_one_line(self, monkeypatch):
        # When OFF, get_config returns market_regime falsy; the debater must
        # not call format_regime_context and the prompt must carry no regime line.
        monkeypatch.setattr(cd, "get_config", lambda: {"market_regime": False})

        def must_not_call(*a, **k):
            raise AssertionError("format_regime_context must not run when off")

        monkeypatch.setattr(cd, "format_regime_context", must_not_call)

        llm = _RecordingLLM()
        cd.create_conservative_debator(llm)(_state())
        off_prompt = llm.prompts[0]
        assert "Market Regime:" not in off_prompt
        assert "turbulence" not in off_prompt

        # When ON, exactly the injected line is added; nothing else changes.
        # (2026-08-15: the injected reading is the composite regime line — a
        # superset of the old turbulence-only format_market_regime.)
        monkeypatch.setattr(cd, "get_config", lambda: {"market_regime": True})
        monkeypatch.setattr(cd, "format_regime_context", lambda *a, **k: "SENTINEL_READING")
        cd.create_conservative_debator(llm)(_state())
        on_prompt = llm.prompts[1]
        assert "Market Regime: SENTINEL_READING" in on_prompt

        # The ONLY difference between off and on is the injected line.
        assert on_prompt.replace("Market Regime: SENTINEL_READING\n", "") == off_prompt

    def test_on_but_reading_none_marks_unavailable(self, monkeypatch):
        # Fail-soft but VISIBLE: with the flag on and the composite regime
        # line returning None (no data), the prompt must say the reading is
        # unavailable — silently omitting the cue the operator opted into
        # made "configured but broken" indistinguishable from "configured
        # and calm".
        monkeypatch.setattr(cd, "get_config", lambda: {"market_regime": True})
        monkeypatch.setattr(cd, "format_regime_context", lambda *a, **k: None)
        llm = _RecordingLLM()
        cd.create_conservative_debator(llm)(_state())
        on_none_prompt = llm.prompts[0]
        assert "Market Regime: unavailable" in on_none_prompt
        assert "fetch failed" in on_none_prompt

        monkeypatch.setattr(cd, "get_config", lambda: {"market_regime": False})
        cd.create_conservative_debator(llm)(_state())
        off_prompt = llm.prompts[1]
        # The unavailable line is the ONLY difference from the off prompt.
        assert on_none_prompt.replace(
            "Market Regime: unavailable (fetch failed or insufficient "
            "history)\n", ""
        ) == off_prompt
