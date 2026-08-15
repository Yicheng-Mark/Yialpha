"""Unit tests for the market turbulence signal (yiagents.dataflows.market_regime)
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

import yiagents.agents.risk_mgmt.conservative_debator as cd
import yiagents.dataflows.market_regime as mr
from yiagents.dataflows.market_regime import (
    compute_turbulence,
    format_market_regime,
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


@pytest.mark.unit
class TestFormatMarketRegime:
    def test_formats_a_line(self, monkeypatch):
        monkeypatch.setattr(mr, "get_config", lambda: {"benchmark_map": {"": "SPY"}})
        monkeypatch.setattr(
            mr, "compute_turbulence", lambda *a, **k: 6.25
        )  # sqrt = 2.5 -> elevated
        line = format_market_regime("AAPL", "2024-01-15")
        assert line is not None
        assert "SPY" in line
        assert "6.25" in line
        assert "elevated" in line

    def test_normal_label_below_threshold(self, monkeypatch):
        monkeypatch.setattr(mr, "get_config", lambda: {"benchmark_map": {"": "SPY"}})
        monkeypatch.setattr(mr, "compute_turbulence", lambda *a, **k: 1.0)  # 1.0σ
        line = format_market_regime("AAPL", "2024-01-15")
        assert "normal" in line

    def test_none_propagates(self, monkeypatch):
        monkeypatch.setattr(mr, "get_config", lambda: {"benchmark_map": {"": "SPY"}})
        monkeypatch.setattr(mr, "compute_turbulence", lambda *a, **k: None)
        assert format_market_regime("AAPL", "2024-01-15") is None


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
