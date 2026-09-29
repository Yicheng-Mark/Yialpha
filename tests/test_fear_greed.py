"""Unit tests for the crypto Fear & Greed advisory line
(yialpha.dataflows.fear_greed) and its opt-in wiring into the conservative
risk debater.

Covers: PIT selection (entries after the analysis date never leak; the
prior-day delta renders only when a prior visible entry exists), string field
coercion, malformed-entry skipping, applicability gating (crypto family on;
stock runs and tokenized-stock perps off), the None short-coverage /
fetch-failure paths, and the byte-equivalence contract of the conservative
debater (off = prompt identical; on+applicable = exactly one injected line;
on+inapplicable = byte-identical and zero fetches). All ``@pytest.mark.unit``.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

import yialpha.agents.risk_mgmt.conservative_debator as cd
import yialpha.dataflows.fear_greed as fg


def _ts(iso_date: str) -> str:
    dt = datetime.strptime(iso_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return str(int(dt.timestamp()))


def _series_bytes(entries) -> bytes:
    """Payload in the API's verified shape: string fields, newest-first."""
    payload = {
        "name": "Fear and Greed Index",
        "data": [
            {
                "value": str(value),
                "value_classification": label,
                "timestamp": _ts(day),
            }
            for day, value, label in entries
        ],
    }
    return json.dumps(payload).encode()


@pytest.mark.unit
class TestFormatFearGreedLine:
    def test_selects_latest_visible_with_prior_day(self, monkeypatch):
        raw = _series_bytes(
            [("2026-09-29", 73, "Greed"),
             ("2026-09-28", 74, "Greed"),
             ("2026-09-27", 71, "Greed")]
        )
        monkeypatch.setattr(fg, "cached_or_fetch", lambda *a, **k: raw)

        assert (
            fg.format_fear_greed_line("2026-09-29")
            == "alternative.me crypto Fear & Greed, as of 2026-09-29: "
            "73 (Greed; prior day 74 (Greed))"
        )

    def test_entries_after_analysis_date_never_leak(self, monkeypatch):
        raw = _series_bytes(
            [("2026-09-29", 73, "Greed"),
             ("2026-09-28", 74, "Greed"),
             ("2026-09-27", 71, "Greed")]
        )
        monkeypatch.setattr(fg, "cached_or_fetch", lambda *a, **k: raw)

        line = fg.format_fear_greed_line("2026-09-27")

        # Only the 09-27 entry is visible: no 73, no 74, and no prior-day
        # fragment (the older series is empty for this date).
        assert line == (
            "alternative.me crypto Fear & Greed, as of 2026-09-27: 71 (Greed)"
        )

    def test_date_older_than_coverage_returns_none(self, monkeypatch):
        raw = _series_bytes([("2026-09-29", 73, "Greed")])
        monkeypatch.setattr(fg, "cached_or_fetch", lambda *a, **k: raw)

        assert fg.format_fear_greed_line("2020-01-01") is None

    def test_malformed_entries_skipped_not_zero_filled(self, monkeypatch):
        payload = {
            "name": "Fear and Greed Index",
            "data": [
                # Newest entry malformed (non-numeric value) -> skipped.
                {"value": "soon", "value_classification": "Greed",
                 "timestamp": _ts("2026-09-29")},
                # Missing timestamp key -> skipped.
                {"value": "70", "value_classification": "Neutral"},
                {"value": "69", "value_classification": "Neutral",
                 "timestamp": _ts("2026-09-28")},
            ],
        }
        monkeypatch.setattr(
            fg, "cached_or_fetch", lambda *a, **k: json.dumps(payload).encode()
        )

        line = fg.format_fear_greed_line("2026-09-29")

        assert line == (
            "alternative.me crypto Fear & Greed, as of 2026-09-28: 69 (Neutral)"
        )

    def test_fetch_failure_returns_none_not_raise(self, monkeypatch):
        def boom(*a, **k):
            raise RuntimeError("network down")

        monkeypatch.setattr(fg, "cached_or_fetch", boom)
        # Advisory signal must be fail-soft, not raise.
        assert fg.format_fear_greed_line("2026-09-29") is None

    def test_empty_or_dataless_body_returns_none(self, monkeypatch):
        monkeypatch.setattr(fg, "cached_or_fetch", lambda *a, **k: b"")
        assert fg.format_fear_greed_line("2026-09-29") is None

        monkeypatch.setattr(
            fg, "cached_or_fetch", lambda *a, **k: json.dumps({"data": []}).encode()
        )
        assert fg.format_fear_greed_line("2026-09-29") is None

    def test_malformed_curr_date_returns_none(self, monkeypatch):
        raw = _series_bytes([("2026-09-29", 73, "Greed")])
        monkeypatch.setattr(fg, "cached_or_fetch", lambda *a, **k: raw)

        assert fg.format_fear_greed_line("not-a-date") is None


@pytest.mark.unit
class TestApplicability:
    def test_pure_crypto_perp_applies(self, monkeypatch):
        monkeypatch.setattr(
            "yialpha.dataflows.binance.stock_perp_underlying", lambda t: None
        )
        assert fg.fear_greed_applicable("crypto_perp", "BTCUSDT") is True

    def test_tokenized_stock_perp_does_not_apply(self, monkeypatch):
        monkeypatch.setattr(
            "yialpha.dataflows.binance.stock_perp_underlying", lambda t: "MU"
        )
        assert fg.fear_greed_applicable("crypto_perp", "MUUSDT") is False

    def test_spot_and_legacy_crypto_apply(self):
        assert fg.fear_greed_applicable("crypto_spot", "BTCUSDT") is True
        assert fg.fear_greed_applicable("crypto", "BTC-USD") is True

    def test_stock_and_unknown_do_not_apply(self):
        assert fg.fear_greed_applicable("stock", "AAPL") is False
        assert fg.fear_greed_applicable(None, "AAPL") is False


# ---------------------------------------------------------------------------
# Byte-equivalence of the conservative debater wiring
# ---------------------------------------------------------------------------


def _state():
    return {
        "company_of_interest": "BTCUSDT",
        "trade_date": "2026-09-29",
        "market_report": "market-report-body",
        "sentiment_report": "sentiment-report-body",
        "news_report": "news-report-body",
        "fundamentals_report": "fundamentals-report-body",
        "instrument_context": "BTCUSDT instrument context",
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


_OFF_CONFIG = {"market_regime": False, "crypto_fear_greed": False}
_ON_CONFIG = {"market_regime": False, "crypto_fear_greed": True}


@pytest.mark.unit
class TestConservativeDebaterWiring:
    def test_off_is_byte_equivalent_on_injects_one_line(self, monkeypatch):
        # When OFF the debater must not even resolve applicability, and the
        # prompt must carry no Fear & Greed line.
        monkeypatch.setattr(cd, "get_config", lambda: dict(_OFF_CONFIG))

        def must_not_call(*a, **k):
            raise AssertionError("fear_greed must not run when off")

        monkeypatch.setattr(cd, "fear_greed_applicable", must_not_call)
        monkeypatch.setattr(cd, "format_fear_greed_line", must_not_call)

        llm = _RecordingLLM()
        cd.create_conservative_debator(llm)(_state())
        off_prompt = llm.prompts[0]
        assert "Fear & Greed" not in off_prompt

        # When ON + applicable, exactly the injected line is added.
        monkeypatch.setattr(cd, "get_config", lambda: dict(_ON_CONFIG))
        monkeypatch.setattr(cd, "fear_greed_applicable", lambda *a, **k: True)
        monkeypatch.setattr(
            cd, "format_fear_greed_line", lambda *a, **k: "SENTINEL_FG"
        )
        cd.create_conservative_debator(llm)(_state())
        on_prompt = llm.prompts[1]
        assert "Fear & Greed: SENTINEL_FG" in on_prompt

        # The ONLY difference between off and on is the injected line.
        assert on_prompt.replace("Fear & Greed: SENTINEL_FG\n", "") == off_prompt

    def test_on_but_inapplicable_is_byte_identical_and_never_fetches(
        self, monkeypatch
    ):
        # A stock / tokenized-stock-perp run with the flag ON: applicability
        # returns False, the module's fetch must never run, and the prompt
        # stays byte-identical to the off prompt (not-applicable is silent —
        # unlike applicable-but-failed, which must be visible).
        monkeypatch.setattr(cd, "get_config", lambda: dict(_ON_CONFIG))
        monkeypatch.setattr(cd, "fear_greed_applicable", lambda *a, **k: False)

        def must_not_fetch(*a, **k):
            raise AssertionError("format_fear_greed_line must not run")

        monkeypatch.setattr(cd, "format_fear_greed_line", must_not_fetch)

        llm = _RecordingLLM()
        cd.create_conservative_debator(llm)(_state())
        on_prompt = llm.prompts[0]

        monkeypatch.setattr(cd, "get_config", lambda: dict(_OFF_CONFIG))
        monkeypatch.setattr(cd, "fear_greed_applicable", lambda *a, **k: True)
        cd.create_conservative_debator(llm)(_state())
        off_prompt = llm.prompts[1]

        assert on_prompt == off_prompt

    def test_on_applicable_but_none_marks_unavailable(self, monkeypatch):
        # Fail-soft but VISIBLE: flag on, applicable, reading None (fetch
        # failed / date outside coverage) -> the prompt must say unavailable
        # (same fail-visible rule as the market-regime cue).
        monkeypatch.setattr(cd, "get_config", lambda: dict(_ON_CONFIG))
        monkeypatch.setattr(cd, "fear_greed_applicable", lambda *a, **k: True)
        monkeypatch.setattr(cd, "format_fear_greed_line", lambda *a, **k: None)

        llm = _RecordingLLM()
        cd.create_conservative_debator(llm)(_state())
        on_none_prompt = llm.prompts[0]
        assert "Fear & Greed: unavailable" in on_none_prompt
        assert "outside index coverage" in on_none_prompt

        monkeypatch.setattr(cd, "get_config", lambda: dict(_OFF_CONFIG))
        cd.create_conservative_debator(llm)(_state())
        off_prompt = llm.prompts[1]
        # The unavailable line is the ONLY difference from the off prompt.
        assert on_none_prompt.replace(
            "Fear & Greed: unavailable (fetch failed or date "
            "outside index coverage)\n",
            "",
        ) == off_prompt
