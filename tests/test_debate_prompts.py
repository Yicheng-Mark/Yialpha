"""Direct prompt-assembly and state-wiring pins for the five debate-stage
agents (known-issue B3): the bull/bear researchers and the three risk
debators had no direct tests — a silent prompt regression (a dropped report
line, a swapped counterpart slot, a state update writing the wrong ring)
would surface only as degraded LLM behaviour downstream, never as a red
test.

Each agent is driven with a recording fake LLM (the test_fear_greed.py /
test_market_regime.py pattern) against a fully-sentinel state, pinning:

* every ``*_report`` field + instrument_context reaches the prompt in the
  builder's fixed order (index monotonicity, not full-string equality);
* the debate history and the correct counterpart slot(s) land in the prompt
  (bull/bear share ``current_response``; each risk debator reads the OTHER
  two ``current_<role>_response`` slots, never its own);
* the returned dict updates the RIGHT sub-state (investment vs risk) with
  the correct ``"<X> Analyst: "`` prefix and fills the slot the counterpart
  will read next round (pinned end-to-end by chained round-robin tests);
* ``get_language_instruction()`` is appended (these files are already in the
  i18n fixture contract, tests/test_i18n_coverage.py:24-30 — here we pin the
  runtime behaviour, not just the source-level call);
* bull/bear asset_type label branches (stock/default vs non-stock).

The conservative debator's cue-ON variants (market_regime_line /
fear_greed_line) are already pinned byte-exactly by
tests/test_market_regime.py::TestConservativeDebaterWiring and
tests/test_fear_greed.py::TestConservativeDebaterWiring — only the
cues-off byte-identical baseline is pinned here. All ``@pytest.mark.unit``.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from yialpha.agents.researchers.bear_researcher import create_bear_researcher
from yialpha.agents.researchers.bull_researcher import create_bull_researcher
from yialpha.agents.risk_mgmt.aggressive_debator import create_aggressive_debator
from yialpha.agents.risk_mgmt.conservative_debator import create_conservative_debator
from yialpha.agents.risk_mgmt.neutral_debator import create_neutral_debator
from yialpha.dataflows.config import set_config

# --- Sentinel bodies: each state slot gets a name-unique value so a prompt
# --- containing it proves the builder read exactly that slot.
_MARKET = "market-report-body"
_SENTIMENT = "sentiment-report-body"
_NEWS = "news-report-body"
_FUNDAMENTALS = "fundamentals-report-body"
_INSTRUMENT = "TESTCO instrument context"
_TRADER_PLAN = "trader-plan-body"


def _debate_state() -> dict:
    """Superset state: both debate rings + every field any of the five
    agents reads. Each node only touches its own slice."""
    return {
        "company_of_interest": "TESTCO",
        "trade_date": "2026-06-15",
        "asset_type": "stock",
        "market_report": _MARKET,
        "sentiment_report": _SENTIMENT,
        "news_report": _NEWS,
        "fundamentals_report": _FUNDAMENTALS,
        "instrument_context": _INSTRUMENT,
        "trader_investment_plan": _TRADER_PLAN,
        "investment_debate_state": {
            "history": "invest-history-body",
            "current_response": "counterpart-current-response",
            "bull_history": "bull-history-body",
            "bear_history": "bear-history-body",
            "judge_decision": "judge-decision-body",
            "count": 2,
        },
        "risk_debate_state": {
            "history": "risk-history-body",
            "current_aggressive_response": "aggressive-current-body",
            "current_conservative_response": "conservative-current-body",
            "current_neutral_response": "neutral-current-body",
            "aggressive_history": "aggressive-history-body",
            "conservative_history": "conservative-history-body",
            "neutral_history": "neutral-history-body",
            "judge_decision": "risk-judge-decision-body",
            "count": 4,
        },
    }


class _RecordingLLM:
    """Fake LLM capturing every prompt; returns a fixed reply."""

    def __init__(self, content: str):
        self.prompts: list[str] = []
        self._content = content

    def invoke(self, prompt):
        self.prompts.append(prompt)
        return MagicMock(content=self._content)


def _assert_fixed_order(prompt: str, markers: list[str]) -> None:
    """Every marker present, each strictly after the previous one."""
    prev = -1
    for marker in markers:
        idx = prompt.find(marker)
        assert idx != -1, f"prompt is missing {marker!r}"
        assert idx > prev, (
            f"{marker!r} missing or out of the builder's fixed order "
            f"(idx={idx}, prev={prev})"
        )
        prev = idx


def _conservative_cues_off(monkeypatch) -> None:
    """Pin both opt-in cues OFF for every conservative-node run in this
    file. The conftest config reset bakes a dev-box ``.env`` into
    DEFAULT_CONFIG (``YIALPHA_MARKET_REGIME`` is not held off there, unlike
    crypto_fear_greed), and cues-on with a trade_date in the state would
    issue a real regime fetch — monkeypatching the module-level get_config
    is hermetic regardless of environment."""
    import yialpha.agents.risk_mgmt.conservative_debator as cd

    monkeypatch.setattr(
        cd, "get_config",
        lambda: {"market_regime": False, "crypto_fear_greed": False},
    )


# ---------------------------------------------------------------------------
# Bull researcher
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestBullResearcher:
    def test_prompt_carries_every_source_in_fixed_order(self):
        llm = _RecordingLLM("BULL-REPLY")
        create_bull_researcher(llm)(_debate_state())
        _assert_fixed_order(llm.prompts[0], [
            _INSTRUMENT,
            f"Market research report: {_MARKET}",
            f"Social media sentiment report: {_SENTIMENT}",
            f"Latest world affairs news: {_NEWS}",
            f"Company fundamentals report: {_FUNDAMENTALS}",
            "Conversation history of the debate: invest-history-body",
            # The counterpart input: bull reads the shared current_response
            # slot, which bear's previous turn wrote ("Last bear argument").
            "Last bear argument: counterpart-current-response",
        ])

    def test_state_update_writes_investment_ring_bull_slot(self):
        llm = _RecordingLLM("BULL-REPLY")
        result = create_bull_researcher(llm)(_debate_state())
        # Right ring, and nothing else: a stray "risk_debate_state" key here
        # would corrupt the graph merge.
        assert set(result) == {"investment_debate_state"}
        argument = "Bull Analyst: BULL-REPLY"
        new = result["investment_debate_state"]
        # current_response is the slot bear reads next round.
        assert new["current_response"] == argument
        assert new["bull_history"] == f"bull-history-body\n{argument}"
        assert new["bear_history"] == "bear-history-body"
        assert new["history"] == f"invest-history-body\n{argument}"
        assert new["judge_decision"] == "judge-decision-body"
        assert new["count"] == 3

    def test_asset_type_stock_and_missing_default_use_stock_labels(self):
        llm = _RecordingLLM("x")
        create_bull_researcher(llm)(_debate_state())
        prompt = llm.prompts[0]
        assert "investing in the stock" in prompt
        assert f"Company fundamentals report: {_FUNDAMENTALS}" in prompt

        bare = _debate_state()
        del bare["asset_type"]
        llm2 = _RecordingLLM("x")
        create_bull_researcher(llm2)(bare)
        assert "investing in the stock" in llm2.prompts[0]

    def test_asset_type_non_stock_uses_asset_labels(self):
        state = _debate_state()
        state["asset_type"] = "crypto_perp"
        llm = _RecordingLLM("x")
        create_bull_researcher(llm)(state)
        prompt = llm.prompts[0]
        assert "investing in the asset" in prompt
        assert "investing in the stock" not in prompt
        assert (
            f"Asset fundamentals report (may be unavailable for crypto): "
            f"{_FUNDAMENTALS}" in prompt
        )
        assert "Company fundamentals report" not in prompt


# ---------------------------------------------------------------------------
# Bear researcher
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestBearResearcher:
    def test_prompt_carries_every_source_in_fixed_order(self):
        llm = _RecordingLLM("BEAR-REPLY")
        create_bear_researcher(llm)(_debate_state())
        _assert_fixed_order(llm.prompts[0], [
            _INSTRUMENT,
            f"Market research report: {_MARKET}",
            f"Social media sentiment report: {_SENTIMENT}",
            f"Latest world affairs news: {_NEWS}",
            f"Company fundamentals report: {_FUNDAMENTALS}",
            "Conversation history of the debate: invest-history-body",
            # Bear reads the same shared slot bull's previous turn wrote.
            "Last bull argument: counterpart-current-response",
        ])

    def test_state_update_writes_investment_ring_bear_slot(self):
        llm = _RecordingLLM("BEAR-REPLY")
        result = create_bear_researcher(llm)(_debate_state())
        assert set(result) == {"investment_debate_state"}
        argument = "Bear Analyst: BEAR-REPLY"
        new = result["investment_debate_state"]
        # current_response is the slot bull reads next round.
        assert new["current_response"] == argument
        assert new["bear_history"] == f"bear-history-body\n{argument}"
        assert new["bull_history"] == "bull-history-body"
        assert new["history"] == f"invest-history-body\n{argument}"
        assert new["judge_decision"] == "judge-decision-body"
        assert new["count"] == 3

    def test_asset_type_stock_and_missing_default_use_stock_labels(self):
        llm = _RecordingLLM("x")
        create_bear_researcher(llm)(_debate_state())
        prompt = llm.prompts[0]
        assert "investing in the stock" in prompt
        assert f"Company fundamentals report: {_FUNDAMENTALS}" in prompt

        bare = _debate_state()
        del bare["asset_type"]
        llm2 = _RecordingLLM("x")
        create_bear_researcher(llm2)(bare)
        assert "investing in the stock" in llm2.prompts[0]

    def test_asset_type_non_stock_uses_asset_labels(self):
        state = _debate_state()
        state["asset_type"] = "crypto_perp"
        llm = _RecordingLLM("x")
        create_bear_researcher(llm)(state)
        prompt = llm.prompts[0]
        assert "investing in the asset" in prompt
        assert "investing in the stock" not in prompt
        assert (
            f"Asset fundamentals report (may be unavailable for crypto): "
            f"{_FUNDAMENTALS}" in prompt
        )
        assert "Company fundamentals report" not in prompt


# ---------------------------------------------------------------------------
# Aggressive risk debator
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAggressiveDebator:
    def test_prompt_carries_every_source_in_fixed_order(self):
        llm = _RecordingLLM("AGG-REPLY")
        create_aggressive_debator(llm)(_debate_state())
        prompt = llm.prompts[0]
        assert "Here is the trader's decision:" in prompt
        _assert_fixed_order(prompt, [
            _TRADER_PLAN,
            _INSTRUMENT,
            f"Market Research Report: {_MARKET}",
            f"Social Media Sentiment Report: {_SENTIMENT}",
            f"Latest World Affairs Report: {_NEWS}",
            f"Company Fundamentals Report: {_FUNDAMENTALS}",
            "Here is the current conversation history: risk-history-body",
            "Here are the last arguments from the conservative analyst: "
            "conservative-current-body",
            "Here are the last arguments from the neutral analyst: "
            "neutral-current-body",
        ])
        # Reads the OTHER two ring members' slots, never its own — an agent
        # echoing its own previous response would degenerate the debate.
        assert "aggressive-current-body" not in prompt

    def test_state_update_writes_risk_ring_aggressive_slot(self):
        llm = _RecordingLLM("AGG-REPLY")
        result = create_aggressive_debator(llm)(_debate_state())
        assert set(result) == {"risk_debate_state"}
        argument = "Aggressive Analyst: AGG-REPLY"
        new = result["risk_debate_state"]
        assert new["current_aggressive_response"] == argument
        assert new["latest_speaker"] == "Aggressive"
        assert new["aggressive_history"] == f"aggressive-history-body\n{argument}"
        assert new["history"] == f"risk-history-body\n{argument}"
        # Counterpart slots the other two debators read stay untouched.
        assert new["current_conservative_response"] == "conservative-current-body"
        assert new["current_neutral_response"] == "neutral-current-body"
        assert new["conservative_history"] == "conservative-history-body"
        assert new["neutral_history"] == "neutral-history-body"
        assert new["judge_decision"] == "risk-judge-decision-body"
        assert new["count"] == 5


# ---------------------------------------------------------------------------
# Conservative risk debator
# ---------------------------------------------------------------------------

# Cues-off, English-language byte-identical baseline. Any drift in the
# conservative prompt (a reworded persona line, a moved data source, an
# accidental newline) fails this pin. The cue-ON variants (exactly one
# injected "Market Regime:" / "Fear & Greed:" line; explicit "unavailable"
# line) are pinned byte-exactly by tests/test_market_regime.py
# TestConservativeDebaterWiring and tests/test_fear_greed.py
# TestConservativeDebaterWiring — deliberately not duplicated here.
_EXPECTED_CONSERVATIVE_BASELINE = f"""As the Conservative Risk Analyst, your primary objective is to protect assets, minimize volatility, and ensure steady, reliable growth. You prioritize stability, security, and risk mitigation, carefully assessing potential losses, economic downturns, and market volatility. When evaluating the trader's decision or plan, critically examine high-risk elements, pointing out where the decision may expose the firm to undue risk and where more cautious alternatives could secure long-term gains. Here is the trader's decision:

{_TRADER_PLAN}

Your task is to actively counter the arguments of the Aggressive and Neutral Analysts, highlighting where their views may overlook potential threats or fail to prioritize sustainability. Respond directly to their points, drawing from the following data sources to build a convincing case for a low-risk approach adjustment to the trader's decision:

{_INSTRUMENT}
Market Research Report: {_MARKET}
Social Media Sentiment Report: {_SENTIMENT}
Latest World Affairs Report: {_NEWS}
Company Fundamentals Report: {_FUNDAMENTALS}
Here is the current conversation history: risk-history-body Here is the last response from the aggressive analyst: aggressive-current-body Here is the last response from the neutral analyst: neutral-current-body. If there are no responses from the other viewpoints yet, present your own argument based on the available data.

Engage by questioning their optimism and emphasizing the potential downsides they may have overlooked. Address each of their counterpoints to showcase why a conservative stance is ultimately the safest path for the firm's assets. Focus on debating and critiquing their arguments to demonstrate the strength of a low-risk strategy over their approaches. Output conversationally as if you are speaking without any special formatting."""


@pytest.mark.unit
class TestConservativeDebator:
    def test_prompt_carries_every_source_in_fixed_order(self, monkeypatch):
        _conservative_cues_off(monkeypatch)
        llm = _RecordingLLM("CON-REPLY")
        create_conservative_debator(llm)(_debate_state())
        prompt = llm.prompts[0]
        assert "Here is the trader's decision:" in prompt
        _assert_fixed_order(prompt, [
            _TRADER_PLAN,
            _INSTRUMENT,
            f"Market Research Report: {_MARKET}",
            f"Social Media Sentiment Report: {_SENTIMENT}",
            f"Latest World Affairs Report: {_NEWS}",
            f"Company Fundamentals Report: {_FUNDAMENTALS}",
            "Here is the current conversation history: risk-history-body",
            "Here is the last response from the aggressive analyst: "
            "aggressive-current-body",
            "Here is the last response from the neutral analyst: "
            "neutral-current-body",
        ])
        assert "conservative-current-body" not in prompt

    def test_state_update_writes_risk_ring_conservative_slot(self, monkeypatch):
        _conservative_cues_off(monkeypatch)
        llm = _RecordingLLM("CON-REPLY")
        result = create_conservative_debator(llm)(_debate_state())
        assert set(result) == {"risk_debate_state"}
        argument = "Conservative Analyst: CON-REPLY"
        new = result["risk_debate_state"]
        assert new["current_conservative_response"] == argument
        assert new["latest_speaker"] == "Conservative"
        assert new["conservative_history"] == (
            f"conservative-history-body\n{argument}"
        )
        assert new["history"] == f"risk-history-body\n{argument}"
        assert new["current_aggressive_response"] == "aggressive-current-body"
        assert new["current_neutral_response"] == "neutral-current-body"
        assert new["aggressive_history"] == "aggressive-history-body"
        assert new["neutral_history"] == "neutral-history-body"
        assert new["judge_decision"] == "risk-judge-decision-body"
        assert new["count"] == 5

    def test_cues_off_prompt_is_byte_identical_baseline(self, monkeypatch):
        _conservative_cues_off(monkeypatch)
        set_config({"output_language": "English"})
        llm = _RecordingLLM("CON-REPLY")
        create_conservative_debator(llm)(_debate_state())
        assert llm.prompts[0] == _EXPECTED_CONSERVATIVE_BASELINE


# ---------------------------------------------------------------------------
# Neutral risk debator
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestNeutralDebator:
    def test_prompt_carries_every_source_in_fixed_order(self):
        llm = _RecordingLLM("NEU-REPLY")
        create_neutral_debator(llm)(_debate_state())
        prompt = llm.prompts[0]
        assert "Here is the trader's decision:" in prompt
        _assert_fixed_order(prompt, [
            _TRADER_PLAN,
            _INSTRUMENT,
            f"Market Research Report: {_MARKET}",
            f"Social Media Sentiment Report: {_SENTIMENT}",
            f"Latest World Affairs Report: {_NEWS}",
            f"Company Fundamentals Report: {_FUNDAMENTALS}",
            "Here is the current conversation history: risk-history-body",
            "Here is the last response from the aggressive analyst: "
            "aggressive-current-body",
            "Here is the last response from the conservative analyst: "
            "conservative-current-body",
        ])
        assert "neutral-current-body" not in prompt

    def test_state_update_writes_risk_ring_neutral_slot(self):
        llm = _RecordingLLM("NEU-REPLY")
        result = create_neutral_debator(llm)(_debate_state())
        assert set(result) == {"risk_debate_state"}
        argument = "Neutral Analyst: NEU-REPLY"
        new = result["risk_debate_state"]
        assert new["current_neutral_response"] == argument
        assert new["latest_speaker"] == "Neutral"
        assert new["neutral_history"] == f"neutral-history-body\n{argument}"
        assert new["history"] == f"risk-history-body\n{argument}"
        assert new["current_aggressive_response"] == "aggressive-current-body"
        assert new["current_conservative_response"] == "conservative-current-body"
        assert new["aggressive_history"] == "aggressive-history-body"
        assert new["conservative_history"] == "conservative-history-body"
        assert new["judge_decision"] == "risk-judge-decision-body"
        assert new["count"] == 5


# ---------------------------------------------------------------------------
# Round-robin wiring: one agent's write is exactly what its counterpart's
# next-round prompt reads (the wiring a silent key rename would corrupt).
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestDebateRoundRobinWiring:
    def test_bull_argument_is_bears_next_round_input(self):
        state = _debate_state()
        result = create_bull_researcher(_RecordingLLM("BULL-REPLY"))(state)
        state["investment_debate_state"] = result["investment_debate_state"]
        bear_llm = _RecordingLLM("BEAR-REPLY")
        create_bear_researcher(bear_llm)(state)
        assert (
            "Last bull argument: Bull Analyst: BULL-REPLY"
            in bear_llm.prompts[0]
        )

    def test_bear_argument_is_bulls_next_round_input(self):
        state = _debate_state()
        result = create_bear_researcher(_RecordingLLM("BEAR-REPLY"))(state)
        state["investment_debate_state"] = result["investment_debate_state"]
        bull_llm = _RecordingLLM("BULL-REPLY")
        create_bull_researcher(bull_llm)(state)
        assert (
            "Last bear argument: Bear Analyst: BEAR-REPLY"
            in bull_llm.prompts[0]
        )

    def test_risk_ring_arguments_reach_both_counterparts(self, monkeypatch):
        _conservative_cues_off(monkeypatch)
        state = _debate_state()

        a_llm = _RecordingLLM("AGG-REPLY")
        state["risk_debate_state"] = create_aggressive_debator(a_llm)(state)[
            "risk_debate_state"
        ]
        c_llm = _RecordingLLM("CON-REPLY")
        state["risk_debate_state"] = create_conservative_debator(c_llm)(state)[
            "risk_debate_state"
        ]
        n_llm = _RecordingLLM("NEU-REPLY")
        create_neutral_debator(n_llm)(state)

        # Conservative saw aggressive's fresh argument (and neutral's still
        # untouched pre-round slot).
        assert (
            "Here is the last response from the aggressive analyst: "
            "Aggressive Analyst: AGG-REPLY" in c_llm.prompts[0]
        )
        assert (
            "Here is the last response from the neutral analyst: "
            "neutral-current-body" in c_llm.prompts[0]
        )
        # Neutral saw BOTH fresh arguments.
        assert (
            "Here is the last response from the aggressive analyst: "
            "Aggressive Analyst: AGG-REPLY" in n_llm.prompts[0]
        )
        assert (
            "Here is the last response from the conservative analyst: "
            "Conservative Analyst: CON-REPLY" in n_llm.prompts[0]
        )


# ---------------------------------------------------------------------------
# Language instruction appended at prompt-assembly time
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestLanguageInstructionAppended:
    @pytest.mark.parametrize(
        "create_agent",
        [
            pytest.param(create_bull_researcher, id="bull"),
            pytest.param(create_bear_researcher, id="bear"),
            pytest.param(create_aggressive_debator, id="aggressive"),
            pytest.param(create_conservative_debator, id="conservative"),
            pytest.param(create_neutral_debator, id="neutral"),
        ],
    )
    def test_english_adds_nothing_non_english_appends_directive(
        self, create_agent, monkeypatch
    ):
        # No-op for the four non-conservative agents; keeps the conservative
        # case hermetic (see _conservative_cues_off).
        _conservative_cues_off(monkeypatch)

        set_config({"output_language": "English"})
        llm = _RecordingLLM("x")
        create_agent(llm)(_debate_state())
        assert "Write your entire response" not in llm.prompts[0]

        set_config({"output_language": "中文"})
        create_agent(llm)(_debate_state())
        assert llm.prompts[1].endswith(" Write your entire response in 中文.")
