"""Field-contract pins for the TypedDict state schema (round 13).

``agent_states.py`` is the LLM-reachable execution surface every graph node
reads and writes, yet until now no test referenced it: the two debate
carry-forward helpers had byte-equivalence guards
(``test_investment_debate_state.py`` / ``test_risk_debate_state.py``) but the
TypedDicts themselves — the *schema* those helpers must satisfy — were pinned
nowhere. A silently renamed or dropped field would not fail any test until a
node read it at runtime.

These tests pin three contracts:

* **Key-set snapshots** — the exact key set of ``InvestDebateState``,
  ``RiskDebateState`` and ``AgentState`` (all required, none optional), so a
  rename/drop is a red test, not a silent state drift.
* **Last-write-wins** — every field's ``Annotated`` metadata is a non-empty
  docstring, NOT a reducer callable (the module docstring's core merge
  contract; ``messages`` is the single exception, inherited from
  ``MessagesState``'s additive reducer).
* **Helper/schema cross-link** — the dict each carry-forward helper returns
  has exactly the TypedDict's key set: the parent state replaces the whole
  sub-dict, so a helper that forgets a field (or a TypedDict that grows one)
  would blank state with no error anywhere. Plus the missing-key behaviour
  (``count`` is the one hard-subscripted KeyError; every other absent key
  defaults) and input non-mutation.
"""

from __future__ import annotations

from typing import Annotated, get_args, get_origin

import pytest

from yialpha.agents.utils.agent_states import (
    AgentState,
    InvestDebateState,
    RiskDebateState,
)
from yialpha.agents.utils.agent_utils import (
    build_investment_debate_update,
    build_risk_debate_update,
)

# --- Hand-transcribed key sets (source of truth: agent_states.py) -----------

INVEST_DEBATE_KEYS = frozenset(
    {
        "bull_history",
        "bear_history",
        "history",
        "current_response",
        "judge_decision",
        "count",
    }
)

RISK_DEBATE_KEYS = frozenset(
    {
        "aggressive_history",
        "conservative_history",
        "neutral_history",
        "history",
        "latest_speaker",
        "current_aggressive_response",
        "current_conservative_response",
        "current_neutral_response",
        "judge_decision",
        "count",
    }
)

# AgentState = its own 18 fields + ``messages`` inherited from MessagesState.
AGENT_STATE_KEYS = frozenset(
    {
        "messages",
        "company_of_interest",
        "asset_type",
        "instrument_context",
        "trade_date",
        "market_report",
        "positioning_report",
        "sentiment_report",
        "news_report",
        "fundamentals_report",
        "investment_debate_state",
        "investment_plan",
        "trader_investment_plan",
        "risk_debate_state",
        "final_trade_decision",
        "pm_rating",
        "pm_decision_fields",
        "past_context",
        "portfolio_state",
    }
)


@pytest.mark.unit
class TestTypedDictKeySnapshots:
    def test_invest_debate_state_keys(self):
        assert frozenset(InvestDebateState.__required_keys__) == INVEST_DEBATE_KEYS

    def test_risk_debate_state_keys(self):
        assert frozenset(RiskDebateState.__required_keys__) == RISK_DEBATE_KEYS

    def test_agent_state_keys(self):
        assert frozenset(AgentState.__required_keys__) == AGENT_STATE_KEYS

    def test_all_fields_required_none_optional(self):
        # total=True (the default): create_initial_state seeds every field, so
        # a field silently becoming optional would also hide a missing seed.
        for cls in (InvestDebateState, RiskDebateState, AgentState):
            assert cls.__optional_keys__ == frozenset(), cls.__name__

    def test_agent_state_inherits_messages_state(self):
        # TypedDict rejects issubclass and flattens the MRO, so the checkable
        # evidence of ``class AgentState(MessagesState)`` is the inherited
        # ``messages`` field (with the additive reducer — see the
        # last-write-wins tests) present in AgentState's own merged
        # annotation dict and required-key set.
        assert "messages" in AgentState.__annotations__
        assert "messages" in AgentState.__required_keys__


@pytest.mark.unit
class TestLastWriteWinsContract:
    """Every Annotated second argument is a docstring, never a reducer."""

    @pytest.mark.parametrize(
        "cls",
        [InvestDebateState, RiskDebateState],
        ids=["invest", "risk"],
    )
    def test_annotated_metadata_is_docstring_not_reducer(self, cls):
        for field, annotation in cls.__annotations__.items():
            assert get_origin(annotation) is Annotated, (cls.__name__, field)
            args = get_args(annotation)
            assert len(args) == 2, (cls.__name__, field)
            metadata = args[1]
            # A reducer here would silently change LangGraph merge semantics
            # from last-write-wins to something else. Must be a non-empty
            # human docstring (str), never a callable.
            assert isinstance(metadata, str) and metadata.strip(), (
                cls.__name__,
                field,
            )

    def test_agent_state_own_fields_are_docstring_not_reducer(self):
        for field, annotation in AgentState.__annotations__.items():
            if field == "messages":
                continue  # inherited additive reducer — see test below
            assert get_origin(annotation) is Annotated, ("AgentState", field)
            metadata = get_args(annotation)[1]
            assert isinstance(metadata, str) and metadata.strip(), ("AgentState", field)

    def test_messages_field_is_the_additive_reducer_exception(self):
        # AgentState inherits ``messages`` from MessagesState with the real
        # ``add_messages`` reducer (append + RemoveMessage) — the single
        # sanctioned departure from last-write-wins. The unresolved string
        # annotation names it.
        ann = AgentState.__annotations__["messages"]
        assert isinstance(ann, str) or type(ann).__name__ == "ForwardRef"
        assert "add_messages" in repr(ann)

    def test_invest_debate_field_types(self):
        ann = InvestDebateState.__annotations__
        assert ann["count"] == Annotated[int, "Length of the current conversation"]
        assert ann["bull_history"] == Annotated[str, "Bullish Conversation history"]

    def test_risk_debate_field_types(self):
        ann = RiskDebateState.__annotations__
        assert ann["count"] == Annotated[int, "Length of the current conversation"]
        assert ann["latest_speaker"] == Annotated[str, "Analyst that spoke last"]

    def test_agent_state_field_types(self):
        from typing import Any

        ann = AgentState.__annotations__
        assert get_args(ann["investment_debate_state"])[0] is InvestDebateState
        assert get_args(ann["risk_debate_state"])[0] is RiskDebateState
        # The two deliberately schema-free fields are Annotated[Any, "..."].
        assert get_args(ann["pm_decision_fields"])[0] is Any
        assert get_args(ann["portfolio_state"])[0] is Any


@pytest.mark.unit
class TestCarryForwardHelpersMatchSchema:
    """The helpers must return EXACTLY the TypedDict key set — no more, no
    less. The parent state replaces the whole sub-dict (last-write-wins on the
    field), so an omitted key blanks prior state and an extra key drifts the
    schema (and would break the TypedDict-equivalence of the runtime dict).
    """

    _INVEST_BASE = {
        "history": "H0",
        "bull_history": "BH0",
        "bear_history": "RH0",
        "current_response": "CR0",
        "judge_decision": "JD0",
        "count": 2,
    }

    _RISK_BASE = {
        "history": "H0",
        "aggressive_history": "AH0",
        "conservative_history": "CH0",
        "neutral_history": "NH0",
        "latest_speaker": "Prior",
        "current_aggressive_response": "CAR0",
        "current_conservative_response": "CCR0",
        "current_neutral_response": "CNR0",
        "judge_decision": "JD0",
        "count": 2,
    }

    @pytest.mark.parametrize("speaker", ["bull", "bear"])
    def test_investment_update_key_set_matches_typeddict(self, speaker):
        result = build_investment_debate_update(self._INVEST_BASE, speaker, "arg")
        assert frozenset(result) == INVEST_DEBATE_KEYS

    @pytest.mark.parametrize(
        "speaker", ["aggressive", "conservative", "neutral"]
    )
    def test_risk_update_key_set_matches_typeddict(self, speaker):
        result = build_risk_debate_update(self._RISK_BASE, speaker, "arg")
        assert frozenset(result) == RISK_DEBATE_KEYS

    def test_investment_carry_forward_and_update(self):
        arg = "Bull: demand"
        result = build_investment_debate_update(self._INVEST_BASE, "bull", arg)
        assert result["history"] == "H0\n" + arg          # shared log appends
        assert result["bull_history"] == "BH0\n" + arg    # speaker appends
        assert result["bear_history"] == "RH0"            # opponent unchanged
        assert result["current_response"] == arg          # overwritten
        assert result["judge_decision"] == "JD0"          # carried through
        assert result["count"] == 3                        # advanced by 1

    def test_risk_carry_forward_and_update(self):
        arg = "Neutral: hold"
        result = build_risk_debate_update(self._RISK_BASE, "neutral", arg)
        assert result["history"] == "H0\n" + arg
        assert result["neutral_history"] == "NH0\n" + arg
        assert result["aggressive_history"] == "AH0"
        assert result["conservative_history"] == "CH0"
        assert result["latest_speaker"] == "Neutral"
        assert result["current_neutral_response"] == arg
        assert result["current_aggressive_response"] == "CAR0"
        assert result["current_conservative_response"] == "CCR0"
        assert result["judge_decision"] == "JD0"
        assert result["count"] == 3


@pytest.mark.unit
class TestMissingKeyBehaviour:
    """``count`` is the ONLY hard-subscripted key (KeyError when absent);
    every other absent key defaults (empty-string histories / carried fields).
    """

    def test_investment_missing_count_raises_keyerror(self):
        with pytest.raises(KeyError):
            build_investment_debate_update(
                {"history": "h", "bull_history": "b", "bear_history": "r"}, "bull", "x"
            )

    def test_risk_missing_count_raises_keyerror(self):
        with pytest.raises(KeyError):
            build_risk_debate_update({"history": "h", "latest_speaker": "P"}, "aggressive", "x")

    def test_investment_absent_fields_default(self):
        result = build_investment_debate_update({"count": 0}, "bull", "arg")
        assert result == {
            "history": "\narg",
            "bull_history": "\narg",
            "bear_history": "",
            "current_response": "arg",
            "judge_decision": "",
            "count": 1,
        }

    def test_risk_absent_fields_default(self):
        result = build_risk_debate_update({"count": 0}, "conservative", "arg")
        assert result == {
            "history": "\narg",
            "aggressive_history": "",
            "conservative_history": "\narg",
            "neutral_history": "",
            "latest_speaker": "Conservative",
            "current_aggressive_response": "",
            "current_conservative_response": "arg",
            "current_neutral_response": "",
            "judge_decision": "",
            "count": 1,
        }


@pytest.mark.unit
class TestInputNotMutated:
    """The graph may reuse the incoming sub-state after the helper runs
    (e.g. checkpoint replay); the helpers must be pure readers of it.
    """

    def test_investment_input_unchanged(self):
        base = dict(TestCarryForwardHelpersMatchSchema._INVEST_BASE)
        build_investment_debate_update(base, "bear", "new")
        assert base == TestCarryForwardHelpersMatchSchema._INVEST_BASE

    def test_risk_input_unchanged(self):
        base = dict(TestCarryForwardHelpersMatchSchema._RISK_BASE)
        build_risk_debate_update(base, "aggressive", "new")
        assert base == TestCarryForwardHelpersMatchSchema._RISK_BASE
