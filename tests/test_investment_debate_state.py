"""Byte-equivalence guard for the shared investment-debate state builder.

The bull/bear researchers previously rebuilt the 5-field
``investment_debate_state`` dict inline; ``build_investment_debate_update`` now
owns it. These tests pin the EXACT dict each researcher produced
(hand-transcribed from the prior inline code) so the centralisation cannot
change state-merge behaviour downstream.

The one deliberate deviation: ``judge_decision`` is carried through unchanged
(neither researcher WRITES a judge decision, but the whole-sub-dict replacement
used to blank one the Research Manager had already written — e.g. on a resumed
run — until the RM ran again).

Mirrors ``test_risk_debate_state.py`` — the same guard pattern already proven
for the three risk debators' shared helper.
"""
import unittest

from yialpha.agents.utils.agent_utils import build_investment_debate_update

# A representative incoming state with every field populated distinctly.
_BASE = {
    "history": "H0",
    "bull_history": "BH0",
    "bear_history": "RH0",
    "current_response": "CR0",
    "count": 2,
}


class TestBuildInvestmentDebateUpdate(unittest.TestCase):
    def test_bull_matches_prior_inline_dict(self):
        b = "Bull Analyst: growth story"
        expected = {
            "history": "H0\n" + b,
            "bull_history": "BH0\n" + b,
            "bear_history": "RH0",
            "current_response": b,
            "judge_decision": "",
            "count": 3,
        }
        self.assertEqual(build_investment_debate_update(_BASE, "bull", b), expected)

    def test_bear_matches_prior_inline_dict(self):
        r = "Bear Analyst: downside risk"
        expected = {
            "history": "H0\n" + r,
            "bull_history": "BH0",
            "bear_history": "RH0\n" + r,
            "current_response": r,
            "judge_decision": "",
            "count": 3,
        }
        self.assertEqual(build_investment_debate_update(_BASE, "bear", r), expected)

    def test_count_uses_hard_subscript_unchanged(self):
        # The prior code used investment_debate_state["count"] + 1 (hard
        # subscript, KeyError if absent) -- preserved exactly, not relaxed to
        # .get. Matches the risk-debate helper's contract.
        base = dict(_BASE)
        del base["count"]
        with self.assertRaises(KeyError):
            build_investment_debate_update(base, "bull", "x")

    def test_judge_decision_carried_through(self):
        # Neither researcher WRITES a judge decision, but the whole-sub-dict
        # replacement used to blank one the Research Manager had already
        # written (e.g. a resumed run replaying a researcher turn) until the
        # RM ran again. The value now passes through untouched.
        base = dict(_BASE, judge_decision="bull wins on fundamentals")
        result = build_investment_debate_update(base, "bull", "x")
        self.assertEqual(result["judge_decision"], "bull wins on fundamentals")

    def test_judge_decision_defaults_to_empty_string(self):
        result = build_investment_debate_update(_BASE, "bear", "x")
        self.assertEqual(result["judge_decision"], "")


if __name__ == "__main__":
    unittest.main()
