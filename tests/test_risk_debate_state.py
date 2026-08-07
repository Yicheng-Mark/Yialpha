"""Byte-equivalence guard for the shared risk-debate state builder.

Each risk debator previously rebuilt the ~13-field ``risk_debate_state`` dict
inline; ``build_risk_debate_update`` now owns it. These tests pin the EXACT dict
each debator produced (hand-transcribed from the prior inline code) so the
centralisation cannot change state-merge behaviour downstream.
"""
import unittest

from yiagents.agents.utils.agent_utils import build_risk_debate_update

# A representative incoming state with every field populated distinctly.
_BASE = {
    "history": "H0",
    "aggressive_history": "AH0",
    "conservative_history": "CH0",
    "neutral_history": "NH0",
    "latest_speaker": "Prior",
    "current_aggressive_response": "CAR0",
    "current_conservative_response": "CCR0",
    "current_neutral_response": "CNR0",
    "count": 2,
}


class TestBuildRiskDebateUpdate(unittest.TestCase):
    def test_aggressive_matches_prior_inline_dict(self):
        a = "Aggressive Analyst: lean in"
        expected = {
            "history": "H0\n" + a,
            "aggressive_history": "AH0\n" + a,
            "conservative_history": "CH0",
            "neutral_history": "NH0",
            "latest_speaker": "Aggressive",
            "current_aggressive_response": a,
            "current_conservative_response": "CCR0",
            "current_neutral_response": "CNR0",
            "count": 3,
        }
        self.assertEqual(build_risk_debate_update(_BASE, "aggressive", a), expected)

    def test_conservative_matches_prior_inline_dict(self):
        c = "Conservative Analyst: pull back"
        expected = {
            "history": "H0\n" + c,
            "aggressive_history": "AH0",
            "conservative_history": "CH0\n" + c,
            "neutral_history": "NH0",
            "latest_speaker": "Conservative",
            "current_aggressive_response": "CAR0",
            "current_conservative_response": c,
            "current_neutral_response": "CNR0",
            "count": 3,
        }
        self.assertEqual(build_risk_debate_update(_BASE, "conservative", c), expected)

    def test_neutral_matches_prior_inline_dict(self):
        n = "Neutral Analyst: balance"
        expected = {
            "history": "H0\n" + n,
            "aggressive_history": "AH0",
            "conservative_history": "CH0",
            "neutral_history": "NH0\n" + n,
            "latest_speaker": "Neutral",
            "current_aggressive_response": "CAR0",
            "current_conservative_response": "CCR0",
            "current_neutral_response": n,
            "count": 3,
        }
        self.assertEqual(build_risk_debate_update(_BASE, "neutral", n), expected)

    def test_count_uses_hard_subscript_unchanged(self):
        # The prior code used risk_debate_state["count"] + 1 (hard subscript,
        # KeyError if absent) -- preserved exactly, not relaxed to .get.
        base = dict(_BASE)
        del base["count"]
        with self.assertRaises(KeyError):
            build_risk_debate_update(base, "aggressive", "x")


if __name__ == "__main__":
    unittest.main()
