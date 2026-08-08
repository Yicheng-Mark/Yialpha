"""Unit tests for pure routing/config methods on ``YiAgentsGraph``.

The full constructor wires real LLM clients and a compiled LangGraph, so these
tests follow the established pattern (test_risk_overlay.py, test_p0_p1a.py):
``YiAgentsGraph.__new__`` bypasses ``__init__`` and we set only the attributes
the method-under-test reads. This pins the checkpoint-signature and
benchmark-resolution contracts without any network or LLM call.
"""

from __future__ import annotations

import pytest

from yiagents.graph.trading_graph import YiAgentsGraph

pytestmark = pytest.mark.unit


def _bare_graph(config: dict, selected_analysts=("market", "social", "news", "fundamentals")) -> YiAgentsGraph:
    """A graph instance with __init__ skipped — only config + analysts set."""
    g = YiAgentsGraph.__new__(YiAgentsGraph)
    g.config = config
    g.selected_analysts = selected_analysts
    return g


class TestResolveBenchmark:
    """``_resolve_benchmark`` picks the alpha-calc benchmark by priority:
    explicit override > suffix-map match > empty-suffix default (SPY)."""

    def test_explicit_override_wins(self):
        g = _bare_graph({"benchmark_ticker": "^GSPC", "benchmark_map": {"": "SPY"}})
        assert g._resolve_benchmark("AAPL") == "^GSPC"

    def test_empty_suffix_defaults_to_spy(self):
        g = _bare_graph({"benchmark_map": {"": "SPY"}})
        assert g._resolve_benchmark("AAPL") == "SPY"

    def test_tokyo_suffix_maps_to_nikkei(self):
        g = _bare_graph({"benchmark_map": {"": "SPY", ".T": "^N225"}})
        assert g._resolve_benchmark("7203.T") == "^N225"

    def test_suffix_match_is_case_insensitive(self):
        g = _bare_graph({"benchmark_map": {"": "SPY", ".T": "^N225"}})
        assert g._resolve_benchmark("7203.t") == "^N225"

    def test_dotted_us_ticker_falls_back_to_empty_suffix(self):
        # BRK.B has a dot but no recognised suffix — alpha calc in USD, so SPY.
        g = _bare_graph({"benchmark_map": {"": "SPY", ".T": "^N225"}})
        assert g._resolve_benchmark("BRK.B") == "SPY"

    def test_no_benchmark_map_at_all_defaults_spy(self):
        g = _bare_graph({})
        assert g._resolve_benchmark("anything") == "SPY"


class TestRunSignature:
    """``_run_signature`` folds graph-shape inputs into a string that becomes
    part of the checkpoint thread_id. Changing any input MUST change the
    signature, so a resume under a different shape starts fresh instead of
    continuing an incompatible graph."""

    def _base_config(self) -> dict:
        return {
            "max_debate_rounds": 1,
            "max_risk_discuss_rounds": 1,
            "analyst_parallel": False,
        }

    def test_basic_signature_shape(self):
        g = _bare_graph(self._base_config(), selected_analysts=("market", "news"))
        sig = g._run_signature("stock")
        assert "analysts=market,news" in sig
        assert "debate=1" in sig
        assert "risk=1" in sig
        assert "asset=stock" in sig
        assert "parallel=False" in sig

    def test_different_analyst_set_changes_signature(self):
        cfg = self._base_config()
        g1 = _bare_graph(cfg, selected_analysts=("market",))
        g2 = _bare_graph(cfg, selected_analysts=("market", "news"))
        assert g1._run_signature("stock") != g2._run_signature("stock")

    def test_different_asset_type_changes_signature(self):
        g = _bare_graph(self._base_config())
        assert g._run_signature("stock") != g._run_signature("crypto")

    def test_parallel_mode_changes_signature(self):
        cfg_off = self._base_config()
        cfg_on = {**self._base_config(), "analyst_parallel": True}
        assert _bare_graph(cfg_off)._run_signature("stock") != _bare_graph(cfg_on)._run_signature("stock")

    def test_debate_depth_changes_signature(self):
        cfg1 = self._base_config()
        cfg2 = {**self._base_config(), "max_debate_rounds": 3}
        assert _bare_graph(cfg1)._run_signature("stock") != _bare_graph(cfg2)._run_signature("stock")
