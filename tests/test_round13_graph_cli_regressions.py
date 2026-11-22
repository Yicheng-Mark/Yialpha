"""Round-13 regression pins — graph/CLI shard (independent file).

Lives in its own file (NOT the main-session-owned test_round13_regressions.py)
after the 17:01 git-reset + concurrent-append truncation incident wiped the
graph shard from that file twice. Scope — the two clear_checkpoint call
sites that must be fail-soft at the END of a finished run:

  [P3] ``_run_graph`` finalize seam (~trading_graph.py:2776) and
       ``finalize_streamed_run`` (~:3060) both cleared the per-ticker
       checkpoint after the run was complete and on disk. A DB failure
       escaping ``clear_checkpoint`` (its own connect can raise even though
       the function handles sqlite3.Error internally) propagated out of
       propagate()/finalize_streamed_run and discarded the completed run —
       run_robust would re-bill the whole graph on retry. Both seams are now
       fail-soft like their round-10 neighbours (_log_state /
       store_decision): only the cleanup is lost, never the run.

The two tests intentionally patch DIFFERENT namespaces: the _run_graph seam
resolves the module-level ``clear_checkpoint`` imported at trading_graph
load, while finalize_streamed_run imports it locally from the checkpointer
module — two genuinely distinct seams.
"""
from __future__ import annotations

import logging
from unittest.mock import MagicMock

import pytest


@pytest.mark.unit
def test_propagate_survives_clear_checkpoint_failure(tmp_path, monkeypatch, caplog):
    """[P3] A clear_checkpoint that RAISES at the _run_graph finalize seam
    (a DB so broken the failure escapes even the widened sqlite3.Error
    handler inside, e.g. connect itself failing) used to propagate out of
    propagate() and discard the completed, already-on-disk run — run_robust
    would re-bill the whole graph on retry. The seam is now fail-soft like
    its round-10 neighbours (_log_state / store_decision): only the cleanup
    is lost."""
    from tests.test_round10_regressions import _finalize_harness_graph
    from yialpha.graph.trading_graph import YiAlphaGraph

    mock_graph, _state = _finalize_harness_graph(tmp_path)
    # checkpoint_enabled routes propagate() through the real checkpointer
    # branch (real per-ticker DB under tmp_path) and arms the finalize-seam
    # clear_checkpoint call under test; _run_signature needs the two debate
    # depth keys and selected_analysts on the mock.
    mock_graph.config = {
        "results_dir": str(tmp_path),
        "data_cache_dir": str(tmp_path),
        "checkpoint_enabled": True,
        "max_debate_rounds": 1,
        "max_risk_discuss_rounds": 1,
    }
    mock_graph.selected_analysts = ("market",)
    from yialpha.agents.utils.memory import TradingMemoryLog

    mock_graph._log_state = MagicMock(return_value=None)
    mock_graph.memory_log = TradingMemoryLog(
        {"memory_log_path": str(tmp_path / "mem.md")}
    )
    # The seam resolves the MODULE-LEVEL name (imported at trading_graph
    # load) — patch it in trading_graph's namespace, not checkpointer's.
    monkeypatch.setattr(
        "yialpha.graph.trading_graph.clear_checkpoint",
        MagicMock(side_effect=RuntimeError("unable to open database file")),
    )
    with caplog.at_level(logging.WARNING):
        final_state, signal = YiAlphaGraph.propagate(
            mock_graph, "NVDA", "2026-01-10"
        )
    # The completed analysis is returned, not raised away.
    assert final_state["final_trade_decision"].startswith("Rating: Buy")
    assert signal is not None
    assert any(
        "post-run checkpoint clear failed" in r.message for r in caplog.records
    )


@pytest.mark.unit
def test_finalize_streamed_run_survives_clear_checkpoint_failure(
    tmp_path, monkeypatch, caplog
):
    """[P3] Same seam, streamed-CLI path: finalize_streamed_run clears the
    pre-run crashed checkpoint AFTER the run is streamed and logged — an
    exception there used to blow up the whole call with the run already
    spent. Patched at the checkpointer MODULE (this site imports
    clear_checkpoint locally, so patching trading_graph's module-level name
    would not intercept it — the two sites genuinely have two seams)."""
    from tests.test_round10_regressions import _finalize_harness_graph
    from yialpha.graph.trading_graph import YiAlphaGraph

    mock_graph, state = _finalize_harness_graph(tmp_path)
    mock_graph.config = {
        "results_dir": str(tmp_path),
        "data_cache_dir": str(tmp_path),
        "checkpoint_enabled": True,
    }
    mock_graph._log_state = MagicMock(return_value=None)
    mock_graph._run_signature = MagicMock(return_value="sig=stub")
    monkeypatch.setattr(
        "yialpha.graph.checkpointer.clear_checkpoint",
        MagicMock(side_effect=RuntimeError("disk I/O error")),
    )
    with caplog.at_level(logging.WARNING):
        out = YiAlphaGraph.finalize_streamed_run(
            mock_graph, "NVDA", "2026-01-10", state
        )
    # The streamed run's state survives verbatim, warning logged, no raise.
    assert out is state
    assert out["data_quality"] is None
    assert any(
        "post-run checkpoint clear failed" in r.message for r in caplog.records
    )
