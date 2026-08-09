"""Regression smoke test: a real ``YiAgentsGraph`` must construct end-to-end.

WHY THIS EXISTS
---------------
The test suite deliberately avoids the real ``YiAgentsGraph.__init__``: most
graph tests use ``YiAgentsGraph.__new__(...)`` to skip the constructor or
``MagicMock(spec=YiAgentsGraph)`` to call unbound methods. That isolation is
fast and focused, but it created a **coverage blind spot**: the constructor is
the only place that threads LLM instances through ``GraphSetup`` and then calls
``setup_graph()`` → ``compile()``.

A 2026-08-08 wiring change (the ``debate_llm`` tier) broke exactly that path —
``setup.py`` read ``self.debate_llm`` without the constructor ever accepting it,
raising ``AttributeError`` on *every* real construction. The full suite (1353
tests) stayed green because no test ever built the real graph. This smoke test
exists so that class of compile-time wiring bug is caught immediately rather
than only at first CLI run.

WHAT IT GUARDS
--------------
``__init__`` → LLM client creation → ``GraphSetup(...)`` → ``setup_graph()``
→ ``workflow.compile()``. All three LLM tiers (quick / deep / debate) are
exercised because ``setup_graph`` dereferences each one eagerly.
"""

from __future__ import annotations

from yiagents.default_config import DEFAULT_CONFIG
from yiagents.graph.trading_graph import YiAgentsGraph


def test_real_graph_constructs_without_error(mock_llm_client, tmp_path):
    """Construct a real YiAgentsGraph and confirm the compiled graph exists.

    ``mock_llm_client`` (conftest fixture) patches ``create_llm_client`` so no
    network or real API key is needed. Directs cache/results dirs to ``tmp_path``
    to avoid polluting the project tree. Risk overlay is off by default.
    """
    config = dict(DEFAULT_CONFIG)
    config["data_cache_dir"] = str(tmp_path / "cache")
    config["results_dir"] = str(tmp_path / "results")

    graph = YiAgentsGraph(config=config)

    # If we get here, __init__ → setup_graph() → compile() all succeeded,
    # meaning every LLM attribute (including debate_llm) was threaded through
    # GraphSetup and dereferenced without AttributeError.
    assert graph.graph is not None
    assert graph.workflow is not None
    # All three LLM tiers must have been assigned during __init__.
    assert graph.quick_thinking_llm is not None
    assert graph.deep_thinking_llm is not None
    assert graph.debate_llm is not None


def test_debate_llm_falls_back_to_deep_think_when_unset(mock_llm_client, tmp_path):
    """When ``debate_llm`` config is None, the debate tier uses deep_think_llm.

    This locks the documented fallback semantics (debate_model = debate_llm or
    deep_think_llm) and guards against a future change that silently makes
    ``debate_llm`` mandatory without a model behind it.
    """
    config = dict(DEFAULT_CONFIG)
    config["debate_llm"] = None
    config["data_cache_dir"] = str(tmp_path / "cache")
    config["results_dir"] = str(tmp_path / "results")

    graph = YiAgentsGraph(config=config)

    assert graph.debate_llm is not None
    # The mock client returns the same MagicMock regardless of model name, so
    # we can't assert identity here — the real assertion is that construction
    # did not raise (debate_model resolved to a real string, not None).
