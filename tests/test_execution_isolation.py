"""Regression guard: the execution layer stays decoupled from agents & graph.

Two invariants this plan promised:

1. **No agent or graph node imports the execution layer.** The execution
   skeleton (domain / gateway / bridge) is new code with no callers; if a
   future change wires it into an agent prompt or the LangGraph topology, the
   default-off / byte-equivalence contract silently breaks. This test greps
   ``yiagents/agents`` and ``yiagents/graph`` for any such import and fails
   loudly if one appears.

2. **The existing BrowserBroker public surface is unchanged.** This round
   does not touch ``browser_broker.py``; the symbols the existing test suite
   relies on must still import.

All ``@pytest.mark.unit``.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

import yiagents

_EXEC_IMPORT_RE = re.compile(
    r"^\s*(?:from\s+yiagents\.execution\b|import\s+yiagents\.execution\b)"
)

# Directories whose import graphs must stay free of the execution layer.
_GUARDED_DIRS = [
    Path(yiagents.__file__).resolve().parent / "agents",
    Path(yiagents.__file__).resolve().parent / "graph",
]


def _python_files(roots):
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*.py"):
            if "__pycache__" in path.parts:
                continue
            yield path


@pytest.mark.unit
class TestExecutionDecoupledFromAgentsAndGraph:
    def test_no_execution_import_in_agents_or_graph(self):
        offenders = []
        for path in _python_files(_GUARDED_DIRS):
            for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                if _EXEC_IMPORT_RE.match(line):
                    offenders.append(f"{path}:{lineno}: {line.strip()}")
        assert not offenders, (
            "Execution layer must not be imported by agents or graph nodes "
            "(breaks default-off / byte-equivalence):\n" + "\n".join(offenders)
        )

    def test_bridge_imports_schemas_not_the_reverse(self):
        # bridge.py lives in execution/ and depends on agents/schemas (allowed,
        # execution -> analysis). Confirm the dependency is one-way by making
        # sure the import works and lives in execution, not agents.
        import yiagents.execution.bridge as bridge  # noqa: F401

        assert hasattr(bridge, "decision_to_order_requests")


@pytest.mark.unit
class TestBrowserBrokerSurfaceUnchanged:
    def test_public_symbols_still_importable(self):
        # The exact set tests/test_browser_broker.py relies on. This round
        # must not remove or rename any of them.
        # OrderResult is the order-result dataclass the broker returns; keep it
        # importable too (used by the future Track B adapter).
        from yiagents.execution.browser_broker import (  # noqa: F401
            BrowserBroker,
            KillSwitch,
            OrderAction,
            OrderResult,  # noqa: F401
            OrderStatus,
            _coerce_bool_env,
        )
