"""_invoke_or_stream: a zero-chunk stream must RAISE, not silently re-invoke.

The old fallback ``return self.graph.invoke(...)`` re-ran the entire graph
(4 analysts + debate + trader + 3-way risk debate + PM ≈ 10 min, many LLM
calls) whenever stream emitted nothing, double-billing the user with no signal.
Now it raises so the failure is visible. Normal streaming (≥1 chunk) and the
non-telemetry ``invoke`` path are byte-identical to before.
"""
import unittest

import pytest

from yiagents.graph.trading_graph import YiAgentsGraph


class _NoChunkGraph:
    """A graph whose stream() emits nothing (the regression scenario)."""
    invoke_calls = 0

    def stream(self, state, **args):  # noqa: ARG002
        return iter([])

    def invoke(self, state, **args):  # noqa: ARG002
        _NoChunkGraph.invoke_calls += 1
        return {"company_of_interest": "X"}


class _HealthyStreamGraph:
    """A graph whose stream() emits two value-chunks (normal path)."""
    def __init__(self):
        self.invoke_calls = 0

    def stream(self, state, **args):  # noqa: ARG002
        yield {"a": 1}
        yield {"a": 1, "b": 2}

    def invoke(self, state, **args):  # noqa: ARG002
        self.invoke_calls += 1
        return {"unexpected": True}


class _Stub:
    """Minimal self for calling the unbound _invoke_or_stream method."""
    def __init__(self, graph, telemetry):
        self.config = {"stream_telemetry": telemetry}
        self.selected_analysts = ["market", "social", "news", "fundamentals"]
        self.graph = graph


class TestInvokeOrStreamGuard(unittest.TestCase):
    def test_zero_chunks_raises_and_does_not_re_invoke(self):
        _NoChunkGraph.invoke_calls = 0  # reset shared class counter
        stub = _Stub(_NoChunkGraph(), telemetry=True)
        with self.assertRaises(RuntimeError):
            YiAgentsGraph._invoke_or_stream(stub, {}, {})
        # The silent full-graph re-invoke (double billing) must NOT happen.
        self.assertEqual(_NoChunkGraph.invoke_calls, 0)

    def test_healthy_stream_returns_last_chunk(self):
        g = _HealthyStreamGraph()
        stub = _Stub(g, telemetry=True)
        out = YiAgentsGraph._invoke_or_stream(stub, {}, {})
        # Last values chunk is the merged final state, exactly like invoke.
        self.assertEqual(out, {"a": 1, "b": 2})
        self.assertEqual(g.invoke_calls, 0)

    def test_telemetry_off_uses_plain_invoke(self):
        # Byte-equivalent baseline path: stream_telemetry off -> plain invoke,
        # no streaming at all.
        g = _NoChunkGraph()
        _NoChunkGraph.invoke_calls = 0
        stub = _Stub(g, telemetry=False)
        out = YiAgentsGraph._invoke_or_stream(stub, {"x": 1}, {})
        self.assertEqual(out, {"company_of_interest": "X"})
        self.assertEqual(_NoChunkGraph.invoke_calls, 1)


if __name__ == "__main__":
    unittest.main()
