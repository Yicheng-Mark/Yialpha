"""Tests for P0 (stream telemetry) and P1a (shared httpx keepalive).

Both are gated off by default; these lock (a) the default path is unchanged and
(b) the opt-in behaviour is correct.
"""

from unittest.mock import MagicMock

from yiagents.graph.trading_graph import YiAgentsGraph
from yiagents.llm_clients.http_client import get_shared_http_client, reset_for_test as reset_http

# ---------------------------------------------------------------------------
# P1a — shared httpx.Client keepalive
# ---------------------------------------------------------------------------


def test_http_client_singleton():
    reset_http()
    a = get_shared_http_client()
    b = get_shared_http_client()
    assert a is not None
    assert a is b  # one process-wide client


def test_provider_kwargs_http_keepalive_off_default():
    reset_http()
    mock = MagicMock(spec=YiAgentsGraph)
    mock.config = {"llm_provider": "deepseek"}
    kwargs = YiAgentsGraph._get_provider_kwargs(mock)
    assert "http_client" not in kwargs  # default behaviour unchanged


def test_provider_kwargs_http_keepalive_on_deepseek():
    reset_http()
    mock = MagicMock(spec=YiAgentsGraph)
    mock.config = {"llm_provider": "deepseek", "http_keepalive": True}
    kwargs = YiAgentsGraph._get_provider_kwargs(mock)
    assert "http_client" in kwargs
    assert kwargs["http_client"] is get_shared_http_client()  # the shared singleton


def test_provider_kwargs_http_keepalive_skips_non_openai_provider():
    """google isn't OpenAI-compatible — no http_client attached even with the flag."""
    reset_http()
    mock = MagicMock(spec=YiAgentsGraph)
    mock.config = {"llm_provider": "google", "http_keepalive": True}
    kwargs = YiAgentsGraph._get_provider_kwargs(mock)
    assert "http_client" not in kwargs


# ---------------------------------------------------------------------------
# P0 — stream telemetry in _invoke_or_stream
# ---------------------------------------------------------------------------


def _graph_mock(config):
    """Plain mock with explicit .graph (spec=YiAgentsGraph hides instance attrs)."""
    mock = MagicMock()
    mock.config = config
    mock.graph = MagicMock()
    return mock


def test_invoke_or_stream_off_uses_invoke():
    """Default (telemetry off) calls graph.invoke, never stream."""
    mock = _graph_mock({})
    mock.graph.invoke.return_value = {"final_trade_decision": "BUY"}
    result = YiAgentsGraph._invoke_or_stream(mock, {"s": 1}, {"stream_mode": "values"})
    mock.graph.invoke.assert_called_once()
    mock.graph.stream.assert_not_called()
    assert result == {"final_trade_decision": "BUY"}


def test_invoke_or_stream_on_uses_stream_returns_last_chunk():
    """Telemetry on streams and returns the last values chunk (= invoke's result)."""
    mock = _graph_mock({"stream_telemetry": True})
    mock.selected_analysts = ("market", "social", "news", "fundamentals")
    chunks = [
        {"market_report": "a"},
        {"market_report": "a", "sentiment_report": "b"},
        {"final_trade_decision": "BUY"},
    ]
    mock.graph.stream.return_value = iter(chunks)
    result = YiAgentsGraph._invoke_or_stream(mock, {"s": 1}, {"stream_mode": "values"})
    mock.graph.stream.assert_called_once()
    mock.graph.invoke.assert_not_called()
    assert result == {"final_trade_decision": "BUY"}  # last chunk


def test_invoke_or_stream_empty_falls_back_to_invoke():
    """A run that emits no chunks falls back to invoke (no KeyError downstream)."""
    mock = _graph_mock({"stream_telemetry": True})
    mock.selected_analysts = ("market",)
    mock.graph.stream.return_value = iter([])
    mock.graph.invoke.return_value = {"final_trade_decision": "BUY"}
    result = YiAgentsGraph._invoke_or_stream(mock, {"s": 1}, {"stream_mode": "values"})
    mock.graph.invoke.assert_called_once()
    assert result == {"final_trade_decision": "BUY"}
