"""Tests for the optional shared LLM rate limiter (Phase C)."""

from unittest.mock import MagicMock

from yiagents.graph.trading_graph import YiAgentsGraph
from yiagents.llm_clients.rate_limiter import get_shared_rate_limiter, reset_for_test


def test_singleton_shared_per_key():
    reset_for_test()
    a = get_shared_rate_limiter("deepseek", 60)
    b = get_shared_rate_limiter("deepseek", 60)
    assert a is b  # same (provider, rpm) → one process-wide limiter


def test_singleton_distinct_keys():
    reset_for_test()
    a = get_shared_rate_limiter("deepseek", 60)
    b = get_shared_rate_limiter("deepseek", 120)
    c = get_shared_rate_limiter("openai", 60)
    assert a is not b
    assert a is not c


def test_provider_kwargs_off_by_default():
    """No rate limiter unless explicitly enabled — default behavior unchanged."""
    reset_for_test()
    mock = MagicMock(spec=YiAgentsGraph)
    mock.config = {"llm_provider": "deepseek"}
    kwargs = YiAgentsGraph._get_provider_kwargs(mock)
    assert "rate_limiter" not in kwargs


def test_provider_kwargs_on_attaches_shared_limiter():
    """Flag on → the shared singleton is attached for an OpenAI-compatible provider."""
    reset_for_test()
    mock = MagicMock(spec=YiAgentsGraph)
    mock.config = {"llm_provider": "deepseek", "llm_rate_limiter": True, "llm_rpm": 90}
    kwargs = YiAgentsGraph._get_provider_kwargs(mock)
    assert "rate_limiter" in kwargs
    assert kwargs["rate_limiter"] is get_shared_rate_limiter("deepseek", 90)
