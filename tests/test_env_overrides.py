"""Tests for YIAGENTS_* env-var overlay onto DEFAULT_CONFIG."""

from __future__ import annotations

import importlib

import pytest

import yiagents.default_config as default_config_module


def _reload_with_env(monkeypatch, **overrides):
    """Set/clear env vars then reload default_config to re-evaluate DEFAULT_CONFIG."""
    for key in list(default_config_module._ENV_OVERRIDES):
        monkeypatch.delenv(key, raising=False)
    for key, val in overrides.items():
        monkeypatch.setenv(key, val)
    return importlib.reload(default_config_module)


def test_no_env_uses_built_in_defaults(monkeypatch):
    dc = _reload_with_env(monkeypatch)
    assert dc.DEFAULT_CONFIG["llm_provider"] == "openai"
    assert dc.DEFAULT_CONFIG["deep_think_llm"] == "gpt-5.5"
    assert dc.DEFAULT_CONFIG["quick_think_llm"] == "gpt-5.4-mini"
    assert dc.DEFAULT_CONFIG["backend_url"] is None
    assert dc.DEFAULT_CONFIG["max_debate_rounds"] == 2
    assert dc.DEFAULT_CONFIG["checkpoint_enabled"] is False


def test_string_overrides(monkeypatch):
    dc = _reload_with_env(
        monkeypatch,
        YIAGENTS_LLM_PROVIDER="google",
        YIAGENTS_DEEP_THINK_LLM="gemini-3-pro-preview",
        YIAGENTS_QUICK_THINK_LLM="gemini-3-flash-preview",
        YIAGENTS_LLM_BACKEND_URL="https://example.invalid/v1",
        YIAGENTS_OUTPUT_LANGUAGE="Chinese",
    )
    assert dc.DEFAULT_CONFIG["llm_provider"] == "google"
    assert dc.DEFAULT_CONFIG["deep_think_llm"] == "gemini-3-pro-preview"
    assert dc.DEFAULT_CONFIG["quick_think_llm"] == "gemini-3-flash-preview"
    assert dc.DEFAULT_CONFIG["backend_url"] == "https://example.invalid/v1"
    assert dc.DEFAULT_CONFIG["output_language"] == "Chinese"


def test_int_coercion(monkeypatch):
    dc = _reload_with_env(
        monkeypatch,
        YIAGENTS_MAX_DEBATE_ROUNDS="3",
        YIAGENTS_MAX_RISK_ROUNDS="2",
    )
    assert dc.DEFAULT_CONFIG["max_debate_rounds"] == 3
    assert isinstance(dc.DEFAULT_CONFIG["max_debate_rounds"], int)
    assert dc.DEFAULT_CONFIG["max_risk_discuss_rounds"] == 2
    assert isinstance(dc.DEFAULT_CONFIG["max_risk_discuss_rounds"], int)


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("true", True), ("True", True), ("1", True), ("yes", True), ("on", True),
        ("false", False), ("False", False), ("0", False), ("no", False), ("off", False),
    ],
)
def test_bool_coercion(monkeypatch, raw, expected):
    dc = _reload_with_env(monkeypatch, YIAGENTS_CHECKPOINT_ENABLED=raw)
    assert dc.DEFAULT_CONFIG["checkpoint_enabled"] is expected


def test_reasoning_thinking_overrides(monkeypatch):
    """The provider reasoning/thinking knobs are env-configurable (non-interactive runs)."""
    dc = _reload_with_env(
        monkeypatch,
        YIAGENTS_OPENAI_REASONING_EFFORT="high",
        YIAGENTS_GOOGLE_THINKING_LEVEL="minimal",
        YIAGENTS_ANTHROPIC_EFFORT="low",
    )
    assert dc.DEFAULT_CONFIG["openai_reasoning_effort"] == "high"
    assert dc.DEFAULT_CONFIG["google_thinking_level"] == "minimal"
    assert dc.DEFAULT_CONFIG["anthropic_effort"] == "low"


def test_reasoning_effort_defaults_to_none(monkeypatch):
    """Unset reasoning/thinking knobs stay None so each provider uses its own default."""
    dc = _reload_with_env(monkeypatch)
    assert dc.DEFAULT_CONFIG["openai_reasoning_effort"] is None
    assert dc.DEFAULT_CONFIG["google_thinking_level"] is None
    assert dc.DEFAULT_CONFIG["anthropic_effort"] is None


def test_empty_env_value_is_passthrough(monkeypatch):
    """Empty YIAGENTS_* values must not clobber the built-in default."""
    dc = _reload_with_env(
        monkeypatch,
        YIAGENTS_LLM_PROVIDER="",
        YIAGENTS_MAX_DEBATE_ROUNDS="",
    )
    assert dc.DEFAULT_CONFIG["llm_provider"] == "openai"
    assert dc.DEFAULT_CONFIG["max_debate_rounds"] == 2


def test_invalid_int_raises(monkeypatch):
    """Garbage int values should surface a ValueError at import, not silently misconfigure."""
    monkeypatch.setenv("YIAGENTS_MAX_DEBATE_ROUNDS", "not-a-number")
    with pytest.raises(ValueError, match="YIAGENTS_MAX_DEBATE_ROUNDS"):
        importlib.reload(default_config_module)
    # Restore module state for subsequent tests in this process
    monkeypatch.delenv("YIAGENTS_MAX_DEBATE_ROUNDS", raising=False)
    importlib.reload(default_config_module)


@pytest.mark.parametrize("bad", ["treu", "flase", "maybe", "2", "enabled"])
def test_invalid_bool_raises(monkeypatch, bad):
    """A misspelled boolean must fail loudly (like ints) instead of silently False."""
    monkeypatch.setenv("YIAGENTS_CHECKPOINT_ENABLED", bad)
    with pytest.raises(ValueError, match="YIAGENTS_CHECKPOINT_ENABLED"):
        importlib.reload(default_config_module)
    monkeypatch.delenv("YIAGENTS_CHECKPOINT_ENABLED", raising=False)
    importlib.reload(default_config_module)


def test_unknown_env_var_is_ignored(monkeypatch):
    """Env vars outside _ENV_OVERRIDES must not bleed into DEFAULT_CONFIG."""
    dc = _reload_with_env(
        monkeypatch,
        YIAGENTS_NONEXISTENT_KEY="oops",
    )
    assert "nonexistent_key" not in dc.DEFAULT_CONFIG


def test_perf_parallel_defaults(monkeypatch):
    """T0/T1.3/T2 knobs default to off / byte-equivalent values when unset."""
    dc = _reload_with_env(monkeypatch)
    assert dc.DEFAULT_CONFIG["node_perf_telemetry"] is False
    assert dc.DEFAULT_CONFIG["llm_max_retries"] == 2          # == langchain default
    assert isinstance(dc.DEFAULT_CONFIG["llm_max_retries"], int)
    assert dc.DEFAULT_CONFIG["analyst_parallel"] is False     # gate stays shut
    assert dc.DEFAULT_CONFIG["analyst_parallel_max_threads"] == 16
    assert isinstance(dc.DEFAULT_CONFIG["analyst_parallel_max_threads"], int)


def test_perf_parallel_overrides(monkeypatch):
    """T0/T1.3/T2 env vars coerce bool/int correctly."""
    dc = _reload_with_env(
        monkeypatch,
        YIAGENTS_NODE_PERF_TELEMETRY="true",
        YIAGENTS_LLM_MAX_RETRIES="0",
        YIAGENTS_ANALYST_PARALLEL="on",
        YIAGENTS_ANALYST_PARALLEL_MAX_THREADS="8",
    )
    assert dc.DEFAULT_CONFIG["node_perf_telemetry"] is True
    assert dc.DEFAULT_CONFIG["llm_max_retries"] == 0
    assert isinstance(dc.DEFAULT_CONFIG["llm_max_retries"], int)
    assert dc.DEFAULT_CONFIG["analyst_parallel"] is True
    assert dc.DEFAULT_CONFIG["analyst_parallel_max_threads"] == 8


@pytest.mark.parametrize("bad", ["treu", "flase", "maybe", "2", "enabled"])
def test_invalid_analyst_parallel_bool_raises(monkeypatch, bad):
    """A misspelled analyst_parallel bool must fail loudly, like other bools."""
    monkeypatch.setenv("YIAGENTS_ANALYST_PARALLEL", bad)
    with pytest.raises(ValueError, match="YIAGENTS_ANALYST_PARALLEL"):
        importlib.reload(default_config_module)
    monkeypatch.delenv("YIAGENTS_ANALYST_PARALLEL", raising=False)
    importlib.reload(default_config_module)


def test_invalid_max_retries_int_raises(monkeypatch):
    """A non-numeric llm_max_retries must fail loudly at import."""
    monkeypatch.setenv("YIAGENTS_LLM_MAX_RETRIES", "not-a-number")
    with pytest.raises(ValueError, match="YIAGENTS_LLM_MAX_RETRIES"):
        importlib.reload(default_config_module)
    monkeypatch.delenv("YIAGENTS_LLM_MAX_RETRIES", raising=False)
    importlib.reload(default_config_module)
