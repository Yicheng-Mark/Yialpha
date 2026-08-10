"""Provider-aware LLM read-timeout defaults.

Cloud providers (DeepSeek, OpenAI, xAI, …) get a 120s built-in timeout so a
half-open socket can't hang the batch forever. Local providers (Ollama,
openai_compatible) are exempt — slow local generation would be killed by a
ceiling. Precedence: caller kwarg > ``YIAGENTS_LLM_TIMEOUT_S`` > provider default.

These tests mock ``ChatOpenAI.__init__`` to capture the kwargs without making a
network call or requiring a real API key.
"""
import pytest

from yiagents.llm_clients.factory import create_llm_client
from yiagents.llm_clients.openai_client import _DEFAULT_CLOUD_TIMEOUT

# Providers that need an API key must see one, otherwise get_llm() raises before
# reaching the timeout logic. Use a dummy key for cloud providers.
_DUMMY_KEY = "sk-test-dummy"


def _captured_llm(monkeypatch, provider, *, model="m", base_url=None, env=None):
    """Create a client, intercept the ChatOpenAI kwargs, return the captured dict.

    ``env`` is an optional dict of env vars to set before constructing the client
    (YIAGENTS_LLM_TIMEOUT_S, provider API keys, etc.).
    """
    captured = {}

    # Patch ChatOpenAI.__init__ (the real base class every chat_class subclasses)
    # to capture kwargs without constructing a live HTTP client.
    from langchain_openai import ChatOpenAI

    def _fake_init(self, **kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(ChatOpenAI, "__init__", _fake_init)

    if env:
        for k, v in env.items():
            monkeypatch.setenv(k, v)

    client = create_llm_client(provider=provider, model=model, base_url=base_url)
    client.get_llm()
    return captured


@pytest.mark.unit
def test_cloud_provider_gets_default_timeout(monkeypatch):
    """DeepSeek (cloud) with no explicit timeout gets the 120s built-in default."""
    monkeypatch.delenv("YIAGENTS_LLM_TIMEOUT_S", raising=False)
    captured = _captured_llm(
        monkeypatch, "deepseek", env={"DEEPSEEK_API_KEY": _DUMMY_KEY}
    )
    assert captured["timeout"] == _DEFAULT_CLOUD_TIMEOUT


@pytest.mark.unit
def test_cloud_provider_openai_gets_default_timeout(monkeypatch):
    """Native OpenAI (cloud) also gets the default when timeout is unset."""
    monkeypatch.delenv("YIAGENTS_LLM_TIMEOUT_S", raising=False)
    captured = _captured_llm(
        monkeypatch, "openai", env={"OPENAI_API_KEY": _DUMMY_KEY}
    )
    assert captured["timeout"] == _DEFAULT_CLOUD_TIMEOUT


@pytest.mark.unit
def test_local_provider_ollama_no_timeout(monkeypatch):
    """Ollama is a local server: no read-timeout (slow local gen is expected)."""
    monkeypatch.delenv("YIAGENTS_LLM_TIMEOUT_S", raising=False)
    captured = _captured_llm(monkeypatch, "ollama")
    assert "timeout" not in captured


@pytest.mark.unit
def test_local_provider_openai_compatible_no_timeout(monkeypatch):
    """Generic openai_compatible (vLLM / LM Studio) is also local: no timeout."""
    monkeypatch.delenv("YIAGENTS_LLM_TIMEOUT_S", raising=False)
    monkeypatch.delenv("OPENAI_COMPATIBLE_API_KEY", raising=False)
    captured = _captured_llm(
        monkeypatch, "openai_compatible", base_url="http://localhost:8000/v1"
    )
    assert "timeout" not in captured


@pytest.mark.unit
def test_explicit_env_overrides_provider_default(monkeypatch):
    """YIAGENTS_LLM_TIMEOUT_S wins over the cloud default for cloud providers."""
    captured = _captured_llm(
        monkeypatch,
        "deepseek",
        env={"DEEPSEEK_API_KEY": _DUMMY_KEY, "YIAGENTS_LLM_TIMEOUT_S": "60"},
    )
    assert captured["timeout"] == 60.0


@pytest.mark.unit
def test_explicit_env_applies_to_local_provider(monkeypatch):
    """YIAGENTS_LLM_TIMEOUT_S also works for local providers (explicit override)."""
    captured = _captured_llm(
        monkeypatch,
        "ollama",
        env={"YIAGENTS_LLM_TIMEOUT_S": "300"},
    )
    assert captured["timeout"] == 300.0


@pytest.mark.unit
def test_caller_timeout_kwarg_wins(monkeypatch):
    """A caller-supplied timeout kwarg takes precedence over everything."""
    monkeypatch.setenv("YIAGENTS_LLM_TIMEOUT_S", "60")
    captured = {}

    from langchain_openai import ChatOpenAI

    def _fake_init(self, **kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(ChatOpenAI, "__init__", _fake_init)
    monkeypatch.setenv("DEEPSEEK_API_KEY", _DUMMY_KEY)

    client = create_llm_client(provider="deepseek", model="m", timeout=30)
    client.get_llm()
    assert captured["timeout"] == 30
