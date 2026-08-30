"""Provider-aware LLM read-timeout defaults.

Cloud providers (DeepSeek, OpenAI, xAI, …) get a 120s built-in timeout so a
half-open socket can't hang the batch forever. Local providers (Ollama,
openai_compatible) are exempt — slow local generation would be killed by a
ceiling. Precedence: caller kwarg > ``YIALPHA_LLM_TIMEOUT_S`` > provider default.

These tests mock ``ChatOpenAI.__init__`` to capture the kwargs without making a
network call or requiring a real API key.
"""
import pytest

from yialpha.llm_clients._timeout import _DEFAULT_CLOUD_TIMEOUT
from yialpha.llm_clients.factory import create_llm_client

# Providers that need an API key must see one, otherwise get_llm() raises before
# reaching the timeout logic. Use a dummy key for cloud providers.
_DUMMY_KEY = "sk-test-dummy"


def _captured_llm(monkeypatch, provider, *, model="m", base_url=None, env=None):
    """Create a client, intercept the ChatOpenAI kwargs, return the captured dict.

    ``env`` is an optional dict of env vars to set before constructing the client
    (YIALPHA_LLM_TIMEOUT_S, provider API keys, etc.).
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
    monkeypatch.delenv("YIALPHA_LLM_TIMEOUT_S", raising=False)
    captured = _captured_llm(
        monkeypatch, "deepseek", env={"DEEPSEEK_API_KEY": _DUMMY_KEY}
    )
    assert captured["timeout"] == _DEFAULT_CLOUD_TIMEOUT


@pytest.mark.unit
def test_cloud_provider_openai_gets_default_timeout(monkeypatch):
    """Native OpenAI (cloud) also gets the default when timeout is unset."""
    monkeypatch.delenv("YIALPHA_LLM_TIMEOUT_S", raising=False)
    captured = _captured_llm(
        monkeypatch, "openai", env={"OPENAI_API_KEY": _DUMMY_KEY}
    )
    assert captured["timeout"] == _DEFAULT_CLOUD_TIMEOUT


@pytest.mark.unit
def test_local_provider_ollama_no_timeout(monkeypatch):
    """Ollama is a local server: no read-timeout (slow local gen is expected)."""
    monkeypatch.delenv("YIALPHA_LLM_TIMEOUT_S", raising=False)
    captured = _captured_llm(monkeypatch, "ollama")
    assert "timeout" not in captured


@pytest.mark.unit
def test_local_provider_openai_compatible_no_timeout(monkeypatch):
    """Generic openai_compatible (vLLM / LM Studio) is also local: no timeout."""
    monkeypatch.delenv("YIALPHA_LLM_TIMEOUT_S", raising=False)
    monkeypatch.delenv("OPENAI_COMPATIBLE_API_KEY", raising=False)
    captured = _captured_llm(
        monkeypatch, "openai_compatible", base_url="http://localhost:8000/v1"
    )
    assert "timeout" not in captured


@pytest.mark.unit
def test_explicit_env_overrides_provider_default(monkeypatch):
    """YIALPHA_LLM_TIMEOUT_S wins over the cloud default for cloud providers."""
    captured = _captured_llm(
        monkeypatch,
        "deepseek",
        env={"DEEPSEEK_API_KEY": _DUMMY_KEY, "YIALPHA_LLM_TIMEOUT_S": "60"},
    )
    assert captured["timeout"] == 60.0


@pytest.mark.unit
def test_explicit_env_applies_to_local_provider(monkeypatch):
    """YIALPHA_LLM_TIMEOUT_S also works for local providers (explicit override)."""
    captured = _captured_llm(
        monkeypatch,
        "ollama",
        env={"YIALPHA_LLM_TIMEOUT_S": "300"},
    )
    assert captured["timeout"] == 300.0


@pytest.mark.unit
def test_caller_timeout_kwarg_wins(monkeypatch):
    """A caller-supplied timeout kwarg takes precedence over everything."""
    monkeypatch.setenv("YIALPHA_LLM_TIMEOUT_S", "60")
    captured = {}

    from langchain_openai import ChatOpenAI

    def _fake_init(self, **kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(ChatOpenAI, "__init__", _fake_init)
    monkeypatch.setenv("DEEPSEEK_API_KEY", _DUMMY_KEY)

    client = create_llm_client(provider="deepseek", model="m", timeout=30)
    client.get_llm()
    assert captured["timeout"] == 30


# ---------------------------------------------------------------------------
# Gap A — a non-numeric YIALPHA_LLM_TIMEOUT_S must NOT be silently swallowed.
#
# Before the fix, ``contextlib.suppress(ValueError)`` around ``float(env)`` ate
# the bad value AND skipped the cloud-default branch (the env was truthy), so
# the call ran with NO timeout and NO warning — a silent-degradation fail-open.
# Now it logs a WARNING and falls through to the local-exempt / cloud-default.
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_non_numeric_env_warns_and_falls_back_to_cloud_default(monkeypatch, caplog):
    """Bad YIALPHA_LLM_TIMEOUT_S warns and uses the 120s cloud default."""
    import logging

    from yialpha.llm_clients import _timeout as _t
    # Reset the one-shot cloud-default info guard so this test is independent.
    monkeypatch.setattr(_t, "_timeout_warned", False)

    captured = _captured_llm(
        monkeypatch,
        "deepseek",
        env={"DEEPSEEK_API_KEY": _DUMMY_KEY, "YIALPHA_LLM_TIMEOUT_S": "120s"},
    )
    assert captured["timeout"] == _DEFAULT_CLOUD_TIMEOUT
    assert any(
        "not a positive finite number" in r.message and "120s" in r.message
        for r in caplog.records
        if r.levelno >= logging.WARNING
    )


@pytest.mark.unit
@pytest.mark.parametrize("bad", ["0", "-5", "nan", "inf"])
def test_non_positive_env_warns_and_falls_back(monkeypatch, caplog, bad):
    """0/negative/NaN/inf parse fine but are not usable timeouts (0 => every
    call times out immediately); they must warn and fall back like garbage."""
    import logging

    captured = _captured_llm(
        monkeypatch,
        "deepseek",
        env={"DEEPSEEK_API_KEY": _DUMMY_KEY, "YIALPHA_LLM_TIMEOUT_S": bad},
    )
    assert captured["timeout"] == _DEFAULT_CLOUD_TIMEOUT
    assert any(
        "not a positive finite number" in r.message
        for r in caplog.records
        if r.levelno >= logging.WARNING
    )


@pytest.mark.unit
def test_non_numeric_env_warns_and_local_stays_timeoutless(monkeypatch, caplog):
    """Bad env on a local provider warns and stays timeout-less (exempt)."""
    import logging

    captured = _captured_llm(
        monkeypatch,
        "ollama",
        env={"YIALPHA_LLM_TIMEOUT_S": "120s"},
    )
    assert "timeout" not in captured
    assert any(
        "not a positive finite number" in r.message for r in caplog.records
        if r.levelno >= logging.WARNING
    )


# ---------------------------------------------------------------------------
# Gap B — the provider-aware timeout must apply to ALL native clients, not just
# the OpenAI-compatible family. Before the fix, Anthropic / Google / Azure /
# Bedrock only forwarded a caller-supplied timeout and had NO env fallback or
# cloud default, so a half-open socket could hang the batch indefinitely.
#
# Each native provider's chat base class is patched (not just ChatOpenAI), then
# the same 3 core scenarios as the OpenAI family are asserted.
# ---------------------------------------------------------------------------

def _native_captured_llm(monkeypatch, provider, *, model, env=None, client_kwargs=None):
    """Like _captured_llm but patches the native provider's chat base class.

    ``provider`` must be one of anthropic / google / azure. Bedrock is covered
    separately (langchain-aws is an optional extra not installed in CI).
    """
    import importlib

    _BASE = {
        "anthropic": ("langchain_anthropic", "ChatAnthropic"),
        "google": ("langchain_google_genai", "ChatGoogleGenerativeAI"),
        "azure": ("langchain_openai", "AzureChatOpenAI"),
    }
    mod_name, cls_name = _BASE[provider]
    base_cls = getattr(importlib.import_module(mod_name), cls_name)

    captured = {}

    def _fake_init(self, **kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(base_cls, "__init__", _fake_init)

    if env:
        for k, v in env.items():
            monkeypatch.setenv(k, v)

    client = create_llm_client(
        provider=provider, model=model, **(client_kwargs or {})
    )
    client.get_llm()
    return captured


@pytest.mark.unit
def test_anthropic_cloud_default_timeout(monkeypatch):
    """Anthropic gets the 120s cloud default when timeout is unset (gap B)."""
    monkeypatch.delenv("YIALPHA_LLM_TIMEOUT_S", raising=False)
    captured = _native_captured_llm(
        monkeypatch, "anthropic",
        model="claude-3-5-sonnet-20240620",
        env={"ANTHROPIC_API_KEY": _DUMMY_KEY},
    )
    assert captured["timeout"] == _DEFAULT_CLOUD_TIMEOUT


@pytest.mark.unit
def test_anthropic_env_overrides_default(monkeypatch):
    """YIALPHA_LLM_TIMEOUT_S overrides the default for Anthropic."""
    captured = _native_captured_llm(
        monkeypatch, "anthropic",
        model="claude-3-5-sonnet-20240620",
        env={
            "ANTHROPIC_API_KEY": _DUMMY_KEY,
            "YIALPHA_LLM_TIMEOUT_S": "45",
        },
    )
    assert captured["timeout"] == 45.0


@pytest.mark.unit
def test_anthropic_caller_kwarg_wins(monkeypatch):
    """A caller timeout kwarg beats env for Anthropic."""
    monkeypatch.setenv("YIALPHA_LLM_TIMEOUT_S", "45")
    captured = _native_captured_llm(
        monkeypatch, "anthropic",
        model="claude-3-5-sonnet-20240620",
        env={"ANTHROPIC_API_KEY": _DUMMY_KEY},
        client_kwargs={"timeout": 15},
    )
    assert captured["timeout"] == 15


@pytest.mark.unit
def test_google_cloud_default_timeout(monkeypatch):
    """Google Gemini gets the 120s cloud default when timeout is unset (gap B)."""
    monkeypatch.delenv("YIALPHA_LLM_TIMEOUT_S", raising=False)
    captured = _native_captured_llm(
        monkeypatch, "google",
        model="gemini-1.5-pro",
        env={"GOOGLE_API_KEY": _DUMMY_KEY},
    )
    assert captured["timeout"] == _DEFAULT_CLOUD_TIMEOUT


@pytest.mark.unit
def test_google_env_overrides_default(monkeypatch):
    """YIALPHA_LLM_TIMEOUT_S overrides the default for Google."""
    captured = _native_captured_llm(
        monkeypatch, "google",
        model="gemini-1.5-pro",
        env={"GOOGLE_API_KEY": _DUMMY_KEY, "YIALPHA_LLM_TIMEOUT_S": "50"},
    )
    assert captured["timeout"] == 50.0


@pytest.mark.unit
def test_google_caller_kwarg_wins(monkeypatch):
    """A caller timeout kwarg beats env for Google."""
    monkeypatch.setenv("YIALPHA_LLM_TIMEOUT_S", "50")
    captured = _native_captured_llm(
        monkeypatch, "google",
        model="gemini-1.5-pro",
        env={"GOOGLE_API_KEY": _DUMMY_KEY},
        client_kwargs={"timeout": 20},
    )
    assert captured["timeout"] == 20


@pytest.mark.unit
def test_azure_cloud_default_timeout(monkeypatch):
    """Azure gets the 120s cloud default when timeout is unset (gap B)."""
    monkeypatch.delenv("YIALPHA_LLM_TIMEOUT_S", raising=False)
    captured = _native_captured_llm(
        monkeypatch, "azure",
        model="gpt-4",
        env={
            "AZURE_OPENAI_API_KEY": _DUMMY_KEY,
            "AZURE_OPENAI_ENDPOINT": "https://x.openai.azure.com",
            "AZURE_OPENAI_DEPLOYMENT_NAME": "dep",
            "OPENAI_API_VERSION": "2024-01-01",
        },
    )
    assert captured["timeout"] == _DEFAULT_CLOUD_TIMEOUT


@pytest.mark.unit
def test_azure_env_overrides_default(monkeypatch):
    """YIALPHA_LLM_TIMEOUT_S overrides the default for Azure."""
    captured = _native_captured_llm(
        monkeypatch, "azure",
        model="gpt-4",
        env={
            "AZURE_OPENAI_API_KEY": _DUMMY_KEY,
            "AZURE_OPENAI_ENDPOINT": "https://x.openai.azure.com",
            "AZURE_OPENAI_DEPLOYMENT_NAME": "dep",
            "OPENAI_API_VERSION": "2024-01-01",
            "YIALPHA_LLM_TIMEOUT_S": "55",
        },
    )
    assert captured["timeout"] == 55.0


@pytest.mark.unit
def test_azure_caller_kwarg_wins(monkeypatch):
    """A caller timeout kwarg beats env for Azure."""
    monkeypatch.setenv("YIALPHA_LLM_TIMEOUT_S", "55")
    captured = _native_captured_llm(
        monkeypatch, "azure",
        model="gpt-4",
        env={
            "AZURE_OPENAI_API_KEY": _DUMMY_KEY,
            "AZURE_OPENAI_ENDPOINT": "https://x.openai.azure.com",
            "AZURE_OPENAI_DEPLOYMENT_NAME": "dep",
            "OPENAI_API_VERSION": "2024-01-01",
        },
        client_kwargs={"timeout": 25},
    )
    assert captured["timeout"] == 25


# ---------------------------------------------------------------------------
# Bedrock — the [bedrock] extra (langchain-aws) is optional and not installed
# in the default / CI dev environment. The helper wiring is verified at the
# source level instead (resolve_timeout call present + timeout in passthrough).
# ---------------------------------------------------------------------------

def test_bedrock_get_llm_wires_timeout_helper():
    """Bedrock get_llm() reaches resolve_timeout (gap B wiring, source-level).

    langchain-aws is an optional extra absent from the default install, so we
    assert the wiring statically. Since the 2026-08-16 client dedup, the call
    site is ``apply_passthrough_kwargs`` (base_client), which routes through
    resolve_timeout with the provider name; ``timeout`` must stay in its
    passthrough tuple.
    """
    from pathlib import Path

    src = Path(__file__).resolve().parent.parent / "yialpha" / "llm_clients" / "bedrock_client.py"
    text = src.read_text(encoding="utf-8")
    assert "from .base_client import" in text
    assert "apply_passthrough_kwargs(" in text
    assert '"bedrock"' in text                # provider name for resolve_timeout
    assert '"timeout"' in text                # in the passthrough tuple

    # The helper itself must keep routing to resolve_timeout (cloud default).
    base_src = (
        Path(__file__).resolve().parent.parent
        / "yialpha" / "llm_clients" / "base_client.py"
    ).read_text(encoding="utf-8")
    assert "resolve_timeout(llm_kwargs, is_local=False, provider_name=timeout_provider)" in base_src

