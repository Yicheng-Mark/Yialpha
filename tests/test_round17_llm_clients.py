"""Round-17 audit tests for the llm_clients shard.

Hermetic: no provider is contacted. The chat classes of the native / optional
SDKs are replaced by a recorder standing at the seam each client module uses,
``httpx.MockTransport`` answers the key pool, and the process-wide singletons
(key pool, keep-alive client, rate limiter, the timeout one-shot flag) are reset
around every test. Each test pins ONE contract named in its docstring.
"""

from __future__ import annotations

import json
import logging
import re
import ssl
import sys
import threading
import types
import warnings
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, ClassVar

import httpx
import pytest
from langchain_core.globals import get_llm_cache, set_llm_cache
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, Generation
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from yialpha.llm_clients import (
    _timeout,
    anthropic_client,
    api_key_env,
    base_client,
    bedrock_client,
    capabilities,
    factory,
    google_client,
    http_client,
    key_pool,
    model_catalog,
    openai_client,
    rate_limiter,
    response_cache,
    validators,
)
from yialpha.llm_clients.anthropic_client import AnthropicClient
from yialpha.llm_clients.azure_client import AzureOpenAIClient
from yialpha.llm_clients.bedrock_client import BedrockClient
from yialpha.llm_clients.google_client import GoogleClient
from yialpha.llm_clients.openai_client import (
    DeepSeekChatOpenAI,
    NormalizedChatOpenAI,
    OpenAIClient,
)
from yialpha.llm_clients.response_cache import DiskLLMCache, _key


class _RecordingChat:
    """Stand-in for a provider's chat class: records every construction."""

    instances: ClassVar[list[_RecordingChat]] = []

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        type(self).instances.append(self)

    def invoke(self, input: Any, config: Any = None, **kwargs: Any) -> Any:
        return types.SimpleNamespace(
            content=[
                {"type": "reasoning", "text": "thinking"},
                {"type": "text", "text": "hello"},
                "raw block",
                42,
            ]
        )


# The single-key variables of the pooled providers must stay UNSET unless a test
# sets one: api_key_pool merges them into the pool.
_POOLS_SINGLE_KEY_VARS = ("ZHIPU_API_KEY", "ZHIPU_CN_API_KEY")
_KEY_VARS = tuple(
    sorted(
        {name for name in api_key_env.PROVIDER_API_KEY_ENV.values() if name}
        - set(_POOLS_SINGLE_KEY_VARS)
    )
)
_POOL_AND_TUNING_VARS = (
    "ZHIPU_API_KEYS",
    "ZHIPU_CN_API_KEYS",
    *_POOLS_SINGLE_KEY_VARS,
    "YIALPHA_LLM_KEY_COOLDOWN_S",
    "YIALPHA_LLM_TIMEOUT_S",
    "YIALPHA_LLM_POOL_STREAMING",
    "YIALPHA_LLM_POOL_STREAM_USAGE",
    "ZHIPU_BASE_URL",
    "ZHIPU_CN_BASE_URL",
    "OLLAMA_BASE_URL",
    "AWS_REGION",
    "AWS_DEFAULT_REGION",
    "AZURE_OPENAI_DEPLOYMENT_NAME",
)


@pytest.fixture(autouse=True)
def _clean_llm_env(monkeypatch, _dummy_api_keys):
    """Start from the documented defaults, whatever the dev-box ``.env`` baked in.

    The real ``.env`` sets ``YIALPHA_LLM_TIMEOUT_S`` and a four-key
    ``ZHIPU_CN_API_KEYS``; a test that does not opt in explicitly must not see
    either (the pool would engage with real keys in memory). conftest keeps a single
    key variable's REAL value when one is set, so every provider's key variable is
    replaced by a fixed dummy here (the pooled providers' single-key variables are
    removed instead — they would join the pool): no object this file builds ever
    holds a real key, and an assertion diff can never print one.

    Takes ``_dummy_api_keys`` as an explicit argument so it is guaranteed to run
    after conftest's own pass over the key variables.
    """
    for name in _POOL_AND_TUNING_VARS:
        monkeypatch.delenv(name, raising=False)
    for name in _KEY_VARS:
        monkeypatch.setenv(name, "r17-test-key")


@pytest.fixture(autouse=True)
def _reset_process_wide_state(monkeypatch):
    """Nothing a previous test left in a process-wide singleton may leak in."""
    for mod in (key_pool, http_client, rate_limiter):
        mod.reset_for_test()
    monkeypatch.setattr(_timeout, "_timeout_warned", False)
    monkeypatch.setattr(bedrock_client, "_BEDROCK_CLASS", None)
    yield
    for mod in (key_pool, http_client, rate_limiter):
        mod.reset_for_test()


@pytest.fixture(autouse=True)
def _live_openai_client_classes(monkeypatch):
    """``tests/test_ollama_base_url.py`` reloads ``yialpha.llm_clients.openai_client``
    and leaves the new module in ``sys.modules``: clients built afterwards are that
    module's classes, while the names this file imported at collection time are the
    old ones. Rebind them to the live module for each test (restored afterwards), so
    identity checks hold whatever order the suite runs in."""
    live = sys.modules["yialpha.llm_clients.openai_client"]
    this = sys.modules[__name__]
    for name in ("OpenAIClient", "NormalizedChatOpenAI", "DeepSeekChatOpenAI"):
        monkeypatch.setattr(this, name, getattr(live, name))
    monkeypatch.setattr(this, "openai_client", live)


@pytest.fixture(autouse=True)
def _cheap_client_construction(monkeypatch):
    """Nothing in this file opens a socket, yet every real ``httpx.Client`` built here
    (the shared keep-alive client, the key pool's) creates one SSL context per
    transport and loads the system CA store into it — ~110 ms each on this Windows
    host, which made this the slowest file of the suite. Hand out an empty context
    instead: routing, timeouts and key rotation do not depend on it."""
    monkeypatch.setattr(
        ssl,
        "create_default_context",
        lambda *args, **kwargs: ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT),
    )


@pytest.fixture()
def fake_bedrock_extra(monkeypatch) -> Iterator[types.ModuleType]:
    """``langchain_aws`` present, as a recorder: the optional extra is installed."""
    fake = types.ModuleType("langchain_aws")
    fake.ChatBedrockConverse = _RecordingChat
    monkeypatch.setitem(sys.modules, "langchain_aws", fake)
    yield fake


# ===========================================================================
# bedrock_client — the optional extra is imported on demand
# ===========================================================================
class TestBedrockLazyClass:
    def test_r17_bedrock_class_is_built_once_and_cached(self, fake_bedrock_extra, monkeypatch):
        """The normalized class is built from ``langchain_aws`` exactly once; later
        calls return the cached object without consulting ``sys.modules`` again."""
        first = bedrock_client._bedrock_class()
        assert bedrock_client._BEDROCK_CLASS is first
        monkeypatch.setitem(sys.modules, "langchain_aws", None)  # importing now raises
        assert bedrock_client._bedrock_class() is first

    def test_r17_bedrock_class_subclasses_the_sdk_class_and_normalizes_content(
        self, fake_bedrock_extra
    ):
        """The returned class is the SDK's chat class behind the normalizing mixin
        (readable name, documented behaviour): text blocks and plain strings are
        joined, reasoning blocks and anything else are dropped."""
        cls = bedrock_client._bedrock_class()
        assert cls.__name__ == "NormalizedChatBedrockConverse"
        assert cls.__doc__ == "ChatBedrockConverse with normalized (string) content output."
        assert cls.__mro__[:3] == (
            cls,
            base_client._NormalizedInvokeMixin,
            fake_bedrock_extra.ChatBedrockConverse,
        )
        chat = cls(model="m", region_name="us-west-2")
        assert chat.invoke([{"role": "user", "content": "hi"}]).content == "hello\nraw block"

    def test_r17_bedrock_class_import_error_names_the_extra_and_is_not_cached(
        self, monkeypatch
    ):
        """A missing extra raises an ImportError that names the pip extra, chains the
        original error, and leaves nothing cached — a later install is picked up."""
        monkeypatch.setitem(sys.modules, "langchain_aws", None)
        with pytest.raises(ImportError, match=r"langchain-aws.*yialpha\[bedrock\]") as caught:
            bedrock_client._bedrock_class()
        assert bedrock_client._BEDROCK_CLASS is None
        assert isinstance(caught.value.__cause__, ImportError)

        fake = types.ModuleType("langchain_aws")
        fake.ChatBedrockConverse = _RecordingChat
        monkeypatch.setitem(sys.modules, "langchain_aws", fake)
        assert issubclass(bedrock_client._bedrock_class(), _RecordingChat)

    def test_r17_bedrock_get_llm_without_the_extra_raises_the_named_import_error(
        self, monkeypatch
    ):
        """The factory constructs the client (no SDK import); ``get_llm`` is where the
        missing extra surfaces, with the install hint."""
        monkeypatch.setitem(sys.modules, "langchain_aws", None)
        with pytest.raises(ImportError, match=r'pip install "yialpha\[bedrock\]"'):
            factory.create_llm_client("bedrock", "us.anthropic.claude-opus-4-8-v1:0").get_llm()


class TestBedrockGetLlm:
    @pytest.mark.parametrize(
        ("env", "expected"),
        [
            ({}, "us-west-2"),
            ({"AWS_REGION": "eu-west-1"}, "eu-west-1"),
            ({"AWS_DEFAULT_REGION": "ap-southeast-2"}, "ap-southeast-2"),
            ({"AWS_REGION": "eu-west-1", "AWS_DEFAULT_REGION": "ap-southeast-2"}, "eu-west-1"),
            ({"AWS_REGION": "", "AWS_DEFAULT_REGION": "ap-southeast-2"}, "ap-southeast-2"),
            ({"AWS_REGION": "", "AWS_DEFAULT_REGION": ""}, "us-west-2"),
        ],
    )
    def test_r17_bedrock_get_llm_region_precedence(
        self, fake_bedrock_extra, monkeypatch, env, expected
    ):
        """Region: ``AWS_REGION`` > ``AWS_DEFAULT_REGION`` > ``us-west-2``; an empty
        variable counts as unset."""
        for name, value in env.items():
            monkeypatch.setenv(name, value)
        BedrockClient("us.anthropic.claude-opus-4-8-v1:0").get_llm()
        assert _RecordingChat.instances[-1].kwargs == {
            "model": "us.anthropic.claude-opus-4-8-v1:0",
            "region_name": expected,
            "timeout": 120.0,
        }

    def test_r17_bedrock_get_llm_forwards_exactly_the_whitelisted_kwargs(
        self, fake_bedrock_extra
    ):
        """Five kwargs pass through (``temperature``, ``max_tokens``, ``max_retries``,
        ``callbacks``, ``timeout``); everything else — including ``base_url``, which
        the constructor accepts — is dropped."""
        sentinel = object()
        BedrockClient(
            "my-model-id",
            "https://ignored.example",
            temperature=0.3,
            max_tokens=2048,
            max_retries=4,
            callbacks=[sentinel],
            timeout=30.0,
            api_key="not-a-real-aws-key",
            reasoning_effort="low",
            effort="high",
            top_p=0.9,
            streaming=True,
            http_client=sentinel,
            thinking_level="minimal",
        ).get_llm()
        assert _RecordingChat.instances[-1].kwargs == {
            "model": "my-model-id",
            "region_name": "us-west-2",
            "temperature": 0.3,
            "max_tokens": 2048,
            "max_retries": 4,
            "callbacks": [sentinel],
            "timeout": 30.0,  # the caller's value wins over env / cloud default
        }

    def test_r17_bedrock_get_llm_timeout_follows_the_env_when_not_given(
        self, fake_bedrock_extra, monkeypatch
    ):
        """Without a caller ``timeout`` the shared precedence chain fills it in: the env
        value here, the 120s cloud default when unset."""
        monkeypatch.setenv("YIALPHA_LLM_TIMEOUT_S", "45")
        BedrockClient("m").get_llm()
        assert _RecordingChat.instances[-1].kwargs["timeout"] == 45.0
        monkeypatch.delenv("YIALPHA_LLM_TIMEOUT_S")
        BedrockClient("m").get_llm()
        assert _RecordingChat.instances[-1].kwargs["timeout"] == 120.0

    def test_r17_bedrock_accepts_any_model_id_without_warning(self, fake_bedrock_extra):
        """Bedrock model IDs / inference-profile IDs are user-defined: nothing is
        ``unknown``, so no RuntimeWarning and construction still works."""
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            for model in ("my-model-id", "us.anthropic.claude-opus-4-8-v1:0", "not-a-model"):
                assert BedrockClient(model).validate_model() is True
                assert BedrockClient(model).get_llm().kwargs["model"] == model


# ===========================================================================
# factory — routing and the unsupported-provider error
# ===========================================================================
_MODULE_OF_CLIENT = {
    "AnthropicClient": "anthropic_client",
    "GoogleClient": "google_client",
    "AzureOpenAIClient": "azure_client",
    "BedrockClient": "bedrock_client",
    "OpenAIClient": "openai_client",
}


class TestFactory:
    @pytest.mark.parametrize(
        ("provider", "class_name", "expected_provider"),
        [
            ("anthropic", "AnthropicClient", None),
            ("Anthropic", "AnthropicClient", None),
            ("google", "GoogleClient", None),
            ("azure", "AzureOpenAIClient", None),
            ("bedrock", "BedrockClient", None),
            ("deepseek", "OpenAIClient", "deepseek"),
            ("DeepSeek", "OpenAIClient", "deepseek"),
            ("GLM-cn", "OpenAIClient", "glm-cn"),
            ("ollama", "OpenAIClient", "ollama"),
        ],
    )
    def test_r17_factory_routes_each_provider_and_forwards_arguments(
        self, provider, class_name, expected_provider
    ):
        """Provider names are case-insensitive; model, base_url and extra kwargs reach
        the client unchanged; only the OpenAI-compatible family stores a provider tag
        (already lowercased).

        The expected class is looked up when the test runs, not when it is collected:
        ``tests/test_ollama_base_url.py`` reloads ``openai_client`` without restoring
        it, so a class object captured at collection can be a stale one."""
        client = factory.create_llm_client(
            provider, "some-model", "https://bu.example", effort="low"
        )
        module = sys.modules[f"yialpha.llm_clients.{_MODULE_OF_CLIENT[class_name]}"]
        assert type(client) is getattr(module, class_name)
        assert client.model == "some-model"
        assert client.base_url == "https://bu.example"
        assert client.kwargs == {"effort": "low"}
        if expected_provider is None:
            assert not hasattr(client, "provider")
        else:
            assert client.provider == expected_provider

    def test_r17_factory_returns_the_default_base_url_when_none_is_given(self):
        client = factory.create_llm_client("anthropic", "claude-opus-4-8")
        assert client.base_url is None

    @pytest.mark.parametrize("provider", ["nonexistent", "claude", " Azure", "azure-openai", ""])
    def test_r17_factory_unsupported_provider_names_the_original_input(self, provider):
        """An unknown provider raises ValueError carrying the operator's own spelling."""
        with pytest.raises(ValueError) as caught:
            factory.create_llm_client(provider, "m")
        assert str(caught.value) == f"Unsupported LLM provider: {provider}"


# ===========================================================================
# google_client / anthropic_client / azure_client — base_url and kwargs
# ===========================================================================
class TestGoogleGetLlm:
    def test_r17_google_get_llm_forwards_base_url(self, monkeypatch):
        monkeypatch.setattr(google_client, "NormalizedChatGoogleGenerativeAI", _RecordingChat)
        GoogleClient("gemini-3.5-flash", "https://gen.example/v1beta").get_llm()
        assert _RecordingChat.instances[-1].kwargs == {
            "model": "gemini-3.5-flash",
            "base_url": "https://gen.example/v1beta",
            "timeout": 120.0,
        }

    def test_r17_google_get_llm_without_base_url_omits_the_key(self, monkeypatch):
        monkeypatch.setattr(google_client, "NormalizedChatGoogleGenerativeAI", _RecordingChat)
        GoogleClient("gemini-3.5-flash", "").get_llm()
        assert "base_url" not in _RecordingChat.instances[-1].kwargs

    def test_r17_google_get_llm_forwards_exactly_the_whitelisted_kwargs(self, monkeypatch):
        """Six kwargs pass through; the key is forwarded as ``google_api_key``
        (unified ``api_key`` wins over ``google_api_key``); ``max_tokens`` and
        ``reasoning_effort`` are dropped."""
        monkeypatch.setattr(google_client, "NormalizedChatGoogleGenerativeAI", _RecordingChat)
        sentinel = object()
        GoogleClient(
            "gemini-3.5-flash",
            timeout=30,
            max_retries=3,
            temperature=0.2,
            callbacks=[sentinel],
            http_client=sentinel,
            http_async_client=sentinel,
            api_key="google-key-1",
            google_api_key="google-key-2",
            thinking_level="minimal",
            max_tokens=99,
            effort="high",
            reasoning_effort="low",
        ).get_llm()
        assert _RecordingChat.instances[-1].kwargs == {
            "model": "gemini-3.5-flash",
            "timeout": 30,
            "max_retries": 3,
            "temperature": 0.2,
            "callbacks": [sentinel],
            "http_client": sentinel,
            "http_async_client": sentinel,
            "google_api_key": "google-key-1",
            "thinking_level": "minimal",  # Flash accepts minimal: unmapped
        }


class TestAnthropicGetLlm:
    def test_r17_anthropic_get_llm_forwards_base_url(self, monkeypatch):
        monkeypatch.setattr(anthropic_client, "NormalizedChatAnthropic", _RecordingChat)
        AnthropicClient("claude-opus-4-8", "https://anthropic.example").get_llm()
        assert _RecordingChat.instances[-1].kwargs == {
            "model": "claude-opus-4-8",
            "base_url": "https://anthropic.example",
            "timeout": 120.0,
        }

    def test_r17_anthropic_get_llm_forwards_exactly_the_whitelisted_kwargs(self, monkeypatch):
        monkeypatch.setattr(anthropic_client, "NormalizedChatAnthropic", _RecordingChat)
        sentinel = object()
        AnthropicClient(
            "claude-opus-4-8",
            timeout=30,
            max_retries=3,
            api_key="anthropic-key",
            max_tokens=4096,
            temperature=0.1,
            callbacks=[sentinel],
            http_client=sentinel,
            http_async_client=sentinel,
            effort="high",
            reasoning_effort="low",
        ).get_llm()
        assert _RecordingChat.instances[-1].kwargs == {
            "model": "claude-opus-4-8",
            "timeout": 30,
            "max_retries": 3,
            "api_key": "anthropic-key",
            "max_tokens": 4096,
            "temperature": 0.1,
            "callbacks": [sentinel],
            "http_client": sentinel,
            "http_async_client": sentinel,
            "effort": "high",
        }


class TestAzureGetLlm:
    def test_r17_azure_get_llm_ignores_the_constructor_base_url(self, monkeypatch):
        """``AzureChatOpenAI`` takes ``azure_endpoint`` (read from the environment by the
        SDK); the ``base_url`` the constructor accepts is silently dropped."""
        monkeypatch.setattr(
            "yialpha.llm_clients.azure_client.NormalizedAzureChatOpenAI", _RecordingChat
        )
        AzureOpenAIClient("gpt-deploy", "https://ignored.example", temperature=0.5).get_llm()
        assert _RecordingChat.instances[-1].kwargs == {
            "model": "gpt-deploy",
            "azure_deployment": "gpt-deploy",
            "temperature": 0.5,
            "timeout": 120.0,
        }


# ===========================================================================
# key_pool — the empty-pool guard of the auth flow
# ===========================================================================
class TestKeyPoolEmptyGuard:
    def test_r17_key_pool_auth_flow_with_no_key_to_serve_raises_before_sending(self):
        """Defensive guard (round-start line 235): when neither ``pick`` nor
        ``force_pick`` has a key, ``auth_flow`` returns before yielding anything.

        Production wiring cannot reach it — ``get_pool_http_client`` builds the state
        with ``size=len(keys)`` under the same lock as the client, and with
        ``size >= 1`` a failed ``pick`` always leaves something for ``force_pick``.
        Reached by direct construction, the in-code comment ("SDK surfaces the 401")
        does not hold on httpx 0.28: ``Auth.sync_auth_flow`` does an unguarded
        ``next(flow)``, and PEP 479 turns the early return into a RuntimeError. No
        request is sent at all."""
        state = key_pool._PoolState(0, 120.0)
        assert state.pick() is None
        assert state.force_pick() is None
        sent: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            sent.append(request.url.path)
            return httpx.Response(401)

        auth = key_pool.RotatingKeyAuth(["pooled-key-1"], state)
        with (
            httpx.Client(
                transport=httpx.MockTransport(handler), auth=auth, trust_env=False
            ) as client,
            pytest.raises(RuntimeError, match="generator raised StopIteration"),
        ):
            client.get("https://llm.invalid/v1/models")
        assert sent == []


# ===========================================================================
# key_pool — what earlier rounds did not pin: parsing, rotation while keys
# cool and revive, cooldown length against a clock, process-wide memory,
# secrecy of the keys, and the wire flags of pooled streaming
# ===========================================================================
_KA, _KB, _KC = "pool-key-a", "pool-key-b", "pool-key-c"
_POOL_URL = "https://pool.test/v1/chat/completions"


class _FakeClock:
    """The monotonic clock ``key_pool`` reads; tests advance it by hand."""

    def __init__(self) -> None:
        self.now = 1000.0

    def monotonic(self) -> float:
        return self.now


@pytest.fixture()
def clock(monkeypatch) -> _FakeClock:
    fake = _FakeClock()
    monkeypatch.setattr(key_pool, "time", fake)
    return fake


def _wire(
    status_by_key: dict[str, int], auth: key_pool.RotatingKeyAuth
) -> tuple[httpx.Client, list[str]]:
    """``(client, seen)``: a client whose transport answers each key by status."""
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        key = request.headers["Authorization"].removeprefix("Bearer ")
        seen.append(key)
        status = status_by_key.get(key, 200)
        return httpx.Response(status, json={"ok": status == 200}, request=request)

    client = httpx.Client(
        transport=httpx.MockTransport(handler), auth=auth, trust_env=False
    )
    return client, seen


def _one_request(status_by_key: dict[str, int], auth: key_pool.RotatingKeyAuth) -> list[str]:
    """One request through a throwaway client sharing ``auth``; the keys tried."""
    client, seen = _wire(status_by_key, auth)
    try:
        client.post(_POOL_URL, json={})
    finally:
        client.close()
    return seen


def _set_pool(monkeypatch, *keys: str) -> None:
    monkeypatch.setenv("ZHIPU_CN_API_KEYS", ",".join(keys))


class TestKeyPoolParsing:
    def test_r17_key_pool_parse_drops_blank_entries_and_dedupes_in_order(self, monkeypatch):
        """Blank / whitespace-only entries never become keys; a duplicate keeps its
        first position; the single-key var is appended only when it is new; the
        provider name is case-insensitive."""
        monkeypatch.setenv("ZHIPU_CN_API_KEYS", " , pool-key-b, ,pool-key-a,pool-key-b,  ,")
        monkeypatch.setenv("ZHIPU_CN_API_KEY", " pool-key-b ")
        assert key_pool.api_key_pool("glm-cn") == [_KB, _KA]
        monkeypatch.setenv("ZHIPU_CN_API_KEY", " pool-key-c ")
        assert key_pool.api_key_pool("glm-cn") == [_KB, _KA, _KC]
        assert key_pool.api_key_pool("GLM-CN") == [_KB, _KA, _KC]

    def test_r17_key_pool_blank_only_pool_var_engages_nothing(self, monkeypatch):
        """A pool variable that parses to zero keys is "not configured": no client,
        no state, nothing cached for the next valid caller."""
        monkeypatch.setenv("ZHIPU_CN_API_KEYS", " , ,")
        assert key_pool.api_key_pool("glm-cn") == []
        assert key_pool.has_pool("glm-cn") is False
        assert key_pool.get_pool_http_client("glm-cn") is None
        assert key_pool._clients == {}
        assert key_pool._states == {}

    def test_r17_key_pool_single_key_var_alone_never_builds_a_client(self, monkeypatch):
        """The single-key variable joins the parsed list (so ``config-check`` can
        show it) but does not engage the rotating transport."""
        monkeypatch.setenv("ZHIPU_CN_API_KEY", "single-key")
        assert key_pool.api_key_pool("glm-cn") == ["single-key"]
        assert key_pool.has_pool("glm-cn") is False
        assert key_pool.get_pool_http_client("glm-cn") is None

    def test_r17_key_pool_provider_without_a_pool_variable_ignores_a_keys_env(
        self, monkeypatch
    ):
        """Only ``glm`` / ``glm-cn`` register a pool variable. DeepSeek is single-key
        by design: a ``DEEPSEEK_API_KEYS`` variable is not consulted."""
        monkeypatch.setenv("DEEPSEEK_API_KEYS", "key-1,key-2")
        monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-single")
        assert key_pool.api_key_pool("deepseek") == []
        assert key_pool.has_pool("deepseek") is False
        assert key_pool.get_pool_http_client("deepseek") is None


class TestKeyPoolRotation:
    def test_r17_key_pool_round_robin_serves_each_live_key_once_per_cycle(self):
        """Three live keys, nine requests: every key exactly once per cycle — no key
        skipped, none served twice in a row."""
        auth = key_pool.RotatingKeyAuth([_KA, _KB, _KC], key_pool._PoolState(3, 60.0))
        client, seen = _wire({}, auth)
        with client:
            for _ in range(9):
                assert client.post(_POOL_URL, json={}).status_code == 200
        assert seen == [_KA, _KB, _KC] * 3

    def test_r17_key_pool_cooling_key_is_skipped_until_its_deadline_then_rejoins(
        self, clock
    ):
        """A 429'd key leaves the rotation for exactly the cooldown; the other keys
        alternate in its absence; at the deadline (inclusive) it is served again at
        its own round-robin slot, and a new 429 there starts a fresh cooldown."""
        keys = [_KA, _KB, _KC]
        state = key_pool._PoolState(3, cooldown_s=120.0)
        auth = key_pool.RotatingKeyAuth(keys, state)

        assert _one_request({}, auth) == [_KA]
        assert _one_request({_KB: 429}, auth) == [_KB, _KC]  # same request, next key
        assert state.dead_until == {1: clock.now + 120.0}
        assert [_one_request({}, auth) for _ in range(4)] == [[_KA], [_KC], [_KA], [_KC]]

        clock.now = state.dead_until[1] - 0.1  # one tick before the deadline
        assert [_one_request({}, auth) for _ in range(2)] == [[_KA], [_KC]]
        clock.now = state.dead_until[1]  # the deadline itself counts as "live again"
        assert _one_request({}, auth) == [_KA]
        assert _one_request({_KB: 429}, auth) == [_KB, _KC]
        assert state.dead_until[1] == clock.now + 120.0  # it was live: fresh cooldown
        assert [_one_request({}, auth) for _ in range(2)] == [[_KA], [_KC]]

    def test_r17_key_pool_rotation_never_repeats_a_key_before_its_live_peers(self, clock):
        """Invariant under failures and clock jumps: between two live attempts on the
        same key, every key that was live at both moments is attempted in between —
        no live key starves while another is served twice."""
        keys = [_KA, _KB, _KC, "pool-key-d"]
        state = key_pool._PoolState(4, cooldown_s=50.0)
        auth = key_pool.RotatingKeyAuth(keys, state)
        # (key index, keys live at the moment this one was picked)
        attempts: list[tuple[int, frozenset[int]]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            key = request.headers["Authorization"].removeprefix("Bearer ")
            attempts.append(
                (keys.index(key), frozenset(i for i in range(4) if state.is_live(i)))
            )
            step = len(attempts)
            status = 429 if step % 5 == 2 else 401 if step % 23 == 7 else 200
            return httpx.Response(status, json={"ok": status == 200}, request=request)

        client = httpx.Client(
            transport=httpx.MockTransport(handler), auth=auth, trust_env=False
        )
        with client:
            for step in range(60):
                clock.now += (step * 17) % 70
                client.post(_POOL_URL, json={})

        # The scenario must have exercised every branch the invariant covers.
        assert len(state.permanent) >= 1
        assert len(state.dead_until) >= 3
        assert any(not live for _, live in attempts)  # the saturated (forced) path
        checked = 0
        for i, (idx, live_i) in enumerate(attempts):
            if idx not in live_i:
                continue  # force-served on a dead key: round-robin promises nothing
            for j in range(i + 1, len(attempts)):
                other, live_j = attempts[j]
                if other != idx or idx not in live_j:
                    continue
                between = {peer for peer, _ in attempts[i + 1 : j]}
                for peer in live_i & live_j - {idx}:  # the key itself is not a peer
                    assert peer in between, (i, j, peer)
                checked += 1
                break
        assert checked >= 20

    def test_r17_key_pool_one_request_tries_each_key_at_most_once_then_settles(self, clock):
        """A request walks the live keys once at most; when the whole pool is cooling
        each later request costs ONE attempt on the soonest-to-revive key, and those
        forced 429s do not push the revival deadlines out."""
        keys = [_KA, _KB, _KC]
        state = key_pool._PoolState(3, cooldown_s=60.0)
        auth = key_pool.RotatingKeyAuth(keys, state)
        saturated = {_KA: 429, _KB: 429, _KC: 429}

        # A and B refuse, C answers: one request walks A, B, C in order, succeeds.
        assert _one_request({_KA: 429, _KB: 429}, auth) == [_KA, _KB, _KC]
        assert set(state.dead_until) == {0, 1}
        assert _one_request({_KA: 429, _KB: 429}, auth) == [_KC]  # only C is live
        assert _one_request(saturated, auth) == [_KC, _KA]  # C cools too; forced A
        assert set(state.dead_until) == {0, 1, 2}
        deadlines = dict(state.dead_until)
        for _ in range(3):
            assert _one_request(saturated, auth) == [_KA]
        assert state.dead_until == deadlines

    @pytest.mark.parametrize("status", [400, 404, 408, 409, 413, 500, 502, 503, 504])
    def test_r17_key_pool_non_key_status_is_returned_unrotated(self, status):
        """Only 401 / 403 / 429 mean "this key is the problem". Anything else is the
        request's or the server's fault: one attempt, response untouched, no key
        marked, the rotation moves on."""
        state = key_pool._PoolState(3, 60.0)
        auth = key_pool.RotatingKeyAuth([_KA, _KB, _KC], state)
        client, seen = _wire({_KA: status}, auth)
        with client:
            assert client.post(_POOL_URL, json={}).status_code == status
        assert seen == [_KA]
        assert state.permanent == set()
        assert state.dead_until == {}
        assert state.pick() == 1


class TestKeyPoolMemory:
    @pytest.mark.parametrize(("raw", "seconds"), [(None, 120.0), ("30", 30.0), ("0", 1.0)])
    def test_r17_key_pool_429_cooldown_lasts_the_configured_seconds(
        self, monkeypatch, clock, raw, seconds
    ):
        """``YIALPHA_LLM_KEY_COOLDOWN_S`` (floored at 1s, default 120s) is the exact
        time a 429'd key sits out — read once, when the pool is built."""
        if raw is not None:
            monkeypatch.setenv("YIALPHA_LLM_KEY_COOLDOWN_S", raw)
        _set_pool(monkeypatch, _KA, _KB)
        pooled = key_pool.get_pool_http_client("glm-cn")
        assert pooled is not None
        state = pooled.auth.state
        assert state.cooldown_s == seconds

        assert _one_request({_KA: 429}, pooled.auth) == [_KA, _KB]
        assert state.dead_until == {0: clock.now + seconds}
        clock.now += seconds - 0.001
        assert _one_request({_KA: 429}, pooled.auth) == [_KB]  # not even tried
        clock.now += 0.001
        assert _one_request({_KA: 429}, pooled.auth) == [_KA, _KB]  # back, fails again
        assert state.dead_until == {0: clock.now + seconds}

    @pytest.mark.parametrize("auth_status", [401, 403])
    def test_r17_key_pool_auth_failure_drops_the_key_for_the_rest_of_the_process(
        self, monkeypatch, clock, auth_status
    ):
        """401 / 403 is final for the process: the key is never tried again — not after
        any amount of time, not after the key "heals" — and every client built later
        (the graph's own LLM included) shares that knowledge."""
        keys = [_KA, _KB]
        _set_pool(monkeypatch, *keys)
        pooled = key_pool.get_pool_http_client("glm-cn")
        assert pooled is not None
        state = pooled.auth.state

        assert _one_request({_KA: auth_status}, pooled.auth) == [_KA, _KB]
        assert state.permanent == {0}
        clock.now += 10 * 365 * 24 * 3600  # years later, the key answers fine now
        for _ in range(6):
            assert _one_request({}, pooled.auth) == [_KB]
        assert state.dead_until == {}  # an auth failure is not a cooldown

        assert key_pool.get_pool_http_client("glm-cn") is pooled
        llm = OpenAIClient("glm-5.3", provider="glm-cn").get_llm()
        assert llm.http_client is pooled
        # The SDK's construction-time key is pool[0] — the dead one. Harmless: the
        # Authorization header is decided per request by the transport.
        assert llm.openai_api_key.get_secret_value() == _KA

    def test_r17_key_pool_never_exposes_a_key_in_logs_repr_or_errors(self, monkeypatch, caplog):
        """The pool's own surfaces — its WARNING lines, the auth / state / client
        objects, the status error a caller raises — identify a key by index only."""
        keys = [_KA, _KB]
        _set_pool(monkeypatch, *keys)
        pooled = key_pool.get_pool_http_client("glm-cn")
        assert pooled is not None
        seen: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request.headers["Authorization"].removeprefix("Bearer "))
            return httpx.Response(429, json={"error": "rate limited"}, request=request)

        with caplog.at_level(logging.WARNING, logger="yialpha.llm_clients.key_pool"):
            client = httpx.Client(
                transport=httpx.MockTransport(handler), auth=pooled.auth, trust_env=False
            )
            with client:
                response = client.post(_POOL_URL, json={})
            with pytest.raises(httpx.HTTPStatusError) as caught:
                response.raise_for_status()
        assert seen == keys
        # Both attempts were ordinary picks, so both lines end the same way — the
        # second although the cycle ends with it and nothing is retried (wording).
        assert all("retrying on the next key" in record.getMessage() for record in caplog.records)
        assert all("cooling down for 120s" in record.getMessage() for record in caplog.records)
        assert [record.getMessage().split(" for this provider")[0] for record in caplog.records] == [
            "key_pool: key 1/2",
            "key_pool: key 2/2",
        ]
        surfaces = "\n".join(
            [
                caplog.text,
                str(caught.value),
                repr(pooled),
                repr(pooled.auth),
                repr(vars(pooled.auth.state)),
                repr(response),
                repr(response.request),
            ]
        )
        for key in keys:
            assert key not in surfaces


class TestKeyPoolTransport:
    def test_r17_key_pool_client_shape(self, monkeypatch):
        """Env proxies honoured, redirects followed (the SDK's own default client
        does), the failed body drained inside the auth flow, and the transport
        ceiling of 120s read / 15s connect / 10s pool."""
        _set_pool(monkeypatch, _KA, _KB)
        pooled = key_pool.get_pool_http_client("glm-cn")
        assert pooled is not None
        assert pooled.trust_env is True
        assert pooled.follow_redirects is True
        assert pooled.auth.requires_response_body is True
        assert (pooled.timeout.read, pooled.timeout.write) == (120.0, 120.0)
        assert (pooled.timeout.connect, pooled.timeout.pool) == (15.0, 10.0)
        assert pooled.auth.state.size == 2

    @pytest.mark.parametrize(
        "raw", [None, "0", "0.0", "-0", "45.5", " 300 ", "", "  ", "abc", "-1", "nan", "inf"]
    )
    def test_r17_key_pool_transport_ceiling_agrees_with_the_model_level_timeout(
        self, monkeypatch, raw
    ):
        """For every value of ``YIALPHA_LLM_TIMEOUT_S`` the pool client's read ceiling
        is the same number the chat model receives (both ``None`` for "off")."""
        if raw is not None:
            monkeypatch.setenv("YIALPHA_LLM_TIMEOUT_S", raw)
        kwargs: dict[str, Any] = {}
        _timeout.resolve_timeout(kwargs, is_local=False, provider_name="glm-cn")
        ceiling = key_pool._pool_transport_timeout()
        assert (ceiling.read if ceiling is not None else None) == kwargs["timeout"]
        if ceiling is not None:
            assert (ceiling.connect, ceiling.pool, ceiling.write) == (15.0, 10.0, ceiling.read)


class _ChatWire:
    """Records Chat Completions request bodies and answers them — in the plain and in
    the SSE dialect — for any mock-backed transport a test hands to a chat model."""

    USAGE = {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}

    def __init__(self) -> None:
        self.bodies: list[dict[str, Any]] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.bodies.append(body)
        if body.get("stream"):
            chunk = {"id": "c1", "object": "chat.completion.chunk", "created": 1, "model": "m"}
            events = [
                {
                    **chunk,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"role": "assistant", "content": "po"},
                            "finish_reason": None,
                        }
                    ],
                },
                {
                    **chunk,
                    "choices": [
                        {"index": 0, "delta": {"content": "ked"}, "finish_reason": "stop"}
                    ],
                },
                {**chunk, "choices": [], "usage": self.USAGE},
            ]
            text = "".join(f"data: {json.dumps(event)}\n\n" for event in events)
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                text=text + "data: [DONE]\n\n",
                request=request,
            )
        message: dict[str, Any] = {"role": "assistant", "content": "poked"}
        finish_reason = "stop"
        if body.get("tools"):
            # A tool-calling request: answer with one tool call whose arguments
            # satisfy the first tool's schema ("BUY" for every string property).
            function = body["tools"][0]["function"]
            properties = function.get("parameters", {}).get("properties", {})
            message = {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": function["name"],
                            "arguments": json.dumps(dict.fromkeys(properties, "BUY")),
                        },
                    }
                ],
            }
            finish_reason = "tool_calls"
        return httpx.Response(
            200,
            json={
                "id": "c1",
                "object": "chat.completion",
                "created": 1,
                "model": body["model"],
                "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
                "usage": self.USAGE,
            },
            request=request,
        )

    def client(self) -> httpx.Client:
        """A legacy ``httpx.Client`` on this wire — what the keep-alive path forwards."""
        return httpx.Client(transport=httpx.MockTransport(self.handler), trust_env=False)


@pytest.fixture()
def wire(monkeypatch) -> _ChatWire:
    """Chat Completions recorded at the transport. Every client the key pool builds
    gets a MockTransport; the SDK's own default client is out of reach of any
    ``httpx.Client`` seam (see TestKeyPoolWireFlags)."""
    recorder = _ChatWire()

    class _MockedClient(httpx.Client):
        """``httpx.Client`` with the transport swapped. It has to be a real class: the
        openai SDK type-checks the client it is handed (``isinstance`` against
        ``httpx.Client``), so a stand-in function is rejected with a TypeError."""

        def __init__(self, **kwargs: Any) -> None:
            kwargs["transport"] = httpx.MockTransport(recorder.handler)
            super().__init__(**kwargs)

    monkeypatch.setattr(key_pool.httpx, "Client", _MockedClient)
    return recorder


class TestKeyPoolWireFlags:
    def test_r17_key_pool_streaming_and_usage_flags_follow_the_switches(
        self, wire, monkeypatch
    ):
        """On the wire: ``stream`` and ``stream_options.include_usage`` are sent only
        when the pool engages; ``YIALPHA_LLM_POOL_STREAM_USAGE=false`` keeps the
        stream but drops the usage request; ``YIALPHA_LLM_POOL_STREAMING=false``
        restores the plain request.

        The single-key leg hands the model a mock-backed client the way the keep-alive
        path does. Given no ``http_client`` at all the SDK builds its own default one
        (openai 3.x: an ``httpx2`` client), which no ``httpx.Client`` seam reaches and
        which dials the provider for real."""
        monkeypatch.setenv("ZHIPU_CN_API_KEY", "single-key")
        single = OpenAIClient("glm-5.3", provider="glm-cn", http_client=wire.client())
        single.get_llm().invoke("hi")
        assert not wire.bodies[-1].get("stream")
        assert "stream_options" not in wire.bodies[-1]

        key_pool.reset_for_test()
        _set_pool(monkeypatch, _KA, _KB)
        result = OpenAIClient("glm-5.3", provider="glm-cn").get_llm().invoke("hi")
        assert wire.bodies[-1]["stream"] is True
        assert wire.bodies[-1]["stream_options"] == {"include_usage": True}
        assert result.content == "poked"
        assert result.usage_metadata["total_tokens"] == 5

        key_pool.reset_for_test()
        monkeypatch.setenv("YIALPHA_LLM_POOL_STREAM_USAGE", "false")
        OpenAIClient("glm-5.3", provider="glm-cn").get_llm().invoke("hi")
        assert wire.bodies[-1]["stream"] is True
        assert "stream_options" not in wire.bodies[-1]

        key_pool.reset_for_test()
        monkeypatch.setenv("YIALPHA_LLM_POOL_STREAMING", "false")
        OpenAIClient("glm-5.3", provider="glm-cn").get_llm().invoke("hi")
        assert not wire.bodies[-1].get("stream")
        assert "stream_options" not in wire.bodies[-1]


class TestOpenAIClientKeyResolution:
    @pytest.mark.parametrize(
        ("provider", "model", "pool_raw", "key_env", "expected"),
        [
            (
                "glm-cn",
                "glm-5.3",
                " , ",
                "ZHIPU_CN_API_KEY",
                "API key for provider 'glm-cn' is not set. Please set the ZHIPU_CN_API_KEY "
                "environment variable (the ZHIPU_CN_API_KEYS pool variable is set but contains "
                "no parseable key — fix it, or add ZHIPU_CN_API_KEY=your_key to your .env file).",
            ),
            (
                "glm-cn",
                "glm-5.3",
                None,
                "ZHIPU_CN_API_KEY",
                "API key for provider 'glm-cn' is not set. Please set the ZHIPU_CN_API_KEY "
                "environment variable (e.g. add ZHIPU_CN_API_KEY=your_key or a "
                "ZHIPU_CN_API_KEYS pool to your .env file).",
            ),
            (
                "deepseek",
                "deepseek-flash",
                None,
                "DEEPSEEK_API_KEY",
                "API key for provider 'deepseek' is not set. Please set the DEEPSEEK_API_KEY "
                "environment variable (e.g. add DEEPSEEK_API_KEY=your_key to your .env file).",
            ),
        ],
    )
    def test_r17_openai_client_missing_key_error_names_what_to_set(
        self, monkeypatch, provider, model, pool_raw, key_env, expected
    ):
        """No usable key: the error names the single-key variable, suggests the pool
        variable only for providers that have one, and tells an operator whose pool
        variable parses to nothing that this is the thing to fix."""
        monkeypatch.setenv(key_env, "")  # present but empty counts as missing
        if pool_raw is not None:
            monkeypatch.setenv("ZHIPU_CN_API_KEYS", pool_raw)
        with pytest.raises(ValueError) as caught:
            OpenAIClient(model, provider=provider).get_llm()
        assert str(caught.value) == expected

    def test_r17_openai_client_whitespace_only_key_is_forwarded_verbatim(self, monkeypatch):
        """Only a strictly empty value counts as missing. A whitespace-only value is
        truthy and reaches the SDK as the key itself (the provider then answers 401)."""
        monkeypatch.setenv("DEEPSEEK_API_KEY", "   ")
        llm = OpenAIClient("deepseek-flash", provider="deepseek").get_llm()
        assert llm.openai_api_key.get_secret_value() == "   "

    def test_r17_openai_client_unregistered_provider_falls_back_to_sdk_defaults(
        self, monkeypatch
    ):
        """``OpenAIClient`` built directly (the factory rejects unknown names) with a
        provider the registry does not know: no base URL, no key handling, any model
        "valid" — it silently becomes a plain client for api.openai.com."""
        monkeypatch.setenv("OPENAI_API_KEY", "openai-test-key")
        client = OpenAIClient("some-model", provider="not-a-provider")
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            llm = client.get_llm()
        assert type(llm) is NormalizedChatOpenAI
        assert llm.openai_api_base is None  # unset: the SDK's own default endpoint
        assert llm.request_timeout == 120.0  # unknown providers are "cloud"
        assert client.get_provider_name() == "not-a-provider"
        assert client.validate_model() is True


# ===========================================================================
# response_cache — fidelity, malformed entries, failure paths of the write
# side, concurrent writers, and langchain's own cache path end to end
# ===========================================================================
_RICH_MESSAGE = {
    "content": '递归 "quote" \\ back',
    "id": "chatcmpl-abc",
    "name": "market_analyst",
    "tool_calls": [
        {"name": "get_stock_data", "args": {"ticker": "NVDA", "days": [1, 5]}, "id": "call_1"}
    ],
    "invalid_tool_calls": [
        {"name": "broken", "args": "{not json", "id": "call_2", "error": "could not parse"}
    ],
    "usage_metadata": {
        "input_tokens": 11,
        "output_tokens": 7,
        "total_tokens": 18,
        "input_token_details": {"cache_read": 4},
        "output_token_details": {"reasoning": 3},
    },
    "response_metadata": {
        "model_name": "deepseek-flash",
        "finish_reason": "tool_calls",
        "token_usage": {"prompt_tokens": 11, "completion_tokens": 7},
    },
    "additional_kwargs": {"reasoning_content": "because the tape is clean"},
}


@pytest.fixture()
def global_cache(tmp_path):
    """A fresh DiskLLMCache as langchain's global cache for one test; restored after."""
    cache = DiskLLMCache(tmp_path / "llm_responses")
    previous = get_llm_cache()
    set_llm_cache(cache)
    yield cache
    set_llm_cache(previous)


def _chat(wire, **kwargs: Any) -> ChatOpenAI:
    """A real ChatOpenAI on the wire's mock-backed client (no provider is contacted)."""
    params: dict[str, Any] = {"model": "m-a", "api_key": "test-key"}
    params.update(kwargs)
    return ChatOpenAI(http_client=wire.client(), **params)


class TestResponseCacheFidelity:
    def test_r17_cache_round_trip_preserves_every_message_field(self, tmp_path):
        """Everything downstream reads off a replayed generation — content, tool calls
        (valid and invalid), usage, response metadata, DeepSeek's reasoning_content,
        the message id and name — comes back equal to the original."""
        message = AIMessage(**_RICH_MESSAGE)
        info = {"finish_reason": "tool_calls", "logprobs": None}
        cache = DiskLLMCache(tmp_path)
        cache.update("p", "s", [ChatGeneration(message=message, generation_info=info)])

        (hit,) = cache.lookup("p", "s")
        assert type(hit.message) is AIMessage
        assert hit.message.model_dump() == message.model_dump()
        assert hit.generation_info == {**info, "cached_replay": True}
        assert hit.message.tool_calls[0]["args"] == {"ticker": "NVDA", "days": [1, 5]}
        assert hit.message.invalid_tool_calls[0]["args"] == "{not json"
        assert hit.message.additional_kwargs["reasoning_content"] == (
            "because the tape is clean"
        )

    def test_r17_cache_round_trip_keeps_non_chat_generations_and_order(self, tmp_path):
        """A plain ``Generation`` (text, optional info) survives next to chat
        generations, in the original order."""
        cache = DiskLLMCache(tmp_path)
        cache.update(
            "p",
            "s",
            [
                Generation(text="first", generation_info={"finish_reason": "stop"}),
                ChatGeneration(message=AIMessage("middle")),
                Generation(text="third"),
            ],
        )
        first, middle, third = cache.lookup("p", "s")
        kinds = [Generation, ChatGeneration, Generation]
        assert [type(generation) for generation in (first, middle, third)] == kinds
        assert (first.text, middle.message.content, third.text) == ("first", "middle", "third")
        assert first.generation_info == {"finish_reason": "stop", "cached_replay": True}
        assert third.generation_info == {"cached_replay": True}

    def test_r17_cache_each_lookup_returns_a_fresh_object_and_the_stamp_stays_in_memory(
        self, tmp_path
    ):
        """Stamping a replay must not leak: a caller mutating what it got back cannot
        change the next replay, and ``cached_replay`` never reaches the disk."""
        cache = DiskLLMCache(tmp_path)
        cache.update("p", "s", [ChatGeneration(message=AIMessage("x"))])
        (first,) = cache.lookup("p", "s")
        first.message.content = "mutated by the caller"
        first.generation_info["cached_replay"] = "poisoned"
        (second,) = cache.lookup("p", "s")
        assert second is not first and second.message is not first.message
        assert second.message.content == "x"
        assert second.generation_info == {"cached_replay": True}
        on_disk = (tmp_path / f"{_key('p', 's')}.json").read_text(encoding="utf-8")
        assert "cached_replay" not in on_disk

    @pytest.mark.parametrize(
        "payload",
        [
            b"[]",
            b"null",
            b'"a string"',
            b"{}",
            b'{"v": 999}',
            b'{"v": 1, "gens": "text"}',
            b'{"v": 1, "gens": [{"is_chat": true, "message": "not a dict"}]}',
            b'{"v": 1, "gens": [{"is_chat": true, "message": {}}]}',
            b'{"v": 1, "gens": [{"is_chat": true, "message": {"type": "weird"}}]}',
            b'{"v": 1, "gens": [{"is_chat": true, "message": {"type": "tool"}}]}',
            b'{"v": 1, "gens": [{"is_chat": true, "message": {"type": "ai", '
            b'"content": {"bad": "type"}}}]}',
        ],
        ids=[
            "json-list",
            "json-null",
            "json-string",
            "empty-object",
            "future-version",
            "gens-not-a-list",
            "message-not-a-dict",
            "message-without-type",
            "message-of-unknown-type",
            "tool-message-without-tool-call-id",
            "content-of-wrong-type",
        ],
    )
    def test_r17_cache_malformed_entry_is_a_miss_and_is_evicted(
        self, tmp_path, caplog, payload
    ):
        """Any unusable on-disk payload — not just invalid JSON — reads as a miss with
        one WARNING naming the file, is removed, and never raises."""
        cache = DiskLLMCache(tmp_path)
        path = tmp_path / f"{_key('p', 's')}.json"
        path.write_bytes(payload)
        logger_name = "yialpha.llm_clients.response_cache"
        with caplog.at_level(logging.WARNING, logger=logger_name):
            assert cache.lookup("p", "s") is None
        assert not path.exists()
        assert len(caplog.records) == 1
        assert "corrupt entry" in caplog.records[0].getMessage()
        assert path.name in caplog.records[0].getMessage()

    def test_r17_cache_lookup_that_cannot_open_the_entry_is_a_miss_not_a_corruption(
        self, tmp_path, monkeypatch, caplog
    ):
        """Windows reader/writer race, reader side: while a writer replaces the entry,
        opening it fails with PermissionError for a moment. That is not a corrupt
        entry — the lookup is a miss, the entry stays, the next lookup replays it.

        Round 17: ``lookup`` treated ANY failure as corruption. It logged "corrupt
        entry" and DELETED the valid, paid-for generation a writer had just landed."""
        cache = DiskLLMCache(tmp_path)
        cache.update("p", "s", [ChatGeneration(message=AIMessage("paid for"))])
        path = tmp_path / f"{_key('p', 's')}.json"
        real_open = Path.open
        deny_once = [True]

        def flaky_open(self, *args, **kwargs):
            if self == path and deny_once:
                deny_once.pop()
                raise PermissionError(13, "Permission denied")
            return real_open(self, *args, **kwargs)

        monkeypatch.setattr(Path, "open", flaky_open)
        logger_name = "yialpha.llm_clients.response_cache"
        with caplog.at_level(logging.WARNING, logger=logger_name):
            assert cache.lookup("p", "s") is None
        assert path.exists()
        corrupt = [r.getMessage() for r in caplog.records if "corrupt" in r.getMessage()]
        assert corrupt == [], corrupt
        assert cache.lookup("p", "s")[0].message.content == "paid for"

    def test_r17_cache_entry_vanishing_before_the_read_is_a_silent_miss(
        self, tmp_path, monkeypatch, caplog
    ):
        """Another lookup evicted the entry in the instant between ``exists()`` and
        the read: an ordinary miss, nothing to warn about."""
        cache = DiskLLMCache(tmp_path)
        cache.update("p", "s", [ChatGeneration(message=AIMessage("x"))])
        path = tmp_path / f"{_key('p', 's')}.json"
        real_read = Path.read_bytes

        def vanishing_read(self):
            if self == path:
                raise FileNotFoundError(2, "No such file or directory")
            return real_read(self)

        monkeypatch.setattr(Path, "read_bytes", vanishing_read)
        with caplog.at_level(logging.WARNING, logger="yialpha.llm_clients.response_cache"):
            assert cache.lookup("p", "s") is None
        assert caplog.records == []

    def test_r17_cache_entry_path_that_is_not_a_readable_file_is_a_miss(
        self, tmp_path, caplog
    ):
        """A path where the entry should be but that is not a readable file (here: a
        directory) is a miss without an exception, reported as "could not read". It is
        not called corrupt and it is not removed: nothing is known to be wrong with
        the entry itself."""
        cache = DiskLLMCache(tmp_path)
        path = tmp_path / f"{_key('p', 's')}.json"
        path.mkdir()
        logger_name = "yialpha.llm_clients.response_cache"
        with caplog.at_level(logging.WARNING, logger=logger_name):
            assert cache.lookup("p", "s") is None
        assert len(caplog.records) == 1
        assert "could not read" in caplog.records[0].getMessage()
        assert "corrupt" not in caplog.records[0].getMessage()
        assert path.is_dir()

    def test_r17_cache_instances_in_one_directory_share_entries(self, tmp_path):
        """A second cache instance on the same directory (what a restarted process
        builds) sees what the first one wrote — even if built before the write."""
        writer, reader = DiskLLMCache(tmp_path), DiskLLMCache(tmp_path)
        writer.update("p", "s", [ChatGeneration(message=AIMessage("from the writer"))])
        assert reader.lookup("p", "s")[0].message.content == "from the writer"

    def test_r17_cache_clear_leaves_unrelated_files_alone(self, tmp_path):
        """``clear`` removes the entries and the writer-crash husks — nothing else in
        the directory, nothing below it."""
        cache = DiskLLMCache(tmp_path)
        cache.update("p", "s", [ChatGeneration(message=AIMessage("x"))])
        (tmp_path / "notes.txt").write_text("keep", encoding="utf-8")
        nested = tmp_path / "llm_responses"
        nested.mkdir()
        (nested / "entry.json").write_text("{}", encoding="utf-8")
        cache.clear()
        names = sorted(part.name for part in tmp_path.iterdir())
        assert names == ["llm_responses", "notes.txt"]
        assert (nested / "entry.json").exists()


class TestResponseCacheWriteFailures:
    def test_r17_cache_unserializable_generation_never_breaks_the_run(self, tmp_path, caplog):
        """A generation that cannot be JSON-encoded (a foreign object in
        ``generation_info``) is skipped with one WARNING: no file, no husk, no
        exception — and the key stays writable."""
        cache = DiskLLMCache(tmp_path)
        bad = ChatGeneration(message=AIMessage("x"), generation_info={"obj": object()})
        logger_name = "yialpha.llm_clients.response_cache"
        with caplog.at_level(logging.WARNING, logger=logger_name):
            cache.update("p", "s", [bad])
        assert list(tmp_path.iterdir()) == []
        assert len(caplog.records) == 1
        assert "could not write" in caplog.records[0].getMessage()
        cache.update("p", "s", [ChatGeneration(message=AIMessage("fine"))])
        assert cache.lookup("p", "s")[0].message.content == "fine"

    def test_r17_cache_replace_denied_once_is_retried_then_lands(self, tmp_path, monkeypatch):
        """The Windows reader race: ``os.replace`` refused once, then succeeds — the
        retry inside ``replace_retry_on_windows`` rides through it without sleeping."""
        sleeps: list[float] = []
        monkeypatch.setattr("yialpha.dataflows.utils.time.sleep", sleeps.append)
        real_replace = response_cache.os.replace
        attempts: list[str] = []

        def flaky_replace(src, dst):
            attempts.append(src)
            if len(attempts) == 1:
                raise PermissionError("target open in a reader")
            return real_replace(src, dst)

        monkeypatch.setattr(response_cache.os, "replace", flaky_replace)
        cache = DiskLLMCache(tmp_path)
        cache.update("p", "s", [ChatGeneration(message=AIMessage("landed"))])
        assert len(attempts) == 2
        assert sleeps == [0.05]  # the first back-off step only
        assert cache.lookup("p", "s")[0].message.content == "landed"

    def test_r17_cache_replace_denied_for_good_is_a_warning_and_leaves_no_husk(
        self, tmp_path, caplog, monkeypatch
    ):
        """When the replace stays denied, caching gives up on this entry with one
        WARNING and removes its own temporary file; the generation itself was
        already returned to the caller."""
        sleeps: list[float] = []
        monkeypatch.setattr("yialpha.dataflows.utils.time.sleep", sleeps.append)

        def denied(src, dst):
            raise PermissionError("locked")

        monkeypatch.setattr(response_cache.os, "replace", denied)
        cache = DiskLLMCache(tmp_path)
        logger_name = "yialpha.llm_clients.response_cache"
        with caplog.at_level(logging.WARNING, logger=logger_name):
            cache.update("p", "s", [ChatGeneration(message=AIMessage("paid for"))])
        assert list(tmp_path.iterdir()) == []
        assert len(caplog.records) == 1
        assert "could not write" in caplog.records[0].getMessage()
        assert sleeps == [0.05, 0.1, 0.2, 0.4]  # 0.75s of back-off, not slept here


class TestResponseCacheConcurrentWriters:
    def test_r17_cache_concurrent_writers_of_one_key_land_one_complete_entry(
        self, tmp_path
    ):
        """Twelve threads, each with its own cache instance (as many processes would
        have), write twelve different payloads for ONE key: the entry that remains is
        one complete payload and no temporary file is left behind."""
        payloads = {f"answer {n}" for n in range(12)}
        caches = [DiskLLMCache(tmp_path) for _ in payloads]
        with ThreadPoolExecutor(max_workers=len(caches)) as pool:
            list(
                pool.map(
                    lambda pair: pair[0].update(
                        "p", "s", [ChatGeneration(message=AIMessage(pair[1]))]
                    ),
                    list(zip(caches, payloads, strict=True)),
                )
            )
        names = sorted(part.name for part in tmp_path.iterdir())
        assert names == [f"{_key('p', 's')}.json"]
        final = DiskLLMCache(tmp_path).lookup("p", "s")
        assert final[0].message.content in payloads

    def test_r17_cache_reader_racing_writers_never_sees_a_torn_entry(self, tmp_path, caplog):
        """Writers replace one entry while a reader keeps looking the same key up (two
        workers making the same call at once). Every read is a miss or one complete
        payload, and a read that collides with a replace is never reported as a
        corruption. A writer may still give up under constant reader pressure — that
        is the write side's own fail-soft, so the entry may legitimately be absent."""
        payloads = {f"answer {n}" for n in range(12)}
        caches = [DiskLLMCache(tmp_path) for _ in payloads]
        reader = DiskLLMCache(tmp_path)
        reads: list[str | None] = []

        def read() -> None:
            for _ in range(1500):
                hit = reader.lookup("p", "s")
                reads.append(None if hit is None else hit[0].message.content)

        logger_name = "yialpha.llm_clients.response_cache"
        workers = len(caches) + 1
        with (
            caplog.at_level(logging.WARNING, logger=logger_name),
            ThreadPoolExecutor(max_workers=workers) as pool,
        ):
            reading = pool.submit(read)
            list(
                pool.map(
                    lambda pair: pair[0].update(
                        "p", "s", [ChatGeneration(message=AIMessage(pair[1]))]
                    ),
                    list(zip(caches, payloads, strict=True)),
                )
            )
            reading.result(timeout=60)
        assert len(reads) == 1500
        assert set(reads) <= payloads | {None}
        corrupt = [r.getMessage() for r in caplog.records if "corrupt" in r.getMessage()]
        assert corrupt == [], corrupt[:3]
        names = sorted(part.name for part in tmp_path.iterdir())
        assert names in ([], [f"{_key('p', 's')}.json"]), names
        if names:
            final = DiskLLMCache(tmp_path).lookup("p", "s")
            assert final is not None and final[0].message.content in payloads

    def test_r17_cache_concurrent_writers_of_distinct_keys_keep_every_entry(self, tmp_path):
        """Threads sharing one cache instance writing different keys: all entries land
        and each reads back its own payload."""
        cache = DiskLLMCache(tmp_path)
        keys = [f"key-{n}" for n in range(40)]

        def write(name: str) -> None:
            cache.update(name, "s", [ChatGeneration(message=AIMessage(f"for {name}"))])

        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(write, keys))
        assert len(list(tmp_path.iterdir())) == len(keys)
        for name in keys:
            assert cache.lookup(name, "s")[0].message.content == f"for {name}"


class _Verdict(BaseModel):
    rating: str


class _Trade(BaseModel):
    action: str


class TestResponseCacheThroughLangchain:
    def test_r17_cache_replays_a_streamed_generation(self, wire, global_cache):
        """The pooled path streams. The generation langchain caches from a stream is
        the same plain AIMessage as a non-streamed one, so a re-run replays it without
        a second call."""
        first = _chat(wire, streaming=True).invoke("analyze NVDA")
        assert first.content == "poked"
        assert wire.bodies[-1]["stream"] is True
        replay = _chat(wire, streaming=True).invoke("analyze NVDA")
        assert replay.content == "poked"
        assert replay.usage_metadata["total_tokens"] == first.usage_metadata["total_tokens"]
        assert len(wire.bodies) == 1

    def test_r17_cache_key_ignores_what_cannot_shape_the_answer(self, wire, global_cache):
        """Replayed across: another API key (the pool rotates keys per request),
        another transport object (every process builds its own keep-alive client),
        another trace configuration. All three invocations together cost one live call."""
        _chat(wire).invoke("analyze NVDA")
        _chat(wire, api_key="another-test-key").invoke("analyze NVDA")
        _chat(wire).invoke("analyze NVDA", config={"tags": ["retry"], "run_name": "bull"})
        assert len(wire.bodies) == 1

    def test_r17_cache_key_also_follows_retry_limit_and_timeout(self, wire, global_cache):
        """langchain hashes the whole serialised constructor, so an entry is also
        keyed by ``max_retries`` and ``timeout``: changing either between two runs
        costs a live call again."""
        _chat(wire).invoke("analyze NVDA")
        _chat(wire, max_retries=6).invoke("analyze NVDA")
        _chat(wire, max_retries=6, timeout=30).invoke("analyze NVDA")
        assert len(wire.bodies) == 3

    def test_r17_cache_structured_output_is_keyed_by_schema(self, wire, global_cache):
        """Through the real ``with_structured_output``: the same schema replays (the
        parsed object comes back from the cache), another schema is another entry."""
        first = _chat(wire).with_structured_output(_Verdict, method="function_calling")
        again = _chat(wire).with_structured_output(_Verdict, method="function_calling")
        other = _chat(wire).with_structured_output(_Trade, method="function_calling")
        verdict = first.invoke("analyze NVDA")
        assert verdict == _Verdict(rating="BUY")
        assert again.invoke("analyze NVDA") == verdict
        assert len(wire.bodies) == 1
        assert other.invoke("analyze NVDA") == _Trade(action="BUY")
        assert len(wire.bodies) == 2

    def test_r17_cache_rerun_replays_every_turn_of_a_tool_loop(self, global_cache):
        """The shape of every tool-using analyst node: the model asks for a tool, the ToolNode
        answers, the model writes its report from a history that now CONTAINS its own
        first answer. On a retry both calls must replay.

        Round 17, langchain-core 1.5.3 to 1.8.4: the first answer came back from the
        cache with ``usage_metadata["total_cost"] = 0`` — added by langchain's own
        cache-hit path ("we zero out cost on cache hits"). ``usage_metadata`` is part
        of the prompt string the cache is keyed on, so the second call never matched
        its entry: in every tool-calling node only the first call was ever replayed
        and the rest were re-billed."""
        requests_seen: list[list[dict[str, Any]]] = []

        class _ToolLoopWire(_ChatWire):
            def handler(self, request: httpx.Request) -> httpx.Response:
                body = json.loads(request.content)
                requests_seen.append(body["messages"])
                tool_answered = any(m.get("role") == "tool" for m in body["messages"])
                message: dict[str, Any] = {
                    "role": "assistant",
                    "content": "REPORT",
                    "reasoning_content": "weighed the tape",
                }
                finish_reason = "stop"
                if not tool_answered:
                    message = {
                        "role": "assistant",
                        "content": None,
                        "reasoning_content": "need the price first",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "get_stock_data",
                                    "arguments": '{"ticker": "NVDA"}',
                                },
                            }
                        ],
                    }
                    finish_reason = "tool_calls"
                return httpx.Response(
                    200,
                    json={
                        "id": "chatcmpl-ds",
                        "object": "chat.completion",
                        "model": "deepseek-flash",
                        "choices": [
                            {"index": 0, "message": message, "finish_reason": finish_reason}
                        ],
                        "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
                    },
                    request=request,
                )

        def analyst_node() -> AIMessage:
            wire = _ToolLoopWire()

            def model() -> DeepSeekChatOpenAI:
                return DeepSeekChatOpenAI(
                    model="deepseek-flash", api_key="test-key", http_client=wire.client()
                )

            task = [SystemMessage("you are the market analyst"), HumanMessage("NVDA")]
            call = model().invoke(task)
            assert call.tool_calls[0]["name"] == "get_stock_data"
            history = [
                *task,
                call,
                ToolMessage(content="close=123", tool_call_id=call.tool_calls[0]["id"]),
            ]
            return model().invoke(history)

        report = analyst_node()
        assert report.content == "REPORT"
        # The second request carries the tool request it must echo back to DeepSeek.
        assert requests_seen[1][2]["tool_calls"][0]["function"]["name"] == "get_stock_data"
        assert requests_seen[1][2]["reasoning_content"] == "need the price first"
        assert len(requests_seen) == 2

        requests_seen.clear()
        replayed = analyst_node()
        assert requests_seen == []  # both calls replayed: nothing re-billed
        assert replayed.content == "REPORT"

    def test_r17_cache_live_call_result_is_untouched_when_nothing_is_cached(self, wire):
        """Byte-equivalence of the default (cache off): what ``invoke`` returns for a
        live call is the SDK's message as langchain built it."""
        answer = _chat(wire).invoke("analyze NVDA")
        assert answer.content == "poked"
        assert answer.usage_metadata == {
            "input_tokens": 3,
            "output_tokens": 2,
            "total_tokens": 5,
            "input_token_details": {},
            "output_token_details": {},
        }
        assert "total_cost" not in answer.response_metadata


# ===========================================================================
# rate_limiter — the budget itself, against a clock that never waits
# ===========================================================================
class _LimiterClock:
    """``time`` as langchain's limiter sees it: ``sleep`` advances the clock by exactly
    the interval asked for, so nothing here waits for real."""

    def __init__(self) -> None:
        self.ticks = 0

    def monotonic(self) -> float:
        return self.ticks / 10.0

    def sleep(self, seconds: float) -> None:
        assert seconds == pytest.approx(0.1), seconds  # the limiter's poll interval
        self.ticks += 1


@pytest.fixture()
def limiter_clock(monkeypatch) -> _LimiterClock:
    clock = _LimiterClock()
    monkeypatch.setattr("langchain_core.rate_limiters.time", clock)
    return clock


class TestRateLimiter:
    @pytest.mark.parametrize(
        ("rpm", "requests_per_second", "bucket"),
        [(1, 1 / 60, 1), (6, 0.1, 1), (60, 1.0, 2), (90, 1.5, 3), (600, 10.0, 20)],
    )
    def test_r17_rate_limiter_parameters_derive_from_the_budget(
        self, rpm, requests_per_second, bucket
    ):
        """``rpm`` requests per minute, polled every 100 ms, a burst of ~2s of budget
        (at least one)."""
        limiter = rate_limiter.get_shared_rate_limiter("deepseek", rpm)
        assert limiter.requests_per_second == pytest.approx(requests_per_second)
        assert limiter.max_bucket_size == bucket
        assert limiter.check_every_n_seconds == 0.1

    @pytest.mark.parametrize(
        ("rpm", "clamped_to"),
        [(0, 1), (-5, 1), (90.9, 90), ("60", 60)],
        ids=["zero", "negative", "float", "numeric-string"],
    )
    def test_r17_rate_limiter_non_positive_rpm_is_clamped_to_one(self, rpm, clamped_to):
        """A zero or negative budget never becomes a limiter that admits nothing: it is
        the 1-rpm limiter (the same shared object). Floats truncate; digit strings work."""
        limiter = rate_limiter.get_shared_rate_limiter("deepseek", rpm)
        assert limiter is rate_limiter.get_shared_rate_limiter("deepseek", clamped_to)

    @pytest.mark.parametrize(
        ("rpm", "error"),
        [("abc", ValueError), (None, TypeError), (float("nan"), ValueError), (float("inf"), OverflowError)],
        ids=["text", "none", "nan", "inf"],
    )
    def test_r17_rate_limiter_non_numeric_rpm_raises_instead_of_guessing(self, rpm, error):
        """No default is invented for an unusable budget — the caller hears about it, and
        nothing is cached under the bad key."""
        with pytest.raises(error):
            rate_limiter.get_shared_rate_limiter("deepseek", rpm)
        assert rate_limiter._limiters == {}

    @pytest.mark.parametrize(("rpm", "first_admission"), [(6, 10.0), (60, 1.0), (90, 0.7)])
    def test_r17_rate_limiter_bucket_starts_empty(self, limiter_clock, rpm, first_admission):
        """langchain initialises the bucket on the first call, so the very first request
        waits for one token to accrue: 60/rpm seconds (10s at 6 rpm)."""
        limiter = rate_limiter.get_shared_rate_limiter("deepseek", rpm)
        assert limiter.acquire(blocking=False) is False
        assert limiter_clock.monotonic() == 0.0
        assert limiter.acquire() is True
        assert limiter_clock.monotonic() == first_admission

    @pytest.mark.parametrize(("rpm", "bucket"), [(1, 1), (6, 1), (60, 2), (90, 3)])
    def test_r17_rate_limiter_burst_after_idle_is_the_bucket_nothing_more(
        self, limiter_clock, rpm, bucket
    ):
        """After a long idle period exactly ``max_bucket_size`` requests go through at
        once; the next one is refused until tokens accrue again. (Idle time only
        counts once the limiter has been used: the bucket is initialised on the
        first call.)"""
        limiter = rate_limiter.get_shared_rate_limiter("deepseek", rpm)
        limiter.acquire()
        limiter_clock.ticks += 10 * 600  # ten idle minutes
        granted = [limiter.acquire(blocking=False) for _ in range(50)]
        assert sum(granted) == bucket
        assert granted == [True] * bucket + [False] * (50 - bucket)

    @pytest.mark.parametrize(("rpm", "bucket"), [(6, 1), (60, 2), (90, 3)])
    def test_r17_rate_limiter_sustained_demand_holds_the_budget(
        self, limiter_clock, rpm, bucket
    ):
        """A caller that always wants more: no 60s window admits more than
        ``rpm + bucket`` requests, and once the initial burst is spent every window
        admits ``rpm`` (give or take one, the window edges)."""
        limiter = rate_limiter.get_shared_rate_limiter("deepseek", rpm)
        limiter.acquire()  # start the clock
        limiter_clock.ticks += 10 * 120  # two idle minutes: the bucket is full
        admitted: list[float] = []
        for _ in range(200):
            limiter.acquire()
            admitted.append(limiter_clock.monotonic())
        assert admitted == sorted(admitted)
        # Fractional tokens accrue in whole steps, so two admissions can share an
        # instant; the window count is what the budget is about.
        windows = [
            sum(1 for t in admitted if start <= t < start + 60.0)
            for start in admitted
            if start + 60.0 <= admitted[-1]  # windows with a full minute observed
        ]
        assert len(windows) >= 100
        assert max(windows) <= rpm + bucket
        steady = windows[bucket:]  # the initial burst is spent
        assert min(steady) >= rpm - 1
        assert max(steady) <= rpm + 1

    def test_r17_rate_limiter_concurrent_non_blocking_acquires_never_overspend(
        self, limiter_clock
    ):
        """Twelve threads ask at the same instant for a token out of a two-token bucket:
        exactly two get one."""
        limiter = rate_limiter.get_shared_rate_limiter("deepseek", 60)
        limiter.acquire()  # start the clock, then let the two-token bucket fill
        limiter_clock.ticks += 10 * 120
        barrier = threading.Barrier(12)
        outcomes: list[bool] = []
        lock = threading.Lock()

        def ask() -> None:
            barrier.wait(timeout=30)
            granted = limiter.acquire(blocking=False)
            with lock:
                outcomes.append(granted)

        threads = [threading.Thread(target=ask) for _ in range(12)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=30)
        assert not any(thread.is_alive() for thread in threads)
        assert sorted(outcomes) == [False] * 10 + [True] * 2

    def test_r17_rate_limiter_concurrent_first_get_builds_one_limiter(self):
        """Sixteen threads race the first call for one (provider, rpm): one object."""
        barrier = threading.Barrier(16)
        results: list[Any] = [None] * 16

        def get(index: int) -> None:
            barrier.wait(timeout=30)
            results[index] = rate_limiter.get_shared_rate_limiter("DeepSeek", 60)

        threads = [threading.Thread(target=get, args=(index,)) for index in range(16)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=30)
        assert len({id(limiter) for limiter in results}) == 1
        assert list(rate_limiter._limiters) == [("deepseek", 60)]


# ===========================================================================
# http_client — proxy routing of the shared keep-alive client
# ===========================================================================
_PROXY_VARS = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "NO_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
    "no_proxy",
)


@pytest.fixture()
def proxies(monkeypatch):
    """``proxies({...})``: what urllib's ``getproxies()`` reports from then on. Scrubbed
    environment and a stubbed ``getproxies`` keep the Windows registry fallback and
    the developer's proxy variables out of the picture."""
    for var in _PROXY_VARS:
        monkeypatch.delenv(var, raising=False)
    reported: dict[str, str] = {}
    monkeypatch.setattr("httpx._utils.getproxies", lambda: dict(reported))

    def configure(mapping: dict[str, str]) -> None:
        reported.clear()
        reported.update(mapping)

    return configure


def _proxy_host(client: httpx.Client, url: str) -> str | None:
    """The proxy a request to ``url`` would go through, or None for a direct route."""
    transport = client._transport_for_url(httpx.URL(url))
    if transport is client._transport:
        return None
    proxy_url = transport._pool._proxy_url
    host = proxy_url.host
    return host.decode() if isinstance(host, bytes) else host


class TestSharedHttpClientProxies:
    @pytest.mark.parametrize(
        ("no_proxy", "url", "direct"),
        [
            ("api.deepseek.com", "https://api.deepseek.com", True),
            ("api.deepseek.com", "https://sub.api.deepseek.com", True),  # and subdomains
            ("api.deepseek.com", "https://xdeepseek.com", False),  # not a suffix match
            ("api.deepseek.com", "https://api.deepseek.com:8443", True),  # port ignored
            ("api.deepseek.com", "http://api.deepseek.com", True),  # scheme ignored
            ("API.DEEPSEEK.COM", "https://api.deepseek.com", True),  # case-insensitive
            (".deepseek.com", "https://sub.deepseek.com", True),
            ("*.deepseek.com", "https://api.deepseek.com", False),  # matches NOTHING
            ("api.deepseek.com:443", "https://api.deepseek.com", False),  # implicit port
            ("api.deepseek.com:8443", "https://api.deepseek.com:8443", True),
            ("10.0.0.5", "https://10.0.0.5", True),
            ("10.0.0.0/8", "https://10.0.0.5", False),  # CIDR is not supported
            ("api.deepseek.com;10.0.0.5", "https://10.0.0.5", False),  # ';' is no separator
            (" api.deepseek.com , 10.0.0.5 ", "https://10.0.0.5", True),
            ("*", "https://anything.example", True),
        ],
    )
    def test_r17_shared_client_no_proxy_entry_forms(self, proxies, no_proxy, url, direct):
        """What a NO_PROXY entry means on the shared client (httpx 0.28.1): a bare host
        covers itself and its subdomains on any port and scheme; ``*`` covers
        everything; a ``*.host`` entry, a CIDR range, a ``;``-separated list and a
        ``host:443`` entry (for a URL that omits the port) cover nothing."""
        proxies({"https": "http://proxy.example:9999", "http": "http://proxy.example:9999"})
        assert _proxy_host(http_client.get_shared_http_client(), url) == "proxy.example"
        proxies({"https": "http://proxy.example:9999", "no": no_proxy})
        http_client.reset_for_test()
        route = _proxy_host(http_client.get_shared_http_client(), url)
        assert (route is None) is direct, (no_proxy, url, route)

    def test_r17_shared_client_routes_each_scheme_through_its_own_proxy(self, proxies):
        proxies({"https": "http://https-proxy.example:3128", "http": "http://http-proxy.example:3128"})
        client = http_client.get_shared_http_client()
        assert _proxy_host(client, "https://api.deepseek.com") == "https-proxy.example"
        assert _proxy_host(client, "http://api.deepseek.com") == "http-proxy.example"
        assert _proxy_host(client, "https://other.example") == "https-proxy.example"

    def test_r17_shared_client_https_only_proxy_leaves_http_direct(self, proxies):
        proxies({"https": "http://proxy.example:3128"})
        client = http_client.get_shared_http_client()
        assert _proxy_host(client, "https://api.deepseek.com") == "proxy.example"
        assert _proxy_host(client, "http://api.deepseek.com") is None

    def test_r17_shared_client_all_proxy_covers_both_schemes(self, proxies):
        proxies({"all": "http://proxy.example:3128"})
        client = http_client.get_shared_http_client()
        assert _proxy_host(client, "https://api.deepseek.com") == "proxy.example"
        assert _proxy_host(client, "http://api.deepseek.com") == "proxy.example"

    def test_r17_shared_client_proxy_configuration_is_read_once_per_process(self, proxies):
        """The shared client is built on first use and keeps that routing: a proxy
        variable changed afterwards (or a new NO_PROXY entry) has no effect on it until
        the process restarts. One client per process — two proxy setups are never
        mixed inside it."""
        proxies({"https": "http://proxy-a.example:3128"})
        first = http_client.get_shared_http_client()
        proxies({"https": "http://proxy-b.example:3128", "no": "api.deepseek.com"})
        again = http_client.get_shared_http_client()
        assert again is first
        assert _proxy_host(again, "https://api.deepseek.com") == "proxy-a.example"
        http_client.reset_for_test()  # what a restart does
        fresh = http_client.get_shared_http_client()
        assert _proxy_host(fresh, "https://api.deepseek.com") is None
        assert _proxy_host(fresh, "https://other.example") == "proxy-b.example"

    def test_r17_shared_client_shape(self, proxies):
        """Like the SDK's own default client (redirects followed, proxies from the
        environment), with httpx's 5s default for requests that carry no timeout."""
        client = http_client.get_shared_http_client()
        assert http_client.get_shared_http_client() is client
        assert client.trust_env is True
        assert client.follow_redirects is True
        assert client.is_closed is False
        assert client.timeout.read == 5.0


# ===========================================================================
# _timeout — the rest of the precedence chain
# ===========================================================================
_TIMEOUT_LOGGER = "yialpha.llm_clients._timeout"


class TestResolveTimeout:
    @pytest.mark.parametrize(
        ("raw", "expected", "warns"),
        [
            ("", 120.0, False),  # present but empty = unset
            ("  ", 120.0, True),
            ("0", None, False),  # explicitly off
            ("-0", None, False),
            ("0.0", None, False),
            ("1e3", 1000.0, False),
            ("1_000", 1000.0, False),
            (" 240 ", 240.0, False),
            ("inf", 120.0, True),
            ("-inf", 120.0, True),
            ("nan", 120.0, True),
            ("0x10", 120.0, True),
            ("120s", 120.0, True),
        ],
    )
    def test_r17_timeout_env_value_matrix(self, monkeypatch, caplog, raw, expected, warns):
        """Every shape of ``YIALPHA_LLM_TIMEOUT_S``: 0 in any spelling disables the
        ceiling; a positive finite number (``float()`` syntax) is used; anything else
        warns and falls back to the 120s cloud default."""
        monkeypatch.setenv("YIALPHA_LLM_TIMEOUT_S", raw)
        kwargs: dict[str, Any] = {}
        with caplog.at_level(logging.WARNING, logger=_TIMEOUT_LOGGER):
            _timeout.resolve_timeout(kwargs, is_local=False, provider_name="deepseek")
        warnings_seen = [r for r in caplog.records if "not a positive finite number" in r.getMessage()]
        assert kwargs["timeout"] == expected
        assert len(warnings_seen) == (1 if warns else 0), [r.getMessage() for r in caplog.records]
        if warns:
            assert f"YIALPHA_LLM_TIMEOUT_S={raw!r}" in warnings_seen[0].getMessage()
            assert "for provider 'deepseek'" in warnings_seen[0].getMessage()

    def test_r17_timeout_bad_value_warns_on_every_call_but_the_default_info_once(
        self, monkeypatch, caplog
    ):
        """A bad value is reported per client built (it is a live misconfiguration); the
        "built-in default" notice is logged once per process in total, on whichever
        path reaches the cloud default first — the bad-value fallback included."""
        monkeypatch.setenv("YIALPHA_LLM_TIMEOUT_S", "120s")
        with caplog.at_level(logging.INFO, logger=_TIMEOUT_LOGGER):
            for provider in ("deepseek", "anthropic", "google"):
                _timeout.resolve_timeout({}, is_local=False, provider_name=provider)
        bad = [r for r in caplog.records if "not a positive finite number" in r.getMessage()]
        assert [r.getMessage().split("for provider ")[1].split(".")[0] for r in bad] == [
            "'deepseek'",
            "'anthropic'",
            "'google'",
        ]
        assert sum("built-in" in r.getMessage() for r in caplog.records) == 1

        caplog.clear()
        monkeypatch.delenv("YIALPHA_LLM_TIMEOUT_S")
        with caplog.at_level(logging.INFO, logger=_TIMEOUT_LOGGER):
            for provider in ("deepseek", "anthropic", "google"):
                _timeout.resolve_timeout({}, is_local=False, provider_name=provider)
        assert caplog.records == []  # already said once in this process

    def test_r17_timeout_explicit_none_wins_and_does_not_consult_the_env(self, monkeypatch, caplog):
        """A caller passing ``timeout=None`` asked for no ceiling: the env (even a bad
        one) is not read, so nothing overrides it and nothing is logged."""
        monkeypatch.setenv("YIALPHA_LLM_TIMEOUT_S", "120s")
        kwargs = {"timeout": None}
        with caplog.at_level(logging.INFO, logger=_TIMEOUT_LOGGER):
            _timeout.resolve_timeout(kwargs, is_local=False, provider_name="deepseek")
        assert kwargs == {"timeout": None}
        assert caplog.records == []

    def test_r17_timeout_local_provider_matrix(self, monkeypatch, caplog):
        """Local servers: no ceiling unless asked for; a bad value names the local
        fallback in its warning."""
        kwargs: dict[str, Any] = {}
        _timeout.resolve_timeout(kwargs, is_local=True, provider_name="ollama")
        assert kwargs == {}
        monkeypatch.setenv("YIALPHA_LLM_TIMEOUT_S", "300")
        _timeout.resolve_timeout(kwargs, is_local=True, provider_name="ollama")
        assert kwargs == {"timeout": 300.0}
        kwargs = {}
        monkeypatch.setenv("YIALPHA_LLM_TIMEOUT_S", "later")
        with caplog.at_level(logging.WARNING, logger=_TIMEOUT_LOGGER):
            _timeout.resolve_timeout(kwargs, is_local=True, provider_name="ollama")
        assert kwargs == {}
        assert "no timeout (local provider)" in caplog.records[0].getMessage()

    def test_r17_timeout_default_notice_fires_once_under_contention(self, monkeypatch, caplog):
        """Sixteen threads build a cloud client at once with the env unset: the one-shot
        notice is logged exactly once."""
        barrier = threading.Barrier(16)

        def resolve(index: int) -> None:
            barrier.wait(timeout=30)
            _timeout.resolve_timeout({}, is_local=False, provider_name=f"p{index}")

        with caplog.at_level(logging.INFO, logger=_TIMEOUT_LOGGER):
            threads = [threading.Thread(target=resolve, args=(index,)) for index in range(16)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=30)
        assert not any(thread.is_alive() for thread in threads)
        notices = [r for r in caplog.records if "built-in" in r.getMessage()]
        assert len(notices) == 1


# ===========================================================================
# base_client — the replay stamp, provider names, the unknown-model warning
# ===========================================================================
class TestNormalizeContentReplayStamp:
    def test_r17_normalize_content_drops_the_zero_cost_stamp_of_a_replayed_message(self):
        """langchain's cache-hit path copies a replayed message and stamps
        ``usage_metadata["total_cost"] = 0``; ``normalize_content`` removes it (same
        object, every other field untouched) so the answer hashes like the live one."""
        message = AIMessage(
            content="answer",
            usage_metadata={"input_tokens": 3, "output_tokens": 2, "total_tokens": 5, "total_cost": 0},
            response_metadata={"model_name": "m"},
        )
        out = base_client.normalize_content(message)
        assert out is message
        assert out.content == "answer"
        assert out.usage_metadata == {"input_tokens": 3, "output_tokens": 2, "total_tokens": 5}
        assert out.response_metadata == {"model_name": "m"}

    @pytest.mark.parametrize("cost", [0.42, 12, -1.5])
    def test_r17_normalize_content_keeps_a_cost_that_is_not_the_stamp(self, cost):
        usage = {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2, "total_cost": cost}
        message = AIMessage(content="x", usage_metadata=usage)
        assert base_client.normalize_content(message).usage_metadata["total_cost"] == cost

    def test_r17_normalize_content_leaves_messages_without_usage_alone(self):
        message = AIMessage(content="x", usage_metadata=None)
        assert base_client.normalize_content(message).usage_metadata is None
        bare = types.SimpleNamespace(content="plain")
        assert base_client.normalize_content(bare) is bare
        assert not hasattr(bare, "usage_metadata")

    def test_r17_normalize_content_stamp_removal_reaches_the_native_providers_too(self):
        """Anthropic / Google / Azure / Bedrock go through the invoke mixin, not
        through NormalizedChatOpenAI: the stamp is dropped there as well."""
        replayed = AIMessage(
            content=[{"type": "text", "text": "ok"}],
            usage_metadata={"input_tokens": 1, "output_tokens": 1, "total_tokens": 2, "total_cost": 0},
        )

        class _ReplayingChat:
            def invoke(self, input, config=None, **kwargs):
                return replayed

        cls = base_client.make_normalized_chat_class(_ReplayingChat, "ReplayingChat", "doc")
        out = cls().invoke("x")
        assert out is replayed
        assert out.content == "ok"
        assert out.usage_metadata == {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}


class TestBaseClientNamesAndWarning:
    @pytest.mark.parametrize(
        ("client", "provider_name"),
        [
            (OpenAIClient("m", provider="GLM-cn"), "glm-cn"),
            (AnthropicClient("m"), "anthropic"),
            (GoogleClient("m"), "google"),
            (AzureOpenAIClient("m"), "azureopenai"),
            (BedrockClient("m"), "bedrock"),
        ],
    )
    def test_r17_provider_name_in_warnings(self, client, provider_name):
        """The name a warning carries: the registry tag for the OpenAI family, the class
        name minus ``Client`` for the native clients."""
        assert client.get_provider_name() == provider_name

    def test_r17_unknown_model_warning_text_and_category(self):
        """One RuntimeWarning with the exact text per call; a known model warns nothing;
        providers that accept any model never warn."""
        text = (
            "Model 'not-a-real-model' is not in the known model list for provider "
            "'deepseek'. Continuing anyway."
        )
        with pytest.warns(RuntimeWarning) as caught:
            OpenAIClient("not-a-real-model", provider="deepseek").warn_if_unknown_model()
        assert [str(w.message) for w in caught] == [text]
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            OpenAIClient("deepseek-flash", provider="deepseek").warn_if_unknown_model()
            OpenAIClient("whatever", provider="openrouter").warn_if_unknown_model()
            BedrockClient("whatever").warn_if_unknown_model()
        with pytest.warns(RuntimeWarning, match=re.escape("for provider 'anthropic'.")):
            AnthropicClient("claude-experimental-x").warn_if_unknown_model()


# ===========================================================================
# the four provider registries, the model catalog and the capability table
# ===========================================================================
_NATIVES = {"anthropic", "google", "azure", "bedrock"}


class TestProviderRegistries:
    def test_r17_every_provider_has_a_key_variable_and_a_client(self):
        """api_key_env lists exactly the OpenAI-compatible registry plus the four
        native clients — no provider one table knows and another does not."""
        compatible = set(openai_client.OPENAI_COMPATIBLE_PROVIDERS)
        assert set(api_key_env.PROVIDER_API_KEY_ENV) == compatible | _NATIVES

    def test_r17_catalog_covers_every_provider_but_the_two_open_ended_ones(self):
        """The static catalog has no list for OpenRouter (fetched live) and Azure (any
        deployment name); everything else it lists is buildable."""
        known = set(api_key_env.PROVIDER_API_KEY_ENV)
        assert set(model_catalog.MODEL_OPTIONS) == known - {"openrouter", "azure"}

    @pytest.mark.parametrize(
        ("international", "china"), [("qwen", "qwen-cn"), ("glm", "glm-cn"), ("minimax", "minimax-cn")]
    )
    def test_r17_dual_region_providers_share_models_but_not_keys_or_endpoints(
        self, international, china
    ):
        """Same model IDs on both sides; separate accounts: a different key variable
        and a different endpoint; the same client class and the same locale flag."""
        here = openai_client.OPENAI_COMPATIBLE_PROVIDERS[international]
        there = openai_client.OPENAI_COMPATIBLE_PROVIDERS[china]
        assert model_catalog.MODEL_OPTIONS[international] is model_catalog.MODEL_OPTIONS[china]
        assert here.base_url != there.base_url
        assert here.chat_class is there.chat_class
        assert here.is_local is there.is_local is False
        assert api_key_env.get_api_key_env(international) != api_key_env.get_api_key_env(china)

    def test_r17_model_validation_is_strict_exactly_for_the_cataloged_providers(self):
        """A model is checked against a list only for providers that have a list and
        are not "any model" (local servers, relays, user-named deployments)."""
        cataloged = set(model_catalog.MODEL_OPTIONS)
        assert set(validators.VALID_MODELS) == cataloged - set(validators._ANY_MODEL_PROVIDERS)
        for provider in validators._ANY_MODEL_PROVIDERS:
            assert validators.validate_model(provider, "anything-at-all") is True
        assert validators.validate_model("not-a-provider", "anything-at-all") is True

    def test_r17_the_custom_placeholder_counts_as_a_known_model(self):
        """``"custom"`` is the CLI's "Custom model ID" choice, not a model. The catalog
        hands it to ``get_known_models``, so ``validate_model`` accepts it and an
        unknown-model warning is suppressed for it even on strict providers. (The CLI
        resolves the choice into a typed ID before any client is built.)"""
        with_sentinel = {
            provider
            for provider, models in model_catalog.get_known_models().items()
            if "custom" in models
        }
        assert {"deepseek", "glm", "glm-cn", "ollama", "bedrock"} <= with_sentinel
        assert not with_sentinel & {"openai", "anthropic", "google", "xai"}
        for provider in with_sentinel:
            assert validators.validate_model(provider, "custom") is True
            for mode in ("quick", "deep"):
                assert model_catalog.get_model_options(provider, mode)[-1][1] == "custom"
        assert "custom" in validators.VALID_MODELS["deepseek"]

    def test_r17_model_validation_folds_case_on_the_provider_not_on_the_model(self):
        assert validators.validate_model("DeepSeek", "deepseek-flash") is True
        assert validators.validate_model("deepseek", "DeepSeek-Flash") is False

    def test_r17_catalog_lists_are_well_formed(self):
        """Two tiers per provider, non-empty, no option listed twice in a tier, every
        id and label non-blank."""
        for provider, modes in model_catalog.MODEL_OPTIONS.items():
            assert set(modes) == {"quick", "deep"}, provider
            for options in modes.values():
                values = [value for _, value in options]
                assert options and len(set(values)) == len(values), (provider, values)
                assert all(display.strip() and value.strip() for display, value in options)

    @pytest.mark.parametrize(("provider", "mode"), [("openrouter", "quick"), ("azure", "deep"), ("openai", "debate")])
    def test_r17_get_model_options_has_no_list_for_unknown_provider_or_tier(
        self, provider, mode
    ):
        """KeyError, not an empty list: the CLI handles OpenRouter and Azure before it
        ever asks the catalog, and there is no third tier."""
        with pytest.raises(KeyError):
            model_catalog.get_model_options(provider, mode)


class TestCapabilitiesAgainstCatalog:
    @pytest.mark.parametrize(
        ("provider", "profile"),
        [
            ("deepseek", "thinking"),
            ("minimax", "thinking"),
            ("minimax-cn", "thinking"),
            ("glm", "default"),
            ("glm-cn", "default"),
            ("qwen", "default"),
            ("qwen-cn", "default"),
            ("openai", "default"),
            ("xai", "default"),
            ("anthropic", "default"),
            ("google", "default"),
        ],
    )
    def test_r17_every_catalog_model_resolves_to_one_expected_profile(self, provider, profile):
        """No catalog model is ambiguous: DeepSeek's and MiniMax's catalog lines are the
        thinking profiles (no ``tool_choice``), every other catalog model the permissive
        default."""
        for model in model_catalog.get_known_models()[provider]:
            if model == "custom":
                continue
            caps = capabilities.get_capabilities(model)
            if profile == "thinking":
                assert caps.supports_tool_choice is False, model
                assert caps.preferred_structured_method == "function_calling", model
            else:
                assert caps is capabilities._DEFAULT, model

    def test_r17_capability_exact_ids_stay_unique_when_folded_to_lower_case(self):
        assert len(capabilities._BY_ID_LOWER) == len(capabilities._BY_ID)
        assert "minimax-m2" in capabilities._BY_ID_LOWER  # mixed-case source, lowered key

    @pytest.mark.parametrize(
        ("model", "thinking"),
        [
            ("  DeepSeek-Flash  ", True),
            ("DeepSeek-FLASH", True),
            ("deepseek-v4.1-flash", True),
            ("deepseek-v30", True),  # two-digit minor: still v4+
            ("deepseek-v3.2-exp", False),
            ("deepseek-v3", False),
            ("MINIMAX-M2.7", True),
            ("MiniMax-Coding-Plan", False),
        ],
    )
    def test_r17_capability_lookup_folds_case_strips_and_keeps_the_version_floor(
        self, model, thinking
    ):
        caps = capabilities.get_capabilities(model)
        assert caps.supports_tool_choice is (not thinking)
        assert caps.requires_reasoning_content_roundtrip is (thinking and "minimax" not in model.lower())
        assert caps.requires_reasoning_split is (thinking and "minimax" in model.lower())

    @pytest.mark.parametrize("model", [None, "", "   ", 12345])
    def test_r17_capability_lookup_of_odd_inputs_is_the_default(self, model):
        assert capabilities.get_capabilities(model) is capabilities._DEFAULT

    @pytest.mark.parametrize(
        ("namespaced", "bare"),
        [
            ("deepseek/deepseek-v4-pro", "deepseek-v4-pro"),
            ("deepseek/deepseek-flash", "deepseek-flash"),
            ("minimax/MiniMax-M2.7", "MiniMax-M2.7"),
        ],
    )
    def test_r17_capabilities_for_a_namespaced_id_fall_to_the_default(self, namespaced, bare):
        """Pins current behaviour (reported, not changed): OpenRouter-style ids — the
        CLI's OpenRouter shortlist prefers exactly the deepseek / minimax / z-ai vendors
        — do not match the capability table, so the thinking profiles' quirks (no
        ``tool_choice``, MiniMax ``reasoning_split``) are lost on that provider, while
        ``_supports_reasoning_effort`` does look at the segment after the last slash."""
        assert capabilities.get_capabilities(bare).supports_tool_choice is False
        assert capabilities.get_capabilities(namespaced) is capabilities._DEFAULT
        assert capabilities.get_capabilities(namespaced).requires_reasoning_split is False


class TestOpenAIClientForwarding:
    @pytest.mark.parametrize(
        ("provider", "model", "forwarded"),
        [
            ("openai", "gpt-5.5", True),
            ("openai", "o3-mini", True),
            ("openai", "gpt-4.1", False),
            ("openrouter", "openai/gpt-5.5", True),
            ("openrouter", "openai/o3-mini", True),
            ("openrouter", "openai/gpt-4.1", False),
            ("openrouter", "deepseek/deepseek-v3", False),
            ("xai", "grok-4.3", False),
        ],
    )
    def test_r17_reasoning_effort_is_forwarded_by_model_segment_on_any_provider(
        self, monkeypatch, provider, model, forwarded
    ):
        """``reasoning_effort`` passes for GPT-5 / o-series model segments — with or
        without an OpenRouter ``vendor/`` prefix — and is silently dropped otherwise,
        on every OpenAI-compatible provider."""
        monkeypatch.setenv(api_key_env.get_api_key_env(provider), "test-key")
        with warnings.catch_warnings():
            # o3-mini / gpt-4.1 are not in OpenAI's static list: "continuing anyway".
            warnings.simplefilter("ignore", RuntimeWarning)
            llm = OpenAIClient(model, provider=provider, reasoning_effort="low").get_llm()
        assert (llm.reasoning_effort == "low") is forwarded

    def test_r17_unknown_model_warns_once_per_client_and_still_works(self, monkeypatch):
        """A model outside the provider's list: one RuntimeWarning per client built, and
        the client is still the provider's own chat class."""
        monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
        with pytest.warns(RuntimeWarning) as caught:
            llm = OpenAIClient("deepseek-not-listed", provider="deepseek").get_llm()
        assert len(caught) == 1
        assert type(llm) is DeepSeekChatOpenAI
        assert OpenAIClient("deepseek-not-listed", provider="deepseek").validate_model() is False


class TestAnthropicEffortGate:
    @pytest.mark.parametrize(
        "model",
        ["claude-opus-4-5-20251101", "claude-sonnet-4-6-20261001", "claude-opus-4-8-20260101"],
    )
    def test_r17_anthropic_dated_snapshot_ids_are_treated_as_unknown_no_effort(
        self, monkeypatch, model
    ):
        """Pins current behaviour (reported, not changed). The version pattern is
        anchored at the version, so a dated snapshot id of an effort-capable model does
        not receive the configured ``effort`` although its alias does. Anthropic's model
        table lists an alias and its dated snapshot as the same model; the conservative
        default ("unknown models get no effort, to avoid a 400") is what fires here."""
        alias = model.rsplit("-", 1)[0]
        assert anthropic_client._supports_effort(alias) is True
        assert anthropic_client._supports_effort(model) is False
        monkeypatch.setattr(anthropic_client, "NormalizedChatAnthropic", _RecordingChat)
        with warnings.catch_warnings():
            # none of these ids is in the static list: "continuing anyway".
            warnings.simplefilter("ignore", RuntimeWarning)
            AnthropicClient(model, effort="high", api_key="k").get_llm()
            assert "effort" not in _RecordingChat.instances[-1].kwargs
            AnthropicClient(alias, effort="high", api_key="k").get_llm()
        assert _RecordingChat.instances[-1].kwargs["effort"] == "high"
