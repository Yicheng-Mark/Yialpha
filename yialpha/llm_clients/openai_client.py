import logging
import os
import re
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI

from ._timeout import resolve_timeout
from .api_key_env import get_api_key_env
from .base_client import BaseLLMClient, normalize_content
from .capabilities import get_capabilities
from .key_pool import api_key_pool, get_pool_http_client
from .validators import validate_model

logger = logging.getLogger(__name__)


class NormalizedChatOpenAI(ChatOpenAI):
    """ChatOpenAI with normalized content output and capability-aware binding.

    The Responses API returns content as a list of typed blocks
    (reasoning, text, etc.). ``invoke`` normalizes to string for
    consistent downstream handling.

    ``with_structured_output`` consults the per-model capability table
    (``capabilities.get_capabilities``) to pick the method and to decide
    whether ``tool_choice`` may be sent. Models that reject ``tool_choice``
    (e.g. DeepSeek V4 and reasoner — per their official tool-calling
    guide) still bind the schema as a tool, but no ``tool_choice``
    parameter is sent.

    Provider-specific quirks beyond structured-output (e.g. DeepSeek's
    reasoning_content roundtrip) live in subclasses so this base class
    stays small.
    """

    def invoke(self, input: Any, config: Any = None, **kwargs: Any) -> Any:
        return normalize_content(super().invoke(input, config, **kwargs))

    def with_structured_output(self, schema: Any, *, method: Any = None, include_raw: bool = False, **kwargs: Any) -> Any:  # type: ignore[override]  # intentional: adds method-dispatch logic the base signature doesn't model
        caps = get_capabilities(self.model_name)
        if caps.preferred_structured_method == "none":
            raise NotImplementedError(
                f"{self.model_name} has no structured-output method available; "
                f"agent factories will fall back to free-text generation."
            )
        method = method or caps.preferred_structured_method
        # When the model rejects tool_choice, suppress langchain's hardcoded
        # value. The schema is still bound as a tool — exactly what
        # DeepSeek's official tool-calling examples do.
        if method == "function_calling" and not caps.supports_tool_choice:
            kwargs.setdefault("tool_choice", None)
        return super().with_structured_output(schema, method=method, **kwargs)


class LocalCompatibleChatOpenAI(NormalizedChatOpenAI):
    """OpenAI-compatible client for arbitrary local servers (LM Studio, vLLM,
    llama.cpp via the generic ``openai_compatible`` provider).

    Their tool-calling support varies, and many reject the object-form
    ``tool_choice`` langchain sends for function-calling structured output. Bind
    the schema as a tool but don't force tool_choice, so structured output works
    across local servers regardless of the model ID's capabilities (#1057).
    """

    def with_structured_output(self, schema: Any, *, method: Any = None, include_raw: bool = False, **kwargs: Any) -> Any:  # type: ignore[override]  # intentional: adds tool_choice suppression the base signature doesn't model
        resolved = method or get_capabilities(self.model_name).preferred_structured_method
        if resolved == "function_calling":
            kwargs.setdefault("tool_choice", None)
        return super().with_structured_output(schema, method=method, **kwargs)


def _input_to_messages(input_: Any) -> list[Any]:
    """Normalise a langchain LLM input to a list of message objects.

    Accepts a list of messages, a ``ChatPromptValue`` (from a
    ChatPromptTemplate), or anything else (treated as no messages).
    Used by providers that need to walk the outgoing message history;
    in particular DeepSeek thinking-mode propagation must work for
    both bare-list invocations and ChatPromptTemplate-driven ones, so
    treating only ``list`` here would silently skip half the call sites.
    """
    if isinstance(input_, list):
        return input_
    if hasattr(input_, "to_messages"):
        return input_.to_messages()
    return []


class DeepSeekChatOpenAI(NormalizedChatOpenAI):
    """DeepSeek-specific overrides on top of the OpenAI-compatible client.

    Thinking-mode round-trip is the only DeepSeek-specific behavior that
    stays here. When DeepSeek's thinking models return a response with
    ``reasoning_content``, that field must be echoed back as part of the
    assistant message on the next turn or the API fails with HTTP 400.
    ``_create_chat_result`` captures it on receive and
    ``_get_request_payload`` re-attaches it on send.

    Tool-choice handling for V4 and reasoner — those models reject the
    ``tool_choice`` parameter — is handled by the capability dispatch in
    ``NormalizedChatOpenAI.with_structured_output``, not here.
    """

    def _get_request_payload(self, input_: Any, *, stop: Any = None, **kwargs: Any) -> Any:
        payload = super()._get_request_payload(input_, stop=stop, **kwargs)
        outgoing = payload.get("messages", [])
        for message_dict, message in zip(outgoing, _input_to_messages(input_), strict=False):
            if not isinstance(message, AIMessage):
                continue
            reasoning = message.additional_kwargs.get("reasoning_content")
            if reasoning is not None:
                message_dict["reasoning_content"] = reasoning
        return payload

    def _create_chat_result(self, response: Any, generation_info: Any = None) -> Any:
        chat_result = super()._create_chat_result(response, generation_info)
        response_dict = (
            response
            if isinstance(response, dict)
            else response.model_dump(
                exclude={"choices": {"__all__": {"message": {"parsed"}}}}
            )
        )
        for generation, choice in zip(
            chat_result.generations, response_dict.get("choices", []), strict=False
        ):
            reasoning = choice.get("message", {}).get("reasoning_content")
            if reasoning is not None:
                generation.message.additional_kwargs["reasoning_content"] = reasoning
        return chat_result


class MinimaxChatOpenAI(NormalizedChatOpenAI):
    """MiniMax-specific overrides on top of the OpenAI-compatible client.

    M2.x reasoning models embed ``<think>...</think>`` blocks directly in
    ``message.content`` by default, which would pollute saved reports.
    Per platform.minimax.io/docs/api-reference/text-openai-api,
    ``reasoning_split=True`` redirects the thinking block into
    ``reasoning_details`` so ``content`` stays clean. It is sent via
    ``extra_body`` (not a top-level kwarg) because the openai SDK validates
    top-level params and rejects unknown ones like reasoning_split (#826).

    The flag is gated by ``ModelCapabilities.requires_reasoning_split`` so
    only M2.x reasoning models receive it; non-reasoning MiniMax endpoints
    (Coding Plan, MiniMax-Text-01) never see it.

    Tool-choice handling for M2.x — those models accept only the string
    enum ``{"none", "auto"}`` and reject langchain's function-spec dict —
    is handled by the capability dispatch in
    ``NormalizedChatOpenAI.with_structured_output``, not here.
    """

    def _get_request_payload(self, input_: Any, *, stop: Any = None, **kwargs: Any) -> Any:
        payload = super()._get_request_payload(input_, stop=stop, **kwargs)
        if get_capabilities(self.model_name).requires_reasoning_split:
            # Pass via extra_body, not as a top-level kwarg: the openai SDK
            # (>=1.56) validates top-level params against Completions.create
            # and rejects unknown ones like reasoning_split (#826). extra_body
            # is forwarded into the request body untouched.
            extra_body = payload.setdefault("extra_body", {})
            extra_body.setdefault("reasoning_split", True)
        return payload


# Kwargs forwarded from user config to ChatOpenAI
_PASSTHROUGH_KWARGS = (
    "timeout", "max_retries", "reasoning_effort", "temperature",
    "api_key", "callbacks", "http_client", "http_async_client", "rate_limiter",
)

# OpenAI's ``reasoning_effort`` is only accepted by reasoning models — the GPT-5
# family and the o-series. Non-reasoning models (gpt-4.1, gpt-4o, ...) 400 with
# "Unsupported parameter: 'reasoning.effort' is not supported with this model".
# Drop the kwarg for those rather than crash the run.
_OPENAI_REASONING_MODEL = re.compile(r"^(gpt-5|o[1-9])")

# The read-timeout safety net (the 120s cloud default, the YIALPHA_LLM_TIMEOUT_S
# precedence chain) lives in ``._timeout`` — the single source shared by all LLM
# clients (OpenAI family + native Anthropic / Google / Azure / Bedrock). See
# ``OpenAIClient.get_llm`` for the call site.


def _supports_reasoning_effort(model: str) -> bool:
    """Whether the (native OpenAI) model accepts ``reasoning_effort``.

    Matches the MODEL SEGMENT after the last ``/`` so OpenRouter-style
    namespaced IDs (``openai/gpt-5.5``, ``openai/o3-mini``) resolve too —
    the former whole-string match silently dropped an explicitly configured
    ``reasoning_effort`` for them.
    """
    segment = model.lower().strip().rsplit("/", 1)[-1]
    return bool(_OPENAI_REASONING_MODEL.match(segment))


@dataclass(frozen=True)
class ProviderSpec:
    """Declarative config for one OpenAI-compatible provider.

    The OpenAI-compatible family (OpenAI, xAI, DeepSeek, Qwen, GLM, MiniMax,
    OpenRouter, Ollama, and any user endpoint) all speak the same Chat
    Completions API and differ only by these fields — so one row here replaces
    the former per-provider base-URL dict, auth handling, and client-class
    branches. Native Anthropic / Google use their own clients (genuinely
    different APIs) and are intentionally NOT in this registry.

    The API-key env var stays in ``api_key_env.PROVIDER_API_KEY_ENV`` (the single
    source consulted by both this client and the CLI prompt); only behavior that
    is provider-specific (base URL, key optionality, wire-format quirks via
    ``chat_class``) lives here.
    """

    chat_class: type = NormalizedChatOpenAI   # provider quirks live in the subclass
    base_url: str | None = None            # default endpoint (None -> SDK default)
    base_url_env: str | None = None        # env var that overrides base_url (e.g. OLLAMA_BASE_URL)
    key_optional: bool = False                # don't require/prompt; send a placeholder if unset
    placeholder_key: str = "EMPTY"            # sent when no key is available (keyless local servers)
    require_base_url: bool = False            # error if no base_url is resolved (generic endpoint)
    use_responses_api: bool = False           # native OpenAI Responses API
    is_local: bool = False                    # local model server: exempt from default read-timeout


# Single source of truth for the OpenAI-compatible provider family. Dual-region
# providers (qwen/glm/minimax) keep separate endpoints because international and
# China accounts cannot share credentials (#758).
#
# The GLM providers carry a base_url_env because Coding Plan subscriptions
# (per-key quota, the typical multi-key pool setup) live on a DEDICATED
# endpoint that rejects pay-as-you-go traffic and vice versa:
#   Z.AI:       https://api.z.ai/api/coding/paas/v4/
#   BigModel:   https://open.bigmodel.cn/api/coding/paas/v4/
# so the spec keeps the pay-as-you-go default and Coding Plan users point
# ZHIPU_BASE_URL / ZHIPU_CN_BASE_URL at the coding endpoint in .env.
OPENAI_COMPATIBLE_PROVIDERS: dict[str, ProviderSpec] = {
    "openai":     ProviderSpec(use_responses_api=True),
    "xai":        ProviderSpec(base_url="https://api.x.ai/v1"),
    "deepseek":   ProviderSpec(base_url="https://api.deepseek.com", chat_class=DeepSeekChatOpenAI),
    "qwen":       ProviderSpec(base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1"),
    "qwen-cn":    ProviderSpec(base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"),
    "glm":        ProviderSpec(base_url="https://api.z.ai/api/paas/v4/", base_url_env="ZHIPU_BASE_URL"),
    "glm-cn":     ProviderSpec(base_url="https://open.bigmodel.cn/api/paas/v4/", base_url_env="ZHIPU_CN_BASE_URL"),
    "minimax":    ProviderSpec(base_url="https://api.minimax.io/v1", chat_class=MinimaxChatOpenAI),
    "minimax-cn": ProviderSpec(base_url="https://api.minimaxi.com/v1", chat_class=MinimaxChatOpenAI),
    "openrouter": ProviderSpec(base_url="https://openrouter.ai/api/v1"),
    "mistral":    ProviderSpec(base_url="https://api.mistral.ai/v1"),
    "kimi":       ProviderSpec(base_url="https://api.moonshot.ai/v1"),
    "groq":       ProviderSpec(base_url="https://api.groq.com/openai/v1"),
    "nvidia":     ProviderSpec(base_url="https://integrate.api.nvidia.com/v1"),
    "ollama":     ProviderSpec(base_url="http://localhost:11434/v1", base_url_env="OLLAMA_BASE_URL",
                               key_optional=True, placeholder_key="ollama", is_local=True),
    # Generic endpoint: user supplies base_url; key optional (keyless local).
    "openai_compatible": ProviderSpec(
        require_base_url=True, key_optional=True, chat_class=LocalCompatibleChatOpenAI, is_local=True
    ),
}


def is_openai_compatible(provider: str) -> bool:
    """Whether ``provider`` is served by the OpenAI-compatible registry."""
    return provider.lower() in OPENAI_COMPATIBLE_PROVIDERS


def _is_native_openai_base_url(base_url: str | None) -> bool:
    """True when ``base_url`` is unset or points at api.openai.com.

    The Responses API (/v1/responses) only exists on native OpenAI. A custom
    base_url on the ``openai`` provider (a proxy, gateway, or local server)
    speaks only Chat Completions, so the Responses API must stay off there even
    though the provider spec enables it (#1024).
    """
    if not base_url:
        return True
    if "://" not in base_url:
        base_url = "https://" + base_url
    host = urlparse(base_url).hostname or ""
    return host == "api.openai.com" or host.endswith(".openai.com")


def _pool_streaming_enabled() -> bool:
    """Whether pooled providers issue their LLM calls over a wire-level stream.

    Default ON: see get_llm's key-pool branch for the read-timeout semantics
    that make slow reasoning endpoints workable. YIALPHA_LLM_POOL_STREAMING=false
    restores the non-streaming request path exactly.
    """
    return os.getenv("YIALPHA_LLM_POOL_STREAMING", "true").strip().lower() not in (
        "false", "0", "no",
    )


class OpenAIClient(BaseLLMClient):
    """Client for OpenAI, Ollama, OpenRouter, and xAI providers.

    For native OpenAI models, uses the Responses API (/v1/responses) which
    supports reasoning_effort with function tools across all model families
    (GPT-4.1, GPT-5). Third-party compatible providers (xAI, OpenRouter,
    Ollama) use standard Chat Completions.
    """

    def __init__(
        self,
        model: str,
        base_url: str | None = None,
        provider: str = "openai",
        **kwargs,
    ):
        super().__init__(model, base_url, **kwargs)
        self.provider = provider.lower()

    def get_llm(self) -> Any:
        """Return a configured ChatOpenAI instance, driven by the provider registry."""
        self.warn_if_unknown_model()
        llm_kwargs: dict[str, Any] = {"model": self.model}
        spec = OPENAI_COMPATIBLE_PROVIDERS.get(self.provider)
        chat_cls: type = NormalizedChatOpenAI
        pool_client = None  # set when a key-pool provider (key_pool.py) engages

        if spec is not None:
            chat_cls = spec.chat_class

            # base_url precedence: explicit client base_url (carries the config /
            # YIALPHA_LLM_BACKEND_URL value) > provider env override (e.g.
            # OLLAMA_BASE_URL) > provider default. None means use the SDK default.
            env_base_url = os.environ.get(spec.base_url_env) if spec.base_url_env else None
            base_url = self.base_url or env_base_url or spec.base_url
            if spec.require_base_url and not base_url:
                raise ValueError(
                    f"Provider '{self.provider}' requires a base_url. Set it via "
                    "backend_url / YIALPHA_LLM_BACKEND_URL to your endpoint, "
                    "e.g. http://localhost:8000/v1 (vLLM) or http://localhost:1234/v1 "
                    "(LM Studio)."
                )
            if base_url:
                llm_kwargs["base_url"] = base_url

            # API key: required unless key_optional; keyless local servers get a
            # placeholder. The env-var name is the single source in api_key_env.
            # Key-pool providers (GLM Coding Plan, see key_pool.py) instead get
            # a rotating transport: ChatOpenAI still needs one key for its own
            # validation, but the Authorization header is decided per request.
            api_key_env = get_api_key_env(self.provider)
            api_key = os.environ.get(api_key_env) if api_key_env else None
            pool_client = get_pool_http_client(self.provider)
            if pool_client is not None:
                pool = api_key_pool(self.provider)
                llm_kwargs["api_key"] = pool[0]
                llm_kwargs["http_client"] = pool_client
                # Wire-level streaming for pooled providers: reasoning models
                # behind slow endpoints (GLM Coding Plan measured ~40 tok/s
                # with minutes-long analyst-scale generations) exceed any
                # sane read timeout in NON-streaming mode, where the read
                # timeout bounds the TOTAL response time. Streaming keeps
                # invoke()'s aggregated-message semantics but applies the
                # read timeout per chunk, so a continuously-generating call
                # never trips it. Opt out with YIALPHA_LLM_POOL_STREAMING=false.
                if _pool_streaming_enabled():
                    llm_kwargs["streaming"] = True
                logger.info(
                    "key_pool: %s active with %d key(s) — round-robin per "
                    "request (401/403 drops a key for the process, 429 cools "
                    "it down); streaming=%s",
                    self.provider, len(pool), llm_kwargs.get("streaming", False),
                )
            elif api_key:
                llm_kwargs["api_key"] = api_key
            elif spec.key_optional:
                llm_kwargs["api_key"] = spec.placeholder_key
            elif api_key_env:
                raise ValueError(
                    f"API key for provider '{self.provider}' is not set. "
                    f"Please set the {api_key_env} environment variable "
                    f"(e.g. add {api_key_env}=your_key to your .env file)."
                )

            # The Responses API only exists on native OpenAI; if the user points
            # the openai provider at a custom base_url (proxy/gateway/local), it
            # only speaks Chat Completions, so keep Responses off there (#1024).
            if spec.use_responses_api and _is_native_openai_base_url(base_url):
                llm_kwargs["use_responses_api"] = True
        elif self.base_url:
            llm_kwargs["base_url"] = self.base_url

        # Forward user-provided kwargs
        for key in _PASSTHROUGH_KWARGS:
            if key not in self.kwargs:
                continue
            if key == "reasoning_effort" and not _supports_reasoning_effort(self.model):
                continue
            llm_kwargs[key] = self.kwargs[key]

        # A key-pool transport beats the generic keepalive http_client that
        # _get_provider_kwargs may have forwarded: the pool client is itself a
        # shared keepalive client (one per provider), and without it the
        # rotating auth — the whole point of the pool — would be dropped.
        if pool_client is not None and llm_kwargs.get("http_client") is not pool_client:
            logger.debug(
                "key_pool: replacing forwarded http_client with the rotating "
                "pool client for provider %s", self.provider,
            )
            llm_kwargs["http_client"] = pool_client

        # Read-timeout safety net (shared with all LLM clients). ChatOpenAI has
        # no default read timeout, so a half-open socket blocks forever; setting
        # ``timeout`` makes such a call raise so the agent layer's "retry once
        # as free text" fallback (see agents/utils/structured.py) can fire
        # instead of stalling. See ``_timeout.resolve_timeout`` for the full
        # precedence chain; local providers (Ollama, openai_compatible) are
        # exempt from the cloud ceiling.
        resolve_timeout(
            llm_kwargs,
            is_local=(spec is not None and spec.is_local),
            provider_name=self.get_provider_name(),
        )

        # The subclass (provider quirks) comes from the registry spec.
        return chat_cls(**llm_kwargs)

    def validate_model(self) -> bool:
        """Validate model for the provider."""
        return validate_model(self.provider, self.model)
