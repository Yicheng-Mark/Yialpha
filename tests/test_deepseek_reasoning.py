"""Tests for DeepSeekChatOpenAI thinking-mode behaviour.

Two pieces verified:

1. ``reasoning_content`` is captured on receive into the AIMessage's
   ``additional_kwargs`` and re-attached on send so DeepSeek's API
   sees the same value across turns.
2. ``with_structured_output`` consults the capability table and
   suppresses ``tool_choice`` for models that reject it (V4 + reasoner),
   matching DeepSeek's official tool-calling pattern at
   https://api-docs.deepseek.com/guides/tool_calls.

The deepseek provider (chat class + capability rows + key mapping) is
still live in the codebase, so the unit tests above stay. The former
DeepSeek live-call test was replaced by a GLM equivalent: the GLM
providers are the actively-wired stack (key pool, Coding Plan
endpoints), so the end-to-end structured-output round-trip now runs
against GLM — and only when a real GLM key is configured; without one
it skips cleanly (key-gated, never a permanent skip).
"""

import os

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompt_values import ChatPromptValue
from pydantic import BaseModel

from yialpha.llm_clients.openai_client import (
    DeepSeekChatOpenAI,
    NormalizedChatOpenAI,
    _input_to_messages,
)

# ---------------------------------------------------------------------------
# _input_to_messages — the helper that handles list / ChatPromptValue / other
# (Gemini bot review note: non-list inputs must also work)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestInputToMessages:
    def test_list_input_returned_as_is(self):
        msgs = [HumanMessage(content="hi")]
        assert _input_to_messages(msgs) is msgs

    def test_chat_prompt_value_unwrapped(self):
        msgs = [HumanMessage(content="hi")]
        prompt_value = ChatPromptValue(messages=msgs)
        assert _input_to_messages(prompt_value) == msgs

    def test_string_input_yields_empty_list(self):
        # A bare string isn't a message-bearing input; the caller's normal
        # langchain conversion happens upstream of _get_request_payload.
        assert _input_to_messages("hello") == []


# ---------------------------------------------------------------------------
# Reasoning content propagation across turns
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestDeepSeekReasoningContent:
    def _client(self):
        os.environ.setdefault("DEEPSEEK_API_KEY", "placeholder")
        return DeepSeekChatOpenAI(
            model="deepseek-v4-flash",
            api_key="placeholder",
            base_url="https://api.deepseek.com",
        )

    def test_capture_on_receive(self):
        """When the response carries reasoning_content, it lands on the
        AIMessage's additional_kwargs so the next turn can echo it back."""
        client = self._client()
        result = client._create_chat_result(
            {
                "model": "deepseek-v4-flash",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "Plan: buy NVDA.",
                            "reasoning_content": "Step 1: trend is up. Step 2: ...",
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            }
        )
        ai = result.generations[0].message
        assert ai.additional_kwargs["reasoning_content"] == "Step 1: trend is up. Step 2: ..."

    def test_propagate_on_send(self):
        """When an outgoing AIMessage carries reasoning_content, the request
        payload echoes it on the corresponding message dict."""
        client = self._client()
        prior = AIMessage(
            content="Plan",
            additional_kwargs={"reasoning_content": "weighed bull case"},
        )
        new_user = HumanMessage(content="Refine.")
        payload = client._get_request_payload([prior, new_user])
        # Find the assistant message in the payload
        assistant_dicts = [m for m in payload["messages"] if m.get("role") == "assistant"]
        assert assistant_dicts, "assistant message missing from outgoing payload"
        assert assistant_dicts[0]["reasoning_content"] == "weighed bull case"

    def test_propagate_through_chat_prompt_value(self):
        """Gemini bot review note: non-list inputs (ChatPromptValue) must
        also propagate reasoning_content."""
        client = self._client()
        prior = AIMessage(
            content="Plan",
            additional_kwargs={"reasoning_content": "weighed bull case"},
        )
        prompt_value = ChatPromptValue(messages=[prior, HumanMessage(content="Refine.")])
        payload = client._get_request_payload(prompt_value)
        assistant_dicts = [m for m in payload["messages"] if m.get("role") == "assistant"]
        assert assistant_dicts[0]["reasoning_content"] == "weighed bull case"


# ---------------------------------------------------------------------------
# Capability-driven structured output: tool_choice suppressed for V4 + reasoner
# ---------------------------------------------------------------------------


def _bound_kwargs(runnable):
    """Extract bind() kwargs from a with_structured_output result."""
    first = runnable.steps[0] if hasattr(runnable, "steps") else runnable
    return getattr(first, "kwargs", {})


@pytest.mark.unit
class TestStructuredOutputCapabilityDispatch:
    """DeepSeek V4 and reasoner reject the tool_choice parameter
    (official guide: api-docs.deepseek.com/guides/tool_calls passes
    tools=[...] without tool_choice). Verify the capability dispatch
    suppresses tool_choice for those models and sends it for chat."""

    class _Sample(BaseModel):
        answer: str

    def _client(self, model):
        return DeepSeekChatOpenAI(
            model=model, api_key="placeholder", base_url="https://api.deepseek.com",
        )

    def test_chat_sends_tool_choice(self):
        bound = self._client("deepseek-chat").with_structured_output(self._Sample)
        assert _bound_kwargs(bound).get("tool_choice") is not None

    def test_reasoner_suppresses_tool_choice(self):
        bound = self._client("deepseek-reasoner").with_structured_output(self._Sample)
        # tool_choice is either absent or explicitly None — both are valid
        # signals that langchain's bind_tools will skip the parameter.
        assert _bound_kwargs(bound).get("tool_choice") in (None, ...) or \
            "tool_choice" not in _bound_kwargs(bound)

    def test_v4_flash_suppresses_tool_choice(self):
        bound = self._client("deepseek-v4-flash").with_structured_output(self._Sample)
        assert _bound_kwargs(bound).get("tool_choice") is None or \
            "tool_choice" not in _bound_kwargs(bound)

    def test_v4_pro_suppresses_tool_choice(self):
        bound = self._client("deepseek-v4-pro").with_structured_output(self._Sample)
        assert _bound_kwargs(bound).get("tool_choice") is None or \
            "tool_choice" not in _bound_kwargs(bound)

    def test_future_v_variant_via_regex(self):
        """Forward-compat: unknown deepseek-v\\d-* IDs inherit V4 quirks."""
        bound = self._client("deepseek-v5-hypothetical").with_structured_output(self._Sample)
        assert _bound_kwargs(bound).get("tool_choice") is None or \
            "tool_choice" not in _bound_kwargs(bound)

    def test_schema_is_still_bound_as_tool(self):
        """tool_choice is suppressed, but the schema is still bound as a tool —
        exactly matching DeepSeek's official tool-calling examples."""
        bound = self._client("deepseek-reasoner").with_structured_output(self._Sample)
        kwargs = _bound_kwargs(bound)
        tools = kwargs.get("tools", [])
        assert any(
            t.get("function", {}).get("name") == "_Sample" for t in tools
        ), f"schema not bound as a tool: {tools}"


# ---------------------------------------------------------------------------
# Live API: structured output round-trips against the real GLM backend
# ---------------------------------------------------------------------------


def _glm_live_provider() -> str | None:
    """Return ``"glm"`` / ``"glm-cn"`` when a REAL GLM key is configured.

    Both the single-key vars and the comma-separated key-pool vars count
    (the pool is the typical GLM Coding Plan setup, and ``get_llm``
    prefers the pool transport when present). ``placeholder`` — what
    tests/conftest.py's autouse fixture injects for unset vars — never
    counts, so a CI run without secrets skips cleanly instead of firing
    a doomed request.
    """
    for provider, single, pool in (
        ("glm", "ZHIPU_API_KEY", "ZHIPU_API_KEYS"),
        ("glm-cn", "ZHIPU_CN_API_KEY", "ZHIPU_CN_API_KEYS"),
    ):
        for var in (single, pool):
            val = os.environ.get(var, "").strip()
            if val and val != "placeholder":
                return provider
    return None


@pytest.mark.integration
@pytest.mark.skipif(
    _glm_live_provider() is None,
    reason="no real GLM key (ZHIPU_API_KEY / ZHIPU_CN_API_KEY / *_API_KEYS "
    "pool); skipping live API call",
)
class TestGLMLiveStructuredOutput:
    """End-to-end: a real GLM call returns a typed instance.

    Exercises the current primary OpenAI-compatible stack — glm / glm-cn
    through the provider registry (NormalizedChatOpenAI, default
    capability row: function_calling with tool_choice) — the same
    structured-output round-trip the former DeepSeek live test covered,
    now against the provider the repo actually wires.
    """

    class _Pick(BaseModel):
        action: str
        confidence: float

    def test_glm_returns_structured_output(self):
        from yialpha.llm_clients.openai_client import OpenAIClient

        llm = OpenAIClient(
            "glm-5.3", provider=_glm_live_provider(), timeout=60,
        ).get_llm()
        bound = llm.with_structured_output(self._Pick)
        result = bound.invoke(
            "Pick BUY or SELL or HOLD for a tech stock with strong earnings. "
            "Confidence is a float between 0 and 1."
        )
        assert isinstance(result, self._Pick)
        assert result.action in {"BUY", "SELL", "HOLD"}
        assert 0.0 <= result.confidence <= 1.0


# ---------------------------------------------------------------------------
# Base class isolation: NormalizedChatOpenAI does NOT have DeepSeek behaviour
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestBaseClassIsolation:
    def test_normalized_does_not_propagate_reasoning_content(self):
        """The general-purpose NormalizedChatOpenAI must not carry
        DeepSeek-specific behaviour. Only the subclass does."""
        assert not hasattr(NormalizedChatOpenAI, "_get_request_payload") or (
            NormalizedChatOpenAI._get_request_payload
            is NormalizedChatOpenAI.__bases__[0]._get_request_payload
        )
