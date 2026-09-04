"""Stats callback handler: counters, token extraction, thread safety.

Covers yialpha/cli/stats_handler.py — ``StatsCallbackHandler``, the CLI's
run-level tally of LLM/chat/tool calls and token usage. Everything here is
in-process only (langchain callback objects are constructed directly; no
network, no LLM).

Two pinned behaviors worth calling out (documented, not fixed here):

* the handler inherits langchain's no-op ``on_llm_error``, so a failed LLM
  call never raises here — but it is also never counted;
* ``usage_metadata`` values are added with ``+=`` without type checking,
  so post-construction mutation to non-numeric/non-dict shapes raises.
  (Real provider AIMessages cannot carry such shapes — pydantic rejects
  them at construction — so these tests mutate the message afterwards.)
"""

from __future__ import annotations

import threading
import uuid
from concurrent.futures import ThreadPoolExecutor

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, Generation, LLMResult

from yialpha.cli.stats_handler import StatsCallbackHandler


def _result(input_tokens, output_tokens, extra=None):
    """An LLMResult shaped like a chat-model completion with usage metadata."""
    usage = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
    }
    if extra:
        usage.update(extra)
    message = AIMessage(content="ok", usage_metadata=usage)
    return LLMResult(generations=[[ChatGeneration(message=message)]])


# --------------------------------------------------------------------------- #
# Counters
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_fresh_handler_stats_are_zero():
    stats = StatsCallbackHandler().get_stats()
    assert stats == {
        "llm_calls": 0,
        "tool_calls": 0,
        "tokens_in": 0,
        "tokens_out": 0,
    }


@pytest.mark.unit
def test_llm_and_chat_model_starts_both_count():
    handler = StatsCallbackHandler()
    handler.on_llm_start({"id": ["m"]}, ["prompt"], run_id=uuid.uuid4())
    handler.on_chat_model_start({"id": ["m"]}, [[AIMessage(content="hi")]])
    handler.on_llm_start({"id": ["m"]}, ["p1", "p2"])  # kwargs-less form ok too
    assert handler.get_stats()["llm_calls"] == 3


@pytest.mark.unit
def test_tool_starts_count():
    handler = StatsCallbackHandler()
    for _ in range(4):
        handler.on_tool_start({"name": "t"}, "input")
    assert handler.get_stats()["tool_calls"] == 4


@pytest.mark.unit
def test_on_llm_end_accumulates_tokens_across_calls():
    handler = StatsCallbackHandler()
    handler.on_llm_end(_result(10, 3))
    handler.on_chat_model_start({"id": ["m"]}, [[AIMessage(content="hi")]])
    handler.on_llm_end(_result(7, 9))
    assert handler.get_stats() == {
        "llm_calls": 1,
        "tool_calls": 0,
        "tokens_in": 17,
        "tokens_out": 12,
    }


# --------------------------------------------------------------------------- #
# usage-metadata robustness (missing / malformed / nested)
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_on_llm_end_without_usage_metadata_is_noop():
    handler = StatsCallbackHandler()
    message = AIMessage(content="ok")  # usage_metadata defaults to None
    handler.on_llm_end(LLMResult(generations=[[ChatGeneration(message=message)]]))
    assert handler.get_stats()["tokens_in"] == 0
    assert handler.get_stats()["tokens_out"] == 0


@pytest.mark.unit
def test_on_llm_end_partial_usage_metadata_defaults_missing_keys_to_zero():
    """The handler reads with ``.get(key, 0)``, so a partial metadata dict
    contributes only the keys it carries. Pydantic requires all three token
    keys at AIMessage construction, so the partial shape is applied by
    post-construction mutation to reach the handler directly."""
    handler = StatsCallbackHandler()
    message = AIMessage(
        content="ok",
        usage_metadata={"input_tokens": 5, "output_tokens": 1, "total_tokens": 6},
    )
    message.usage_metadata = {"input_tokens": 5}
    handler.on_llm_end(LLMResult(generations=[[ChatGeneration(message=message)]]))
    stats = handler.get_stats()
    assert stats["tokens_in"] == 5
    assert stats["tokens_out"] == 0


@pytest.mark.unit
def test_on_llm_end_empty_usage_metadata_is_skipped():
    handler = StatsCallbackHandler()
    # Empty dict is falsy -> the accumulation block is skipped entirely.
    message = AIMessage(content="ok")
    message.usage_metadata = {}
    handler.on_llm_end(LLMResult(generations=[[ChatGeneration(message=message)]]))
    assert handler.get_stats()["tokens_in"] == 0


@pytest.mark.unit
def test_on_llm_end_nested_token_details_do_not_affect_totals():
    handler = StatsCallbackHandler()
    result = _result(
        10,
        4,
        extra={"input_token_details": {"cache_read": 6},
               "output_token_details": {"reasoning": 2}},
    )
    handler.on_llm_end(result)
    stats = handler.get_stats()
    assert stats["tokens_in"] == 10  # nested details are NOT re-counted
    assert stats["tokens_out"] == 4


@pytest.mark.unit
def test_on_llm_end_empty_generations_is_noop():
    """IndexError path: ``generations[0][0]`` on an empty result returns early."""
    handler = StatsCallbackHandler()
    handler.on_llm_end(LLMResult(generations=[]))
    handler.on_llm_end(LLMResult(generations=[[]]))
    assert handler.get_stats() == {
        "llm_calls": 0,
        "tool_calls": 0,
        "tokens_in": 0,
        "tokens_out": 0,
    }


@pytest.mark.unit
def test_on_llm_end_non_chat_generation_is_noop():
    """A plain Generation has no ``.message`` at all, and a chat generation
    whose message is not an AIMessage is skipped by the isinstance guard —
    nothing to extract, no raise."""
    handler = StatsCallbackHandler()
    handler.on_llm_end(LLMResult(generations=[[Generation(text="plain")]]))
    handler.on_llm_end(
        LLMResult(generations=[[ChatGeneration(message=HumanMessage(content="x"))]])
    )
    assert handler.get_stats()["tokens_in"] == 0


@pytest.mark.unit
def test_on_llm_end_malformed_usage_metadata_pins_current_brittleness():
    """KNOWN GAP (documented, intentionally not fixed by this test suite):
    the accumulation block does ``tokens += value`` without checking the
    value type, and truthy non-dict metadata has no ``.get``. Pydantic
    blocks such shapes at AIMessage construction, but the handler itself
    does not defend in depth. These assertions pin today's behavior so a
    future hardening shows up as a deliberate diff."""
    handler = StatsCallbackHandler()

    def _message_with(metadata):
        """Valid AIMessage, then swap in a shape pydantic would reject —
        langchain does not validate attribute assignment, so this is how a
        malformed payload could reach the handler in practice."""
        message = AIMessage(
            content="ok",
            usage_metadata={"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
        )
        message.usage_metadata = metadata
        return message

    non_numeric = _message_with(
        {"input_tokens": "many", "output_tokens": 3, "total_tokens": 4}
    )
    with pytest.raises(TypeError):
        handler.on_llm_end(LLMResult(generations=[[ChatGeneration(message=non_numeric)]]))

    non_dict = _message_with("garbage")
    with pytest.raises(AttributeError):
        handler.on_llm_end(LLMResult(generations=[[ChatGeneration(message=non_dict)]]))


@pytest.mark.unit
def test_on_llm_error_is_inherited_noop_and_not_counted():
    """The handler defines no on_llm_error, so langchain's base no-op runs:
    an errored LLM call neither raises here nor increments llm_calls (a
    failed call is invisible in the stats — known gap, documented)."""
    handler = StatsCallbackHandler()
    handler.on_llm_start({"id": ["m"]}, ["p"])
    handler.on_llm_error(RuntimeError("provider 500"), run_id=uuid.uuid4())
    stats = handler.get_stats()
    assert stats["llm_calls"] == 1  # only the start counted
    assert stats["tokens_in"] == 0


# --------------------------------------------------------------------------- #
# Thread safety: interleaved start/end across threads
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_concurrent_start_end_tool_interleave_keeps_counts_exact():
    """Workers interleave on_chat_model_start / on_llm_start, on_llm_end and
    on_tool_start concurrently; the lock must make every increment survive
    (no lost updates) so the final tallies are exactly the per-work sums."""
    handler = StatsCallbackHandler()
    threads_n, iters_n = 16, 25
    barrier = threading.Barrier(threads_n)

    def work(worker: int) -> None:
        barrier.wait()
        for i in range(iters_n):
            if (worker + i) % 2 == 0:
                handler.on_chat_model_start({"id": ["m"]}, [[AIMessage(content="x")]])
            else:
                handler.on_llm_start({"id": ["m"]}, ["p"])
            if i % 3 == 0:
                handler.on_tool_start({"name": "t"}, "in")
            handler.on_llm_end(_result(2, 1))

    with ThreadPoolExecutor(max_workers=threads_n) as pool:
        list(pool.map(work, range(threads_n)))

    total = threads_n * iters_n
    stats = handler.get_stats()
    assert stats["llm_calls"] == total
    # tool call every 3rd iteration (i % 3 == 0): iters 0,3,...,24 -> 9 of 25
    assert stats["tool_calls"] == threads_n * 9
    assert stats["tokens_in"] == 2 * total
    assert stats["tokens_out"] == total
