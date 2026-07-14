"""Unit tests for ``yiagents.llm_clients.response_cache.DiskLLMCache``.

Exercises the cache directly (no network, no real LLM): round-trip fidelity
for content + tool_calls, disabled=no-op, corrupt-entry eviction, version
mismatch, key isolation across bound-tools, clear(), and a global-cache
install/teardown smoke. langchain's global cache is always restored in
teardown so nothing leaks across the suite.
"""

from __future__ import annotations

import json

import pytest
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration

from yiagents.llm_clients.response_cache import DiskLLMCache


def _gen(content: str = "hello", tool_calls=None) -> ChatGeneration:
    msg = AIMessage(content=content, tool_calls=tool_calls or [])
    return ChatGeneration(message=msg)


@pytest.mark.unit
def test_update_then_lookup_roundtrips_content_and_tool_calls(tmp_path):
    cache = DiskLLMCache(tmp_path)
    gen = _gen(
        "buy",
        tool_calls=[{"name": "decide", "args": {"rating": "BUY"}, "id": "c1"}],
    )
    cache.update("prompt-a", "llm-str-a", [gen])

    hit = cache.lookup("prompt-a", "llm-str-a")
    assert hit is not None
    assert len(hit) == 1
    got = hit[0]
    assert isinstance(got, ChatGeneration)
    assert got.message.content == "buy"
    assert got.message.tool_calls[0]["name"] == "decide"
    assert got.message.tool_calls[0]["args"] == {"rating": "BUY"}


@pytest.mark.unit
def test_disabled_cache_is_noop_and_writes_nothing(tmp_path):
    cache = DiskLLMCache(tmp_path, enabled=False)
    cache.update("p", "l", [_gen("x")])
    assert cache.lookup("p", "l") is None
    assert list(tmp_path.glob("*.json")) == []


@pytest.mark.unit
def test_corrupt_entry_is_miss_and_evicted(tmp_path):
    cache = DiskLLMCache(tmp_path)
    cache.update("p", "l", [_gen("ok")])
    files = list(tmp_path.glob("*.json"))
    assert len(files) == 1
    files[0].write_text("{not valid json", encoding="utf-8")

    assert cache.lookup("p", "l") is None
    assert not files[0].exists()  # evicted so next time it's a clean miss


@pytest.mark.unit
def test_version_mismatch_is_miss(tmp_path):
    cache = DiskLLMCache(tmp_path)
    cache.update("p", "l", [_gen("ok")])
    files = list(tmp_path.glob("*.json"))
    data = json.loads(files[0].read_text(encoding="utf-8"))
    data["v"] = 999  # simulate a future schema change
    files[0].write_text(json.dumps(data), encoding="utf-8")

    assert cache.lookup("p", "l") is None


@pytest.mark.unit
def test_distinct_llm_string_keys_do_not_collide(tmp_path):
    """A plain invoke and a structured-output call share the same prompt but
    differ in bound tools, so langchain gives them different llm_string — the
    cache must keep them as two separate entries."""
    cache = DiskLLMCache(tmp_path)
    cache.update("same-prompt", "llm-plain", [_gen("plain")])
    cache.update("same-prompt", "llm-with-tools", [_gen("structured")])

    assert cache.lookup("same-prompt", "llm-plain")[0].message.content == "plain"
    assert (
        cache.lookup("same-prompt", "llm-with-tools")[0].message.content
        == "structured"
    )
    assert len(list(tmp_path.glob("*.json"))) == 2


@pytest.mark.unit
def test_clear_removes_entries(tmp_path):
    cache = DiskLLMCache(tmp_path)
    cache.update("p1", "l1", [_gen("a")])
    cache.update("p2", "l2", [_gen("b")])
    assert len(list(tmp_path.glob("*.json"))) == 2

    cache.clear()

    assert list(tmp_path.glob("*.json")) == []
    assert cache.lookup("p1", "l1") is None


@pytest.mark.unit
def test_global_cache_install_round_trip_and_teardown(tmp_path):
    """Installing via set_llm_cache makes the global accessor serve stored
    generations back. The previous global cache is always restored so this
    never leaks into other tests."""
    from langchain_core.globals import get_llm_cache, set_llm_cache

    prev = get_llm_cache()
    cache = DiskLLMCache(tmp_path)
    set_llm_cache(cache)
    try:
        cache.update("gp", "gl", [_gen("replay")])
        active = get_llm_cache()
        assert active is cache
        assert active.lookup("gp", "gl")[0].message.content == "replay"
    finally:
        set_llm_cache(prev)
    assert get_llm_cache() is prev


@pytest.mark.unit
def test_lookup_miss_for_unseen_key_returns_none(tmp_path):
    cache = DiskLLMCache(tmp_path)
    cache.update("p", "l", [_gen("x")])
    assert cache.lookup("other-prompt", "l") is None
    assert cache.lookup("p", "other-llm") is None
