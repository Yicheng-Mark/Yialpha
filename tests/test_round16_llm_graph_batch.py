"""Round 16 — llm_graph_batch shard: audit regressions + coverage gaps.

Owned modules under test: ``yialpha/llm_clients`` (key pool, response cache,
providers), ``yialpha/graph`` (routing / checkpointer / execution),
``yialpha/batch`` (runner + locks), monitoring, regime, logging/config helpers.

Every test is hermetic (conftest's network guard is autouse): LLM objects are
real SDK clients driven by ``httpx.MockTransport`` or plain fakes — never a
live call. Dates are frozen, nothing sleeps.
"""

from __future__ import annotations

import json
import shutil
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from types import SimpleNamespace
from typing import Any, TypedDict

import httpx
import pytest
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

from yialpha.dataflows.config import set_config
from yialpha.graph import routing
from yialpha.graph.analyst_execution import (
    AnalystWallTimeTracker,
    build_analyst_execution_plan,
)
from yialpha.graph.checkpointer import (
    checkpoint_step,
    clear_checkpoint,
    get_checkpointer,
    has_checkpoint,
    thread_id,
)
from yialpha.graph.routing import (
    UNKNOWN_PERP,
    describe_instrument,
    instrument_class,
    is_exchange_traded_fund,
)
from yialpha.graph.trading_graph import _SIGNATURE_CONFIG_KEYS
from yialpha.instruments.registry import SOURCE_EXCHANGEINFO, SOURCE_REGISTRY_EMPTY
from yialpha.instruments.sessions import (
    SESSION_BINANCE_TRADFI,
    SESSION_CONTINUOUS,
)
from yialpha.llm_clients import key_pool
from yialpha.llm_clients.key_pool import (
    DEFAULT_COOLDOWN_S,
    RotatingKeyAuth,
    _PoolState,
    get_pool_http_client,
    reset_for_test,
)
from yialpha.llm_clients.response_cache import DiskLLMCache, _key

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

_K1 = "k1aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
_K2 = "k2aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
_K3 = "k3aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
_URL = "https://pool.test/v1/chat/completions"


@pytest.fixture(autouse=True)
def _clean_pool_env(monkeypatch):
    """Isolate the pool env vars + drop process-wide pooled clients per test."""
    for env_var in (
        "ZHIPU_API_KEYS",
        "ZHIPU_CN_API_KEYS",
        "ZHIPU_CN_BASE_URL",
        "ZHIPU_BASE_URL",
        "YIALPHA_LLM_KEY_COOLDOWN_S",
        "YIALPHA_LLM_TIMEOUT_S",
        "ZHIPU_API_KEY",
        "ZHIPU_CN_API_KEY",
    ):
        monkeypatch.delenv(env_var, raising=False)
    reset_for_test()
    yield
    reset_for_test()


def _set_pool(monkeypatch, *keys):
    monkeypatch.setenv("ZHIPU_CN_API_KEYS", ",".join(keys))


def _mock_client(status_by_key, keys, state):
    """httpx.Client over a MockTransport answering by Authorization key."""
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        key = request.headers.get("Authorization", "").removeprefix("Bearer ")
        seen.append(key)
        status = status_by_key.get(key, 200)
        return httpx.Response(status, json={"ok": status == 200}, request=request)

    client = httpx.Client(
        transport=httpx.MockTransport(handler),
        auth=RotatingKeyAuth(list(keys), state),
    )
    return client, seen


# ---------------------------------------------------------------------------
# key_pool: rotation, failure policy, thread safety
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_r16_key_pool_force_pick_prefers_cooling_then_lowest_permanent():
    """force_pick: a cooling key outranks a permanently-dead one (even one with
    a stale cooldown entry); among cooling keys the soonest-to-revive wins;
    with nothing cooling it falls back to the LOWEST dead index so the request
    still reaches the provider and fails with its real status."""
    state = _PoolState(3, cooldown_s=60.0)
    for idx in (0, 2):
        state.mark(idx, 401)
    assert state.force_pick() == 0  # nothing cooling -> lowest permanent index

    state.mark(1, 429)
    assert state.force_pick() == 1  # a cooling key always wins

    state.dead_until[2] = time.monotonic() + 5.0  # permanent + stale cooldown
    state.dead_until[1] = time.monotonic() + 90.0
    assert state.force_pick() == 1

    two_cooling = _PoolState(2, cooldown_s=60.0)
    two_cooling.dead_until = {0: time.monotonic() + 90.0, 1: time.monotonic() + 5.0}
    assert two_cooling.force_pick() == 1
    assert _PoolState(0, 1.0).force_pick() is None


@pytest.mark.unit
def test_r16_key_pool_every_key_auth_dead_each_tried_once_then_forced_once():
    """3 keys all answer 401: ONE request cycle tries every key exactly once
    (so each is kicked out of the process) and the last 401 propagates; every
    later request costs exactly ONE attempt on the lowest dead index (no
    rotation storm while the SDK retries)."""
    state = _PoolState(3, cooldown_s=60.0)
    client, seen = _mock_client({_K1: 401, _K2: 401, _K3: 401}, [_K1, _K2, _K3], state)

    assert client.post(_URL, json={}).status_code == 401
    assert seen == [_K1, _K2, _K3]
    assert state.permanent == {0, 1, 2}

    seen.clear()
    assert client.post(_URL, json={}).status_code == 401
    assert seen == [_K1]


@pytest.mark.unit
def test_r16_key_pool_forced_429_keeps_original_cooldown_deadline():
    """Round-6 contract: when every key is cooling and the force-served key 429s
    again, its revival deadline is NOT pushed out — otherwise a saturated pool
    never revives."""
    state = _PoolState(2, cooldown_s=60.0)
    client, _ = _mock_client({_K1: 429, _K2: 429}, [_K1, _K2], state)
    assert client.post(_URL, json={}).status_code == 429
    deadlines = dict(state.dead_until)
    assert set(deadlines) == {0, 1}

    # Three more saturated requests: the forced key's deadline stays put.
    for _ in range(3):
        assert client.post(_URL, json={}).status_code == 429
    assert state.dead_until == deadlines


@pytest.mark.unit
def test_r16_key_pool_cooldown_revives_key_back_into_rotation():
    """A 429'd key leaves the rotation while cooling and rejoins at its
    round-robin slot once its cooldown elapsed (time advanced by rewinding the
    deadline — no sleep)."""
    state = _PoolState(2, cooldown_s=60.0)
    status = {_K1: 429}
    client, seen = _mock_client(status, [_K1, _K2], state)

    # k1 429s -> the SAME request is re-sent on k2 and succeeds.
    assert client.post(_URL, json={}).status_code == 200
    assert seen == [_K1, _K2]
    # k1 is cooling: the next request skips straight to k2.
    seen.clear()
    assert client.post(_URL, json={}).status_code == 200
    assert seen == [_K2]

    # Cooldown elapsed (and k1 healed) -> it is served again.
    state.dead_until[0] = time.monotonic() - 1.0
    status.clear()
    seen.clear()
    assert client.post(_URL, json={}).status_code == 200
    assert seen == [_K1]
    assert state.is_live(0) is True


@pytest.mark.unit
def test_r16_key_pool_forced_401_promotes_cooling_key_to_permanent():
    """A cooling key that answers 401 while force-served is auth-dead, not
    merely cooling: it must never come back when its cooldown lapses."""
    state = _PoolState(1, cooldown_s=60.0)
    client, seen = _mock_client({_K1: 401}, [_K1], state)
    state.mark(0, 429)  # cooling
    assert client.post(_URL, json={}).status_code == 401
    assert seen == [_K1]
    assert state.permanent == {0}

    state.dead_until[0] = time.monotonic() - 1.0  # cooldown would have lapsed
    assert state.is_live(0) is False
    assert state.pick() is None
    assert state.force_pick() == 0  # still force-servable, never "live"


@pytest.mark.unit
def test_r16_key_pool_concurrent_picks_are_exactly_balanced():
    """Thread safety: N threads x M picks over K keys under the module lock must
    hand out each key exactly N*M/K times (cursor increments never get lost)."""
    keys_n, threads_n, per_thread = 4, 8, 250
    state = _PoolState(keys_n, cooldown_s=60.0)

    def worker():
        return [state.pick() for _ in range(per_thread)]

    with ThreadPoolExecutor(max_workers=threads_n) as pool:
        results = list(pool.map(lambda _: worker(), range(threads_n)))
    counts = Counter(idx for batch in results for idx in batch)
    assert counts == dict.fromkeys(range(keys_n), threads_n * per_thread // keys_n)


@pytest.mark.unit
def test_r16_key_pool_concurrent_requests_rotate_without_duplicate_keys():
    """Thread safety at the transport: concurrent requests sharing one pooled
    client each carry exactly one well-formed key, and the load is spread
    evenly across the pool (no key starved / double-served)."""
    keys = [_K1, _K2, _K3, "k4" + _K1[2:]]
    state = _PoolState(len(keys), cooldown_s=60.0)
    seen: list[str] = []
    lock = threading.Lock()

    def handler(request: httpx.Request) -> httpx.Response:
        with lock:
            seen.append(request.headers.get("Authorization", ""))
        return httpx.Response(200, json={"ok": True}, request=request)

    client = httpx.Client(
        transport=httpx.MockTransport(handler),
        auth=RotatingKeyAuth(keys, state),
    )

    def send(_):
        for _ in range(50):
            assert client.post(_URL, json={}).status_code == 200

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(send, range(8)))
    assert len(seen) == 400
    assert Counter(seen) == dict.fromkeys((f"Bearer {k}" for k in keys), 100)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (None, 120.0),  # unset -> cloud default
        ("0", None),  # explicit off: no ceiling (watchdog owns hang recovery)
        ("0.0", None),
        ("45.5", 45.5),
        (" 300 ", 300.0),
        ("", 120.0),
        ("abc", 120.0),  # invalid falls back silently (resolve_timeout warned)
        ("-1", 120.0),
        ("nan", 120.0),
        ("inf", 120.0),
    ],
)
def test_r16_key_pool_transport_timeout_matrix(monkeypatch, raw, expected):
    """The pooled client's httpx ceiling mirrors resolve_timeout's precedence:
    explicit positive finite -> that value, 0 -> no ceiling, anything else ->
    the 120s cloud default; connect stays 15s whenever a ceiling applies."""
    if raw is not None:
        monkeypatch.setenv("YIALPHA_LLM_TIMEOUT_S", raw)
    timeout = key_pool._pool_transport_timeout()
    if expected is None:
        assert timeout is None
    else:
        assert timeout.read == expected
        assert timeout.connect == 15.0
        assert timeout.pool == 10.0


@pytest.mark.unit
def test_r16_key_pool_client_is_process_wide_and_ignores_env_change(monkeypatch):
    """One pooled client per provider per process: the first caller's key set and
    cooldown are frozen — a mid-process env change must not re-arm the pool;
    ``reset_for_test`` is the only thing that re-arms it (and closes the old
    client)."""
    _set_pool(monkeypatch, _K1, _K2)
    monkeypatch.setenv("YIALPHA_LLM_KEY_COOLDOWN_S", "7")
    first = get_pool_http_client("glm-cn")
    assert isinstance(first, httpx.Client)
    assert get_pool_http_client("glm-cn") is first
    # Upper-cased provider id resolves to the SAME process-wide client.
    assert get_pool_http_client("GLM-CN") is first

    monkeypatch.setenv("ZHIPU_CN_API_KEYS", _K3)
    monkeypatch.setenv("YIALPHA_LLM_KEY_COOLDOWN_S", "999")
    assert get_pool_http_client("glm-cn") is first
    assert first.auth.keys == [_K1, _K2]
    assert first.auth.state.cooldown_s == 7.0

    reset_for_test()
    fresh = get_pool_http_client("glm-cn")
    assert fresh is not first
    assert first.is_closed
    assert fresh.auth.keys == [_K3]
    assert fresh.auth.state.cooldown_s == 999.0


@pytest.mark.unit
def test_r16_key_pool_empty_parsed_pool_creates_no_client(monkeypatch):
    """A pool env var that engages the pool gate but parses to zero keys must
    not build or cache a pooled client, and must not poison the process-wide
    cache for the next (valid) caller."""
    _set_pool(monkeypatch, _K1)
    real_parse = key_pool.api_key_pool
    monkeypatch.setattr(key_pool, "api_key_pool", lambda provider: [])
    assert get_pool_http_client("glm-cn") is None
    assert key_pool._clients == {}
    assert key_pool._states == {}

    monkeypatch.setattr(key_pool, "api_key_pool", real_parse)
    assert isinstance(get_pool_http_client("glm-cn"), httpx.Client)


@pytest.mark.unit
def test_r16_key_pool_cooldown_seconds_matrix(monkeypatch, caplog):
    """YIALPHA_LLM_KEY_COOLDOWN_S floor: negative / NaN clamp to the 1s floor,
    a non-numeric value warns and falls back to the default."""
    monkeypatch.setenv("YIALPHA_LLM_KEY_COOLDOWN_S", "-5")
    assert key_pool.cooldown_seconds() == 1.0
    monkeypatch.setenv("YIALPHA_LLM_KEY_COOLDOWN_S", "nan")
    assert key_pool.cooldown_seconds() == 1.0
    with caplog.at_level("WARNING", logger="yialpha.llm_clients.key_pool"):
        monkeypatch.setenv("YIALPHA_LLM_KEY_COOLDOWN_S", "later")
        assert key_pool.cooldown_seconds() == DEFAULT_COOLDOWN_S
    assert "YIALPHA_LLM_KEY_COOLDOWN_S" in caplog.text


# ---------------------------------------------------------------------------
# response_cache: the cache key must contain everything that shapes a call
# ---------------------------------------------------------------------------


@tool
def _get_price(ticker: str) -> float:
    """Look up the last close for a ticker."""
    return 1.0


@tool
def _get_volume(ticker: str) -> float:
    """Look up the traded volume for a ticker."""
    return 1.0


def _gen(content: str = "hi") -> ChatGeneration:
    return ChatGeneration(message=AIMessage(content))


class _CountingTransport:
    """MockTransport whose every live call answers ``live-<n>`` — the body a
    test observes tells it whether the model was really called or replayed."""

    def __init__(self, tool_call: bool = False):
        self.calls = 0
        self.bodies: list[dict] = []
        self.tool_call = tool_call

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.calls += 1
        self.bodies.append(json.loads(request.content))
        message: dict = {"role": "assistant", "content": f"live-{self.calls}"}
        if self.tool_call:
            message = {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "_get_price", "arguments": '{"ticker": "AAPL"}'},
                    }
                ],
            }
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-1",
                "object": "chat.completion",
                "created": 1700000000,
                "model": "m-a",
                "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
            },
            request=request,
        )


def _chat(transport: _CountingTransport, **kwargs) -> ChatOpenAI:
    params: dict = {"model": "m-a", "api_key": "x"}
    params.update(kwargs)
    return ChatOpenAI(
        **params,
        http_client=httpx.Client(transport=httpx.MockTransport(transport.handler)),
    )


@pytest.fixture
def _global_cache(tmp_path):
    """Install a fresh DiskLLMCache as langchain's global cache for one test."""
    from langchain_core.globals import set_llm_cache

    cache = DiskLLMCache(tmp_path / "llm_cache")
    set_llm_cache(cache)
    yield cache
    set_llm_cache(None)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("label", "variant"),
    [
        ("model", lambda t: _chat(t, model="m-b")),
        ("temperature", lambda t: _chat(t, temperature=0.7)),
        ("bound_tools", lambda t: _chat(t).bind_tools([_get_price])),
        ("different_tools", lambda t: _chat(t).bind_tools([_get_volume])),
        ("tool_choice", lambda t: _chat(t).bind_tools([_get_price], tool_choice="required")),
        (
            "structured_schema",
            lambda t: _chat(t).bind(
                response_format={"type": "json_schema", "json_schema": {"name": "Verdict"}}
            ),
        ),
        (
            "other_schema",
            lambda t: _chat(t).bind(
                response_format={"type": "json_schema", "json_schema": {"name": "Trade"}}
            ),
        ),
    ],
)
def test_r16_response_cache_key_separates_call_shape(_global_cache, label, variant):
    """End-to-end through langchain's own cache path: an entry written under one
    call shape must NOT be replayed to a call that differs in model /
    temperature / bound tools / tool_choice / structured schema — the variant
    reaches the transport (a real call) and gets its own live answer."""
    transport = _CountingTransport()
    plain = _chat(transport)
    assert plain.invoke("prompt").content == "live-1"
    assert plain.invoke("prompt").content == "live-1"  # replayed: no 2nd call
    assert transport.calls == 1

    assert variant(transport).invoke("prompt").content == "live-2"
    assert transport.calls == 2
    # The variant is cached under its OWN key: replaying it is free too, and
    # the plain call still replays its original answer.
    assert variant(transport).invoke("prompt").content == "live-2"
    assert plain.invoke("prompt").content == "live-1"
    assert transport.calls == 2
    assert len(list(_global_cache.cache_dir.iterdir())) == 2


@pytest.mark.unit
def test_r16_response_cache_replays_tool_calls_and_usage_end_to_end(_global_cache):
    """A cached tool-calling generation round-trips through the real cache path
    with ``tool_calls`` + ``usage_metadata`` intact — no second transport call."""
    transport = _CountingTransport(tool_call=True)
    llm = _chat(transport).bind_tools([_get_price])
    first = llm.invoke("prompt")
    assert first.tool_calls[0]["name"] == "_get_price"
    assert first.usage_metadata["total_tokens"] == 5

    replay = llm.invoke("prompt")
    assert transport.calls == 1
    assert replay.content == first.content
    assert replay.tool_calls == first.tool_calls
    # Token counts survive the round-trip (langchain may add cost bookkeeping
    # on the replay path, so compare the counted fields, not the whole dict).
    for field in ("input_tokens", "output_tokens", "total_tokens"):
        assert replay.usage_metadata[field] == first.usage_metadata[field] == {
            "input_tokens": 3,
            "output_tokens": 2,
            "total_tokens": 5,
        }[field]


@pytest.mark.unit
def test_r16_response_cache_key_separates_prompt_and_is_stable(tmp_path):
    """Same llm_string + different prompt = different key; the key is a stable
    filename-safe hex digest, so a second process finds the same entry."""
    cache = DiskLLMCache(tmp_path)
    cache.update("prompt A", "llm-string", [_gen("answer A")])
    cache.update("prompt B", "llm-string", [_gen("answer B")])

    assert cache.lookup("prompt B", "llm-string")[0].message.content == "answer B"
    (hit,) = cache.lookup("prompt A", "llm-string")
    assert hit.message.content == "answer A"
    assert hit.generation_info["cached_replay"] is True

    digest = _key("prompt A", "llm-string")
    assert len(digest) == 40
    assert set(digest) <= set("0123456789abcdef")
    assert (tmp_path / f"{digest}.json").is_file()


@pytest.mark.unit
def test_r16_response_cache_default_off_no_dir_no_io(tmp_path):
    """Default-off contract: a disabled cache creates no directory, answers
    every lookup with None, writes nothing on update and is a no-op on clear."""
    target = tmp_path / "llm"
    cache = DiskLLMCache(target, enabled=False)
    assert not target.exists()
    cache.update("p", "s", [_gen()])
    cache.clear()
    assert not target.exists()
    assert cache.lookup("p", "s") is None


@pytest.mark.unit
def test_r16_response_cache_lookup_stamps_replay_on_existing_dict_info(tmp_path):
    """A replayed generation carries ``cached_replay`` even when its on-disk
    generation_info already holds a dict (token accounting must see it); the
    stamp is in-memory only — the on-disk entry keeps the original info."""
    cache = DiskLLMCache(tmp_path)
    cache.update(
        "p",
        "s",
        [
            ChatGeneration(
                message=AIMessage("x"),
                generation_info={"model_name": "m-a", "finish_reason": "stop"},
            )
        ],
    )
    (hit,) = cache.lookup("p", "s")
    assert hit.generation_info == {
        "model_name": "m-a",
        "finish_reason": "stop",
        "cached_replay": True,
    }
    on_disk = json.loads((tmp_path / f"{_key('p', 's')}.json").read_text(encoding="utf-8"))
    assert on_disk["gens"][0]["generation_info"] == {
        "model_name": "m-a",
        "finish_reason": "stop",
    }


@pytest.mark.unit
def test_r16_response_cache_unknown_message_type_is_miss_and_evicted(tmp_path):
    """A payload whose message ``type`` is not a known subclass reads as a
    miss AND is evicted — a foreign on-disk schema can never break a run."""
    cache = DiskLLMCache(tmp_path)
    path = tmp_path / f"{_key('p', 's')}.json"
    path.write_text(
        json.dumps(
            {"v": 1, "gens": [{"is_chat": True, "message": {"type": "weird", "content": ""}}]}
        ),
        encoding="utf-8",
    )
    assert cache.lookup("p", "s") is None
    assert not path.exists()


@pytest.mark.unit
def test_r16_response_cache_clear_noop_when_disabled_or_dir_missing(tmp_path):
    """clear() is a no-op for a disabled cache and must not raise when the
    directory vanished between construction and the call."""
    target = tmp_path / "llm"
    disabled = DiskLLMCache(target, enabled=False)
    target.mkdir()
    keep = target / "a.json"
    keep.write_text("{}", encoding="utf-8")
    disabled.clear()
    assert keep.exists()

    live = DiskLLMCache(target)
    shutil.rmtree(target)
    live.clear()  # must not raise on the vanished dir
    assert not target.exists()


@pytest.mark.unit
def test_r16_response_cache_clear_sweeps_tmp_husks(tmp_path):
    """clear() also removes writer-crash ``*.tmp`` husks that the ``*.json``
    glob never matches."""
    cache = DiskLLMCache(tmp_path)
    cache.update("p", "s", [_gen()])
    (tmp_path / "x.json.1.2.tmp").write_text("{}", encoding="utf-8")
    (tmp_path / f"{_key('p', 's')}.json.4242.abc.tmp").write_text("{partial", encoding="utf-8")
    assert len(list(tmp_path.iterdir())) == 3
    DiskLLMCache(tmp_path).clear()
    assert list(tmp_path.iterdir()) == []


# ---------------------------------------------------------------------------
# graph: checkpointer — signature flip + the None-input resume contract
# ---------------------------------------------------------------------------

_DATE = "2026-01-05"


class _ResumeState(TypedDict):
    count: int


def _counting_graph(saver, runs: list[str], *, crash_once: bool):
    """Two-node graph (analyst -> trader) that logs every node execution and,
    on demand, crashes the FIRST time the trader runs — like a mid-analysis
    LLM failure."""
    def analyst(state):
        runs.append("analyst")
        return {"count": state["count"] + 1}

    def trader(state):
        runs.append("trader")
        if crash_once and runs.count("trader") == 1:
            raise RuntimeError("simulated mid-analysis crash")
        return {"count": state["count"] + 10}

    builder = StateGraph(_ResumeState)
    builder.add_node("analyst", analyst)
    builder.add_node("trader", trader)
    builder.set_entry_point("analyst")
    builder.add_edge("analyst", "trader")
    builder.add_edge("trader", END)
    return builder.compile(checkpointer=saver)


@pytest.mark.unit
def test_r16_checkpointer_none_input_resumes_where_dict_input_reruns(tmp_path):
    """The contract _run_graph relies on: after a crash a ``None`` input resumes
    at the first un-checkpointed node (no re-billing), while ANY dict input
    re-runs the whole graph from ``__start__`` — and a ``None`` input on an
    already-finished thread replays the final state with zero node runs."""
    runs: list[str] = []
    cfg = {"configurable": {"thread_id": thread_id("TEST", _DATE, "sig")}}
    with (
        get_checkpointer(tmp_path, "TEST") as saver,
        pytest.raises(RuntimeError, match="simulated"),
    ):
        _counting_graph(saver, runs, crash_once=True).invoke({"count": 0}, config=cfg)
    assert runs == ["analyst", "trader"]
    assert checkpoint_step(tmp_path, "TEST", _DATE, "sig") == 1  # analyst done

    # Resume with None: the analyst is NOT re-executed.
    with get_checkpointer(tmp_path, "TEST") as saver:
        final = _counting_graph(saver, runs, crash_once=False).invoke(None, config=cfg)
    assert final["count"] == 11
    assert runs == ["analyst", "trader", "trader"]

    # Finished thread + None input: final state replayed, nothing re-runs.
    assert checkpoint_step(tmp_path, "TEST", _DATE, "sig") is not None
    with get_checkpointer(tmp_path, "TEST") as saver:
        again = _counting_graph(saver, runs, crash_once=False).invoke(None, config=cfg)
    assert again["count"] == 11
    assert runs == ["analyst", "trader", "trader"]

    # Contrast — same crash, but the retry feeds a fresh init-state dict.
    other: list[str] = []
    cfg2 = {"configurable": {"thread_id": thread_id("TEST", _DATE, "sig2")}}
    with (
        get_checkpointer(tmp_path, "TEST") as saver,
        pytest.raises(RuntimeError, match="simulated"),
    ):
        _counting_graph(saver, other, crash_once=True).invoke({"count": 0}, config=cfg2)
    with get_checkpointer(tmp_path, "TEST") as saver:
        restarted = _counting_graph(saver, other, crash_once=False).invoke(
            {"count": 0}, config=cfg2
        )
    assert restarted["count"] == 11
    assert other == ["analyst", "trader", "analyst", "trader"]  # from __start__


@pytest.mark.unit
def test_r16_checkpointer_signature_flip_never_resumes_stale_thread(tmp_path):
    """A checkpoint written under graph shape A is invisible under shape B
    (fresh run instead of resuming a differently-shaped graph), and clearing
    shape B's thread leaves shape A's checkpoint untouched."""
    runs: list[str] = []
    cfg_a = {"configurable": {"thread_id": thread_id("TEST", _DATE, "shape-A")}}
    with (
        get_checkpointer(tmp_path, "TEST") as saver,
        pytest.raises(RuntimeError, match="simulated"),
    ):
        _counting_graph(saver, runs, crash_once=True).invoke({"count": 0}, config=cfg_a)

    assert thread_id("TEST", _DATE, "shape-A") != thread_id("TEST", _DATE, "shape-B")
    assert checkpoint_step(tmp_path, "TEST", _DATE, "shape-A") == 1
    assert checkpoint_step(tmp_path, "TEST", _DATE, "shape-B") is None
    assert has_checkpoint(tmp_path, "TEST", _DATE, "shape-B") is False

    clear_checkpoint(tmp_path, "TEST", _DATE, "shape-B")  # B has nothing to clear
    assert checkpoint_step(tmp_path, "TEST", _DATE, "shape-A") == 1

    clear_checkpoint(tmp_path, "TEST", _DATE, "shape-A")
    assert checkpoint_step(tmp_path, "TEST", _DATE, "shape-A") is None
    assert has_checkpoint(tmp_path, "TEST", _DATE, "shape-A") is False


@pytest.mark.unit
def test_r16_checkpointer_step_on_missing_db_is_none_without_creating_db(tmp_path):
    """The pre-run probe on a ticker that never checkpointed is a pure read:
    None / False, and no DB file is materialised for it."""
    assert checkpoint_step(tmp_path, "NEVER", _DATE, "sig") is None
    assert has_checkpoint(tmp_path, "NEVER", _DATE, "sig") is False
    assert list(tmp_path.rglob("*.db")) == []


# ---------------------------------------------------------------------------
# graph: run signature flips with every graph-shape input
# ---------------------------------------------------------------------------


def _signature_shell(**config: Any):
    """YiAlphaGraph shell with ``__init__`` skipped (established suite pattern)."""
    from yialpha.graph.trading_graph import YiAlphaGraph

    graph = YiAlphaGraph.__new__(YiAlphaGraph)
    graph.config = {"max_debate_rounds": 1, "max_risk_discuss_rounds": 1, **config}
    graph.selected_analysts = ["market", "news"]
    return graph


def _flip_config_key(key: str, graph) -> None:
    graph.config[key] = {"flipped": key}


_SIGNATURE_FLIPS = [
    ("analysts", lambda g: setattr(g, "selected_analysts", ["market"])),
    ("analyst_order", lambda g: setattr(g, "selected_analysts", ["news", "market"])),
    ("debate", lambda g: g.config.__setitem__("max_debate_rounds", 2)),
    ("risk", lambda g: g.config.__setitem__("max_risk_discuss_rounds", 3)),
    ("parallel", lambda g: g.config.__setitem__("analyst_parallel", True)),
] + [("config:" + key, partial(_flip_config_key, key)) for key in _SIGNATURE_CONFIG_KEYS]


@pytest.mark.unit
@pytest.mark.parametrize(("label", "mutate"), _SIGNATURE_FLIPS)
def test_r16_run_signature_flips_with_graph_shape_inputs(label, mutate):
    """Every graph-shape input — analyst set and its order, debate/risk depth,
    parallel mode, and EVERY key listed in ``_SIGNATURE_CONFIG_KEYS`` — changes
    the run signature, so a checkpoint restored under a different shape starts
    fresh instead of serving stale node outputs."""
    graph = _signature_shell()
    before = graph._run_signature("stock")
    assert graph._run_signature("stock") == before  # stable for a fixed config
    mutate(graph)
    after = graph._run_signature("stock")
    assert after != before, f"signature must flip on {label}"
    if label.startswith("config:"):
        # The flip is visible in the machine-parseable sorted-JSON segment.
        key = label.removeprefix("config:")
        assert json.loads(after.split("config=", 1)[1])[key] == {"flipped": key}
        assert key not in json.loads(before.split("config=", 1)[1]) or json.loads(
            before.split("config=", 1)[1]
        )[key] != {"flipped": key}


@pytest.mark.unit
def test_r16_run_signature_flips_with_asset_type():
    """``asset_type`` is part of the signature: same config, other pipeline."""
    graph = _signature_shell()
    signatures = {
        a: graph._run_signature(a)
        for a in ("stock", "crypto", "crypto_spot", "crypto_perp")
    }
    assert len(set(signatures.values())) == 4
    assert "asset=crypto_perp" in signatures["crypto_perp"]


def _record(
    source: str,
    klass: str = "stock_perp",
    underlying: str | None = None,
    onboard: str | None = None,
    status: str | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        classification_source=source,
        instrument_class=klass,
        underlying_symbol=underlying,
        onboard_date=onboard,
        status=status,
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("registry_on", "warm", "source", "record_class", "expected"),
    [
        # Flag OFF: byte-identical legacy predicate — the registry is never read.
        (False, "MU", SOURCE_EXCHANGEINFO, "stock_perp", "stock_perp"),
        (False, None, SOURCE_EXCHANGEINFO, "unknown_perp", "pure_crypto_perp"),
        # Flag ON: persisted exchangeInfo evidence wins in BOTH directions.
        (True, "MU", SOURCE_EXCHANGEINFO, "stock_perp", "stock_perp"),
        (True, None, SOURCE_EXCHANGEINFO, "stock_perp", "stock_perp"),
        (True, "MU", SOURCE_EXCHANGEINFO, "unknown_perp", "unknown_perp"),
        (True, "MU", SOURCE_EXCHANGEINFO, "pure_crypto_perp", "pure_crypto_perp"),
        # Flag ON, registry empty: a positive warm answer stands, pure-by-default
        # is honestly downgraded to unknown.
        (True, "MU", SOURCE_REGISTRY_EMPTY, "stock_perp", "stock_perp"),
        (True, None, SOURCE_REGISTRY_EMPTY, "stock_perp", UNKNOWN_PERP),
        # Other source tiers never override the warm answer at this stage.
        (True, "MU", "static_seed", "pure_crypto_perp", "stock_perp"),
        (True, None, "static_seed", "stock_perp", "pure_crypto_perp"),
    ],
)
def test_r16_routing_instrument_class_registry_precedence(
    monkeypatch, registry_on, warm, source, record_class, expected
):
    """``instrument_class`` precedence: warm/seed answer (flag off) -> persisted
    exchangeInfo record (flag on) -> positive-warm fallback / unknown downgrade
    (registry empty); lower evidence tiers never override the warm answer."""
    consulted: list[str] = []
    monkeypatch.setattr(routing, "stock_perp_underlying", lambda ticker: warm)
    monkeypatch.setattr(
        routing,
        "classify_perp",
        lambda ticker: (consulted.append(ticker), _record(source, record_class))[1],
    )
    set_config({"instrument_registry": registry_on})
    assert instrument_class("crypto_perp", "MUUSDT") == expected
    assert consulted == (["MUUSDT"] if registry_on else [])


@pytest.mark.unit
def test_r16_routing_describe_crypto_spot_descriptor():
    """The crypto_spot descriptor: spot contract, continuous session, the
    upper-cased ticker as its only vendor symbol, no underlying, and no
    as-of listing verdict (spot has no onboard-date evidence)."""
    descriptor = describe_instrument("crypto_spot", "btc-usd", as_of=_DATE)
    assert descriptor.instrument_class == "crypto_spot"
    assert descriptor.contract_kind == "spot"
    assert descriptor.session_calendar == SESSION_CONTINUOUS
    assert descriptor.vendor_symbols == {"binance": "BTC-USD"}
    assert descriptor.underlying_equity is None
    assert descriptor.fundamentals_symbol == "btc-usd"
    assert descriptor.listed_asof is None
    assert descriptor.as_disclosure().startswith(f"listing as of {_DATE}: unknown")


@pytest.mark.unit
@pytest.mark.parametrize(
    ("listing", "as_of", "onboard", "status", "listed"),
    [
        # Fresh process, warm never ran: the registry row alone resolves the
        # underlying AND carries the onboard/status the listing cache lacks.
        ({}, "2025-01-05", "2025-03-10", "TRADING", False),
        # Listing cache entry wins over the registry for onboard/status.
        (
            {"MU": {"onboard_date": "2024-12-01", "status": "TRADING"}},
            "2025-06-01",
            "2024-12-01",
            "TRADING",
            True,
        ),
    ],
)
def test_r16_routing_describe_perp_resolves_underlying_from_registry(
    monkeypatch, listing, as_of, onboard, status, listed
):
    """After a restart (no in-memory warm) a persisted registry row still
    yields the underlying equity for vendor remaps; the listing cache's
    onboard/status outranks the registry's, the registry's fills the gaps."""
    monkeypatch.setattr(routing, "stock_perp_underlying", lambda ticker: None)
    monkeypatch.setattr(
        routing,
        "classify_perp",
        lambda ticker: _record(SOURCE_EXCHANGEINFO, "stock_perp", "MU", "2025-03-10", "TRADING"),
    )
    monkeypatch.setattr(routing, "equity_perp_listing_info", lambda: listing)
    set_config({"instrument_registry": True})

    descriptor = describe_instrument("crypto_perp", "muusdt", as_of=as_of)
    assert descriptor.instrument_class == "stock_perp"
    assert descriptor.underlying_equity == "MU"
    assert descriptor.fundamentals_symbol == "MU"
    assert descriptor.vendor_symbols == {"binance": "MUUSDT", "yfinance": "MU", "sec_edgar": "MU"}
    assert descriptor.session_calendar == SESSION_BINANCE_TRADFI
    assert descriptor.contract_kind == "perpetual"
    assert descriptor.onboard_date == onboard
    assert descriptor.listing_status == status
    assert descriptor.listed_asof is listed
    disclosure = descriptor.as_disclosure()
    assert disclosure.startswith(f"listing as of {as_of}: ")
    assert ("NOT yet listed" in disclosure) is (not listed)
    assert f", onboard {onboard}" in disclosure
    assert descriptor.as_of == as_of


@pytest.mark.unit
@pytest.mark.parametrize(
    ("asset_type", "ticker", "expected"),
    [
        ("stock", "SPY", True),
        ("stock", " spy ", True),
        ("stock", "AAPL", False),
        ("crypto_perp", "SPYUSDT", True),
        ("crypto_perp", "SPYUSDC", True),
        ("crypto_perp", "BTCUSDT", False),
        ("crypto_spot", "SPY", True),
        ("crypto_spot", "BTC-USD", False),
        ("crypto_perp", "USDT", False),  # bare quote asset: nothing to strip
        ("crypto_perp", "USDC", False),
        (None, "QQQ", True),  # None asset_type reads as plain equity
        (None, "SPYUSDT", False),  # ...and so keeps the suffix (no strip)
    ],
)
def test_r16_routing_etf_predicate_strips_perp_suffix_for_crypto_family(
    asset_type, ticker, expected
):
    """ETF detection strips a USDT/USDC perp suffix ONLY for crypto-family
    runs, so the perp form (SPYUSDT) and the plain form (SPY) both match the
    high-confidence ETF set, while a non-crypto run keeps the suffix."""
    assert is_exchange_traded_fund(asset_type, ticker) is expected


# ---------------------------------------------------------------------------
# graph: analyst execution plan + wall-time tracker edges
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_r16_analyst_plan_rejects_empty_selection():
    """An empty analyst selection is rejected loudly — never a plan with no
    entry node (``get_initial_analyst_node`` would raise IndexError)."""
    with pytest.raises(ValueError, match="at least one analyst"):
        build_analyst_execution_plan([])


@pytest.mark.unit
@pytest.mark.parametrize("method", ["mark_started", "mark_completed"])
def test_r16_analyst_tracker_rejects_unknown_key(method):
    """Tracker bookkeeping is keyed by the known analyst keys only."""
    plan = build_analyst_execution_plan(["market"])
    tracker = AnalystWallTimeTracker(plan)
    with pytest.raises(ValueError, match="unknown analyst key"):
        getattr(tracker, method)("astrologer", 0.0)


@pytest.mark.unit
def test_r16_analyst_tracker_completion_before_start_is_ignored():
    """A completion with no start on record records NO wall time (not 0.0):
    an unmeasured analyst must not be reported as instantaneous."""
    tracker = AnalystWallTimeTracker(build_analyst_execution_plan(["market"]))
    tracker.mark_completed("market", completed_at=10.0)
    assert tracker.get_wall_times() == {}

    tracker.mark_started("market", started_at=5.0)
    tracker.mark_completed("market", completed_at=8.0)
    assert tracker.get_wall_times() == {"market": 3.0}
    assert tracker.format_summary() == "Analyst wall time: Market 3.00s"


@pytest.mark.unit
def test_r16_analyst_tracker_first_completion_wins_and_never_negative():
    """First completion wins (later ones can't rewrite a measured duration);
    the first start sticks; a completion earlier than its start clamps to 0."""
    tracker = AnalystWallTimeTracker(build_analyst_execution_plan(["market", "news"]))
    tracker.mark_started("market", started_at=0.0)
    tracker.mark_completed("market", completed_at=5.0)
    tracker.mark_completed("market", completed_at=100.0)  # ignored
    tracker.mark_started("market", started_at=50.0)  # ignored (already started)
    assert tracker.get_wall_times() == {"market": 5.0}

    tracker.mark_started("news", started_at=10.0)
    tracker.mark_completed("news", completed_at=4.0)  # clock rewound
    assert tracker.get_wall_times() == {"market": 5.0, "news": 0.0}
