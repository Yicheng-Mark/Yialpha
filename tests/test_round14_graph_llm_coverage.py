"""Round-14 coverage backfill: graph modules + LLM client base/openai paths.

Targets the branches the suite reaches last, asserting behavior (not just
execution):

* ``Propagator.get_graph_args`` — the whole method was previously unexercised:
  stream_mode / recursion_limit assembly and the falsy-callbacks skip.
* ``memory_resolution`` fail-closed date gates — malformed dates on pending
  entries, the stale-pending WARNING gate, and the invalid ``as_of_date``
  early abort (no price fetch, no durable write).
* ``perf_telemetry`` defensive branches — empty node names, the negative
  duration clamp, ``reset()``, empty-batch / cached-replay / non-AIMessage
  generation skips, the reasoning-only ``llm_output`` fallback, and the
  explicit no-op callback hooks.
* ``base_client`` — ``normalize_content`` block joining, the
  ``make_normalized_chat_class`` invoke mixin, and the abstract stub bodies.
* ``openai_client`` — ``invoke`` content normalization,
  ``with_structured_output`` method="none" rejection, native-base-url host
  parsing (NOTE: the classes in tests/test_openai_responses_base_url.py are
  NOT collected — they lack the ``Test*`` prefix — so those contracts are
  re-pinned here in collected form), the missing-API-key error hints
  (generic / pool-suggestion / pool-set-but-unparseable), and the
  unknown-provider explicit base_url path.

All hermetic: no network, no real LLM. SDK chat classes are monkeypatched or
fed fakes; env vars are pinned per test.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from yialpha.graph.memory_resolution import (
    _entry_precedes_cutoff,
    _warn_if_stale_pending,
    resolve_pending_entries,
)
from yialpha.graph.perf_telemetry import (
    UNATTRIBUTED,
    NodePerfTokenCallback,
    NodePerfTracker,
)
from yialpha.graph.propagation import Propagator
from yialpha.llm_clients.base_client import (
    BaseLLMClient,
    make_normalized_chat_class,
    normalize_content,
)
from yialpha.llm_clients.capabilities import ModelCapabilities
from yialpha.llm_clients.openai_client import (
    NormalizedChatOpenAI,
    OpenAIClient,
    _is_native_openai_base_url,
)

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _isolated_llm_env(monkeypatch):
    """Pin every key/pool/timeout env var the openai_client reads.

    conftest's autouse ``_dummy_api_keys`` sets the single-key vars to a
    placeholder; delete them so the missing-key error paths are reachable,
    and keep the pool machinery stateless across tests in this module.
    Tests that need a key set it explicitly.
    """
    from yialpha.llm_clients import key_pool

    for var in (
        "OPENAI_API_KEY",
        "ZHIPU_API_KEY", "ZHIPU_CN_API_KEY",
        "ZHIPU_API_KEYS", "ZHIPU_CN_API_KEYS",
        "ZHIPU_BASE_URL", "ZHIPU_CN_BASE_URL",
        "YIALPHA_LLM_TIMEOUT_S", "YIALPHA_LLM_KEY_COOLDOWN_S",
        "YIALPHA_LLM_POOL_STREAMING", "YIALPHA_LLM_POOL_STREAM_USAGE",
    ):
        monkeypatch.delenv(var, raising=False)
    key_pool.reset_for_test()
    yield
    key_pool.reset_for_test()


# --------------------------------------------------------------------------- #
# 1. Propagator.get_graph_args (previously the module's only uncovered lines)
# --------------------------------------------------------------------------- #
class TestGetGraphArgs:
    def test_defaults_without_callbacks(self):
        args = Propagator().get_graph_args()
        assert args == {
            "stream_mode": "values",
            "config": {"recursion_limit": 100},
        }
        # Falsy-callbacks skip: an absent (not passed) callbacks list must
        # not surface as a None entry langgraph would choke on.
        assert "callbacks" not in args["config"]

    def test_custom_recursion_limit_forwarded(self):
        args = Propagator(max_recur_limit=42).get_graph_args()
        assert args["config"]["recursion_limit"] == 42
        assert args["stream_mode"] == "values"

    def test_callbacks_forwarded_into_config(self):
        handler = object()
        args = Propagator().get_graph_args(callbacks=[handler])
        assert args["config"]["callbacks"] == [handler]
        assert args["stream_mode"] == "values"

    def test_empty_callbacks_list_is_omitted(self):
        """``if callbacks:`` is truthiness, not identity: an empty list adds
        no key — same shape as not passing callbacks at all."""
        args = Propagator().get_graph_args(callbacks=[])
        assert "callbacks" not in args["config"]

    def test_recursion_limit_matches_constructor_default(self):
        # The production default ceiling is 100 superteps; pin it so an
        # accidental tightening cannot slip through as a "test-only" change.
        assert Propagator().max_recur_limit == 100


# --------------------------------------------------------------------------- #
# 2. memory_resolution fail-closed date gates
# --------------------------------------------------------------------------- #
#: Midnight cutoff: the gate parses dates at day precision (``[:10]``), so a
#: midnight cutoff makes "same day" exactly NOT precede (strict ``<``).
_CUTOFF = datetime(2026, 6, 30)


class TestEntryPrecedesCutoff:
    def test_missing_date_key_fails_closed(self):
        # A hand-corrupted row with no date must not crash the sweep nor be
        # resolved on a bogus parse: it is simply not eligible yet.
        assert _entry_precedes_cutoff({"ticker": "AAPL"}, _CUTOFF) is False

    def test_non_string_date_fails_closed(self):
        assert _entry_precedes_cutoff({"date": None}, _CUTOFF) is False
        assert _entry_precedes_cutoff({"date": 12345}, _CUTOFF) is False

    def test_garbage_date_string_fails_closed(self):
        assert _entry_precedes_cutoff({"date": "not-a-date"}, _CUTOFF) is False
        # A datetime-ish prefix that is not a plain ISO calendar date.
        assert _entry_precedes_cutoff({"date": "15/06/2026"}, _CUTOFF) is False

    def test_valid_dates_gate_both_directions(self):
        assert _entry_precedes_cutoff({"date": "2026-06-15"}, _CUTOFF) is True
        assert _entry_precedes_cutoff({"date": "2026-07-01"}, _CUTOFF) is False
        # Same-day is NOT precede (strict <): it resolves on a later sweep.
        assert _entry_precedes_cutoff({"date": "2026-06-30"}, _CUTOFF) is False


class TestWarnIfStalePending:
    def test_invalid_trade_date_is_silent(self, caplog):
        # An unparseable date cannot compute an age; the helper must return
        # without logging rather than raise (it runs inside the fail-soft
        # price leg of the sweep).
        with caplog.at_level("WARNING", logger="yialpha.graph.memory_resolution"):
            assert _warn_if_stale_pending("AAPL", "garbage-date", _CUTOFF) is None
            assert _warn_if_stale_pending("AAPL", None, _CUTOFF) is None
        assert caplog.records == []

    def test_fresh_pending_entry_does_not_warn(self, caplog):
        now = datetime(2026, 11, 23, 12, 0, 0)
        with caplog.at_level("WARNING", logger="yialpha.graph.memory_resolution"):
            _warn_if_stale_pending("AAPL", "2026-11-20", now)
        assert caplog.records == []

    def test_old_pending_entry_warns_with_age(self, caplog):
        now = datetime(2026, 11, 23, 12, 0, 0)
        trade_date = (now - timedelta(days=22)).strftime("%Y-%m-%d")
        with caplog.at_level("WARNING", logger="yialpha.graph.memory_resolution"):
            _warn_if_stale_pending("NVDA", trade_date, now)
        assert len(caplog.records) == 1
        msg = caplog.records[0].getMessage()
        assert "NVDA" in msg
        assert trade_date in msg
        assert "22 days old" in msg


class _FakeMemoryLog:
    """Minimal stand-in for TradingMemoryLog (only the two methods the
    resolver touches); records durable writes for assertion."""

    def __init__(self, entries):
        self._entries = entries
        self.updates = None

    def get_pending_entries(self):
        return list(self._entries)

    def batch_update_with_outcomes(self, updates):
        self.updates = updates


class _FakeReflector:
    def __init__(self):
        self.calls = 0

    def reflect_on_final_decision(self, **kwargs):
        self.calls += 1
        return "lesson"


class TestResolvePendingEntriesStaleSkip:
    def test_unavailable_price_on_old_entry_warns_and_leaves_pending(
        self, monkeypatch, caplog,
    ):
        """Price unavailable + entry older than STALE_PENDING_DAYS: the
        entry must be skipped (left pending, no reflection LLM call, no
        durable write) AND logged at WARNING so a permanently-stuck ticker
        (delisted / permanent data gap) does not silently accumulate."""
        import yialpha.accuracy as accuracy_mod

        monkeypatch.setattr(
            accuracy_mod, "fetch_returns_yf", lambda *a, **k: (None, None, None)
        )
        log = _FakeMemoryLog(
            [{"ticker": "DELIST", "date": "2026-06-01", "decision": "Buy"}]
        )
        reflector = _FakeReflector()

        with caplog.at_level("WARNING", logger="yialpha.graph.memory_resolution"):
            resolved = resolve_pending_entries(
                log, reflector, "DELIST", as_of_date="2026-06-23"
            )

        assert resolved == 0
        assert log.updates is None  # nothing durably resolved
        assert reflector.calls == 0  # the LLM reflection leg never ran
        stale_warnings = [
            r.getMessage()
            for r in caplog.records
            if "days old" in r.getMessage()
        ]
        assert len(stale_warnings) == 1
        assert "DELIST" in stale_warnings[0]
        assert "2026-06-01" in stale_warnings[0]
        assert "22 days old" in stale_warnings[0]


class TestResolvePendingEntriesInvalidAsOfDate:
    def test_invalid_as_of_date_aborts_before_any_fetch(self, monkeypatch, caplog):
        import yialpha.accuracy as accuracy_mod

        fetches = []

        def _no_fetch(*args, **kwargs):
            fetches.append((args, kwargs))
            return (0.05, 0.02, 5)

        monkeypatch.setattr(accuracy_mod, "fetch_returns_yf", _no_fetch)

        log = _FakeMemoryLog(
            [{"ticker": "AAPL", "date": "2026-06-01", "decision": "Buy"}]
        )

        with caplog.at_level("WARNING", logger="yialpha.graph.memory_resolution"):
            resolved = resolve_pending_entries(
                log, reflector=object(), ticker="AAPL", as_of_date="31/12/2026"
            )

        assert resolved == 0
        assert fetches == []  # the abort happens before any price leg runs
        assert log.updates is None  # and before any durable write
        assert any(
            "Invalid memory as-of date" in r.getMessage() for r in caplog.records
        )


# --------------------------------------------------------------------------- #
# 3. perf_telemetry defensive branches
# --------------------------------------------------------------------------- #
class TestTrackerDefensiveBranches:
    def test_record_ignores_empty_node_name(self):
        tracker = NodePerfTracker()
        tracker.record("", 1.0)
        tracker.record(None, 2.0)
        assert tracker.serialize()["nodes"] == {}

    def test_record_clamps_negative_duration_to_zero(self):
        tracker = NodePerfTracker()
        tracker.record("clock_skew", -7.5)
        node = tracker.serialize()["nodes"]["clock_skew"]
        # Clamped, not dropped: the call still counts.
        assert node["wall_seconds"] == 0.0
        assert node["calls"] == 1

    def test_record_tokens_empty_name_goes_to_unattributed(self):
        tracker = NodePerfTracker()
        tracker.record_tokens("", input_tokens=7, output_tokens=3)
        node = tracker.serialize()["nodes"][UNATTRIBUTED]
        assert node["tokens_in"] == 7
        assert node["tokens_out"] == 3

    def test_reset_clears_all_nodes(self):
        tracker = NodePerfTracker()
        tracker.record("A", 1.0)
        tracker.record_tokens("B", input_tokens=5)
        assert tracker.serialize()["nodes"]
        tracker.reset()
        # Post-reset snapshot is byte-identical to a fresh tracker's.
        assert tracker.serialize() == {
            "nodes": {},
            "totals": {
                "wall_seconds": 0.0,
                "tokens_in": 0,
                "tokens_out": 0,
                "tokens_reasoning": 0,
            },
        }


def _ai_generation(text: str, usage: dict | None) -> ChatGeneration:
    message = AIMessage(content=text)
    if usage is not None:
        message.usage_metadata = usage
    return ChatGeneration(message=message)


class TestTokenCallbackSkips:
    def test_empty_generation_batches_are_skipped(self):
        tracker = NodePerfTracker()
        cb = NodePerfTokenCallback(tracker)
        result = LLMResult(
            generations=[
                [],  # empty batch: skipped without indexing errors
                [_ai_generation("ok", {"input_tokens": 11, "output_tokens": 4})],
                [],  # trailing empty batch too
            ]
        )
        tracker.set_active_node("Market Analyst")
        try:
            cb.on_llm_end(result)
        finally:
            tracker.set_active_node(None)
        node = tracker.serialize()["nodes"]["Market Analyst"]
        assert node["tokens_in"] == 11
        assert node["tokens_out"] == 4

    def test_cached_replay_generations_are_skipped(self):
        """Round-6 contract: a generation stamped ``cached_replay`` by the
        LLM response cache contributes ZERO tokens — its usage belongs to
        the original call the cache replayed, not to this run's node."""
        tracker = NodePerfTracker()
        cb = NodePerfTokenCallback(tracker)
        message = AIMessage(content="cached")
        message.usage_metadata = {"input_tokens": 100, "output_tokens": 50}
        generation = ChatGeneration(
            message=message, generation_info={"cached_replay": True}
        )
        tracker.set_active_node("Trader")
        try:
            cb.on_llm_end(LLMResult(generations=[[generation]]))
        finally:
            tracker.set_active_node(None)
        assert "Trader" not in tracker.serialize()["nodes"]

    def test_non_ai_message_generations_are_skipped(self):
        tracker = NodePerfTracker()
        cb = NodePerfTokenCallback(tracker)
        result = LLMResult(
            generations=[[ChatGeneration(message=HumanMessage(content="hi"))]]
        )
        tracker.set_active_node("Trader")
        try:
            cb.on_llm_end(result)
        finally:
            tracker.set_active_node(None)
        assert "Trader" not in tracker.serialize()["nodes"]

    def test_llm_output_fills_all_fields_when_generations_silent(self):
        """Providers that put usage ONLY on ``response.llm_output`` (no
        ``usage_metadata`` on the AIMessage) still get full attribution —
        input, output AND the nested reasoning detail."""
        tracker = NodePerfTracker()
        cb = NodePerfTokenCallback(tracker)
        result = LLMResult(
            generations=[[_ai_generation("ok", None)]],
            llm_output={
                "token_usage": {
                    "input_tokens": 50,
                    "output_tokens": 25,
                    "output_token_details": {"reasoning": 9},
                }
            },
        )
        tracker.set_active_node("Research Manager")
        try:
            cb.on_llm_end(result)
        finally:
            tracker.set_active_node(None)
        node = tracker.serialize()["nodes"]["Research Manager"]
        assert node["tokens_in"] == 50
        assert node["tokens_out"] == 25
        assert node["tokens_reasoning"] == 9


class TestTokenCallbackExplicitNoOps:
    def test_lifecycle_hooks_return_none_and_record_nothing(self):
        """The explicit no-op hooks must stay side-effect free: a
        misbehaving handler there must never corrupt the tracker."""
        tracker = NodePerfTracker()
        cb = NodePerfTokenCallback(tracker)
        boom = RuntimeError("boom")

        assert cb.on_llm_start({}, ["prompt"]) is None
        assert cb.on_chat_model_start({}, [["m"]]) is None
        assert cb.on_chain_start({}, {"input": 1}) is None
        assert cb.on_chain_end({"output": 1}) is None
        assert cb.on_chain_error(boom) is None
        assert cb.on_tool_start({}, "tool input") is None
        assert cb.on_tool_end("tool output") is None
        assert cb.on_tool_error(boom) is None
        assert cb.on_llm_error(boom) is None

        assert tracker.serialize() == {
            "nodes": {},
            "totals": {
                "wall_seconds": 0.0,
                "tokens_in": 0,
                "tokens_out": 0,
                "tokens_reasoning": 0,
            },
        }


# --------------------------------------------------------------------------- #
# 4. base_client: normalize_content + invoke mixin + abstract stubs
# --------------------------------------------------------------------------- #
class _FakeResponse:
    def __init__(self, content):
        self.content = content


class TestNormalizeContent:
    def test_joins_text_blocks_and_drops_non_text(self):
        resp = _FakeResponse(
            [
                {"type": "reasoning", "reasoning": "internal thinking"},
                {"type": "text", "text": "Part A"},
                "bare string block",
                {"type": "text", "text": "Part B"},
                {"type": "image_url", "image_url": "http://x/img.png"},
                12345,  # unknown block shape: not dict-with-text, not str
            ]
        )
        out = normalize_content(resp)
        assert out is resp  # normalized in place, same object returned
        assert out.content == "Part A\nbare string block\nPart B"

    def test_string_content_passes_through_untouched(self):
        resp = _FakeResponse("already a string")
        assert normalize_content(resp).content == "already a string"

    def test_empty_block_list_normalizes_to_empty_string(self):
        resp = _FakeResponse([])
        assert normalize_content(resp).content == ""

    def test_all_non_text_blocks_normalize_to_empty_string(self):
        resp = _FakeResponse([{"type": "reasoning", "reasoning": "x"}, {"z": 1}])
        assert normalize_content(resp).content == ""

    def test_non_list_content_is_left_alone(self):
        resp = _FakeResponse(None)
        assert normalize_content(resp).content is None


class _StubChat:
    """Plain-callable stand-in for a provider chat class."""

    def __init__(self, reply):
        self._reply = reply
        self.calls = []

    def invoke(self, input, config=None, **kwargs):
        self.calls.append((input, config, kwargs))
        return self._reply


class TestMakeNormalizedChatClass:
    def test_mixin_invoke_forwards_and_normalizes(self):
        cls = make_normalized_chat_class(_StubChat, "Round14StubChat", "stub doc")
        assert cls.__name__ == "Round14StubChat"
        assert cls.__doc__ == "stub doc"
        assert issubclass(cls, _StubChat)

        reply = _FakeResponse([{"type": "text", "text": "hello"}])
        chat = cls(reply)
        out = chat.invoke("prompt", {"tags": ["t"]}, extra=1)

        assert out is reply
        assert out.content == "hello"
        # The underlying chat saw the exact invoke arguments, positionally
        # forwarded through the mixin (input, config) plus kwargs.
        assert chat.calls == [("prompt", {"tags": ["t"]}, {"extra": 1})]


class TestBaseLLMClientAbstractStubs:
    def test_abstract_stub_bodies_delegate_to_none(self):
        """A concrete subclass MAY delegate to the abstract stubs (e.g. a
        ``get_llm`` that conditionally builds); the stub bodies are no-ops
        returning None rather than raising."""

        class _DelegatingClient(BaseLLMClient):
            def get_llm(self):
                return super().get_llm()

            def validate_model(self):
                return super().validate_model()

        client = _DelegatingClient("some-model")
        assert client.get_llm() is None
        assert client.validate_model() is None


# --------------------------------------------------------------------------- #
# 5. openai_client
# --------------------------------------------------------------------------- #
class TestNormalizedChatOpenAIInvoke:
    def test_invoke_joins_typed_content_blocks(self, monkeypatch):
        from langchain_openai import ChatOpenAI

        reply = AIMessage(
            content=[
                {"type": "reasoning", "reasoning": "thinking hard"},
                {"type": "text", "text": "Answer A"},
                {"type": "text", "text": "Answer B"},
            ]
        )
        seen = {}

        def _fake_super_invoke(self, input, config=None, **kwargs):
            seen["input"] = input
            seen["config"] = config
            return reply

        monkeypatch.setattr(ChatOpenAI, "invoke", _fake_super_invoke)

        llm = NormalizedChatOpenAI(model="gpt-5.5", api_key="sk-test")
        prompt = [HumanMessage(content="hi")]
        out = llm.invoke(prompt, {"tags": ["t"]})

        assert out is reply
        # Downstream agents get the joined plain string, not typed blocks.
        assert out.content == "Answer A\nAnswer B"
        assert seen["input"] is prompt
        assert seen["config"] == {"tags": ["t"]}


class TestWithStructuredOutputNoneCapability:
    def test_none_method_raises_not_implemented_with_fallback_hint(
        self, monkeypatch,
    ):
        none_caps = ModelCapabilities(
            supports_tool_choice=False,
            supports_json_mode=False,
            supports_json_schema=False,
            preferred_structured_method="none",
        )
        monkeypatch.setattr(
            "yialpha.llm_clients.openai_client.get_capabilities",
            lambda model_name: none_caps,
        )
        llm = NormalizedChatOpenAI(model="mystery-foss-model", api_key="sk-test")
        with pytest.raises(NotImplementedError) as excinfo:
            llm.with_structured_output(object())
        msg = str(excinfo.value)
        assert "mystery-foss-model" in msg
        assert "no structured-output method available" in msg
        assert "fall back to free-text generation" in msg


class TestNativeBaseUrlDetection:
    @pytest.mark.parametrize(
        "base_url,expected",
        [
            # Unset/empty is native (SDK default endpoint).
            (None, True),
            ("", True),
            # Scheme-less strings get https:// prepended before host parsing.
            ("api.openai.com/v1", True),
            ("eu.api.openai.com/v1", True),
            ("https://api.openai.com/v1", True),
            ("https://eu.api.openai.com/v1", True),
            # Custom endpoints are not native — even ones crafted to look
            # like OpenAI hosts.
            ("http://localhost:1234/v1", False),
            ("my-gateway.example.com", False),
            ("https://proxy.example.com/v1", False),
            ("https://api.openai.com.evil.com/v1", False),
        ],
    )
    def test_host_parsing(self, base_url, expected):
        assert _is_native_openai_base_url(base_url) is expected

    def test_get_llm_responses_api_on_native_off_on_proxy(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        native = OpenAIClient("gpt-5.5", provider="openai").get_llm()
        assert getattr(native, "use_responses_api", False) is True

        proxied = OpenAIClient(
            "gpt-5.5", base_url="https://proxy.example.com/v1", provider="openai"
        ).get_llm()
        # A custom base_url only speaks Chat Completions (#1024). (An unset
        # flag surfaces as the class-default None, never True.)
        assert getattr(proxied, "use_responses_api", False) in (False, None)
        assert str(proxied.openai_api_base) == "https://proxy.example.com/v1"


class TestMissingApiKeyErrorHints:
    """The 407-427 block: the missing-key ValueError with key-pool-aware
    hints. Three hint variants, each pinned by message content."""

    def test_generic_hint_for_provider_without_pool(self):
        client = OpenAIClient("gpt-5.5", provider="openai")
        with pytest.raises(ValueError) as excinfo:
            client.get_llm()
        msg = str(excinfo.value)
        assert "API key for provider 'openai' is not set" in msg
        assert "OPENAI_API_KEY environment variable" in msg
        assert "add OPENAI_API_KEY=your_key to your .env file" in msg
        # openai has no pool registry entry: no pool mention in the hint.
        assert "pool" not in msg

    def test_pool_suggested_when_pool_var_unset(self):
        client = OpenAIClient("glm-5.3", provider="glm-cn")
        with pytest.raises(ValueError) as excinfo:
            client.get_llm()
        msg = str(excinfo.value)
        assert "API key for provider 'glm-cn' is not set" in msg
        assert "or a ZHIPU_CN_API_KEYS pool to your .env file" in msg

    def test_pool_set_but_unparseable_names_the_pool_var(self, monkeypatch):
        # Set but strips to zero parseable keys: has_pool() stays False (no
        # rotating transport engages), yet the hint must point the operator
        # at the broken pool variable instead of the single-key var they
        # never needed (round-6).
        monkeypatch.setenv("ZHIPU_CN_API_KEYS", " , ,")
        client = OpenAIClient("glm-5.3", provider="glm-cn")
        with pytest.raises(ValueError) as excinfo:
            client.get_llm()
        msg = str(excinfo.value)
        assert (
            "the ZHIPU_CN_API_KEYS pool variable is set but contains no "
            "parseable key" in msg
        )


class TestUnknownProviderExplicitBaseUrl:
    def test_unknown_provider_honors_explicit_base_url(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        llm = OpenAIClient(
            "some-model", base_url="http://localhost:9999/v1", provider="custom-relay"
        ).get_llm()
        # No registry spec -> plain NormalizedChatOpenAI against the
        # caller-supplied endpoint; the Responses API never engages there.
        assert type(llm).__name__ == "NormalizedChatOpenAI"
        assert str(llm.openai_api_base) == "http://localhost:9999/v1"
        assert getattr(llm, "use_responses_api", False) in (False, None)

    def test_unknown_provider_without_base_url_uses_sdk_default(
        self, monkeypatch,
    ):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        llm = OpenAIClient("some-model", provider="custom-relay").get_llm()
        assert type(llm).__name__ == "NormalizedChatOpenAI"
