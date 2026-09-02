"""Per-run prefetch scope (yialpha.dataflows.run_scope).

Pins the ContextVar contract the bundles/news prefetches rely on:
* ``run_cached`` fetches exactly ONCE per key per scope — the LLM tool loop
  re-enters analyst nodes on every tool call and the prefetch must not
  re-run (nor re-charge live budgets) per round;
* the cache is run-scoped: reset drops it, and a fresh scope never sees the
  previous run's values (no long-lived global cache serving stale live
  snapshots across runs);
* a scope-less caller degrades to a direct call (unit-test friendly, no
  silent cross-test leak);
* the scope dict is the SAME object across ``submit_with_context`` workers
  and copied node contexts — the same inheritance the quality ledger uses.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pytest

from yialpha.dataflows import run_scope
from yialpha.dataflows.config import submit_with_context


@pytest.fixture(autouse=True)
def _clean_scope():
    run_scope.reset_run_scope()
    yield
    run_scope.reset_run_scope()


@pytest.mark.unit
def test_run_cached_fetches_once_per_scope():
    run_scope.ensure_run_scope()
    calls = []

    def fetch():
        calls.append(1)
        return {"value": 42}

    out1 = run_scope.run_cached(("bundle", "MU", "2026-09-01"), fetch)
    out2 = run_scope.run_cached(("bundle", "MU", "2026-09-01"), fetch)
    assert out1 == out2 == {"value": 42}
    assert len(calls) == 1  # second entry (tool-loop re-entry) is free


@pytest.mark.unit
def test_distinct_keys_fetch_independently():
    run_scope.ensure_run_scope()
    calls = []

    def fetch(name):
        calls.append(name)
        return name

    assert run_scope.run_cached(("a",), lambda: fetch("a")) == "a"
    assert run_scope.run_cached(("b",), lambda: fetch("b")) == "b"
    assert calls == ["a", "b"]


@pytest.mark.unit
def test_reset_drops_cache_across_runs():
    run_scope.ensure_run_scope()
    calls = []
    run_scope.run_cached(("k",), lambda: calls.append(1) or "first")
    run_scope.reset_run_scope()
    run_scope.ensure_run_scope()  # fresh run in the same process
    out = run_scope.run_cached(("k",), lambda: calls.append(1) or "second")
    assert out == "second"
    assert len(calls) == 2  # the new run refetched — no cross-run leak


@pytest.mark.unit
def test_scopeless_caller_degrades_to_direct_call():
    calls = []
    out = run_scope.run_cached(("k",), lambda: calls.append(1) or "direct")
    assert out == "direct"
    assert len(calls) == 1


@pytest.mark.unit
def test_exceptions_are_not_cached():
    run_scope.ensure_run_scope()
    attempts = []

    def flaky():
        attempts.append(1)
        if len(attempts) == 1:
            raise RuntimeError("transient")
        return "ok"

    with pytest.raises(RuntimeError):
        run_scope.run_cached(("k",), flaky)
    assert run_scope.run_cached(("k",), flaky) == "ok"
    assert len(attempts) == 2  # failure retried on the next node re-entry


@pytest.mark.unit
def test_scope_object_shared_with_submit_with_context_workers():
    # The worker contexts must see the SAME cache dict the node context
    # mutates — otherwise a bundle fetched in one worker would refetch in
    # the next (the exact bug the quality ledger hit before
    # submit_with_context).
    run_scope.ensure_run_scope()
    calls = []

    def fetch():
        calls.append(1)
        return "payload"

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = submit_with_context(
            pool, run_scope.run_cached, ("worker_key",), fetch,
        ).result()
        second = submit_with_context(
            pool, run_scope.run_cached, ("worker_key",), fetch,
        ).result()
    assert first == second == "payload"
    assert len(calls) == 1


@pytest.mark.unit
def test_fundamentals_analyst_single_fetch_across_tool_loop_reentries(monkeypatch):
    # Node-level pin: invoking the analyst node twice in one run (what the
    # tool loop does between tool calls) fetches the bundle ONCE, and the
    # second entry still renders the block from the cached payload.
    from langchain_core.messages import AIMessage
    from langchain_core.runnables import Runnable

    import yialpha.agents.analysts.fundamentals_analyst as fa
    import yialpha.dataflows.fundamentals_bundle as fb
    from yialpha.dataflows.config import get_config, set_config

    class _CaptureLLM(Runnable):
        def __init__(self):
            super().__init__()
            self.prompts = []

        def invoke(self, inp, config=None, **kwargs):  # noqa: ARG002
            self.prompts.append(inp)
            return AIMessage(content="MOCK REPORT", tool_calls=[])

        def bind_tools(self, tools, **kwargs):  # noqa: ARG002
            return self

    calls = []
    orig = get_config()
    try:
        set_config({**orig, "fundamentals_bundle": True})
        monkeypatch.setattr(
            fb, "fetch_fundamentals_bundle",
            lambda at, t, d: calls.append(1)
            or {"symbol": t, "fundamentals_symbol": t, "as_of": d},
        )
        monkeypatch.setattr(
            fb, "render_fundamentals_bundle_block", lambda b: "BUNDLE_BLOCK",
        )
        run_scope.ensure_run_scope()
        llm = _CaptureLLM()
        state = {
            "trade_date": "2026-09-01",
            "company_of_interest": "MU",
            "asset_type": "stock",
            "instrument_context": "CTX",
            "messages": [],
        }
        fa.create_fundamentals_analyst(llm)(state)  # first entry
        fa.create_fundamentals_analyst(llm)(state)  # tool-loop re-entry
    finally:
        set_config(orig)
        run_scope.reset_run_scope()
    assert len(calls) == 1  # exactly one fetch per run
    assert len(llm.prompts) == 2  # both entries rendered the evidence
    assert all("BUNDLE_BLOCK" in str(p) for p in llm.prompts)


@pytest.mark.unit
def test_market_analyst_single_fetch_across_tool_loop_reentries(monkeypatch):
    # Same pin for the perp market bundle.
    from langchain_core.messages import AIMessage
    from langchain_core.runnables import Runnable

    import yialpha.agents.analysts.market_analyst as ma
    import yialpha.dataflows.perp_bundle as pb
    from yialpha.dataflows.config import get_config, set_config

    class _CaptureLLM(Runnable):
        def __init__(self):
            super().__init__()
            self.prompts = []

        def invoke(self, inp, config=None, **kwargs):  # noqa: ARG002
            self.prompts.append(inp)
            return AIMessage(content="MOCK REPORT", tool_calls=[])

        def bind_tools(self, tools, **kwargs):  # noqa: ARG002
            return self

    calls = []
    orig = get_config()
    try:
        set_config({**orig, "perp_market_bundle": True})
        monkeypatch.setattr(
            pb, "fetch_perp_market_bundle",
            lambda t, d: calls.append(1) or {"symbol": t, "as_of": d},
        )
        monkeypatch.setattr(
            pb, "render_perp_bundle_block", lambda b: "MARKET_BLOCK",
        )
        run_scope.ensure_run_scope()
        llm = _CaptureLLM()
        state = {
            "trade_date": "2026-09-01",
            "company_of_interest": "BTCUSDT",
            "asset_type": "crypto_perp",
            "instrument_context": "CTX",
            "messages": [],
        }
        ma.create_market_analyst(llm)(state)
        ma.create_market_analyst(llm)(state)
    finally:
        set_config(orig)
        run_scope.reset_run_scope()
    assert len(calls) == 1
    assert len(llm.prompts) == 2
    assert all("MARKET_BLOCK" in str(p) for p in llm.prompts)
    # Evidence posture parity: the market block rides a human evidence
    # message too, never the system role.
    messages = llm.prompts[0].to_messages()
    assert messages[-1].type == "human"
    assert "<start_of_perp_market_bundle>" in messages[-1].content
    system = next(m for m in messages if m.type == "system")
    assert "MARKET_BLOCK" not in system.content
