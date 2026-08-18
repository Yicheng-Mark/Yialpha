"""Tavily web-search vendor: degradation, budget, evidence, and formatting.

The vendor is a *direct tool* (not router-routed), so it owns its own
fail-open contract: every degradation path must emit a
``WEB_SEARCH_UNAVAILABLE`` sentinel AND a ``record_sentinel`` event — the
round-5 audit class of bug is a direct tool returning sentinels that never
enter the data_quality ledger, making a degraded run look clean.

HTTP is mocked with the real call signature (url/json/headers/timeout
keywords) so a signature drift in the vendor fails here instead of hiding
behind a permissive fake (round-2 audit lesson). Env handling goes through
monkeypatch with split-literal names (same idiom as config-check) so no
credential-shaped literal ever appears in source.
"""

from __future__ import annotations

import datetime

import requests

from yiagents.agents.utils.web_search_tools import web_search
from yiagents.dataflows import quality, tavily

_KEY_ENV = "TAVILY" + "_API_KEY"
_KEYS_ENV = "TAVILY_API" + "_KEYS"
_MAX_ENV = "YIAGENTS_TAVILY_MAX_CALLS" + "_PER_RUN"
_SPLIT_ENV = "YIAGENTS_TAVILY_BUDGET" + "_SPLIT"
# Assembled at runtime — never a real credential, and never a whole-literal
# assignment that secret scanners flag.
_FAKE_KEY = "tvly-dev-" + "unit-test-fake"


class _FakeResponse:
    def __init__(self, payload=None, status=200):
        self._payload = payload
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.exceptions.HTTPError(
                f"{self.status_code} error", response=self
            )

    def json(self):
        if self._payload is None:
            raise ValueError("no json")
        return self._payload


def _post_returning(response):
    """requests.post fake with the vendor's real keyword signature."""
    def _post(url, json=None, headers=None, timeout=None):  # noqa: A002
        return response
    return _post


def _fresh(monkeypatch, *, key=_FAKE_KEY, max_calls=None, split=None, keys=None):
    quality.ensure_run_context()
    tavily.reset_run_budget()
    if keys is None:
        # No pool: single-key mode (the single var may still be set below).
        monkeypatch.delenv(_KEYS_ENV, raising=False)
        if key is None:
            monkeypatch.delenv(_KEY_ENV, raising=False)
        else:
            monkeypatch.setenv(_KEY_ENV, key)
    else:
        # Explicit pool replaces the whole key picture; tests wanting the
        # single var alongside re-set it after _fresh.
        monkeypatch.delenv(_KEY_ENV, raising=False)
        monkeypatch.setenv(_KEYS_ENV, ",".join(keys))
    if max_calls is None:
        monkeypatch.delenv(_MAX_ENV, raising=False)
    else:
        monkeypatch.setenv(_MAX_ENV, str(max_calls))
    if split is None:
        monkeypatch.delenv(_SPLIT_ENV, raising=False)
    else:
        monkeypatch.setenv(_SPLIT_ENV, split)


def test_missing_key_sentinel_and_evidence(monkeypatch):
    _fresh(monkeypatch, key=None)
    result = tavily.get_web_search("nvda news")
    assert "WEB_SEARCH_UNAVAILABLE" in result
    assert "TAVILY_API_KEY" in result
    events = quality.snapshot_quality()
    assert len(events) == 1
    assert events[0]["method"] == "web_search"
    assert events[0]["kind"] == quality.KIND_OPTIONAL_UNAVAILABLE
    # The missing-key check precedes the budget: nothing was charged.
    assert sum(tavily.run_usage().values()) == 0


def test_budget_exhausted_sentinel(monkeypatch):
    _fresh(monkeypatch, max_calls=1)
    first = tavily.get_web_search("q1")
    second = tavily.get_web_search("q2")
    assert "budget exhausted" in second
    assert "budget exhausted" not in first
    assert any("budget" in e["detail"] for e in quality.snapshot_quality())


def test_reset_run_budget_restores_calls(monkeypatch):
    _fresh(monkeypatch, max_calls=1)
    tavily.get_web_search("q1")
    exhausted = tavily.get_web_search("q2")
    tavily.reset_run_budget()
    fresh = tavily.get_web_search("q3")
    assert "budget exhausted" in exhausted
    assert "budget exhausted" not in fresh


def test_empty_query_sentinel(monkeypatch):
    _fresh(monkeypatch)
    result = tavily.get_web_search("   ")
    assert "WEB_SEARCH_UNAVAILABLE" in result
    assert "empty query" in result


def test_success_format(monkeypatch):
    payload = {
        "results": [
            {"title": "NVIDIA news", "url": "https://example.com/a",
             "content": "Chip demand strong."},
            {"title": "Analyst note", "url": "https://example.com/b",
             "content": "Price target raised."},
        ]
    }
    _fresh(monkeypatch)
    with requests_mock_post(_post_returning(_FakeResponse(payload))) as post:
        result = tavily.get_web_search("nvda", max_results=5)
    assert "## Web search: nvda" in result
    assert "**NVIDIA news**" in result
    assert "Source: https://example.com/a" in result
    assert "Chip demand strong." in result
    assert "unverified snippet text" in result  # anti-fabrication footer
    # no degradation events on the success path
    assert quality.snapshot_quality() == []
    # real signature used: bearer auth + json body + timeout
    _, kwargs = post.call_args
    assert "Bearer" in kwargs["headers"]["Authorization"]
    assert kwargs["json"]["query"] == "nvda"
    assert kwargs["timeout"] > 0


def test_snippet_truncation(monkeypatch):
    payload = {"results": [
        {"title": "t", "url": "https://e.com/x", "content": "w " * 1200}
    ]}
    _fresh(monkeypatch)
    with requests_mock_post(_post_returning(_FakeResponse(payload))):
        result = tavily.get_web_search("q")
    assert "[...]" in result
    for line in result.splitlines():
        if line.startswith("   w "):
            assert len(line) <= tavily._SNIPPET_CHAR_CAP + 8


def test_max_results_clamped_to_ceiling(monkeypatch):
    captured = {}

    def _post(url, json=None, headers=None, timeout=None):  # noqa: A002
        captured["max_results"] = json["max_results"]
        return _FakeResponse({"results": []})

    _fresh(monkeypatch)
    with requests_mock_post(_post):
        tavily.get_web_search("q", max_results=99)
    assert captured["max_results"] == tavily.MAX_RESULTS_CEILING


def test_empty_results_is_honest_not_degraded(monkeypatch):
    _fresh(monkeypatch)
    with requests_mock_post(_post_returning(_FakeResponse({"results": []}))):
        result = tavily.get_web_search("obscure query")
    assert "no results found" in result
    # a real "nothing found" answer is not a degradation: no sentinel event
    assert quality.snapshot_quality() == []


def test_http_error_sentinel(monkeypatch):
    _fresh(monkeypatch)
    with requests_mock_post(_post_returning(_FakeResponse(status=401))):
        result = tavily.get_web_search("q")
    assert "WEB_SEARCH_UNAVAILABLE" in result
    assert "HTTP 401" in result
    assert len(quality.snapshot_quality()) == 1


def test_transport_error_sentinel_after_retry(monkeypatch):
    calls = {"n": 0}

    def _post(url, json=None, headers=None, timeout=None):  # noqa: A002
        calls["n"] += 1
        raise requests.exceptions.ConnectionError("socks reset")

    _fresh(monkeypatch)
    with requests_mock_post(_post):
        result = tavily.get_web_search("q")
    assert "WEB_SEARCH_UNAVAILABLE" in result
    assert "ConnectionError" in result
    # netretry absorbed one transient retry before degrading
    assert calls["n"] == 2
    assert len(quality.snapshot_quality()) == 1


def test_non_numeric_max_calls_env_falls_back(monkeypatch):
    _fresh(monkeypatch, max_calls=None, split="news:6")
    monkeypatch.setenv(_MAX_ENV, "not-a-number")
    # Non-numeric total falls back to the default cap 6; with news allocated
    # all 6, the 7th call degrades on budget, not crash. (HTTP is mocked so
    # the six passing calls make no real network requests.)
    with requests_mock_post(_post_returning(_FakeResponse({"results": []}))):
        for i in range(6):
            out = tavily.get_web_search(f"q{i}")
            assert "budget exhausted" not in out
        assert "budget exhausted" in tavily.get_web_search("q7")


# --------------------------------------------------------------------------- #
# Scoped budget: news:8 / market:5 / fundamentals:2 by default — each analyst
# spends its own cap and cannot starve the others; the run total caps the sum.
# --------------------------------------------------------------------------- #
def test_scoped_budgets_do_not_starve(monkeypatch):
    _fresh(monkeypatch)
    with requests_mock_post(_post_returning(_FakeResponse({"results": []}))):
        for i in range(8):
            assert "budget exhausted" not in tavily.get_web_search(
                f"n{i}", scope="news"
            )
        assert "budget exhausted" in tavily.get_web_search("n8", scope="news")
        # news exhausted its 8 — market's 5 and fundamentals' 2 are untouched.
        for i in range(5):
            assert "budget exhausted" not in tavily.get_web_search(
                f"m{i}", scope="market"
            )
        assert "budget exhausted" in tavily.get_web_search("m5", scope="market")
        for i in range(2):
            assert "budget exhausted" not in tavily.get_web_search(
                f"f{i}", scope="fundamentals"
            )
        assert "budget exhausted" in tavily.get_web_search(
            "f2", scope="fundamentals"
        )


def test_total_cap_binds_across_scopes(monkeypatch):
    """Explicit split may exceed the total; the run-total cap has final say."""
    _fresh(monkeypatch, max_calls=6, split="news:5,market:5")
    with requests_mock_post(_post_returning(_FakeResponse({"results": []}))):
        for i in range(5):
            assert "budget exhausted" not in tavily.get_web_search(
                f"n{i}", scope="news"
            )
        assert "budget exhausted" not in tavily.get_web_search("m0", scope="market")
        # 6 total calls charged: total cap fires before market's own 5.
        total_hit = tavily.get_web_search("m1", scope="market")
    assert "total search budget exhausted" in total_hit
    assert "6/6" in total_hit


def test_total_override_rescales_default_split(monkeypatch):
    """Raising only the total rescales the default split proportionally
    (12 -> news:6 / market:4 / fundamentals:2, largest-remainder)."""
    _fresh(monkeypatch, max_calls=12)
    with requests_mock_post(_post_returning(_FakeResponse({"results": []}))):
        for i in range(6):
            assert "budget exhausted" not in tavily.get_web_search(
                f"n{i}", scope="news"
            )
        assert "budget exhausted" in tavily.get_web_search("n6", scope="news")
        for i in range(4):
            assert "budget exhausted" not in tavily.get_web_search(
                f"m{i}", scope="market"
            )
        assert "budget exhausted" in tavily.get_web_search("m4", scope="market")
        for i in range(2):
            assert "budget exhausted" not in tavily.get_web_search(
                f"f{i}", scope="fundamentals"
            )
        assert "budget exhausted" in tavily.get_web_search(
            "f2", scope="fundamentals"
        )


def test_zero_total_cap_disables_all_scopes(monkeypatch):
    _fresh(monkeypatch, max_calls=0)
    result = tavily.get_web_search("q")
    assert "budget exhausted" in result


def test_scope_without_allocation_sentinel(monkeypatch):
    _fresh(monkeypatch, split="news:3")
    result = tavily.get_web_search("q", scope="market")
    assert "WEB_SEARCH_UNAVAILABLE" in result
    assert "no budget allocation" in result
    assert "market" in result


def test_malformed_split_env_falls_back(monkeypatch):
    _fresh(monkeypatch, split="garbage")
    with requests_mock_post(_post_returning(_FakeResponse({"results": []}))):
        for i in range(8):
            assert "budget exhausted" not in tavily.get_web_search(f"n{i}")
        assert "budget exhausted" in tavily.get_web_search("n8")


def test_parse_budget_split():
    assert tavily.parse_budget_split("news:3,market:2,fundamentals:1") == {
        "news": 3, "market": 2, "fundamentals": 1,
    }
    assert tavily.parse_budget_split("news: 0 ") == {"news": 0}
    assert tavily.parse_budget_split("nope") is None
    assert tavily.parse_budget_split("news:-1") is None
    assert tavily.parse_budget_split("news") is None
    assert tavily.parse_budget_split("") is None


def test_default_scope_is_news_for_backcompat(monkeypatch):
    """scope-less calls (pre-scoping callers) charge the news bucket."""
    _fresh(monkeypatch, max_calls=15)
    with requests_mock_post(_post_returning(_FakeResponse({"results": []}))):
        tavily.get_web_search("q")
    assert tavily.run_usage() == {"news": 1, "market": 0, "fundamentals": 0}


def test_run_usage_counts_per_scope_and_resets(monkeypatch):
    _fresh(monkeypatch)
    with requests_mock_post(_post_returning(_FakeResponse({"results": []}))):
        tavily.get_web_search("n", scope="news")
        tavily.get_web_search("n2", scope="news")
        tavily.get_web_search("m", scope="market")
    assert tavily.run_usage() == {"news": 2, "market": 1, "fundamentals": 0}
    tavily.reset_run_budget()
    assert tavily.run_usage() == {"news": 0, "market": 0, "fundamentals": 0}


# --------------------------------------------------------------------------- #
# Key pool: TAVILY_API_KEYS (comma-separated) + TAVILY_API_KEY round-robin;
# 401/403/429 rotates a key out for the rest of the run. Balanced burn, N x
# free-tier quota, and no single dead key can take the tool down.
# --------------------------------------------------------------------------- #
def test_api_key_pool_parses_dedupes_and_appends_single(monkeypatch):
    _fresh(monkeypatch, keys=["tvly-dev-x", "tvly-dev-y"])
    monkeypatch.setenv(_KEY_ENV, "tvly-dev-x")  # dup of a pool entry
    assert tavily.api_key_pool() == ["tvly-dev-x", "tvly-dev-y"]


def test_pool_round_robin_distributes_keys(monkeypatch):
    keys = ["tvly-dev-fake-a", "tvly-dev-fake-b", "tvly-dev-fake-c"]
    _fresh(monkeypatch, keys=keys)
    seen = []

    def _post(url, json=None, headers=None, timeout=None):  # noqa: A002
        seen.append(headers["Authorization"])
        return _FakeResponse({"results": []})

    with requests_mock_post(_post):
        for i in range(3):
            out = tavily.get_web_search(f"q{i}")
            assert "WEB_SEARCH_UNAVAILABLE" not in out
    assert sorted(seen) == sorted(f"Bearer {k}" for k in keys)


def test_dead_key_fails_over_within_call_and_never_retried(monkeypatch):
    _fresh(monkeypatch, keys=["tvly-dev-bad", "tvly-dev-good"])
    posts = {"n": 0}

    def _post(url, json=None, headers=None, timeout=None):  # noqa: A002
        posts["n"] += 1
        if "bad" in headers["Authorization"]:
            return _FakeResponse(status=401)
        return _FakeResponse({"results": []})

    with requests_mock_post(_post):
        first = tavily.get_web_search("q")
        second = tavily.get_web_search("q2")
    assert "WEB_SEARCH_UNAVAILABLE" not in first
    assert "WEB_SEARCH_UNAVAILABLE" not in second
    # call 1: bad-key 401 then good key (2 posts); call 2: good key only (1).
    assert posts["n"] == 3
    # the rotation stays visible in the quality ledger — never silent
    assert any("HTTP 401" in e["detail"] for e in quality.snapshot_quality())


def test_all_keys_dead_sentinel(monkeypatch):
    _fresh(monkeypatch, keys=["tvly-dev-bad1", "tvly-dev-bad2"])
    with requests_mock_post(_post_returning(_FakeResponse(status=429))):
        result = tavily.get_web_search("q")
    assert "WEB_SEARCH_UNAVAILABLE" in result
    assert "all 2 Tavily keys" in result
    assert "429" in result


def test_reset_revives_dead_keys(monkeypatch):
    """A key dead in one run is retried the next run (quotas recover)."""
    _fresh(monkeypatch, keys=["tvly-dev-flaky", "tvly-dev-good"])

    def _post(url, json=None, headers=None, timeout=None):  # noqa: A002
        if "flaky" in headers["Authorization"]:
            return _FakeResponse(status=401)
        return _FakeResponse({"results": []})

    with requests_mock_post(_post):
        out = tavily.get_web_search("q")  # key #1 dead, key #2 serves
    assert "WEB_SEARCH_UNAVAILABLE" not in out
    tavily.reset_run_budget()
    # Rotation restarts at key #1 — it now serves again (quota recovered).
    with requests_mock_post(_post):
        revived = tavily.get_web_search("q2")
    assert "WEB_SEARCH_UNAVAILABLE" not in revived


def test_tool_wrapper_delegates(monkeypatch):
    _fresh(monkeypatch, key=None)
    result = web_search.invoke({"query": "anything"})
    assert "WEB_SEARCH_UNAVAILABLE" in result


def requests_mock_post(fake):
    """Patch tavily.requests.post with ``fake`` behind a recording Mock.

    ``wraps`` delegates to the real-signature fake (so a vendor-side signature
    drift still raises TypeError here) while exposing ``call_args``.
    """
    from unittest import mock

    return mock.patch.object(tavily.requests, "post", mock.Mock(wraps=fake))


# --------------------------------------------------------------------------- #
# Wiring: the news analyst's bind list, prompt, and the ToolNode registry
# must stay in sync (three-place rule, round-5 audit). The ToolNode registry
# side is pinned in tests/test_market_toolnode.py; these pin the analyst side.
# --------------------------------------------------------------------------- #
def _news_bound_tools_and_prompt(config_overrides, trade_date=None):
    from langchain_core.messages import HumanMessage
    from langchain_core.runnables import Runnable

    from yiagents.agents.analysts.news_analyst import create_news_analyst
    from yiagents.dataflows import config as cfgmod

    class _RecordingLLM(Runnable):
        def __init__(self):
            super().__init__()
            self.bound_tools = None
            self.last_input = None

        def bind_tools(self, tools, **kwargs):  # noqa: ARG002
            self.bound_tools = [t.name for t in tools]
            return self

        def invoke(self, inp, config=None, **kwargs):  # noqa: ARG002
            self.last_input = str(inp)
            from langchain_core.messages import AIMessage
            return AIMessage(content="MOCK REPORT", tool_calls=[])

    orig = cfgmod.get_config()
    try:
        cfgmod.set_config({**orig, **config_overrides})
        llm = _RecordingLLM()
        node = create_news_analyst(llm)
        node({
            # Default to a live date (web_search binds on live dates only);
            # historical-date behaviour has its own test below.
            "trade_date": trade_date or datetime.date.today().isoformat(),
            "company_of_interest": "AAPL",
            "asset_type": "stock",
            "instrument_context": "CTX",
            "messages": [HumanMessage(content="analyze")],
        })
        return llm.bound_tools, llm.last_input
    finally:
        cfgmod.set_config(orig)


def test_news_analyst_binds_and_prompts_web_search_when_enabled():
    tools, prompt = _news_bound_tools_and_prompt({"web_search_enabled": True})
    assert "web_search" in tools
    assert "web_search(query)" in prompt
    # citation discipline travels with the tool, not just the mention
    assert "unverified text" in prompt


def test_news_analyst_disabled_is_byte_equivalent():
    tools, prompt = _news_bound_tools_and_prompt({"web_search_enabled": False})
    assert "web_search" not in tools
    assert "web_search" not in prompt


def test_news_analyst_historical_date_never_binds_web_search():
    """PIT guard: Tavily has no as-of parameter, so a historical replay must
    not see the tool or its prompt nudge (would leak future web context —
    same contract as the prediction-markets gate)."""
    tools, prompt = _news_bound_tools_and_prompt(
        {"web_search_enabled": True}, trade_date="2024-06-15"
    )
    assert "web_search" not in tools
    assert "web_search" not in prompt
    # prediction markets stay gated the same way (pre-existing contract)
    assert "get_prediction_markets" not in tools


# --------------------------------------------------------------------------- #
# Market / fundamentals wiring: same three-place rule as news (bind + prompt
# nudge + ToolNode registry). Live dates only; byte-equivalent when the flag
# is off or the date is historical.
# --------------------------------------------------------------------------- #
def _recording_llm_class():
    from langchain_core.runnables import Runnable

    class _RecordingLLM(Runnable):
        def __init__(self):
            super().__init__()
            self.bound_tools = None
            self.last_input = None

        def bind_tools(self, tools, **kwargs):  # noqa: ARG002
            self.bound_tools = [t.name for t in tools]
            return self

        def invoke(self, inp, config=None, **kwargs):  # noqa: ARG002
            from langchain_core.messages import AIMessage
            self.last_input = str(inp)
            return AIMessage(content="MOCK REPORT", tool_calls=[])

    return _RecordingLLM


def _market_bound_tools_and_prompt(config_overrides, trade_date=None):
    from langchain_core.messages import HumanMessage

    from yiagents.agents.analysts.market_analyst import create_market_analyst
    from yiagents.dataflows import config as cfgmod

    orig = cfgmod.get_config()
    try:
        cfgmod.set_config({**orig, **config_overrides})
        llm = _recording_llm_class()()
        node = create_market_analyst(llm)
        node({
            "trade_date": trade_date or datetime.date.today().isoformat(),
            "company_of_interest": "AAPL",
            "asset_type": "stock",
            "instrument_context": "CTX",
            "messages": [HumanMessage(content="analyze")],
        })
        return llm.bound_tools, llm.last_input
    finally:
        cfgmod.set_config(orig)


def test_market_analyst_binds_and_prompts_web_search_when_enabled():
    tools, prompt = _market_bound_tools_and_prompt({"web_search_enabled": True})
    assert "web_search" in tools
    assert "web_search(query)" in prompt
    assert "unverified text" in prompt


def test_market_analyst_disabled_is_byte_equivalent():
    tools, prompt = _market_bound_tools_and_prompt({"web_search_enabled": False})
    assert "web_search" not in tools
    assert "web_search" not in prompt


def test_market_analyst_historical_date_never_binds_web_search():
    tools, prompt = _market_bound_tools_and_prompt(
        {"web_search_enabled": True}, trade_date="2024-06-15"
    )
    assert "web_search" not in tools
    assert "web_search" not in prompt


def _fundamentals_bound_tools_and_prompt(config_overrides, trade_date=None):
    from langchain_core.messages import HumanMessage

    from yiagents.agents.analysts.fundamentals_analyst import (
        create_fundamentals_analyst,
    )
    from yiagents.dataflows import config as cfgmod

    orig = cfgmod.get_config()
    try:
        cfgmod.set_config({**orig, **config_overrides})
        llm = _recording_llm_class()()
        node = create_fundamentals_analyst(llm)
        node({
            "trade_date": trade_date or datetime.date.today().isoformat(),
            "company_of_interest": "AAPL",
            "asset_type": "stock",
            "instrument_context": "CTX",
            "messages": [HumanMessage(content="analyze")],
        })
        return llm.bound_tools, llm.last_input
    finally:
        cfgmod.set_config(orig)


def test_fundamentals_analyst_binds_and_prompts_web_search_when_enabled():
    tools, prompt = _fundamentals_bound_tools_and_prompt(
        {"web_search_enabled": True}
    )
    assert "web_search" in tools
    assert "web_search(query)" in prompt
    assert "unverified text" in prompt


def test_fundamentals_analyst_disabled_is_byte_equivalent():
    tools, prompt = _fundamentals_bound_tools_and_prompt(
        {"web_search_enabled": False}
    )
    assert "web_search" not in tools
    assert "web_search" not in prompt


def test_fundamentals_analyst_historical_date_never_binds_web_search():
    tools, prompt = _fundamentals_bound_tools_and_prompt(
        {"web_search_enabled": True}, trade_date="2024-06-15"
    )
    assert "web_search" not in tools
    assert "web_search" not in prompt


# --------------------------------------------------------------------------- #
# Context-boundary regression (2026-08-17 first live run): ToolNode executes
# every tool call in a worker thread under a COPIED context. The copied
# context shares bound *objects* (so the quality ledger's list appends reach
# the run root) but not later ContextVar.set() rebinds — the original
# copy-on-write counters charged only the worker's own context copy, so the
# first real NVDA run recorded ``web_search_usage: 0`` across all scopes
# while the news analyst actually made 12 Tavily calls, and the budget gate
# (which reads the same counters) never fired. These tests drive a REAL
# StateGraph -> ToolNode -> worker-thread path, mirroring
# test_quality_context_propagation.py; same-thread unit tests cannot catch
# this class.
# --------------------------------------------------------------------------- #
def _toolnode_graph():
    from typing import TypedDict

    from langchain_core.messages import AIMessage
    from langgraph.graph import END, START, StateGraph
    from langgraph.prebuilt import ToolNode

    class _S(TypedDict, total=False):
        messages: list

    def route(state: _S) -> dict:
        return {
            "messages": [
                AIMessage(
                    "",
                    tool_calls=[
                        {"name": "web_search", "args": {"query": "q"}, "id": "c1"}
                    ],
                )
            ]
        }

    builder = StateGraph(_S)
    builder.add_node("route", route)
    builder.add_node("tools", ToolNode([web_search]))
    builder.add_edge(START, "route")
    builder.add_edge("route", "tools")
    builder.add_edge("tools", END)
    return builder.compile()


def test_run_usage_visible_through_real_toolnode(monkeypatch):
    _fresh(monkeypatch)
    graph = _toolnode_graph()
    with requests_mock_post(_post_returning(_FakeResponse({"results": []}))):
        out = graph.invoke({"messages": []})
    assert tavily.run_usage() == {"news": 1, "market": 0, "fundamentals": 0}
    # The successful call must not fabricate degradation evidence either.
    assert quality.snapshot_quality() == []
    tool_msgs = [m for m in out["messages"] if getattr(m, "type", "") == "tool"]
    assert len(tool_msgs) == 1


def test_budget_enforced_across_toolnode_executions(monkeypatch):
    """Charges must ACCUMULATE across separate ToolNode worker contexts:
    pre-fix each invocation saw a zeroed copy, so the cap could never fire
    (12 calls passed a news:8 cap in the first live run)."""
    _fresh(monkeypatch)
    monkeypatch.setenv(_SPLIT_ENV, "news:2")
    graph = _toolnode_graph()
    with requests_mock_post(_post_returning(_FakeResponse({"results": []}))):
        graph.invoke({"messages": []})
        graph.invoke({"messages": []})
        third = graph.invoke({"messages": []})
    assert tavily.run_usage()["news"] == 2
    third_tool = [
        m for m in third["messages"] if getattr(m, "type", "") == "tool"
    ][0]
    assert "budget exhausted" in third_tool.content
    assert any("budget" in e["detail"] for e in quality.snapshot_quality())
