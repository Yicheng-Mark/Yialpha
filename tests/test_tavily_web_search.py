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

import requests

from yiagents.agents.utils.web_search_tools import web_search
from yiagents.dataflows import quality, tavily

_KEY_ENV = "TAVILY" + "_API_KEY"
_MAX_ENV = "YIAGENTS_TAVILY_MAX_CALLS" + "_PER_RUN"
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


def _fresh(monkeypatch, *, key=_FAKE_KEY, max_calls=None):
    quality.ensure_run_context()
    tavily.reset_run_budget()
    if key is None:
        monkeypatch.delenv(_KEY_ENV, raising=False)
    else:
        monkeypatch.setenv(_KEY_ENV, key)
    if max_calls is None:
        monkeypatch.delenv(_MAX_ENV, raising=False)
    else:
        monkeypatch.setenv(_MAX_ENV, str(max_calls))


def test_missing_key_sentinel_and_evidence(monkeypatch):
    _fresh(monkeypatch, key=None)
    result = tavily.get_web_search("nvda news")
    assert "WEB_SEARCH_UNAVAILABLE" in result
    assert "TAVILY_API_KEY" in result
    events = quality.snapshot_quality()
    assert len(events) == 1
    assert events[0]["method"] == "web_search"
    assert events[0]["kind"] == quality.KIND_OPTIONAL_UNAVAILABLE


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
    _fresh(monkeypatch, max_calls=None)
    monkeypatch.setenv(_MAX_ENV, "not-a-number")
    # default cap is 6: the 7th call degrades on budget, not crash
    for i in range(6):
        out = tavily.get_web_search(f"q{i}")
        assert "budget exhausted" not in out
    assert "budget exhausted" in tavily.get_web_search("q7")


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
def _news_bound_tools_and_prompt(config_overrides):
    from langchain_core.messages import AIMessage, HumanMessage
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
            return AIMessage(content="MOCK REPORT", tool_calls=[])

    orig = cfgmod.get_config()
    try:
        cfgmod.set_config({**orig, **config_overrides})
        llm = _RecordingLLM()
        node = create_news_analyst(llm)
        node({
            "trade_date": "2024-06-15",  # historical: no prediction_markets
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
