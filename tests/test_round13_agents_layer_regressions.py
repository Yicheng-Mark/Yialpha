"""Round-13 adversarial-sweep regression pins — agent-layer defense shard.

Independent shard (owned by the agents-layer pass; do NOT append to
tests/test_round13_regressions.py). The graph entry now normalizes
``propagate()``'s trade_date, but four defensive gaps survived behind it
(LLMs echoing prompt dates, weak models serializing containers,
programmatic direct calls). This file pins each fix:

* P1 — ``get_news`` normalizes timestamp bounds at the TOOL layer, so both
  the bound-tool path (LLM echoing "2026-08-01 16:00:00" from the prompt)
  and the programmatic ``.func`` path reach the vendor as ``YYYY-MM-DD``;
  a bare strptime at the vendor (yfinance) / Alpha Vantage's
  ``format_datetime_for_api`` ValueError'd, and news_data is a CORE
  category (fail-closed) — the sentiment node's prefetch aborted the run.
* P3 — the tool-loop analysts' final round with empty content and no
  (in)valid tool calls returned ``""``; round-12's EMPTY_RESPONSE_SENTINEL
  had closed that hole only on the structured free-text path.
* P3 — ``PortfolioDecision.probabilities`` / ``invalidation`` received a
  weak model's STRING form (container serialized to JSON text / one
  sentence) and the pydantic ValidationError vetoed the whole decision —
  an optional field's shape degrading every primary output to free text.
* P3 — an unparseable trade_date ("2026/08/01") ValueError'd out of the
  sentiment/news lookback helpers and killed the node before any LLM call;
  they now degrade fail-closed (unavailable placeholders), mirroring
  ``is_historical_date``'s malformed-date branch.

Zero network, zero real LLM. Config is reset per test by conftest.
"""

from __future__ import annotations

import logging

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import ValidationError

import yialpha.agents.analysts.news_analyst as news
import yialpha.agents.analysts.sentiment_analyst as sent
import yialpha.agents.utils.news_data_tools as ndt
from yialpha.agents.schemas import PortfolioDecision
from yialpha.agents.utils.agent_utils import final_analyst_report
from yialpha.agents.utils.structured import EMPTY_RESPONSE_SENTINEL

# --------------------------------------------------------------------------- #
# P1 — get_news tool-layer date normalization
# --------------------------------------------------------------------------- #


def _spy_router(seen: list[tuple], monkeypatch) -> None:
    """In-memory vendor-router spy: records every routed call, no network."""

    def spy(*args):
        seen.append(args)
        return "MOCK"

    monkeypatch.setattr(ndt, "route_to_vendor", spy)
    monkeypatch.setattr(
        "yialpha.dataflows.binance.stock_perp_underlying", lambda t: None,
    )


@pytest.mark.unit
def test_get_news_strips_timestamp_form_from_bounds(monkeypatch):
    """A bound-tool call echoing the prompt's full-ISO header date must reach
    the vendor as the date part — the vendor strptime/AV formatter ValueError
    on the time part, and the core-category fail-closed contract turned that
    into a run abort."""
    seen: list[tuple] = []
    _spy_router(seen, monkeypatch)

    ndt.get_news.invoke(
        {
            "ticker": "AAPL",
            "start_date": "2026-08-01 16:00:00",
            "end_date": "2026-09-03 09:30:00",
        }
    )
    assert seen == [("get_news", "AAPL", "2026-08-01", "2026-09-03")]


@pytest.mark.unit
def test_get_news_direct_func_call_normalizes_too(monkeypatch):
    """The programmatic direct-call path (sentiment / news prefetch legs use
    ``.func``) gets the same single-point normalization."""
    seen: list[tuple] = []
    _spy_router(seen, monkeypatch)

    ndt.get_news.func("AAPL", "2026-08-01", "2026-09-03 16:00:00")
    assert seen == [("get_news", "AAPL", "2026-08-01", "2026-09-03")]


@pytest.mark.unit
def test_get_news_canonical_bounds_byte_identical(monkeypatch):
    """``str(x)[:10]`` on an already-canonical date changes nothing — the
    canonical-input contract (pinned at length in test_stock_perp_news.py)
    must hold through the new normalization."""
    seen: list[tuple] = []
    _spy_router(seen, monkeypatch)

    ndt.get_news.func("AAPL", "2026-08-27", "2026-09-03")
    assert seen == [("get_news", "AAPL", "2026-08-27", "2026-09-03")]


# --------------------------------------------------------------------------- #
# P3 — tool-loop analysts: empty final round
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_final_round_empty_content_yields_empty_response_sentinel(caplog):
    """Final round, NO tool calls of any kind, empty content (a thinking
    model that answered entirely in reasoning content): the "" that used to
    flow out read as "section absent" in every downstream truthiness guard.
    Now the round-12 marker fires — same string, same semantics as the
    structured free-text path."""
    with caplog.at_level(
        logging.WARNING, logger="yialpha.agents.utils.agent_utils",
    ):
        out = final_analyst_report(
            AIMessage(content=""), agent_name="News Analyst", ticker="AMD",
        )
    assert out == EMPTY_RESPONSE_SENTINEL
    assert any(
        "News Analyst" in r.message and "AMD" in r.message
        for r in caplog.records
    )


@pytest.mark.unit
def test_final_round_whitespace_only_content_yields_sentinel():
    assert (
        final_analyst_report(AIMessage(content="  \n\t"), agent_name="X", ticker="A")
        == EMPTY_RESPONSE_SENTINEL
    )


@pytest.mark.unit
def test_pending_tool_calls_still_return_empty_string():
    """The loop-state artifact is unchanged: a round with PENDING tool calls
    still reports "" — the graph routes on ``tool_calls``, and the report is
    only meaningful once the loop ends."""
    msg = AIMessage(content="", tool_calls=[{"name": "t", "args": {}, "id": "1"}])
    assert final_analyst_report(msg, agent_name="X", ticker="A") == ""


@pytest.mark.unit
def test_final_round_real_content_still_passthrough():
    assert (
        final_analyst_report(AIMessage(content="A report."), agent_name="X", ticker="A")
        == "A report."
    )


@pytest.mark.unit
def test_news_analyst_empty_final_round_writes_sentinel_to_state():
    """Node-level pin: a tool-loop analyst whose final reply carries neither
    content nor tool calls writes the marker into its report slot."""

    class _ScriptedLLM:
        def bind_tools(self, tools):  # noqa: ANN001
            def _invoke(_messages):
                return AIMessage(content="")

            return _invoke

    node = news.create_news_analyst(_ScriptedLLM())
    update = node({
        "messages": [],
        "company_of_interest": "AMD",
        "asset_type": "stock",
        "instrument_context": "The instrument to analyze is `AMD`.",
        "trade_date": "2026-07-01",
    })
    assert update["news_report"] == EMPTY_RESPONSE_SENTINEL


# --------------------------------------------------------------------------- #
# P3 — PortfolioDecision string-form container rescue
# --------------------------------------------------------------------------- #


def _decision_kwargs() -> dict:
    return {
        "rating": "Buy",
        "executive_summary": "Add on strength.",
        "investment_thesis": "Momentum with support.",
    }


@pytest.mark.unit
def test_probabilities_json_string_form_rescued():
    """A weak model serializing the distribution to a JSON string must not
    veto the whole decision — the container shape is rescued, values pass
    through the same unit-interval coercers as a native dict."""
    d = PortfolioDecision(
        **_decision_kwargs(),
        probabilities='{"bull": 0.61, "neutral": 0.24, "bear": 0.15}',
    )
    assert d.probabilities is not None
    assert d.probabilities.bull == pytest.approx(0.61)
    assert d.probabilities.neutral == pytest.approx(0.24)
    assert d.probabilities.bear == pytest.approx(0.15)
    assert d.rating.value == "Buy"


@pytest.mark.unit
def test_probabilities_json_string_percent_form_rescued():
    # Percent twins inside the rescued container degrade through the same
    # unit-interval coercer as the dict form (round-7 semantics preserved).
    d = PortfolioDecision(
        **_decision_kwargs(),
        probabilities='{"bull": "70%", "neutral": 0.2, "bear": 0.1}',
    )
    assert d.probabilities is not None
    assert d.probabilities.bull == pytest.approx(0.70)


@pytest.mark.unit
def test_probabilities_non_json_string_still_fails_validation():
    # Garbage prose is NOT rescued: json.loads fails, the original value
    # flows on, and pydantic vetoes the field honestly.
    with pytest.raises(ValidationError):
        PortfolioDecision(
            **_decision_kwargs(),
            probabilities="bull 61 / neutral 24 / bear 16",
        )


@pytest.mark.unit
def test_probabilities_valid_json_non_object_still_fails_validation():
    # Valid JSON of the wrong shape (a bare array) is a real veto too —
    # only the container-as-string form is rescued, nothing is mapped.
    with pytest.raises(ValidationError):
        PortfolioDecision(
            **_decision_kwargs(),
            probabilities="[0.61, 0.24, 0.15]",
        )


@pytest.mark.unit
def test_invalidation_single_sentence_string_wrapped():
    d = PortfolioDecision(
        **_decision_kwargs(),
        invalidation="closes below 180 support; funding flips negative",
    )
    assert d.invalidation == ["closes below 180 support; funding flips negative"]
    assert d.rating.value == "Buy"


@pytest.mark.unit
def test_invalidation_nullish_string_degrades_to_none():
    # Placeholder strings mean "not provided" (same nullish contract as the
    # optional floats), not a one-element list carrying the placeholder.
    for junk in ("", "   ", "N/A", "none"):
        d = PortfolioDecision(**_decision_kwargs(), invalidation=junk)
        assert d.invalidation is None


@pytest.mark.unit
def test_invalidation_native_list_form_unchanged():
    d = PortfolioDecision(
        **_decision_kwargs(),
        invalidation=["closes below 180 support", "funding flips negative"],
    )
    assert d.invalidation == ["closes below 180 support", "funding flips negative"]


# --------------------------------------------------------------------------- #
# P3 — unparseable trade_date in the sentiment/news lookbacks
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_seven_days_back_malformed_date_returns_none_with_warning(caplog):
    """Slash/other malformed forms used to ValueError out of strptime and
    kill the node; now WARNING + None (no window invented), and the canonical
    dual forms keep their round-11 behaviour."""
    with caplog.at_level(
        logging.WARNING, logger="yialpha.agents.analysts.sentiment_analyst",
    ):
        assert sent._seven_days_back("2026/08/01") is None
    assert any("not a parseable" in r.message for r in caplog.records)
    # Canonical forms unchanged.
    assert sent._seven_days_back("2026-08-01 16:00:00") == "2026-07-25"
    assert sent._seven_days_back("2026-08-01") == "2026-07-25"


@pytest.mark.unit
def test_lookback_start_malformed_date_returns_none_with_warning(caplog):
    with caplog.at_level(
        logging.WARNING, logger="yialpha.agents.analysts.news_analyst",
    ):
        assert news._lookback_start("01.08.2026") is None
    assert any("not a parseable" in r.message for r in caplog.records)
    # Canonical forms unchanged.
    assert news._lookback_start("2026-08-01 16:00:00") == "2026-07-25"
    assert news._lookback_start("2026-08-01") == "2026-07-25"


@pytest.mark.unit
def test_fetch_sentiment_sources_none_start_degrades_without_vendor(monkeypatch):
    """None start (= unparseable trade date) routes the news leg to the
    explicit placeholder and never reaches the vendor seam; a VALID
    historical date keeps fetching through it (the guard must not
    over-fire)."""
    calls: list[tuple] = []

    def fake_impl(ticker, start_date, end_date):
        calls.append((ticker, start_date, end_date))
        return "NEWS BODY"

    monkeypatch.setattr(sent, "_get_news_impl", fake_impl)

    news_block, stocktwits, reddit, square = sent._fetch_sentiment_sources(
        "BTCUSDT", None, "2026/08/01", asset_type="crypto_perp",
    )
    assert news_block.startswith("<unavailable: trade date")
    assert "unavailable" in stocktwits  # social legs fail closed too
    assert "unavailable" in reddit
    assert square is not None and "unavailable" in square
    assert calls == []  # the vendor never saw the undefined window

    # Valid historical date: the date-bounded fetch still happens.
    news_block, _stocktwits, _reddit, _square = sent._fetch_sentiment_sources(
        "BTCUSDT", "2026-07-25", "2026-08-01", asset_type="crypto_perp",
    )
    assert news_block == "NEWS BODY"
    assert calls == [("BTCUSDT", "2026-07-25", "2026-08-01")]


@pytest.mark.unit
def test_fetch_company_news_none_start_degrades_without_vendor(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("vendor must not be called without a window")

    monkeypatch.setattr(news, "_get_news_impl", boom)
    out = news._fetch_company_news("MU", None, "2026/08/01")
    assert out.startswith("<company-angle news unavailable: unparseable trade date")


class _FreeTextLLM:
    """Structured binding fails -> free-text path; records the messages."""

    def __init__(self, content: str):
        self._content = content
        self.seen_messages = None

    def with_structured_output(self, schema):  # noqa: ANN001
        raise NotImplementedError("no structured output")

    def invoke(self, messages):
        self.seen_messages = messages
        return AIMessage(content=self._content)


@pytest.mark.unit
def test_sentiment_node_malformed_trade_date_degrades_not_crashes(monkeypatch):
    """End-to-end pin for the chosen fallback semantics: an unparseable
    trade_date no longer aborts the run — the news leg degrades to the
    unavailable placeholder, social legs fail closed, and the analyst still
    produces its report from the honest gaps."""
    def boom(*a, **k):
        raise AssertionError("vendor must not be called for a malformed date")

    monkeypatch.setattr(sent, "_get_news_impl", boom)
    llm = _FreeTextLLM("**Overall Sentiment:** **Neutral** (Score: 5.0/10)")
    node = sent.create_sentiment_analyst(llm)
    out = node({
        "trade_date": "2026/08/01",  # unparseable form an LLM can emit
        "company_of_interest": "BTCUSDT",
        "asset_type": "crypto_perp",
        "instrument_context": "CTX",
        "messages": [HumanMessage(content="analyze")],
    })

    assert out["sentiment_report"].startswith("**Overall Sentiment:**")
    evidence = llm.seen_messages[-1]
    assert evidence.content.startswith("[EXTERNAL EVIDENCE")
    assert "<unavailable: trade date" in evidence.content


@pytest.mark.unit
def test_news_node_malformed_trade_date_degrades_not_crashes(monkeypatch):
    """The perp dual-angle prefetch survives an unparseable trade_date: both
    angles degrade to per-angle placeholders (company via the None-window
    guard, contract via is_historical_date's malformed branch) and the tool
    loop still completes."""

    class _ScriptedLLM:
        def __init__(self):
            self.seen_messages = None

        def bind_tools(self, tools):  # noqa: ANN001
            outer = self

            def _invoke(messages):
                outer.seen_messages = messages
                return AIMessage(content="A report.")

            return _invoke

    monkeypatch.setattr(news, "stock_perp_underlying", lambda t: "MU")
    llm = _ScriptedLLM()
    node = news.create_news_analyst(llm)
    update = node({
        "trade_date": "2026/08/01",
        "company_of_interest": "MUUSDT",
        "asset_type": "crypto_perp",
        "instrument_context": "CTX",
        "messages": [HumanMessage(content="analyze")],
    })

    assert update["news_report"] == "A report."
    assert llm.seen_messages is not None
    # The coerced RunnableLambda receives the prompt's ChatPromptValue.
    prompt_value = llm.seen_messages
    messages = (
        prompt_value.to_messages()
        if hasattr(prompt_value, "to_messages")
        else prompt_value
    )
    evidence_text = messages[-1].content
    assert "<company-angle news unavailable: unparseable trade date" in evidence_text
    assert "<unavailable: contract-side news" in evidence_text
