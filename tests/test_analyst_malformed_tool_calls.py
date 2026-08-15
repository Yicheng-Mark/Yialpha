"""C4 regression: malformed-tool-call responses must not yield silent empty
reports.

When the model emits a corrupt tool call, langchain surfaces it on the final
``AIMessage`` as ``invalid_tool_calls`` with ``tool_calls == []`` and (usually)
empty ``content``. The three tool-using analysts used to write that empty
string straight into their report slot with no warning — downstream nodes and
reports could not distinguish "analyst said nothing" from "analyst broke".

The shared :func:`final_analyst_report` helper now logs a WARNING with agent /
ticker context and returns a visible sentinel instead of an empty report.

Hermetic: fake LLM objects, no network.
"""

from __future__ import annotations

import logging

import pytest
from langchain_core.messages import AIMessage

from yiagents.agents.analysts.fundamentals_analyst import create_fundamentals_analyst
from yiagents.agents.analysts.market_analyst import create_market_analyst
from yiagents.agents.analysts.news_analyst import create_news_analyst
from yiagents.agents.utils.agent_utils import (
    MALFORMED_TOOL_CALLS_SENTINEL,
    final_analyst_report,
)


def _state(ticker: str = "AAPL") -> dict:
    return {
        "messages": [],
        "company_of_interest": ticker,
        "asset_type": "stock",
        "instrument_context": f"The instrument to analyze is `{ticker}`.",
        "trade_date": "2026-07-01",
    }


class _ScriptedLLM:
    """Duck-typed LLM: ``bind_tools`` returns a callable that always yields
    the scripted AIMessage (langchain coerces plain callables via
    RunnableLambda, so no real Runnable is needed)."""

    def __init__(self, message: AIMessage) -> None:
        self._message = message
        self.tools: list = []

    def bind_tools(self, tools):
        self.tools = list(tools)

        def _invoke(_messages):
            return self._message

        return _invoke


@pytest.mark.unit
class TestFinalAnalystReportHelper:
    def test_pending_tool_calls_return_empty(self):
        msg = AIMessage(
            content="",
            tool_calls=[{"name": "t", "args": {}, "id": "1"}],
        )
        assert final_analyst_report(msg, agent_name="X", ticker="AAPL") == ""

    def test_clean_final_message_returns_content(self):
        msg = AIMessage(content="The report body.")
        assert (
            final_analyst_report(msg, agent_name="X", ticker="AAPL")
            == "The report body."
        )

    def test_invalid_tool_calls_with_empty_content_yield_sentinel_and_warning(
        self, caplog,
    ):
        msg = AIMessage(content="", invalid_tool_calls=[
            {"name": "get_news", "args": "{bad json", "error": "parsing"},
        ])
        with caplog.at_level(logging.WARNING, logger="yiagents.agents.utils.agent_utils"):
            out = final_analyst_report(msg, agent_name="Market Analyst", ticker="NVDA")
        assert out == MALFORMED_TOOL_CALLS_SENTINEL
        assert any(
            "Market Analyst" in r.message and "NVDA" in r.message
            for r in caplog.records
        )

    def test_invalid_tool_calls_with_content_keeps_content_but_warns(self, caplog):
        msg = AIMessage(
            content="partial but real text",
            invalid_tool_calls=[{"name": "t", "args": "{", "error": "e"}],
        )
        with caplog.at_level(logging.WARNING, logger="yiagents.agents.utils.agent_utils"):
            out = final_analyst_report(msg, agent_name="News Analyst", ticker="TSLA")
        assert out == "partial but real text"
        assert any(r.levelno == logging.WARNING for r in caplog.records)

    def test_message_without_invalid_tool_calls_attribute_is_safe(self):
        class Bare:
            tool_calls = []
            content = "ok"

        assert final_analyst_report(Bare(), agent_name="X", ticker="A") == "ok"


def _invalid_tool_message() -> AIMessage:
    return AIMessage(content="", invalid_tool_calls=[
        {"name": "get_stock_data", "args": "{corrupt", "error": "json parse"},
    ])


@pytest.mark.unit
class TestAnalystsEmitSentinel:
    """Each factory-built analyst node surfaces the sentinel + WARNING."""

    @pytest.mark.parametrize("factory,report_key,agent_name", [
        (create_market_analyst, "market_report", "Market Analyst"),
        (create_news_analyst, "news_report", "News Analyst"),
        (create_fundamentals_analyst, "fundamentals_report", "Fundamentals Analyst"),
    ])
    def test_sentinel_written_to_state(self, factory, report_key, agent_name, caplog):
        llm = _ScriptedLLM(_invalid_tool_message())
        node = factory(llm)
        with caplog.at_level(logging.WARNING, logger="yiagents.agents.utils.agent_utils"):
            update = node(_state("AMD"))
        assert update[report_key] == MALFORMED_TOOL_CALLS_SENTINEL
        assert any(
            agent_name in r.message and "AMD" in r.message for r in caplog.records
        )

    @pytest.mark.parametrize("factory,report_key", [
        (create_market_analyst, "market_report"),
        (create_news_analyst, "news_report"),
        (create_fundamentals_analyst, "fundamentals_report"),
    ])
    def test_clean_response_unchanged(self, factory, report_key):
        llm = _ScriptedLLM(AIMessage(content="A perfectly normal report."))
        node = factory(llm)
        update = node(_state())
        assert update[report_key] == "A perfectly normal report."

    @pytest.mark.parametrize("factory,report_key", [
        (create_market_analyst, "market_report"),
        (create_news_analyst, "news_report"),
        (create_fundamentals_analyst, "fundamentals_report"),
    ])
    def test_tool_round_still_returns_empty_report(self, factory, report_key):
        llm = _ScriptedLLM(AIMessage(
            content="", tool_calls=[{"name": "t", "args": {}, "id": "1"}],
        ))
        node = factory(llm)
        update = node(_state())
        assert update[report_key] == ""
