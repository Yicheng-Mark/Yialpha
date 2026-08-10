"""Unit tests for pure-logic helpers in ``yiagents.cli.main``.

Focuses on: format_tokens, format_tool_args, extract_content_string,
classify_message_type, MessageBuffer state machine, update_analyst_statuses,
get_analysis_date, and batch-command validation.

Deliberately defers run_analysis (~300 stmts streaming/UI orchestrator) and
update_display (~200 stmts Rich rendering) — high effort, low logical risk.
"""
from __future__ import annotations

import datetime

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from typer.testing import CliRunner

from yiagents.cli.main import (
    MessageBuffer,
    app,
    classify_message_type,
    extract_content_string,
    format_tokens,
    format_tool_args,
    get_analysis_date,
    update_analyst_statuses,
    update_research_team_status,
)

runner = CliRunner()


# ---------------------------------------------------------------------------
# format_tokens
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestFormatTokens:
    def test_under_1000_returns_plain(self):
        assert format_tokens(0) == "0"
        assert format_tokens(999) == "999"

    def test_exactly_1000_returns_k(self):
        assert format_tokens(1000) == "1.0k"

    def test_above_1000_returns_k(self):
        assert format_tokens(1500) == "1.5k"
        assert format_tokens(25000) == "25.0k"


# ---------------------------------------------------------------------------
# format_tool_args
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestFormatToolArgs:
    def test_short_returns_as_is(self):
        result = format_tool_args({"a": 1})
        assert result == str({"a": 1})

    def test_long_truncates_with_ellipsis(self):
        long_args = {"key": "x" * 200}
        result = format_tool_args(long_args, max_length=80)
        assert len(result) == 80
        assert result.endswith("...")

    def test_custom_max_length(self):
        result = format_tool_args("abcdefghij", max_length=5)
        assert result == "ab..."


# ---------------------------------------------------------------------------
# extract_content_string
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestExtractContentString:
    def test_none_returns_none(self):
        assert extract_content_string(None) is None

    def test_empty_string_returns_none(self):
        assert extract_content_string("") is None

    def test_whitespace_returns_none(self):
        assert extract_content_string("   ") is None

    def test_literal_zero_returns_none(self):
        """ast.literal_eval('0') is falsy → treated as empty."""
        assert extract_content_string("0") is None

    def test_literal_false_returns_none(self):
        assert extract_content_string("False") is None

    def test_real_text_returns_stripped(self):
        assert extract_content_string("  hello  ") == "hello"

    def test_unparseable_string_returned_as_is(self):
        """Non-literal text (e.g. '123abc') is returned as-is."""
        assert extract_content_string("123abc") == "123abc"

    def test_dict_with_text_key(self):
        assert extract_content_string({"text": " hi "}) == "hi"

    def test_dict_with_empty_text_returns_none(self):
        assert extract_content_string({"text": ""}) is None

    def test_list_of_text_blocks_concatenates(self):
        content = [
            {"type": "text", "text": "a"},
            {"type": "text", "text": "b"},
        ]
        assert extract_content_string(content) == "a b"

    def test_list_filters_non_text_items(self):
        content = [
            {"type": "image", "url": "x"},  # non-text
            {"type": "text", "text": "real"},
            "plain string",
        ]
        result = extract_content_string(content)
        assert "real" in result
        assert "plain string" in result


# ---------------------------------------------------------------------------
# classify_message_type
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestClassifyMessageType:
    def test_human_continue_is_control(self):
        msg = HumanMessage(content="Continue")
        msg_type, content = classify_message_type(msg)
        assert msg_type == "Control"
        assert content == "Continue"

    def test_human_message_is_user(self):
        msg = HumanMessage(content="hello there")
        msg_type, _ = classify_message_type(msg)
        assert msg_type == "User"

    def test_tool_message_is_data(self):
        msg = ToolMessage(content="result data", tool_call_id="x")
        msg_type, _ = classify_message_type(msg)
        assert msg_type == "Data"

    def test_ai_message_is_agent(self):
        msg = AIMessage(content="thinking...")
        msg_type, _ = classify_message_type(msg)
        assert msg_type == "Agent"

    def test_unknown_falls_back_to_system(self):
        class Unknown:
            content = "mystery"

        msg_type, _ = classify_message_type(Unknown())
        assert msg_type == "System"


# ---------------------------------------------------------------------------
# MessageBuffer
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestMessageBufferInit:
    def test_init_adds_selected_and_fixed_agents_pending(self):
        buf = MessageBuffer()
        buf.init_for_analysis(["market", "news"])
        assert buf.agent_status["Market Analyst"] == "pending"
        assert buf.agent_status["News Analyst"] == "pending"
        # Fixed teams present
        assert buf.agent_status["Bull Researcher"] == "pending"
        assert buf.agent_status["Trader"] == "pending"
        assert buf.agent_status["Portfolio Manager"] == "pending"
        # Unselected analysts absent
        assert "Sentiment Analyst" not in buf.agent_status
        assert "Fundamentals Analyst" not in buf.agent_status

    def test_init_builds_report_sections_for_selected_only(self):
        buf = MessageBuffer()
        buf.init_for_analysis(["market"])
        assert "market_report" in buf.report_sections
        assert "investment_plan" in buf.report_sections  # always included (analyst_key=None)
        assert "trader_investment_plan" in buf.report_sections
        assert "final_trade_decision" in buf.report_sections
        assert "sentiment_report" not in buf.report_sections

    def test_init_resets_state(self):
        buf = MessageBuffer()
        buf.init_for_analysis(["market"])
        buf.agent_status["Market Analyst"] = "completed"
        buf.final_report = "stale"
        # Re-init should reset
        buf.init_for_analysis(["market"])
        assert buf.agent_status["Market Analyst"] == "pending"
        assert buf.final_report is None


@pytest.mark.unit
class TestCompletedReportsCount:
    def test_zero_when_no_content(self):
        buf = MessageBuffer()
        buf.init_for_analysis(["market"])
        assert buf.get_completed_reports_count() == 0

    def test_counts_only_finalized_sections(self):
        """A section with content but its agent NOT completed should not count."""
        buf = MessageBuffer()
        buf.init_for_analysis(["market"])
        buf.update_report_section("market_report", "content here")
        # Market Analyst still pending → not counted
        assert buf.get_completed_reports_count() == 0
        # Now mark Market Analyst completed → counted
        buf.update_agent_status("Market Analyst", "completed")
        assert buf.get_completed_reports_count() == 1


@pytest.mark.unit
class TestMessageBufferUpdates:
    def test_update_agent_status_only_updates_known(self):
        buf = MessageBuffer()
        buf.init_for_analysis(["market"])
        buf.update_agent_status("Unknown Agent", "completed")
        assert "Unknown Agent" not in buf.agent_status
        buf.update_agent_status("Market Analyst", "completed")
        assert buf.agent_status["Market Analyst"] == "completed"
        assert buf.current_agent == "Market Analyst"

    def test_update_report_section_ignores_unknown(self):
        buf = MessageBuffer()
        buf.init_for_analysis(["market"])
        buf.update_report_section("nonexistent_section", "data")
        # Should not crash; unknown section is a no-op

    def test_final_report_assembles_sections(self):
        buf = MessageBuffer()
        buf.init_for_analysis(["market"])
        buf.update_report_section("market_report", "market data")
        buf.update_report_section("trader_investment_plan", "buy 100")
        buf.update_report_section("final_trade_decision", "HOLD")
        assert buf.final_report is not None
        assert "Analyst Team Reports" in buf.final_report
        assert "Trading Team Plan" in buf.final_report
        assert "Portfolio Management Decision" in buf.final_report

    def test_final_report_none_when_no_sections(self):
        buf = MessageBuffer()
        buf.init_for_analysis(["market"])
        # No content written
        buf._update_final_report()
        assert buf.final_report is None


# ---------------------------------------------------------------------------
# update_analyst_statuses
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestUpdateAnalystStatuses:
    def test_first_selected_without_report_is_in_progress(self):
        buf = MessageBuffer()
        buf.init_for_analysis(["market", "news"])
        update_analyst_statuses(buf, {})  # empty chunk, no reports
        assert buf.agent_status["Market Analyst"] == "in_progress"
        assert buf.agent_status["News Analyst"] == "pending"

    def test_report_marks_completed_and_next_in_progress(self):
        buf = MessageBuffer()
        buf.init_for_analysis(["market", "news"])
        update_analyst_statuses(buf, {"market_report": "done"})
        assert buf.agent_status["Market Analyst"] == "completed"
        assert buf.agent_status["News Analyst"] == "in_progress"

    def test_all_complete_transitions_researcher(self):
        buf = MessageBuffer()
        buf.init_for_analysis(["market", "news"])
        update_analyst_statuses(buf, {"market_report": "d1", "news_report": "d2"})
        assert buf.agent_status["Market Analyst"] == "completed"
        assert buf.agent_status["News Analyst"] == "completed"
        assert buf.agent_status["Bull Researcher"] == "in_progress"


@pytest.mark.unit
class TestUpdateResearchTeamStatus:
    def test_sets_all_three_members(self, monkeypatch):
        # update_research_team_status operates on the module-level message_buffer.
        from yiagents.cli import main as cli_main

        buf = MessageBuffer()
        buf.init_for_analysis(["market"])
        monkeypatch.setattr(cli_main, "message_buffer", buf)
        update_research_team_status("in_progress")
        assert buf.agent_status["Bull Researcher"] == "in_progress"
        assert buf.agent_status["Bear Researcher"] == "in_progress"
        assert buf.agent_status["Research Manager"] == "in_progress"


# ---------------------------------------------------------------------------
# get_analysis_date
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestGetAnalysisDate:
    def test_accepts_valid_past_date(self, monkeypatch):
        monkeypatch.setattr("typer.prompt", lambda *a, **kw: "2020-01-01")
        assert get_analysis_date() == "2020-01-01"

    def test_rejects_future_date_then_accepts(self, monkeypatch):
        tomorrow = (datetime.datetime.now().date() + datetime.timedelta(days=1)).strftime("%Y-%m-%d")
        answers = iter([tomorrow, "2020-01-01"])
        monkeypatch.setattr("typer.prompt", lambda *a, **kw: next(answers))
        assert get_analysis_date() == "2020-01-01"

    def test_rejects_bad_format_then_accepts(self, monkeypatch):
        answers = iter(["01/02/2020", "2020-01-02"])
        monkeypatch.setattr("typer.prompt", lambda *a, **kw: next(answers))
        assert get_analysis_date() == "2020-01-02"


# ---------------------------------------------------------------------------
# batch command validation
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestBatchValidation:
    def test_rejects_invalid_ticker_exits_2(self, monkeypatch):
        monkeypatch.setenv("YIAGENTS_LLM_PROVIDER", "deepseek")
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-x")
        result = runner.invoke(app, ["batch", "-t", "bad symbol!", "-d", "2026-01-10"])
        assert result.exit_code == 2
        assert "Invalid ticker" in result.output

    def test_rejects_bad_date_exits_2(self, monkeypatch):
        monkeypatch.setenv("YIAGENTS_LLM_PROVIDER", "deepseek")
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-x")
        result = runner.invoke(app, ["batch", "-t", "AAPL", "-d", "2026/01/10"])
        assert result.exit_code == 2
        assert "Bad date" in result.output

    def test_rejects_workers_below_one_exits_2(self, monkeypatch):
        monkeypatch.setenv("YIAGENTS_LLM_PROVIDER", "deepseek")
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-x")
        result = runner.invoke(app, ["batch", "-t", "AAPL", "-d", "2026-01-10", "-w", "0"])
        assert result.exit_code == 2

    def test_rejects_mixed_asset_classes_exits_2(self, monkeypatch):
        monkeypatch.setenv("YIAGENTS_LLM_PROVIDER", "deepseek")
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-x")
        result = runner.invoke(
            app, ["batch", "-t", "AAPL", "-t", "BTC-USD", "-d", "2026-01-10", "--asset-type", "auto"]
        )
        assert result.exit_code == 2
        assert "Mixed asset classes" in result.output


# ---------------------------------------------------------------------------
# config-check: timeout provider-awareness
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestConfigCheckTimeout:
    def test_cloud_shows_default_message(self, monkeypatch):
        monkeypatch.setenv("YIAGENTS_LLM_PROVIDER", "deepseek")
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
        monkeypatch.delenv("YIAGENTS_LLM_TIMEOUT_S", raising=False)
        result = runner.invoke(app, ["config-check"])
        assert result.exit_code == 0
        assert "120s" in result.output
        assert "default" in result.output

    def test_local_shows_expected_message(self, monkeypatch):
        monkeypatch.setenv("YIAGENTS_LLM_PROVIDER", "ollama")
        monkeypatch.delenv("YIAGENTS_LLM_TIMEOUT_S", raising=False)
        result = runner.invoke(app, ["config-check"])
        assert result.exit_code == 0
        assert "local provider" in result.output.lower()
        assert "expected" in result.output.lower()

    # Native (non-OpenAI-compatible) providers are always cloud, so config-check
    # must show the 120s default message for them too (gap B — the timeout
    # safety net now applies to all clients, and the CLI reflects that).

    def test_anthropic_shows_default_message(self, monkeypatch):
        monkeypatch.setenv("YIAGENTS_LLM_PROVIDER", "anthropic")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        monkeypatch.delenv("YIAGENTS_LLM_TIMEOUT_S", raising=False)
        result = runner.invoke(app, ["config-check"])
        assert result.exit_code == 0
        assert "120s" in result.output
        assert "default" in result.output

    def test_google_shows_default_message(self, monkeypatch):
        monkeypatch.setenv("YIAGENTS_LLM_PROVIDER", "google")
        monkeypatch.setenv("GOOGLE_API_KEY", "sk-test")
        monkeypatch.delenv("YIAGENTS_LLM_TIMEOUT_S", raising=False)
        result = runner.invoke(app, ["config-check"])
        assert result.exit_code == 0
        assert "120s" in result.output
        assert "default" in result.output

