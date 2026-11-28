"""Round-15 coverage for the CLI surface: ``yialpha.cli.utils`` helpers and the
non-interactive, terminable paths of ``yialpha.cli.main``.

Scope decisions (mirrors the round brief):

- Interactive questionary prompts are driven with the established
  SimpleNamespace-ask mock (see test_cli_quality_chain.py /
  test_openrouter_model_select.py) so every branch — including the
  exit-on-cancel contracts — runs headlessly.
- Typer commands whose bodies terminate without interaction (config-check
  extras, batch happy-path, verify-history, memory-resolve gates, ic-cycle,
  analyze --clear-checkpoints) run through CliRunner with their heavy
  collaborators (BatchRunner, verify_history, run_ic_cycle, run_analysis)
  monkeypatched at their call-time import seams.
- NOT covered here (needs live graph/LLM/network): the ``run_analysis``
  stream loop and Live display, the interactive ``analyze`` wizard beyond
  its non-interactive entry checks, and main.py's ``if __name__`` guard.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest
import typer
from typer.testing import CliRunner

# Module scope on purpose: yialpha.cli.main calls setup_logging() at import
# time, which replaces every root handler (same note as
# test_cli_quality_chain.py) — the import side effect must be long done
# before any caplog-based test runs.
import yialpha.cli.main as cli_main
from yialpha.cli import utils as cli_utils
from yialpha.cli.main import (
    MessageBuffer,
    _apply_batch_worker_override,
    _store_cli_decision,
    app,
    create_layout,
    display_complete_report,
    extract_content_string,
    save_report_to_disk,
    update_analyst_statuses,
)
from yialpha.cli.models import AnalystType, AssetType
from yialpha.graph.analyst_execution import (
    AnalystWallTimeTracker,
    build_analyst_execution_plan,
)

runner = CliRunner()


def _ask(value):
    """questionary-prompt stand-in whose .ask() returns ``value``."""
    return SimpleNamespace(ask=lambda: value)


def _round15_full_state() -> dict:
    """A merged final_state carrying every report/display section."""
    return {
        "market_report": "market text",
        "positioning_report": "positioning text",
        "sentiment_report": "sentiment text",
        "news_report": "news text",
        "fundamentals_report": "fundamentals text",
        "investment_debate_state": {
            "bull_history": "bull case",
            "bear_history": "bear case",
            "judge_decision": "judge decision",
        },
        "trader_investment_plan": "trade plan",
        "risk_debate_state": {
            "aggressive_history": "aggressive view",
            "conservative_history": "conservative view",
            "neutral_history": "neutral view",
            "judge_decision": "risk judge",
        },
        "final_trade_decision": "Rating: BUY",
    }


# ---------------------------------------------------------------------------
# cli.utils — get_ticker guard rails
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRound15GetTickerPromptPaths:
    def test_cancelled_ticker_prompt_exits_1(self, monkeypatch):
        monkeypatch.setattr(cli_utils.questionary, "text", lambda *a, **k: _ask(None))
        with pytest.raises(SystemExit) as exc:
            cli_utils.get_ticker()
        assert exc.value.code == 1

    def test_unsafe_component_exit_is_last_line_of_defense(self, monkeypatch):
        """A resolved symbol the data layer's safe_ticker_component rejects
        must exit 1 with the typed message, never flow into path building."""
        monkeypatch.setattr(
            cli_utils.questionary, "text", lambda *a, **k: _ask("AAPL")
        )

        def _boom(symbol):
            raise ValueError(f"unsafe symbol {symbol!r}")

        monkeypatch.setattr(cli_utils, "safe_ticker_component", _boom)
        with pytest.raises(SystemExit) as exc:
            cli_utils.get_ticker()
        assert exc.value.code == 1


@pytest.mark.unit
class TestRound15NormalizeTickerFallback:
    def test_data_layer_failure_falls_back_to_upper(self, monkeypatch):
        import yialpha.dataflows.symbol_utils as symbol_utils

        def _boom(raw):
            raise RuntimeError("catalog on fire")

        monkeypatch.setattr(symbol_utils, "normalize_symbol", _boom)
        assert cli_utils.normalize_ticker_symbol(" brk.b ") == "BRK.B"


# ---------------------------------------------------------------------------
# cli.utils — interactive selectors (headless via questionary mocks)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRound15SelectAnalystsPrompt:
    def test_returns_selection_and_stock_shows_all_four(self, monkeypatch):
        captured: dict = {}

        def fake_checkbox(message, **kwargs):
            captured["values"] = [c.value for c in kwargs["choices"]]
            return _ask([AnalystType.MARKET, AnalystType.NEWS])

        monkeypatch.setattr(cli_utils.questionary, "checkbox", fake_checkbox)
        out = cli_utils.select_analysts(AssetType.STOCK, "AAPL")
        assert out == [AnalystType.MARKET, AnalystType.NEWS]
        assert AnalystType.FUNDAMENTALS in captured["values"]

    def test_crypto_drops_fundamentals_from_choices(self, monkeypatch):
        captured: dict = {}

        def fake_checkbox(message, **kwargs):
            captured["values"] = [c.value for c in kwargs["choices"]]
            return _ask([AnalystType.MARKET])

        monkeypatch.setattr(cli_utils.questionary, "checkbox", fake_checkbox)
        out = cli_utils.select_analysts(AssetType.CRYPTO, "BTC-USD")
        assert out == [AnalystType.MARKET]
        # crypto_native is off in tests (conftest) -> fundamentals dropped
        assert AnalystType.FUNDAMENTALS not in captured["values"]

    def test_cancelled_or_empty_selection_exits_1(self, monkeypatch):
        def _select_with(answer):
            monkeypatch.setattr(
                cli_utils.questionary, "checkbox", lambda *a, **k: _ask(answer)
            )
            with pytest.raises(SystemExit) as exc:
                cli_utils.select_analysts(AssetType.STOCK, "AAPL")
            return exc.value.code

        assert _select_with(None) == 1
        assert _select_with([]) == 1


@pytest.mark.unit
class TestRound15SelectResearchDepthPrompt:
    def test_returns_selected_depth(self, monkeypatch):
        monkeypatch.setattr(cli_utils.questionary, "select", lambda *a, **k: _ask(5))
        assert cli_utils.select_research_depth() == 5

    def test_cancelled_depth_exits_1(self, monkeypatch):
        monkeypatch.setattr(cli_utils.questionary, "select", lambda *a, **k: _ask(None))
        with pytest.raises(SystemExit) as exc:
            cli_utils.select_research_depth()
        assert exc.value.code == 1


@pytest.mark.unit
class TestRound15FetchOpenrouterModelsFailure:
    def test_network_failure_returns_empty_list(self):
        with mock.patch("requests.get", side_effect=OSError("offline")), \
             mock.patch.object(cli_utils.console, "print"):
            assert cli_utils._fetch_openrouter_models() == []


@pytest.mark.unit
class TestRound15SelectModelDispatch:
    def test_openrouter_delegates_to_openrouter_picker(self, monkeypatch):
        picked = mock.Mock(return_value="openrouter/some-model")
        monkeypatch.setattr(cli_utils, "select_openrouter_model", picked)
        assert cli_utils._select_model("OpenRouter", "deep") == "openrouter/some-model"
        picked.assert_called_once_with("deep")

    def test_azure_prompts_for_deployment_name(self, monkeypatch):
        monkeypatch.setattr(
            cli_utils.questionary, "text", lambda *a, **k: _ask("my-deploy ")
        )
        out = cli_utils._select_model("azure", "quick")
        assert out == "my-deploy"

    def test_generic_provider_returns_catalog_choice(self, monkeypatch):
        monkeypatch.setattr(
            cli_utils.questionary, "select", lambda *a, **k: _ask("gpt-5.5")
        )
        assert cli_utils._select_model("openai", "quick") == "gpt-5.5"

    def test_generic_custom_choice_prompts_for_model_id(self, monkeypatch):
        monkeypatch.setattr(
            cli_utils.questionary, "select", lambda *a, **k: _ask("custom")
        )
        monkeypatch.setattr(
            cli_utils.questionary, "text", lambda *a, **k: _ask(" my-custom-model ")
        )
        assert cli_utils._select_model("qwen", "deep") == "my-custom-model"

    def test_generic_cancel_exits_1(self, monkeypatch):
        monkeypatch.setattr(
            cli_utils.questionary, "select", lambda *a, **k: _ask(None)
        )
        with pytest.raises(SystemExit) as exc:
            cli_utils._select_model("openai", "quick")
        assert exc.value.code == 1


@pytest.mark.unit
class TestRound15ThinkingAgentSelectors:
    def test_shallow_uses_quick_mode(self, monkeypatch):
        inner = mock.Mock(return_value="fast-model")
        monkeypatch.setattr(cli_utils, "_select_model", inner)
        assert cli_utils.select_shallow_thinking_agent("openai") == "fast-model"
        inner.assert_called_once_with("openai", "quick")

    def test_deep_uses_deep_mode(self, monkeypatch):
        inner = mock.Mock(return_value="smart-model")
        monkeypatch.setattr(cli_utils, "_select_model", inner)
        assert cli_utils.select_deep_thinking_agent("openai") == "smart-model"
        inner.assert_called_once_with("openai", "deep")


@pytest.mark.unit
class TestRound15PromptOpenaiCompatibleUrl:
    def test_returns_stripped_url(self, monkeypatch):
        monkeypatch.setattr(
            cli_utils.questionary,
            "text",
            lambda *a, **k: _ask(" http://localhost:8000/v1 "),
        )
        assert cli_utils.prompt_openai_compatible_url() == "http://localhost:8000/v1"

    def test_cancelled_url_exits_1(self, monkeypatch):
        monkeypatch.setattr(cli_utils.questionary, "text", lambda *a, **k: _ask(None))
        with pytest.raises(SystemExit) as exc:
            cli_utils.prompt_openai_compatible_url()
        assert exc.value.code == 1


@pytest.mark.unit
class TestRound15SelectLlmProviderPrompt:
    def test_returns_provider_and_url(self, monkeypatch):
        pair = ("deepseek", "https://api.deepseek.com")
        monkeypatch.setattr(cli_utils.questionary, "select", lambda *a, **k: _ask(pair))
        assert cli_utils.select_llm_provider() == pair

    def test_cancelled_provider_exits_1(self, monkeypatch):
        monkeypatch.setattr(cli_utils.questionary, "select", lambda *a, **k: _ask(None))
        with pytest.raises(SystemExit) as exc:
            cli_utils.select_llm_provider()
        assert exc.value.code == 1


@pytest.mark.unit
class TestRound15ProviderKnobPrompts:
    def test_openai_reasoning_effort_returns_choice(self, monkeypatch):
        monkeypatch.setattr(
            cli_utils.questionary, "select", lambda *a, **k: _ask("high")
        )
        assert cli_utils.ask_openai_reasoning_effort() == "high"

    def test_anthropic_effort_returns_choice(self, monkeypatch):
        monkeypatch.setattr(
            cli_utils.questionary, "select", lambda *a, **k: _ask("low")
        )
        assert cli_utils.ask_anthropic_effort() == "low"

    def test_gemini_thinking_returns_choice(self, monkeypatch):
        monkeypatch.setattr(
            cli_utils.questionary, "select", lambda *a, **k: _ask("minimal")
        )
        assert cli_utils.ask_gemini_thinking_config() == "minimal"


@pytest.mark.unit
class TestRound15EnsureApiKeyOptionalProviders:
    def test_key_optional_provider_returns_env_without_prompt(self, monkeypatch):
        prompt = mock.Mock()
        monkeypatch.setattr(cli_utils.questionary, "password", prompt)
        monkeypatch.setenv("OPENAI_COMPATIBLE_API_KEY", "sk-local-xyz")
        assert cli_utils.ensure_api_key("openai_compatible") == "sk-local-xyz"
        monkeypatch.delenv("OPENAI_COMPATIBLE_API_KEY")
        assert cli_utils.ensure_api_key("openai_compatible") is None
        prompt.assert_not_called()

    def test_unknown_provider_returns_none_without_prompt(self, monkeypatch):
        prompt = mock.Mock()
        monkeypatch.setattr(cli_utils.questionary, "password", prompt)
        assert cli_utils.ensure_api_key("not-a-provider") is None
        prompt.assert_not_called()


@pytest.mark.unit
class TestRound15AskOutputLanguageChoice:
    def test_plain_choice_returned_verbatim(self, monkeypatch):
        monkeypatch.setattr(
            cli_utils.questionary, "select", lambda *a, **k: _ask("Chinese")
        )
        assert cli_utils.ask_output_language() == "Chinese"

    def test_custom_language_text_returned(self, monkeypatch):
        monkeypatch.setattr(
            cli_utils.questionary, "select", lambda *a, **k: _ask("custom")
        )
        monkeypatch.setattr(
            cli_utils.questionary, "text", lambda *a, **k: _ask(" Turkish ")
        )
        assert cli_utils.ask_output_language() == "Turkish"


# ---------------------------------------------------------------------------
# cli.main — MessageBuffer low-level paths
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRound15MessageBufferLowLevel:
    def test_unknown_section_skipped_in_completed_count(self):
        buf = MessageBuffer()
        buf.init_for_analysis(["market"])
        buf.update_report_section("market_report", "content")
        buf.update_agent_status("Market Analyst", "completed")
        # A foreign key injected after init must be skipped, not KeyError.
        buf.report_sections["ghost_section"] = "spooky"
        assert buf.get_completed_reports_count() == 1

    def test_add_message_and_tool_call_record_timestamped_tuples(self):
        buf = MessageBuffer()
        buf.add_message("System", "hello")
        buf.add_tool_call("get_stock_data", {"ticker": "AAPL"})
        timestamp, msg_type, content = buf.messages[-1]
        assert timestamp and msg_type == "System" and content == "hello"
        ts2, tool_name, tool_args = buf.tool_calls[-1]
        assert ts2 and tool_name == "get_stock_data" and tool_args == {"ticker": "AAPL"}


@pytest.mark.unit
class TestRound15FinalReportAssembly:
    def test_every_section_renders_in_pinned_order(self):
        buf = MessageBuffer()
        buf.init_for_analysis(
            ["market", "positioning", "social", "news", "fundamentals"]
        )
        buf.update_report_section("market_report", "m")
        buf.update_report_section("positioning_report", "p")
        buf.update_report_section("sentiment_report", "s")
        buf.update_report_section("news_report", "n")
        buf.update_report_section("fundamentals_report", "f")
        buf.update_report_section("investment_plan", "ip")
        buf.update_report_section("trader_investment_plan", "tp")
        buf.update_report_section("final_trade_decision", "fd")

        report = buf.final_report
        assert report is not None
        # Positioning renders right after market (the pinned run order).
        order = [
            report.index("### Market Analysis"),
            report.index("### Positioning Analysis"),
            report.index("### Social Sentiment"),
            report.index("### News Analysis"),
            report.index("### Fundamentals Analysis"),
        ]
        assert order == sorted(order)
        assert "## Research Team Decision" in report
        assert "## Trading Team Plan" in report
        assert "## Portfolio Management Decision" in report

    def test_current_report_tracks_latest_section(self):
        buf = MessageBuffer()
        buf.init_for_analysis(["market", "news"])
        buf.update_report_section("market_report", "first")
        assert "Market Analysis" in buf.current_report
        buf.update_report_section("news_report", "second")
        assert "News Analysis" in buf.current_report

    def test_create_layout_builds_named_panes(self):
        layout = create_layout()
        for pane in ("header", "progress", "messages", "analysis", "footer"):
            assert layout[pane] is not None


# ---------------------------------------------------------------------------
# cli.main — get_user_selections (interactive branches, fully mocked)
# ---------------------------------------------------------------------------

_SELECTION_ENV_VARS = (
    "YIALPHA_OUTPUT_LANGUAGE",
    "YIALPHA_MAX_DEBATE_ROUNDS",
    "YIALPHA_MAX_RISK_ROUNDS",
    "YIALPHA_LLM_PROVIDER",
    "YIALPHA_LLM_BACKEND_URL",
    "YIALPHA_QUICK_THINK_LLM",
    "YIALPHA_DEEP_THINK_LLM",
    "YIALPHA_GOOGLE_THINKING_LEVEL",
    "YIALPHA_OPENAI_REASONING_EFFORT",
    "YIALPHA_ANTHROPIC_EFFORT",
)


def _mock_selection_step(monkeypatch, **overrides):
    """Install the standard get_user_selections mock set; return the mocks.

    Every prompt main.get_user_selections can reach is replaced by a Mock so
    the function terminates headlessly; individual tests override the pieces
    whose behavior they pin. DEFAULT_CONFIG is swapped for a copy with
    backend_url=None so a developer-box YIALPHA_LLM_BACKEND_URL baked into the
    real defaults cannot leak into the resolution-under-test.
    """
    for var in _SELECTION_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    mocks = {
        "get_ticker": mock.Mock(return_value="AAPL"),
        "get_analysis_date": mock.Mock(return_value="2026-01-10"),
        "ask_output_language": mock.Mock(return_value="English"),
        "select_analysts": mock.Mock(return_value=[]),
        "select_research_depth": mock.Mock(return_value=1),
        "select_llm_provider": mock.Mock(return_value=("openai", None)),
        "ask_qwen_region": mock.Mock(return_value=("qwen", "intl-url")),
        "ask_minimax_region": mock.Mock(return_value=("minimax", "global-url")),
        "ask_glm_region": mock.Mock(return_value=("glm", "zai-url")),
        "prompt_openai_compatible_url": mock.Mock(return_value="http://vllm:8000/v1"),
        "confirm_ollama_endpoint": mock.Mock(),
        "ensure_api_key": mock.Mock(return_value=None),
        "select_shallow_thinking_agent": mock.Mock(return_value="quick-model"),
        "select_deep_thinking_agent": mock.Mock(return_value="deep-model"),
        "ask_gemini_thinking_config": mock.Mock(return_value="high"),
        "ask_openai_reasoning_effort": mock.Mock(return_value="medium"),
        "ask_anthropic_effort": mock.Mock(return_value="low"),
    }
    mocks.update(overrides)
    for name, value in mocks.items():
        monkeypatch.setattr(cli_main, name, value)
    fake_cfg = dict(cli_main.DEFAULT_CONFIG)
    fake_cfg["backend_url"] = None
    monkeypatch.setattr(cli_main, "DEFAULT_CONFIG", fake_cfg)
    return mocks


@pytest.mark.unit
class TestRound15UserSelectionsInteractive:
    def test_interactive_output_language_prompt_used(self, monkeypatch):
        mocks = _mock_selection_step(monkeypatch)
        sel = cli_main.get_user_selections()
        mocks["ask_output_language"].assert_called_once()
        assert sel["output_language"] == "English"

    def test_research_depth_prompt_used_when_env_absent(self, monkeypatch):
        mocks = _mock_selection_step(monkeypatch)
        sel = cli_main.get_user_selections()
        mocks["select_research_depth"].assert_called_once()
        assert sel["research_depth"] == 1

    def test_qwen_region_prompt_switches_provider_and_url(self, monkeypatch):
        mocks = _mock_selection_step(
            monkeypatch,
            select_llm_provider=mock.Mock(return_value=("qwen", "intl-url")),
            ask_qwen_region=mock.Mock(return_value=("qwen-cn", "https://cn.example/v1")),
        )
        sel = cli_main.get_user_selections()
        mocks["ask_qwen_region"].assert_called_once()
        mocks["ensure_api_key"].assert_called_once_with("qwen-cn")
        assert sel["llm_provider"] == "qwen-cn"
        assert sel["backend_url"] == "https://cn.example/v1"

    def test_minimax_region_prompt_switches_provider(self, monkeypatch):
        mocks = _mock_selection_step(
            monkeypatch,
            select_llm_provider=mock.Mock(return_value=("minimax", "global-url")),
            ask_minimax_region=mock.Mock(
                return_value=("minimax-cn", "https://cn.minimaxi/v1")
            ),
        )
        sel = cli_main.get_user_selections()
        mocks["ensure_api_key"].assert_called_once_with("minimax-cn")
        assert sel["llm_provider"] == "minimax-cn"

    def test_glm_region_prompt_switches_provider(self, monkeypatch):
        mocks = _mock_selection_step(
            monkeypatch,
            select_llm_provider=mock.Mock(return_value=("glm", None)),
            ask_glm_region=mock.Mock(
                return_value=("glm-cn", "https://open.bigmodel.cn/api/paas/v4/")
            ),
        )
        sel = cli_main.get_user_selections()
        mocks["ensure_api_key"].assert_called_once_with("glm-cn")
        assert sel["llm_provider"] == "glm-cn"

    def test_openai_compatible_prompts_for_base_url(self, monkeypatch):
        mocks = _mock_selection_step(
            monkeypatch,
            select_llm_provider=mock.Mock(return_value=("openai_compatible", None)),
        )
        sel = cli_main.get_user_selections()
        mocks["prompt_openai_compatible_url"].assert_called_once()
        assert sel["backend_url"] == "http://vllm:8000/v1"

    def test_ollama_confirms_resolved_endpoint(self, monkeypatch):
        mocks = _mock_selection_step(
            monkeypatch,
            select_llm_provider=mock.Mock(
                return_value=("ollama", "http://localhost:11434/v1")
            ),
        )
        sel = cli_main.get_user_selections()
        mocks["confirm_ollama_endpoint"].assert_called_once_with(
            "http://localhost:11434/v1"
        )
        mocks["ensure_api_key"].assert_called_once_with("ollama")
        assert sel["backend_url"] == "http://localhost:11434/v1"

    def test_google_step8_prompts_thinking_mode(self, monkeypatch):
        mocks = _mock_selection_step(
            monkeypatch,
            select_llm_provider=mock.Mock(return_value=("google", None)),
        )
        sel = cli_main.get_user_selections()
        mocks["ask_gemini_thinking_config"].assert_called_once()
        assert sel["google_thinking_level"] == "high"
        assert sel["openai_reasoning_effort"] is None
        assert sel["anthropic_effort"] is None

    def test_openai_step8_prompts_reasoning_effort(self, monkeypatch):
        mocks = _mock_selection_step(monkeypatch)
        sel = cli_main.get_user_selections()  # default provider is openai
        mocks["ask_openai_reasoning_effort"].assert_called_once()
        assert sel["openai_reasoning_effort"] == "medium"

    def test_anthropic_step8_prompts_effort(self, monkeypatch):
        mocks = _mock_selection_step(
            monkeypatch,
            select_llm_provider=mock.Mock(return_value=("anthropic", None)),
        )
        sel = cli_main.get_user_selections()
        mocks["ask_anthropic_effort"].assert_called_once()
        assert sel["anthropic_effort"] == "low"

    def test_google_step8_env_skips_prompt(self, monkeypatch):
        mocks = _mock_selection_step(
            monkeypatch,
            select_llm_provider=mock.Mock(return_value=("google", None)),
        )
        # Set AFTER _mock_selection_step: the helper clears every YIALPHA_*
        # selection env var to keep the interactive branch deterministic.
        monkeypatch.setenv("YIALPHA_GOOGLE_THINKING_LEVEL", "high")
        fake_cfg = dict(cli_main.DEFAULT_CONFIG)
        fake_cfg["google_thinking_level"] = "high"
        monkeypatch.setattr(cli_main, "DEFAULT_CONFIG", fake_cfg)
        sel = cli_main.get_user_selections()
        mocks["ask_gemini_thinking_config"].assert_not_called()
        assert sel["google_thinking_level"] == "high"

    def test_detected_crypto_asset_type_flows_through(self, monkeypatch):
        mocks = _mock_selection_step(
            monkeypatch, get_ticker=mock.Mock(return_value="BTC-USD")
        )
        sel = cli_main.get_user_selections()
        mocks["select_analysts"].assert_called_once_with(AssetType.CRYPTO, "BTC-USD")
        assert sel["asset_type"] == "crypto"
        assert sel["ticker"] == "BTC-USD"


@pytest.mark.unit
class TestRound15AssetTypeOverridePaths:
    def test_invalid_override_exits_2_with_valid_values_listed(self, monkeypatch):
        _mock_selection_step(monkeypatch)
        cli_main.console.begin_capture()
        with pytest.raises(typer.Exit) as exc:
            cli_main.get_user_selections(asset_type_override="banana")
        out = cli_main.console.end_capture()
        assert exc.value.exit_code == 2
        assert "Invalid --asset-type 'banana'" in out
        assert "crypto_perp" in out  # the valid values are enumerated

    def test_perp_override_renames_ticker_to_venue_symbol(self, monkeypatch):
        warm = mock.Mock()
        monkeypatch.setattr(
            "yialpha.dataflows.binance.warm_equity_perp_bases", warm
        )
        _mock_selection_step(
            monkeypatch,
            get_ticker=mock.Mock(return_value="BTC-USD"),
            select_analysts=mock.Mock(return_value=[AnalystType.MARKET]),
        )
        cli_main.console.begin_capture()
        sel = cli_main.get_user_selections(asset_type_override="crypto_perp")
        out = cli_main.console.end_capture()
        warm.assert_called_once()
        # The Binance-venue contract symbol keys every ledger identity.
        assert sel["ticker"] == "BTCUSDT"
        assert sel["asset_type"] == "crypto_perp"
        assert "Venue symbol" in out

    def test_degenerate_venue_symbol_exits_2(self, monkeypatch):
        _mock_selection_step(
            monkeypatch, get_ticker=mock.Mock(return_value="USDT")
        )
        cli_main.console.begin_capture()
        with pytest.raises(typer.Exit) as exc:
            cli_main.get_user_selections(asset_type_override="crypto_perp")
        out = cli_main.console.end_capture()
        assert exc.value.exit_code == 2
        assert "degenerate" in out


# ---------------------------------------------------------------------------
# cli.main — report persistence + display
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRound15ReportPersistenceAndDisplay:
    def test_save_report_to_disk_writes_full_tree(self, tmp_path):
        target = tmp_path / "reports" / "AAPL_run"
        path = save_report_to_disk(_round15_full_state(), "AAPL", target)
        assert path == target / "complete_report.md"
        assert path.exists()
        assert (target / "1_analysts" / "market.md").exists()
        assert (target / "1_analysts" / "positioning.md").exists()
        assert (target / "2_research" / "bull.md").exists()
        assert (target / "3_trading" / "trader.md").exists()
        assert (target / "4_risk" / "aggressive.md").exists()
        assert (target / "5_portfolio" / "decision.md").exists()
        assert "Rating: BUY" in path.read_text(encoding="utf-8")

    def test_display_complete_report_renders_all_five_teams(self):
        cli_main.console.begin_capture()
        display_complete_report(_round15_full_state())
        out = cli_main.console.end_capture()
        assert "Complete Analysis Report" in out
        assert "I. Analyst Team Reports" in out
        assert "Positioning Analyst" in out
        assert "II. Research Team Decision" in out
        assert "III. Trading Team Plan" in out
        assert "IV. Risk Management Team Decision" in out
        assert "V. Portfolio Manager Decision" in out

    def test_display_complete_report_empty_state_prints_only_rule(self):
        cli_main.console.begin_capture()
        display_complete_report({})
        out = cli_main.console.end_capture()
        assert "Complete Analysis Report" in out
        assert "I. Analyst Team Reports" not in out


# ---------------------------------------------------------------------------
# cli.main — analyst status sync + misc pure helpers
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRound15AnalystStatusTrackerSync:
    def test_wall_time_tracker_synced_from_chunk(self):
        plan = build_analyst_execution_plan(["market", "news"])
        tracker = AnalystWallTimeTracker(plan)
        buf = MessageBuffer()
        buf.init_for_analysis(["market", "news"])
        update_analyst_statuses(buf, {"market_report": "done"}, wall_time_tracker=tracker)
        assert "market" in tracker.get_wall_times()
        assert buf.agent_status["Market Analyst"] == "completed"
        assert buf.agent_status["News Analyst"] == "in_progress"


@pytest.mark.unit
class TestRound15ExtractContentScalarFallback:
    def test_numeric_content_stringified(self):
        assert extract_content_string(42) == "42"
        assert extract_content_string(3.14) == "3.14"


@pytest.mark.unit
class TestRound15BatchWorkerOverride:
    def test_none_preserves_configured_switch(self):
        cfg = {"batch_concurrency": False}
        assert _apply_batch_worker_override(cfg, None) is cfg
        assert cfg["batch_concurrency"] is False

    def test_explicit_one_forces_serial(self):
        cfg = {"batch_concurrency": True}
        _apply_batch_worker_override(cfg, 1)
        assert cfg["batch_concurrency"] is False

    def test_explicit_pool_forces_concurrency(self):
        cfg = {"batch_concurrency": False}
        _apply_batch_worker_override(cfg, 4)
        assert cfg["batch_concurrency"] is True

    def test_zero_raises_value_error(self):
        with pytest.raises(ValueError, match="at least 1"):
            _apply_batch_worker_override({"batch_concurrency": False}, 0)


@pytest.mark.unit
class TestRound15StoreDecisionFailSoft:
    def test_store_decision_io_failure_warns_not_raises(self, caplog):
        def _boom(**kwargs):
            raise OSError("disk full")

        graph = SimpleNamespace(
            memory_log=SimpleNamespace(store_decision=_boom)
        )
        with caplog.at_level("WARNING", logger="yialpha.cli.main"):
            _store_cli_decision(
                graph, "NVDA", "2026-06-10", {"final_trade_decision": "BUY"},
                asset_type="crypto_perp",
            )
        assert any("memory store_decision failed" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# cli.main — analyze command entry (non-interactive plumbing only)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRound15AnalyzeCommandEntry:
    def test_clear_checkpoints_reports_count_and_runs(self, monkeypatch):
        cleared = mock.Mock(return_value=3)
        monkeypatch.setattr(
            "yialpha.graph.checkpointer.clear_all_checkpoints", cleared
        )
        run = mock.Mock()
        monkeypatch.setattr(cli_main, "run_analysis", run)
        monkeypatch.setattr(cli_main, "_warn_config_drift", mock.Mock())

        result = runner.invoke(app, ["analyze", "--clear-checkpoints"])
        assert result.exit_code == 0, result.output
        cleared.assert_called_once()
        assert "Cleared 3 checkpoint(s)" in result.output
        run.assert_called_once_with(checkpoint=None, asset_type="auto")

    def test_default_invocation_skips_clear_and_checks_drift(self, monkeypatch):
        cleared = mock.Mock(return_value=0)
        monkeypatch.setattr(
            "yialpha.graph.checkpointer.clear_all_checkpoints", cleared
        )
        run = mock.Mock()
        monkeypatch.setattr(cli_main, "run_analysis", run)
        drift = mock.Mock()
        monkeypatch.setattr(cli_main, "_warn_config_drift", drift)

        result = runner.invoke(app, ["analyze"])
        assert result.exit_code == 0, result.output
        cleared.assert_not_called()
        drift.assert_called_once()
        run.assert_called_once_with(checkpoint=None, asset_type="auto")

    def test_asset_type_flag_forwarded(self, monkeypatch):
        monkeypatch.setattr(
            "yialpha.graph.checkpointer.clear_all_checkpoints", mock.Mock()
        )
        run = mock.Mock()
        monkeypatch.setattr(cli_main, "run_analysis", run)
        monkeypatch.setattr(cli_main, "_warn_config_drift", mock.Mock())
        result = runner.invoke(app, ["analyze", "--asset-type", "crypto_perp"])
        assert result.exit_code == 0, result.output
        run.assert_called_once_with(checkpoint=None, asset_type="crypto_perp")


# ---------------------------------------------------------------------------
# cli.main — batch command happy path (runner mocked at its import seam)
# ---------------------------------------------------------------------------


class _Round15FakeBatchRunner:
    """Stands in for BatchRunner; records ctor args, serves canned results."""

    instances: list[_Round15FakeBatchRunner] = []

    def __init__(self, config, workers=None, progress=False, selected_analysts=None):
        self.config = config
        self.workers = workers
        self.selected_analysts = selected_analysts
        # Canned per-invocation results staged on the class by the test.
        self.results = list(_Round15FakeBatchRunner.results)
        _Round15FakeBatchRunner.instances.append(self)

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def run(self, tickers, date, asset_type=None):
        return self.results


@pytest.mark.unit
class TestRound15BatchCommandRun:
    def _patch_runner_module(self, monkeypatch, results, resolved="stock"):
        import yialpha.batch.runner as runner_mod

        _Round15FakeBatchRunner.instances = []
        _Round15FakeBatchRunner.results = results
        monkeypatch.setattr(
            runner_mod,
            "prepare_batch_run",
            mock.Mock(return_value=({"results_dir": "unused"}, resolved)),
        )
        monkeypatch.setattr(
            runner_mod,
            "batch_selected_analysts",
            mock.Mock(return_value=("market", "social", "news")),
        )
        monkeypatch.setattr(runner_mod, "BatchRunner", _Round15FakeBatchRunner)

    def test_all_ok_reports_inferred_type_and_exits_zero(self, monkeypatch):
        self._patch_runner_module(
            monkeypatch,
            [
                {
                    "ticker": "AAPL",
                    "error": None,
                    "elapsed": 1.25,
                    "report_path": Path("results/AAPL/report.md"),
                }
            ],
        )
        result = runner.invoke(app, ["batch", "-t", "AAPL", "-d", "2026-01-10"])
        assert result.exit_code == 0, result.output
        assert "Asset type inferred: stock (from AAPL)" in result.output
        assert "Batch results (1/1 ok)" in result.output
        assert "AAPL" in result.output

    def test_any_failure_exits_one_with_error_row(self, monkeypatch):
        self._patch_runner_module(
            monkeypatch,
            [
                {
                    "ticker": "AAPL",
                    "error": None,
                    "elapsed": 1.0,
                    "report_path": Path("results/AAPL/report.md"),
                },
                {
                    "ticker": "MSFT",
                    "error": ValueError("boom"),
                    "elapsed": 0.0,
                    "report_path": None,
                },
            ],
        )
        result = runner.invoke(app, ["batch", "-t", "AAPL", "-t", "MSFT", "-d", "2026-01-10"])
        assert result.exit_code == 1
        assert "Batch results (1/2 ok)" in result.output
        assert "ValueError: boom" in result.output

    def test_explicit_asset_type_skips_inference_print(self, monkeypatch):
        self._patch_runner_module(
            monkeypatch,
            [{"ticker": "BTCUSDT", "error": None, "elapsed": 2.0, "report_path": None}],
            resolved="crypto_perp",
        )
        result = runner.invoke(
            app,
            ["batch", "-t", "BTCUSDT", "-d", "2026-01-10", "--asset-type", "crypto_perp"],
        )
        assert result.exit_code == 0, result.output
        assert "Asset type inferred" not in result.output
        assert "type=crypto_perp" in result.output


# ---------------------------------------------------------------------------
# cli.main — verify-history / memory-resolve / ic-cycle command bodies
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRound15VerifyHistoryCommand:
    def _report(self, hit_rate, hits, n):
        return {
            "direction": {"hit_rate": hit_rate, "hits": hits, "n": n},
            "holding_days": 5,
            "total_runs": 10,
            "scored": 4,
            "pending": 6,
        }

    def test_hit_rate_branch_prints_stats_and_paths(self, monkeypatch, tmp_path):
        # Rich folds console output at 80 columns when no terminal is attached, and
        # WHERE it folds depends on the length of tmp_path: with a temp directory one
        # to nineteen characters longer than this box's, the file name came out split
        # over two lines ("accuracy_report.jso" / "n") -- a GitHub ubuntu runner's
        # /tmp/pytest-of-runner/pytest-0 is in that range. Keep the paths on one line.
        monkeypatch.setenv("COLUMNS", "400")
        import yialpha.accuracy as accuracy_mod

        fake = mock.Mock(
            return_value=(
                self._report(0.75, 3, 4),
                tmp_path / "accuracy_report.json",
                tmp_path / "accuracy_report.md",
            )
        )
        monkeypatch.setattr(accuracy_mod, "verify_history", fake)
        result = runner.invoke(app, ["verify-history", "--holding-days", "5"])
        assert result.exit_code == 0, result.output
        fake.assert_called_once_with(None, holding_days=5)
        assert "75.0%" in result.output
        assert "3/4" in result.output
        assert "10 scanned | 4 scored | 6 pending" in result.output
        assert "accuracy_report.json" in result.output

    def test_paths_stay_whole_in_a_narrow_console(self, monkeypatch, tmp_path):
        # Round-17 follow-up: the command used to fold a long path at the console
        # width, so in a narrow terminal the file name itself came out split
        # ("accuracy_report.jso" / "n"). Paths are printed with soft_wrap now: one
        # line whatever the width (the terminal wraps it visually).
        monkeypatch.setenv("COLUMNS", "40")
        import yialpha.accuracy as accuracy_mod

        json_path = tmp_path / "accuracy_report.json"
        md_path = tmp_path / "accuracy_report.md"
        fake = mock.Mock(
            return_value=(self._report(0.75, 3, 4), json_path, md_path)
        )
        monkeypatch.setattr(accuracy_mod, "verify_history", fake)
        result = runner.invoke(app, ["verify-history", "--holding-days", "5"])
        assert result.exit_code == 0, result.output
        assert f"JSON: {json_path}" in result.output
        assert f"Markdown: {md_path}" in result.output

    def test_no_elapsed_decisions_branch(self, monkeypatch, tmp_path):
        import yialpha.accuracy as accuracy_mod

        monkeypatch.setattr(
            accuracy_mod,
            "verify_history",
            mock.Mock(
                return_value=(
                    self._report(None, 0, 0),
                    tmp_path / "accuracy_report.json",
                    tmp_path / "accuracy_report.md",
                )
            ),
        )
        result = runner.invoke(
            app, ["verify-history", "--results-dir", str(tmp_path)]
        )
        assert result.exit_code == 0, result.output
        assert "No fully-elapsed directional decisions yet." in result.output


@pytest.mark.unit
class TestRound15MemoryResolveGates:
    def test_no_pending_entries_prints_notice_and_exits_zero(self, monkeypatch, tmp_path):
        monkeypatch.setitem(
            cli_main.DEFAULT_CONFIG,
            "memory_log_path",
            str(tmp_path / "nested" / "trading_memory.md"),
        )
        result = runner.invoke(app, ["memory-resolve"])
        assert result.exit_code == 0, result.output
        assert "No pending memory entries." in result.output

    def test_missing_log_file_with_pending_entries_exits_two(self, monkeypatch, tmp_path):
        class _FakeMemoryLog:
            def __init__(self, config):
                self._log_path = tmp_path / "does-not-exist.md"

            def get_pending_entries(self):
                return [{"ticker": "AAPL"}]

        monkeypatch.setattr(
            "yialpha.agents.utils.memory.TradingMemoryLog", _FakeMemoryLog
        )
        result = runner.invoke(app, ["memory-resolve"])
        assert result.exit_code == 2
        assert "Memory log file not found" in result.output


def _round15_ic_result(prune: bool) -> dict:
    keep = ["rsi", "atr", "macd"]
    pruned = ["stoch"] if prune else []
    per_indicator = {
        "rsi": {"verdict": "keep", "mean_abs_ic": 0.12, "finite_windows": 40},
        "atr": {"verdict": "keep", "mean_abs_ic": 0.08, "finite_windows": 41},
        "macd": {"verdict": "keep", "mean_abs_ic": 0.05, "finite_windows": 42},
    }
    if prune:
        per_indicator["stoch"] = {"verdict": "prune", "mean_abs_ic": None, "finite_windows": 5}
    verdict = {"keep": keep, "prune": pruned, "per_indicator": per_indicator}
    return {
        "csv": {"NVDA": "ic_data/NVDA_5d.csv", "AMD": "ic_data/AMD_5d.csv"},
        "verdict": {"NVDA": verdict, "AMD": verdict},
    }


@pytest.mark.unit
class TestRound15IcCycleCommand:
    def test_runtime_error_exits_one(self, monkeypatch):
        import yialpha.backtest.ic_dataset as ic_mod

        monkeypatch.setattr(
            ic_mod, "run_ic_cycle", mock.Mock(side_effect=RuntimeError("no dataset"))
        )
        result = runner.invoke(app, ["ic-cycle"])
        assert result.exit_code == 1
        assert "no dataset" in result.output

    def test_prune_suggestion_prints_intersection_and_review_warning(self, monkeypatch):
        import yialpha.backtest.ic_dataset as ic_mod

        monkeypatch.setattr(
            ic_mod, "run_ic_cycle", mock.Mock(return_value=_round15_ic_result(prune=True))
        )
        result = runner.invoke(app, ["ic-cycle"])
        assert result.exit_code == 0, result.output
        assert "NVDA — verdict" in result.output
        assert "stoch" in result.output
        assert "1 indicator(s) suggested for pruning" in result.output
        assert "indicator_battery (intersection across 2 tickers)" in result.output
        assert "REVIEW BEFORE APPLYING" in result.output
        assert "snapshot record" in result.output

    def test_no_prune_reports_battery_unchanged(self, monkeypatch):
        import yialpha.backtest.ic_dataset as ic_mod

        monkeypatch.setattr(
            ic_mod, "run_ic_cycle", mock.Mock(return_value=_round15_ic_result(prune=False))
        )
        result = runner.invoke(app, ["ic-cycle"])
        assert result.exit_code == 0, result.output
        assert "No indicator meets the prune thresholds" in result.output
        assert "REVIEW BEFORE APPLYING" not in result.output


# ---------------------------------------------------------------------------
# cli.main — config-check verdicts beyond the covered provider-key paths
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRound15ConfigCheckPools:
    def test_glm_key_pool_branch(self, monkeypatch):
        monkeypatch.setenv("YIALPHA_LLM_PROVIDER", "glm")
        # The pool merges the single-key var in; drop it so the count is exact.
        monkeypatch.delenv("ZHIPU_API_KEY", raising=False)
        monkeypatch.setenv("ZHIPU_API_KEYS", "key-a,key-b")
        result = runner.invoke(app, ["config-check"])
        assert result.exit_code == 0, result.output
        assert "pool with 2 keys" in result.output
        assert "Configuration ready" in result.output


@pytest.mark.unit
class TestRound15ConfigCheckTavily:
    def test_enabled_without_key_warns(self, monkeypatch):
        monkeypatch.delenv("TAVILY_API_KEY", raising=False)
        monkeypatch.delenv("TAVILY_API_KEYS", raising=False)
        monkeypatch.setenv("YIALPHA_LLM_PROVIDER", "deepseek")
        result = runner.invoke(app, ["config-check"])
        assert result.exit_code == 0, result.output
        assert "web_search_enabled is on but no Tavily key" in result.output

    def test_multi_key_pool_advertised(self, monkeypatch):
        monkeypatch.delenv("TAVILY_API_KEY", raising=False)
        monkeypatch.setenv("TAVILY_API_KEYS", "t1,t2")
        monkeypatch.setenv("YIALPHA_LLM_PROVIDER", "deepseek")
        result = runner.invoke(app, ["config-check"])
        assert result.exit_code == 0, result.output
        assert "Tavily key pool: 2 keys" in result.output

    def test_valid_budget_split_acknowledged(self, monkeypatch):
        monkeypatch.setenv("YIALPHA_TAVILY_BUDGET_SPLIT", "news:3,market:2")
        monkeypatch.setenv("YIALPHA_LLM_PROVIDER", "deepseek")
        result = runner.invoke(app, ["config-check"])
        assert result.exit_code == 0, result.output
        assert "news:3,market:2" in result.output
        assert "malformed" not in result.output

    def test_malformed_budget_split_warns(self, monkeypatch):
        monkeypatch.setenv("YIALPHA_TAVILY_BUDGET_SPLIT", "banana")
        monkeypatch.setenv("YIALPHA_LLM_PROVIDER", "deepseek")
        result = runner.invoke(app, ["config-check"])
        assert result.exit_code == 0, result.output
        assert "is malformed" in result.output
        assert "default split" in result.output


@pytest.mark.unit
class TestRound15ConfigCheckDataGate:
    def test_invalid_vacuum_policy_flagged_not_gating(self, monkeypatch):
        from yialpha.dataflows.config import set_config

        set_config({"data_vacuum_policy": "banana"})
        monkeypatch.setenv("YIALPHA_LLM_PROVIDER", "deepseek")
        result = runner.invoke(app, ["config-check"])
        assert result.exit_code == 0, result.output
        assert "data_vacuum_policy='banana' is invalid" in result.output
        assert "fails closed to 'reject'" in result.output

    def test_alpha_vantage_chain_leg_warns_when_key_unset(self, monkeypatch):
        monkeypatch.delenv("ALPHA_VANTAGE_API_KEY", raising=False)
        result = runner.invoke(app, ["config-check"])
        assert result.exit_code == 0, result.output
        assert "vendor chain includes alpha_vantage" in result.output
        assert "ALPHA_VANTAGE_API_KEY is unset" in result.output

    def test_single_vendor_config_reports_no_chains(self, monkeypatch):
        from yialpha.dataflows.config import set_config

        set_config(
            {
                "data_vendors": {
                    "core_stock_apis": "yfinance",
                    "technical_indicators": "yfinance",
                    "fundamental_data": "sec_edgar",
                    "news_data": "yfinance",
                    "macro_data": "fred",
                    "prediction_markets": "polymarket",
                }
            }
        )
        result = runner.invoke(app, ["config-check"])
        assert result.exit_code == 0, result.output
        assert "no multi-vendor chains configured" in result.output


@pytest.mark.unit
class TestRound15ConfigCheckProxy:
    def test_no_proxy_reports_direct_connection(self, monkeypatch):
        monkeypatch.delenv("SOCKS5_PROXY", raising=False)
        monkeypatch.delenv("ALL_PROXY", raising=False)
        result = runner.invoke(app, ["config-check"])
        assert result.exit_code == 0, result.output
        assert "No proxy configured (direct connection)" in result.output

    def test_proxy_creds_stripped_to_host(self, monkeypatch):
        monkeypatch.setenv("ALL_PROXY", "socks5://user:secret@proxyhost:1080")
        result = runner.invoke(app, ["config-check"])
        assert result.exit_code == 0, result.output
        assert "Proxy configured: proxyhost:1080" in result.output
        assert "secret" not in result.output
