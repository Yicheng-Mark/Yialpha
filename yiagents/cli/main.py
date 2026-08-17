import contextlib
import datetime
import logging
import os
import sys
import time
from collections import deque
from functools import wraps
from pathlib import Path

# Windows 控制台默认 GBK(cp936)，打印 ✅/❌ 等 Unicode 会触发 UnicodeEncodeError；
# 强制标准输出/错误流用 utf-8（Python 3.7+），让所有模式的中文与符号都能正常显示。
# 修 batch 命令打印 Rich 结果表（含 ✅/❌）时的 GBK 崩溃，对齐 scripts/run_baseline.py。
for _stream in (sys.stdout, sys.stderr):
    with contextlib.suppress(AttributeError, ValueError):
        reconfigure = _stream.reconfigure  # type: ignore[union-attr]
        reconfigure(encoding="utf-8", errors="replace")

# noqa: E402 — all imports sit after the UTF-8 reconfigure guard above, which must
# run first so ✅/❌/中文 render on a GBK Windows console.  The guard is a platform
# shim, not a code-ordering bug; reordering would re-introduce UnicodeEncodeError.
from yiagents.logging_config import setup_logging  # noqa: E402

setup_logging()

import typer  # noqa: E402
from rich import box  # noqa: E402
from rich.align import Align  # noqa: E402
from rich.console import Console, Group  # noqa: E402
from rich.layout import Layout  # noqa: E402
from rich.live import Live  # noqa: E402
from rich.markdown import Markdown  # noqa: E402
from rich.panel import Panel  # noqa: E402
from rich.rule import Rule  # noqa: E402
from rich.spinner import Spinner  # noqa: E402
from rich.table import Table  # noqa: E402
from rich.text import Text  # noqa: E402

from yiagents.dataflows import quality  # noqa: E402
from yiagents.default_config import DEFAULT_CONFIG  # noqa: E402
from yiagents.graph.analyst_execution import (  # noqa: E402
    AnalystWallTimeTracker,
    build_analyst_execution_plan,
    get_initial_analyst_node,
    sync_analyst_tracker_from_chunk,
)
from yiagents.graph.trading_graph import YiAgentsGraph  # noqa: E402
from yiagents.reporting import write_report_tree  # noqa: E402

from .stats_handler import StatsCallbackHandler  # noqa: E402
from .utils import (  # noqa: E402
    ask_anthropic_effort,
    ask_gemini_thinking_config,
    ask_glm_region,
    ask_minimax_region,
    ask_openai_reasoning_effort,
    ask_output_language,
    ask_qwen_region,
    confirm_ollama_endpoint,
    detect_asset_type,
    ensure_api_key,
    get_ticker,
    prompt_openai_compatible_url,
    resolve_backend_url,
    select_analysts,
    select_deep_thinking_agent,
    select_llm_provider,
    select_research_depth,
    select_shallow_thinking_agent,
)

console = Console()

logger = logging.getLogger(__name__)

app = typer.Typer(
    name="YiAgents",
    help="YiAgents CLI: Multi-Agent LLM Financial Analysis Framework",
    add_completion=True,  # Enable shell completion
)


# Create a deque to store recent messages with a maximum length
class MessageBuffer:
    # Fixed teams that always run (not user-selectable)
    FIXED_AGENTS = {
        "Research Team": ["Bull Researcher", "Bear Researcher", "Research Manager"],
        "Trading Team": ["Trader"],
        "Risk Management": ["Aggressive Analyst", "Neutral Analyst", "Conservative Analyst"],
        "Portfolio Management": ["Portfolio Manager"],
    }

    # Analyst name mapping
    ANALYST_MAPPING = {
        "market": "Market Analyst",
        "social": "Sentiment Analyst",
        "news": "News Analyst",
        "fundamentals": "Fundamentals Analyst",
    }

    # Report section mapping: section -> (analyst_key for filtering, finalizing_agent)
    # analyst_key: which analyst selection controls this section (None = always included)
    # finalizing_agent: which agent must be "completed" for this report to count as done
    REPORT_SECTIONS = {
        "market_report": ("market", "Market Analyst"),
        "sentiment_report": ("social", "Sentiment Analyst"),
        "news_report": ("news", "News Analyst"),
        "fundamentals_report": ("fundamentals", "Fundamentals Analyst"),
        "investment_plan": (None, "Research Manager"),
        "trader_investment_plan": (None, "Trader"),
        "final_trade_decision": (None, "Portfolio Manager"),
    }

    def __init__(self, max_length=100):
        self.messages = deque(maxlen=max_length)
        self.tool_calls = deque(maxlen=max_length)
        self.current_report = None
        self.final_report = None  # Store the complete final report
        self.agent_status = {}
        self.current_agent = None
        self.report_sections = {}
        self.selected_analysts = []
        self._processed_message_ids = set()

    def init_for_analysis(self, selected_analysts):
        """Initialize agent status and report sections based on selected analysts.

        Args:
            selected_analysts: List of analyst type strings (e.g., ["market", "news"])
        """
        self.selected_analysts = [a.lower() for a in selected_analysts]

        # Build agent_status dynamically
        self.agent_status = {}

        # Add selected analysts
        for analyst_key in self.selected_analysts:
            if analyst_key in self.ANALYST_MAPPING:
                self.agent_status[self.ANALYST_MAPPING[analyst_key]] = "pending"

        # Add fixed teams
        for team_agents in self.FIXED_AGENTS.values():
            for agent in team_agents:
                self.agent_status[agent] = "pending"

        # Build report_sections dynamically
        self.report_sections = {}
        for section, (analyst_key, _) in self.REPORT_SECTIONS.items():
            if analyst_key is None or analyst_key in self.selected_analysts:
                self.report_sections[section] = None

        # Reset other state
        self.current_report = None
        self.final_report = None
        self.current_agent = None
        self.messages.clear()
        self.tool_calls.clear()
        self._processed_message_ids.clear()

    def get_completed_reports_count(self):
        """Count reports that are finalized (their finalizing agent is completed).

        A report is considered complete when:
        1. The report section has content (not None), AND
        2. The agent responsible for finalizing that report has status "completed"

        This prevents interim updates (like debate rounds) from counting as completed.
        """
        count = 0
        for section in self.report_sections:
            if section not in self.REPORT_SECTIONS:
                continue
            _, finalizing_agent = self.REPORT_SECTIONS[section]
            # Report is complete if it has content AND its finalizing agent is done
            has_content = self.report_sections.get(section) is not None
            agent_done = self.agent_status.get(finalizing_agent) == "completed"
            if has_content and agent_done:
                count += 1
        return count

    def add_message(self, message_type, content):
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        self.messages.append((timestamp, message_type, content))

    def add_tool_call(self, tool_name, args):
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        self.tool_calls.append((timestamp, tool_name, args))

    def update_agent_status(self, agent, status):
        if agent in self.agent_status:
            self.agent_status[agent] = status
            self.current_agent = agent

    def update_report_section(self, section_name, content):
        if section_name in self.report_sections:
            self.report_sections[section_name] = content
            self._update_current_report()

    def _update_current_report(self):
        # For the panel display, only show the most recently updated section
        latest_section = None
        latest_content = None

        # Find the most recently updated section
        for section, content in self.report_sections.items():
            if content is not None:
                latest_section = section
                latest_content = content

        if latest_section and latest_content:
            # Format the current section for display
            section_titles = {
                "market_report": "Market Analysis",
                "sentiment_report": "Social Sentiment",
                "news_report": "News Analysis",
                "fundamentals_report": "Fundamentals Analysis",
                "investment_plan": "Research Team Decision",
                "trader_investment_plan": "Trading Team Plan",
                "final_trade_decision": "Portfolio Management Decision",
            }
            self.current_report = (
                f"### {section_titles[latest_section]}\n{latest_content}"
            )

        # Update the final complete report
        self._update_final_report()

    def _update_final_report(self):
        report_parts = []

        # Analyst Team Reports - use .get() to handle missing sections
        analyst_sections = ["market_report", "sentiment_report", "news_report", "fundamentals_report"]
        if any(self.report_sections.get(section) for section in analyst_sections):
            report_parts.append("## Analyst Team Reports")
            if self.report_sections.get("market_report"):
                report_parts.append(
                    f"### Market Analysis\n{self.report_sections['market_report']}"
                )
            if self.report_sections.get("sentiment_report"):
                report_parts.append(
                    f"### Social Sentiment\n{self.report_sections['sentiment_report']}"
                )
            if self.report_sections.get("news_report"):
                report_parts.append(
                    f"### News Analysis\n{self.report_sections['news_report']}"
                )
            if self.report_sections.get("fundamentals_report"):
                report_parts.append(
                    f"### Fundamentals Analysis\n{self.report_sections['fundamentals_report']}"
                )

        # Research Team Reports
        if self.report_sections.get("investment_plan"):
            report_parts.append("## Research Team Decision")
            report_parts.append(f"{self.report_sections['investment_plan']}")

        # Trading Team Reports
        if self.report_sections.get("trader_investment_plan"):
            report_parts.append("## Trading Team Plan")
            report_parts.append(f"{self.report_sections['trader_investment_plan']}")

        # Portfolio Management Decision
        if self.report_sections.get("final_trade_decision"):
            report_parts.append("## Portfolio Management Decision")
            report_parts.append(f"{self.report_sections['final_trade_decision']}")

        self.final_report = "\n\n".join(report_parts) if report_parts else None


message_buffer = MessageBuffer()


def create_layout():
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="main"),
        Layout(name="footer", size=3),
    )
    layout["main"].split_column(
        Layout(name="upper", ratio=3), Layout(name="analysis", ratio=5)
    )
    layout["upper"].split_row(
        Layout(name="progress", ratio=2), Layout(name="messages", ratio=3)
    )
    return layout


def format_tokens(n):
    """Format token count for display."""
    if n >= 1000:
        return f"{n/1000:.1f}k"
    return str(n)


def update_display(layout, spinner_text=None, stats_handler=None, start_time=None):
    # Header with welcome message
    layout["header"].update(
        Panel(
            "[bold green]Welcome to YiAgents CLI[/bold green]",
            title="Welcome to YiAgents",
            border_style="green",
            padding=(1, 2),
            expand=True,
        )
    )

    # Progress panel showing agent status
    progress_table = Table(
        show_header=True,
        header_style="bold magenta",
        show_footer=False,
        box=box.SIMPLE_HEAD,  # Use simple header with horizontal lines
        title=None,  # Remove the redundant Progress title
        padding=(0, 2),  # Add horizontal padding
        expand=True,  # Make table expand to fill available space
    )
    progress_table.add_column("Team", style="cyan", justify="center", width=20)
    progress_table.add_column("Agent", style="green", justify="center", width=20)
    progress_table.add_column("Status", style="yellow", justify="center", width=20)

    # Group agents by team - filter to only include agents in agent_status
    all_teams = {
        "Analyst Team": [
            "Market Analyst",
            "Sentiment Analyst",
            "News Analyst",
            "Fundamentals Analyst",
        ],
        "Research Team": ["Bull Researcher", "Bear Researcher", "Research Manager"],
        "Trading Team": ["Trader"],
        "Risk Management": ["Aggressive Analyst", "Neutral Analyst", "Conservative Analyst"],
        "Portfolio Management": ["Portfolio Manager"],
    }

    # Filter teams to only include agents that are in agent_status
    teams = {}
    for team, agents in all_teams.items():
        active_agents = [a for a in agents if a in message_buffer.agent_status]
        if active_agents:
            teams[team] = active_agents

    for team, agents in teams.items():
        # Add first agent with team name
        first_agent = agents[0]
        status = message_buffer.agent_status.get(first_agent, "pending")
        if status == "in_progress":
            spinner = Spinner(
                "dots", text="[blue]in_progress[/blue]", style="bold cyan"
            )
            status_cell = spinner
        else:
            status_color = {
                "pending": "yellow",
                "completed": "green",
                "error": "red",
            }.get(status, "white")
            status_cell = f"[{status_color}]{status}[/{status_color}]"
        progress_table.add_row(team, first_agent, status_cell)

        # Add remaining agents in team
        for agent in agents[1:]:
            status = message_buffer.agent_status.get(agent, "pending")
            if status == "in_progress":
                spinner = Spinner(
                    "dots", text="[blue]in_progress[/blue]", style="bold cyan"
                )
                status_cell = spinner
            else:
                status_color = {
                    "pending": "yellow",
                    "completed": "green",
                    "error": "red",
                }.get(status, "white")
                status_cell = f"[{status_color}]{status}[/{status_color}]"
            progress_table.add_row("", agent, status_cell)

        # Add horizontal line after each team
        progress_table.add_row("─" * 20, "─" * 20, "─" * 20, style="dim")

    layout["progress"].update(
        Panel(progress_table, title="Progress", border_style="cyan", padding=(1, 2))
    )

    # Messages panel showing recent messages and tool calls
    messages_table = Table(
        show_header=True,
        header_style="bold magenta",
        show_footer=False,
        expand=True,  # Make table expand to fill available space
        box=box.MINIMAL,  # Use minimal box style for a lighter look
        show_lines=True,  # Keep horizontal lines
        padding=(0, 1),  # Add some padding between columns
    )
    messages_table.add_column("Time", style="cyan", width=8, justify="center")
    messages_table.add_column("Type", style="green", width=10, justify="center")
    messages_table.add_column(
        "Content", style="white", no_wrap=False, ratio=1
    )  # Make content column expand

    # Combine tool calls and messages
    all_messages = []

    # Add tool calls
    for timestamp, tool_name, args in message_buffer.tool_calls:
        formatted_args = format_tool_args(args)
        all_messages.append((timestamp, "Tool", f"{tool_name}: {formatted_args}"))

    # Add regular messages
    for timestamp, msg_type, content in message_buffer.messages:
        content_str = str(content) if content else ""
        if len(content_str) > 200:
            content_str = content_str[:197] + "..."
        all_messages.append((timestamp, msg_type, content_str))

    # Sort by timestamp descending (newest first)
    all_messages.sort(key=lambda x: x[0], reverse=True)

    # Calculate how many messages we can show based on available space
    max_messages = 12

    # Get the first N messages (newest ones)
    recent_messages = all_messages[:max_messages]

    # Add messages to table (already in newest-first order)
    for timestamp, msg_type, content in recent_messages:
        # Format content with word wrapping
        wrapped_content = Text(content, overflow="fold")
        messages_table.add_row(timestamp, msg_type, wrapped_content)

    layout["messages"].update(
        Panel(
            messages_table,
            title="Messages & Tools",
            border_style="blue",
            padding=(1, 2),
        )
    )

    # Analysis panel showing current report
    if message_buffer.current_report:
        layout["analysis"].update(
            Panel(
                Markdown(message_buffer.current_report),
                title="Current Report",
                border_style="green",
                padding=(1, 2),
            )
        )
    else:
        layout["analysis"].update(
            Panel(
                "[italic]Waiting for analysis report...[/italic]",
                title="Current Report",
                border_style="green",
                padding=(1, 2),
            )
        )

    # Footer with statistics
    # Agent progress - derived from agent_status dict
    agents_completed = sum(
        1 for status in message_buffer.agent_status.values() if status == "completed"
    )
    agents_total = len(message_buffer.agent_status)

    # Report progress - based on agent completion (not just content existence)
    reports_completed = message_buffer.get_completed_reports_count()
    reports_total = len(message_buffer.report_sections)

    # Build stats parts
    stats_parts = [f"Agents: {agents_completed}/{agents_total}"]

    # LLM and tool stats from callback handler
    if stats_handler:
        stats = stats_handler.get_stats()
        stats_parts.append(f"LLM: {stats['llm_calls']}")
        stats_parts.append(f"Tools: {stats['tool_calls']}")

        # Token display with graceful fallback
        if stats["tokens_in"] > 0 or stats["tokens_out"] > 0:
            tokens_str = f"Tokens: {format_tokens(stats['tokens_in'])}\u2191 {format_tokens(stats['tokens_out'])}\u2193"
        else:
            tokens_str = "Tokens: --"
        stats_parts.append(tokens_str)

    stats_parts.append(f"Reports: {reports_completed}/{reports_total}")

    # Elapsed time
    if start_time:
        elapsed = time.time() - start_time
        elapsed_str = f"\u23f1 {int(elapsed // 60):02d}:{int(elapsed % 60):02d}"
        stats_parts.append(elapsed_str)

    stats_table = Table(show_header=False, box=None, padding=(0, 2), expand=True)
    stats_table.add_column("Stats", justify="center")
    stats_table.add_row(" | ".join(stats_parts))

    layout["footer"].update(Panel(stats_table, border_style="grey50"))


def get_user_selections():
    """Get all user selections before starting the analysis display."""
    # Display ASCII art welcome message
    with open(Path(__file__).parent / "static" / "welcome.txt", encoding="utf-8") as f:
        welcome_ascii = f.read()

    # Drop surrounding blank lines so the logo centers cleanly.
    ascii_lines = welcome_ascii.splitlines()
    while ascii_lines and ascii_lines[0].strip() == "":
        ascii_lines.pop(0)
    while ascii_lines and ascii_lines[-1].strip() == "":
        ascii_lines.pop()
    welcome_ascii = "\n".join(ascii_lines)

    # Center the logo as a single block. Rich's Align.center works line by
    # line and trims trailing whitespace, so centering a raw multi-line
    # string would shear the slant-font letterforms (each line shifted by
    # its own width). Wrapping the logo in a fixed-width (expand=False),
    # borderless panel makes every rendered line the same width, so
    # Align.center shifts the whole block evenly and the logo's internal
    # alignment is preserved.
    _invisible_box = box.Box("\n".join(["    "] * 8))
    logo_block = Panel(welcome_ascii, expand=False, box=_invisible_box, padding=0)

    # Welcome box content — logo and workflow, each centered as its own block.
    welcome_content = Group(
        Align.center(logo_block),
        "",
        Align.center("[bold white]Workflow Steps:[/bold white]"),
        Align.center(
            "[white]I. Analyst Team → II. Research Team → III. Trader → "
            "IV. Risk Management → V. Portfolio Management[/white]"
        ),
    )

    # Create and center the welcome box
    welcome_box = Panel(
        welcome_content,
        border_style="orange1",
        padding=(1, 2),
        title="Welcome to YiAgents",
        subtitle="Multi-Agents LLM Financial Trading Framework",
    )
    console.print(Align.center(welcome_box))
    console.print()

    # Create a boxed questionnaire for each step
    def create_question_box(title, prompt, default=None):
        box_content = f"[bold]{title}[/bold]\n"
        box_content += f"[dim]{prompt}[/dim]"
        if default:
            box_content += f"\n[dim]Default: {default}[/dim]"
        return Panel(box_content, border_style="blue", padding=(1, 2))

    def thinking_value_or_prompt(env_var, config_key, label, box_title, box_body, prompt_fn):
        """Return the env-configured reasoning/thinking value, or prompt for it.

        When ``env_var`` is set the interactive choice is skipped and the value
        the env overlay placed on DEFAULT_CONFIG is used — mirroring the
        env-precedence rule applied to the other selection steps.
        """
        if os.environ.get(env_var):
            value = DEFAULT_CONFIG[config_key]
            console.print(f"[green]✓ {label} from environment:[/green] {value}")
            return value
        console.print(create_question_box(box_title, box_body))
        return prompt_fn()

    # Step 1: Ticker symbol
    console.print(
        create_question_box(
            "Step 1: Ticker Symbol",
            "Enter the ticker, with exchange suffix when needed (e.g. SPY, 0700.HK, BTC-USD)",
            "SPY",
        )
    )
    selected_ticker = get_ticker()
    asset_type = detect_asset_type(selected_ticker)
    # Only announce when it's not the default stock path, to avoid printing
    # "stock" on every run.
    if asset_type.value != "stock":
        console.print(
            f"[green]Detected asset type:[/green] {asset_type.value}"
        )

    # Step 2: Analysis date
    default_date = datetime.datetime.now().strftime("%Y-%m-%d")
    console.print(
        create_question_box(
            "Step 2: Analysis Date",
            "Enter the analysis date (YYYY-MM-DD)",
            default_date,
        )
    )
    analysis_date = get_analysis_date()

    # Step 3: Output language (skipped when set via YIAGENTS_OUTPUT_LANGUAGE)
    if os.environ.get("YIAGENTS_OUTPUT_LANGUAGE"):
        output_language = DEFAULT_CONFIG["output_language"]
        console.print(
            f"[green]✓ Output language from environment:[/green] {output_language}"
        )
    else:
        console.print(
            create_question_box(
                "Step 3: Output Language",
                "Select the language for analyst reports and final decision"
            )
        )
        output_language = ask_output_language()

    # Step 4: Select analysts
    console.print(
        create_question_box(
            "Step 4: Analysts Team", "Select your LLM analyst agents for the analysis"
        )
    )
    selected_analysts = select_analysts(asset_type)
    console.print(
        f"[green]Selected analysts:[/green] {', '.join(analyst.value for analyst in selected_analysts)}"
    )

    # Step 5: Research depth (skipped when both round counts are set via env).
    # Research depth maps to the debate + risk round counts; when both are
    # supplied through YIAGENTS_MAX_DEBATE_ROUNDS / _MAX_RISK_ROUNDS we keep
    # the run non-interactive and honor the env values (#977).
    depth_from_env = bool(os.environ.get("YIAGENTS_MAX_DEBATE_ROUNDS")) and bool(
        os.environ.get("YIAGENTS_MAX_RISK_ROUNDS")
    )
    if depth_from_env:
        selected_research_depth = DEFAULT_CONFIG["max_debate_rounds"]
        console.print(
            f"[green]✓ Research depth from environment:[/green] "
            f"{DEFAULT_CONFIG['max_debate_rounds']} debate / "
            f"{DEFAULT_CONFIG['max_risk_discuss_rounds']} risk rounds"
        )
    else:
        console.print(
            create_question_box(
                "Step 5: Research Depth", "Select your research depth level"
            )
        )
        selected_research_depth = select_research_depth()

    # Step 6: LLM Provider (skipped when set via YIAGENTS_LLM_PROVIDER).
    # The backend URL comes from YIAGENTS_LLM_BACKEND_URL when set,
    # otherwise the provider's default endpoint — the same value the menu
    # would have picked.
    provider_from_env = bool(os.environ.get("YIAGENTS_LLM_PROVIDER"))
    if provider_from_env:
        selected_llm_provider = DEFAULT_CONFIG["llm_provider"].lower()
        backend_url = resolve_backend_url(
            selected_llm_provider, env_url=DEFAULT_CONFIG["backend_url"]
        )
        console.print(f"[green]✓ LLM provider from environment:[/green] {selected_llm_provider}")
        console.print(f"[green]✓ Backend URL:[/green] {backend_url}")
        # Still confirm/persist the API key so the run doesn't fail later.
        ensure_api_key(selected_llm_provider)
    else:
        console.print(
            create_question_box(
                "Step 6: LLM Provider", "Select your LLM provider"
            )
        )
        selected_llm_provider, backend_url = select_llm_provider()

        # Providers with regional endpoints prompt for the region as a secondary
        # step so the main dropdown stays clean (mainland China and international
        # accounts cannot share API keys).
        if selected_llm_provider == "qwen":
            selected_llm_provider, backend_url = ask_qwen_region()
        elif selected_llm_provider == "minimax":
            selected_llm_provider, backend_url = ask_minimax_region()
        elif selected_llm_provider == "glm":
            selected_llm_provider, backend_url = ask_glm_region()

        # Honor an explicit env backend URL even when the provider was chosen
        # interactively, so it isn't overwritten by the menu default (#978).
        backend_url = resolve_backend_url(
            selected_llm_provider, backend_url, env_url=DEFAULT_CONFIG["backend_url"]
        )

        # The generic OpenAI-compatible endpoint has no default; ask for it if
        # neither the menu nor the environment supplied one.
        if selected_llm_provider == "openai_compatible" and not backend_url:
            backend_url = prompt_openai_compatible_url()

        # For Ollama, surface the resolved endpoint (OLLAMA_BASE_URL vs default)
        # before model selection so it's obvious where we're connecting.
        if selected_llm_provider == "ollama":
            confirm_ollama_endpoint(backend_url)

        # Confirm the provider's API key is present; prompt the user to paste
        # one and persist it to .env if it's missing, so the analysis run
        # doesn't fail later at the first API call.
        ensure_api_key(selected_llm_provider)

    # Step 7: Thinking agents (skipped when either model is set via environment)
    if os.environ.get("YIAGENTS_QUICK_THINK_LLM") or os.environ.get("YIAGENTS_DEEP_THINK_LLM"):
        selected_shallow_thinker = DEFAULT_CONFIG["quick_think_llm"]
        selected_deep_thinker = DEFAULT_CONFIG["deep_think_llm"]
        console.print(
            f"[green]✓ Thinking agents from environment:[/green] "
            f"quick={selected_shallow_thinker}, deep={selected_deep_thinker}"
        )
    else:
        console.print(
            create_question_box(
                "Step 7: Thinking Agents", "Select your thinking agents for analysis"
            )
        )
        selected_shallow_thinker = select_shallow_thinking_agent(selected_llm_provider)
        selected_deep_thinker = select_deep_thinking_agent(selected_llm_provider)

    # Step 8: Provider-specific reasoning/thinking configuration. Each knob is
    # settable via its YIAGENTS_* env var; when that var is set (or the
    # provider itself came from env) the prompt is skipped and the configured
    # value is used — same env-precedence rule as the steps above. None = each
    # provider's own default.
    thinking_level = None
    reasoning_effort = None
    anthropic_effort = None

    provider_lower = selected_llm_provider.lower()
    if provider_from_env:
        thinking_level = DEFAULT_CONFIG["google_thinking_level"]
        reasoning_effort = DEFAULT_CONFIG["openai_reasoning_effort"]
        anthropic_effort = DEFAULT_CONFIG["anthropic_effort"]
    elif provider_lower == "google":
        thinking_level = thinking_value_or_prompt(
            "YIAGENTS_GOOGLE_THINKING_LEVEL", "google_thinking_level",
            "Gemini thinking mode", "Step 8: Thinking Mode",
            "Configure Gemini thinking mode", ask_gemini_thinking_config,
        )
    elif provider_lower == "openai":
        reasoning_effort = thinking_value_or_prompt(
            "YIAGENTS_OPENAI_REASONING_EFFORT", "openai_reasoning_effort",
            "Reasoning effort", "Step 8: Reasoning Effort",
            "Configure OpenAI reasoning effort level", ask_openai_reasoning_effort,
        )
    elif provider_lower == "anthropic":
        anthropic_effort = thinking_value_or_prompt(
            "YIAGENTS_ANTHROPIC_EFFORT", "anthropic_effort",
            "Claude effort", "Step 8: Effort Level",
            "Configure Claude effort level", ask_anthropic_effort,
        )

    return {
        "ticker": selected_ticker,
        "asset_type": asset_type.value,
        "analysis_date": analysis_date,
        "analysts": selected_analysts,
        "research_depth": selected_research_depth,
        "llm_provider": selected_llm_provider.lower(),
        "backend_url": backend_url,
        "shallow_thinker": selected_shallow_thinker,
        "deep_thinker": selected_deep_thinker,
        "google_thinking_level": thinking_level,
        "openai_reasoning_effort": reasoning_effort,
        "anthropic_effort": anthropic_effort,
        "output_language": output_language,
    }


def get_analysis_date():
    """Get the analysis date from user input."""
    while True:
        date_str = typer.prompt(
            "", default=datetime.datetime.now().strftime("%Y-%m-%d")
        )
        try:
            # Validate date format and ensure it's not in the future
            analysis_date = datetime.datetime.strptime(date_str, "%Y-%m-%d")
            if analysis_date.date() > datetime.datetime.now().date():
                console.print("[red]Error: Analysis date cannot be in the future[/red]")
                continue
            return date_str
        except ValueError:
            console.print(
                "[red]Error: Invalid date format. Please use YYYY-MM-DD[/red]"
            )


def save_report_to_disk(final_state, ticker: str, save_path: Path):
    """Save the complete analysis report to disk (shared CLI/API writer)."""
    return write_report_tree(final_state, ticker, save_path)


def display_complete_report(final_state):
    """Display the complete analysis report sequentially (avoids truncation)."""
    console.print()
    console.print(Rule("Complete Analysis Report", style="bold green"))

    # I. Analyst Team Reports
    analysts = []
    if final_state.get("market_report"):
        analysts.append(("Market Analyst", final_state["market_report"]))
    if final_state.get("sentiment_report"):
        analysts.append(("Sentiment Analyst", final_state["sentiment_report"]))
    if final_state.get("news_report"):
        analysts.append(("News Analyst", final_state["news_report"]))
    if final_state.get("fundamentals_report"):
        analysts.append(("Fundamentals Analyst", final_state["fundamentals_report"]))
    if analysts:
        console.print(Panel("[bold]I. Analyst Team Reports[/bold]", border_style="cyan"))
        for title, content in analysts:
            console.print(Panel(Markdown(content), title=title, border_style="blue", padding=(1, 2)))

    # II. Research Team Reports
    if final_state.get("investment_debate_state"):
        debate = final_state["investment_debate_state"]
        research = []
        if debate.get("bull_history"):
            research.append(("Bull Researcher", debate["bull_history"]))
        if debate.get("bear_history"):
            research.append(("Bear Researcher", debate["bear_history"]))
        if debate.get("judge_decision"):
            research.append(("Research Manager", debate["judge_decision"]))
        if research:
            console.print(Panel("[bold]II. Research Team Decision[/bold]", border_style="magenta"))
            for title, content in research:
                console.print(Panel(Markdown(content), title=title, border_style="blue", padding=(1, 2)))

    # III. Trading Team
    if final_state.get("trader_investment_plan"):
        console.print(Panel("[bold]III. Trading Team Plan[/bold]", border_style="yellow"))
        console.print(Panel(Markdown(final_state["trader_investment_plan"]), title="Trader", border_style="blue", padding=(1, 2)))

    # IV. Risk Management Team
    if final_state.get("risk_debate_state"):
        risk = final_state["risk_debate_state"]
        risk_reports = []
        if risk.get("aggressive_history"):
            risk_reports.append(("Aggressive Analyst", risk["aggressive_history"]))
        if risk.get("conservative_history"):
            risk_reports.append(("Conservative Analyst", risk["conservative_history"]))
        if risk.get("neutral_history"):
            risk_reports.append(("Neutral Analyst", risk["neutral_history"]))
        if risk_reports:
            console.print(Panel("[bold]IV. Risk Management Team Decision[/bold]", border_style="red"))
            for title, content in risk_reports:
                console.print(Panel(Markdown(content), title=title, border_style="blue", padding=(1, 2)))

    # V. Portfolio Manager Decision. Prefer the post-processed decision because
    # it includes the deterministic risk overlay; the risk judge text is only
    # a compatibility fallback for partial/legacy states.
    risk = final_state.get("risk_debate_state") or {}
    portfolio_decision = final_state.get("final_trade_decision") or risk.get(
        "judge_decision"
    )
    if portfolio_decision:
        console.print(
            Panel(
                "[bold]V. Portfolio Manager Decision[/bold]",
                border_style="green",
            )
        )
        console.print(
            Panel(
                Markdown(portfolio_decision),
                title="Portfolio Manager",
                border_style="blue",
                padding=(1, 2),
            )
        )


def update_research_team_status(status):
    """Update status for research team members (not Trader)."""
    research_team = ["Bull Researcher", "Bear Researcher", "Research Manager"]
    for agent in research_team:
        message_buffer.update_agent_status(agent, status)


# Ordered list of analysts for status transitions
ANALYST_ORDER = ["market", "social", "news", "fundamentals"]
ANALYST_AGENT_NAMES = {
    "market": "Market Analyst",
    "social": "Sentiment Analyst",
    "news": "News Analyst",
    "fundamentals": "Fundamentals Analyst",
}
ANALYST_REPORT_MAP = {
    "market": "market_report",
    "social": "sentiment_report",
    "news": "news_report",
    "fundamentals": "fundamentals_report",
}


def update_analyst_statuses(message_buffer, chunk, wall_time_tracker=None):
    """Update analyst statuses based on accumulated report state.

    Logic:
    - Store new report content from the current chunk if present
    - Check accumulated report_sections (not just current chunk) for status
    - Analysts with reports = completed
    - First analyst without report = in_progress
    - Remaining analysts without reports = pending
    - When all analysts done, set Bull Researcher to in_progress
    """
    selected = message_buffer.selected_analysts
    found_active = False

    if wall_time_tracker is not None:
        sync_analyst_tracker_from_chunk(wall_time_tracker, chunk)

    for analyst_key in ANALYST_ORDER:
        if analyst_key not in selected:
            continue

        agent_name = ANALYST_AGENT_NAMES[analyst_key]
        report_key = ANALYST_REPORT_MAP[analyst_key]

        # Capture new report content from current chunk
        if chunk.get(report_key):
            message_buffer.update_report_section(report_key, chunk[report_key])

        # Determine status from accumulated sections, not just current chunk
        has_report = bool(message_buffer.report_sections.get(report_key))

        if has_report:
            message_buffer.update_agent_status(agent_name, "completed")
        elif not found_active:
            message_buffer.update_agent_status(agent_name, "in_progress")
            found_active = True
        else:
            message_buffer.update_agent_status(agent_name, "pending")

    # When all analysts complete, transition research team to in_progress
    if (
        not found_active
        and selected
        and message_buffer.agent_status.get("Bull Researcher") == "pending"
    ):
        message_buffer.update_agent_status("Bull Researcher", "in_progress")

def extract_content_string(content):
    """Extract string content from various message formats.
    Returns None if no meaningful text content is found.
    """
    import ast

    def is_empty(val):
        """Check if value is empty using Python's truthiness."""
        if val is None or val == '':
            return True
        if isinstance(val, str):
            s = val.strip()
            if not s:
                return True
            try:
                return not bool(ast.literal_eval(s))
            except (ValueError, SyntaxError):
                return False  # Can't parse = real text
        return not bool(val)

    if is_empty(content):
        return None

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, dict):
        text = content.get('text', '')
        return text.strip() if not is_empty(text) else None

    if isinstance(content, list):
        text_parts = [
            item.get('text', '').strip() if isinstance(item, dict) and item.get('type') == 'text'
            else (item.strip() if isinstance(item, str) else '')
            for item in content
        ]
        result = ' '.join(t for t in text_parts if t and not is_empty(t))
        return result if result else None

    return str(content).strip() if not is_empty(content) else None


def classify_message_type(message) -> tuple[str, str | None]:
    """Classify LangChain message into display type and extract content.

    Returns:
        (type, content) - type is one of: User, Agent, Data, Control
                        - content is extracted string or None
    """
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

    content = extract_content_string(getattr(message, 'content', None))

    if isinstance(message, HumanMessage):
        if content and content.strip() == "Continue":
            return ("Control", content)
        return ("User", content)

    if isinstance(message, ToolMessage):
        return ("Data", content)

    if isinstance(message, AIMessage):
        return ("Agent", content)

    # Fallback for unknown types
    return ("System", content)


def format_tool_args(args, max_length=80) -> str:
    """Format tool arguments for terminal display."""
    result = str(args)
    if len(result) > max_length:
        return result[:max_length - 3] + "..."
    return result

def _build_run_config(selections: dict, checkpoint: bool | None) -> dict:
    """Assemble the run config from interactive selections, honoring env precedence.

    Round counts and checkpoint follow "explicit env/flag wins": an env-applied
    value on DEFAULT_CONFIG is preserved unless the user overrode it on the CLI.
    """
    config = DEFAULT_CONFIG.copy()
    # Research depth sets both round counts, but an explicit env override
    # (YIAGENTS_MAX_DEBATE_ROUNDS / _MAX_RISK_ROUNDS) wins over the
    # interactive selection — leave the env-applied value in place (#977).
    if not os.environ.get("YIAGENTS_MAX_DEBATE_ROUNDS"):
        config["max_debate_rounds"] = selections["research_depth"]
    if not os.environ.get("YIAGENTS_MAX_RISK_ROUNDS"):
        config["max_risk_discuss_rounds"] = selections["research_depth"]
    config["quick_think_llm"] = selections["shallow_thinker"]
    config["deep_think_llm"] = selections["deep_thinker"]
    config["backend_url"] = selections["backend_url"]
    config["llm_provider"] = selections["llm_provider"].lower()
    # Provider-specific thinking configuration
    config["google_thinking_level"] = selections.get("google_thinking_level")
    config["openai_reasoning_effort"] = selections.get("openai_reasoning_effort")
    config["anthropic_effort"] = selections.get("anthropic_effort")
    config["output_language"] = selections.get("output_language", "English")
    # --checkpoint/--no-checkpoint overrides only when explicitly given; omitting
    # the flag preserves YIAGENTS_CHECKPOINT_ENABLED / the default (#976).
    if checkpoint is not None:
        config["checkpoint_enabled"] = checkpoint
    return config


def _apply_batch_worker_override(config: dict, workers: int | None) -> dict:
    """Make an explicit ``--workers`` value authoritative.

    With no CLI value, the configured ``batch_concurrency`` master switch is
    preserved. Supplying ``--workers`` is itself an explicit opt-in (or an
    explicit serial request for ``1``), so the default-off switch cannot
    silently force a requested pool back to one worker.
    """
    if workers is not None:
        if workers < 1:
            raise ValueError("workers must be at least 1")
        config["batch_concurrency"] = workers > 1
    return config


def _store_cli_decision(
    graph: YiAgentsGraph, ticker: str, trade_date: str, final_state: dict
) -> None:
    """Persist the streamed run's decision for deferred reflection.

    Reads ``final_trade_decision`` with ``.get`` (not bare indexing): a
    streamed state whose PM output never landed would otherwise raise
    KeyError right before the report is displayed, crashing the UI at the
    finish line. An absent/empty decision still gets stored (keeps the
    memory-log contract) but is logged at WARNING so the gap is observable.
    """
    decision = final_state.get("final_trade_decision") or ""
    if not decision:
        logger.warning(
            "Streamed run for %s on %s produced no final_trade_decision; "
            "storing an empty decision.",
            ticker,
            trade_date,
        )
    graph.memory_log.store_decision(
        ticker=ticker, trade_date=trade_date, final_trade_decision=decision
    )


def run_analysis(checkpoint: bool | None = None):
    # First get all user selections
    selections = get_user_selections()

    config = _build_run_config(selections, checkpoint)

    # Interactive path: a human is watching the live display, so soften the
    # data-vacuum gate from reject to warn — the run still completes with the
    # DEGRADED banner + data_quality evidence for the human to judge. An
    # explicit YIAGENTS_DATA_VACUUM_POLICY wins over this default (someone who
    # sets it deliberately wants the typed failure even interactively).
    if "YIAGENTS_DATA_VACUUM_POLICY" not in os.environ:
        config["data_vacuum_policy"] = "warn"

    # Create stats callback handler for tracking LLM/tool calls
    stats_handler = StatsCallbackHandler()

    # Normalize analyst selection to predefined order (selection is a 'set', order is fixed)
    selected_set = {analyst.value for analyst in selections["analysts"]}
    selected_analyst_keys = [a for a in ANALYST_ORDER if a in selected_set]
    analyst_execution_plan = build_analyst_execution_plan(selected_analyst_keys)
    analyst_wall_time_tracker = AnalystWallTimeTracker(analyst_execution_plan)

    # Initialize the graph with callbacks bound to LLMs
    graph = YiAgentsGraph(
        selected_analyst_keys,
        config=config,
        debug=True,
        callbacks=[stats_handler],
    )

    # Initialize message buffer with selected analysts
    message_buffer.init_for_analysis(selected_analyst_keys)

    # Track start time for elapsed display
    start_time = time.time()

    # Create result directory
    results_dir = Path(config["results_dir"]) / selections["ticker"] / selections["analysis_date"]
    results_dir.mkdir(parents=True, exist_ok=True)
    report_dir = results_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    log_file = results_dir / "message_tool.log"
    log_file.touch(exist_ok=True)

    def save_message_decorator(obj, func_name):
        func = getattr(obj, func_name)
        @wraps(func)
        def wrapper(*args, **kwargs):
            func(*args, **kwargs)
            timestamp, message_type, content = obj.messages[-1]
            content = content.replace("\n", " ")  # Replace newlines with spaces
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(f"{timestamp} [{message_type}] {content}\n")
        return wrapper

    def save_tool_call_decorator(obj, func_name):
        func = getattr(obj, func_name)
        @wraps(func)
        def wrapper(*args, **kwargs):
            func(*args, **kwargs)
            timestamp, tool_name, args = obj.tool_calls[-1]
            args_str = ", ".join(f"{k}={v}" for k, v in args.items())
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(f"{timestamp} [Tool Call] {tool_name}({args_str})\n")
        return wrapper

    def save_report_section_decorator(obj, func_name):
        func = getattr(obj, func_name)
        @wraps(func)
        def wrapper(section_name, content):
            func(section_name, content)
            if section_name in obj.report_sections and obj.report_sections[section_name] is not None:
                content = obj.report_sections[section_name]
                if content:
                    file_name = f"{section_name}.md"
                    text = "\n".join(str(item) for item in content) if isinstance(content, list) else content
                    with open(report_dir / file_name, "w", encoding="utf-8") as f:
                        f.write(text)
        return wrapper

    message_buffer.add_message = save_message_decorator(  # type: ignore[method-assign]
        message_buffer, "add_message"
    )
    message_buffer.add_tool_call = save_tool_call_decorator(  # type: ignore[method-assign]
        message_buffer, "add_tool_call"
    )
    message_buffer.update_report_section = save_report_section_decorator(  # type: ignore[method-assign]
        message_buffer, "update_report_section"
    )

    # Now start the display layout
    layout = create_layout()

    # BatchRunner (including the web/run_robust path) takes this same
    # cross-process lock. The interactive streaming path bypasses BatchRunner,
    # so it must join the shared run-lock contract explicitly. The same goes
    # for the analysis-date PIT clamp: _run_graph pins it around propagate(),
    # but this path streams graph.stream directly — without the pin a
    # historical analysis date leaves every vendor clamp in live mode and
    # rows after the analysis date leak into the prompt.
    from yiagents.batch.runner import serialized_run
    from yiagents.dataflows.utils import pinned_analysis_date

    with serialized_run(
        config,
        selections["ticker"],
        selections["analysis_date"],
        selections["asset_type"],
    ), pinned_analysis_date(selections["analysis_date"]), Live(layout, refresh_per_second=4):
        # Initial display
        update_display(layout, stats_handler=stats_handler, start_time=start_time)

        # Add initial messages
        message_buffer.add_message("System", f"Selected ticker: {selections['ticker']}")
        if selections["asset_type"] != "stock":
            message_buffer.add_message("System", f"Detected asset type: {selections['asset_type']}")
        message_buffer.add_message(
            "System", f"Analysis date: {selections['analysis_date']}"
        )
        message_buffer.add_message(
            "System",
            f"Selected analysts: {', '.join(analyst.value for analyst in selections['analysts'])}",
        )
        update_display(layout, stats_handler=stats_handler, start_time=start_time)

        # Update agent status to in_progress for the first analyst
        first_analyst = get_initial_analyst_node(analyst_execution_plan)
        message_buffer.update_agent_status(first_analyst, "in_progress")
        analyst_wall_time_tracker.mark_started(selected_analyst_keys[0])
        update_display(layout, stats_handler=stats_handler, start_time=start_time)

        # Create spinner text
        spinner_text = (
            f"Analyzing {selections['ticker']} on {selections['analysis_date']}..."
        )
        update_display(layout, spinner_text, stats_handler=stats_handler, start_time=start_time)

        # Initialize state and get graph args with callbacks.
        # Bind the data-quality event accumulator in THIS (parent) context
        # before the graph runs — the same contract as
        # YiAgentsGraph._run_graph. langgraph executes node tasks inside
        # copied contexts, so without this pre-bound list the sentinel events
        # recorded by vendor routers inside nodes would never be visible to
        # finalize_streamed_run below and the DEGRADED evidence chain goes dead.
        quality.ensure_run_context()
        # Mirror propagate()'s run contract for the per-run Tavily search
        # budget: fresh counter per CLI run, never inherited from a prior one.
        from yiagents.dataflows import tavily as tavily_vendor

        tavily_vendor.reset_run_budget()
        # Resolve the instrument identity once here so all agents anchor to
        # the real company (#814); the CLI builds state directly rather than
        # going through propagate(), so this must happen on the CLI path too.
        graph._resolve_pending_entries(  # noqa: SLF001 -- mirror propagate's run contract
            selections["ticker"],
            as_of_date=str(selections["analysis_date"]),
        )
        past_context = graph.memory_log.get_past_context(
            selections["ticker"],
            as_of_date=str(selections["analysis_date"]),
        )
        instrument_context = graph.resolve_instrument_context(
            selections["ticker"], selections["asset_type"],
            trade_date=str(selections["analysis_date"]),
        )
        init_agent_state = graph.propagator.create_initial_state(
            selections["ticker"],
            selections["analysis_date"],
            asset_type=selections["asset_type"],
            past_context=past_context,
            instrument_context=instrument_context,
        )
        # Pass callbacks to graph config for tool execution tracking
        # (LLM tracking is handled separately via LLM constructor)
        args = graph.propagator.get_graph_args(callbacks=[stats_handler])

        # Stream the analysis
        trace = []
        for chunk in graph.graph.stream(init_agent_state, **args):
            # Process all messages in chunk, deduplicating by message ID
            for message in chunk.get("messages", []):
                msg_id = getattr(message, "id", None)
                if msg_id is not None:
                    if msg_id in message_buffer._processed_message_ids:
                        continue
                    message_buffer._processed_message_ids.add(msg_id)

                msg_type, content = classify_message_type(message)
                if content and content.strip():
                    message_buffer.add_message(msg_type, content)

                if hasattr(message, "tool_calls") and message.tool_calls:
                    for tool_call in message.tool_calls:
                        if isinstance(tool_call, dict):
                            message_buffer.add_tool_call(tool_call["name"], tool_call["args"])
                        else:
                            message_buffer.add_tool_call(tool_call.name, tool_call.args)

            # Update analyst statuses based on report state (runs on every chunk)
            update_analyst_statuses(
                message_buffer,
                chunk,
                wall_time_tracker=analyst_wall_time_tracker,
            )

            # Research Team - Handle Investment Debate State
            if chunk.get("investment_debate_state"):
                debate_state = chunk["investment_debate_state"]
                bull_hist = debate_state.get("bull_history", "").strip()
                bear_hist = debate_state.get("bear_history", "").strip()
                judge = debate_state.get("judge_decision", "").strip()

                # Only update status when there's actual content
                if bull_hist or bear_hist:
                    update_research_team_status("in_progress")
                if bull_hist:
                    message_buffer.update_report_section(
                        "investment_plan", f"### Bull Researcher Analysis\n{bull_hist}"
                    )
                if bear_hist:
                    message_buffer.update_report_section(
                        "investment_plan", f"### Bear Researcher Analysis\n{bear_hist}"
                    )
                if judge:
                    message_buffer.update_report_section(
                        "investment_plan", f"### Research Manager Decision\n{judge}"
                    )
                    update_research_team_status("completed")
                    message_buffer.update_agent_status("Trader", "in_progress")

            # Trading Team
            if chunk.get("trader_investment_plan"):
                message_buffer.update_report_section(
                    "trader_investment_plan", chunk["trader_investment_plan"]
                )
                if message_buffer.agent_status.get("Trader") != "completed":
                    message_buffer.update_agent_status("Trader", "completed")
                    message_buffer.update_agent_status("Aggressive Analyst", "in_progress")

            # Risk Management Team - Handle Risk Debate State
            if chunk.get("risk_debate_state"):
                risk_state = chunk["risk_debate_state"]
                agg_hist = risk_state.get("aggressive_history", "").strip()
                con_hist = risk_state.get("conservative_history", "").strip()
                neu_hist = risk_state.get("neutral_history", "").strip()
                judge = risk_state.get("judge_decision", "").strip()

                if agg_hist:
                    if message_buffer.agent_status.get("Aggressive Analyst") != "completed":
                        message_buffer.update_agent_status("Aggressive Analyst", "in_progress")
                    message_buffer.update_report_section(
                        "final_trade_decision", f"### Aggressive Analyst Analysis\n{agg_hist}"
                    )
                if con_hist:
                    if message_buffer.agent_status.get("Conservative Analyst") != "completed":
                        message_buffer.update_agent_status("Conservative Analyst", "in_progress")
                    message_buffer.update_report_section(
                        "final_trade_decision", f"### Conservative Analyst Analysis\n{con_hist}"
                    )
                if neu_hist:
                    if message_buffer.agent_status.get("Neutral Analyst") != "completed":
                        message_buffer.update_agent_status("Neutral Analyst", "in_progress")
                    message_buffer.update_report_section(
                        "final_trade_decision", f"### Neutral Analyst Analysis\n{neu_hist}"
                    )
                if judge and message_buffer.agent_status.get("Portfolio Manager") != "completed":
                    message_buffer.update_agent_status("Portfolio Manager", "in_progress")
                    message_buffer.update_report_section(
                        "final_trade_decision", f"### Portfolio Manager Decision\n{judge}"
                    )
                    message_buffer.update_agent_status("Aggressive Analyst", "completed")
                    message_buffer.update_agent_status("Conservative Analyst", "completed")
                    message_buffer.update_agent_status("Neutral Analyst", "completed")
                    message_buffer.update_agent_status("Portfolio Manager", "completed")

            # Update the display
            update_display(layout, stats_handler=stats_handler, start_time=start_time)

            trace.append(chunk)

        # Streamed chunks are per-node deltas, not full state. Merge them
        # so every report field populated across the run is present.
        final_state = {}
        for chunk in trace:
            final_state.update(chunk)

        if not final_state:
            raise RuntimeError("graph.stream emitted no chunks for this run")

        # The interactive path streams the graph directly for live UI updates,
        # so apply the same post-processing contract as YiAgentsGraph.propagate
        # before updating, displaying, or saving report sections.
        final_state = graph._apply_risk_overlay(  # noqa: SLF001
            selections["ticker"],
            selections["analysis_date"],
            final_state,
            portfolio_state=None,
        )
        # Land the same on-disk evidence a propagate() run produces
        # (full_states_log_<date>.json + the data_quality block on the state)
        # so the CLI run shows up in the web history and its reports can
        # render the DEGRADED banner.
        final_state = graph.finalize_streamed_run(
            selections["ticker"], selections["analysis_date"], final_state
        )
        graph.curr_state = final_state
        _store_cli_decision(
            graph, selections["ticker"], selections["analysis_date"], final_state
        )

        # Update all agent statuses to completed
        for agent in message_buffer.agent_status:
            message_buffer.update_agent_status(agent, "completed")

        message_buffer.add_message(
            "System", f"Completed analysis for {selections['analysis_date']}"
        )
        message_buffer.add_message("System", analyst_wall_time_tracker.format_summary())

        # Update final report sections
        for section in message_buffer.report_sections:
            if section in final_state:
                message_buffer.update_report_section(section, final_state[section])

        update_display(layout, stats_handler=stats_handler, start_time=start_time)

    # Post-analysis prompts (outside Live context for clean interaction)
    console.print("\n[bold cyan]Analysis Complete![/bold cyan]\n")
    console.print(f"[dim]{analyst_wall_time_tracker.format_summary()}[/dim]")

    # Prompt to save report
    save_choice = typer.prompt("Save report?", default="Y").strip().upper()
    if save_choice in ("Y", "YES", ""):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        default_path = Path.cwd() / "reports" / f"{selections['ticker']}_{timestamp}"
        save_path_str = typer.prompt(
            "Save path (press Enter for default)",
            default=str(default_path)
        ).strip()
        save_path = Path(save_path_str)
        try:
            report_file = save_report_to_disk(final_state, selections["ticker"], save_path)
            console.print(f"\n[green]✓ Report saved to:[/green] {save_path.resolve()}")
            console.print(f"  [dim]Complete report:[/dim] {report_file.name}")
        except Exception as e:
            console.print(f"[red]Error saving report: {e}[/red]")

    # Prompt to display full report
    display_choice = typer.prompt("\nDisplay full report on screen?", default="Y").strip().upper()
    if display_choice in ("Y", "YES", ""):
        display_complete_report(final_state)


def _warn_config_drift() -> None:
    """Interactive-entry helper: warn when the config drifted from the last snapshot.

    The self-improvement loop's audit trail is only useful if drift is
    *noticed*: ``yiagents snapshot record`` pins a config, and this check
    surfaces (non-blocking, fail-soft) that the live config no longer matches
    it — e.g. an ``indicator_battery`` prune was applied without a follow-up
    snapshot. Point the user at ``yiagents snapshot diff`` for the detail.
    """
    try:
        from yiagents.config_snapshot import diff_against_last_snapshot

        diffs = diff_against_last_snapshot(dict(DEFAULT_CONFIG))
        if diffs:
            console.print(
                f"[yellow]⚠️  Config drifted from the last recorded snapshot "
                f"({len(diffs)} key(s): {', '.join(sorted(diffs)[:8])}"
                f"{'…' if len(diffs) > 8 else ''}). Run `yiagents snapshot diff` "
                f"for details, then `yiagents snapshot record` after applying "
                f"intentional changes.[/yellow]"
            )
    except Exception:  # noqa: BLE001 -- advisory only, never blocks the CLI
        import logging

        logging.getLogger(__name__).debug("config drift check failed", exc_info=True)


@app.command()
def analyze(
    checkpoint: bool | None = typer.Option(
        None,
        "--checkpoint/--no-checkpoint",
        help="Enable/disable checkpoint-resume (save state after each node so a "
        "crashed run can resume). Omit to honor YIAGENTS_CHECKPOINT_ENABLED.",
    ),
    clear_checkpoints: bool = typer.Option(
        False,
        "--clear-checkpoints",
        help="Delete all saved checkpoints before running (force fresh start).",
    ),
):
    if clear_checkpoints:
        from yiagents.graph.checkpointer import clear_all_checkpoints
        n = clear_all_checkpoints(DEFAULT_CONFIG["data_cache_dir"])
        console.print(f"[yellow]Cleared {n} checkpoint(s).[/yellow]")
    _warn_config_drift()
    run_analysis(checkpoint=checkpoint)


@app.command()
def batch(
    tickers: list[str] = typer.Option(
        ...,
        "--ticker",
        "-t",
        help="Ticker symbol; repeat -t for each (e.g. -t AAPL -t NVDA -t MSFT). "
        "One asset class per batch.",
    ),
    date: str = typer.Option(..., "--date", "-d", help="Analysis date YYYY-MM-DD."),
    asset_type: str = typer.Option(
        "auto",
        "--asset-type",
        help="stock | crypto | crypto_spot | crypto_perp | auto (auto = infer "
        "from the first ticker; crypto_spot = Binance spot, crypto_perp = "
        "Binance USDT-M perpetual — both explicit opt-in).",
    ),
    workers: int | None = typer.Option(
        None,
        "--workers",
        "-w",
        help="Concurrency K (pool size). An explicit K>1 enables concurrency; "
        "when omitted, honor YIAGENTS_BATCH_CONCURRENCY/BATCH_WORKERS.",
    ),
):
    """Analyze many tickers concurrently (one API key drives many agents).

    Each ticker runs the exact same analysis as ``analyze``; concurrency is
    layered ABOVE propagate() — agent inputs/depth/reasoning are unchanged, so
    every ticker is byte-equivalent to a serial run. One batch = one asset
    class (all workers share one config). Reports land under results_dir per
    ticker.

    Concurrency is OFF by default (strictly serial, K=1, byte-equivalent to
    per-ticker runs). Pass ``--workers K`` with K>1 to fan the ticker list out
    across a pool of worker graphs; ``--workers 1`` is explicit serial. The
    env switch ``YIAGENTS_BATCH_CONCURRENCY=true`` also enables the pool.
    """
    from yiagents.batch.runner import BatchInputError, BatchRunner, prepare_batch_run

    try:
        config, resolved = prepare_batch_run(tickers, date, asset_type, workers)
    except BatchInputError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=2) from None
    if asset_type == "auto":
        console.print(f"[cyan]Asset type inferred: {resolved} (from {tickers[0]})[/cyan]")

    console.print(
        f"[bold]Batch: {len(tickers)} tickers | date={date} | type={resolved}[/bold]"
    )
    with BatchRunner(config, workers=workers, progress=False) as runner:
        results = runner.run(tickers, date, asset_type=resolved)

    ok = sum(1 for r in results if r["error"] is None)
    table = Table(title=f"Batch results ({ok}/{len(results)} ok)", box=box.SIMPLE)
    table.add_column("ticker")
    table.add_column("status")
    table.add_column("elapsed", justify="right")
    table.add_column("report")
    for r in results:
        if r["error"] is None:
            table.add_row(
                r["ticker"],
                "[green]✅[/green]",
                f"{r['elapsed']:.1f}s",
                str(r["report_path"] or ""),
            )
        else:
            table.add_row(
                r["ticker"],
                "[red]❌[/red]",
                "-",
                f"{type(r['error']).__name__}: {r['error']}",
            )
    console.print(table)
    if ok != len(results):
        raise typer.Exit(code=1)


@app.command("verify-history")
def verify_history_cmd(
    holding_days: int = typer.Option(
        5, "--holding-days", help="Forward horizon in sessions to score against.",
    ),
    results_dir: str = typer.Option(
        "", "--results-dir",
        help="Results dir to scan (default: results_dir from config).",
    ),
):
    """Score archived ratings against realized forward returns.

    Scans every ``full_states_log_<date>.json`` under the results dir, fetches
    the PIT forward return per (ticker, date, rating), and writes
    ``accuracy/accuracy_report.{json,md}`` — directional hit rate, per-rating
    and per-ticker tables (all with sample sizes). Decisions whose horizon
    has not fully elapsed are counted as pending, never scored.
    """
    from yiagents.accuracy import verify_history as run_verify

    report, json_path, md_path = run_verify(
        results_dir or None, holding_days=holding_days
    )
    d = report["direction"]
    if d["hit_rate"] is not None:
        console.print(
            f"[bold]Directional hit rate:[/bold] {d['hits']}/{d['n']} "
            f"({d['hit_rate']:.1%}) over {report['holding_days']}d"
        )
    else:
        console.print("[yellow]No fully-elapsed directional decisions yet.[/yellow]")
    console.print(
        f"Runs: {report['total_runs']} scanned | {report['scored']} scored | "
        f"{report['pending']} pending"
    )
    console.print(f"[dim]JSON: {json_path}[/dim]")
    console.print(f"[dim]Markdown: {md_path}[/dim]")


@app.command("memory-resolve")
def memory_resolve_cmd(
    tickers: list[str] = typer.Option(
        None, "--ticker", "-t",
        help="Only resolve these tickers (repeatable). Default: all pending.",
    ),
    as_of: str = typer.Option(
        "", "--as-of",
        help="Resolve as of this date YYYY-MM-DD (default: today). Outcomes "
        "are PIT-clamped to it.",
    ),
):
    """Resolve pending memory-log entries WITHOUT running a full analysis.

    Pending entries normally only resolve when their ticker is analyzed
    again; entries for tickers you stopped analyzing stayed pending forever.
    This sweeps them on demand — one reflection LLM call per resolved entry
    (the same resolution the graph runs at the start of a same-ticker run).
    """
    import copy as _copy
    from pathlib import Path as _Path

    from yiagents.agents.utils.memory import TradingMemoryLog
    from yiagents.dataflows.market_regime import resolve_market_benchmark
    from yiagents.graph.memory_resolution import resolve_pending_entries
    from yiagents.graph.reflection import Reflector
    from yiagents.llm_clients import create_llm_client

    config = _copy.deepcopy(DEFAULT_CONFIG)
    # The memory log is opt-in at run time; resolution reads + updates the
    # SAME file, so force-enable the path — but only proceed if it exists
    # (nothing to resolve otherwise; never create a log from here).
    config["memory_enabled"] = True
    memory_log = TradingMemoryLog(config)
    pending_all = memory_log.get_pending_entries()
    if not pending_all:
        console.print("[green]No pending memory entries.[/green]")
        return
    if memory_log._log_path is None or not _Path(memory_log._log_path).is_file():  # noqa: SLF001
        console.print(
            "[yellow]Memory log file not found — pending entries require "
            "memory_enabled at analysis time to have been written.[/yellow]"
        )
        raise typer.Exit(code=2)

    wanted = {t.upper() for t in tickers} if tickers else None
    pending = [
        e for e in pending_all
        if wanted is None or str(e.get("ticker", "")).upper() in wanted
    ]
    tickers_to_sweep = sorted({str(e["ticker"]) for e in pending})
    if not tickers_to_sweep:
        console.print("[green]No pending entries match the given ticker filter.[/green]")
        return

    llm = create_llm_client(
        provider=config["llm_provider"],
        model=config["quick_think_llm"],
        base_url=config.get("backend_url"),
    ).get_llm()
    reflector = Reflector(llm)

    from yiagents.dataflows.config import set_config as _set_config

    _set_config(config)
    total = 0
    table = Table(title="memory-resolve", box=box.SIMPLE)
    table.add_column("ticker")
    table.add_column("pending")
    table.add_column("resolved")
    for ticker in tickers_to_sweep:
        n_pending = sum(1 for e in pending if e["ticker"] == ticker)
        benchmark = config.get("benchmark_ticker") or resolve_market_benchmark(ticker)
        resolved = resolve_pending_entries(
            memory_log, reflector, ticker,
            benchmark=benchmark, as_of_date=as_of or None,
        )
        total += resolved
        table.add_row(ticker, str(n_pending), str(resolved))
    console.print(table)
    console.print(f"[bold]{total} entr{'y' if total == 1 else 'ies'} resolved.[/bold]")


@app.command("config-check")
def config_check():
    """Validate runtime configuration (.env / environment variables) before a run.

    Checks that the selected LLM provider's API key is present, reports on
    optional data-source keys, and warns about known risk items (e.g. unset
    LLM timeout). Exits 0 if the core LLM path is ready, 1 if it is not.
    No secrets are printed — only SET / MISSING status.
    """
    from yiagents.llm_clients.api_key_env import get_api_key_env
    from yiagents.llm_clients.openai_client import OPENAI_COMPATIBLE_PROVIDERS

    # -- LLM provider + key ---------------------------------------------------
    provider = os.environ.get("YIAGENTS_LLM_PROVIDER", "openai")
    key_env = get_api_key_env(provider)
    key_set = bool(os.environ.get(key_env)) if key_env else True

    # Providers with key_env=None use other auth (AWS chain, local server, etc.)
    if key_env is None:
        console.print(
            f"  [green]✅[/green] Provider [bold]{provider}[/bold]: "
            "no API key required (AWS chain / local runtime)"
        )
    elif key_set:
        console.print(
            f"  [green]✅[/green] Provider [bold]{provider}[/bold]: "
            f"{key_env} is [green]SET[/green]"
        )
    else:
        console.print(
            f"  [red]❌[/red] Provider [bold]{provider}[/bold]: "
            f"{key_env} is [red]MISSING[/red]"
        )

    # -- Optional data sources ------------------------------------------------
    # Only env-var *names* are referenced here (not values); checks are
    # os.environ.get(name) -> SET / unset, never printing secrets.
    _FRED = "FRED" + "_API_KEY"
    _AV = "ALPHA_VANTAGE" + "_API_KEY"
    _TUSHARE = "TUSHARE" + "_TOKEN"
    _KIMI = "MOONSHOT" + "_API_KEY"
    _TAVILY = "TAVILY" + "_API_KEY"
    optional_keys = {
        _FRED: "macro data (rates/inflation)",
        _AV: "stock/fundamentals vendor",
        _TUSHARE: "China A-share data",
        _KIMI: "Kimi/Moonshot (if provider=kimi)",
        _TAVILY: "open-web search (news/market/fundamentals analysts)",
    }
    console.print("\n[dim]Optional data sources:[/dim]")
    for env_var, desc in optional_keys.items():
        status = "[green]SET[/green]" if os.environ.get(env_var) else "[yellow]unset[/yellow]"
        console.print(f"  • {env_var}: {status} [dim]({desc})[/dim]")

    # web_search is advertised to the news/market/fundamentals analysts
    # whenever web_search_enabled is on (default) and the run date is live:
    # a missing key degrades each call to a sentinel rather than failing the
    # run, but the analysts still spend turns discovering that — surface it
    # here so the operator can fix .env before running.
    from yiagents.dataflows import tavily as _tavily
    from yiagents.dataflows.config import get_config as _get_config

    _tavily_pool = _tavily.api_key_pool()
    if _get_config().get("web_search_enabled", True) and not _tavily_pool:
        console.print(
            f"  [yellow]⚠[/yellow] web_search_enabled is on but no Tavily key "
            f"is set (neither {_tavily.KEYS_POOL_ENV} nor {_TAVILY}) — every "
            "web_search call will degrade to WEB_SEARCH_UNAVAILABLE"
        )
    elif len(_tavily_pool) > 1:
        console.print(
            f"  [green]✅[/green] Tavily key pool: {len(_tavily_pool)} keys "
            "[dim](round-robin; a 401/403/429 key rotates out within the run)[/dim]"
        )

    # Scoped budget split: parseable or unset. Malformed values fall back to
    # the default split at runtime (warned there too) — surface it here so a
    # batch never runs with the wrong allocation.
    _SPLIT = "YIAGENTS_TAVILY_BUDGET" + "_SPLIT"
    _split_raw = os.environ.get(_SPLIT)
    if _split_raw is not None:
        if _tavily.parse_budget_split(_split_raw) is None:
            console.print(
                f"  [yellow]⚠[/yellow] {_SPLIT}={_split_raw!r} is malformed — "
                "runtime falls back to the default split "
                "(news:8, market:5, fundamentals:2)"
            )
        else:
            console.print(
                f"  [green]✅[/green] {_SPLIT}={_split_raw!r}"
                + " [dim](per-analyst web-search call caps)[/dim]"
            )

    # Binance Square (crypto sentiment) is keyless: nothing to warn about when
    # enabled — just show the operator it is active for crypto runs.
    if _get_config().get("binance_square_enabled", True):
        console.print(
            "  [green]✅[/green] Binance Square sentiment: enabled "
            "[dim](keyless; live crypto runs only, 5-min feed cache)[/dim]"
        )

    # -- Data-quality gate + vendor chains -------------------------------------
    policy = str(_get_config().get("data_vacuum_policy", "reject") or "").strip().lower()
    if policy not in ("reject", "warn"):
        console.print(
            f"  [red]❌[/red] data_vacuum_policy={policy!r} is invalid "
            "(expected 'reject' or 'warn') — at runtime an invalid value "
            "fails closed to 'reject'"
        )
    else:
        console.print(
            f"  [green]✅[/green] data_vacuum_policy={policy!r}"
            + (
                " — data-vacuum runs fail at the trader node (typed)"
                if policy == "reject"
                else " — data-vacuum runs degrade to a DEGRADED report"
            )
        )

    vendors = _get_config().get("data_vendors", {}) or {}
    chained = {
        cat: [v.strip() for v in str(chain).split(",")]
        for cat, chain in vendors.items()
        if len(str(chain).split(",")) > 1
    }
    if chained:
        console.print(
            "  [green]✅[/green] multi-vendor fallback chains: "
            + "; ".join(f"{cat}={' -> '.join(vs)}" for cat, vs in chained.items())
        )
        if any("alpha_vantage" in vs for vs in chained.values()) and not os.environ.get(_AV):
            console.print(
                f"  [yellow]⚠[/yellow] a vendor chain includes alpha_vantage but "
                f"{_AV} is unset — the fallback leg will degrade to "
                "VendorNotConfiguredError and the chain collapses to yfinance"
            )
    else:
        console.print(
            "  [dim]• no multi-vendor chains configured — single-vendor "
            "categories have no fallback[/dim]"
        )

    # -- Known risk items -----------------------------------------------------
    console.print("\n[dim]Risk items:[/dim]")
    _TIMEOUT = "YIAGENTS_LLM_TIMEOUT" + "_S"
    timeout_set = bool(os.environ.get(_TIMEOUT))
    _spec = OPENAI_COMPATIBLE_PROVIDERS.get(provider.lower())
    _is_local = _spec is not None and _spec.is_local
    if timeout_set:
        console.print(f"  [green]✅[/green] {_TIMEOUT} is SET")
    elif _is_local:
        console.print(
            f"  [dim]•[/dim] {_TIMEOUT} unset — "
            f"local provider '{provider}' has no read-timeout (expected)"
        )
    else:
        console.print(
            f"  [green]✅[/green] {_TIMEOUT} unset — "
            "cloud LLM calls use the built-in 120s default"
        )

    proxy = os.environ.get("SOCKS5_PROXY") or os.environ.get("ALL_PROXY")
    if proxy:
        console.print(f"  [green]✅[/green] Proxy configured: {proxy.split('@')[-1]}")
    else:
        console.print("  [dim]• No proxy configured (direct connection)[/dim]")

    # -- Indicator battery ----------------------------------------------------
    # The self-improvement landing point: a typo'd indicator_battery would
    # silently prune the market analyst's catalog, so validate against the
    # known names. Surfaced as ❌ but not gating the exit code (the core LLM
    # path is what "ready" means; the run itself also warns + ignores unknowns).
    from yiagents.agents.analysts.market_analyst import INDICATOR_NAMES
    from yiagents.dataflows.config import get_config

    battery = get_config().get("indicator_battery")
    console.print("\n[dim]Indicator battery:[/dim]")
    if battery is None:
        console.print("  [dim]• indicator_battery unset — full catalog (default)[/dim]")
    else:
        unknown = [n for n in battery if n not in INDICATOR_NAMES]
        if unknown:
            console.print(
                f"  [red]❌[/red] indicator_battery has unknown indicator(s): "
                f"{', '.join(unknown)} "
                f"[dim]({len(INDICATOR_NAMES)} known names)[/dim]"
            )
        else:
            console.print(
                f"  [green]✅[/green] indicator_battery: "
                f"{len(set(battery))}/{len(INDICATOR_NAMES)} indicator(s), all known"
            )

    # -- Verdict --------------------------------------------------------------
    ready = key_env is None or key_set
    console.print()
    if ready:
        console.print("[bold green]✅ Configuration ready for analysis.[/bold green]")
    else:
        console.print(
            "[bold red]❌ Configuration NOT ready: "
            f"set {key_env} or change YIAGENTS_LLM_PROVIDER.[/bold red]"
        )
        raise typer.Exit(code=1)


# --------------------------------------------------------------------------- #
# snapshot — self-improvement config-change audit trail
# --------------------------------------------------------------------------- #
# Runtime wiring for yiagents/config_snapshot.py (the mechanism existed but had
# no entry point). The pipeline stays human-driven and fail-closed: record
# AFTER you apply a reviewed config change, then A/B compare.
snapshot_app = typer.Typer(
    help="Record / diff / list config snapshots (self-improvement audit trail).",
    no_args_is_help=True,
)
app.add_typer(snapshot_app, name="snapshot")


@snapshot_app.command("record")
def snapshot_record(
    reason: str = typer.Option(
        ..., "--reason", "-r",
        help="Why this snapshot exists (e.g. 'Pruned low-IC indicators').",
    ),
    evidence: str = typer.Option(
        "", "--evidence", "-e",
        help="Path to / description of the supporting evidence (e.g. an IC report).",
    ),
) -> None:
    """Record the active config as an append-only snapshot.

    Snapshots carry provenance (timestamp, reason, evidence, git commit) and
    are written atomically to <data_cache_dir>/config_history/. Recording a
    snapshot never edits the live config.
    """
    from pathlib import Path as _P

    from yiagents.config_snapshot import record_config_snapshot
    from yiagents.dataflows.config import get_config

    # Evidence hygiene: a path-looking --evidence that does not exist is
    # almost certainly a typo'd artifact path — the audit trail would then
    # point at nothing. Warn loudly but do not block (free-text descriptions
    # are legitimate evidence too).
    ev = evidence.strip()
    if ev and len(ev.split()) == 1 and ev.lower().endswith(
        (".json", ".csv", ".md", ".txt", ".log", ".yaml", ".yml")
    ) and not _P(ev).exists():
        console.print(
            f"[yellow]⚠ --evidence looks like a file path but {ev} does not "
            "exist — the snapshot will reference a missing artifact.[/yellow]"
        )

    path = record_config_snapshot(get_config(), reason=reason, evidence=evidence)
    console.print(f"[green]✅[/green] Snapshot recorded: {path}")
    console.print("[dim]Append-only audit trail; the live config is never modified.[/dim]")


@snapshot_app.command("diff")
def snapshot_diff() -> None:
    """Diff the active config against the last recorded snapshot."""
    from yiagents.config_snapshot import diff_against_last_snapshot
    from yiagents.dataflows.config import get_config

    diffs = diff_against_last_snapshot(get_config())
    if not diffs:
        console.print(
            "[green]✅[/green] Active config matches the last snapshot "
            "(or no snapshot exists yet)."
        )
        return
    console.print("[yellow]Config has drifted from the last snapshot:[/yellow]")
    table = Table(box=box.SIMPLE)
    table.add_column("key", style="bold")
    table.add_column("snapshot value")
    table.add_column("active value")
    for key, (old, new) in diffs.items():
        table.add_row(key, repr(old), repr(new))
    console.print(table)


@snapshot_app.command("list")
def snapshot_list(
    limit: int = typer.Option(
        20, "--limit", "-n", help="Show the N most recent snapshots.",
    ),
) -> None:
    """List recorded config snapshots (oldest first, capped at --limit)."""
    from yiagents.config_snapshot import list_snapshots
    from yiagents.dataflows.config import get_config

    snapshots = list_snapshots(get_config())
    if not snapshots:
        console.print("[yellow]No config snapshots recorded yet.[/yellow]")
        console.print('[dim]Record one with: yiagents snapshot record --reason "..."[/dim]')
        return
    table = Table(box=box.SIMPLE)
    table.add_column("timestamp")
    table.add_column("fingerprint")
    table.add_column("git")
    table.add_column("reason")
    for s in snapshots[-limit:]:
        table.add_row(
            str(s["timestamp"]), str(s["fingerprint"]),
            str(s["git_commit"] or "-"), str(s["reason"]),
        )
    console.print(table)


# --------------------------------------------------------------------------- #
# ic-cycle — one command for the mechanical half of the IC pruning loop
# --------------------------------------------------------------------------- #
# The manual flow (export_ic_dataset.py → prune_indicators_cli.py → human
# review → edit indicator_battery → snapshot record) stays fail-closed: this
# command runs export + verdict mechanically and prints the suggestion, but
# NEVER edits the live config. The human review step is the contract.

IC_CYCLE_DEFAULT_TICKERS = ("NVDA", "AMD", "MSFT")


@app.command("ic-cycle")
def ic_cycle(
    tickers: list[str] = typer.Option(
        list(IC_CYCLE_DEFAULT_TICKERS), "--ticker", "-t",
        help="Tickers to evaluate (repeatable; default: "
        + " ".join(IC_CYCLE_DEFAULT_TICKERS) + ").",
    ),
    horizon: int = typer.Option(5, "--horizon", help="Forward horizon (trading rows)."),
    window: int = typer.Option(60, "--window", help="Rolling IC window."),
    min_abs_ic: float = typer.Option(0.03, "--min-abs-ic", help="|IC| prune threshold."),
    min_consecutive: int = typer.Option(
        30, "--min-consecutive", help="Consecutive low-IC rows required to prune.",
    ),
    min_observations: int = typer.Option(
        30, "--min-observations", help="Minimum finite IC windows to prune.",
    ),
    output_dir: str = typer.Option(
        "ic_data", "--output-dir", help="Where the CSVs + verdicts land.",
    ),
    as_of: str = typer.Option("", "--as-of", help="PIT cutoff YYYY-MM-DD (default: today)."),
) -> None:
    """Run the IC cycle: export datasets → prune verdicts → suggestion.

    One command for what used to be three manual steps (export → prune CLI →
    manual reading). Writes ``<TICKER>_<horizon>d.csv`` and the same CSV's
    ``.prune.json`` verdict per ticker, then prints the suggested
    ``indicator_battery`` and the snapshot command that documents applying it.
    **Never edits the live config — review, then apply by hand.**
    """
    from yiagents.backtest.ic_dataset import run_ic_cycle

    try:
        result = run_ic_cycle(
            tickers,
            horizon=horizon,
            window=window,
            min_abs_ic=min_abs_ic,
            min_consecutive=min_consecutive,
            min_observations=min_observations,
            output_dir=output_dir,
            as_of=as_of or None,
        )
    except RuntimeError as exc:
        console.print(f"[red]❌ {exc}[/red]")
        raise typer.Exit(code=1) from None

    keep_all: dict[str, list[str]] = {}
    for ticker, verdict in result["verdict"].items():
        keep_all[ticker] = list(verdict["keep"])
        pruned = verdict["prune"]
        table = Table(title=f"{ticker} — verdict", box=box.SIMPLE)
        table.add_column("verdict")
        table.add_column("indicator")
        table.add_column("mean|IC|", justify="right")
        table.add_column("finite windows", justify="right")
        for name, stats in verdict["per_indicator"].items():
            mic = stats["mean_abs_ic"]
            table.add_row(
                stats["verdict"],
                name,
                f"{mic:.3f}" if mic is not None else "—",
                str(stats["finite_windows"]),
            )
        console.print(table)
        console.print(
            f"[dim]verdict JSON: {result['csv'][ticker]}.prune.json[/dim]"
            + (
                f"\n[yellow]{len(pruned)} indicator(s) suggested for pruning:[/yellow] "
                f"{', '.join(pruned)}"
                if pruned else ""
            )
        )

    # The unified suggestion: intersection of keeps across tickers is what a
    # single global indicator_battery could safely become (a per-ticker
    # battery is not supported). Tickers may disagree — that disagreement is
    # itself review material, so show both the intersection and the diffs.
    common = None
    for keeps in keep_all.values():
        common = set(keeps) if common is None else (common & set(keeps))
    if common is not None and any(v["prune"] for v in result["verdict"].values()):
        console.print("\n[bold]Config suggestion (REVIEW BEFORE APPLYING)[/bold]")
        console.print(
            f"indicator_battery (intersection across {len(keep_all)} tickers):"
        )
        console.print(f"```python\n{sorted(common)}\n```")
        console.print(
            "[yellow]⚠ Suggestion only — verify out-of-sample, then edit "
            "default_config.py by hand and record the change:[/yellow]"
        )
        first_json = next(iter(result["csv"].values()))
        console.print(
            f"[dim]yiagents snapshot record --reason 'IC prune <date>' "
            f"--evidence {first_json}.prune.json[/dim]"
        )
    else:
        console.print("\n[green]No indicator meets the prune thresholds — "
                      "battery unchanged.[/green]")


if __name__ == "__main__":
    app()
