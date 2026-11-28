"""Round 17 — ``yialpha.cli.main``: run_analysis save-decorators, live display,
interactive run orchestration.

Everything here is driven with recording fakes: the Typer/questionary prompts,
the LLM graph, the Rich ``Live`` context and the network are never touched.
"""
from __future__ import annotations

import datetime
import io
import itertools
import os
import re
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import Annotated, Any, TypedDict

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from rich.console import Console
from rich.layout import Layout

import yialpha.batch.runner as batch_runner
import yialpha.dataflows.run_scope as run_scope
import yialpha.dataflows.tavily as tavily_vendor
import yialpha.dataflows.utils as dataflows_utils
import yialpha.graph.analyst_execution as analyst_execution
from yialpha.cli import main as cli_main
from yialpha.graph.analyst_execution import AnalystWallTimeTracker
from yialpha.graph.propagation import Propagator

SELECTIONS: dict[str, Any] = {
    "ticker": "DEMO",
    "analysis_date": "2024-06-12",
    "analysts": ["market"],
    "asset_type": "stock",
}


class _StopBeforeDisplay(Exception):
    """Raised from the patched ``create_layout``: ``run_analysis`` stops right
    after the three save-decorators were defined and installed on the buffer."""


class _Stub:
    """Stand-in for the collaborators constructed before the decorators."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.init_args = (args, kwargs)


def _run_analysis_frame(exc: BaseException):
    """The ``run_analysis`` frame the sentinel escaped from."""
    tb = exc.__traceback__
    code = cli_main.run_analysis.__code__
    while tb is not None and tb.tb_frame.f_code is not code:
        tb = tb.tb_next
    assert tb is not None, "sentinel did not pass through run_analysis"
    return tb.tb_frame


def _decorator_env(monkeypatch, tmp_path) -> SimpleNamespace:
    """Drive ``run_analysis`` up to the live display and hand back the three
    nested decorator factories with the closure state they captured
    (``log_file`` / ``report_dir``) and the buffer they were installed on."""
    monkeypatch.setattr(
        cli_main, "get_user_selections", lambda asset_type_override="auto": dict(SELECTIONS)
    )
    monkeypatch.setattr(
        cli_main,
        "_build_run_config",
        lambda selections, checkpoint: {"results_dir": str(tmp_path / "results")},
    )
    monkeypatch.setattr(cli_main, "StatsCallbackHandler", _Stub)
    monkeypatch.setattr(cli_main, "resolve_run_analysts", lambda *a, **k: ["market"])
    monkeypatch.setattr(cli_main, "build_analyst_execution_plan", lambda keys: {"plan": keys})
    monkeypatch.setattr(cli_main, "AnalystWallTimeTracker", _Stub)
    monkeypatch.setattr(cli_main, "YiAlphaGraph", _Stub)
    # run_analysis rebinds the buffer's methods: never let that touch the
    # module-level singleton another test may be looking at.
    monkeypatch.setattr(cli_main, "message_buffer", cli_main.MessageBuffer())
    monkeypatch.delenv("YIALPHA_DATA_VACUUM_POLICY", raising=False)

    def _stop():
        raise _StopBeforeDisplay

    monkeypatch.setattr(cli_main, "create_layout", _stop)

    with pytest.raises(_StopBeforeDisplay) as info:
        cli_main.run_analysis(checkpoint=False)

    local = _run_analysis_frame(info.value).f_locals
    return SimpleNamespace(
        save_message_decorator=local["save_message_decorator"],
        save_tool_call_decorator=local["save_tool_call_decorator"],
        save_report_section_decorator=local["save_report_section_decorator"],
        log_file=local["log_file"],
        report_dir=local["report_dir"],
        buffer=cli_main.message_buffer,
    )


class _FakeBuffer:
    """Stand-in for MessageBuffer: the wrapped methods append to the same
    attributes the real ones do and record how they were called."""

    def __init__(self) -> None:
        self.messages: list[tuple] = []
        self.tool_calls: list[tuple] = []
        self.report_sections: dict[str, Any] = {}
        self.calls: list[tuple[str, tuple, dict]] = []

    def add_message(self, message_type, content, **extra):
        self.calls.append(("add_message", (message_type, content), extra))
        self.messages.append(("12:01:02", message_type, content))

    def add_tool_call(self, tool_name, args, **extra):
        self.calls.append(("add_tool_call", (tool_name, args), extra))
        self.tool_calls.append(("12:02:03", tool_name, args))

    def update_report_section(self, section_name, content, **extra):
        self.calls.append(("update_report_section", (section_name, content), extra))
        self.report_sections[section_name] = content


# --------------------------------------------------------------------------- #
# format_tokens
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("count", "expected"),
    [
        (0, "0"),
        (999, "999"),
        (1000, "1.0k"),
        (1500, "1.5k"),
        (123_456, "123.5k"),
        (1_234_567, "1234.6k"),
    ],
)
def test_r17_format_tokens_threshold_and_scaling(count, expected):
    """Below 1000 the raw count is shown; from 1000 it is one-decimal 'k'."""
    assert cli_main.format_tokens(count) == expected


# --------------------------------------------------------------------------- #
# run_analysis save-decorators (nested closures, lines 1292-1326)
# --------------------------------------------------------------------------- #


def test_r17_save_decorators_installed_on_buffer_with_wrapped_metadata(monkeypatch, tmp_path):
    """The three wrappers replace the buffer's methods, and ``@wraps`` keeps the
    method name and points ``__wrapped__`` at the original bound method."""
    env = _decorator_env(monkeypatch, tmp_path)

    for name in ("add_message", "add_tool_call", "update_report_section"):
        installed = getattr(env.buffer, name)
        original = getattr(cli_main.MessageBuffer, name)
        assert installed.__name__ == name
        assert installed.__wrapped__.__func__ is original
        assert installed.__wrapped__.__self__ is env.buffer

    # Closure state: the log sits next to the reports dir of this run.
    assert env.log_file == tmp_path / "results" / "DEMO" / "2024-06-12" / "message_tool.log"
    assert env.report_dir == tmp_path / "results" / "DEMO" / "2024-06-12" / "reports"
    assert env.log_file.exists()


def test_r17_save_message_decorator_appends_last_message_flattened(monkeypatch, tmp_path):
    """The wrapper forwards every argument, then appends the message the wrapped
    method stored, with newlines turned into spaces."""
    env = _decorator_env(monkeypatch, tmp_path)
    obj = _FakeBuffer()
    wrapper = env.save_message_decorator(obj, "add_message")

    env.log_file.write_text("10:00:00 [System] earlier line\n", encoding="utf-8")
    result = wrapper("Analyst", "line one\nline two\n", ignored="kwarg is forwarded")

    assert result is None
    assert obj.calls == [
        ("add_message", ("Analyst", "line one\nline two\n"), {"ignored": "kwarg is forwarded"})
    ]
    assert env.log_file.read_text(encoding="utf-8") == (
        "10:00:00 [System] earlier line\n12:01:02 [Analyst] line one line two \n"
    )


def test_r17_save_message_decorator_reraises_wrapped_error_without_logging(monkeypatch, tmp_path):
    """An exception from the wrapped method propagates unchanged and nothing is
    logged for the call that failed."""
    env = _decorator_env(monkeypatch, tmp_path)
    boom = ValueError("buffer rejected the message")

    class _Refusing:
        def add_message(self, message_type, content):
            raise boom

    wrapper = env.save_message_decorator(_Refusing(), "add_message")

    with pytest.raises(ValueError) as info:
        wrapper("Analyst", "never stored")

    assert info.value is boom
    assert env.log_file.read_text(encoding="utf-8") == ""


def test_r17_save_tool_call_decorator_writes_kwargs_of_last_call(monkeypatch, tmp_path):
    """Tool calls land as ``HH:MM:SS [Tool Call] name(k=v, ...)``."""
    env = _decorator_env(monkeypatch, tmp_path)
    obj = _FakeBuffer()
    wrapper = env.save_tool_call_decorator(obj, "add_tool_call")

    assert wrapper("get_price", {"ticker": "DEMO", "days": 5}) is None
    wrapper("think", {})

    assert obj.calls == [
        ("add_tool_call", ("get_price", {"ticker": "DEMO", "days": 5}), {}),
        ("add_tool_call", ("think", {}), {}),
    ]
    assert env.log_file.read_text(encoding="utf-8") == (
        "12:02:03 [Tool Call] get_price(ticker=DEMO, days=5)\n"
        "12:02:03 [Tool Call] think()\n"
    )


@pytest.mark.parametrize(
    ("stored", "expected_text"),
    [
        ("### Bull Researcher Analysis\nbody", "### Bull Researcher Analysis\nbody"),
        (["first line", 42], "first line\n42"),
    ],
)
def test_r17_save_report_section_decorator_writes_markdown_file(
    monkeypatch, tmp_path, stored, expected_text
):
    """The file holds what the wrapped method left in ``report_sections`` (not
    the argument), a list is joined with newlines, and 'w' replaces the file."""
    env = _decorator_env(monkeypatch, tmp_path)

    class _SectionBuffer:
        def __init__(self) -> None:
            self.report_sections: dict[str, Any] = {"investment_plan": None}
            self.seen: list[tuple] = []

        def update_report_section(self, section_name, content):
            self.seen.append((section_name, content))
            self.report_sections[section_name] = stored

    obj = _SectionBuffer()
    wrapper = env.save_report_section_decorator(obj, "update_report_section")
    target = env.report_dir / "investment_plan.md"
    target.write_text("stale content from an earlier update", encoding="utf-8")

    assert wrapper("investment_plan", "argument passed through") is None

    assert obj.seen == [("investment_plan", "argument passed through")]
    assert target.read_text(encoding="utf-8") == expected_text


@pytest.mark.parametrize("stored", [None, ""])
def test_r17_save_report_section_decorator_skips_empty_section(monkeypatch, tmp_path, stored):
    """A section the wrapped method did not fill (None / empty) writes no file."""

    class _EmptySection:
        report_sections = {"investment_plan": None}

        def update_report_section(self, section_name, content):
            self.report_sections[section_name] = stored

    env = _decorator_env(monkeypatch, tmp_path)
    wrapper = env.save_report_section_decorator(_EmptySection(), "update_report_section")

    wrapper("investment_plan", "ignored")

    assert list(env.report_dir.iterdir()) == []


# --------------------------------------------------------------------------- #
# update_display — the Rich "Live" layout (lines 330-538)
# --------------------------------------------------------------------------- #

_SEP = "─" * 20
_SEP_ROW = f"│ {_SEP} {_SEP} {_SEP} │"
# An empty content cell renders as "│ │" after whitespace is collapsed, so the
# single space before the content is optional.
_TIMED_ROW = re.compile(r"│ (\d\d:\d\d:\d\d) │ (\S+) │ ?(.*?) │$")
_HEAD_ROW = object()
_SPINNER_ROW = object()


class _SpinnerRow:
    """Placeholder like ``_SPINNER_ROW`` for any agent that is in progress:
    its spinner's frame glyph depends on the clock. ``team`` is the Team cell
    (filled on the first row of a team, empty on the rows below)."""

    def __init__(self, agent: str, team: str = "") -> None:
        self.agent = agent
        self.team = team

    @property
    def pattern(self) -> str:
        cells = " ".join(part for part in (self.team, self.agent) if part)
        return rf"│ {cells} \S in_progress │"


def _fresh_buffer(monkeypatch, analysts=("market", "news")):
    """A private MessageBuffer installed as the display's data source."""
    buffer = cli_main.MessageBuffer()
    buffer.init_for_analysis(list(analysts))
    monkeypatch.setattr(cli_main, "message_buffer", buffer)
    return buffer


def _render_lines(renderable, width=120):
    """Plain text of ``renderable`` at a fixed width, whitespace collapsed."""
    console = Console(
        record=True,
        file=io.StringIO(),
        width=width,
        color_system=None,
        legacy_windows=False,
        highlight=False,
    )
    console.print(renderable)
    return [" ".join(line.split()) for line in console.export_text().splitlines()]


def _content_lines(renderable, width=120):
    """The rows between a panel's borders, blank rows dropped."""
    return [
        line for line in _render_lines(renderable, width) if line.startswith("│") and line != "│ │"
    ]


def _assert_progress(renderable, expected):
    """The Progress panel's rows equal ``expected``, top to bottom.

    Two placeholders stand for rows whose text is not fixed: ``_HEAD_ROW`` is
    the rule ``box.SIMPLE_HEAD`` draws under the header (its length follows
    the panel width) and ``_SPINNER_ROW`` is the Bull Researcher row while it
    is in progress (the spinner's frame glyph depends on the clock);
    ``_SpinnerRow(agent)`` is the same for any other agent of its team.
    """
    lines = _content_lines(renderable)
    assert len(lines) == len(expected)
    for got, want in zip(lines, expected, strict=True):
        if want is _HEAD_ROW:
            assert re.fullmatch(r"│ ─+ │", got), got
        elif want is _SPINNER_ROW:
            assert re.fullmatch(r"│ Research Team Bull Researcher \S in_progress │", got)
        elif isinstance(want, _SpinnerRow):
            assert re.fullmatch(want.pattern, got), got
        else:
            assert got == want


def _panel_title(renderable):
    return _render_lines(renderable)[0].strip("─╭╮ ")


def _message_rows(layout):
    """(time, type, content) of every row the Messages & Tools panel shows.

    Rendered 400 columns wide so a 200-character message stays on one line.
    """
    return [
        match.groups()
        for line in _content_lines(layout["messages"].renderable, width=400)
        if (match := _TIMED_ROW.match(line))
    ]


class _Stats:
    """StatsCallbackHandler stand-in returning a canned get_stats() dict."""

    def __init__(self, **stats: Any) -> None:
        self.stats = {
            "llm_calls": 0,
            "tool_calls": 0,
            "tokens_in": 0,
            "tokens_out": 0,
            **stats,
        }

    def get_stats(self) -> dict[str, Any]:
        return dict(self.stats)


def _footer(layout) -> str:
    (line,) = _content_lines(layout["footer"].renderable)
    assert line.startswith("│ ") and line.endswith(" │")
    return line[2:-2]


_FRESH_PROGRESS = [
    "│ Team Agent Status │",
    _HEAD_ROW,
    "│ Analyst Team Market Analyst pending │",
    "│ News Analyst pending │",
    _SEP_ROW,
    "│ Research Team Bull Researcher pending │",
    "│ Bear Researcher pending │",
    "│ Research Manager pending │",
    _SEP_ROW,
    "│ Trading Team Trader pending │",
    _SEP_ROW,
    "│ Risk Management Aggressive Analyst pending │",
    "│ Neutral Analyst pending │",
    "│ Conservative Analyst pending │",
    _SEP_ROW,
    "│ Portfolio Management Portfolio Manager pending │",
    _SEP_ROW,
]


def test_r17_update_display_fresh_run_renders_every_region(monkeypatch):
    """Before the first chunk: every agent pending, an empty message table, the
    waiting panel, and a footer without LLM/tool/token/elapsed parts."""
    _fresh_buffer(monkeypatch)
    layout = cli_main.create_layout()

    cli_main.update_display(layout)

    assert _panel_title(layout["header"].renderable) == "Welcome to YiAlpha"
    assert _content_lines(layout["header"].renderable) == ["│ Welcome to YiAlpha CLI │"]
    assert _panel_title(layout["progress"].renderable) == "Progress"
    _assert_progress(layout["progress"].renderable, _FRESH_PROGRESS)
    assert _panel_title(layout["messages"].renderable) == "Messages & Tools"
    assert [line for line in _render_lines(layout["messages"].renderable) if "Time" in line] == [
        "│ Time │ Type │ Content │"
    ]
    assert _message_rows(layout) == []
    assert _panel_title(layout["analysis"].renderable) == "Current Report"
    assert _content_lines(layout["analysis"].renderable) == [
        "│ Waiting for analysis report... │"
    ]
    assert _footer(layout) == "Agents: 0/10 | Reports: 0/5"


def test_r17_update_display_only_lists_teams_with_a_pending_or_running_agent(monkeypatch):
    """A one-analyst run (market) lists the Market Analyst only; a perp run that
    carries the positioning analyst gets its row right after Market."""
    buffer = _fresh_buffer(monkeypatch, analysts=("market", "positioning"))
    buffer.agent_status.pop("Conservative Analyst")
    layout = cli_main.create_layout()

    cli_main.update_display(layout)

    _assert_progress(
        layout["progress"].renderable,
        [
            "│ Team Agent Status │",
            _HEAD_ROW,
            "│ Analyst Team Market Analyst pending │",
            "│ Positioning Analyst pending │",
            _SEP_ROW,
            "│ Research Team Bull Researcher pending │",
            "│ Bear Researcher pending │",
            "│ Research Manager pending │",
            _SEP_ROW,
            "│ Trading Team Trader pending │",
            _SEP_ROW,
            "│ Risk Management Aggressive Analyst pending │",
            "│ Neutral Analyst pending │",
            _SEP_ROW,
            "│ Portfolio Management Portfolio Manager pending │",
            _SEP_ROW,
        ],
    )


def test_r17_update_display_progress_statuses_and_spinner(monkeypatch):
    """Status cells: completed stays text, in_progress becomes a spinner (its
    frame glyph is clock-dependent), anything unknown is shown verbatim."""
    buffer = _fresh_buffer(monkeypatch)
    buffer.agent_status.update(
        {
            "Market Analyst": "completed",
            "Bull Researcher": "in_progress",
            "News Analyst": "error",
            "Trader": "paused",
        }
    )
    layout = cli_main.create_layout()

    cli_main.update_display(layout)

    _assert_progress(
        layout["progress"].renderable,
        [
            "│ Team Agent Status │",
            _HEAD_ROW,
            "│ Analyst Team Market Analyst completed │",
            "│ News Analyst error │",
            _SEP_ROW,
            _SPINNER_ROW,
            "│ Bear Researcher pending │",
            "│ Research Manager pending │",
            _SEP_ROW,
            "│ Trading Team Trader paused │",
            _SEP_ROW,
            "│ Risk Management Aggressive Analyst pending │",
            "│ Neutral Analyst pending │",
            "│ Conservative Analyst pending │",
            _SEP_ROW,
            "│ Portfolio Management Portfolio Manager pending │",
            _SEP_ROW,
        ],
    )


def test_r17_update_display_messages_newest_first_truncated_and_capped(monkeypatch):
    """Tool calls and messages share one newest-first table; a message is cut
    to 200 characters, tool args to 80, and only the 12 newest rows are shown."""
    buffer = _fresh_buffer(monkeypatch)
    buffer.tool_calls.append(("10:00:03", "get_price", {"ticker": "DEMO", "days": 5}))
    buffer.tool_calls.append(("10:00:00", "long_args", {"x": "y" * 100}))
    buffer.messages.append(("10:00:05", "Analyst", "x" * 250))
    buffer.messages.append(("10:00:04", "System", None))
    for minute in range(1, 13):
        buffer.messages.append((f"09:{minute:02d}:00", "System", f"old {minute}"))
    layout = cli_main.create_layout()

    cli_main.update_display(layout)

    assert _message_rows(layout) == [
        ("10:00:05", "Analyst", "x" * 197 + "..."),
        ("10:00:04", "System", ""),
        ("10:00:03", "Tool", "get_price: {'ticker': 'DEMO', 'days': 5}"),
        ("10:00:00", "Tool", "long_args: {'x': '" + "y" * 70 + "..."),
        *[
            (f"09:{minute:02d}:00", "System", f"old {minute}")
            for minute in range(12, 4, -1)
        ],
    ]
    # Same-second rows keep their arrival order (tool call, then message).
    buffer.tool_calls.clear()
    buffer.messages.clear()
    buffer.tool_calls.append(("11:00:00", "think", {}))
    buffer.messages.append(("11:00:00", "System", "after the tool call"))
    cli_main.update_display(layout)
    assert _message_rows(layout) == [
        ("11:00:00", "Tool", "think: {}"),
        ("11:00:00", "System", "after the tool call"),
    ]


def test_r17_update_display_messages_newest_first_across_midnight(monkeypatch):
    """Regression: rows are ordered by their wall-clock "HH:MM:SS" stamp.

    Sorted as text, everything stamped after 00:00:00 ranks below the
    pre-midnight rows, so once 12 pre-midnight rows exist the panel keeps
    showing those and never shows a later row for the rest of the run.
    """
    buffer = _fresh_buffer(monkeypatch)
    for second in range(50, 60):
        buffer.tool_calls.append((f"23:59:{second:02d}", "tool", {}))
        buffer.messages.append((f"23:59:{second:02d}", "System", f"late {second}"))
    buffer.messages.append(("00:00:01", "System", "first message after midnight"))
    layout = cli_main.create_layout()

    cli_main.update_display(layout)

    # 21 rows exist; the panel shows the 12 newest, so 23:59:54 keeps only its
    # tool call (which arrived before that second's message).
    expected = [("00:00:01", "System", "first message after midnight")]
    for second in range(59, 54, -1):
        stamp = f"23:59:{second:02d}"
        expected += [(stamp, "Tool", "tool: {}"), (stamp, "System", f"late {second}")]
    expected += [("23:59:54", "Tool", "tool: {}")]
    assert _message_rows(layout) == expected


@pytest.mark.parametrize(
    ("stats", "elapsed", "expected"),
    [
        (
            {"llm_calls": 5, "tool_calls": 2, "tokens_in": 1200, "tokens_out": 345},
            125.7,
            "Agents: 0/10 | LLM: 5 | Tools: 2 | Tokens: 1.2k↑ 345↓ | Reports: 0/5 | ⏱ 02:05",
        ),
        (
            {"llm_calls": 3, "tool_calls": 0, "tokens_in": 0, "tokens_out": 0},
            0.0,
            "Agents: 0/10 | LLM: 3 | Tools: 0 | Tokens: -- | Reports: 0/5 | ⏱ 00:00",
        ),
        (
            {"llm_calls": 1, "tool_calls": 1, "tokens_in": 0, "tokens_out": 7},
            None,
            "Agents: 0/10 | LLM: 1 | Tools: 1 | Tokens: 0↑ 7↓ | Reports: 0/5",
        ),
        (
            {"llm_calls": 1, "tool_calls": 1, "tokens_in": 2_500_000, "tokens_out": 999_999},
            3599.9,
            "Agents: 0/10 | LLM: 1 | Tools: 1 | Tokens: 2500.0k↑ 1000.0k↓ | Reports: 0/5"
            " | ⏱ 59:59",
        ),
        (
            {"llm_calls": 1, "tool_calls": 1, "tokens_in": 1_000, "tokens_out": 0},
            3661.2,
            "Agents: 0/10 | LLM: 1 | Tools: 1 | Tokens: 1.0k↑ 0↓ | Reports: 0/5 | ⏱ 61:01",
        ),
    ],
)
def test_r17_update_display_footer_stats_tokens_and_elapsed(monkeypatch, stats, elapsed, expected):
    """Footer format: LLM / Tools counts, one-decimal 'k' tokens (in↑ out↓,
    '--' until the first usage arrives), elapsed as MM:SS (None start_time =
    no clock)."""
    buffer = _fresh_buffer(monkeypatch)
    layout = cli_main.create_layout()
    start_time = None
    if elapsed is not None:
        start_time = 1_000.0
        monkeypatch.setattr(cli_main, "time", SimpleNamespace(time=lambda: 1_000.0 + elapsed))
    assert cli_main.message_buffer is buffer  # the panel reads this buffer

    cli_main.update_display(layout, stats_handler=_Stats(**stats), start_time=start_time)

    assert _footer(layout) == expected


def test_r17_update_display_finished_run_counts_and_report_panel(monkeypatch):
    """End of a run: every agent completed, the reports whose finalizing agent
    is done count as complete, and the panel shows the current report as
    Markdown (its '###' heading is rendered as a title, not as raw text)."""
    buffer = _fresh_buffer(monkeypatch)
    layout = cli_main.create_layout()
    buffer.update_report_section("market_report", "MARKET BODY")
    buffer.update_report_section("investment_plan", "debate so far")
    buffer.update_agent_status("Market Analyst", "completed")
    cli_main.update_display(layout)
    assert _footer(layout) == "Agents: 1/10 | Reports: 1/5"

    for agent in buffer.agent_status:
        buffer.update_agent_status(agent, "completed")
    cli_main.update_display(layout)

    assert _footer(layout) == "Agents: 10/10 | Reports: 2/5"
    assert _panel_title(layout["analysis"].renderable) == "Current Report"
    assert _content_lines(layout["analysis"].renderable) == [
        "│ Research Team Decision │",
        "│ debate so far │",
    ]
    _assert_progress(
        layout["progress"].renderable,
        [
            line.replace("pending", "completed") if isinstance(line, str) else line
            for line in _FRESH_PROGRESS
        ],
    )


def _whole_layout_text(monkeypatch, height):
    """The complete layout of a freshly started run, rendered in a 120-column
    console of ``height`` rows."""
    _fresh_buffer(monkeypatch)
    monkeypatch.setattr(cli_main, "time", SimpleNamespace(time=lambda: 1_125.7))
    layout = cli_main.create_layout()

    cli_main.update_display(
        layout,
        "Analyzing DEMO...",
        stats_handler=_Stats(llm_calls=1, tool_calls=1, tokens_in=1200, tokens_out=345),
        start_time=1_000.0,
    )

    console = Console(
        record=True,
        file=io.StringIO(),
        width=120,
        height=height,
        color_system=None,
        legacy_windows=False,
        highlight=False,
    )
    console.print(layout)
    return " ".join(console.export_text().split())


def test_r17_update_display_whole_layout_renders_inside_a_console(monkeypatch):
    """The five regions of the layout create_layout() builds render together in
    one pass. At 120 columns the Progress cells wrap, so its rows are matched
    word by word."""
    text = _whole_layout_text(monkeypatch, height=40)

    for visible in (
        "Welcome to YiAlpha",
        "Progress",
        "Analyst Market pending",
        "News pending",
        "Messages & Tools",
        "Waiting for analysis report...",
        "Agents: 0/10 | LLM: 1 | Tools: 1 | Tokens: 1.2k↑ 345↓ | Reports: 0/5 | ⏱ 02:05",
    ):
        assert visible in text


@pytest.mark.parametrize("height", [24, 40, 80])
def test_r17_create_layout_header_welcome_text_is_visible(monkeypatch, height):
    """Regression: the header region is fixed at 3 rows, while update_display
    filled it with a Panel padded by one blank line above and below the text
    (5 rows). The layout cropped the middle, so the welcome text was never
    shown at any terminal height -- only the titled top border and the bottom
    one were.
    """
    text = _whole_layout_text(monkeypatch, height=height)

    assert "Welcome to YiAlpha CLI" in text


# --------------------------------------------------------------------------- #
# run_analysis — the streaming loop and everything after it
# --------------------------------------------------------------------------- #

_STREAM_SELECTIONS = {
    **SELECTIONS,
    "analysts": [SimpleNamespace(value="market"), SimpleNamespace(value="news")],
}


class _RecordingContext:
    """Context-manager stand-in (run lock, date pin, Rich ``Live``) that logs
    its enter / exit to a shared trace."""

    def __init__(self, trace, details, name, *args, **kwargs):
        self._trace, self.name = trace, name
        self.args = details[name] = (args, kwargs)

    def __enter__(self):
        self._trace.append(f"enter:{self.name}")
        return self

    def __exit__(self, *exc_info):
        self._trace.append(f"exit:{self.name}")
        return False


class _Tracker(AnalystWallTimeTracker):
    """The real tracker; remembers its instance so a test can tell whether
    run_analysis handed it to the graph."""

    last: AnalystWallTimeTracker | None = None

    def __init__(self, plan):
        super().__init__(plan)
        _Tracker.last = self


class _FakeGraph:
    """Stand-in for YiAlphaGraph: records every seam run_analysis touches and
    streams FULL-state chunks, the shape ``stream_mode="values"`` produces."""

    def __init__(
        self, keys, config, debug, callbacks, deltas, trace, details, stream_error=None, full_state_chunks=False
    ):
        trace.append("graph.init")
        details["graph.init"] = (list(keys), config, debug, callbacks)
        self._deltas, self._trace, self._details = deltas, trace, details
        self._stream_error = stream_error
        self._full_state = full_state_chunks
        self._curr_state = "never assigned"
        self._tracker = "never published"
        self.memory_log = SimpleNamespace(
            get_past_context=self._past_context, store_decision=self._store_decision
        )
        self.propagator = SimpleNamespace(
            create_initial_state=self._create_initial_state, get_graph_args=self._get_graph_args
        )
        self.graph = SimpleNamespace(stream=self._stream)

    @property
    def curr_state(self):
        return self._curr_state

    @curr_state.setter
    def curr_state(self, value):
        self._trace.append("assign curr_state")
        self._details["assign curr_state"] = value
        self._curr_state = value

    @property
    def _active_wall_time_tracker(self):
        return self._tracker

    @_active_wall_time_tracker.setter
    def _active_wall_time_tracker(self, value):
        self._trace.append("tracker cleared" if value is None else "tracker published")
        self._details.setdefault("tracker assignments", []).append(value)
        self._tracker = value

    def _note(self, name, args, kwargs):
        self._trace.append(name)
        self._details[name] = (args, kwargs)

    def _past_context(self, *args, **kwargs):
        self._note("get_past_context", args, kwargs)
        return "PAST CONTEXT"

    def _store_decision(self, **kwargs):
        self._note("store_decision", (), kwargs)

    def _create_initial_state(self, *args, **kwargs):
        self._note("create_initial_state", args, kwargs)
        return "INITIAL STATE"

    def _get_graph_args(self, *args, **kwargs):
        self._note("get_graph_args", args, kwargs)
        return {"stream_mode": "values", "config": {"recursion_limit": 7}}

    def _stream(self, *args, **kwargs):
        self._note("stream", args, kwargs)
        return self._chunks()

    def _chunks(self):
        self._trace.append("tracker while streaming")
        self._details["tracker while streaming"] = self._active_wall_time_tracker
        if self._full_state:
            # Chunks come from a real graph already: hand them over untouched.
            yield from self._deltas
        else:
            state = {}
            for delta in self._deltas:
                state = {
                    **state,
                    **delta,
                    "messages": [*state.get("messages", []), *delta.get("messages", [])],
                }
                yield dict(state)
        if self._stream_error is not None:
            raise self._stream_error

    def _bind_run_record_stage(self, *args, **kwargs):
        self._note("_bind_run_record_stage", args, kwargs)
        return "ledger-run-1", "regime-run-1"

    def _resolve_pending_entries(self, *args, **kwargs):
        self._note("_resolve_pending_entries", args, kwargs)

    def resolve_instrument_context(self, *args, **kwargs):
        self._note("resolve_instrument_context", args, kwargs)
        return "INSTRUMENT CONTEXT"

    def _apply_risk_overlay(self, ticker, date, final_state, portfolio_state=None, asset_type=None):
        self._note(
            "_apply_risk_overlay",
            (ticker, date, dict(final_state)),
            {"portfolio_state": portfolio_state, "asset_type": asset_type},
        )
        return {
            **final_state,
            "final_trade_decision": "### Portfolio Manager Decision\nPM JUDGE\n\n**overlay**",
        }

    def _stamp_run_record_ids(self, final_state, ledger_run_id, regime_run_id):
        self._note("_stamp_run_record_ids", (dict(final_state), ledger_run_id, regime_run_id), {})

    def finalize_streamed_run(self, ticker, date, final_state, asset_type=None):
        self._note("finalize_streamed_run", (ticker, date, dict(final_state)), {"asset_type": asset_type})
        return {**final_state, "data_quality": {}}


def _recording_console():
    return Console(
        record=True,
        file=io.StringIO(),
        width=120,
        color_system=None,
        legacy_windows=False,
        highlight=False,
    )


def _streaming_env(
    monkeypatch,
    tmp_path,
    deltas,
    *,
    selections=_STREAM_SELECTIONS,
    stream_error=None,
    answers=("n", "n"),
    save_error=None,
    clock_step=100.0,
    full_state_chunks=False,
):
    """Drive ``run_analysis`` through the streaming loop with scripted chunks.

    The buffer, the decorators, ``update_display`` and the tracker are real;
    the graph, the run lock, the date pin, ``Live``, the two per-run scope
    resets and the Typer prompts are stand-ins that record how they were used.

    ``deltas`` are per-node deltas that the fake graph accumulates into
    full-state chunks, unless ``full_state_chunks`` says they already are the
    chunks of a real graph.
    """
    trace: list[Any] = []
    details: dict[str, Any] = {}
    prompts: list[tuple[Any, Any]] = []
    saved: list[tuple[Any, ...]] = []
    snapshots: list[tuple[Any, dict[str, str], Any]] = []
    console = _recording_console()
    selections = dict(selections)
    config = {"results_dir": str(tmp_path / "results")}

    def _build_config(selections, checkpoint):
        # The env overlay lands on DEFAULT_CONFIG when it is loaded, so this
        # stub stands in for that step: an explicit YIALPHA_DATA_VACUUM_POLICY
        # is already the configured policy, else the shipped default.
        config["data_vacuum_policy"] = os.environ.get("YIALPHA_DATA_VACUUM_POLICY", "reject")
        return config

    monkeypatch.setattr(cli_main, "get_user_selections", lambda asset_type_override="auto": dict(selections))
    monkeypatch.setattr(cli_main, "_build_run_config", _build_config)
    monkeypatch.setattr(cli_main, "StatsCallbackHandler", _Stats)
    monkeypatch.setattr(cli_main, "resolve_run_analysts", lambda analysts, asset, ticker: [a.value for a in analysts])
    monkeypatch.setattr(cli_main, "AnalystWallTimeTracker", _Tracker)
    monkeypatch.setattr(
        cli_main,
        "YiAlphaGraph",
        lambda keys, config, debug, callbacks: _FakeGraph(
            keys,
            config,
            debug,
            callbacks,
            deltas,
            trace,
            details,
            stream_error,
            full_state_chunks,
        ),
    )
    # run_analysis rebinds the methods of the buffer: keep the module singleton clean.
    monkeypatch.setattr(cli_main, "message_buffer", cli_main.MessageBuffer())
    monkeypatch.setattr(cli_main, "console", console)
    monkeypatch.delenv("YIALPHA_DATA_VACUUM_POLICY", raising=False)
    # Imported lazily inside run_analysis -> patch the source modules.
    monkeypatch.setattr(
        batch_runner,
        "serialized_run",
        partial(_RecordingContext, trace, details, "serialized_run"),
    )
    monkeypatch.setattr(
        dataflows_utils,
        "pinned_analysis_date",
        partial(_RecordingContext, trace, details, "pinned_analysis_date"),
    )
    monkeypatch.setattr(tavily_vendor, "reset_run_budget", lambda: trace.append("tavily.reset_run_budget"))
    monkeypatch.setattr(run_scope, "ensure_run_scope", lambda: trace.append("run_scope.ensure_run_scope"))
    monkeypatch.setattr(cli_main, "Live", partial(_RecordingContext, trace, details, "Live"))

    pending = iter(answers)

    def _prompt(text, default=None, **kwargs):
        prompts.append((text, default))
        return next(pending)

    def _save(final_state, ticker, save_path):
        if save_error is not None:
            raise save_error
        saved.append(("saved", final_state, ticker, save_path))
        return Path("complete_report.md")

    def _display(final_state):
        saved.append(("displayed", final_state))

    monkeypatch.setattr(cli_main.typer, "prompt", _prompt)
    monkeypatch.setattr(cli_main, "save_report_to_disk", _save)
    monkeypatch.setattr(cli_main, "display_complete_report", _display)

    real_update_display = cli_main.update_display

    def _spy(layout, spinner_text=None, stats_handler=None, start_time=None):
        real_update_display(layout, spinner_text, stats_handler, start_time)
        buffer = cli_main.message_buffer
        snapshots.append((spinner_text, dict(buffer.agent_status), buffer.current_report))

    monkeypatch.setattr(cli_main, "update_display", _spy)
    clock = itertools.count(step=clock_step)
    monkeypatch.setattr(analyst_execution, "monotonic", lambda: next(clock))

    run_dir = tmp_path / "results" / selections["ticker"] / selections["analysis_date"]
    return SimpleNamespace(
        trace=trace,
        details=details,
        prompts=prompts,
        saved=saved,
        snapshots=snapshots,
        console=console,
        config=config,
        buffer=cli_main.message_buffer,
        run_dir=run_dir,
        report_dir=run_dir / "reports",
        log_file=run_dir / "message_tool.log",
    )


def _ai(content, msg_id, **extra):
    return AIMessage(content=content, id=msg_id, **extra)


_PM_DECISION = "### Portfolio Manager Decision\nPM JUDGE"
_OVERLAID_DECISION = _PM_DECISION + "\n\n**overlay**"

# One serial run as graph deltas (the fake graph turns each into a full-state
# chunk): two analysts, the research debate, the trader, the risk debate.
_PLAIN_MESSAGE = SimpleNamespace(  # not a LangChain type, not even dict tool calls
    content="plain object message",
    id="m4",
    tool_calls=[
        {"name": "get_price", "args": {"ticker": "DEMO"}, "id": "t1", "type": "tool_call"},
        SimpleNamespace(name="legacy_tool", args={"x": 1}),
    ],
)
_FULL_RUN = [
    {"messages": [_ai("Market looks bullish", "m1")], "market_report": "MARKET BODY"},
    {
        "messages": [
            _PLAIN_MESSAGE,
            AIMessage(content="", id="m3"),
            AIMessage(content="   ", id="m3b"),
            ToolMessage(content="price 123", tool_call_id="t1", id="m5"),
        ]
    },
    {"news_report": "NEWS BODY"},
    {"investment_debate_state": {"bull_history": "BULL", "bear_history": "  ", "judge_decision": ""}},
    {"investment_debate_state": {"bull_history": "BULL", "bear_history": "BEAR", "judge_decision": "JUDGE"}},
    {"trader_investment_plan": "TRADER PLAN"},
    {
        "risk_debate_state": {
            "aggressive_history": "AGG", "conservative_history": "", "neutral_history": "NEU", "judge_decision": ""
        }
    },
    {
        "risk_debate_state": {
            "aggressive_history": "AGG",
            "conservative_history": "CON",
            "neutral_history": "NEU",
            "judge_decision": "PM JUDGE",
        },
        "final_trade_decision": _PM_DECISION,
    },
]
_MERGED_STATE = {
    "messages": [m for delta in _FULL_RUN[:2] for m in delta["messages"]],
    "market_report": "MARKET BODY",
    "news_report": "NEWS BODY",
    "investment_debate_state": {"bull_history": "BULL", "bear_history": "BEAR", "judge_decision": "JUDGE"},
    "trader_investment_plan": "TRADER PLAN",
    "risk_debate_state": _FULL_RUN[7]["risk_debate_state"],
    "final_trade_decision": _PM_DECISION,
}
_OVERLAID_STATE = {**_MERGED_STATE, "final_trade_decision": _OVERLAID_DECISION}
_FINAL_STATE = {**_OVERLAID_STATE, "data_quality": {}}
_WALL_SUMMARY = "Analyst wall time: Market 100.00s | News 200.00s"


def test_r17_run_analysis_full_stream_calls_every_seam_in_order(monkeypatch, tmp_path):
    """A whole run, seam by seam: what run_analysis hands the graph and when the
    run lock, the date pin and the live display are entered and left."""
    env = _streaming_env(monkeypatch, tmp_path, _FULL_RUN)

    cli_main.run_analysis(checkpoint=False)

    assert env.trace == [
        "graph.init",
        "enter:serialized_run",
        "enter:pinned_analysis_date",
        "enter:Live",
        "tavily.reset_run_budget",
        "run_scope.ensure_run_scope",
        "_bind_run_record_stage",
        "_resolve_pending_entries",
        "get_past_context",
        "resolve_instrument_context",
        "create_initial_state",
        "get_graph_args",
        "tracker published",
        "stream",
        "tracker while streaming",
        "tracker cleared",
        "_apply_risk_overlay",
        "_stamp_run_record_ids",
        "finalize_streamed_run",
        "assign curr_state",
        "store_decision",
        "exit:Live",
        "exit:pinned_analysis_date",
        "exit:serialized_run",
    ]
    d = env.details
    keys, config, debug, callbacks = d["graph.init"]
    assert (keys, debug) == (["market", "news"], True)
    assert config is env.config
    [stats_handler] = callbacks
    assert isinstance(stats_handler, _Stats)

    assert d["serialized_run"] == ((config, "DEMO", "2024-06-12", "stock"), {})
    assert d["pinned_analysis_date"] == (("2024-06-12",), {})
    (layout,), live_kwargs = d["Live"]
    assert live_kwargs == {"refresh_per_second": 4}
    # The display gets the real five-region layout create_layout() builds.
    assert isinstance(layout, Layout)
    assert [region.name for region in (layout["header"], layout["main"], layout["footer"])] == [
        "header", "main", "footer",
    ]
    upper = layout["main"]["upper"]
    assert [region.name for region in (upper, layout["main"]["analysis"])] == ["upper", "analysis"]
    assert [region.name for region in (upper["progress"], upper["messages"])] == ["progress", "messages"]

    assert d["_bind_run_record_stage"] == (("DEMO", "2024-06-12", "stock"), {})
    assert d["_resolve_pending_entries"] == (
        ("DEMO",), {"as_of_date": "2024-06-12", "asset_type": "stock"}
    )
    assert d["get_past_context"] == (("DEMO",), {"as_of_date": "2024-06-12", "asset_type": "stock"})
    assert d["resolve_instrument_context"] == (("DEMO", "stock"), {"trade_date": "2024-06-12"})
    assert d["create_initial_state"] == (
        ("DEMO", "2024-06-12"),
        {"asset_type": "stock", "past_context": "PAST CONTEXT", "instrument_context": "INSTRUMENT CONTEXT"},
    )
    assert d["get_graph_args"] == ((), {"callbacks": [stats_handler]})
    assert d["stream"] == (("INITIAL STATE",), {"stream_mode": "values", "config": {"recursion_limit": 7}})
    # The display tracker is published to the graph while chunks stream ...
    assert d["tracker while streaming"] is _Tracker.last
    assert d["tracker assignments"] == [_Tracker.last, None]  # ... and dropped afterwards

    # Streamed chunks are per-node deltas: the overlay gets the merged state.
    assert d["_apply_risk_overlay"] == (
        ("DEMO", "2024-06-12", _MERGED_STATE),
        {"portfolio_state": None, "asset_type": "stock"},
    )
    assert d["_stamp_run_record_ids"] == ((_OVERLAID_STATE, "ledger-run-1", "regime-run-1"), {})
    assert d["finalize_streamed_run"] == (("DEMO", "2024-06-12", _OVERLAID_STATE), {"asset_type": "stock"})
    assert d["assign curr_state"] == _FINAL_STATE
    assert d["store_decision"] == (
        (),
        {
            "ticker": "DEMO",
            "trade_date": "2024-06-12",
            "final_trade_decision": _OVERLAID_DECISION,
            "asset_type": "stock",
        },
    )
    # ... and dropped again once the loop is over.
    assert env.saved == []


@pytest.mark.parametrize(
    ("env_value", "expected"),
    [(None, "warn"), ("reject", "reject"), ("bogus", "bogus")],
)
def test_r17_run_analysis_interactive_run_softens_the_data_vacuum_gate(
    monkeypatch, tmp_path, env_value, expected
):
    """CLAUDE.md contract, both halves: ``yialpha analyze`` defaults the policy
    to warn (a human is watching the live display); an explicit
    YIALPHA_DATA_VACUUM_POLICY is passed through untouched."""
    env = _streaming_env(monkeypatch, tmp_path, _FULL_RUN)
    if env_value is not None:
        monkeypatch.setenv("YIALPHA_DATA_VACUUM_POLICY", env_value)

    cli_main.run_analysis(checkpoint=False)

    assert env.details["graph.init"][1]["data_vacuum_policy"] == expected


def test_r17_run_analysis_full_stream_fills_display_log_and_report_files(monkeypatch, tmp_path):
    """What the run leaves behind: buffer rows, the message_tool.log lines, one
    markdown file per report section, and the console text after the run."""
    env = _streaming_env(monkeypatch, tmp_path, _FULL_RUN)

    cli_main.run_analysis(checkpoint=False)

    buffer = env.buffer
    # Full-state chunks re-carry earlier messages: each is shown exactly once.
    # Empty / whitespace content is skipped; a non-LangChain object is "System".
    assert [(kind, content) for _, kind, content in buffer.messages] == [
        ("System", "Selected ticker: DEMO"),
        ("System", "Analysis date: 2024-06-12"),
        ("System", "Selected analysts: market, news"),
        ("Agent", "Market looks bullish"),
        ("System", "plain object message"),
        ("Data", "price 123"),
        ("System", "Completed analysis for 2024-06-12"),
        ("System", _WALL_SUMMARY),
    ]
    # No "Detected asset type" line for the default stock path.
    assert buffer._processed_message_ids == {"m1", "m3", "m3b", "m4", "m5"}
    assert [(name, args) for _, name, args in buffer.tool_calls] == [
        ("get_price", {"ticker": "DEMO"}),  # dict form (AIMessage.tool_calls)
        ("legacy_tool", {"x": 1}),  # object form (.name / .args)
    ]
    lines = [re.sub(r"^\d\d:\d\d:\d\d ", "", line) for line in env.log_file.read_text(encoding="utf-8").splitlines()]
    assert lines == [
        "[System] Selected ticker: DEMO",
        "[System] Analysis date: 2024-06-12",
        "[System] Selected analysts: market, news",
        "[Agent] Market looks bullish",
        "[System] plain object message",
        "[Tool Call] get_price(ticker=DEMO)",
        "[Tool Call] legacy_tool(x=1)",
        "[Data] price 123",
        "[System] Completed analysis for 2024-06-12",
        f"[System] {_WALL_SUMMARY}",
    ]
    assert buffer.report_sections == {
        "market_report": "MARKET BODY",
        "news_report": "NEWS BODY",
        "investment_plan": "### Research Manager Decision\nJUDGE",
        "trader_investment_plan": "TRADER PLAN",
        "final_trade_decision": _OVERLAID_DECISION,
    }
    assert {
        path.stem: path.read_text(encoding="utf-8") for path in env.report_dir.iterdir()
    } == buffer.report_sections
    assert sorted(path.name for path in env.run_dir.iterdir()) == ["message_tool.log", "reports"]

    text = " ".join(env.console.export_text().split())
    assert "Analysis Complete!" in text
    assert _WALL_SUMMARY in text
    assert env.prompts == [("Save report?", "Y"), ("\nDisplay full report on screen?", "Y")]
    assert env.saved == []


def test_r17_run_analysis_full_stream_drives_agent_statuses_and_the_report_panel(
    monkeypatch, tmp_path
):
    """One snapshot per display refresh of what was handed to update_display
    as ``spinner_text``, the agent statuses and the current report: from the
    empty progress table to the finished run.

    The "Analyzing ..." text is passed once, right before the stream starts;
    the in-loop refreshes pass none.
    """
    env = _streaming_env(monkeypatch, tmp_path, _FULL_RUN)

    cli_main.run_analysis(checkpoint=False)

    def statuses(**overrides):
        return {
            "Market Analyst": "pending",
            "News Analyst": "pending",
            "Bull Researcher": "pending",
            "Bear Researcher": "pending",
            "Research Manager": "pending",
            "Trader": "pending",
            "Aggressive Analyst": "pending",
            "Neutral Analyst": "pending",
            "Conservative Analyst": "pending",
            "Portfolio Manager": "pending",
            **overrides,
        }

    spinner = "Analyzing DEMO on 2024-06-12..."
    market_active = {"Market Analyst": "in_progress"}
    market_done = {"Market Analyst": "completed"}
    news_active = {"News Analyst": "in_progress"}
    analysts_done = {"Market Analyst": "completed", "News Analyst": "completed"}
    bull_active = {"Bull Researcher": "in_progress"}
    research_running = {
        "Bull Researcher": "in_progress",
        "Bear Researcher": "in_progress",
        "Research Manager": "in_progress",
    }
    research_done = dict.fromkeys(research_running, "completed")
    trader_active = {"Trader": "in_progress"}
    trader_done = {"Trader": "completed"}
    aggressive_active = {"Aggressive Analyst": "in_progress"}
    risk_active = {"Aggressive Analyst": "in_progress", "Neutral Analyst": "in_progress"}
    risk_done = {
        "Aggressive Analyst": "completed",
        "Conservative Analyst": "completed",
        "Neutral Analyst": "completed",
    }
    pm_done = {"Portfolio Manager": "completed"}

    assert env.snapshots == [
        (None, statuses(), None),
        (None, statuses(), None),
        (None, statuses(**market_active), None),
        (spinner, statuses(**market_active), None),
        (None, statuses(**market_done, **news_active), "### Market Analysis\nMARKET BODY"),
        (None, statuses(**market_done, **news_active), "### Market Analysis\nMARKET BODY"),
        (None, statuses(**analysts_done, **bull_active), "### News Analysis\nNEWS BODY"),
        (
            None,
            statuses(**analysts_done, **research_running),
            "### Research Team Decision\n### Bull Researcher Analysis\nBULL",
        ),
        (
            None,
            statuses(**analysts_done, **research_done, **trader_active),
            "### Research Team Decision\n### Research Manager Decision\nJUDGE",
        ),
        (
            None,
            statuses(**analysts_done, **research_done, **trader_done, **aggressive_active),
            "### Trading Team Plan\nTRADER PLAN",
        ),
        (
            None,
            statuses(**analysts_done, **research_done, **trader_done, **risk_active),
            "### Portfolio Management Decision\n### Neutral Analyst Analysis\nNEU",
        ),
        (
            None,
            statuses(**analysts_done, **research_done, **trader_done, **risk_done, **pm_done),
            "### Portfolio Management Decision\n" + _PM_DECISION,
        ),
        (
            None,
            dict.fromkeys(statuses(), "completed"),
            "### Portfolio Management Decision\n" + _OVERLAID_DECISION,
        ),
    ]


def _frozen_clock(monkeypatch):
    """Freeze ``datetime.datetime.now()`` as run_analysis's module sees it."""
    monkeypatch.setattr(
        cli_main,
        "datetime",
        SimpleNamespace(datetime=SimpleNamespace(now=lambda: datetime.datetime(2026, 3, 10, 14, 30, 5))),
    )


@pytest.mark.parametrize(
    ("save_answer", "display_answer", "expect_saved", "expect_displayed"),
    [
        ("Y", "Y", True, True),
        ("YES", "NO", True, False),
        ("", "", True, True),  # Enter accepts both defaults
        ("n", "y", False, True),
        ("no", "N", False, False),
    ],
)
def test_r17_run_analysis_post_analysis_prompts(
    monkeypatch, tmp_path, save_answer, display_answer, expect_saved, expect_displayed
):
    """After the run: "Save report?", then "Display full report on screen?";
    Y / YES / Enter mean yes. The save path is asked only when saving."""
    target = tmp_path / "chosen" / "report"
    answers = (save_answer, str(target), display_answer) if expect_saved else (save_answer, display_answer)
    env = _streaming_env(monkeypatch, tmp_path, _FULL_RUN, answers=answers)
    _frozen_clock(monkeypatch)

    cli_main.run_analysis(checkpoint=False)

    expected_prompts = [("Save report?", "Y")]
    expected_saved = []
    if expect_saved:
        expected_prompts.append(
            (
                "Save path (press Enter for default)",
                str(Path.cwd() / "reports" / "DEMO_20260310_143005"),
            )
        )
        expected_saved.append(("saved", _FINAL_STATE, "DEMO", target))
    expected_prompts.append(("\nDisplay full report on screen?", "Y"))
    if expect_displayed:
        expected_saved.append(("displayed", _FINAL_STATE))
    assert env.prompts == expected_prompts
    assert env.saved == expected_saved
    text = " ".join(env.console.export_text().split())
    assert ("✓ Report saved to:" in text) is expect_saved
    assert ("Complete report: complete_report.md" in text) is expect_saved


def test_r17_run_analysis_a_failed_save_is_reported_and_the_run_still_finishes(
    monkeypatch, tmp_path
):
    env = _streaming_env(
        monkeypatch,
        tmp_path,
        _FULL_RUN,
        answers=("y", str(tmp_path / "chosen"), "y"),
        save_error=OSError("disk full"),
    )

    cli_main.run_analysis(checkpoint=False)

    text = " ".join(env.console.export_text().split())
    assert "Error saving report: disk full" in text
    assert "✓ Report saved to:" not in text
    # The run itself is complete and the report can still be shown.
    assert env.saved == [("displayed", _FINAL_STATE)]


def test_r17_run_analysis_an_empty_stream_raises_after_releasing_lock_pin_and_display(
    monkeypatch, tmp_path
):
    env = _streaming_env(monkeypatch, tmp_path, [])

    with pytest.raises(RuntimeError, match="graph.stream emitted no chunks for this run"):
        cli_main.run_analysis(checkpoint=False)

    assert env.trace == [
        "graph.init",
        "enter:serialized_run",
        "enter:pinned_analysis_date",
        "enter:Live",
        "tavily.reset_run_budget",
        "run_scope.ensure_run_scope",
        "_bind_run_record_stage",
        "_resolve_pending_entries",
        "get_past_context",
        "resolve_instrument_context",
        "create_initial_state",
        "get_graph_args",
        "tracker published",
        "stream",
        "tracker while streaming",
        "tracker cleared",
        # Left in reverse order: display, date pin, run lock.
        "exit:Live",
        "exit:pinned_analysis_date",
        "exit:serialized_run",
    ]
    assert env.details["tracker assignments"] == [_Tracker.last, None]
    assert env.prompts == []
    assert list(env.report_dir.iterdir()) == []  # nothing streamed, nothing filed


def test_r17_run_analysis_a_crash_mid_stream_propagates_after_releasing_everything(
    monkeypatch, tmp_path
):
    crash = RuntimeError("node crashed")
    env = _streaming_env(monkeypatch, tmp_path, _FULL_RUN[:2], stream_error=crash)

    with pytest.raises(RuntimeError) as info:
        cli_main.run_analysis(checkpoint=False)

    assert info.value is crash
    assert env.trace[env.trace.index("tracker while streaming") :] == [
        "tracker while streaming",
        "exit:Live",
        "exit:pinned_analysis_date",
        "exit:serialized_run",
    ]
    assert "_apply_risk_overlay" not in env.trace  # the billed run is not finalized
    assert env.prompts == []
    assert env.saved == []


def test_r17_run_analysis_announces_a_detected_asset_type_and_threads_it_everywhere(
    monkeypatch, tmp_path
):
    """``asset_type`` reaches every seam of a non-stock run: the lock, the
    record stage, the pending-entry resolution, the risk overlay, the state
    log and the memory decision. Defaulting it to "stock" used to void the
    perp machinery on interactive perp runs."""
    selections = {**_STREAM_SELECTIONS, "asset_type": "crypto_perp"}
    env = _streaming_env(monkeypatch, tmp_path, _FULL_RUN[:1], selections=selections)

    cli_main.run_analysis(checkpoint=False)

    assert [(kind, content) for _, kind, content in env.buffer.messages][:5] == [
        ("System", "Selected ticker: DEMO"),
        ("System", "Detected asset type: crypto_perp"),
        ("System", "Analysis date: 2024-06-12"),
        ("System", "Selected analysts: market, news"),
        ("Agent", "Market looks bullish"),
    ]
    d = env.details
    assert d["serialized_run"][0][1:] == ("DEMO", "2024-06-12", "crypto_perp")
    assert d["_bind_run_record_stage"] == (("DEMO", "2024-06-12", "crypto_perp"), {})
    assert d["_resolve_pending_entries"] == (
        ("DEMO",), {"as_of_date": "2024-06-12", "asset_type": "crypto_perp"}
    )
    assert d["get_past_context"] == (
        ("DEMO",), {"as_of_date": "2024-06-12", "asset_type": "crypto_perp"}
    )
    assert d["resolve_instrument_context"] == (("DEMO", "crypto_perp"), {"trade_date": "2024-06-12"})
    assert d["create_initial_state"] == (
        ("DEMO", "2024-06-12"),
        {"asset_type": "crypto_perp", "past_context": "PAST CONTEXT", "instrument_context": "INSTRUMENT CONTEXT"},
    )
    assert d["_apply_risk_overlay"][1] == {"portfolio_state": None, "asset_type": "crypto_perp"}
    assert d["finalize_streamed_run"][1] == {"asset_type": "crypto_perp"}
    assert d["store_decision"][1]["asset_type"] == "crypto_perp"


def test_r17_run_analysis_an_id_less_message_is_shown_again_in_every_chunk(monkeypatch, tmp_path):
    """The loop deduplicates by ``message.id`` and ``stream_mode="values"``
    re-carries every earlier message in each chunk, so a message whose id is
    None is shown once per chunk that carries it. Not reachable with graph
    state today -- see the real-stream test below -- pinned as it stands."""
    id_less = SimpleNamespace(content="no id here", id=None)

    env = _streaming_env(monkeypatch, tmp_path, [{"messages": [id_less]}, {}])

    cli_main.run_analysis(checkpoint=False)

    shown = [content for _, kind, content in env.buffer.messages if content == "no id here"]
    assert shown == ["no id here", "no id here"]
    assert env.buffer._processed_message_ids == set()


class _StreamState(TypedDict):
    """State of the tiny graph the real-stream test streams."""

    messages: Annotated[list, add_messages]
    market_report: str


def _real_graph_chunks():
    """Full-state chunks of a real LangGraph graph, streamed the way production
    streams it (``Propagator.get_graph_args``), from nodes that inject messages
    without ids -- as the analysts do with their evidence / placeholder
    HumanMessages."""

    def _analyst(_state):
        return {
            "messages": [HumanMessage(content="evidence block"), AIMessage(content="analyst says")],
            "market_report": "MARKET BODY",
        }

    def _trader(_state):
        return {
            "messages": [
                AIMessage(
                    content="trader says",
                    tool_calls=[{"name": "get_price", "args": {"ticker": "DEMO"}, "id": "call-1"}],
                )
            ]
        }

    builder = StateGraph(_StreamState)
    builder.add_node("analyst", _analyst)
    builder.add_node("trader", _trader)
    builder.add_edge(START, "analyst")
    builder.add_edge("analyst", "trader")
    builder.add_edge("trader", END)
    args = Propagator.get_graph_args(SimpleNamespace(max_recur_limit=100))
    assert args["stream_mode"] == "values"
    return list(builder.compile().stream({"messages": [HumanMessage(content="DEMO")]}, **args))


def test_r17_run_analysis_shows_each_message_of_a_real_stream_once(monkeypatch, tmp_path):
    """A real ``values``-mode stream, where every chunk carries all messages so
    far. The id-based deduplication also covers the messages a node injected
    without an id, because the graph's ``add_messages`` reducer assigns them
    one (verified 2026-11-24 on langgraph 1.2.10 and 1.2.24). If a future
    langgraph stopped doing that, this is where the display would start
    repeating rows."""
    chunks = _real_graph_chunks()
    assert [len(chunk["messages"]) for chunk in chunks] == [1, 3, 4]
    assert all(message.id for chunk in chunks for message in chunk["messages"])

    env = _streaming_env(monkeypatch, tmp_path, chunks, full_state_chunks=True)

    cli_main.run_analysis(checkpoint=False)

    assert [(kind, content) for _, kind, content in env.buffer.messages] == [
        ("System", "Selected ticker: DEMO"),
        ("System", "Analysis date: 2024-06-12"),
        ("System", "Selected analysts: market, news"),
        ("User", "DEMO"),
        ("User", "evidence block"),
        ("Agent", "analyst says"),
        ("Agent", "trader says"),
        ("System", "Completed analysis for 2024-06-12"),
        ("System", "Analyst wall time: Market 200.00s"),
    ]
    assert [(name, args) for _, name, args in env.buffer.tool_calls] == [
        ("get_price", {"ticker": "DEMO"})
    ]
    assert len(env.buffer._processed_message_ids) == 4
    assert env.buffer.agent_status["Market Analyst"] == "completed"


# --------------------------------------------------------------------------- #
# the remaining gap lines of main.py
# --------------------------------------------------------------------------- #


def test_r17_update_display_spinner_on_an_agent_that_is_not_the_first_row_of_its_team(
    monkeypatch,
):
    """Every agent of a team gets the same status cell: the first row of the
    team (Team column filled) and the rows below it (empty Team column)."""
    buffer = _fresh_buffer(monkeypatch)
    buffer.agent_status.update(
        {
            "Market Analyst": "completed",
            "News Analyst": "in_progress",
            "Bear Researcher": "in_progress",
            "Neutral Analyst": "in_progress",
        }
    )
    layout = cli_main.create_layout()

    cli_main.update_display(layout)

    _assert_progress(
        layout["progress"].renderable,
        [
            "│ Team Agent Status │",
            _HEAD_ROW,
            "│ Analyst Team Market Analyst completed │",
            _SpinnerRow("News Analyst"),
            _SEP_ROW,
            "│ Research Team Bull Researcher pending │",
            _SpinnerRow("Bear Researcher"),
            "│ Research Manager pending │",
            _SEP_ROW,
            "│ Trading Team Trader pending │",
            _SEP_ROW,
            "│ Risk Management Aggressive Analyst pending │",
            _SpinnerRow("Neutral Analyst"),
            "│ Conservative Analyst pending │",
            _SEP_ROW,
            "│ Portfolio Management Portfolio Manager pending │",
            _SEP_ROW,
        ],
    )


def test_r17_update_display_spinner_text_argument_is_not_rendered(monkeypatch):
    """Known observation: ``update_display`` accepts ``spinner_text`` but never
    uses it, so the "Analyzing <ticker> on <date>..." text run_analysis builds
    is not shown anywhere. Reported as it stands; rendering it would be a new
    UI behaviour."""
    buffer = _fresh_buffer(monkeypatch)
    buffer.update_agent_status("Market Analyst", "in_progress")
    with_text = cli_main.create_layout()
    cli_main.update_display(with_text, "Analyzing DEMO on 2024-06-12...")
    without = cli_main.create_layout()
    cli_main.update_display(without)

    assert _render_lines(with_text) == _render_lines(without)
    assert "Analyzing DEMO" not in " ".join(_render_lines(with_text))
    # The "in_progress" cells are where the spinner actually is.
    _assert_progress(
        with_text["progress"].renderable,
        [
            "│ Team Agent Status │",
            _HEAD_ROW,
            _SpinnerRow("Market Analyst", "Analyst Team"),
            "│ News Analyst pending │",
            *_FRESH_PROGRESS[4:],
        ],
    )


@pytest.mark.parametrize(
    ("art", "logo_rows"),
    [
        ("LOGO-A\nLOGO-B", [3, 4]),  # the shipped shape: no blank line around the art
        ("\n\nLOGO-A\nLOGO-B", [3, 4]),
        ("LOGO-A\nLOGO-B\n\n\n", [3, 4]),
        ("LOGO-A\r\nLOGO-B", [3, 4]),  # the file is stored with CRLF line endings
        ("\r\n\r\nLOGO-A\r\nLOGO-B\r\n", [3, 4]),
        ("LOGO-A\n\nLOGO-B", [3, 5]),  # a blank line INSIDE the art is kept
    ],
)
def test_r17_get_user_selections_drops_blank_lines_around_the_welcome_art(
    monkeypatch, art, logo_rows
):
    """``static/welcome.txt`` has its surrounding blank lines trimmed before it
    is rendered, so adding some to the file changes nothing on screen: the
    first and the last line of the art keep their rows. The shipped file has
    none, which is why the two trimming loops never ran."""
    console = _recording_console()
    monkeypatch.setattr(cli_main, "console", console)
    # Stand-in for reading static/welcome.txt. Text mode: CRLF reads back as LF.
    monkeypatch.setattr(
        cli_main, "open", lambda *args, **kwargs: io.StringIO(art, newline=None), raising=False
    )

    def _stop():
        raise _StopBeforeDisplay  # stop right after the welcome box

    monkeypatch.setattr(cli_main, "get_ticker", _stop)

    with pytest.raises(_StopBeforeDisplay):
        cli_main.get_user_selections()

    lines = [" ".join(line.split()) for line in console.export_text().splitlines()]
    assert [i for i, line in enumerate(lines) if "LOGO-A" in line] == [logo_rows[0]]
    assert [i for i, line in enumerate(lines) if "LOGO-B" in line] == [logo_rows[1]]
    first, last = logo_rows
    assert "╭" in lines[0] and "Welcome to YiAlpha" in lines[0]
    # Two blank rows above the art (the padding of the box itself) and two
    # below it, whatever the file had around the art.
    assert lines[1:first] == ["│ │"] * 2
    assert lines[last + 1 : last + 3] == ["│ │"] * 2
    assert "Workflow Steps:" in lines[last + 3]
    # (Where the box closes depends on the terminal width: the workflow line wraps.)
    assert any("╰" in line for line in lines[last + 4 :])


def test_r17_update_current_report_shows_the_last_declared_section_not_the_last_updated_one(
    monkeypatch,
):
    """Known observation (``MessageBuffer._update_current_report``): the comment
    says "most recently updated", the code takes the LAST non-None section in
    declaration order. Decided NOT to fix: the two only disagree about which of
    several already-arrived analyst reports the panel shows, never about the
    stage of the run (later stages are declared after earlier ones)."""
    buffer = _fresh_buffer(monkeypatch, analysts=("market", "positioning", "news"))
    updates: list[str] = []
    original = cli_main.MessageBuffer.update_report_section

    def _spy(self, section_name, content):
        updates.append(section_name)
        original(self, section_name, content)

    monkeypatch.setattr(cli_main.MessageBuffer, "update_report_section", _spy)

    # Per chunk, update_analyst_statuses refreshes the arrived reports in
    # ANALYST_ORDER: market, positioning, news. News is updated last ...
    for section, text in (("market_report", "M"), ("positioning_report", "P"), ("news_report", "N")):
        buffer.update_report_section(section, text)

    assert updates[-1] == "news_report"
    # ... but the panel shows positioning: it is declared after the other
    # analyst reports, so it is the last non-None section in dict order.
    assert buffer.current_report == "### Positioning Analysis\nP"
    # Out-of-order arrival across chunks (analyst-parallel mode) looks the same.
    buffer.update_report_section("market_report", "M2")
    assert buffer.current_report == "### Positioning Analysis\nP"
    # A later stage always wins: it is declared last.
    buffer.update_report_section("final_trade_decision", "DECISION")
    assert buffer.current_report == "### Portfolio Management Decision\nDECISION"


def test_r17_save_report_section_decorator_writes_only_sections_of_this_run(monkeypatch, tmp_path):
    """The on-disk file name is the section name: only a section the run's
    buffer actually has can ever become a file, so an unknown or path-like
    name writes nothing."""
    env = _decorator_env(monkeypatch, tmp_path)
    buffer = cli_main.MessageBuffer()
    buffer.init_for_analysis(["market"])
    wrapper = env.save_report_section_decorator(buffer, "update_report_section")

    for name in ("positioning_report", "not_a_section", "../escape"):
        assert wrapper(name, "content") is None

    assert buffer.report_sections == {
        "market_report": None,
        "investment_plan": None,
        "trader_investment_plan": None,
        "final_trade_decision": None,
    }
    assert list(env.report_dir.iterdir()) == []

    wrapper("market_report", "MARKET BODY")

    assert [path.name for path in env.report_dir.iterdir()] == ["market_report.md"]
    assert buffer.report_sections["market_report"] == "MARKET BODY"
