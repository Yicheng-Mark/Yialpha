"""Unit tests for the FinCoT structured prompt builder (Phase 2b)."""

from __future__ import annotations

import pytest

from yialpha.agents.utils.prompt_builder import (
    build_collaborator_prompt,
    build_fincot_prompt,
    build_mermaid_workflow,
)


def _render_collaborator(include_tools, **partials):
    p = build_collaborator_prompt(include_tools)
    for k, v in partials.items():
        p = p.partial(**{k: v})
    return p.format_messages(messages=[])[0].content


# Golden rendering of the shared "collaborating with other assistants" scaffolding.
# Captured from each analyst's prior inline ChatPromptTemplate (byte-identical across
# news / fundamentals / market for the tool variant; sentiment uses the no-tool variant).
# Any drift in build_collaborator_prompt changes a tool-calling analyst's LLM input, so
# these exact strings are the byte-equivalence contract.
_TOOLS_TRUE_GOLDEN = (
    "You are a helpful AI assistant, collaborating with other assistants."
    " Use the provided tools to progress towards answering the question."
    " If you are unable to fully answer, that's OK; another assistant with different tools"
    " will help where you left off. Execute what you can to make progress."
    " If you or any other assistant has the FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** or deliverable,"
    " prefix your response with FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** so the team knows to stop."
    " You have access to the following tools: A, B."
    " Today's date is 2026-01-01; treat it as 'now' for all analysis and tool-call date ranges. CTX\n"
    "SYS"
)
_TOOLS_FALSE_GOLDEN = (
    "You are a helpful AI assistant, collaborating with other assistants."
    " If you or any other assistant has the FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** or deliverable,"
    " prefix your response with FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** so the team knows to stop."
    " Today's date is 2026-01-01; treat it as 'now' for all analysis. CTX\n"
    "SYS"
)


@pytest.mark.unit
def test_collaborator_prompt_tools_variant_matches_golden():
    out = _render_collaborator(
        True,
        system_message="SYS",
        tool_names="A, B",
        current_date="2026-01-01",
        instrument_context="CTX",
    )
    assert out == _TOOLS_TRUE_GOLDEN


@pytest.mark.unit
def test_collaborator_prompt_no_tools_variant_matches_golden():
    out = _render_collaborator(
        False,
        system_message="SYS",
        current_date="2026-01-01",
        instrument_context="CTX",
    )
    assert out == _TOOLS_FALSE_GOLDEN


@pytest.mark.unit
def test_collaborator_prompt_no_tools_has_no_tool_names_slot():
    # The no-tool variant must NOT carry a {tool_names} placeholder (sentiment
    # never partials one), and must NOT advertise tools.
    out = _render_collaborator(False, system_message="SYS", current_date="D", instrument_context="I")
    assert "{tool_names}" not in out
    assert "Use the provided tools" not in out
    assert "tool-call date ranges" not in out


@pytest.mark.unit
def test_mermaid_workflow_basic_shape():
    out = build_mermaid_workflow(["Load data", "Compute signals", "Summarize"])
    assert out.startswith("flowchart TD")
    assert 'S([Start]) --> N1["Load data"]' in out
    assert "N3" in out
    assert 'E([Output])' in out


@pytest.mark.unit
def test_mermaid_workflow_empty():
    assert build_mermaid_workflow([]) == ""


@pytest.mark.unit
def test_fincot_prompt_has_three_sections():
    prompt = build_fincot_prompt(
        task="Produce a sentiment band and score.",
        reasoning_steps=["Read sources", "Score each", "Aggregate"],
        output_constraints=["Cite evidence", "No invented numbers"],
    )
    assert "## Task" in prompt
    assert "## Reasoning steps" in prompt
    assert "## Output constraints" in prompt
    # De-persona: no "You are a" framing.
    assert "You are a" not in prompt


@pytest.mark.unit
def test_fincot_prompt_numbered_steps():
    prompt = build_fincot_prompt(
        task="t", reasoning_steps=["a", "b", "c"], output_constraints=["x"],
    )
    assert "1. a" in prompt
    assert "2. b" in prompt
    assert "3. c" in prompt


@pytest.mark.unit
def test_fincot_prompt_includes_mermaid_by_default():
    prompt = build_fincot_prompt(
        task="t", reasoning_steps=["a", "b"], output_constraints=["x"],
    )
    assert "```mermaid" in prompt
    assert "flowchart TD" in prompt


@pytest.mark.unit
def test_fincot_prompt_can_omit_workflow():
    prompt = build_fincot_prompt(
        task="t", reasoning_steps=["a"], output_constraints=["x"],
        include_workflow=False,
    )
    assert "```mermaid" not in prompt


@pytest.mark.unit
def test_fincot_prompt_context_prepended():
    prompt = build_fincot_prompt(
        task="t", reasoning_steps=["a"], output_constraints=["x"],
        context="Instrument: AAPL. Tools: get_stock_data.",
    )
    # Context appears before the Task section.
    assert prompt.index("Instrument: AAPL") < prompt.index("## Task")


@pytest.mark.unit
def test_fincot_prompt_constraints_bulleted():
    prompt = build_fincot_prompt(
        task="t", reasoning_steps=["a"],
        output_constraints=["No lookahead", "Cite dates"],
    )
    assert "- No lookahead" in prompt
    assert "- Cite dates" in prompt
