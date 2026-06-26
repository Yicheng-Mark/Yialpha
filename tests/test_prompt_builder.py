"""Unit tests for the FinCoT structured prompt builder (Phase 2b)."""

from __future__ import annotations

import pytest

from tradingagents.agents.utils.prompt_builder import (
    build_fincot_prompt,
    build_mermaid_workflow,
)


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
