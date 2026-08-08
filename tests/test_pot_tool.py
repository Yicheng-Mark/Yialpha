"""Tests for the PoT compute tool wrapper (pot_tool.py).

Verifies that the tool correctly delegates to PotAnalyzer and returns a
formatted result string. The LLM and sandbox are mocked so no real model
call or code execution happens.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from yiagents.agents.utils.pot_tool import make_pot_compute_tool


@dataclass
class _FakeAnalysis:
    question: str
    code: str
    result: Any
    ok: bool
    error: str | None
    attempts: int


@pytest.mark.unit
class TestPotComputeTool:
    def test_tool_name_and_description(self):
        """The tool has the expected name for prompt injection."""
        llm = MagicMock()
        t = make_pot_compute_tool(llm)
        assert t.name == "pot_compute"

    def test_successful_computation_returns_result(self):
        """When PotAnalyzer succeeds, the tool returns the computed value."""
        llm = MagicMock()
        fake = _FakeAnalysis(
            question="What is 6 times 7?", code="result = 6 * 7",
            result=42.0, ok=True, error=None, attempts=1,
        )
        with patch(
            "yiagents.agents.utils.pot_tool.PotAnalyzer"
        ) as MockAnalyzer:
            MockAnalyzer.return_value.compute.return_value = fake
            t = make_pot_compute_tool(llm)
            result = t.invoke({"question": "What is 6 times 7?", "data_json": '{"a": 6, "b": 7}'})
        assert "PoT result: 42.0" in result
        assert "result = 6 * 7" in result

    def test_failed_computation_returns_error_message(self):
        """When PotAnalyzer fails, the tool returns a helpful fallback message."""
        llm = MagicMock()
        fake = _FakeAnalysis(
            question="bad", code="", result=None, ok=False,
            error="NameError: name 'foo' is not defined", attempts=2,
        )
        with patch(
            "yiagents.agents.utils.pot_tool.PotAnalyzer"
        ) as MockAnalyzer:
            MockAnalyzer.return_value.compute.return_value = fake
            t = make_pot_compute_tool(llm)
            result = t.invoke({"question": "bad computation", "data_json": "{}"})
        assert "PoT failed" in result
        assert "NameError" in result

    def test_invalid_json_returns_error(self):
        """Malformed data_json is reported without crashing."""
        llm = MagicMock()
        with patch("yiagents.agents.utils.pot_tool.PotAnalyzer"):
            t = make_pot_compute_tool(llm)
            result = t.invoke({"question": "anything", "data_json": "not valid json{"})
        assert "PoT error" in result
        assert "invalid JSON" in result

    def test_empty_data_json_works(self):
        """Empty data_json defaults to {} and delegates to the analyzer."""
        llm = MagicMock()
        fake = _FakeAnalysis(
            question="q", code="result = 1", result=1, ok=True, error=None, attempts=1,
        )
        with patch(
            "yiagents.agents.utils.pot_tool.PotAnalyzer"
        ) as MockAnalyzer:
            MockAnalyzer.return_value.compute.return_value = fake
            t = make_pot_compute_tool(llm)
            result = t.invoke({"question": "q", "data_json": ""})
        assert "PoT result: 1" in result
