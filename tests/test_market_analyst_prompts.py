"""Tests for the market analyst's selectable prompt forms (Phase 2b wiring)."""

from __future__ import annotations

import pytest

from tradingagents.agents.analysts.market_analyst import (
    INDICATOR_CATALOG,
    _fincot_system_message,
    _legacy_system_message,
    _system_message,
)


@pytest.mark.unit
def test_legacy_prompt_keeps_persona_and_catalog():
    msg = _legacy_system_message()
    # Baseline persona framing preserved.
    assert "trading assistant" in msg
    # Indicator vocabulary intact in both forms.
    assert "close_50_sma" in msg
    assert "atr" in msg
    assert "get_verified_market_snapshot" in msg


@pytest.mark.unit
def test_fincot_prompt_is_structured_and_depersona():
    msg = _fincot_system_message()
    assert "## Task" in msg
    assert "## Reasoning steps" in msg
    assert "## Output constraints" in msg
    assert "```mermaid" in msg
    # De-persona: no "You are a ... Analyst" framing.
    assert "You are a trading assistant" not in msg
    # Same indicator vocabulary.
    assert "close_50_sma" in msg
    assert INDICATOR_CATALOG.splitlines()[0] in msg


@pytest.mark.unit
def test_system_message_defaults_to_legacy():
    # Default config has fin_cot_prompts=False.
    msg = _system_message()
    assert "trading assistant" in msg  # legacy form


@pytest.mark.unit
def test_system_message_switches_to_fincot_when_enabled(monkeypatch):
    import tradingagents.dataflows.config as cfg_mod
    enabled = dict(cfg_mod._config)
    enabled["fin_cot_prompts"] = True
    monkeypatch.setattr(cfg_mod, "_config", enabled)
    msg = _system_message()
    assert "## Task" in msg
    assert "trading assistant" not in msg
