"""Tests for the ``yialpha config-check`` CLI subcommand.

Verifies that config-check:
* exits 0 when the selected provider's key is present;
* exits 1 when the key is missing;
* warns (but exits 0) when optional data sources are unset.
"""
from __future__ import annotations

import pytest
from typer.testing import CliRunner

from yialpha.cli.main import app

runner = CliRunner()


@pytest.mark.unit
class TestConfigCheck:
    def test_provider_key_present_exits_zero(self, monkeypatch):
        """When the provider's key is set, config-check exits 0."""
        monkeypatch.setenv("YIALPHA_LLM_PROVIDER", "deepseek")
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test-dummy")
        result = runner.invoke(app, ["config-check"])
        assert result.exit_code == 0
        assert "ready" in result.output.lower()

    def test_provider_key_missing_exits_one(self, monkeypatch):
        """When the provider's key is missing, config-check exits 1."""
        monkeypatch.setenv("YIALPHA_LLM_PROVIDER", "deepseek")
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        result = runner.invoke(app, ["config-check"])
        assert result.exit_code == 1
        assert "MISSING" in result.output

    def test_keyless_provider_exits_zero(self, monkeypatch):
        """Ollama (no key required) always passes."""
        monkeypatch.setenv("YIALPHA_LLM_PROVIDER", "ollama")
        result = runner.invoke(app, ["config-check"])
        assert result.exit_code == 0
        assert "no API key required" in result.output

    def test_optional_keys_reported_as_set_or_unset(self, monkeypatch):
        """Optional data sources show SET/unset status without failing."""
        monkeypatch.setenv("YIALPHA_LLM_PROVIDER", "deepseek")
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test-dummy")
        monkeypatch.setenv("FRED_API_KEY", "dummy-fred")
        monkeypatch.delenv("TUSHARE_TOKEN", raising=False)
        result = runner.invoke(app, ["config-check"])
        assert result.exit_code == 0
        assert "SET" in result.output  # FRED should be SET
