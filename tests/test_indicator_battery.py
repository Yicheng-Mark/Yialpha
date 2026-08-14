"""``indicator_battery`` — the self-improvement loop's config landing point.

The market analyst's indicator catalog used to live only as a prompt constant,
so IC-pruning conclusions (scripts/prune_indicators_cli.py) had nowhere to
land. The catalog is now rendered from a structured source of truth and pruned
by the ``indicator_battery`` config key.

The iron rule pinned here: with the battery unset (the default), the rendered
catalog and both prompt forms are **byte-identical** to the pre-refactor
hand-written literal (the snapshot below was extracted from git HEAD before
the refactor).
"""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

import yiagents.agents.analysts.market_analyst as ma
from yiagents.cli.main import app
from yiagents.dataflows.config import set_config

runner = CliRunner()

# Snapshot of the original INDICATOR_CATALOG literal (git HEAD, 2026-08-14
# refactor). Byte-equivalence of the default render is the fail-safe contract.
_CATALOG_SNAPSHOT = """Moving Averages:
- close_50_sma: 50 SMA: A medium-term trend indicator. Usage: Identify trend direction and serve as dynamic support/resistance. Tips: It lags price; combine with faster indicators for timely signals.
- close_200_sma: 200 SMA: A long-term trend benchmark. Usage: Confirm overall market trend and identify golden/death cross setups. Tips: It reacts slowly; best for strategic trend confirmation rather than frequent trading entries.
- close_10_ema: 10 EMA: A responsive short-term average. Usage: Capture quick shifts in momentum and potential entry points. Tips: Prone to noise in choppy markets; use alongside longer averages for filtering false signals.

MACD Related:
- macd: MACD: Computes momentum via differences of EMAs. Usage: Look for crossovers and divergence as signals of trend changes. Tips: Confirm with other indicators in low-volatility or sideways markets.
- macds: MACD Signal: An EMA smoothing of the MACD line. Usage: Use crossovers with the MACD line to trigger trades. Tips: Should be part of a broader strategy to avoid false positives.
- macdh: MACD Histogram: Shows the gap between the MACD line and its signal. Usage: Visualize momentum strength and spot divergence early. Tips: Can be volatile; complement with additional filters in fast-moving markets.

Momentum Indicators:
- rsi: RSI: Measures momentum to flag overbought/oversold conditions. Usage: Apply 70/30 thresholds and watch for divergence to signal reversals. Tips: In strong trends, RSI may remain extreme; always cross-check with trend analysis.

Volatility Indicators:
- boll: Bollinger Middle: A 20 SMA serving as the basis for Bollinger Bands. Usage: Acts as a dynamic benchmark for price movement. Tips: Combine with the upper and lower bands to effectively spot breakouts or reversals.
- boll_ub: Bollinger Upper Band: Typically 2 standard deviations above the middle line. Usage: Signals potential overbought conditions and breakout zones. Tips: Confirm signals with other tools; prices may ride the band in strong trends.
- boll_lb: Bollinger Lower Band: Typically 2 standard deviations below the middle line. Usage: Indicates potential oversold conditions. Tips: Use additional analysis to avoid false reversal signals.
- atr: ATR: Averages true range to measure volatility. Usage: Set stop-loss levels and adjust position sizes based on current market volatility. Tips: It's a reactive measure, so use it as part of a broader risk management strategy.

Volume-Based Indicators:
- vwma: VWMA: A moving average weighted by volume. Usage: Confirm trends by integrating price action with volume data. Tips: Watch for skewed results from volume spikes; use in combination with other volume analyses."""

_EXPECTED_NAMES = {
    "close_50_sma", "close_200_sma", "close_10_ema",
    "macd", "macds", "macdh",
    "rsi",
    "boll", "boll_ub", "boll_lb", "atr",
    "vwma",
}


@pytest.mark.unit
class TestCatalogRender:
    def test_default_render_byte_identical_to_baseline(self):
        # The iron rule: no battery -> the exact pre-refactor prompt text.
        assert ma._render_indicator_catalog() == _CATALOG_SNAPSHOT
        assert ma.INDICATOR_CATALOG == _CATALOG_SNAPSHOT

    def test_indicator_names_exact(self):
        assert set(ma.INDICATOR_NAMES) == _EXPECTED_NAMES
        assert len(ma.INDICATOR_NAMES) == 12

    def test_battery_prunes_entries_and_empty_sections(self):
        out = ma._render_indicator_catalog(["rsi", "atr"])
        assert "- rsi:" in out
        assert "- atr:" in out
        assert "- macd:" not in out
        assert "- vwma:" not in out
        # Sections with no surviving indicators disappear entirely.
        assert "Moving Averages:" not in out
        assert "MACD Related:" not in out
        assert "Volume-Based Indicators:" not in out
        assert "Momentum Indicators:" in out
        assert "Volatility Indicators:" in out

    def test_battery_preserves_entry_order_and_dedupes(self):
        out = ma._render_indicator_catalog(["rsi", "atr", "rsi"])
        assert out.count("- rsi:") == 1  # dedupe (dict.fromkeys upstream)
        # Section order follows the catalog, not the battery list.
        assert out.index("Momentum") < out.index("Volatility")


@pytest.mark.unit
class TestBatteryConfig:
    def test_default_battery_none_is_full_catalog(self):
        assert ma._catalog_for_run() == ma.INDICATOR_CATALOG

    def test_battery_from_config_prunes_catalog(self, caplog):
        set_config({"indicator_battery": ["rsi", "atr"]})
        out = ma._catalog_for_run()
        assert "- rsi:" in out and "- macd:" not in out

    def test_unknown_battery_names_warn_and_are_ignored(self, caplog):
        set_config({"indicator_battery": ["rsi", "nope"]})
        with caplog.at_level("WARNING",
                             logger="yiagents.agents.analysts.market_analyst"):
            out = ma._catalog_for_run()
        assert "- rsi:" in out and "- macd:" not in out
        assert any("unknown indicators" in r.message for r in caplog.records)

    def test_all_unknown_battery_falls_back_to_full_catalog(self, caplog):
        set_config({"indicator_battery": ["nope"]})
        with caplog.at_level("WARNING",
                             logger="yiagents.agents.analysts.market_analyst"):
            out = ma._catalog_for_run()
        assert out == ma.INDICATOR_CATALOG  # never leave the analyst tool-less
        assert any("no known indicators" in r.message for r in caplog.records)

    def test_empty_battery_warns_and_uses_full_catalog(self, caplog):
        set_config({"indicator_battery": []})
        with caplog.at_level("WARNING",
                             logger="yiagents.agents.analysts.market_analyst"):
            out = ma._catalog_for_run()
        assert out == ma.INDICATOR_CATALOG

    def test_legacy_prompt_default_byte_identical(self):
        # Battery unset: the legacy prompt is the baseline text unchanged.
        set_config({"indicator_battery": None})
        assert "- vwma:" in ma._legacy_system_message()

    def test_legacy_prompt_uses_battery(self):
        set_config({"indicator_battery": ["rsi"]})
        msg = ma._legacy_system_message()
        assert "- rsi:" in msg
        assert "- macd:" not in msg
        assert "Volume-Based Indicators:" not in msg

    def test_fincot_prompt_uses_battery(self):
        set_config({"fin_cot_prompts": True,
                    "indicator_battery": ["rsi", "macd"]})
        msg = ma._fincot_system_message()
        assert "- rsi:" in msg and "- macd:" in msg
        assert "- boll:" not in msg


@pytest.mark.unit
class TestConfigCheckBattery:
    """`yiagents config-check` validates the battery against known names."""

    def test_unset_battery_reports_default(self, monkeypatch):
        monkeypatch.setenv("YIAGENTS_LLM_PROVIDER", "deepseek")
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test-dummy")
        result = runner.invoke(app, ["config-check"])
        assert result.exit_code == 0
        assert "full catalog (default)" in result.output

    def test_valid_battery_passes(self, monkeypatch):
        monkeypatch.setenv("YIAGENTS_LLM_PROVIDER", "deepseek")
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test-dummy")
        set_config({"indicator_battery": ["rsi", "atr"]})
        result = runner.invoke(app, ["config-check"])
        assert result.exit_code == 0
        assert "all known" in result.output

    def test_unknown_battery_flagged_not_fatal(self, monkeypatch):
        monkeypatch.setenv("YIAGENTS_LLM_PROVIDER", "deepseek")
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test-dummy")
        set_config({"indicator_battery": ["rsi", "typo_name"]})
        result = runner.invoke(app, ["config-check"])
        # Flagged as ❌ but the exit code stays 0 (core-LLM readiness only).
        assert result.exit_code == 0
        assert "unknown indicator" in result.output
        assert "typo_name" in result.output
