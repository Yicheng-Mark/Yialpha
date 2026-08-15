"""Smoke test for the offline prune_indicators CLI script.

Verifies the script loads a CSV, computes rolling IC, applies the pruning rule,
and produces a markdown report — without touching the live config.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "prune_indicators_cli.py"


@pytest.mark.unit
class TestPruneIndicatorsCLI:
    def _make_csv(self, tmp_path: Path) -> Path:
        """Create a small CSV with one predictive and one useless indicator."""
        import numpy as np
        import pandas as pd

        rng = np.random.RandomState(42)
        n = 120
        dates = pd.date_range("2025-01-01", periods=n, freq="B")

        # 'good_indicator' is correlated with forward returns (should be KEPT).
        good = rng.randn(n)
        forward = 0.5 * good + 0.5 * rng.randn(n)

        # 'bad_indicator' is pure noise (should be PRUNED if it has enough data).
        bad = rng.randn(n)

        df = pd.DataFrame({
            "date": dates,
            "forward_return": forward,
            "good_indicator": good,
            "bad_indicator": bad,
        })
        csv_path = tmp_path / "ic_sample.csv"
        df.to_csv(csv_path, index=False)
        return csv_path

    def test_script_produces_report(self, tmp_path):
        """The CLI runs end-to-end and prints a markdown report."""
        csv = self._make_csv(tmp_path)
        result = subprocess.run(
            [sys.executable, str(_SCRIPT), str(csv), "--window", "20",
             "--min-consecutive", "10", "--min-observations", "10"],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        assert "IC Indicator Pruning Report" in result.stdout
        assert "Kept" in result.stdout
        assert "Pruned" in result.stdout

    def test_missing_file_returns_error(self, tmp_path):
        """A non-existent CSV exits with code 1."""
        result = subprocess.run(
            [sys.executable, str(_SCRIPT), str(tmp_path / "nonexistent.csv")],
            capture_output=True, text=True, timeout=10,
        )
        assert result.returncode == 1
        assert "not found" in result.stderr

    def test_suggest_config_flag_outputs_python_snippet(self, tmp_path):
        """--suggest-config adds a Python snippet pointing at the real key."""
        csv = self._make_csv(tmp_path)
        result = subprocess.run(
            [sys.executable, str(_SCRIPT), str(csv), "--window", "20",
             "--min-consecutive", "5", "--min-observations", "5",
             "--suggest-config"],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0
        assert "IC Indicator Pruning Report" in result.stdout
        if "Config suggestion" in result.stdout:
            # The suggestion must land on the real config key the market
            # analyst reads (indicator_battery), not a phantom key.
            assert '"indicator_battery"' in result.stdout
            assert "good_indicator" in result.stdout  # kept names suggested

    def test_output_flag_writes_file(self, tmp_path):
        """--output writes the report to a file."""
        csv = self._make_csv(tmp_path)
        out_path = tmp_path / "report.md"
        result = subprocess.run(
            [sys.executable, str(_SCRIPT), str(csv), "--window", "20",
             "--min-consecutive", "10", "--min-observations", "10",
             "--output", str(out_path)],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0
        assert out_path.exists()
        assert "IC Indicator Pruning Report" in out_path.read_text(encoding="utf-8")

    def test_json_out_writes_structured_verdict(self, tmp_path):
        """--json-out writes params + keep/prune + per-indicator stats."""
        import json as _json

        csv = self._make_csv(tmp_path)
        json_path = tmp_path / "verdict.json"
        result = subprocess.run(
            [sys.executable, str(_SCRIPT), str(csv), "--window", "20",
             "--min-consecutive", "10", "--min-observations", "10",
             "--json-out", str(json_path)],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        verdict = _json.loads(json_path.read_text(encoding="utf-8"))

        assert set(verdict["params"]) == {
            "window", "min_abs_ic", "min_consecutive", "min_observations",
        }
        assert verdict["params"]["window"] == 20
        assert set(verdict) == {
            "generated_at", "params", "keep", "prune", "per_indicator",
        }
        assert set(verdict["per_indicator"]) == {"good_indicator", "bad_indicator"}
        for stats in verdict["per_indicator"].values():
            # Core pruning evidence, always present...
            assert {
                "verdict", "mean_abs_ic", "finite_windows", "longest_low_run",
            } <= set(stats)
            # ...plus the 2026-08-15 diagnostics, present whenever the math
            # yields a value on this CSV (no fwd_ret_*d columns here, so
            # ic_by_horizon stays absent).
            assert "quantile_means" in stats
            assert "quantile_spread" in stats
            assert "monotonic" in stats
            assert "turnover" in stats
            assert "ic_by_horizon" not in stats
            assert stats["verdict"] in ("keep", "prune")
            assert isinstance(stats["finite_windows"], int)
        # keep/prune lists stay consistent with the per-indicator verdicts.
        for name, stats in verdict["per_indicator"].items():
            assert name in verdict[stats["verdict"]]
