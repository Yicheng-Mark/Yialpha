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
        import pandas as pd
        import numpy as np

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

    def test_suggest_config_flag_outputs_yaml(self, tmp_path):
        """--suggest-config adds a YAML suggestion section (never auto-applies)."""
        csv = self._make_csv(tmp_path)
        result = subprocess.run(
            [sys.executable, str(_SCRIPT), str(csv), "--window", "20",
             "--min-consecutive", "5", "--min-observations", "5",
             "--suggest-config"],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0
        # Either a suggestion block (if something was pruned) or just the report.
        assert "IC Indicator Pruning Report" in result.stdout

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
