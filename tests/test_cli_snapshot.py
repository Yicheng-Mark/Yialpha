"""Tests for the ``yiagents snapshot`` CLI subcommand group.

The record/diff/list wiring gives yiagents/config_snapshot.py (previously
mechanism-complete but unreachable at runtime) its human-driven entry point.
Snapshots are isolated into a tmp data_cache_dir so tests never touch the
user's real history.
"""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from yiagents.cli.main import app
from yiagents.dataflows.config import set_config

runner = CliRunner()


@pytest.fixture()
def isolated_history(tmp_path):
    """Point data_cache_dir at a tmp dir so snapshots land in isolation."""
    set_config({"data_cache_dir": str(tmp_path)})
    return tmp_path / "config_history"


@pytest.mark.unit
class TestSnapshotRecord:
    def test_record_writes_snapshot_with_provenance(self, isolated_history):
        result = runner.invoke(
            app,
            ["snapshot", "record", "--reason", "pruned low-IC indicators",
             "--evidence", "reports/ic_2026-08-14.md"],
        )
        assert result.exit_code == 0, result.output
        assert "Snapshot recorded" in result.output
        files = list(isolated_history.glob("config_*.json"))
        assert len(files) == 1
        data = json.loads(files[0].read_text(encoding="utf-8"))
        assert data["reason"] == "pruned low-IC indicators"
        assert data["evidence"] == "reports/ic_2026-08-14.md"
        assert data["fingerprint"]
        assert "config" in data  # full config payload stored

    def test_record_requires_reason(self, isolated_history):
        result = runner.invoke(app, ["snapshot", "record"])
        assert result.exit_code != 0  # --reason is required (audit anchor)


@pytest.mark.unit
class TestSnapshotDiff:
    def test_diff_empty_history_reports_match(self, isolated_history):
        result = runner.invoke(app, ["snapshot", "diff"])
        assert result.exit_code == 0
        assert "matches the last snapshot" in result.output

    def test_diff_detects_drift_after_config_change(self, isolated_history):
        runner.invoke(app, ["snapshot", "record", "--reason", "baseline"])
        # Simulate the human applying a reviewed change.
        set_config({"indicator_battery": ["rsi", "atr"]})
        result = runner.invoke(app, ["snapshot", "diff"])
        assert result.exit_code == 0
        assert "drifted" in result.output
        assert "indicator_battery" in result.output

    def test_diff_identical_config_reports_match(self, isolated_history):
        runner.invoke(app, ["snapshot", "record", "--reason", "baseline"])
        result = runner.invoke(app, ["snapshot", "diff"])
        assert result.exit_code == 0
        assert "matches the last snapshot" in result.output


@pytest.mark.unit
class TestSnapshotList:
    def test_list_empty_history(self, isolated_history):
        result = runner.invoke(app, ["snapshot", "list"])
        assert result.exit_code == 0
        assert "No config snapshots" in result.output

    def test_list_shows_recorded_snapshots(self, isolated_history):
        runner.invoke(app, ["snapshot", "record", "--reason", "first change"])
        runner.invoke(app, ["snapshot", "record", "--reason", "second change"])
        result = runner.invoke(app, ["snapshot", "list"])
        assert result.exit_code == 0
        assert "first change" in result.output
        assert "second change" in result.output

    def test_list_respects_limit(self, isolated_history):
        for i in range(5):
            runner.invoke(app, ["snapshot", "record", "--reason", f"change {i}"])
        result = runner.invoke(app, ["snapshot", "list", "--limit", "2"])
        assert result.exit_code == 0
        assert "change 4" in result.output
        assert "change 3" in result.output
        assert "change 0" not in result.output
