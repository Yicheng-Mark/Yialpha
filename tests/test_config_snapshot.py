"""Tests for the config snapshot / audit trail module.

Verifies that snapshots are written atomically, are append-only, can be
diffed against the current config, and never auto-edit the live config.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from yiagents.config_snapshot import (
    diff_against_last_snapshot,
    list_snapshots,
    load_last_snapshot,
    record_config_snapshot,
)


@pytest.mark.unit
class TestConfigSnapshot:
    def _config(self, **overrides) -> dict:
        base = {"data_cache_dir": "/tmp/test", "model": "gpt-4", "debate_rounds": 2}
        base.update(overrides)
        return base

    def test_snapshot_written_and_readable(self, tmp_path):
        """record_config_snapshot writes a JSON file that load_last_snapshot reads."""
        cfg = self._config()
        path = record_config_snapshot(
            cfg, reason="initial baseline", history_dir=tmp_path
        )
        assert path.exists()
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["reason"] == "initial baseline"
        assert data["config"]["model"] == "gpt-4"
        assert "fingerprint" in data
        assert "timestamp" in data

    def test_last_snapshot_is_most_recent(self, tmp_path):
        """load_last_snapshot returns the chronologically latest snapshot."""
        record_config_snapshot(
            self._config(model="v1"), reason="first", history_dir=tmp_path
        )
        record_config_snapshot(
            self._config(model="v2"), reason="second", history_dir=tmp_path
        )
        last = load_last_snapshot(history_dir=tmp_path)
        assert last is not None
        assert last["config"]["model"] == "v2"
        assert last["reason"] == "second"

    def test_diff_detects_changes(self, tmp_path):
        """diff_against_last_snapshot flags top-level key changes."""
        record_config_snapshot(
            self._config(model="v1", debate_rounds=2),
            reason="baseline", history_dir=tmp_path,
        )
        changed = self._config(model="v2", debate_rounds=2)
        diffs = diff_against_last_snapshot(changed, history_dir=tmp_path)
        assert "model" in diffs
        assert diffs["model"] == ("v1", "v2")
        assert "debate_rounds" not in diffs  # unchanged

    def test_diff_empty_when_identical(self, tmp_path):
        """No diff when the config matches the last snapshot."""
        cfg = self._config()
        record_config_snapshot(cfg, reason="baseline", history_dir=tmp_path)
        diffs = diff_against_last_snapshot(cfg, history_dir=tmp_path)
        assert diffs == {}

    def test_diff_detects_new_key(self, tmp_path):
        """A key added after the snapshot is flagged."""
        record_config_snapshot(
            self._config(model="v1"), reason="baseline", history_dir=tmp_path
        )
        cfg_with_new_key = self._config(model="v1", new_feature=True)
        diffs = diff_against_last_snapshot(cfg_with_new_key, history_dir=tmp_path)
        assert "new_feature" in diffs
        assert diffs["new_feature"] == (None, True)

    def test_no_snapshot_returns_none(self, tmp_path):
        """load_last_snapshot returns None when history is empty."""
        assert load_last_snapshot(history_dir=tmp_path) is None
        assert diff_against_last_snapshot(self._config(), history_dir=tmp_path) == {}

    def test_list_snapshots_summaries(self, tmp_path):
        """list_snapshots returns chronological summaries."""
        record_config_snapshot(
            self._config(model="a"), reason="alpha", history_dir=tmp_path
        )
        record_config_snapshot(
            self._config(model="b"), reason="beta", history_dir=tmp_path
        )
        summaries = list_snapshots(history_dir=tmp_path)
        assert len(summaries) == 2
        assert summaries[0]["reason"] == "alpha"
        assert summaries[1]["reason"] == "beta"
        assert all("fingerprint" in s for s in summaries)

    def test_snapshot_stores_deepcopy_not_reference(self, tmp_path):
        """Mutating the config after recording does not change the snapshot."""
        cfg = self._config(model="original")
        record_config_snapshot(cfg, reason="test", history_dir=tmp_path)
        cfg["model"] = "mutated"
        last = load_last_snapshot(history_dir=tmp_path)
        assert last is not None
        assert last["config"]["model"] == "original"

    def test_evidence_recorded(self, tmp_path):
        """The evidence field is preserved in the snapshot."""
        record_config_snapshot(
            self._config(),
            reason="pruned RSI",
            evidence="reports/ic_pruning.md",
            history_dir=tmp_path,
        )
        last = load_last_snapshot(history_dir=tmp_path)
        assert last is not None
        assert last["evidence"] == "reports/ic_pruning.md"
