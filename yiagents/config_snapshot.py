"""Config change audit + snapshot mechanism for the self-improvement loop.

This module provides the "snap to history" half of the self-improvement
pipeline described in the optimization plan:

    IC evidence (prune_indicators_cli)
        → change suggestion (human reviews)
        → snapshot lands here (human applies + records)
        → A/B comparison (run_analyst_parallel_ab.py)

Design constraints (fail-closed self-improvement):
- **Never auto-edits the live config.** This module only *records* snapshots
  and diffs. The actual ``set_config`` / env-var change is always done by a
  human.
- **Every snapshot carries provenance**: timestamp, reason, the source evidence
  (e.g. which IC report prompted it), and the git commit hash if available.
- **Snapshots are append-only JSON** written to ``<data_cache_dir>/config_history/``,
  so the audit trail is tamper-evident (any overwrite is visible on disk).

Usage::

    from yiagents.config_snapshot import record_config_snapshot

    record_config_snapshot(
        config=graph.config,
        reason="Pruned RSI_14 after IC < 0.03 for 35 consecutive days",
        evidence="reports/ic_pruning_2026-08-08.md",
    )

To diff the current config against the last snapshot::

    from yiagents.config_snapshot import diff_against_last_snapshot
    diff = diff_against_last_snapshot(graph.config)
    if diff:
        print("Config has drifted from the last recorded snapshot:")
        for key, (old, new) in diff.items():
            print(f"  {key}: {old!r} -> {new!r}")
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

#: Subdirectory under data_cache_dir for config history.
_HISTORY_SUBDIR = "config_history"


def _history_dir(config: dict[str, Any] | None = None) -> Path:
    """Resolve the config history directory, creating it if missing."""
    cache_dir = (
        (config or {}).get("data_cache_dir")
        or os.path.join(os.path.expanduser("~"), ".yiagents", "cache")
    )
    path = Path(cache_dir) / _HISTORY_SUBDIR
    path.mkdir(parents=True, exist_ok=True)
    return path


def _git_commit() -> str | None:
    """Best-effort current git commit hash (None if not in a repo or git missing)."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass
    return None


def _config_fingerprint(config: dict[str, Any]) -> str:
    """A short SHA-256 fingerprint of the config for quick identity checks."""
    blob = json.dumps(config, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:12]


def record_config_snapshot(
    config: dict[str, Any],
    *,
    reason: str,
    evidence: str = "",
    history_dir: Path | None = None,
) -> Path:
    """Write a timestamped JSON snapshot of ``config`` to the history dir.

    Args:
        config: The full config dict to snapshot (a deepcopy is stored).
        reason: Human-readable reason for this snapshot (e.g. "Pruned RSI_14
            after IC collapse"). This is the audit-trail anchor.
        evidence: Optional path to or description of the supporting evidence
            (e.g. an IC pruning report file).
        history_dir: Override the history directory (for testing). Defaults to
            ``<data_cache_dir>/config_history/``.

    Returns:
        The path to the written snapshot file.
    """
    import copy

    target_dir = history_dir or _history_dir(config)
    target_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now()
    snapshot = {
        "timestamp": ts.isoformat(timespec="milliseconds"),
        "reason": reason,
        "evidence": evidence,
        "git_commit": _git_commit(),
        "fingerprint": _config_fingerprint(config),
        "config": copy.deepcopy(config),
    }

    # Use millisecond precision in the filename so two snapshots recorded in
    # the same second still sort chronologically.
    filename = f"config_{ts.strftime('%Y%m%d_%H%M%S_%f')}_{snapshot['fingerprint']}.json"
    path = target_dir / filename

    # Atomic write: temp file + os.replace so a crash never leaves a partial file.
    tmp_path = path.with_suffix(".tmp")
    tmp_path.write_text(
        json.dumps(snapshot, indent=2, default=str, ensure_ascii=False),
        encoding="utf-8",
    )
    tmp_path.replace(path)

    logger.info(
        "Config snapshot recorded: %s (fingerprint %s, reason: %s)",
        path.name, snapshot["fingerprint"], reason,
    )
    return path


def load_last_snapshot(
    config: dict[str, Any] | None = None,
    history_dir: Path | None = None,
) -> dict[str, Any] | None:
    """Load the most recent snapshot from the history dir, or ``None`` if empty."""
    target_dir = history_dir or _history_dir(config)
    snapshots = sorted(target_dir.glob("config_*.json"))
    if not snapshots:
        return None
    try:
        return json.loads(snapshots[-1].read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Could not read last config snapshot %s: %s", snapshots[-1], exc)
        return None


def diff_against_last_snapshot(
    config: dict[str, Any],
    *,
    history_dir: Path | None = None,
) -> dict[str, tuple[Any, Any]]:
    """Diff ``config`` against the last recorded snapshot at the top level.

    Returns a ``{key: (old_value, new_value)}`` dict for every top-level key
    that differs. An empty dict means the config matches the last snapshot.
    Keys present in one but not the other are included with ``None`` for the
    missing side.
    """
    last = load_last_snapshot(config, history_dir=history_dir)
    if last is None:
        return {}
    old_config = last.get("config", {})
    diffs: dict[str, tuple[Any, Any]] = {}
    all_keys = set(config) | set(old_config)
    for key in all_keys:
        old_val = old_config.get(key)
        new_val = config.get(key)
        if old_val != new_val:
            diffs[key] = (old_val, new_val)
    return diffs


def list_snapshots(
    config: dict[str, Any] | None = None,
    history_dir: Path | None = None,
) -> list[dict[str, Any]]:
    """Return a summary list of all snapshots (timestamp, reason, fingerprint)."""
    target_dir = history_dir or _history_dir(config)
    summaries: list[dict[str, Any]] = []
    for path in sorted(target_dir.glob("config_*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            summaries.append({
                "file": path.name,
                "timestamp": data.get("timestamp", ""),
                "reason": data.get("reason", ""),
                "fingerprint": data.get("fingerprint", ""),
                "git_commit": data.get("git_commit", ""),
            })
        except (json.JSONDecodeError, OSError):
            continue
    return summaries
