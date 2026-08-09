"""Tests for the unified ``results_dir`` resolution in the web store.

The web read side (``web.store``) must derive its logs root from
``YIAGENTS_RESULTS_DIR`` — the same env the CLI/batch write side uses — so a
custom results dir stays aligned between writer and reader. Before this fix
the web hardcoded ``~/.yiagents/logs``, diverging whenever the env was set.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest


@pytest.mark.unit
class TestResultsDirUnification:
    def test_env_results_dir_is_used(self, monkeypatch):
        monkeypatch.setenv("YIAGENTS_RESULTS_DIR", str(Path("/tmp/custom-yiagents")))
        import web.store as store
        importlib.reload(store)
        assert Path("/tmp/custom-yiagents") == store.LOGS_ROOT
        assert Path("/tmp/custom-yiagents") / "reports" == store.REPORTS_ROOT

    def test_fallback_to_default_when_env_unset(self, monkeypatch):
        monkeypatch.delenv("YIAGENTS_RESULTS_DIR", raising=False)
        import web.store as store
        importlib.reload(store)
        expected = Path.home() / ".yiagents" / "logs"
        assert expected == store.LOGS_ROOT
        assert expected / "reports" == store.REPORTS_ROOT

    def test_resolve_function_matches_default_config_source(self, monkeypatch):
        # The store's resolver must agree with default_config's source env.
        # Both read YIAGENTS_RESULTS_DIR at module-eval time, so reload
        # default_config under the same env to compare apples-to-apples.
        custom = str(Path("/opt/yiagents-logs"))
        monkeypatch.setenv("YIAGENTS_RESULTS_DIR", custom)
        import yiagents.default_config as dc
        importlib.reload(dc)
        import web.store as store
        importlib.reload(store)
        assert str(store.LOGS_ROOT) == dc.DEFAULT_CONFIG["results_dir"] == custom
