"""Tests for the unified ``results_dir`` resolution in the web store.

The web read side (``web.store``) must derive its logs root from
``YIALPHA_RESULTS_DIR`` — the same env the CLI/batch write side uses — so a
custom results dir stays aligned between writer and reader. Before this fix
the web hardcoded ``~/.yialpha/logs``, diverging whenever the env was set.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _undo_module_reloads():
    """Put the modules this file ``importlib.reload()``s back as they were.

    ``reload`` re-executes a module into its existing namespace, so after a
    test that reloaded ``web.store`` / ``yialpha.default_config`` under a
    patched ``YIALPHA_RESULTS_DIR`` the modules keep the roots built from it
    for the rest of the session -- ``monkeypatch`` restores ``os.environ``
    only. Snapshot both namespaces before the test and restore them after
    (same objects as before, whatever order the fixtures are torn down in).
    """
    import web.store
    import yialpha.default_config

    modules = (web.store, yialpha.default_config)
    saved = [(module, dict(module.__dict__)) for module in modules]
    yield
    for module, namespace in saved:
        module.__dict__.clear()
        module.__dict__.update(namespace)


@pytest.mark.unit
class TestResultsDirUnification:
    def test_env_results_dir_is_used(self, monkeypatch):
        monkeypatch.setenv("YIALPHA_RESULTS_DIR", str(Path("/tmp/custom-yialpha")))
        import web.store as store
        importlib.reload(store)
        assert Path("/tmp/custom-yialpha") == store.LOGS_ROOT
        assert Path("/tmp/custom-yialpha") / "reports" == store.REPORTS_ROOT

    def test_fallback_to_default_when_env_unset(self, monkeypatch):
        monkeypatch.delenv("YIALPHA_RESULTS_DIR", raising=False)
        import web.store as store
        importlib.reload(store)
        expected = Path.home() / ".yialpha" / "logs"
        assert expected == store.LOGS_ROOT
        assert expected / "reports" == store.REPORTS_ROOT

    def test_resolve_function_matches_default_config_source(self, monkeypatch):
        # The store's resolver must agree with default_config's source env.
        # Both read YIALPHA_RESULTS_DIR at module-eval time, so reload
        # default_config under the same env to compare apples-to-apples.
        custom = str(Path("/opt/yialpha-logs"))
        monkeypatch.setenv("YIALPHA_RESULTS_DIR", custom)
        import yialpha.default_config as dc
        importlib.reload(dc)
        import web.store as store
        importlib.reload(store)
        assert str(store.LOGS_ROOT) == dc.DEFAULT_CONFIG["results_dir"] == custom
