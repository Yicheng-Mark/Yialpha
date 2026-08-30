"""Packaging regressions for the collision-free YiAlpha CLI namespace."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.unit
def test_pyproject_entrypoint_and_package_data_use_unique_namespace():
    source = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'yialpha = "yialpha.cli.main:app"' in source
    assert 'include = ["yialpha*"]' in source
    assert '"yialpha.cli" = ["static/*"]' in source


@pytest.mark.unit
def test_unique_cli_starts_even_when_top_level_cli_is_not_a_package(tmp_path):
    # Reproduce the original failure mode: an unrelated ``cli.py`` shadows the
    # generic package name. The installed entrypoint must never import it.
    (tmp_path / "cli.py").write_text("MARKER = 'unrelated cli module'\n", encoding="utf-8")
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(tmp_path), str(ROOT), env.get("PYTHONPATH", "")]
    )

    completed = subprocess.run(
        [sys.executable, "-m", "yialpha.cli.main", "--help"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=30,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "YiAlpha CLI" in completed.stdout


@pytest.mark.unit
def test_docker_context_excludes_environment_secrets():
    patterns = (ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()
    assert ".env" in patterns
    assert ".env.*" in patterns
    assert "!.env.example" in patterns
