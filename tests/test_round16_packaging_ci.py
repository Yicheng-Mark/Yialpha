"""Drift guards for the packaging, CI and documentation surface.

Three invariants that nothing else in the suite (or in CI) checks. Each one
failed silently in this repo's history or was one edit away from doing so:

* ``requirements.lock`` — what the README tells users to install from — must
  satisfy every dependency ``pyproject.toml`` declares for the scopes it
  covers (runtime + the extras named in its own generation header). CI's
  ``lockfile`` job installs the lock and then imports ``yialpha`` *from the
  checkout*, so the project's own metadata is never validated against it: a
  floor bumped in pyproject without regenerating the lock leaves every CI job
  green and breaks the user's first install.
* the lock must stay a real lock — exactly one ``==`` pin per distribution.
* the CI mypy gate must name the ``yialpha`` package root. It used to be a
  hand-maintained module list and had drifted far enough that four whole
  packages (instruments, ledger, perp, regime) sat outside the gate.

A fourth guard keeps the docs honest: every ``yialpha <command>`` the docs
tell a user to run must be registered on the real CLI.
"""
from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from packaging.version import Version

REPO_ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = REPO_ROOT / "pyproject.toml"
LOCK = REPO_ROOT / "requirements.lock"
CI_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "ci.yml"
DOCS = ("README.md", "README.zh-CN.md", "CLAUDE.md", "web/README.md")

_pyproject = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
_lock_text = LOCK.read_text(encoding="utf-8")

# The header records how the lock was generated, e.g.
#   uv pip compile --generate-hashes --python-version 3.12 \
#       --extra dev --extra web --output-file requirements.lock pyproject.toml
_header_extras = frozenset(re.findall(r"--extra\s+([A-Za-z0-9_-]+)", _lock_text))
_extra_names = frozenset(_pyproject["project"]["optional-dependencies"])


def _lock_requirements() -> list[Requirement]:
    """Every requirement line of the lock, continuations joined and hashes /
    ``# via`` comments stripped."""
    requirements = []
    for raw in _lock_text.splitlines():
        # one requirement per entry: `name==version \` then `--hash=` lines,
        # then a `# via` comment; the hash lines become empty and drop out
        parts = [
            part
            for part in raw.split("#", 1)[0].split()
            if not part.startswith("--hash")
        ]
        spec = " ".join(parts).rstrip("\\").strip()
        if spec:
            requirements.append(Requirement(spec))
    return requirements


def _declared() -> list[tuple[str, str]]:
    """(scope, requirement) for runtime deps and for every extra the lock
    actually covers — the lock cannot vouch for an extra it was not compiled
    with, so those are outside this guard by construction."""
    declared = [("runtime", req) for req in _pyproject["project"]["dependencies"]]
    for extra in sorted(_header_extras):
        declared.extend(
            (extra, req) for req in _pyproject["project"]["optional-dependencies"][extra]
        )
    return declared


# --------------------------------------------------------------------------- #
# requirements.lock <-> pyproject.toml
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_r16_lock_header_names_only_known_extras():
    """The covered scopes are read off the lock header; a typo there (or an
    extra renamed in pyproject) would silently narrow the guard below."""
    assert _header_extras, "lock header no longer records --extra flags"
    unknown = _header_extras - _extra_names
    assert not unknown, f"lock covers extras pyproject does not define: {sorted(unknown)}"


@pytest.mark.unit
@pytest.mark.parametrize(
    ("scope", "requirement"), _declared(), ids=lambda v: v if isinstance(v, str) else None
)
def test_r16_lock_satisfies_declared_dependency(scope, requirement):
    """A floor raised in pyproject must be matched by regenerating the lock."""
    name = canonicalize_name(Requirement(requirement).name)
    pins = [
        next(iter(req.specifier)).version
        for req in _lock_requirements()
        if canonicalize_name(req.name) == name
    ]
    assert pins, (
        f"{requirement} (scope '{scope}') is not in requirements.lock; regenerate the "
        "lock with the command in its header"
    )
    assert Requirement(requirement).specifier.contains(
        Version(pins[0]), prereleases=True
    ), f"lock pins {name}=={pins[0]}, below the declared {requirement}"


@pytest.mark.unit
def test_r16_lock_has_one_exact_pin_per_distribution():
    """A lock with a floating or a duplicated requirement is not a lock: the
    same file can resolve differently on two machines."""
    seen: dict[str, str] = {}
    duplicates: list[str] = []
    loose: list[str] = []
    for req in _lock_requirements():
        name = canonicalize_name(req.name)
        specifiers = list(req.specifier)
        if len(specifiers) != 1 or specifiers[0].operator not in ("==", "==="):
            loose.append(str(req))
        pin = specifiers[0].version if specifiers else "?"
        if name in seen and seen[name] != pin:
            duplicates.append(f"{name}: {seen[name]} vs {pin}")
        seen[name] = pin
    assert not loose, f"not pinned to an exact version: {loose}"
    assert not duplicates, f"conflicting pins: {duplicates}"
    assert len(seen) > 50, f"only {len(seen)} distributions in the lock?"


# --------------------------------------------------------------------------- #
# CI workflow
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_r16_ci_mypy_gate_names_the_package_root():
    """The gate must be ``mypy ... yialpha``: naming sub-packages by hand is
    how four of them fell outside it."""
    workflow = CI_WORKFLOW.read_text(encoding="utf-8")
    match = re.search(r"^\s*run: mypy (.+?)\s*$", workflow, re.M)
    assert match, "no `run: mypy ...` step in ci.yml"
    targets = [token for token in match.group(1).split() if not token.startswith("-")]
    assert "yialpha" in targets, (
        f"mypy gate targets {targets}; it must name the package root so every "
        "sub-package is covered"
    )


@pytest.mark.unit
def test_r16_coverage_floor_is_not_twenty_points_below_reality():
    """The floor sat at 75 while every lane measured 95.5%, so it gated
    nothing. Lowering it again should be a deliberate edit of this test."""
    floor = _pyproject["tool"]["coverage"]["report"]["fail_under"]
    assert floor >= 90, f"coverage fail_under is {floor}; it was re-based to 90 on 2026-11-23"


def _pytest_cov_targets_in_ci() -> list[frozenset[str]]:
    """The ``--cov=<target>`` set of every ``run: pytest ...`` line in ci.yml.

    A ``--cov`` flag on the command line replaces ``[tool.coverage.run]
    source``, so a job that names fewer targets than the config measures less
    than the config says — and nothing fails, because the floor still holds on
    the narrower scope."""
    workflow = CI_WORKFLOW.read_text(encoding="utf-8")
    scopes = []
    for line in workflow.splitlines():
        if not re.match(r"^\s*run: pytest\b", line):
            continue
        targets = frozenset(re.findall(r"--cov=([A-Za-z0-9_.-]+)", line))
        assert targets, f"pytest step names no --cov target: {line.strip()}"
        scopes.append(targets)
    return scopes


@pytest.mark.unit
def test_r17_ci_coverage_scope_matches_pyproject_source():
    """ci.yml measured only ``yialpha`` while ``web/`` and ``scripts/`` (the
    run-in-place trees) were outside every gate; the config and the command
    lines must name the same scope."""
    scopes = _pytest_cov_targets_in_ci()
    # the ubuntu, weekly and windows test jobs — a job that loses its pytest
    # step would otherwise drop out of this guard silently
    assert len(scopes) >= 3, f"only {len(scopes)} pytest steps in ci.yml"
    configured = frozenset(_pyproject["tool"]["coverage"]["run"]["source"])
    for scope in scopes:
        assert scope == configured, f"ci.yml measures {sorted(scope)}; pyproject says {sorted(configured)}"


@pytest.mark.unit
def test_r17_coverage_scope_includes_the_run_in_place_trees():
    """``web/`` and ``scripts/`` are not packaged modules, so they are only
    measured when they are named as source directories."""
    configured = set(_pyproject["tool"]["coverage"]["run"]["source"])
    missing = {"yialpha", "web", "scripts"} - configured
    assert not missing, f"coverage source is missing: {sorted(missing)}"


# --------------------------------------------------------------------------- #
# docs <-> CLI
# --------------------------------------------------------------------------- #
def _cli_commands() -> tuple[set[str], dict[str, set[str]]]:
    from yialpha.cli.main import app

    def _name(entry):
        return entry.name or (entry.callback.__name__ or "").replace("_", "-")

    commands = {_name(cmd) for cmd in app.registered_commands}
    groups = {
        _name(group): {_name(cmd) for cmd in group.typer_instance.registered_commands}
        for group in app.registered_groups
    }
    return commands, groups


# `yialpha` as the command word: at the start of a line, after a shell prompt
# or operator, or as the first word of an inline code span. That excludes
# `conda create -n yialpha python=3.12` / `conda activate yialpha`.
_MENTION = re.compile(
    r"(?:^|(?<=[`$;&|]))\s*yialpha\s+([a-z][a-z0-9-]*)(?:\s+([a-z][a-z0-9-]*))?",
    re.M,
)


def _documented_mentions() -> list[tuple[str, str, str]]:
    mentions = []
    for doc in DOCS:
        text = (REPO_ROOT / doc).read_text(encoding="utf-8")
        for match in _MENTION.finditer(text):
            mentions.append((doc, match.group(1), match.group(2) or ""))
    return mentions


@pytest.mark.unit
def test_r16_docs_mention_the_cli_at_all():
    """If the regex stops matching anything, the parametrized guard below
    would pass vacuously."""
    assert len(_documented_mentions()) >= 10


@pytest.mark.unit
@pytest.mark.parametrize(
    ("doc", "command", "subcommand"),
    sorted(set(_documented_mentions())),
    ids=lambda v: v,
)
def test_r16_documented_cli_command_exists(doc, command, subcommand):
    """A command renamed or removed in code must be updated in the docs too."""
    commands, groups = _cli_commands()
    assert command in commands | set(groups), (
        f"{doc} tells users to run `yialpha {command}`, which is not a "
        f"registered command (have: {sorted(commands | set(groups))})"
    )
    if command in groups and subcommand:
        assert subcommand in groups[command], (
            f"{doc} tells users to run `yialpha {command} {subcommand}`; "
            f"`{command}` only has {sorted(groups[command])}"
        )


def _readme_structure_block(readme: str) -> str:
    text = (REPO_ROOT / readme).read_text(encoding="utf-8")
    for block in re.findall(r"```text\n(.*?)```", text, re.S):
        if "├── yialpha/" in block:
            return block
    raise AssertionError(f"{readme} has no project-structure tree")


def _yialpha_packages() -> list[str]:
    return sorted(
        path.name
        for path in (REPO_ROOT / "yialpha").iterdir()
        if path.is_dir() and (path / "__init__.py").exists()
    )


@pytest.mark.unit
@pytest.mark.parametrize("readme", ("README.md", "README.zh-CN.md"))
def test_r16_readme_tree_lists_every_package(readme):
    """The tree had drifted to 8 of 13 sub-packages — the same four that sat
    outside the mypy gate. A package added without a tree line fails here."""
    tree = _readme_structure_block(readme)
    missing = [
        name for name in _yialpha_packages()
        if not re.search(rf"[├└]── {re.escape(name)}/", tree)
    ]
    assert not missing, f"{readme} project tree is missing sub-packages: {missing}"
    assert _yialpha_packages(), "no sub-packages found under yialpha/?"


@pytest.mark.unit
@pytest.mark.parametrize("readme", ("README.md", "README.zh-CN.md"))
def test_r16_readme_tree_lists_every_cli_command(readme):
    """The `cli/` line names the commands a reader can run; a command added
    to the app without a mention here goes undocumented in the structure map."""
    tree = _readme_structure_block(readme)
    cli_line = next(
        (line for line in tree.splitlines() if re.search(r"[├└]── cli/", line)),
        None,
    )
    assert cli_line, f"{readme} project tree has no cli/ line"
    commands, _groups = _cli_commands()
    missing = sorted(name for name in commands if not re.search(rf"\b{name}\b", cli_line))
    assert not missing, f"{readme} cli/ line does not mention: {missing}"
