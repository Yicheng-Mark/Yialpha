"""Round-17 S12 (test_hygiene): pins that keep the test suite itself honest.

Cheap AST / text checks over ``tests/`` plus a few object-identity checks. Each
one closes a hole this round actually fell into:

* a test file that ``importlib.reload``-ed a module and left the reloaded
  objects behind for the rest of the session (two round-17 sessions went red,
  6 and 20 tests, on state leaked that way);
* a real ``time.sleep`` standing in for a hand-shake in a concurrency test;
* a ``C:/...`` literal used as "an absolute path" (it is relative on Linux);
* warnings nobody asserted (45 at the start of this card).

Every detector is a plain function over ``(node, context)`` pairs, and each
pin has an in-memory self-test that feeds it a known offender: a detector that
stops detecting fails here instead of passing silently.
"""

from __future__ import annotations

import ast
import functools
import re
import sys
import textwrap
import tomllib
import types
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, replace
from pathlib import Path

import pytest

import web.store as _web_store
import yialpha.agents.analysts.sentiment_analyst as _sentiment
import yialpha.cli.utils as _cli_utils
import yialpha.dataflows.perp_bundle as _perp_bundle
import yialpha.dataflows.stockstats_utils as _stockstats
import yialpha.default_config as _default_config
import yialpha.llm_clients.openai_client as _openai_client

TESTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TESTS_DIR.parent

# Allow-lists are exact sets and are checked in both directions: no new site
# without a listed reason, and no entry whose site is gone.
_MAX_SLEEP_SECONDS = 0.05
_SLEEP_ALLOWLIST = {
    ("test_perp_bundle.py", "test_bundle_leg_timeout_reads_module_constant"): (
        "The wait under test is a wall-clock timeout (future.result(timeout=...)), and "
        "the pool joins that leg before returning: the leg cannot be cancelled, so it "
        "has to outlast the threshold and then end on its own."
    ),
}
_DRIVE_LETTER_ALLOWLIST = {
    ("test_round17_dataflows_vendors.py", "test_r17_disk_cache_path_escaping_to_another_drive_is_refused"): (
        "The subject of the test IS a Windows drive letter."
    ),
    ("test_round17_dataflows_vendors.py", "test_r17_disk_cache_same_drive_escape_refuses_before_any_fetch"): (
        "The subject of the test IS a Windows drive letter."
    ),
    ("test_disk_cache.py", "test_unsafe_filenames_rejected"): (
        "A cache filename the sanitizer has to refuse; never opened."
    ),
    ("test_round17_scripts_e2e.py", "TestRobustOneChildEnv.test_r17_robust_one_child_env_exported_value_wins_over_the_default"): (
        "An exported env value the child must receive verbatim; never opened."
    ),
    ("test_round17_web_runtime.py", "test_r17_app_analyze_option_like_and_non_ascii_tickers_never_spawn"): (
        "A ticker the endpoint has to refuse; never used as a path."
    ),
}
_SYSPATH_ALLOWLIST = {
    ("test_rank_signals.py", "<module>"): (
        "scripts/ is not a package, and the script's own bootstrap inserts its directory."
    ),
    ("test_round13_ticket_shard_regressions.py", "<module>"): (
        "scripts/ is not a package, and the script's own bootstrap inserts its directory."
    ),
}

Finding = tuple[int, str, str]  # line, outermost enclosing function, detail
Site = tuple[str, str]  # file relative to tests/, outermost enclosing function
Nodes = Iterable[tuple[ast.AST, "_Ctx"]]


# --- one walk with context ----------------------------------------------------
_FUNC_TYPES = (ast.FunctionDef, ast.AsyncFunctionDef)
_OWNER_TYPES = _FUNC_TYPES + (ast.ClassDef,)
_CASE_TYPES = {"TestCase", "FunctionTestCase"}


def _dotted(node: ast.AST | None) -> str:
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return ""


def _last(node: ast.AST | None) -> str:
    return _dotted(node).rsplit(".", 1)[-1]


def _qual(parent: str, name: str) -> str:
    return name if parent == "<module>" else f"{parent}.{name}"


@dataclass(frozen=True)
class _Ctx:
    """Where a node sits: the enclosing function, and the scopes that excuse it."""

    func: str = "<module>"  # innermost enclosing function (a class name for methods)
    root: str = "<module>"  # outermost enclosing function ("Class.method" for methods)
    in_testcase: bool = False  # inside a unittest.TestCase subclass
    guarded: bool = False  # inside a try that has a finally
    scoped: bool = False  # inside a `with warnings.catch_warnings()` block


_YIELDED = frozenset({
    ast.Call, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Assign, ast.AugAssign,
    ast.Delete, ast.JoinedStr,
})
_CONTEXT_CHANGERS = frozenset({
    ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef, ast.Try, ast.With, ast.AsyncWith,
})


class _Exit:
    """Sentinel that restores the context that held one level up."""

    __slots__ = ("ctx",)

    def __init__(self, ctx: _Ctx) -> None:
        self.ctx = ctx


def _child_context(node: ast.AST, kind: type[ast.AST], ctx: _Ctx) -> _Ctx:
    if kind is ast.ClassDef:
        is_case = any(_last(base) in _CASE_TYPES for base in node.bases)  # type: ignore[attr-defined]
        return replace(ctx, func=_qual(ctx.func, node.name), in_testcase=ctx.in_testcase or is_case)  # type: ignore[attr-defined]
    if kind in _FUNC_TYPES:
        name = _qual(ctx.func, node.name)  # type: ignore[attr-defined]
        return replace(
            ctx, func=name, root=name if ctx.root == "<module>" else ctx.root, guarded=False, scoped=False,
        )
    if kind is ast.Try:
        return replace(ctx, guarded=True) if node.finalbody else ctx  # type: ignore[attr-defined]
    if any(  # `with warnings.catch_warnings(): ...`
        isinstance(item.context_expr, ast.Call)
        and _last(item.context_expr.func) == "catch_warnings"
        for item in node.items  # type: ignore[attr-defined]
    ):
        return replace(ctx, scoped=True)
    return ctx


def _walk(tree: ast.AST) -> Iterator[tuple[ast.AST, _Ctx]]:
    """Yield the nodes the detectors look at, each with the context it sits in."""
    stack: list[ast.AST | _Exit] = [tree]
    ctx = _Ctx()
    while stack:
        item = stack.pop()
        if type(item) is _Exit:
            ctx = item.ctx
            continue
        node = item
        kind = type(node)
        if kind in _YIELDED or (kind is ast.Constant and type(node.value) is str):
            yield node, ctx
        if kind in _CONTEXT_CHANGERS:
            child = _child_context(node, kind, ctx)
            if child is not ctx:
                stack.append(_Exit(ctx))
                ctx = child
        stack.extend(ast.iter_child_nodes(node))


def _fixture_autouse(node: ast.AST) -> bool | None:
    """``None`` when the function is not a fixture, else whether it is autouse."""
    for deco in node.decorator_list:
        target = deco.func if isinstance(deco, ast.Call) else deco
        if _last(target) == "fixture":
            return isinstance(deco, ast.Call) and any(
                kw.arg == "autouse"
                and isinstance(kw.value, ast.Constant)
                and kw.value.value is True
                for kw in deco.keywords
            )
    return None


def _numeric_literal(node: ast.AST | None) -> float | None:
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return float(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USUB):
        inner = _numeric_literal(node.operand)
        return None if inner is None else -inner
    return None


def _literal_head(node: ast.AST) -> str | None:
    """The leading text of a string literal or of an f-string, else ``None``."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr) and node.values:
        first = node.values[0]
        if isinstance(first, ast.Constant) and isinstance(first.value, str):
            return first.value
    return None


# --- detectors -----------------------------------------------------------------
def _sleep_findings(nodes: Nodes) -> list[Finding]:
    out: list[Finding] = []
    for node, ctx in nodes:
        if isinstance(node, ast.Call) and _last(node.func) == "sleep" and node.args:
            seconds = _numeric_literal(node.args[0])
            if seconds is not None and abs(seconds) > _MAX_SLEEP_SECONDS:
                out.append((node.lineno, ctx.root, f"sleep({seconds:g})"))
    return out


def _is_place_holder(stmt: ast.stmt) -> bool:
    return isinstance(stmt, ast.Pass) or (
        isinstance(stmt, ast.Expr)
        and isinstance(stmt.value, ast.Constant)
        and stmt.value.value is Ellipsis
    )


def _body_findings(nodes: Nodes) -> list[Finding]:
    out: list[Finding] = []
    for node, ctx in nodes:
        if not (isinstance(node, _FUNC_TYPES) and node.name.startswith("test_")):
            continue
        body = node.body
        if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
            body = body[1:]  # the docstring
        if all(_is_place_holder(stmt) for stmt in body):
            out.append((node.lineno, ctx.root, "empty body"))
        for sub in ast.walk(node):
            if isinstance(sub, ast.Assert) and isinstance(sub.test, ast.Constant):
                out.append((sub.lineno, ctx.root, f"assert {ast.unparse(sub.test)}"))
    return out


_PYTEST_SKIP = {"skip": "skip", "skipif": "skipif", "xfail": "xfail"}
_UNITTEST_SKIP = {
    "unittest.skip": "skip", "unittest.skipIf": "uskipif", "unittest.skipUnless": "uskipif",
    "unittest.expectedFailure": "xfail",
}


def _skip_api(name: str) -> str | None:
    """The skip / xfail API a dotted name denotes, else ``None``.

    pytest's markers are recognised under any alias of the module
    (``pt.mark.skipif``), plus the call forms ``pytest.skip(...)`` and their
    bare imports, plus unittest's. ``pytest.importorskip`` is not one: the
    module it names is the reason.
    """
    parts = name.split(".")
    if len(parts) >= 2 and parts[-2] == "mark":  # the markers, under any alias of pytest
        return _PYTEST_SKIP.get(parts[-1])
    if name in {"skip", "skipif", "xfail", "pytest.skip", "pytest.skipif", "pytest.xfail"}:
        return _PYTEST_SKIP[parts[-1]]
    if name in _UNITTEST_SKIP:
        return _UNITTEST_SKIP[name]
    return "skip" if parts[-1] == "skipTest" else None


def _skip_reason_ok(call: ast.Call, api: str) -> bool:
    for kw in call.keywords:
        if kw.arg == "reason":
            return not (isinstance(kw.value, ast.Constant) and kw.value.value in ("", None))
    if api == "skipif":
        return False  # pytest's positionals are the condition
    if api == "uskipif":
        return len(call.args) >= 2  # unittest's second positional is the reason
    return bool(call.args)


def _skip_findings(nodes: Nodes) -> list[Finding]:
    out: list[Finding] = []
    seen_decorators: set[int] = set()
    for node, ctx in nodes:
        if isinstance(node, ast.Call):
            if id(node) in seen_decorators:
                continue  # reported against the function it decorates
            api = _skip_api(_dotted(node.func))
            if api is not None and not _skip_reason_ok(node, api):
                out.append((node.lineno, ctx.root, f"{_dotted(node.func)}() without a reason"))
        elif isinstance(node, _OWNER_TYPES):
            for deco in node.decorator_list:
                seen_decorators.add(id(deco))
                bare = deco.func if isinstance(deco, ast.Call) else deco
                name = _dotted(bare)
                api = _skip_api(name)
                if api is None:
                    continue
                if not isinstance(deco, ast.Call) or not _skip_reason_ok(deco, api):
                    out.append((bare.lineno, ctx.root, f"@{name} without a reason"))
    return out


def _is_environ(node: ast.AST) -> bool:
    """``os.environ`` through any alias: a name chain that ends in ``environ``."""
    name = _dotted(node)
    return name == "environ" or name.endswith(".environ")


def _is_env_write_call(name: str) -> bool:
    parts = name.split(".")
    mutating = {"pop", "update", "setdefault"}
    return (
        len(parts) >= 2 and parts[-2] == "environ" and parts[-1] in mutating
    ) or parts[-1] in {"putenv", "unsetenv"}


def _env_findings(nodes: Nodes) -> list[Finding]:
    out: list[Finding] = []
    for node, ctx in nodes:
        if ctx.in_testcase or ctx.guarded:
            continue  # a unittest setUp/tearDown pair, or a try/finally that restores
        if isinstance(node, (ast.Assign, ast.AugAssign, ast.Delete)):
            targets = node.targets if isinstance(node, (ast.Assign, ast.Delete)) else [node.target]
            if any(isinstance(t, ast.Subscript) and _is_environ(t.value) for t in targets):
                out.append((node.lineno, ctx.root, ast.unparse(node)[:80]))
        elif isinstance(node, ast.Call) and _is_env_write_call(_dotted(node.func)):
            out.append((node.lineno, ctx.root, ast.unparse(node)[:80]))
    return out


_DRIVE_RE = re.compile(r"^[A-Za-z]:[\\/]")


def _drive_findings(nodes: Nodes) -> list[Finding]:
    out: list[Finding] = []
    for node, ctx in nodes:
        head = _literal_head(node)
        if head is not None and _DRIVE_RE.match(head):
            out.append((node.lineno, ctx.root, repr(head[:40])))
    return out


_WARN_VERBS = {"simplefilter", "filterwarnings", "resetwarnings"}


def _warn_findings(nodes: Nodes) -> list[Finding]:
    out: list[Finding] = []
    for node, ctx in nodes:
        if isinstance(node, ast.Call) and _last(node.func) in _WARN_VERBS:
            action = node.args[0] if node.args else None
            ignoring = (
                isinstance(action, ast.Constant)
                and isinstance(action.value, str)
                and action.value.strip().lower().startswith("ignore")
            )
            if ctx.func == "<module>":
                out.append((node.lineno, ctx.root, "warning filter set at import"))
            elif ignoring and not ctx.scoped:
                out.append((node.lineno, ctx.root, f"unscoped {_dotted(node.func)}(...)"))
        elif isinstance(node, _OWNER_TYPES):
            for deco in node.decorator_list:
                first = deco.args[0] if isinstance(deco, ast.Call) and deco.args else None
                if (
                    isinstance(deco, ast.Call)
                    and _last(deco.func) == "filterwarnings"
                    and isinstance(first, ast.Constant)
                    and str(first.value).lower().startswith("ignore")
                ):
                    out.append((deco.lineno, ctx.root, f"@{_dotted(deco.func)}({first.value})"))
    return out


_SYSPATH_MUTATORS = {"insert", "append", "remove", "extend"}


def _is_sys_path_edit_call(name: str) -> bool:
    """``<any alias of sys>.path.<mutator>(...)``; ``self.path`` is a test's own attribute."""
    parts = name.split(".")
    return (
        len(parts) == 3 and parts[1] == "path" and parts[2] in _SYSPATH_MUTATORS
        and parts[0] not in {"self", "cls"}
    )


def _syspath_findings(nodes: Nodes) -> list[Finding]:
    out: list[Finding] = []
    for node, ctx in nodes:
        if ctx.guarded:
            continue
        if isinstance(node, ast.Call) and _is_sys_path_edit_call(_dotted(node.func)):
            out.append((node.lineno, ctx.root, ast.unparse(node)[:80]))
        elif isinstance(node, (ast.Assign, ast.AugAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            touched = any(
                _dotted(t) == "sys.path" or _dotted(getattr(t, "value", None)) == "sys.path"
                for t in targets
            )
            if touched:
                out.append((node.lineno, ctx.root, ast.unparse(node)[:80]))
    return out


def _restores_namespace(node: ast.AST) -> bool:
    """The snapshot / restore idiom: ``m.__dict__.clear()`` then ``m.__dict__.update(saved)``."""
    seen = set()
    for sub in ast.walk(node):
        if (
            isinstance(sub, ast.Call)
            and isinstance(sub.func, ast.Attribute)
            and sub.func.attr in {"clear", "update"}
            and _dotted(sub.func.value).endswith("__dict__")
        ):
            seen.add(sub.func.attr)
    return seen == {"clear", "update"}


def _callee(func: ast.AST) -> str | None:
    """The same-file function a call names: ``helper()`` or ``self.helper()``."""
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
        return f"{func.value.id}.{func.attr}"
    return None


def _reload_findings(nodes: Nodes) -> list[Finding]:
    """Tests that reach an ``importlib.reload`` nothing undoes.

    A reload is undone in place when the reload sits in a ``try/finally`` of its
    own function, or its function is a fixture that restores the namespace on
    teardown. Any other reload leaks the reloaded module's new objects unless the
    test is covered by a restoring fixture: an autouse one at module or class
    level, or one the test asks for.
    """
    functions: dict[str, ast.AST] = {}
    classes: set[str] = set()
    owner: dict[str, str] = {}
    restores: set[str] = set()  # functions holding the snapshot / restore idiom
    fixtures: set[str] = set()  # every fixture, restoring or not
    autouse_by_scope: dict[str, set[str]] = {}  # "" (module) or a class -> restoring autouse fixtures
    calls: dict[str, set[str]] = {}
    reloads: dict[str, list[tuple[bool, int]]] = {}
    out: list[Finding] = []
    pairs = list(nodes)

    # Pass 1 -- register every class and function before resolving any call:
    # the walk visits siblings in reverse source order.
    for node, ctx in pairs:
        if isinstance(node, ast.ClassDef):
            classes.add(_qual(ctx.func, node.name))
        elif isinstance(node, _FUNC_TYPES):
            qual = _qual(ctx.func, node.name)
            functions[qual] = node
            owner[qual] = ctx.func if ctx.func in classes else ""
            if _restores_namespace(node):
                restores.add(qual)
            autouse = _fixture_autouse(node)
            if autouse is not None:
                fixtures.add(qual)
                if autouse and qual in restores:
                    autouse_by_scope.setdefault(owner[qual], set()).add(qual)

    # Pass 2 -- the reloads and the same-file call graph.
    for node, ctx in pairs:
        if not isinstance(node, ast.Call):
            continue
        if _last(node.func) == "reload":
            if ctx.func == "<module>":
                out.append((node.lineno, "<module>", "reload at import"))
            else:
                reloads.setdefault(ctx.func, []).append((ctx.guarded, node.lineno))
        callee = _callee(node.func)
        if callee in functions and ctx.func != "<module>":
            calls.setdefault(ctx.func, set()).add(callee)

    undone = {func for func, sites in reloads.items() if all(g for g, _ in sites)}
    undone |= fixtures & restores
    # A helper's leaked reload is the caller's leak too -- unless the caller restores.
    leaky = set(reloads) - undone
    changed = True
    while changed:
        changed = False
        for func, callees in calls.items():
            if func not in leaky and func not in restores and any(c in leaky for c in callees):
                leaky.add(func)
                changed = True

    for qual, node in functions.items():
        if qual not in leaky:
            continue
        if qual in fixtures:
            out.append((node.lineno, qual, "fixture reloads and restores nothing"))
            continue
        if not node.name.startswith("test_"):
            continue
        params = {arg.arg for arg in node.args.args + node.args.kwonlyargs}
        covered = (
            bool(autouse_by_scope.get("", set()))
            or bool(autouse_by_scope.get(owner[qual], set()))
            or any(name in fixtures and name in restores for name in params)
        )
        if not covered:
            out.append((node.lineno, qual, "reaches a reload nothing undoes"))
    return out


# --- the tree ------------------------------------------------------------------
@functools.cache
def _parsed() -> tuple[tuple[str, ast.Module | SyntaxError], ...]:
    """Every ``*.py`` under ``tests/``: its path relative to ``tests/`` and either
    the module or the ``SyntaxError`` that keeps pytest from collecting it."""
    out: list[tuple[str, ast.Module | SyntaxError]] = []
    for path in sorted(TESTS_DIR.rglob("*.py")):
        rel = path.relative_to(TESTS_DIR).as_posix()
        try:
            out.append((rel, ast.parse(path.read_bytes(), filename=str(path))))
        except SyntaxError as exc:
            out.append((rel, exc))
    return tuple(out)


@functools.cache
def _trees() -> tuple[tuple[str, ast.Module], ...]:
    return tuple((rel, tree) for rel, tree in _parsed() if isinstance(tree, ast.Module))


@functools.cache
def _nodes_of(tree: ast.AST) -> list[tuple[ast.AST, _Ctx]]:
    return list(_walk(tree))


def _scan(detector) -> dict[Site, list[Finding]]:
    """Every finding of ``detector`` in ``tests/``, keyed by (file, function)."""
    hits: dict[Site, list[Finding]] = {}
    for rel, tree in _trees():
        for finding in detector(_nodes_of(tree)):
            hits.setdefault((rel, finding[1]), []).append(finding)
    return hits


def _vs_allowlist(hits: dict[Site, list[Finding]], allowlist: dict[Site, str]) -> str:
    """'' when the sites are exactly the allow-listed ones, else what differs."""
    sites = set(hits)
    lines = []
    for site in sorted(sites - set(allowlist)):
        lines.append(f"unlisted: {site[0]}::{site[1]}  ({hits[site][0][2]})")
    for site in sorted(set(allowlist) - sites):
        lines.append(f"stale allow-list entry (no such site any more): {site[0]}::{site[1]}")
    for site, reason in allowlist.items():
        if not isinstance(reason, str) or not reason.strip():
            lines.append(f"allow-list entry without a reason: {site[0]}::{site[1]}")
    return "\n".join(lines)


# --- pins ----------------------------------------------------------------------
@pytest.mark.unit
def test_r17_hygiene_every_python_file_under_tests_parses():
    broken = [
        f"{rel}:{exc.lineno}: {exc.msg}" for rel, exc in _parsed() if isinstance(exc, SyntaxError)
    ]
    assert broken == [], "a test file that does not parse is silently not collected"


@pytest.mark.unit
def test_r17_hygiene_the_pins_see_the_whole_tree():
    test_functions = sum(
        1
        for _, tree in _trees()
        for node, _ in _nodes_of(tree)
        if isinstance(node, _FUNC_TYPES) and node.name.startswith("test_")
    )
    assert len(_trees()) >= 250, "the scan is looking at a directory that is not tests/"
    assert test_functions >= 4000


@pytest.mark.unit
def test_r17_hygiene_no_test_body_is_empty_or_constant_true():
    hits = _scan(_body_findings)
    assert hits == {}, f"tests that cannot fail: {sorted(hits)}"


@pytest.mark.unit
def test_r17_hygiene_skip_and_xfail_always_say_why():
    hits = _scan(_skip_findings)
    assert hits == {}, f"skip / xfail without a reason: {sorted(hits)}"


@pytest.mark.unit
def test_r17_hygiene_real_sleep_needs_a_listed_reason():
    report = _vs_allowlist(_scan(_sleep_findings), _SLEEP_ALLOWLIST)
    assert report == "", report


@pytest.mark.unit
def test_r17_hygiene_pytest_style_tests_never_write_os_environ_directly():
    hits = _scan(_env_findings)
    assert hits == {}, f"bare os.environ writes (use monkeypatch): {sorted(hits)}"


@pytest.mark.unit
def test_r17_hygiene_drive_letter_literal_needs_a_listed_reason():
    report = _vs_allowlist(_scan(_drive_findings), _DRIVE_LETTER_ALLOWLIST)
    assert report == "", report


@pytest.mark.unit
def test_r17_hygiene_sys_path_edit_needs_a_listed_reason():
    report = _vs_allowlist(_scan(_syspath_findings), _SYSPATH_ALLOWLIST)
    assert report == "", report


@pytest.mark.unit
def test_r17_hygiene_warnings_are_never_silenced_unscoped():
    hits = _scan(_warn_findings)
    assert hits == {}, f"warning filters that outlive their block: {sorted(hits)}"


@pytest.mark.unit
def test_r17_hygiene_pyproject_silences_only_deprecation_warnings():
    config = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    filters = config["tool"]["pytest"]["ini_options"].get("filterwarnings", [])
    ignored = [f for f in filters if f.split("::", 1)[0].strip().lower().startswith("ignore")]
    assert ignored == ["ignore::DeprecationWarning"]


@pytest.mark.unit
def test_r17_hygiene_every_importlib_reload_in_a_test_is_undone():
    hits = _scan(_reload_findings)
    assert hits == {}, (
        f"tests that reach an importlib.reload nothing undoes: {sorted(hits)}"
    )


# Objects that earlier test files used to replace for the rest of the session.
_PINNED = (
    (_default_config, "DEFAULT_CONFIG"),
    (_openai_client, "OpenAIClient"),
    (_openai_client, "NormalizedChatOpenAI"),
    (_cli_utils, "ensure_api_key"),
    (_cli_utils, "confirm_ollama_endpoint"),
    (_stockstats, "load_ohlcv"),
    (_stockstats, "YF_HTTP_TIMEOUT"),
    (_perp_bundle, "fetch_perp_market_bundle"),
    (_perp_bundle, "LEG_TIMEOUT_S"),
    (_sentiment, "_fetch_sentiment_sources"),
    (_sentiment, "_SENTIMENT_PARALLEL_FETCH"),
    (_web_store, "load_run"),
    (_web_store, "LOGS_ROOT"),
)
_BY_IDENTITY = (type, types.FunctionType, dict)
# Taken while this module is imported, i.e. during collection, before any test runs.
_AT_COLLECTION = [(module, attr, getattr(module, attr)) for module, attr in _PINNED]
_CLI_MAIN = sys.modules.get("yialpha.cli.main")
# The names cli.main binds at import time: a reload of cli.utils used to leave them stale.
_CLI_MAIN_FROM_UTILS = (
    "ask_anthropic_effort", "ask_gemini_thinking_config", "ask_glm_region",
    "ask_minimax_region", "ask_openai_reasoning_effort", "ask_output_language",
    "ask_qwen_region", "confirm_ollama_endpoint", "detect_asset_type", "ensure_api_key",
    "get_ticker", "is_canonical_ymd", "prompt_openai_compatible_url", "resolve_backend_url",
    "resolve_run_analysts", "select_analysts", "select_deep_thinking_agent",
    "select_llm_provider", "select_research_depth", "select_shallow_thinking_agent",
)
_AT_COLLECTION_CLI_MAIN = {
    name: getattr(_CLI_MAIN, name)
    for name in _CLI_MAIN_FROM_UTILS
    if _CLI_MAIN is not None and hasattr(_CLI_MAIN, name)
}


@pytest.mark.unit
def test_r17_hygiene_earlier_files_leave_session_objects_as_they_found_them():
    replaced = []
    for module, attr, was in _AT_COLLECTION:
        now = getattr(module, attr)
        same = now is was if isinstance(was, _BY_IDENTITY) else type(now) is type(was) and now == was
        if not same:
            replaced.append(f"{module.__name__}.{attr}")
    for name, was in _AT_COLLECTION_CLI_MAIN.items():
        if not (getattr(_CLI_MAIN, name) is was and getattr(_cli_utils, name, None) is was):
            replaced.append(f"yialpha.cli.main.{name}")
    assert replaced == [], f"replaced by an earlier test file and not restored: {replaced}"


# --- self-tests: each detector on a known offender ------------------------------
def _details(detector, source: str) -> list[str]:
    tree = ast.parse(textwrap.dedent(source).lstrip("\n"))
    return [detail for _, _, detail in detector(_walk(tree))]


@pytest.mark.unit
@pytest.mark.parametrize(
    "source, flagged",
    [
        ("def test_a():\n    time.sleep(0.2)\n", True),
        ("def test_a():\n    time_mod.sleep(1)\n", True),
        ("def test_a():\n    time.sleep(0.05)\n    time.sleep(0.01)\n", False),
        ("def test_a(monkeypatch):\n    monkeypatch.setattr(mod.time, 'sleep', lambda s: None)\n", False),
        ("def slow():\n    time.sleep(0.2)\n\ndef test_a():\n    slow()\n", True),
    ],
)
def test_r17_hygiene_sleep_detector_sees_a_real_sleep(source, flagged):
    assert bool(_details(_sleep_findings, source)) is flagged


@pytest.mark.unit
@pytest.mark.parametrize(
    "source, flagged",
    [
        ("def test_a():\n    pass\n", True),
        ('def test_a():\n    """What it checks."""\n    ...\n', True),
        ("def test_a():\n    assert True\n", True),
        ("def test_a():\n    assert f() == 1\n", False),
        ("def helper():\n    pass\n\ndef test_a():\n    assert helper() is None\n", False),
    ],
)
def test_r17_hygiene_body_detector_sees_a_test_that_cannot_fail(source, flagged):
    assert bool(_details(_body_findings, source)) is flagged


@pytest.mark.unit
@pytest.mark.parametrize(
    "source, flagged",
    [
        ("@pytest.mark.skipif(COND)\ndef test_a():\n    pass\n", True),
        ('@pytest.mark.skipif(COND, reason="needs a tty")\ndef test_a():\n    pass\n', False),
        ("@pytest.mark.xfail\ndef test_a():\n    pass\n", True),
        ('@pytest.mark.skip(reason="no fixture data yet")\ndef test_a():\n    pass\n', False),
        ("@unittest.expectedFailure\ndef test_a(self):\n    pass\n", True),
        (
            '@unittest.skipUnless(sys.platform == "win32", "needs symlinks")\n'
            "def test_a(self):\n    pass\n",
            False,
        ),
        ('def test_a():\n    pytest.skip("symlinks not permitted on this host")\n', False),
        ("def test_a():\n    pytest.skip()\n", True),
        ('def test_a():\n    pytest.importorskip("scipy.stats")\n', False),
        ("import pytest as pt\n\n\n@pt.mark.skipif(True)\ndef test_a():\n    pass\n", True),
        (
            "import pytest as pt\n\n\n@pt.mark.skipif(True, reason='why')\ndef test_a():\n    pass\n",
            False,
        ),
    ],
)
def test_r17_hygiene_skip_detector_sees_a_skip_without_a_reason(source, flagged):
    assert bool(_details(_skip_findings, source)) is flagged


@pytest.mark.unit
@pytest.mark.parametrize(
    "source, flagged",
    [
        ("import os\n\ndef test_a():\n    os.environ['X'] = '1'\n", True),
        ("import os\n\ndef test_a():\n    os.environ.setdefault('X', '1')\n", True),
        (
            "import os\n\ndef test_a():\n    os.environ['X'] = '1'\n"
            "    try:\n        f()\n    finally:\n        os.environ.pop('X', None)\n",
            True,
        ),
        (
            "import os\n\ndef test_a():\n    try:\n        os.environ['X'] = '1'\n        f()\n"
            "    finally:\n        os.environ.pop('X', None)\n",
            False,
        ),
        (
            "import os, unittest\n\nclass T(unittest.TestCase):\n"
            "    def test_a(self):\n        os.environ['X'] = '1'\n",
            False,
        ),
        ("def test_a(monkeypatch):\n    monkeypatch.setenv('X', '1')\n", False),
        ("import os as _os\n\ndef test_a():\n    _os.environ['X'] = '1'\n", True),
        ("def test_a():\n    env = {}\n    env['X'] = '1'\n    assert env == {'X': '1'}\n", False),
    ],
)
def test_r17_hygiene_env_detector_sees_a_bare_environment_write(source, flagged):
    assert bool(_details(_env_findings, source)) is flagged


@pytest.mark.unit
@pytest.mark.parametrize(
    "source, flagged",
    [
        ('import os\n\ndef test_a():\n    assert not os.path.exists("C:/x/y.json")\n', True),
        ('def test_a():\n    p = Path("/tmp/x")\n    q = tmp_path / "C.txt"\n', False),
        ('def test_a(name):\n    p = f"C:/cache/{name}.json"\n', True),
        ('def test_a():\n    p = "C:\\\\x"\n', True),
        ('def test_a():\n    p = "C:/x"\n    assert p == "C:/x"\n', True),
    ],
)
def test_r17_hygiene_drive_detector_sees_a_drive_letter_literal(source, flagged):
    assert bool(_details(_drive_findings, source)) is flagged


@pytest.mark.unit
@pytest.mark.parametrize(
    "source, flagged",
    [
        ('def test_a():\n    warnings.simplefilter("ignore")\n', True),
        (
            'def test_a():\n    with warnings.catch_warnings():\n'
            '        warnings.simplefilter("ignore", RuntimeWarning)\n        f()\n',
            False,
        ),
        ('warnings.simplefilter("error")\n', True),
        ('@pytest.mark.filterwarnings("ignore::UserWarning")\ndef test_a():\n    pass\n', True),
        ("import warnings as w\n\nw.simplefilter('ignore')\n", True),
        (
            'def test_a():\n    with warnings.catch_warnings(record=True) as caught:\n'
            '        warnings.simplefilter("always")\n    assert caught\n',
            False,
        ),
    ],
)
def test_r17_hygiene_warning_detector_sees_an_unscoped_filter(source, flagged):
    assert bool(_details(_warn_findings, source)) is flagged


@pytest.mark.unit
@pytest.mark.parametrize(
    "source, flagged",
    [
        ("import sys\nsys.path.insert(0, 'scripts')\n", True),
        (
            "def test_a():\n    sys.path.insert(0, 'scripts')\n    f()\n",
            True,
        ),
        (
            "import sys\n\ndef test_a():\n    saved = list(sys.path)\n"
            "    try:\n        sys.path.insert(0, 'x')\n        f()\n"
            "    finally:\n        sys.path[:] = saved\n",
            False,
        ),
        ("def test_a(monkeypatch):\n    monkeypatch.setattr(sys, 'path', list(sys.path))\n", False),
        ("import sys as _sys\n\n_sys.path.insert(0, 'nowhere')\n", True),
        ("class T:\n    def test_a(self):\n        self.path.append('x')\n", False),
    ],
)
def test_r17_hygiene_sys_path_detector_sees_a_sys_path_edit(source, flagged):
    assert bool(_details(_syspath_findings, source)) is flagged


_RESTORE_FIXTURE = (
    "import importlib\n"
    "import pytest\n"
    "import mod\n"
    "\n"
    "\n"
    "@pytest.fixture(autouse=True)\n"
    "def _undo():\n"
    "    saved = dict(mod.__dict__)\n"
    "    yield\n"
    "    mod.__dict__.clear()\n"
    "    mod.__dict__.update(saved)\n"
    "\n"
    "\n"
)
_RESTORE_FIXTURE_ON_REQUEST = _RESTORE_FIXTURE.replace("@pytest.fixture(autouse=True)", "@pytest.fixture")


@pytest.mark.unit
@pytest.mark.parametrize(
    "source, flagged",
    [
        ("import importlib\nimport mod\n\nimportlib.reload(mod)\n", True),
        ("import importlib\nimport mod\n\n\ndef test_a():\n    importlib.reload(mod)\n", True),
        (_RESTORE_FIXTURE + "def test_a():\n    importlib.reload(mod)\n", False),
        (
            "import importlib\nimport mod\n\n\ndef test_a():\n"
            "    saved = dict(mod.__dict__)\n"
            "    try:\n"
            "        importlib.reload(mod)\n"
            "        assert mod.X == 1\n"
            "    finally:\n"
            "        mod.__dict__.clear()\n"
            "        mod.__dict__.update(saved)\n",
            False,
        ),
        (
            "import importlib\nimport mod\n\n\ndef _reload():\n"
            "    return importlib.reload(mod)\n\n\ndef test_a():\n    _reload()\n",
            True,
        ),
        (
            _RESTORE_FIXTURE
            + "def _reload():\n    return importlib.reload(mod)\n\n\ndef test_a():\n    _reload()\n",
            False,
        ),
        (
            _RESTORE_FIXTURE_ON_REQUEST
            + "def _reload():\n    return importlib.reload(mod)\n\n\ndef test_a():\n    _reload()\n",
            True,
        ),
        (
            "import importlib\nimport pytest\nimport mod\n\n\n@pytest.fixture\n"
            "def undone():\n    yield importlib.reload(mod)\n\n\n"
            "def test_a(undone):\n    assert undone.X\n",
            True,
        ),
        (
            "import importlib\nimport pytest\nimport mod\n\n\n@pytest.fixture\n"
            "def reloaded():\n    saved = dict(mod.__dict__)\n"
            "    yield importlib.reload(mod)\n"
            "    mod.__dict__.clear()\n    mod.__dict__.update(saved)\n\n\n"
            "def test_a(reloaded):\n    assert reloaded.X\n",
            False,
        ),
    ],
)
def test_r17_hygiene_reload_detector_sees_a_reload_nothing_undoes(source, flagged):
    assert bool(_details(_reload_findings, source)) is flagged
