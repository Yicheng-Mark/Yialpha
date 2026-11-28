"""Round 17 (pot_tools shard): PoT executor gate, timeout runners, prediction tools.

PoT executor (flag parsing, gate matrix, guarded import, timeout plumbing),
prediction tools (per-entry validation, flush / settle / dispatch) and the
crypto-native / global-news ``@tool`` wrappers.

Every test pins an exact value, exception or message. The PoT gate tests drive
``PoTEnableSwitch`` with the env x runtime-config matrix; the config store is
reset per test by the autouse ``_isolate_config`` fixture, env vars by
``monkeypatch``.
"""

from __future__ import annotations

import importlib
import importlib.machinery
import logging
import sys
import threading
import time
import types
from pathlib import Path
from typing import Any

import pytest
from langchain_core.messages import AIMessage
from pydantic import ValidationError

import yialpha.agents.utils.crypto_native_tools as crypto_native_tools
import yialpha.agents.utils.news_data_tools as news_data_tools
import yialpha.agents.utils.pot_executor as pot_executor_module
import yialpha.agents.utils.prediction_tools as prediction_tools
import yialpha.dataflows.config as config_module
import yialpha.dataflows.interface as vendor_interface
from yialpha.agents.utils.pot_executor import PoTEnableSwitch, PoTExecutor
from yialpha.dataflows import quality
from yialpha.dataflows.config import set_config
from yialpha.dataflows.utils import pinned_analysis_date
from yialpha.ledger.run_context import reset_ledger_run_context, set_ledger_run_context

_POT_ENV = "YIALPHA_POT_ENABLED"
_ANALYSIS_ONLY_ENV = "YIALPHA_ANALYSIS_ONLY"

_TRUTHY = ["true", "1", "yes", "on", "TRUE", "True", " Yes ", "\tON\n"]
_FALSY = ["false", "0", "no", "off", "FALSE", "False", " Off ", "\tNO\n"]
_GARBAGE = ["tru", "y", "t", "f", "n", "2", "-1", "truee", "falsee", "nope", "enabled"]


@pytest.fixture
def posix_gate(monkeypatch):
    """PoT gate on a POSIX host with both env flags unset: the runtime config is the only input."""
    for name in (_POT_ENV, _ANALYSIS_ONLY_ENV):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(pot_executor_module.platform, "system", lambda: "Linux")


@pytest.mark.unit
class TestR17PotFlagParsing:
    @pytest.mark.parametrize("raw", _TRUTHY)
    def test_r17_pot_env_truthy_overrides_config_false(self, posix_gate, monkeypatch, raw):
        # A truthy env value wins over the config on both flags the parser serves.
        set_config({"pot_enabled": False, "analysis_only": True})
        monkeypatch.setenv(_POT_ENV, raw)
        assert (
            PoTEnableSwitch._env_or_config_flag(_POT_ENV, "pot_enabled", default=False) is True
        )
        monkeypatch.setenv(_ANALYSIS_ONLY_ENV, raw)
        assert (
            PoTEnableSwitch._env_or_config_flag(
                _ANALYSIS_ONLY_ENV, "analysis_only", default=False
            )
            is True
        )

    @pytest.mark.parametrize("raw", _FALSY)
    def test_r17_pot_env_falsy_overrides_config_true(self, posix_gate, monkeypatch, raw):
        # "false" / "0" / "no" / "off" are parsed False (never truthy via bool("false")).
        set_config({"pot_enabled": True, "analysis_only": True})
        monkeypatch.setenv(_POT_ENV, raw)
        assert (
            PoTEnableSwitch._env_or_config_flag(_POT_ENV, "pot_enabled", default=True) is False
        )
        monkeypatch.setenv(_ANALYSIS_ONLY_ENV, raw)
        assert (
            PoTEnableSwitch._env_or_config_flag(
                _ANALYSIS_ONLY_ENV, "analysis_only", default=True
            )
            is False
        )

    @pytest.mark.parametrize("raw", _GARBAGE)
    def test_r17_pot_env_malformed_fails_closed_and_ignores_config(
        self, posix_gate, monkeypatch, raw
    ):
        # A malformed value resolves to the caller's default and does NOT consult the
        # config: pot_enabled=True / analysis_only=False in the config cannot rescue it.
        set_config({"pot_enabled": True, "analysis_only": False})
        monkeypatch.setenv(_POT_ENV, raw)
        monkeypatch.setenv(_ANALYSIS_ONLY_ENV, raw)
        assert (
            PoTEnableSwitch._env_or_config_flag(_POT_ENV, "pot_enabled", default=False) is False
        )
        assert (
            PoTEnableSwitch._env_or_config_flag(
                _ANALYSIS_ONLY_ENV, "analysis_only", default=True
            )
            is True
        )

    @pytest.mark.parametrize("raw", [None, "", "   ", "\t\n"])
    def test_r17_pot_env_unset_or_blank_defers_to_config(self, posix_gate, monkeypatch, raw):
        if raw is not None:
            monkeypatch.setenv(_POT_ENV, raw)
        set_config({"pot_enabled": True})
        assert (
            PoTEnableSwitch._env_or_config_flag(_POT_ENV, "pot_enabled", default=False) is True
        )
        set_config({"pot_enabled": False})
        assert (
            PoTEnableSwitch._env_or_config_flag(_POT_ENV, "pot_enabled", default=True) is False
        )

    @pytest.mark.parametrize(
        "value", ["true", "false", "yes", 1, 0, None, [True], {"v": True}, 1.0]
    )
    def test_r17_pot_config_non_bool_falls_back_to_default(self, posix_gate, value):
        # Only real booleans are honoured; strings such as "true" fail closed.
        set_config({"pot_enabled": value})
        assert PoTEnableSwitch._config_flag("pot_enabled", False) is False
        assert PoTEnableSwitch._config_flag("pot_enabled", True) is True

    def test_r17_pot_config_bool_and_absent_key(self, posix_gate):
        set_config({"pot_enabled": True, "analysis_only": False})
        assert PoTEnableSwitch._config_flag("pot_enabled", False) is True
        assert PoTEnableSwitch._config_flag("analysis_only", True) is False
        # An absent key resolves to the default, in both directions.
        assert "r17_key_that_does_not_exist" not in config_module.get_config()
        assert PoTEnableSwitch._config_flag("r17_key_that_does_not_exist", False) is False
        assert PoTEnableSwitch._config_flag("r17_key_that_does_not_exist", True) is True

    @pytest.mark.parametrize("default", [False, True])
    def test_r17_pot_config_read_error_fails_closed_to_default(
        self, posix_gate, monkeypatch, default
    ):
        # get_config is imported lazily inside _config_flag -> patch the source module.
        def _boom():
            raise RuntimeError("config store unavailable")

        monkeypatch.setattr(config_module, "get_config", _boom)
        assert PoTEnableSwitch._config_flag("pot_enabled", default) is default


@pytest.mark.unit
class TestR17PotGateMatrix:
    @pytest.mark.parametrize(
        ("analysis_only_raw", "pot_raw", "config", "expected"),
        [
            (None, None, {}, False),  # shipped default: analysis-only, PoT off
            ("false", None, {}, False),  # analysis_only lifted, PoT never opted in
            ("false", "true", {}, True),
            ("false", "1", {}, True),
            ("0", "yes", {}, True),
            ("off", "on", {}, True),
            ("FALSE", " True ", {}, True),
            # env wins over the config in both directions
            ("false", "false", {"pot_enabled": True}, False),
            ("true", "true", {"analysis_only": False}, False),
            ("false", "true", {"pot_enabled": False}, True),
            # pure config opt-in (both env flags unset / blank)
            (None, None, {"analysis_only": False, "pot_enabled": True}, True),
            ("   ", None, {"analysis_only": False, "pot_enabled": True}, True),
            (None, None, {"analysis_only": False}, False),
            (None, "true", {"analysis_only": False}, True),
            (None, "true", {}, False),  # analysis_only still active
            # malformed values fail closed even against an explicit config opt-in/out
            ("garbage", "true", {"analysis_only": False}, False),
            ("false", "garbage", {"pot_enabled": True}, False),
            ("", "tru", {}, False),
            # non-bool config values fail closed
            (None, "true", {"analysis_only": "false", "pot_enabled": True}, False),
            ("false", None, {"pot_enabled": "true"}, False),
            ("false", None, {"pot_enabled": 1}, False),
        ],
    )
    def test_r17_pot_is_enabled_matrix(
        self, posix_gate, monkeypatch, analysis_only_raw, pot_raw, config, expected
    ):
        set_config(config)
        for name, raw in ((_ANALYSIS_ONLY_ENV, analysis_only_raw), (_POT_ENV, pot_raw)):
            if raw is None:
                monkeypatch.delenv(name, raising=False)
            else:
                monkeypatch.setenv(name, raw)
        assert PoTEnableSwitch.is_enabled() is expected

    def test_r17_pot_windows_refuses_before_reading_pot_flag(self, monkeypatch):
        # On Windows the PoT flag is never consulted; analysis_only is checked first.
        monkeypatch.delenv(_ANALYSIS_ONLY_ENV, raising=False)
        monkeypatch.setenv(_POT_ENV, "true")
        set_config({"analysis_only": False, "pot_enabled": True})
        monkeypatch.setattr(pot_executor_module.platform, "system", lambda: "Windows")
        assert PoTEnableSwitch.is_enabled() is False
        # ...and the analysis-only gate comes before the platform gate.
        monkeypatch.setenv(_ANALYSIS_ONLY_ENV, "true")
        assert PoTEnableSwitch.reason() == (
            "analysis_only is active; host-side LLM code execution is blocked."
        )


@pytest.mark.unit
class TestR17PotReasonText:
    """``reason()`` must name the cause that actually keeps the gate shut."""

    _ANALYSIS_ONLY = "analysis_only is active; host-side LLM code execution is blocked."
    _WINDOWS = (
        "PoT host execution is blocked on Windows until it uses a "
        "terminable, process-isolated sandbox."
    )
    _FROM_CONFIG = "pot_enabled=True in runtime config."
    _NOT_OPTED_IN = (
        "YIALPHA_POT_ENABLED is unset and pot_enabled is not explicitly true; "
        "LLM-generated code execution is disabled."
    )
    _TRUTHY_SET = {"true", "1", "yes", "on"}

    @pytest.mark.parametrize(
        ("analysis_only_raw", "config"),
        [
            (None, {}),
            (None, {"analysis_only": True}),
            ("true", {}),
            ("garbage", {"analysis_only": False}),  # malformed fails closed
            ("", {"analysis_only": True}),
        ],
    )
    def test_r17_pot_reason_analysis_only_wins_over_every_other_flag(
        self, posix_gate, monkeypatch, analysis_only_raw, config
    ):
        # Checked first: an explicit PoT opt-in via env AND config is never named.
        monkeypatch.setenv(_POT_ENV, "true")
        set_config({**config, "pot_enabled": True})
        if analysis_only_raw is None:
            monkeypatch.delenv(_ANALYSIS_ONLY_ENV, raising=False)
        else:
            monkeypatch.setenv(_ANALYSIS_ONLY_ENV, analysis_only_raw)
        assert PoTEnableSwitch.reason() == self._ANALYSIS_ONLY
        assert PoTEnableSwitch.is_enabled() is False

    def test_r17_pot_reason_names_windows_before_the_pot_flag(self, monkeypatch):
        monkeypatch.setenv(_ANALYSIS_ONLY_ENV, "false")
        monkeypatch.setenv(_POT_ENV, "true")
        set_config({"analysis_only": False, "pot_enabled": True})
        monkeypatch.setattr(pot_executor_module.platform, "system", lambda: "Windows")
        assert PoTEnableSwitch.reason() == self._WINDOWS
        assert PoTEnableSwitch.is_enabled() is False

    @pytest.mark.parametrize("pot_raw", [None, "", "   "])
    def test_r17_pot_reason_env_unset_config_opt_in(self, posix_gate, monkeypatch, pot_raw):
        monkeypatch.setenv(_ANALYSIS_ONLY_ENV, "false")
        set_config({"pot_enabled": True})
        if pot_raw is None:
            monkeypatch.delenv(_POT_ENV, raising=False)
        else:
            monkeypatch.setenv(_POT_ENV, pot_raw)
        assert PoTEnableSwitch.reason() == self._FROM_CONFIG
        assert PoTEnableSwitch.is_enabled() is True

    @pytest.mark.parametrize(
        "config", [{}, {"pot_enabled": False}, {"pot_enabled": "true"}, {"pot_enabled": 1}]
    )
    def test_r17_pot_reason_env_unset_not_opted_in(
        self, posix_gate, monkeypatch, config
    ):
        # A non-bool config value is "not explicitly true" -> the same honest reason.
        monkeypatch.setenv(_ANALYSIS_ONLY_ENV, "false")
        set_config(config)
        assert PoTEnableSwitch.reason() == self._NOT_OPTED_IN
        assert PoTEnableSwitch.is_enabled() is False

    @pytest.mark.parametrize(
        ("pot_raw", "expected"),
        [
            ("tru", "YIALPHA_POT_ENABLED='tru' is invalid; execution is disabled."),
            ("2", "YIALPHA_POT_ENABLED='2' is invalid; execution is disabled."),
            (" true ", "YIALPHA_POT_ENABLED=' true '."),
            ("false", "YIALPHA_POT_ENABLED='false'."),
            ("0", "YIALPHA_POT_ENABLED='0'."),
        ],
    )
    def test_r17_pot_reason_env_set_matches_verdict(
        self, posix_gate, monkeypatch, pot_raw, expected
    ):
        monkeypatch.setenv(_ANALYSIS_ONLY_ENV, "false")
        set_config({"pot_enabled": False})
        monkeypatch.setenv(_POT_ENV, pot_raw)
        assert PoTEnableSwitch.reason() == expected
        # The verdict matches the reason: only a truthy value opens the gate.
        assert PoTEnableSwitch.is_enabled() is (pot_raw.strip().lower() in self._TRUTHY_SET)


@pytest.mark.unit
class TestR17PotGuardedImport:
    """``_make_guarded_import()``: only already-loaded numpy/pandas internals resolve."""

    @pytest.fixture
    def guard(self):
        return pot_executor_module._make_guarded_import()

    @pytest.mark.parametrize("level", [1, 2, -1])
    def test_r17_pot_guard_relative_import_blocked_before_the_allowlist(self, guard, level):
        # The level check comes first: even an allowed (or disallowed) name is refused
        # as a relative import.
        for name in ("linalg", "numpy.linalg", "os"):
            with pytest.raises(ImportError) as excinfo:
                guard(name, None, None, (), level)
            assert str(excinfo.value) == (
                f"PoTExecutor: relative imports are blocked (attempted {name!r})."
            )

    @pytest.mark.parametrize("name", ["os", "sys", "subprocess", "math", "builtins", "", "Numpy"])
    def test_r17_pot_guard_disallowed_top_level_blocked_even_when_loaded(self, guard, name):
        # The allowlist precedes the sys.modules lookup: "os" is loaded, still refused.
        if name in {"os", "sys", "builtins"}:
            assert name in sys.modules
        with pytest.raises(ImportError) as excinfo:
            guard(name)
        assert str(excinfo.value) == (
            f"PoTExecutor: import of {name!r} is not permitted in the sandbox "
            f"(only numpy/pandas internals resolve, and only if already loaded)."
        )

    def test_r17_pot_guard_allowed_top_level_but_not_loaded_is_unavailable(self, guard):
        missing = "numpy.r17_no_such_submodule"
        assert missing not in sys.modules
        with pytest.raises(ImportError) as excinfo:
            guard(missing)
        assert str(excinfo.value) == (
            f"PoTExecutor: module {missing!r} is not available in the sandbox."
        )
        # Never touched the import machinery: the module is still not loaded.
        assert missing not in sys.modules

    def test_r17_pot_guard_fromlist_returns_the_package_module(self, guard):
        import numpy

        assert "numpy.linalg" in sys.modules
        # ``from numpy import linalg`` -> the package, Python pulls the attribute.
        assert guard("numpy", fromlist=("linalg",)) is numpy
        # ``from numpy.linalg import norm`` -> the submodule itself.
        assert guard("numpy.linalg", fromlist=("norm",)) is sys.modules["numpy.linalg"]

    def test_r17_pot_guard_plain_import_returns_the_top_level_package(self, guard):
        import numpy
        import pandas

        assert guard("numpy.linalg") is numpy
        assert guard("pandas.core.frame") is pandas
        assert guard("numpy") is numpy
        assert guard("pandas", None, None, None, 0) is pandas

    @pytest.fixture
    def pot_on(self, posix_gate, monkeypatch):
        monkeypatch.setenv(_ANALYSIS_ONLY_ENV, "false")
        monkeypatch.setenv(_POT_ENV, "true")

    def _exec(self, code, **kwargs):
        return PoTExecutor().run_sandboxed(code=code, **kwargs)

    def test_r17_pot_guard_end_to_end_lazy_numpy_import_runs(self, pot_on):
        res = self._exec("import numpy.linalg\nresult = float(np.linalg.norm([3.0, 4.0]))")
        assert res.ok is True
        assert res.error is None
        assert res.result == 5.0

    def test_r17_pot_guard_end_to_end_from_pandas_import_runs(self, pot_on):
        res = self._exec("from pandas import Series\nresult = int(Series([1, 2, 3]).sum())")
        assert res.ok is True
        assert res.result == 6

    # -- ``from X import Y``: who decides what ``Y`` is -------------------------
    #
    # The guard is an ``__import__`` replacement, so it only ever sees the import
    # statement. For a from-import CPython calls it once, with the package name
    # and the ``fromlist``; it returns the package and the interpreter resolves
    # each name by attribute access. A name that is not an attribute is refused
    # by the interpreter itself (``cannot import name``): it does not call
    # ``__import__`` again and starts no import, so the guard's
    # "already-loaded only" rule holds end to end. Verified on CPython 3.11 /
    # 3.12 / 3.13 on Windows and Linux (evidence 09, 11, 12 under
    # scratch_pot_tools/evidence).

    def test_r17_pot_guard_fromlist_names_are_left_to_the_interpreter(self, guard):
        import numpy
        import pandas

        # The guard decides about modules only. Real attributes, ``*``, a name
        # that does not exist, both mixed: the package comes back regardless.
        assert guard("numpy", None, None, ("ndarray", "r17_no_such_name")) is numpy
        assert guard("pandas", None, None, ("r17_no_such_name",)) is pandas
        assert guard("numpy", None, None, ("*",)) is numpy

    @pytest.mark.parametrize(
        ("name", "fromlist", "level", "refusal"),
        [
            # relative import: refused first, fromlist or not
            ("numpy", ("ndarray",), 1, "relative"),
            ("numpy.linalg", ("norm",), -1, "relative"),
            # disallowed top level: the allowlist wins over any fromlist
            ("os", ("path",), 0, "not permitted"),
            ("math", ("pi",), 0, "not permitted"),
            # allowed top level but not loaded: unavailable with a fromlist too
            ("numpy.r17_no_such_submodule", ("r17_name",), 0, "unavailable"),
        ],
    )
    def test_r17_pot_guard_fromlist_never_overrides_a_refusal(
        self, guard, name, fromlist, level, refusal
    ):
        messages = {
            "relative": f"relative imports are blocked (attempted {name!r}).",
            "not permitted": (
                f"import of {name!r} is not permitted in the sandbox "
                f"(only numpy/pandas internals resolve, and only if already loaded)."
            ),
            "unavailable": f"module {name!r} is not available in the sandbox.",
        }
        with pytest.raises(ImportError) as excinfo:
            guard(name, None, None, fromlist, level)
        assert str(excinfo.value) == "PoTExecutor: " + messages[refusal]

    @pytest.fixture
    def find_requests(self, monkeypatch):
        """Names the import machinery is asked to locate while the fixture is up."""
        asked: list[str] = []

        class _Recorder:
            def find_spec(self, fullname, path=None, target=None):
                asked.append(fullname)
                return None  # resolves nothing: the real finders stay in charge

        monkeypatch.setattr(sys, "meta_path", [_Recorder(), *sys.meta_path])
        return asked

    def test_r17_pot_guard_end_to_end_from_import_missing_name_is_refused_without_import(
        self, pot_on, monkeypatch, find_requests
    ):
        # An earlier version of this test expected the guard's "not available in
        # the sandbox" text and was red. That expectation was wrong: the guard is
        # called once, for ``numpy``, and returns it; the name is then resolved by
        # attribute access and refused by the interpreter -- which is where the
        # wording and the host path come from. No import is started, so the
        # guard's contract holds and there is nothing for the guard to refuse.
        guard_calls: list[tuple[Any, ...]] = []
        real_factory = pot_executor_module._make_guarded_import

        def _recording_factory():
            guard = real_factory()

            def _spy(name, globals=None, locals=None, fromlist=(), level=0):
                guard_calls.append((name, fromlist, level))
                return guard(name, globals, locals, fromlist, level)

            return _spy

        monkeypatch.setattr(pot_executor_module, "_make_guarded_import", _recording_factory)
        res = self._exec("from numpy import r17_no_such_attribute\nresult = 1")
        assert res.ok is False
        assert res.code_ran is False
        assert res.result is None
        assert res.error is not None
        assert "ImportError: cannot import name 'r17_no_such_attribute' from 'numpy'" in res.error
        # Exactly one guard call: the package. The name was never handed to it.
        assert guard_calls == [("numpy", ("r17_no_such_attribute",), 0)]
        # And never to the import machinery: no lookup on disk, nothing loaded.
        assert "numpy.r17_no_such_attribute" not in find_requests
        assert "numpy.r17_no_such_attribute" not in sys.modules

    @pytest.mark.parametrize("pkg_name", ["numpy", "pandas"])
    def test_r17_pot_guard_end_to_end_from_import_never_loads_a_module_from_disk(
        self, pot_on, monkeypatch, tmp_path, find_requests, pkg_name
    ):
        # The contract of ``_make_guarded_import``: "only modules that are
        # already present in sys.modules can be resolved ... No new filesystem
        # or network imports can ever occur." For ``from <pkg> import <name>``
        # that holds even though the guard never sees ``<name>``: the
        # interpreter refuses a name that is not an attribute instead of
        # importing a submodule. Pinned with a real package directory in
        # tmp_path appended to the package's search path, so the module IS
        # importable (the precondition below proves it) and only the sandbox
        # keeps it out. Nothing under site-packages is touched.
        pkg = importlib.import_module(pkg_name)
        shadow = "r17_shadow_sub"
        qualified = f"{pkg_name}.{shadow}"
        init = tmp_path / shadow / "__init__.py"
        init.parent.mkdir()
        init.write_text("MARKER = 'r17 loaded from disk'\n", encoding="utf-8")
        assert qualified not in sys.modules
        spec = importlib.machinery.PathFinder.find_spec(qualified, [str(tmp_path)])
        assert spec is not None
        assert Path(spec.origin).samefile(init)
        monkeypatch.setattr(pkg, "__path__", [*pkg.__path__, str(tmp_path)])
        try:
            res = self._exec(f"from {pkg_name} import {shadow}\nresult = {shadow}.MARKER")
            assert res.ok is False
            assert res.code_ran is False
            assert res.result is None
            assert res.error is not None
            assert f"ImportError: cannot import name {shadow!r} from {pkg_name!r}" in res.error
            # Not one request to the import machinery, nothing loaded.
            assert qualified not in find_requests
            assert qualified not in sys.modules
        finally:
            # Has work to do only if this contract ever breaks: undo what an
            # unguarded import would have installed in the process.
            sys.modules.pop(qualified, None)
            if hasattr(pkg, shadow):
                delattr(pkg, shadow)

    def test_r17_pot_guard_end_to_end_import_math_refused_by_guard(self, pot_on):
        # "import math" passes the token scan; the guarded __import__ refuses it.
        res = self._exec("import math\nresult = math.pi")
        assert res.ok is False
        assert res.code_ran is False
        assert res.error is not None
        assert "PoTExecutor: import of 'math' is not permitted in the sandbox" in res.error

    def test_r17_pot_guard_end_to_end_disabled_gate_runs_nothing(self, posix_gate):
        # Flag off (shipped default): the code string is never handed to exec and the
        # message names analysis_only.
        marker = []
        res = PoTExecutor().run_sandboxed(
            code="marker.append(1)\nresult = 1", data={"marker": marker}
        )
        assert res == pot_executor_module.PoTResult(
            ok=False,
            result=None,
            stdout="",
            error=(
                "PoTExecutor disabled (fail-closed). analysis_only is active; "
                "host-side LLM code execution is blocked."
            ),
            code_ran=False,
        )
        assert marker == []


def _previous_handler(signum, frame):
    """Stand-in for a host-installed SIGALRM handler that must be restored."""


class _FakeSignal:
    """Stand-in for the ``signal`` module: records the SIGALRM plumbing calls.

    Lets the POSIX signal path run verbatim on hosts without ``SIGALRM``
    (Windows) and without ever touching the real process timer.
    """

    SIGALRM = 14
    ITIMER_REAL = 0

    def __init__(self, previous_handler):
        self._previous_handler = previous_handler
        self.installed = previous_handler
        self.log: list[tuple[Any, ...]] = []

    def getsignal(self, signum):
        return self._previous_handler

    def signal(self, signum, handler):
        self.log.append(("signal", signum, handler))
        previous, self.installed = self.installed, handler
        return previous

    def setitimer(self, which, seconds):
        self.log.append(("setitimer", which, seconds))


@pytest.mark.unit
class TestR17PotTimeoutRunners:
    """Timeout plumbing: signal path (via a fake ``signal``), thread path, clamp."""

    @pytest.fixture
    def pot_on(self, posix_gate, monkeypatch):
        monkeypatch.setenv(_ANALYSIS_ONLY_ENV, "false")
        monkeypatch.setenv(_POT_ENV, "true")

    @pytest.fixture
    def fake_signal(self, monkeypatch):
        fake = _FakeSignal(previous_handler=_previous_handler)
        monkeypatch.setattr(pot_executor_module, "signal", fake)
        return fake

    @pytest.fixture
    def thread_path(self, monkeypatch):
        """Force ``_run_with_timeout`` onto the thread watchdog on every host.

        Production takes it when SIGALRM is unavailable (Windows) or the call
        is off the main thread. Here the ``signal`` stand-in has no SIGALRM at
        all, so a POSIX main thread cannot silently switch these tests to the
        signal path (where ``func`` runs on the caller's own thread).
        """
        monkeypatch.setattr(pot_executor_module, "signal", types.SimpleNamespace())

    def test_r17_pot_signal_path_arms_disarms_then_restores_handler(self, pot_on, fake_signal):
        res = PoTExecutor(timeout_seconds=5.0).run_sandboxed(code="result = 1 + 1")
        assert (res.ok, res.result, res.error) == (True, 2, None)
        # install -> arm -> disarm -> restore, in exactly that order
        assert [entry[0] for entry in fake_signal.log] == [
            "signal",
            "setitimer",
            "setitimer",
            "signal",
        ]
        assert [entry[1] for entry in fake_signal.log] == [
            fake_signal.SIGALRM,
            fake_signal.ITIMER_REAL,
            fake_signal.ITIMER_REAL,
            fake_signal.SIGALRM,
        ]
        assert (fake_signal.log[1][2], fake_signal.log[2][2]) == (5.0, 0)
        assert callable(fake_signal.log[0][2])
        assert fake_signal.log[0][2] is not _previous_handler
        assert fake_signal.log[3][2] is _previous_handler
        assert fake_signal.installed is _previous_handler

    def test_r17_pot_signal_handler_raises_the_internal_timeout(self, fake_signal):
        def _fire_alarm():
            # The handler the runner installed, invoked as SIGALRM would.
            fake_signal.log[0][2](fake_signal.SIGALRM, None)

        with pytest.raises(pot_executor_module._TimeoutError):
            PoTExecutor(timeout_seconds=5.0)._run_with_signal_timeout(_fire_alarm)
        # Exception path: the timer is disarmed BEFORE the handler is restored.
        assert [entry[0] for entry in fake_signal.log] == [
            "signal",
            "setitimer",
            "setitimer",
            "signal",
        ]
        assert fake_signal.log[2][2] == 0
        assert fake_signal.log[3][2] is _previous_handler

    def test_r17_pot_signal_alarm_inside_exec_reports_timeout(self, pot_on, fake_signal):
        def _alarm():
            fake_signal.log[0][2](fake_signal.SIGALRM, None)  # SIGALRM firing mid-exec

        res = PoTExecutor(timeout_seconds=5.0).run_sandboxed(
            code="print('hello')\nalarm()\nresult = 1", data={"alarm": _alarm}
        )
        # Normalized timeout message, no raw private-exception traceback; output
        # produced before the alarm is kept; nothing is credited as a result.
        assert res == pot_executor_module.PoTResult(
            ok=False,
            result=None,
            stdout="hello\n",
            error="PoTExecutor: execution timed out after 5.0s.",
            code_ran=False,
        )
        assert fake_signal.installed is _previous_handler

    def test_r17_pot_late_alarm_after_exec_keeps_the_finished_result(self, pot_on, monkeypatch):
        # The SIGALRM race: exec completed, the timer fires before it is disarmed.
        def _late_alarm(self, func):
            func()
            raise pot_executor_module._TimeoutError()

        monkeypatch.setattr(PoTExecutor, "_run_with_signal_timeout", _late_alarm)
        monkeypatch.setattr(PoTExecutor, "_run_with_thread_timeout", _late_alarm)
        res = PoTExecutor(timeout_seconds=5.0).run_sandboxed(code="result = 41 + 1")
        assert res == pot_executor_module.PoTResult(
            ok=True, result=42, stdout="", error=None, code_ran=True
        )

    def test_r17_pot_thread_path_joins_the_worker_when_func_completes(self):
        seen = {}

        def _func():
            seen["worker"] = threading.current_thread()

        PoTExecutor(timeout_seconds=5.0)._run_with_thread_timeout(_func)
        worker = seen["worker"]
        assert worker is not threading.main_thread()
        assert worker.daemon is True
        assert not worker.is_alive()

    @pytest.mark.parametrize("exc", [ValueError("boom"), KeyboardInterrupt()])
    def test_r17_pot_thread_path_reraises_worker_exception(self, exc):
        def _func():
            raise exc

        with pytest.raises(type(exc)) as excinfo:
            PoTExecutor(timeout_seconds=5.0)._run_with_thread_timeout(_func)
        assert excinfo.value is exc

    def test_r17_pot_thread_path_timeout_returns_and_worker_is_released(self):
        blocker = threading.Event()
        started = threading.Event()
        seen = {}

        def _func():
            seen["worker"] = threading.current_thread()
            started.set()
            blocker.wait(5)  # bounded: never outlives the test even if release fails

        t0 = time.monotonic()
        with pytest.raises(pot_executor_module._TimeoutError):
            PoTExecutor(timeout_seconds=0.2)._run_with_thread_timeout(_func)
        elapsed = time.monotonic() - t0
        assert elapsed < 1.0
        # Documented limitation of the thread path: the caller gets its timeout,
        # the daemon worker is still running until its own code returns.
        assert started.wait(5)
        worker = seen["worker"]
        assert worker.daemon is True
        assert worker.is_alive()
        blocker.set()
        worker.join(5)
        assert not worker.is_alive()

    @pytest.mark.parametrize("bad", [None, 0, -5])
    def test_r17_pot_timeout_invalidated_after_construction_still_watchdogs(
        self, monkeypatch, bad
    ):
        # The constructor clamps None/0/negative, but ``timeout_seconds`` is a public
        # attribute: a later assignment must not hand LLM code an unbounded run.
        calls: list[str] = []

        def _spy(name):
            def _run(self, func):
                calls.append(name)
                func()

            return _run

        monkeypatch.setattr(PoTExecutor, "_run_with_signal_timeout", _spy("signal"))
        monkeypatch.setattr(PoTExecutor, "_run_with_thread_timeout", _spy("thread"))
        executor = PoTExecutor(timeout_seconds=0.05)
        executor.timeout_seconds = bad
        executor._run_with_timeout(lambda: calls.append("func"))
        assert len(calls) == 2
        assert calls[0] in {"signal", "thread"}
        assert calls[1] == "func"

    def test_r17_pot_thread_watchdog_uses_the_clamped_deadline(self, thread_path, monkeypatch):
        monkeypatch.setattr(PoTExecutor, "_MIN_TIMEOUT_SECONDS", 0.05)
        blocker = threading.Event()
        workers: list[threading.Thread] = []

        def _block() -> None:
            workers.append(threading.current_thread())
            blocker.wait(0.5)  # bounded: on the unbounded path the test must fail, not hang

        executor = PoTExecutor(timeout_seconds=30)
        executor.timeout_seconds = 0
        t0 = time.monotonic()
        try:
            with pytest.raises(pot_executor_module._TimeoutError):
                executor._run_with_timeout(_block)
            assert time.monotonic() - t0 < 0.5
        finally:
            blocker.set()
            for worker in workers:
                if worker is not threading.current_thread():
                    worker.join(5)
        assert [worker for worker in workers if worker is threading.current_thread()] == []
        assert all(not worker.is_alive() for worker in workers)

    @pytest.mark.parametrize(
        ("assigned", "expected_join"), [(None, 0.05), (0, 0.05), (-5, 0.05), (0.25, 0.25)]
    )
    def test_r17_pot_thread_runner_is_given_the_clamped_deadline(
        self, thread_path, monkeypatch, assigned, expected_join
    ):
        # Regression: ``_run_with_timeout`` skipped the watchdog entirely when
        # ``timeout_seconds`` was None / <= 0. The constructor clamps those, but
        # the attribute is public and was re-read on every call. Pins the exact
        # join deadline: 0 or negative would time out before the worker even
        # starts, so "a timeout was raised" alone does not prove the clamp.
        monkeypatch.setattr(PoTExecutor, "_MIN_TIMEOUT_SECONDS", 0.05)
        joins: list[float | None] = []
        real = pot_executor_module.threading

        class _RecordingThread(real.Thread):
            def join(self, timeout=None):
                joins.append(timeout)
                super().join(timeout)

        monkeypatch.setattr(
            pot_executor_module,
            "threading",
            types.SimpleNamespace(
                Thread=_RecordingThread,
                current_thread=real.current_thread,
                main_thread=real.main_thread,
            ),
        )
        ran = threading.Event()
        executor = PoTExecutor(timeout_seconds=30)
        executor.timeout_seconds = assigned
        executor._run_with_timeout(ran.set)
        assert joins == [expected_join]
        assert ran.is_set()

    @pytest.mark.parametrize("assigned", [None, 0, -5])
    def test_r17_pot_signal_runner_is_given_the_clamped_deadline(
        self, posix_gate, fake_signal, monkeypatch, assigned
    ):
        # Same regression on the POSIX main-thread path: the timer is armed
        # with the clamp, not with the invalid value.
        monkeypatch.setattr(PoTExecutor, "_MIN_TIMEOUT_SECONDS", 0.05)
        ran: list[str] = []
        executor = PoTExecutor(timeout_seconds=30)
        executor.timeout_seconds = assigned
        executor._run_with_timeout(lambda: ran.append("func"))
        assert ran == ["func"]
        timers = [entry[2] for entry in fake_signal.log if entry[0] == "setitimer"]
        assert timers == [0.05, 0]  # armed with the clamp, disarmed with 0
        assert fake_signal.installed is _previous_handler

    @pytest.mark.parametrize("assigned", [None, 0])
    def test_r17_pot_timeout_message_reports_the_deadline_actually_enforced(
        self, pot_on, thread_path, monkeypatch, assigned
    ):
        # The reason text must name the deadline that was really applied (the
        # clamp), never the invalid value that happened to be in the attribute.
        monkeypatch.setattr(PoTExecutor, "_MIN_TIMEOUT_SECONDS", 0.05)
        blocker = threading.Event()
        workers: list[threading.Thread] = []
        executor = PoTExecutor(timeout_seconds=30)
        executor.timeout_seconds = assigned
        res = executor.run_sandboxed(
            code="grab()\nblocker.wait(0.5)\nresult = 1",
            data={"grab": lambda: workers.append(threading.current_thread()), "blocker": blocker},
        )
        try:
            assert res == pot_executor_module.PoTResult(
                ok=False,
                result=None,
                stdout="",
                error="PoTExecutor: execution timed out after 0.05s.",
                code_ran=False,
            )
        finally:
            blocker.set()
            for worker in workers:
                if worker is not threading.current_thread():
                    worker.join(5)
        # The code ran on a worker, never on the caller's own thread.
        assert [worker for worker in workers if worker is threading.current_thread()] == []
        assert all(not worker.is_alive() for worker in workers)


# --------------------------------------------------------------------------- #
# prediction_tools: per-entry validation, flush / settle / dispatch
# --------------------------------------------------------------------------- #

_PRED_RUN = "run-r17-prediction"
_PRED_INSTRUMENT = "BTCUSDT"
_PRED_SCOPE = "CONTRACT"
_PRED_AS_OF = "2026-03-10"  # fixed, far from any boundary (CI runs with TZ=UTC)
_PRED_LOGGER = "yialpha.agents.utils.prediction_tools"
_ENTRY_KEYS = [
    "confidence",
    "direction",
    "expected_return",
    "horizon_days",
    "price_basis",
    "prob_up",
    "target_currency",
    "target_price",
]


def _bind_prediction_run(run_id: str = _PRED_RUN) -> None:
    set_ledger_run_context(
        run_id, _PRED_INSTRUMENT, "crypto_perp", "pure_crypto_perp", _PRED_AS_OF
    )


@pytest.fixture
def prediction_stage():
    """Start and end with no run context, no capture, no quality events."""
    reset_ledger_run_context()
    prediction_tools.reset_prediction_capture_for_test()
    quality.reset_quality()
    yield
    reset_ledger_run_context()
    prediction_tools.reset_prediction_capture_for_test()
    quality.reset_quality()


def _armed(analyst: str = "market") -> Any:
    _bind_prediction_run()
    prediction_tools.begin_prediction_capture(analyst, _PRED_INSTRUMENT, _PRED_SCOPE)
    return prediction_tools._active_capture()


def _horizons(capture: Any) -> list[int]:
    return [entry.horizon_days for entry in capture.entries]


@pytest.mark.unit
class TestR17PredictionToolEntries:
    @pytest.mark.parametrize("currency", ["BTC", "usdt", "", "U"])
    def test_r17_prediction_tool_rejects_unknown_target_currency_per_entry(
        self, prediction_stage, currency
    ):
        # target_currency is free-form in the ledger row; the tool is the only
        # gate. Exact, case-sensitive membership: the bad horizon is refused
        # with its reason, its neighbours land.
        capture = _armed()
        reply = prediction_tools.submit_prediction.invoke(
            {
                "predictions": [
                    {"horizon_days": 1, "direction": "up", "prob_up": 0.6},
                    {
                        "horizon_days": 5,
                        "direction": "up",
                        "target_price": 95000.0,
                        "target_currency": currency,
                        "price_basis": "mark",
                    },
                    {"horizon_days": 21, "direction": "down", "prob_up": 0.4},
                ]
            }
        )
        assert reply == (
            f"submit_prediction: accepted horizons [1, 21] for market on "
            f"{_PRED_INSTRUMENT} ({_PRED_SCOPE}) — committed. Rejected entries: "
            f"entry 2: target_currency {currency!r} not in ['USD', 'USDT']"
        )
        assert _horizons(capture) == [1, 21]

    @pytest.mark.parametrize(
        ("entry", "field"),
        [
            ({"horizon_days": 5, "direction": "up", "prob_up": True}, "prob_up"),
            ({"horizon_days": 5, "direction": "down", "prob_up": False}, "prob_up"),
            ({"horizon_days": 5, "direction": "up", "confidence": True}, "confidence"),
            ({"horizon_days": True, "direction": "up"}, "horizon_days"),
        ],
    )
    def test_r17_prediction_tool_refuses_a_boolean_in_a_numeric_field(
        self, prediction_stage, entry, field
    ):
        # Regression. The tool argument is a pydantic model with numeric fields,
        # and pydantic turns True into 1.0 (and into the 1-day horizon for an
        # int field) BEFORE any ledger validator runs: a JSON true for prob_up
        # was FILED as a certain forecast, false as a certain fall, and rows are
        # immutable once accepted. The ledger validators refuse booleans since
        # round 17, but they only ever received the coerced float here.
        capture = _armed()

        with pytest.raises(ValidationError) as exc_info:
            prediction_tools.submit_prediction.invoke(
                {
                    "predictions": [
                        {"horizon_days": 1, "direction": "up", "prob_up": 0.6},
                        entry,
                    ]
                }
            )

        assert f"{field} must be a number, not a boolean" in str(exc_info.value)
        # Like every other type-level error, a boolean rejects the WHOLE call:
        # nothing is filed, not even the valid neighbour.
        assert capture.entries == []

    @pytest.mark.parametrize(
        "entry",
        [
            {"horizon_days": 5, "direction": "up", "prob_up": 1},
            {"horizon_days": 5, "direction": "down", "prob_up": 0},
            {"horizon_days": "5", "direction": "up", "prob_up": "0.7", "confidence": 1},
        ],
        ids=["int-one", "int-zero", "numeric-strings"],
    )
    def test_r17_prediction_tool_still_accepts_ints_and_numeric_strings(
        self, prediction_stage, entry
    ):
        # Only the boolean is refused: an int or a quoted number is a number.
        capture = _armed()

        reply = prediction_tools.submit_prediction.invoke({"predictions": [entry]})

        assert reply == (
            f"submit_prediction: accepted horizons [5] for market on {_PRED_INSTRUMENT} "
            f"({_PRED_SCOPE}) — committed."
        )
        assert capture.entries[0].prob_up == pytest.approx(float(entry["prob_up"]))

    @pytest.mark.parametrize(
        ("entry", "unknown"),
        [
            ({"horizon_days": 5, "direction": "up", "prob_pu": 0.7}, "prob_pu"),
            ({"horizon_days": 5, "direction": "up", "Prob_Up": 0.7}, "Prob_Up"),
            (
                {"horizon_days": 5, "direction": "up", "confidence": 0.8, "conviction": "high"},
                "conviction",
            ),
        ],
    )
    def test_r17_prediction_tool_unknown_key_is_rejected_not_silently_dropped(
        self, prediction_stage, entry, unknown
    ):
        # Regression. ``coerce_prediction_entry`` documents "a typo'd key must
        # fail loudly instead of silently dropping", but the tool argument is a
        # pydantic model that ignored unknown keys, so the check never saw them:
        # ``{"horizon_days": 5, "direction": "up", "prob_pu": 0.7}`` was FILED
        # with prob_up=None and the reply said "committed". Rows are immutable
        # once accepted, so the corrected entry was then refused ("immutable
        # once accepted"): a typo'd field was lost for good, unreported.
        capture = _armed()
        reply = prediction_tools.submit_prediction.invoke(
            {
                "predictions": [
                    {"horizon_days": 1, "direction": "up", "prob_up": 0.6},
                    entry,
                    {"horizon_days": 21, "direction": "down", "prob_up": 0.4},
                ]
            }
        )
        assert reply == (
            f"submit_prediction: accepted horizons [1, 21] for market on {_PRED_INSTRUMENT} "
            f"({_PRED_SCOPE}) — committed. Rejected entries: entry 2: unknown prediction "
            f"entry keys ['{unknown}']; allowed keys: {_ENTRY_KEYS}"
        )
        assert _horizons(capture) == [1, 21]
        # Nothing was filed for that horizon, so the corrected entry still can be.
        corrected = prediction_tools.submit_prediction.invoke(
            {"predictions": [{"horizon_days": 5, "direction": "up", "prob_up": 0.7}]}
        )
        assert corrected == (
            f"submit_prediction: accepted horizons [5] for market on {_PRED_INSTRUMENT} "
            f"({_PRED_SCOPE}) — committed."
        )
        assert [entry.prob_up for entry in capture.entries] == [0.6, 0.4, 0.7]

    def test_r17_prediction_tool_schema_gains_no_field_from_the_fix(self):
        # The LLM-facing tool-call schema: still exactly the eight entry fields
        # (same titles, types and descriptions as before); unknown keys are
        # explicitly allowed through so the tool body can reject them per entry.
        schema = prediction_tools.submit_prediction.tool_call_schema.model_json_schema()
        entry = schema["$defs"]["PredictionEntryInput"]
        assert sorted(entry["properties"]) == sorted(_ENTRY_KEYS)
        assert entry["required"] == ["horizon_days", "direction"]
        assert entry["additionalProperties"] is True


def _must_not_write(*_args: Any, **_kwargs: Any) -> Any:
    raise AssertionError("nothing may reach the ledger on this path")


def _file_one_horizon() -> str:
    return prediction_tools.submit_prediction.invoke(
        {"predictions": [{"horizon_days": 1, "direction": "up", "prob_up": 0.6}]}
    )


@pytest.mark.unit
class TestR17PredictionFlush:
    @pytest.fixture
    def writes(self, monkeypatch):
        """Every ledger write ``flush_predictions`` would make."""
        calls: list[dict[str, Any]] = []

        def _submit(**kwargs):
            calls.append(kwargs)
            return [f"pred-{len(calls)}"]

        monkeypatch.setattr(prediction_tools, "submit_predictions", _submit)
        return calls

    def test_r17_prediction_flush_with_no_capture_armed_writes_nothing(
        self, prediction_stage, writes, monkeypatch
    ):
        # Run context bound, ``begin_prediction_capture`` never ran.
        _bind_prediction_run()
        monkeypatch.setattr(prediction_tools, "evidence_ids_for_run", _must_not_write)
        assert prediction_tools.flush_predictions() == []
        assert prediction_tools.flush_predictions(analyst="market") == []
        assert writes == []

    def test_r17_prediction_flush_for_another_analyst_leaves_the_capture_pending(
        self, prediction_stage, writes, monkeypatch
    ):
        capture = _armed("market")
        assert "accepted horizons [1]" in _file_one_horizon()
        monkeypatch.setattr(prediction_tools, "evidence_ids_for_run", _must_not_write)
        assert prediction_tools.flush_predictions(analyst="news") == []
        assert writes == []
        # Not retired on the sibling's behalf: the rows can still be written.
        assert capture.submitted is False
        assert prediction_tools.prediction_capture_pending() is True

    def test_r17_prediction_flush_never_writes_a_capture_from_another_run(
        self, prediction_stage, writes
    ):
        # Armed under run A; the context now names run B. Attributing A's
        # forecasts to B would file them against the wrong run's evidence.
        capture = _armed("market")
        assert "accepted horizons [1]" in _file_one_horizon()
        set_ledger_run_context(
            "run-r17-other", _PRED_INSTRUMENT, "crypto_perp", "pure_crypto_perp", _PRED_AS_OF
        )
        assert prediction_tools.flush_predictions() == []
        assert prediction_tools.flush_predictions(analyst="market") == []
        assert writes == []
        assert (capture.run_id, capture.submitted) == (_PRED_RUN, False)
        # Under its own run the same capture flushes, carrying run A's id.
        set_ledger_run_context(
            _PRED_RUN, _PRED_INSTRUMENT, "crypto_perp", "pure_crypto_perp", _PRED_AS_OF
        )
        assert prediction_tools.flush_predictions(analyst="market") == ["pred-1"]
        assert writes[0]["run_id"] == _PRED_RUN
        assert writes[0]["analysis_as_of"] == _PRED_AS_OF
        assert capture.submitted is True
        # Retired: a second flush neither writes nor asks again.
        assert prediction_tools.flush_predictions(analyst="market") == []
        assert len(writes) == 1

    def test_r17_prediction_flush_other_analysts_capture_is_untouched_by_a_scoped_flush(
        self, prediction_stage, writes
    ):
        # The analyst filter selects one capture out of two for the same run.
        _armed("market")
        assert "accepted horizons [1]" in _file_one_horizon()
        _armed("news")
        assert "accepted horizons [1]" in _file_one_horizon()
        assert prediction_tools.flush_predictions(analyst="news") == ["pred-1"]
        assert [call["analyst"] for call in writes] == ["news"]
        assert prediction_tools.prediction_capture_pending() is True  # market still pending
        assert prediction_tools.flush_predictions(analyst="news") == []  # news retired
        assert prediction_tools.flush_predictions() == ["pred-2"]
        assert [call["analyst"] for call in writes] == ["news", "market"]
        assert prediction_tools.prediction_capture_pending() is False


@pytest.mark.unit
class TestR17PredictionSettle:
    @staticmethod
    def _only_record(caplog):
        records = [r for r in caplog.records if r.name == _PRED_LOGGER]
        assert len(records) == 1
        return records[0]

    def test_r17_prediction_settle_swallows_a_failing_sentinel_recorder(
        self, prediction_stage, monkeypatch, caplog
    ):
        _armed("market")  # armed and empty: settlement wants to record the sentinel

        def _boom(*_args, **_kwargs):
            raise RuntimeError("quality ledger down")

        monkeypatch.setattr(quality, "record_sentinel", _boom)
        with caplog.at_level(logging.WARNING, logger=_PRED_LOGGER):
            assert prediction_tools.settle_prediction_capture("market", _PRED_INSTRUMENT) is None
        record = self._only_record(caplog)
        assert record.levelno == logging.WARNING
        assert record.getMessage() == "prediction capture settlement failed for market"
        assert record.exc_info is not None
        assert (record.exc_info[0], str(record.exc_info[1])) == (RuntimeError, "quality ledger down")

    def test_r17_prediction_settle_swallows_a_failing_flush(
        self, prediction_stage, monkeypatch, caplog
    ):
        capture = _armed("market")
        assert "accepted horizons [1]" in _file_one_horizon()

        def _boom(**_kwargs):
            raise RuntimeError("ledger locked")

        monkeypatch.setattr(prediction_tools, "flush_predictions", _boom)
        with caplog.at_level(logging.WARNING, logger=_PRED_LOGGER):
            prediction_tools.settle_prediction_capture("market", _PRED_INSTRUMENT)
        record = self._only_record(caplog)
        assert record.getMessage() == "prediction capture settlement failed for market"
        assert (record.exc_info[0], str(record.exc_info[1])) == (RuntimeError, "ledger locked")
        # The buffer is left as it was: nothing was filed, nothing was retired,
        # so a later flush can still write it.
        assert _horizons(capture) == [1]
        assert (capture.submitted, capture.written_horizons) == (False, set())
        assert quality.snapshot_quality() == []  # no sentinel claims "never filed"


def _tool_call(name: Any, args: Any, call_id: str) -> dict[str, Any]:
    return {"name": name, "args": args, "id": call_id}


class TestR17PredictionDispatch:
    def test_r17_prediction_dispatch_counts_only_submit_prediction_calls(
        self, prediction_stage, caplog
    ):
        _armed("market")
        response = types.SimpleNamespace(
            tool_calls=[
                _tool_call("get_binance_klines", {"symbol": "BTCUSDT"}, "c1"),
                ("submit_prediction", {"predictions": []}),  # not a dict
                _tool_call("submit_prediction ", {"predictions": []}, "c3"),  # other name
                _tool_call(None, {"predictions": []}, "c4"),
                _tool_call(
                    "submit_prediction",
                    {"predictions": [{"horizon_days": 1, "direction": "up", "prob_up": 0.6}]},
                    "c5",
                ),
            ]
        )
        with caplog.at_level(logging.INFO, logger=_PRED_LOGGER):
            assert prediction_tools.dispatch_prediction_tool_calls(response) == 1
        assert _horizons(prediction_tools._active_capture()) == [1]
        assert [r.getMessage() for r in caplog.records if r.name == _PRED_LOGGER] == [
            f"submit_prediction dispatch: submit_prediction: accepted horizons [1] for market "
            f"on {_PRED_INSTRUMENT} ({_PRED_SCOPE}) — committed."
        ]

    @pytest.mark.parametrize(
        "response",
        [
            object(),
            types.SimpleNamespace(),
            types.SimpleNamespace(tool_calls=None),
            types.SimpleNamespace(tool_calls=[]),
            types.SimpleNamespace(tool_calls=[None, "submit_prediction", 3]),
        ],
    )
    def test_r17_prediction_dispatch_without_a_usable_call(self, prediction_stage, response):
        _armed("market")
        assert prediction_tools.dispatch_prediction_tool_calls(response) == 0
        assert prediction_tools._active_capture().entries == []

    def test_r17_prediction_dispatch_accepts_a_real_ai_message(self, prediction_stage):
        _armed("market")
        response = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "submit_prediction",
                    "args": {"predictions": [{"horizon_days": 21, "direction": "down"}]},
                    "id": "call-1",
                    "type": "tool_call",
                }
            ],
        )
        assert prediction_tools.dispatch_prediction_tool_calls(response) == 1
        assert _horizons(prediction_tools._active_capture()) == [21]

    def test_r17_prediction_dispatch_one_failing_call_is_logged_and_the_rest_still_run(
        self, prediction_stage, monkeypatch, caplog
    ):
        _armed("market")
        entries: list[dict[str, Any]] = [
            {"horizon_days": 1, "direction": "up", "prob_up": 0.6},  # first call: fine
            {"horizon_days": 5, "direction": "up", "prob_up": 0.7, "note": "ignored"},
            {"horizon_days": 21, "direction": "down"},  # last call: fine
        ]
        real_invoke = prediction_tools.submit_prediction.invoke
        seen: list[dict[str, Any]] = []

        def _flaky_invoke(payload, config=None, **kwargs):
            seen.append(payload)
            if payload["predictions"] and "note" in payload["predictions"][0]:
                raise RuntimeError("tool exploded")
            return real_invoke(payload, config, **kwargs)

        # Replaces the module-level tool object dispatch looks up by name.
        monkeypatch.setattr(
            prediction_tools, "submit_prediction", types.SimpleNamespace(invoke=_flaky_invoke)
        )
        response = types.SimpleNamespace(
            tool_calls=[
                _tool_call("submit_prediction", {"predictions": entries[0:1]}, "c1"),
                _tool_call("submit_prediction", "not-a-mapping", "c2"),
                _tool_call("submit_prediction", {"predictions": entries[1:2]}, "c3"),
                _tool_call("submit_prediction", {"predictions": entries[2:3]}, "c4"),
                _tool_call("submit_prediction", {}, "c5"),  # no predictions at all
            ]
        )
        with caplog.at_level(logging.INFO, logger=_PRED_LOGGER):
            assert prediction_tools.dispatch_prediction_tool_calls(response) == 3
        assert _horizons(prediction_tools._active_capture()) == [1, 21]
        # The call with non-mapping args never reaches the tool; the one without
        # a "predictions" key reaches it with an empty list.
        assert seen == [
            {"predictions": entries[0:1]},
            {"predictions": entries[1:2]},
            {"predictions": entries[2:3]},
            {"predictions": []},
        ]
        records = [r for r in caplog.records if r.name == _PRED_LOGGER]
        assert [r.getMessage() for r in records] == [
            "submit_prediction dispatch: submit_prediction: accepted horizons [1] for market "
            f"on {_PRED_INSTRUMENT} ({_PRED_SCOPE}) — committed.",
            "submit_prediction dispatch failed",
            "submit_prediction dispatch failed",
            "submit_prediction dispatch: submit_prediction: accepted horizons [21] for market "
            f"on {_PRED_INSTRUMENT} ({_PRED_SCOPE}) — committed.",
            "submit_prediction dispatch: submit_prediction: no entries accepted.",
        ]
        failed = [r for r in records if r.getMessage() == "submit_prediction dispatch failed"]
        assert [r.levelno for r in failed] == [logging.WARNING] * 2
        # ``dict("not-a-mapping")`` -> ValueError; the second one is ours.
        assert [type(r.exc_info[1]).__name__ for r in failed] == ["ValueError", "RuntimeError"]
        assert str(failed[1].exc_info[1]) == "tool exploded"


# --------------------------------------------------------------------------- #
# crypto_native_tools / news_data_tools: the thin ``@tool`` wrappers
# --------------------------------------------------------------------------- #


def _routed_spy(monkeypatch, module: Any) -> list[tuple[str, tuple[Any, ...], dict[str, Any]]]:
    """Replace ``route_to_vendor`` as the module sees it; return the call log."""
    calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []

    def _route(method, *args, **kwargs):
        calls.append((method, args, kwargs))
        return f"ROUTED {method}"

    monkeypatch.setattr(module, "route_to_vendor", _route)
    return calls


@pytest.mark.unit
class TestR17CryptoNetworkTool:
    _NETWORK = "get_crypto_network_fundamentals"

    @pytest.fixture
    def routed(self, monkeypatch):
        return _routed_spy(monkeypatch, crypto_native_tools)

    def test_r17_crypto_network_tool_forwards_ticker_date_and_default_window(self, routed):
        result = crypto_native_tools.get_crypto_network_fundamentals.invoke(
            {"ticker": "UNIUSDT", "curr_date": "2026-03-10"}
        )
        assert result == "ROUTED get_crypto_network_fundamentals"
        assert routed == [(self._NETWORK, ("UNIUSDT", "2026-03-10", 180), {})]

    def test_r17_crypto_network_tool_forwards_an_explicit_window(self, routed):
        crypto_native_tools.get_crypto_network_fundamentals.invoke(
            {"ticker": "UNIUSDT", "curr_date": "2026-03-10", "look_back_days": 30}
        )
        assert routed == [(self._NETWORK, ("UNIUSDT", "2026-03-10", 30), {})]

    def test_r17_crypto_network_tool_clamps_a_date_past_the_pinned_analysis_date(self, routed):
        # Backtest pinned to 2026-03-10; the model echoes a later date (today's
        # cached header). The vendor PIT-filters by the date it is handed, so a
        # post-decision date here would admit rows the backtest could not see.
        with pinned_analysis_date("2026-03-10"):
            crypto_native_tools.get_crypto_network_fundamentals.invoke(
                {"ticker": "UNIUSDT", "curr_date": "2026-11-24", "look_back_days": 30}
            )
        assert routed == [(self._NETWORK, ("UNIUSDT", "2026-03-10", 30), {})]

    def test_r17_crypto_network_tool_keeps_a_date_on_or_before_the_pinned_one(self, routed):
        with pinned_analysis_date("2026-03-10"):
            crypto_native_tools.get_crypto_network_fundamentals.invoke(
                {"ticker": "UNIUSDT", "curr_date": "2026-03-10"}
            )
            crypto_native_tools.get_crypto_network_fundamentals.invoke(
                {"ticker": "UNIUSDT", "curr_date": "2026-02-01"}
            )
        assert [args for _, args, _ in routed] == [
            ("UNIUSDT", "2026-03-10", 180),
            ("UNIUSDT", "2026-02-01", 180),
        ]

    def test_r17_crypto_network_tool_empty_date_takes_the_pinned_date(self, routed):
        # Pinned run: an empty date must not sidestep the historical window.
        with pinned_analysis_date("2026-03-10"):
            crypto_native_tools.get_crypto_network_fundamentals.invoke(
                {"ticker": "UNIUSDT", "curr_date": ""}
            )
        assert routed == [(self._NETWORK, ("UNIUSDT", "2026-03-10", 180), {})]
        # Live run: nothing is pinned, the empty value is left to the vendor.
        crypto_native_tools.get_crypto_network_fundamentals.invoke(
            {"ticker": "UNIUSDT", "curr_date": ""}
        )
        assert routed[-1] == (self._NETWORK, ("UNIUSDT", "", 180), {})

    def test_r17_crypto_network_tool_reaches_the_registered_defillama_route(self, monkeypatch):
        # Through the real router: the method name resolves and the arguments
        # arrive at the registered vendor in its positional order.
        seen: list[tuple[Any, ...]] = []

        def _defillama(*args):
            seen.append(args)
            return "## Network fundamentals\nSTUB"

        monkeypatch.setitem(
            vendor_interface.VENDOR_METHODS[self._NETWORK], "defillama", _defillama
        )
        result = crypto_native_tools.get_crypto_network_fundamentals.invoke(
            {"ticker": "UNIUSDT", "curr_date": "2026-03-10", "look_back_days": 60}
        )
        assert result == "## Network fundamentals\nSTUB"
        assert seen == [("UNIUSDT", "2026-03-10", 60)]


@pytest.mark.unit
class TestR17GlobalNewsTool:
    _NEWS = "get_global_news"

    @pytest.fixture
    def routed(self, monkeypatch):
        return _routed_spy(monkeypatch, news_data_tools)

    def test_r17_global_news_tool_omitted_options_reach_the_vendor_as_none(self, routed):
        # The wrapper substitutes nothing: the lookback / limit defaults live in
        # DEFAULT_CONFIG and are the vendor's fallbacks.
        result = news_data_tools.get_global_news.invoke({"curr_date": "2026-03-10"})
        assert result == "ROUTED get_global_news"
        assert routed == [(self._NEWS, ("2026-03-10", None, None), {})]

    def test_r17_global_news_tool_forwards_explicit_options(self, routed):
        news_data_tools.get_global_news.invoke(
            {"curr_date": "2026-03-10", "look_back_days": 3, "limit": 12}
        )
        assert routed == [(self._NEWS, ("2026-03-10", 3, 12), {})]

    def test_r17_global_news_tool_numeric_strings_are_integers_for_the_vendor(self, routed):
        # A model that quotes the numbers still hands the vendor ints.
        news_data_tools.get_global_news.invoke(
            {"curr_date": "2026-03-10", "look_back_days": "3", "limit": "12"}
        )
        assert routed == [(self._NEWS, ("2026-03-10", 3, 12), {})]

    def test_r17_global_news_tool_passes_the_date_through_unchanged(self, routed):
        # No clamp in this wrapper, by design: both registered vendors clamp
        # against the pinned analysis date and cut a timestamp form to its date
        # head themselves. Pinned here so that adding one would be deliberate.
        with pinned_analysis_date("2026-03-10"):
            news_data_tools.get_global_news.invoke(
                {"curr_date": "2026-03-10 16:00:00", "look_back_days": 7, "limit": 5}
            )
        assert routed == [(self._NEWS, ("2026-03-10 16:00:00", 7, 5), {})]

    def test_r17_global_news_tool_reaches_the_first_vendor_of_the_news_chain(self, monkeypatch):
        seen: list[tuple[Any, ...]] = []

        def _yfinance(curr_date, look_back_days=None, limit=None):
            seen.append(("yfinance", curr_date, look_back_days, limit))
            return "STUB HEADLINES"

        def _alpha_vantage(*args):
            seen.append(("alpha_vantage", *args))
            return "MUST NOT BE ASKED"

        table = vendor_interface.VENDOR_METHODS[self._NEWS]
        monkeypatch.setitem(table, "yfinance", _yfinance)
        monkeypatch.setitem(table, "alpha_vantage", _alpha_vantage)
        result = news_data_tools.get_global_news.invoke(
            {"curr_date": "2026-03-10", "look_back_days": 7, "limit": 5}
        )
        assert result == "STUB HEADLINES"
        assert seen == [("yfinance", "2026-03-10", 7, 5)]
