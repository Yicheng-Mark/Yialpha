"""Program-of-Thoughts (PoT) sandbox executor for analyst numerical reasoning.

Analyst LLMs currently do arithmetic in their heads and get roughly half of the
calculations wrong. The PoT pattern externalizes the math: the LLM emits Python
code that performs the computation, this module runs that code in a restricted
namespace seeded with the *verified* market data, and the computed numbers are
returned to the caller. Externalizing the arithmetic raises numerical accuracy
considerably.

SECURITY MODEL — read this before relying on this module:

    The sandbox reduces risk for *trusted*, LLM-generated analysis code. It is
    NOT a hard security boundary against adversarial input. Defense is layered:

    1. ``__builtins__`` is replaced with a fixed allowlist, removing
       ``__import__``, ``open``, ``eval``, ``exec``, ``compile``, ``getattr``,
       ``globals``, ``locals``, ``input``, ``breakpoint``, etc. Without
       ``__import__`` the literal ``import os`` statement cannot succeed even
       if it slips past the token guard.
    2. Only ``numpy`` (as ``np``) and ``pandas`` (as ``pd``) are exposed. No
       ``os``/``sys``/``subprocess``.
    3. A pre-exec token scan rejects obviously dangerous substrings
       (``__import__``, ``import os``, ``subprocess``, ``open(``, ...) before
       any code runs. This is defense-in-depth; the restricted builtins are the
       real barrier.
    4. Code size (line count and character count) is capped to bound runtime.
    5. A best-effort timeout is applied. On POSIX main threads this uses
       ``SIGALRM``. Windows host execution is refused entirely until PoT runs
       in a terminable, process-isolated sandbox.

    6. Execution is disabled by default and unconditionally blocked whenever
       the project is in its default ``analysis_only`` mode.

If you need to run genuinely untrusted code, do not use this module — reach for
a container/seccomp-based sandbox instead.

CONVENTION: the LLM MUST assign its final answer to a variable named ``result``
(or to the name passed as ``result_var``). ``exec`` cannot reliably surface the
value of a bare trailing expression, so the result is read back from the
namespace by name.
"""

from __future__ import annotations

import contextlib
import io
import os
import platform
import signal
import sys
import threading
import traceback
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

__all__ = ["PoTEnableSwitch", "PoTResult", "PoTExecutor"]


class PoTEnableSwitch:
    """Runtime opt-in for host-side execution of LLM-generated Python.

    The restricted namespace is a convenience guard, not an OS security
    boundary: numpy/pandas are powerful host objects.  PoT therefore defaults
    off and is checked immediately before every execution. It can never run
    while ``analysis_only`` is active. Windows host execution is also refused:
    its timeout cannot terminate a Python worker safely without process-level
    isolation. Missing, malformed, or non-boolean values fail closed.
    """

    _ENV_VAR = "YIALPHA_POT_ENABLED"
    _ANALYSIS_ONLY_ENV = "YIALPHA_ANALYSIS_ONLY"
    _TRUTHY = frozenset({"true", "1", "yes", "on"})
    _FALSY = frozenset({"false", "0", "no", "off"})

    @classmethod
    def _config_flag(cls, name: str, default: bool) -> bool:
        try:
            from yialpha.dataflows.config import get_config

            value = get_config().get(name, default)
            return value if isinstance(value, bool) else default
        except Exception:  # noqa: BLE001 - uncertainty must fail closed
            return default

    @classmethod
    def _env_or_config_flag(
        cls,
        env_name: str,
        config_name: str,
        *,
        default: bool,
    ) -> bool:
        raw = os.environ.get(env_name)
        if raw is None or raw.strip() == "":
            return cls._config_flag(config_name, default)
        value = raw.strip().lower()
        if value in cls._TRUTHY:
            return True
        if value in cls._FALSY:
            return False
        return default

    @classmethod
    def is_enabled(cls) -> bool:
        analysis_only = cls._env_or_config_flag(
            cls._ANALYSIS_ONLY_ENV,
            "analysis_only",
            default=True,
        )
        if analysis_only:
            return False
        if platform.system() == "Windows":
            return False
        return cls._env_or_config_flag(
            cls._ENV_VAR,
            "pot_enabled",
            default=False,
        )

    @classmethod
    def reason(cls) -> str:
        analysis_only = cls._env_or_config_flag(
            cls._ANALYSIS_ONLY_ENV,
            "analysis_only",
            default=True,
        )
        if analysis_only:
            return "analysis_only is active; host-side LLM code execution is blocked."
        if platform.system() == "Windows":
            return (
                "PoT host execution is blocked on Windows until it uses a "
                "terminable, process-isolated sandbox."
            )
        raw = os.environ.get(cls._ENV_VAR)
        if raw is None or raw.strip() == "":
            if cls._config_flag("pot_enabled", False):
                return "pot_enabled=True in runtime config."
            return (
                f"{cls._ENV_VAR} is unset and pot_enabled is not explicitly true; "
                "LLM-generated code execution is disabled."
            )
        if raw.strip().lower() not in cls._TRUTHY | cls._FALSY:
            return f"{cls._ENV_VAR}={raw!r} is invalid; execution is disabled."
        return f"{cls._ENV_VAR}={raw!r}."


def _make_guarded_import() -> Any:
    """Return an ``__import__`` callable that only resolves already-loaded modules.

    Why this exists: numpy and pandas perform *lazy* submodule imports at
    runtime (e.g. ``np.diff(...).mean()`` pulls in ``numpy._core._methods``
    on first call). If ``__import__`` is absent from ``__builtins__`` those
    internal calls raise ``KeyError: '__import__'`` and crash otherwise-valid
    analysis code.

    The guard restores the name but tightens its behavior to a strict
    allowlist: only modules that are *already present in ``sys.modules``*
    (i.e. were imported by the host before the sandbox ran, never by the
    sandboxed code itself) and whose top-level package is numpy or pandas
    can be resolved. Any attempt to import ``os``, ``sys``, ``subprocess``,
    or anything else the LLM might reach for, raises ``ImportError``. No
    new filesystem or network imports can ever occur.
    """

    _ALLOWED_TOP_LEVELS = frozenset({"numpy", "pandas"})

    def _guarded_import(
        name: str,
        globals: dict | None = None,  # noqa: A002 - matches builtin signature
        locals: dict | None = None,  # noqa: A002 - matches builtin signature
        fromlist: tuple = (),
        level: int = 0,
    ) -> Any:
        if level != 0:
            raise ImportError(
                f"PoTExecutor: relative imports are blocked (attempted {name!r})."
            )
        top = name.split(".")[0]
        if top not in _ALLOWED_TOP_LEVELS:
            raise ImportError(
                f"PoTExecutor: import of {name!r} is not permitted in the "
                f"sandbox (only numpy/pandas internals resolve, and only if "
                f"already loaded)."
            )
        # Resolve only from already-loaded modules — never hit the import
        # machinery / filesystem.
        if name not in sys.modules:
            raise ImportError(
                f"PoTExecutor: module {name!r} is not available in the sandbox."
            )
        module = sys.modules[name]
        # Honor ``from X import Y`` by returning the package so Python can
        # pull the named attribute off it.
        if fromlist:
            return module
        # Plain ``import X.Y`` returns the top-level package per Python semantics.
        return sys.modules[top]

    return _guarded_import


# Safe builtins allowlist. Setting ``__builtins__`` to this dict inside the
# exec namespace is what removes ``__import__``, ``open``, ``eval``, ``exec``,
# ``compile``, ``getattr``, ``setattr``, ``globals``, ``locals``, ``input``,
# ``breakpoint`` and everything else not listed here.
_SAFE_BUILTINS: dict[str, Any] = {
    "abs": abs,
    "min": min,
    "max": max,
    "round": round,
    "sum": sum,
    "len": len,
    "range": range,
    "enumerate": enumerate,
    "zip": zip,
    "sorted": sorted,
    "any": any,
    "all": all,
    "int": int,
    "float": float,
    "str": str,
    "bool": bool,
    "list": list,
    "tuple": tuple,
    "dict": dict,
    "set": set,
    "True": True,
    "False": False,
    "None": None,
    "print": print,
    "isinstance": isinstance,
    "type": type,
    "map": map,
    "filter": filter,
    "pow": pow,
    "divmod": divmod,
    "reversed": reversed,
    "frozenset": frozenset,
}

# A guarded ``__import__`` is added to the allowlist at runtime (see
# ``_make_guarded_import``) so numpy/pandas lazy submodule resolution works
# while every other import path is blocked. It is injected into the dict in
# ``_sandbox_namespace`` rather than here, to keep the module-level constant
# pristine.

# Dangerous substrings scanned for in the *user code string* before exec. These
# are defense-in-depth on top of the restricted builtins. Kept deliberately
# broad; a false positive simply rejects code the LLM should not be writing in
# a numerics sandbox anyway. Note the bare ``__`` rule — dunder access is the
# usual escape hatch from restricted exec, so any dunder use is rejected.
_DANGEROUS_TOKENS: tuple[str, ...] = (
    "__import__",
    "import os",
    "import sys",
    "import subprocess",
    "from os",
    "from sys",
    "from subprocess",
    "subprocess",
    "open(",
    "eval(",
    "exec(",
    "compile(",
    "globals(",
    "locals(",
    "getattr(",
    "setattr(",
    "delattr(",
    "breakpoint(",
    "input(",
    # pandas/numpy deserialization and host filesystem IO. ``read_pickle`` is
    # arbitrary code execution (pickle deserialization RCE), as is
    # ``np.load(..., allow_pickle=True)``; ``to_csv``/``read_csv``/
    # ``to_pickle`` are arbitrary host file read/write. A numerics sandbox has
    # no legitimate use for any of them.
    "read_pickle",
    "to_pickle",
    "allow_pickle",
    "read_csv(",
    "to_csv(",
    "__",
)

# Upper bound on raw source size (characters). The line cap alone is not
# enough: a single pathological multi-megabyte line would otherwise pass it.
_MAX_CODE_CHARS = 100_000

# Namespace keys injected ``data`` may never occupy: the builtins barrier and
# module aliases are sandbox escapes, and the result slots must stay empty so
# generated code that never assigns ``result`` is not credited with a stale
# injected value from the caller's data.
_RESERVED_NAMESPACE_KEYS = ("__builtins__", "np", "pd", "result", "result_var")


def _sandbox_namespace(data: dict | None) -> dict:
    """Build the restricted globals dict for ``exec``.

    The namespace contains the safe builtins allowlist under
    ``__builtins__`` (augmented with a guarded ``__import__``), ``numpy`` as
    ``np`` and ``pandas`` as ``pd``, and every entry of ``data`` as a
    top-level variable name.
    """
    builtins = dict(_SAFE_BUILTINS)
    builtins["__import__"] = _make_guarded_import()
    namespace: dict[str, Any] = {
        "__builtins__": builtins,
        "np": np,
        "pd": pd,
    }
    if data:
        for key, value in data.items():
            # Never let injected data clobber the builtins barrier, the
            # canonical module aliases, or the result slots — that would be a
            # sandbox escape (or a stale value masquerading as the answer).
            if key in _RESERVED_NAMESPACE_KEYS:
                continue
            namespace[key] = value
    return namespace


@dataclass
class PoTResult:
    """Outcome of a sandboxed PoT execution.

    Attributes:
        ok: True iff the code ran to completion AND a result value was
            extracted without exception.
        result: The value of the variable named ``result`` (or
            ``result_var``) from the sandbox namespace, or ``None`` if it
            could not be found.
        stdout: Captured ``print`` output produced by the code.
        error: Exception traceback/message when ``ok`` is False, else None.
        code_ran: True if ``exec`` completed (even if result extraction
            failed afterwards). Distinguishes "code crashed" from "code ran
            but did not assign ``result``".
    """

    ok: bool
    result: Any | None
    stdout: str
    error: str | None
    code_ran: bool


class _TimeoutError(Exception):
    """Internal: raised when the watchdog trips the execution deadline."""


class PoTExecutor:
    """Run LLM-generated Python in a restricted namespace and return the result.

    Parameters:
        timeout_seconds: Best-effort wall-clock cap on supported hosts. On a
            POSIX main thread a hard ``SIGALRM`` is used. Windows calls are
            rejected by :class:`PoTEnableSwitch` before execution. A value of
            ``None``/``0``/negative is clamped to a minimal 1-second watchdog
            rather than disabling the deadline entirely.
        max_lines: Maximum number of lines of source code accepted. Longer
            inputs are rejected before execution.
    """

    _MIN_TIMEOUT_SECONDS = 1.0

    def __init__(self, timeout_seconds: float = 10.0, max_lines: int = 200) -> None:
        # A non-positive or absent timeout would run LLM-generated code with
        # no watchdog at all; clamp to a minimal positive deadline instead.
        if timeout_seconds is None or timeout_seconds <= 0:
            timeout_seconds = self._MIN_TIMEOUT_SECONDS
        self.timeout_seconds = timeout_seconds
        self.max_lines = max_lines

    # -- public API ---------------------------------------------------------

    def run_sandboxed(
        self,
        code: str,
        data: dict | None = None,
        result_var: str | None = None,
    ) -> PoTResult:
        """Execute ``code`` with ``data`` injected as variables, sandboxed.

        See the module docstring for the security model and the result
        extraction convention (the LLM must assign its answer to ``result``
        or to ``result_var``).
        """
        # 0. Runtime opt-in. This is deliberately checked per call so PoT can
        # be disabled after an executor/analyzer has already been constructed.
        if not PoTEnableSwitch.is_enabled():
            return PoTResult(
                ok=False,
                result=None,
                stdout="",
                error=f"PoTExecutor disabled (fail-closed). {PoTEnableSwitch.reason()}",
                code_ran=False,
            )

        # 1. Validate input.
        if not isinstance(code, str) or not code.strip():
            return PoTResult(
                ok=False,
                result=None,
                stdout="",
                error="PoTExecutor: empty code string.",
                code_ran=False,
            )

        line_count = code.count("\n") + 1
        if line_count > self.max_lines:
            return PoTResult(
                ok=False,
                result=None,
                stdout="",
                error=(
                    f"PoTExecutor: code exceeds max_lines "
                    f"({line_count} > {self.max_lines})."
                ),
                code_ran=False,
            )

        if len(code) > _MAX_CODE_CHARS:
            return PoTResult(
                ok=False,
                result=None,
                stdout="",
                error=(
                    f"PoTExecutor: code exceeds max characters "
                    f"({len(code)} > {_MAX_CODE_CHARS})."
                ),
                code_ran=False,
            )

        # 2. Token guard (defense-in-depth on the user code string).
        guard_hit = _scan_for_dangerous_tokens(code)
        if guard_hit is not None:
            return PoTResult(
                ok=False,
                result=None,
                stdout="",
                error=(
                    f"PoTExecutor: code rejected by security token guard "
                    f"(blocked token: {guard_hit!r})."
                ),
                code_ran=False,
            )

        # 3. Build the restricted namespace.
        sandbox_globals = _sandbox_namespace(data)

        # 4. Execute with stdout capture and a best-effort timeout.
        stdout_buf = io.StringIO()
        code_ran = False
        run_error: str | None = None

        with contextlib.redirect_stdout(stdout_buf):

            def _actually_exec() -> None:
                nonlocal code_ran, run_error
                try:
                    exec(code, sandbox_globals)  # noqa: S102 - intentional restricted exec
                    code_ran = True
                except _TimeoutError:
                    # POSIX SIGALRM fires the timeout INSIDE the exec block, so
                    # without this branch the interrupt is captured below as a
                    # raw traceback of this private exception — instead of the
                    # normalized message the thread-watchdog path produces.
                    run_error = (
                        f"PoTExecutor: execution timed out after "
                        f"{self.timeout_seconds}s."
                    )
                except BaseException as exc:  # noqa: BLE001 - report any failure
                    run_error = "".join(
                        traceback.format_exception(type(exc), exc, exc.__traceback__)
                    )

            try:
                self._run_with_timeout(_actually_exec)
            except _TimeoutError:
                run_error = (
                    f"PoTExecutor: execution timed out after "
                    f"{self.timeout_seconds}s."
                )

        stdout = stdout_buf.getvalue()

        if not code_ran:
            return PoTResult(
                ok=False,
                result=None,
                stdout=stdout,
                error=run_error,
                code_ran=False,
            )

        # 5. Extract the result by name from the namespace.
        result_value: Any | None = None
        result_found = False
        if result_var is not None:
            if result_var in sandbox_globals:
                result_value = sandbox_globals[result_var]
                result_found = True
            else:
                return PoTResult(
                    ok=False,
                    result=None,
                    stdout=stdout,
                    error=(
                        f"PoTExecutor: result_var {result_var!r} not found "
                        f"in sandbox namespace."
                    ),
                    code_ran=True,
                )
        elif "result" in sandbox_globals:
            result_value = sandbox_globals["result"]
            result_found = True

        if not result_found:
            return PoTResult(
                ok=False,
                result=None,
                stdout=stdout,
                error=(
                    "PoTExecutor: no result extracted. The LLM must assign "
                    "its final answer to a variable named 'result' (or to "
                    "the provided result_var)."
                ),
                code_ran=True,
            )

        return PoTResult(
            ok=True,
            result=result_value,
            stdout=stdout,
            error=None,
            code_ran=True,
        )

    # -- timeout plumbing ---------------------------------------------------

    def _run_with_timeout(self, func: Any) -> None:
        """Run ``func`` with a best-effort timeout, platform-aware.

        POSIX main thread: ``signal.SIGALRM`` delivers a hard interrupt.
        Other supported contexts use a bounded daemon-thread fallback. Windows
        never reaches this method through :meth:`run_sandboxed`; the runtime
        policy rejects host execution before any worker is started.
        """
        if self.timeout_seconds is None or self.timeout_seconds <= 0:
            func()
            return

        # POSIX path: real signal-based timeout.
        can_use_signal = (
            hasattr(signal, "SIGALRM")
            and platform.system() != "Windows"
            and threading.current_thread() is threading.main_thread()
        )
        if can_use_signal:
            self._run_with_signal_timeout(func)
            return

        # Windows / non-main-thread path: thread-based watchdog.
        self._run_with_thread_timeout(func)

    def _run_with_signal_timeout(self, func: Any) -> None:
        previous_handler = signal.getsignal(signal.SIGALRM)  # type: ignore[attr-defined]

        def _handler(signum: int, frame: Any) -> None:  # noqa: ARG001
            raise _TimeoutError()

        try:
            signal.signal(signal.SIGALRM, _handler)  # type: ignore[attr-defined]
            signal.setitimer(signal.ITIMER_REAL, float(self.timeout_seconds))  # type: ignore[attr-defined]
            try:
                func()
            finally:
                signal.setitimer(signal.ITIMER_REAL, 0)  # type: ignore[attr-defined]
        finally:
            signal.signal(signal.SIGALRM, previous_handler)  # type: ignore[attr-defined]

    def _run_with_thread_timeout(self, func: Any) -> None:
        worker_exc: list[BaseException] = []

        def _worker() -> None:
            try:
                func()
            except BaseException as exc:  # noqa: BLE001
                worker_exc.append(exc)

        thread = threading.Thread(target=_worker, daemon=True)
        thread.start()
        # Never join without a deadline: on Windows the worker cannot be
        # forcibly interrupted, so an unbounded join defeated the watchdog and
        # hung the caller forever. The daemon may finish later, but this call
        # returns promptly and reports a timeout.
        thread.join(timeout=float(self.timeout_seconds))

        if thread.is_alive():
            raise _TimeoutError()

        if worker_exc:
            raise worker_exc[0]


def _scan_for_dangerous_tokens(code: str) -> str | None:
    """Return the first dangerous token found in ``code``, or None.

    Operates on the raw source string (case-sensitive for the dunder rule,
    case-insensitive for the import/keyword rules). The bare ``__`` rule is
    the broadest layer: dunder access is the standard escape hatch from a
    restricted-exec sandbox, so any occurrence is rejected.
    """
    lowered = code.lower()
    for token in _DANGEROUS_TOKENS:
        if token == "__":
            if "__" in code:
                return token
        elif token.lower() in lowered:
            return token
    return None
