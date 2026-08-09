"""Unit tests for the Program-of-Thoughts sandbox executor.

No network, no API keys. Covers arithmetic correctness, data injection, stdout
capture, the security token guard, runtime-error handling, the result_var
override, the oversized-code rejection, and that dangerous tokens are blocked.
"""

import threading
import time

import pytest

import yiagents.agents.utils.pot_executor as pot_executor_module
from yiagents.agents.utils.pot_executor import PoTEnableSwitch, PoTExecutor, PoTResult


@pytest.fixture(autouse=True)
def _explicitly_enable_pot_for_executor_tests(monkeypatch):
    # Production/default behavior is covered separately below. Arithmetic and
    # guard tests need an explicit opt-in to reach the executor.
    monkeypatch.setenv("YIAGENTS_POT_ENABLED", "true")
    monkeypatch.setenv("YIAGENTS_ANALYSIS_ONLY", "false")
    # Production Windows is intentionally blocked until PoT is process-isolated.
    # Unit tests exercise the restricted executor and portable timeout plumbing.
    monkeypatch.setattr(pot_executor_module.platform, "system", lambda: "Linux")


@pytest.mark.unit
class TestPoTExecutor:
    def _exec(self, **kwargs):
        return PoTExecutor().run_sandboxed(**kwargs)

    def test_runtime_switch_off_rejects_without_running(self, monkeypatch):
        monkeypatch.setenv("YIAGENTS_POT_ENABLED", "false")
        res = self._exec(code="result = 42")
        assert PoTEnableSwitch.is_enabled() is False
        assert res.ok is False
        assert res.code_ran is False
        assert "disabled" in (res.error or "").lower()

    def test_analysis_only_cannot_be_overridden_by_pot_flag(self, monkeypatch):
        monkeypatch.setenv("YIAGENTS_ANALYSIS_ONLY", "true")
        res = self._exec(code="result = 42")
        assert PoTEnableSwitch.is_enabled() is False
        assert res.ok is False
        assert res.code_ran is False

    def test_switch_is_rechecked_after_executor_construction(self, monkeypatch):
        executor = PoTExecutor()
        assert executor.run_sandboxed(code="result = 1").ok is True
        monkeypatch.setenv("YIAGENTS_ANALYSIS_ONLY", "true")
        blocked = executor.run_sandboxed(code="result = 2")
        assert blocked.ok is False
        assert blocked.code_ran is False

    def test_windows_host_exec_is_refused(self, monkeypatch):
        monkeypatch.setattr(pot_executor_module.platform, "system", lambda: "Windows")
        res = self._exec(code="result = 42")
        assert res.ok is False
        assert res.code_ran is False
        assert "windows" in (res.error or "").lower()

    def test_malformed_runtime_switch_fails_closed(self, monkeypatch):
        monkeypatch.setenv("YIAGENTS_POT_ENABLED", "tru")
        res = self._exec(code="result = 42")
        assert res.ok is False
        assert res.code_ran is False

    # 1. Simple arithmetic.
    def test_simple_arithmetic(self):
        res = self._exec(code="result = (105.0/100.0 - 1) * 100", data=None)
        assert res.ok is True
        assert res.result == pytest.approx(5.0)
        assert res.code_ran is True
        assert res.error is None

    # 2. Data injection with numpy exposed as np.
    def test_data_injection_with_numpy(self):
        code = "ret = np.diff(prices)/prices[:-1]\nresult = float(ret.mean())"
        res = self._exec(code=code, data={"prices": [100, 105, 103, 108]})
        assert res.ok is True
        # diffs: [5, -2, 5]; returns: [0.05, -0.0190476, 0.0485437]
        # mean ~= 0.026499
        assert res.result == pytest.approx(0.026499, abs=1e-4)

    # 3. Print capture.
    def test_print_capture(self):
        res = self._exec(code="print('hello')\nresult = 42")
        assert res.ok is True
        assert "hello" in res.stdout
        assert res.result == 42

    # 4. Security: dangerous imports blocked by the token guard.
    def test_security_blocks_import_os(self):
        res = self._exec(code="import os\nos.system('ls')")
        assert res.ok is False
        assert res.code_ran is False
        assert res.error is not None
        assert "reject" in res.error.lower() or "block" in res.error.lower()

    def test_security_blocks_open(self):
        res = self._exec(code="open('/etc/passwd')")
        assert res.ok is False
        assert res.code_ran is False
        assert res.error is not None
        assert "reject" in res.error.lower() or "block" in res.error.lower()

    def test_security_blocks_dunder_escape(self):
        # Dunder access is the classic restricted-exec escape hatch.
        res = self._exec(code="result = ().__class__.__bases__")
        assert res.ok is False
        assert res.code_ran is False

    # 5. Runtime error propagates as ok=False, no host crash.
    def test_division_by_zero(self):
        res = self._exec(code="result = 1 / 0")
        assert res.ok is False
        assert res.code_ran is False
        assert res.error is not None
        assert "ZeroDivisionError" in res.error or "division" in res.error.lower()

    def test_missing_result_variable(self):
        res = self._exec(code="x = 5")
        assert res.ok is False
        assert res.code_ran is True
        assert res.error is not None
        assert "result" in res.error.lower()

    # 6. result_var override.
    def test_result_var_override(self):
        res = self._exec(code="answer = 7", result_var="answer")
        assert res.ok is True
        assert res.result == 7

    def test_result_var_not_found(self):
        res = self._exec(code="answer = 7", result_var="missing")
        assert res.ok is False
        assert res.code_ran is True
        assert res.error is not None
        assert "missing" in res.error

    # 7. Oversized code rejected.
    def test_oversized_code_rejected(self):
        executor = PoTExecutor(max_lines=5)
        code = "\n".join(f"x{i} = {i}" for i in range(10))
        res = executor.run_sandboxed(code=code)
        assert res.ok is False
        assert res.code_ran is False
        assert res.error is not None
        assert "max_lines" in res.error

    def test_empty_code_rejected(self):
        res = self._exec(code="")
        assert res.ok is False
        assert res.code_ran is False
        assert res.error is not None

    # Sanity: the restricted builtins actually remove __import__.
    def test_import_inside_sandbox_fails(self):
        # `import` would need __import__; even if it slipped the token guard,
        # the restricted builtins block it. Use a phrased import that the
        # token guard does NOT catch by substring to exercise the builtins
        # barrier directly.
        res = self._exec(code="import math\nresult = math.pi")
        # "import math" is not in _DANGEROUS_TOKENS, so this reaches exec and
        # then fails because __import__ is absent from the sandbox builtins.
        assert res.ok is False
        assert res.code_ran is False

    # DataFrame injection via pandas works end-to-end.
    def test_pandas_dataframe_injection(self):
        code = "result = float(pd.Series(prices).pct_change().dropna().mean())"
        res = self._exec(code=code, data={"prices": [100, 110, 99]})
        assert res.ok is True
        # pct changes: [0.1, -0.1]; mean = 0.0
        assert res.result == pytest.approx(0.0, abs=1e-9)

    def test_result_dataclass_fields(self):
        res = self._exec(code="result = 1")
        # Confirm the documented fields all exist with the right types.
        assert isinstance(res, PoTResult)
        assert isinstance(res.ok, bool)
        assert isinstance(res.code_ran, bool)
        assert isinstance(res.stdout, str)
        # error is None on success
        assert res.error is None

    def test_thread_timeout_returns_without_waiting_for_worker(self):
        blocker = threading.Event()
        executor = PoTExecutor(timeout_seconds=0.05)
        started = time.monotonic()
        res = executor.run_sandboxed(
            code="blocker.wait(2)\nresult = 1",
            data={"blocker": blocker},
        )
        elapsed = time.monotonic() - started
        blocker.set()  # let the daemon worker finish promptly after the assertion
        assert res.ok is False
        assert "timed out" in (res.error or "").lower()
        assert elapsed < 0.5
