"""run_robust.py watchdog resilience: a per-ticker exception must NOT crash the
whole batch, and ``_reap`` must absorb a ``TimeoutExpired`` instead of letting
it escape the watchdog loop.

These load the script as an isolated module (importlib) so its ``__main__``
guard never fires, and never spawn a real subprocess.
"""
import contextlib
import importlib.util
import io
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

_SCRIPT_PATH = (
    Path(__file__).resolve().parent.parent / "scripts" / "run_robust.py"
)
_spec = importlib.util.spec_from_file_location("run_robust_under_test", _SCRIPT_PATH)
assert _spec is not None and _spec.loader is not None
rr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rr)


class TestCoreSentinelCount(unittest.TestCase):
    """The DEGRADED verdict source: full_states_log's data_quality block."""

    def _write_log(self, states_root: Path, ticker: str, date: str, payload) -> Path:
        # Real layout: full_states_log lives under the CHILD's results dir
        # (~/.yialpha/logs/<TICKER>/...), NEXT TO (not inside) reports/.
        log_dir = states_root / ticker / "YiAlphaStrategy_logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log = log_dir / f"full_states_log_{date}.json"
        import json as _json

        log.write_text(_json.dumps(payload), encoding="utf-8")
        return log

    def test_reads_core_sentinel_count(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "reports"
            root.mkdir()
            self._write_log(
                root.parent, "NVDA", "2026-06-10",
                {"data_quality": {"core_sentinel_count": 3,
                                  "optional_sentinel_count": 1, "sentinels": []}},
            )
            self.assertEqual(
                rr._core_sentinel_count(root.parent, "NVDA", "2026-06-10"), 3
            )

    def test_states_root_independent_of_reports_root(self):
        # An operator overriding --reports-root to a copied directory must
        # not decouple the DEGRADED evidence: the count is read from the
        # child's ACTUAL results dir (states_root), never from
        # reports_root.parent — under the old anchoring strict mode burned
        # every retry on phantom "evidence unavailable" DEGRADED verdicts.
        with tempfile.TemporaryDirectory() as td_a, \
                tempfile.TemporaryDirectory() as td_b:
            copied_reports = Path(td_a) / "reports"  # offline copy, no logs
            copied_reports.mkdir()
            states = Path(td_b)
            self._write_log(
                states, "NVDA", "2026-06-10",
                {"data_quality": {"core_sentinel_count": 1}},
            )
            self.assertEqual(
                rr._core_sentinel_count(states, "NVDA", "2026-06-10"), 1
            )
            # And the copied tree reads as unknown, not zero, when alone.
            self.assertIsNone(
                rr._core_sentinel_count(copied_reports.parent, "NVDA", "2026-06-10")
            )

    def test_missing_log_returns_none(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "reports"
            root.mkdir()
            self.assertIsNone(
                rr._core_sentinel_count(root.parent, "NVDA", "2026-06-10")
            )

    def test_legacy_log_without_block_returns_none(self):
        # Older runs (pre-data_quality) must read as UNKNOWN, not zero —
        # "0 sentinels" would wrongly certify a data-vacuum report as clean.
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "reports"
            root.mkdir()
            self._write_log(root.parent, "AAPL", "2026-01-05", {"final_trade_decision": "x"})
            self.assertIsNone(
                rr._core_sentinel_count(root.parent, "AAPL", "2026-01-05")
            )


class TestStatesRootDotenv(unittest.TestCase):
    """Round-11 footgun: the child resolves its results dir through
    ``load_dotenv(usecwd=True)`` with cwd=_PROJECT_ROOT, so an .env-only
    ``YIALPHA_RESULTS_DIR`` is what the child WRITES — while the parent used
    to read the ``~/.yialpha/logs`` default and never find the evidence.

    dotenv_values is MOCKED (no real .env file is written) to keep these pins
    order-independent — no later full-suite test that touches a real ``.env``
    can perturb them; file parsing itself is python-dotenv's contract, both
    sides use the same lib on the same file in production."""

    def test_env_var_beats_dotenv(self):
        with mock.patch("dotenv.dotenv_values", return_value={
            "YIALPHA_RESULTS_DIR": "C:/from/dotenv",
        }) as mdv, mock.patch.dict(rr.os.environ,
                                   {"YIALPHA_RESULTS_DIR": "C:/from/env"}):
            self.assertEqual(rr._states_root(), Path("C:/from/env"))
        # An exported var must not even consult the .env file.
        mdv.assert_not_called()

    def test_dotenv_only_value_resolves_for_parent(self):
        env = {k: v for k, v in rr.os.environ.items()
               if k != "YIALPHA_RESULTS_DIR"}
        with mock.patch("dotenv.dotenv_values", return_value={
            "YIALPHA_RESULTS_DIR": "C:/from/dotenv",
        }), mock.patch.dict(rr.os.environ, env, clear=True):
            self.assertEqual(rr._states_root(), Path("C:/from/dotenv"))

    def test_dotenv_without_key_falls_back_to_home_default(self):
        # Covers every no-value shape with one pin: .env missing, .env with
        # no YIALPHA_RESULTS_DIR key — dotenv_values yields nothing usable
        # either way.
        env = {k: v for k, v in rr.os.environ.items()
               if k != "YIALPHA_RESULTS_DIR"}
        with mock.patch("dotenv.dotenv_values", return_value={}), \
                mock.patch.dict(rr.os.environ, env, clear=True):
            self.assertEqual(
                rr._states_root(),
                Path.home() / ".yialpha" / "logs",
            )

    def test_dotenv_relative_value_anchors_to_project_root(self):
        """Round-11 remediation (C3/D2): a RELATIVE ``YIALPHA_RESULTS_DIR``
        from ``.env`` anchors to the PROJECT ROOT — the resolution-equivalent
        of the child's ``load_dotenv(usecwd=True)`` with cwd=_PROJECT_ROOT.
        The old parent resolved it against its OWN cwd, so launching
        run_robust from another directory read quality evidence from a
        different tree than the child wrote. The pin must hold no matter
        which cwd this process happens to run under."""
        env = {k: v for k, v in rr.os.environ.items()
               if k != "YIALPHA_RESULTS_DIR"}
        for cwd in (str(rr._PROJECT_ROOT), str(Path(rr._PROJECT_ROOT).parent)):
            with mock.patch("dotenv.dotenv_values", return_value={
                "YIALPHA_RESULTS_DIR": "logs",
            }), mock.patch.dict(rr.os.environ, env, clear=True), \
                    mock.patch.object(rr.os, "getcwd", return_value=cwd), \
                    mock.patch("os.getcwd", return_value=cwd):
                self.assertEqual(
                    rr._states_root(), (rr._PROJECT_ROOT / "logs").resolve(),
                    f"anchor must not depend on cwd ({cwd})",
                )

    def test_exported_relative_value_passes_through_verbatim(self):
        # An EXPORTED relative value is the operator's own resolution
        # semantics — the parent must not second-guess it (only .env-derived
        # values anchor to the project root).
        env = dict(rr.os.environ)
        env["YIALPHA_RESULTS_DIR"] = "logs"
        with mock.patch.dict(rr.os.environ, env, clear=True):
            self.assertEqual(rr._states_root(), Path("logs"))


class TestKillTreeReturn(unittest.TestCase):
    """Round-11 footgun: _kill_tree reports landed-vs-already-gone so the
    verdict attributes the attempt honestly (taskkill 128 / ESRCH = the child
    exited on its own in the poll→kill window; indeterminate failures stay
    "landed" — never mistake a real kill for a natural exit)."""

    def _windows_run(self, rc: int) -> bool:
        completed = subprocess.CompletedProcess(args=[], returncode=rc)
        with mock.patch.object(rr, "IS_WINDOWS", True), \
                mock.patch.object(
                    rr.subprocess, "run", return_value=completed) as mrun:
            landed = rr._kill_tree(4242)
        mrun.assert_called_once()
        self.assertEqual(
            mrun.call_args.args[0][:2], ["taskkill", "/F"]
        )
        return landed

    def test_windows_rc0_is_landed(self):
        self.assertTrue(self._windows_run(0))

    def test_windows_rc128_is_already_gone(self):
        self.assertFalse(self._windows_run(128))

    def test_windows_other_rc_conservatively_landed(self):
        # Access denied etc. — indeterminate: claim landed so the verdict
        # never evaluates a possibly half-killed attempt as finished.
        self.assertTrue(self._windows_run(1))

    def test_posix_esrch_is_already_gone(self):
        # getpgid/killpg are POSIX-only attributes (absent on Windows os),
        # hence create=True.
        with mock.patch.object(rr, "IS_WINDOWS", False), \
                mock.patch.object(
                    rr.os, "getpgid",
                    side_effect=ProcessLookupError(), create=True), \
                mock.patch.object(rr.os, "killpg", create=True) as mkill:
            self.assertFalse(rr._kill_tree(4242))
        mkill.assert_not_called()

    def test_posix_success_is_landed(self):
        with mock.patch.object(rr, "IS_WINDOWS", False), \
                mock.patch.object(
                    rr.os, "getpgid", return_value=99, create=True), \
                mock.patch.object(rr.os, "killpg", create=True) as mkill:
            self.assertTrue(rr._kill_tree(4242))
        mkill.assert_called_once_with(99, 9)

    def test_posix_eperm_conservatively_landed(self):
        with mock.patch.object(rr, "IS_WINDOWS", False), \
                mock.patch.object(
                    rr.os, "getpgid", return_value=99, create=True), \
                mock.patch.object(
                    rr.os, "killpg",
                    side_effect=PermissionError(), create=True):
            self.assertTrue(rr._kill_tree(4242))


class TestArgValidation(unittest.TestCase):
    """Bad numeric flags exit 2 with a usage error, not a bare traceback
    (--workers 0 used to escape as ValueError from ThreadPoolExecutor, and
    --max-attempts 0 failed every ticker with an empty reason)."""

    def _expect_exit_2(self, extra_args):
        argv = ["run_robust.py", "--tickers", "AAA", "--date", "2026-07-01",
                *extra_args]
        buf = io.StringIO()
        with mock.patch("sys.argv", argv), \
                contextlib.redirect_stderr(buf), self.assertRaises(SystemExit) as cm:
            rr._parse_args()
        self.assertEqual(cm.exception.code, 2)

    def test_workers_zero_rejected(self):
        self._expect_exit_2(["--workers", "0"])

    def test_max_attempts_zero_rejected(self):
        self._expect_exit_2(["--max-attempts", "0"])

    def test_negative_timeout_rejected(self):
        self._expect_exit_2(["--per-ticker-timeout", "0"])

    def test_negative_backoff_rejected(self):
        self._expect_exit_2(["--backoff", "-1"])

    def _parse_with_date(self, bad_date: str):
        argv = ["run_robust.py", "--tickers", "AAA", "--date", bad_date]
        buf = io.StringIO()
        with mock.patch("sys.argv", argv), \
                contextlib.redirect_stderr(buf), self.assertRaises(SystemExit) as cm:
            rr._parse_args()
        return cm.exception.code, buf.getvalue()

    def test_slash_date_rejected(self):
        # Round-10: a garbage date (2026/07/01) used to propagate a malformed
        # as-of through the whole graph before failing mid-run; it now dies
        # at the F5 validation site with the same exit 2 as the numeric
        # guards.
        code, err = self._parse_with_date("2026/07/01")
        self.assertEqual(code, 2)
        self.assertIn("YYYY-MM-DD", err)

    def test_non_padded_date_rejected(self):
        # Bare strptime accepts "2026-7-1"; the round-trip check enforces
        # zero-padded ISO (downstream as-of parsing / log globbing expect it).
        code, _ = self._parse_with_date("2026-7-1")
        self.assertEqual(code, 2)

    def test_valid_date_accepted(self):
        argv = ["run_robust.py", "--tickers", "AAA", "--date", "2026-07-01"]
        with mock.patch("sys.argv", argv):
            opts = rr._parse_args()
        self.assertEqual(opts.date, "2026-07-01")


class _FakeProc:
    """Popen stand-in for the watchdog-loop race tests."""

    def __init__(self, exit_code_after_kill: int):
        self.returncode = None
        self._exit_code = exit_code_after_kill
        self.pid = 424242

    def poll(self):
        # Alive at every watchdog poll — the loop must take the kill branch.
        return None

    def wait(self, timeout=None):
        # _reap's wait reaps the child: its true exit code surfaces here.
        self.returncode = self._exit_code
        return self._exit_code

    def kill(self):
        self.returncode = self._exit_code


class TestKillRaceAndReason(unittest.TestCase):
    """The kill branch must (a) recognize a child that beat the kill and
    evaluate it normally, and (b) record the kill reason in result["reason"]
    for the summary table (it used to print only, leaving "" or the previous
    attempt's stale reason in the row)."""

    def _run_one(self, exit_code_after_kill: int,
                 kill_landed: bool = True) -> dict:
        """Shared watchdog-loop harness: child alive at every poll, the wall
        clock trips the timeout, and ``_kill_tree`` reports landed (default —
        a plain MagicMock's truthy return) or already-gone (``False``:
        taskkill 128 / ESRCH, the child beat the kill out the door)."""
        argv = [
            "run_robust.py",
            "--tickers", "AAA",
            "--date", "2026-07-01",
            "--workers", "1",
            "--per-ticker-timeout", "1800",
            "--max-attempts", "1",
        ]
        with tempfile.TemporaryDirectory() as td:
            argv += ["--reports-root", str(Path(td) / "reports")]
            buf = io.StringIO()
            clock = iter([1000.0, 4000.0, 4005.0])

            proc = _FakeProc(exit_code_after_kill)
            kill_patch = (
                mock.patch.object(rr, "_kill_tree")
                if kill_landed
                else mock.patch.object(rr, "_kill_tree", return_value=False)
            )
            with mock.patch("sys.argv", argv), \
                    mock.patch.object(rr.subprocess, "Popen", return_value=proc), \
                    mock.patch.object(rr, "_register_proc"), \
                    mock.patch.object(rr, "_unregister_proc"), \
                    kill_patch, \
                    mock.patch.object(rr, "_reap", side_effect=lambda p, _sig, **k: p.wait()), \
                    mock.patch.object(rr.time, "time", side_effect=lambda: next(clock, 4010.0)), \
                    mock.patch.object(rr.time, "sleep"), \
                    mock.patch.object(rr, "_find_new_report", return_value=Path("fake.md")), \
                    mock.patch.object(rr, "_core_sentinel_count", return_value=0), \
                    contextlib.redirect_stdout(buf), \
                    contextlib.redirect_stderr(buf):
                result = rr._run_one_ticker("AAA", "2026-07-01",
                                            rr._parse_args())
        return result

    def test_kill_raced_with_clean_exit_is_success(self):
        # Child completed (rc=0, report on disk) just as the watchdog fired:
        # the attempt must be evaluated normally, not burned as a kill.
        result = self._run_one(exit_code_after_kill=0)
        self.assertTrue(result["ok"])
        self.assertIn("ok", result["reason"])

    def test_real_kill_records_reason(self):
        result = self._run_one(exit_code_after_kill=1)
        self.assertFalse(result["ok"])
        self.assertIn("killed", result["reason"])
        self.assertIn("wall_clock", result["reason"])

    def test_kill_not_landed_natural_crash_reported_as_exit(self):
        # Round-11: a child that CRASHED naturally (rc=1) inside the
        # poll→kill window used to be misreported as "killed: wall_clock";
        # with _kill_tree reporting already-gone, it flows to the exit-N
        # branch with its own code.
        result = self._run_one(exit_code_after_kill=1, kill_landed=False)
        self.assertFalse(result["ok"])
        self.assertIn("exit 1", result["reason"])
        self.assertNotIn("killed", result["reason"])

    def test_kill_not_landed_clean_exit_is_success(self):
        # Already-gone + rc=0 + report on disk: finished attempt, evaluated
        # normally (previously relied on the rc==0 heuristic alone).
        result = self._run_one(exit_code_after_kill=0, kill_landed=False)
        self.assertTrue(result["ok"])
        self.assertIn("ok", result["reason"])



    def test_clean_exit_does_not_force_kill(self):
        killed = {"n": 0}

        class FakeProc:
            def wait(self, timeout=None):
                return 0

            def kill(self):
                killed["n"] += 1

        rr._reap(FakeProc(), "TCK")
        self.assertEqual(killed["n"], 0)

    def test_timeout_expired_then_force_killed(self):
        # proc.wait(timeout=30) raises TimeoutExpired -> _reap must escalate to
        # proc.kill() + unconditional wait(), and NOT re-raise. Without this,
        # the exception escaped _run_one_ticker -> fut.result() -> main() and
        # crashed the whole batch.
        events = {"kill": 0, "reap_wait": 0}

        class FakeProc:
            def wait(self, timeout=None):
                if timeout is not None:
                    raise subprocess.TimeoutExpired(cmd="child", timeout=timeout)
                events["reap_wait"] += 1
                return 0

            def kill(self):
                events["kill"] += 1

        rr._reap(FakeProc(), "TCK")
        self.assertEqual(events["kill"], 1)
        self.assertEqual(events["reap_wait"], 1)


class TestRobustLLMCacheEnv(unittest.TestCase):
    """The robust child subprocess gets YIALPHA_LLM_CACHE set so retries
    replay completed nodes (hang recovery) instead of re-billing the graph."""

    def test_default_enables_cache(self):
        env = {}
        rr._apply_robust_llm_cache(env, no_llm_cache=False)
        self.assertEqual(env.get("YIALPHA_LLM_CACHE"), "true")

    def test_no_llm_cache_forces_off(self):
        env = {"YIALPHA_LLM_CACHE": "true"}  # user pre-set on
        rr._apply_robust_llm_cache(env, no_llm_cache=True)
        self.assertEqual(env.get("YIALPHA_LLM_CACHE"), "false")

    def test_respects_user_preset_off(self):
        # A user who exported YIALPHA_LLM_CACHE=false is respected (setdefault).
        env = {"YIALPHA_LLM_CACHE": "false"}
        rr._apply_robust_llm_cache(env, no_llm_cache=False)
        self.assertEqual(env.get("YIALPHA_LLM_CACHE"), "false")

    def test_respects_user_preset_on(self):
        env = {"YIALPHA_LLM_CACHE": "true"}
        rr._apply_robust_llm_cache(env, no_llm_cache=False)
        self.assertEqual(env.get("YIALPHA_LLM_CACHE"), "true")


class TestMainResilience(unittest.TestCase):
    def test_per_ticker_exception_does_not_crash_batch(self):
        # If _run_one_ticker raises (any unexpected error), main() must catch
        # it via the fut.result() guard, record a failure dict, and keep the
        # other tickers' results — not crash the orchestrator.
        def _boom(*a, **k):
            raise RuntimeError("simulated watchdog explosion")

        tmp = tempfile.mkdtemp()
        argv = [
            "run_robust.py",
            "--tickers", "AAA", "BBB",
            "--date", "2026-07-01",
            "--workers", "2",
            "--reports-root", tmp,
        ]
        buf = io.StringIO()
        with mock.patch.object(rr, "_run_one_ticker", _boom), \
                mock.patch("sys.argv", argv), \
                contextlib.redirect_stdout(buf), \
                contextlib.redirect_stderr(buf):
            rc = rr.main()
        # Both tickers failed -> rc 1, but main returned (did not propagate).
        self.assertEqual(rc, 1)
        out = buf.getvalue()
        # The orchestrator logged each per-ticker failure rather than crashing.
        self.assertIn("orchestrator error", out)
        self.assertIn("AAA", out)
        self.assertIn("BBB", out)


if __name__ == "__main__":
    unittest.main()
