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

    def _write_log(self, reports_root: Path, ticker: str, date: str, payload) -> Path:
        # Real layout: full_states_log lives NEXT TO the reports/ dir
        # (~/.yialpha/logs/<TICKER>/...), i.e. under reports_root.parent.
        log_dir = reports_root.parent / ticker / "YiAlphaStrategy_logs"
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
                root, "NVDA", "2026-06-10",
                {"data_quality": {"core_sentinel_count": 3,
                                  "optional_sentinel_count": 1, "sentinels": []}},
            )
            self.assertEqual(
                rr._core_sentinel_count(root, "NVDA", "2026-06-10"), 3
            )

    def test_missing_log_returns_none(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "reports"
            root.mkdir()
            self.assertIsNone(
                rr._core_sentinel_count(root, "NVDA", "2026-06-10")
            )

    def test_legacy_log_without_block_returns_none(self):
        # Older runs (pre-data_quality) must read as UNKNOWN, not zero —
        # "0 sentinels" would wrongly certify a data-vacuum report as clean.
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "reports"
            root.mkdir()
            self._write_log(root, "AAPL", "2026-01-05", {"final_trade_decision": "x"})
            self.assertIsNone(
                rr._core_sentinel_count(root, "AAPL", "2026-01-05")
            )



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
