"""Pins for BatchRunner (yialpha/batch/runner.py) — the per-ticker isolation
and context-propagation seams shared by every batch frontend
(scripts/run_batch.py and `yialpha batch` in yialpha/cli/main.py).

A prior review found the runner clean against the DataVacuumError taxonomy;
these tests pin that finding hermetically (graph construction stubbed via
the public graph_factory seam — same pattern as tests/test_batch_concurrency.py):

* one ticker raising DataVacuumError (the typed "decided with zero core
  data" failure, a RuntimeError subclass) is captured into ITS result dict
  as ``error`` — every other ticker still runs, and the batch returns one
  row per input ticker in input order, never raising;
* a generic Exception (and a save_reports-stage failure) is isolated the
  same way;
* submit_with_context pins the config ContextVar per submission so a
  ThreadPoolExecutor worker sees the submitting context's config (a plain
  executor.submit does NOT — that is the hazard the helper exists for);
* the serial path (_run_one) defensively re-binds the runner's own config
  snapshot, so a runner constructed under one config and invoked after the
  ambient context changed still runs on its own config;
* prepare_batch_run (the single validation contract both frontends map to
  exit code 2) rejects bad tickers/dates/workers and mixed asset classes.
"""

from concurrent.futures import ThreadPoolExecutor

import pytest

from yialpha.batch.runner import BatchInputError, BatchRunner, prepare_batch_run
from yialpha.dataflows.config import get_config, set_config, submit_with_context
from yialpha.dataflows.quality import DataVacuumError
from yialpha.default_config import DEFAULT_CONFIG

_ROW_KEYS = {"ticker", "state", "signal", "report_path", "elapsed", "error"}


# --------------------------------------------------------------------------- #
# Hermetic graph stubs (the graph_factory seam is public API for this)
# --------------------------------------------------------------------------- #


class _FakeGraph:
    """Minimal YiAlphaGraph stand-in: .config for the uniformity guard,
    propagate + save_reports for the run loop."""

    def __init__(self, config):
        self.config = config

    def propagate(self, ticker, trade_date, asset_type="stock"):
        return ({"company_of_interest": ticker}, f"signal:{ticker}")

    def save_reports(self, final_state, ticker):
        return None


class _VacuumGraph(_FakeGraph):
    """Raises the typed vacuum failure for one ticker, succeeds elsewhere."""

    fail_ticker = "VACUUM"

    def propagate(self, ticker, trade_date, asset_type="stock"):
        if ticker == type(self).fail_ticker:
            raise DataVacuumError(
                "data vacuum: zero successful core-category data calls"
            )
        return super().propagate(ticker, trade_date, asset_type=asset_type)


class _CrashGraph(_FakeGraph):
    """Raises a generic RuntimeError for one ticker."""

    fail_ticker = "BOOM"

    def propagate(self, ticker, trade_date, asset_type="stock"):
        if ticker == type(self).fail_ticker:
            raise RuntimeError("LLM stream died mid-response")
        return super().propagate(ticker, trade_date, asset_type=asset_type)


class _SaveCrashGraph(_FakeGraph):
    """propagate succeeds but the report save stage fails."""

    fail_ticker = "SAVEFAIL"

    def save_reports(self, final_state, ticker):
        if ticker == type(self).fail_ticker:
            raise OSError("disk full")
        return None


def _config(tmp_path, **extra):
    """A runner config whose lock roots point at the per-test tmp dir
    (serialized_run requires data_cache_dir and results_dir)."""
    cfg = {
        "data_cache_dir": str(tmp_path / "cache"),
        "results_dir": str(tmp_path / "results"),
    }
    cfg.update(extra)
    return cfg


def _assert_isolation(results, failed_ticker, error_type):
    """One row per input ticker, input order, only the failed row degraded."""
    assert [r["ticker"] for r in results] == ["AAPL", failed_ticker, "NVDA"]
    failed = results[1]
    assert isinstance(failed["error"], error_type)
    assert failed["state"] is None
    assert failed["signal"] is None
    assert failed["report_path"] is None
    assert failed["elapsed"] >= 0.0
    for ok in (results[0], results[2]):
        assert ok["error"] is None
        assert ok["state"] == {"company_of_interest": ok["ticker"]}
        assert ok["signal"] == f"signal:{ok['ticker']}"


# --------------------------------------------------------------------------- #
# DataVacuumError taxonomy (the review finding these tests pin)
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_data_vacuum_error_isolated_serial(tmp_path):
    """Serial batch: the typed vacuum failure for one ticker lands in that
    ticker's result dict; siblings complete; run() never raises. DataVacuumError
    is a RuntimeError subclass, so the generic per-ticker handler catches it —
    frontends then surface it via exit code 1 (error row), not a crash."""
    assert issubclass(DataVacuumError, RuntimeError)
    config = _config(tmp_path, batch_concurrency=False)
    with BatchRunner(config, graph_factory=_VacuumGraph, progress=False) as br:
        results = br.run(["AAPL", "VACUUM", "NVDA"], "2026-01-10")
    _assert_isolation(results, "VACUUM", DataVacuumError)
    # The error keeps its message: frontends print type + str(exc).
    assert "data vacuum" in str(results[1]["error"])


@pytest.mark.unit
def test_data_vacuum_error_isolated_serial_real(tmp_path):
    """Same pin via the run_batch.py-style entry (dedup on, all-unique tickers)."""
    config = _config(tmp_path, batch_concurrency=False, batch_dedup_tickers=True)
    with BatchRunner(config, graph_factory=_VacuumGraph, progress=False) as br:
        results = br.run(["AAPL", "VACUUM", "NVDA", "VACUUM"], "2026-01-10")
    # Dedup collapses the duplicate vacuum ticker: one row per unique ticker.
    assert [r["ticker"] for r in results] == ["AAPL", "VACUUM", "NVDA"]
    assert isinstance(results[1]["error"], DataVacuumError)
    assert all(r["error"] is None for r in (results[0], results[2]))


@pytest.mark.unit
def test_data_vacuum_error_isolated_concurrent(tmp_path):
    """Same taxonomy under the worker pool: DataVacuumError must not break the
    ThreadPoolExecutor loop or reorder results."""
    config = _config(tmp_path, batch_concurrency=True, batch_workers=2)
    with BatchRunner(config, graph_factory=_VacuumGraph, progress=False) as br:
        results = br.run(["AAPL", "VACUUM", "NVDA"], "2026-01-10")
    _assert_isolation(results, "VACUUM", DataVacuumError)


@pytest.mark.unit
def test_generic_exception_isolated_serial(tmp_path):
    config = _config(tmp_path, batch_concurrency=False)
    with BatchRunner(config, graph_factory=_CrashGraph, progress=False) as br:
        results = br.run(["AAPL", "BOOM", "NVDA"], "2026-01-10")
    _assert_isolation(results, "BOOM", RuntimeError)


@pytest.mark.unit
def test_save_reports_failure_is_isolated_like_propagate(tmp_path):
    """The per-ticker try block covers BOTH propagate and save_reports: a
    reporting-stage failure degrades only that ticker's row."""
    config = _config(tmp_path, batch_concurrency=False)
    with BatchRunner(config, graph_factory=_SaveCrashGraph, progress=False) as br:
        results = br.run(["AAPL", "SAVEFAIL", "NVDA"], "2026-01-10")
    _assert_isolation(results, "SAVEFAIL", OSError)


@pytest.mark.unit
def test_result_row_schema(tmp_path):
    """Every row (ok and failed alike) exposes exactly the six documented keys."""
    config = _config(tmp_path, batch_concurrency=False)
    with BatchRunner(config, graph_factory=_VacuumGraph, progress=False) as br:
        results = br.run(["AAPL", "VACUUM"], "2026-01-10")
    assert len(results) == 2
    for row in results:
        assert set(row) == _ROW_KEYS
        assert isinstance(row["elapsed"], float)
        assert row["elapsed"] >= 0.0


# --------------------------------------------------------------------------- #
# Config ContextVar propagation
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_submit_with_context_pins_config_per_submission():
    """The seam the runner relies on: copy_context() per submission carries
    the config ContextVar into the pool worker. A plain executor.submit
    does NOT (Python deliberately does not copy contexts into pool threads)
    — that contrast is why every _run_concurrent submission goes through the
    helper."""
    probe = "test_submit_ctx_probe"
    with ThreadPoolExecutor(max_workers=2) as ex:
        set_config({probe: "pinned-one"})
        pinned = submit_with_context(ex, lambda: get_config().get(probe))
        set_config({probe: "pinned-two"})
        other = submit_with_context(ex, lambda: get_config().get(probe))
        plain = ex.submit(lambda: get_config().get(probe))

        assert pinned.result() == "pinned-one"
        # Each submission snapshots the context AT SUBMIT TIME: an earlier
        # submission is not retroactively overwritten by a later set_config.
        assert other.result() == "pinned-two"
        # The control: no context copy -> the worker thread's own (default)
        # config, which never carries this test's probe key.
        assert plain.result() is None


@pytest.mark.unit
def test_serial_run_one_rebinds_runner_config(tmp_path):
    """The defensive re-bind: _run_one calls set_config(self._runtime_config)
    before running, so a runner constructed under one config still runs on it
    after the ambient context has been changed by someone else."""
    probe = "test_serial_rebind_probe"
    config = _config(tmp_path, batch_concurrency=False)
    config[probe] = "runner-config"

    class _ProbeGraph(_FakeGraph):
        def propagate(self, ticker, trade_date, asset_type="stock"):
            state, signal = super().propagate(ticker, trade_date, asset_type)
            state[probe] = get_config().get(probe)
            return state, signal

    with BatchRunner(config, graph_factory=_ProbeGraph, progress=False) as br:
        # Simulate the ambient context drifting after construction.
        set_config({probe: "mutated-after-construction"})
        results = br.run(["AAPL"], "2026-01-10")

    assert results[0]["error"] is None
    assert results[0]["state"][probe] == "runner-config"


@pytest.mark.unit
def test_concurrent_workers_see_runner_config(tmp_path):
    """Pool path: the __init__ context snapshot flows through
    submit_with_context into every worker thread."""
    probe = "test_concurrent_probe"
    config = _config(tmp_path, batch_concurrency=True, batch_workers=2)
    config[probe] = "pinned-987"

    class _ProbeGraph(_FakeGraph):
        def propagate(self, ticker, trade_date, asset_type="stock"):
            state, signal = super().propagate(ticker, trade_date, asset_type)
            state[probe] = get_config().get(probe)
            return state, signal

    with BatchRunner(config, graph_factory=_ProbeGraph, progress=False) as br:
        results = br.run(["AAPL", "NVDA"], "2026-01-10")

    assert [r["state"][probe] for r in results] == ["pinned-987", "pinned-987"]


@pytest.mark.unit
def test_graph_pool_is_built_per_worker(tmp_path):
    """One graph per pool slot, built up front: K graphs for K workers, and
    exactly 1 when the master switch forces serial."""
    built = []

    def counting_factory(config):
        built.append(config)
        return _FakeGraph(config)

    serial_cfg = _config(tmp_path, batch_concurrency=False, batch_workers=3)
    with BatchRunner(serial_cfg, graph_factory=counting_factory, progress=False):
        assert len(built) == 1  # switch off -> K=1 regardless of batch_workers

    built.clear()
    pool_cfg = _config(tmp_path / "pool", batch_concurrency=True)
    with BatchRunner(pool_cfg, workers=3, graph_factory=counting_factory,
                     progress=False) as br:
        assert len(built) == 3
        assert br.workers == 3
        assert br._pool.qsize() == 3


# --------------------------------------------------------------------------- #
# prepare_batch_run: the shared validation contract (exit code 2 in both
# frontends: scripts/run_batch.py and `yialpha batch`)
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_prepare_batch_run_rejects_bad_ticker():
    with pytest.raises(BatchInputError, match="Invalid ticker"):
        prepare_batch_run(["AAPL!"], "2026-01-10")


@pytest.mark.unit
def test_prepare_batch_run_rejects_bad_date():
    with pytest.raises(BatchInputError, match="Bad date"):
        prepare_batch_run(["AAPL"], "2026/01/10")
    with pytest.raises(BatchInputError, match="Bad date"):
        prepare_batch_run(["AAPL"], "2026-13-01")


@pytest.mark.unit
def test_prepare_batch_run_rejects_non_positive_workers():
    with pytest.raises(BatchInputError, match="at least 1"):
        prepare_batch_run(["AAPL"], "2026-01-10", workers=0)
    with pytest.raises(BatchInputError, match="at least 1"):
        prepare_batch_run(["AAPL"], "2026-01-10", workers=-2)


@pytest.mark.unit
def test_prepare_batch_run_rejects_mixed_asset_classes():
    """auto detection is per-batch: one class only (NVDA=stock, BTC-USD=crypto)."""
    with pytest.raises(BatchInputError, match="Mixed asset classes"):
        prepare_batch_run(["NVDA", "BTC-USD"], "2026-01-10", asset_type="auto")


@pytest.mark.unit
def test_prepare_batch_run_worker_override_semantics():
    """Explicit workers>1 forces the pool ON, workers=1 forces it OFF, and
    None honors whatever the env/default baked into DEFAULT_CONFIG."""
    config, resolved = prepare_batch_run(
        ["AAPL", "NVDA"], "2026-01-10", "stock", workers=2,
    )
    assert config["batch_concurrency"] is True

    config, _ = prepare_batch_run(["AAPL"], "2026-01-10", "stock", workers=1)
    assert config["batch_concurrency"] is False

    config, resolved = prepare_batch_run(["BTC-USD", "ETH-USD"], "2026-01-10")
    assert resolved == "crypto"
    assert config["batch_concurrency"] == DEFAULT_CONFIG["batch_concurrency"]


@pytest.mark.unit
def test_prepare_batch_run_results_dir_override_and_defaults():
    config, resolved = prepare_batch_run(
        ["AAPL"], "2026-01-10", "stock", results_dir="/tmp/xyz",
    )
    assert config["results_dir"] == "/tmp/xyz"
    assert resolved == "stock"
    # Without the override the config keeps the default root.
    config, _ = prepare_batch_run(["AAPL"], "2026-01-10", "stock")
    assert config["results_dir"] == DEFAULT_CONFIG["results_dir"]


@pytest.mark.unit
def test_prepare_batch_run_empty_tickers_raises_batch_input_error():
    """Programmatic empty list hits the exit-2 contract, not IndexError
    (round-5 fix; both frontends require >=1 ticker via argparse/typer)."""
    import pytest as _pytest

    from yialpha.batch.runner import BatchInputError, prepare_batch_run

    with _pytest.raises(BatchInputError):
        prepare_batch_run([], "2026-09-19", "auto")
