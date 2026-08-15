"""Multi-ticker batch runner: analyze many tickers concurrently.

Concurrency is layered ABOVE ``propagate()`` — each ticker runs the exact same
graph the single-ticker path runs; nothing inside any agent changes. The iron
law (byte-equivalence to a serial run) holds because the concurrency layer adds
no new inputs, depth, or randomness to any ticker's analysis.

Why this is safe (see plan yiagents-prancy-acorn.md):
  * A pool of K long-lived YiAgentsGraph instances, one per worker, never shared
    across threads. Each instance is touched by exactly one thread, so the
    instance-mutation hazards (self.ticker at trading_graph.py:437, self.graph
    recompile at :448/:468, self.curr_state at :535, the memory_log object) are
    all single-writer and race-free.
  * Shared BACKING files (memory log, OHLCV cache) are serialized by their own
    file locks (yiagents.batch.locks) — installed unconditionally, harmless when
    uncontended.
  * One batch = one config. The submitting ContextVar snapshot is copied into
    every worker; a uniformity guard rejects an accidentally divergent graph
    config before any work begins.

Master switch ``batch_concurrency`` (default False): when off, the runner runs
strictly serial (K=1, deterministic order) and is byte-equivalent to today.
"""
from __future__ import annotations

import hashlib
import json
import logging
import queue
import threading
import time
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import ExitStack, contextmanager
from pathlib import Path

from tqdm import tqdm

from yiagents.batch.locks import FileLock
from yiagents.dataflows.config import get_config, set_config, submit_with_context
from yiagents.dataflows.utils import safe_ticker_component
from yiagents.graph.trading_graph import YiAgentsGraph

logger = logging.getLogger(__name__)


@contextmanager
def serialized_run(
    config: dict, ticker: str, trade_date: str, asset_type: str
) -> Iterator[None]:
    """Serialize one logical run across threads *and* processes.

    A run touches both checkpoint/cache state and final report/temp paths. Lock
    under both configured roots so independently launched CLI/Web processes
    coordinate whenever either shared resource would collide. User-controlled
    values are validated or hashed before becoming path components.
    """
    safe_ticker = safe_ticker_component(ticker.strip().upper())
    # Reports, checkpoints and temporary state are currently keyed only by
    # ticker/date. Keep asset_type in this public API for caller compatibility,
    # but deliberately leave it out of the lock key: unlike asset modes must
    # still serialize instead of racing on those shared paths.
    del asset_type
    run_digest = hashlib.sha256(
        json.dumps(
            [safe_ticker, str(trade_date)],
            ensure_ascii=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()[:24]

    lock_paths: set[Path] = set()
    for config_key in ("data_cache_dir", "results_dir"):
        root_value = config.get(config_key)
        if not root_value:
            raise ValueError(f"{config_key} is required for batch run locking")
        lock_dir = Path(root_value).expanduser().resolve() / ".run_locks"
        lock_dir.mkdir(parents=True, exist_ok=True)
        lock_paths.add(lock_dir / f"{safe_ticker}-{run_digest}")

    with ExitStack() as stack:
        # Stable acquisition order prevents deadlock when the two configured
        # roots differ between otherwise overlapping callers.
        for path in sorted(lock_paths, key=str):
            stack.enter_context(FileLock(path))
        yield


# Config keys that drive LLM + vendor behaviour. Workers intentionally share
# one submitting context snapshot, so divergent graph configs are rejected;
# results/cache roots are shared and intentionally excluded from this check.
_UNIFORM_CONFIG_KEYS = (
    "llm_provider",
    "deep_think_llm",
    "quick_think_llm",
    "backend_url",
    "data_vendors",
    "tool_vendors",
    "benchmark_ticker",
)


def _config_signature(config: dict) -> str:
    """Stable hash of the config keys that must be uniform across workers."""
    payload = {k: config.get(k) for k in _UNIFORM_CONFIG_KEYS}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


class _TickerLogFilter(logging.Filter):
    """Tag log records with the ticker of the worker thread emitting them.

    Lets interleaved stderr stay attributable under concurrency. Reads a
    threading.local so each worker thread tags only its own records; records
    from the main thread (no ticker set) pass through unmodified. Idempotent.
    """

    def __init__(self, ctx: threading.local):
        super().__init__()
        self._ctx = ctx

    def filter(self, record: logging.LogRecord) -> bool:
        ticker = getattr(self._ctx, "ticker", None)
        if ticker and not getattr(record, "_batch_tagged", False):
            record._batch_tagged = True  # type: ignore[attr-defined]
            record.msg = f"[{ticker}] {record.getMessage()}"
            record.args = ()  # msg already formatted; don't re-interpolate
        return True


class BatchRunner:
    """Run propagate() for many tickers across a pool of K worker graphs.

    Parameters
    ----------
    config : dict
        Base config (DEFAULT_CONFIG + env overrides). One asset class only.
    workers : int, optional
        Pool size K. Defaults to ``config["batch_workers"]``. Ignored (forced to
        1) when ``config["batch_concurrency"]`` is False.
    selected_analysts : tuple, optional
        Passed through to every YiAgentsGraph (default: all four analysts).
    graph_factory : callable, optional
        Override graph construction (testing). Must build graphs sharing the
        passed config's uniform keys.
    progress : bool, optional
        Show a tqdm progress bar (default True). Set False under a Rich UI.
    """

    def __init__(
        self,
        config: dict,
        *,
        workers: int | None = None,
        selected_analysts: tuple = ("market", "social", "news", "fundamentals"),
        graph_factory: Callable[[dict], YiAgentsGraph] | None = None,
        progress: bool = True,
    ):
        self.config = config
        self.progress = progress
        self.selected_analysts = selected_analysts
        self._graph_factory = graph_factory or self._default_graph_factory
        self._signature = _config_signature(config)

        # Graph construction and every worker submission below inherit this
        # exact context snapshot. ContextVar values do not otherwise cross a
        # ThreadPoolExecutor boundary.
        set_config(config)
        self._runtime_config = get_config()

        concurrency = bool(config.get("batch_concurrency", False))
        requested = workers if workers is not None else int(config.get("batch_workers", 3))
        # Master switch off → strictly serial (K=1), byte-equivalent to today.
        self.workers = max(1, requested) if concurrency else 1

        # Per-thread ticker context for log attribution. The filter must sit
        # on the root logger's HANDLERS, not on the root logger itself: a
        # logger-level filter only sees records emitted directly on that
        # logger, so records from ``yiagents.*`` child loggers (where all of
        # our modules log) propagate straight past it — the tagging never
        # fired. Handler-level filters run for every record the handler
        # processes, propagated ones included. Requires setup_logging() to
        # have run first (every entry point does, at import); a handler added
        # later is picked up by the next BatchRunner construction.
        self._worker_ctx = threading.local()
        self._log_filter: _TickerLogFilter | None = _TickerLogFilter(self._worker_ctx)
        for handler in logging.getLogger().handlers:
            handler.addFilter(self._log_filter)

        # Build the K-graph pool up front. set_config() is idempotent for
        # identical configs, so the last construction leaves the global in the
        # right state for every worker.
        self._pool: queue.Queue = queue.Queue()
        for _ in range(self.workers):
            graph = self._graph_factory(config)
            self._assert_uniform(graph.config)
            self._pool.put(graph)

    # -- public API ---------------------------------------------------------

    def run(
        self,
        tickers: list[str],
        trade_date: str,
        asset_type: str = "stock",
    ) -> list[dict]:
        """Analyze every ticker for ``trade_date``; one result dict per ticker.

        Single-ticker failures are recorded (``error`` key), not raised, so one
        bad symbol doesn't abort the batch — unless ``batch_fail_fast`` is set.
        Results are returned in the input ticker order.
        """
        tickers = self._dedup(tickers)

        if self.workers == 1:
            return self._run_serial(tickers, trade_date, asset_type)
        return self._run_concurrent(tickers, trade_date, asset_type)

    def close(self) -> None:
        """Remove the log filter from every root handler (idempotent).

        Safe to skip at process exit. Removal scans all current root handlers
        (not just the ones seen at construction) so a ``setup_logging`` handler
        swap between construction and close cannot leak the filter.
        """
        flt = getattr(self, "_log_filter", None)
        if flt is not None:
            for handler in logging.getLogger().handlers:
                handler.removeFilter(flt)
            self._log_filter = None

    def __enter__(self) -> BatchRunner:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- internals ----------------------------------------------------------

    def _default_graph_factory(self, config: dict) -> YiAgentsGraph:
        return YiAgentsGraph(self.selected_analysts, debug=False, config=config)

    def _assert_uniform(self, graph_config: dict) -> None:
        if _config_signature(graph_config) != self._signature:
            raise ValueError(
                "All batch workers must share an identical config for the "
                "LLM/vendor keys. YiAgentsGraph.__init__ mutates a module-global "
                "via set_config(), so divergent configs would clobber each other "
                "(one worker fetching another asset class's data vendors)."
            )

    def _dedup(self, tickers: list[str]) -> list[str]:
        if not self.config.get("batch_dedup_tickers", True):
            return list(tickers)
        seen, unique = set(), []
        for t in tickers:
            key = t.strip().upper()
            if key not in seen:
                seen.add(key)
                unique.append(t)
        if len(unique) != len(tickers):
            logger.warning(
                "Removed %d duplicate ticker(s) from the batch "
                "(same-ticker concurrency would race the OHLCV cache/checkpoint DB).",
                len(tickers) - len(unique),
            )
        return unique

    def _run_one(self, ticker: str, trade_date: str, asset_type: str) -> dict:
        """Acquire a graph from the pool, run propagate + save_reports, release."""
        # Defensive re-bind: normal submissions already carry a copied context,
        # but this also covers a runner constructed in one context and invoked
        # from another. Nested fan-out submissions then inherit the same config.
        set_config(self._runtime_config)
        with serialized_run(self._runtime_config, ticker, trade_date, asset_type):
            self._worker_ctx.ticker = ticker
            graph = self._pool.get()
            start = time.monotonic()
            try:
                final_state, signal = graph.propagate(
                    ticker, trade_date, asset_type=asset_type
                )
                report_path = graph.save_reports(final_state, ticker)
                return {
                    "ticker": ticker,
                    "state": final_state,
                    "signal": signal,
                    "report_path": report_path,
                    "elapsed": time.monotonic() - start,
                    "error": None,
                }
            except Exception as exc:
                # One ticker failing must not abort the batch.
                logger.exception("Ticker %s failed in batch", ticker)
                return {
                    "ticker": ticker,
                    "state": None,
                    "signal": None,
                    "report_path": None,
                    "elapsed": time.monotonic() - start,
                    "error": exc,
                }
            finally:
                self._pool.put(graph)
                self._worker_ctx.ticker = None

    def _run_serial(
        self, tickers: list[str], trade_date: str, asset_type: str
    ) -> list[dict]:
        iterable = (
            tqdm(tickers, desc="batch", unit="ticker") if self.progress else tickers
        )
        return [self._run_one(t, trade_date, asset_type) for t in iterable]

    def _run_concurrent(
        self, tickers: list[str], trade_date: str, asset_type: str
    ) -> list[dict]:
        fail_fast = bool(self.config.get("batch_fail_fast", False))
        results_by_future: dict = {}
        failed_ticker: str | None = None
        with ThreadPoolExecutor(max_workers=self.workers) as ex:
            future_to_ticker = {
                submit_with_context(ex, self._run_one, t, trade_date, asset_type): t
                for t in tickers
            }
            pbar = (
                tqdm(total=len(tickers), desc="batch", unit="ticker")
                if self.progress
                else None
            )
            try:
                for fut in as_completed(future_to_ticker):
                    t = future_to_ticker[fut]
                    try:
                        res = fut.result()
                    except Exception as exc:
                        # _run_one already catches per-ticker errors; this guards
                        # the pool itself (e.g. BrokenThreadPool).
                        res = {
                            "ticker": t,
                            "state": None,
                            "signal": None,
                            "report_path": None,
                            "elapsed": 0.0,
                            "error": exc,
                        }
                    results_by_future[fut] = res
                    if pbar:
                        pbar.update(1)
                    if fail_fast and res["error"] is not None:
                        failed_ticker = t
                        for f in future_to_ticker:
                            if f is not fut:
                                f.cancel()
                        break
            finally:
                if pbar:
                    pbar.close()

        # Leaving the executor waits for already-running work and cancels only
        # futures that had not started. Record every input position explicitly:
        # the old ticker-keyed reconstruction raised KeyError as soon as
        # fail-fast left a future uncollected (and also collapsed duplicates).
        for fut, ticker in future_to_ticker.items():
            if fut in results_by_future:
                continue
            if fut.cancelled():
                error = RuntimeError(
                    f"cancelled by batch_fail_fast after {failed_ticker or 'another ticker'} failed"
                )
                results_by_future[fut] = {
                    "ticker": ticker,
                    "state": None,
                    "signal": None,
                    "report_path": None,
                    "elapsed": 0.0,
                    "error": error,
                }
                continue
            try:
                results_by_future[fut] = fut.result()
            except Exception as exc:
                results_by_future[fut] = {
                    "ticker": ticker,
                    "state": None,
                    "signal": None,
                    "report_path": None,
                    "elapsed": 0.0,
                    "error": exc,
                }

        return [results_by_future[fut] for fut in future_to_ticker]
