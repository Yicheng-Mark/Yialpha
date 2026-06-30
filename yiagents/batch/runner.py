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
  * One batch = one config. YiAgentsGraph.__init__ mutates a module-global
    config via set_config(), so all K workers MUST carry identical config; we
    assert it (otherwise a worker fetches data with another asset class's
    vendors).

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
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable

from tqdm import tqdm

from yiagents.graph.trading_graph import YiAgentsGraph

logger = logging.getLogger(__name__)

# Config keys that drive LLM + vendor behaviour. The module-global set_config()
# means all workers must agree on these; results_dir / cache paths are shared
# and intentionally excluded from the uniformity check.
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

        concurrency = bool(config.get("batch_concurrency", False))
        requested = workers if workers is not None else int(config.get("batch_workers", 3))
        # Master switch off → strictly serial (K=1), byte-equivalent to today.
        self.workers = max(1, requested) if concurrency else 1

        # Per-thread ticker context for log attribution.
        self._worker_ctx = threading.local()
        self._log_filter = _TickerLogFilter(self._worker_ctx)
        logging.getLogger().addFilter(self._log_filter)

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
        """Remove the log filter (idempotent). Safe to skip at process exit."""
        flt = getattr(self, "_log_filter", None)
        if flt is not None:
            logging.getLogger().removeFilter(flt)
            self._log_filter = None

    def __enter__(self) -> "BatchRunner":
        return self

    def __exit__(self, *exc) -> bool:
        self.close()
        return False

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
        results: list[dict] = []
        with ThreadPoolExecutor(max_workers=self.workers) as ex:
            future_to_ticker = {
                ex.submit(self._run_one, t, trade_date, asset_type): t
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
                            "error": exc,
                        }
                    results.append(res)
                    if pbar:
                        pbar.update(1)
                    if fail_fast and res["error"] is not None:
                        for f in future_to_ticker:
                            f.cancel()
                        break
            finally:
                if pbar:
                    pbar.close()
        # as_completed yields completion order; restore input order.
        by_ticker = {r["ticker"]: r for r in results}
        return [by_ticker[t] for t in tickers]
