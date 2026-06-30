"""Concurrency safety tests for the multi-ticker batch layer (Phase A gates G2/G3/G5).

These validate the iron-law foundation: under racing worker threads, the shared
backing files (memory log) lose no updates and stay parseable, the per-path
FileLock serializes same-path access while the master switch correctly forces a
serial (K=1) fallback. No network/LLM — graph construction is stubbed.
"""

import threading
import time

import pytest

from yiagents.agents.utils.memory import TradingMemoryLog
from yiagents.batch.locks import FileLock
from yiagents.batch.runner import BatchRunner

DECISION = "Rating: Buy\nEnter at $190, 6% portfolio cap."


# ---------------------------------------------------------------------------
# Stubs (avoid real graph / LLM / network)
# ---------------------------------------------------------------------------


class _FakeGraph:
    """Minimal stand-in for YiAgentsGraph used by BatchRunner."""

    def __init__(self, config):
        self.config = config
        self.ticker = None

    def propagate(self, ticker, trade_date, asset_type="stock"):
        self.ticker = ticker
        return ({"final_trade_decision": "Rating: Buy", "company_of_interest": ticker}, "Buy")

    def save_reports(self, final_state, ticker):
        return None


class _FailGraph(_FakeGraph):
    """A worker that blows up for one specific ticker (graceful-degradation probe)."""

    def __init__(self, config, fail_ticker="BAD"):
        super().__init__(config)
        self._fail_ticker = fail_ticker

    def propagate(self, ticker, trade_date, asset_type="stock"):
        if ticker == self._fail_ticker:
            raise ValueError("unknown symbol")
        return super().propagate(ticker, trade_date, asset_type=asset_type)


def _fake_factory(config):
    return _FakeGraph(config)


# ---------------------------------------------------------------------------
# FileLock primitive (G5)
# ---------------------------------------------------------------------------


def test_filelock_serializes_same_path(tmp_path):
    """Eight FileLock instances on the same path never overlap (max concurrency 1)."""
    lockpath = tmp_path / "shared"
    current = {"n": 0}
    peak = {"v": 0}
    guard = threading.Lock()

    def worker():
        # Each worker makes its OWN instance — they must still exclude each other
        # via the path-keyed process-local registry.
        with FileLock(lockpath):
            with guard:
                current["n"] += 1
                peak["v"] = max(peak["v"], current["n"])
            time.sleep(0.02)
            with guard:
                current["n"] -= 1

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert peak["v"] == 1, f"same-path locks overlapped: peak concurrency {peak['v']}"


# ---------------------------------------------------------------------------
# Memory log under concurrency (G5: no lost updates, no corruption)
# ---------------------------------------------------------------------------


def test_memory_log_concurrent_distinct_tickers_no_loss(tmp_path):
    """30 threads appending distinct tickers → all 30 entries present, file parses."""
    log = TradingMemoryLog({"memory_log_path": str(tmp_path / "mem.md")})
    tickers = [f"T{i:03d}" for i in range(30)]
    threads = [
        threading.Thread(target=log.store_decision, args=(t, "2026-01-10", DECISION))
        for t in tickers
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    entries = log.load_entries()  # parses cleanly => no corruption
    assert len(entries) == 30
    assert {e["ticker"] for e in entries} == set(tickers)


def test_memory_log_concurrent_same_key_stays_idempotent(tmp_path):
    """20 threads storing the SAME (ticker, date) → exactly one entry.

    Without the lock, the check-then-act idempotency scan races the append and
    several workers would each pass the scan before any write lands, double- or
    triple-appending. The lock makes scan+append atomic.
    """
    log = TradingMemoryLog({"memory_log_path": str(tmp_path / "mem.md")})
    threads = [
        threading.Thread(target=log.store_decision, args=("NVDA", "2026-01-10", DECISION))
        for _ in range(20)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(log.load_entries()) == 1


def test_memory_log_concurrent_update_no_loss(tmp_path):
    """Concurrent update_with_outcome on different pending entries resolves all of them."""
    log = TradingMemoryLog({"memory_log_path": str(tmp_path / "mem.md")})
    tickers = [f"T{i:03d}" for i in range(15)]
    for t in tickers:
        log.store_decision(t, "2026-01-10", DECISION)

    def update(t):
        log.update_with_outcome(t, "2026-01-10", 0.05, 0.02, 5, f"Lesson {t}.")

    threads = [threading.Thread(target=update, args=(t,)) for t in tickers]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    entries = log.load_entries()
    assert len(entries) == 15
    assert all(not e["pending"] for e in entries)
    assert all(e["reflection"] == f"Lesson {e['ticker']}." for e in entries)


# ---------------------------------------------------------------------------
# BatchRunner switches, dedup, uniformity (G2)
# ---------------------------------------------------------------------------


def test_batchrunner_off_switch_forces_serial():
    config = {"batch_concurrency": False, "batch_workers": 5}
    with BatchRunner(config, graph_factory=_fake_factory, progress=False) as br:
        assert br.workers == 1  # master switch off → K=1 regardless of batch_workers


def test_batchrunner_on_respects_workers():
    config = {"batch_concurrency": True, "batch_workers": 4}
    with BatchRunner(config, graph_factory=_fake_factory, progress=False) as br:
        assert br.workers == 4
        assert br._pool.qsize() == 4


def test_batchrunner_dedup():
    config = {"batch_concurrency": False, "batch_dedup_tickers": True}
    with BatchRunner(config, graph_factory=_fake_factory, progress=False) as br:
        # strip().upper() keying: case/whitespace variants collapse to one.
        assert br._dedup(["AAPL", "aapl", "NVDA", " NVDA "]) == ["AAPL", "NVDA"]


def test_batchrunner_rejects_divergent_config():
    base = {"batch_concurrency": True, "batch_workers": 2, "llm_provider": "openai"}

    def divergent(config):
        g = _FakeGraph(config)
        g.config = {**config, "llm_provider": "anthropic"}  # different signature
        return g

    with pytest.raises(ValueError, match="identical config"):
        BatchRunner(base, graph_factory=divergent, progress=False)


def test_batchrunner_run_serial_end_to_end():
    config = {"batch_concurrency": False, "batch_dedup_tickers": True}
    with BatchRunner(config, graph_factory=_fake_factory, progress=False) as br:
        results = br.run(["AAPL", "NVDA", "AAPL"], "2026-01-10")
    assert len(results) == 2  # dedup dropped the second AAPL
    assert [r["ticker"] for r in results] == ["AAPL", "NVDA"]
    assert all(r["error"] is None for r in results)
    assert all(r["signal"] == "Buy" for r in results)


def test_batchrunner_concurrent_graceful_degradation():
    """G3: one bad ticker doesn't abort the batch; others complete."""
    config = {
        "batch_concurrency": True,
        "batch_workers": 3,
        "batch_fail_fast": False,
    }
    with BatchRunner(config, graph_factory=_FailGraph, progress=False) as br:
        results = br.run(["AAPL", "BAD", "NVDA"], "2026-01-10")
    # Results restored to input order.
    assert [r["ticker"] for r in results] == ["AAPL", "BAD", "NVDA"]
    by_ticker = {r["ticker"]: r for r in results}
    assert by_ticker["BAD"]["error"] is not None
    assert by_ticker["AAPL"]["error"] is None
    assert by_ticker["NVDA"]["error"] is None
