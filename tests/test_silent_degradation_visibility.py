"""Regressions for the last silent-degradation closures (2026-08 audit).

Each test pins one previously-invisible failure path to its new observable
behavior (a raised error or a logged warning):

- H1/H2: yfinance news vendors must raise (not return an error string) so the
  core-category router can fail closed — an error string used to masquerade
  as successfully fetched news.
- M1: a failed BaoStock quarterly fetch is logged, not silently skipped.
- M2: the backtest benchmark SPY fallback is logged (alpha/beta against the
  wrong index must be observable).
- M4: checkpoint cleanup failure is logged (stale-resume risk).
- M5: fin_cot prompt-selection failure logs its fallback to the legacy prompt.
- L4: an unparseable memory as_of_date logs why no history was injected.
- L5: a failed Binance order-recovery lookup is logged even though None
  already drives the fail-closed REJECTED path.
"""

from __future__ import annotations

from datetime import date
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import yiagents.dataflows.yfinance_news as ynews
from yiagents.agents.utils.memory import TradingMemoryLog
from yiagents.backtest.engine import _resolve_index_benchmark
from yiagents.dataflows import baostock_vendor
from yiagents.execution.binance_gateway import BinanceGateway
from yiagents.execution.domain import (
    Direction,
    Exchange,
    OrderRequest,
    OrderType,
)
from yiagents.graph import checkpointer


@pytest.fixture(autouse=True)
def _isolated_search_cache(tmp_path, monkeypatch):
    """Route the global-news Search disk cache into a per-test tmp dir.

    ``_cached_search_news`` serves repeats from disk, which would otherwise
    bypass the mocked ``yf.Search`` of a later test (and pollute the real
    user cache).
    """
    def _cache_dir(name):
        d = tmp_path / name
        d.mkdir(parents=True, exist_ok=True)
        return str(d)

    monkeypatch.setattr(ynews, "vendor_cache_dir", _cache_dir)



# --------------------------------------------------------------------------- #
# H1/H2 — yfinance news vendors raise instead of returning error prose
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_ticker_news_failure_raises_not_prose(monkeypatch, caplog):
    class BoomTicker:
        def get_news(self, count):
            raise RuntimeError("network down")

    monkeypatch.setattr(ynews.yf, "Ticker", lambda t: BoomTicker())
    monkeypatch.setattr(ynews, "yf_retry", lambda fn: fn())
    with (caplog.at_level("ERROR", logger="yiagents.dataflows.yfinance_news"),
          pytest.raises(RuntimeError, match="network down")):
        ynews.get_news_yfinance("AAPL", "2025-01-01", "2025-01-10")
    assert any("news retrieval failed" in r.message for r in caplog.records)


@pytest.mark.unit
def test_global_news_failure_raises_not_prose(monkeypatch, caplog):
    def boom_search(*args, **kwargs):
        raise RuntimeError("network down")

    monkeypatch.setattr(ynews.yf, "Search", boom_search)
    monkeypatch.setattr(ynews, "yf_retry", lambda fn: fn())
    with (caplog.at_level("ERROR", logger="yiagents.dataflows.yfinance_news"),
          pytest.raises(RuntimeError, match="network down")):
        ynews.get_global_news_yfinance("2025-05-09", look_back_days=7, limit=10)
    assert any("global news retrieval failed" in r.message for r in caplog.records)


# --------------------------------------------------------------------------- #
# M1 — BaoStock quarterly fetch failure is logged, statement survives
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_baostock_quarter_fetch_failure_warns(caplog):
    class FakeRs:
        error_code = "0"

        def next(self):
            return False

    def raising_query(**kwargs):
        if (kwargs.get("year"), kwargs.get("quarter")) == ("2024", "1"):
            raise RuntimeError("boom")
        return FakeRs()

    bs = SimpleNamespace(profit=raising_query)
    with caplog.at_level("WARNING", logger="yiagents.dataflows.baostock_vendor"):
        fetch = baostock_vendor._query_statement(
            bs, "600519.SH", "profit", anchor=date(2025, 6, 1))
    assert fetch.rows == []  # failed quarter skipped, the statement call survives
    # The failed quarter is returned to the renderer for an in-band note.
    assert fetch.failed == ["2024Q1"]
    hits = [r for r in caplog.records if "fetch failed" in r.message]
    assert hits and "600519.SH" in hits[0].message and "2024Q1" in hits[0].message


# --------------------------------------------------------------------------- #
# M2 — benchmark SPY fallback stays observable
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_benchmark_spy_fallback_warns(caplog):
    class Graph:
        def _resolve_benchmark(self, ticker):
            raise RuntimeError("no benchmark for this market")

    with caplog.at_level("WARNING", logger="yiagents.backtest.engine"):
        assert _resolve_index_benchmark(Graph(), "600519.SS") == "SPY"
    assert any("falling back to SPY" in r.message for r in caplog.records)


# --------------------------------------------------------------------------- #
# M4 — checkpoint cleanup failure warns (stale-resume risk)
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_checkpoint_cleanup_failure_warns(tmp_path, caplog):
    db_dir = tmp_path / "checkpoints"
    db_dir.mkdir()
    # An empty DB file: connect succeeds but the DELETE hits missing tables,
    # i.e. the sqlite3.OperationalError cleanup path.
    (db_dir / "AAPL.db").touch()
    with caplog.at_level("WARNING", logger="yiagents.graph.checkpointer"):
        checkpointer.clear_checkpoint(tmp_path, "AAPL", "2024-01-01")
    assert any("checkpoint cleanup failed" in r.message for r in caplog.records)


# --------------------------------------------------------------------------- #
# M5 — fin_cot prompt-selection failure logs the legacy fallback
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_fincot_prompt_selection_failure_warns(monkeypatch, caplog):
    # Config itself is healthy; only the fin_cot prompt build blows up — the
    # run must fall back to the legacy prompt, visibly.
    from yiagents.dataflows.config import set_config

    set_config({"fin_cot_prompts": True})

    def boom():
        raise RuntimeError("fincot build broken")

    import yiagents.agents.analysts.market_analyst as ma

    monkeypatch.setattr(ma, "_fincot_system_message", boom)
    with caplog.at_level("WARNING", logger="yiagents.agents.analysts.market_analyst"):
        msg = ma._system_message()
    assert "You are a trading assistant" in msg   # legacy fallback still served
    assert any("fin_cot prompt selection failed" in r.message for r in caplog.records)


# --------------------------------------------------------------------------- #
# L4 — unparseable memory as_of_date explains why no history was injected
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_memory_bad_as_of_date_warns(tmp_path, caplog):
    log = TradingMemoryLog(
        {"memory_enabled": True, "memory_log_path": str(tmp_path / "mem.md")}
    )
    with caplog.at_level("WARNING", logger="yiagents.agents.utils.memory"):
        out = log.get_past_context("AAPL", as_of_date="not-a-date")
    assert out == ""    # fail closed: no history rather than a look-ahead leak
    assert any("unparseable as_of_date" in r.message for r in caplog.records)


# --------------------------------------------------------------------------- #
# L5 — Binance order-recovery lookup failure is logged (None -> REJECTED)
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_binance_recovery_failure_warns(caplog):
    gw = BinanceGateway()
    gw._client = MagicMock()
    gw._client.rest_api.query_order.side_effect = RuntimeError("boom")
    req = OrderRequest(
        symbol="BTCUSDT",
        exchange=Exchange.BINANCE,
        direction=Direction.LONG,
        type=OrderType.MARKET,
        volume=0.5,
    )
    with caplog.at_level("WARNING", logger="yiagents.execution.binance_gateway"):
        assert gw._safe_query_by_client_id(req, "coid-1") is None
    assert any("recovery lookup failed" in r.message for r in caplog.records)
