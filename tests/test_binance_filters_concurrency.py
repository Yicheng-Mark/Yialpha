"""Concurrency tests for binance_filters.get_symbol_filters (double-checked TTL cache).

The TTL check runs under ``_filters_lock`` but the network fetch must not hold
it, so a naive implementation lets N threads that all miss the TTL together
issue N identical exchangeInfo requests. The fix gates the miss path per key
and re-checks the cache inside the gate, so a concurrent first fetch is shared
by every waiter.
"""

from __future__ import annotations

import threading
import time

import pytest

from yialpha.dataflows import binance_filters as bf


def _exchange_info_payload(symbol: str = "BTCUSDT") -> dict:
    return {
        "symbols": [
            {
                "symbol": symbol,
                "status": "TRADING",
                "filters": [
                    {"filterType": "PRICE_FILTER", "tickSize": "0.10"},
                    {
                        "filterType": "LOT_SIZE",
                        "stepSize": "0.001",
                        "minQty": "0.001",
                        "maxQty": "1000",
                    },
                    {"filterType": "NOTIONAL", "notional": "5.0"},
                ],
                "pricePrecision": 2,
                "quantityPrecision": 3,
            }
        ]
    }


@pytest.fixture(autouse=True)
def _fresh_cache():
    bf.reset_for_test()
    yield
    bf.reset_for_test()


@pytest.mark.unit
def test_concurrent_first_fetch_hits_network_once(monkeypatch):
    calls = {"n": 0}
    release = threading.Event()

    def _slow_http(path, params, symbol, canonical, **kwargs):  # noqa: ANN001
        calls["n"] += 1
        # Hold the first fetcher open so every other worker piles up on the
        # per-key gate before the result is published.
        assert release.wait(5), "main thread never released the fetch"
        return _exchange_info_payload(canonical)

    monkeypatch.setattr(bf, "_http_get", _slow_http)

    n_threads = 6
    barrier = threading.Barrier(n_threads, timeout=10)
    results: list = []
    errors: list = []
    start = threading.Event()

    def worker():
        start.wait(5)
        barrier.wait()
        try:
            results.append(bf.get_symbol_filters("BTCUSDT"))
        except Exception as exc:  # noqa: BLE001 — surfaced via the assert below
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(n_threads)]
    for t in threads:
        t.start()
    start.set()

    # Wait until the (single) in-flight fetch is being held open, then let it
    # complete. If more than one thread had slipped past the gate, calls would
    # exceed 1 while we wait.
    deadline = time.monotonic() + 5
    while calls["n"] < 1 and time.monotonic() < deadline:
        time.sleep(0.005)
    release.set()

    for t in threads:
        t.join(timeout=15)

    assert errors == []
    assert calls["n"] == 1, "concurrent first lookups must share one request"
    assert len(results) == n_threads
    assert all(r == results[0] for r in results)
    assert results[0].symbol == "BTCUSDT"
    assert str(results[0].tick_size) == "0.10"


@pytest.mark.unit
def test_second_lookup_within_ttl_does_not_refetch(monkeypatch):
    calls = {"n": 0}

    def _counting_http(path, params, symbol, canonical, **kwargs):  # noqa: ANN001
        calls["n"] += 1
        return _exchange_info_payload(canonical)

    monkeypatch.setattr(bf, "_http_get", _counting_http)
    first = bf.get_symbol_filters("ETHUSDT")
    second = bf.get_symbol_filters("ETHUSDT")
    assert calls["n"] == 1
    assert second == first


@pytest.mark.unit
def test_different_symbols_do_not_block_each_other(monkeypatch):
    """The gate is per-key: a slow BTCUSDT fetch must not serialize a
    concurrent ETHUSDT miss behind it."""
    gate_open = threading.Event()

    def _http(path, params, symbol, canonical, **kwargs):  # noqa: ANN001
        if canonical == "BTCUSDT":
            assert gate_open.wait(5)
        return _exchange_info_payload(canonical)

    monkeypatch.setattr(bf, "_http_get", _http)

    done = {}

    def fetch(name):
        done[name] = bf.get_symbol_filters(name)

    t_btc = threading.Thread(target=fetch, args=("BTCUSDT",))
    t_btc.start()
    time.sleep(0.05)  # let BTCUSDT enter its slow fetch first
    t_eth = threading.Thread(target=fetch, args=("ETHUSDT",))
    t_eth.start()
    t_eth.join(5)
    assert "ETHUSDT" in done  # completed while BTCUSDT still blocked
    gate_open.set()
    t_btc.join(5)
    assert "BTCUSDT" in done
