"""Round-13 regression pins — binance/dataflows shard (standalone file).

The round-13 adversarial sweep's dataflows fixes live here, separate from
the backtest/misc shards in tests/test_round13_regressions.py — concurrent
full-file appends to that one shared file truncated it once already, so
each shard owns its own file this round.

Four verified fixes pinned here:

  D1 [P3] perp_bundle._fetch_adl: zero-match adlQuantile responses fell
         back to data[0] (a wrong contract's quantiles rendered as ok)
  D2 [P3] get_binance_spot_perp_basis: default look_back_days=7 window
         spanned 8 days under a "last 7 days" header (round-12 funding
         off-by-one, tool-layer twin)
  D3 [P3] coingecko/defillama _PARSE_MEMO: unbounded growth (no FIFO cap
         like binance's _HISTORY_MEMO_MAX)
  D4 [P3] warm_equity_perp_bases: exchangeInfo request inside the data lock
         froze every cache-only equity-perp reader for the request duration

Zero network, zero LLM. Config is reset per test by conftest.
"""
from __future__ import annotations

import json
import threading
import time
from datetime import UTC, datetime, timedelta

import pytest

import yialpha.dataflows.binance as bn
import yialpha.dataflows.perp_bundle as pb
from yialpha.dataflows import coingecko_vendor as cg, defillama_vendor as dl

# --------------------------------------------------------------------------- #
# D1 [P3] — adlQuantile zero-match must degrade, never render data[0]
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_adl_zero_match_response_degrades_unavailable(monkeypatch):
    """D1 [P3]: the round-12 symbol matcher still fell back to ``data[0]``
    when NO entry matched the canonical contract — a wrong-contract payload
    (endpoint regression / mangling proxy) rendered some OTHER contract's
    quantiles as this one's ok-status ADL. Zero matches must degrade the leg
    to unavailable (the assembler's optional-unavailable note) naming the
    contract we asked for — never a wrong-contract ok."""
    import yialpha.dataflows.binance_brackets as bb

    monkeypatch.setenv("BINANCE_API_KEY", "test-key")
    monkeypatch.setenv("BINANCE_API_SECRET", "test-secret")

    def fake_signed_get(path, params, symbol, canonical):  # noqa: ARG001
        return [
            {"symbol": "ETHUSDT", "adlQuantile": {"LONG": "3", "SHORT": 3}},
            {"symbol": "SOLUSDT", "adlQuantile": {"LONG": "2", "SHORT": 2}},
        ]

    monkeypatch.setattr(bb, "signed_fapi_get", fake_signed_get)
    out = pb._fetch_adl("BTCUSDT")
    assert out["status"] == pb.STATUS_UNAVAILABLE  # degraded, never ok
    assert "BTCUSDT" in out["reason"]  # names the contract we asked for
    # data[0]'s ETHUSDT quantiles must not leak through as this leg's —
    # the unavailable dict carries status+reason only, no long/short.
    assert "long" not in out and "short" not in out


# --------------------------------------------------------------------------- #
# D2 [P3] — spot-perp basis default window vs its "last N days" label
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_spot_perp_basis_default_window_matches_label(monkeypatch):
    """D2 [P3]: look_back_days=7 built the default window start at
    ``end - 7d`` — an 8-day end-inclusive span fetched and rendered under a
    header claiming "last 7 days" (the tool-layer twin of the round-12
    funding off-by-one). The start is now limit-1 days back, so the default
    ask fetches AND renders exactly 7 daily rows."""
    end = datetime(2026, 9, 30, tzinfo=UTC)
    windows = {}

    def fake_paginate(path, params, page_limit, cursor_of, start_ms, end_ms,
                      symbol, canonical, base=None, weight_key=None):  # noqa: ARG001
        windows[path] = (start_ms, end_ms)
        # Honest-klines fake: daily bars filling the REQUESTED window only
        # (so the test measures the window we ask for, not what we render).
        days = (end_ms - start_ms) // 86_400_000 + 1  # end-of-day inclusive
        rows = []
        for i in range(days):
            open_ms = int((end - timedelta(days=days - 1 - i)).timestamp() * 1000)
            close = str(100.0 + i)
            rows.append([open_ms, "0", "0", "0", close, "0",
                         open_ms, "0", 0, "0", "0", "0"])
        return rows

    monkeypatch.setattr(bn, "_paginate_history", fake_paginate)
    out = bn.get_binance_spot_perp_basis("BTCUSDT", end_date="2026-09-30")

    # The request span: exactly 7 calendar days end-inclusive (D-6 .. D).
    start_ms, end_ms = windows["/fapi/v1/klines"]
    assert (end_ms - start_ms) // 86_400_000 == 6  # was 7 pre-fix (8 days)
    assert windows["/api/v3/klines"] == windows["/fapi/v1/klines"]  # aligned

    # The render: header label and row count agree on 7.
    assert "last 7 days" in out
    lines = [ln for ln in out.splitlines() if ln and not ln.startswith("#")]
    assert len(lines) == 8  # CSV column row + 7 data rows
    assert lines[1].startswith("2026-09-24")  # D-6: end-inclusive start
    assert lines[-1].startswith("2026-09-30")


# --------------------------------------------------------------------------- #
# D3 [P3] — vendor parse-memo FIFO cap (coingecko + defillama)
# --------------------------------------------------------------------------- #


@pytest.mark.unit
@pytest.mark.parametrize("mod", [cg, dl], ids=["coingecko", "defillama"])
def test_parse_memo_fifo_cap_evicts_oldest_inserted(mod, monkeypatch, tmp_path):
    """D3 [P3]: ``_PARSE_MEMO`` had no bound — the key carries the cache dir,
    so anything that rotates ``data_cache_dir`` (or an unbounded artifact
    fan-out) grew it for the whole process lifetime. Same FIFO cap as
    binance's ``_HISTORY_MEMO_MAX``: the OLDEST INSERTED key is evicted at
    64, present keys refresh in place, and mtime-hit semantics are intact."""
    assert mod._PARSE_MEMO_MAX == 64  # the binance precedent's value
    mod._PARSE_MEMO.clear()
    cache = tmp_path / "vc"
    cache.mkdir()

    def fake_cached_or_fetch(cache_dir, filename, fetch, ttl_days=None,  # noqa: ARG001
                             vendor=None):  # noqa: ARG001
        return json.dumps({"file": filename}).encode("utf-8")

    monkeypatch.setattr(mod, "cached_or_fetch", fake_cached_or_fetch)
    monkeypatch.setattr(mod, "_cache_dir", lambda: str(cache))

    def _key(i: int) -> tuple[str, str]:
        return (str(cache), f"artifact{i:02d}.json")

    # Real files on disk -> real mtimes, so the memo-HIT branch is exercised
    # (the -1.0 no-file sentinel never memo-hits by design).
    parsed_by_key: dict[tuple[str, str], object] = {}
    for i in range(65):
        name = f"artifact{i:02d}.json"
        (cache / name).write_text('{"x": 1}')
        parsed_by_key[_key(i)] = mod._cached_json(name, lambda: {}, 1.0)

    # The 65th insert evicted the OLDEST (artifact00); the cap holds.
    assert len(mod._PARSE_MEMO) == mod._PARSE_MEMO_MAX
    assert _key(0) not in mod._PARSE_MEMO
    assert _key(1) in mod._PARSE_MEMO

    # A still-cached key HITS on mtime: the memo serves the SAME parsed
    # object (the memo skips the json.loads re-parse; cached_or_fetch is
    # still consulted for the disk read — that is its contract).
    out = mod._cached_json("artifact63.json", lambda: {}, 1.0)
    assert out is parsed_by_key[_key(63)]  # identity: not re-parsed

    # The evicted key honestly re-parses (a NEW object, equal by value) and
    # its re-insert evicts the next oldest — FIFO by INSERTION order, not
    # mtime order.
    out = mod._cached_json("artifact00.json", lambda: {}, 1.0)
    assert out == {"file": "artifact00.json"}
    assert out is not parsed_by_key[_key(0)]  # re-parsed, not memo-served
    assert _key(1) not in mod._PARSE_MEMO


# --------------------------------------------------------------------------- #
# D4 [P3] — warm_equity_perp_bases: network fetch outside the data lock
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_warm_equity_perp_bases_reader_not_blocked_by_fetch(monkeypatch):
    """D4 [P3]: the warm ran its exchangeInfo request INSIDE the data lock,
    so every cache-only reader (``equity_perp_bases``) stalled for the
    request's whole duration. The fetch now runs outside the lock
    (binance_filters._fetch_gate pattern): a reader during an in-flight
    fetch is served the previous warmed snapshot immediately, and the swap
    lands atomically once the fetch completes."""
    first = {"symbols": [
        {"symbol": "AAPLUSDT", "quoteAsset": "USDT",
         "underlyingType": "EQUITY", "status": "TRADING"},
    ]}
    second = {"symbols": [
        {"symbol": "TSLAUSDT", "quoteAsset": "USDT",
         "underlyingType": "EQUITY", "status": "TRADING"},
    ]}

    bn.refresh_equity_perp_bases()
    try:
        monkeypatch.setattr(bn, "_http_get", lambda *a, **k: first)
        assert bn.warm_equity_perp_bases() == frozenset({"AAPL"})

        # Age the successful warm past its TTL: the next warm re-fetches.
        monkeypatch.setattr(
            bn, "_EQUITY_PERP_WARMED_AT_MONO",
            time.monotonic() - bn._EQUITY_PERP_WARM_TTL_S - 60.0,
        )

        started, release = threading.Event(), threading.Event()

        def slow_http(path, params, **kw):  # noqa: ARG001
            started.set()
            release.wait(timeout=5.0)  # a stalled exchangeInfo round trip
            return second

        monkeypatch.setattr(bn, "_http_get", slow_http)
        warmer = threading.Thread(target=bn.warm_equity_perp_bases)
        warmer.start()
        assert started.wait(timeout=5.0)

        # Reader while the fetch is in flight: the OLD snapshot, served
        # without waiting for the network call (pre-fix this blocked on
        # the data lock until the 5s release timeout burned down).
        t0 = time.perf_counter()
        got = bn.equity_perp_bases()
        served_in = time.perf_counter() - t0
        assert got == frozenset({"AAPL"})  # stale-but-real, never torn
        assert served_in < 1.0

        release.set()
        warmer.join(timeout=5.0)
        assert not warmer.is_alive()
        # The swap landed once the fetch completed — all-new, all at once.
        assert bn.equity_perp_bases() == frozenset({"TSLA"})
        assert bn.equity_perp_listing_info().keys() == {"TSLA"}
    finally:
        bn.refresh_equity_perp_bases()  # back to seed for later tests


@pytest.mark.unit
def test_warm_equity_perp_bases_single_flight_double_check(monkeypatch):
    """D4 companion: two warmers arriving past the TTL together must still
    collapse to ONE exchangeInfo request — the gate's double-check serves
    the second warmer from the first one's fresh swap (the fetch moving out
    of the data lock must not reintroduce per-thread duplicate fetches)."""
    second = {"symbols": [
        {"symbol": "TSLAUSDT", "quoteAsset": "USDT",
         "underlyingType": "EQUITY", "status": "TRADING"},
    ]}
    fetches: list[str] = []
    gate_arrivals: list[int] = []
    both_at_gate = threading.Event()
    gate = bn._EQUITY_PERP_WARM_GATE

    class _GateCountingWarmers:
        """``_EQUITY_PERP_WARM_GATE`` plus one signal: a warmer enters the
        gate right after its first TTL check, so the second arrival means the
        other warmer has found the snapshot stale as well."""

        def __enter__(self):
            gate_arrivals.append(1)
            if len(gate_arrivals) == 2:
                both_at_gate.set()
            gate.acquire()
            return self

        def __exit__(self, *exc):
            gate.release()

    def slow_http(path, params, **kw):  # noqa: ARG001
        fetches.append(path)
        # Hold the round trip until the other warmer has passed its first TTL
        # check and arrived at the gate. The swap has not landed yet, so that
        # is the arrival order the double-check exists for; a handshake pins
        # it, where a fixed sleep could only make it likely.
        if not both_at_gate.wait(timeout=3.0):
            raise AssertionError("the second warmer never reached the gate")
        return second

    bn.refresh_equity_perp_bases()
    try:
        monkeypatch.setattr(bn, "_http_get", slow_http)
        monkeypatch.setattr(bn, "_EQUITY_PERP_WARM_GATE", _GateCountingWarmers())
        barrier = threading.Barrier(2)
        results: dict[str, frozenset] = {}

        def worker(name):
            barrier.wait()  # maximize the simultaneous-miss window
            results[name] = bn.warm_equity_perp_bases()

        threads = [
            threading.Thread(target=worker, args=(n,)) for n in ("a", "b")
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5.0)
        assert not any(t.is_alive() for t in threads)
        assert sorted(results) == ["a", "b"]  # both warmers returned
        assert len(fetches) == 1  # one request per stale window, not two
        assert results["a"] == results["b"] == frozenset({"TSLA"})
    finally:
        bn.refresh_equity_perp_bases()
