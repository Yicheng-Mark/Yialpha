"""Round-12 adversarial sweep regression pins (2026-09-30, USDT-perp track).

Six read-only shards scanned the perp main path (vendor core / vision+bundle /
agent wiring / risk overlay+ticket / backtest+accuracy / telemetry+logs);
19 findings were verified by hand and fixed. This file pins every fix except
the two whose fixtures live elsewhere:

* ``test_node_degrades_when_bundle_fetch_fails`` (positioning honest-skip,
  C2) — updated in ``tests/test_positioning_analyst.py``;
* the vision no-data backoff disclosure (B4 repeat-query) — pinned in
  ``tests/test_binance_vision.py`` next to its archive fixture.

Findings index (shard-letter = scanning shard):
  B1 [P1] adlQuantile missing symbol param -> all-symbols [0] rendered
  B2 [P3] chg_1d/chg_7d/ATR on a duplicated-tail kline frame
  A1 [P3] negative YIALPHA_BINANCE_HTTP_RETRIES -> zero attempts crash
  A2 [P3] LSR per-series leg: raw transport error discarded fetched series
  A3 [P3] unwindowed OI look-back > retention claims "last N days"
  F1 [P2] Square base-asset mangling of hyphenated legacy-crypto tickers
  F2 [P2] dual-form trade_date ("YYYY-MM-DD HH:MM:00") kills F&G/regime
         cues and makes the states-log filename illegal on Windows
  F3 [P3] onchain None payload cached for the full 6h TTL
  F4 [P3] router ledger: perp price-book failures recorded optional-kind
         while classify_quality graded them DEGRADED_CRITICAL
  C1 [P2] run_cached check-then-set: fanout double-fetch of the shared
         perp bundle (nondeterministic live snapshots, doubled sentinels)
  C3 [P3] structured free-text fallback: empty content -> silent empty
         report (billed, then truthiness-dropped everywhere)
  E1 [P2] event-study CAR window ignored holding_days (docstring promised
         "the holding window")
  E2 [P3] funding coverage: unconditional final-bar exemption silently
         zero-dragged a fully-historical window's last day
  E3 [P3] Binance-shaped benchmark override fetched via yfinance
         (BTCUSDT -> BTC-USD silent instrument substitution)
  D1 [P2] trade_ticket ignored the exchange initialLeverage bracket cap
         (script leverage contradicted the same report's overlay)
  B3 [P3] vision store lockless reader: Windows sharing violation
         misclassified as "partition unreadable - delete it"
  B5 [P3] bookDepth multi-year load materialized tens of millions of rows
         before any cap (no memory budget)

Zero network, zero LLM. Config is reset per test by conftest.
"""

from __future__ import annotations

import importlib.util
import threading
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import requests

import yialpha.backtest.engine as engine
import yialpha.dataflows.binance as bn
import yialpha.dataflows.binance_vision_store as bvs
import yialpha.dataflows.perp_bundle as pb
from tests.test_backtest_engine import (
    FakeGraph,
    _decision_dates,
    _flat_index,
    _rising_prices,
)
from tests.test_perp_bundle import _kline_frame
from tests.test_perp_engine import _PERP_KW, _funding, _funding_dropping
from yialpha.agents.utils.structured import (
    EMPTY_RESPONSE_SENTINEL,
    STRUCTURED_FALLBACK,
    invoke_structured_or_freetext,
)
from yialpha.dataflows import quality
from yialpha.dataflows.binance_square import base_asset_candidates
from yialpha.dataflows.config import set_config
from yialpha.dataflows.errors import NoMarketDataError
from yialpha.dataflows.fear_greed import _select_pit
from yialpha.dataflows.run_scope import (
    ensure_run_scope,
    reset_run_scope,
    run_cached,
)

_tt_spec = importlib.util.spec_from_file_location(
    "trade_ticket_round12_test",
    Path(__file__).resolve().parents[1] / "scripts" / "trade_ticket.py",
)
tt = importlib.util.module_from_spec(_tt_spec)
_tt_spec.loader.exec_module(tt)


# --------------------------------------------------------------------------- #
# B1 [P1] — adlQuantile must carry the symbol param
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_adl_request_params_carry_the_symbol():
    """The symbol param is OPTIONAL on /fapi/v1/adlQuantile — an empty params
    dict serves the ALL-symbols list whose [0] entry is an arbitrary symbol.
    The existing pin's comment claimed "params carry the canonical symbol"
    but only asserted the function ARG; this pins the actual request."""
    import os

    import yialpha.dataflows.binance_brackets as bb

    saved_key, saved_secret = (
        os.environ.get("BINANCE_API_KEY"),
        os.environ.get("BINANCE_API_SECRET"),
    )
    os.environ["BINANCE_API_KEY"] = "test-key"
    os.environ["BINANCE_API_SECRET"] = "test-secret"
    seen = {}

    def fake_signed_get(path, params, symbol, canonical):
        seen.update(path=path, params=dict(params))
        return [{"symbol": canonical, "adlQuantile": {"LONG": "1", "SHORT": 2}}]

    orig = bb.signed_fapi_get
    bb.signed_fapi_get = fake_signed_get
    try:
        out = pb._fetch_adl("BTCUSDT")
    finally:
        bb.signed_fapi_get = orig
        for k, v in (("BINANCE_API_KEY", saved_key),
                     ("BINANCE_API_SECRET", saved_secret)):
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    assert out["status"] == pb.STATUS_OK
    assert seen["params"].get("symbol") == "BTCUSDT"  # the round-12 pin


@pytest.mark.unit
def test_adl_selects_matching_entry_from_all_symbols_response(monkeypatch):
    """Defense in depth: even if a future regression drops the param again,
    the entry whose symbol MATCHES the analyzed contract is selected from
    the all-symbols list — never a blind data[0]."""
    import yialpha.dataflows.binance_brackets as bb

    monkeypatch.setenv("BINANCE_API_KEY", "test-key")
    monkeypatch.setenv("BINANCE_API_SECRET", "test-secret")

    def fake_signed_get(path, params, symbol, canonical):  # noqa: ARG001
        return [
            {"symbol": "ETHUSDT", "adlQuantile": {"LONG": "3", "SHORT": 3}},
            {"symbol": "BTCUSDT", "adlQuantile": {"LONG": "1", "SHORT": 2}},
        ]

    monkeypatch.setattr(bb, "signed_fapi_get", fake_signed_get)
    out = pb._fetch_adl("BTCUSDT")
    assert out["status"] == pb.STATUS_OK
    assert out["long"] == 1 and out["short"] == 2  # BTCUSDT's, not ETHUSDT's


# --------------------------------------------------------------------------- #
# B2 [P3] — chg/ATR on a duplicated-tail frame
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_chg_and_atr_survive_duplicated_tail_date(monkeypatch):
    """The round-11 duplicate-tail defence covered only the basis pairing;
    chg_1d/chg_7d/ATR ran on the raw frame — with the tail date twice,
    ``iloc[-2]`` was the SAME date's earlier page, so chg_1d reported a
    phantom same-day "change" into the analyst prompt."""
    base = [100.0 + i for i in range(29)]  # 2026-07-01..07-29
    dup_last = pd.concat([
        _kline_frame(base),
        _kline_frame([111.0], start="2026-07-30"),
        _kline_frame([199.0], start="2026-07-30"),
    ])
    assert dup_last.index.duplicated().sum() == 1  # fixture really is dup
    clean_tail = _kline_frame(base + [199.0])

    def fake_klines(symbol, start, end, interval="1d", venue="binance_perp",
                    price_type="last"):  # noqa: ARG001
        return dup_last if price_type == "last" else clean_tail

    monkeypatch.setattr(pb, "binance_klines_frame", fake_klines)
    out = pb._fetch_prices("BTCUSDT", "2026-08-18")
    assert out["last"]["close"] == pytest.approx(199.0)
    # prev day is 128.0 (07-29) — the buggy path read 111.0 (the SAME date's
    # earlier page) and reported 199/111 as a "1-day change".
    assert out["last"]["chg_1d"] == pytest.approx(199.0 / 128.0 - 1.0)
    assert out["last"]["chg_7d"] == pytest.approx(199.0 / 122.0 - 1.0)
    assert out["last"]["atr_14"] is not None  # ATR window deduped too


# --------------------------------------------------------------------------- #
# A1 [P3] — negative binance_http_retries
# --------------------------------------------------------------------------- #


class _FakeResp:
    def __init__(self, status_code=200, json_data=None, text=""):
        self.status_code = status_code
        self.text = text
        self._json = json_data

    def json(self):
        return self._json


@pytest.mark.unit
def test_negative_http_retries_clamped_to_zero(monkeypatch):
    """-1 made ``range(max_retries + 1)`` execute zero attempts and crash
    every Binance call inside the transport layer with an untyped error
    naming neither the env var nor the vendor. Negative now means exactly
    what 0 means: single attempt, raw propagation."""
    set_config({"binance_http_retries": -1})
    monkeypatch.setattr(
        bn.requests, "get", lambda url, **k: _FakeResp(200, [[1]], text="[[1]]"),
    )
    assert bn._http_get("/fapi/v1/klines", {"symbol": "X"}, "X", "X") == [[1]]

    scripted = []

    def conn_error(url, **k):
        scripted.append(url)
        raise requests.ConnectionError("dns boom")

    monkeypatch.setattr(bn.requests, "get", conn_error)
    with pytest.raises(requests.ConnectionError):
        bn._http_get("/fapi/v1/klines", {"symbol": "X"}, "X", "X")
    assert len(scripted) == 1  # no retries, same as retries=0


# --------------------------------------------------------------------------- #
# A2 [P3] — LSR per-series leg vs raw transport error
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_lsr_raw_transport_error_degrades_one_series_only(monkeypatch):
    """retries=0 (default) re-raises RAW transport errors; the per-series
    except tuple missed requests.RequestException, so one blip on the
    second leg discarded the already-fetched first series through the
    router's generic catch — violating the "only an outright failure of
    all three raises" contract."""
    rows = [{"timestamp": 1_700_000_000_000, "longAccount": 0.6,
             "longShortRatio": 1.5, "shortAccount": 0.4}]

    def fake_http(path, params, symbol, canonical):  # noqa: ARG001
        if path.endswith("topLongShortPositionRatio"):
            raise requests.ConnectionError("mid-response drop")
        return list(rows)

    monkeypatch.setattr(bn, "_http_get", fake_http)
    out = bn.get_binance_long_short_ratio("BTCUSDT")
    assert "top_account" in out and "global_account" in out
    assert "unavailable series: top_position" in out


# --------------------------------------------------------------------------- #
# A3 [P3] — unwindowed OI look-back beyond retention
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_oi_unwindowed_beyond_retention_discloses_cap(monkeypatch):
    """look_back_days=90 unwindowed: the endpoint retains only the last 30
    days, so ~31 rows served under a header claiming "last 90 days + live"
    with no retention note (that note was windowed-only)."""
    hist = [
        {"timestamp": i * 86_400_000, "sumOpenInterest": "12.5",
         "sumOpenInterestValue": "125"}
        for i in range(31)
    ]

    def fake_http(path, params, symbol, canonical):  # noqa: ARG001
        if path.endswith("openInterest"):
            return {"openInterest": "12.5"}
        return hist

    monkeypatch.setattr(bn, "_http_get", fake_http)
    out = bn.get_binance_open_interest("BTCUSDT", look_back_days=90)
    assert "last 90 days + live" in out  # the (unchanged) request framing
    assert "retains the last 30 days only (90 requested)" in out  # the cap
    # Within retention, nothing changes: no disclosure on a 30-day ask.
    out30 = bn.get_binance_open_interest("BTCUSDT", look_back_days=30)
    assert "retains the last 30 days only" not in out30


# --------------------------------------------------------------------------- #
# F1 [P2] — Square base-asset for hyphenated legacy-crypto tickers
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_square_base_asset_candidates_hyphenated_forms():
    """Legacy-crypto tickers arrive yfinance-shaped ("BTC-USD"); stripping
    "USD" across the hyphen left base "BTC-" that can never match a code
    or cashtag — the block always rendered "0 posts" for the most-
    discussed assets on the platform."""
    assert base_asset_candidates("BTC-USD") == ("BTC",)
    assert base_asset_candidates("BTC-USDT") == ("BTC",)
    assert base_asset_candidates("1000PEPE-USDT") == ("1000PEPE", "PEPE")
    # Existing compact behaviour unchanged.
    assert base_asset_candidates("BTCUSDT") == ("BTC",)
    assert base_asset_candidates("1000PEPEUSDT") == ("1000PEPE", "PEPE")


# --------------------------------------------------------------------------- #
# F2 [P2] — dual-form trade_date
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_fear_greed_select_pit_accepts_dual_form_date():
    """The bare %Y-%m-%d strptime raised on live programmatic entries
    ("2026-08-01 16:00:00" — the propagate() form documented in the
    sentiment/news analysts), killing the cue and mislabelling it a fetch
    failure."""
    entries = [
        {"ts": 1_753_900_000, "value": 40, "label": "Fear"},     # 07-31
        {"ts": 1_753_986_400, "value": 55, "label": "Greed"},    # 08-01
    ]
    for date_form in ("2026-08-01", "2026-08-01 16:00:00"):
        cur, prev = _select_pit(entries, date_form)
        assert cur["value"] == 55 and prev["value"] == 40


@pytest.mark.unit
def test_market_regime_binance_branch_accepts_dual_form_date(monkeypatch):
    import yialpha.dataflows.binance as binance_mod
    import yialpha.dataflows.market_regime as mr

    captured = {}

    def fake_frame(*args, **kwargs):  # noqa: ARG001
        captured.update(symbol=args[0], start=args[1], end=args[2])
        idx = pd.bdate_range(args[1], args[2])
        return pd.DataFrame(
            {"Close": [100.0 + i for i in range(len(idx))]}, index=idx,
        )

    monkeypatch.setattr(binance_mod, "binance_klines_frame", fake_frame)
    # The lookback strptime + the klines end edge must both take the date
    # part (the old code raised ValueError before any fetch).
    mr._load_regime_ohlcv("BTCUSDT", "2026-08-01 16:00:00", "crypto_perp")
    assert captured["end"] == "2026-08-01"
    from datetime import timedelta

    assert captured["start"] == (
        datetime(2026, 8, 1) - timedelta(days=500)
    ).strftime("%Y-%m-%d")  # 500 calendar days back


@pytest.mark.unit
def test_states_log_stem_date_parts_timestamp_form():
    """A timestamped trade_date embedded verbatim made the filename ILLEGAL
    on Windows (colons) — the run's entire on-disk evidence silently never
    wrote. The stem takes the date part; perp/spot suffixes unchanged."""
    from yialpha.graph.trading_graph import YiAlphaGraph

    stem = YiAlphaGraph._states_log_stem
    assert stem("2026-08-01 16:00:00", "crypto_perp") == "full_states_log_2026-08-01_perp"
    assert stem("2026-08-01 16:00:00", "crypto_spot") == "full_states_log_2026-08-01_spot"
    assert stem("2026-08-01", "stock") == "full_states_log_2026-08-01"
    assert (
        stem(datetime(2026, 8, 1, 16, 0), "crypto_perp")
        == "full_states_log_2026-08-01_perp"
    )


# --------------------------------------------------------------------------- #
# F3 [P3] — onchain null payload must not be cached
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_onchain_null_payload_not_cached(tmp_path):
    """A 200 with a valid-JSON NON-dict body returned None from the fetch,
    and the wrapper serialized {"payload": null} — caching the failure for
    the full 6h TTL (one transient maintenance page -> 6h outage). The
    fetch slot now raises, so fail_open leaves the entry uncached."""
    import yialpha.dataflows.onchain_flows as oc

    set_config({"data_cache_dir": str(tmp_path)})
    resp = SimpleNamespace(
        raise_for_status=lambda: None, text='["not", "a", "dict"]',
    )
    monkey = pytest.MonkeyPatch()
    monkey.setattr(oc, "requests_get", lambda *a, **k: resp)
    try:
        assert oc._fetch_chart_cached("n-unique-addresses") is None
    finally:
        monkey.undo()
    # Nothing was written: the next call refetches instead of replaying the
    # outage for 6 hours.
    assert not list(tmp_path.rglob("*.json"))


# --------------------------------------------------------------------------- #
# F4 [P3] — router ledger severity for the perp price book
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_sentinel_is_core_predicate():
    assert quality.sentinel_is_core("binance_perp", "get_binance_klines", "")
    assert quality.sentinel_is_core("binance_perp", "get_binance_indicators", "")
    # Index-kline misses stay auxiliary (settlement anchor, not the traded
    # price book).
    assert not quality.sentinel_is_core("binance_perp", "get_binance_klines", "index")
    # Enrichment legs stay optional.
    assert not quality.sentinel_is_core("binance_perp", "get_binance_funding_rate", "")
    # Stock core categories unchanged.
    assert quality.sentinel_is_core("core_stock_apis", "get_stock_data", "")
    assert not quality.sentinel_is_core("sec_ownership", "get_sec_form4_filings", "")


@pytest.mark.unit
def test_router_records_core_kind_for_perp_price_book_failure(monkeypatch):
    """The success side already treated perp klines as core inside the
    optional category; the sentinel side recorded optional-unavailable, so
    core_sentinel_count (run_robust's DEGRADED verdict) read 0 while
    classify_quality NO_TRADE'd the same evidence as DEGRADED_CRITICAL."""
    from yialpha.dataflows import interface as iface

    def boom(*a, **k):
        raise RuntimeError("fapi down")

    monkeypatch.setattr(iface, "get_vendor", lambda category, m: "default")

    def _kinds(method):
        quality.reset_quality()
        saved = dict(iface.VENDOR_METHODS[method])
        try:
            iface.VENDOR_METHODS[method] = dict.fromkeys(saved, boom)
            out = iface.route_to_vendor(method, "BTCUSDT", "2026-01-01", "2026-01-31")
        finally:
            iface.VENDOR_METHODS[method] = saved
        kinds = [e["kind"] for e in quality.snapshot_quality()]
        summary = quality.summarize_quality(
            quality.snapshot_quality(), quality.snapshot_core_successes(),
        )
        return out, kinds, summary

    out, kinds, summary = _kinds("get_binance_klines")
    assert out.startswith("DATA_UNAVAILABLE")  # return-vs-raise unchanged
    assert kinds == [quality.KIND_CORE_ERROR]
    assert summary["core_sentinel_count"] == 1  # run_robust now sees it

    out, kinds, summary = _kinds("get_binance_funding_rate")
    assert kinds == [quality.KIND_OPTIONAL_UNAVAILABLE]  # enrichment intact
    assert summary["core_sentinel_count"] == 0


@pytest.mark.unit
def test_router_records_no_data_kind_for_perp_klines(monkeypatch):
    from yialpha.dataflows import interface as iface

    def no_data(*a, **k):
        raise NoMarketDataError("BTCUSDT", "BTCUSDT", "no rows anywhere")

    monkeypatch.setattr(iface, "get_vendor", lambda category, m: "default")
    quality.reset_quality()
    saved = dict(iface.VENDOR_METHODS["get_binance_klines"])
    try:
        iface.VENDOR_METHODS["get_binance_klines"] = dict.fromkeys(saved, no_data)
        out = iface.route_to_vendor(
            "get_binance_klines", "BTCUSDT", "2026-01-01", "2026-01-31",
        )
    finally:
        iface.VENDOR_METHODS["get_binance_klines"] = saved
    assert out.startswith("NO_DATA_AVAILABLE")
    kinds = [e["kind"] for e in quality.snapshot_quality()]
    assert kinds == [quality.KIND_NO_DATA]  # was KIND_OPTIONAL_UNAVAILABLE


# --------------------------------------------------------------------------- #
# C1 [P2] — run_cached single-flight
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_run_cached_single_flight_under_threads():
    """The analyst fanout runs market + positioning subgraphs concurrently
    over ONE shared perp-bundle key; the old check-then-set let both fetch
    (duplicate live endpoints sampled seconds apart inside one decision —
    nondeterminism serial mode cannot produce — plus doubled sentinels)."""
    ensure_run_scope()
    try:
        import contextvars
        import time

        calls = []
        lock = threading.Lock()

        def slow_fetch():
            with lock:
                calls.append(1)
            time.sleep(0.15)
            return {"bundle": "bytes"}

        results = []
        barrier = threading.Barrier(8)

        def worker():
            barrier.wait()  # maximize the check-then-set collision window
            results.append(
                run_cached(("perp_market_bundle", "BTCUSDT", "2026-09-30"), slow_fetch)
            )

        # Production shape: submit_with_context gives each worker a COPY of
        # the parent context — plain Thread.start does NOT inherit
        # contextvars, so a bare thread would degrade to direct calls.
        contexts = [contextvars.copy_context() for _ in range(8)]
        threads = [
            threading.Thread(target=c.run, args=(worker,)) for c in contexts
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(calls) == 1  # fetched exactly once (the doc's contract)
        assert all(r == {"bundle": "bytes"} for r in results)
    finally:
        reset_run_scope()


# --------------------------------------------------------------------------- #
# C3 [P3] — structured free-text empty content
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_freetext_empty_content_gets_sentinel():
    """A thinking model can answer a plain invoke entirely in reasoning
    content; the empty string flowed out as the report (analyst billed,
    section silently dropped by every consumer's truthiness guard)."""

    class _Plain:
        def __init__(self, content):
            self.content = content

        def invoke(self, prompt):  # noqa: ARG002
            return SimpleNamespace(content=self.content)

    for empty in ("", "   \n\t"):
        out = invoke_structured_or_freetext(
            None, _Plain(empty), "p", lambda r: "x", "T",
        )
        assert out == EMPTY_RESPONSE_SENTINEL

    out, marker = invoke_structured_or_freetext(
        None, _Plain(""), "p", lambda r: "x", "T", extract=lambda r: "v",
    )
    assert out == EMPTY_RESPONSE_SENTINEL
    assert marker is STRUCTURED_FALLBACK

    # Real content passes through untouched.
    assert invoke_structured_or_freetext(
        None, _Plain("real report"), "p", lambda r: "x", "T",
    ) == "real report"


# --------------------------------------------------------------------------- #
# E1 [P2] — event-study window tracks holding_days
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_event_study_window_tracks_holding_days():
    """The opt-in event study always used the module default (0, 20) — a
    21-session drift — while the engine's docstring promised "the holding
    window". The window now follows holding_days and is disclosed."""

    def prices(ticker, start, end):
        if ticker == "SPY":
            return _flat_index(ticker, start, end)
        return _rising_prices(ticker, start, end)

    for holding, expected in ((3, (0, 3)), (10, (0, 10))):
        dates = _decision_dates(6, start="2024-06-01")
        result = engine.run_backtest(
            FakeGraph(dict.fromkeys(dates, "Buy"), benchmark="SPY"),
            "AAPL", dates, holding_days=holding,
            price_provider=prices, compute_index_alpha=True, event_study=True,
        )
        assert result.metrics.event_study_n > 0
        assert result.metrics.event_study_window == expected


@pytest.mark.unit
def test_report_renders_event_study_window():
    from yialpha.backtest.report import render_backtest_report

    metrics = {"event_study_n": 4, "event_study_mean_car": 0.01,
               "event_study_t_stat": 2.0, "event_study_p_value": 0.05,
               "event_study_ci": (0.001, 0.02), "event_study_benchmark": "SPY",
               "event_study_window": (0, 5)}
    out = render_backtest_report(_fake_result_with_metrics(metrics))
    assert "CAR window: [0, +5]" in out


def _fake_result_with_metrics(metrics):
    """Minimal BacktestResult stand-in for the report renderer."""
    from yialpha.backtest.engine import BacktestResult
    from yialpha.backtest.metrics import BacktestMetrics

    m = BacktestMetrics(
        total_return=0.1, cagr=0.1, volatility=0.1, sharpe=1.0,
        sortino=1.0, max_drawdown=0.05, calmar=2.0, deflated_sharpe=1.0,
        alpha_vs_buyhold=0.0, num_periods=10, periods_per_year=252,
    )
    for k, v in metrics.items():
        setattr(m, k, v)
    return BacktestResult(
        ticker="AAPL", initial_capital=100_000.0, holding_days=5,
        equity=[100_000.0], equity_dates=["2024-06-01"], trades=[],
        benchmark_equity=[100_000.0], benchmark_name="SPY",
        metrics=m, config_summary={},
    )


# --------------------------------------------------------------------------- #
# E2 [P3] — funding coverage final-bar exemption narrowed
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_funding_gap_on_final_historical_bar_fails_closed():
    """The final charge bar of a FULLY-HISTORICAL window (no forming day,
    mark_end inside the fetch window) genuinely lacking settlements used to
    slip through the round-11 unconditional exemption and silently accrue
    zero drag — the exact silent-zero failure the gate exists to prevent."""
    with pytest.raises(ValueError, match="missing 1 priced day"):
        engine.run_backtest(
            FakeGraph({"2024-01-01": "Buy"}), "BTCUSDT", ["2024-01-01"],
            asset_type="crypto_perp",
            funding_provider=_funding_dropping(0.001, {"2024-01-15"}),
            **_PERP_KW,
        )
    # Same window with full coverage still runs (the gate did not widen).
    res = engine.run_backtest(
        FakeGraph({"2024-01-01": "Buy"}), "BTCUSDT", ["2024-01-01"],
        asset_type="crypto_perp", funding_provider=_funding(0.001),
        **_PERP_KW,
    )
    assert res.metrics.total_return is not None


# --------------------------------------------------------------------------- #
# E3 [P3] — Binance-shaped benchmark leaves the yfinance leg
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_index_leg_provider_routes_binance_shaped_benchmarks(monkeypatch):
    """BTCUSDT through yfinance silently resolved to the Yahoo spot pair
    BTC-USD while config_summary still said BTCUSDT — alpha and the event
    study market model ran on a different instrument than named."""

    def fake_yf(ticker, start, end):
        return _rising_prices(ticker, start, end)

    def fake_perp(ticker, start, end):
        return _rising_prices(ticker, start, end)

    monkeypatch.setattr(engine, "_yfinance_price_provider", fake_yf)
    routed = engine._index_leg_provider("BTCUSDT", fake_yf, fake_perp, True)
    assert routed is fake_perp
    # Non-Binance names and non-swapped runs keep the original leg.
    assert engine._index_leg_provider("SPY", fake_yf, fake_perp, True) is fake_yf
    assert engine._index_leg_provider("BTCUSDT", fake_yf, fake_perp, False) is fake_yf
    assert engine._index_leg_provider("000300.SS", fake_yf, fake_perp, True) is fake_yf


# --------------------------------------------------------------------------- #
# D1 [P2] — trade_ticket respects the exchange bracket cap
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_build_ticket_respects_bracket_leverage_cap(tmp_path, monkeypatch):
    """All three runtime consumers thread advisory_leverage_cap into
    compute_leverage; the script alone did not — a 5x-bracket contract got
    a 10x ticket (half the true margin, a liquidation price computed at the
    wrong leverage) contradicting the same report's overlay bullet."""
    from tests.test_trade_ticket import PERP_OVERLAY, _write_report

    decision = "**Rating**: Buy\n\nMomentum continuation.\n\n" + PERP_OVERLAY
    report_dir = _write_report(tmp_path, decision)

    monkeypatch.setattr(tt, "advisory_leverage_cap", lambda t, n=None: 5.0)
    capped = tt.build_ticket(report_dir, capital=10_000.0, profile="moderate")
    assert capped["status"] == "ok"
    assert capped["leverage"] == pytest.approx(5.0)
    assert capped["leverage_caps"]["L_bracket"] == pytest.approx(5.0)

    monkeypatch.setattr(tt, "advisory_leverage_cap", lambda t, n=None: None)
    uncapped = tt.build_ticket(report_dir, capital=10_000.0, profile="moderate")
    assert uncapped["leverage"] == pytest.approx(10.0)  # conviction cap binds
    # 5x margin is exactly double the 10x margin on the same notional.
    assert capped["margin"] == pytest.approx(uncapped["margin"] * 2.0)
    # The cap the report's overlay would have shown is the one honoured.
    assert capped["leverage"] < uncapped["leverage"]


# --------------------------------------------------------------------------- #
# B3 [P3] — vision store reader retry
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_partition_read_retries_sharing_violation(monkeypatch):
    """A lockless reader can lose the Windows sharing-violation race with a
    concurrent writer's os.replace; the transient PermissionError was
    classified "partition unreadable — delete it" for a healthy partition.
    Retry once and the read succeeds."""
    calls = []

    def scripted_read(path, **kwargs):  # noqa: ARG001
        calls.append(path)
        if len(calls) == 1:
            raise PermissionError(
                13, "The process cannot access the file (sharing violation)"
            )
        return pd.DataFrame({"a": [1, 2]})

    monkeypatch.setattr(bvs.pd, "read_csv", scripted_read)
    frame = bvs._read_partition_with_retry(Path("dummy.csv.gz"))
    assert list(frame["a"]) == [1, 2]
    assert len(calls) == 2

    # A persistent violation still raises after the budget (fail honestly,
    # and the caller's "unreadable" verdict is then genuinely true).
    monkeypatch.setattr(
        bvs.pd, "read_csv",
        lambda path, **k: (_ for _ in ()).throw(PermissionError("locked")),
    )
    with pytest.raises(PermissionError):
        bvs._read_partition_with_retry(Path("dummy.csv.gz"))


# --------------------------------------------------------------------------- #
# B5 [P3] — bookDepth memory budget
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_window_row_estimate_math(tmp_path):
    set_config({"data_cache_dir": str(tmp_path)})
    store = bvs.VisionStore()
    store._dir("bookDepth", "BTCUSDT").mkdir(parents=True, exist_ok=True)
    store._write_manifest(
        "bookDepth", "BTCUSDT",
        {"days": {"2024-01-01": 10, "2024-01-02": 20,
                  "2024-01-03": 30, "2024-01-04": 40},
         "missing": {}, "updated": "x"},
    )
    est = store.window_row_estimate(
        "bookDepth", "BTCUSDT",
        datetime(2024, 1, 1), datetime(2024, 1, 6),
    )
    # 4 known days sum 100; 2 unknown days x median(10,20,30,40)=30 -> 160.
    assert est == 160
    # A window the manifest knows nothing about estimates nothing.
    assert store.window_row_estimate(
        "bookDepth", "BTCUSDT", datetime(2025, 1, 1), datetime(2025, 1, 6),
    ) is None


@pytest.mark.unit
def test_bookdepth_row_budget_guard(monkeypatch):
    """A multi-year bookDepth call materialized tens of millions of rows
    before any cap; oversized windows now refuse up front with the budget
    and the remedy in the error."""
    import yialpha.dataflows.binance_vision as bnv

    class _FatStore:
        def __init__(self, *a, **k):
            pass

        def window_row_estimate(self, *a, **k):
            return 99_999_999

    monkeypatch.setattr(bvs, "VisionStore", _FatStore)
    with pytest.raises(NoMarketDataError, match="memory budget"):
        bnv.get_binance_vision_book_depth("BTCUSDT", "2022-01-01", "2024-12-31")
