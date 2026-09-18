"""Regression pins for the 2026-09-19 crypto_perp bug-fix batch.

Each test pins one verified defect (see CHANGELOG "Unreleased → Fixed"):

* funding ``sum_7d`` window spanned 8 days (24 settlements at the 8h
  cadence) and over-stated ``sum_7d``/``annualized`` by ~14%;
* one RECOVERABLE klines failure (model typo → instructive sentinel →
  successful retry) permanently vetoed the ticket NO_TRADE because the
  append-only quality ledger had no success-erasure;
* the perp bundle's core-sentinel detail reported failed price legs as
  ``ok`` (key-presence instead of status check);
* the perp bundle's ThreadPoolExecutor dropped the parent contextvars;
* ``reaches_now`` used a single UTC anchor (dual-anchor contract miss);
* accuracy/memory resolution scored perp decisions on the still-forming
  daily bar and priced memory outcomes on the Yahoo SPOT venue;
  tokenized-stock perps (MUUSDT) never resolved at all;
* ``take_profits`` rounded to 6 DECIMALS, destroying micro-price TP
  levels (PEPEUSDT-class ~1e-5 contracts);
* the interactive CLI's streamed path applied the risk overlay with the
  default ``asset_type="stock"`` (perp machinery void), stored decisions
  without the venue tag, and never bound the ledger run context;
* the overlay price/mark failures recorded optional-unavailable while
  their own docstrings (and run_robust's DEGRADED counter) claim core.

Hermetic: every network/clock seam is monkeypatched.
"""

from __future__ import annotations

import importlib.util
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest

import yialpha.dataflows.perp_bundle as pb
from yialpha.accuracy import DecisionRecord, fetch_returns_yf, forward_outcome
from yialpha.dataflows import quality
from yialpha.dataflows.quality import classify_quality, is_perp_core_method
from yialpha.risk.perp_ticket import take_profits


def _frame(closes: list[float], dates: list[str] | None = None) -> pd.DataFrame:
    idx = pd.to_datetime(dates or [f"2026-09-{i:02d}" for i in range(1, len(closes) + 1)])
    return pd.DataFrame(
        {"Close": closes, "Low": closes, "High": closes, "Volume": [1.0] * len(closes)},
        index=idx,
    )


# ---- funding window: exactly 7 days end-inclusive ----------------------------


@pytest.mark.unit
def test_overlay_trailing_funding_window_is_seven_days(monkeypatch):
    """`_trailing_funding_total` (the risk gate's annualized-funding input)
    had the same 8-day window as the bundle's _fetch_funding: dt-7 spans 8
    calendar days (24 settlements), inflating /7*365 by ~14%."""
    from datetime import date as date_cls

    from yialpha.graph.trading_graph import YiAlphaGraph

    captured: dict[str, str] = {}

    def fake_provider(ticker, start, end):
        captured["window"] = (start, end)
        return pd.Series([0.0001] * 21)  # 7 days × 3 settlements

    monkeypatch.setattr(
        "yialpha.backtest.engine._binance_funding_provider", fake_provider
    )
    today = date_cls.today().strftime("%Y-%m-%d")
    total = YiAlphaGraph._trailing_funding_total("BTCUSDT", today)
    start, end = captured["window"]
    assert (date_cls.fromisoformat(end) - date_cls.fromisoformat(start)).days == 6
    assert total == pytest.approx(21 * 0.0001)


@pytest.mark.unit
def test_funding_window_is_seven_days_end_inclusive(monkeypatch):
    """The trailing-7d funding window must span exactly 7 calendar days
    (21 settlements at 8h), not 8 days (24) — ``sum_7d`` and the annualized
    carry divide by 7."""
    captured: dict[str, int] = {}

    def fake_http_get(path, params, *args, **kwargs):
        captured.update(params)
        # 8h cadence over any window the caller asks for
        start = params["startTime"]
        rows = [
            {"fundingTime": start + i * 8 * 3600 * 1000, "fundingRate": "0.0001"}
            for i in range((params["endTime"] - start) // (8 * 3600 * 1000) + 1)
        ]
        return rows

    monkeypatch.setattr(pb, "is_historical_date", lambda d: False)
    monkeypatch.setattr(pb, "_http_get", fake_http_get)

    out = pb._fetch_funding("BTCUSDT", "2026-09-18")

    start_ms = captured["startTime"]
    end_ms = captured["endTime"]
    # [D-6 00:00 UTC, D 23:59:59.999] — 7 calendar days end-inclusive.
    expected_start = int(
        datetime(2026, 9, 12, tzinfo=UTC).timestamp() * 1000
    )
    assert start_ms == expected_start
    assert end_ms - start_ms < 7 * 86_400_000
    assert out["status"] == pb.STATUS_OK
    assert out["settlements"] == 21  # 7 days × 3 settlements, not 8×3
    assert out["annualized"] == pytest.approx(out["sum_7d"] / 7.0 * 365.0)


# ---- quality recovery: a retried-and-succeeded core call must not veto -------


@pytest.mark.unit
def test_classify_quality_downgrades_recovered_core_sentinel():
    events = [
        {
            "method": "get_binance_klines", "kind": quality.KIND_OPTIONAL_UNAVAILABLE,
            "detail": "bad interval 4H",
        },
    ]
    # Without the later success: critical (the historical NO_TRADE trap).
    no_recovery = classify_quality(events, set())
    assert no_recovery["tier"] == "DEGRADED_CRITICAL"
    # With the SAME method having served data later: disclosed as
    # auxiliary "(recovered)" — the price book exists, no veto.
    recovered = classify_quality(events, {"get_binance_klines"})
    assert recovered["tier"] == "DEGRADED_AUXILIARY"
    assert any("(recovered)" in a for a in recovered["auxiliary_degraded"])
    assert recovered["critical_missing"] == []


@pytest.mark.unit
def test_classify_recovery_is_method_scoped():
    """An indicators success must NOT forgive a klines sentinel."""
    events = [
        {
            "method": "get_binance_klines", "kind": quality.KIND_OPTIONAL_UNAVAILABLE,
            "detail": "vendor down",
        },
    ]
    out = classify_quality(events, {"get_binance_indicators"})
    assert out["tier"] == "DEGRADED_CRITICAL"


@pytest.mark.unit
def test_qualified_success_does_not_recover_overlay_sentinel():
    """Round-2 regression pin (the P1-fix-vs-P1-fix collision): perp runs
    ALWAYS carry qualified router klines successes by overlay time, and the
    overlay's own decision-time sentinel is UNqualified and never retried —
    a bare-method recovery rule downgraded it to auxiliary and voided the
    NO_TRADE veto. Recovery keys on (method, qualifier) now."""
    overlay_sentinel = {
        "method": "get_binance_klines", "kind": quality.KIND_CORE_ERROR,
        "detail": "overlay price/ATR unavailable",
    }
    # The analysts' successful klines calls (qualified by price_type):
    successes = {"get_binance_klines[last]", "get_binance_klines[mark]"}
    out = classify_quality(overlay_sentinel and [overlay_sentinel], successes)
    assert out["tier"] == "DEGRADED_CRITICAL"
    assert out["critical_missing"] == [
        "get_binance_klines(core_error)",
    ]


@pytest.mark.unit
def test_summarize_core_count_is_recovery_aware():
    """run_robust's DEGRADED verdict reads core_sentinel_count; a recovered
    sentinel (same method AND qualifier later served) must not re-run a
    completed run — the ticket gate and the orchestrator cannot disagree
    about the same event."""
    events = [
        {
            "method": "get_stock_data", "kind": quality.KIND_NO_DATA,
            "detail": "bad arg",
        },
    ]
    summary = quality.summarize_quality(events, {"get_stock_data"})
    assert summary["core_sentinel_count"] == 0
    assert summary["data_vacuum"] is False
    assert quality.summarize_quality(events, set())["core_sentinel_count"] == 1


@pytest.mark.unit
def test_is_perp_core_method_matches_the_price_book_engines():
    assert is_perp_core_method("get_binance_klines")
    assert is_perp_core_method("get_binance_indicators")
    assert not is_perp_core_method("get_binance_funding_rate")
    assert not is_perp_core_method("get_YFin_history_cached")


@pytest.mark.unit
def test_router_records_success_for_perp_core_methods_in_optional_category(monkeypatch):
    """interface.py must record the success of a perp price-book core method
    served from the (optional) binance_perp category — that record is what
    lets classify_quality tell a recoverable failure from a hard outage."""
    from yialpha.dataflows import interface

    successes: list[str] = []
    monkeypatch.setattr(interface, "get_category_for_method", lambda m: "binance_perp")
    monkeypatch.setattr(interface, "get_vendor", lambda cat, meth: "default")
    monkeypatch.setattr(
        interface, "VENDOR_METHODS",
        {
            "get_binance_klines": {"binance_perp": lambda *a, **k: "csv"},
            "get_binance_funding_rate": {"binance_perp": lambda *a, **k: "csv"},
        },
    )

    def fake_record_success(method, qualifier=""):
        successes.append(f"{method}[{qualifier}]" if qualifier else method)

    monkeypatch.setattr(interface.quality, "record_success", fake_record_success)
    # The core price engine's success IS recorded (with its qualifier); a
    # category-level optional enrichment method's is not (unchanged contract).
    interface.route_to_vendor("get_binance_klines", "BTCUSDT", "2026-09-18", "2026-09-18")
    interface.route_to_vendor("get_binance_funding_rate", "BTCUSDT", "a", "b")
    assert successes == ["get_binance_klines"]


# ---- perp bundle sentinel detail: failed legs must read "missing" -----------


@pytest.mark.unit
def test_bundle_core_sentinel_reports_failed_leg_as_missing(monkeypatch):
    def fake_prices(symbol, end_date):
        # last ok, mark FAILED (writes its key with STATUS_UNAVAILABLE —
        # the old key-presence check reported this as "ok").
        return {
            "status": pb.STATUS_UNAVAILABLE,
            "last": {"status": pb.STATUS_OK, "close": 1.0},
            "mark": {"status": pb.STATUS_UNAVAILABLE, "reason": "boom"},
            "core_complete": False,
        }

    monkeypatch.setattr(pb, "is_historical_date", lambda d: True)
    monkeypatch.setattr(pb, "_fetch_prices", fake_prices)
    monkeypatch.setattr(pb, "_fetch_open_interest",
                        lambda s, d: {"status": pb.STATUS_OK})
    monkeypatch.setattr(pb, "_fetch_lsr", lambda s, d: {"status": pb.STATUS_OK})
    monkeypatch.setattr(pb, "_fetch_taker", lambda s, d: {"status": pb.STATUS_OK})
    monkeypatch.setattr(pb, "_fetch_spot_basis",
                        lambda s, d, c: {"status": pb.STATUS_CAPABILITY_ABSENT})
    monkeypatch.setattr(pb, "_last_close_hint", lambda s, d: None)

    quality.ensure_run_context()
    try:
        pb.fetch_perp_market_bundle("BTCUSDT", "2020-01-10")
        events = quality.snapshot_quality()
    finally:
        quality.reset_quality()

    core = [e for e in events if e["method"] == "get_binance_klines"]
    assert core, "bundle price failure must reach the quality chain"
    assert "last=ok" in core[0]["detail"]
    assert "mark=missing" in core[0]["detail"]


# ---- perp bundle workers inherit the parent contextvars ---------------------


@pytest.mark.unit
def test_bundle_workers_see_parent_pinned_analysis_date(monkeypatch):
    """The bundle's ThreadPoolExecutor must submit with the caller's context
    (submit_with_context): a worker that re-initializes config/analysis-date
    ContextVars from defaults diverges from the run's transport/PIT pins."""
    from yialpha.dataflows.utils import pinned_analysis_date

    seen: list[str | None] = []

    def spy_frame(symbol, start, end, *args, **kwargs):
        from yialpha.dataflows.utils import get_analysis_date

        seen.append(get_analysis_date())
        return _frame([1.0, 1.1])  # canned — hermetic, no vendor call

    monkeypatch.setattr(pb, "is_historical_date", lambda d: True)
    monkeypatch.setattr(pb, "binance_klines_frame", spy_frame)
    monkeypatch.setattr(pb, "_fetch_open_interest",
                        lambda s, d: {"status": pb.STATUS_OK})
    monkeypatch.setattr(pb, "_fetch_lsr", lambda s, d: {"status": pb.STATUS_OK})
    monkeypatch.setattr(pb, "_fetch_taker", lambda s, d: {"status": pb.STATUS_OK})
    monkeypatch.setattr(pb, "_fetch_spot_basis",
                        lambda s, d, c: {"status": pb.STATUS_CAPABILITY_ABSENT})
    monkeypatch.setattr(pb, "_last_close_hint", lambda s, d: None)
    with pinned_analysis_date("2020-01-10"):
        pb.fetch_perp_market_bundle("BTCUSDT", "2020-01-10")
    assert seen and all(v == "2020-01-10" for v in seen), (
        f"bundle workers lost the pinned analysis date: {seen}"
    )


# ---- reaches_now delegates to the shared dual-anchor live predicate ---------


@pytest.mark.unit
def test_futures_data_window_reaches_now_on_local_anchor(monkeypatch):
    """A host BEHIND UTC labels its live run with the host-local date; a
    strict single-anchor "is today" drops the live snapshot row from that
    run. reaches_now delegates to the ONE shared predicate
    (is_historical_date), so the contract is pinned relative to the real
    clock (no fixed-date time bomb in the retention guard)."""
    from datetime import date, timedelta

    import yialpha.dataflows.binance as bn

    local = date.today()                    # host-local "today" (UTC-5 host)
    utc = local + timedelta(days=1)         # venue UTC date, already rolled
    live_labels = {local.isoformat(), utc.isoformat()}
    monkeypatch.setattr(
        bn, "is_historical_date", lambda d: d not in live_labels
    )

    _extra, _end, reaches_now, _end_ms, _note = bn._futures_data_window(
        "BTCUSDT", "BTCUSDT", 7, None, local.isoformat(), period="1d",
    )
    assert reaches_now is True

    # Past AND future labels stay non-live — no live row leaks into a
    # replay (and none is appended to a future-dated window either).
    for label in ((local - timedelta(days=1)), (local + timedelta(days=2))):
        _extra, _end, reaches, _end_ms, _note = bn._futures_data_window(
            "BTCUSDT", "BTCUSDT", 7, None, label.isoformat(), period="1d",
        )
        assert reaches is False


# ---- accuracy: forming bar + perp venue routing ------------------------------


@pytest.mark.unit
def test_dated_close_drops_forming_bar_via_closed_as_of(monkeypatch):

    kwargs_seen: list[dict] = []

    def fake_frame(symbol, start, end, interval="1d", venue="binance_perp", **kw):
        kwargs_seen.append(kw)
        return _frame([1.0, 1.1, 1.2, 1.3, 1.4, 1.5])

    monkeypatch.setattr(
        "yialpha.dataflows.binance.binance_klines_frame", fake_frame
    )
    record = DecisionRecord(
        ticker="BTCUSDT", trade_date="2026-09-10", rating="buy",
        asset_type="crypto_perp", price_at_decision=None, price_basis="",
        asset_source="logged", log_path="",
    )
    # Frame carries 6 rows (horizon + the forming bar); closed_as_of is
    # what drops the boundary-day partial close in production — here the
    # fake ignores it, so the 6-row frame scores day 5. The regression pin
    # is that the seam is USED at all:
    out = forward_outcome(record, holding_days=5)
    assert out is not None
    assert kwargs_seen and kwargs_seen[0].get("closed_as_of") is not None


@pytest.mark.unit
def test_forward_outcome_pending_when_only_forming_bar_closes_horizon(monkeypatch):
    """End-to-end pending semantics: with the forming bar filtered out (as
    the closed_as_of seam does), a horizon that only completes intraday
    must stay pending, never scored on the partial close."""
    closes = [100.0, 101.0, 102.0, 103.0, 104.0, 200.0]  # last = forming bar
    dates = ["2026-09-10", "09-11", "09-12", "09-13", "09-14", "2026-09-15"]
    dates = [d if len(d) == 10 else f"2026-{d}" for d in dates]

    def fake_frame(symbol, start, end, interval="1d", venue="binance_perp",
                   *, closed_as_of=None, **kw):
        f = _frame(closes, dates)
        if closed_as_of is not None:
            f = f.iloc[:-1]  # emulate the seam: drop the unclosed bar
        return f

    monkeypatch.setattr(
        "yialpha.dataflows.binance.binance_klines_frame", fake_frame
    )
    record = DecisionRecord(
        ticker="BTCUSDT", trade_date="2026-09-10", rating="buy",
        asset_type="crypto_perp", price_at_decision=None, price_basis="",
        asset_source="logged", log_path="",
    )
    assert forward_outcome(record, holding_days=5) is None  # pending


@pytest.mark.unit
def test_fetch_returns_routes_perp_to_binance_venue(monkeypatch):
    """A perp memory entry must resolve on the venue it traded, not the
    Yahoo spot symbol normalize_symbol() picks (BTCUSDT → BTC-USD)."""
    calls: list[tuple] = []

    def fake_binance(symbol, start, end, interval="1d", venue="binance_perp",
                     price_type="last", *, closed_as_of=None):
        calls.append(("binance", venue))
        return _frame([100.0, 110.0])

    def fake_yf(symbol, start, end):
        calls.append(("yf", symbol))
        return _frame([400.0, 404.0])

    monkeypatch.setattr(
        "yialpha.dataflows.binance.binance_klines_frame", fake_binance
    )
    monkeypatch.setattr(
        "yialpha.dataflows.y_finance.get_YFin_history_cached", fake_yf
    )

    raw, alpha, days = fetch_returns_yf(
        "BTCUSDT", "2026-09-01", holding_days=1, asset_type="crypto_perp",
    )
    assert ("binance", "binance_perp") in calls
    assert ("yf", "SPY") in calls          # benchmark leg stays yfinance
    assert "BTC-USD" not in [c[1] for c in calls]
    assert raw == pytest.approx(0.10)
    assert alpha == pytest.approx(0.10 - 0.01)


@pytest.mark.unit
def test_resolve_pending_routes_on_entry_asset_tag(monkeypatch, tmp_path):
    """The entry's own asset= tag outranks the run-level fallback: a perp
    entry resolves on the perp venue even when a later spot run of the
    same ticker triggers the sweep."""
    from yialpha.agents.utils.memory import TradingMemoryLog
    from yialpha.graph.memory_resolution import resolve_pending_entries

    log = TradingMemoryLog({"memory_log_path": str(tmp_path / "mem.md")})
    # One tagged perp entry, one legacy untagged entry, same ticker.
    log.store_decision("BTCUSDT", "2026-09-01", "Buy BTC.",
                       asset_type="crypto_perp")
    log.store_decision("BTCUSDT", "2026-09-02", "Buy BTC again.")

    seen: list[str | None] = []

    def fake_fetch(ticker, trade_date, **kw):
        seen.append(kw.get("asset_type"))
        return 0.05, 0.02, 5

    reflector = type("R", (), {})()
    reflector.reflect_on_final_decision = lambda **kw: "lesson"
    monkeypatch.setattr("yialpha.accuracy.fetch_returns_yf", fake_fetch)

    resolved = resolve_pending_entries(
        log, reflector, "BTCUSDT", holding_days=1,
        as_of_date="2026-09-20", asset_type="crypto_spot",
    )
    assert resolved == 2
    # Tagged entry keeps its own venue; the untagged one falls back.
    assert "crypto_perp" in seen and "crypto_spot" in seen


# ---- take_profits: significant-digit rounding for micro-price perps ---------


@pytest.mark.unit
def test_take_profits_keeps_precision_on_micro_price_perp():
    """PEPEUSDT-class entry (~1.14e-5): round(x, 6) left ONE significant
    digit; the three R-multiple targets must keep six."""
    entry, stop = 0.0000114, 0.0000102
    tps = take_profits(entry, stop, "long")
    assert len(tps) == 3
    r = entry - stop
    assert tps[0] == pytest.approx(entry + 1.5 * r, rel=1e-4)
    assert tps[2] == pytest.approx(entry + 5.0 * r, rel=1e-4)
    for tp in tps:
        # six significant digits at the 1e-5 magnitude = sub-1e-10 grid
        assert tp != round(tp, 6) or tp > 0.01 or abs(tp - round(tp, 6)) < 1e-12
    # The old bug collapsed TP1/TP2 to the same 6-decimal value.
    assert len({round(tp, 10) for tp in tps}) == 3


@pytest.mark.unit
def test_take_profits_unchanged_at_normal_prices():
    tps = take_profits(100.0, 98.0, "long")
    assert tps == [103.0, 106.0, 110.0]


# ---- overlay price failures count as CORE in summarize ----------------------


@pytest.mark.unit
def test_overlay_price_failure_counts_in_core_sentinel_count(monkeypatch):
    from yialpha.graph.trading_graph import YiAlphaGraph

    def boom(ticker, date, asset_type):
        raise RuntimeError("vendor down")

    monkeypatch.setattr(
        "yialpha.graph.trading_graph._memoized_close_and_atr", boom
    )
    quality.ensure_run_context()
    try:
        close, atr = YiAlphaGraph._latest_close_and_atr(
            None, "BTCUSDT", "2026-09-18", "crypto_perp"
        )
        summary = quality.summarize_quality(
            quality.snapshot_quality(), quality.snapshot_core_successes()
        )
    finally:
        quality.reset_quality()
    assert (close, atr) == (None, None)
    assert summary["core_sentinel_count"] >= 1, (
        "run_robust's DEGRADED verdict reads core_sentinel_count — an "
        "overlay price failure must be visible to it"
    )


# ---- overlay markdown ↔ web parsing survives micro-price perps ---------------


@pytest.mark.unit
def test_overlay_fields_parse_scientific_notation():
    """A PEPEUSDT-class liquidation price renders as ``1.07e-05`` (``%g``
    below 1e-4); the old ``[-0-9.]+`` regex class stopped at the ``e`` and
    the web KPI tile displayed "1.07" — five orders of magnitude off."""
    from yialpha.graph.overlay_fields import parse_overlay

    md = (
        "### Quantitative Risk Overlay\n"
        "- **Action**: enter\n"
        "- **Target Weight**: 8.0% (960)\n"
        "- **Stop Loss**: 1.02e-05\n"
        "- **Entry Reference**: 1.14e-05\n"
        "- **Suggested Leverage**: ≤ 4.0x\n"
        "- **Est. Liquidation Price**: 1.07e-05\n"
    )
    out = parse_overlay(md)
    assert out is not None
    assert float(out["liquidation_price"]) == pytest.approx(1.07e-05)
    assert float(out["stop_loss"]) == pytest.approx(1.02e-05)
    assert float(out["entry"]) == pytest.approx(1.14e-05)
    # Normal magnitudes keep parsing unchanged.
    md2 = md.replace("1.07e-05", "90123.5").replace(
        "1.02e-05", "88000.25"
    ).replace("1.14e-05", "90000.0")
    out2 = parse_overlay(md2)
    assert float(out2["liquidation_price"]) == pytest.approx(90123.5)


# ---- /futures/data retention clamp + vision edge-day disclosure --------------


@pytest.mark.unit
def test_futures_data_window_clamps_start_to_retention_horizon():
    """The server rejects (HTTP 400 -1130) any startTime older than the
    30-day retention horizon — live-probed 2026-09-19; it does NOT silently
    truncate. A wide window must clamp into the retained tail with a
    disclosure note instead of degrading the whole call to a sentinel."""
    from datetime import UTC as _UTC, datetime as _dt, timedelta as _td

    import yialpha.dataflows.binance as bn

    now = _dt.now(_UTC)
    start = (now - _td(days=90)).strftime("%Y-%m-%d")
    end = (now - _td(days=1)).strftime("%Y-%m-%d")
    extra, _end_iso, _reaches, _end_ms, note = bn._futures_data_window(
        "BTCUSDT", "BTCUSDT", 7, start, end,
    )
    horizon_ms = int((now - _td(days=30)).timestamp() * 1000)
    assert extra["startTime"] >= horizon_ms - 86_400_000  # clamped into retention
    assert "retains only the last" in note
    assert "dropped from the head" in note


@pytest.mark.unit
def test_vision_edge_day_404_is_disclosed():
    """A window whose END day itself 404s (the ~06:30 UTC pre-publication
    window: end == yesterday, the file not landed yet) must disclose the
    absent edge day — previously only interior holes / budget-unsynced /
    low-row days were surfaced, so the newest, decision-adjacent day was
    silently absent while the header named the full window."""
    from types import SimpleNamespace

    from yialpha.dataflows.binance_vision import _coverage_notes

    report = SimpleNamespace(
        unsynced=0, synced_days_count=17, window_days=18, budget=400,
        missing_days=["2026-09-17"],
    )
    qa = {"interior_missing_days": [], "low_row_days": []}
    note = _coverage_notes(
        report, qa,
        _dt(2026, 9, 1), _dt(2026, 9, 17),
    )
    assert "end 2026-09-17" in note
    assert "not published yet" in note
    # No edge miss → no edge note.
    clean = _coverage_notes(
        SimpleNamespace(
            unsynced=0, synced_days_count=18, window_days=18, budget=400,
            missing_days=[],
        ),
        qa, _dt(2026, 9, 1), _dt(2026, 9, 17),
    )
    assert "not published yet" not in clean


def _dt(y, m, d):
    from datetime import datetime as _datetime

    return _datetime(y, m, d)


# ---- execution-advice script: exponent parsing + breakout robustness ---------

_tt_spec = importlib.util.spec_from_file_location(
    "trade_ticket_under_test",
    Path(__file__).resolve().parents[1] / "scripts" / "trade_ticket.py",
)
_tt = importlib.util.module_from_spec(_tt_spec)
_tt_spec.loader.exec_module(_tt)


@pytest.mark.unit
def test_trade_ticket_fnum_parses_exponent_notation():
    """trade_ticket._fnum had the same exponent truncation the overlay
    regexes carried: the overlay renders stop/entry with %g, so a
    PEPEUSDT-class stop of 1.14e-05 parsed as 1.14 — the execution ticket's
    notional/TP/liquidation outputs all off by five orders of magnitude."""
    assert _tt._fnum("1.14e-05") == pytest.approx(1.14e-05)
    assert _tt._fnum("1.02E-05") == pytest.approx(1.02e-05)
    assert _tt._fnum("**Stop Loss**: 1.14e-05") == pytest.approx(1.14e-05)
    # Legacy formats unchanged.
    assert _tt._fnum("1,234.56") == pytest.approx(1234.56)
    assert _tt._fnum("65000.0") == pytest.approx(65000.0)
    assert _tt._fnum("no number here") is None


@pytest.mark.unit
def test_trade_ticket_money_renders_micro_prices():
    assert "1.14e-05" in _tt._money(1.14e-05)
    assert _tt._money(65_000.0) == "65,000.00"
    assert _tt._money(103.5) == "103.5"


@pytest.mark.unit
def test_trade_ticket_breakout_resistance_layout_does_not_crash():
    """A long ticket where EVERY parsed resistance is below entry (breakout
    narrative) crashed render_ticket with ``min() arg is an empty sequence``
    — the whole script died instead of printing the ticket."""
    ticket = {
        "ticker": "BTCUSDT", "asset_type": "crypto_perp",
        "report_dir": "x", "rating": "Buy", "trader_action": "Buy",
        "capital": 100_000.0, "risk_profile": "balanced",
        "status": "ok", "direction": "long",
        "direction_reason": "rating Buy", "conviction_score": 0.8,
        "entry": 0.0000114, "stop": 0.0000102, "stop_dist_pct": 0.10526,
        "R_dollar": 120.0, "risk_dollar": 1000.0, "risk_frac": 0.01,
        "notional": 9500.0, "quantity": 833_333_333.0,
        "take_profits": [0.0000132, 0.0000174, 0.0000174],
        # breakout: every parsed resistance BELOW the entry
        "resistances": [0.0000090, 0.0000100], "supports": [0.0000080],
        "price_target_pm": None, "liquidation_price": None,
        "liquidation_dist_pct": None, "leverage": 4.0,
        "leverage_caps": {"L_liq": 6.0, "L_vol": 8.0, "L_conv": 5.0, "L_hard": 20.0},
        "margin": 2375.0, "margin_pct_of_capital": 0.024,
        "margin_warning": None, "atr": 6e-7, "atr_pct": 0.05,
        "current_price": 0.0000114, "sizing": None, "action": "Buy",
        "framework_positioning": "", "overlay_target_weight_pct": None,
        "ovl_action": "enter", "ovl_target_weight": 8.0, "ovl_stop": None,
        "ovl_entry_ref": None, "ovl_drawdown_regime": "normal",
        "ovl_is_perp": True, "drawdown_regime": "normal",
    }
    out = _tt.render_ticket(ticket)  # must not raise
    assert "做多" in out
    assert "结构参考·最近阻力" not in out  # no qualifying resistance → row skipped


# ---- memory: rotation keeps tagged pending; guard keys include venue ---------


@pytest.mark.unit
def test_memory_rotation_keeps_asset_tagged_pending(tmp_path):
    """The rotation suffix test (``endswith('| pending]')``) misread every
    asset-tagged pending entry as resolved — the exact 'unprocessed work'
    rotation promises to keep became droppable."""
    from yialpha.agents.utils.memory import TradingMemoryLog

    log = TradingMemoryLog({
        "memory_log_path": str(tmp_path / "mem.md"),
        "memory_log_max_entries": 1,
    })
    # Two asset-tagged PENDING entries + one genuinely resolved one.
    log.store_decision("BTCUSDT", "2026-09-01", "Buy A.", asset_type="crypto_perp")
    log.store_decision("BTCUSDT", "2026-09-02", "Buy B.", asset_type="crypto_perp")
    log.store_decision("BTCUSDT", "2026-09-03", "Buy C.", asset_type="crypto_perp")
    log.update_with_outcome(
        "BTCUSDT", "2026-09-03", 0.05, 0.02, 5, "lesson",
        asset_type="crypto_perp",
    )

    raw = (tmp_path / "mem.md").read_text(encoding="utf-8")
    blocks = [b for b in raw.split(log._SEPARATOR) if b.strip()]
    kept = log._apply_rotation(blocks)
    pending_blocks = [
        b for b in kept if "| pending" in b.splitlines()[0]
    ]
    assert len(pending_blocks) == 2, (
        "rotation must keep BOTH pending entries (the resolved one is "
        "droppable, pending ones are not)"
    )


@pytest.mark.unit
def test_batch_outcome_attaches_to_matching_venue(tmp_path):
    """Same-day perp+spot pending entries: an update carrying asset_type
    must land on THAT venue's entry, not the first pending match."""
    from yialpha.agents.utils.memory import TradingMemoryLog

    log = TradingMemoryLog({"memory_log_path": str(tmp_path / "mem.md")})
    log.store_decision("BTCUSDT", "2026-09-18", "Perp view: Buy.",
                       asset_type="crypto_perp")
    log.store_decision("BTCUSDT", "2026-09-18", "Spot view: Hold.",
                       asset_type="crypto_spot")
    log.batch_update_with_outcomes([{
        "ticker": "BTCUSDT", "trade_date": "2026-09-18",
        "raw_return": 0.05, "alpha_return": 0.02, "holding_days": 5,
        "reflection": "spot lesson", "available_date": "2026-09-25",
        "asset_type": "crypto_spot",
    }])
    entries = log.load_entries()
    by_asset = {e["asset"]: e for e in entries}
    assert by_asset["crypto_perp"]["pending"] is True       # untouched
    assert by_asset["crypto_spot"]["pending"] is False      # resolved
    assert "spot lesson" in (by_asset["crypto_spot"]["reflection"] or "")


@pytest.mark.unit
def test_store_decision_allows_same_day_perp_and_spot(tmp_path):
    """The idempotency guard keyed on (date, ticker) only: whichever venue
    stored first silently dropped the OTHER venue's same-day decision —
    defeating the asset= tag's stated purpose."""
    from yialpha.agents.utils.memory import TradingMemoryLog

    log = TradingMemoryLog({"memory_log_path": str(tmp_path / "mem.md")})
    log.store_decision("BTCUSDT", "2026-09-18", "Perp view: Buy.",
                       asset_type="crypto_perp")
    log.store_decision("BTCUSDT", "2026-09-18", "Spot view: Hold.",
                       asset_type="crypto_spot")
    # Same venue again → still idempotent (no duplicate).
    log.store_decision("BTCUSDT", "2026-09-18", "Perp view: Buy.",
                       asset_type="crypto_perp")
    entries = log.load_entries()
    assert len(entries) == 2
    assert {e["asset"] for e in entries} == {"crypto_perp", "crypto_spot"}


# ---- CLI streamed path mirrors the propagate run contract --------------------


@pytest.mark.unit
def test_bind_run_record_stage_is_shared_by_run_graph_and_cli(monkeypatch):
    """The extraction contract: _run_graph delegates its record stage to
    _bind_run_record_stage (the same seam the CLI streamed path calls), so
    an interactive run records like a propagate run."""
    from yialpha.graph.trading_graph import YiAlphaGraph

    captured: dict = {}

    def fake_bind(self, ticker, trade_date, asset_type):
        captured["args"] = (ticker, trade_date, asset_type)
        return "run-1", "regime-1"

    monkeypatch.setattr(YiAlphaGraph, "_bind_run_record_stage", fake_bind)

    import inspect

    src = inspect.getsource(YiAlphaGraph._run_graph)
    assert "self._bind_run_record_stage(" in src, (
        "_run_graph must delegate to the shared record-stage seam"
    )
    # And the CLI's streamed block calls the same seam and stamps the ids.
    cli_src = inspect.getsource(
        __import__("yialpha.cli.main", fromlist=["run_analysis"])
    )
    assert "graph._bind_run_record_stage(" in cli_src
    # The stamping half is the SAME shared seam (not a hand-copied mirror).
    assert "graph._stamp_run_record_ids(" in cli_src


@pytest.mark.unit
def test_cli_streamed_overlay_passes_asset_type():
    """The P1: the interactive perp run's overlay call defaulted to
    asset_type='stock', voiding the perp ticket/funding/fair-value bridge."""
    import inspect

    cli_src = inspect.getsource(
        __import__("yialpha.cli.main", fromlist=["run_analysis"])
    )
    assert "asset_type=selections[\"asset_type\"]," in cli_src.replace(
        "'", '"'
    ), "the streamed overlay call must forward the selected asset_type"


@pytest.mark.unit
def test_store_cli_decision_tags_asset_type(tmp_path):
    from yialpha.agents.utils.memory import TradingMemoryLog
    from yialpha.cli.main import _store_cli_decision

    class _Graph:
        def __init__(self):
            self.memory_log = TradingMemoryLog(
                {"memory_log_path": str(tmp_path / "mem.md")}
            )

    g = _Graph()
    _store_cli_decision(
        g, "BTCUSDT", "2026-09-18", {"final_trade_decision": "Buy."},
        asset_type="crypto_perp",
    )
    entries = g.memory_log.load_entries()
    assert entries[0]["asset"] == "crypto_perp"
