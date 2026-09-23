"""Round-7 perp sweep regression pins (2026-09-23).

Five review shards (binance+vision / p0-shadow / backtest economics / risk
micro-layer / analysts+tool-binding) plus the main-line sweep; every landed
fix carries its pin here or in an updated dedicated file (see CHANGELOG
round 7). Highlights pinned in this file:

* regime funding window 7 days END-INCLUSIVE (the third 8-day off-by-one);
* the perp bundle's ``/futures/data`` legs clamp their start into the
  retained tail (live probe: the old D-30 OI window 400'd with -1130 on
  EVERY historical replay);
* taker ``mean_7d`` = exactly 7 daily values; OI render states the span
  actually served;
* the tradeability gate refuses non-finite/non-positive targets and
  non-finite costs (new stable reasons);
* ``RiskManager.decide`` consults the breaker's bad-tick path
  unconditionally (no fabricated permissive state on NaN equity);
* the fair-value bridge refuses non-positive targets, renders micro prices
  adaptively, and USD-quoted verdicts re-price against the converted
  contract target (graph-level pin in test_v21_ledger_wiring);
* constraint ``others`` sums exclude by the (symbol, class) replacement key
  (same-symbol other-venue rows counted); beta filters pairwise-aligned;
* vision edge-day disclosure survives the 24h retry backoff (manifest-
  derived ``known_missing_days``);
* the shared midnight-cross grace predicate admits the scorer's rows in
  the scoreboard and audit;
* schema percent/case/comma rescue (junk degrades optional fields instead
  of discarding whole structured decisions);
* the indicator/snapshot tool entries clamp the LLM-supplied date;
* ``normalize_symbol_for_venue`` accepts ccxt/TradingView pair forms;
* IC-context fallback never blends venues; the accuracy inference label
  and the lowercase confidence cap behave as documented;
* the sentiment prompt's best-practice #1 gates on the StockTwits source
  policy (closeout fix);
* the REST paginator's malformed-tail break carries a distinguishing
  warning (closeout fix; in-band header fix remains a known issue).
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import pandas as pd
import pytest

import yialpha.dataflows.binance as bz
import yialpha.dataflows.perp_bundle as pb
import yialpha.regime.compute as rc

_DAY_MS = 86_400_000


def _frame(closes: list[float]) -> pd.DataFrame:
    idx = pd.date_range("2025-01-01", periods=len(closes), freq="D")
    idx.name = "Date"
    return pd.DataFrame(
        {
            "Open": [c * 0.999 for c in closes],
            "High": [c * 1.01 for c in closes],
            "Low": [c * 0.99 for c in closes],
            "Close": closes,
            "Volume": [1000.0] * len(closes),
        },
        index=idx,
    )


_UPTREND = [100.0 + 0.5 * i for i in range(260)]
_FUNDING_CSV = "fundingTime,fundingRate\n2026-08-26 00:00:00,0.00030\n"


@pytest.mark.unit
def test_regime_funding_window_is_seven_days_end_inclusive(monkeypatch):
    """The funding seam is asked for [D-6, D] — 7 calendar days end-inclusive,
    the settled seam semantics — not the 8-day [D-7, D] span that overstated
    ``funding_pct`` (a frozen RegimeState field and stress trigger) ~14%."""
    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(
        rc, "binance_klines_frame",
        lambda *a, **k: _frame([c * 0.999 for c in _UPTREND])
        if k.get("venue") == "binance_spot" else _frame(_UPTREND),
    )
    monkeypatch.setattr(
        rc, "get_binance_funding_rate",
        lambda s, a, b: (calls.append((a, b)), _FUNDING_CSV)[1],
    )
    monkeypatch.setattr(
        rc, "_fetch_open_interest",
        lambda s, d: {"status": "ok", "latest": 1.0, "percentile": 42.0,
                      "chg_1d": 0.01},
    )
    monkeypatch.setattr(
        rc, "_fetch_lsr",
        lambda s, d: {"status": "ok", "global_account": {"status": "ok",
                                                          "latest": 1.3}},
    )
    monkeypatch.setattr(
        rc, "_fetch_taker", lambda s, d: {"status": "ok", "latest": 1.1}
    )
    monkeypatch.setattr(
        rc, "_fetch_depth_bands", lambda s: {"status": "ok"}
    )

    end = date.today().isoformat()
    state = rc.compute_regime_state(
        "BTCUSDT", "crypto_perp", "pure_crypto_perp", end, end_date=end,
    )
    assert state is not None and state.funding_pct is not None
    assert calls, "funding seam was never consulted"
    start, got_end = calls[0]
    assert got_end == end
    assert start == (date.fromisoformat(end) - timedelta(days=6)).isoformat()


@pytest.mark.unit
def test_futures_data_start_ms_clamps_into_retained_tail(monkeypatch):
    frozen = int(datetime(2026, 9, 23, 12, 0, tzinfo=UTC).timestamp() * 1000)
    monkeypatch.setattr(bz, "_now_ms", lambda: frozen)
    retention = frozen - bz._FUTURES_DATA_RETENTION_DAYS * _DAY_MS
    # A 40-day-back start (the old OI window on a 10-day-back replay) clamps
    # up to the retention boundary instead of leaving the 400 -1130 request.
    assert pb._futures_data_start_ms(frozen - 40 * _DAY_MS) == retention
    # An in-retention start passes through untouched.
    assert pb._futures_data_start_ms(frozen - 5 * _DAY_MS) == frozen - 5 * _DAY_MS


@pytest.mark.unit
def test_oi_series_sends_clamped_start_and_pit_end(monkeypatch):
    """A historical replay's OI request keeps its PIT end-of-day endTime but
    clamps startTime into the retained tail — the exact request shape the
    2026-09-23 live probe validated against -1130."""
    frozen = int(datetime(2026, 9, 23, 12, 0, tzinfo=UTC).timestamp() * 1000)
    monkeypatch.setattr(bz, "_now_ms", lambda: frozen)
    captured: dict[str, int] = {}

    def fake_get(path, params, symbol, canonical):
        captured.update(params)
        return [{"timestamp": frozen - i * _DAY_MS, "sumOpenInterest": 1.0}
                for i in range(1, 4)]

    monkeypatch.setattr(pb, "_http_get", fake_get)
    replay = (date.today() - timedelta(days=10)).isoformat()
    rows = pb._oi_series("BTCUSDT", replay, 30)
    assert rows  # sorted (ts, value) tuples came back
    retention = frozen - bz._FUTURES_DATA_RETENTION_DAYS * _DAY_MS
    assert captured["startTime"] >= retention
    end_dt = datetime.strptime(replay, "%Y-%m-%d").replace(tzinfo=UTC)
    assert captured["endTime"] == int((end_dt.timestamp() + 86399) * 1000)


@pytest.mark.unit
def test_taker_mean_7d_averages_exactly_seven_values(monkeypatch):
    values = [round(0.5 + 0.01 * i, 4) for i in range(10)]  # 10 daily rows
    monkeypatch.setattr(
        pb, "_http_get",
        lambda path, params, symbol, canonical: [
            {"buySellRatio": str(v)} for v in values
        ],
    )
    out = pb._fetch_taker("BTCUSDT", date.today().isoformat())
    assert out["status"] == "ok"
    expected = sum(values[-7:]) / 7.0  # [D-6..D], the 7d end-inclusive span
    assert out["mean_7d"] == pytest.approx(expected)
    assert out["mean_7d"] != pytest.approx(sum(values[-8:]) / 8.0)


@pytest.mark.unit
def test_oi_render_states_span_actually_served():
    bundle = {
        "symbol": "BTCUSDT", "as_of": "2026-09-23", "live_run": True,
        "open_interest": {
            "status": "ok", "latest": 1234.0, "percentile": 55.0,
            "chg_1d": 0.01, "days": 20,  # clamped replay / young contract
        },
    }
    rendered = pb.render_perp_bundle_block(bundle)
    assert "trailing 20d" in rendered
    # A component without the field (legacy fixtures) keeps the 30d default.
    bundle["open_interest"].pop("days")
    assert "trailing 30d" in pb.render_perp_bundle_block(bundle)


# ---------------------------------------------------------------------------
# Shard D (risk micro-layer) pins — round 7
# ---------------------------------------------------------------------------
from yialpha.perp.fair_value import (  # noqa: E402
    MISSING_UNDERLYING,
    fair_value_bridge,
)
from yialpha.risk.constraints import (  # noqa: E402
    PortfolioLimits,
    PortfolioSnapshot,
    PositionView,
    advisory_metrics,
    global_gross,
)
from yialpha.risk.cost_model import CostEstimate, estimate_round_trip_cost  # noqa: E402
from yialpha.risk.manager import PortfolioState, RiskManager  # noqa: E402
from yialpha.risk.tradeability import (  # noqa: E402
    REASON_COST_NON_FINITE,
    REASON_INVALID_TARGET,
    REASON_OK,
    TicketSide,
    Tradeability,
    evaluate_tradeability,
    gross_edge,
)


@pytest.mark.unit
def test_gross_edge_refuses_nonpositive_and_nan_target():
    """A zero/negative target is short-rewarding junk (1 - 0/ref = +100%),
    NaN defeats the comparisons — both refuse instead of laundering an
    edge. A legitimate below-reference short target still computes."""
    for bad in (0.0, -5.0, float("nan"), float("inf")):
        assert gross_edge(TicketSide.SHORT, bad, 100.0) is None
        assert gross_edge(TicketSide.LONG, bad, 100.0) is None
    assert gross_edge(TicketSide.SHORT, 50.0, 100.0) == pytest.approx(0.5)


@pytest.mark.unit
def test_evaluate_tradeability_invalid_target_reason():
    cost = estimate_round_trip_cost("crypto_perp", "short")
    for bad in (0.0, -5.0, float("nan")):
        verdict = evaluate_tradeability(
            rating="sell", target_price=bad, reference_price=100.0, cost=cost,
        )
        assert verdict.tradeability == Tradeability.UNEVALUATED
        assert verdict.tradeability_reason == REASON_INVALID_TARGET


@pytest.mark.unit
def test_evaluate_tradeability_non_finite_cost_fails_closed():
    """NaN funding through series.sum() makes total_bps NaN, which used to
    defeat net_edge <= 0 and fall through to TRADEABLE."""
    nan_cost = CostEstimate(funding_bps=float("nan"))
    verdict = evaluate_tradeability(
        rating="sell", target_price=90.0, reference_price=100.0, cost=nan_cost,
    )
    assert verdict.tradeability == Tradeability.NO_TRADE
    assert verdict.tradeability_reason == REASON_COST_NON_FINITE
    # Sanity: the same trade with a finite cost still prices TRADEABLE.
    ok = evaluate_tradeability(
        rating="sell", target_price=50.0, reference_price=100.0,
        cost=estimate_round_trip_cost("crypto_perp", "short"),
    )
    assert ok.tradeability_reason == REASON_OK


@pytest.mark.unit
def test_manager_consults_breaker_on_nan_equity():
    """One NaN equity tick must NOT skip a hard_stop breaker: the old guard
    fabricated BreakerState(normal, can_open_new=True) and a new Buy sized
    at full Kelly. The breaker's own bad-tick path replays the last valid
    regime fail-closed."""
    rm = RiskManager()
    state = PortfolioState(equity=100.0)
    rm.decide("BTCUSDT", "Buy", state)
    state.equity = 80.0  # -20% from peak -> hard_stop
    rm.decide("BTCUSDT", "Buy", state)
    decision = rm.decide("BTCUSDT", "Buy", PortfolioState(equity=float("nan")))
    assert decision.action == "blocked"


@pytest.mark.unit
def test_fair_value_refuses_nonpositive_underlying_target():
    for bad in (0.0, -50.0):
        result = fair_value_bridge(bad, 0.9993, None)
        assert result.contract_target_usdt is None
        assert MISSING_UNDERLYING in result.missing_inputs
        assert result.chain == ()
    # A sub-1 USD target renders adaptively (the .2f micro-price defect).
    ok = fair_value_bridge(0.004, 0.9993, None)
    assert ok.contract_target_usdt is not None
    assert "underlying target 0.004 USD" in ok.chain[0]
    # 100-scale renders keep the pinned fixed-decimal form.
    assert "underlying target 100.00 USD" in fair_value_bridge(
        100.0, 0.9993, None
    ).chain[0]


def _pv(
    weight: float,
    symbol: str = "BTCUSDT",
    cls: str = "pure_crypto_perp",
) -> PositionView:
    side = "LONG" if weight >= 0 else "SHORT"
    return PositionView(symbol=symbol, instrument_class=cls, side=side, weight=weight)


@pytest.mark.unit
def test_constraints_count_same_symbol_other_venue_row():
    """A same-symbol different-venue row (perp + spot books of one ticker)
    stays in the post book — the class-agnostic sums must COUNT it, not
    lose it between the symbol-only `others` filter and the candidate."""
    snap = PortfolioSnapshot(
        equity=100_000.0,
        positions=(_pv(0.7, symbol="BTCUSDT", cls="stock_perp"),),
        returns_history=(),
    )
    # Candidate re-sizes the pure-crypto perp book; the same-symbol
    # stock_perp-class row is a DIFFERENT row and must stay counted.
    candidate = _pv(0.4, symbol="BTCUSDT", cls="pure_crypto_perp")
    decision = global_gross(candidate, snap, PortfolioLimits())
    # True post gross = 0.7 (other-venue row kept) + 0.4 = 1.1 > 1.0 cap —
    # the contract sizes the candidate into the UNUSED headroom: clipped to
    # (1.0 - 0.7) / 0.4 = 0.75, not the full weight the old code allowed
    # (others=0 under the symbol-only filter -> PASS at 1.0).
    assert decision.metrics["post_gross"] == pytest.approx(1.1)
    assert decision.action == "RESIZE"
    assert decision.multiplier == pytest.approx(0.75)


@pytest.mark.unit
def test_beta_uses_pairwise_aligned_filtering():
    """Independent per-side finite filters left equal lengths over DIFFERENT
    dates and fabricated a date-shifted covariance; the aligned pairs here
    are (0.01, 0.01) and (0.05, 0.03) -> beta 2.0 exactly."""
    nan = float("nan")
    snap = PortfolioSnapshot(
        equity=100_000.0,
        positions=(),
        returns_history=[0.01, nan, 0.02, 0.05],
    )
    out = advisory_metrics(
        _pv(0.1), snap, PortfolioLimits(),
        btc_returns=[0.01, 0.02, nan, 0.03],
    )
    assert out["beta_vs_btc"] == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# Shard A / B / E pins — round 7
# ---------------------------------------------------------------------------
import json  # noqa: E402

import yialpha.accuracy as acc  # noqa: E402
import yialpha.agents.analysts.market_analyst as ma  # noqa: E402
import yialpha.agents.analysts.sentiment_analyst as sa  # noqa: E402
from yialpha.dataflows.symbol_utils import normalize_symbol_for_venue  # noqa: E402
from yialpha.ledger.time_contract import (  # noqa: E402
    entry_boundary_admitted,
    midnight_cross_grace_day,
)


@pytest.mark.unit
def test_indicator_and_snapshot_tools_clamp_llm_date(monkeypatch):
    """get_indicators / get_verified_market_snapshot clamp the LLM-supplied
    date to the pinned analysis date — the round-5 guard's two missed
    entries (a model echoing the cache's 'retrieved on' header date used to
    serve post-decision rows on a historical run)."""
    from yialpha.agents.utils import (
        market_data_validation_tools as mdv,
        technical_indicators_tools as tit,
    )
    from yialpha.dataflows.utils import set_analysis_date

    pinned = (date.today() - timedelta(days=3)).isoformat()
    set_analysis_date(pinned)
    try:
        echoed = date.today().isoformat()  # the cache header the model echoes

        seen: list[str] = []

        def fake_route(method, symbol, ind, curr_date, look_back_days):
            seen.append(curr_date)
            return "ok"

        monkeypatch.setattr(tit, "route_to_vendor", fake_route)
        tit.get_indicators.func("AAPL", "rsi", echoed, 30)
        assert seen == [pinned]

        def fake_snapshot(sym, curr, look):
            seen.append(curr)
            return "ok"

        monkeypatch.setattr(
            mdv, "build_verified_market_snapshot", fake_snapshot,
        )
        mdv.get_verified_market_snapshot.func("AAPL", echoed, 30)
        assert seen == [pinned, pinned]
    finally:
        set_analysis_date(None)


@pytest.mark.unit
def test_vision_edge_day_disclosure_survives_backoff(tmp_path):
    """A previously-404ed edge day stays disclosed on later calls: the 24h
    retry backoff keeps it out of the call-scoped attempt set, so qa()'s
    manifest-derived ``known_missing_days`` is the durable source."""
    import yialpha.dataflows.binance_vision as bnv
    from yialpha.dataflows.binance_vision_store import VisionStore

    store = VisionStore(tmp_path)
    end = date.today() - timedelta(days=1)
    start = end - timedelta(days=2)
    store._dir("metrics", "BTCUSDT").mkdir(parents=True, exist_ok=True)
    manifest = store._read_manifest("metrics", "BTCUSDT")
    manifest["missing"][end.isoformat()] = "2026-09-23T00:00:00+00:00"
    manifest["days"][start.isoformat()] = 1  # one synced day
    store._write_manifest("metrics", "BTCUSDT", manifest)

    start_dt = datetime.combine(start, datetime.min.time(), tzinfo=UTC)
    end_dt = datetime.combine(end, datetime.min.time(), tzinfo=UTC)
    qa = store.qa("metrics", "BTCUSDT", start_dt, end_dt)
    assert qa["known_missing_days"] == [end.isoformat()]

    class _NoReport:
        missing_days = []  # the backoff-suppressed later call: no fresh 404

    notes = bnv._coverage_notes(_NoReport(), qa, start_dt, end_dt)
    assert f"end {end} day(s) not published yet" in notes


@pytest.mark.unit
def test_entry_boundary_grace_predicate():
    """The shared predicate admits exactly the scorer's midnight-cross grace
    boundary — the fix for the scoreboard/audit silently rejecting rows the
    compute layer scored complete."""
    formed = datetime(2026, 9, 23, 0, 3, tzinfo=UTC)  # minutes past midnight
    # Expected reference day for a 09-23 formation is 09-22; its bar closes
    # at 09-23T00:00Z. The grace boundary is one bar-day older: 09-22T00:00Z.
    strict = datetime(2026, 9, 23, 0, 0, tzinfo=UTC)
    grace = datetime(2026, 9, 22, 0, 0, tzinfo=UTC)
    assert midnight_cross_grace_day(formed, equity=False) == date(2026, 9, 21)
    assert entry_boundary_admitted(grace, formed, equity=False)
    assert entry_boundary_admitted(strict, formed, equity=False)
    # Formation outside the midnight seam: only the strict boundary passes.
    late = datetime(2026, 9, 23, 12, 0, tzinfo=UTC)
    assert not entry_boundary_admitted(grace, late, equity=False)
    assert entry_boundary_admitted(strict, late, equity=False)


@pytest.mark.unit
def test_schema_percent_and_case_rescue():
    from yialpha.agents.schemas import (
        PortfolioDecision,
        PositioningReport,
        SentimentReport,
        TraderProposal,
    )

    d = PortfolioDecision(
        rating="Buy", action="Buy", reasoning="x",
        executive_summary="s", investment_thesis="t", confidence=85,
        evidence_coverage="70%",
        probabilities={"bull": 61, "neutral": 24, "bear": 16},
        price_target="1,250.50",
    )
    assert d.confidence == pytest.approx(0.85)
    assert d.evidence_coverage == pytest.approx(0.70)
    assert d.price_target == pytest.approx(1250.50)
    assert d.probabilities is not None
    assert d.probabilities.bull == pytest.approx(0.61)
    # One unusable branch degrades the whole OPTIONAL block, never the decision.
    d2 = PortfolioDecision(
        rating="Buy", action="Buy", reasoning="x",
        executive_summary="s", investment_thesis="t",
        probabilities={"bull": 0.6, "neutral": 0.2},  # bear missing
    )
    assert d2.probabilities is None
    assert d2.rating.value == "Buy"
    # Junk optional floats degrade to None instead of raising.
    d3 = PortfolioDecision(
        rating="Buy", action="Buy", reasoning="x",
        executive_summary="s", investment_thesis="t", confidence="garbage",
    )
    assert d3.confidence is None
    # Comma-grouped prices parse; required Literal case is rescued.
    t = TraderProposal(action="Buy", reasoning="x", entry_price="1,250.50")
    assert t.entry_price == pytest.approx(1250.50)
    s = SentimentReport(
        overall_score=5.0, overall_band="Neutral",
        confidence="Medium", narrative="x",
    )
    assert s.confidence == "medium"
    p = PositioningReport(
        funding_bias="long_pays_expensive", crowding="balanced",
        liquidity_risk="normal", confidence="High", narrative="x",
    )
    assert p.confidence == "high"
    # Percent-form strings via the direct model path too.
    probs = PortfolioDecision.model_validate(
        {"rating": "Buy", "action": "Buy", "reasoning": "x",
         "executive_summary": "s", "investment_thesis": "t",
         "probabilities": {"bull": "70%", "neutral": 0.2, "bear": 0.1}}
    ).probabilities
    assert probs is not None and probs.bull == pytest.approx(0.70)


@pytest.mark.unit
def test_accuracy_labels_overridden_stock_tag_as_inferred():
    venue, source = acc._resolve_asset_type("BTCUSDT", "stock")
    assert venue == "crypto_perp"
    assert source == "inferred"  # the override is visible in the report caveat
    venue, source = acc._resolve_asset_type("AAPL", "stock")
    assert (venue, source) == ("stock", "logged")


@pytest.mark.unit
def test_ccxt_and_colon_symbol_forms_normalize():
    assert normalize_symbol_for_venue("BTC/USDT", "binance_perp") == "BTCUSDT"
    assert normalize_symbol_for_venue("btc:usdt", "binance_spot") == "BTCUSDT"
    assert normalize_symbol_for_venue("BTC-USD", "binance_perp") == "BTCUSDT"


@pytest.mark.unit
def test_ic_context_fallback_never_blends_venues(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ic_dir = tmp_path / "ic_data"
    ic_dir.mkdir()
    verdict = {"per_indicator": {"rsi": {"mean_abs_ic": 0.05}}}
    (ic_dir / "BTC_5d_perp.csv.prune.json").write_text(
        json.dumps(verdict), encoding="utf-8",
    )
    # A perp run reads its tagged verdict; a SPOT/stock run must NOT fall
    # back to the perp file (the old every-file fallback blended venues).
    assert ma._format_indicator_ic_context("crypto_perp") is not None
    assert ma._format_indicator_ic_context("crypto_spot") is None
    assert ma._format_indicator_ic_context("stock") is None


@pytest.mark.unit
def test_confidence_cap_fires_on_lowercase_free_text():
    capped = sa._enforce_confidence_cap(
        "**Confidence:** high\n", "low", "source policy",
    )
    assert "capped by source policy" in capped
    assert "**Confidence:** Low" in capped  # normalized capitalized output


@pytest.mark.unit
def test_sentiment_practice_1_gated_on_stocktwits_policy():
    """Best-practice #1 no longer instructs reading the StockTwits ratio on
    runs whose StockTwits block is a policy placeholder (the square_only
    profile — pure crypto); the fetched wording is untouched."""
    kw = {"ticker": "BTCUSDT", "start_date": "2026-01-01",
          "end_date": "2026-01-08"}
    off = sa._build_system_message(has_square=True, has_stocktwits=False, **kw)
    assert "No StockTwits evidence on this run" in off
    assert "Bullish/Bearish ratio as a leading" not in off
    on = sa._build_system_message(has_square=True, **kw)  # default: fetched
    assert "Bullish/Bearish ratio as a leading retail-sentiment signal" in on
    assert "No StockTwits evidence on this run" not in on


@pytest.mark.unit
def test_pagination_malformed_tail_warns(monkeypatch, caplog):
    """Cursor-extraction failure on the last page used to break as a normal
    final page — silently dropping any remaining RECENT rows. The break now
    carries a distinguishing warning naming the truncation direction."""
    import logging

    base = int(datetime(2024, 1, 1, tzinfo=UTC).timestamp() * 1000)

    def malformed_page(path, params, symbol, canonical, **kwargs):  # noqa: ARG001
        # A full page whose LAST row lacks the cursor key: shape alone
        # cannot tell this from a normal final page.
        return [{"fundingTime": base, "fundingRate": "0.0001"},
                {"fundingRate": "0.0001"}]

    monkeypatch.setattr(bz, "_http_get", malformed_page)
    with caplog.at_level(logging.WARNING, logger="yialpha.dataflows.binance"):
        rows = bz._paginate_history(
            "/fapi/v1/fundingRate", {"symbol": "BTCUSDT"}, 1000,
            lambda r: r["fundingTime"], base, base + 3 * _DAY_MS,
            "BTCUSDT", "BTCUSDT",
        )
    assert len(rows) == 2  # the served page is kept; the walk just stops
    assert any("cursor extraction failed" in r.getMessage() for r in caplog.records)
