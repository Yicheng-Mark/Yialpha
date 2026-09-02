"""Integration tests for the Phase-1 risk overlay wired into YiAlphaGraph.

These exercise the live-flow wiring (``_apply_risk_overlay``) directly, without
the full graph ``__init__`` (no LLM clients, no network). The backtest-side
wiring of the same :class:`RiskManager` is covered in ``test_risk_manager.py``.
"""

from __future__ import annotations

import pytest

from yialpha.agents.utils.rating import parse_rating
from yialpha.graph.signal_processing import SignalProcessor
from yialpha.graph.trading_graph import YiAlphaGraph


def _make_graph(risk_enabled: bool) -> YiAlphaGraph:
    """Build a graph shell with just enough state for the overlay to run."""
    g = YiAlphaGraph.__new__(YiAlphaGraph)
    g.config = {
        "risk_enabled": risk_enabled,
        "kelly_fraction": 0.25,
        "max_single_position": 0.20,
        "max_single_sector": 0.30,
        "max_drawdown_hard_stop": 0.15,
        "atr_stop_mult": 2.0,
    }
    g._risk_overlay_degraded = False
    g.risk_manager = g._build_risk_manager()
    g.signal_processor = SignalProcessor(None)
    return g


@pytest.mark.unit
def test_overlay_disabled_is_noop():
    g = _make_graph(risk_enabled=False)
    assert g.risk_manager is None
    original = "**Rating**: Buy\n\nThe thesis."
    state = {"final_trade_decision": original}
    out = g._apply_risk_overlay("AAPL", "2024-01-15", state, None)
    assert out["final_trade_decision"] == original  # untouched


@pytest.mark.unit
def test_overlay_appends_section_and_preserves_rating(monkeypatch):
    g = _make_graph(risk_enabled=True)
    monkeypatch.setattr(g, "_latest_close_and_atr", lambda t, d, at="stock": (190.0, 3.0))

    state = {"final_trade_decision": "**Rating**: Buy\n\nStrong momentum thesis."}
    out = g._apply_risk_overlay("AAPL", "2024-01-15", state, {"equity": 100_000})

    md = out["final_trade_decision"]
    assert "## Quantitative Risk Overlay" in md
    assert "**Target Weight**" in md
    assert "**Stop Loss**" in md
    # The appended section never hijacks rating extraction.
    assert parse_rating(md) == "Buy"
    # The original thesis is still present above the overlay.
    assert md.index("Strong momentum thesis.") < md.index("Quantitative Risk Overlay")


@pytest.mark.unit
def test_overlay_sell_has_no_stop(monkeypatch):
    g = _make_graph(risk_enabled=True)
    monkeypatch.setattr(g, "_latest_close_and_atr", lambda t, d, at="stock": (190.0, 3.0))
    state = {"final_trade_decision": "**Rating**: Sell"}
    out = g._apply_risk_overlay("AAPL", "2024-01-15", state, {"equity": 100_000})
    md = out["final_trade_decision"]
    assert "**Stop Loss**" not in md
    assert "Target Weight**: 0.0%" in md


@pytest.mark.unit
def test_overlay_survives_missing_price(monkeypatch):
    """No price/ATR available -> overlay still runs, just without a stop."""
    g = _make_graph(risk_enabled=True)
    monkeypatch.setattr(g, "_latest_close_and_atr", lambda t, d, at="stock": (None, None))
    state = {"final_trade_decision": "**Rating**: Overweight"}
    out = g._apply_risk_overlay("AAPL", "2024-01-15", state, {"equity": 100_000})
    md = out["final_trade_decision"]
    assert "Quantitative Risk Overlay" in md
    assert "**Stop Loss**" not in md
    assert "**Entry Reference**" not in md
    # Bullish position with no price data -> the missing-stop is now visible.
    assert "Stop-loss not set" in md


@pytest.mark.unit
def test_overlay_marks_missing_atr_warning(monkeypatch):
    """Bullish position + price/ATR unavailable -> visible degradation marker.

    Completes the three-path fail-open visibility coverage alongside
    ``test_overlay_build_failure_marks_decision`` and
    ``test_overlay_decide_failure_marks_decision``.
    """
    g = _make_graph(risk_enabled=True)
    monkeypatch.setattr(g, "_latest_close_and_atr", lambda t, d, at="stock": (None, None))
    state = {"final_trade_decision": "**Rating**: Buy\n\nThesis."}
    out = g._apply_risk_overlay("AAPL", "2024-01-15", state, {"equity": 100_000})
    md = out["final_trade_decision"]
    # Overlay ran (not disabled), but the missing stop is now visible.
    assert "Quantitative Risk Overlay" in md
    assert "⚠️ Quantitative Risk Overlay DISABLED" not in md
    assert "Stop-loss not set" in md
    assert "price/ATR data unavailable" in md
    # Rating and thesis are untouched.
    assert parse_rating(md) == "Buy"
    assert "Thesis." in md


@pytest.mark.unit
def test_overlay_no_marker_when_price_available(monkeypatch):
    """Price loaded normally -> no false-positive missing-stop marker."""
    g = _make_graph(risk_enabled=True)
    monkeypatch.setattr(g, "_latest_close_and_atr", lambda t, d, at="stock": (190.0, 3.0))
    state = {"final_trade_decision": "**Rating**: Buy"}
    out = g._apply_risk_overlay("AAPL", "2024-01-15", state, {"equity": 100_000})
    md = out["final_trade_decision"]
    assert "Stop-loss not set" not in md
    assert "**Stop Loss**" in md  # stop was computed normally


@pytest.mark.unit
def test_overlay_no_marker_for_sell_without_price(monkeypatch):
    """Sell (target_weight <= 0) + no price -> no marker (no stop expected anyway)."""
    g = _make_graph(risk_enabled=True)
    monkeypatch.setattr(g, "_latest_close_and_atr", lambda t, d, at="stock": (None, None))
    state = {"final_trade_decision": "**Rating**: Sell"}
    out = g._apply_risk_overlay("AAPL", "2024-01-15", state, {"equity": 100_000})
    md = out["final_trade_decision"]
    assert "Stop-loss not set" not in md


@pytest.mark.unit
def test_overlay_marks_stop_computation_failure(monkeypatch, caplog):
    """Bullish position + price present but ATR invalid -> visible marker.

    Closes the F1 silent-degradation gap: when ``close`` is available but the
    ATR stop *computation* throws inside RiskManager.decide (atr <= 0, non-numeric),
    the guard ``target_weight > 0.0 and stop_loss is None`` must still fire --
    unlike the old ``close is None`` guard which only caught missing-data, not
    computation-failure. The manager-side warning is also asserted via caplog.
    """
    import logging

    g = _make_graph(risk_enabled=True)
    # close valid, atr negative -> atr_stop_from_values raises ValueError.
    monkeypatch.setattr(g, "_latest_close_and_atr", lambda t, d, at="stock": (190.0, -1.0))
    state = {"final_trade_decision": "**Rating**: Buy\n\nThesis."}
    with caplog.at_level(logging.WARNING, logger="yialpha.risk.manager"):
        out = g._apply_risk_overlay("AAPL", "2024-01-15", state, {"equity": 100_000})
    md = out["final_trade_decision"]
    # Position is sized but stop failed -> marker visible (the F1 fix).
    assert "Stop-loss not set" in md
    assert "**Stop Loss**" not in md
    # Manager logged the computation failure (root cause traceable).
    assert any(
        "ATR stop-loss computation failed" in r.message for r in caplog.records
    )


@pytest.mark.unit
def test_overlay_coerces_dict_portfolio_state(monkeypatch):
    g = _make_graph(risk_enabled=True)
    monkeypatch.setattr(g, "_latest_close_and_atr", lambda t, d, at="stock": (100.0, 2.0))
    portfolio = {
        "cash": 50_000, "equity": 120_000,
        "positions": {"AAPL": 70_000}, "sectors": {"Tech": 70_000},
        "returns_history": [], "trade_history": [],
    }
    state = {"final_trade_decision": "**Rating**: Buy"}
    out = g._apply_risk_overlay("AAPL", "2024-01-15", state, portfolio)
    assert "Quantitative Risk Overlay" in out["final_trade_decision"]


@pytest.mark.unit
def test_overlay_drawdown_regime_recorded(monkeypatch):
    g = _make_graph(risk_enabled=True)
    monkeypatch.setattr(g, "_latest_close_and_atr", lambda t, d, at="stock": (100.0, 2.0))
    # Deep drawdown: equity 80k vs a 100k peak forces hard-stop via the breaker.
    portfolio = {"equity": 80_000, "cash": 80_000}
    # Prime the breaker by deciding once at the lower equity so it tracks peak.
    state = {"final_trade_decision": "**Rating**: Buy"}
    out = g._apply_risk_overlay("AAPL", "2024-01-15", state, portfolio)
    md = out["final_trade_decision"]
    # Regime line always present; with a fresh peak == 80k the drawdown is 0,
    # so this asserts the section renders without error and includes the field.
    assert "Drawdown Regime" in md


@pytest.mark.unit
def test_overlay_build_failure_marks_decision():
    """When RiskManager build fails, the decision gets a visible DISABLED warning."""
    g = _make_graph(risk_enabled=True)
    # Simulate build failure: risk_manager is None but the degraded flag is set.
    g.risk_manager = None
    g._risk_overlay_degraded = True

    original = "**Rating**: Buy\n\nStrong thesis."
    state = {"final_trade_decision": original}
    out = g._apply_risk_overlay("AAPL", "2024-01-15", state, None)
    md = out["final_trade_decision"]
    # The warning is visible in the decision text.
    assert "⚠️ Quantitative Risk Overlay DISABLED" in md
    assert "no Kelly sizing" in md
    # The original decision is still there, and the rating still parses.
    assert "Strong thesis." in md
    assert parse_rating(md) == "Buy"


@pytest.mark.unit
def test_overlay_decide_failure_marks_decision(monkeypatch):
    """When decide() throws, the decision gets a visible DISABLED warning."""
    g = _make_graph(risk_enabled=True)
    monkeypatch.setattr(g, "_latest_close_and_atr", lambda t, d, at="stock": (190.0, 3.0))
    # Force decide() to raise.
    original_decide = g.risk_manager.decide
    g.risk_manager.decide = lambda *a, **k: (_ for _ in ()).throw(
        RuntimeError("simulated Kelly failure")
    )

    state = {"final_trade_decision": "**Rating**: Buy\n\nThesis."}
    out = g._apply_risk_overlay("AAPL", "2024-01-15", state, {"equity": 100_000})
    md = out["final_trade_decision"]
    assert "⚠️ Quantitative Risk Overlay DISABLED" in md
    assert "overlay computation failed" in md
    # Rating still intact.
    assert parse_rating(md) == "Buy"
    # Restore so other tests using the same graph instance aren't affected.
    g.risk_manager.decide = original_decide


@pytest.mark.unit
def test_overlay_risk_off_does_not_mark():
    """When risk_enabled=False (user choice, not failure), no warning appears."""
    g = _make_graph(risk_enabled=False)
    assert g.risk_manager is None
    assert g._risk_overlay_degraded is False
    original = "**Rating**: Buy\n\nThesis."
    state = {"final_trade_decision": original}
    out = g._apply_risk_overlay("AAPL", "2024-01-15", state, None)
    assert out["final_trade_decision"] == original


@pytest.mark.unit
def test_overlay_prefers_structured_pm_rating(monkeypatch):
    """pm_rating (structured) is preferred over parse_rating(markdown).

    When the PM node extracted the rating directly from its PortfolioDecision,
    the overlay must use it — never re-parsing the markdown. This decouples
    rating extraction from text ordering so a prompt change that moves the
    rating off the first line cannot corrupt the overlay.
    """
    g = _make_graph(risk_enabled=True)
    captured: dict = {}
    original_decide = g.risk_manager.decide

    def spy_decide(ticker, rating, *a, **kw):
        captured["rating"] = rating
        return original_decide(ticker, rating, *a, **kw)

    monkeypatch.setattr(g, "_latest_close_and_atr", lambda t, d, at="stock": (190.0, 3.0))
    monkeypatch.setattr(g.risk_manager, "decide", spy_decide)
    # Markdown says "Sell" but pm_rating says "Buy" — the structured value wins.
    state = {
        "final_trade_decision": "**Rating**: Sell\n\nConfusing markdown.",
        "pm_rating": "Buy",
    }
    out = g._apply_risk_overlay("AAPL", "2024-01-15", state, {"equity": 100_000})
    assert captured["rating"] == "Buy"
    assert "Quantitative Risk Overlay" in out["final_trade_decision"]


@pytest.mark.unit
def test_overlay_falls_back_to_parse_rating_when_pm_rating_empty(monkeypatch):
    """Empty pm_rating -> overlay falls back to parse_rating on the markdown."""
    g = _make_graph(risk_enabled=True)
    captured: dict = {}
    original_decide = g.risk_manager.decide

    def spy_decide(ticker, rating, *a, **kw):
        captured["rating"] = rating
        return original_decide(ticker, rating, *a, **kw)

    monkeypatch.setattr(g, "_latest_close_and_atr", lambda t, d, at="stock": (190.0, 3.0))
    monkeypatch.setattr(g.risk_manager, "decide", spy_decide)
    # pm_rating absent (free-text fallback / old checkpoint) -> parse markdown.
    state = {"final_trade_decision": "**Rating**: Hold\n\nThesis."}
    out = g._apply_risk_overlay("AAPL", "2024-01-15", state, {"equity": 100_000})
    assert captured["rating"] == "Hold"
    assert "Quantitative Risk Overlay" in out["final_trade_decision"]


@pytest.mark.unit
def test_overlay_threads_asset_type_into_price_loader(monkeypatch):
    """crypto_perp runs must price the overlay on the perp venue.

    ``_apply_risk_overlay`` receives ``asset_type`` from ``_run_graph`` and
    forwards it to the close/ATR loader; a perp that fell back to the Yahoo
    path would compute its entry reference / ATR stop on the wrong
    instrument (BTC-USD spot instead of the BTCUSDT perp book).
    """
    g = _make_graph(risk_enabled=True)
    seen: list[str] = []

    def capture(ticker, date, asset_type="stock"):
        seen.append(asset_type)
        return (65000.0, 900.0)

    monkeypatch.setattr(g, "_latest_close_and_atr", capture)
    state = {"final_trade_decision": "**Rating**: Buy\n\nThesis.", "pm_rating": "Buy"}
    g._apply_risk_overlay(
        "BTCUSDT", "2026-06-01", state, {"equity": 100_000}, asset_type="crypto_perp",
    )
    assert seen == ["crypto_perp"]

    # Default stays the historical stock path (byte-stable for non-perp runs).
    seen.clear()
    g._apply_risk_overlay("AAPL", "2026-06-01", state, {"equity": 100_000})
    assert seen == ["stock"]


# --------------------------------------------------------------------------- #
# Perp advisory ticket: deterministic leverage / liquidation bullets
# --------------------------------------------------------------------------- #


def _perp_overlay_md(g, ticker="BTCUSDT", rating="Buy", weight=None):
    """Run the overlay on a perp-shaped decision and return the markdown."""
    state = {
        "final_trade_decision": f"**Rating**: {rating}\n\nThesis.",
        "pm_rating": rating,
    }
    out = g._apply_risk_overlay(
        ticker, "2020-01-15", state, {"equity": 100_000},
        asset_type="crypto_perp",
    )
    return out["final_trade_decision"]


@pytest.mark.unit
def test_perp_run_renders_ticket_bullets(monkeypatch):
    g = _make_graph(risk_enabled=True)
    monkeypatch.setattr(
        g, "_latest_close_and_atr",
        lambda t, d, at="stock": (100.0, 2.0),
    )
    md = _perp_overlay_md(g)
    assert "**Suggested Leverage**" in md
    assert "**Est. Liquidation Price**" in md
    assert "**Funding (7d)**: n/a (historical run; not fetched)" in md
    # 2020-01-15 is far in the past: the funding note must NOT have hit the
    # network — the n/a text is the explicit historical marker.
    # Deterministic math: entry 100, atr 2 -> atr_pct 2%, strength Buy=2.
    # L_vol = 0.30/0.02 = 15x; L_conv(2, conservative) = 5x -> L = 5x.
    assert "≤ 5.0x" in md
    # MMR-aware liq at 100×(1−1/5+0.005+0.0005) = 80.55 — MMR tiers at the
    # DISCLOSED assumed 50,000 USDT notional (0.50%), never at the entry
    # price; stop = 100−2×2 = 96 fires first by design (verified at render).
    # The trigger basis is disclosed on its own bullet.
    assert "80.55" in md and "96" in md
    assert "**Stop Trigger Basis**: CONTRACT_PRICE" in md
    assert "MARK price" in md


@pytest.mark.unit
def test_perp_ticket_short_side_mirrors():
    # Force a negative overlay weight: Sell rating + allow-short semantics via
    # a spied manager would be heavy; instead render through the static helper
    # with a synthesized decision-shaped object.
    from types import SimpleNamespace

    from yialpha.graph.trading_graph import YiAlphaGraph

    decision = SimpleNamespace(
        rating="Sell", action="exit", target_weight=-0.10,
        stop_loss=None, entry_price=100.0,
    )
    out = YiAlphaGraph._render_perp_ticket(
        "BTCUSDT", "2020-01-15", "Sell", decision, 100.0, 2.0,
    )
    assert "**Suggested Leverage**" in out
    # Short: stop mirrored to 100+2*2=104; liq above entry.
    assert "104" in out


@pytest.mark.unit
def test_stock_run_has_no_perp_bullets(monkeypatch):
    g = _make_graph(risk_enabled=True)
    monkeypatch.setattr(
        g, "_latest_close_and_atr", lambda t, d, at="stock": (190.0, 3.0),
    )
    state = {"final_trade_decision": "**Rating**: Buy\n\nThesis.", "pm_rating": "Buy"}
    out = g._apply_risk_overlay("AAPL", "2024-01-15", state, {"equity": 100_000})
    md = out["final_trade_decision"]
    assert "Suggested Leverage" not in md
    assert "Liquidation Price" not in md
    assert "Funding (7d)" not in md


@pytest.mark.unit
def test_perp_ticket_skipped_without_price_or_atr(monkeypatch):
    g = _make_graph(risk_enabled=True)
    monkeypatch.setattr(
        g, "_latest_close_and_atr", lambda t, d, at="stock": (None, None),
    )
    md = _perp_overlay_md(g)
    assert "Suggested Leverage" not in md
    # The pre-existing no-stop warning still fires for the sized position.
    assert "Stop-loss not set" in md


@pytest.mark.unit
def test_perp_ticket_renders_mark_reference(monkeypatch):
    # PR3: the ticket carries the MARK-price basis next to the last-price
    # entry reference — liquidation distance claims anchor on mark. Entry
    # 100 (mocked last close), mark 99.5 -> last−mark ≈ +50.3 bps; liq
    # 80.55 at 5x (MMR @ assumed 50k notional) is 19.0% BELOW the mark.
    g = _make_graph(risk_enabled=True)
    monkeypatch.setattr(
        g, "_latest_close_and_atr",
        lambda t, d, at="stock": (100.0, 2.0),
    )
    monkeypatch.setattr(g, "_latest_mark_close", lambda t, d: 99.5)
    md = _perp_overlay_md(g)
    assert "**Mark Reference**: 99.5" in md
    assert "+50.3 bps" in md
    assert "19.0% from current mark 99.5 to liq" in md
    assert "liquidations trigger on MARK" in md


@pytest.mark.unit
def test_perp_ticket_mark_already_breached_renders_warning():
    # P0: a long whose mark has fallen THROUGH the estimated liquidation
    # level must never read as a calm positive "X% to liq" — the breach is
    # the headline and the stop-first claim is withdrawn.
    from types import SimpleNamespace

    from yialpha.graph.trading_graph import YiAlphaGraph

    decision = SimpleNamespace(
        rating="Buy", action="buy", target_weight=0.10,
        stop_loss=None, entry_price=100.0,
    )
    out = YiAlphaGraph._render_perp_ticket(
        "BTCUSDT", "2020-01-15", "Buy", decision, 100.0, 2.0,
        mark_close=79.0,  # below the long liq estimate 80.55
    )
    assert "ALREADY breached the estimated liquidation level" in out
    assert "stop-first ordering no longer holds" in out
    assert "from current mark" not in out


@pytest.mark.unit
def test_perp_overlay_mark_unavailable_omits_bullet(monkeypatch):
    g = _make_graph(risk_enabled=True)
    monkeypatch.setattr(
        g, "_latest_close_and_atr",
        lambda t, d, at="stock": (100.0, 2.0),
    )
    monkeypatch.setattr(g, "_latest_mark_close", lambda t, d: None)
    md = _perp_overlay_md(g)
    assert "Mark Reference" not in md
    assert "Est. Liquidation Price" in md  # the ticket itself still renders


@pytest.mark.unit
def test_perp_price_book_failure_records_core_sentinel(monkeypatch):
    # PR3 gate: when the overlay's perp price/ATR load FAILS, the failure
    # must reach the quality chain (get_binance_klines core sentinel →
    # DEGRADED_CRITICAL → ticket NO_TRADE) instead of vanishing into a
    # "Stop-loss not set" warning on an otherwise-tradable-looking report.
    from yialpha.dataflows import quality
    from yialpha.graph import trading_graph as tg

    def boom(ticker, trade_date, asset_type="stock"):
        raise RuntimeError("klines endpoint down")

    monkeypatch.setattr(tg, "_memoized_close_and_atr", boom)
    g = _make_graph(risk_enabled=True)
    quality.ensure_run_context()
    try:
        md = _perp_overlay_md(g)
        events = quality.snapshot_quality()
    finally:
        quality.reset_quality()

    assert "Stop-loss not set" in md
    core = [e for e in events if e["method"] == "get_binance_klines"]
    assert core and core[0]["kind"] == quality.KIND_OPTIONAL_UNAVAILABLE
    # The veto lands on the ticket rendered into the same decision markdown.
    assert "NO_TRADE (critical_data_missing)" in md


@pytest.mark.unit
def test_perp_ticket_math_module_pinned():
    """The extracted module reproduces the script's documented caps exactly."""
    from yialpha.risk.perp_ticket import (
        compute_leverage,
        liquidation_price,
        take_profits,
    )

    L, detail = compute_leverage(0.04, 0.03, "crypto_perp", 2, "conservative")
    assert pytest.approx(5.0) == L  # conviction cap binds
    assert detail["L_liq"] == pytest.approx(12.5)
    assert detail["L_vol"] == pytest.approx(10.0)
    assert detail["L_hard"] == pytest.approx(20.0)

    # MMR-aware trigger (the backtest engine's formula): long
    # 100×(1−1/5+0.005+0.0005) = 80.55, short mirrored 119.45. The static
    # default bracket at the ASSUMED 50k USDT notional gives MMR 0.50% +
    # 5bps fee est.
    assert liquidation_price(100.0, 5.0, "long", "crypto_perp") == pytest.approx(80.55)
    assert liquidation_price(100.0, 5.0, "short", "crypto_perp") == pytest.approx(119.45)
    assert liquidation_price(100.0, 1.0, "long", "crypto_perp") is None
    assert liquidation_price(100.0, 5.0, "long", "crypto_spot") is None

    assert take_profits(100.0, 96.0, "long") == pytest.approx([106.0, 112.0, 120.0])
    assert take_profits(100.0, 104.0, "short") == pytest.approx([94.0, 88.0, 80.0])
    assert take_profits(None, 96.0, "long") == []


@pytest.mark.unit
def test_liquidation_formula_matches_backtest_engine():
    """The advisory liq price is the SAME MMR-aware trigger the backtest
    engine simulates (backtest/engine.py): entry×(1−1/L+MMR+fee) for longs,
    mirrored for shorts — ticket and simulated equity curve cannot disagree.
    MMR tiers at the notional PASSED to the ticket (the same
    notional→bracket lookup engine.py does), not at the entry price."""
    from yialpha.dataflows.binance_brackets import (
        DEFAULT_USDT_M_BRACKETS,
        mmr_for_notional,
    )
    from yialpha.risk.perp_ticket import LIQ_FEE_RATE_EST, liquidation_price

    for entry, lev, notional in (
        (100.0, 5.0, 10_000.0), (60_000.0, 10.0, 250_000.0), (123.4, 3.0, 60_000.0),
    ):
        mmr = mmr_for_notional(DEFAULT_USDT_M_BRACKETS, notional)
        assert liquidation_price(
            entry, lev, "long", "crypto_perp", notional=notional,
        ) == pytest.approx(entry * (1.0 - 1.0 / lev + mmr + LIQ_FEE_RATE_EST))
        assert liquidation_price(
            entry, lev, "short", "crypto_perp", notional=notional,
        ) == pytest.approx(entry * (1.0 + 1.0 / lev - mmr - LIQ_FEE_RATE_EST))


@pytest.mark.unit
def test_liquidation_gets_closer_as_mmr_tier_rises():
    """Bigger notional → higher MMR bracket → liq CLOSER to entry. The old
    1/L-only formula could not see position-size risk at all and placed the
    estimate farther than reality (systematically understating it)."""
    from yialpha.risk.perp_ticket import liquidation_price

    ratio_small = (
        liquidation_price(100.0, 10.0, "long", "crypto_perp", notional=1_000.0)
        / 100.0
    )
    ratio_large = (
        liquidation_price(100.0, 10.0, "long", "crypto_perp", notional=200_000.0)
        / 100.0
    )
    # 1k notional: MMR 0.40% → long liq ratio 0.9045; 200k notional: MMR
    # 0.60% → 0.9065 — higher tier, less room to the trigger. Assert the
    # ordering, not the decimals.
    assert ratio_small < ratio_large


@pytest.mark.unit
def test_perp_ticket_numbers_disclose_liq_assumptions():
    from yialpha.risk.perp_ticket import ASSUMED_NOTIONAL_USDT, perp_ticket_numbers

    result = perp_ticket_numbers(100.0, 2.0, "Buy", None, 0.1)
    assert result is not None
    _lev, detail, liq, _stop = result
    assert liq is not None
    # Default: MMR tiers at the DISCLOSED assumed notional (50k USDT →
    # 0.50%), never at the entry price pretending to be a position size.
    assert detail["liq_mmr"] == pytest.approx(0.005)
    assert detail["liq_fee_rate"] == pytest.approx(0.0005)
    assert "assumed 50,000 USDT notional" in detail["liq_mmr_basis"]
    assert "rises with size" in detail["liq_mmr_basis"]
    # An explicit real notional overrides the assumption and says so.
    sized = perp_ticket_numbers(100.0, 2.0, "Buy", None, 0.1, notional=200_000.0)
    assert sized is not None
    _l, sized_detail, sized_liq, _s = sized
    assert sized_liq is not None
    assert sized_detail["liq_mmr"] == pytest.approx(0.006)
    assert "notional 200,000 USDT" in sized_detail["liq_mmr_basis"]
    assert ASSUMED_NOTIONAL_USDT == 50_000.0


@pytest.mark.unit
def test_perp_live_run_funding_gate_scales_overlay(monkeypatch):
    """A live perp run feeds trailing funding into the risk gate: strongly
    adverse carry halves the overlay's target weight and says so in the
    rationale; the funding-note bullet reads the same number."""
    from datetime import date

    g = _make_graph(risk_enabled=True)
    monkeypatch.setattr(
        g, "_latest_close_and_atr", lambda t, d, at="stock": (100.0, 2.0),
    )
    today = date.today().strftime("%Y-%m-%d")

    # Adverse 50%/yr carry: 7-day sum = 0.50 * 7 / 365. The closure switch
    # lets the historical twin flip the fetch off without re-patching.
    fetch = {"total": 0.50 * 7 / 365}
    monkeypatch.setattr(
        g, "_trailing_funding_total", lambda t, d: fetch["total"],
    )
    state = {"final_trade_decision": "**Rating**: Buy\n\nThesis.", "pm_rating": "Buy"}
    out = g._apply_risk_overlay(
        "BTCUSDT", today, state, {"equity": 100_000}, asset_type="crypto_perp",
    )
    md = out["final_trade_decision"]
    assert "Funding drag +50%/yr" in md
    assert "Funding (7d)" in md and "longs pay" in md

    # Same setup with the funding fetch unavailable (historical twin): full
    # weight and an explicit n/a note. Fresh state dict — the overlay mutates
    # its input in place, so reusing `state` would stack both overlays.
    fetch["total"] = None
    out2 = g._apply_risk_overlay(
        "BTCUSDT", today,
        {"final_trade_decision": "**Rating**: Buy\n\nThesis.", "pm_rating": "Buy"},
        {"equity": 100_000}, asset_type="crypto_perp",
    )
    md2 = out2["final_trade_decision"]
    assert "Funding drag" not in md2
    assert "n/a (fetch failed)" in md2
    # The gate halved the first run's target weight relative to this twin.
    import re

    def _weight(text):
        m = re.search(r"\*\*Target Weight\*\*:\s*([0-9.]+)%", text)
        return float(m.group(1))

    assert _weight(md) == pytest.approx(_weight(md2) * 0.5, rel=1e-6)


@pytest.mark.unit
def test_stock_run_never_fetches_funding(monkeypatch):
    g = _make_graph(risk_enabled=True)
    calls = {"n": 0}

    def boom(t, d):
        calls["n"] += 1
        raise AssertionError("stock runs must not touch the funding vendor")

    monkeypatch.setattr(g, "_trailing_funding_total", boom)
    monkeypatch.setattr(
        g, "_latest_close_and_atr", lambda t, d, at="stock": (190.0, 3.0),
    )
    state = {"final_trade_decision": "**Rating**: Buy\n\nThesis.", "pm_rating": "Buy"}
    g._apply_risk_overlay("AAPL", "2024-01-15", state, {"equity": 100_000})
    assert calls["n"] == 0
