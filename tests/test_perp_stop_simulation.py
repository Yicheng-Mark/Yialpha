"""Stop-trigger simulation: the risk overlay's ATR stop as a GTC order.

2026-08-17 round — the overlay's stop-loss used to be advisory metadata only
(``manager.py`` warned "this close-price backtest does not simulate
stop-trigger fills"). With leverage, the stop is the PRIMARY defence before
liquidation, so a levered backtest that never fires stops overstates risk
taken and understates exit discipline. Pinned here:

* Default: ON for crypto_perp, OFF (byte-identical) for spot/stock.
* Long/short stop exits fill AT the stop level with slippage + taker fee,
  triggered by the bar's adverse extreme even when the close recovers.
* A bar piercing both stop and liquidation conservatively liquidates first.
* GTC semantics: a decision without a stop keeps the previous level resting;
  a full close disarms it; a direction-flip invalidates the old level.
* A signal landing on the stop bar may still re-enter at that bar's close.

All hermetic: synthetic providers, weight_fn stamps ctx["risk_decision"],
no network, no LLM.
"""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from tests.test_backtest_engine import FakeGraph
from yialpha.backtest.engine import run_backtest


def _flat(ticker, start, end, level: float = 100.0):
    idx = pd.bdate_range(start, end)
    return pd.Series([level] * len(idx), index=idx.strftime("%Y-%m-%d"),
                     dtype=float)


def _extremes(lows: list[float], highs: list[float], pad_from_start: int):
    """Extremes provider: first ``pad_from_start`` bars get neutral lows/highs,
    then the scripted low/high sequences (used to pierce a stop on one bar)."""

    def provider(ticker, start, end):
        idx = pd.bdate_range(start, end)
        n = len(idx)
        low_vals = [99.5] * pad_from_start + lows + [99.5] * (
            n - pad_from_start - len(lows))
        high_vals = [100.5] * pad_from_start + highs + [100.5] * (
            n - pad_from_start - len(highs))
        return (
            pd.Series(low_vals[:n], index=idx.strftime("%Y-%m-%d"), dtype=float),
            pd.Series(high_vals[:n], index=idx.strftime("%Y-%m-%d"), dtype=float),
        )

    return provider


def _stop_weight_fn(weight: float = 1.0, stop_after_fill: float = 97.0):
    """weight_fn that arms a stop from the second fill onward (GTC).

    The first decision opens the position and publishes its stop; every later
    decision re-publishes the same level (GTC replace). ``stop_after_fill``
    simulates the overlay's ``close - mult*ATR`` level computed at decision
    time.
    """

    def fn(rating: str, date: str, ctx: dict) -> float | None:
        ctx["risk_decision"] = SimpleNamespace(
            action="Buy" if weight > 0 else "Sell", stop_loss=stop_after_fill,
        )
        return weight

    return fn


_KW = {
    "initial_capital": 100_000.0, "holding_days": 5,
    "price_provider": _flat, "compute_index_alpha": False,
    "periods_per_year": 365, "cost_bps": 0.0,
}


def _funding(rate: float):
    def provider(ticker, start, end):
        idx = pd.bdate_range(start, end)
        return pd.Series([rate] * len(idx), index=idx.strftime("%Y-%m-%d"),
                         dtype=float)
    return provider


# --------------------------------------------------------------------------- #
# Triggering + fill mechanics
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_long_stop_wick_exits_at_level():
    """Bar low pierces the 97 stop while the close stays 100: force-exit AT 97."""
    # Signal 2024-01-01 fills at 2024-01-02's close (100), stop 97 armed.
    # Bar 2024-01-03 wicks to 94 -> stop exit at 97.0, position flat after.
    res = run_backtest(
        FakeGraph({"2024-01-01": "Buy"}), "BTCUSDT", ["2024-01-01"],
        asset_type="crypto_perp", funding_provider=_funding(0.0),
        weight_fn=_stop_weight_fn(1.0, 97.0),
        extremes_provider=_extremes([94.0], [100.5], 2),
        **_KW,
    )
    events = res.config_summary.get("perp_stop_triggers")
    assert events and len(events) == 1
    ev = events[0]
    assert ev["side"] == "long"
    assert ev["stop_price"] == pytest.approx(97.0)
    assert ev["exit_price"] == pytest.approx(97.0)  # slip=0 fee=0
    assert ev["date"] == "2024-01-03"
    # 1000 shares in at 100, out at 97: equity locks the -3% and stays flat.
    assert res.equity[-1] == pytest.approx(97_000.0)
    assert res.config_summary["perp_stop_trigger_count"] == 1


@pytest.mark.unit
def test_stop_exit_pays_slippage_and_taker_fee():
    res = run_backtest(
        FakeGraph({"2024-01-01": "Buy"}), "BTCUSDT", ["2024-01-01"],
        asset_type="crypto_perp", funding_provider=_funding(0.0),
        weight_fn=_stop_weight_fn(1.0, 97.0),
        extremes_provider=_extremes([94.0], [100.5], 2),
        taker_bps=5.0, slippage_bps=10.0, **{**_KW, "cost_bps": 0.0},
    )
    ev = res.config_summary["perp_stop_triggers"][0]
    # Adverse slippage on the stop fill: 97 x (1 - 10bps).
    assert ev["exit_price"] == pytest.approx(97.0 * (1 - 10 / 10_000), rel=1e-9)
    # Taker fee on the exit notional (no BNB discount here).
    assert ev["fee"] == pytest.approx(1000.0 * ev["exit_price"] * 5 / 10_000)
    # Entry fill slipped too, so the locked loss is a touch worse than -3%.
    assert res.equity[-1] < 97_000.0


@pytest.mark.unit
def test_no_trigger_when_extreme_stays_above_stop():
    """Low 98 vs stop 97: nothing fires, position marks flat at 100."""
    res = run_backtest(
        FakeGraph({"2024-01-01": "Buy"}), "BTCUSDT", ["2024-01-01"],
        asset_type="crypto_perp", funding_provider=_funding(0.0),
        weight_fn=_stop_weight_fn(1.0, 97.0),
        extremes_provider=_extremes([98.0], [100.5], 2),
        **_KW,
    )
    assert "perp_stop_triggers" not in res.config_summary
    assert res.equity[-1] == pytest.approx(100_000.0)


@pytest.mark.unit
def test_short_stop_triggers_on_upside_extreme():
    """Short with a stop ABOVE entry: the bar's HIGH pierces it."""
    res = run_backtest(
        FakeGraph({"2024-01-01": "Sell"}), "BTCUSDT", ["2024-01-01"],
        asset_type="crypto_perp", allow_short=True,
        funding_provider=_funding(0.0),
        weight_fn=_stop_weight_fn(-1.0, 103.0),
        extremes_provider=_extremes([99.5], [105.0], 2),
        **_KW,
    )
    ev = res.config_summary["perp_stop_triggers"][0]
    assert ev["side"] == "short"
    assert ev["stop_price"] == pytest.approx(103.0)
    # Short 1000 @ 100, stopped at 103: -3%.
    assert res.equity[-1] == pytest.approx(97_000.0)


# --------------------------------------------------------------------------- #
# Ordering vs liquidation + defaults
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_liq_wins_when_bar_pierces_both_levels():
    """10x long: liq at 90.4% of entry, stop at 95. A low of 85 pierces both —
    the conservative order liquidates (stop never fires on a dead position)."""
    res = run_backtest(
        FakeGraph({"2024-01-01": "Buy"}), "BTCUSDT", ["2024-01-01"],
        asset_type="crypto_perp", funding_provider=_funding(0.0),
        weight_fn=_stop_weight_fn(1.0, 95.0),
        extremes_provider=_extremes([85.0], [100.5], 2),
        leverage=10.0, brackets_provider=lambda t: [(0.0, 0.004)],
        **_KW,
    )
    assert res.config_summary.get("perp_liquidations")
    assert "perp_stop_triggers" not in res.config_summary
    # Liquidation math pinned elsewhere (test_perp_engine B4): margin
    # remainder at the trigger.
    assert res.equity[-1] == pytest.approx(
        100_000.0 * (1.0 - 10.0 * (1.0 - 0.904)), rel=1e-6)


@pytest.mark.unit
def test_perp_default_on_stock_default_off():
    """config_summary discloses the simulation mode; a stock run with the same
    stop-supplying weight_fn stays byte-identical (no simulated exit)."""
    perp = run_backtest(
        FakeGraph({"2024-01-01": "Buy"}), "BTCUSDT", ["2024-01-01"],
        asset_type="crypto_perp", funding_provider=_funding(0.0),
        weight_fn=_stop_weight_fn(1.0, 97.0),
        extremes_provider=_extremes([94.0], [100.5], 2),
        **_KW,
    )
    assert perp.config_summary["perp_stop_simulation"] is True
    assert "GTC stop-market" in perp.config_summary["perp_model_note"]

    stock = run_backtest(
        FakeGraph({"2024-01-01": "Buy"}), "AAPL", ["2024-01-01"],
        weight_fn=_stop_weight_fn(1.0, 97.0),
        **{k: v for k, v in _KW.items() if k != "periods_per_year"},
    )
    # Historical behaviour: the stop is stamped on the row but never fires —
    # the position marks flat through the 94 wick.
    assert stock.trades[0].stop_loss == pytest.approx(97.0)
    assert "perp_stop_triggers" not in stock.config_summary
    assert stock.equity[-1] == pytest.approx(100_000.0)


@pytest.mark.unit
def test_perp_opt_out_restores_advisory_only():
    res = run_backtest(
        FakeGraph({"2024-01-01": "Buy"}), "BTCUSDT", ["2024-01-01"],
        asset_type="crypto_perp", funding_provider=_funding(0.0),
        weight_fn=_stop_weight_fn(1.0, 97.0),
        extremes_provider=_extremes([94.0], [100.5], 2),
        simulate_stop_triggers=False, **_KW,
    )
    assert res.config_summary["perp_stop_simulation"] is False
    assert "advisory only" in res.config_summary["perp_model_note"]
    assert "perp_stop_triggers" not in res.config_summary
    assert res.equity[-1] == pytest.approx(100_000.0)


# --------------------------------------------------------------------------- #
# GTC semantics + re-entry
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_stop_survives_hold_and_reentry_same_bar():
    """A second Buy signal on the stop bar re-enters at that bar's close."""
    res = run_backtest(
        FakeGraph({"2024-01-01": "Buy", "2024-01-03": "Buy"}),
        "BTCUSDT", ["2024-01-01", "2024-01-03"],
        asset_type="crypto_perp", funding_provider=_funding(0.0),
        weight_fn=_stop_weight_fn(1.0, 97.0),
        extremes_provider=_extremes([94.0], [100.5], 2),
        **_KW,
    )
    ev = res.config_summary["perp_stop_triggers"][0]
    assert ev["date"] == "2024-01-03"
    # Second trade executes 2024-01-04 close and re-opens: final equity marks
    # a live position again (flat prices -> back to the 100k entry value).
    assert res.trades[-1].executed_weight == pytest.approx(1.0)
    assert res.equity[-1] == pytest.approx(97_000.0, rel=1e-6)


@pytest.mark.unit
def test_full_close_disarms_stop():
    """Weight 0 on the second decision closes the position; a later wick
    through the OLD stop level must not fire on a flat book."""
    calls = {"n": 0}

    def fn(rating: str, date: str, ctx: dict) -> float | None:
        calls["n"] += 1
        weight = 1.0 if calls["n"] == 1 else 0.0
        ctx["risk_decision"] = SimpleNamespace(action="Buy", stop_loss=97.0)
        return weight

    res = run_backtest(
        FakeGraph({"2024-01-01": "Buy", "2024-01-02": "Sell"}),
        "BTCUSDT", ["2024-01-01", "2024-01-02"],
        asset_type="crypto_perp", funding_provider=_funding(0.0),
        weight_fn=fn,
        extremes_provider=_extremes([94.0], [100.5], 3),
        **_KW,
    )
    assert "perp_stop_triggers" not in res.config_summary
    assert res.equity[-1] == pytest.approx(100_000.0)


@pytest.mark.unit
def test_gtc_stop_kept_when_later_decision_lacks_one():
    """Decision 2 publishes no stop: the decision-1 level keeps resting and
    still fires — broker GTC behaviour, not fire-and-forget metadata."""
    calls = {"n": 0}

    def fn(rating: str, date: str, ctx: dict) -> float | None:
        calls["n"] += 1
        ctx["risk_decision"] = SimpleNamespace(
            action="Buy",
            stop_loss=97.0 if calls["n"] == 1 else None,
        )
        return 1.0

    res = run_backtest(
        FakeGraph({"2024-01-01": "Buy", "2024-01-02": "Buy"}),
        "BTCUSDT", ["2024-01-01", "2024-01-02"],
        asset_type="crypto_perp", funding_provider=_funding(0.0),
        weight_fn=fn,
        extremes_provider=_extremes([94.0], [100.5], 3),
        **_KW,
    )
    assert len(res.config_summary["perp_stop_triggers"]) == 1
    assert res.equity[-1] == pytest.approx(97_000.0)


@pytest.mark.unit
def test_wrong_side_stop_never_arms():
    """A long-formula stop (below entry) arriving while SHORT must not arm an
    instantly-triggering order."""
    calls = {"n": 0}

    def fn(rating: str, date: str, ctx: dict) -> float | None:
        calls["n"] += 1
        # Long-shaped stop 97 on a short entered at 100 — invalid side.
        ctx["risk_decision"] = SimpleNamespace(action="Sell", stop_loss=97.0)
        return -1.0

    res = run_backtest(
        FakeGraph({"2024-01-01": "Sell"}), "BTCUSDT", ["2024-01-01"],
        asset_type="crypto_perp", allow_short=True,
        funding_provider=_funding(0.0),
        weight_fn=fn,
        extremes_provider=_extremes([99.5], [105.0], 2),
        **_KW,
    )
    assert "perp_stop_triggers" not in res.config_summary
    # Short held through the 105 INTRABAR high; closes stayed flat at 100,
    # so the book marks flat — the wrong-side stop neither fired (which would
    # have been a phantom profit exit at 97) nor armed.
    assert res.equity[-1] == pytest.approx(100_000.0)
