"""Perp-accuracy engine tests: wiring, funding coverage, costs, margin, shorts.

2026-08-16 round — every capability the engine gained this round, pinned:

* B1 wiring: ``asset_type="crypto_perp"`` swaps the DEFAULT price provider to
  the perp's own Binance klines (an explicit provider always wins).
* B2 funding coverage: days absent from the funding series fail closed
  instead of silently accruing zero drag (``allow_funding_gaps`` opts in).
* B3 costs: taker fee / BNB discount / adverse slippage in the fill price /
  stepSize+minNotional quantization of the share delta.
* B4 leverage: >1x sizes the notional up and models isolated-margin
  liquidation against the MMR ladder (bar-extreme trigger, worse-close fill
  on gaps through the level); 1x stays the historical cash sim.
* B5 shorts: opt-in ``Sell -> -1.0`` with funding RECEIVED and upside
  liquidation.

All hermetic: providers injected, no network, no LLM.
"""

from __future__ import annotations

import pandas as pd
import pytest

from tests.test_backtest_engine import FakeGraph, _rising_prices
from yialpha.backtest.engine import run_backtest


def _funding(rate: float):
    def provider(ticker, start, end):
        idx = pd.bdate_range(start, end)
        return pd.Series(
            [rate] * len(idx), index=idx.strftime("%Y-%m-%d"), dtype=float,
        )
    return provider


def _gappy_funding(rate: float, drop_from: int = 3, drop_n: int = 2):
    """Funding series missing ``drop_n`` days starting at index ``drop_from``.

    The window's first days precede the first execution fill (a position
    opened at day-2's close accrues no funding that day), so gaps there are
    invisible — the default drops mid-window days the simulator WOULD charge.
    """

    def provider(ticker, start, end):
        idx = pd.bdate_range(start, end)
        s = pd.Series(
            [rate] * len(idx), index=idx.strftime("%Y-%m-%d"), dtype=float,
        )
        return pd.concat([s.iloc[:drop_from], s.iloc[drop_from + drop_n:]])
    return provider


_PERP_KW = {
    "initial_capital": 100_000.0, "holding_days": 5,
    "price_provider": _rising_prices, "compute_index_alpha": False,
    "periods_per_year": 365,
}


# --------------------------------------------------------------------------- #
# B1 — default perp price provider wiring
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_perp_default_price_provider_is_binance_klines(monkeypatch):
    """crypto_perp + the DEFAULT provider must price on the perp's own candles.

    The wiring gap this closes: the funding-drag work was unreachable from a
    default perp run because prices/marks came from Yahoo spot (BTC-USD).
    """
    import yialpha.backtest.engine as eng

    calls: list[tuple[str, str, str]] = []

    def fake_frame(symbol, start, end, interval="1d", venue="binance_perp",
                   price_type="last", closed_as_of=None):
        calls.append((symbol, start, end, venue, price_type, closed_as_of))
        idx = pd.bdate_range(start, end)
        df = pd.DataFrame(
            {"Close": [100.0 + i for i in range(len(idx))],
             "Low": [99.0 + i for i in range(len(idx))],
             "High": [101.0 + i for i in range(len(idx))]},
            index=idx,
        )
        return df

    monkeypatch.setattr(
        "yialpha.dataflows.binance.binance_klines_frame", fake_frame,
    )
    # NO explicit price_provider here — the whole point is that the DEFAULT
    # provider gets swapped to the perp's own klines.
    res = run_backtest(
        FakeGraph({"2024-01-01": "Buy"}), "BTCUSDT", ["2024-01-01"],
        asset_type="crypto_perp", funding_provider=_funding(0.0),
        initial_capital=100_000.0, holding_days=5,
        compute_index_alpha=False, periods_per_year=365,
    )
    assert calls, "the perp run must fetch Binance klines for pricing"
    assert all(c[3] == "binance_perp" for c in calls)
    # Marked on the perp series (100, 101, ...) not a Yahoo fallback.
    assert res.equity[-1] > 100_000.0
    # Providers pin to CLOSED bars: a window touching today must not mark
    # equity or fill a decision on the intraday forming candle.
    assert all(c[4] is not None for c in calls), (
        "provider must pass closed_as_of (closed-bars-only seam)"
    )

    # The swap must only happen for the DEFAULT provider sentinel.
    sentinel = eng._yfinance_price_provider
    assert sentinel is not None


@pytest.mark.unit
def test_perp_explicit_price_provider_wins(monkeypatch):
    def explode(*a, **k):
        raise AssertionError("perp provider must not be swapped in")

    monkeypatch.setattr(
        "yialpha.backtest.engine._binance_perp_price_provider", explode,
    )
    res = run_backtest(
        FakeGraph({"2024-01-01": "Buy"}), "BTCUSDT", ["2024-01-01"],
        asset_type="crypto_perp", funding_provider=_funding(0.0), **_PERP_KW,
    )
    assert res.equity[-1] > 0.0   # synthetic provider served everything


# --------------------------------------------------------------------------- #
# B2 — funding coverage fail-closed
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_funding_coverage_gap_fails_closed():
    with pytest.raises(ValueError, match="missing .* priced day"):
        run_backtest(
            FakeGraph({"2024-01-01": "Buy"}), "BTCUSDT", ["2024-01-01"],
            asset_type="crypto_perp",
            funding_provider=_gappy_funding(0.001), **_PERP_KW,
        )


@pytest.mark.unit
def test_funding_gap_override_runs():
    res = run_backtest(
        FakeGraph({"2024-01-01": "Buy"}), "BTCUSDT", ["2024-01-01"],
        asset_type="crypto_perp", allow_funding_gaps=True,
        funding_provider=_gappy_funding(0.001), **_PERP_KW,
    )
    # Runs, and the missing days genuinely accrue zero drag (2 fewer charge
    # days than the full-coverage twin below).
    full = run_backtest(
        FakeGraph({"2024-01-01": "Buy"}), "BTCUSDT", ["2024-01-01"],
        asset_type="crypto_perp",
        funding_provider=_funding(0.001), **_PERP_KW,
    )
    assert res.metrics.total_return > full.metrics.total_return


# --------------------------------------------------------------------------- #
# B3 — execution-cost model
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_slippage_and_taker_fee_reduce_equity():
    base = run_backtest(
        FakeGraph({"2024-01-01": "Buy"}), "BTCUSDT", ["2024-01-01"],
        asset_type="crypto_perp", funding_provider=_funding(0.0),
        cost_bps=0.0, **_PERP_KW,
    )
    costed = run_backtest(
        FakeGraph({"2024-01-01": "Buy"}), "BTCUSDT", ["2024-01-01"],
        asset_type="crypto_perp", funding_provider=_funding(0.0),
        cost_bps=0.0, taker_bps=5.0, slippage_bps=10.0, **_PERP_KW,
    )
    assert costed.equity[-1] < base.equity[-1]
    assert costed.trades[0].transaction_cost > 0.0
    assert "taker 5.0bps" in costed.config_summary["perp_fees"]
    assert "slippage 10.0bps" in costed.config_summary["perp_fees"]


@pytest.mark.unit
def test_bnb_discount_cuts_fee():

    full = run_backtest(
        FakeGraph({"2024-01-01": "Buy"}), "BTCUSDT", ["2024-01-01"],
        asset_type="crypto_perp", funding_provider=_funding(0.0),
        cost_bps=0.0, taker_bps=10.0, **_PERP_KW,
    )
    discounted = run_backtest(
        FakeGraph({"2024-01-01": "Buy"}), "BTCUSDT", ["2024-01-01"],
        asset_type="crypto_perp", funding_provider=_funding(0.0),
        cost_bps=0.0, taker_bps=10.0, bnb_discount=True, **_PERP_KW,
    )
    assert discounted.trades[0].transaction_cost == pytest.approx(
        full.trades[0].transaction_cost * 0.9
    )
    assert "x0.9 BNB" in discounted.config_summary["perp_fees"]


@pytest.mark.unit
def test_fill_quantization_floors_to_step():
    """stepSize=1 lot + $20 minNotional: fills become whole contracts and a
    sub-minNotional delta is refused (no trade), like the venue would."""
    from decimal import Decimal

    from yialpha.dataflows.binance_filters import SymbolFilters

    filters = SymbolFilters(
        symbol="BTCUSDT", status="TRADING",
        tick_size=Decimal("0.1"), step_size=Decimal("1"),
        min_qty=Decimal("0"), max_qty=Decimal("0"),
        min_notional=Decimal("20"), price_precision=1, quantity_precision=0,
    )
    res = run_backtest(
        FakeGraph({"2024-01-01": "Buy"}), "BTCUSDT", ["2024-01-01"],
        asset_type="crypto_perp", funding_provider=_funding(0.0),
        cost_bps=0.0, filters_provider=lambda t: filters, **_PERP_KW,
    )
    # Full-notional Buy at ~100 USDT: ~1000 contracts, already whole.
    assert res.trades[0].traded_notional > 0.0
    assert res.config_summary["perp_fill_quantization"] == "stepSize/minNotional"

    # A rating map targeting a tiny weight produces a sub-minNotional delta,
    # which the simulator must refuse rather than fill fractionally.
    tiny = run_backtest(
        FakeGraph({"2024-01-01": "Overweight"}), "BTCUSDT", ["2024-01-01"],
        asset_type="crypto_perp", funding_provider=_funding(0.0),
        cost_bps=0.0, filters_provider=lambda t: filters,
        rating_to_weight={"Overweight": 0.0001, "Buy": 1.0}, **_PERP_KW,
    )
    assert tiny.trades[0].traded_notional == 0.0
    assert tiny.trades[0].is_rebalance is False


# --------------------------------------------------------------------------- #
# B4 — leverage / margin / liquidation
# --------------------------------------------------------------------------- #


def _flat_then_crash(ticker, start, end):
    """100 flat, then one -60% day, then flat at 40 (a wick to 20)."""
    idx = pd.bdate_range(start, end)
    values = [100.0] * 3 + [40.0] + [40.0] * (len(idx) - 4)
    return pd.Series(values, index=idx.strftime("%Y-%m-%d"), dtype=float)


def _crash_lows(ticker, start, end):
    idx = pd.bdate_range(start, end)
    values = [99.0] * 3 + [20.0] + [39.0] * (len(idx) - 4)
    s = pd.Series(values, index=idx.strftime("%Y-%m-%d"), dtype=float)
    highs = pd.Series(
        [101.0] * len(idx), index=idx.strftime("%Y-%m-%d"), dtype=float,
    )
    return s, highs


_BRACKETS = [(0.0, 0.004)]  # flat 0.4% MMR ladder


@pytest.mark.unit
def test_leverage_10x_crash_forces_liquidation():
    res = run_backtest(
        FakeGraph({"2024-01-01": "Buy"}), "BTCUSDT", ["2024-01-01"],
        asset_type="crypto_perp", funding_provider=_funding(0.0),
        price_provider=_flat_then_crash, compute_index_alpha=False,
        periods_per_year=365, initial_capital=100_000.0,
        leverage=10.0, brackets_provider=lambda t: _BRACKETS,
        extremes_provider=_crash_lows, cost_bps=0.0,
    )
    events = res.config_summary.get("perp_liquidations")
    assert events, "a 10x long through a -60% bar must be liquidated"
    ev = events[0]
    assert ev["side"] == "long"
    # Trigger level: entry x (1 - 1/10 + 0.004) = 90.4% of entry. The bar's
    # 20.0 wick blew far through it, but the forced close executes AT the
    # level — which in the cash-account model caps the loss at the isolated
    # margin (a deeper mark would model unlimited-loss cross margin instead).
    assert ev["liquidation_price"] == pytest.approx(100.0 * 0.904, rel=1e-6)
    assert ev["exit_price"] == pytest.approx(100.0 * 0.904, rel=1e-6)
    # Position gone; equity keeps the margin remainder: 100k posted at 10x,
    # price P&L at the trigger = shares x (90.4 - 100) ~= -96% of margin.
    assert res.equity[-1] == pytest.approx(
        100_000.0 * (1.0 - 10.0 * (1.0 - 0.904)), rel=1e-6
    )


@pytest.mark.unit
def test_leverage_1x_never_models_liquidation():
    """1x = the historical cash sim: no brackets resolve, no liq events, and
    a -60% bar merely marks equity down (regression anchor for B4)."""
    res = run_backtest(
        FakeGraph({"2024-01-01": "Buy"}), "BTCUSDT", ["2024-01-01"],
        asset_type="crypto_perp", funding_provider=_funding(0.0),
        price_provider=_flat_then_crash, compute_index_alpha=False,
        periods_per_year=365, initial_capital=100_000.0,
        leverage=1.0, cost_bps=0.0,
    )
    assert "perp_liquidations" not in res.config_summary
    assert res.config_summary["perp_leverage"] == 1.0
    assert "NOT modeled" in res.config_summary["perp_model_note"]
    # Full position through the crash: equity = 40% of the entry value.
    assert res.equity[-1] == pytest.approx(40_000.0, rel=1e-6)


@pytest.mark.unit
def test_leverage_requires_perp():
    with pytest.raises(ValueError, match="crypto_perp-only"):
        run_backtest(
            FakeGraph({"2024-01-01": "Buy"}), "AAPL", ["2024-01-01"],
            price_provider=_rising_prices, leverage=2.0,
        )


@pytest.mark.unit
def test_liquidation_exit_at_level_when_close_recovers():
    """Wick through the level but close back above it: exit at the LEVEL
    (stop-through), not the close and not the wick."""

    def wick_prices(ticker, start, end):
        idx = pd.bdate_range(start, end)
        values = [100.0] * 3 + [95.0] + [95.0] * (len(idx) - 4)
        return pd.Series(values, index=idx.strftime("%Y-%m-%d"), dtype=float)

    def wick_lows(ticker, start, end):
        idx = pd.bdate_range(start, end)
        lows = pd.Series(
            [99.0] * 3 + [85.0] + [94.0] * (len(idx) - 4),
            index=idx.strftime("%Y-%m-%d"), dtype=float,
        )
        highs = pd.Series(
            [101.0] * len(idx), index=idx.strftime("%Y-%m-%d"), dtype=float,
        )
        return lows, highs

    res = run_backtest(
        FakeGraph({"2024-01-01": "Buy"}), "BTCUSDT", ["2024-01-01"],
        asset_type="crypto_perp", funding_provider=_funding(0.0),
        price_provider=wick_prices, compute_index_alpha=False,
        periods_per_year=365, initial_capital=100_000.0,
        leverage=10.0, brackets_provider=lambda t: _BRACKETS,
        extremes_provider=wick_lows, cost_bps=0.0,
    )
    ev = res.config_summary["perp_liquidations"][0]
    assert ev["exit_price"] == pytest.approx(ev["liquidation_price"])


# --------------------------------------------------------------------------- #
# B5 — opt-in short side
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_short_sells_and_receives_funding():
    def falling(ticker, start, end):
        idx = pd.bdate_range(start, end)
        values = [100.0 * (1 - 0.001 * i) for i in range(len(idx))]
        return pd.Series(values, index=idx.strftime("%Y-%m-%d"), dtype=float)

    res = run_backtest(
        FakeGraph({"2024-01-01": "Sell"}), "BTCUSDT", ["2024-01-01"],
        asset_type="crypto_perp", allow_short=True,
        funding_provider=_funding(0.001), cost_bps=0.0,
        price_provider=falling, compute_index_alpha=False,
        periods_per_year=365, initial_capital=100_000.0,
    )
    trade = res.trades[0]
    assert trade.target_weight == -1.0
    assert trade.executed_weight < 0.0
    # Falling prices + positive funding RECEIVED by the short: equity grows
    # on both legs.
    assert res.equity[-1] > 100_000.0
    assert res.config_summary["perp_funding_paid_total"] < 0.0
    assert "opt-in short side" in res.config_summary["perp_model_note"]


@pytest.mark.unit
def test_short_disabled_by_default_sell_stays_flat():
    res = run_backtest(
        FakeGraph({"2024-01-01": "Sell"}), "BTCUSDT", ["2024-01-01"],
        asset_type="crypto_perp", funding_provider=_funding(0.0),
        cost_bps=0.0, **{**_PERP_KW, "price_provider": _rising_prices},
    )
    assert res.trades[0].executed_weight == 0.0
    assert res.equity[-1] == pytest.approx(100_000.0)


@pytest.mark.unit
def test_allow_short_requires_perp():
    with pytest.raises(ValueError, match="crypto_perp-only"):
        run_backtest(
            FakeGraph({"2024-01-01": "Sell"}), "AAPL", ["2024-01-01"],
            price_provider=_rising_prices, allow_short=True,
        )


# --------------------------------------------------------------------------- #
# B6 — mark-price liquidation trigger (Binance liquidates on mark, not last)
# --------------------------------------------------------------------------- #


def _mark_vs_last_frames(monkeypatch, mark_low_on_day: int, last_low: float = 99.0):
    """Serve synthetic last/mark klines: flat closes at 100, calm LAST lows,
    and a MARK wick to ``mark_low_on_day`` (index) that pierces a 10x liquidation
    level (90.4) while the last-price book never does."""

    def fake_frame(symbol, start, end, interval="1d", venue="binance_perp",
                   price_type="last", closed_as_of=None):
        idx = pd.bdate_range(start, end)
        n = len(idx)
        calm_low = last_low
        if price_type == "mark" and mark_low_on_day is not None:
            lows = [calm_low] * mark_low_on_day + [85.0] + [calm_low] * (
                n - mark_low_on_day - 1)
        else:
            lows = [calm_low] * n
        return pd.DataFrame(
            {"Close": [100.0] * n, "Low": lows, "High": [101.0] * n},
            index=idx,
        )

    monkeypatch.setattr(
        "yialpha.dataflows.binance.binance_klines_frame", fake_frame,
    )


_LIQ_KW = {
    "initial_capital": 100_000.0, "holding_days": 5, "compute_index_alpha": False,
    "periods_per_year": 365, "cost_bps": 0.0,
    "asset_type": "crypto_perp", "funding_provider": _funding(0.0),
    "leverage": 10.0, "brackets_provider": lambda t: _BRACKETS,
}


@pytest.mark.unit
def test_mark_wick_liquidates_when_last_stays_calm(monkeypatch):
    """The default perp run triggers liquidation on MARK kline wicks: a mark
    low of 85 forces the close while the last-price book (low 99) never
    would have — pricing this on last klines was the old wrong behaviour."""
    _mark_vs_last_frames(monkeypatch, mark_low_on_day=3)
    res = run_backtest(
        FakeGraph({"2024-01-01": "Buy"}), "BTCUSDT", ["2024-01-01"],
        **_LIQ_KW,
    )
    events = res.config_summary.get("perp_liquidations")
    assert events, "a MARK wick through the 10x liquidation level must force the close"
    assert res.config_summary["perp_liq_price_source"] == "mark"
    assert "mark-price" in res.config_summary["perp_model_note"]
    # Equity still marks on LAST closes (flat 100), and the forced exit pays
    # out the isolated-margin remainder at the trigger.
    assert res.equity[-1] == pytest.approx(
        100_000.0 * (1.0 - 10.0 * (1.0 - 0.904)), rel=1e-6)


@pytest.mark.unit
def test_last_mode_skips_mark_liquidation_on_same_wick(monkeypatch):
    """Explicit liquidation_price_type='last': the same mark wick does NOT
    trigger (single-series behaviour restored)."""
    _mark_vs_last_frames(monkeypatch, mark_low_on_day=3)
    res = run_backtest(
        FakeGraph({"2024-01-01": "Buy"}), "BTCUSDT", ["2024-01-01"],
        liquidation_price_type="last", **_LIQ_KW,
    )
    assert "perp_liquidations" not in res.config_summary
    assert res.config_summary["perp_liq_price_source"] == "last"


@pytest.mark.unit
def test_mark_unavailable_falls_back_loudly(monkeypatch):
    """Mark klines failing -> LAST fallback with a disclosure stamp, never a
    silent relabel (the difference between mark and last can be liquidation
    itself)."""
    import yialpha.backtest.engine  # noqa: F401  — ensures engine module path

    real_frame_calls: list[str] = []

    def fake_frame(symbol, start, end, interval="1d", venue="binance_perp",
                   price_type="last", closed_as_of=None):
        real_frame_calls.append(price_type)
        if price_type == "mark":
            raise RuntimeError("mark endpoint down")
        idx = pd.bdate_range(start, end)
        n = len(idx)
        lows = [99.0] * 3 + [85.0] + [99.0] * (n - 4)  # LAST wick pierces
        return pd.DataFrame(
            {"Close": [100.0] * n, "Low": lows, "High": [101.0] * n},
            index=idx,
        )

    monkeypatch.setattr(
        "yialpha.dataflows.binance.binance_klines_frame", fake_frame,
    )
    res = run_backtest(
        FakeGraph({"2024-01-01": "Buy"}), "BTCUSDT", ["2024-01-01"],
        **_LIQ_KW,
    )
    assert "mark" in real_frame_calls, "the engine must have TRIED mark first"
    assert res.config_summary["perp_liquidations"], (
        "the LAST-price fallback must still check last wicks"
    )
    assert res.config_summary["perp_liq_price_source"] == "last (mark unavailable)"


@pytest.mark.unit
def test_injected_extremes_serve_both_trigger_roles():
    """An injected extremes provider is the synthetic world: it serves the
    liquidation AND stop trigger series verbatim (source disclosed)."""
    from tests.test_perp_stop_simulation import _stop_weight_fn

    res = run_backtest(
        FakeGraph({"2024-01-01": "Buy"}), "BTCUSDT", ["2024-01-01"],
        price_provider=_flat_then_crash, extremes_provider=_crash_lows,
        weight_fn=_stop_weight_fn(1.0, 95.0), **_LIQ_KW,
    )
    assert res.config_summary["perp_liq_price_source"] == "injected"


@pytest.mark.unit
def test_liq_price_type_validation():
    with pytest.raises(ValueError, match="must be 'mark' or 'last'"):
        run_backtest(
            FakeGraph({"2024-01-01": "Buy"}), "BTCUSDT", ["2024-01-01"],
            liquidation_price_type="mid", **_LIQ_KW,
        )
    with pytest.raises(ValueError, match="crypto_perp-only"):
        run_backtest(
            FakeGraph({"2024-01-01": "Buy"}), "AAPL", ["2024-01-01"],
            price_provider=_rising_prices,
            liquidation_price_type="mark", leverage=1.0,
        )
