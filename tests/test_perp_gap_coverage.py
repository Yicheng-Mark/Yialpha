"""Perp gap coverage: engine branches and data helpers with zero prior tests.

2026-08-17 round — every path the exploratory audit found unexercised:

* Short + leverage > 1 upside liquidation (the mirrored trigger branch had
  never run in any test; the short tests all ran at default 1x).
* Negative funding: longs RECEIVE when the settlement sum is negative.
* MMR ladder tiering + signed-bracket fetch/cache (binance_brackets had no
  direct unit tests at all).
* pos_avg_entry blending: extend blends, reduce keeps, flip resets — observed
  through the liquidation event's ``entry`` field.
* ``_binance_funding_provider`` day-bucketing of 8h settlements.
* The REAL risk overlay (build_backtest_weight_fn + RiskManager) wired into a
  perp run: stamped stops now actually fire (and the caveat text flips from
  "advisory metadata only" to "GTC stop-market" via stop_simulation_mode).

All hermetic: providers injected / vendor functions monkeypatched, no network.
"""

from __future__ import annotations

import pandas as pd
import pytest

from tests.test_backtest_engine import FakeGraph
from yiagents.backtest.engine import run_backtest
from yiagents.dataflows import binance_brackets
from yiagents.dataflows.binance_brackets import (
    Bracket,
    default_brackets,
    get_leverage_brackets,
    mmr_for_notional,
)
from yiagents.dataflows.errors import NoMarketDataError


def _funding(rate: float):
    def provider(ticker, start, end):
        idx = pd.bdate_range(start, end)
        return pd.Series([rate] * len(idx), index=idx.strftime("%Y-%m-%d"),
                         dtype=float)
    return provider


def _series(values_by_offset: dict[int, float], level: float = 100.0):
    """Price series: flat ``level`` except scripted offsets."""

    def provider(ticker, start, end):
        idx = pd.bdate_range(start, end)
        vals = [level] * len(idx)
        for off, v in values_by_offset.items():
            if off < len(vals):
                vals[off] = v
        return pd.Series(vals, index=idx.strftime("%Y-%m-%d"), dtype=float)

    return provider


def _wick_extremes(low_offset: int, low: float, high: float = 100.5):
    def provider(ticker, start, end):
        idx = pd.bdate_range(start, end)
        n = len(idx)
        lows = [99.5] * n
        highs = [100.5] * n
        if low_offset < n:
            lows[low_offset] = low
            highs[low_offset] = high
        return (
            pd.Series(lows, index=idx.strftime("%Y-%m-%d"), dtype=float),
            pd.Series(highs, index=idx.strftime("%Y-%m-%d"), dtype=float),
        )

    return provider


_PERP = {
    "asset_type": "crypto_perp", "initial_capital": 100_000.0, "holding_days": 5,
    "compute_index_alpha": False, "periods_per_year": 365, "cost_bps": 0.0,
    "funding_provider": _funding(0.0),
}
_BRACKETS = [(0.0, 0.004)]


# --------------------------------------------------------------------------- #
# Short + leverage: the mirrored (upside) liquidation branch
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_short_leverage_10x_upside_liquidation():
    """Short @100 at 10x: liq at entry*(1 + 1/10 - MMR - fee) = 109.6 with the
    flat ladder. A high wick to 120 forces the close even though the close
    (150) gapped far through — exit AT the level, loss capped at margin."""

    def rally(ticker, start, end):
        idx = pd.bdate_range(start, end)
        vals = [100.0] * 3 + [150.0] * (len(idx) - 3)
        return pd.Series(vals, index=idx.strftime("%Y-%m-%d"), dtype=float)

    def wick_highs(ticker, start, end):
        idx = pd.bdate_range(start, end)
        n = len(idx)
        highs = [100.5] * 3 + [120.0] + [150.5] * (n - 4)
        lows = [99.5] * n
        return (
            pd.Series(lows, index=idx.strftime("%Y-%m-%d"), dtype=float),
            pd.Series(highs, index=idx.strftime("%Y-%m-%d"), dtype=float),
        )

    res = run_backtest(
        FakeGraph({"2024-01-01": "Sell"}), "BTCUSDT", ["2024-01-01"],
        allow_short=True, leverage=10.0, brackets_provider=lambda t: _BRACKETS,
        price_provider=rally, extremes_provider=wick_highs, **_PERP,
    )
    events = res.config_summary.get("perp_liquidations")
    assert events, "a 10x short through a +20% upside wick must be liquidated"
    ev = events[0]
    assert ev["side"] == "short"
    assert ev["liquidation_price"] == pytest.approx(100.0 * 1.096, rel=1e-6)
    assert ev["exit_price"] == pytest.approx(100.0 * 1.096, rel=1e-6)
    # Margin remainder: -10 x (1.096 - 1) = -96% of posted capital.
    assert res.equity[-1] == pytest.approx(
        100_000.0 * (1.0 - 10.0 * 0.096), rel=1e-6)


# --------------------------------------------------------------------------- #
# Negative funding: longs receive
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_negative_funding_pays_the_long():
    base = run_backtest(
        FakeGraph({"2024-01-01": "Buy"}), "BTCUSDT", ["2024-01-01"],
        price_provider=_series({}), **_PERP,
    )
    neg = run_backtest(
        FakeGraph({"2024-01-01": "Buy"}), "BTCUSDT", ["2024-01-01"],
        price_provider=_series({}), funding_provider=_funding(-0.001),
        **{k: v for k, v in _PERP.items() if k != "funding_provider"},
    )
    assert neg.config_summary["perp_funding_paid_total"] < 0.0
    assert neg.equity[-1] > base.equity[-1]


# --------------------------------------------------------------------------- #
# MMR ladder tiering + signed bracket fetch
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_mmr_for_notional_tier_boundaries():
    ladder = default_brackets()
    cases = {
        0.0: 0.004,
        49_999.99: 0.004,
        50_000.0: 0.005,        # at the floor -> the higher tier applies
        150_000.0: 0.006,
        4_999_999.99: 0.0125,
        5_000_000.0: 0.05,
        99_999_999.0: 0.125,
        100_000_000.0: 0.5,     # above the last floor -> last tier
        1e12: 0.5,
    }
    for notional, expected in cases.items():
        assert mmr_for_notional(ladder, notional) == pytest.approx(expected), (
            f"notional {notional}"
        )


@pytest.mark.unit
def test_mmr_for_notional_accepts_tuples_and_empty():
    assert mmr_for_notional([(0.0, 0.004), (100.0, 0.01)], 150.0) == 0.01
    assert mmr_for_notional([(0.0, 0.004), (100.0, 0.01)], 50.0) == 0.004
    # Below every floor (degenerate ladder): the loop never assigns -> 0.0,
    # which would make the liquidation model optimistically wrong; pin the
    # actual behaviour so any future change is deliberate.
    assert mmr_for_notional([(1_000.0, 0.004)], 500.0) == 0.0
    assert mmr_for_notional([], 1.0) == 0.0
    # Bracket dataclass form is accepted alongside tuples.
    assert mmr_for_notional([Bracket(0.0, 0.004)], 10.0) == 0.004


@pytest.mark.unit
def test_get_leverage_brackets_needs_keys(monkeypatch):
    monkeypatch.delenv("BINANCE_API_KEY", raising=False)
    monkeypatch.delenv("BINANCE_API_SECRET", raising=False)
    binance_brackets.reset_for_test()
    with pytest.raises(NoMarketDataError, match="signed endpoint"):
        get_leverage_brackets("BTCUSDT")


@pytest.mark.unit
def test_get_leverage_brackets_fetch_parse_and_cache(monkeypatch):
    monkeypatch.setenv("BINANCE_API_KEY", "k")
    monkeypatch.setenv("BINANCE_API_SECRET", "s")
    binance_brackets.reset_for_test()
    calls = {"n": 0}

    def fake_http(path, params, symbol, canonical, headers=None, **kw):
        calls["n"] += 1
        assert path == "/fapi/v1/leverageBracket"
        assert "signature" in params and params["signature"]
        assert headers == {"X-MBX-APIKEY": "k"}
        # Deliberately unsorted + one garbage bracket to pin parsing.
        return [{
            "symbol": canonical,
            "brackets": [
                {"notionalFloor": "250000", "maintMarginRatio": "0.007"},
                {"notionalFloor": "0", "maintMarginRatio": "0.004"},
                {"garbage": True},
            ],
        }]

    monkeypatch.setattr(binance_brackets, "_http_get", fake_http)
    first = get_leverage_brackets("BTCUSDT")
    second = get_leverage_brackets("BTCUSDT")  # TTL cache hit
    assert calls["n"] == 1
    assert second == first
    assert [b.notional_floor for b in first] == [0.0, 250_000.0]
    assert [b.mmr for b in first] == [0.004, 0.007]


@pytest.mark.unit
def test_default_brackets_is_a_fresh_mutable_copy():
    a = default_brackets()
    a.pop()
    b = default_brackets()
    assert len(b) == len(default_brackets())
    assert a != b


# --------------------------------------------------------------------------- #
# pos_avg_entry blending: extend blends / reduce keeps / flip resets
# --------------------------------------------------------------------------- #


def _run_with_weights(weights: list[float], prices: list[float],
                      wick_at: int, wick_low: float = 99.5,
                      wick_high: float = 100.5):
    """Buy-signals with scripted target weights at 10x; a crash/squeeze wick
    liquidates at the end and exposes pos_avg_entry via the event's ``entry``.

    Weights are pre-leverage target weights (w * 10 = effective notional
    fraction), so w=0.05 behaves like the historical 0.5x sizing."""

    def price_provider(ticker, start, end):
        idx = pd.bdate_range(start, end)
        vals = list(prices) + [prices[-1]] * (len(idx) - len(prices))
        return pd.Series(vals[:len(idx)], index=idx.strftime("%Y-%m-%d"),
                         dtype=float)

    def extremes(ticker, start, end):
        idx = pd.bdate_range(start, end)
        n = len(idx)
        lows = [99.5] * n
        highs = [100.5] * n
        lows[wick_at] = wick_low
        highs[wick_at] = wick_high
        return (
            pd.Series(lows, index=idx.strftime("%Y-%m-%d"), dtype=float),
            pd.Series(highs, index=idx.strftime("%Y-%m-%d"), dtype=float),
        )

    dates = [f"2024-01-{i:02d}" for i in range(1, len(weights) + 1)]
    it = iter(weights)

    def wfn(rating, date, ctx):
        ctx["risk_decision"] = None
        return next(it)

    res = run_backtest(
        FakeGraph(dict.fromkeys(dates, "Buy")), "BTCUSDT", dates,
        allow_short=True, leverage=10.0, brackets_provider=lambda t: _BRACKETS,
        price_provider=price_provider, extremes_provider=extremes,
        weight_fn=wfn, **_PERP,
    )
    return res


@pytest.mark.unit
def test_extend_blends_entry_basis():
    # w=0.05 fills @100 (500 sh); w=0.10 fills @120 (equity 110k, +416.67 sh).
    # Blended entry = (100*500 + 120*416.667)/916.667 = 109.0909.
    res = _run_with_weights(
        [0.05, 0.10, 0.10], [100.0, 100.0, 120.0, 120.0, 120.0],
        wick_at=4, wick_low=20.0,
    )
    ev = res.config_summary["perp_liquidations"][0]
    assert ev["entry"] == pytest.approx((100.0 * 500 + 120.0 * (1250 / 3))
                                        / (500 + 1250 / 3), rel=1e-6)


@pytest.mark.unit
def test_reduce_keeps_entry_basis():
    # Same build-up, then w=0.025 trims the position: entry basis unchanged.
    res = _run_with_weights(
        [0.05, 0.10, 0.025, 0.025], [100.0, 100.0, 120.0, 120.0, 120.0],
        wick_at=4, wick_low=20.0,
    )
    ev = res.config_summary["perp_liquidations"][0]
    assert ev["entry"] == pytest.approx((100.0 * 500 + 120.0 * (1250 / 3))
                                        / (500 + 1250 / 3), rel=1e-6)


@pytest.mark.unit
def test_flip_resets_entry_basis():
    # Long w=0.05 @100, then a flip to w=-0.10 @120: entry restarts at the
    # flip fill price; the upside wick (140 >= 120*1.096) liquidates the
    # fresh SHORT — entered at 120, not at the stale long basis ~100.
    res = _run_with_weights(
        [0.05, -0.10, -0.10], [100.0, 100.0, 120.0, 120.0, 120.0],
        wick_at=4, wick_high=140.0,
    )
    ev = res.config_summary["perp_liquidations"][0]
    assert ev["side"] == "short"
    assert ev["entry"] == pytest.approx(120.0)


# --------------------------------------------------------------------------- #
# _binance_funding_provider: 8h settlements bucketed per UTC day
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_funding_provider_sums_each_utc_day(monkeypatch):
    from yiagents.backtest.engine import _binance_funding_provider

    def ms(day: str, hour: int) -> int:
        return int(pd.Timestamp(f"{day} {hour:02d}:00", tz="UTC").timestamp()
                   * 1000)

    rows = [
        {"fundingTime": ms("2024-01-01", 0), "fundingRate": "0.0001"},
        {"fundingTime": ms("2024-01-01", 8), "fundingRate": "0.0001"},
        {"fundingTime": ms("2024-01-01", 16), "fundingRate": "0.0001"},
        {"fundingTime": ms("2024-01-02", 0), "fundingRate": "-0.0002"},
        {"fundingTime": ms("2024-01-02", 8), "fundingRate": "0.0001"},
        {"fundingTime": ms("2024-01-03", 0), "fundingRate": "garbage"},  # skipped
        {"fundingTime": None, "fundingRate": "0.5"},                    # skipped
        "not-a-dict",                                                    # skipped
    ]

    def fake_paginate(path, params, limit, key, start_ms, end_ms, *a, **k):
        assert path == "/fapi/v1/fundingRate"
        return rows

    monkeypatch.setattr(
        "yiagents.dataflows.binance._paginate_history", fake_paginate,
    )
    s = _binance_funding_provider("BTCUSDT", "2024-01-01", "2024-01-05")
    assert s["2024-01-01"] == pytest.approx(0.0003)
    assert s["2024-01-02"] == pytest.approx(-0.0001)
    assert "2024-01-03" not in s.index


# --------------------------------------------------------------------------- #
# Real risk overlay x perp: stamped stops now actually fire
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_risk_overlay_stop_fires_in_perp_engine():
    from yiagents.risk.manager import RiskManager, build_backtest_weight_fn

    wfn = build_backtest_weight_fn(
        RiskManager(atr_mult=2.0), "BTCUSDT", atr_lookup=lambda date: 2.0,
    )
    res = run_backtest(
        FakeGraph({"2024-01-01": "Buy"}), "BTCUSDT", ["2024-01-01"],
        weight_fn=wfn, price_provider=_series({}),
        extremes_provider=_wick_extremes(2, 94.0),
        **_PERP,
    )
    trade = res.trades[0]
    assert trade.stop_loss is not None
    assert "GTC stop-market" in (trade.risk_warning or "")
    events = res.config_summary.get("perp_stop_triggers")
    assert events, "the overlay's stamped stop must actually fire on the wick"
    assert events[0]["stop_price"] == pytest.approx(trade.stop_loss)

    # Stock twin (stop simulation off): same overlay, the warning keeps the
    # honest "advisory metadata only" caveat and nothing fires.
    dates = ["2024-01-01"]
    res2 = run_backtest(
        FakeGraph({"2024-01-01": "Buy"}), "AAPL", dates,
        weight_fn=wfn, price_provider=_series({}),
        initial_capital=100_000.0, holding_days=5, compute_index_alpha=False,
        cost_bps=0.0,
    )
    assert res2.trades[0].stop_loss is not None
    assert "advisory metadata only" in (res2.trades[0].risk_warning or "")
    assert "perp_stop_triggers" not in res2.config_summary


# --------------------------------------------------------------------------- #
# Perp facts as first-class metrics + report rendering
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_perp_facts_in_metrics_and_report():
    from tests.test_perp_stop_simulation import _extremes, _stop_weight_fn
    from yiagents.backtest.report import render_backtest_report

    res = run_backtest(
        FakeGraph({"2024-01-01": "Buy"}), "BTCUSDT", ["2024-01-01"],
        weight_fn=_stop_weight_fn(1.0, 97.0),
        price_provider=_series({}),
        extremes_provider=_extremes([94.0], [100.5], 2),
        leverage=10.0, brackets_provider=lambda t: _BRACKETS,
        funding_provider=_funding(0.001),
        **{k: v for k, v in _PERP.items() if k != "funding_provider"},
    )
    m = res.metrics
    assert m is not None
    # Stop fires (wick 94 < 97), liquidation never (94 > 90.4 trigger).
    assert m.stop_trigger_count == 1
    assert m.liquidation_count == 0
    assert m.funding_paid_total is not None and m.funding_paid_total > 0.0
    assert m.funding_drag_annualized is not None and m.funding_drag_annualized > 0.0

    text = render_backtest_report(res)
    assert "Stop-trigger exits: 1" in text
    assert "Liquidation events: 0" in text
    assert "Liquidation trigger source: injected" in text
    assert "net paid (drag)" in text
    assert "/yr of initial capital" in text

    # Non-perp runs keep the new fields at None (byte-stable contract).
    stock = run_backtest(
        FakeGraph({"2024-01-01": "Buy"}), "AAPL", ["2024-01-01"],
        price_provider=_series({}),
        initial_capital=100_000.0, holding_days=5, compute_index_alpha=False,
        cost_bps=0.0,
    )
    assert stock.metrics is not None
    assert stock.metrics.liquidation_count is None
    assert stock.metrics.stop_trigger_count is None
    assert stock.metrics.funding_paid_total is None
    assert stock.metrics.funding_drag_annualized is None
