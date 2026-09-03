"""V2.4 batch M — the backtest engine keyed on instrument_class, not the string.

Pins the perp-class split:

* Classification seam: the engine delegates to the ONE shared routing
  classifier (memoized per instrument) and discloses the resolved class in
  ``config_summary``; the perp-family gates (leverage / allow_short /
  mark-price liquidation) accept a stock perp while equity keeps the
  historical rejection.
* Annualization: ``periods_per_year`` defaults resolve through
  ``periods_per_year_for`` — stock_perp 261 / pure_crypto_perp 365 /
  equity 252 / crypto_spot 365 (unchanged) / unknown_perp 252; an explicit
  override still wins.
* Next-tradable-bar fills on BOTH series shapes: a 24/7 calendar-daily
  series and a stock-perp-like weekday series with session gaps (the fill
  lands on the next PRESENT bar, never an absent calendar day).
* Corporate actions (honest skeleton): caller-supplied records back-adjust
  the close series before returns; ``None`` stays byte-identical.
* MMR/stop coverage NOT already pinned elsewhere: stops firing on
  LAST-price wicks in the DEFAULT wiring while mark stays calm, and the
  short-side same-bar conservative resolution (liquidation first).

Already pinned in other suites (referenced, not duplicated): mark-wick
liquidation on the MMR ladder (test_perp_engine::
test_mark_wick_liquidates_when_last_stays_calm / test_leverage_10x_crash_
forces_liquidation; ladder math test_perp_gap_coverage::
test_mmr_for_notional_tier_boundaries), the long same-bar liq-wins order
(test_perp_stop_simulation::test_liq_wins_when_bar_pierces_both_levels),
the short upside liquidation mirror (test_perp_gap_coverage::
test_short_leverage_10x_upside_liquidation), and the contiguous next-bar
fill (test_backtest_engine::test_signal_executes_on_next_bar_not_same_close).

All hermetic: providers injected, no network, no LLM.
"""

from __future__ import annotations

import pandas as pd
import pytest

from tests.test_backtest_engine import FakeGraph, _rising_prices
from tests.test_perp_stop_simulation import (
    _extremes,
    _flat,
    _stop_weight_fn,
)
from yialpha.backtest.engine import run_backtest
from yialpha.instruments.corporate_actions import CorporateActionRecord


def _funding(rate: float):
    def provider(ticker, start, end):
        idx = pd.bdate_range(start, end)
        return pd.Series(
            [rate] * len(idx), index=idx.strftime("%Y-%m-%d"), dtype=float,
        )
    return provider


@pytest.fixture(autouse=True)
def _deterministic_classification():
    """Reset the seam memo and the warm listing cache around each test.

    The engine memoizes routing classifications per ``(asset_type, symbol)``
    for the process lifetime; without the reset, a class frozen under
    another test's flag/warm state (or under this file's own monkeypatching)
    would leak into later assertions. Seed mode + registry flag off (conftest
    autouse) make routing fully deterministic: MU is in the static equity
    seed, BTC/ZZZ are not.
    """
    from yialpha.backtest import engine as engine_mod
    from yialpha.dataflows.binance import refresh_equity_perp_bases

    engine_mod._INSTRUMENT_CLASS_CACHE.clear()
    refresh_equity_perp_bases()
    yield
    engine_mod._INSTRUMENT_CLASS_CACHE.clear()
    refresh_equity_perp_bases()


_PERP_KW = {
    "asset_type": "crypto_perp", "initial_capital": 100_000.0,
    "holding_days": 5, "compute_index_alpha": False, "cost_bps": 0.0,
    "funding_provider": _funding(0.0), "price_provider": _rising_prices,
}


# --------------------------------------------------------------------------- #
# Classification seam
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_seam_delegates_to_routing():
    from yialpha.backtest.engine import _perp_instrument_class

    # The routing truth table (seed mode, registry flag off): MU is in the
    # static equity-perp seed, BTC is not.
    assert _perp_instrument_class("crypto_perp", "BTCUSDT") == "pure_crypto_perp"
    assert _perp_instrument_class("crypto_perp", "MUUSDT") == "stock_perp"
    assert _perp_instrument_class("stock", "AAPL") == "equity"
    assert _perp_instrument_class("crypto", "BTC-USD") == "crypto_spot"


@pytest.mark.unit
def test_seam_memoizes_per_instrument(monkeypatch):
    import yialpha.graph.routing as routing
    from yialpha.backtest import engine as engine_mod

    first = engine_mod._perp_instrument_class("crypto_perp", "BTCUSDT")
    assert first == "pure_crypto_perp"

    def explode(*args, **kwargs):
        raise AssertionError("a memoized classification must not re-call routing")

    monkeypatch.setattr(routing, "instrument_class", explode)
    assert (
        engine_mod._perp_instrument_class("crypto_perp", "BTCUSDT")
        == "pure_crypto_perp"
    )


@pytest.mark.unit
def test_pure_crypto_perp_run_class_and_annualization():
    res = run_backtest(
        FakeGraph({"2024-01-01": "Buy"}), "BTCUSDT", ["2024-01-01"],
        **_PERP_KW,
    )
    assert res.config_summary["instrument_class"] == "pure_crypto_perp"
    assert res.config_summary["periods_per_year"] == 365


@pytest.mark.unit
def test_stock_perp_run_class_and_annualization_261():
    """A tokenized-stock perp enters through asset_type="crypto_perp" — the
    SEAM (not the string) resolves stock_perp and annualizes its
    weekday-session candles at 261 instead of the 24/7 factor 365."""
    res = run_backtest(
        FakeGraph({"2024-01-01": "Buy"}), "MUUSDT", ["2024-01-01"],
        **_PERP_KW,
    )
    assert res.config_summary["instrument_class"] == "stock_perp"
    assert res.config_summary["periods_per_year"] == 261
    assert res.metrics is not None  # 261 flowed through the metric suite


@pytest.mark.unit
def test_equity_run_class_and_annualization_252():
    res = run_backtest(
        FakeGraph({"2024-01-01": "Buy"}), "AAPL", ["2024-01-01"],
        initial_capital=100_000.0, holding_days=5, cost_bps=0.0,
        price_provider=_rising_prices, compute_index_alpha=False,
    )
    assert res.config_summary["instrument_class"] == "equity"
    assert res.config_summary["periods_per_year"] == 252


@pytest.mark.unit
def test_crypto_spot_annualization_stays_365():
    """Regression guard for the seam's class gating: spot crypto must keep
    the 365 asset-type rule. Blindly threading the resolved class
    ("crypto_spot" -> the 252 equity convention in periods_per_year_for)
    would silently change every spot-crypto backtest."""
    res = run_backtest(
        FakeGraph({"2024-01-01": "Buy"}), "BTC-USD", ["2024-01-01"],
        asset_type="crypto", initial_capital=100_000.0, holding_days=5,
        cost_bps=0.0, price_provider=_rising_prices, compute_index_alpha=False,
    )
    assert res.config_summary["instrument_class"] == "crypto_spot"
    assert res.config_summary["periods_per_year"] == 365


@pytest.mark.unit
def test_unknown_perp_annualizes_at_252():
    """Registry flag on + empty registry + a base outside the seed: the
    honest unknown_perp verdict, which the V2.2 convention annualizes at 252
    (NOT the 365 the raw crypto_perp string used to imply)."""
    from yialpha.dataflows.config import set_config

    set_config({"instrument_registry": True})
    res = run_backtest(
        FakeGraph({"2024-01-01": "Buy"}), "ZZZUSDT", ["2024-01-01"],
        **_PERP_KW,
    )
    assert res.config_summary["instrument_class"] == "unknown_perp"
    assert res.config_summary["periods_per_year"] == 252


@pytest.mark.unit
def test_explicit_periods_per_year_override_wins():
    res = run_backtest(
        FakeGraph({"2024-01-01": "Buy"}), "MUUSDT", ["2024-01-01"],
        periods_per_year=300, **_PERP_KW,
    )
    assert res.config_summary["periods_per_year"] == 300


# --------------------------------------------------------------------------- #
# Perp-family gates
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_stock_perp_accepted_by_perp_family_gates():
    """leverage / allow_short / liquidation_price_type="mark" validation keys
    on the perp FAMILY: a stock perp routed through the crypto_perp entrance
    gets perp semantics instead of the spot/stock rejection."""
    res = run_backtest(
        FakeGraph({"2024-01-01": "Buy"}), "MUUSDT", ["2024-01-01"],
        leverage=2.0, allow_short=True, liquidation_price_type="mark",
        **_PERP_KW,
    )
    # The levered target weight's notional: 1.0 clipped target x 2x leverage.
    assert res.trades[0].executed_weight == pytest.approx(2.0)


@pytest.mark.unit
def test_equity_keeps_perp_param_rejection():
    with pytest.raises(ValueError, match="crypto_perp-only"):
        run_backtest(
            FakeGraph({"2024-01-01": "Buy"}), "AAPL", ["2024-01-01"],
            initial_capital=100_000.0, holding_days=5, cost_bps=0.0,
            price_provider=_rising_prices, compute_index_alpha=False,
            leverage=2.0,
        )
    with pytest.raises(ValueError, match="crypto_perp-only"):
        run_backtest(
            FakeGraph({"2024-01-01": "Sell"}), "AAPL", ["2024-01-01"],
            initial_capital=100_000.0, holding_days=5, cost_bps=0.0,
            price_provider=_rising_prices, compute_index_alpha=False,
            allow_short=True,
        )


# --------------------------------------------------------------------------- #
# Next-tradable-bar fills on both series shapes
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_fill_lands_on_next_present_bar_24_7_series():
    """A 24/7 calendar-daily series: Saturday bars EXIST, so a Friday signal
    fills the very next present bar — Saturday's close."""
    idx = pd.date_range("2024-01-01", "2024-01-31", freq="D")

    def calendar_prices(ticker, start, end):
        return pd.Series(
            [100.0 + i for i in range(len(idx))],
            index=idx.strftime("%Y-%m-%d"), dtype=float,
        )

    res = run_backtest(
        FakeGraph({"2024-01-05": "Buy"}), "BTCUSDT", ["2024-01-05"],
        initial_capital=100_000.0, holding_days=5, cost_bps=0.0,
        price_provider=calendar_prices, compute_index_alpha=False,
    )
    trade = res.trades[0]
    assert trade.date == "2024-01-05"        # a Friday
    assert trade.execution_date == "2024-01-06"  # Saturday: present in 24/7
    assert trade.price == pytest.approx(105.0)    # idx position 5 -> 100+5


@pytest.mark.unit
def test_fill_lands_on_next_present_bar_weekday_gap_series():
    """A stock-perp-like weekday-only series: the Fri->Mon gap has NO bars,
    so the fill must land on Monday's PRESENT bar (2024-01-08), never on the
    absent calendar Saturday (which a naive +1-day fill would pick)."""
    idx = pd.bdate_range("2024-01-01", "2024-01-31")

    def weekday_prices(ticker, start, end):
        return pd.Series(
            [100.0 + 10 * i for i in range(len(idx))],
            index=idx.strftime("%Y-%m-%d"), dtype=float,
        )

    res = run_backtest(
        FakeGraph({"2024-01-05": "Buy"}), "MUUSDT", ["2024-01-05"],
        initial_capital=100_000.0, holding_days=5, cost_bps=0.0,
        price_provider=weekday_prices, compute_index_alpha=False,
    )
    trade = res.trades[0]
    assert trade.date == "2024-01-05"        # Friday signal
    assert trade.execution_date == "2024-01-08"  # next PRESENT bar = Monday
    assert "2024-01-06" not in res.equity_dates
    assert trade.price == pytest.approx(150.0)    # idx position 5 -> 100+50


# --------------------------------------------------------------------------- #
# Corporate actions (honest skeleton wiring)
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_apply_corporate_actions_direct_chaining_and_symbols():
    """Helper-level pin: split (to/from spelling) halves prior bars, a later
    dividend subtracts from the already-adjusted series, the event_date bar
    is the ex bar, and records for a different symbol are ignored."""
    from yialpha.backtest.engine import _apply_corporate_actions

    idx = pd.bdate_range("2024-01-01", "2024-01-12")
    raw = pd.Series(
        [100.0] * len(idx), index=idx.strftime("%Y-%m-%d"), dtype=float,
    )
    out, applied = _apply_corporate_actions(raw, "MUUSDT", [
        # Different instrument: ignored entirely.
        CorporateActionRecord(
            symbol="TSLA", action_type="split", event_date="2024-01-02",
            details={"ratio": 2.0},
        ),
        # 2:1 split via the to/from spelling, effective Friday 01-05.
        CorporateActionRecord(
            symbol="MU", action_type="split", event_date="2024-01-05",
            details={"to": 2, "from": 1},
        ),
        # $2/share dividend effective Monday 01-08 (underlying symbol form).
        CorporateActionRecord(
            symbol="MUUSDT", action_type="dividend", event_date="2024-01-08",
            details={"amount": 2.0, "currency": "USD"},
        ),
    ])
    assert applied == 2
    # Bars before the split ex-date: halved, then the dividend subtracted
    # (chained on the running adjusted series) -> 100/2 - 2 = 48.
    assert out["2024-01-02"] == pytest.approx(48.0)
    assert out["2024-01-04"] == pytest.approx(48.0)
    # The split ex bar itself stays raw (100) but sits before the dividend
    # ex-date, so only the dividend touches it: 100 - 2 = 98.
    assert out["2024-01-05"] == pytest.approx(98.0)
    # Ex both actions: the raw bar, untouched.
    assert out["2024-01-08"] == pytest.approx(100.0)
    assert out["2024-01-11"] == pytest.approx(100.0)
    # The input series is never mutated.
    assert raw["2024-01-02"] == pytest.approx(100.0)


@pytest.mark.unit
def test_split_adjustment_removes_mechanical_jump():
    """A 2:1 underlying split halves the perp's price overnight — mechanics,
    not P&L. With the record supplied, the pre-split bars halve and the
    backtest sees a flat series; the unadjusted twin eats a fake -50% crash."""

    def split_prices(ticker, start, end):
        idx = pd.bdate_range(start, end)
        values = [100.0] * 4 + [50.0] * (len(idx) - 4)
        return pd.Series(values, index=idx.strftime("%Y-%m-%d"), dtype=float)

    adjusted = run_backtest(
        FakeGraph({"2024-01-01": "Buy"}), "MUUSDT", ["2024-01-01"],
        initial_capital=100_000.0, holding_days=5, cost_bps=0.0,
        price_provider=split_prices, compute_index_alpha=False,
        corporate_actions=[
            CorporateActionRecord(
                symbol="MUUSDT", action_type="split", event_date="2024-01-05",
                details={"ratio": 2.0},
            ),
        ],
    )
    assert adjusted.config_summary["corporate_actions_applied"] == 1
    # Flat adjusted series: entry and every later mark at 50 — no P&L.
    assert adjusted.equity[-1] == pytest.approx(100_000.0)

    raw = run_backtest(
        FakeGraph({"2024-01-01": "Buy"}), "MUUSDT", ["2024-01-01"],
        initial_capital=100_000.0, holding_days=5, cost_bps=0.0,
        price_provider=split_prices, compute_index_alpha=False,
    )
    # The raw series books the split as a -50% move the strategy cannot dodge
    # (and must not be credited with avoiding, either).
    assert raw.equity[-1] == pytest.approx(50_000.0)
    assert "corporate_actions_applied" not in raw.config_summary


@pytest.mark.unit
def test_dividend_adjustment_lifts_prior_bars():
    """A $2 dividend: bars before the ex-date subtract the amount, so a long
    entered pre-ex keeps the economic (dividend-inclusive) return instead of
    eating the ex-day price drop."""

    def flat_prices(ticker, start, end):
        idx = pd.bdate_range(start, end)
        return pd.Series(
            [100.0] * len(idx), index=idx.strftime("%Y-%m-%d"), dtype=float,
        )

    adjusted = run_backtest(
        FakeGraph({"2024-01-01": "Buy"}), "MUUSDT", ["2024-01-01"],
        initial_capital=100_000.0, holding_days=5, cost_bps=0.0,
        price_provider=flat_prices, compute_index_alpha=False,
        corporate_actions=[
            CorporateActionRecord(
                symbol="MU", action_type="dividend", event_date="2024-01-08",
                details={"amount": 2.0},
            ),
        ],
    )
    assert adjusted.config_summary["corporate_actions_applied"] == 1
    # Entry at the adjusted 98 (01-02), series steps back to the raw 100 on
    # the ex-date: +100/98, the dividend-compensated holding return.
    assert adjusted.equity[-1] == pytest.approx(100_000.0 * 100.0 / 98.0)

    raw = run_backtest(
        FakeGraph({"2024-01-01": "Buy"}), "MUUSDT", ["2024-01-01"],
        initial_capital=100_000.0, holding_days=5, cost_bps=0.0,
        price_provider=flat_prices, compute_index_alpha=False,
    )
    assert raw.equity[-1] == pytest.approx(100_000.0)


@pytest.mark.unit
def test_none_corporate_actions_is_byte_identical():
    """None (the only production value today) must change nothing: identical
    equity, identical config_summary, and no disclosure key appears."""
    explicit = run_backtest(
        FakeGraph({"2024-01-01": "Buy"}), "MUUSDT", ["2024-01-01"],
        corporate_actions=None, **_PERP_KW,
    )
    omitted = run_backtest(
        FakeGraph({"2024-01-01": "Buy"}), "MUUSDT", ["2024-01-01"],
        **_PERP_KW,
    )
    assert explicit.equity == omitted.equity
    assert explicit.benchmark_equity == omitted.benchmark_equity
    assert explicit.config_summary == omitted.config_summary
    assert "corporate_actions_applied" not in explicit.config_summary


@pytest.mark.unit
def test_bad_split_ratio_fails_closed():
    from yialpha.backtest.engine import _apply_corporate_actions

    idx = pd.bdate_range("2024-01-01", "2024-01-12")
    raw = pd.Series(
        [100.0] * len(idx), index=idx.strftime("%Y-%m-%d"), dtype=float,
    )
    with pytest.raises(ValueError, match="positive finite ratio"):
        _apply_corporate_actions(raw, "MUUSDT", [
            CorporateActionRecord(
                symbol="MUUSDT", action_type="split", event_date="2024-01-05",
                details={},
            ),
        ])
    with pytest.raises(ValueError, match="details\\['amount'\\]"):
        _apply_corporate_actions(raw, "MUUSDT", [
            CorporateActionRecord(
                symbol="MUUSDT", action_type="dividend",
                event_date="2024-01-05", details={},
            ),
        ])


# --------------------------------------------------------------------------- #
# MMR / stop coverage not already pinned
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_stop_fires_on_last_wick_while_mark_stays_calm(monkeypatch):
    """Default perp wiring uses TWO trigger series: liquidation on MARK kline
    wicks, stops on LAST kline wicks. The already-pinned twin
    (test_perp_engine::test_mark_wick_liquidates_when_last_stays_calm) proves
    the mark side; this pins the mirror — a LAST wick through the stop fires
    it while the mark book (and the 10x liquidation level) is never pierced."""

    def fake_frame(symbol, start, end, interval="1d", venue="binance_perp",
                   price_type="last"):
        idx = pd.bdate_range(start, end)
        n = len(idx)
        # LAST book: a wick to 94 on bar 3; MARK book: calm all window.
        lows = (
            [99.0] * 2 + [94.0] + [99.0] * (n - 3)
            if price_type == "last" else [99.5] * n
        )
        return pd.DataFrame(
            {"Close": [100.0] * n, "Low": lows, "High": [100.5] * n},
            index=idx,
        )

    monkeypatch.setattr(
        "yialpha.dataflows.binance.binance_klines_frame", fake_frame,
    )
    res = run_backtest(
        FakeGraph({"2024-01-01": "Buy"}), "BTCUSDT", ["2024-01-01"],
        weight_fn=_stop_weight_fn(1.0, 97.0),
        leverage=10.0, brackets_provider=lambda t: [(0.0, 0.004)],
        **{k: v for k, v in _PERP_KW.items() if k != "price_provider"},
    )
    events = res.config_summary.get("perp_stop_triggers")
    assert events and len(events) == 1
    assert events[0]["date"] == "2024-01-03"
    assert events[0]["stop_price"] == pytest.approx(97.0)
    assert events[0]["exit_price"] == pytest.approx(97.0)  # slip=0 fee=0
    assert "perp_liquidations" not in res.config_summary
    assert res.config_summary["perp_liq_price_source"] == "mark"
    # 10x long in at 100, stopped at 97: -3% x 10 = -30% locked.
    assert res.equity[-1] == pytest.approx(70_000.0)


@pytest.mark.unit
def test_short_same_bar_liquidation_wins_over_stop():
    """Short-side mirror of test_liq_wins_when_bar_pierces_both_levels: an
    upside wick through BOTH the resting stop (103) and the 10x liquidation
    level (109.6) resolves conservatively as LIQUIDATION first — the stop
    never fires on a dead position."""
    res = run_backtest(
        FakeGraph({"2024-01-01": "Sell"}), "BTCUSDT", ["2024-01-01"],
        allow_short=True, weight_fn=_stop_weight_fn(-1.0, 103.0),
        price_provider=_flat, extremes_provider=_extremes([99.5], [115.0], 2),
        leverage=10.0, brackets_provider=lambda t: [(0.0, 0.004)],
        **{k: v for k, v in _PERP_KW.items() if k != "price_provider"},
    )
    events = res.config_summary.get("perp_liquidations")
    assert events and len(events) == 1
    ev = events[0]
    assert ev["side"] == "short"
    assert ev["date"] == "2024-01-03"
    # Short trigger: entry x (1 + 1/L - MMR - fee) = 100 x 1.096.
    assert ev["liquidation_price"] == pytest.approx(100.0 * 1.096, rel=1e-6)
    assert ev["exit_price"] == pytest.approx(100.0 * 1.096, rel=1e-6)
    assert "perp_stop_triggers" not in res.config_summary
    # Margin remainder at the trigger: -10 x (1.096 - 1) = -96% of capital.
    assert res.equity[-1] == pytest.approx(100_000.0 * (1.0 - 10.0 * 0.096))
