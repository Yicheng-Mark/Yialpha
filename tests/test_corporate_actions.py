"""Pins for corporate-action OHLCV back-adjustment
(yialpha/backtest/engine.py: _apply_corporate_actions + _split_ratio).

The record vocabulary lives in yialpha/instruments/corporate_actions.py
(record stage only — no storage/vendor); the price math lives in the engine.
Prior review verified but never pinned, on small synthetic close series:

* split direction — ratio is NEW shares per OLD share, so bars strictly
  BEFORE the ex-date divide by the ratio; the ex-date bar itself and every
  bar after it stay untouched (strict ``<`` comparison);
* dividend sign — cash dividends SUBTRACT the per-share amount from prior
  closes (additive convention), fail-closed when the subtraction would drive
  any prior close to <= 0;
* date-boundary behavior — an action dated before the price window moves
  nothing; an action dated after the window end currently adjusts ALL bars
  (back-adjustment to the post-action basis — PIT filtering of the records
  themselves is the CALLER's contract; production passes None).

All hermetic: hand-built pd.Series, no vendors, no network.
"""

import pandas as pd
import pytest

from yialpha.backtest.engine import _apply_corporate_actions, _split_ratio
from yialpha.instruments.corporate_actions import CorporateActionRecord

# A flat 100-dollar week: business days 2024-01-02 .. 2024-01-08.
_DATES = ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-08"]


def _flat_series(values=None, dates=None):
    return pd.Series(
        values if values is not None else [100.0] * len(dates or _DATES),
        index=dates or _DATES,
        dtype=float,
    )


def _record(action_type, event_date, symbol="MUUSDT", **details):
    return CorporateActionRecord(
        symbol=symbol,
        action_type=action_type,
        event_date=event_date,
        details=details,
    )


# --------------------------------------------------------------------------- #
# Split direction (new -> old convention)
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_split_divides_prior_bars_new_per_old():
    """2:1 split (ratio 2.0 = 2 new shares per old): prior bars HALVE; the
    ex-date bar and later bars are untouched."""
    out, applied = _apply_corporate_actions(
        _flat_series(), "MUUSDT",
        [_record("split", "2024-01-05", ratio=2.0)],
    )
    assert applied == 1
    assert out["2024-01-02"] == pytest.approx(50.0)
    assert out["2024-01-04"] == pytest.approx(50.0)
    # The ex bar itself stays raw.
    assert out["2024-01-05"] == pytest.approx(100.0)
    # Rows after the ex-date are untouched.
    assert out["2024-01-08"] == pytest.approx(100.0)


@pytest.mark.unit
def test_reverse_split_ratio_below_one_doubles_prior_bars():
    """A 1:2 reverse split is ratio 0.5 (fewer new shares per old): dividing
    prior bars by 0.5 DOUBLES them — the direction is pure arithmetic on the
    ratio, not a hardcoded halving."""
    out, applied = _apply_corporate_actions(
        _flat_series(), "MUUSDT",
        [_record("split", "2024-01-05", ratio=0.5)],
    )
    assert applied == 1
    assert out["2024-01-04"] == pytest.approx(200.0)
    assert out["2024-01-05"] == pytest.approx(100.0)


@pytest.mark.unit
def test_split_exactly_on_first_bar_adjusts_nothing():
    """Event on the FIRST bar: no bar is strictly before it, so nothing moves
    and the record does not count as applied."""
    raw = _flat_series()
    out, applied = _apply_corporate_actions(
        raw, "MUUSDT", [_record("split", "2024-01-02", ratio=2.0)],
    )
    assert applied == 0
    assert out.equals(raw)


@pytest.mark.unit
def test_split_to_from_spelling_matches_ratio():
    """details['to']/details['from'] is the venue spelling of the same ratio
    (to/from = 3 -> prior bars divide by 3)."""
    out, applied = _apply_corporate_actions(
        _flat_series(), "MUUSDT",
        [_record("split", "2024-01-05", to=3, **{"from": 1})],
    )
    assert applied == 1
    assert out["2024-01-04"] == pytest.approx(100.0 / 3.0)
    assert out["2024-01-05"] == pytest.approx(100.0)


@pytest.mark.unit
def test_input_series_is_never_mutated():
    raw = _flat_series()
    _apply_corporate_actions(
        raw, "MUUSDT", [_record("split", "2024-01-05", ratio=2.0)],
    )
    assert (raw == 100.0).all()


# --------------------------------------------------------------------------- #
# Dividend sign (additive convention)
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_dividend_subtracts_from_prior_closes():
    """Cash dividend: prior bars SUBTRACT the per-share amount (the ex-day
    drop is economics the holder receives, so pre-ex levels come down); the
    ex bar and later bars are untouched."""
    out, applied = _apply_corporate_actions(
        _flat_series(), "MUUSDT",
        [_record("dividend", "2024-01-05", amount=2.5, currency="USD")],
    )
    assert applied == 1
    assert out["2024-01-02"] == pytest.approx(97.5)
    assert out["2024-01-04"] == pytest.approx(97.5)
    assert out["2024-01-05"] == pytest.approx(100.0)
    assert out["2024-01-08"] == pytest.approx(100.0)


@pytest.mark.unit
def test_dividend_driving_prior_close_nonpositive_fails_closed():
    """A dividend at or above a prior close is a units error (raw amount
    applied to a split-adjusted series, most likely): the adjuster refuses
    rather than emit a non-positive price."""
    cheap = _flat_series(values=[2.0] * len(_DATES))
    with pytest.raises(ValueError, match="drives a prior close"):
        _apply_corporate_actions(
            cheap, "MUUSDT", [_record("dividend", "2024-01-05", amount=2.5)],
        )
    # Exactly zero after subtraction is equally refused (strictly <= 0).
    with pytest.raises(ValueError, match="drives a prior close"):
        _apply_corporate_actions(
            cheap, "MUUSDT", [_record("dividend", "2024-01-05", amount=2.0)],
        )


# --------------------------------------------------------------------------- #
# Date-boundary / point-in-time behavior
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_record_dated_before_the_window_is_a_noop():
    """An action whose ex-date precedes every bar moves nothing: the whole
    window is already on the post-action basis. applied == 0, byte-identical
    output."""
    raw = _flat_series()
    out, applied = _apply_corporate_actions(
        raw, "MUUSDT", [_record("split", "2023-12-01", ratio=2.0)],
    )
    assert applied == 0
    assert out.equals(raw)


@pytest.mark.unit
def test_record_dated_after_the_window_end_is_skipped():
    """PIT window guard (2026-09-19): an ex-date AFTER the window's last bar
    is not yet effective anywhere in the window and must be skipped — the
    naive ``prior = index < event_date`` mask would select EVERY bar, and a
    future-dated dividend would subtract its amount from every close
    (booking a not-yet-ex dividend into past levels). A uniform split
    rescale happens to preserve returns, but the skip is unconditional.
    Publication PIT (available_at <= analysis date) stays the caller's
    contract (see the next test).
    """
    raw = _flat_series()
    out, applied = _apply_corporate_actions(
        raw, "MUUSDT", [_record("split", "2024-02-01", ratio=2.0)],
    )
    assert applied == 0
    assert out.equals(raw)

    out_div, applied_div = _apply_corporate_actions(
        _flat_series(), "MUUSDT",
        [_record("dividend", "2024-02-01", amount=1.0)],
    )
    assert applied_div == 0
    assert out_div.equals(raw)


@pytest.mark.unit
def test_adjuster_never_consults_available_at():
    """PIT contract boundary: the ENGINE does no knowability filtering of its
    own — two records identical except available_at (one blatantly future,
    one long past) produce byte-identical adjustments. Bounding records by
    available_at <= analysis date belongs to the caller (V2.4 replay)."""
    raw = _flat_series()
    future = _record("split", "2024-01-05", ratio=2.0)
    past = CorporateActionRecord(
        symbol="MUUSDT", action_type="split", event_date="2024-01-05",
        available_at="2023-01-01", details={"ratio": 2.0},
    )
    out_future, _ = _apply_corporate_actions(raw, "MUUSDT", [future])
    out_past, _ = _apply_corporate_actions(raw, "MUUSDT", [past])
    assert out_future.equals(out_past)


@pytest.mark.unit
def test_datetime_index_is_supported():
    """The prior-bar mask is index < str(event_date); a DatetimeIndex
    compares against the ISO string the same way a string index does."""
    idx = pd.bdate_range("2024-01-02", "2024-01-08")
    raw = pd.Series([100.0] * len(idx), index=idx, dtype=float)
    out, applied = _apply_corporate_actions(
        raw, "MUUSDT", [_record("split", "2024-01-05", ratio=2.0)],
    )
    assert applied == 1
    assert out.loc[pd.Timestamp("2024-01-04")] == pytest.approx(50.0)
    assert out.loc[pd.Timestamp("2024-01-05")] == pytest.approx(100.0)


# --------------------------------------------------------------------------- #
# Composition, symbol scoping, identity actions
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_chained_splits_compose_and_order_is_normalized():
    """Records apply chronologically on the RUNNING adjusted series: split 2
    then split 4 leaves the earliest bars at 100/2/4 = 12.5, the middle at
    100/4 = 25, the latest raw. Feeding them in reverse order gives the same
    answer (the loop sorts by event_date)."""
    dates = ["2024-01-02", "2024-01-03", "2024-01-04",
             "2024-01-05", "2024-01-08", "2024-01-09", "2024-01-10"]
    first = _record("split", "2024-01-05", ratio=2.0)
    second = _record("split", "2024-01-09", ratio=4.0)
    out, applied = _apply_corporate_actions(
        _flat_series(dates=dates), "MUUSDT", [second, first],
    )
    assert applied == 2
    assert out["2024-01-02"] == pytest.approx(12.5)
    assert out["2024-01-04"] == pytest.approx(12.5)
    assert out["2024-01-05"] == pytest.approx(25.0)
    assert out["2024-01-08"] == pytest.approx(25.0)
    assert out["2024-01-09"] == pytest.approx(100.0)


@pytest.mark.unit
def test_foreign_symbol_record_is_ignored():
    """A record for a different instrument is data for something else: never
    mis-applied (symbol universe = exact perp symbol and its base form)."""
    raw = _flat_series()
    out, applied = _apply_corporate_actions(
        raw, "MUUSDT", [_record("split", "2024-01-05", symbol="TSLA", ratio=2.0)],
    )
    assert applied == 0
    assert out.equals(raw)


@pytest.mark.unit
def test_identity_actions_move_no_prices():
    """merger / ticker_change / other are identity actions in this skeleton:
    prices unchanged, not counted as applied."""
    raw = _flat_series()
    out, applied = _apply_corporate_actions(
        raw, "MUUSDT",
        [
            _record("merger", "2024-01-03"),
            _record("ticker_change", "2024-01-04"),
            _record("other", "2024-01-05"),
        ],
    )
    assert applied == 0
    assert out.equals(raw)


# --------------------------------------------------------------------------- #
# Fail-closed validation
# --------------------------------------------------------------------------- #


@pytest.mark.unit
@pytest.mark.parametrize(
    "details",
    [{}, {"ratio": 0.0}, {"ratio": -1.0}, {"ratio": float("nan")},
     {"ratio": "two"}, {"to": 2}, {"from": 1}, {"to": 1, "from": 0}],
)
def test_malformed_split_ratio_fails_closed(details):
    with pytest.raises(ValueError, match="positive finite ratio"):
        _apply_corporate_actions(
            _flat_series(), "MUUSDT",
            [_record("split", "2024-01-05", **details)],
        )


@pytest.mark.unit
@pytest.mark.parametrize(
    "details",
    [{}, {"amount": 0.0}, {"amount": -1.0}, {"amount": float("nan")},
     {"amount": "free"}],
)
def test_malformed_dividend_amount_fails_closed(details):
    with pytest.raises(ValueError, match="amount"):
        _apply_corporate_actions(
            _flat_series(), "MUUSDT",
            [_record("dividend", "2024-01-05", **details)],
        )


# --------------------------------------------------------------------------- #
# _split_ratio: record details parsing
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_split_ratio_parsing_precedence_and_rejections():
    def rec(**d):
        return CorporateActionRecord(
            symbol="MU", action_type="split", event_date="2024-01-05", details=d,
        )
    # details['ratio'] wins over a conflicting to/from pair.
    assert _split_ratio(rec(ratio=3.0, to=99, **{"from": 1})) == 3.0
    assert _split_ratio(rec(to=3, **{"from": 1})) == 3.0
    assert _split_ratio(rec(to=1.5, **{"from": 2})) == 0.75
    # Numeric strings are accepted (vendor payloads arrive as strings).
    assert _split_ratio(rec(ratio="2.0")) == 2.0
    # Unparseable / incomplete / zero-denominator -> None (caller raises).
    assert _split_ratio(rec(ratio="two")) is None
    assert _split_ratio(rec(ratio=None, to=None, **{"from": 2})) is None
    assert _split_ratio(rec(to=1, **{"from": 0})) is None
    assert _split_ratio(rec()) is None
