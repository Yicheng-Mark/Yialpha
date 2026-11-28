"""Round 17 — ledger shard: outcome computation, scoreboard, time contract.

Pure helpers first (funding CSV / cadence / window sum, finiteness, horizon
ladder), then the acceptance-time contract, the row models, and the
database-backed seams further down the file.
"""

from __future__ import annotations

import json
import logging
import math
import sqlite3
from collections.abc import Mapping
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from yialpha.dataflows.config import get_config, set_config
from yialpha.ledger import (
    models as ledger_models,
    outcome_compute as oc,
    outcomes as outcomes_mod,
    portfolio,
    scoreboard,
    sqlite as ledger_sqlite,
    time_contract as tc,
)
from yialpha.ledger.evidence import register_run
from yialpha.ledger.outcomes import outcomes_for_prediction
from yialpha.ledger.predictions import submit_predictions
from yialpha.ledger.run_context import (
    LedgerRunContext,
    current_ledger_run_context,
    ensure_ledger_run_context,
    reset_ledger_run_context,
)
from yialpha.ledger.sqlite import ledger_transaction, utc_now_iso
from yialpha.versions import (
    LEGACY_OUTCOME_VERSION,
    OUTCOME_COMPUTE_VERSION,
    PREDICTION_TIME_VERSION,
)

# --------------------------------------------------------------------------- #
# outcome_compute._parse_funding_csv
# --------------------------------------------------------------------------- #


def test_r17_parse_funding_csv_sorts_and_skips_malformed_rows():
    text = (
        "# fundingRate history for BTCUSDT\n"
        "\n"
        "fundingTime,fundingRate\n"
        "2025-03-01 16:00:00,-0.0002\n"
        "2025-03-01 00:00:00,0.0001\n"
        "not-a-time,0.5\n"  # unparseable instant
        "2025-03-01T08:00:00,0.5\n"  # ISO "T" is not the vendor format
        "2025-03-01 12:00:00,abc\n"  # unparseable rate
        "2025-03-01 20:00:00\n"  # short row: rate column missing
        "2025-03-01 08:00:00, 0.0003 \n"
    )
    assert oc._parse_funding_csv(text) == [
        (datetime(2025, 3, 1, 0, 0, tzinfo=UTC), 0.0001),
        (datetime(2025, 3, 1, 8, 0, tzinfo=UTC), 0.0003),
        (datetime(2025, 3, 1, 16, 0, tzinfo=UTC), -0.0002),
    ]


def test_r17_parse_funding_csv_empty_and_comment_only_text_is_empty():
    assert oc._parse_funding_csv("") == []
    assert oc._parse_funding_csv("# nothing\n\n  \n") == []


# --------------------------------------------------------------------------- #
# outcome_compute._modal_cadence_hours
# --------------------------------------------------------------------------- #

_GRID_ORIGIN = datetime(2025, 3, 1, tzinfo=UTC)


@pytest.mark.parametrize(("spacing", "expected"), [(1, 1), (4, 4), (8, 8)])
def test_r17_modal_cadence_hours_regular_grid(spacing, expected):
    times = [_GRID_ORIGIN + timedelta(hours=spacing * i) for i in range(9)]
    assert oc._modal_cadence_hours(times) == expected


def test_r17_modal_cadence_hours_survives_one_missing_settlement():
    times = [_GRID_ORIGIN + timedelta(hours=8 * i) for i in range(9)]
    del times[4]  # one vendor gap: diffs 8,8,8,16,8,8,8 — the 16 must not win
    assert oc._modal_cadence_hours(times) == 8


@pytest.mark.parametrize(
    "times",
    [
        [],
        [_GRID_ORIGIN],
        [_GRID_ORIGIN] * 3,  # repeated instant: no spacing at all
        [_GRID_ORIGIN + timedelta(hours=8), _GRID_ORIGIN],  # strictly decreasing
    ],
    ids=["empty", "single", "all-equal", "decreasing"],
)
def test_r17_modal_cadence_hours_unknown(times):
    assert oc._modal_cadence_hours(times) is None


# --------------------------------------------------------------------------- #
# outcome_compute._funding_window_sum
# --------------------------------------------------------------------------- #

_WINDOW_START = datetime(2025, 3, 1, tzinfo=UTC)
_WINDOW_END = datetime(2025, 3, 2, tzinfo=UTC)


def _funding_csv(*rows: tuple[str, str]) -> str:
    return "fundingTime,fundingRate\n" + "".join(f"{moment},{rate}\n" for moment, rate in rows)


def _funding_rows(start: datetime, count: int, step_hours: int, rate: str = "0.0001"):
    return [
        ((start + timedelta(hours=step_hours * i)).strftime("%Y-%m-%d %H:%M:%S"), rate)
        for i in range(count)
    ]


class _FundingSpy:
    """Stands in for ``get_binance_funding_rate`` and records every call."""

    def __init__(self, csv_text: str) -> None:
        self.csv_text = csv_text
        self.calls: list[tuple[str, str, str]] = []

    def __call__(self, symbol: str, start: str, end: str) -> str:
        self.calls.append((symbol, start, end))
        return self.csv_text


@pytest.mark.parametrize("cadence", [1, 4, 8])
def test_r17_funding_window_sum_charges_every_settlement_of_the_cadence(monkeypatch, cadence):
    step = timedelta(hours=cadence)
    rows: list[tuple[str, str]] = []
    moment = _WINDOW_START - step
    while moment <= _WINDOW_END + step:  # one slot of margin on both sides
        rows.append((moment.strftime("%Y-%m-%d %H:%M:%S"), "0.0001"))
        moment += step
    spy = _FundingSpy(_funding_csv(*rows))
    monkeypatch.setattr(oc, "get_binance_funding_rate", spy)

    total, last_iso = oc._funding_window_sum("BTCUSDT", _WINDOW_START, _WINDOW_END)

    # 24h window: the start instant is excluded, the end instant included.
    assert total == pytest.approx(0.0001 * 24 / cadence)
    assert last_iso == "2025-03-02T00:00:00+00:00"
    # the vendor seam is asked for the DATE span of the window
    assert spy.calls == [("BTCUSDT", "2025-03-01", "2025-03-02")]


def test_r17_funding_window_sum_ignores_start_instant_and_rows_past_the_end(monkeypatch):
    text = _funding_csv(
        ("2025-03-01 00:00:00", "9.0"),  # exactly at the start: NOT charged
        ("2025-03-01 08:00:00", "0.0001"),
        ("2025-03-01 16:00:00", "-0.0002"),
        ("2025-03-02 00:00:00", "0.0003"),  # exactly at the end: charged
        ("2025-03-02 08:00:00", "9.0"),  # past the end: cannot change anything
    )
    monkeypatch.setattr(oc, "get_binance_funding_rate", _FundingSpy(text))

    total, last_iso = oc._funding_window_sum("BTCUSDT", _WINDOW_START, _WINDOW_END)

    assert total == pytest.approx(0.0002)
    assert last_iso == "2025-03-02T00:00:00+00:00"


def test_r17_funding_window_sum_duplicate_vendor_row_is_not_another_settlement(monkeypatch):
    rows = _funding_rows(_WINDOW_START, 4, 8)
    text = _funding_csv(*rows, rows[1], rows[3])  # same instant, same rate
    monkeypatch.setattr(oc, "get_binance_funding_rate", _FundingSpy(text))

    total, last_iso = oc._funding_window_sum("BTCUSDT", _WINDOW_START, _WINDOW_END)

    assert total == pytest.approx(0.0003)  # 3 in-window settlements, not 5
    assert last_iso == "2025-03-02T00:00:00+00:00"


def test_r17_funding_window_sum_unparseable_rate_past_the_end_is_ignored(monkeypatch):
    # "data past the fixed endpoint cannot change cadence or any attribution":
    # a nan row after the end must not fail the window closed.
    text = _funding_csv(*_funding_rows(_WINDOW_START, 4, 8), ("2025-03-02 08:00:00", "nan"))
    monkeypatch.setattr(oc, "get_binance_funding_rate", _FundingSpy(text))

    assert oc._funding_window_sum("BTCUSDT", _WINDOW_START, _WINDOW_END) == (
        pytest.approx(0.0003),
        "2025-03-02T00:00:00+00:00",
    )


def test_r17_funding_window_sum_leading_gap_fails_closed(monkeypatch):
    # Window (03-01 00:00, 03-01 16:00] on a 4h grid whose 04:00 settlement is
    # missing. The first returned settlement (08:00) is inside the window, so
    # only the backward grid extension can see the hole.
    window_end = datetime(2025, 3, 1, 16, tzinfo=UTC)
    holed = _funding_csv(*_funding_rows(datetime(2025, 3, 1, 8, tzinfo=UTC), 3, 4))
    monkeypatch.setattr(oc, "get_binance_funding_rate", _FundingSpy(holed))
    assert oc._funding_window_sum("BTCUSDT", _WINDOW_START, window_end) == (None, None)

    # ... and the same window with the 04:00 settlement present sums 4 slots.
    whole = _funding_csv(*_funding_rows(_WINDOW_START, 5, 4))
    monkeypatch.setattr(oc, "get_binance_funding_rate", _FundingSpy(whole))
    assert oc._funding_window_sum("BTCUSDT", _WINDOW_START, window_end) == (
        pytest.approx(0.0004),
        "2025-03-01T16:00:00+00:00",
    )


@pytest.mark.parametrize(
    ("rows", "why"),
    [
        ([], "empty vendor text"),
        (_funding_rows(datetime(2025, 2, 20, tzinfo=UTC), 4, 8), "nothing inside the window"),
        ([("2025-03-01 08:00:00", "0.0001")], "one settlement: cadence unknown"),
        (
            _funding_rows(_WINDOW_START, 4, 8)[:1]
            + [("2025-03-01 08:00:00", "nan")]
            + _funding_rows(_WINDOW_START, 4, 8)[2:],
            "non-finite rate",
        ),
        (
            _funding_rows(datetime(2025, 2, 28, tzinfo=UTC), 3, 8)
            + _funding_rows(_WINDOW_START, 2, 8)
            + [("2025-03-02 00:00:00", "0.0001")],  # 03-01 16:00 absent
            "interior gap",
        ),
    ],
    ids=["empty", "outside", "single", "nan", "gap"],
)
def test_r17_funding_window_sum_fails_closed(monkeypatch, rows, why):
    monkeypatch.setattr(oc, "get_binance_funding_rate", _FundingSpy(_funding_csv(*rows)))
    assert oc._funding_window_sum("BTCUSDT", _WINDOW_START, _WINDOW_END) == (None, None), why


def test_r17_funding_window_sum_conflicting_rates_at_one_instant_fail_closed(monkeypatch):
    rows = _funding_rows(_WINDOW_START, 4, 8)
    # same instant, a different rate: ambiguous, never silently summed
    text = _funding_csv(*rows, ("2025-03-01 08:00:00", "0.0007"))
    monkeypatch.setattr(oc, "get_binance_funding_rate", _FundingSpy(text))
    assert oc._funding_window_sum("BTCUSDT", _WINDOW_START, _WINDOW_END) == (None, None)


# --------------------------------------------------------------------------- #
# scoreboard._finite
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0.0, True),
        (1, True),
        ("0.25", True),  # anything float() accepts
        (None, False),
        (math.nan, False),
        (math.inf, False),
        (-math.inf, False),
        ("n/a", False),  # ValueError
        ([0.5], False),  # TypeError
        (10**400, False),  # OverflowError: int too large to convert to float
    ],
    ids=str,
)
def test_r17_finite_rejects_every_non_finite_shape(value, expected):
    assert scoreboard._finite(value) is expected


# --------------------------------------------------------------------------- #
# models.validate_horizon_days
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("bad", [None, "week", "5.5", "", [5]])
def test_r17_validate_horizon_days_non_integer_raises_with_cause(bad):
    with pytest.raises(ValueError) as exc_info:
        ledger_models.validate_horizon_days(bad)

    assert str(exc_info.value) == f"horizon_days must be an integer, got {bad!r}"
    assert isinstance(exc_info.value.__cause__, (TypeError, ValueError))


def test_r17_validate_horizon_days_off_ladder_raises():
    with pytest.raises(ValueError) as exc_info:
        ledger_models.validate_horizon_days(7)
    assert str(exc_info.value) == "horizon_days 7 not in horizon ladder (1, 5, 21)"


@pytest.mark.parametrize(("given", "expected"), [(1, 1), ("5", 5), (21, 21)])
def test_r17_validate_horizon_days_ladder_values_returned_as_int(given, expected):
    assert ledger_models.validate_horizon_days(given) == expected


def test_r17_validate_horizon_days_rejects_a_boolean():
    """Regression. ``True`` is an ``int`` in Python: ``int(True)`` is 1, so a
    JSON ``true`` for the horizon was filed as the 1-day horizon."""
    with pytest.raises(ValueError) as exc_info:
        ledger_models.validate_horizon_days(True)

    assert str(exc_info.value) == "horizon_days must be an integer, got True"


@pytest.mark.parametrize("name", ["prob_up", "confidence"])
@pytest.mark.parametrize("value", [True, False])
def test_r17_validate_probability_rejects_a_boolean(name, value):
    """Regression. ``float(True)`` is 1.0: a JSON ``true`` was filed as a
    certain forecast. One such row on the wrong side costs log(1e6) = 13.8 in
    log loss (150 honest rows: 0.53 -> 0.62)."""
    with pytest.raises(ValueError) as exc_info:
        ledger_models.validate_probability(name, value)

    assert str(exc_info.value) == f"{name} must be within [0, 1], got {value}"


def test_r17_prediction_entry_never_carries_a_boolean_in_a_numeric_field():
    """The row model is the last gate: a boolean in a numeric field is refused,
    while plain ints remain valid numbers."""
    with pytest.raises(ValueError, match="horizon_days must be an integer, got True"):
        ledger_models.PredictionEntry(horizon_days=True, direction="up")
    with pytest.raises(ValueError, match=r"prob_up must be within \[0, 1\], got True"):
        ledger_models.coerce_prediction_entry({"horizon_days": 1, "direction": "up", "prob_up": True})
    with pytest.raises(ValueError, match=r"confidence must be within \[0, 1\], got False"):
        ledger_models.coerce_prediction_entry(
            {"horizon_days": 1, "direction": "up", "confidence": False}
        )

    entry = ledger_models.PredictionEntry(horizon_days=1, direction="up", prob_up=1, confidence=0)
    assert (entry.prob_up, entry.confidence) == (1.0, 0.0)


# --------------------------------------------------------------------------- #
# time_contract.midnight_cross_grace_day
# --------------------------------------------------------------------------- #

#: A Tuesday 00:02 UTC formation: the expected reference is Monday's close, so
#: the grace day is the bar-day before it.
_FORMED_TUESDAY = datetime(2025, 3, 4, 0, 2, tzinfo=UTC)


def test_r17_grace_day_equity_seam_walks_back_over_the_weekend():
    # Monday 03-03 -> previous bar-day Sunday 03-02 -> Saturday -> Friday.
    assert tc.midnight_cross_grace_day(_FORMED_TUESDAY, equity=True) == date(2025, 2, 28)
    # the perp seam settles every calendar day: no walk.
    assert tc.midnight_cross_grace_day(_FORMED_TUESDAY, equity=False) == date(2025, 3, 2)
    # formed Monday: previous bar-day is Thursday, one weekend walk earlier.
    formed_monday = datetime(2025, 3, 3, 0, 2, tzinfo=UTC)
    assert tc.midnight_cross_grace_day(formed_monday, equity=True) == date(2025, 2, 27)


@pytest.mark.parametrize(
    ("formed", "perp_grace", "equity_grace"),
    [
        (datetime(2025, 3, 4, 0, 0, 0, tzinfo=UTC), date(2025, 3, 2), date(2025, 2, 28)),
        (datetime(2025, 3, 4, 0, 5, 0, tzinfo=UTC), date(2025, 3, 2), date(2025, 2, 28)),
        (datetime(2025, 3, 4, 0, 6, 0, tzinfo=UTC), None, None),
        (datetime(2025, 3, 4, 0, 59, 0, tzinfo=UTC), None, None),
        (datetime(2025, 3, 4, 1, 0, 0, tzinfo=UTC), None, None),
        (datetime(2025, 3, 3, 23, 59, 0, tzinfo=UTC), None, None),
    ],
    ids=["00:00:00", "00:05:00", "00:06:00", "00:59:00", "01:00:00", "day-before"],
)
def test_r17_grace_day_is_admitted_only_inside_the_first_minutes_after_utc_midnight(
    formed, perp_grace, equity_grace
):
    assert tc.midnight_cross_grace_day(formed, equity=False) == perp_grace
    assert tc.midnight_cross_grace_day(formed, equity=True) == equity_grace


# --------------------------------------------------------------------------- #
# time_contract.validate_timing
# --------------------------------------------------------------------------- #

_DROP = object()


def _timing(**overrides: Any) -> dict[str, Any]:
    """One valid ``close_reference_v1`` snapshot, with overrides applied."""
    base: dict[str, Any] = {
        "version": PREDICTION_TIME_VERSION,
        "prediction_formed_at": "2025-03-10T12:00:00+00:00",
        "reference_price_at": "2025-03-10T00:00:00+00:00",
        "reference_available_at": "2025-03-10T00:00:00+00:00",
        "reference_observed_at": "2025-03-10T11:59:00+00:00",
        "reference_price": 100.0,
        "reference_source": "binance_perp:1d:last",
        "reference_error": None,
    }
    for key, value in overrides.items():
        if value is _DROP:
            del base[key]
        else:
            base[key] = value
    return base


def test_r17_validate_timing_round_trips_a_valid_snapshot():
    assert tc.validate_timing(_timing()) == _timing()


@pytest.mark.parametrize("not_a_mapping", ["close", [("version", 1)], 5])
def test_r17_validate_timing_rejects_anything_but_a_mapping(not_a_mapping):
    with pytest.raises(ValueError) as exc_info:
        tc.validate_timing(not_a_mapping)
    assert str(exc_info.value) == "timing must be a mapping or None"


def test_r17_validate_timing_accepts_any_mapping_type():
    class _PairTuples(Mapping):  # type: ignore[type-arg]
        def __iter__(self):
            return iter(_timing())

        def __len__(self):
            return len(_timing())

        def __getitem__(self, key):
            return _timing()[key]

    assert tc.validate_timing(_PairTuples()) == _timing()


def test_r17_validate_timing_requires_exactly_the_close_reference_fields():
    message = "timing must contain exactly the close-reference fields"
    without_observed = _timing(reference_observed_at=_DROP)
    with pytest.raises(ValueError, match=message):
        tc.validate_timing(without_observed)
    with pytest.raises(ValueError, match=message):
        tc.validate_timing(_timing(reference_close=99.0))
    assert set(tc._FIELDS) == set(_timing())


@pytest.mark.parametrize(
    ("timing", "message"),
    [
        (_timing(reference_source=""), "reference_source must be a nonempty string or None"),
        (_timing(reference_error=5), "reference_error must be a nonempty string or None"),
        (_timing(reference_error=["x"]), "reference_error must be a nonempty string or None"),
        (
            _timing(reference_observed_at=None),
            "a reference price requires all timestamps and its source",
        ),
        (
            _timing(reference_source=None),
            "a reference price requires all timestamps and its source",
        ),
        (
            _timing(reference_error="reference_bar_missing:2025-03-09"),
            "a valid reference price cannot carry reference_error",
        ),
    ],
    ids=["empty-source", "int-error", "list-error", "no-observed", "no-source", "price-and-error"],
)
def test_r17_validate_timing_inconsistent_reference_fields_raise(timing, message):
    with pytest.raises(ValueError) as exc_info:
        tc.validate_timing(timing)
    assert str(exc_info.value) == message


@pytest.mark.parametrize("key", ["reference_price_at", "reference_available_at"])
def test_r17_validate_timing_price_timestamps_without_a_price_are_rejected(key):
    other = "reference_available_at" if key == "reference_price_at" else "reference_price_at"
    with pytest.raises(ValueError) as exc_info:
        tc.validate_timing(_timing(reference_price=None, **{other: None}))
    assert str(exc_info.value) == "reference price timestamps require a reference_price"


def test_r17_validate_timing_error_snapshot_keeps_only_the_observation_instant():
    """A failed fetch is an explicit new-contract snapshot: no price, no price
    timestamps, but the instant the vendor was observed stays on record."""
    failed = _timing(
        reference_price=None,
        reference_price_at=None,
        reference_available_at=None,
        reference_error="reference_fetch_failed:RuntimeError",
    )
    assert tc.validate_timing(failed) == failed


def test_r17_validate_timing_canonicalises_instants_to_utc_without_mutating_input():
    given = _timing(
        prediction_formed_at="2025-03-10T20:00:00+08:00",
        reference_observed_at="2025-03-10T11:59:00Z",
        reference_price=100,  # int is widened to float
    )
    untouched = dict(given)

    canonical = tc.validate_timing(given)

    assert given == untouched
    assert canonical == _timing()
    assert type(canonical["reference_price"]) is float


# --------------------------------------------------------------------------- #
# time_contract.capture_prediction_timing — closed-at guard
# --------------------------------------------------------------------------- #


def _frame(dates: tuple[str, ...] = ("2026-09-04",), prices: tuple[float, ...] = (100.0,)):
    return pd.DataFrame({"Close": prices}, index=pd.to_datetime(dates))


def test_r17_capture_refuses_a_bar_whose_close_is_after_the_observation_instant(monkeypatch):
    """Defensive closed-at guard (a wall clock that steps back between request
    and observation). Requested 09-05 12:00 -> expected bar 09-04, closing
    09-05 00:00; the frame is "observed" at 09-04 23:00, an instant at which
    that bar had NOT closed yet — it must not become the reference."""
    moments = iter(
        datetime.fromisoformat(value)
        for value in (
            "2026-09-05T12:00:00+00:00",  # requested_at
            "2026-09-04T23:00:00+00:00",  # observed_at: the clock stepped back
            "2026-09-04T23:00:01+00:00",  # formed_at
        )
    )
    monkeypatch.setattr(tc, "_now_utc", lambda: next(moments))
    monkeypatch.setattr(tc, "binance_klines_frame", lambda *a, **k: _frame())

    timing = tc.capture_prediction_timing("BTCUSDT", "CONTRACT", asset_type="crypto_perp")

    assert timing["reference_price"] is None
    assert timing["reference_error"] == "reference_bar_not_closed"
    assert timing["reference_observed_at"] == "2026-09-04T23:00:00+00:00"
    assert timing["reference_price_at"] is None
    assert timing["reference_available_at"] is None


# --------------------------------------------------------------------------- #
# models — entry coercion and row converters
# --------------------------------------------------------------------------- #


def test_r17_coerce_prediction_entry_rejects_non_mapping_with_type_name():
    with pytest.raises(ValueError) as exc_info:
        ledger_models.coerce_prediction_entry(["horizon_days"])  # type: ignore[arg-type]
    assert str(exc_info.value) == (
        "prediction entry must be a mapping or PredictionEntry, got list"
    )


@pytest.mark.parametrize(
    ("entry", "missing"),
    [
        ({"direction": "up"}, "['horizon_days']"),
        ({"horizon_days": 5}, "['direction']"),
        ({"prob_up": 0.6}, "['direction', 'horizon_days']"),
    ],
    ids=["no-horizon", "no-direction", "neither"],
)
def test_r17_coerce_prediction_entry_requires_horizon_and_direction(entry, missing):
    with pytest.raises(ValueError) as exc_info:
        ledger_models.coerce_prediction_entry(entry)
    assert str(exc_info.value) == (
        f"prediction entry requires 'horizon_days' and 'direction'; missing {missing}"
    )


def _sqlite_row(**columns: Any) -> sqlite3.Row:
    """A genuine ``sqlite3.Row`` carrying exactly ``columns`` (no ledger file)."""
    connection = sqlite3.connect(":memory:")
    try:
        connection.row_factory = sqlite3.Row
        # bound parameters with explicit aliases: an unaliased "?" column has
        # no name, and a bare double-quoted name is a string literal here.
        projection = ", ".join(f'? AS "{name}"' for name in columns)
        return connection.execute(  # type: ignore[return-value]
            f"SELECT {projection}", tuple(columns.values())
        ).fetchone()
    finally:
        connection.close()


def test_r17_row_to_run_maps_every_column_and_keeps_nulls():
    row = _sqlite_row(
        run_id="run-1",
        ticker="BTCUSDT",
        asset_type="crypto_perp",
        instrument_class=None,
        analysis_as_of="2026-08-25",
        created_at="2026-08-25T10:00:00+00:00",
        config_digest=None,
        schema_version="v2.1",
    )
    assert ledger_models.row_to_run(row) == ledger_models.RunRecord(
        run_id="run-1",
        ticker="BTCUSDT",
        asset_type="crypto_perp",
        instrument_class=None,
        analysis_as_of="2026-08-25",
        created_at="2026-08-25T10:00:00+00:00",
        config_digest=None,
        schema_version="v2.1",
    )


def test_r17_row_to_ticket_mirror_maps_every_column_and_keeps_nulls():
    row = _sqlite_row(
        ticket_id="ticket-1",
        decision_id=None,
        run_id="run-1",
        payload='{"side": "long"}',
        ticket_version="v1",
        written_at="2026-08-25T10:00:00+00:00",
    )
    assert ledger_models.row_to_ticket_mirror(row) == ledger_models.TicketMirror(
        ticket_id="ticket-1",
        decision_id=None,
        run_id="run-1",
        payload='{"side": "long"}',
        ticket_version="v1",
        written_at="2026-08-25T10:00:00+00:00",
    )


# --------------------------------------------------------------------------- #
# outcome_compute — vendor seams and the per-test ledger
# --------------------------------------------------------------------------- #

_RUN = "run-r17-ledger"
_AS_OF = "2026-08-25"  # a Tuesday; the 5-day horizon ends on Sunday 2026-08-30
_NOW = "2026-09-05"  # horizon end + 6 days: inside the 7-day retry grace
_AFTER_GRACE = "2026-09-07"  # horizon end + 8 days: a gap is terminal now
_COST_DETAIL = {
    "entry_fee_bps": 5.0,
    "exit_fee_bps": 5.0,
    "entry_slippage_bps": 2.0,
    "exit_slippage_bps": 2.0,
}
_PERP_CLOSES = {
    "2026-08-24": 99.0,
    "2026-08-25": 100.0,  # legacy entry: the analysis date's own bar
    "2026-08-26": 103.0,
    "2026-08-27": 105.0,
    "2026-08-28": 107.0,
    "2026-08-29": 109.0,
    "2026-08-30": 110.0,  # legacy exit: the horizon end's own bar (perps print daily)
    "2026-09-01": 130.0,  # beyond the horizon: must not move the exit
}
_EQUITY_CLOSES = {
    "2026-08-24": 99.0,
    "2026-08-25": 100.0,  # entry
    "2026-08-26": 103.0,
    "2026-08-27": 105.0,
    "2026-08-28": 107.0,  # exit: the last weekday at/before the Sunday horizon end
}


def _closes_frame(closes: dict[str, float]) -> pd.DataFrame:
    """Vendor-shaped daily close frame (DatetimeIndex + ``Close`` column)."""
    days = sorted(closes)
    return pd.DataFrame({"Close": [closes[day] for day in days]}, index=pd.to_datetime(days))


def _funding_grid(rate: float = 0.0001) -> str:
    """The funding seam's CSV: an 8h grid from 2026-08-20 through 2026-08-31."""
    stamps = pd.date_range("2026-08-20", "2026-08-31", freq="8h")
    return "\n".join(
        ["# funding (fixture)", "fundingTime,fundingRate,symbol"]
        + [f"{stamp:%Y-%m-%d %H:%M:%S},{rate},BTCUSDT" for stamp in stamps]
    )


class _Seams:
    """Synthetic vendor answers installed on outcome_compute's bindings."""

    def __init__(
        self,
        *,
        perp: dict[str, float] | None = None,
        equity: dict[str, float] | None = None,
        funding: str | None = None,
        equity_error: Exception | None = None,
        funding_error: Exception | None = None,
    ) -> None:
        self.perp = dict(perp if perp is not None else _PERP_CLOSES)
        self.equity = dict(equity or {})
        self.funding = funding if funding is not None else _funding_grid()
        self.equity_error = equity_error
        self.funding_error = funding_error
        self.equity_calls: list[str] = []

    def install(self, monkeypatch: pytest.MonkeyPatch) -> _Seams:
        monkeypatch.setattr(oc, "binance_klines_frame", self._klines)
        monkeypatch.setattr(oc, "get_YFin_history_cached", self._equity_history)
        monkeypatch.setattr(oc, "get_binance_funding_rate", self._funding)
        return self

    def _klines(self, symbol: str, start_date: str, end_date: str, **_kwargs: object):
        return _closes_frame(self.perp)

    def _equity_history(self, symbol: str, start_date: str, end_date: str):
        self.equity_calls.append(symbol)
        if self.equity_error is not None:
            raise self.equity_error
        return _closes_frame(self.equity)

    def _funding(self, symbol: str, start_date: str, end_date: str) -> str:
        if self.funding_error is not None:
            raise self.funding_error
        return self.funding


def _seed(
    run_id: str = _RUN,
    *,
    analyst: str = "market",
    direction: str = "up",
    scope: str = ledger_models.SCOPE_CONTRACT,
    ticker: str = "BTCUSDT",
    asset_type: str = "crypto_perp",
    instrument_class: str | None = "pure_crypto_perp",
    horizon: int = 5,
) -> str:
    """One run + one prediction WITHOUT timing (the legacy path); returns its id."""
    register_run(run_id, ticker, asset_type, instrument_class, _AS_OF)
    (prediction_id,) = submit_predictions(
        run_id,
        analyst,
        ticker,
        scope,
        [{"horizon_days": horizon, "direction": direction, "prob_up": 0.6}],
        _AS_OF,
    )
    return prediction_id


def _write_ticket(run_id: str = _RUN, payload: dict[str, Any] | None = None) -> None:
    """One ticket-mirror row carrying the cost-model decomposition."""
    with ledger_transaction() as cur:
        cur.execute(
            "INSERT INTO tickets (ticket_id, decision_id, run_id, payload, "
            "ticket_version, written_at) VALUES (?, ?, ?, ?, ?, ?)",
            ("T-r17", None, run_id, json.dumps(payload or {"cost_detail": _COST_DETAIL}), "v1", utc_now_iso()),
        )


# --------------------------------------------------------------------------- #
# outcome_compute, legacy path — invalid closes must fail closed
# --------------------------------------------------------------------------- #


def _assert_gap_fails_closed(prediction_id: str, reason: str) -> ledger_models.OutcomeRecord:
    """Inside the retry grace nothing is written and the row stays pending;
    past it a terminal ``incomplete`` row names the reason and carries no
    fabricated number."""
    early = oc.compute_outcomes(_NOW)
    assert (early.considered, early.completed, early.incomplete, early.failed) == (1, 0, 0, 0)
    (detail,) = early.details
    assert (detail.status, detail.reason, detail.legs_missing) == (
        "pending", reason, (oc.LEG_CONTRACT_PRICE,)
    )
    assert outcomes_for_prediction(prediction_id) == []

    late = oc.compute_outcomes(_AFTER_GRACE)
    assert (late.considered, late.completed, late.incomplete, late.failed) == (1, 0, 1, 0)
    (record,) = outcomes_for_prediction(prediction_id)
    assert record.status == "incomplete"
    assert record.legs_missing == (oc.LEG_CONTRACT_PRICE,)
    assert (record.contract_price_return, record.net_return, record.fees) == (None, None, None)
    assert record.scoring_context["reason"] == reason
    return record


@pytest.mark.parametrize("bad", [math.nan, math.inf, 0.0, -110.0], ids=["nan", "inf", "zero", "negative"])
def test_r17_legacy_exit_close_that_is_not_a_finite_positive_number_fails_closed(monkeypatch, bad):
    """Regression (money path). The legacy scorer divided by whatever close the
    vendor returned. A non-finite exit close was written as status='complete'
    with its return columns NULL (NaN) or Inf in SQLite; a zero or negative
    close produced a fabricated -100% or worse, which the scoreboard admits
    because it is finite. The reference path already rejects such an exit bar
    (``horizon_end_price_invalid``); the legacy path had no gate."""
    _Seams(perp={**_PERP_CLOSES, "2026-08-30": bad}).install(monkeypatch)
    prediction_id = _seed()
    _write_ticket()

    record = _assert_gap_fails_closed(prediction_id, "horizon_end_price_invalid")

    assert record.outcome_id is not None


@pytest.mark.parametrize("bad", [math.nan, 0.0, -100.0], ids=["nan", "zero", "negative"])
def test_r17_legacy_entry_close_that_is_not_a_finite_positive_number_fails_closed(monkeypatch, bad):
    """Regression (money path). The legacy entry bar had no validity gate
    either: NaN / 0 raised out of the row handler (JSON "nan", division by
    zero), so the row was "failed" and re-fetched every batch without ever
    retiring; a negative close was scored complete at a fabricated return."""
    _Seams(perp={**_PERP_CLOSES, "2026-08-25": bad}).install(monkeypatch)
    prediction_id = _seed()
    _write_ticket()

    _assert_gap_fails_closed(prediction_id, "legacy_entry_price_invalid")


@pytest.mark.parametrize("bad", [math.nan, 0.0], ids=["nan", "zero"])
@pytest.mark.parametrize(("bad_day", "reason"), [("2026-08-28", "horizon_end_price_invalid"), ("2026-08-25", "legacy_entry_price_invalid")])
def test_r17_legacy_equity_close_that_is_not_a_finite_positive_number_fails_closed(
    monkeypatch, bad, bad_day, reason
):
    """Same gate on the equity seam (weekday bars, the Sunday horizon resolves
    on Friday's close)."""
    _Seams(equity={**_EQUITY_CLOSES, bad_day: bad}).install(monkeypatch)
    prediction_id = _seed(ticker="DEMO", asset_type="stock", instrument_class=None)
    _write_ticket()

    _assert_gap_fails_closed(prediction_id, reason)


def test_r17_legacy_underlying_diagnostic_with_an_invalid_close_is_reported_missing(monkeypatch):
    """Regression. The stock-perp underlying diagnostic divided the underlying
    closes without checking them: a NaN close was stored as NULL in a row whose
    ``legs_missing`` claimed every leg was present."""
    _Seams(
        perp=_PERP_CLOSES,
        equity={
            "2026-08-24": 199.0,
            "2026-08-25": 200.0,
            "2026-08-28": math.nan,
        },
    ).install(monkeypatch)
    prediction_id = _seed(ticker="TSLAUSDT", instrument_class="stock_perp")
    _write_ticket()

    report = oc.compute_outcomes(_NOW)

    # A diagnostic gap never blocks the contract label ...
    assert (report.completed, report.incomplete, report.failed) == (1, 0, 0)
    (record,) = outcomes_for_prediction(prediction_id)
    assert record.status == "complete"
    assert record.contract_price_return == pytest.approx(0.10)
    assert record.net_return == pytest.approx(0.0971)
    # ... but it is reported, and carries no number.
    assert record.legs_missing == (oc.LEG_UNDERLYING,)
    assert (record.underlying_return, record.basis_return) == (None, None)


@pytest.mark.parametrize(
    ("equity_error", "funding_error", "status", "missing", "reason"),
    [
        (RuntimeError("equity vendor down"), None, "complete", (oc.LEG_UNDERLYING,), None),
        (None, RuntimeError("funding vendor down"), "incomplete", (oc.LEG_FUNDING,), "attribution_legs_missing"),
    ],
    ids=["underlying-fetch-fails", "funding-fetch-fails"],
)
def test_r17_legacy_vendor_failure_on_one_leg_degrades_that_leg_and_retires_the_row(
    monkeypatch, caplog, equity_error, funding_error, status, missing, reason
):
    """A vendor exception on the underlying diagnostic or the funding leg is
    logged at DEBUG and fails that leg closed: the row is written (complete
    with the diagnostic gap, or incomplete with the mandatory leg missing).
    It used to raise out of the row handler as "failed" without retiring, so
    every later batch re-fetched the same dead window."""
    _Seams(
        equity_error=equity_error,
        funding_error=funding_error,
        equity={"2026-08-25": 200.0, "2026-08-28": 210.0},
    ).install(monkeypatch)
    prediction_id = _seed(ticker="TSLAUSDT", instrument_class="stock_perp")
    _write_ticket()

    with caplog.at_level(logging.DEBUG, logger=oc.logger.name):
        report = oc.compute_outcomes(_NOW)

    assert (report.completed, report.incomplete, report.failed) == (
        (1, 0, 0) if status == "complete" else (0, 1, 0)
    )
    (record,) = outcomes_for_prediction(prediction_id)
    assert record.status == status
    assert record.legs_missing == missing
    assert record.contract_price_return == pytest.approx(0.10)
    debug_messages = [
        record_.getMessage()
        for record_ in caplog.records
        if record_.name == oc.logger.name and record_.levelno == logging.DEBUG
    ]
    assert len(debug_messages) == 1
    assert debug_messages[0].startswith(
        "legacy underlying diagnostic unavailable" if equity_error else "legacy funding leg unavailable"
    )


# --------------------------------------------------------------------------- #
# outcome_compute — ticket cost legs
# --------------------------------------------------------------------------- #


def _ticket(payload_text: str, *, ticket_id: str = "T-r17", written_at: str = _AS_OF + "T10:00:00+00:00") -> None:
    with ledger_transaction() as cur:
        cur.execute(
            "INSERT INTO tickets (ticket_id, decision_id, run_id, payload, "
            "ticket_version, written_at) VALUES (?, ?, ?, ?, ?, ?)",
            (ticket_id, None, _RUN, payload_text, "v1", written_at),
        )


@pytest.mark.parametrize("bad", ["five", None, [5.0], object()], ids=["str", "none", "list", "object"])
def test_r17_cost_split_from_mapping_rejects_a_non_numeric_bps_field(bad):
    # A decomposition is usable only when all four fields parse as numbers.
    assert oc._cost_split_from_mapping({**_COST_DETAIL, "entry_fee_bps": bad}) is None


def test_r17_cost_split_from_mapping_sums_each_side_and_turns_bps_into_a_fraction():
    # fees = entry + exit fee bps over 1e4; slippage likewise: a fraction, never percent.
    assert oc._cost_split_from_mapping(_COST_DETAIL) == (pytest.approx(0.0010), pytest.approx(0.0004))
    assert oc._cost_split_from_mapping(
        {"entry_fee_bps": "5", "exit_fee_bps": 5, "entry_slippage_bps": 0, "exit_slippage_bps": 0.0}
    ) == (pytest.approx(0.0010), 0.0)
    # An omitted field is not a decomposition.
    incomplete = {key: value for key, value in _COST_DETAIL.items() if key != "exit_fee_bps"}
    assert oc._cost_split_from_mapping(incomplete) is None


@pytest.mark.parametrize(
    ("payload_text", "why"),
    [
        ("not json", "not JSON at all"),
        ('["not", "an object"]', "JSON that is not an object"),
        ('{"note": "no cost model here"}', "neither a decomposition nor a scalar total"),
        ('{"estimated_cost": "0.02"}', "a scalar total that is a string"),
        ('{"estimated_cost": true}', "a scalar total that is a bool"),
    ],
)
def test_r17_ticket_cost_legs_unusable_payload_keeps_the_ticket_id_and_marks_the_legs_missing(
    payload_text, why
):
    _ticket(payload_text)

    assert oc._ticket_cost_legs(_RUN) == ("T-r17", None, None), why
    assert oc._ticket_cost_legs(_RUN, require_decomposition=True) == ("T-r17", None, None)


def test_r17_ticket_cost_legs_without_a_ticket_row_or_a_ledger_file(tmp_path):
    assert oc._ticket_cost_legs("run-without-a-ticket") == (None, None, None)
    # No ledger file at all: the optional read degrades to "no row".
    set_config({"ledger_db_path": str(tmp_path / "absent" / "ledger.db")})

    assert oc._ticket_cost_legs(_RUN) == (None, None, None)


def test_r17_ticket_cost_legs_latest_ticket_wins_and_reads_a_nested_decomposition():
    _ticket(json.dumps({"estimated_cost": 0.02}), ticket_id="T-old", written_at=_AS_OF + "T10:00:00+00:00")
    _ticket(json.dumps({"cost_estimate": _COST_DETAIL}), ticket_id="T-new", written_at=_AS_OF + "T11:00:00+00:00")

    expected = ("T-new", pytest.approx(0.0010), pytest.approx(0.0004))

    assert oc._ticket_cost_legs(_RUN) == expected
    assert oc._ticket_cost_legs(_RUN, require_decomposition=True) == expected


def test_r17_ticket_cost_legs_same_second_takes_the_row_written_last():
    # ticket_id is a content hash: its order says nothing about write order.
    stamp = _AS_OF + "T10:00:00+00:00"
    _ticket(json.dumps({"estimated_cost": 0.02}), ticket_id="T-zzz", written_at=stamp)
    _ticket(json.dumps({"estimated_cost": 0.03}), ticket_id="T-aaa", written_at=stamp)

    assert oc._ticket_cost_legs(_RUN) == ("T-aaa", pytest.approx(0.03), 0.0)


def test_r17_ticket_cost_legs_scalar_total_only_where_a_decomposition_is_not_required():
    _ticket(json.dumps({"estimated_cost": 0.02}))

    # Legacy rows book the whole total under fees, slippage 0.0 ...
    assert oc._ticket_cost_legs(_RUN) == ("T-r17", pytest.approx(0.02), 0.0)
    # ... new outcomes refuse it: the total can contain expected carry.
    assert oc._ticket_cost_legs(_RUN, require_decomposition=True) == ("T-r17", None, None)


@pytest.mark.parametrize(
    "bad",
    [
        {**_COST_DETAIL, "entry_fee_bps": math.nan},
        {**_COST_DETAIL, "entry_slippage_bps": math.inf},
        {**_COST_DETAIL, "entry_fee_bps": -10.0, "exit_fee_bps": -10.0},  # a negative fee total
    ],
    ids=["nan", "inf", "negative-total"],
)
def test_r17_ticket_cost_legs_skips_a_decomposition_whose_side_total_is_unusable(bad):
    good = dict.fromkeys(_COST_DETAIL, 1.0)
    _ticket(json.dumps({"cost_detail": bad, "cost": good}))

    # The first decomposition that qualifies wins, whichever key carries it.
    assert oc._ticket_cost_legs(_RUN, require_decomposition=True) == (
        "T-r17", pytest.approx(0.0002), pytest.approx(0.0002),
    )


def test_r17_ticket_cost_legs_gate_is_on_each_side_total_not_on_each_field():
    # fees and slippage are booked as sums, and only the sums are gated: a
    # negative field whose side still sums to a non-negative number passes.
    _ticket(json.dumps({"cost_detail": {**_COST_DETAIL, "entry_fee_bps": -5.0, "exit_fee_bps": 5.0}}))

    assert oc._ticket_cost_legs(_RUN, require_decomposition=True) == ("T-r17", 0.0, pytest.approx(0.0004))


# --------------------------------------------------------------------------- #
# outcome_compute — the reference path (predictions with a frozen timing)
# --------------------------------------------------------------------------- #

_REFERENCE_NOW = "2026-09-05T12:00:00+00:00"  # horizon end + 5d 12h: inside the grace
_REFERENCE_LATE = "2026-09-07T12:00:00+00:00"  # horizon end + 8 days: terminal


def _reference_timing(**overrides: Any) -> dict[str, Any]:
    """A frozen daily-close reference for a perp formed 2026-08-25T12:00Z. The
    latest bar CLOSED at formation is 08-24 (final at 08-25T00:00Z)."""
    timing: dict[str, Any] = {
        "version": PREDICTION_TIME_VERSION,
        "prediction_formed_at": _AS_OF + "T12:00:00+00:00",
        "reference_source": tc.REFERENCE_SOURCE_PERP,
        "reference_price": 99.0,
        "reference_price_at": _AS_OF + "T00:00:00+00:00",
        "reference_available_at": _AS_OF + "T00:00:00+00:00",
        "reference_observed_at": _AS_OF + "T11:00:00+00:00",
        "reference_error": None,
    }
    timing.update(overrides)
    return tc.validate_timing(timing)


def _seed_timed(
    timing: dict[str, Any] | None,
    *,
    run_id: str = _RUN,
    analyst: str = "market",
    direction: str = "up",
    scope: str = ledger_models.SCOPE_CONTRACT,
    ticker: str = "BTCUSDT",
    asset_type: str = "crypto_perp",
    instrument_class: str | None = "pure_crypto_perp",
    horizon: int = 5,
) -> str:
    """One run + one prediction carrying ``timing`` (the reference path)."""
    register_run(run_id, ticker, asset_type, instrument_class, _AS_OF)
    (prediction_id,) = submit_predictions(
        run_id,
        analyst,
        ticker,
        scope,
        [{"horizon_days": horizon, "direction": direction, "prob_up": 0.6}],
        _AS_OF,
        timing=timing,
    )
    return prediction_id


def _worklist_item(prediction_id: str) -> dict[str, Any]:
    """The worklist row of one prediction, exactly as compute_outcomes sees it."""
    rows = [
        row for row in outcomes_mod.pending_predictions(_REFERENCE_NOW) if row["prediction_id"] == prediction_id
    ]
    (item,) = rows
    return item


def test_r17_reference_outcome_complete_row_is_measured_between_the_frozen_boundaries(monkeypatch):
    """The benchmark in one row: entry = the frozen reference (08-24 close),
    exit = the last daily close knowable by the fixed horizon end (08-29 bar,
    final at 08-30T00:00Z); funding integrates exactly inside those closes."""
    _Seams().install(monkeypatch)
    prediction_id = _seed_timed(_reference_timing())
    _write_ticket()

    report = oc.compute_outcomes(_REFERENCE_NOW)

    assert (report.considered, report.completed, report.incomplete, report.failed) == (1, 1, 0, 0)
    (record,) = outcomes_for_prediction(prediction_id)
    assert record.status == "complete"
    assert record.legs_missing == ()
    assert record.contract_price_return == pytest.approx(109.0 / 99.0 - 1.0)
    assert record.funding_pnl == pytest.approx(-15 * 0.0001)  # 15 settlements of 8h, long pays
    assert (record.fees, record.slippage) == (pytest.approx(0.0010), pytest.approx(0.0004))
    assert record.net_return == pytest.approx(109.0 / 99.0 - 1.0 - 15 * 0.0001 - 0.0010 - 0.0004)
    assert record.outcome_available_at == "2026-08-30T00:00:00+00:00"  # the exit bar's close instant
    context = record.scoring_context
    assert context["version"] == oc.OUTCOME_COMPUTE_VERSION
    assert "entry_grace" not in context  # the entry was the exact expected bar
    assert (context["window_start"], context["window_end"]) == (
        "2026-08-25T00:00:00+00:00", "2026-08-30T00:00:00+00:00",
    )
    assert context["horizon_end"] == "2026-08-30T12:00:00+00:00"
    assert context["retry_deadline"] == "2026-09-06T12:00:00+00:00"
    assert context["reference_price"] == 99.0
    assert context["reason"] == "strategy_view_return=+0.098110"


def test_r17_reference_outcome_before_the_horizon_end_is_pending_and_writes_nothing(monkeypatch):
    """The worklist only nominates rows whose horizon has elapsed, so this is
    the scorer's own re-check, driven directly: before the horizon end the row
    is pending, whatever the vendor already knows about later bars."""
    _Seams().install(monkeypatch)
    prediction_id = _seed_timed(_reference_timing())
    _write_ticket()

    detail = oc._compute_one(_worklist_item(prediction_id), "2026-08-30T11:59:00+00:00")

    assert detail == oc.OutcomeComputeDetail(
        prediction_id=prediction_id, run_id=_RUN, horizon_days=5,
        status="pending", legs_missing=(), net_return=None, outcome_id=None,
        reason="horizon_not_elapsed",
    )
    assert outcomes_for_prediction(prediction_id) == []


def test_r17_reference_outcome_never_scores_an_exit_bar_that_has_not_closed_yet(monkeypatch):
    """Belt and braces on the exit boundary: if the endpoint bar's close instant
    were still in the future, the row stays pending and nothing is written."""
    _Seams().install(monkeypatch)
    prediction_id = _seed_timed(_reference_timing())
    _write_ticket()
    real_close_moment = oc._bar_close_moment

    # A seam whose endpoint bar closes ten days later than the convention says
    # (2026-09-09T00:00Z: after `now`).
    monkeypatch.setattr(
        oc,
        "_bar_close_moment",
        lambda bar_date: real_close_moment(bar_date)
        + (timedelta(days=10) if bar_date == "2026-08-29" else timedelta(0)),
    )

    detail = oc._compute_one(_worklist_item(prediction_id), _REFERENCE_NOW)

    assert (detail.status, detail.reason, detail.legs_missing, detail.outcome_id) == (
        "pending", "horizon_end_bar_not_closed", (oc.LEG_CONTRACT_PRICE,), None,
    )
    assert outcomes_for_prediction(prediction_id) == []


@pytest.mark.parametrize("bad", [math.nan, 0.0], ids=["nan", "zero"])
def test_r17_reference_outcome_exit_price_that_is_not_a_finite_positive_number_fails_closed(
    monkeypatch, bad
):
    _Seams(perp={**_PERP_CLOSES, "2026-08-29": bad}).install(monkeypatch)
    prediction_id = _seed_timed(_reference_timing())
    _write_ticket()

    early = oc.compute_outcomes(_REFERENCE_NOW)

    assert (early.completed, early.incomplete, early.failed) == (0, 0, 0)
    (detail,) = early.details
    assert (detail.status, detail.reason, detail.legs_missing) == (
        "pending", "horizon_end_price_invalid", (oc.LEG_CONTRACT_PRICE,),
    )
    assert outcomes_for_prediction(prediction_id) == []

    late = oc.compute_outcomes(_REFERENCE_LATE)

    assert (late.completed, late.incomplete, late.failed) == (0, 1, 0)
    (record,) = outcomes_for_prediction(prediction_id)
    assert record.status == "incomplete"
    assert record.legs_missing == (oc.LEG_CONTRACT_PRICE,)
    assert (record.contract_price_return, record.net_return) == (None, None)
    assert record.scoring_context["reason"] == "horizon_end_price_invalid"


def test_r17_reference_outcome_reference_from_the_other_seam_is_terminal_incomplete(monkeypatch):
    _Seams().install(monkeypatch)
    prediction_id = _seed_timed(_reference_timing(reference_source=tc.REFERENCE_SOURCE_EQUITY))
    _write_ticket()

    report = oc.compute_outcomes(_REFERENCE_NOW)

    assert (report.completed, report.incomplete, report.failed) == (0, 1, 0)
    (record,) = outcomes_for_prediction(prediction_id)
    assert record.status == "incomplete"
    assert record.legs_missing == (oc.LEG_CONTRACT_PRICE,)
    assert record.scoring_context["reason"] == "reference_source_mismatch"


_MIDNIGHT_TIMING = {
    "prediction_formed_at": _AS_OF + "T00:03:00+00:00",
    "reference_price": 98.0,
    "reference_price_at": "2026-08-24T00:00:00+00:00",  # the 08-23 bar
    "reference_available_at": "2026-08-24T00:00:00+00:00",
    "reference_observed_at": "2026-08-24T23:58:00+00:00",  # the fetch straddled midnight
}
_GRACE_CLOSES = {**_PERP_CLOSES, "2026-08-22": 97.0, "2026-08-23": 98.0}


def test_r17_reference_outcome_accepts_one_older_bar_when_formation_crossed_utc_midnight(monkeypatch):
    """A reference fetch that straddled UTC midnight stored the latest CLOSED
    bar at REQUEST time, one bar-day older than formation implies. Exactly one
    bar-day of skew is admitted, only while formation sits within minutes of
    midnight, and the admission is disclosed in the scoring context."""
    _Seams(perp=_GRACE_CLOSES).install(monkeypatch)
    prediction_id = _seed_timed(_reference_timing(**_MIDNIGHT_TIMING))
    _write_ticket()

    report = oc.compute_outcomes("2026-09-05T00:03:00+00:00")

    assert (report.completed, report.incomplete, report.failed) == (1, 0, 0)
    (record,) = outcomes_for_prediction(prediction_id)
    # Entry = 08-23 close (98.0, final 08-24T00:00Z); horizon end 08-30T00:03Z
    # -> endpoint 08-29 (109.0). 18 settlements of 8h inside the two closes.
    assert record.contract_price_return == pytest.approx(109.0 / 98.0 - 1.0)
    assert record.funding_pnl == pytest.approx(-18 * 0.0001)
    context = record.scoring_context
    assert context["entry_grace"] == "midnight_cross"
    assert (context["window_start"], context["window_end"]) == (
        "2026-08-24T00:00:00+00:00", "2026-08-30T00:00:00+00:00",
    )


@pytest.mark.parametrize(
    ("timing", "why"),
    [
        (
            {
                "reference_price": 98.0,
                "reference_price_at": "2026-08-24T00:00:00+00:00",
                "reference_available_at": "2026-08-24T00:00:00+00:00",
                "reference_observed_at": _AS_OF + "T11:00:00+00:00",
            },
            "one bar older, but formed at noon: no midnight cross",
        ),
        (
            {
                "prediction_formed_at": _AS_OF + "T00:03:00+00:00",
                "reference_price": 97.0,
                "reference_price_at": "2026-08-23T00:00:00+00:00",  # the 08-22 bar
                "reference_available_at": "2026-08-23T00:00:00+00:00",
                "reference_observed_at": "2026-08-24T23:58:00+00:00",
            },
            "two bars older: more skew than the grace allows",
        ),
    ],
)
def test_r17_reference_outcome_reference_that_is_not_the_latest_closed_bar_is_terminal_incomplete(
    monkeypatch, timing, why
):
    _Seams(perp=_GRACE_CLOSES).install(monkeypatch)
    prediction_id = _seed_timed(_reference_timing(**timing))
    _write_ticket()

    report = oc.compute_outcomes(_REFERENCE_NOW)

    assert (report.completed, report.incomplete, report.failed) == (0, 1, 0), why
    (record,) = outcomes_for_prediction(prediction_id)
    assert record.scoring_context["reason"] == "reference_not_latest_closed_daily_bar"
    assert record.legs_missing == (oc.LEG_CONTRACT_PRICE,)


def test_r17_reference_outcome_without_a_timing_object_is_terminal_incomplete(monkeypatch):
    """``validate_timing(None)`` means legacy; reached with the reference scorer
    itself, it can only be an invalid prediction timing."""
    _Seams().install(monkeypatch)
    prediction_id = _seed_timed(_reference_timing())
    _write_ticket()
    item = {**_worklist_item(prediction_id), "timing": None}

    detail = oc._compute_reference_outcome(item, _REFERENCE_NOW)

    assert (detail.status, detail.reason, detail.legs_missing) == (
        "incomplete", "invalid_prediction_timing: missing prediction timing", (oc.LEG_CONTRACT_PRICE,),
    )
    (record,) = outcomes_for_prediction(prediction_id)
    assert record.status == "incomplete"


def _positioning_timing() -> dict[str, Any]:
    """POSITIONING carries no price reference: only the formation instant."""
    timing: dict[str, Any] = {
        "version": PREDICTION_TIME_VERSION,
        "prediction_formed_at": _AS_OF + "T12:00:00+00:00",
        "reference_source": None,
        "reference_price": None,
        "reference_price_at": None,
        "reference_available_at": None,
        "reference_observed_at": None,
        "reference_error": None,
    }
    return tc.validate_timing(timing)


def _seed_positioning(analyst: str) -> str:
    return _seed_timed(
        _positioning_timing(), analyst=analyst, direction="up", scope=ledger_models.SCOPE_POSITIONING
    )


def test_r17_reference_positioning_outcome_is_the_funding_sum_of_its_own_window(monkeypatch):
    """(formed, horizon end]: 15 settlements of 8h between 08-25T16:00Z and
    08-30T08:00Z. net_return IS the cumulative rate (positive = longs paid);
    no price and no cost leg applies."""
    _Seams().install(monkeypatch)
    prediction_id = _seed_positioning("positioning")

    report = oc.compute_outcomes(_REFERENCE_NOW)

    assert (report.completed, report.incomplete, report.failed) == (1, 0, 0)
    (record,) = outcomes_for_prediction(prediction_id)
    assert record.status == "complete"
    assert record.legs_missing == ()
    assert (record.funding_pnl, record.net_return) == (pytest.approx(0.0015), pytest.approx(0.0015))
    assert (record.contract_price_return, record.fees, record.slippage) == (None, None, None)
    assert record.outcome_available_at == "2026-08-30T12:00:00+00:00"
    context = record.scoring_context
    assert context["reason"] == "funding_window_complete"
    assert (context["window_start"], context["window_end"]) == (
        "2026-08-25T12:00:00+00:00", "2026-08-30T12:00:00+00:00",
    )


@pytest.mark.parametrize(
    ("seams", "reason"),
    [
        (_Seams(funding=""), "funding_window_missing"),
        (_Seams(funding_error=RuntimeError("funding history unavailable")), "funding_fetch_failed:RuntimeError"),
    ],
    ids=["empty-window", "fetch-raises"],
)
def test_r17_reference_positioning_without_a_provable_funding_window_is_not_scored(
    monkeypatch, seams, reason
):
    """No settlement grid, or a vendor exception: nothing is scored. Inside the
    grace the row stays pending; past it a terminal incomplete names the gap."""
    seams.install(monkeypatch)
    prediction_id = _seed_positioning("positioning")

    early = oc.compute_outcomes(_REFERENCE_NOW)

    assert (early.completed, early.incomplete, early.failed) == (0, 0, 0)
    (detail,) = early.details
    assert (detail.status, detail.reason, detail.legs_missing) == (
        "pending", reason, (oc.LEG_FUNDING_WINDOW,),
    )
    assert outcomes_for_prediction(prediction_id) == []

    late = oc.compute_outcomes(_REFERENCE_LATE)

    assert (late.completed, late.incomplete, late.failed) == (0, 1, 0)
    (record,) = outcomes_for_prediction(prediction_id)
    assert (record.status, record.legs_missing, record.net_return) == (
        "incomplete", (oc.LEG_FUNDING_WINDOW,), None,
    )
    assert record.scoring_context["reason"] == reason


def test_r17_reference_stock_perp_underlying_diagnostic_failure_does_not_block_the_label(
    monkeypatch, caplog
):
    """The contract is priced on the perp seam; the underlying equity leg is a
    diagnostic. When its vendor fails, the row is still scored complete, the
    gap is declared, and nothing is invented for basis."""
    _Seams(equity_error=RuntimeError("equity vendor down")).install(monkeypatch)
    prediction_id = _seed_timed(_reference_timing(), ticker="TSLAUSDT", instrument_class="stock_perp")
    _write_ticket()

    with caplog.at_level(logging.DEBUG, logger=oc.logger.name):
        report = oc.compute_outcomes(_REFERENCE_NOW)

    assert (report.completed, report.incomplete, report.failed) == (1, 0, 0)
    (record,) = outcomes_for_prediction(prediction_id)
    assert record.status == "complete"
    assert record.legs_missing == (oc.LEG_UNDERLYING,)
    assert (record.underlying_return, record.basis_return) == (None, None)
    assert record.contract_price_return == pytest.approx(109.0 / 99.0 - 1.0)
    assert record.net_return == pytest.approx(109.0 / 99.0 - 1.0 - 15 * 0.0001 - 0.0010 - 0.0004)
    [debug] = [
        record_.getMessage()
        for record_ in caplog.records
        if record_.name == oc.logger.name and record_.levelno == logging.DEBUG
    ]
    assert debug == "underlying diagnostic unavailable for TSLAUSDT"


# --------------------------------------------------------------------------- #
# scoreboard._scoring_version — the admission gate for a scored fact
# --------------------------------------------------------------------------- #

_FORMED = _AS_OF + "T12:00:00+00:00"
_HORIZON_END = "2026-08-30T12:00:00+00:00"
_ENTRY = _AS_OF + "T00:00:00+00:00"  # the 08-24 bar, final at 08-25T00:00Z
_EXIT = "2026-08-30T00:00:00+00:00"  # the 08-29 bar
_PRICE_RETURN = 109.0 / 99.0 - 1.0
_NET = _PRICE_RETURN - 15 * 0.0001 - 0.0010 - 0.0004


def _canonical(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _versioned_row(**overrides: Any) -> dict[str, Any]:
    """A complete CONTRACT outcome row joined to its prediction and run, in the
    shape ``_scored_rows`` yields: the perp baseline the gate admits."""
    context = {
        "version": OUTCOME_COMPUTE_VERSION,
        "prediction_formed_at": _FORMED,
        "horizon_end": _HORIZON_END,
        "window_start": _ENTRY,
        "window_end": _EXIT,
        "reference_source": tc.REFERENCE_SOURCE_PERP,
        "reference_price": 99.0,
        "reference_available_at": _ENTRY,
        "reference_observed_at": _AS_OF + "T11:00:00+00:00",
    }
    row: dict[str, Any] = {
        "prediction_id": "P-r17",
        "horizon_days": 5,
        "prediction_horizon_days": 5,
        "prediction_scope": ledger_models.SCOPE_CONTRACT,
        "instrument_class": "pure_crypto_perp",
        "asset_type": "crypto_perp",
        "contract_price_return": _PRICE_RETURN,
        "funding_pnl": -15 * 0.0001,
        "fees": 0.0010,
        "slippage": 0.0004,
        "net_return": _NET,
        "legs_missing": None,
        "outcome_available_at": _EXIT,
        "scoring_context": _canonical(context),
        "prediction_timing": _canonical(_reference_timing()),
    }
    timing = overrides.pop("timing", None)
    if timing is not None:
        row["prediction_timing"] = _canonical(timing)
    raw = overrides.pop("context", None)
    if raw is not None:
        context.update(raw)
        row["scoring_context"] = _canonical(context)
    row.update(overrides)
    return row


def _legacy_row(scope: str = ledger_models.SCOPE_CONTRACT, **overrides: Any) -> dict[str, Any]:
    """A complete row scored under ``legacy_daily_v1`` (no prediction timing)."""
    positioning = scope == ledger_models.SCOPE_POSITIONING
    context = {
        "version": LEGACY_OUTCOME_VERSION,
        "window_start": _ENTRY if positioning else "2026-08-26T00:00:00+00:00",
        "window_end": "2026-08-30T00:00:00+00:00" if positioning else "2026-08-31T00:00:00+00:00",
        "horizon_end": "2026-08-30T00:00:00+00:00",
        "reason": "strategy_view_return=+0.097100",
    }
    row: dict[str, Any] = {
        "prediction_id": "P-r17-legacy",
        "horizon_days": 5,
        "prediction_horizon_days": 5,
        "prediction_scope": scope,
        "instrument_class": "pure_crypto_perp",
        "asset_type": "crypto_perp",
        "contract_price_return": None if positioning else 0.10,
        "funding_pnl": 0.0015 if positioning else -0.0015,
        "fees": None if positioning else 0.0010,
        "slippage": None if positioning else 0.0004,
        "net_return": 0.0015 if positioning else 0.0971,
        "legs_missing": None,
        "outcome_available_at": "2026-08-29T16:00:00+00:00" if positioning else "2026-08-31T00:00:00+00:00",
        "scoring_context": _canonical(context),
        "prediction_timing": None,
    }
    raw = overrides.pop("context", None)
    if raw is not None:
        context.update(raw)
        row["scoring_context"] = _canonical(context)
    row.update(overrides)
    return row


def test_r17_scoring_version_admits_the_three_baseline_shapes():
    assert scoreboard._scoring_version(_versioned_row()) == OUTCOME_COMPUTE_VERSION
    assert scoreboard._scoring_version(_legacy_row()) == LEGACY_OUTCOME_VERSION
    assert scoreboard._scoring_version(_legacy_row(ledger_models.SCOPE_POSITIONING)) == LEGACY_OUTCOME_VERSION
    assert (
        scoreboard._scoring_version(
            _versioned_row(
                prediction_scope=ledger_models.SCOPE_POSITIONING,
                context={
                    "window_start": _FORMED,
                    "window_end": _HORIZON_END,
                    "reference_source": None,
                    "reference_price": None,
                    "reference_available_at": None,
                    "reference_observed_at": None,
                },
                timing=_positioning_timing(),
                contract_price_return=None,
                funding_pnl=0.0015,
                fees=None,
                slippage=None,
                net_return=0.0015,
                outcome_available_at=_HORIZON_END,
            )
        )
        == OUTCOME_COMPUTE_VERSION
    )


_NOON = _AS_OF + "T12:00:00+00:00"
_ERROR_SNAPSHOT = {
    "version": PREDICTION_TIME_VERSION,
    "prediction_formed_at": _FORMED,
    "reference_source": None,
    "reference_price": None,
    "reference_price_at": None,
    "reference_available_at": None,
    "reference_observed_at": None,
    "reference_error": "reference_fetch_failed:RuntimeError",
}


@pytest.mark.parametrize(
    ("row", "why"),
    [
        # --- legacy rows -------------------------------------------------
        (
            _legacy_row(
                ledger_models.SCOPE_POSITIONING, outcome_available_at="2026-08-25T00:00:00+00:00"
            ),
            "legacy POSITIONING: availability outside (window start, window end]",
        ),
        (
            _legacy_row(outcome_available_at="2026-08-30T12:00:00+00:00"),
            "legacy price label: available before the window closed",
        ),
        (
            _legacy_row(legs_missing='["funding"]'),
            "legacy: a mandatory leg is missing",
        ),
        # --- versioned rows ----------------------------------------------
        (_versioned_row(prediction_timing="null"), "versioned: no prediction timing object"),
        (_versioned_row(prediction_horizon_days=1), "versioned: horizon differs from the prediction's"),
        (_versioned_row(context={"horizon_end": "2026-08-31T12:00:00+00:00"}), "versioned: horizon end is not formation + horizon"),
        (_versioned_row(prediction_scope="MACRO"), "versioned: a scope the gate does not score"),
        (_versioned_row(timing=_ERROR_SNAPSHOT, context={"reference_price": None, "reference_available_at": None, "reference_observed_at": None}), "versioned: the prediction had no reference price"),
        (_versioned_row(context={"reference_price": True}), "versioned: the context's reference price is a bool"),
        (
            _versioned_row(
                prediction_scope=ledger_models.SCOPE_POSITIONING,
                context={
                    "window_start": _FORMED,
                    "window_end": _HORIZON_END,
                    "reference_source": None,
                    "reference_price": None,
                    "reference_available_at": None,
                    "reference_observed_at": None,
                },
                timing=_positioning_timing(),
                contract_price_return=None,
                fees=None,
                slippage=None,
                net_return=0.0015,
                outcome_available_at=_HORIZON_END,
                funding_pnl=None,
            ),
            "versioned POSITIONING: no finite funding label",
        ),
        (
            _versioned_row(asset_type="stock", instrument_class=None),
            "versioned: reference source is not the seam the row's instrument prices on",
        ),
        (
            _versioned_row(
                timing=_reference_timing(reference_price_at="2026-08-24T00:00:00+00:00", reference_available_at="2026-08-24T00:00:00+00:00"),
                context={"window_start": "2026-08-24T00:00:00+00:00", "reference_available_at": "2026-08-24T00:00:00+00:00"},
            ),
            "versioned: entry is one bar older than formation allows at noon",
        ),
        (
            _versioned_row(
                timing=_reference_timing(
                    prediction_formed_at=_AS_OF + "T23:00:00+00:00",
                ),
                context={
                    "prediction_formed_at": _AS_OF + "T23:00:00+00:00",
                    "horizon_end": "2026-08-30T23:00:00+00:00",
                    "window_end": "2026-08-30T12:00:00+00:00",
                },
                outcome_available_at="2026-08-30T12:00:00+00:00",
            ),
            "versioned: exit is not the fixed horizon's own daily close",
        ),
        (_versioned_row(net_return=_NET + 1e-3), "a stored score that no longer equals its legs"),
        (_versioned_row(fees=-0.0010, net_return=_NET + 0.002), "versioned: a negative cost"),
    ],
    ids=[
        "legacy-positioning-availability",
        "legacy-available-too-early",
        "legacy-missing-leg",
        "no-timing-object",
        "horizon-mismatch",
        "horizon-end-mismatch",
        "scope-unscored",
        "no-reference-price",
        "bool-reference-price",
        "positioning-no-funding",
        "source-mismatch",
        "entry-boundary",
        "exit-boundary",
        "net-tampered",
        "negative-cost",
    ],
)
def test_r17_scoring_version_rejects_a_fact_that_breaks_its_contract(row, why):
    assert scoreboard._scoring_version(row) is None, why


@pytest.mark.parametrize(
    "column", ["net_return", "contract_price_return", "funding_pnl", "fees", "slippage"]
)
@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf, None], ids=["nan", "inf", "-inf", "null"])
def test_r17_scoring_version_rejects_a_non_finite_leg(column, bad):
    # NaN / inf / NULL never enter an aggregate, whichever leg carries them.
    assert scoreboard._scoring_version(_versioned_row(**{column: bad})) is None


@pytest.mark.parametrize(
    "column", ["net_return", "contract_price_return", "funding_pnl", "fees", "slippage"]
)
@pytest.mark.parametrize("bad", ["0.1", True], ids=["str", "bool"])
def test_r17_scoring_version_rejects_a_leg_that_only_parses_as_a_number(column, bad):
    # _finite() accepts anything float() accepts, so the net-consistency check
    # is what stops a text / bool leg: the stored score no longer equals its legs.
    assert scoreboard._scoring_version(_versioned_row(**{column: bad})) is None


def test_r17_scoring_version_tolerates_only_float_noise_in_the_stored_score():
    base = scoreboard._scoring_version(_versioned_row())
    assert base == OUTCOME_COMPUTE_VERSION
    assert scoreboard._scoring_version(_versioned_row(net_return=_NET * (1 + 1e-10))) == OUTCOME_COMPUTE_VERSION
    assert scoreboard._scoring_version(_versioned_row(net_return=_NET * (1 + 1e-6))) is None
    # A diagnostic gap is declared, not fatal; a mandatory one is.
    assert scoreboard._scoring_version(_versioned_row(legs_missing='["underlying"]')) == OUTCOME_COMPUTE_VERSION
    assert scoreboard._scoring_version(_versioned_row(legs_missing='["funding"]')) is None
    assert scoreboard._scoring_version(_versioned_row(scoring_context="{not json")) is None
    assert scoreboard._scoring_version(_versioned_row(scoring_context="[1, 2]")) is None
    assert scoreboard._scoring_version(_versioned_row(scoring_context=None)) is None  # timed row, no metadata
    assert scoreboard._scoring_version(_legacy_row(scoring_context=None)) == LEGACY_OUTCOME_VERSION  # historical NULL
    assert scoreboard._scoring_version(_legacy_row(scoring_context="{broken")) is None
    # A new prediction cannot pass as legacy.
    assert scoreboard._scoring_version(_legacy_row(prediction_timing=_canonical(_reference_timing()))) is None


# --------------------------------------------------------------------------- #
# outcomes.write_outcome — argument validation before any write
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("bad", [["version"], "x", 5], ids=["list", "str", "int"])
def test_r17_write_outcome_rejects_a_scoring_context_that_is_not_an_object(bad):
    with pytest.raises(ValueError, match="scoring_context must be an object or None"):
        outcomes_mod.write_outcome("P-x", "run-x", 5, status="complete", scoring_context=bad)
    assert outcomes_mod.outcomes_for_prediction("P-x") == []


def test_r17_write_outcome_refuses_non_finite_metadata_before_touching_the_ledger():
    """The JSON audit trail is written with ``allow_nan=False``: a NaN anywhere
    in ``scoring_context`` is refused before the row exists."""
    prediction_id = _seed()

    with pytest.raises(ValueError, match="Out of range float values are not JSON compliant"):
        outcomes_mod.write_outcome(
            prediction_id, _RUN, 5, status="incomplete", scoring_context={"note": math.nan}
        )

    assert outcomes_for_prediction(prediction_id) == []


# --------------------------------------------------------------------------- #
# run_context.ensure_ledger_run_context
# --------------------------------------------------------------------------- #


def test_r17_ensure_ledger_run_context_binds_only_when_nothing_is_bound():
    reset_ledger_run_context()
    try:
        ensure_ledger_run_context("run-a", "BTCUSDT", "crypto_perp", "pure_crypto_perp", _AS_OF)
        bound = current_ledger_run_context()
        assert bound == LedgerRunContext(
            run_id="run-a", ticker="BTCUSDT", asset_type="crypto_perp",
            instrument_class="pure_crypto_perp", analysis_as_of=_AS_OF, regime_id=None,
        )

        # A retried pipeline keeps the original binding: the run row and
        # everything anchored to it are never rewritten mid-flight.
        ensure_ledger_run_context("run-b", "ETHUSDT", "crypto_perp", "2026-09-01", regime_id="R-1")
        assert current_ledger_run_context() is bound

        reset_ledger_run_context()
        ensure_ledger_run_context("run-b", "ETHUSDT", "crypto_perp", None, "2026-09-01", regime_id="R-1")
        assert current_ledger_run_context().run_id == "run-b"
        assert current_ledger_run_context().regime_id == "R-1"
    finally:
        reset_ledger_run_context()


# --------------------------------------------------------------------------- #
# sqlite — migrations
# --------------------------------------------------------------------------- #


def _schema_fingerprint() -> list[tuple[str, str, str]]:
    connection = ledger_sqlite.get_connection()
    return [
        tuple(row)
        for row in connection.execute(
            "SELECT type, name, sql FROM sqlite_master WHERE name NOT LIKE 'sqlite_%' ORDER BY name"
        ).fetchall()
    ]


def _schema_version() -> str:
    row = ledger_sqlite.get_connection().execute(
        "SELECT value FROM schema_meta WHERE key = 'schema_version'"
    ).fetchone()
    return row[0]


def test_r17_migrate_refuses_a_database_written_by_a_newer_binary():
    ledger_sqlite.get_connection()

    with ledger_sqlite.ledger_transaction() as cur:
        cur.execute("UPDATE schema_meta SET value = '99' WHERE key = 'schema_version'")
    ledger_sqlite.reset_ledger_state_for_test()  # a later launch finds it cold

    with pytest.raises(RuntimeError) as exc_info:
        ledger_sqlite.get_connection()

    assert str(exc_info.value) == (
        f"ledger DB schema version 99 is newer than this binary "
        f"(knows {ledger_sqlite._KNOWN_SCHEMA_VERSION}); upgrade yialpha before "
        f"touching {ledger_sqlite.ledger_db_path()}"
    )
    # Nothing was migrated: the file still says what it said.
    raw = ledger_sqlite.get_connection(readonly=True).execute(
        "SELECT value FROM schema_meta WHERE key = 'schema_version'"
    ).fetchone()
    assert raw[0] == "99"


def test_r17_migrate_leaves_an_up_to_date_database_untouched_when_it_is_opened_again():
    with ledger_sqlite.ledger_transaction() as cur:
        cur.execute("INSERT INTO schema_meta (key, value) VALUES ('r17_probe', 'kept')")
    before = _schema_fingerprint()

    ledger_sqlite._migrate(ledger_sqlite.get_connection())  # same connection, once more
    ledger_sqlite.reset_ledger_state_for_test()  # a second process: cold connection, migration re-run
    assert _schema_version() == str(ledger_sqlite._KNOWN_SCHEMA_VERSION)

    assert _schema_fingerprint() == before
    probe = ledger_sqlite.get_connection().execute(
        "SELECT value FROM schema_meta WHERE key = 'r17_probe'"
    ).fetchone()
    assert probe[0] == "kept"


def test_r17_migrate_reraises_an_alter_error_that_is_not_a_duplicate_column(tmp_path):
    """A v3 database whose ``predictions`` is a view (a foreign or corrupted
    file): the v4 ``ALTER TABLE`` fails for a reason other than "duplicate
    column name", and that error must surface instead of being swallowed."""
    path = Path(get_config()["ledger_db_path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    foreign = sqlite3.connect(path)
    foreign.executescript(
        "CREATE TABLE schema_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);"
        "INSERT INTO schema_meta VALUES ('schema_version', '3');"
        "CREATE VIEW predictions AS SELECT 1 AS x;"
    )
    foreign.close()
    ledger_sqlite.reset_ledger_state_for_test()

    with pytest.raises(sqlite3.OperationalError, match="view"):
        ledger_sqlite.get_connection()

    assert path.exists()  # nothing dropped, nothing rewritten


@pytest.mark.parametrize(
    ("helper", "table"),
    [
        (ledger_sqlite._alter_predictions_add_regime_id, "predictions"),
        (ledger_sqlite._alter_outcomes_add_regime_id, "outcomes"),
    ],
    ids=["predictions", "outcomes"],
)
def test_r17_alter_add_regime_id_swallows_only_the_duplicate_column_error(helper, table):
    conn = sqlite3.connect(":memory:", isolation_level=None)
    try:
        conn.execute(f"CREATE VIEW {table} AS SELECT 1 AS x")
        with pytest.raises(sqlite3.OperationalError, match="view"):
            helper(conn)

        conn.execute(f"DROP VIEW {table}")
        conn.execute(f"CREATE TABLE {table} (id TEXT PRIMARY KEY)")
        helper(conn)
        helper(conn)  # the column is already there: idempotent

        columns = [row[1] for row in conn.execute(f"PRAGMA table_info({table})")]
        assert columns == ["id", "regime_id"]
    finally:
        conn.close()


# --------------------------------------------------------------------------- #
# portfolio — degraded reads are labelled, never an "empty book"
# --------------------------------------------------------------------------- #


def _broken_connection(*, readonly: bool = False):
    raise sqlite3.DatabaseError("file is not a database")


def test_r17_open_positions_distinguishes_no_ledger_degraded_read_and_empty_book(tmp_path, caplog):
    set_config({"ledger_db_path": str(tmp_path / "absent" / "ledger.db")})
    assert portfolio.open_positions_with_status() == ([], True)  # no ledger: legally empty

    set_config({"ledger_db_path": str(tmp_path / "present" / "ledger.db")})
    ledger_sqlite.reset_ledger_state_for_test()
    portfolio.get_connection()  # now the file exists
    assert portfolio.open_positions_with_status() == ([], True)  # ledger, no positions

    with (
        caplog.at_level(logging.WARNING, logger=portfolio.logger.name),
        pytest.MonkeyPatch.context() as patch,
    ):
        patch.setattr(portfolio, "get_connection", _broken_connection)
        assert portfolio.open_positions_with_status() == ([], False)  # unreadable
        assert portfolio.open_positions() == []

    # One line per degraded read, each carrying its traceback.
    warnings = [record for record in caplog.records if record.name == portfolio.logger.name]
    assert [record.levelno for record in warnings] == [logging.WARNING, logging.WARNING]
    assert {record.getMessage() for record in warnings} == {
        "open_positions() read failed; degraded to [] — callers receive "
        "'book unreadable', not a legal empty book"
    }
    assert all(
        record.exc_info is not None and record.exc_info[0] is sqlite3.DatabaseError
        for record in warnings
    )


def test_r17_load_positions_input_labels_an_unreadable_ledger_beside_a_positions_file(tmp_path, monkeypatch):
    book = tmp_path / "book.json"
    book.write_text(
        json.dumps([{"symbol": "BTCUSDT", "side": "LONG", "signed_weight": 0.1}]), encoding="utf-8"
    )
    set_config({"portfolio_positions_file": str(book)})
    portfolio.get_connection()

    monkeypatch.setattr(portfolio, "get_connection", _broken_connection)
    rows, source = portfolio.load_positions_input()

    # The file still feeds the preview, and the label says the ledger half was not read.
    assert source == "ledger-unreadable+file:book.json"
    assert [(row["position_id"], row["symbol"], row["side"], row["signed_weight"]) for row in rows] == [
        ("file:BTCUSDT", "BTCUSDT", "LONG", 0.1)
    ]


def test_r17_snapshot_by_id_is_none_whenever_it_cannot_return_a_parsed_payload(monkeypatch):
    portfolio.get_connection()
    assert portfolio.snapshot_by_id("S-nobody") is None  # no such snapshot

    with portfolio.ledger_transaction() as cur:
        for snapshot_id, payload in (("S-good", '{"equity": 1}'), ("S-bad", "{not json"), ("S-null", "null")):
            cur.execute(
                "INSERT INTO portfolio_snapshots (snapshot_id, run_id, payload, created_at) "
                "VALUES (?, ?, ?, ?)",
                (snapshot_id, "run-1", payload, "2026-08-25T10:00:00+00:00"),
            )

    assert portfolio.snapshot_by_id("S-good") == {"equity": 1}
    assert portfolio.snapshot_by_id("S-bad") is None  # malformed JSON
    assert portfolio.snapshot_by_id("S-null") is None  # a JSON `null` payload

    monkeypatch.setattr(portfolio, "get_connection", _broken_connection)
    assert portfolio.snapshot_by_id("S-good") is None  # read failed


def test_r17_snapshot_by_id_without_a_ledger_file_is_none(tmp_path):
    set_config({"ledger_db_path": str(tmp_path / "absent" / "ledger.db")})

    assert portfolio.snapshot_by_id("S-nobody") is None


def test_r17_open_positions_keeps_a_row_whose_payload_does_not_parse():
    """The book is the positions table, not the payload JSON: a row whose
    payload is unparseable still counts as open, with an empty payload."""
    portfolio.get_connection()
    with portfolio.ledger_transaction() as cur:
        cur.executemany(
            "INSERT INTO positions (position_id, ticket_id, run_id, symbol, side, "
            "signed_weight, opened_at, closed_at, payload) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                ("N-ok", "T-1", "run-1", "BTCUSDT", "LONG", 0.1, "2026-08-25T10:00:00+00:00", None,
                 '{"instrument_class": "pure_crypto_perp"}'),
                ("N-bad", "T-1", "run-1", "ETHUSDT", "SHORT", -0.2, "2026-08-25T11:00:00+00:00", None,
                 "{not json"),
                ("N-closed", "T-1", "run-1", "SOLUSDT", "LONG", 0.3, "2026-08-25T09:00:00+00:00",
                 "2026-08-26T12:00:00+00:00", "{}"),
            ],
        )

    rows, ok = portfolio.open_positions_with_status()

    assert ok is True
    assert [
        (row["position_id"], row["symbol"], row["side"], row["signed_weight"], row["payload"])
        for row in rows
    ] == [
        ("N-ok", "BTCUSDT", "LONG", 0.1, {"instrument_class": "pure_crypto_perp"}),
        ("N-bad", "ETHUSDT", "SHORT", -0.2, {}),
    ]
