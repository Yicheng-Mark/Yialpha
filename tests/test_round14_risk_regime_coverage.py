"""Round-14 coverage top-up: risk numeric guards + regime serialization.

Fills the remaining uncovered branches in five modules (all hermetic —
synthetic frames, pure state machines, and per-test tmp ledger DBs; no
network, no shared fixtures beyond conftest's isolation):

* :mod:`yialpha.risk.atr_stop` — the degenerate-frame guards (None frame,
  len-less input, non-datetime index, unparseable Date cell, non-positive
  ATR, ATR column with no finite row, no finite close) and the ``atr_stop``
  wrapper's symbol path plus its ``mult`` validation;
* :mod:`yialpha.regime.state` — every ``render_regime_block`` disclosure
  line for the fields no existing fixture ever populated (the pure-crypto
  positioning legs and the whole stock-perp group);
* :mod:`yialpha.ledger.regime_store` — ``regime_by_id``'s fail-soft reads
  (absent DB, corrupt DB file, unparseable payload JSON);
* :mod:`yialpha.risk.cvar` / :mod:`yialpha.risk.breaker` — the tail-count
  floor clamp and constructor / exposure-cap input validation.

These are capital-preservation paths: every assertion pins the ERROR
behaviour on NaN/inf/extreme inputs, not just "it ran".
"""

from __future__ import annotations

import importlib
import logging

import pandas as pd
import pytest

from yialpha.ledger.regime_store import regime_by_id
from yialpha.ledger.sqlite import ledger_exists, ledger_transaction
from yialpha.regime.state import RegimeState, render_regime_block
from yialpha.risk.atr_stop import atr_stop, latest_atr_from_frame
from yialpha.risk.breaker import DrawdownBreaker
from yialpha.risk.cvar import cvar_position_multiplier, historical_cvar

_NAN = float("nan")


def _hist_frame(n: int = 40) -> pd.DataFrame:
    """Deterministic historical (2024) OHLCV frame with a real 14-period ATR."""
    closes = [100.0 + i * 0.1 for i in range(n)]
    return pd.DataFrame(
        {
            "Open": closes,
            "High": [c + 1.0 for c in closes],
            "Low": [c - 1.0 for c in closes],
            "Close": closes,
            "Volume": [10.0] * n,
        },
        index=pd.date_range("2024-01-01", periods=n, freq="D"),
    )


# --------------------------------------------------------------------------- #
# atr_stop: degenerate-frame guards
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_none_and_lenless_frames_rejected():
    """The wrapper must refuse unusable frames with a named error, not a
    TypeError from inside stockstats/pandas."""
    with pytest.raises(ValueError, match="frame is None"):
        latest_atr_from_frame(None)
    with pytest.raises(ValueError, match="frame has no length"):
        latest_atr_from_frame(42.5)  # a float carries no len()
    with pytest.raises(ValueError, match="frame is empty"):
        latest_atr_from_frame(pd.DataFrame())
    with pytest.raises(ValueError, match="missing a 'Close' column"):
        latest_atr_from_frame(pd.DataFrame({"Open": [1.0], "High": [2.0]}))


@pytest.mark.unit
def test_non_datetime_index_keeps_every_row():
    """A frame whose index does not parse as dates falls back to historical
    behaviour: NO trailing row is split off for a forming-bar guard that
    cannot even evaluate.

    Discriminator: exactly 15 rows (the ATR minimum) still yields an ATR —
    had the guard dropped the last row, 14 completed bars would return
    ``None`` (fewer than :data:`_ATR_MIN_COMPLETED_ROWS`).
    """
    closes = [100.0 + i * 0.1 for i in range(15)]
    frame = pd.DataFrame(
        {
            "Open": closes,
            "High": [c + 1.0 for c in closes],
            "Low": [c - 1.0 for c in closes],
            "Close": closes,
            "Volume": [10.0] * 15,
        },
        index=[f"row-{i}" for i in range(15)],  # non-datetime strings
    )
    close, atr = latest_atr_from_frame(frame)
    assert close == pytest.approx(closes[-1])
    assert atr is not None and atr > 0.0
    # The boundary pairing: one row fewer -> no ATR (None, not an error).
    _, atr_short = latest_atr_from_frame(frame.iloc[:-1])
    assert atr_short is None


@pytest.mark.unit
def test_unparseable_date_cell_keeps_every_row():
    """Same fallback through the ``Date``-column path: a garbage last Date
    cell makes the forming-bar timestamp unparseable, and the guard must
    degrade to "no split" (all 15 rows -> ATR exists) rather than silently
    dropping the row or crashing."""
    closes = [100.0 + i * 0.1 for i in range(15)]
    frame = pd.DataFrame(
        {
            "Date": ["2024-01-01"] * 14 + ["not-a-date"],
            "Open": closes,
            "High": [c + 1.0 for c in closes],
            "Low": [c - 1.0 for c in closes],
            "Close": closes,
            "Volume": [10.0] * 15,
        }
    )
    close, atr = latest_atr_from_frame(frame)
    assert close == pytest.approx(closes[-1])
    assert atr is not None and atr > 0.0


@pytest.mark.unit
def test_frozen_feed_zero_atr_rejected():
    """A frame of identical bars (delisted feed / frozen prints) computes a
    literal 0.0 ATR (stockstats' SMMA of all-zero true ranges). A 0 ATR is
    NOT a usable volatility: ``close - mult * 0 == close`` would place the
    stop AT the live price — an instant stop-out — so the guard refuses."""
    frame = pd.DataFrame(
        {
            "Open": [100.0] * 20,
            "High": [100.0] * 20,
            "Low": [100.0] * 20,
            "Close": [100.0] * 20,
            "Volume": [10.0] * 20,
        },
        index=pd.date_range("2024-01-01", periods=20, freq="D"),
    )
    with pytest.raises(ValueError, match="computed ATR is non-positive"):
        latest_atr_from_frame(frame)


@pytest.mark.unit
def test_atr_column_with_no_finite_row_rejected():
    """An ATR series that exists but carries no finite value (stockstats
    produced nothing usable) must raise, never return ``nan`` as a stop
    distance. Modelled by pre-seeding an all-NaN ``atr`` column: wrap
    returns an existing column as-is, so the walk-back finds no anchor."""
    frame = _hist_frame()
    frame["atr"] = [_NAN] * len(frame)
    with pytest.raises(ValueError, match="no finite ATR row"):
        latest_atr_from_frame(frame)
    # Fixture sanity: the identical frame WITHOUT the poisoned column has
    # a finite positive ATR — the guard is what fired, not the fixture.
    _, atr = latest_atr_from_frame(_hist_frame())
    assert atr is not None and atr > 0.0


@pytest.mark.unit
def test_no_finite_close_on_short_frame_rejected():
    """All-NaN closes on a sub-minimum frame: the early "too few completed
    bars" return still needs a close for the tuple — and there is none, so
    it raises (a NaN close must never leak out as a live price)."""
    frame = pd.DataFrame(
        {
            "Open": [100.0] * 5,
            "High": [101.0] * 5,
            "Low": [99.0] * 5,
            "Close": [_NAN] * 5,
            "Volume": [10.0] * 5,
        },
        index=pd.date_range("2024-01-01", periods=5, freq="D"),
    )
    with pytest.raises(ValueError, match="no finite close row"):
        latest_atr_from_frame(frame)


# --------------------------------------------------------------------------- #
# atr_stop: the wrapper (symbol path, mult validation)
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_atr_stop_without_enough_bars_refuses_to_place_a_stop():
    """Fewer than 15 completed bars -> no ATR -> NO stop. A newly listed
    instrument must not get a stop from a single-bar range (round-8 guard);
    the wrapper raises rather than inventing ``close - mult * None``."""
    short = _hist_frame().iloc[:10]  # 10 completed bars < the 15 minimum
    with pytest.raises(ValueError, match="ATR undefined"):
        atr_stop(short)
    # The tuple helper stays soft for the same frame: close valid, ATR None.
    close, atr = latest_atr_from_frame(short)
    assert atr is None
    assert close == pytest.approx(100.9)


@pytest.mark.unit
def test_symbol_path_requires_curr_date():
    with pytest.raises(ValueError, match="curr_date is required"):
        atr_stop("AAPL")  # symbol without a PIT date: refuse, do not guess "today"


@pytest.mark.unit
def test_symbol_path_routes_through_loader(monkeypatch):
    """``atr_stop`` on a symbol fetches via ``load_ohlcv`` (PIT-truncated)
    and computes ``close - mult * atr`` from that frame."""
    calls: list[tuple[str, str]] = []

    def fake_loader(symbol, curr_date):
        calls.append((symbol, curr_date))
        return _hist_frame()

    # importlib, not a dotted monkeypatch target: yialpha.risk.__init__
    # re-exports the atr_stop FUNCTION, shadowing the submodule attribute.
    monkeypatch.setattr(
        importlib.import_module("yialpha.risk.atr_stop"), "load_ohlcv", fake_loader
    )
    expected_close, expected_atr = latest_atr_from_frame(_hist_frame())

    stop = atr_stop("FAKE", "2024-03-01")
    assert calls == [("FAKE", "2024-03-01")]  # the PIT date threads through
    assert stop == pytest.approx(expected_close - 2.0 * expected_atr)
    tripled = atr_stop("FAKE", "2024-03-01", mult=3.0)
    assert tripled == pytest.approx(expected_close - 3.0 * expected_atr)


@pytest.mark.unit
def test_mult_must_be_finite_and_non_negative():
    frame = _hist_frame()
    with pytest.raises(ValueError, match="mult is not finite"):
        atr_stop(frame, mult=_NAN)
    with pytest.raises(ValueError, match="mult is not finite"):
        atr_stop(frame, mult=float("inf"))
    with pytest.raises(ValueError, match="mult must be non-negative"):
        atr_stop(frame, mult=-0.1)
    # Boundary: mult == 0 is valid (non-negative) — the stop sits AT the
    # close, mathematically the weakest allowed stop.
    close, atr = latest_atr_from_frame(frame)
    assert atr_stop(frame, mult=0.0) == pytest.approx(close)


# --------------------------------------------------------------------------- #
# regime/state.py: render_regime_block full disclosure
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_render_regime_block_full_state_disclosure():
    """Every disclosure line renders — a fully populated state (both field
    groups) must disclose ALL of its inputs; the block is the frozen,
    deterministic disclosure contract, pinned line by line."""
    state = RegimeState(
        trend_regime="range",
        realized_vol_pct=2.718,
        liquidity_regime="constrained",
        spread_depth_regime="wide",
        market_stress=True,
        confidence_components={"book": 1.0, "depth": 0.5},
        funding_pct=0.0012,
        oi_pct=91.6,
        lsr_crowding=1.337,
        taker_aggression=0.871,
        spot_perp_basis_bps=-63.4,
        liquidation_cascade_risk="elevated",
        underlying_trend="down",
        earnings_window="pre_earnings",
        sector_index_trend="down",
        session_state="continuous_24_7",
        overnight_gap_bps=12.34,
        index_mark_basis_bps=-7.86,
        contract_liquidity="thin",
        listing_age_days=1234,
        regime_id="Gabc123def456",
        regime_version="v5",
        analysis_as_of="2026-11-20",
        missing_inputs=("earnings_calendar",),
    )
    expected = [
        "### Regime State — version v5, as of 2026-11-20",
        "- **Trend**: range (close vs SMA50/SMA200, daily)",
        "- **Realized vol (20d)**: 2.718% daily",
        "- **Market stress**: true",
        "- **Liquidity**: constrained",
        "- **Spread regime**: wide (live book)",
        "- **Liquidation cascade risk**: elevated",
        "- **Funding (7d net)**: +0.1200% (longs pay when +)",
        "- **Open interest**: 92th pct of trailing 30d",
        "- **LSR crowding**: 1.34 global accounts (>1 = long-heavy)",
        "- **Taker aggression**: 0.87 buy/sell (>1 = aggressive buying)",
        "- **Spot-perp basis**: -63.4 bps",
        "- **Underlying trend**: down (equity close vs SMA50/SMA200)",
        "- **Overnight gap**: +12.3 bps",
        "- **Last-index basis**: -7.9 bps",
        "- **Contract liquidity**: thin (±50bps depth)",
        "- **Session state**: continuous_24_7 (24/7 continuous — "
        "klines-verified 2026-09-04)",
        "- **Listing age**: 1234 days",
        "- **Earnings window**: pre_earnings",
        "- **Sector index trend**: down",
        "- **Input coverage**: book=1.00, depth=0.50",
        "- **Missing inputs**: earnings_calendar",
        "- **Regime id**: Gabc123def456",
    ]
    assert render_regime_block(state).splitlines() == expected


@pytest.mark.unit
def test_render_regime_block_minimal_state_and_false_stress():
    """A state with only ``market_stress=False`` discloses exactly three
    lines: header (with ``n/a`` as-of), the greppable ``false`` stress
    line, and the ``n/a`` regime id. None/empty fields render NOTHING —
    an empty regime never masquerades as a populated one."""
    block = render_regime_block(RegimeState(market_stress=False, regime_version="v5"))
    assert block.splitlines() == [
        "### Regime State — version v5, as of n/a",
        "- **Market stress**: false",
        "- **Regime id**: n/a",
    ]


# --------------------------------------------------------------------------- #
# ledger/regime_store.py: regime_by_id fail-soft reads
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_regime_by_id_none_when_ledger_absent_or_id_empty():
    """conftest points ``ledger_db_path`` at a per-test tmp path that no
    write has created yet: the optional read degrades to ``None`` without
    ever creating the file (a read must not materialise a ledger)."""
    from pathlib import Path

    from yialpha.ledger.sqlite import ledger_db_path

    assert not ledger_exists()  # fixture sanity: this test created nothing
    assert regime_by_id("") is None  # falsy id short-circuits
    assert regime_by_id("G" + "0" * 12) is None
    assert not ledger_exists()  # the miss did NOT create the DB file
    assert not Path(ledger_db_path()).exists()


@pytest.mark.unit
def test_regime_by_id_corrupt_db_file_degrades_to_none(caplog):
    """A ledger path occupied by a non-sqlite file raises
    ``sqlite3.DatabaseError`` at read time; the optional read must swallow
    it with a visible WARNING and answer ``None`` — a corrupt advisory
    store must never abort the run reading it."""
    from pathlib import Path

    from yialpha.ledger.sqlite import ledger_db_path

    path = Path(ledger_db_path())
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"this is definitely not a sqlite database" * 8)
    with caplog.at_level(logging.WARNING, logger="yialpha.ledger.regime_store"):
        assert regime_by_id("G" + "a" * 12) is None
    assert any("read failed" in rec.message for rec in caplog.records)


@pytest.mark.unit
def test_regime_by_id_unparseable_payload_degrades_to_empty_dict():
    """A regimes row whose payload blob is not valid JSON still returns the
    scalar columns with ``payload == {}`` — the linkage columns survive a
    corrupted blob instead of poisoning the whole read."""
    with ledger_transaction() as cur:
        cur.execute(
            "INSERT INTO regimes (regime_id, payload, regime_version, "
            "analysis_as_of, ticker, instrument_class, computed_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                "G" + "b" * 12,
                "{not-json",
                "v5",
                "2026-09-03",
                "BTCUSDT",
                "pure_crypto_perp",
                "2026-09-03T00:00:00+00:00",
            ),
        )
    row = regime_by_id("G" + "b" * 12)
    assert row is not None  # found: the miss paths above are not what fired
    assert row["regime_version"] == "v5"
    assert row["ticker"] == "BTCUSDT"
    assert row["payload"] == {}


# --------------------------------------------------------------------------- #
# risk/cvar.py: tail-count clamps
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_historical_cvar_tail_floored_at_one_observation():
    """A confidence so high that ``n * (1 - confidence)`` rounds below 1
    still keeps the single WORST observation in the tail — one catastrophic
    day must never be averaged away by a razor-thin tail."""
    razor = 1.0 - 1e-12  # 2 * 1e-12 rounds to 0.0 at 9dp -> ceil 0 -> floor 1
    assert historical_cvar([0.05, -0.02], confidence=razor) == pytest.approx(-0.02)
    assert historical_cvar([0.05, 0.04, 0.03], confidence=razor) == pytest.approx(0.03)
    # Through the de-risk gate: the floored tail still trips the breach.
    assert cvar_position_multiplier([-0.9] * 30, confidence=razor) == 0.5


@pytest.mark.unit
def test_historical_cvar_tail_may_consume_whole_sample():
    """A confidence near 0 puts every observation in the tail (ceil of
    ``n * (1 - c)`` lands exactly on ``n`` — the size ceiling holds without
    overrunning the sort slice). Note: the ``tail_count > arr.size`` clamp
    in cvar.py is a defensive ceiling — with ``0 < c < 1`` the product is
    always < n, so it cannot be driven through the public API."""
    returns = [0.01, -0.02, 0.03]
    assert historical_cvar(returns, confidence=1e-9) == pytest.approx(
        sum(returns) / len(returns)
    )


# --------------------------------------------------------------------------- #
# risk/breaker.py: constructor validation + check_exposure equity guards
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_constructor_rejects_disordered_thresholds_and_bad_caps():
    with pytest.raises(ValueError, match="thresholds must satisfy"):
        DrawdownBreaker(warn_drawdown=0.2, no_new_drawdown=0.1)  # warn > no_new
    with pytest.raises(ValueError, match="thresholds must satisfy"):
        DrawdownBreaker(warn_drawdown=-0.01)  # below the 0 floor
    with pytest.raises(ValueError, match="exposure caps must be positive"):
        DrawdownBreaker(max_single_position=0.0)
    with pytest.raises(ValueError, match="exposure caps must be positive"):
        DrawdownBreaker(max_single_sector=-0.3)
    with pytest.raises(ValueError, match="cooldown_steps must be non-negative"):
        DrawdownBreaker(cooldown_steps=-1)
    # Boundary: an all-equal inclusive chain is valid, and the shared
    # threshold fires the deepest regime at exactly that drawdown.
    b = DrawdownBreaker(warn_drawdown=0.1, no_new_drawdown=0.1, hard_stop_drawdown=0.1)
    assert b.update(100.0).regime == "normal"
    assert b.update(90.0).regime == "hard_stop"  # 10% >= 10% == hard stop


@pytest.mark.unit
def test_check_exposure_rejects_unparseable_and_non_finite_equity():
    """Bad equity rejects with the SAME message on every path: a
    non-numeric string (float() ValueError), and — after a successful
    float() — zero, negative, NaN, and ±inf. A NaN equity divides the
    position fraction into NaN, which slips past every ``>`` cap
    comparison; it must be caught by the finiteness guard instead."""
    b = DrawdownBreaker()
    for bad in ("not-a-number", 0.0, -5.0, _NAN, float("inf"), float("-inf")):
        allowed, reason = b.check_exposure(10.0, equity_value=bad)
        assert allowed is False, bad
        assert reason == "equity_value must be a positive finite number", bad
    # The guard does not over-reject: a sane equity passes untouched.
    assert b.check_exposure(10.0, equity_value=100.0) == (True, "ok")
