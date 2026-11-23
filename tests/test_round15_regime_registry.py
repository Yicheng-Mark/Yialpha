"""Round-15 coverage hardening: regime math edges, registry PIT edges, config snapshot edges.

Targets the branches the round-15 report flagged as uncovered:

* ``yialpha/regime/compute.py`` — the frozen turbulence/trend classifiers
  (down/range buckets, zero-denominator vol filter, hand-pinned SMA and
  stdev values), the funding-CSV parser's malformed-row paths, the dormant
  ET machine's instant-weekend bucket, the depth-regime threshold table,
  the ``_market_stress`` trigger matrix (incl. the OI trigger), and the
  per-family fail-soft branches inside ``compute_regime_state`` (depth
  exception, positioning legs, spot/index same-date legs, registry
  onboard/underlying degradation).
* ``yialpha/instruments/registry.py`` — out-of-range ``onboardDate`` ms,
  SQLite failure on every read helper (fail-soft to registry_empty/None),
  blank/dash-only symbol candidates, and the ``_asof_bound`` normalization
  matrix (naive datetimes, unparseable input).
* ``yialpha/config_snapshot.py`` — the hermetic ``_git_commit`` failure
  paths (missing binary, nonzero exit, pinned success) and corrupt-JSON
  handling in ``load_last_snapshot`` / ``list_snapshots``.

All tests monkeypatch the same module-level seams the existing regime /
registry suites patch; no network, no real git subprocess, no sleeps.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import subprocess
from datetime import date, timedelta
from typing import Any

import pandas as pd
import pytest

import yialpha.config_snapshot as cs
import yialpha.instruments.registry as registry
import yialpha.regime.compute as rc
from yialpha.instruments.models import InstrumentRecord
from yialpha.instruments.registry import (
    SOURCE_REGISTRY_EMPTY,
    classify_perp,
    registry_available_at_values,
    reset_registry_cache_for_test,
    snapshot_instruments,
)
from yialpha.ledger.sqlite import get_connection

_TODAY = date.today().isoformat()
_PAST = (date.today() - timedelta(days=40)).isoformat()
_T1 = "2026-03-01T00:00:00+00:00"


@pytest.fixture(autouse=True)
def _round15_reset_registry_memo():
    """The classify_perp memo must never leak a record across tests."""
    reset_registry_cache_for_test()
    yield
    reset_registry_cache_for_test()


# --------------------------------------------------------------------------- #
# Shared regime fixtures (same seams test_regime_state.py patches)
# --------------------------------------------------------------------------- #
def _round15_frame(closes: list[float], *, opens: list[float] | None = None) -> pd.DataFrame:
    idx = pd.date_range("2025-01-01", periods=len(closes), freq="D")
    idx.name = "Date"
    return pd.DataFrame(
        {
            "Open": opens if opens is not None else [c * 0.999 for c in closes],
            "High": [abs(c) * 1.01 for c in closes],
            "Low": [abs(c) * 0.99 for c in closes],
            "Close": closes,
            "Volume": [1000.0] * len(closes),
        },
        index=idx,
    )


_UPTREND = [100.0 + 0.5 * i for i in range(260)]  # close > SMA50 > SMA200
# 210 closes at 100.0, 49 at 80.0, tail 85.0: SMA50 = 80.1 < 85.0 < SMA200 = 95.025
_RANGE_TREND = [100.0] * 210 + [80.0] * 49 + [85.0]
# Descending slope: tail 70.5 < SMA50 82.75 < SMA200 120.25 -> "down"
_DOWNTREND = [200.0 - 0.5 * i for i in range(260)]
# 259 descending closes then a NEGATIVE tail: trend "down", closes[-1] <= 0
_NEG_TAIL = [100.0 - 0.1 * i for i in range(259)] + [-1.0]

_FUNDING_CSV = (
    "fundingTime,fundingRate\n"
    "2026-08-26 00:00:00,0.00030\n"
    "2026-08-26 08:00:00,0.00020\n"
)
_EMPTY_FUNDING_CSV = "fundingTime,fundingRate\n"
_DEPTH_OK = {
    "status": "ok",
    "spread_bps": 1.5,
    "bands": {"50": {"bid_notional": 1_000_000.0, "ask_notional": 1_000_000.0}},
}
_OI_OK = {"status": "ok", "latest": 100.0, "percentile": 42.0, "chg_1d": 0.01}
_LSR_OK = {"status": "ok", "global_account": {"status": "ok", "latest": 1.30}}
_TAKER_OK = {"status": "ok", "latest": 1.10}


def _round15_klines_stub(
    perp: pd.DataFrame | None = None,
    *,
    spot: pd.DataFrame | None = None,
    index: pd.DataFrame | None = None,
    raise_spot: bool = False,
    raise_index: bool = False,
    calls: list[tuple[str, str]] | None = None,
):
    """Stub for ``binance_klines_frame`` routing by (venue, price_type)."""
    perp_frame = _round15_frame(_UPTREND) if perp is None else perp
    spot_frame = (perp_frame if spot is None else spot)
    index_frame = (perp_frame if index is None else index)

    def fake_klines(
        symbol, start_date, end_date, interval="1d", venue="binance_perp",
        price_type="last", closed_as_of=None,
    ):
        if calls is not None:
            calls.append((venue, price_type))
        if venue == "binance_spot":
            if raise_spot:
                raise RuntimeError("spot seam down")
            return spot_frame
        if price_type == "index":
            if raise_index:
                raise RuntimeError("index seam down")
            return index_frame
        return perp_frame

    return fake_klines


def _round15_mock_seams(
    monkeypatch, calls: list[tuple[str, str]] | None = None,
) -> None:
    monkeypatch.setattr(rc, "binance_klines_frame", _round15_klines_stub(calls=calls))
    monkeypatch.setattr(
        rc, "get_binance_funding_rate", lambda s, a, b: _FUNDING_CSV
    )
    monkeypatch.setattr(rc, "_fetch_open_interest", lambda s, d: dict(_OI_OK))
    monkeypatch.setattr(rc, "_fetch_lsr", lambda s, d: dict(_LSR_OK))
    monkeypatch.setattr(rc, "_fetch_taker", lambda s, d: dict(_TAKER_OK))
    monkeypatch.setattr(rc, "_fetch_depth_bands", lambda s: dict(_DEPTH_OK))


def _round15_mock_stock_registry(
    monkeypatch, *, onboard: str | None = "2024-06-01", underlying: str | None = "MU",
) -> None:
    monkeypatch.setattr(
        rc, "classify_perp",
        lambda symbol, as_of=None: InstrumentRecord(
            symbol="MUUSDT", instrument_class="stock_perp",
            classification_source="binance_exchangeinfo",
            classification_confidence=1.0,
            underlying_symbol=underlying, onboard_date=onboard,
        ),
    )
    monkeypatch.setattr(
        rc, "get_YFin_history_cached",
        lambda symbol, start, end: _round15_frame(_UPTREND),
    )


# --------------------------------------------------------------------------- #
# 1. Frozen classifier math — hand-pinned values
# --------------------------------------------------------------------------- #
@pytest.mark.unit
class TestRound15RegimeMath:
    def test_round15_trend_classifier_down_and_range_buckets(self):
        # "down": tail 70.5 strictly below BOTH SMAs (hand-computed).
        assert rc._sma(_DOWNTREND, 50) == pytest.approx(82.75)
        assert rc._sma(_DOWNTREND, 200) == pytest.approx(120.25)
        assert _DOWNTREND[-1] == pytest.approx(70.5)
        assert rc._trend_from_closes(_DOWNTREND) == "down"
        # "range": 85.0 sits BETWEEN SMA50 (80.1) and SMA200 (95.025).
        assert rc._sma(_RANGE_TREND, 50) == pytest.approx(80.1)
        assert rc._sma(_RANGE_TREND, 200) == pytest.approx(95.025)
        assert rc._trend_from_closes(_RANGE_TREND) == "range"
        # short history: no verdict, never a guess
        assert rc._sma([1.0, 2.0], 3) is None
        assert rc._trend_from_closes([100.0] * 199) is None

    def test_round15_realized_vol_pinned_and_zero_denominator_filter(self):
        # Constant +1% daily returns -> zero sample stdev (ddof=1).
        constant = [100.0 * 1.01**i for i in range(21)]
        assert rc._realized_vol_pct(constant) == pytest.approx(0.0, abs=1e-9)
        # Alternating +10%/-10% returns: stdev = sqrt(0.2/19) in percent.
        alternating = [100.0]
        for _ in range(20):
            alternating.append(alternating[-1] * (1.1 if len(alternating) % 2 else 0.9))
        assert rc._realized_vol_pct(alternating) == pytest.approx((0.2 / 19) ** 0.5 * 100)
        # A zero prior close drops that return; fewer than 20 surviving
        # returns -> None (unknown vol), not a silently skewed stdev.
        assert rc._realized_vol_pct([0.0] + [100.0] * 20) is None
        assert rc._realized_vol_pct([100.0] * 20) is None  # window gate

    def test_round15_parse_funding_csv_malformed_rows_and_sorting(self):
        text = (
            "fundingTime,fundingRate\n"
            "# a comment line the parser must skip\n"
            "\n"
            "2026-08-26 08:00:00,0.00020\n"       # out of order on purpose
            "not-a-date,0.00010\n"                # unparseable time -> skip
            "2026-08-26 00:00:00,oops\n"          # unparseable rate -> skip
            "2026-08-26 00:00:00\n"               # missing rate column -> skip
            "2026-08-25 16:00:00,0.00005\n"
        )
        settlements = rc._parse_funding_csv(text)
        assert [rate for _moment, rate in settlements] == [0.00005, 0.00020]
        assert [moment.isoformat() for moment, _rate in settlements] == [
            "2026-08-25T16:00:00+00:00", "2026-08-26T08:00:00+00:00",
        ]
        # A header the funding seam never ships: every row misses the keys.
        assert rc._parse_funding_csv("time,rate\n2026-01-01 00:00:00,0.01\n") == []
        assert rc._parse_funding_csv(_EMPTY_FUNDING_CSV) == []

    def test_round15_et_session_state_instant_weekend_closed(self):
        # Full instants on a weekend land "closed" via the LOCAL weekday
        # gate (the date-only approximation path is a different branch).
        assert rc._et_session_state("2026-09-05T14:30:00+00:00") == "closed"  # Saturday
        assert rc._et_session_state("2026-09-06T03:00:00") == "closed"  # naive Sunday
        # Sanity contrast: the same clock time on the following Monday is
        # inside the regular session (14:30 UTC = 10:30 EDT).
        assert rc._et_session_state("2026-09-07T14:30:00+00:00") == "regular"

    def test_round15_depth_regimes_threshold_table(self):
        # (spread_bps, bid50, ask50) -> (combined, spread, contract liquidity)
        def depth(spread: float, bid: float, ask: float) -> dict[str, Any]:
            return {
                "status": "ok", "spread_bps": spread,
                "bands": {"50": {"bid_notional": bid, "ask_notional": ask}},
            }

        assert rc._depth_regimes(depth(2.0, 1_000_000.0, 1_000_000.0)) == (
            "ample", "tight", "deep",
        )
        # tight spread but a thin book -> constrained
        assert rc._depth_regimes(depth(1.0, 200_000.0, 200_000.0)) == (
            "constrained", "tight", "thin",
        )
        # normal spread with a healthy book -> normal combined
        assert rc._depth_regimes(depth(5.0, 300_000.0, 300_000.0)) == (
            "normal", "normal", "normal",
        )
        # boundary: spread exactly 10.0 is still "normal", not "wide"
        assert rc._depth_regimes(depth(10.0, 1_500_000.0, 1_000_000.0)) == (
            "normal", "normal", "deep",
        )
        # wide spread dominates even a deep book -> constrained
        assert rc._depth_regimes(depth(12.0, 1_500_000.0, 1_500_000.0)) == (
            "constrained", "wide", "deep",
        )
        # boundary: spread exactly 2.0 is "tight"
        assert rc._depth_regimes(depth(2.0, 300_000.0, 300_000.0))[1] == "tight"
        # spread missing / non-numeric -> no verdict at all
        assert rc._depth_regimes({"bands": {}}) == (None, None, None)
        assert rc._depth_regimes({"spread_bps": "n/a", "bands": {}}) == (None, None, None)
        # no bands at all -> notional 0 -> thin + constrained
        assert rc._depth_regimes({"spread_bps": 1.0}) == ("constrained", "tight", "thin")
        # unparseable band notional -> fail-soft to no verdict
        bad = {"spread_bps": 1.0, "bands": {"50": {"bid_notional": "junk"}}}
        assert rc._depth_regimes(bad) == (None, None, None)
        # component gate: only status == ok counts
        assert rc._component_ok({"status": "ok"}) is True
        assert rc._component_ok({"status": "unavailable"}) is False
        assert rc._component_ok("not-a-dict") is False
        assert rc._component_ok(None) is False

    def test_round15_market_stress_trigger_matrix(self):
        # OI trigger: >= 90th percentile AND a strictly > +10% 1d build.
        fires = {
            "funding_pct": None, "spot_basis_bps": None,
            "oi_pct": 90.0, "oi_chg_1d": 0.11,
            "index_basis_bps": None, "overnight_gap_bps": None,
        }
        assert rc._market_stress("pure_crypto_perp", **fires) is True
        # The OI leg only ever ESCALATES (it appends nothing when calm), so
        # an OI-only record that does not fire reports stress UNKNOWN — None,
        # never a calm False. Funding/basis legs below their thresholds DO
        # append False and can report a calm composite.
        below_pct = {**fires, "oi_pct": 89.99}
        assert rc._market_stress("pure_crypto_perp", **below_pct) is None
        flat_build = {**fires, "oi_chg_1d": 0.10}  # exactly +10%: not a build
        assert rc._market_stress("pure_crypto_perp", **flat_build) is None
        # pure-crypto single triggers
        neg_funding = {**fires, "oi_pct": None, "oi_chg_1d": None,
                       "funding_pct": -0.01}
        assert rc._market_stress("pure_crypto_perp", **neg_funding) is True
        mild_basis = {**fires, "oi_pct": None, "oi_chg_1d": None,
                      "spot_basis_bps": 49.9}
        assert rc._market_stress("pure_crypto_perp", **mild_basis) is False
        # stock-perp triggers: |index basis| >= 50 bps, |gap| >= 100 bps
        assert rc._market_stress(
            "stock_perp", funding_pct=None, spot_basis_bps=None, oi_pct=None,
            oi_chg_1d=None, index_basis_bps=50.0, overnight_gap_bps=None,
        ) is True
        assert rc._market_stress(
            "stock_perp", funding_pct=None, spot_basis_bps=None, oi_pct=None,
            oi_chg_1d=None, index_basis_bps=None, overnight_gap_bps=-100.0,
        ) is True
        assert rc._market_stress(
            "stock_perp", funding_pct=None, spot_basis_bps=None, oi_pct=None,
            oi_chg_1d=None, index_basis_bps=None, overnight_gap_bps=99.9,
        ) is False
        # no trigger input computable anywhere -> stress UNKNOWN (None)
        no_inputs = {
            "funding_pct": None, "spot_basis_bps": None,
            "oi_pct": None, "oi_chg_1d": None,
            "index_basis_bps": None, "overnight_gap_bps": None,
        }
        assert rc._market_stress("pure_crypto_perp", **no_inputs) is None
        assert rc._market_stress("stock_perp", **no_inputs) is None
        assert rc._market_stress("unknown_perp", **no_inputs) is None
        # funding inputs are pure-crypto triggers only: a stock_perp run
        # carrying them alone still reports unknown stress.
        assert rc._market_stress("stock_perp", **neg_funding) is None


# --------------------------------------------------------------------------- #
# 2. compute_regime_state — fail-soft families and window contracts
# --------------------------------------------------------------------------- #
@pytest.mark.unit
class TestRound15RegimeAssembly:
    def test_round15_seam_window_contracts_pinned(self, monkeypatch):
        funding_calls: list[tuple[str, str]] = []
        kline_calls: list[tuple[str, str, str]] = []

        def fake_funding(symbol, start, end):
            funding_calls.append((start, end))
            return _FUNDING_CSV

        def fake_klines(
            symbol, start_date, end_date, interval="1d", venue="binance_perp",
            price_type="last", closed_as_of=None,
        ):
            kline_calls.append((start_date, end_date, venue))
            return _round15_frame(_UPTREND)

        monkeypatch.setattr(rc, "binance_klines_frame", fake_klines)
        monkeypatch.setattr(rc, "get_binance_funding_rate", fake_funding)
        monkeypatch.setattr(rc, "_fetch_open_interest", lambda s, d: dict(_OI_OK))
        monkeypatch.setattr(rc, "_fetch_lsr", lambda s, d: dict(_LSR_OK))
        monkeypatch.setattr(rc, "_fetch_taker", lambda s, d: dict(_TAKER_OK))
        state = rc.compute_regime_state(
            "BTCUSDT", "crypto_perp", "pure_crypto_perp", _PAST, end_date=_PAST,
        )
        assert state is not None  # fixture sanity
        past = date.fromisoformat(_PAST)
        # Frozen with REGIME_VERSION: 420-day kline lookback, END-INCLUSIVE
        # 7-day funding window (start 6 back), 8-day spot-basis window.
        perp_starts = {call[0] for call in kline_calls if call[2] == "binance_perp"}
        assert perp_starts == {(past - timedelta(days=420)).isoformat()}
        assert funding_calls == [
            ((past - timedelta(days=6)).isoformat(), _PAST)
        ]
        spot_starts = {call[0] for call in kline_calls if call[2] == "binance_spot"}
        assert spot_starts == {(past - timedelta(days=8)).isoformat()}

    def test_round15_live_depth_exception_disclosed_missing(self, monkeypatch):
        _round15_mock_seams(monkeypatch)

        def boom(symbol):
            raise RuntimeError("depth endpoint down")

        monkeypatch.setattr(rc, "_fetch_depth_bands", boom)
        state = rc.compute_regime_state(
            "BTCUSDT", "crypto_perp", "pure_crypto_perp", _TODAY, end_date=_TODAY,
        )
        assert state is not None  # fail-soft per family, not per run
        assert "depth_bands" in state.missing_inputs
        assert state.liquidity_regime is None
        assert state.spread_depth_regime is None
        assert state.confidence_components["depth"] == 0.0
        # stress composite still computable from the PIT legs -> low cascade
        assert state.market_stress is False
        assert state.liquidation_cascade_risk == "low"

    @pytest.mark.parametrize(
        ("seam", "kind", "payload", "missing_key", "field"),
        [
            (
                "get_binance_funding_rate", "return", _EMPTY_FUNDING_CSV,
                "funding_history", "funding_pct",
            ),
            (
                "get_binance_funding_rate", "raise", RuntimeError("csv seam down"),
                "funding_history", "funding_pct",
            ),
            (
                "_fetch_open_interest", "return", {"status": "unavailable"},
                "oi_history", "oi_pct",
            ),
            (
                "_fetch_open_interest", "raise", RuntimeError("oi seam down"),
                "oi_history", "oi_pct",
            ),
            (
                "_fetch_lsr", "return", {"status": "ok"},
                "lsr", "lsr_crowding",
            ),
            (
                "_fetch_lsr", "raise", RuntimeError("lsr seam down"),
                "lsr", "lsr_crowding",
            ),
            (
                "_fetch_taker", "return", {"status": "degraded"},
                "taker_flow", "taker_aggression",
            ),
            (
                "_fetch_taker", "raise", RuntimeError("taker seam down"),
                "taker_flow", "taker_aggression",
            ),
        ],
    )
    def test_round15_positioning_family_fail_soft_per_leg(
        self, monkeypatch, seam, kind, payload, missing_key, field,
    ):
        _round15_mock_seams(monkeypatch)
        if kind == "raise":
            exc = payload

            def boom(*args, **kwargs):
                raise exc

            monkeypatch.setattr(rc, seam, boom)
        else:
            monkeypatch.setattr(rc, seam, lambda *args, **kwargs: payload)
        state = rc.compute_regime_state(
            "BTCUSDT", "crypto_perp", "pure_crypto_perp", _PAST, end_date=_PAST,
        )
        assert state is not None
        assert missing_key in state.missing_inputs
        assert getattr(state, field) is None
        # the other four positioning legs still carry the family
        assert state.confidence_components["positioning"] == pytest.approx(0.8)

    def test_round15_spot_leg_exception_disclosed_missing(self, monkeypatch):
        _round15_mock_seams(monkeypatch)
        monkeypatch.setattr(
            rc, "binance_klines_frame", _round15_klines_stub(raise_spot=True)
        )
        state = rc.compute_regime_state(
            "BTCUSDT", "crypto_perp", "pure_crypto_perp", _PAST, end_date=_PAST,
        )
        assert state is not None
        assert state.spot_perp_basis_bps is None
        assert "spot_klines" in state.missing_inputs
        assert state.confidence_components["positioning"] == pytest.approx(0.8)

    def test_round15_non_positive_tail_skips_spot_leg(self, monkeypatch):
        calls: list[tuple[str, str]] = []
        _round15_mock_seams(monkeypatch)
        monkeypatch.setattr(
            rc, "binance_klines_frame",
            _round15_klines_stub(perp=_round15_frame(_NEG_TAIL), calls=calls),
        )
        state = rc.compute_regime_state(
            "BTCUSDT", "crypto_perp", "pure_crypto_perp", _PAST, end_date=_PAST,
        )
        assert state is not None
        assert state.trend_regime == "down"  # negative tail is still a verdict
        assert state.spot_perp_basis_bps is None
        assert "spot_klines" in state.missing_inputs
        # the spot venue is never queried for a non-positive denominator
        assert all(venue != "binance_spot" for venue, _price in calls)

    def test_round15_stock_perp_missing_onboard_date_disclosed(self, monkeypatch):
        _round15_mock_seams(monkeypatch)
        _round15_mock_stock_registry(monkeypatch, onboard="")
        state = rc.compute_regime_state(
            "MUUSDT", "crypto_perp", "stock_perp", _PAST, end_date=_PAST,
        )
        assert state is not None
        assert "onboard_date" in state.missing_inputs
        assert state.listing_age_days is None
        assert state.confidence_components["calendar"] == 0.0
        # the mapping itself survived: the equity leg is unaffected
        assert state.underlying_trend == "up"

    def test_round15_stock_perp_underlying_history_fail_soft(self, monkeypatch):
        _round15_mock_seams(monkeypatch)
        _round15_mock_stock_registry(monkeypatch)

        def boom(symbol, start, end):
            raise RuntimeError("equity vendor down")

        monkeypatch.setattr(rc, "get_YFin_history_cached", boom)
        state = rc.compute_regime_state(
            "MUUSDT", "crypto_perp", "stock_perp", _PAST, end_date=_PAST,
        )
        assert state is not None
        assert state.underlying_trend is None
        assert "underlying_history" in state.missing_inputs
        assert state.confidence_components["underlying"] == 0.0
        # the perp-side legs are untouched by the equity outage
        assert state.trend_regime == "up"
        assert state.index_mark_basis_bps is not None

    def test_round15_stock_perp_index_leg_exception_disclosed(self, monkeypatch):
        _round15_mock_seams(monkeypatch)
        _round15_mock_stock_registry(monkeypatch)
        monkeypatch.setattr(
            rc, "binance_klines_frame", _round15_klines_stub(raise_index=True)
        )
        state = rc.compute_regime_state(
            "MUUSDT", "crypto_perp", "stock_perp", _PAST, end_date=_PAST,
        )
        assert state is not None
        assert state.index_mark_basis_bps is None
        assert "index_klines" in state.missing_inputs
        assert state.underlying_trend == "up"  # other legs unaffected

    def test_round15_non_positive_tail_skips_index_leg(self, monkeypatch):
        calls: list[tuple[str, str]] = []
        _round15_mock_seams(monkeypatch)
        _round15_mock_stock_registry(monkeypatch)
        monkeypatch.setattr(
            rc, "binance_klines_frame",
            _round15_klines_stub(perp=_round15_frame(_NEG_TAIL), calls=calls),
        )
        state = rc.compute_regime_state(
            "MUUSDT", "crypto_perp", "stock_perp", _PAST, end_date=_PAST,
        )
        assert state is not None
        assert state.trend_regime == "down"
        assert state.index_mark_basis_bps is None
        assert "index_klines" in state.missing_inputs
        assert all(price != "index" for _venue, price in calls)


# --------------------------------------------------------------------------- #
# 3. registry — edge branches
# --------------------------------------------------------------------------- #
def _round15_sym(
    symbol: str = "XUSDT",
    underlying_type: str = "EQUITY",
    onboard_ms: int | float | None = None,
) -> dict[str, Any]:
    """Minimal exchangeInfo row (no filters: filter parsing is covered elsewhere)."""
    return {
        "symbol": symbol,
        "underlyingType": underlying_type,
        "status": "TRADING",
        "quoteAsset": "USDT",
        "marginAsset": "USDT",
        "onboardDate": onboard_ms,
    }


@pytest.mark.unit
class TestRound15RegistryEdges:
    def test_round15_onboard_date_from_ms_out_of_range_is_none(self):
        # A value far past datetime's range raises on Windows (OSError) and
        # elsewhere (OverflowError/ValueError) — every flavor degrades to None.
        assert registry._onboard_date_from_ms(10**18) is None
        assert registry._onboard_date_from_ms(1e30) is None
        assert registry._onboard_date_from_ms(0) is None  # non-positive gate
        assert registry._onboard_date_from_ms("junk") is None  # non-numeric
        # in-range value still converts (UTC date, ms epoch)
        assert registry._onboard_date_from_ms(1_770_000_000_000) == "2026-02-02"
        # and through the write side: the row persists with a NULL onboard
        assert snapshot_instruments(
            [_round15_sym("FARUSDT", onboard_ms=10**18)], _T1
        ) == 1
        row = (
            get_connection(readonly=True)
            .execute(
                "SELECT onboard_date FROM instrument_snapshots WHERE symbol = ?",
                ("FARUSDT",),
            )
            .fetchone()
        )
        assert row is not None and row["onboard_date"] is None

    def test_round15_symbol_candidates_blank_and_dash_only(self):
        assert registry._symbol_candidates("   ") == []
        assert registry._symbol_candidates(" - ") == []
        assert registry._symbol_candidates("MU-USDT") == ["MUUSDT"]
        assert registry._symbol_candidates("mu-usdt ") == ["MUUSDT"]
        assert registry._symbol_candidates("BTCUSDC") == ["BTCUSDC", "BTCUSDT"]
        assert registry._symbol_candidates("USDC") == ["USDC"]  # len 4: no twin
        # the public path: a blank ticker is registry_empty, never a crash
        record = classify_perp("  ")
        assert record.classification_source == SOURCE_REGISTRY_EMPTY
        assert record.instrument_class == "unknown_perp"

    def test_round15_asof_bound_normalization_matrix(self, caplog):
        assert registry._asof_bound(None) is None
        # bare date -> conservative midnight UTC at the START of the day
        assert registry._asof_bound("2026-03-01") == "2026-03-01T00:00:00+00:00"
        # naive datetimes (T or space separated) are ASSUMED UTC
        assert registry._asof_bound("2026-03-02T08:00:00") == (
            "2026-03-02T08:00:00+00:00"
        )
        assert registry._asof_bound("2026-03-02 08:00:00") == (
            "2026-03-02T08:00:00+00:00"
        )
        # aware datetimes are coerced to UTC seconds-precision ISO
        assert registry._asof_bound("2026-03-02T08:00:00+02:00") == (
            "2026-03-02T06:00:00+00:00"
        )
        # unparseable -> UNBOUNDED (None) with a warning, never garbage bytes
        with caplog.at_level(logging.WARNING, logger="yialpha.instruments.registry"):
            assert registry._asof_bound("not-a-date") is None
        assert any(
            "unparseable as_of" in rec.message for rec in caplog.records
        ), [rec.message for rec in caplog.records]

    def test_round15_classify_perp_sqlite_failure_is_registry_empty(
        self, monkeypatch, caplog,
    ):
        snapshot_instruments([_round15_sym("MUUSDT")], _T1)  # DB exists now

        def broken_conn(*args: Any, **kwargs: Any):
            raise sqlite3.OperationalError("ledger locked")

        monkeypatch.setattr(registry, "get_connection", broken_conn)
        with caplog.at_level(logging.WARNING, logger="yialpha.instruments.registry"):
            record = classify_perp("MUUSDT")
        # both the memo probe (_max_available_at) and the row query fail
        # soft: no evidence, never an exception into the run.
        assert record.classification_source == SOURCE_REGISTRY_EMPTY
        assert record.instrument_class == "unknown_perp"
        assert record.classification_confidence == 0.0
        assert any(
            "registry read failed" in rec.message for rec in caplog.records
        ), [rec.message for rec in caplog.records]

    def test_round15_registry_available_at_values_sqlite_failure(self, monkeypatch):
        snapshot_instruments([_round15_sym("MUUSDT")], _T1)  # DB exists now

        def broken_conn(*args: Any, **kwargs: Any):
            raise sqlite3.OperationalError("ledger locked")

        monkeypatch.setattr(registry, "get_connection", broken_conn)
        assert registry_available_at_values("MUUSDT") == []


# --------------------------------------------------------------------------- #
# 4. config_snapshot — hermetic git + corrupt-file edges
# --------------------------------------------------------------------------- #
@pytest.mark.unit
class TestRound15ConfigSnapshotEdges:
    def test_round15_git_commit_unavailable_paths_recorded_as_none(
        self, monkeypatch, tmp_path,
    ):
        # git binary missing -> FileNotFoundError swallowed -> None
        def missing_git(cmd, **kwargs):
            raise FileNotFoundError("git not installed")

        monkeypatch.setattr(cs.subprocess, "run", missing_git)
        path = cs.record_config_snapshot(
            {"model": "a"}, reason="no git binary", history_dir=tmp_path / "d1",
        )
        assert json.loads(path.read_text(encoding="utf-8"))["git_commit"] is None

        # nonzero exit (not a repo) -> falls through to None
        monkeypatch.setattr(
            cs.subprocess, "run",
            lambda cmd, **kwargs: subprocess.CompletedProcess(cmd, 128),
        )
        path = cs.record_config_snapshot(
            {"model": "a"}, reason="not a repo", history_dir=tmp_path / "d2",
        )
        assert json.loads(path.read_text(encoding="utf-8"))["git_commit"] is None

        # hermetic success path: stdout stripped into the snapshot
        monkeypatch.setattr(
            cs.subprocess, "run",
            lambda cmd, **kwargs: subprocess.CompletedProcess(
                cmd, 0, stdout="abc1234\n",
            ),
        )
        path = cs.record_config_snapshot(
            {"model": "a"}, reason="clean repo", history_dir=tmp_path / "d3",
        )
        assert json.loads(path.read_text(encoding="utf-8"))["git_commit"] == "abc1234"

    def test_round15_load_last_snapshot_corrupt_latest_returns_none(
        self, monkeypatch, tmp_path, caplog,
    ):
        monkeypatch.setattr(
            cs.subprocess, "run",
            lambda cmd, **kwargs: subprocess.CompletedProcess(cmd, 1),
        )
        cs.record_config_snapshot({"model": "v1"}, reason="alpha", history_dir=tmp_path)
        cs.record_config_snapshot({"model": "v2"}, reason="beta", history_dir=tmp_path)
        # a corrupt file sorting AFTER every real snapshot (timestamp prefix)
        corrupt = tmp_path / "config_99991231_235959_999999_c0rrupt0bad.json"
        corrupt.write_text("{ not valid json", encoding="utf-8")
        with caplog.at_level(logging.WARNING, logger="yialpha.config_snapshot"):
            assert cs.load_last_snapshot(history_dir=tmp_path) is None
        assert any(
            "Could not read last config snapshot" in rec.message
            for rec in caplog.records
        ), [rec.message for rec in caplog.records]
        # no usable last snapshot -> the diff reports no drift, not a crash
        assert cs.diff_against_last_snapshot(
            {"model": "changed"}, history_dir=tmp_path,
        ) == {}

    def test_round15_list_snapshots_skips_unreadable_files(
        self, monkeypatch, tmp_path,
    ):
        monkeypatch.setattr(
            cs.subprocess, "run",
            lambda cmd, **kwargs: subprocess.CompletedProcess(cmd, 1),
        )
        cs.record_config_snapshot({"model": "v1"}, reason="alpha", history_dir=tmp_path)
        (tmp_path / "config_88880101_000000_000000_deadbeefcafe.json").write_text(
            "[ broken", encoding="utf-8",
        )
        cs.record_config_snapshot({"model": "v2"}, reason="beta", history_dir=tmp_path)
        summaries = cs.list_snapshots(history_dir=tmp_path)
        assert [item["reason"] for item in summaries] == ["alpha", "beta"]
        assert all("deadbeefcafe" not in item["file"] for item in summaries)
        # the unavailable-commit sentinel persisted as None and read back as None
        assert all(item["git_commit"] is None for item in summaries)
