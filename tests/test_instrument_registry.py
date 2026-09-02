"""Persistent point-in-time instrument registry (V2.1 record stage).

Pins the yialpha.instruments package and its two integration seams:

  * the exchangeInfo classification truth table (unsupported contract types
    — non-US equity, commodity, index, premarket — land in ``unknown_perp``
    with a machine-readable reason instead of silently masquerading as US
    stocks or pure crypto),
  * snapshot persistence (append-only PK dedupe, filter parsing, ms-epoch
    onboard conversion, fail-soft),
  * point-in-time ``classify_perp`` (as-of bounds, listed_asof transitions,
    bare-date conservatism, memo auto-invalidation),
  * the routing flag matrix (flag OFF = legacy bytes; flag ON = registry
    evidence beats warm/seed, pure-by-default becomes ``unknown_perp``),
  * session-calendar constants (canonical home + routing re-export) with
    pinned session counts,
  * the exchangeInfo warm hook appending registry rows from the SAME payload
    (zero extra HTTP; off by default; warm behavior unchanged),
  * corporate-action validation (V2.4 skeleton).

Hermetic: the ledger DB is conftest's per-test tmp file, the HTTP seam is
monkeypatched (``_http_get``), and the warm caches + registry memo are
dropped around every test.
"""

from __future__ import annotations

import contextlib
import json
import logging
import sqlite3
from datetime import UTC, datetime
from typing import Any

import pytest

import yialpha.dataflows.binance as bnb
import yialpha.graph.routing as routing
import yialpha.instruments.registry as registry
from yialpha.dataflows.config import set_config
from yialpha.instruments.corporate_actions import (
    CorporateActionRecord,
    render_corporate_action_disclosure,
    validate_corporate_action,
)
from yialpha.instruments.models import InstrumentRecord, classify_exchangeinfo_row
from yialpha.instruments.registry import (
    SOURCE_EXCHANGEINFO,
    SOURCE_REGISTRY_EMPTY,
    classify_perp,
    registry_available_at_values,
    reset_registry_cache_for_test,
    snapshot_instruments,
)
from yialpha.instruments.sessions import (
    SESSION_BINANCE_TRADFI,
    SESSION_CONTINUOUS,
    SESSION_EXCHANGE,
    trading_sessions_between,
)
from yialpha.ledger.sqlite import get_connection

# 2026-02-02T05:20:00Z — same fixture epoch test_instrument_routing.py pins.
_ONBOARD_MS = 1_770_000_000_000
_ONBOARD_DATE = "2026-02-02"
_T1 = "2026-03-01T00:00:00+00:00"
_T2 = "2026-04-01T00:00:00+00:00"


def _ms_of(iso_date: str) -> int:
    y, m, d = (int(part) for part in iso_date.split("-"))
    return int(datetime(y, m, d, tzinfo=UTC).timestamp() * 1000)


def _sym(
    symbol: str = "MUUSDT",
    underlying_type: str | None = "EQUITY",
    status: str = "TRADING",
    quote: str = "USDT",
    onboard_ms: int | float | None = _ONBOARD_MS,
    base: str | None = None,
    filters: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """One exchangeInfo symbol row shaped like the fapi /exchangeInfo payload."""
    row: dict[str, Any] = {
        "symbol": symbol,
        "underlyingType": underlying_type,
        "status": status,
        "baseAsset": base if base is not None else symbol[: -len(quote)],
        "quoteAsset": quote,
        "marginAsset": quote,
        "onboardDate": onboard_ms,
        "filters": filters
        if filters is not None
        else [
            {
                "filterType": "PRICE_FILTER",
                "tickSize": "0.0010",
                "stepSize": "1.00",
                "minQty": "0.10",
            },
            {"filterType": "MIN_NOTIONAL", "notional": 5.0},
        ],
    }
    return row


@pytest.fixture(autouse=True)
def _isolated_warm_and_registry():
    """Drop the process warm caches and the registry memo around each test."""
    bnb.refresh_equity_perp_bases()
    reset_registry_cache_for_test()
    yield
    bnb.refresh_equity_perp_bases()
    reset_registry_cache_for_test()


def _registry_row(symbol: str) -> sqlite3.Row | None:
    """Raw ledger read of the newest registry row for ``symbol`` (tests)."""
    conn = get_connection(readonly=True)
    return conn.execute(
        "SELECT symbol, snapshot_available_at, classification_source, "
        "instrument_class, unsupported_reason, underlying_type, "
        "underlying_symbol, quote_asset, margin_asset, onboard_date, "
        "status, session_calendar, tick_size, step_size, min_qty, "
        "min_notional, leverage_bracket_version, classification_confidence, "
        "raw_payload, created_at "
        "FROM instrument_snapshots WHERE symbol = ? "
        "ORDER BY snapshot_available_at DESC LIMIT 1",
        (symbol,),
    ).fetchone()


# ---- classification truth table ----------------------------------------------


@pytest.mark.unit
class TestClassifyExchangeinfoRow:
    def test_equity_is_stock_perp(self):
        assert classify_exchangeinfo_row({"underlyingType": "EQUITY"}) == (
            "stock_perp", None,
        )

    def test_coin_is_pure_crypto(self):
        assert classify_exchangeinfo_row({"underlyingType": "COIN"}) == (
            "pure_crypto_perp", None,
        )

    @pytest.mark.parametrize(
        "underlying_type",
        ["HK_EQUITY", "KR_EQUITY", "CN_EQUITY", "COMMODITY", "INDEX", "PREMARKET"],
    )
    def test_unsupported_contract_types_are_unknown_with_reason(
        self, underlying_type: str,
    ):
        klass, reason = classify_exchangeinfo_row({"underlyingType": underlying_type})
        assert klass == "unknown_perp"
        assert reason == f"unsupported-contract-type:{underlying_type}"

    def test_missing_underlying_type(self):
        assert classify_exchangeinfo_row({}) == ("unknown_perp", "underlying_type_missing")

    def test_unknown_underlying_type_value(self):
        assert classify_exchangeinfo_row({"underlyingType": "EXOTIC"}) == (
            "unknown_perp", "underlying_type_missing",
        )


@pytest.mark.unit
class TestInstrumentRecordDisclosure:
    def test_disclosure_is_one_line_with_the_essentials(self):
        record = InstrumentRecord(
            symbol="MUUSDT",
            instrument_class="stock_perp",
            classification_source=SOURCE_EXCHANGEINFO,
            classification_confidence=1.0,
            underlying_symbol="MU",
            onboard_date=_ONBOARD_DATE,
            status="TRADING",
            snapshot_available_at=_T1,
        )
        line = record.as_disclosure()
        assert "\n" not in line
        for fragment in (
            "MUUSDT", "stock_perp", "binance_exchangeinfo", "confidence=1.00",
            "underlying=MU", f"onboard={_ONBOARD_DATE}", "status=TRADING",
        ):
            assert fragment in line, fragment

    def test_disclosure_carries_unsupported_reason(self):
        record = InstrumentRecord(
            symbol="TENCENTUSDT",
            instrument_class="unknown_perp",
            classification_source=SOURCE_EXCHANGEINFO,
            classification_confidence=1.0,
            unsupported_reason="unsupported-contract-type:HK_EQUITY",
        )
        assert "reason=unsupported-contract-type:HK_EQUITY" in record.as_disclosure()


# ---- snapshot persistence ------------------------------------------------------


@pytest.mark.unit
class TestSnapshotInstruments:
    def test_persists_every_symbol_row_with_parsed_fields(self):
        rows = [
            _sym("MUUSDT", "EQUITY"),
            _sym("BTCUSDT", "COIN"),
            _sym("TENCENTUSDT", "HK_EQUITY"),
        ]
        assert snapshot_instruments(rows, _T1) == 3

        mu = _registry_row("MUUSDT")
        assert mu is not None
        assert mu["classification_source"] == SOURCE_EXCHANGEINFO
        assert mu["classification_confidence"] == 1.0
        assert mu["instrument_class"] == "stock_perp"
        assert mu["underlying_type"] == "EQUITY"
        assert mu["underlying_symbol"] == "MU"
        assert mu["quote_asset"] == "USDT"
        assert mu["margin_asset"] == "USDT"
        assert mu["onboard_date"] == _ONBOARD_DATE
        assert mu["status"] == "TRADING"
        assert mu["session_calendar"] == SESSION_BINANCE_TRADFI
        assert mu["tick_size"] == pytest.approx(0.001)
        assert mu["step_size"] == pytest.approx(1.0)
        assert mu["min_qty"] == pytest.approx(0.10)
        assert mu["min_notional"] == pytest.approx(5.0)
        assert mu["leverage_bracket_version"] is None
        assert json.loads(mu["raw_payload"])["symbol"] == "MUUSDT"
        assert mu["created_at"]

        btc = _registry_row("BTCUSDT")
        assert btc is not None
        assert btc["instrument_class"] == "pure_crypto_perp"
        assert btc["underlying_symbol"] is None
        assert btc["session_calendar"] == SESSION_CONTINUOUS
        assert btc["unsupported_reason"] is None

        tencent = _registry_row("TENCENTUSDT")
        assert tencent is not None
        assert tencent["instrument_class"] == "unknown_perp"
        assert tencent["unsupported_reason"] == "unsupported-contract-type:HK_EQUITY"
        assert tencent["session_calendar"] is None

    def test_primary_key_dedupes_same_available_at(self):
        rows = [_sym("MUUSDT", "EQUITY")]
        assert snapshot_instruments(rows, _T1) == 1
        assert snapshot_instruments(rows, _T1) == 0  # INSERT OR IGNORE dedupe
        assert registry_available_at_values("MUUSDT") == [_T1]
        assert snapshot_instruments(rows, _T2) == 1  # append-only new snapshot
        assert registry_available_at_values("MUUSDT") == [_T1, _T2]

    def test_min_notional_accepts_legacy_minNotional_key(self):
        rows = [
            _sym(
                "AAPLUSDT", "EQUITY",
                filters=[{"filterType": "MIN_NOTIONAL", "minNotional": "20"}],
            )
        ]
        assert snapshot_instruments(rows, _T1) == 1
        aapl = _registry_row("AAPLUSDT")
        assert aapl is not None
        assert aapl["min_notional"] == pytest.approx(20.0)
        assert aapl["tick_size"] is None  # no PRICE_FILTER in this payload

    def test_underlying_symbol_falls_back_to_suffix_strip(self):
        row = _sym("BRKBUSDT", "EQUITY", base=None)
        del row["baseAsset"]  # symbol minus quote suffix must take over
        assert snapshot_instruments([row], _T1) == 1
        brkb = _registry_row("BRKBUSDT")
        assert brkb is not None
        assert brkb["underlying_symbol"] == "BRKB"

    def test_non_positive_or_garbage_onboard_date_is_none(self):
        assert snapshot_instruments(
            [_sym("AUSDT", "EQUITY", onboard_ms=0), _sym("BUSDT", "EQUITY", onboard_ms="x")],
            _T1,
        ) == 2
        for symbol in ("AUSDT", "BUSDT"):
            persisted = _registry_row(symbol)
            assert persisted is not None
            assert persisted["onboard_date"] is None

    def test_skips_non_dict_and_symbol_less_rows(self):
        assert snapshot_instruments(["junk", {"status": "TRADING"}], _T1) == 0

    def test_sqlite_failure_is_fail_soft(self, monkeypatch):
        @contextlib.contextmanager
        def _failing_transaction(*_args: Any, **_kwargs: Any):
            raise sqlite3.OperationalError("ledger locked")
            yield  # pragma: no cover

        monkeypatch.setattr(registry, "ledger_transaction", _failing_transaction)
        assert snapshot_instruments([_sym()], _T1) == 0  # never raises


# ---- point-in-time classify_perp ----------------------------------------------


@pytest.mark.unit
class TestClassifyPerpPIT:
    def test_latest_snapshot_without_asof(self):
        snapshot_instruments(
            [_sym("MUUSDT", "EQUITY", onboard_ms=_ms_of("2026-03-10"))], _T1
        )
        record = classify_perp("MUUSDT")
        assert record.instrument_class == "stock_perp"
        assert record.classification_source == SOURCE_EXCHANGEINFO
        assert record.classification_confidence == 1.0
        assert record.snapshot_available_at == _T1
        assert record.onboard_date == "2026-03-10"
        assert record.listed_asof is None  # no as-of -> no verdict, not False

    def test_asof_before_first_snapshot_is_registry_empty(self):
        snapshot_instruments([_sym()], _T1)
        record = classify_perp("MUUSDT", as_of="2026-02-01")
        assert record.instrument_class == "unknown_perp"
        assert record.classification_source == SOURCE_REGISTRY_EMPTY
        assert record.classification_confidence == 0.0
        assert record.listed_asof is None

    def test_missing_ledger_db_is_registry_empty(self):
        # No snapshot written this test -> the per-test tmp DB does not exist.
        record = classify_perp("BTCUSDT")
        assert record.classification_source == SOURCE_REGISTRY_EMPTY
        assert record.instrument_class == "unknown_perp"

    def test_bare_date_asof_is_conservative_start_of_day(self):
        # Snapshot taken at 10:00 UTC on 2026-03-01; a bare-date as-of means
        # midnight UTC at the START of that day, so the row is NOT yet
        # available — the leak-safe reading.
        snapshot_instruments([_sym()], "2026-03-01T10:00:00+00:00")
        before = classify_perp("MUUSDT", as_of="2026-03-01")
        after = classify_perp("MUUSDT", as_of="2026-03-02")
        assert before.classification_source == SOURCE_REGISTRY_EMPTY
        assert after.instrument_class == "stock_perp"

    def test_listed_asof_transitions_at_onboard_date(self):
        snapshot_instruments(
            [_sym("MUUSDT", "EQUITY", onboard_ms=_ms_of("2026-03-10"))], _T1
        )
        pre = classify_perp("MUUSDT", as_of="2026-03-01T12:00:00+00:00")
        post = classify_perp("MUUSDT", as_of="2026-06-01")
        assert pre.instrument_class == "stock_perp"
        assert pre.listed_asof is False  # 2026-03-01 < onboard 2026-03-10
        assert post.listed_asof is True

    def test_asof_datetime_normalization(self):
        snapshot_instruments([_sym("BTCUSDT", "COIN")], _T1)
        record = classify_perp("BTCUSDT", as_of="2026-03-02T08:00:00+02:00")
        # +02:00 offset coerced to UTC (2026-03-02T06:00Z) — still after T1.
        assert record.instrument_class == "pure_crypto_perp"

    def test_symbol_input_normalization_matches_exchangeinfo_keys(self):
        snapshot_instruments([_sym()], _T1)
        for spelling in ("MUUSDT", "muusdt", "MU-USDT", " mu-usdt "):
            assert classify_perp(spelling).instrument_class == "stock_perp", spelling

    def test_usdc_input_probes_the_usdt_twin(self):
        snapshot_instruments([_sym("MUUSDT", "EQUITY")], _T1)
        assert classify_perp("MUUSDC").instrument_class == "stock_perp"

    def test_pit_query_picks_the_latest_qualifying_snapshot(self):
        snapshot_instruments([_sym("MUUSDT", "EQUITY")], _T1)
        snapshot_instruments([_sym("MUUSDT", "COIN")], _T2)
        latest = classify_perp("MUUSDT")
        pinned_to_t1 = classify_perp("MUUSDT", as_of="2026-03-15")
        assert latest.instrument_class == "pure_crypto_perp"  # T2 reclass wins
        assert pinned_to_t1.instrument_class == "stock_perp"  # PIT holds at T1

    def test_memo_auto_invalidates_when_a_newer_snapshot_appears(self):
        snapshot_instruments([_sym("MUUSDT", "EQUITY")], _T1)
        assert classify_perp("MUUSDT").instrument_class == "stock_perp"
        snapshot_instruments([_sym("MUUSDT", "COIN")], _T2)  # same key, new row
        assert classify_perp("MUUSDT").instrument_class == "pure_crypto_perp"

    def test_registry_available_at_values_test_helper(self):
        snapshot_instruments([_sym()], _T1)
        snapshot_instruments([_sym()], _T2)
        assert registry_available_at_values("MUUSDT") == [_T1, _T2]
        assert registry_available_at_values("NOVAUSDT") == []


# ---- routing flag matrix --------------------------------------------------------


@pytest.mark.unit
class TestRoutingFlagMatrix:
    def test_flag_off_is_the_legacy_pure_crypto_default(self):
        # conftest holds instrument_registry off; an unresolvable symbol keeps
        # the legacy pure_crypto_perp answer.
        assert routing.instrument_class("crypto_perp", "ZZQAUSDT") == "pure_crypto_perp"

    def test_flag_off_positive_paths_unchanged(self):
        assert routing.instrument_class("crypto_perp", "MUUSDT") == "stock_perp"
        assert routing.instrument_class("crypto_perp", "BTCUSDT") == "pure_crypto_perp"
        assert routing.instrument_class("stock", "AAPL") == "equity"
        assert routing.instrument_class("crypto", "BTC-USD") == "crypto_spot"

    def test_flag_on_unresolvable_becomes_unknown_perp(self):
        set_config({"instrument_registry": True})
        assert routing.instrument_class("crypto_perp", "ZZQAUSDT") == routing.UNKNOWN_PERP
        assert routing.UNKNOWN_PERP == "unknown_perp"

    def test_flag_on_registry_empty_keeps_positive_seed_classification(self):
        set_config({"instrument_registry": True})
        # MU is in the static seed; empty registry must not regress it.
        assert routing.instrument_class("crypto_perp", "MUUSDT") == "stock_perp"
        assert routing.fundamentals_applicable("crypto_perp", "MUUSDT") is True

    def test_flag_on_registry_evidence_beats_seed(self):
        set_config({"instrument_registry": True})
        snapshot_instruments([_sym("MUUSDT", "COIN")], _T1)  # seed says equity
        assert routing.instrument_class("crypto_perp", "MUUSDT") == "pure_crypto_perp"
        assert routing.fundamentals_applicable("crypto_perp", "MUUSDT") is False

    def test_flag_on_registry_evidence_beats_warm_default(self):
        set_config({"instrument_registry": True})
        # NEWCO is neither in the seed nor warmed — only the registry knows.
        snapshot_instruments([_sym("NEWCOUSDT", "EQUITY")], _T1)
        assert routing.instrument_class("crypto_perp", "NEWCOUSDT") == "stock_perp"
        assert routing.fundamentals_applicable("crypto_perp", "NEWCOUSDT") is True

    def test_flag_on_unsupported_contract_type_is_unknown_and_inapplicable(self):
        set_config({"instrument_registry": True})
        snapshot_instruments([_sym("TENCENTUSDT", "HK_EQUITY")], _T1)
        assert routing.instrument_class("crypto_perp", "TENCENTUSDT") == routing.UNKNOWN_PERP
        assert routing.fundamentals_applicable("crypto_perp", "TENCENTUSDT") is False

    def test_flag_off_non_perp_paths_never_consult_the_registry(self):
        # Even with snapshots on disk, equity/spot routing is untouched.
        snapshot_instruments([_sym("AAPLUSDT", "EQUITY")], _T1)
        assert routing.instrument_class("stock", "AAPLUSDT") == "equity"
        assert routing.instrument_class("crypto_spot", "AAPLUSDT") == "crypto_spot"


@pytest.mark.unit
class TestDescribeInstrumentRegistryEnrichment:
    def test_flag_on_prefers_registry_onboard_and_status(self):
        set_config({"instrument_registry": True})
        snapshot_instruments(
            [_sym("BTCUSDT", "COIN", onboard_ms=_ms_of("2019-09-08"))], _T1
        )
        d = routing.describe_instrument("crypto_perp", "BTCUSDT")
        assert d.instrument_class == "pure_crypto_perp"
        assert d.onboard_date == "2019-09-08"  # listing cache knows nothing of BTC
        assert d.listing_status == "TRADING"
        assert d.listed_asof is None  # no as-of
        listed = routing.describe_instrument("crypto_perp", "BTCUSDT", as_of="2020-01-01")
        pre = routing.describe_instrument("crypto_perp", "BTCUSDT", as_of="2019-01-01")
        assert listed.listed_asof is True
        assert pre.listed_asof is False

    def test_flag_on_registry_resolves_underlying_beyond_warm_cache(self):
        set_config({"instrument_registry": True})
        snapshot_instruments([_sym("NEWCOUSDT", "EQUITY")], _T1)
        d = routing.describe_instrument("crypto_perp", "NEWCOUSDT")
        assert d.instrument_class == "stock_perp"
        assert d.underlying_equity == "NEWCO"
        assert d.fundamentals_symbol == "NEWCO"
        assert d.session_calendar == SESSION_BINANCE_TRADFI
        assert d.vendor_symbols["yfinance"] == "NEWCO"

    def test_flag_off_keeps_legacy_descriptor_bytes(self):
        snapshot_instruments(
            [_sym("BTCUSDT", "COIN", onboard_ms=_ms_of("2019-09-08"))], _T1
        )
        d = routing.describe_instrument("crypto_perp", "BTCUSDT")
        assert d.instrument_class == "pure_crypto_perp"
        assert d.onboard_date is None  # flag off: registry never consulted
        assert d.listing_status == "unknown"


# ---- session calendars ----------------------------------------------------------


@pytest.mark.unit
class TestTradingSessions:
    def test_constants_and_routing_reexport(self):
        assert SESSION_CONTINUOUS == "continuous_24_7"
        assert SESSION_BINANCE_TRADFI == "binance_published_tradfi_sessions"
        assert SESSION_EXCHANGE == "listing_exchange_sessions"
        # routing re-exports the SAME names (existing importers unaffected).
        assert routing.SESSION_CONTINUOUS is SESSION_CONTINUOUS
        assert routing.SESSION_BINANCE_TRADFI is SESSION_BINANCE_TRADFI
        assert routing.SESSION_EXCHANGE is SESSION_EXCHANGE

    def test_continuous_is_calendar_day_delta(self):
        # Half-open [start, end): 2026-01-01..2026-01-11 -> 10 days.
        assert trading_sessions_between(SESSION_CONTINUOUS, "2026-01-01", "2026-01-11") == 10
        assert trading_sessions_between(SESSION_CONTINUOUS, "2026-01-01", "2026-01-01") == 0

    def test_tradfi_counts_weekdays_pinned(self):
        # 2026-01-01 is a Thursday. [Jan 1, Jan 11): Jan 1,2,5,6,7,8,9 -> 7.
        assert trading_sessions_between(
            SESSION_BINANCE_TRADFI, "2026-01-01", "2026-01-11"
        ) == 7
        # Mon Jan 5 through Sun Jan 11 -> 5 weekdays.
        assert trading_sessions_between(
            SESSION_BINANCE_TRADFI, "2026-01-05", "2026-01-12"
        ) == 5
        # Three full weeks -> 15.
        assert trading_sessions_between(
            SESSION_BINANCE_TRADFI, "2026-01-05", "2026-01-26"
        ) == 15

    def test_exchange_calendar_shares_the_weekday_approximation(self):
        assert trading_sessions_between(SESSION_EXCHANGE, "2026-01-01", "2026-01-11") == 7

    @pytest.mark.parametrize(
        ("calendar", "start", "end"),
        [
            ("nonsense", "2026-01-01", "2026-01-11"),
            (SESSION_CONTINUOUS, "2026-01-11", "2026-01-01"),  # inverted window
            (SESSION_CONTINUOUS, "2026/01/01", "2026-01-11"),  # malformed date
            (SESSION_BINANCE_TRADFI, "2026-01-01", "not-a-date"),
        ],
    )
    def test_invalid_inputs_raise(self, calendar: str, start: str, end: str):
        with pytest.raises(ValueError):
            trading_sessions_between(calendar, start, end)


# ---- warm-hook integration --------------------------------------------------------


@pytest.mark.unit
class TestWarmRegistryIntegration:
    PAYLOAD = {
        "symbols": [
            _sym("MUUSDT", "EQUITY"),
            _sym("BTCUSDT", "COIN"),
            _sym("TENCENTUSDT", "HK_EQUITY"),
            _sym("OLDCOUSDT", "EQUITY", status="BREAK"),
        ]
    }

    def test_warm_with_flag_on_writes_registry_rows_from_same_payload(
        self, monkeypatch,
    ):
        set_config({"instrument_registry": True})
        calls = {"n": 0}

        def fake_get(path, params, **kwargs):  # noqa: ARG001
            calls["n"] += 1
            return self.PAYLOAD

        monkeypatch.setattr(bnb, "_http_get", fake_get)
        bases = bnb.warm_equity_perp_bases()
        assert bases == frozenset({"MU"})  # warm behavior unchanged
        assert calls["n"] == 1  # ZERO extra HTTP for the registry
        assert bnb.equity_perp_listing_info()["MU"]["onboard_date"] == _ONBOARD_DATE

        assert registry_available_at_values("MUUSDT")
        assert classify_perp("BTCUSDT").instrument_class == "pure_crypto_perp"
        assert classify_perp("TENCENTUSDT").unsupported_reason == (
            "unsupported-contract-type:HK_EQUITY"
        )
        # Non-TRADING rows are evidence too — the registry keeps them.
        assert classify_perp("OLDCOUSDT").status == "BREAK"

        # A second warm inside the TTL serves the cache: no fetch, no new rows.
        before = registry_available_at_values("MUUSDT")
        assert bnb.warm_equity_perp_bases() == bases
        assert calls["n"] == 1
        assert registry_available_at_values("MUUSDT") == before

    def test_warm_with_flag_off_writes_no_registry_rows(self, monkeypatch):
        monkeypatch.setattr(
            bnb, "_http_get", lambda path, params, **kwargs: self.PAYLOAD
        )
        assert bnb.warm_equity_perp_bases() == frozenset({"MU"})
        assert registry_available_at_values("MUUSDT") == []
        assert classify_perp("MUUSDT").classification_source == SOURCE_REGISTRY_EMPTY

    def test_registry_failure_downgrades_to_warning(self, monkeypatch, caplog):
        set_config({"instrument_registry": True})

        def boom(rows, available_at):  # noqa: ARG001
            raise RuntimeError("registry backend down")

        monkeypatch.setattr(registry, "snapshot_instruments", boom)
        monkeypatch.setattr(
            bnb, "_http_get", lambda path, params, **kwargs: self.PAYLOAD
        )
        with caplog.at_level(logging.WARNING, logger="yialpha.dataflows.binance"):
            bases = bnb.warm_equity_perp_bases()
        assert bases == frozenset({"MU"})  # warm still succeeds
        assert any(
            "instrument registry" in rec.message for rec in caplog.records
        ), [rec.message for rec in caplog.records]
        assert registry_available_at_values("MUUSDT") == []


# ---- corporate actions (V2.4 skeleton) ---------------------------------------------


@pytest.mark.unit
class TestCorporateActions:
    def _record(self, **overrides: Any) -> CorporateActionRecord:
        fields: dict[str, Any] = {
            "symbol": "MU",
            "action_type": "dividend",
            "event_date": "2026-02-14",
            "available_at": "2026-02-10T00:00:00+00:00",
            "source": "test_vendor",
            "details": {"amount": 0.115, "currency": "USD"},
        }
        fields.update(overrides)
        return CorporateActionRecord(**fields)

    def test_valid_record_passes(self):
        validate_corporate_action(self._record())  # no raise
        validate_corporate_action(self._record(available_at=None))  # optional

    def test_bad_action_type_raises(self):
        with pytest.raises(ValueError, match="action type"):
            validate_corporate_action(self._record(action_type="ipo"))

    def test_malformed_event_date_raises(self):
        with pytest.raises(ValueError, match="event_date"):
            validate_corporate_action(self._record(event_date="02/14/2026"))

    def test_malformed_available_at_raises(self):
        with pytest.raises(ValueError, match="available_at"):
            validate_corporate_action(self._record(available_at="not-a-date"))

    def test_render_disclosure_lines(self):
        records = [
            self._record(),
            self._record(symbol="TSLA", action_type="split", event_date="2026-03-05",
                         details={"ratio": "5:1"}),
        ]
        block = render_corporate_action_disclosure(records)
        lines = block.splitlines()
        assert len(lines) == 2
        assert "MU: dividend effective 2026-02-14" in lines[0]
        assert "knowable at 2026-02-10T00:00:00+00:00" in lines[0]
        assert "source test_vendor" in lines[0]
        assert "amount=0.115" in lines[0]
        assert "TSLA: split effective 2026-03-05" in lines[1]
        assert "ratio=5:1" in lines[1]

    def test_render_empty_sequence_is_empty_string(self):
        assert render_corporate_action_disclosure([]) == ""
