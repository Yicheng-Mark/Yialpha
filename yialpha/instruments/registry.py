"""Persistent point-in-time instrument registry (V2.1 record stage).

The registry is the ``instrument_snapshots`` table of the central ledger
(:mod:`yialpha.ledger.sqlite`, migration v1). Two entry points:

* :func:`snapshot_instruments` — the WRITE side, called by the exchangeInfo
  warm hook in :mod:`yialpha.dataflows.binance` with the same payload the
  warm already fetched (zero extra HTTP). Every symbol row becomes one
  append-only ``INSERT OR IGNORE`` keyed (symbol, snapshot_available_at,
  classification_source), so re-warming within the same second is a no-op and
  a later warm adds a NEW row rather than mutating history — classifications
  survive restarts and remain replayable.
* :func:`classify_perp` — the READ side, a point-in-time query used by
  :mod:`yialpha.graph.routing`: the latest snapshot for the symbol with
  ``snapshot_available_at <= as_of`` (evidence that existed at or before the
  analysis moment; ``as_of=None`` takes the latest unconditionally).

Symbol-key convention: registry rows are keyed by the exchangeInfo ``symbol``
value — the compact, uppercase perp symbol (``"MUUSDT"``, ``"BTCUSDT"``).
Lookups normalize the caller's ticker the same way the routing/base-matching
path does (``strip().upper()`` with dashes removed), and a ``USDC``-quoted
input additionally probes its ``USDT`` twin, because routing's base matching
collapses USDC/USDT quotes onto the USDT-M book while the registry persists
each venue symbol verbatim.

PIT comparison normalization: ``as_of`` may be an ISO date or datetime. A
bare date is interpreted CONSERVATIVELY as midnight UTC at the START of that
date (``"2026-03-01"`` → ``"2026-03-01T00:00:00+00:00"``) — a snapshot
recorded later during the same day is treated as not yet available to that
analysis, which is the leak-safe reading for replay integrity. Datetimes are
coerced to UTC seconds-precision ISO. All stored ``snapshot_available_at``
values come from :func:`yialpha.ledger.sqlite.utc_now_iso` (identical format),
so a plain lexicographic ``<=`` is chronologically exact.

Fail-soft contract: neither entry point ever raises into a run. Writes catch
``sqlite3.Error`` (returning 0 persisted); reads treat a missing ledger DB or
any SQLite error as "no evidence" and return the ``registry_empty`` record.
"""
from __future__ import annotations

import json
import logging
import sqlite3
import threading
from dataclasses import replace
from datetime import UTC, datetime
from typing import Any

from yialpha.instruments.models import (
    InstrumentRecord,
    classify_exchangeinfo_row,
)
from yialpha.instruments.sessions import SESSION_BINANCE_TRADFI, SESSION_CONTINUOUS
from yialpha.ledger.sqlite import (
    get_connection,
    ledger_exists,
    ledger_transaction,
    utc_now_iso,
)

logger = logging.getLogger(__name__)

#: Source tag for rows persisted from a live exchangeInfo warm (confidence 1.0).
SOURCE_EXCHANGEINFO = "binance_exchangeinfo"

#: Source tag returned when the registry holds no usable row for the symbol
#: (or the ledger DB does not exist yet) — callers fall back to warm/seed.
SOURCE_REGISTRY_EMPTY = "registry_empty"

#: Memo cache for classify_perp: (symbol, normalized as-of bound) ->
#: (record, max snapshot_available_at seen for the symbol at compute time).
#: A later snapshot bumps the observed max, which auto-invalidates the entry.
_MEMO: dict[tuple[str, str | None], tuple[InstrumentRecord, str | None]] = {}
_MEMO_LOCK = threading.Lock()


def reset_registry_cache_for_test() -> None:
    """Drop the classify_perp memo cache (tests / forced re-read)."""
    with _MEMO_LOCK:
        _MEMO.clear()


# ---------------------------------------------------------------------------
# Write side — append-only snapshots from an exchangeInfo payload.
# ---------------------------------------------------------------------------


def snapshot_instruments(rows: list[dict[str, Any]], available_at: str) -> int:
    """Persist one exchangeInfo payload into the instrument registry.

    One ``INSERT OR IGNORE`` per symbol row (same-second re-warms dedupe on
    the table's primary key) inside a single ledger transaction, so a crash
    never strands half a snapshot. Rows without a usable ``symbol`` string
    are skipped. Fail-soft: a ``sqlite3.Error`` is logged and swallowed — the
    warm that called this must never fail because the registry did.

    Returns the count of NEW rows persisted.
    """
    created_at = utc_now_iso()
    persisted = 0
    try:
        with ledger_transaction() as cur:
            for row in rows:
                if not isinstance(row, dict):
                    continue
                symbol = row.get("symbol")
                if not isinstance(symbol, str) or not symbol.strip():
                    continue
                cur.execute(
                    "INSERT OR IGNORE INTO instrument_snapshots ("
                    "symbol, snapshot_available_at, classification_source, "
                    "instrument_class, unsupported_reason, underlying_type, "
                    "underlying_symbol, quote_asset, margin_asset, "
                    "onboard_date, status, session_calendar, tick_size, "
                    "step_size, min_qty, min_notional, leverage_bracket_version, "
                    "classification_confidence, raw_payload, created_at"
                    ") VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    _row_to_params(row, available_at, created_at),
                )
                persisted += max(cur.rowcount, 0)
    except sqlite3.Error as exc:
        logger.warning(
            "instrument registry snapshot failed (nothing persisted): %s", exc
        )
        return 0
    return persisted


def _row_to_params(row: dict[str, Any], available_at: str, created_at: str) -> tuple[Any, ...]:
    """Ledger INSERT parameters for one exchangeInfo symbol row (column order)."""
    symbol = str(row["symbol"]).strip().upper()
    klass, unsupported_reason = classify_exchangeinfo_row(row)
    underlying_type = row.get("underlyingType")
    quote_asset = row.get("quoteAsset")
    quote = quote_asset if isinstance(quote_asset, str) else None
    margin_asset = row.get("marginAsset")
    underlying_symbol: str | None = None
    if klass == "stock_perp":
        base_asset = row.get("baseAsset")
        if isinstance(base_asset, str) and base_asset:
            underlying_symbol = base_asset
        elif quote and symbol.endswith(quote) and len(symbol) > len(quote):
            underlying_symbol = symbol[: -len(quote)]
    session_calendar = (
        SESSION_BINANCE_TRADFI
        if klass == "stock_perp"
        else SESSION_CONTINUOUS
        if klass == "pure_crypto_perp"
        else None
    )
    price_filter, notional_filter = _find_filters(row)
    return (
        symbol,
        available_at,
        SOURCE_EXCHANGEINFO,
        klass,
        unsupported_reason,
        underlying_type if isinstance(underlying_type, str) else None,
        underlying_symbol,
        quote,
        margin_asset if isinstance(margin_asset, str) else None,
        _onboard_date_from_ms(row.get("onboardDate")),
        row.get("status") if isinstance(row.get("status"), str) else None,
        session_calendar,
        _opt_float(price_filter.get("tickSize")),
        _opt_float(price_filter.get("stepSize")),
        _opt_float(price_filter.get("minQty")),
        _opt_float(notional_filter.get("notional", notional_filter.get("minNotional"))),
        None,  # leverage_bracket_version — reserved for a future snapshot tier
        1.0,
        json.dumps(row, sort_keys=True),
        created_at,
    )


def _find_filters(row: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """(PRICE_FILTER, MIN_NOTIONAL) filter dicts from an exchangeInfo row."""
    found: dict[str, dict[str, Any]] = {}
    filters = row.get("filters")
    for item in filters if isinstance(filters, list) else []:
        if isinstance(item, dict) and item.get("filterType") in (
            "PRICE_FILTER", "MIN_NOTIONAL",
        ):
            found.setdefault(str(item.get("filterType")), item)
    return found.get("PRICE_FILTER", {}), found.get("MIN_NOTIONAL", {})


def _onboard_date_from_ms(raw: object) -> str | None:
    """UTC ISO date from the exchangeInfo ``onboardDate`` ms epoch, else None."""
    try:
        ms = int(raw) if isinstance(raw, (int, float)) and raw > 0 else None
        if ms is None:
            return None
        return datetime.fromtimestamp(ms / 1000, tz=UTC).strftime("%Y-%m-%d")
    except (OverflowError, OSError, TypeError, ValueError):
        return None


def _opt_float(raw: object) -> float | None:
    """Best-effort float from a filter value (Binance ships strings)."""
    try:
        return float(raw)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Read side — point-in-time classification.
# ---------------------------------------------------------------------------


def classify_perp(symbol: str, as_of: str | None = None) -> InstrumentRecord:
    """Latest registry classification for ``symbol`` as of ``as_of`` (PIT).

    Returns the newest snapshot row with ``snapshot_available_at <= as_of``
    (``as_of=None`` → the latest row unconditionally). When no row qualifies
    — empty registry, snapshot taken after ``as_of``, or the ledger DB does
    not exist — the result is the ``registry_empty`` unknown record
    (confidence 0.0), never an exception. ``listed_asof`` on the returned
    record is derived from the row's ``onboard_date`` and the as-of DATE part
    (as-of < onboard → False, else True; None when either side is unknown).

    Memoized on (symbol, as-of bound); the memo self-invalidates whenever a
    newer snapshot for the symbol appears (including from another writer),
    so callers can treat this as a cheap repeated call.
    """
    candidates = _symbol_candidates(symbol)
    primary = candidates[0] if candidates else str(symbol)
    bound = _asof_bound(as_of)
    key = (primary, bound)
    current_max = _max_available_at(candidates)
    with _MEMO_LOCK:
        cached = _MEMO.get(key)
    if cached is not None and cached[1] == current_max:
        return cached[0]
    record, row_available_at = _query_latest(candidates, bound, primary)
    if row_available_at is not None and record.onboard_date is not None:
        as_of_date = bound[:10] if bound is not None else None
        listed_asof = None if as_of_date is None else as_of_date >= record.onboard_date
        record = replace(record, listed_asof=listed_asof)
    with _MEMO_LOCK:
        _MEMO[key] = (record, current_max)
    return record


def _query_latest(
    candidates: list[str], bound: str | None, primary: str
) -> tuple[InstrumentRecord, str | None]:
    """Newest qualifying row across the candidate symbols (read-only)."""
    if not candidates or not ledger_exists():
        return _empty_record(primary), None
    try:
        conn = get_connection(readonly=True)
        # Two literal statements: an unbounded as-of (None) must not filter
        # (``<= NULL`` matches nothing in SQL).
        unbounded_sql = (
            "SELECT symbol, snapshot_available_at, classification_source, "
            "instrument_class, unsupported_reason, underlying_type, "
            "underlying_symbol, quote_asset, margin_asset, onboard_date, "
            "status, session_calendar, tick_size, step_size, min_qty, "
            "min_notional, leverage_bracket_version, "
            "classification_confidence "
            "FROM instrument_snapshots WHERE symbol = ? "
            "ORDER BY snapshot_available_at DESC, classification_source ASC "
            "LIMIT 1"
        )
        bounded_sql = (
            "SELECT symbol, snapshot_available_at, classification_source, "
            "instrument_class, unsupported_reason, underlying_type, "
            "underlying_symbol, quote_asset, margin_asset, onboard_date, "
            "status, session_calendar, tick_size, step_size, min_qty, "
            "min_notional, leverage_bracket_version, "
            "classification_confidence "
            "FROM instrument_snapshots WHERE symbol = ? "
            "AND snapshot_available_at <= ? "
            "ORDER BY snapshot_available_at DESC, classification_source ASC "
            "LIMIT 1"
        )
        for candidate in candidates:
            row = (
                conn.execute(unbounded_sql, (candidate,)).fetchone()
                if bound is None
                else conn.execute(bounded_sql, (candidate, bound)).fetchone()
            )
            if row is not None:
                return _record_from_row(row), row["snapshot_available_at"]
    except sqlite3.Error as exc:
        logger.warning("classify_perp: registry read failed (%s); no evidence", exc)
    return _empty_record(primary), None


def _record_from_row(row: sqlite3.Row) -> InstrumentRecord:
    """InstrumentRecord from one instrument_snapshots row."""
    return InstrumentRecord(
        symbol=row["symbol"],
        instrument_class=row["instrument_class"],
        classification_source=row["classification_source"],
        classification_confidence=float(row["classification_confidence"]),
        unsupported_reason=row["unsupported_reason"],
        underlying_type=row["underlying_type"],
        underlying_symbol=row["underlying_symbol"],
        quote_asset=row["quote_asset"],
        margin_asset=row["margin_asset"],
        onboard_date=row["onboard_date"],
        status=row["status"],
        session_calendar=row["session_calendar"],
        tick_size=row["tick_size"],
        step_size=row["step_size"],
        min_qty=row["min_qty"],
        min_notional=row["min_notional"],
        leverage_bracket_version=row["leverage_bracket_version"],
        snapshot_available_at=row["snapshot_available_at"],
        listed_asof=None,  # derived by classify_perp against the caller's as-of
    )


def _empty_record(symbol: str) -> InstrumentRecord:
    """The no-evidence answer: unknown class, registry_empty source."""
    return InstrumentRecord(
        symbol=symbol,
        instrument_class="unknown_perp",
        classification_source=SOURCE_REGISTRY_EMPTY,
        classification_confidence=0.0,
    )


def _symbol_candidates(symbol: str) -> list[str]:
    """Registry keys for a caller ticker, best-first (see module docstring)."""
    compact = str(symbol).strip().upper().replace("-", "")
    if not compact:
        return []
    candidates = [compact]
    if compact.endswith("USDC") and len(compact) > 4:
        candidates.append(compact[:-4] + "USDT")
    return candidates


def _asof_bound(as_of: str | None) -> str | None:
    """Normalize an as-of to the lexicographic comparison bound.

    ``None`` → unbounded (latest). A bare ``YYYY-MM-DD`` becomes midnight UTC
    at the START of that date (conservative: evidence recorded later during
    the day is not yet available to that analysis). Datetimes are coerced to
    UTC seconds-precision ISO so they compare cleanly against the stored
    ``utc_now_iso`` values. Unparseable input is passed through raw with a
    WARNING — the comparison then stays conservative rather than crashing.
    """
    if as_of is None:
        return None
    raw = str(as_of).strip()
    try:
        if "T" not in raw and " " not in raw:
            return (
                datetime.strptime(raw, "%Y-%m-%d")
                .replace(tzinfo=UTC)
                .isoformat(timespec="seconds")
            )
        parsed = datetime.fromisoformat(raw)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=UTC)
        return parsed.astimezone(UTC).isoformat(timespec="seconds")
    except ValueError:
        logger.warning("classify_perp: unparseable as_of %r; comparing raw", as_of)
        return raw


def _max_available_at(candidates: list[str]) -> str | None:
    """Latest snapshot timestamp across the candidate symbols (memo probe)."""
    if not candidates or not ledger_exists():
        return None
    latest: str | None = None
    try:
        conn = get_connection(readonly=True)
        for candidate in candidates:
            row = conn.execute(
                "SELECT MAX(snapshot_available_at) FROM instrument_snapshots "
                "WHERE symbol = ?",
                (candidate,),
            ).fetchone()
            value = row[0] if row is not None else None
            if value is not None and (latest is None or value > latest):
                latest = value
    except sqlite3.Error:
        return None
    return latest


def registry_available_at_values(symbol: str) -> list[str]:
    """Ascending distinct snapshot timestamps recorded for ``symbol`` (tests)."""
    candidates = _symbol_candidates(symbol)
    if not candidates or not ledger_exists():
        return []
    values: list[str] = []
    try:
        conn = get_connection(readonly=True)
        for candidate in candidates:
            rows = conn.execute(
                "SELECT DISTINCT snapshot_available_at FROM instrument_snapshots "
                "WHERE symbol = ? ORDER BY snapshot_available_at",
                (candidate,),
            ).fetchall()
            values.extend(str(row[0]) for row in rows)
    except sqlite3.Error:
        return []
    return sorted(set(values))
