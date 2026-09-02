"""SQLite storage seam for the V2 ledgers (one central append-first database).

All V2 ledgers — instrument registry snapshots, run evidence, blind analyst
predictions, outcomes, ticket mirrors, regime states (V2.2), (and, from V2.4,
positions / portfolio snapshots) — share ONE SQLite file so cross-ledger
joins and atomic multi-table commits stay local to a single database. The
path comes from the config key ``ledger_db_path`` (default
``~/.yialpha/ledger/portfolio.db``, overridable with the ``YIALPHA_LEDGER_DB``
environment variable).

Contract (docs/V2_BASELINE.md):

* **Append-first** — this module exposes no UPDATE path for ledger rows;
  corrections are new rows referencing the row they revise. Row-level
  helpers built by callers write literal SQL with bound parameters inside
  :func:`ledger_transaction` — never string-formatted SQL.
* **Atomic** — multi-row writes run inside one ``BEGIN IMMEDIATE``
  transaction, so a crash between two related writes can never strand one
  without the other (V2.4 relies on this for Final Ticket + Position).
* **Versioned** — the schema version lives in the ``schema_meta`` table
  (``schema_version`` row). Migrations are explicit sequential blocks of
  idempotent statements (``IF NOT EXISTS``), so concurrent processes can
  race a migration harmlessly; an older binary opening a newer database
  fails loudly instead of guessing at unknown columns.
* **Thread-safe** — connections are thread-local (WAL, ``busy_timeout``);
  batch workers each get their own connection, and migrations are
  serialised by a process-wide lock.

Time discipline (V2_BASELINE.md, mandatory): every timestamp column is an
ISO-8601 UTC string — ``YYYY-MM-DD`` for pure dates, otherwise
``YYYY-MM-DDTHH:MM:SS+00:00`` from :func:`utc_now_iso`. Bare
``time`` / ``date`` / ``timestamp`` column names are forbidden.

Key chain (I2): ``run_id → prediction_id → decision_id → ticket_id →
position_id``; Evidence is many-to-many and referenced back via
``evidence_ids[]`` columns on predictions/tickets.
"""

from __future__ import annotations

import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from datetime import UTC, datetime
from pathlib import Path

# Highest schema version this binary knows. Bump + add a migration block in
# _migrate() whenever a ledger table is renamed/removed/redefined (additive
# optional columns with defaults do not need a migration).
_KNOWN_SCHEMA_VERSION = 2

_ledger_lock = threading.Lock()
_local = threading.local()
_migrated_paths: set[str] = set()


def utc_now_iso() -> str:
    """Current UTC time as an ISO-8601 string with seconds precision."""
    return datetime.now(UTC).isoformat(timespec="seconds")


def ledger_db_path() -> str:
    """Resolved ledger DB path from config (``ledger_db_path``)."""
    from yialpha.dataflows.config import get_config

    configured = str(get_config().get("ledger_db_path") or "").strip()
    path = configured or str(Path.home() / ".yialpha" / "ledger" / "portfolio.db")
    return str(Path(path).resolve())


def ledger_exists() -> bool:
    """True when the ledger DB file exists (read-only opens raise otherwise)."""
    return Path(ledger_db_path()).exists()


def _connect(path: str, *, readonly: bool) -> sqlite3.Connection:
    if readonly:
        conn = sqlite3.connect(
            f"file:{path}?mode=ro",
            uri=True,
            timeout=5.0,
            check_same_thread=False,
            isolation_level=None,
        )
    else:
        conn = sqlite3.connect(
            path,
            timeout=5.0,
            check_same_thread=False,
            isolation_level=None,
        )
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout = 5000")
    conn.execute("PRAGMA foreign_keys = ON")
    if not readonly:
        conn.execute("PRAGMA journal_mode = WAL")
    return conn


def _migrate(conn: sqlite3.Connection) -> None:
    """Bring the ledger DB up to ``_KNOWN_SCHEMA_VERSION`` (idempotent)."""
    conn.execute(
        "CREATE TABLE IF NOT EXISTS schema_meta ("
        "key TEXT PRIMARY KEY, value TEXT NOT NULL)"
    )
    row = conn.execute(
        "SELECT value FROM schema_meta WHERE key = 'schema_version'"
    ).fetchone()
    version = int(row[0]) if row is not None else 0
    if version > _KNOWN_SCHEMA_VERSION:
        raise RuntimeError(
            f"ledger DB schema version {version} is newer than this binary "
            f"(knows {_KNOWN_SCHEMA_VERSION}); upgrade yialpha before "
            f"touching {ledger_db_path()}"
        )
    if version < 1:
        # v1 (V2.1 Measurability): runs, instrument registry snapshots,
        # evidence, blind predictions, outcomes, ticket mirrors.
        with ledger_transaction(conn=conn) as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    run_id           TEXT PRIMARY KEY,
                    ticker           TEXT NOT NULL,
                    asset_type       TEXT NOT NULL,
                    instrument_class TEXT,
                    analysis_as_of   TEXT NOT NULL,
                    created_at       TEXT NOT NULL,
                    config_digest    TEXT,
                    schema_version   TEXT NOT NULL
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS instrument_snapshots (
                    symbol                    TEXT NOT NULL,
                    snapshot_available_at     TEXT NOT NULL,
                    classification_source     TEXT NOT NULL,
                    instrument_class          TEXT NOT NULL,
                    unsupported_reason        TEXT,
                    underlying_type           TEXT,
                    underlying_symbol         TEXT,
                    quote_asset               TEXT,
                    margin_asset              TEXT,
                    onboard_date              TEXT,
                    status                    TEXT,
                    session_calendar          TEXT,
                    tick_size                 REAL,
                    step_size                 REAL,
                    min_qty                   REAL,
                    min_notional              REAL,
                    leverage_bracket_version  TEXT,
                    classification_confidence REAL NOT NULL,
                    raw_payload               TEXT,
                    created_at                TEXT NOT NULL,
                    PRIMARY KEY (symbol, snapshot_available_at, classification_source)
                )
                """
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_instrument_snapshots_symbol "
                "ON instrument_snapshots (symbol, snapshot_available_at)"
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS evidence (
                    evidence_id    TEXT PRIMARY KEY,
                    run_id         TEXT NOT NULL REFERENCES runs(run_id),
                    source         TEXT NOT NULL,
                    category       TEXT NOT NULL,
                    symbol         TEXT,
                    scope          TEXT NOT NULL,
                    payload_hash   TEXT NOT NULL,
                    source_url     TEXT,
                    event_time     TEXT,
                    available_at   TEXT NOT NULL,
                    created_at     TEXT NOT NULL,
                    replayability  TEXT NOT NULL,
                    quality_status TEXT
                )
                """
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_evidence_run ON evidence (run_id)"
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS predictions (
                    prediction_id          TEXT PRIMARY KEY,
                    run_id                 TEXT NOT NULL REFERENCES runs(run_id),
                    analyst                TEXT NOT NULL,
                    instrument_id          TEXT NOT NULL,
                    prediction_scope       TEXT NOT NULL,
                    horizon_days           INTEGER NOT NULL,
                    direction              TEXT NOT NULL,
                    prob_up                REAL,
                    expected_return        REAL,
                    target_price           REAL,
                    target_currency        TEXT,
                    price_basis            TEXT,
                    confidence             REAL,
                    evidence_ids           TEXT NOT NULL DEFAULT '[]',
                    analysis_as_of         TEXT NOT NULL,
                    created_at             TEXT NOT NULL,
                    schema_version         TEXT NOT NULL,
                    feature_version        TEXT NOT NULL,
                    original_prediction_id TEXT,
                    debate_revision        INTEGER,
                    revision_reason        TEXT
                )
                """
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_predictions_run ON predictions (run_id)"
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_predictions_slice "
                "ON predictions (analyst, instrument_id, horizon_days)"
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS outcomes (
                    outcome_id            TEXT PRIMARY KEY,
                    prediction_id         TEXT NOT NULL REFERENCES predictions(prediction_id),
                    run_id                TEXT NOT NULL,
                    ticket_id             TEXT,
                    horizon_days          INTEGER NOT NULL,
                    status                TEXT NOT NULL,
                    contract_price_return REAL,
                    underlying_return     REAL,
                    basis_return          REAL,
                    funding_pnl           REAL,
                    fees                  REAL,
                    slippage              REAL,
                    liquidation_loss      REAL,
                    net_return            REAL,
                    legs_missing          TEXT,
                    outcome_available_at  TEXT,
                    computed_at           TEXT NOT NULL,
                    UNIQUE (prediction_id, horizon_days)
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS tickets (
                    ticket_id      TEXT PRIMARY KEY,
                    decision_id    TEXT,
                    run_id         TEXT,
                    payload        TEXT NOT NULL,
                    ticket_version TEXT NOT NULL,
                    written_at     TEXT NOT NULL
                )
                """
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_tickets_run ON tickets (run_id)"
            )
            cur.execute(
                "INSERT INTO schema_meta (key, value) VALUES ('schema_version', ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                ("1",),
            )
    if version < 2:
        # v2 (V2.2 Context): the regimes table + the regime_id linkage
        # columns on predictions/outcomes. CREATE TABLE is idempotent
        # (IF NOT EXISTS); SQLite's ALTER TABLE has no IF NOT EXISTS, so each
        # ALTER runs in its OWN tiny transaction with a duplicate-column
        # recovery — two processes racing this migration both pass the
        # version check above, and the loser must read "duplicate column
        # name" as SUCCESS (the column exists) rather than aborting with a
        # half-migrated DB. A separate transaction per ALTER also means the
        # duplicate error can never roll back the CREATE TABLE above.
        with ledger_transaction(conn=conn) as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS regimes (
                    regime_id       TEXT PRIMARY KEY,
                    payload         TEXT NOT NULL,
                    regime_version  TEXT NOT NULL,
                    analysis_as_of  TEXT,
                    ticker          TEXT,
                    instrument_class TEXT,
                    computed_at     TEXT NOT NULL
                )
                """
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_regimes_ticker ON regimes (ticker)"
            )
        _alter_predictions_add_regime_id(conn)
        _alter_outcomes_add_regime_id(conn)
        with ledger_transaction(conn=conn) as cur:
            cur.execute(
                "INSERT INTO schema_meta (key, value) VALUES ('schema_version', ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                ("2",),
            )


def _alter_predictions_add_regime_id(conn: sqlite3.Connection) -> None:
    """Idempotent ``predictions.regime_id`` column (see the v2 block above)."""
    try:
        with ledger_transaction(conn=conn) as cur:
            cur.execute("ALTER TABLE predictions ADD COLUMN regime_id TEXT")
    except sqlite3.OperationalError as exc:
        if "duplicate column name" not in str(exc).lower():
            raise


def _alter_outcomes_add_regime_id(conn: sqlite3.Connection) -> None:
    """Idempotent ``outcomes.regime_id`` column (see the v2 block above)."""
    try:
        with ledger_transaction(conn=conn) as cur:
            cur.execute("ALTER TABLE outcomes ADD COLUMN regime_id TEXT")
    except sqlite3.OperationalError as exc:
        if "duplicate column name" not in str(exc).lower():
            raise


def get_connection(*, readonly: bool = False) -> sqlite3.Connection:
    """Thread-local ledger connection (migrated on first write-side open).

    ``readonly=True`` opens in SQLite ro URI mode and never creates the file;
    it raises ``sqlite3.OperationalError`` when the DB does not exist yet —
    callers serving optional reads should check :func:`ledger_exists` first.
    """
    path = ledger_db_path()
    conns: dict[str, sqlite3.Connection] | None = getattr(_local, "connections", None)
    if conns is None:
        conns = {}
        _local.connections = conns
    key = f"{path}:{'ro' if readonly else 'rw'}"
    conn = conns.get(key)
    if conn is None:
        if not readonly:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        conn = _connect(path, readonly=readonly)
        if not readonly:
            with _ledger_lock:
                if path not in _migrated_paths:
                    _migrate(conn)
                    _migrated_paths.add(path)
        conns[key] = conn
    return conn


@contextmanager
def ledger_transaction(
    *, conn: sqlite3.Connection | None = None
) -> Iterator[sqlite3.Cursor]:
    """One ``BEGIN IMMEDIATE`` transaction; the only transaction boundary.

    Callers execute literal SQL with bound parameters on the yielded cursor.
    Never nest :func:`ledger_transaction` — SQLite cannot nest BEGIN.
    """
    connection = conn if conn is not None else get_connection()
    connection.execute("BEGIN IMMEDIATE")
    try:
        yield connection.cursor()
    except BaseException:
        connection.execute("ROLLBACK")
        raise
    connection.execute("COMMIT")


def reset_ledger_state_for_test() -> None:
    """Close this thread's connections and forget migration state.

    conftest points ``ledger_db_path`` at a per-test tmp file; without this
    reset, a connection opened for the previous test's path would keep
    serving that database for the current test.
    """
    conns = getattr(_local, "connections", None)
    if conns:
        for conn in conns.values():
            with suppress(sqlite3.Error):
                conn.close()
        conns.clear()
    with _ledger_lock:
        _migrated_paths.clear()
