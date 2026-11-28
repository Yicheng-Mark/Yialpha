"""Read-only scans of ``~/.yialpha/logs`` to serve run history to the web UI.

The single source of truth for a completed run is::

    ~/.yialpha/logs/<TICKER>/YiAlphaStrategy_logs/full_states_log_<date>[_perp|_spot].json

written atomically by ``yialpha.graph.trading_graph._log_state``. A same-date
re-run of the SAME venue atomically overwrites (``os.replace``); crypto venue
runs carry a ``_perp``/``_spot`` suffix so a same-date perp AND spot pair are
two first-class files (the date keys served below are the filename stems).
The date in the filename is the analysis date (not the run wall-clock).

Report directories ``~/.yialpha/logs/reports/<TICKER>_<stamp>/`` carry a
wall-clock stamp, NOT the analysis date, and may be multiple per date; they are
listed separately as download links and never force-paired 1:1 with a date.

Why derive everything from the JSON (and not the markdown report tree):
``reporting.write_report_tree``'s section V uses the pre-overlay
``risk_debate_state.judge_decision`` and does NOT contain ``final_trade_decision``
(the quantitative risk overlay is appended only into the JSON's
``final_trade_decision``). The JSON is therefore the only source that has the
final, overlay-adjusted decision.
"""

from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path

from yialpha.accuracy import sanitize_nonfinite
from yialpha.agents.utils.rating import parse_rating
from yialpha.dataflows.utils import safe_ticker_component
from yialpha.graph.overlay_fields import parse_overlay


def _resolve_logs_root() -> Path:
    """Resolve the logs root from ``YIALPHA_RESULTS_DIR`` (same source as the
    CLI/batch write side) so the web read side stays aligned with a custom
    results dir. Falls back to ``~/.yialpha/logs`` when the env is unset,
    matching :mod:`yialpha.default_config`.
    """
    env = os.getenv("YIALPHA_RESULTS_DIR")
    if env:
        return Path(env)
    return Path.home() / ".yialpha" / "logs"


LOGS_ROOT = _resolve_logs_root()
REPORTS_ROOT = LOGS_ROOT / "reports"

# Subdirs under LOGS_ROOT that are not per-ticker result dirs.
_NON_TICKER_DIRS = {"reports", "robust"}

# How many recent ratings ``list_tickers`` exposes per ticker for the home
# card's timeline dots — enough to read a trend at a glance, few enough that
# the endpoint stays one small JSON even with months of daily runs.
_HISTORY_CAP = 8

# Venue-suffixed states logs (2026-09-19): a same-date perp AND spot run of
# one ticker are two first-class files — full_states_log_<date>_perp.json /
# _spot.json beside the legacy unsuffixed name. The captured group IS the
# filename stem, so every exact-name constructor downstream
# (_latest_rating / load_run / load_node_perf) keeps working with the
# suffixed key unchanged.
_DATE_RE = re.compile(r"full_states_log_(\d{4}-\d{2}-\d{2}(?:_perp|_spot)?)\.json$")

# Alias kept for callers/tests that imported the local name; the marker,
# field map and parser are owned by yialpha.graph.overlay_fields (the
# module beside the renderer) so this side can never drift from the
# analyze_window script's copy again.
parse_overlay_local = parse_overlay


def _strategy_dir(ticker: str) -> Path:
    """``LOGS_ROOT/<ticker>/YiAlphaStrategy_logs`` (ticker already validated)."""
    return LOGS_ROOT / ticker / "YiAlphaStrategy_logs"


def _dates_for(ticker: str) -> list[str]:
    """Sorted (ascending) analysis dates with a saved full_states_log."""
    sdir = _strategy_dir(ticker)
    if not sdir.is_dir():
        return []
    dates: list[str] = []
    for f in sdir.glob("full_states_log_*.json"):
        m = _DATE_RE.search(f.name)
        if m:
            dates.append(m.group(1))
    return sorted(dates)


def _null_constant(_literal: str) -> None:
    """``json`` ``parse_constant`` hook: the bare ``NaN`` / ``Infinity`` /
    ``-Infinity`` literals read as ``None``."""
    return None


def _null_overflow(text: str) -> float | None:
    """``json`` ``parse_float`` hook (round-17): an overflowing number literal
    (``1e999``) is not a bare constant — ``parse_constant`` never sees it and
    ``float()`` turns it into ``inf``. Nulled like the constants; every finite
    spelling keeps ``float()``'s exact value (an underflow is a finite 0.0)."""
    value = float(text)
    return value if math.isfinite(value) else None


def _read_json_object(path: Path) -> dict | None:
    """Parse ``path`` as a JSON **object**, or return ``None`` when unreadable.

    Unreadable means: a missing / locked file, invalid UTF-8, truncated JSON,
    or a document whose root is not an object (``[]`` / ``null`` / ``"text"``).
    One such log must degrade to "no rating / no run" — never raise: the
    listing endpoints touch every ticker's files in one scan, so a single
    escaped exception 500'd the whole home grid.

    Round-16: the old ``except (OSError, json.JSONDecodeError)`` missed two
    cases — ``UnicodeDecodeError`` is a ``ValueError`` but not a
    ``JSONDecodeError`` (one GBK-encoded log took down ``/api/tickers``), and
    ``.get`` on a non-object root raised ``AttributeError``.

    ``parse_constant`` maps the bare ``NaN`` / ``Infinity`` literals to ``None``
    in the same pass as the parse (round-11 semantics, see
    :func:`load_accuracy_report`): the write side dumps with ``json.dump``
    defaults (``allow_nan`` on), and this app renders responses with
    ``allow_nan=False`` — serving such a file as-is was a 500. Round-17:
    ``parse_float`` does the same for an overflowing literal (``1e999``), which
    the constants hook cannot see and which 500'd the same way.
    """
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f, parse_constant=_null_constant, parse_float=_null_overflow)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _text(value: object) -> str:
    """A string-typed log field; a non-string value in a malformed log is ``""``."""
    return value if isinstance(value, str) else ""


def _mapping(value: object) -> dict:
    """An object-typed log field; a non-object value in a malformed log is ``{}``."""
    return value if isinstance(value, dict) else {}


def _latest_rating(ticker: str, latest_date: str) -> str | None:
    """Rating of the most recent run, for the home grid + distribution summary.

    Returns ``None`` only when the JSON can't be read; a readable run always
    yields a rating (``parse_rating`` defaults to ``Hold`` when no tier word
    appears, matching ``load_run``'s behavior).
    """
    path = _strategy_dir(ticker) / f"full_states_log_{latest_date}.json"
    if not path.is_file():
        return None
    state = _read_json_object(path)
    if state is None:
        return None
    return parse_rating(_text(state.get("final_trade_decision")))


def list_tickers() -> list[dict]:
    """One entry per ticker that has at least one completed run."""
    out: list[dict] = []
    if not LOGS_ROOT.is_dir():
        return out
    for d in sorted(LOGS_ROOT.iterdir(), key=lambda p: p.name.lower()):
        if not d.is_dir() or d.name in _NON_TICKER_DIRS:
            continue
        dates = _dates_for(d.name)
        if not dates:
            continue
        out.append(
            {
                "ticker": d.name,
                "latest_date": dates[-1],  # _dates_for is ascending
                "latest_rating": _latest_rating(d.name, dates[-1]),
                "run_count": len(dates),
                # Most recent ratings for the home card's timeline dots — same
                # light per-date parse as ``list_runs``' date_ratings, capped
                # to keep the payload (and disk reads) bounded.
                "history": [
                    {"date": date, "rating": _latest_rating(d.name, date)}
                    for date in dates[-_HISTORY_CAP:]
                ],
            }
        )
    return out


def list_reports(ticker: str) -> list[dict]:
    """Wall-clock-stamped report dirs ``<TICKER>_<stamp>/``, newest first."""
    try:
        safe_ticker_component(ticker)
    except ValueError:
        return []
    if not REPORTS_ROOT.is_dir():
        return []
    # Round-13: glob's "{ticker}_*" prefix match also lists a LONGER ticker
    # that starts with this one plus "_" (FOO lists FOO_BAR_* dirs — glob's
    # "*" matches underscores too). The write side stamps every dir
    # ``<TICKER>_<YYYYMMDD>_<HHMMSS>`` (trading_graph.save_reports), so pin
    # the exact shape with a fullmatch instead of trusting the glob prefix.
    stamp_re = re.compile(rf"{re.escape(ticker)}_\d{{8}}_\d{{6}}")
    reports: list[dict] = []
    for d in REPORTS_ROOT.glob(f"{ticker}_*"):
        if stamp_re.fullmatch(d.name) is None:
            continue
        # Round-17: ``is_dir()`` sits inside the guard — on Python <= 3.13 it
        # re-raises stat errors pathlib does not ignore, and an entry the loop
        # cannot stat raised out of the listing (a 500 here, and a finished
        # analysis then reported as "post-run report lookup failed").
        try:
            if not d.is_dir():
                continue
            cr = d / "complete_report.md"
            mtime = cr.stat().st_mtime if cr.is_file() else d.stat().st_mtime
        except OSError:
            continue
        reports.append(
            {
                "dir": d.name,
                "mtime": mtime,
                "complete": cr.is_file(),
            }
        )
    reports.sort(key=lambda r: r["mtime"], reverse=True)
    return reports


def list_runs(ticker: str) -> dict:
    dates = _dates_for(ticker)
    return {
        "ticker": ticker,
        "dates": dates,
        # Per-date rating lets the detail view show a small badge next to each
        # analysis date (a rating evolution over time). Same light parse as
        # ``_latest_rating``; None only when the JSON is unreadable.
        "date_ratings": [{"date": d, "rating": _latest_rating(ticker, d)} for d in dates],
        "reports": list_reports(ticker),
    }


def list_compare() -> dict:
    """Per-ticker rating series for the compare view.

    One entry per ticker that has at least one *readable* rating — a ticker
    whose JSON files are all unreadable would render as an empty row in the
    comparison chart, so it is skipped here rather than filtered client-side.
    """
    out: list[dict] = []
    if not LOGS_ROOT.is_dir():
        return {"tickers": out}
    for d in sorted(LOGS_ROOT.iterdir(), key=lambda p: p.name.lower()):
        if not d.is_dir() or d.name in _NON_TICKER_DIRS:
            continue
        date_ratings = [
            {"date": dt, "rating": _latest_rating(d.name, dt)} for dt in _dates_for(d.name)
        ]
        if any(dr["rating"] for dr in date_ratings):
            out.append({"ticker": d.name, "date_ratings": date_ratings})
    return {"tickers": out}


def load_accuracy_report() -> dict:
    """Serve the ``yialpha verify-history`` artifact (read-only).

    Scoring fetches forward prices per record — a CLI job, not a request
    handler. Until the operator runs it, the honest payload is
    ``available=False`` plus the command hint (never a fabricated empty
    report that would read as "100% accuracy with no data").

    Non-finite read guard (round-11): ``json.loads`` happily parses the bare
    ``NaN`` / ``Infinity`` literals a pre-fix artifact may contain, and this
    app's JSON responses render with ``allow_nan=False`` — serving such an
    artifact as-is turned /api/accuracy into a 500. ``parse_constant`` maps
    those literals to ``None`` IN THE SAME PASS as the parse, replacing
    the old post-parse recursive walk — one pass, no deep copy, same
    nulling semantics as the write side's ``sanitize_nonfinite``. (Round-17:
    an overflowing literal such as ``1e999`` bypassed the constants hook —
    ``_read_json_object`` now nulls it through ``parse_float``.)
    """
    path = LOGS_ROOT / "accuracy" / "accuracy_report.json"
    # Round-16: routed through _read_json_object so an invalid-UTF-8 artifact
    # or a non-object root (``[]`` — ``**data`` raised TypeError) is "not
    # available" instead of a 500; parse_constant lives there now.
    data = _read_json_object(path)
    if data is None:
        return {"available": False, "hint": "yialpha verify-history"}
    return {"available": True, **data}


def load_node_perf(ticker: str, date: str) -> dict | None:
    """Per-node wall-clock + token telemetry, if ``--profile`` wrote it."""
    path = _strategy_dir(ticker) / f"node_perf_{date}.json"
    if not path.is_file():
        return None
    # Same tolerant reader as the states log (perf_telemetry also dumps with
    # ``allow_nan`` on); a non-object root reads as "no telemetry".
    return _read_json_object(path)


def _ledger_readable() -> bool:
    """False when the ledger file exists but cannot be read as a ledger.

    ``ledger_exists()`` only checks the file. A present-but-unreadable ledger
    (a never-migrated zero-byte file, a corrupt or locked one) makes every
    ledger reader degrade to an empty result internally, which the aggregate
    loaders would serve as an empty-but-AVAILABLE table — the "phantom empty
    book" round-10 fixed for positions. The ``schema_version`` row is written
    by the same transaction that creates the v1 tables, so a ledger that
    answers this probe has every table these loaders read.
    """
    from yialpha.ledger.sqlite import get_connection

    try:
        get_connection(readonly=True).execute(
            "SELECT value FROM schema_meta WHERE key = 'schema_version'"
        ).fetchone()
    except Exception:  # noqa: BLE001 -- read-only UI path degrades, never raises
        return False
    return True


def load_run_predictions(run_id: str) -> dict | None:
    """Blind analyst predictions for one ledger run (read-only).

    Returns None when the run_id is unknown (the UI maps that to 404). Reads
    the central V2 ledger DB; absent DB degrades to an empty-but-known run
    only when the runs table has no row — both paths are honest about what
    was recorded.
    """
    from yialpha.ledger.sqlite import get_connection, ledger_exists

    if not ledger_exists():
        return None
    try:
        row = get_connection(readonly=True).execute(
            "SELECT run_id FROM runs WHERE run_id = ?", (run_id,)
        ).fetchone()
    except Exception:  # noqa: BLE001 -- read-only UI path degrades, never raises
        return None
    if row is None:
        return None
    from dataclasses import asdict, is_dataclass

    from yialpha.ledger.evidence import evidence_ids_for_run
    from yialpha.ledger.predictions import predictions_for_run

    records = predictions_for_run(run_id)
    return {
        "run_id": run_id,
        "prediction_ids": [record.prediction_id for record in records],
        "predictions": [
            # Round-17: ``expected_return`` / ``target_price`` are not validated
            # for finiteness on the write side (a REAL column stores ``inf``) —
            # nulled here instead of failing the whole response render.
            # ``not isinstance(.., type)`` is mypy-only narrowing (typeshed types
            # is_dataclass as ``DataclassInstance | type[DataclassInstance]``); a
            # class object was never a valid record here.
            sanitize_nonfinite(
                asdict(record) if is_dataclass(record) and not isinstance(record, type) else dict(record)
            )
            for record in records
        ],
        "evidence_ids": evidence_ids_for_run(run_id),
    }


def load_outcomes(limit: int = 500) -> dict:
    """Forward-outcome rows with net-return attribution legs (read-only).

    ``available`` is true only when the ledger could actually be read: the
    reader degrades to ``[]`` on any ledger error, which would otherwise be
    served as an empty-but-available table (see :func:`_ledger_readable`).
    """
    from dataclasses import asdict, is_dataclass

    from yialpha.ledger.outcomes import all_outcomes
    from yialpha.ledger.sqlite import ledger_exists

    if not ledger_exists() or not _ledger_readable():
        return {"available": False, "hint": "yialpha scoreboard", "outcomes": []}
    return {
        "available": True,
        # Round-17: the outcome legs are not validated for finiteness on the
        # write side (``inf`` is stored as-is); one such row is nulled rather
        # than failing the whole render.
        "outcomes": [
            sanitize_nonfinite(
                asdict(row) if is_dataclass(row) and not isinstance(row, type) else dict(row)  # see load_run_predictions
            )
            for row in all_outcomes(limit=limit)
        ],
    }


def load_calibration() -> dict:
    """Prediction calibration scoreboard (read-only; scoring is a CLI job).

    Mirrors :func:`load_accuracy_report`'s honesty contract: without the
    ledger DB (or before the operator ever runs ``yialpha scoreboard``),
    return ``available=False`` + the hint instead of a fabricated
    zero-everything table that would read as perfect calibration.

    Round-17: the scoreboard builder swallows ledger errors itself (round-10),
    so a present-but-unreadable ledger also reached the fabricated zero table;
    that answer is now reserved for a ledger that actually reads.
    """
    from yialpha.ledger.scoreboard import build_scoreboard
    from yialpha.ledger.sqlite import ledger_exists

    if not ledger_exists() or not _ledger_readable():
        return {"available": False, "hint": "yialpha scoreboard"}
    try:
        return {"available": True, **build_scoreboard()}
    except Exception:  # noqa: BLE001 -- read-only UI path degrades, never raises
        return {"available": False, "hint": "yialpha scoreboard"}


def load_ticket(ticket_id: str) -> dict | None:
    """One mirrored ticket payload by id (V2.4 read-only surface)."""
    import json as _json

    from yialpha.ledger.sqlite import get_connection, ledger_exists

    if not ledger_exists():
        return None
    try:
        row = get_connection(readonly=True).execute(
            "SELECT ticket_id, run_id, payload, ticket_version, written_at "
            "FROM tickets WHERE ticket_id = ?",
            (ticket_id,),
        ).fetchone()
    except Exception:  # noqa: BLE001 -- read-only UI path degrades, never raises
        return None
    if row is None:
        return None
    try:
        # Round-17: the mirror is dumped with ``json.dumps`` defaults, so a
        # ticket carrying a NaN/inf float lands here as a bare literal — null
        # it at parse time (same hooks as :func:`_read_json_object`); served
        # as-is it failed the response render.
        payload = _json.loads(
            row["payload"], parse_constant=_null_constant, parse_float=_null_overflow
        )
    except (TypeError, ValueError):
        payload = None
    return {
        "ticket_id": row["ticket_id"],
        "run_id": row["run_id"],
        "ticket_version": row["ticket_version"],
        "written_at": row["written_at"],
        "payload": payload,
    }


def load_portfolio_snapshot(snapshot_id: str) -> dict | None:
    """One portfolio snapshot by id (V2.4 read-only surface)."""
    from yialpha.ledger.portfolio import snapshot_by_id
    from yialpha.ledger.sqlite import ledger_exists

    if not ledger_exists():
        return None
    # Round-17: the snapshot payload is dumped with ``json.dumps`` defaults and
    # carries resolver/advisory floats — a NaN/inf must not fail the render.
    return sanitize_nonfinite(snapshot_by_id(snapshot_id))


def load_positions() -> dict:
    """Open positions + the resolver/pipeline state they came from."""
    from yialpha.ledger.portfolio import open_positions_with_status
    from yialpha.ledger.sqlite import ledger_exists

    if not ledger_exists():
        return {"available": False, "positions": []}
    rows, ok = open_positions_with_status()
    # Round-10: a DEGRADED read used to be served as available=True with
    # positions=[] — the phantom empty book; reflect the read status in the
    # flag (same honesty contract as load_outcomes / load_calibration).
    return {
        "available": ok,
        # Round-17: ``signed_weight`` is a REAL column that stores ``inf`` as-is
        # and the row payload embeds the same number — one such position 500'd
        # the whole book.
        "positions": sanitize_nonfinite(rows),
    }


def load_run(ticker: str, date: str) -> dict | None:
    """Full report view: rating badge + overlay card + 5 collapsible sections."""
    path = _strategy_dir(ticker) / f"full_states_log_{date}.json"
    if not path.is_file():
        return None
    state = _read_json_object(path)
    if state is None:
        return None

    # Type-guarded: ``parse_rating`` / ``parse_overlay`` / ``.get`` on a
    # non-string decision or a non-object debate block raised (500) on a
    # malformed log; a malformed field reads as empty.
    final = _text(state.get("final_trade_decision"))
    rating = parse_rating(final)
    overlay = parse_overlay_local(final)

    ideb = _mapping(state.get("investment_debate_state"))
    rdeb = _mapping(state.get("risk_debate_state"))

    return {
        "ticker": ticker,
        "trade_date": state.get("trade_date", date),
        "rating": rating,
        # Asset class the run analyzed (stock default matches pre-asset_type
        # logs) — drives the venue badge on the report page so a BTCUSDT perp
        # run is visually distinguishable from its spot twin.
        "asset_type": state.get("asset_type") or "stock",
        "company_of_interest": state.get("company_of_interest", "") or "",
        "sections": {
            "market_report": state.get("market_report", "") or "",
            # Positioning Analyst only runs for crypto_perp with positioning_split
            # on (default off); null (not "") hides the card when absent.
            "positioning_report": state.get("positioning_report") or None,
            "sentiment_report": state.get("sentiment_report", "") or "",
            "news_report": state.get("news_report", "") or "",
            # Fundamentals Analyst is dropped for crypto (spot + perp); null hides
            # its section in the UI rather than rendering an empty card.
            "fundamentals_report": state.get("fundamentals_report") or None,
            "investment_debate": {
                "bull_history": ideb.get("bull_history", "") or "",
                "bear_history": ideb.get("bear_history", "") or "",
                "history": ideb.get("history", "") or "",
                "current_response": ideb.get("current_response", "") or "",
                "judge_decision": ideb.get("judge_decision", "") or "",
            },
            # JSON write key is trader_investment_decision (renamed from
            # trader_investment_plan in _log_state L705). Reading any other key
            # leaves this section blank.
            "trader_decision": state.get("trader_investment_decision", "") or "",
            "risk_debate": {
                "aggressive_history": rdeb.get("aggressive_history", "") or "",
                "conservative_history": rdeb.get("conservative_history", "") or "",
                "neutral_history": rdeb.get("neutral_history", "") or "",
                "history": rdeb.get("history", "") or "",
                "judge_decision": rdeb.get("judge_decision", "") or "",
            },
            "investment_plan": state.get("investment_plan", "") or "",
            "final_trade_decision": final,
        },
        "overlay": overlay,
        # Router's sentinel evidence for this run: rendered as the degraded-run
        # banner in the report view (None for logs written before the field
        # existed or for fully-fed runs).
        "data_quality": state.get("data_quality") or None,
        "node_perf": load_node_perf(ticker, date),
    }
