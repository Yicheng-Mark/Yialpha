"""Calibration scoreboard — how well did the blind predictions age?

:func:`build_scoreboard` joins every ``complete`` outcome row back to its
prediction (direction / ``prob_up`` / evidence chain) and its run
(``instrument_class``), then computes per-slice calibration metrics:

* ``directional_accuracy`` — hits over directional rows, where
  ``up`` hits when ``net_return > 0`` and ``down`` hits when
  ``net_return < 0``; ``flat`` rows are counted (``flat_n``) but excluded
  from the accuracy denominator (a flat call claims no direction);
* ``brier_score`` — mean ``(prob_up - realized_up)^2`` over directional rows
  with a ``prob_up`` (``realized_up`` = 1.0 iff ``net_return > 0``); rows
  missing ``prob_up`` are excluded and counted (``missing_prob_up``);
* ``log_loss`` — ``-mean(log(p or 1-p))`` on the same row set, with the
  probability clipped to ``[1e-6, 1 - 1e-6]`` so a confident wrong call is
  punished, not made infinite;
* ``calibration_error`` — ECE over 10 equal-width probability bins:
  ``sum(|bin| / n * |accuracy(bin) - confidence(bin)|)``.

Slices: overall, by analyst, by ``instrument_class`` (run row), by
``horizon_days``, by direction, and by evidence-coverage bucket
(``len(evidence_ids)``: ``0`` / ``1-2`` / ``3+``).

**V3 discipline** (:data:`V3_MIN_SAMPLES_PER_CELL`): every cell with fewer
samples is flagged ``below_v3_min_samples`` and rendered with a display-only
note. There is NO weighting logic anywhere — a thin cell never silently
moves an aggregate; the reader sees the sample size next to every number.

The dict returned by :func:`build_scoreboard` is machine-readable (the web
API will serve it); :func:`render_scoreboard_markdown` renders the same
content for humans.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime
from typing import Any

from yialpha.ledger.models import DIRECTION_FLAT, decode_json_list
from yialpha.ledger.sqlite import get_connection, ledger_exists

#: V3 discipline: a calibration cell with fewer samples than this is display
#: only — no weight is ever derived from it (no weighting logic exists here).
V3_MIN_SAMPLES_PER_CELL = 30

#: Probability clip for log loss — a confident wrong call must stay finite.
_PROB_CLIP = 1e-6

#: Number of equal-width probability bins for the ECE.
_ECE_BINS = 10

#: Evidence-count bucket boundaries (len(evidence_ids) -> bucket label).
_EVIDENCE_BUCKETS: tuple[tuple[int, str], ...] = ((0, "0"), (2, "1-2"))


def _evidence_bucket(count: int) -> str:
    """Coverage bucket for an evidence chain length: ``0`` / ``1-2`` / ``3+``."""
    for bound, label in _EVIDENCE_BUCKETS:
        if count <= bound:
            return label
    return "3+"


def _scored_rows() -> list[dict[str, Any]]:
    """Complete outcome rows joined to their prediction and run (literal SQL).

    Only ``status='complete'`` rows with a ``net_return`` are scoreable — an
    incomplete attribution leg must never enter a calibration aggregate.
    """
    if not ledger_exists():
        return []
    rows = (
        get_connection(readonly=True)
        .execute(
            "SELECT o.prediction_id, o.horizon_days, o.net_return, "
            "p.analyst, p.direction, p.prob_up, p.evidence_ids, "
            "r.instrument_class "
            "FROM outcomes o "
            "JOIN predictions p ON p.prediction_id = o.prediction_id "
            "JOIN runs r ON r.run_id = p.run_id "
            "WHERE o.status = 'complete' AND o.net_return IS NOT NULL "
            "ORDER BY p.analysis_as_of, o.prediction_id"
        )
        .fetchall()
    )
    return [
        {
            "prediction_id": str(row["prediction_id"]),
            "horizon_days": int(row["horizon_days"]),
            "net_return": float(row["net_return"]),
            "analyst": str(row["analyst"]),
            "direction": str(row["direction"]),
            "prob_up": (
                float(row["prob_up"]) if row["prob_up"] is not None else None
            ),
            "instrument_class": (
                str(row["instrument_class"]) if row["instrument_class"] else "unknown"
            ),
            "evidence_n": len(decode_json_list(row["evidence_ids"])),
        }
        for row in rows
    ]


def _ece(rows: list[dict[str, Any]]) -> float | None:
    """Expected calibration error over 10 equal-width probability bins."""
    scored = _prob_rows(rows)
    if not scored:
        return None
    bins: list[list[dict[str, Any]]] = [[] for _ in range(_ECE_BINS)]
    for row in scored:
        bins[min(int(row["prob_up"] * _ECE_BINS), _ECE_BINS - 1)].append(row)
    total = len(scored)
    error = 0.0
    for bucket in bins:
        if not bucket:
            continue
        accuracy = sum(_realized(row) for row in bucket) / len(bucket)
        confidence = sum(row["prob_up"] for row in bucket) / len(bucket)
        error += len(bucket) / total * abs(accuracy - confidence)
    return error


def _realized(row: dict[str, Any]) -> float:
    """Realized-up indicator: 1.0 iff the net return was positive."""
    return 1.0 if row["net_return"] > 0 else 0.0


def _directional(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Up/down rows only — flat rows claim no direction to score."""
    return [row for row in rows if row["direction"] != DIRECTION_FLAT]


def _prob_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Directional rows that carry a prob_up (the Brier/log-loss/ECE set)."""
    return [row for row in _directional(rows) if row["prob_up"] is not None]


def _cell_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """All calibration metrics for one slice cell (empty cells report None)."""
    directional = _directional(rows)
    hits = sum(
        1
        for row in directional
        if (row["direction"] == "up" and row["net_return"] > 0)
        or (row["direction"] == "down" and row["net_return"] < 0)
    )
    scored = _prob_rows(rows)
    brier = (
        sum((row["prob_up"] - _realized(row)) ** 2 for row in scored) / len(scored)
        if scored
        else None
    )
    log_loss = None
    if scored:
        clipped = [min(max(row["prob_up"], _PROB_CLIP), 1.0 - _PROB_CLIP) for row in scored]
        terms = [
            math.log(prob) if _realized(row) else math.log(1.0 - prob)
            for prob, row in zip(clipped, scored, strict=True)
        ]
        log_loss = -sum(terms) / len(terms)
    return {
        "n": len(rows),
        "directional_n": len(directional),
        "hits": hits,
        "directional_accuracy": hits / len(directional) if directional else None,
        "brier_score": brier,
        "log_loss": log_loss,
        "calibration_error": _ece(rows),
        "flat_n": len(rows) - len(directional),
        "missing_prob_up": len(directional) - len(scored),
        "below_v3_min_samples": len(rows) < V3_MIN_SAMPLES_PER_CELL,
    }


def _slice(
    rows: list[dict[str, Any]], key: str, min_samples_display: int
) -> dict[str, Any]:
    """Group rows by ``key`` into metric cells, dropping display-thin cells."""
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        label = str(row[key]) if key != "evidence_n" else _evidence_bucket(row[key])
        groups.setdefault(label, []).append(row)
    return {
        label: _cell_metrics(members)
        for label, members in sorted(groups.items())
        if len(members) >= min_samples_display
    }


def build_scoreboard(*, min_samples_display: int = 1) -> dict[str, Any]:
    """Aggregate every complete outcome into the calibration scoreboard.

    ``min_samples_display`` drops ``by_*`` cells with fewer rows from the
    dict (the ``overall`` cell is always present); it is a display filter,
    never a scoring weight. Cells below :data:`V3_MIN_SAMPLES_PER_CELL` stay
    in the dict flagged ``below_v3_min_samples`` for the renderer to annotate.
    """
    rows = _scored_rows()
    return {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "v3_min_samples_per_cell": V3_MIN_SAMPLES_PER_CELL,
        "min_samples_display": min_samples_display,
        "overall": _cell_metrics(rows),
        "by_analyst": _slice(rows, "analyst", min_samples_display),
        "by_instrument_class": _slice(rows, "instrument_class", min_samples_display),
        "by_horizon_days": _slice(rows, "horizon_days", min_samples_display),
        "by_direction": _slice(rows, "direction", min_samples_display),
        "by_evidence_bucket": _slice(rows, "evidence_n", min_samples_display),
    }


def _fmt(value: float | None, spec: str) -> str:
    """Format a metric or an em dash when the cell has no value."""
    return "—" if value is None else format(value, spec)


_V3_NOTE = "below V3 sample threshold — display only, no weight adjustment"


def render_scoreboard_markdown(scoreboard: dict[str, Any]) -> str:
    """Human-facing scoreboard: one table per slice, sample sizes everywhere."""
    overall = scoreboard["overall"]
    lines = [
        "# Prediction calibration scoreboard",
        "",
        f"- Generated: {scoreboard['generated_at']}  ",
        f"- Complete scored outcomes: {overall['n']} "
        f"(directional {overall['directional_n']}, flat {overall['flat_n']}, "
        f"missing prob_up {overall['missing_prob_up']})  ",
        f"- V3 discipline: cells with n < {scoreboard['v3_min_samples_per_cell']} "
        "are display only — no weight adjustment.",
        "",
    ]
    sections: list[tuple[str, str, dict[str, Any]]] = [
        ("Overall", "all", {"all": overall}),
        ("By analyst", "analyst", scoreboard["by_analyst"]),
        ("By instrument class", "instrument_class", scoreboard["by_instrument_class"]),
        ("By horizon (days)", "horizon_days", scoreboard["by_horizon_days"]),
        ("By direction", "direction", scoreboard["by_direction"]),
        ("By evidence coverage", "evidence bucket", scoreboard["by_evidence_bucket"]),
    ]
    for title, column, cells in sections:
        lines += [f"## {title}", ""]
        if not cells:
            lines += ["_(no rows)_", ""]
            continue
        lines += [
            f"| {column} | n | dir n | hits | accuracy | brier | log loss | ECE | note |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
        for label, cell in cells.items():
            note = _V3_NOTE if cell["below_v3_min_samples"] else ""
            lines.append(
                f"| {label} | {cell['n']} | {cell['directional_n']} "
                f"| {cell['hits']} | {_fmt(cell['directional_accuracy'], '.1%')} "
                f"| {_fmt(cell['brier_score'], '.4f')} "
                f"| {_fmt(cell['log_loss'], '.4f')} "
                f"| {_fmt(cell['calibration_error'], '.4f')} | {note} |"
            )
        lines.append("")
    return "\n".join(lines) + "\n"
