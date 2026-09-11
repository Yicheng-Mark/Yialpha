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
  ``sum(|bin| / n * |accuracy(bin) - confidence(bin)|)``;
* ``reliability_bins`` — the per-bin data behind that ECE (non-empty bins
  only: ``bin_low`` / ``bin_high`` / ``n`` / ``mean_prob_up`` /
  ``accuracy``). Like every cell here these are display diagnostics shown
  with their sample size — a thin bin never becomes a weight.

Slices: overall, by analyst, by ``instrument_class`` (run row), by
``horizon_days``, by direction, by evidence-coverage bucket
(``len(evidence_ids)``: ``0`` / ``1-2`` / ``3+``), and by regime
(the prediction's ``regime_id``; rows without one bucket under
``"no_regime"`` — predictions written before V2.2 or with the regime
stage off never silently merge into a real regime cell).

**V3 discipline** (:data:`V3_MIN_SAMPLES_PER_CELL`): every cell with fewer
samples is flagged ``below_v3_min_samples`` and rendered with a display-only
note. There is NO weighting logic anywhere — a thin cell never silently
moves an aggregate; the reader sees the sample size next to every number.

The dict returned by :func:`build_scoreboard` is machine-readable (the web
API will serve it); :func:`render_scoreboard_markdown` renders the same
content for humans.
"""

from __future__ import annotations

import json
import math
from datetime import UTC, datetime, timedelta
from typing import Any

from yialpha.ledger.models import (
    DIRECTION_FLAT,
    SCOPE_CONTRACT,
    SCOPE_POSITIONING,
    SCOPE_UNDERLYING,
    decode_json_list,
)
from yialpha.ledger.sqlite import get_connection, ledger_exists
from yialpha.ledger.time_contract import aware_utc, expected_reference_date, validate_timing
from yialpha.versions import LEGACY_OUTCOME_VERSION, OUTCOME_COMPUTE_VERSION

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


def _finite(value: Any) -> bool:
    try:
        return value is not None and math.isfinite(float(value))
    except (TypeError, ValueError, OverflowError):
        return False


def _scoring_version(row: Any) -> str | None:
    """Eligible version, or None for a malformed/incomplete scored fact.

    Historical NULL metadata retains its original complete/net semantics.
    Versioned outcomes must prove a nonempty, closed holding window and all
    mandatory attribution legs. Diagnostic underlying/basis misses do not
    invalidate a contract calibration label.
    """
    if not _finite(row["net_return"]):
        return None
    raw = row["scoring_context"]
    if raw is None:
        return LEGACY_OUTCOME_VERSION if row["prediction_timing"] is None else None
    try:
        context = json.loads(raw)
        if not isinstance(context, dict):
            return None
        version = context.get("version")
        if version not in {LEGACY_OUTCOME_VERSION, OUTCOME_COMPUTE_VERSION}:
            return None
        if version == LEGACY_OUTCOME_VERSION and row["prediction_timing"] is not None:
            return None
        instants = [
            datetime.fromisoformat(context[key])
            for key in ("window_start", "window_end", "horizon_end")
        ]
        available = datetime.fromisoformat(row["outcome_available_at"])
        if any(instant.tzinfo is None or instant.utcoffset() is None for instant in [*instants, available]):
            return None
        start, end, horizon = instants
        if not start < end:
            return None
        if version == OUTCOME_COMPUTE_VERSION:
            # This is the fixed window's availability boundary, not the
            # later observation/computation time of the outcome row.
            if available != end:
                return None
        elif row["prediction_scope"] == SCOPE_POSITIONING:
            # Legacy funding labels exposed the final settlement instant,
            # which may precede a non-grid deadline (e.g. 08:00 vs 10:00).
            if not start < available <= end:
                return None
        elif available < end:
            return None
        # Legacy date labels admitted their named session close, even when
        # it followed the midnight-normalized deadline. Keep that behavior
        # visible in its own version; the new contract forbids overshoot.
        if version == OUTCOME_COMPUTE_VERSION and end > horizon:
            return None
        if version == OUTCOME_COMPUTE_VERSION:
            timing = validate_timing(json.loads(row["prediction_timing"]))
            if timing is None:
                return None
            formed = aware_utc(context["prediction_formed_at"])
            if formed != aware_utc(timing["prediction_formed_at"]):
                return None
            if int(row["horizon_days"]) != int(row["prediction_horizon_days"]):
                return None
            if horizon != formed + timedelta(days=int(row["horizon_days"])):
                return None
            if row["prediction_scope"] not in {SCOPE_CONTRACT, SCOPE_UNDERLYING, SCOPE_POSITIONING}:
                return None
            if row["prediction_scope"] == SCOPE_POSITIONING:
                if start != formed or end != horizon:
                    return None
            else:
                if timing["reference_price"] is None or timing["reference_error"] is not None:
                    return None
                if start != aware_utc(timing["reference_price_at"]):
                    return None
                if isinstance(context.get("reference_price"), bool):
                    return None
                if context.get("reference_price") != timing["reference_price"]:
                    return None
                if context.get("reference_source") != timing["reference_source"]:
                    return None
                for key in ("reference_available_at", "reference_observed_at"):
                    if aware_utc(context[key]) != aware_utc(timing[key]):
                        return None
        missing = set(decode_json_list(row["legs_missing"]))
        if row["prediction_scope"] == SCOPE_POSITIONING:
            if missing or not _finite(row["funding_pnl"]):
                return None
            return version if math.isclose(float(row["net_return"]), float(row["funding_pnl"]), abs_tol=1e-12) else None
        if version == OUTCOME_COMPUTE_VERSION:
            is_perp = row["asset_type"] == "crypto_perp" or "perp" in str(row["instrument_class"] or "")
            equity = not is_perp or (
                row["prediction_scope"] == SCOPE_UNDERLYING and row["instrument_class"] == "stock_perp"
            )
            source = "yfinance:1d:close" if equity else "binance_perp:1d:last"
            if context.get("reference_source") != source:
                return None
            if equity and context.get("reference_basis_check") != "matched_frozen_close":
                return None
            # The metadata must describe the same exact daily boundaries the
            # scorer admits, not merely any ordered historical window.
            for instant, cutoff in ((start, formed), (end, horizon)):
                expected_day = expected_reference_date(cutoff, equity=equity)
                expected = datetime.combine(expected_day + timedelta(days=1), datetime.min.time(), tzinfo=UTC)
                if instant != expected:
                    return None
        if missing - {"underlying", "underlying_close", "underlying_price_return", "basis_return"}:
            return None
        if not all(_finite(row[key]) for key in ("contract_price_return", "fees", "slippage")):
            return None
        if version == OUTCOME_COMPUTE_VERSION and any(
            float(row[key]) < 0 for key in ("fees", "slippage")
        ):
            return None
        funding = 0.0
        if row["prediction_scope"] == SCOPE_CONTRACT and (
            row["asset_type"] == "crypto_perp" or "perp" in str(row["instrument_class"] or "")
        ):
            if not _finite(row["funding_pnl"]):
                return None
            funding = float(row["funding_pnl"])
        expected_net = float(row["contract_price_return"]) + funding - float(row["fees"]) - float(row["slippage"])
        return version if math.isclose(float(row["net_return"]), expected_net, rel_tol=1e-9, abs_tol=1e-12) else None
    except (ValueError, TypeError, KeyError):
        return None


def _scored_rows() -> list[dict[str, Any]]:
    """Complete outcome rows joined to their prediction and run (literal SQL).

    Only eligible ``status='complete'`` rows with finite returns enter an
    aggregate. Versioned facts must also satisfy their attribution contract.
    """
    if not ledger_exists():
        return []
    connection = get_connection(readonly=True)
    columns = {row["name"] for row in connection.execute("PRAGMA table_info(outcomes)")}
    context_projection = "o.scoring_context" if "scoring_context" in columns else "NULL AS scoring_context"
    prediction_columns = {row["name"] for row in connection.execute("PRAGMA table_info(predictions)")}
    timing_projection = "p.timing AS prediction_timing" if "timing" in prediction_columns else "NULL AS prediction_timing"
    rows = (
        connection
        .execute(
            "SELECT o.prediction_id, o.horizon_days, o.net_return, o.contract_price_return, "
            "o.funding_pnl, o.fees, o.slippage, o.legs_missing, o.outcome_available_at, "
            f"{context_projection}, {timing_projection}, p.horizon_days AS prediction_horizon_days, "
            "p.analyst, p.direction, p.prob_up, p.evidence_ids, p.regime_id, "
            "p.prediction_scope, r.instrument_class, r.asset_type "
            "FROM outcomes o "
            "JOIN predictions p ON p.prediction_id = o.prediction_id "
            "JOIN runs r ON r.run_id = p.run_id "
            "WHERE o.status = 'complete' AND o.net_return IS NOT NULL "
            "ORDER BY p.analysis_as_of, o.prediction_id"
        )
        .fetchall()
    )
    scored = []
    for row in rows:
        version = _scoring_version(row)
        if version is None:
            continue
        scored.append({
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
            "regime_id": (
                str(row["regime_id"]) if row["regime_id"] else "no_regime"
            ),
            "scoring_version": version,
        })
    return scored


def _reliability_bins(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Per-bin reliability data for one cell; non-empty bins only.

    Same row set as the ECE (:func:`_prob_rows` — directional rows that
    carry a ``prob_up``), so the bins and the scalar always tell one story.
    """
    scored = _prob_rows(rows)
    if not scored:
        return []
    bins: list[list[dict[str, Any]]] = [[] for _ in range(_ECE_BINS)]
    for row in scored:
        bins[min(int(row["prob_up"] * _ECE_BINS), _ECE_BINS - 1)].append(row)
    out: list[dict[str, Any]] = []
    for index, bucket in enumerate(bins):
        if not bucket:
            continue
        out.append({
            "bin_low": index / _ECE_BINS,
            "bin_high": (index + 1) / _ECE_BINS,
            "n": len(bucket),
            "mean_prob_up": sum(row["prob_up"] for row in bucket) / len(bucket),
            "accuracy": sum(_realized(row) for row in bucket) / len(bucket),
        })
    return out


def _ece(rows: list[dict[str, Any]]) -> float | None:
    """Expected calibration error, derived from :func:`_reliability_bins`."""
    bins = _reliability_bins(rows)
    total = sum(bucket["n"] for bucket in bins)
    if not total:
        return None
    return sum(
        bucket["n"] / total * abs(bucket["accuracy"] - bucket["mean_prob_up"])
        for bucket in bins
    )


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
        "reliability_bins": _reliability_bins(rows),
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


def build_scoreboard(
    *, min_samples_display: int = 1, scoring_version: str | None = None
) -> dict[str, Any]:
    """Aggregate every complete outcome into the calibration scoreboard.

    ``min_samples_display`` drops ``by_*`` cells with fewer rows from the
    dict (the ``overall`` cell is always present); it is a display filter,
    never a scoring weight. Cells below :data:`V3_MIN_SAMPLES_PER_CELL` stay
    in the dict flagged ``below_v3_min_samples`` for the renderer to annotate.

    ``scoring_version`` optionally selects one known convention. Unfiltered
    mixed-version aggregates are explicitly labeled and split by version.
    """
    if scoring_version not in {None, LEGACY_OUTCOME_VERSION, OUTCOME_COMPUTE_VERSION}:
        raise ValueError(f"unknown scoring version {scoring_version!r}")
    rows = _scored_rows()
    if scoring_version is not None:
        rows = [row for row in rows if row["scoring_version"] == scoring_version]
    versions = sorted({row["scoring_version"] for row in rows})
    return {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "v3_min_samples_per_cell": V3_MIN_SAMPLES_PER_CELL,
        "min_samples_display": min_samples_display,
        "scoring_version_filter": scoring_version,
        "scoring_versions": versions,
        "mixed_scoring_versions": len(versions) > 1,
        "overall": _cell_metrics(rows),
        "by_analyst": _slice(rows, "analyst", min_samples_display),
        "by_instrument_class": _slice(rows, "instrument_class", min_samples_display),
        "by_horizon_days": _slice(rows, "horizon_days", min_samples_display),
        "by_direction": _slice(rows, "direction", min_samples_display),
        "by_evidence_bucket": _slice(rows, "evidence_n", min_samples_display),
        "by_regime": _slice(rows, "regime_id", min_samples_display),
        "by_scoring_version": _slice(rows, "scoring_version", min_samples_display),
    }


def _fmt(value: float | None, spec: str) -> str:
    """Format a metric or an em dash when the cell has no value."""
    return "—" if value is None else format(value, spec)


_V3_NOTE = "below V3 sample threshold — display only, no weight adjustment"


def _reliability_lines(bins: list[dict[str, Any]]) -> list[str]:
    """Markdown reliability table — the bins behind the overall ECE."""
    if not bins:
        return ["## Reliability", "", "_(no probability rows to bin)_", ""]
    lines = [
        "## Reliability",
        "",
        "| prob bin | n | mean prob_up | realized accuracy | gap |",
        "|---|---:|---:|---:|---:|",
    ]
    for bucket in bins:
        gap = bucket["accuracy"] - bucket["mean_prob_up"]
        lines.append(
            f"| [{bucket['bin_low']:.1f}, {bucket['bin_high']:.1f}) "
            f"| {bucket['n']} | {bucket['mean_prob_up']:.4f} "
            f"| {bucket['accuracy']:.1%} | {gap:+.4f} |"
        )
    return [*lines, ""]


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
        f"- Scoring versions: {', '.join(scoreboard.get('scoring_versions', [])) or 'none'}; "
        f"mixed versions: {str(scoreboard.get('mixed_scoring_versions', False)).lower()}. "
        "Use the version slices or scoring_version filter for one convention.",
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
        ("By regime", "regime_id", scoreboard["by_regime"]),
        ("By scoring version", "scoring_version", scoreboard.get("by_scoring_version", {})),
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
        if title == "Overall":
            lines += _reliability_lines(overall.get("reliability_bins") or [])
    return "\n".join(lines) + "\n"
