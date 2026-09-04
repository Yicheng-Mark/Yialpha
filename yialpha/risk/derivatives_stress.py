"""Derivatives Stress Score — deterministic crowding/flow state (V2.0 P0.4).

The perp market analyst used to face a dozen raw derivative numbers
(funding, OI, three long/short ratios, taker flow, basis) with no common
scale. This module compresses them into ONE structured, LLM-stable object —
``crowding_score`` on a 0–100 long-crowding scale plus labelled states and
explicit risk flags — because a single bare number invites misreading while
``{funding: extreme_long, oi: elevated, basis: positive, taker: buy}`` does
not.

Everything here is PURE: series in, report out, no I/O. The fail-open fetcher
that assembles the series lives in
:func:`yialpha.dataflows.binance.derivatives_stress_series`; the overlay
calls it live-perp-only and discloses when it could not run.

Component semantics (percentile of the LATEST value against its own trailing
window, inclusive):

- ``funding_pct`` — high funding = longs pay = crowded long.
- ``lsr_pct``    — high GLOBAL account long/short ratio = crowded long.
- ``basis_pct``  — high perp-over-index premium = long demand.
- ``oi_zscore``  — z-score of the latest OI vs its window (amplifier/state,
  not a direction: entering the score only through flags).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

#: A component needs at least this many trailing observations before its
#: percentile means anything; below it the component is dropped (listed in
#: ``missing``) rather than scored on noise.
MIN_WINDOW = 30
#: The intended window; shorter-but-valid windows still score but flag
#: ``thin_history`` so downstream readers know the estimate is noisier.
TARGET_WINDOW = 90

#: Weights of the direction components in the crowding score (sum to 1 over
#: the components actually available; OI deliberately absent — elevation is
#: not direction).
_SCORE_WEIGHTS = {"funding_pct": 0.35, "lsr_pct": 0.35, "basis_pct": 0.30}


@dataclass(frozen=True)
class DerivativesStressReport:
    """Structured stress output — exactly what the prompt/overlay renders."""

    crowding_score: int                      # 0 extreme-short crowd .. 100 extreme-long
    state: dict[str, str] = field(default_factory=dict)     # funding/oi/basis/taker
    risk_flags: list[str] = field(default_factory=list)
    components: dict[str, float] = field(default_factory=dict)
    windows: dict[str, int] = field(default_factory=dict)
    missing: list[str] = field(default_factory=list)


def _percentile_rank(series: pd.Series) -> tuple[float, int] | None:  # type: ignore[type-arg]
    """(mid-rank percentile 0–100 of the latest value vs its window, length).

    None when the window is too short (``< MIN_WINDOW``) or holds no usable
    numeric data. NaN rows are dropped first (a NaN compares False and would
    otherwise inflate the denominator). MID-RANK (ties count half) so a
    perfectly FLAT window scores the neutral 50 instead of a degenerate 100,
    while a monotonically rising window still lands near 100.
    """
    if series is None:
        return None
    s = series.dropna()
    if len(s) < MIN_WINDOW:
        return None
    if not pd.api.types.is_numeric_dtype(s):
        return None
    try:
        latest_f = float(s.iloc[-1])
    except (TypeError, ValueError):
        return None
    less = float((s < latest_f).sum())
    equal = float((s == latest_f).sum())
    pct = (less + 0.5 * equal) / len(s) * 100.0
    return pct, int(len(s))


def _zscore_latest(series: pd.Series) -> tuple[float, int] | None:  # type: ignore[type-arg]
    """(z-score of the latest value vs its window, window length) or None."""
    if series is None:
        return None
    s = series.dropna()
    if len(s) < MIN_WINDOW:
        return None
    if not pd.api.types.is_numeric_dtype(s):
        return None
    std = float(s.std(ddof=1))
    if std <= 0.0:
        return None
    z = (float(s.iloc[-1]) - float(s.mean())) / std
    return z, int(len(s))


def _funding_label(pct: float) -> str:
    if pct >= 90.0:
        return "extreme_long"
    if pct >= 70.0:
        return "elevated_long"
    if pct > 30.0:
        return "neutral"
    if pct > 10.0:
        return "elevated_short"
    return "extreme_short"


def _basis_label(pct: float) -> str:
    if pct >= 85.0:
        return "elevated_premium"
    if pct <= 15.0:
        return "deep_discount"
    if pct > 50.0:
        return "positive"
    return "negative"


def _taker_label(ratio: float | None) -> str:
    if ratio is None:
        return "unknown"
    if ratio >= 1.1:
        return "buy"
    if ratio <= 0.9:
        return "sell"
    return "balanced"


def compute_stress(
    funding: pd.Series | None = None,  # type: ignore[type-arg]
    global_lsr: pd.Series | None = None,  # type: ignore[type-arg]
    basis: pd.Series | None = None,  # type: ignore[type-arg]
    open_interest: pd.Series | None = None,  # type: ignore[type-arg]
    taker_ratio: float | None = None,
) -> DerivativesStressReport:
    """Build the report from trailing component series.

    Every component is optional and independently fail-open: a missing or too
    short series drops its component (``missing``) instead of failing the
    report. With ZERO direction components the score is the neutral 50 and
    ``insufficient_history`` is flagged — an honest "cannot say", never a
    fabricated crowd reading.
    """
    components: dict[str, float] = {}
    windows: dict[str, int] = {}
    missing: list[str] = []
    flags: list[str] = []

    pairs: list[tuple[str, pd.Series | None]] = [  # type: ignore[type-arg]
        ("funding_pct", funding),
        ("lsr_pct", global_lsr),
        ("basis_pct", basis),
    ]
    for name, series in pairs:
        result = _percentile_rank(series)
        if result is None:
            missing.append(name)
        else:
            components[name] = round(result[0], 1)
            windows[name] = result[1]

    oi_result = _zscore_latest(open_interest)
    if oi_result is None:
        missing.append("oi_zscore")
    else:
        components["oi_zscore"] = round(oi_result[0], 2)
        windows["oi_zscore"] = oi_result[1]

    available = {
        k: components[k] for k in _SCORE_WEIGHTS if k in components
    }
    total_w = sum(_SCORE_WEIGHTS[k] for k in available)
    if total_w > 0.0:
        score = round(sum(v * _SCORE_WEIGHTS[k] for k, v in available.items()) / total_w)
    else:
        # Zero direction components (OI alone is state, not direction): the
        # honest neutral 50 + explicit flag — never a ZeroDivisionError.
        score = 50
        flags.append("insufficient_history")

    if any(w < TARGET_WINDOW for w in windows.values()):
        flags.append("thin_history")

    funding_pct = components.get("funding_pct")
    basis_pct = components.get("basis_pct")
    oi_z = components.get("oi_zscore")

    if score >= 80:
        flags.append("crowded_long")
    if score <= 20:
        flags.append("crowded_short")
    if funding_pct is not None and funding_pct >= 90.0:
        flags.append("funding_expensive")
    if funding_pct is not None and funding_pct <= 10.0:
        flags.append("funding_favorable_shorts")
    if oi_z is not None and oi_z >= 2.0:
        flags.append("oi_elevated")

    state = {
        "funding": _funding_label(funding_pct) if funding_pct is not None else "unknown",
        "oi": (
            "elevated" if (oi_z or 0.0) >= 2.0
            else "expanding" if (oi_z or 0.0) >= 1.0
            else "contracting" if (oi_z or 0.0) <= -1.0
            else "normal"
        ) if oi_z is not None else "unknown",
        "basis": _basis_label(basis_pct) if basis_pct is not None else "unknown",
        "taker": _taker_label(taker_ratio),
    }
    if taker_ratio is None:
        missing.append("taker_ratio")

    return DerivativesStressReport(
        crowding_score=score,
        state=state,
        risk_flags=flags,
        components=components,
        windows=windows,
        missing=missing,
    )


def render_stress_line(report: DerivativesStressReport) -> str:
    """One overlay bullet summarising the report (deterministic text)."""
    s = report.state
    detail = (
        f"funding {s.get('funding', 'unknown')} · oi {s.get('oi', 'unknown')} · "
        f"basis {s.get('basis', 'unknown')} · taker {s.get('taker', 'unknown')}"
    )
    line = (
        f"- **Derivatives Stress**: crowding {report.crowding_score}/100 "
        f"({detail})"
    )
    if report.risk_flags:
        line += f" · flags: {', '.join(report.risk_flags)}"
    return line + "\n"
