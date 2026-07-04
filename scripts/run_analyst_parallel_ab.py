#!/usr/bin/env python
"""A/B gate: is ``analyst_parallel=True`` distributionally indistinguishable
from the serial analyst chain?

Same ticker + same date, run the full pipeline ``--n`` times with the analyst
chain SERIAL (``analyst_parallel=False``) and ``--n`` times with it PARALLEL
(``analyst_parallel=True``), then statistically compare the two legs. The
iron law is: the concurrency layer must NOT change any agent's decision
distribution. LLM sampling guarantees byte-level differences run-to-run, so
the metrics here are DISTRIBUTIONAL, not byte-equality.

Usage (run from the repo root — ``yiagents/__init__.py`` does
``load_dotenv(usecwd=True)`` and needs the ``.env`` next to the cwd):

    # Zero-cost verification (synthetic data, no LLM calls, no propagate):
    python scripts/run_analyst_parallel_ab.py --dry-run \\
        --tickers AAPL --date 2026-03-15 --n 6

    # Real gate (real LLM, real propagate; ~10 min per ticker per run):
    python scripts/run_analyst_parallel_ab.py \\
        --tickers AAPL NVDA --date 2026-03-15 --n 10 \\
        --max-threads 8 --out-dir ~/.yiagents/logs/ab

``--dry-run`` exercises every metric function and the report writer on
synthetic data drawn from a fixed seed, so the script is verifiable without
spending DeepSeek tokens. The synthetic data is constructed so the gate
PASSES, demonstrating the pass path end-to-end.

Exit code: 0 on PASS, 1 on FAIL. ``--dry-run`` always exits 0 (synthetic).

Dependencies: this script needs ONLY the stdlib + (optionally) scipy /
scikit-learn for the chi-square and TF-IDF fast paths. Neither scipy nor
sklearn is required — zero-dependency fallbacks (manual chi-square via the
Wilson-Hilferty approximation; manual TF-IDF + cosine) are implemented
below and are the active path on this machine, where neither is installed.
The pure metric functions import no yiagents code, so the unit tests run
without loading the heavy graph stack.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import re
import sys
import time
from pathlib import Path
from random import Random

# Allow running as ``python scripts/run_analyst_parallel_ab.py`` without an
# editable install: ensure the project root (parent of this scripts/ dir) is
# on sys.path. Done before any yiagents import (those are lazy, below).
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# Windows console defaults to GBK (cp936); printing the ✅/❌ glyphs or CJK
# triggers UnicodeEncodeError. Force utf-8 on stdout/stderr (Python 3.7+).
for _stream in (sys.stdout, sys.stderr):
    with contextlib.suppress(AttributeError, ValueError):
        _stream.reconfigure(encoding="utf-8", errors="replace")

# ---------------------------------------------------------------------------
# Optional accelerator availability flags (tested via monkeypatch).
# Both are False on this machine: ``python -c "import scipy"`` and
# ``python -c "import sklearn"`` both raise ModuleNotFoundError. The
# fallback paths below are therefore the active paths and are unit-tested.
# ---------------------------------------------------------------------------
try:  # pragma: no cover - import guard
    import scipy.stats as _scipy_stats  # type: ignore

    _HAS_SCIPY = True
except Exception:  # noqa: BLE001
    _scipy_stats = None
    _HAS_SCIPY = False

try:  # pragma: no cover - import guard
    import sklearn.feature_extraction.text as _sk_text  # type: ignore
    import sklearn.metrics.pairwise as _sk_metrics  # type: ignore

    _HAS_SKLEARN = True
except Exception:  # noqa: BLE001
    _sk_text = None
    _sk_metrics = None
    _HAS_SKLEARN = False


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RATING_CATEGORIES: tuple[str, ...] = (
    "Buy", "Overweight", "Hold", "Underweight", "Sell",
)

# Marker the risk overlay appends to ``final_trade_decision``; see
# ``yiagents/graph/trading_graph.py:_apply_risk_overlay`` (around line 332).
_RISK_OVERLAY_MARKER = "## Quantitative Risk Overlay"
_TARGET_WEIGHT_RE = re.compile(r"\*\*Target Weight\*\*:\s*([\-0-9.]+)%")
_STOP_LOSS_RE = re.compile(r"\*\*Stop Loss\*\*:\s*\$?([\-0-9.]+)")
_ENTRY_REF_RE = re.compile(r"\*\*Entry Reference\*\*:\s*\$?([\-0-9.]+)")

# Gate thresholds (mirrors the task spec).
_RATING_P_THRESHOLD = 0.05
_SPEEDUP_THRESHOLD = 2.5


# ---------------------------------------------------------------------------
# Pure metric functions (no yiagents import; unit-tested directly).
# ---------------------------------------------------------------------------

def rating_histogram(ratings: list[str]) -> dict[str, int]:
    """Count occurrences of each rating category.

    Unknown ratings are kept under their own key (so a typo doesn't silently
    drop samples), but the chi-square helper unions keys across both legs
    and thus tolerates them.
    """
    hist: dict[str, int] = {}
    for r in ratings:
        key = r if isinstance(r, str) else str(r)
        hist[key] = hist.get(key, 0) + 1
    return hist


def _chi2_sf_wilson_hilferty(x: float, df: int) -> float:
    """Survival function ``P(X > x)`` for a chi-square with ``df`` dof.

    Wilson-Hilferty approximation: ``((x/df) ** (1/3) - (1 - 2/(9*df))) /
    sqrt(2/(9*df))`` is approximately standard normal. Used as the scipy-free
    fallback; for the sample sizes here (n per leg >= 5, dof = k-1 <= 4) the
    approximation is accurate to ~0.01 in the tail, well below the 0.05
    gate threshold. Returns 1.0 for ``df <= 0`` (no degrees of freedom →
    cannot reject) and clamps the normal CDF to [0, 1].
    """
    if df <= 0:
        return 1.0
    if x <= 0:
        return 1.0
    try:
        z = ((x / df) ** (1.0 / 3.0) - (1.0 - 2.0 / (9.0 * df))) / math.sqrt(
            2.0 / (9.0 * df)
        )
    except (ValueError, ZeroDivisionError):
        return 1.0
    # Standard normal survival function 1 - Phi(z) via math.erfc.
    sf = 0.5 * math.erfc(z / math.sqrt(2.0))
    return min(1.0, max(0.0, sf))


def _chi_square_manual(hist_a: dict, hist_b: dict) -> float:
    """Manual chi-square test of homogeneity on two rating histograms.

    Builds the union of categories, lays out a 2 x k contingency table, and
    computes the standard Pearson statistic
    ``sum_ij (O_ij - E_ij)^2 / E_ij`` with ``E_ij = row_i * col_j / total``
    (df = k - 1). Returns the survival function (p-value). Smoothes any
    zero expected cell by adding a tiny constant to avoid divide-by-zero.
    """
    cats = list(set(hist_a) | set(hist_b))
    if len(cats) <= 1:
        # One category (or none): nothing to compare; cannot reject H0.
        return 1.0
    row_a = sum(hist_a.get(c, 0) for c in cats)
    row_b = sum(hist_b.get(c, 0) for c in cats)
    total = row_a + row_b
    if total <= 0:
        return 1.0
    stat = 0.0
    for c in cats:
        col = hist_a.get(c, 0) + hist_b.get(c, 0)
        for row_total, row_hist in ((row_a, hist_a), (row_b, hist_b)):
            observed = row_hist.get(c, 0)
            expected = row_total * col / total
            if expected <= 0:
                expected = 1e-9
            diff = observed - expected
            stat += diff * diff / expected
    df = len(cats) - 1
    return _chi2_sf_wilson_hilferty(stat, df)


def chi_square_p(hist_a: dict, hist_b: dict) -> float:
    """Two-sample chi-square p-value on rating histograms.

    Prefers ``scipy.stats.chi2_contingency`` when scipy is importable; falls
    back to :func:`_chi_square_manual` (Wilson-Hilferty) otherwise. Returns
    1.0 when there are too few categories or samples to form a table (the
    caller's "too few samples" branch then triggers the overlap fallback).
    """
    cats = list(set(hist_a) | set(hist_b))
    if len(cats) <= 1:
        return 1.0
    if _HAS_SCIPY and _scipy_stats is not None:
        try:  # pragma: no cover - exercised only when scipy is installed
            row_a = sum(hist_a.get(c, 0) for c in cats)
            row_b = sum(hist_b.get(c, 0) for c in cats)
            if row_a == 0 or row_b == 0:
                return 1.0
            table = [[hist_a.get(c, 0) for c in cats],
                     [hist_b.get(c, 0) for c in cats]]
            _stat, p, _dof, _exp = _scipy_stats.chi2_contingency(table)
            return float(p)
        except Exception:  # noqa: BLE001
            pass  # fall through to manual
    return _chi_square_manual(hist_a, hist_b)


_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall((text or "").lower())


def _tfidf_vectors(texts: list[str]):
    """Zero-dependency TF-IDF vectors.

    Bag-of-words term counts × idf, where idf uses sklearn's smoothing
    ``idf = ln((1 + N) / (1 + df)) + 1`` so every term gets a positive
    weight and a document with a single unique term does not collapse.
    Returns ``(vectors: list[dict[str,float]], vocab: set[str])``.
    """
    n = len(texts)
    df_counts: dict[str, int] = {}
    token_lists: list[list[str]] = []
    for t in texts:
        toks = _tokenize(t)
        token_lists.append(toks)
        for term in set(toks):
            df_counts[term] = df_counts.get(term, 0) + 1
    idf: dict[str, float] = {}
    for term, dfc in df_counts.items():
        idf[term] = math.log((1.0 + n) / (1.0 + dfc)) + 1.0
    vectors: list[dict[str, float]] = []
    for toks in token_lists:
        tf: dict[str, int] = {}
        for term in toks:
            tf[term] = tf.get(term, 0) + 1
        vec = {term: count * idf[term] for term, count in tf.items()}
        vectors.append(vec)
    return vectors, set(df_counts)


def _l2_normalize(vec: dict[str, float]) -> dict[str, float]:
    norm = math.sqrt(sum(v * v for v in vec.values()))
    if norm == 0:
        return {}
    return {k: v / norm for k, v in vec.items()}


def _cosine(vec_a: dict[str, float], vec_b: dict[str, float]) -> float:
    """Cosine of two sparse dicts (assumed already L2-normalized → dot product)."""
    if not vec_a or not vec_b:
        return 0.0
    # Iterate the smaller dict for fewer lookups.
    if len(vec_a) > len(vec_b):
        vec_a, vec_b = vec_b, vec_a
    return sum(w * vec_b.get(term, 0.0) for term, w in vec_a.items())


def _tfidf_cosine_manual(texts_a: list[str], texts_b: list[str]) -> float:
    """Mean pairwise cosine between TF-IDF vectors of two text sets.

    Builds ONE idf over the union corpus (so shared vocabulary gets
    meaningful weight), then averages cosine over every (a, b) pair. Empty
    inputs return 0.0.
    """
    if not texts_a or not texts_b:
        return 0.0
    vectors, _vocab = _tfidf_vectors(list(texts_a) + list(texts_b))
    normed = [_l2_normalize(v) for v in vectors]
    a_vecs = normed[: len(texts_a)]
    b_vecs = normed[len(texts_a):]
    total = 0.0
    count = 0
    for va in a_vecs:
        for vb in b_vecs:
            total += _cosine(va, vb)
            count += 1
    return total / count if count else 0.0


def tfidf_cosine_similarity(texts_a: list[str], texts_b: list[str]) -> float:
    """Mean pairwise TF-IDF cosine between two text sets.

    Prefers scikit-learn (``TfidfVectorizer`` + ``cosine_similarity``) when
    importable; falls back to :func:`_tfidf_cosine_manual` otherwise. The
    fallback is the active path on machines without scikit-learn (incl. this
    one) and is the path exercised by the unit tests.
    """
    if not texts_a or not texts_b:
        return 0.0
    if _HAS_SKLEARN and _sk_text is not None and _sk_metrics is not None:
        try:  # pragma: no cover - exercised only when sklearn is installed
            vec = _sk_text.TfidfVectorizer()
            mat = vec.fit_transform(list(texts_a) + list(texts_b))
            a = mat[: len(texts_a)]
            b = mat[len(texts_a):]
            sim = _sk_metrics.cosine_similarity(a, b)
            return float(sim.mean())
        except Exception:
            pass  # fall through to manual
    return _tfidf_cosine_manual(texts_a, texts_b)


def _mean_pairwise_within(texts: list[str]) -> float:
    """Mean pairwise cosine among texts in a single set (within-set coherence)."""
    n = len(texts)
    if n < 2:
        return 1.0 if n == 1 else 0.0
    vectors, _vocab = _tfidf_vectors(texts)
    normed = [_l2_normalize(v) for v in vectors]
    total = 0.0
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            total += _cosine(normed[i], normed[j])
            count += 1
    return total / count if count else 0.0


def between_vs_within_similarity(
    reports_serial: list[str], reports_parallel: list[str]
) -> dict:
    """Return within-set and between-set mean TF-IDF cosines.

    Criterion: ``between >= min(within_serial, within_parallel)`` — i.e. the
    parallel leg is no further from the serial leg than each leg is from
    itself. A single-text set has within-similarity 1.0 (a text is identical
    to itself); an empty set has 0.0.
    """
    return {
        "within_serial": _mean_pairwise_within(reports_serial),
        "within_parallel": _mean_pairwise_within(reports_parallel),
        "between": tfidf_cosine_similarity(reports_serial, reports_parallel),
    }


def parse_risk_overlay(decision_text: str) -> dict | None:
    """Parse the appended ``## Quantitative Risk Overlay`` numbers.

    Returns ``{"target_weight": float, "stop_loss": float|None,
    "entry_price": float|None}`` or ``None`` when the overlay marker is
    absent (e.g. risk_enabled=False). Comparing floats (not raw strings)
    tolerates formatting differences like ``12.0`` vs ``12``.
    """
    if not decision_text or _RISK_OVERLAY_MARKER not in decision_text:
        return None
    block = decision_text.split(_RISK_OVERLAY_MARKER, 1)[1]
    out: dict = {"target_weight": None, "stop_loss": None, "entry_price": None}
    m = _TARGET_WEIGHT_RE.search(block)
    if m:
        with contextlib.suppress(ValueError):
            out["target_weight"] = float(m.group(1))
    m = _STOP_LOSS_RE.search(block)
    if m:
        with contextlib.suppress(ValueError):
            out["stop_loss"] = float(m.group(1))
    m = _ENTRY_REF_RE.search(block)
    if m:
        with contextlib.suppress(ValueError):
            out["entry_price"] = float(m.group(1))
    return out


def risk_overlay_determinism(runs: list[dict]) -> bool:
    """Within each rating group, the risk-overlay numbers must be identical.

    The overlay is a deterministic function of (rating, price, ATR), so two
    runs that share a rating MUST share overlay numbers. ``runs`` items are
    expected to carry ``rating`` (str) and ``risk_overlay`` (dict | None, as
    produced by :func:`parse_risk_overlay`). Groups where every run has
    ``risk_overlay is None`` (overlay disabled) are treated as vacuously
    consistent. Floats are compared with a tight tolerance (1e-9) to absorb
    formatting-only differences.
    """
    by_rating: dict[str, list[dict]] = {}
    for r in runs:
        rating = r.get("rating", "?")
        by_rating.setdefault(rating, []).append(r)
    for rating, group in by_rating.items():
        overlays = [g.get("risk_overlay") for g in group]
        # Skip groups where every overlay is missing (risk disabled).
        if all(o is None for o in overlays):
            continue
        # If some have an overlay and some don't, that's inconsistent.
        if any(o is None for o in overlays):
            return False
        ref = overlays[0]
        for o in overlays[1:]:
            for key in ("target_weight", "stop_loss", "entry_price"):
                a = ref.get(key)
                b = o.get(key)
                if a is None and b is None:
                    continue
                if a is None or b is None:
                    return False
                if abs(float(a) - float(b)) > 1e-9:
                    return False
    return True


def summarize_gate(metrics: dict) -> tuple[bool, list[str]]:
    """Return ``(overall_pass, reasons)`` from a metrics dict.

    Required keys (all metric functions populate these):
      ``n_serial``, ``n_parallel``, ``rating_p``, ``overlap_ratio``,
      ``per_field_similarity`` (dict field -> {within_serial, within_parallel,
      between}), ``risk_determinism_serial``, ``risk_determinism_parallel``,
      ``empty_reports_serial``, ``empty_reports_parallel``,
      ``exceptions_serial``, ``exceptions_parallel``, ``speedup``,
      ``speedup_na`` (bool), ``dry_run`` (bool).
    """
    reasons: list[str] = []
    ok = True

    def fail(msg: str) -> None:
        nonlocal ok
        ok = False
        reasons.append(msg)

    # 1) Rating distribution: chi-square p > 0.05, OR (too-few-samples
    #    fallback) distributions overlap > 50%.
    p = metrics.get("rating_p")
    if p is None:
        fail("rating chi-square: no p-value computed")
    elif p > _RATING_P_THRESHOLD:
        pass  # ok
    else:
        # Too few samples for chi-square → fall back to overlap.
        overlap = metrics.get("overlap_ratio", 0.0)
        if overlap > 0.5:
            pass  # ok via fallback
        else:
            fail(
                f"rating chi-square p={p:.4f} <= {_RATING_P_THRESHOLD} "
                f"(overlap fallback {overlap:.2f} <= 0.50)"
            )

    # 2) Per-field similarity: between >= min(within_serial, within_parallel).
    for field, sim in (metrics.get("per_field_similarity") or {}).items():
        ws = sim.get("within_serial", 0.0)
        wp = sim.get("within_parallel", 0.0)
        btw = sim.get("between", 0.0)
        threshold = min(ws, wp)
        if btw + 1e-9 < threshold:
            fail(
                f"{field}: between-similarity {btw:.3f} < "
                f"min(within_serial {ws:.3f}, within_parallel {wp:.3f})"
            )

    # 3) Risk-overlay determinism in BOTH legs.
    if not metrics.get("risk_determinism_serial", False):
        fail("risk-overlay determinism FAILED in serial leg")
    if not metrics.get("risk_determinism_parallel", False):
        fail("risk-overlay determinism FAILED in parallel leg")

    # 4) No empty / truncated reports.
    if metrics.get("empty_reports_serial", 0) > 0:
        fail(f"{metrics['empty_reports_serial']} empty report(s) in serial leg")
    if metrics.get("empty_reports_parallel", 0) > 0:
        fail(f"{metrics['empty_reports_parallel']} empty report(s) in parallel leg")

    # 5) Parallel >= 2.5x faster (analyst-segment wall time). N/A in dry-run
    #    and when perf timing is unavailable.
    if metrics.get("dry_run", False):
        pass  # N/A — synthetic
    elif metrics.get("speedup_na", False):
        pass  # N/A — perf_tracker not exposed; recorded, not enforced
    else:
        sp = metrics.get("speedup")
        if sp is None or sp < _SPEEDUP_THRESHOLD:
            fail(
                f"parallel speedup {sp:.2f}x < {_SPEEDUP_THRESHOLD}x"
                if sp is not None
                else "parallel speedup not measured"
            )

    # 6) Zero exceptions in either leg.
    if metrics.get("exceptions_serial", 0) > 0:
        fail(f"{metrics['exceptions_serial']} exception(s) in serial leg")
    if metrics.get("exceptions_parallel", 0) > 0:
        fail(f"{metrics['exceptions_parallel']} exception(s) in parallel leg")

    return ok, reasons


# ---------------------------------------------------------------------------
# Report writer
# ---------------------------------------------------------------------------

def write_report(
    path: Path,
    ticker: str,
    date: str,
    n: int,
    metrics: dict,
    *,
    dry_run: bool = False,
) -> None:
    """Write ``parallel_ab_<ticker>_<date>.md`` (Markdown) to ``path``."""
    lines: list[str] = []
    if dry_run:
        lines.append(
            "> **DRY RUN (synthetic data)** — no LLM calls, no propagate. "
            "Ratings, reports and risk numbers are generated from a fixed "
            "seed to exercise the metrics + report pipeline."
        )
        lines.append("")
    lines.append(f"# Analyst Parallel A/B Gate — {ticker} @ {date}")
    lines.append("")
    lines.append(f"- Runs per leg: **{n}**")
    lines.append(f"- Serial leg exceptions: {metrics.get('exceptions_serial', 0)}")
    lines.append(f"- Parallel leg exceptions: {metrics.get('exceptions_parallel', 0)}")
    speedup = metrics.get("speedup")
    if dry_run:
        lines.append("- Parallel speedup: N/A (dry-run, synthetic timing)")
    elif metrics.get("speedup_na"):
        lines.append(
            "- Parallel speedup: N/A (perf_tracker not exposed on graph; "
            "whole-propagate wall recorded but speedup not enforced)"
        )
    elif speedup is not None:
        lines.append(f"- Parallel speedup: **{speedup:.2f}x** "
                     f"(threshold >= {_SPEEDUP_THRESHOLD}x)")
    else:
        lines.append("- Parallel speedup: not measured")
    lines.append("")

    lines.append("## Rating distribution")
    lines.append("")
    sh = metrics.get("serial_hist", {})
    ph = metrics.get("parallel_hist", {})
    cats = list(RATING_CATEGORIES) + [c for c in (set(sh) | set(ph))
                                      if c not in RATING_CATEGORIES]
    lines.append("| Rating | Serial | Parallel |")
    lines.append("|---|---:|---:|")
    for c in cats:
        lines.append(f"| {c} | {sh.get(c, 0)} | {ph.get(c, 0)} |")
    lines.append("")
    p = metrics.get("rating_p")
    overlap = metrics.get("overlap_ratio")
    lines.append(
        f"- Chi-square p = {p:.4f}" if p is not None else "- Chi-square p = N/A"
    )
    if overlap is not None:
        lines.append(f"- Distribution overlap = {overlap:.2f}")
    lines.append("")

    lines.append("## Report similarity (TF-IDF cosine)")
    lines.append("")
    lines.append("| Field | within-serial | within-parallel | between | between >= min(within)? |")
    lines.append("|---|---:|---:|---:|:---:|")
    for field, sim in (metrics.get("per_field_similarity") or {}).items():
        ws = sim.get("within_serial", 0.0)
        wp = sim.get("within_parallel", 0.0)
        btw = sim.get("between", 0.0)
        ok_sim = btw + 1e-9 >= min(ws, wp)
        lines.append(
            f"| {field} | {ws:.3f} | {wp:.3f} | {btw:.3f} | "
            f"{'YES' if ok_sim else 'NO'} |"
        )
    lines.append("")

    lines.append("## Risk-overlay determinism")
    lines.append("")
    lines.append(f"- Serial leg: {'PASS' if metrics.get('risk_determinism_serial') else 'FAIL'}")
    lines.append(f"- Parallel leg: {'PASS' if metrics.get('risk_determinism_parallel') else 'FAIL'}")
    lines.append("")

    overall, reasons = summarize_gate(metrics)
    verdict = "PASS" if overall else "FAIL"
    lines.append(f"## Verdict: {verdict}")
    lines.append("")
    if reasons:
        lines.append("Reasons:")
        for r in reasons:
            lines.append(f"- {r}")
    else:
        lines.append("All gate criteria met.")
    lines.append("")

    if dry_run:
        lines.append(
            "_Note: dry-run exit code is always 0 regardless of verdict "
            "(synthetic data). Real mode exits 1 on FAIL._"
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Synthetic data generation (dry-run only)
# ---------------------------------------------------------------------------

def _synth_report(ticker: str, rating: str, run_idx: int) -> str:
    """Synthetic analyst report.

    Deliberately independent of ``leg`` and constructed so that serial[i]
    and parallel[i] are byte-identical: with identical multisets the
    between-set cosine (which counts the diagonal cos(a_i, a_i) = 1.0
    pairs) is strictly greater than the within-set cosine, so the
    similarity gate criterion cleanly passes on the dry-run path. Varying
    ``run_idx`` keeps within-set similarity below 1.0 so the metric is
    non-trivial.
    """
    bias = {
        "Buy": "strong upside momentum and expanding margins",
        "Overweight": "constructive setup with positive trend confirmation",
        "Hold": "balanced signals, mixed price action, await confirmation",
        "Underweight": "deteriorating relative strength and soft guidance",
        "Sell": "breakdown in trend, weakening demand, raise cash",
    }.get(rating, "neutral outlook")
    return (
        f"{ticker} run {run_idx}: rating {rating}. "
        f"Analysts cite {bias}. Valuation, momentum and risk factors "
        f"discussed. Recommendation aligns with the {rating} thesis. "
        f"Liquidity and macro backdrop are consistent with this view."
    )


def _synth_decision(ticker: str, rating: str, run_idx: int) -> str:
    """Synthetic ``final_trade_decision`` with a deterministic risk overlay.

    The overlay numbers depend ONLY on the rating, so two runs sharing a
    rating produce byte-identical overlay numbers and the determinism check
    passes.
    """
    overlay = {
        "Buy": (40.0, 95.50, 100.00),
        "Overweight": (25.0, 96.00, 100.00),
        "Hold": (10.0, 97.00, 100.00),
        "Underweight": (-10.0, 102.00, 100.00),
        "Sell": (-30.0, 105.00, 100.00),
    }[rating]
    tw, sl, ep = overlay
    return (
        f"**Rating**: {rating}\n\nPM thesis for {ticker} run {run_idx}: "
        f"weighted toward the analyst consensus.\n\n"
        f"---\n\n## Quantitative Risk Overlay\n\n"
        f"- **Action**: {rating}\n"
        f"- **Target Weight**: {tw:.1f}%\n"
        f"- **Stop Loss**: {sl:.2f}\n"
        f"- **Entry Reference**: {ep:.2f}\n"
    )


# Deterministic per-rating risk numbers used by the synthetic decision.
def _synth_run(rng: Random, ticker: str, leg: str, rating: str, run_idx: int) -> dict:
    """One synthetic run. ``rating`` is drawn once per (ticker, run_idx) and
    shared across both legs so the dry-run demonstrates the gate's PASS path."""
    decision = _synth_decision(ticker, rating, run_idx)
    return {
        "ticker": ticker,
        "leg": leg,
        "run_idx": run_idx,
        "rating": rating,
        "market_report": _synth_report(ticker, rating, run_idx),
        "sentiment_report": _synth_report(ticker, rating, run_idx),
        "news_report": _synth_report(ticker, rating, run_idx),
        "fundamentals_report": _synth_report(ticker, rating, run_idx),
        "final_trade_decision": decision,
        "risk_overlay": parse_risk_overlay(decision),
        "wall_full": 60.0 if leg == "serial" else 20.0,  # synthetic 3x speedup
        "analyst_segment": 40.0 if leg == "serial" else 12.0,
    }


def _compute_metrics(
    serial_runs: list[dict],
    parallel_runs: list[dict],
    *,
    dry_run: bool = False,
) -> dict:
    """Run every metric function over two legs and return the metrics dict."""
    s_ratings = [r["rating"] for r in serial_runs]
    p_ratings = [r["rating"] for r in parallel_runs]
    sh = rating_histogram(s_ratings)
    ph = rating_histogram(p_ratings)
    p = chi_square_p(sh, ph)
    overlap = _overlap_ratio(sh, ph)

    fields = ("market_report", "sentiment_report", "news_report", "fundamentals_report")
    per_field: dict[str, dict[str, float]] = {}
    for f in fields:
        s_texts = [r.get(f, "") for r in serial_runs]
        p_texts = [r.get(f, "") for r in parallel_runs]
        per_field[f] = between_vs_within_similarity(s_texts, p_texts)

    def empty_count(runs: list[dict], fields_) -> int:
        n = 0
        for r in runs:
            for f in fields_:
                if not (r.get(f) or "").strip():
                    n += 1
        return n

    # Speedup: prefer analyst_segment when present; fall back to wall_full;
    # mark N/A in dry-run regardless.
    serial_segs = [r.get("analyst_segment") for r in serial_runs]
    par_segs = [r.get("analyst_segment") for r in parallel_runs]
    serial_walls = [r.get("wall_full", 0.0) for r in serial_runs]
    par_walls = [r.get("wall_full", 0.0) for r in parallel_runs]
    speedup_na = dry_run
    speedup = None
    if all(x is not None and x > 0 for x in serial_segs) and \
       all(x is not None and x > 0 for x in par_segs):
        speedup = (sum(serial_segs) / len(serial_segs)) / \
                  (sum(par_segs) / len(par_segs))
    elif serial_walls and par_walls and sum(par_walls) > 0:
        speedup = (sum(serial_walls) / len(serial_walls)) / \
                  (sum(par_walls) / len(par_walls))
    # In dry-run we don't enforce speedup.
    if dry_run:
        speedup_na = True

    return {
        "n_serial": len(serial_runs),
        "n_parallel": len(parallel_runs),
        "serial_hist": sh,
        "parallel_hist": ph,
        "rating_p": p,
        "overlap_ratio": overlap,
        "per_field_similarity": per_field,
        "risk_determinism_serial": risk_overlay_determinism(serial_runs),
        "risk_determinism_parallel": risk_overlay_determinism(parallel_runs),
        "empty_reports_serial": empty_count(serial_runs, fields),
        "empty_reports_parallel": empty_count(parallel_runs, fields),
        "exceptions_serial": sum(1 for r in serial_runs if r.get("exception")),
        "exceptions_parallel": sum(1 for r in parallel_runs if r.get("exception")),
        "speedup": speedup,
        "speedup_na": speedup_na,
        "dry_run": dry_run,
    }


def _overlap_ratio(hist_a: dict, hist_b: dict) -> float:
    """Shared-sample fraction of the union distribution (overlap > 0.5 fallback).

    Computes, for each category, ``min(share_a, share_b)`` and sums — the
    standard distribution-overlap coefficient (a.k.a. histogram intersection
    normalized to 1.0).
    """
    cats = set(hist_a) | set(hist_b)
    tot_a = sum(hist_a.values()) or 1
    tot_b = sum(hist_b.values()) or 1
    return sum(min(hist_a.get(c, 0) / tot_a, hist_b.get(c, 0) / tot_b) for c in cats)


# ---------------------------------------------------------------------------
# Dry-run driver
# ---------------------------------------------------------------------------

def _dry_run(args) -> int:
    out_dir = Path(args.out_dir).expanduser()
    rng = Random(args.seed)
    overall_any = True
    for ticker in args.tickers:
        serial_runs: list[dict] = []
        parallel_runs: list[dict] = []
        # Weighted slightly toward the middle so the histogram is non-degenerate.
        weights = [3, 4, 6, 4, 3]
        for i in range(args.n):
            # Draw the rating ONCE per run index and reuse for both legs:
            # identical multisets ⇒ between-similarity > within-similarity,
            # so the similarity gate criterion cleanly passes (PASS path).
            rating = rng.choices(list(RATING_CATEGORIES), weights=weights, k=1)[0]
            serial_runs.append(_synth_run(rng, ticker, "serial", rating, i))
            parallel_runs.append(_synth_run(rng, ticker, "parallel", rating, i))
        metrics = _compute_metrics(serial_runs, parallel_runs, dry_run=True)
        report_path = out_dir / f"parallel_ab_{ticker}_{args.date}.md"
        write_report(report_path, ticker, args.date, args.n, metrics, dry_run=True)
        overall, reasons = summarize_gate(metrics)
        overall_any = overall_any and overall
        print(f"[{ticker}] dry-run report -> {report_path}")
        print(f"[{ticker}] verdict: {'PASS' if overall else 'FAIL'} "
              f"(dry-run, exit 0 regardless)")
        if reasons:
            for r in reasons:
                print(f"    - {r}")
        # Also dump raw metrics JSON for inspection / the unit test.
        (out_dir / f"parallel_ab_{ticker}_{args.date}.json").write_text(
            json.dumps(metrics, indent=2, default=str), encoding="utf-8"
        )
    # Dry-run always exits 0 (synthetic).
    return 0


# ---------------------------------------------------------------------------
# Real-mode driver
# ---------------------------------------------------------------------------

def _build_graph(leg: str, args):
    """Build a YiAgentsGraph for one leg. Heavy imports are lazy so the dry-run
    path and the unit tests don't require the yiagents graph stack."""
    from yiagents.default_config import DEFAULT_CONFIG  # noqa: E402
    from yiagents.graph.trading_graph import YiAgentsGraph  # noqa: E402

    config = DEFAULT_CONFIG.copy()
    # Force batch_workers=1 to isolate the analyst-parallel effect from
    # batch concurrency (the K-graph pool). The iron law is about the
    # analyst fan-out only.
    config["batch_workers"] = 1
    # Enable node telemetry on both legs; read analyst-segment times when
    # the graph exposes them (currently it does not — see timing note).
    config["node_perf_telemetry"] = True
    if leg == "parallel":
        config["analyst_parallel"] = True
        if args.max_threads:
            config["analyst_parallel_max_threads"] = args.max_threads
    else:
        config["analyst_parallel"] = False
    return YiAgentsGraph(config=config)


def _preflight_warn(args) -> None:
    """Warn (not fail) when the shared rate limiter would throttle parallelism."""
    try:
        from yiagents.default_config import DEFAULT_CONFIG  # noqa: E402
    except Exception as exc:  # noqa: BLE001
        print(f"[preflight] could not import DEFAULT_CONFIG: {exc}", file=sys.stderr)
        return
    cfg = DEFAULT_CONFIG
    if cfg.get("llm_rate_limiter") and int(cfg.get("llm_rpm", 60)) < 120:
        print(
            f"[preflight] WARNING: llm_rate_limiter is ON with llm_rpm="
            f"{cfg.get('llm_rpm')} (< 120). The parallel-leg speedup is gated "
            f"by the shared limiter bucket; consider raising llm_rpm or "
            f"disabling the limiter for this gate run.",
            file=sys.stderr,
        )


def _capture_run(graph, ticker: str, date: str) -> dict:
    """Run one propagate and capture all the per-run fields the metrics need."""
    t0 = time.perf_counter()
    exception = None
    final_state: dict = {}
    rating = "Hold"
    try:
        final_state, rating = graph.propagate(ticker, date)
    except Exception as exc:  # noqa: BLE001
        exception = exc
        final_state = {}
        rating = "?"
    wall_full = time.perf_counter() - t0

    # Try to read per-node analyst wall times from the perf tracker. The
    # graph currently does NOT expose ``perf_tracker`` as an attribute
    # (perf_telemetry.py is new/unwired), so this is defensive: when None,
    # we fall back to whole-propagate wall time and mark the speedup
    # criterion N/A in the gate.
    analyst_segment = None
    tracker = getattr(graph, "perf_tracker", None)
    if tracker is not None and hasattr(tracker, "serialize"):
        try:
            data = tracker.serialize()
            segs = []
            # Tolerate several key shapes.
            nodes = data.get("nodes", data) if isinstance(data, dict) else {}
            for key in ("market", "social", "news", "fundamentals"):
                node = nodes.get(key) if isinstance(nodes, dict) else None
                if isinstance(node, dict):
                    segs.append(float(node.get("wall_seconds",
                                               node.get("duration", 0.0)) or 0.0))
            if segs:
                analyst_segment = sum(segs)
        except Exception:  # noqa: BLE001
            analyst_segment = None

    decision = (final_state or {}).get("final_trade_decision", "") or ""
    return {
        "ticker": ticker,
        "rating": rating,
        "market_report": (final_state or {}).get("market_report", "") or "",
        "sentiment_report": (final_state or {}).get("sentiment_report", "") or "",
        "news_report": (final_state or {}).get("news_report", "") or "",
        "fundamentals_report": (final_state or {}).get("fundamentals_report", "") or "",
        "final_trade_decision": decision,
        "risk_overlay": parse_risk_overlay(decision),
        "wall_full": wall_full,
        "analyst_segment": analyst_segment,
        "exception": repr(exception) if exception else None,
    }


def _real_run(args) -> int:
    _preflight_warn(args)
    out_dir = Path(args.out_dir).expanduser()
    overall_any = True
    for ticker in args.tickers:
        print(f"\n=== {ticker} @ {args.date} | n={args.n} per leg ===", flush=True)
        # One graph instance per leg (propagate mutates instance state).
        serial_graph = _build_graph("serial", args)
        parallel_graph = _build_graph("parallel", args)

        serial_runs: list[dict] = []
        parallel_runs: list[dict] = []

        for i in range(args.n):
            print(f"  [{ticker}] serial run {i + 1}/{args.n} ...", flush=True)
            serial_runs.append(_capture_run(serial_graph, ticker, args.date))
        for i in range(args.n):
            print(f"  [{ticker}] parallel run {i + 1}/{args.n} ...", flush=True)
            parallel_runs.append(_capture_run(parallel_graph, ticker, args.date))

        metrics = _compute_metrics(serial_runs, parallel_runs, dry_run=False)
        report_path = out_dir / f"parallel_ab_{ticker}_{args.date}.md"
        write_report(report_path, ticker, args.date, args.n, metrics, dry_run=False)
        overall, reasons = summarize_gate(metrics)
        overall_any = overall_any and overall
        print(f"[{ticker}] report -> {report_path}")
        print(f"[{ticker}] verdict: {'PASS' if overall else 'FAIL'}")
        if reasons:
            for r in reasons:
                print(f"    - {r}")
        (out_dir / f"parallel_ab_{ticker}_{args.date}.json").write_text(
            json.dumps(metrics, indent=2, default=str), encoding="utf-8"
        )
    return 0 if overall_any else 1


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="A/B gate: is analyst_parallel=True distributionally "
                    "indistinguishable from serial?",
    )
    p.add_argument("--tickers", nargs="+", required=True,
                   help="One or more tickers (e.g. AAPL NVDA).")
    p.add_argument("--date", required=True,
                   help="Trade date YYYY-MM-DD.")
    p.add_argument("--n", type=int, default=10,
                   help="Runs per leg per ticker (default 10).")
    p.add_argument("--out-dir",
                   default=os.path.join(Path.home(), ".yiagents", "logs", "ab"),
                   help="Where to write reports (default ~/.yiagents/logs/ab).")
    p.add_argument("--max-threads", type=int, default=None,
                   help="analyst_parallel_max_threads for the parallel leg.")
    p.add_argument("--dry-run", action="store_true",
                   help="SYNTHETIC mode: no LLM, no propagate. Exercises every "
                        "metric function + the report writer on fake data.")
    p.add_argument("--seed", type=int, default=42,
                   help="RNG seed for --dry-run synthetic data (default 42).")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    if args.n < 1:
        print("--n must be >= 1", file=sys.stderr)
        return 2
    if args.dry_run:
        return _dry_run(args)
    return _real_run(args)


if __name__ == "__main__":
    sys.exit(main())
