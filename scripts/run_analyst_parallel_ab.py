#!/usr/bin/env python
"""A/B gate: is ``analyst_parallel=True`` distributionally indistinguishable
from the serial analyst chain?

Same ticker + same date, run the full pipeline ``--n`` times with the analyst
chain SERIAL (``analyst_parallel=False``) and ``--n`` times with it PARALLEL
(``analyst_parallel=True``), then statistically compare the two legs. The
iron law is: the concurrency layer must NOT change any agent's decision
distribution. LLM sampling guarantees byte-level differences run-to-run, so
the metrics here are DISTRIBUTIONAL, not byte-equality.

Usage (run from the repo root — ``yialpha/__init__.py`` does
``load_dotenv(usecwd=True)`` and needs the ``.env`` next to the cwd):

    # Zero-cost verification (synthetic data, no LLM calls, no propagate):
    python scripts/run_analyst_parallel_ab.py --dry-run \\
        --tickers AAPL --date 2026-03-15 --n 6

    # Real gate (real LLM, real propagate; ~10 min per ticker per run):
    python scripts/run_analyst_parallel_ab.py \\
        --tickers AAPL NVDA --date 2026-03-15 --n 10 \\
        --max-threads 8 --out-dir ~/.yialpha/logs/ab

``--dry-run`` exercises every metric function and the report writer on
synthetic data drawn from a fixed seed, so the script is verifiable without
spending DeepSeek tokens. The synthetic data is constructed so the gate
PASSES, demonstrating the pass path end-to-end.

Exit codes (round-13 #7: infra crashes are never conflated with a
statistical FAIL):
    0  PASS (all gate criteria met)
    1  FAIL (a statistical/perf criterion was rejected)
    2  usage error (bad --n / --max-threads / --date, caught before any spend)
    3  infra crash — the harness itself broke before a verdict (graph build
       failure, missing config key, report IO error, ...)
``--dry-run`` always exits 0 (synthetic).

Real-mode cost note (round-13 #3): each leg runs ONE extra uncounted
warm-up propagate before its ``--n`` sampled runs, so a ticker costs
``2 * (n + 1)`` full LLM runs. The warm-up absorbs one-off cold costs (data
fetch, disk caches) that would otherwise land entirely on whichever leg
runs first and bias the speedup criterion.

Dependencies: this script needs ONLY the stdlib + (optionally) scipy /
scikit-learn for the chi-square and TF-IDF fast paths. Neither scipy nor
sklearn is required — zero-dependency fallbacks (manual chi-square via the
Wilson-Hilferty approximation; manual TF-IDF + cosine; exact-integer
Fisher) are implemented below. When the accelerators ARE installed the
scipy/sklearn paths run, and the sparse-table / same-space policies below
keep both paths decision-consistent for the same input.
The pure metric functions import no yialpha code, so the unit tests run
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
from datetime import datetime
from pathlib import Path
from random import Random

# Allow running as `python scripts/run_analyst_parallel_ab.py` without an editable install, and
# make scripts/_bootstrap importable under importlib-based test loading too
# (spec_from_file_location does not put scripts/ on sys.path).
_HERE = str(Path(__file__).resolve().parent)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import _bootstrap  # noqa: E402

_bootstrap.bootstrap()  # project-root sys.path + UTF-8 console (Windows GBK)

# Centralised logging: make INFO/DEBUG from agents/dataflows visible at runtime
# (otherwise Python's default WARNING-only root logger silently swallows them).
from yialpha.logging_config import setup_logging  # noqa: E402

setup_logging()  # noqa: E402

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
# ``yialpha/graph/trading_graph.py:_apply_risk_overlay`` (around line 332).
_RISK_OVERLAY_MARKER = "## Quantitative Risk Overlay"
# Exponent-aware numeric class for the two price bullets: the overlay
# renders Stop Loss / Entry Reference with %.6g, which emits exponent form
# below 1e-4 (PEPEUSDT-class stops, "1.14e-05"). The old [\-0-9.]+ class
# truncated at the "e" and parsed 1.14 — feeding the run dicts and the
# risk_overlay_determinism compare values five orders of magnitude off
# (two different stops could even false-PASS the determinism criterion by
# both truncating to the same mantissa). Mirrors overlay_fields._NUM.
_EXPNUM = r"([\-0-9.]+(?:[eE][-+]?[0-9]+)?)"
_TARGET_WEIGHT_RE = re.compile(r"\*\*Target Weight\*\*:\s*([\-0-9.]+)%")
_STOP_LOSS_RE = re.compile(rf"\*\*Stop Loss\*\*:\s*\$?{_EXPNUM}")
_ENTRY_REF_RE = re.compile(rf"\*\*Entry Reference\*\*:\s*\$?{_EXPNUM}")

# Gate thresholds (mirrors the task spec).
_RATING_P_THRESHOLD = 0.05
_SPEEDUP_THRESHOLD = 2.5


# ---------------------------------------------------------------------------
# Pure metric functions (no yialpha import; unit-tested directly).
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
    (df = k - 1). NO continuity correction — matching the scipy path's
    explicit ``correction=False`` (round-13 #6): scipy used to apply Yates
    for 2x2 tables while this path never did, so the two paths could return
    opposite verdicts for the same table. Returns the survival function
    (p-value) via the Wilson-Hilferty approximation. Smoothes any zero
    expected cell by adding a tiny constant to avoid divide-by-zero.
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


def _fisher_exact_2x2(a: int, b: int, c: int, d: int) -> float:
    """Two-sided Fisher exact p-value for the 2x2 table ``[[a,b],[c,d]]``.

    Exact integer arithmetic (``math.comb``) with the standard
    observed-probability two-sided convention: sum the probabilities of all
    tables no more probable than the observed one. Deliberately used on BOTH
    the scipy and manual paths — identical on every machine, no floating
    point ordering differences — and verified against
    ``scipy.stats.fisher_exact`` (500 random tables, max |diff| < 1e-12).
    """
    row1 = a + b
    col1 = a + c
    col2 = b + d
    n = row1 + c + d
    if row1 <= 0 or (c + d) <= 0 or col1 <= 0 or col2 <= 0 or n <= 0:
        return 1.0
    denom = math.comb(n, row1)
    if denom == 0:  # pragma: no cover - unreachable with positive margins
        return 1.0
    p_obs = math.comb(col1, a) * math.comb(col2, b)
    lo = max(0, row1 - col2)
    hi = min(row1, col1)
    total = 0
    for x in range(lo, hi + 1):
        px = math.comb(col1, x) * math.comb(col2, row1 - x)
        if px <= p_obs:
            total += px
    return total / denom


#: Below this minimum expected cell count the chi-square asymptotics are
#: unreliable (the textbook "expected >= 5" rule). At the gate's default
#: n=10 per leg spread over 5 rating categories this is the norm, not the
#: exception — every cell expectation is ~1-4.
_SPARSE_EXPECTED_MIN = 5.0


def _contingency(hist_a: dict, hist_b: dict):
    """Union-category 2 x k contingency layout from two histograms.

    Returns ``(table, cats, min_expected)`` or ``None`` when no comparison
    is possible (<=1 effective category, or an empty leg). Categories with a
    zero count in BOTH legs are dropped (they contribute df without
    information and can flip the sparse classification).
    """
    cats = sorted(
        c for c in (set(hist_a) | set(hist_b))
        if (hist_a.get(c, 0) or hist_b.get(c, 0))
    )
    if len(cats) <= 1:
        return None
    row_a = sum(hist_a.get(c, 0) for c in cats)
    row_b = sum(hist_b.get(c, 0) for c in cats)
    if row_a <= 0 or row_b <= 0:
        return None
    total = row_a + row_b
    table = [[hist_a.get(c, 0) for c in cats], [hist_b.get(c, 0) for c in cats]]
    min_expected = min(
        row_total * col / total
        for row_total in (row_a, row_b)
        for col in (hist_a.get(c, 0) + hist_b.get(c, 0) for c in cats)
    )
    return table, cats, min_expected


def chi_square_test(hist_a: dict, hist_b: dict) -> tuple[float, str]:
    """Two-sample homogeneity test on rating histograms -> ``(p, method)``.

    Round-13 #6 selection policy (consistent across the scipy and manual
    paths, conservative when sparse):

    * Degenerate table (<=1 effective category / empty leg): ``(1.0,
      "degenerate")`` — cannot reject; the caller's small-sample overlap
      fallback still applies.
    * Any expected cell < 5 (sparse):
        - exactly 2 categories → Fisher exact (two-sided), the conservative
          exact test for 2x2; computed by :func:`_fisher_exact_2x2` on both
          paths (machine-identical).
        - >2 categories → chi-square with the unified no-correction policy
          on both paths, and the method string DISCLOSES the sparse warning
          (asymptotic p on sparse cells; treat borderline 0.03-0.07 with
          caution — a general 2xk exact test is out of scope for stdlib).
    * Otherwise chi-square, scipy path with explicit ``correction=False``
      so both paths compute the same Pearson statistic (pre-round-13 the
      scipy path silently applied Yates continuity for 2x2 tables while
      the manual path never did — the two paths could disagree across the
      0.05 threshold for the same table).
    """
    layout = _contingency(hist_a, hist_b)
    if layout is None:
        return 1.0, "degenerate (too few categories or samples)"
    table, cats, min_expected = layout
    sparse = min_expected < _SPARSE_EXPECTED_MIN

    if sparse and len(cats) == 2:
        (a, b), (c, d) = table[0], table[1]
        return (
            _fisher_exact_2x2(a, b, c, d),
            "Fisher exact (2x2, sparse cells: "
            f"min expected {min_expected:.2f} < {_SPARSE_EXPECTED_MIN:.0f})",
        )

    if sparse:
        method = (
            "chi-square (SPARSE cells: min expected "
            f"{min_expected:.2f} < {_SPARSE_EXPECTED_MIN:.0f} — asymptotic "
            "p, treat borderline 0.03-0.07 with caution)"
        )
    else:
        method = "chi-square"

    if _HAS_SCIPY and _scipy_stats is not None:
        try:
            _stat, p, _dof, _exp = _scipy_stats.chi2_contingency(
                table, correction=False
            )
            return float(p), method
        except Exception:  # noqa: BLE001
            pass  # fall through to manual
    return _chi_square_manual(hist_a, hist_b), method


def chi_square_p(hist_a: dict, hist_b: dict) -> float:
    """Two-sample homogeneity p-value on rating histograms.

    Thin wrapper over :func:`chi_square_test` (see its docstring for the
    sparse-table policy). Returns 1.0 when there are too few categories or
    samples to form a table (the caller's "too few samples" branch then
    triggers the overlap fallback).
    """
    return chi_square_test(hist_a, hist_b)[0]


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


def _pairwise_cosines(texts: list[str]) -> list[list[float]]:
    """N x N cosine-similarity matrix from ONE TF-IDF fit over ``texts``.

    Round-13 #1: the between/within criterion MUST compare vectors from a
    SINGLE fitted space. The pre-fix code fitted the between leg on the
    serial+parallel union (N = 2n) while each within leg was fitted on its
    own texts (N = n): a term unique to one leg gets ~+ln2 idf in the union
    fit, inflating that leg's vector norms while never matching anything
    across legs — the between cosine was systematically biased LOW and the
    criterion false-FAILED 39/40 Monte-Carlo i.i.d. pairs (scanner round 13).

    Both backends fit ONE space with the SAME idf semantics: sklearn's
    ``TfidfVectorizer()`` defaults (smooth_idf=True → ``idf =
    ln((1+N)/(1+df)) + 1``, L2 norm) match :func:`_tfidf_vectors` exactly.
    """
    texts = list(texts)
    if not texts:
        return []
    if _HAS_SKLEARN and _sk_text is not None and _sk_metrics is not None:
        try:
            mat = _sk_text.TfidfVectorizer().fit_transform(texts)
            sim = _sk_metrics.cosine_similarity(mat)
            return [[float(v) for v in row] for row in sim]
        except Exception:  # noqa: BLE001
            pass  # fall through to manual
    vectors, _vocab = _tfidf_vectors(texts)
    normed = [_l2_normalize(v) for v in vectors]
    return [[_cosine(a, b) for b in normed] for a in normed]


def _mean_block(sim: list[list[float]], r0: int, r1: int, c0: int, c1: int,
                *, exclude_diagonal: bool = False) -> float:
    """Mean over the [r0,r1) x [c0,c1) block of a similarity matrix.

    ``exclude_diagonal`` is used for the square within-leg blocks (a text is
    trivially identical to itself, so the diagonal must not inflate within
    coherence). Empty range returns 0.0.
    """
    total = 0.0
    count = 0
    for i in range(r0, r1):
        for j in range(c0, c1):
            if exclude_diagonal and i == j:
                continue
            total += sim[i][j]
            count += 1
    return total / count if count else 0.0


def tfidf_cosine_similarity(texts_a: list[str], texts_b: list[str]) -> float:
    """Mean pairwise TF-IDF cosine between two text sets.

    ONE idf fit over the union corpus (so shared vocabulary gets meaningful
    weight), then the mean cosine over every (a, b) pair. Empty inputs
    return 0.0. sklearn path and manual path share the fit semantics (see
    :func:`_pairwise_cosines`).
    """
    if not texts_a or not texts_b:
        return 0.0
    sim = _pairwise_cosines(list(texts_a) + list(texts_b))
    n, m = len(texts_a), len(texts_b)
    return _mean_block(sim, 0, n, n, n + m)


def between_vs_within_similarity(
    reports_serial: list[str], reports_parallel: list[str]
) -> dict:
    """Return within-set and between-set mean TF-IDF cosines.

    Criterion: ``between >= min(within_serial, within_parallel)`` — i.e. the
    parallel leg is no further from the serial leg than each leg is from
    itself. A single-text set has within-similarity 1.0 (a text is identical
    to itself); an empty set has 0.0.

    Round-13 #1: within and between are computed from the SAME set of
    L2-normalized vectors fitted in ONE union space (serial+parallel
    concatenated through :func:`_pairwise_cosines`), so the comparison is
    apples-to-apples and identically-distributed legs pass the criterion
    (Monte-Carlo H0 pass rate 20/20 on the regression-test construction;
    the pre-fix split-space code passed 13/20 on the same seeds and 1/40 on
    the scanner's construction).
    """
    s = list(reports_serial)
    p = list(reports_parallel)
    n, m = len(s), len(p)
    sim = _pairwise_cosines(s + p)

    def within(r0: int, r1: int) -> float:
        k = r1 - r0
        if k < 2:
            return 1.0 if k == 1 else 0.0
        return _mean_block(sim, r0, r1, r0, r1, exclude_diagonal=True)

    between = _mean_block(sim, 0, n, n, n + m) if (n and m) else 0.0
    return {
        "within_serial": within(0, n),
        "within_parallel": within(n, n + m),
        "between": between,
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


def risk_overlay_determinism(runs: list[dict]) -> bool | None:
    """Within each rating group, the risk-overlay numbers must be identical.

    The overlay is a deterministic function of (rating, price, ATR), so two
    runs that share a rating MUST share overlay numbers. ``runs`` items are
    expected to carry ``rating`` (str) and ``risk_overlay`` (dict | None, as
    produced by :func:`parse_risk_overlay`). Groups where every run has
    ``risk_overlay is None`` (overlay disabled) are treated as vacuously
    consistent. Floats are compared with a tight tolerance (1e-9) to absorb
    formatting-only differences.

    Returns tri-state (round-13 #5):

    * ``True``  — at least one group was actually verified consistent (and
      no group violated). All-vacuous runs (overlay disabled everywhere)
      also return ``True`` — nothing claimed to be checked.
    * ``False`` — a violation: mixed overlay-present/absent within a group,
      or same-rating runs disagreeing on a parsed number.
    * ``None``  — the overlay MARKER was present but no group yielded a
      single parseable field, so nothing could be verified. The pre-fix code
      compared ``None == None`` per field, silently treating such groups as
      "equal" and PASSing criterion 3 vacuously; honest reporting requires
      N/A instead (a parsing regression is not determinism evidence).
    """
    by_rating: dict[str, list[dict]] = {}
    for r in runs:
        rating = r.get("rating", "?")
        by_rating.setdefault(rating, []).append(r)
    any_verified = False
    any_na = False
    for _rating, group in by_rating.items():
        overlays = [g.get("risk_overlay") for g in group]
        # Skip groups where every overlay is missing (risk disabled).
        if all(o is None for o in overlays):
            continue
        # If some have an overlay and some don't, that's inconsistent.
        if any(o is None for o in overlays):
            return False
        # Marker present in every run of this group, but did ANY run parse
        # at least one number? An all-None-fields group is unverifiable —
        # count it N/A rather than silently "equal" (round-13 #5).
        group_parsed = any(
            (o or {}).get(k) is not None
            for o in overlays
            for k in ("target_weight", "stop_loss", "entry_price")
        )
        if not group_parsed:
            any_na = True
            continue
        any_verified = True
        # After the None-check above, all overlays are dicts.
        ref = overlays[0] or {}
        for o in overlays[1:]:
            o = o or {}
            for key in ("target_weight", "stop_loss", "entry_price"):
                a = ref.get(key)
                b = o.get(key)
                if a is None and b is None:
                    # Neither run exposes this field: not evidence either
                    # way (the field-level vacuous case, tolerated as
                    # before — at least one field of the group IS parsed).
                    continue
                if a is None or b is None:
                    return False
                if abs(float(a) - float(b)) > 1e-9:
                    return False
    if any_verified:
        return True
    if any_na:
        return None
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
        # Overlap rescue is a SMALL-SAMPLE fallback only. chi_square_p
        # already returns 1.0 in degenerate cases (<=1 category, empty leg),
        # so a real p <= threshold with decent n is a genuine statistical
        # rejection — round-8: the unconditional overlap>0.5 rescue used to
        # pass a 0.9→0.5 Buy-rate drift at n=20/leg (p≈0.006), exactly the
        # distribution shift this gate exists to catch before the parallel
        # default flips.
        n_small = min(
            metrics.get("n_serial") or 0,
            metrics.get("n_parallel") or 0,
        ) < 5
        overlap = metrics.get("overlap_ratio", 0.0)
        if n_small and overlap > 0.5:
            pass  # ok via small-sample fallback
        else:
            fail(
                f"rating chi-square p={p:.4f} <= {_RATING_P_THRESHOLD} "
                f"(overlap fallback unavailable: "
                f"n_min={min(metrics.get('n_serial') or 0, metrics.get('n_parallel') or 0)}"
                f"{' >=5' if not n_small else ''}, overlap {overlap:.2f})"
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

    # 3) Risk-overlay determinism in BOTH legs. Tri-state (round-13 #5):
    #    ``None`` = N/A — the overlay marker appeared but no numbers could
    #    be parsed in any group, so there is nothing to compare. N/A does
    #    not fail the gate (it is not a determinism violation) but it is
    #    NOT a pass either; the report renders it explicitly.
    if metrics.get("risk_determinism_serial") is False:
        fail("risk-overlay determinism FAILED in serial leg")
    if metrics.get("risk_determinism_parallel") is False:
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

    # 7) Both legs retain usable samples after exception-dropping. An
    #    all-crashed leg has no distribution to compare — that is an infra
    #    failure, not a parallel-vs-serial verdict.
    n_serial_ok = (metrics.get("n_serial") or 0) - (metrics.get("dropped_serial") or 0)
    n_parallel_ok = (metrics.get("n_parallel") or 0) - (metrics.get("dropped_parallel") or 0)
    if n_serial_ok <= 0:
        fail("serial leg: all propagate runs failed (no usable samples)")
    if n_parallel_ok <= 0:
        fail("parallel leg: all propagate runs failed (no usable samples)")

    # 8) Prediction attribution (S3-a race probe): when the parallel leg
    #    recorded settled predictions, at least one run must attribute them
    #    to 2+ distinct analysts — the __active__ single-slot race collapses
    #    every submission onto the last writer.
    if not metrics.get("prediction_attribution_measured", False):
        pass  # N/A — no predictions recorded (ledger off / no submissions)
    elif not metrics.get("prediction_attribution_ok", False):
        fail(
            "prediction attribution: every parallel run's predictions land "
            "on a single analyst (__active__ capture race)"
        )

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
    lines.append(
        f"- Dropped from stats (propagate failed): serial "
        f"{metrics.get('dropped_serial', 0)}, parallel "
        f"{metrics.get('dropped_parallel', 0)}"
    )
    if metrics.get("prediction_attribution_measured", False):
        lines.append(
            "- Prediction attribution (S3-a race probe): "
            + ("OK (2+ distinct analysts in a parallel run)"
               if metrics.get("prediction_attribution_ok", False)
               else "FAILED (single-analyst collapse)")
        )
    speedup = metrics.get("speedup")
    if dry_run:
        lines.append("- Parallel speedup: N/A (dry-run, synthetic timing)")
    elif metrics.get("speedup_na"):
        wf = metrics.get("speedup_wall_full")
        extra = f" (whole-run ratio {wf:.2f}x recorded, informational only)" if wf else ""
        lines.append(
            "- Parallel speedup: N/A (clean per-analyst segment unavailable; "
            f"2.5x gate not enforced{extra})"
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
    method = metrics.get("rating_test_method")
    if method:
        # Round-13 #6: disclose WHICH test produced the p-value (Fisher
        # exact on sparse 2x2 tables / chi-square with its sparse-cells
        # caveat) so a borderline verdict is auditable.
        lines.append(f"- Rating test: {method}")
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

    def _det(v) -> str:
        # Tri-state render (round-13 #5): N/A is honest about "marker hit
        # but nothing parsed" — never shown as PASS, never hidden as FAIL.
        if v is None:
            return "N/A (overlay present but no numbers parsed; not verified)"
        return "PASS" if v else "FAIL"

    lines.append(f"- Serial leg: {_det(metrics.get('risk_determinism_serial'))}")
    lines.append(f"- Parallel leg: {_det(metrics.get('risk_determinism_parallel'))}")
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
    """Run every metric function over two legs and return the metrics dict.

    Runs whose propagate raised are EXCLUDED from every distributional
    statistic: a crashed run's ``"?"`` rating and empty reports are infra
    artifacts, not samples — they used to skew the chi-square / TF-IDF
    inputs and silently flip the speedup criterion to N/A via a None
    analyst segment. They still count in ``exceptions_*`` (gate criterion
    6 fails on them) and in the new ``dropped_*`` keys; criterion 7 fails
    outright when a leg loses ALL its runs.
    """
    def _ok(runs: list[dict]) -> list[dict]:
        return [r for r in runs if not r.get("exception")]

    serial_ok = _ok(serial_runs)
    parallel_ok = _ok(parallel_runs)
    dropped_serial = len(serial_runs) - len(serial_ok)
    dropped_parallel = len(parallel_runs) - len(parallel_ok)

    s_ratings = [r["rating"] for r in serial_ok]
    p_ratings = [r["rating"] for r in parallel_ok]
    sh = rating_histogram(s_ratings)
    ph = rating_histogram(p_ratings)
    p, rating_test_method = chi_square_test(sh, ph)
    overlap = _overlap_ratio(sh, ph)

    fields = ("market_report", "sentiment_report", "news_report", "fundamentals_report")
    per_field: dict[str, dict[str, float]] = {}
    for f in fields:
        s_texts = [r.get(f, "") for r in serial_ok]
        p_texts = [r.get(f, "") for r in parallel_ok]
        per_field[f] = between_vs_within_similarity(s_texts, p_texts)

    def empty_count(runs: list[dict], fields_) -> int:
        n = 0
        for r in runs:
            for f in fields_:
                if not (r.get(f) or "").strip():
                    n += 1
        return n

    # Speedup: gate ONLY on a clean per-analyst segment. When either leg can't
    # expose one (e.g. the parallel leg's per-analyst subgraph nodes are not yet
    # wrapped by telemetry), mark N/A so the gate skips the 2.5x rule honestly.
    # We deliberately do NOT fall back to whole-propagate wall time for the gate:
    # that dilutes the analyst-only speedup with the serial debate/trader/risk/PM
    # tail and would falsely fail a correct implementation. The whole-run ratio
    # is still recorded as informational-only in ``speedup_wall_full``.
    serial_segs: list[float | None] = [r.get("analyst_segment") for r in serial_ok]
    par_segs: list[float | None] = [r.get("analyst_segment") for r in parallel_ok]
    serial_walls = [r.get("wall_full", 0.0) for r in serial_ok]
    par_walls = [r.get("wall_full", 0.0) for r in parallel_ok]
    speedup_na = dry_run
    speedup = None
    speedup_wall_full = None
    if serial_walls and par_walls and sum(par_walls) > 0:
        speedup_wall_full = (sum(serial_walls) / len(serial_walls)) / \
                            (sum(par_walls) / len(par_walls))

    segs_ok = (
        all(x is not None and x > 0 for x in serial_segs)
        and all(x is not None and x > 0 for x in par_segs)
    )
    # Round-17: ``all([])`` is True, so a leg whose every propagate failed (no
    # usable runs left) passed the check above and the mean below divided by
    # zero -- the whole run died as an "infra crash" (exit 3, no report)
    # instead of the criterion-6 / criterion-7 failure those criteria exist to
    # report. An empty leg has no segment to measure: N/A, like a leg without
    # telemetry.
    if segs_ok and serial_segs and par_segs:
        ss = [x for x in serial_segs if x is not None]
        ps = [x for x in par_segs if x is not None]
        speedup = (sum(ss) / len(ss)) / (sum(ps) / len(ps))
    else:
        # Clean analyst segment unavailable for at least one leg — record the
        # diluted whole-run ratio above for visibility, but mark the gate
        # criterion N/A rather than enforcing a misleading number.
        speedup_na = True
    # In dry-run we don't enforce speedup.
    if dry_run:
        speedup_na = True

    # S3-a race probe: a healthy parallel leg attributes one run's settled
    # predictions to 2+ DISTINCT analysts; the __active__ single-slot race
    # collapses every submission onto the last writer. N/A when no
    # predictions were recorded at all (ledger off / no tool submissions) —
    # unmeasured, not passed.
    par_pred_sets = [r.get("prediction_analysts") or [] for r in parallel_ok]
    any_predictions = any(par_pred_sets)
    prediction_attribution_ok = any(len(a) >= 2 for a in par_pred_sets)

    return {
        "n_serial": len(serial_runs),
        "n_parallel": len(parallel_runs),
        "dropped_serial": dropped_serial,
        "dropped_parallel": dropped_parallel,
        "serial_hist": sh,
        "parallel_hist": ph,
        "rating_p": p,
        "rating_test_method": rating_test_method,
        "overlap_ratio": overlap,
        "per_field_similarity": per_field,
        "risk_determinism_serial": risk_overlay_determinism(serial_ok),
        "risk_determinism_parallel": risk_overlay_determinism(parallel_ok),
        "empty_reports_serial": empty_count(serial_ok, fields),
        "empty_reports_parallel": empty_count(parallel_ok, fields),
        "exceptions_serial": sum(1 for r in serial_runs if r.get("exception")),
        "exceptions_parallel": sum(1 for r in parallel_runs if r.get("exception")),
        "prediction_attribution_ok": prediction_attribution_ok,
        "prediction_attribution_measured": any_predictions,
        "speedup": speedup,
        "speedup_na": speedup_na,
        "speedup_wall_full": speedup_wall_full,
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
    """Build a YiAlphaGraph for one leg. Heavy imports are lazy so the dry-run
    path and the unit tests don't require the yialpha graph stack."""
    from yialpha.default_config import DEFAULT_CONFIG  # noqa: E402
    from yialpha.graph.trading_graph import YiAlphaGraph  # noqa: E402

    config = DEFAULT_CONFIG.copy()
    # Round-13 #2: force the per-call LLM disk cache OFF (same defensive
    # shape as run_baseline.py's memory_enabled guard at :106-109, applied
    # here to llm_cache). YIALPHA_LLM_CACHE=true would install the shared
    # DiskLLMCache and replay byte-identical ChatGenerations per call: the
    # parallel leg would replay the serial leg's responses, collapsing all
    # four gate criteria into hollow PASSes (identical texts -> TF-IDF
    # trivially equal; identical ratings -> chi-square p=1; ~zero wall
    # spread -> speedup meaningless). The cache must NOT decide the default
    # flip; hard-disable regardless of env/.env.
    config["llm_cache"] = False
    # Round-13 #4: force persistent memory OFF (mirrors run_baseline.py
    # :106-109 — "validation runs must not depend on persistent lessons
    # written by an earlier ticker/run ordering"). A shared memory log
    # makes later runs depend on earlier ones: the legs stop being i.i.d.
    # samples of the same distribution, which is the null hypothesis every
    # criterion here tests.
    config["memory_enabled"] = False
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
    return YiAlphaGraph(config=config)


def _preflight_warn(args) -> None:
    """Warn (not fail) when environment config would hollow out the gate."""
    try:
        from yialpha.default_config import DEFAULT_CONFIG  # noqa: E402
    except Exception as exc:  # noqa: BLE001
        print(f"[preflight] could not import DEFAULT_CONFIG: {exc}", file=sys.stderr)
        return
    cfg = DEFAULT_CONFIG
    if cfg.get("llm_cache"):
        # Round-13 #2: loud, because the damage is silent — the gate keeps
        # running, "passes", and the parallel default flips on evidence the
        # cache fabricated. _build_graph hard-forces llm_cache=False so the
        # run itself stays valid; this warning explains why the env var had
        # no effect (the user turned it on for a reason and should know it
        # was ignored here).
        print(
            "[preflight] WARNING: YIALPHA_LLM_CACHE=true is set. The per-call "
            "disk cache replays byte-identical LLM responses across runs, "
            "which would hollow out EVERY criterion of this gate (identical "
            "reports, identical ratings, fake speedup). This script FORCES "
            "llm_cache=False on both legs and ignores the env var.",
            file=sys.stderr,
        )
    if cfg.get("memory_enabled"):
        # Round-13 #4: same class of problem — a shared memory log makes the
        # legs stateful, violating the i.i.d. null hypothesis.
        print(
            "[preflight] WARNING: memory_enabled=true. A shared memory log "
            "makes later runs depend on earlier ones (non-i.i.d. legs). This "
            "script FORCES memory_enabled=False on both legs.",
            file=sys.stderr,
        )
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

    # Analyst-segment wall time, leg-aware (invariant only; the telemetry
    # contract lives in graph/setup.py): the parallel leg records the whole
    # analyst stage as ONE "Analyst Fanout" node; the serial leg records it
    # as the per-analyst agent/tool/clear clusters the fanout replaces. Sum
    # EXACTLY that replaced scope — agents-only undercounts the serial stage
    # and biases the 2.5x criterion toward a false FAIL. Nodes absent from
    # telemetry contribute nothing: unselected analysts never run. Serial
    # ToolNode walls count since round-11 (setup.py._wrap_node wraps
    # ToolNode via the Runnable-aware wrapper — tool HTTP previously read
    # as zero on the serial leg, systematically biasing the 2.5x criterion
    # toward a false FAIL); the parallel leg's subgraph tool nodes still
    # fold into the parent "Analyst Fanout" node, so both legs now cover
    # agent + tool + clear over the same replaced scope.
    analyst_segment = None
    tracker = getattr(graph, "perf_tracker", None)
    if tracker is not None and hasattr(tracker, "serialize"):
        try:
            data = tracker.serialize()
            nodes = data.get("nodes", data) if isinstance(data, dict) else {}

            def _wall(name: str) -> float | None:
                node = nodes.get(name) if isinstance(nodes, dict) else None
                if isinstance(node, dict):
                    return float(node.get("wall_seconds",
                                          node.get("duration", 0.0)) or 0.0)
                return None

            fanout = _wall("Analyst Fanout")
            if fanout is not None and fanout > 0:
                # Parallel leg: the fanout node IS the analyst stage.
                analyst_segment = fanout
            else:
                # Serial leg: per-analyst walls are recorded under the node
                # NAME (perf_telemetry serializes by node name). Resolve the
                # replaced cluster's names (agent + tool + clear per spec)
                # from the spec table — unguarded import on purpose:
                # _build_graph already imported yialpha.graph.trading_graph
                # (which pulls this module) before _capture_run can run, so
                # an except-branch could never fire and only masked drift
                # (the deleted fallback list had 4 names vs 5 specs).
                from yialpha.graph.analyst_execution import ANALYST_NODE_SPECS
                names = [
                    node_name
                    for spec in ANALYST_NODE_SPECS.values()
                    for node_name in (
                        spec.agent_node, spec.tool_node, spec.clear_node,
                    )
                ]
                segs = [w for w in (_wall(n) for n in names) if w is not None]
                if segs:
                    analyst_segment = sum(segs)
        except Exception:  # noqa: BLE001
            analyst_segment = None

    decision = (final_state or {}).get("final_trade_decision", "") or ""

    # S3-a race probe material: the distinct analysts this run's settled
    # predictions attribute to. A healthy run with multiple tool-loop
    # analysts attributes to 2+; the __active__ single-slot race collapses
    # every submission onto the last writer (one analyst). Fail-soft — the
    # attribution check must never break the gate run itself.
    prediction_analysts: list[str] = []
    run_id = (final_state or {}).get("run_id")
    if run_id:
        try:
            from yialpha.ledger.predictions import predictions_for_run

            prediction_analysts = sorted(
                {p.analyst for p in predictions_for_run(str(run_id))}
            )
        except Exception:  # noqa: BLE001
            prediction_analysts = []

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
        "prediction_analysts": prediction_analysts,
        "exception": repr(exception) if exception else None,
    }


def _real_run(args) -> int:
    """Real-mode driver. Infra crashes exit 3, statistical FAIL exits 1.

    Round-13 #7: anything that escapes a single propagate (graph build
    failure, missing config key, report IO error, ...) is an INFRA failure,
    not a distributional verdict. Exit 1 is reserved for "the parallel leg
    was measured and rejected"; conflating the two would let a broken
    harness masquerade as a FAIL (blocking the default flip) or hide behind
    it. Per-run propagate exceptions are already caught inside
    :func:`_capture_run` (they become criterion-6 failures), so anything
    raising out of the loop below is by construction harness-level.
    """
    try:
        return _real_run_inner(args)
    except Exception as exc:  # noqa: BLE001 - infra crash boundary
        print(
            f"\n[infra] A/B harness crashed before a verdict could be "
            f"reached (this is NOT a statistical FAIL): {exc!r}",
            file=sys.stderr,
            flush=True,
        )
        return 3


def _real_run_inner(args) -> int:
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

        # Round-13 #3: one uncounted WARM-UP per leg, result discarded. The
        # first propagate on a fresh graph pays one-off cold costs (data
        # fetch, connection/DNS warm-up, disk caches) that have nothing to
        # do with the analyst concurrency under test; with the leg order
        # fixed serial-first they landed entirely on the serial leg and
        # systematically inflated the speedup criterion. Warm-up (rather
        # than interleaving serial/parallel samples) is the simpler and
        # clearer remedy: each leg is measured warm, leg order stays fixed,
        # both legs keep symmetric n-sample statistics, and no metric input
        # changes shape. Cost: +2 propagates per ticker.
        print(f"  [{ticker}] serial warm-up (discarded) ...", flush=True)
        _capture_run(serial_graph, ticker, args.date)
        for i in range(args.n):
            print(f"  [{ticker}] serial run {i + 1}/{args.n} ...", flush=True)
            serial_runs.append(_capture_run(serial_graph, ticker, args.date))

        print(f"  [{ticker}] parallel warm-up (discarded) ...", flush=True)
        _capture_run(parallel_graph, ticker, args.date)
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
                   default=os.path.join(Path.home(), ".yialpha", "logs", "ab"),
                   help="Where to write reports (default ~/.yialpha/logs/ab).")
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
    if args.max_threads is not None and args.max_threads < 1:
        # 0 would reach ThreadPoolExecutor(max_workers=0) inside the fanout
        # on FIRST INVOKE — mid-graph, after billing has started — as a bare
        # ValueError, instead of failing fast here.
        print("--max-threads must be >= 1", file=sys.stderr)
        return 2
    try:
        datetime.strptime(args.date, "%Y-%m-%d")
    except ValueError:
        # A malformed date otherwise surfaces only after 2*n full LLM runs
        # (propagate-level rejection), burning the entire gate budget.
        print(f"--date must be YYYY-MM-DD (got {args.date!r})", file=sys.stderr)
        return 2
    if args.dry_run:
        return _dry_run(args)
    return _real_run(args)


if __name__ == "__main__":
    sys.exit(main())
