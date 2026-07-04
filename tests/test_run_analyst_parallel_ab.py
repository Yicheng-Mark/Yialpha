"""Unit tests for the PURE metric functions in scripts/run_analyst_parallel_ab.py.

These tests do NOT call propagate, do NOT touch the LLM, and do NOT import
the yiagents graph stack (every metric function is stdlib-only). The script
is loaded as an isolated module via importlib so its ``if __name__ ==
"__main__"`` guard never fires.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Load the script as a module (it has no package; importlib is the clean way).
# ---------------------------------------------------------------------------
_SCRIPT_PATH = (
    Path(__file__).resolve().parent.parent / "scripts" / "run_analyst_parallel_ab.py"
)
_spec = importlib.util.spec_from_file_location(
    "run_analyst_parallel_ab", _SCRIPT_PATH
)
assert _spec is not None and _spec.loader is not None
ab = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ab)


# ---------------------------------------------------------------------------
# rating_histogram
# ---------------------------------------------------------------------------

class TestRatingHistogram:
    def test_counts(self):
        ratings = ["Buy", "Buy", "Hold", "Sell", "Buy", "Hold"]
        hist = ab.rating_histogram(ratings)
        assert hist == {"Buy": 3, "Hold": 2, "Sell": 1}

    def test_empty(self):
        assert ab.rating_histogram([]) == {}

    def test_unknown_category_kept(self):
        # Unknown ratings are kept under their own key (not silently dropped).
        hist = ab.rating_histogram(["Buy", "WEIRD"])
        assert hist == {"Buy": 1, "WEIRD": 1}


# ---------------------------------------------------------------------------
# tfidf_cosine_similarity
# ---------------------------------------------------------------------------

class TestTfidfCosine:
    def test_identical_texts_approx_one(self):
        texts = ["alpha beta gamma delta epsilon"]
        sim = ab.tfidf_cosine_similarity(texts, texts)
        # Identical non-empty texts → cosine of a vector with itself = 1.0.
        assert sim == pytest.approx(1.0, abs=1e-6)

    def test_disjoint_vocab_approx_zero(self):
        a = ["alpha beta gamma"]
        b = ["delta epsilon zeta"]
        sim = ab.tfidf_cosine_similarity(a, b)
        assert sim == pytest.approx(0.0, abs=1e-6)

    def test_partial_overlap_intermediate(self):
        a = ["alpha beta gamma"]
        b = ["beta gamma delta"]
        sim = ab.tfidf_cosine_similarity(a, b)
        # Some shared terms, some not → strictly between 0 and 1.
        assert 0.0 < sim < 1.0

    def test_empty_inputs_return_zero(self):
        assert ab.tfidf_cosine_similarity([], ["a"]) == 0.0
        assert ab.tfidf_cosine_similarity(["a"], []) == 0.0

    def test_fallback_path_forced(self, monkeypatch):
        """Force the zero-dependency fallback via the availability flag and
        confirm it produces the same correct results. (On this machine the
        fallback is already the active path because scikit-learn isn't
        installed; this test pins the behavior so a future sklearn install
        can't silently change which code path runs.)"""
        monkeypatch.setattr(ab, "_HAS_SKLEARN", False)
        texts = ["alpha beta gamma delta"]
        assert ab.tfidf_cosine_similarity(texts, texts) == pytest.approx(1.0, abs=1e-6)
        assert ab.tfidf_cosine_similarity(["alpha"], ["beta"]) == pytest.approx(0.0, abs=1e-6)

    def test_sklearn_flag_true_falls_back_gracefully(self, monkeypatch):
        """If the flag claims sklearn but the import isn't really there, the
        function must still return a correct answer via the manual fallback."""
        monkeypatch.setattr(ab, "_HAS_SKLEARN", True)
        # _sk_text remains None on this machine → the sklearn branch raises
        # inside the function and falls through to the manual path.
        texts = ["alpha beta gamma delta"]
        assert ab.tfidf_cosine_similarity(texts, texts) == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# between_vs_within_similarity
# ---------------------------------------------------------------------------

class TestBetweenWithin:
    def test_identical_sets_pass_criterion(self):
        # Identical multisets → between >= min(within_serial, within_parallel)
        # (between includes the diagonal cos(x,x)=1 pairs, so strictly >).
        a = ["alpha beta", "beta gamma", "alpha gamma"]
        b = ["alpha beta", "beta gamma", "alpha gamma"]
        out = ab.between_vs_within_similarity(a, b)
        assert set(out.keys()) == {"within_serial", "within_parallel", "between"}
        assert out["between"] >= min(out["within_serial"], out["within_parallel"]) - 1e-9

    def test_single_text_identical_sets_equal(self):
        # Degenerate case honoring the literal "between == within" property:
        # one text per set, identical → all three equal 1.0.
        a = ["alpha beta gamma"]
        b = ["alpha beta gamma"]
        out = ab.between_vs_within_similarity(a, b)
        assert out["within_serial"] == pytest.approx(1.0, abs=1e-9)
        assert out["within_parallel"] == pytest.approx(1.0, abs=1e-9)
        assert out["between"] == pytest.approx(1.0, abs=1e-9)

    def test_disjoint_legs_fail_criterion(self):
        # High within-set similarity, ~zero between → criterion fails.
        serial = ["alpha alpha alpha", "alpha alpha alpha"]
        parallel = ["beta beta beta", "beta beta beta"]
        out = ab.between_vs_within_similarity(serial, parallel)
        # within should be high (near-identical texts within each leg) ...
        assert out["within_serial"] > 0.9
        assert out["within_parallel"] > 0.9
        # ... but between should be ~0 (disjoint vocab) ...
        assert out["between"] < 0.1
        # ... so the gate criterion between >= min(within) FAILS.
        assert out["between"] < min(out["within_serial"], out["within_parallel"])


# ---------------------------------------------------------------------------
# risk_overlay_determinism
# ---------------------------------------------------------------------------

def _run(rating, tw=None, sl=None, ep=None):
    """Build a minimal run record with the given overlay numbers."""
    return {"rating": rating, "risk_overlay": {
        "target_weight": tw, "stop_loss": sl, "entry_price": ep,
    }}


class TestRiskOverlayDeterminism:
    def test_consistent_groups_pass(self):
        runs = [
            _run("Buy", 40.0, 95.5, 100.0),
            _run("Buy", 40.0, 95.5, 100.0),
            _run("Sell", -30.0, 105.0, 100.0),
            _run("Sell", -30.0, 105.0, 100.0),
        ]
        assert ab.risk_overlay_determinism(runs) is True

    def test_inconsistent_group_fails(self):
        runs = [
            _run("Buy", 40.0, 95.5, 100.0),
            _run("Buy", 35.0, 95.5, 100.0),  # different target_weight
        ]
        assert ab.risk_overlay_determinism(runs) is False

    def test_missing_overlay_in_one_run_fails(self):
        runs = [
            _run("Buy", 40.0, 95.5, 100.0),
            {"rating": "Buy", "risk_overlay": None},  # overlay missing
        ]
        assert ab.risk_overlay_determinism(runs) is False

    def test_all_missing_overlay_passes(self):
        # risk_enabled=False everywhere → vacuously consistent.
        runs = [
            {"rating": "Buy", "risk_overlay": None},
            {"rating": "Buy", "risk_overlay": None},
        ]
        assert ab.risk_overlay_determinism(runs) is True

    def test_float_formatting_tolerated(self):
        # 40.0 vs 40.00 must be treated equal (compared as floats).
        runs = [
            _run("Buy", 40.0, 95.50, 100.0),
            _run("Buy", 40.0, 95.5, 100.00),
        ]
        assert ab.risk_overlay_determinism(runs) is True


# ---------------------------------------------------------------------------
# chi_square_p
# ---------------------------------------------------------------------------

class TestChiSquareP:
    def test_identical_histograms_high_p(self):
        h = {"Buy": 5, "Hold": 3, "Sell": 2}
        assert ab.chi_square_p(h, h) > 0.05

    def test_identical_histograms_near_one(self):
        # Identical distributions → statistic ~0 → p ~1.
        h = {"Buy": 5, "Hold": 5}
        assert ab.chi_square_p(h, h) == pytest.approx(1.0, abs=1e-6)

    def test_disjoint_histograms_low_p(self):
        # Completely opposite distributions → very low p.
        a = {"Buy": 10, "Sell": 0}
        b = {"Buy": 0, "Sell": 10}
        assert ab.chi_square_p(a, b) < 0.05

    def test_single_category_returns_one(self):
        # One category → df=0 → cannot reject → p=1.0.
        assert ab.chi_square_p({"Buy": 5}, {"Buy": 7}) == pytest.approx(1.0)

    def test_fallback_path_forced(self, monkeypatch):
        """Force the manual Wilson-Hilferty fallback and confirm it still
        separates identical vs disjoint distributions."""
        monkeypatch.setattr(ab, "_HAS_SCIPY", False)
        assert ab.chi_square_p({"Buy": 5, "Sell": 5}, {"Buy": 5, "Sell": 5}) > 0.05
        assert ab.chi_square_p({"Buy": 10, "Sell": 0}, {"Buy": 0, "Sell": 10}) < 0.05


# ---------------------------------------------------------------------------
# summarize_gate
# ---------------------------------------------------------------------------

def _passing_metrics(**overrides):
    """A metrics dict that satisfies every gate criterion."""
    base = {
        "n_serial": 6,
        "n_parallel": 6,
        "rating_p": 1.0,
        "overlap_ratio": 1.0,
        "per_field_similarity": {
            "market_report": {"within_serial": 0.5, "within_parallel": 0.5, "between": 0.7},
        },
        "risk_determinism_serial": True,
        "risk_determinism_parallel": True,
        "empty_reports_serial": 0,
        "empty_reports_parallel": 0,
        "exceptions_serial": 0,
        "exceptions_parallel": 0,
        "speedup": 3.0,
        "speedup_na": False,
        "dry_run": False,
    }
    base.update(overrides)
    return base


class TestSummarizeGate:
    def test_passing_metrics(self):
        ok, reasons = ab.summarize_gate(_passing_metrics())
        assert ok is True
        assert reasons == []

    def test_rating_p_too_low_with_low_overlap_fails(self):
        m = _passing_metrics(rating_p=0.01, overlap_ratio=0.2)
        ok, reasons = ab.summarize_gate(m)
        assert ok is False
        assert any("rating chi-square" in r for r in reasons)

    def test_rating_p_low_but_overlap_high_passes(self):
        # Too-few-samples fallback: overlap > 0.5 rescues a low p.
        m = _passing_metrics(rating_p=0.01, overlap_ratio=0.7)
        ok, reasons = ab.summarize_gate(m)
        assert ok is True
        assert reasons == []

    def test_similarity_criterion_fails(self):
        m = _passing_metrics(per_field_similarity={
            "market_report": {"within_serial": 0.9, "within_parallel": 0.9, "between": 0.2},
        })
        ok, reasons = ab.summarize_gate(m)
        assert ok is False
        assert any("market_report" in r for r in reasons)

    def test_risk_determinism_fails(self):
        m = _passing_metrics(risk_determinism_parallel=False)
        ok, reasons = ab.summarize_gate(m)
        assert ok is False
        assert any("parallel" in r and "determinism" in r for r in reasons)

    def test_speedup_below_threshold_fails(self):
        m = _passing_metrics(speedup=1.2)
        ok, reasons = ab.summarize_gate(m)
        assert ok is False
        assert any("speedup" in r for r in reasons)

    def test_speedup_na_passes(self):
        # perf_tracker not exposed → speedup recorded but not enforced.
        m = _passing_metrics(speedup=None, speedup_na=True)
        ok, reasons = ab.summarize_gate(m)
        assert ok is True
        assert reasons == []

    def test_dry_run_skips_speedup(self):
        m = _passing_metrics(speedup=None, dry_run=True)
        ok, reasons = ab.summarize_gate(m)
        assert ok is True
        assert reasons == []

    def test_exceptions_fail(self):
        m = _passing_metrics(exceptions_serial=2)
        ok, reasons = ab.summarize_gate(m)
        assert ok is False
        assert any("serial" in r and "exception" in r for r in reasons)

    def test_empty_reports_fail(self):
        m = _passing_metrics(empty_reports_parallel=1)
        ok, reasons = ab.summarize_gate(m)
        assert ok is False
        assert any("empty" in r for r in reasons)

    def test_multiple_failures_collect_all_reasons(self):
        m = _passing_metrics(
            rating_p=0.01,
            overlap_ratio=0.2,
            risk_determinism_serial=False,
            exceptions_parallel=1,
            speedup=1.0,
        )
        ok, reasons = ab.summarize_gate(m)
        assert ok is False
        # At least one reason per failed criterion.
        assert len(reasons) >= 4


# ---------------------------------------------------------------------------
# parse_risk_overlay (light coverage — used by determinism + real mode)
# ---------------------------------------------------------------------------

class TestParseRiskOverlay:
    def test_parses_all_three_fields(self):
        decision = (
            "## Quantitative Risk Overlay\n\n"
            "- **Action**: Buy\n"
            "- **Target Weight**: 40.0%\n"
            "- **Stop Loss**: 95.50\n"
            "- **Entry Reference**: 100.00\n"
        )
        out = ab.parse_risk_overlay(decision)
        assert out == {"target_weight": 40.0, "stop_loss": 95.5, "entry_price": 100.0}

    def test_missing_marker_returns_none(self):
        assert ab.parse_risk_overlay("just a normal decision, no overlay") is None

    def test_empty_returns_none(self):
        assert ab.parse_risk_overlay("") is None


# ---------------------------------------------------------------------------
# main(["--dry-run", ...]) end-to-end
# ---------------------------------------------------------------------------

class TestDryRunMain:
    def test_writes_report_and_exits_zero(self, tmp_path):
        out_dir = tmp_path / "ab_out"
        rc = ab.main([
            "--dry-run",
            "--tickers", "AAPL",
            "--date", "2026-03-15",
            "--n", "6",
            "--out-dir", str(out_dir),
        ])
        assert rc == 0
        report = out_dir / "parallel_ab_AAPL_2026-03-15.md"
        assert report.exists()
        text = report.read_text(encoding="utf-8")
        assert "DRY RUN" in text
        # Dry-run is constructed to PASS (identical multisets per leg).
        assert "Verdict: PASS" in text

    def test_writes_metrics_json(self, tmp_path):
        out_dir = tmp_path / "ab_json"
        rc = ab.main([
            "--dry-run", "--tickers", "NVDA",
            "--date", "2026-04-01", "--n", "4",
            "--out-dir", str(out_dir),
        ])
        assert rc == 0
        assert (out_dir / "parallel_ab_NVDA_2026-04-01.json").exists()

    def test_multiple_tickers(self, tmp_path):
        out_dir = tmp_path / "ab_multi"
        rc = ab.main([
            "--dry-run",
            "--tickers", "AAPL", "NVDA", "MSFT",
            "--date", "2026-03-15", "--n", "4",
            "--out-dir", str(out_dir),
        ])
        assert rc == 0
        for t in ("AAPL", "NVDA", "MSFT"):
            assert (out_dir / f"parallel_ab_{t}_2026-03-15.md").exists()

    def test_invalid_n_returns_two(self, tmp_path):
        rc = ab.main([
            "--dry-run", "--tickers", "AAPL", "--date", "2026-03-15",
            "--n", "0", "--out-dir", str(tmp_path),
        ])
        assert rc == 2
