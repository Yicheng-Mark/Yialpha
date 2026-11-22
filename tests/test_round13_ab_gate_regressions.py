"""Round-13 A/B gate + baseline regressions (8 confirmed findings).

Covers the round-13 fixes in scripts/run_analyst_parallel_ab.py and
scripts/run_baseline.py:

#1  TF-IDF between/within from ONE fitted space (H0 false-FAIL fixed)
#2  _build_graph forces llm_cache=False (+ loud preflight warning)
#3  one discarded warm-up per leg before the n sampled runs
#4  _build_graph forces memory_enabled=False
#5  risk-overlay determinism is tri-state (N/A, not hollow PASS)
#6  sparse contingency tables: Fisher exact / unified no-Yates chi-square
#7  infra crash exits 3, never conflated with the statistical FAIL (1)
#8  run_baseline ``gate_{t}.md`` sanitizes the ticker before writing

The A/B script is loaded as an isolated module (importlib, same as
tests/test_run_analyst_parallel_ab.py) so no ``__main__`` guard fires.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
from pathlib import Path
from random import Random
from types import SimpleNamespace

import pytest

_SCRIPT_PATH = (
    Path(__file__).resolve().parent.parent / "scripts" / "run_analyst_parallel_ab.py"
)
_spec = importlib.util.spec_from_file_location(
    "run_analyst_parallel_ab_round13", _SCRIPT_PATH
)
assert _spec is not None and _spec.loader is not None
ab = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ab)

import scripts.run_baseline as runner  # noqa: E402


# ---------------------------------------------------------------------------
# #1 — TF-IDF between/within must share ONE fitted space
# ---------------------------------------------------------------------------

#: i.i.d. construction: Zipf-weighted sampling from a shared word pool.
#: Natural text has Zipfian word frequencies, and the frequency spread is
#: what makes per-text "centrality" vary — the first-order effect that
#: keeps the between statistic between the two within statistics under H0.
_MC_POOL = [f"w{i:03d}" for i in range(30)]
_MC_WEIGHTS = [1.0 / (i + 1) ** 1.5 for i in range(30)]
_MC_WORDS_PER_TEXT = 12
_MC_N_PER_LEG = 12


def _iid_leg(rng: Random, n: int) -> list[str]:
    return [
        " ".join(rng.choices(_MC_POOL, weights=_MC_WEIGHTS, k=_MC_WORDS_PER_TEXT))
        for _ in range(n)
    ]


def _criterion2_pass(serial: list[str], parallel: list[str]) -> bool:
    out = ab.between_vs_within_similarity(serial, parallel)
    return out["between"] + 1e-9 >= min(out["within_serial"], out["within_parallel"])


class TestTfidfSameSpaceH0:
    """Round-13 #1: under the i.i.d. null the between/within criterion must
    not systematically reject.

    PRE-FIX evidence (split fitting spaces: between fitted on the serial
    +parallel union N=2n, each within fitted on its own leg N=n — leg-unique
    terms gained ~+ln2 idf, inflating norms while never matching across
    legs): on THIS construction and these exact seeds 0..19 the old code
    passed only 13/20 (7 false FAILs; other seed blocks went as low as
    4/20 passes), and the round-13 scanner's construction false-FAILED
    39/40. The fix fits ONE union space and computes within and between
    from the same L2-normalized vectors.
    """

    def test_iid_legs_pass_rate_at_least_18_of_20(self):
        passes = 0
        for seed in range(20):
            rng = Random(seed)
            serial = _iid_leg(rng, _MC_N_PER_LEG)
            parallel = _iid_leg(rng, _MC_N_PER_LEG)
            if _criterion2_pass(serial, parallel):
                passes += 1
        assert passes >= 18, (
            f"criterion 2 under i.i.d. H0 passed only {passes}/20 "
            f"(seeds 0..19, n={_MC_N_PER_LEG}/leg) — the between/within "
            "comparison must not be systematically biased on identically "
            "distributed legs"
        )

    def test_identical_multisets_between_at_least_min_within(self):
        # Retained semantics: identical multisets pass criterion 2 (between
        # includes the cos(x,x)=1.0 diagonal pairs, so it dominates).
        a = ["alpha beta", "beta gamma", "alpha gamma"]
        out = ab.between_vs_within_similarity(a, list(a))
        assert out["between"] >= min(out["within_serial"],
                                     out["within_parallel"]) - 1e-9

    def test_disjoint_vocab_legs_still_fail(self):
        # Retained semantics: genuinely disjoint vocabulary must FAIL.
        serial = ["alpha alpha alpha", "alpha alpha alpha"]
        parallel = ["beta beta beta", "beta beta beta"]
        out = ab.between_vs_within_similarity(serial, parallel)
        assert out["between"] < min(out["within_serial"], out["within_parallel"])

    def test_single_text_identical_legs_all_one(self):
        out = ab.between_vs_within_similarity(["alpha beta gamma"],
                                              ["alpha beta gamma"])
        assert out["within_serial"] == pytest.approx(1.0, abs=1e-9)
        assert out["within_parallel"] == pytest.approx(1.0, abs=1e-9)
        assert out["between"] == pytest.approx(1.0, abs=1e-9)

    def test_between_field_equals_tfidf_cosine_similarity(self):
        # The between statistic of the criterion helper and the standalone
        # mean-pairwise function now share the SAME union fit (pre-fix they
        # were the only union-fitted side of the comparison).
        rng = Random(7)
        a = _iid_leg(rng, 5)
        b = _iid_leg(rng, 5)
        out = ab.between_vs_within_similarity(a, b)
        assert out["between"] == pytest.approx(
            ab.tfidf_cosine_similarity(a, b), abs=1e-12
        )

    def test_sklearn_and_manual_paths_same_space(self, monkeypatch):
        """Both backends must fit ONE space with the same smooth-idf/L2
        semantics, so the flag flip cannot move the criterion."""
        rng = Random(11)
        a = _iid_leg(rng, 6)
        b = _iid_leg(rng, 6)
        monkeypatch.setattr(ab, "_HAS_SKLEARN", True)
        with_sklearn = ab.between_vs_within_similarity(a, b)
        monkeypatch.setattr(ab, "_HAS_SKLEARN", False)
        manual = ab.between_vs_within_similarity(a, b)
        for key in ("within_serial", "within_parallel", "between"):
            assert with_sklearn[key] == pytest.approx(manual[key], abs=1e-9), key


# ---------------------------------------------------------------------------
# #6 — sparse contingency tables: Fisher exact + unified no-Yates policy
# ---------------------------------------------------------------------------

class TestChiSquareSparsePolicy:
    """Round-13 #6: at the gate's n=10/5 categories the expected cell counts
    are far below 5, and the scipy path (Yates for 2x2 via
    ``chi2_contingency`` default) disagreed with the manual path (never
    corrected) across the 0.05 boundary. Policy now: sparse 2x2 -> Fisher
    exact (exact integer arithmetic, identical on both paths); sparse k>2 ->
    chi-square with the SAME no-correction statistic on both paths and a
    disclosed warning; non-sparse -> chi-square with explicit
    ``correction=False`` on the scipy path.
    """

    def test_sparse_boundary_2x2_both_paths_agree_and_match_scipy(self, monkeypatch):
        # min expected = 3.29 < 5 -> Fisher. PRE-FIX this table straddled
        # the boundary between the two paths: scipy-with-Yates gave
        # p=0.0765 (PASS) while the manual no-correction chi-square gave
        # p=0.0235 (FAIL) — same table, opposite verdicts.
        hist_a = {"Buy": 1, "Hold": 6}
        hist_b = {"Buy": 7, "Hold": 3}

        monkeypatch.setattr(ab, "_HAS_SCIPY", False)
        p_manual, method_manual = ab.chi_square_test(hist_a, hist_b)
        monkeypatch.setattr(ab, "_HAS_SCIPY", True)
        p_scipy, method_scipy = ab.chi_square_test(hist_a, hist_b)

        # Sparse 2x2 runs the exact-integer Fisher on BOTH paths.
        assert "Fisher" in method_manual
        assert p_manual == pytest.approx(p_scipy, abs=1e-12)
        assert (p_manual > 0.05) == (p_scipy > 0.05)
        # Exact value: sum of tables no more probable than the observed one
        # = 968 / C(17,7) = 968/19448.
        assert p_manual == pytest.approx(968.0 / 19448.0, abs=1e-12)
        assert p_manual < 0.05  # rejects (decisively, not at the knife edge)

    def test_fisher_matches_scipy_reference(self):
        pytest.importorskip("scipy")
        from scipy.stats import fisher_exact

        for table in (((1, 6), (7, 3)), ((2, 12), (2, 0)), ((5, 5), (5, 5)),
                      ((9, 1), (2, 8))):
            (a, b), (c, d) = table
            expected = fisher_exact([list(r) for r in table])[1]
            assert ab._fisher_exact_2x2(a, b, c, d) == pytest.approx(
                expected, abs=1e-12
            ), table

    def test_sparse_2x2_identical_histograms_fisher_is_one(self):
        # expected 4*4/8 = 2 < 5 -> Fisher; identical legs make the observed
        # table the most probable one, so every table is "no more improbable"
        # and the exact two-sided p is 1.0.
        p, method = ab.chi_square_test({"Buy": 2, "Hold": 2},
                                       {"Buy": 2, "Hold": 2})
        assert "Fisher" in method
        assert p == pytest.approx(1.0, abs=1e-12)

    def test_sparse_k_gt_2_discloses_warning_and_paths_same_sign(self, monkeypatch):
        # min expected = 2.0 < 5, 3 categories -> chi-square on both paths,
        # with the sparse-cells caveat disclosed in the method string.
        hist_a = {"Buy": 6, "Hold": 3, "Sell": 1}
        hist_b = {"Buy": 2, "Hold": 5, "Sell": 3}

        monkeypatch.setattr(ab, "_HAS_SCIPY", False)
        p_manual, method = ab.chi_square_test(hist_a, hist_b)
        assert "SPARSE" in method
        monkeypatch.setattr(ab, "_HAS_SCIPY", True)
        p_scipy, _ = ab.chi_square_test(hist_a, hist_b)
        assert (p_manual > 0.05) == (p_scipy > 0.05)
        assert 0.0 < p_manual < 1.0 and 0.0 < p_scipy < 1.0

    def test_nonsparse_2x2_unified_no_yates(self, monkeypatch):
        # min expected = 8 >= 5 -> chi-square; scipy path must pass
        # correction=False so both paths compute the SAME statistic (the
        # pre-fix scipy default applied Yates and could flip the verdict).
        hist_a = {"Buy": 16, "Sell": 4}
        hist_b = {"Buy": 8, "Sell": 12}

        monkeypatch.setattr(ab, "_HAS_SCIPY", False)
        p_manual, method = ab.chi_square_test(hist_a, hist_b)
        assert method == "chi-square"
        monkeypatch.setattr(ab, "_HAS_SCIPY", True)
        p_scipy, _ = ab.chi_square_test(hist_a, hist_b)
        # Wilson-Hilferty is accurate to ~0.01 in this tail; the SIGN vs
        # the gate threshold is what must agree.
        assert (p_manual > 0.05) == (p_scipy > 0.05)
        assert abs(p_manual - p_scipy) < 0.02

        pytest.importorskip("scipy")
        from scipy.stats import chi2_contingency

        _s, expected, _dof, _e = chi2_contingency([[16, 4], [8, 12]],
                                                  correction=False)
        assert p_scipy == pytest.approx(expected, abs=1e-12)

    def test_degenerate_and_empty_inputs(self):
        assert ab.chi_square_p({"Buy": 5}, {"Buy": 7}) == pytest.approx(1.0)
        assert ab.chi_square_p({}, {}) == pytest.approx(1.0)
        assert ab.chi_square_p({"Buy": 3}, {}) == pytest.approx(1.0)

    def test_zero_column_dropped_not_sparse_flipped(self):
        # A category with zero count in BOTH legs carries no information;
        # dropping it keeps this table degenerate (p=1) instead of feeding
        # chi-square a structural zero column.
        p, method = ab.chi_square_test({"Buy": 5, "Sell": 0},
                                       {"Buy": 3, "Sell": 0})
        assert p == pytest.approx(1.0)
        assert "degenerate" in method

    def test_compute_metrics_discloses_method(self):
        def runs(ratings):
            return [
                {"rating": rr, "market_report": "m", "sentiment_report": "s",
                 "news_report": "n", "fundamentals_report": "f",
                 "final_trade_decision": "d", "risk_overlay": None,
                 "wall_full": 10.0, "analyst_segment": 5.0}
                for rr in ratings
            ]

        m = ab._compute_metrics(runs(["Buy"]), runs(["Hold"]), dry_run=False)
        assert m["rating_test_method"]
        assert isinstance(m["rating_p"], float)


# ---------------------------------------------------------------------------
# #5 — risk-overlay determinism tri-state
# ---------------------------------------------------------------------------

def _run(rating, overlay):
    return {"rating": rating, "risk_overlay": overlay}


_ALL_NONE_OVERLAY = {"target_weight": None, "stop_loss": None, "entry_price": None}


class TestRiskOverlayDeterminismNA:
    """Round-13 #5: when the overlay marker is present but every regex
    misses (three all-None fields), the pre-fix per-field ``None == None``
    comparison silently counted the group as "equal" and criterion 3
    PASSed vacuously. Such groups are now N/A; ALL groups N/A -> the
    criterion reports N/A instead of PASS.
    """

    def test_marker_without_any_number_is_na(self):
        runs = [_run("Buy", dict(_ALL_NONE_OVERLAY)),
                _run("Buy", dict(_ALL_NONE_OVERLAY))]
        assert ab.risk_overlay_determinism(runs) is None

    def test_parse_marker_without_bullets_feeds_the_na_path(self):
        parsed = ab.parse_risk_overlay("## Quantitative Risk Overlay\n(no bullets)")
        assert parsed is not None
        assert parsed == _ALL_NONE_OVERLAY

    def test_na_group_plus_verified_group_is_true(self):
        runs = [
            _run("Buy", {"target_weight": 40.0, "stop_loss": 95.5,
                         "entry_price": 100.0}),
            _run("Buy", {"target_weight": 40.0, "stop_loss": 95.5,
                         "entry_price": 100.0}),
            _run("Hold", dict(_ALL_NONE_OVERLAY)),
            _run("Hold", dict(_ALL_NONE_OVERLAY)),
        ]
        assert ab.risk_overlay_determinism(runs) is True

    def test_parsed_next_to_unparsed_in_same_group_fails(self):
        # A deterministic overlay renders numbers for the same rating; a
        # numberless sibling is an inconsistency, not an N/A.
        runs = [
            _run("Buy", {"target_weight": 40.0, "stop_loss": 95.5,
                         "entry_price": 100.0}),
            _run("Buy", dict(_ALL_NONE_OVERLAY)),
        ]
        assert ab.risk_overlay_determinism(runs) is False

    def test_all_overlays_absent_still_vacuously_true(self):
        runs = [_run("Buy", None), _run("Buy", None)]
        assert ab.risk_overlay_determinism(runs) is True

    def test_summarize_gate_treats_na_as_neither_pass_nor_fail(self):
        base = {
            "n_serial": 6, "n_parallel": 6, "rating_p": 1.0,
            "overlap_ratio": 1.0, "per_field_similarity": {}, "dry_run": False,
            "empty_reports_serial": 0, "empty_reports_parallel": 0,
            "exceptions_serial": 0, "exceptions_parallel": 0,
            "speedup": 3.0, "speedup_na": False,
            "risk_determinism_serial": None,
            "risk_determinism_parallel": True,
        }
        ok, reasons = ab.summarize_gate(base)
        assert ok is True
        assert not any("determinism" in r for r in reasons)
        # ... while a real violation still fails.
        base["risk_determinism_serial"] = False
        ok2, reasons2 = ab.summarize_gate(base)
        assert ok2 is False
        assert any("determinism" in r for r in reasons2)

    def test_report_renders_na_honestly(self, tmp_path):
        metrics = {
            "n_serial": 2, "n_parallel": 2,
            "serial_hist": {"Buy": 2}, "parallel_hist": {"Buy": 2},
            "rating_p": 1.0, "overlap_ratio": 1.0,
            "per_field_similarity": {}, "rating_test_method": "chi-square",
            "risk_determinism_serial": None,
            "risk_determinism_parallel": None,
            "empty_reports_serial": 0, "empty_reports_parallel": 0,
            "exceptions_serial": 0, "exceptions_parallel": 0,
            "speedup": None, "speedup_na": True, "dry_run": False,
        }
        path = tmp_path / "report.md"
        ab.write_report(path, "AAPL", "2026-03-15", 2, metrics)
        text = path.read_text(encoding="utf-8")
        assert "N/A (overlay present but no numbers parsed; not verified)" in text
        assert "Serial leg: PASS" not in text


# ---------------------------------------------------------------------------
# #2 / #4 — _build_graph forces llm_cache / memory_enabled off (+ warnings)
# ---------------------------------------------------------------------------

class TestBuildGraphForcesStatelessConfig:
    def _captured_config(self, monkeypatch, leg="parallel", **env_overrides):
        import yialpha.default_config as dc
        import yialpha.graph.trading_graph as tg

        captured = {}

        class _FakeGraph:
            def __init__(self, config=None, **_kw):
                captured.update(config or {})

        monkeypatch.setattr(dc, "DEFAULT_CONFIG", {**dc.DEFAULT_CONFIG, **env_overrides})
        monkeypatch.setattr(tg, "YiAlphaGraph", _FakeGraph)
        args = SimpleNamespace(max_threads=None)
        ab._build_graph(leg, args)
        return captured

    def test_llm_cache_forced_off_even_when_env_sets_true(self, monkeypatch):
        # YIALPHA_LLM_CACHE=true replays byte-identical LLM responses across
        # runs — both legs would be the same tape and all four criteria
        # would hollow-PASS. The gate must hard-disable it.
        captured = self._captured_config(monkeypatch, llm_cache=True)
        assert captured["llm_cache"] is False

    def test_memory_enabled_forced_off_even_when_env_sets_true(self, monkeypatch):
        # Mirrors run_baseline.py:106-109: a shared memory log makes the
        # legs stateful (non-i.i.d.), violating every criterion's null.
        captured = self._captured_config(monkeypatch, memory_enabled=True)
        assert captured["memory_enabled"] is False

    def test_leg_flags_and_batch_isolation_unchanged(self, monkeypatch):
        captured = self._captured_config(monkeypatch, leg="parallel")
        assert captured["analyst_parallel"] is True
        assert captured["batch_workers"] == 1
        captured_s = self._captured_config(monkeypatch, leg="serial")
        assert captured_s["analyst_parallel"] is False


class TestPreflightWarnings:
    def _warn(self, monkeypatch, **overrides):
        import yialpha.default_config as dc

        monkeypatch.setattr(dc, "DEFAULT_CONFIG",
                            {**dc.DEFAULT_CONFIG, **overrides})
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            ab._preflight_warn(SimpleNamespace())
        return err.getvalue()

    def test_warns_loudly_when_llm_cache_env_true(self, monkeypatch):
        err = self._warn(monkeypatch, llm_cache=True)
        assert "YIALPHA_LLM_CACHE" in err
        assert "llm_cache=False" in err

    def test_warns_when_memory_enabled_env_true(self, monkeypatch):
        err = self._warn(monkeypatch, memory_enabled=True)
        assert "memory_enabled=False" in err

    def test_no_state_warning_on_clean_config(self, monkeypatch):
        err = self._warn(monkeypatch, llm_cache=False, memory_enabled=False,
                         llm_rate_limiter=False)
        assert "YIALPHA_LLM_CACHE" not in err
        assert "memory_enabled" not in err


# ---------------------------------------------------------------------------
# #3 — one discarded warm-up per leg, symmetric n-sample statistics
# ---------------------------------------------------------------------------

class _LegGraph:
    def __init__(self, leg: str):
        self.leg = leg


def _canned_run(leg: str) -> dict:
    return {
        "ticker": "AAPL", "rating": "Hold",
        "market_report": "shared analysis text about momentum",
        "sentiment_report": "shared analysis text about sentiment",
        "news_report": "shared analysis text about news flow",
        "fundamentals_report": "shared analysis text about fundamentals",
        "final_trade_decision": "**Rating**: Hold",
        "risk_overlay": None,
        "wall_full": 100.0 if leg == "serial" else 30.0,
        "analyst_segment": 80.0 if leg == "serial" else 20.0,
        "prediction_analysts": [],
        "exception": None,
    }


class TestWarmupRuns:
    def test_each_leg_runs_one_discarded_warmup_before_its_samples(
        self, monkeypatch, tmp_path, capsys
    ):
        legs: list[str] = []

        def fake_build(leg, _args):
            return _LegGraph(leg)

        def fake_capture(graph, _ticker, _date):
            legs.append(graph.leg)
            return _canned_run(graph.leg)

        monkeypatch.setattr(ab, "_preflight_warn", lambda _a: None)
        monkeypatch.setattr(ab, "_build_graph", fake_build)
        monkeypatch.setattr(ab, "_capture_run", fake_capture)

        rc = ab.main(["--tickers", "AAPL", "--date", "2026-03-15", "--n", "2",
                      "--out-dir", str(tmp_path)])
        assert rc == 0
        # warm-up first on each leg, then exactly n sampled runs per leg.
        assert legs == (["serial"] + ["serial"] * 2
                        + ["parallel"] + ["parallel"] * 2)
        out = capsys.readouterr().out
        assert "serial warm-up (discarded)" in out
        assert "parallel warm-up (discarded)" in out
        # The warm-up runs are NOT counted into the statistics.
        metrics = json.loads(
            (tmp_path / "parallel_ab_AAPL_2026-03-15.json").read_text("utf-8")
        )
        assert metrics["n_serial"] == 2
        assert metrics["n_parallel"] == 2
        assert metrics["speedup"] == pytest.approx(4.0)


# ---------------------------------------------------------------------------
# #7 — infra crash exits 3, distinct from FAIL's 1
# ---------------------------------------------------------------------------

class TestInfraCrashExitThree:
    def test_graph_build_crash_returns_three_not_one(self, monkeypatch, capsys,
                                                     tmp_path):
        def boom(_leg, _args):
            raise RuntimeError("missing key 'analyst_parallel_max_threads'")

        monkeypatch.setattr(ab, "_preflight_warn", lambda _a: None)
        monkeypatch.setattr(ab, "_build_graph", boom)
        rc = ab.main(["--tickers", "AAPL", "--date", "2026-03-15", "--n", "2",
                      "--out-dir", str(tmp_path)])
        assert rc == 3
        assert rc != 1  # never conflated with the statistical FAIL code
        err = capsys.readouterr().err
        assert "NOT a statistical FAIL" in err

    def test_usage_errors_still_two(self, monkeypatch):
        monkeypatch.setattr(ab, "_build_graph",
                            lambda *_: (_ for _ in ()).throw(RuntimeError()))
        rc = ab.main(["--tickers", "AAPL", "--date", "bad", "--n", "1"])
        assert rc == 2


# ---------------------------------------------------------------------------
# #8 — run_baseline gate_{t}.md ticker sanitization
# ---------------------------------------------------------------------------

class TestGateFilenameSanitized:
    @pytest.mark.parametrize("raw,expected", [
        ("BTC:USDT", "BTC_USDT"),   # ':' reserved on Windows
        ("a/b", "a_b"),             # path separator -> nested dir / crash
        ("a\\b", "a_b"),
        ("AAPL", "AAPL"),           # normal tickers unchanged
        ("600519.SS", "600519.SS"), # '.' kept (A-share suffix convention)
    ])
    def test_safe_filename_component(self, raw, expected):
        assert runner._safe_filename_component(raw) == expected

    def test_full_ab_writes_sanitized_gate_file_for_hostile_ticker(
        self, monkeypatch, tmp_path
    ):
        """End-to-end pin: the gate write is the LAST step of full_ab (after
        the whole LLM spend) — an unsanitized 'BTC:USDT' used to raise
        OSError here and void the entire run's spend."""
        monkeypatch.setattr(runner, "_rebalance_dates",
                            lambda *a: ["2024-01-02"])
        monkeypatch.setattr(
            runner, "_map_tickers",
            lambda tickers, task, workers, risk_enabled=None, asset_type="stock":
                [task(t, object()) for t in tickers],
        )
        monkeypatch.setattr(runner.RiskManager, "from_config",
                            lambda cfg: object())
        monkeypatch.setattr(runner, "build_backtest_weight_fn",
                            lambda mgr, t: (lambda rating, date, ctx: 0.05))
        monkeypatch.setattr(
            runner, "run_backtest",
            lambda graph, ticker, dates, **kw: SimpleNamespace(
                ticker=ticker,
                metrics=SimpleNamespace(total_return=0.1, sharpe=1.0,
                                        max_drawdown=-0.05,
                                        deflated_sharpe=0.6),
            ),
        )
        monkeypatch.setattr(
            runner, "evaluate_gate",
            lambda base, improved: SimpleNamespace(
                passes=True, mean_dsr=0.6, beats_buyhold=True,
                recommendation="ok",
                render=lambda: "# gate verdict",
            ),
        )
        monkeypatch.setattr(runner, "write_report",
                            lambda *a, **k: tmp_path / "report.md")
        monkeypatch.setattr(runner, "write_dashboard", lambda *a, **k: None)

        ok = runner.full_ab(["BTC:USDT", "a/b"], "2024-01-01", "2024-02-01",
                            5, 1, 5, 1.0, 1, str(tmp_path), workers=1)
        assert ok is True
        assert (tmp_path / "gate_BTC_USDT.md").read_text("utf-8") == "# gate verdict"
        assert (tmp_path / "gate_a_b.md").exists()
        # No raw-ticker file (would have crashed / nested) was produced.
        assert not (tmp_path / "gate_BTC:USDT.md").exists()
