"""Round-8 known-issue landing pins (agents / ledger seam / A-B gate).

Covers the landed S3 items (prediction-capture isolation in the fan-out,
parse_rating negation window, positioning direction clamp), the S6-b shared
reference-seam verdict, and the A/B gate's exception-leg filtering plus the
two new gate criteria. Hermetic: no network, no LLM.
"""

from __future__ import annotations

import contextvars
import importlib.util
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

import yialpha.agents.utils.prediction_tools as pt
from yialpha.agents.schemas import clamp_direction_language
from yialpha.agents.utils.rating import parse_rating
from yialpha.ledger import time_contract as tc
from yialpha.ledger.time_contract import (
    PERP_INSTRUMENT_CLASSES,
    REFERENCE_SOURCE_EQUITY,
    REFERENCE_SOURCE_PERP,
    reference_seam_equity,
)

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _clean_capture_scope():
    pt.reset_prediction_capture_for_test()
    yield
    pt.reset_prediction_capture_for_test()


# ---------------------------------------------------------------------------
# S3-a: per-subgraph prediction-capture isolation
# ---------------------------------------------------------------------------


class TestBindIsolatedPredictionCaptureScope:
    def test_fresh_registry_in_bound_context_only(self):
        pt.ensure_prediction_capture_scope()
        parent_registry = pt._CAPTURE_SCOPE.get()
        assert parent_registry is not None

        seen = {}

        def worker():
            pt.bind_isolated_prediction_capture_scope()
            seen["worker"] = pt._CAPTURE_SCOPE.get()

        ctx = contextvars.copy_context()
        ctx.run(worker)

        worker_registry = seen["worker"]
        assert worker_registry is not None
        assert worker_registry is not parent_registry  # a FRESH dict
        assert worker_registry == {}                    # and empty
        # The parent context still sees its original registry — the bind in
        # the copied context never leaks out.
        assert pt._CAPTURE_SCOPE.get() is parent_registry

    def test_double_bind_replaces_with_fresh_dict(self):
        pt.bind_isolated_prediction_capture_scope()
        first = pt._CAPTURE_SCOPE.get()
        first["stale"] = 1
        pt.bind_isolated_prediction_capture_scope()
        second = pt._CAPTURE_SCOPE.get()
        assert second is not first
        assert second == {}  # no residue from the previous registry


# ---------------------------------------------------------------------------
# S3-b: parse_rating negation window
# ---------------------------------------------------------------------------


class TestParseRatingNegation:
    def test_against_sell_falls_through_to_next_unnegated(self):
        text = "The thesis argues against Sell, maintain Overweight."
        assert parse_rating(text) == "Overweight"

    def test_counter_to_sell_not_captured(self):
        assert parse_rating("Position is counter to Sell for now.") == "Hold"

    def test_not_sell_not_captured(self):
        assert parse_rating("This is not Sell.") == "Hold"

    def test_negators_within_window(self):
        assert parse_rating("argues against Buy") == "Hold"
        assert parse_rating("counter to Sell elsewhere") == "Hold"
        # Falsifiable variant (review P3-7): the skipped word must differ
        # from the default so the assertion can actually fail.
        assert parse_rating("rather than Sell, maintain Overweight.") == "Overweight"

    def test_negator_outside_window_still_captures(self):
        # Window is 2 tokens: "not" sits 4 tokens before "Sell" — the
        # negation no longer applies, the rating word wins.
        assert parse_rating("not the case here, Sell") == "Sell"

    def test_label_path_unaffected(self):
        assert parse_rating("Rating: **Sell**") == "Sell"
        assert parse_rating("Rating - Buy") == "Buy"

    def test_plain_positive_rating_unchanged(self):
        assert parse_rating("We rate it a Buy after the review.") == "Buy"


# ---------------------------------------------------------------------------
# S3-c: positioning direction clamp (free-text fallback enforcement)
# ---------------------------------------------------------------------------


class TestClampDirectionLanguage:
    def test_direction_line_removed_with_banner(self):
        text = "Funding is elevated.\nI would go long here at these levels.\nWatch crowding."
        out = clamp_direction_language(text)
        assert "go long" not in out
        assert "[direction statement removed — positioning freeze]" in out
        assert "Funding is elevated." in out   # untouched lines survive
        assert "Watch crowding." in out
        assert "positioning direction freeze" in out  # observable banner

    def test_imperative_variants_removed(self):
        for line in (
            "Traders should open a short position",
            "This is a strong bullish call",
            "The bullish setup is intact",
            "Consider a long entry on strength",
            "Enter a short on a break",
            # Review P3-4: -ing forms are direction claims too.
            "We are going long here at these levels",
            "Position is entering a long on strength",
            "Managers are opening a short into the print",
        ):
            out = clamp_direction_language(line)
            assert "[direction statement removed — positioning freeze]" in out, line

    def test_descriptive_entry_narrative_survives(self):
        # Review P3-4: "long entry" as a descriptive noun (observed levels)
        # is legitimate content — only verb-led imperative phrasing strips.
        text = "long entry levels were observed by takers today"
        assert clamp_direction_language(text) == text

    def test_taker_flow_narrative_survives(self):
        # buy/sell FLOW is legitimate positioning content — the clamp only
        # strips imperative position phrasing.
        text = "Taker buy flow remains heavy into the close; sell flow faded."
        assert clamp_direction_language(text) == text

    def test_clean_text_unchanged(self):
        text = "Funding bias: warm. Crowding: moderate. Liquidity: ample."
        assert clamp_direction_language(text) == text


# ---------------------------------------------------------------------------
# S6-b: the ONE shared reference-seam verdict
# ---------------------------------------------------------------------------


class TestReferenceSeamEquity:
    @pytest.mark.parametrize(
        ("scope", "instrument_class", "asset_type", "expected"),
        [
            ("UNDERLYING", "stock_perp", None, True),
            ("UNDERLYING", "stock_perp", "crypto_perp", True),
            ("UNDERLYING", "unknown_perp", None, False),   # the KI divergence
            ("UNDERLYING", "pure_crypto_perp", None, False),
            ("CONTRACT", "pure_crypto_perp", None, False),
            ("CONTRACT", "stock_perp", None, False),
            ("CONTRACT", "stock_perp", "stock", False),
            ("CONTRACT", None, None, True),                # unclassified: equity
            ("CONTRACT", None, "crypto_perp", False),
            ("CONTRACT", "equity", None, True),
            (None, None, None, True),
        ],
    )
    def test_truth_table(self, scope, instrument_class, asset_type, expected):
        assert (
            reference_seam_equity(scope, instrument_class, asset_type)
            is expected
        )

    def test_perp_classes_shared_spelling(self):
        # outcome_compute and the backtest engine import the SAME frozenset.
        from yialpha.backtest import engine as be
        from yialpha.ledger import outcome_compute as oc

        assert oc._PERP_INSTRUMENT_CLASSES is PERP_INSTRUMENT_CLASSES
        assert be._PERP_INSTRUMENT_CLASSES is PERP_INSTRUMENT_CLASSES
        assert "unknown_perp" in PERP_INSTRUMENT_CLASSES

    def test_capture_adopts_scoring_seam(self, monkeypatch):
        # The unknown_perp underlying row that used to store the equity
        # source (and retire as a permanent mismatch) now stores perp.
        monkeypatch.setattr(
            tc, "_reference_frame",
            lambda *a, **k: pd.DataFrame(
                {"Close": [100.0]}, index=pd.Index(["2026-09-04"])),
        )
        instants = iter(
            datetime(2026, 9, 5, 12, tzinfo=UTC) for _ in range(10))
        monkeypatch.setattr(tc, "_now_utc", lambda: next(instants))
        timing = tc.capture_prediction_timing(
            "MUUSDT", "UNDERLYING",
            instrument_class="unknown_perp", asset_type="crypto_perp",
        )
        assert timing["reference_source"] == REFERENCE_SOURCE_PERP

    def test_capture_equity_default_for_unclassified(self, monkeypatch):
        monkeypatch.setattr(
            tc, "_reference_frame",
            lambda *a, **k: pd.DataFrame(
                {"Close": [100.0]}, index=pd.Index(["2026-09-04"])),
        )
        instants = iter(
            datetime(2026, 9, 5, 12, tzinfo=UTC) for _ in range(10))
        monkeypatch.setattr(tc, "_now_utc", lambda: next(instants))
        timing = tc.capture_prediction_timing("MU", "UNDERLYING")
        assert timing["reference_source"] == REFERENCE_SOURCE_EQUITY


# ---------------------------------------------------------------------------
# S5-a + S3-a gate: A/B script exception filtering and new criteria
# ---------------------------------------------------------------------------

_SCRIPT_PATH = (
    Path(__file__).resolve().parent.parent / "scripts" / "run_analyst_parallel_ab.py"
)
_spec = importlib.util.spec_from_file_location(
    "run_analyst_parallel_ab_ki", _SCRIPT_PATH
)
assert _spec is not None and _spec.loader is not None
ab = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ab)


def _ok_run(leg: str, rating: str = "Buy", text: str = "solid analysis text") -> dict:
    return {
        "ticker": "TST",
        "rating": rating,
        "market_report": text,
        "sentiment_report": text,
        "news_report": text,
        "fundamentals_report": text,
        "final_trade_decision": "Rating: " + rating,
        "risk_overlay": {"entry_price": 10.0, "stop": 9.0, "target": 12.0},
        "wall_full": 60.0 if leg == "serial" else 20.0,
        "analyst_segment": 40.0 if leg == "serial" else 12.0,
        "prediction_analysts": [],
        "exception": None,
    }


def _failed_run(leg: str) -> dict:
    run = _ok_run(leg)
    run["rating"] = "?"
    for f in ("market_report", "sentiment_report", "news_report",
              "fundamentals_report", "final_trade_decision"):
        run[f] = ""
    run["analyst_segment"] = None
    run["exception"] = "RuntimeError('propagate died')"
    return run


class TestComputeMetricsExceptionFiltering:
    def test_failed_runs_excluded_from_stats_counted_in_exceptions(self):
        serial = [_ok_run("serial"), _failed_run("serial"), _ok_run("serial")]
        parallel = [_ok_run("parallel"), _failed_run("parallel"), _ok_run("parallel")]
        m = ab._compute_metrics(serial, parallel)
        assert m["n_serial"] == 3 and m["n_parallel"] == 3
        assert m["dropped_serial"] == 1 and m["dropped_parallel"] == 1
        assert "?" not in m["serial_hist"] and "?" not in m["parallel_hist"]
        assert m["exceptions_serial"] == 1 and m["exceptions_parallel"] == 1
        # The "?" never entered the TF-IDF corpus: all fields non-empty for
        # the counted similarity inputs (empty_reports stays 0 on OK runs).
        assert m["empty_reports_serial"] == 0
        assert m["empty_reports_parallel"] == 0
        # The failed run's None segment no longer forces speedup N/A.
        assert m["speedup_na"] is False
        assert m["speedup"] is not None and m["speedup"] > 1

    def test_all_ok_unchanged_counts(self):
        serial = [_ok_run("serial") for _ in range(3)]
        parallel = [_ok_run("parallel") for _ in range(3)]
        m = ab._compute_metrics(serial, parallel)
        assert m["dropped_serial"] == 0 and m["dropped_parallel"] == 0
        assert m["exceptions_serial"] == 0 and m["exceptions_parallel"] == 0


class TestGateNewCriteria:
    def _base_metrics(self) -> dict:
        serial = [_ok_run("serial") for _ in range(3)]
        parallel = [_ok_run("parallel") for _ in range(3)]
        return ab._compute_metrics(serial, parallel)

    def test_all_failed_leg_fails_gate(self):
        m = self._base_metrics()
        m["n_serial"] = 3
        m["dropped_serial"] = 3
        ok, reasons = ab.summarize_gate(m)
        assert ok is False
        assert any("no usable samples" in r for r in reasons)

    def test_parallel_leg_all_failed_also_fails(self):
        m = self._base_metrics()
        m["dropped_parallel"] = m["n_parallel"]
        ok, reasons = ab.summarize_gate(m)
        assert ok is False
        assert any("parallel leg" in r and "no usable samples" in r for r in reasons)

    def test_single_analyst_prediction_collapse_fails(self):
        m = self._base_metrics()
        m["prediction_attribution_measured"] = True
        m["prediction_attribution_ok"] = False
        ok, reasons = ab.summarize_gate(m)
        assert ok is False
        assert any("single analyst" in r for r in reasons)

    def test_unmeasured_attribution_is_na(self):
        m = self._base_metrics()
        m["prediction_attribution_measured"] = False
        m["prediction_attribution_ok"] = False
        ok, reasons = ab.summarize_gate(m)
        assert ok is True
        assert not any("single analyst" in r for r in reasons)

    def test_multi_analyst_attribution_passes(self):
        m = self._base_metrics()
        m["prediction_attribution_measured"] = True
        m["prediction_attribution_ok"] = True
        ok, reasons = ab.summarize_gate(m)
        assert ok is True


class TestCaptureRunPredictionAnalysts:
    def test_prediction_analysts_collected(self, monkeypatch):
        import yialpha.ledger.predictions as preds

        class _FakeGraph:
            def propagate(self, ticker, date):
                return {"run_id": "r-1", "final_trade_decision": "d"}, "Buy"

        monkeypatch.setattr(
            preds, "predictions_for_run",
            lambda run_id: [
                SimpleNamespace(analyst="market"),
                SimpleNamespace(analyst="news"),
                SimpleNamespace(analyst="market"),
            ],
        )
        run = ab._capture_run(_FakeGraph(), "TST", "2026-01-10")
        assert run["prediction_analysts"] == ["market", "news"]
        assert run["exception"] is None

    def test_no_run_id_leaves_list_empty(self, monkeypatch):
        class _FakeGraph:
            def propagate(self, ticker, date):
                return {"final_trade_decision": "d"}, "Buy"

        run = ab._capture_run(_FakeGraph(), "TST", "2026-01-10")
        assert run["prediction_analysts"] == []
