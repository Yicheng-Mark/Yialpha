"""Scoreboard tests (yialpha/ledger/scoreboard) + the ``scoreboard`` CLI.

Hand-seeded predictions + outcomes via the ledger APIs; every metric is
computed by hand in the comments next to its assertion.

Base dataset (all ``complete`` outcomes, analyst/direction/prob_up/net):

==== ============ ======== ======= ======== ========== ======== =========
run  analyst      direction prob_up net      evidence   class    horizon
==== ============ ======== ======= ======== ========== ======== =========
r1   fundamentals up       0.8     +0.05    0          pure     5
r2   fundamentals up       0.6     -0.02    2          pure     5
r3   fundamentals up       0.9     +0.01    3          pure     5
r4   macro        flat     0.5     +0.03    0          stock    1
r5   sentiment    down     None    -0.04    5          linear   21
==== ============ ======== ======= ======== ========== ======== =========

Directional rows = r1, r2, r3, r5 (flat excluded). Hits: r1 (up, net>0),
r3, r5 (down, net<0) -> 3; r2 misses -> overall accuracy 3/4. The
fundamentals cell (r1-r3): accuracy 2/3, and over probs {0.8, 0.6, 0.9}:

* brier = (0.2^2 + 0.6^2 + 0.1^2) / 3 = (0.04 + 0.36 + 0.01) / 3 = 0.136667
* log_loss = (ln(1/0.8) + ln(1/0.4) + ln(1/0.9)) / 3
            = (0.223144 + 0.916291 + 0.105361) / 3 = 0.414932
* ECE: each prob lands alone in its 0.1-wide bin, so
  (|1-0.8| + |0-0.6| + |1-0.9|) / 3 = 0.9 / 3 = 0.3
"""

from __future__ import annotations

import json
import math

import pytest
from typer.testing import CliRunner

from yialpha.cli.main import app
from yialpha.ledger.evidence import register_run
from yialpha.ledger.models import SCOPE_CONTRACT
from yialpha.ledger.outcomes import write_outcome
from yialpha.ledger.predictions import submit_predictions
from yialpha.ledger.scoreboard import (
    V3_MIN_SAMPLES_PER_CELL,
    build_scoreboard,
    render_scoreboard_markdown,
)

runner = CliRunner()


def _seed_row(
    *,
    run_id: str,
    analyst: str,
    direction: str,
    prob_up: float | None,
    net_return: float,
    evidence_n: int = 0,
    horizon: int = 5,
    instrument_class: str = "pure_crypto_perp",
    complete: bool = True,
) -> None:
    """One run + one prediction + its outcome row (complete by default)."""
    register_run(run_id, "BTCUSDT", "crypto_perp", instrument_class, "2026-08-25")
    entry: dict[str, object] = {"horizon_days": horizon, "direction": direction}
    if prob_up is not None:
        entry["prob_up"] = prob_up
    evidence = [f"E{i:02d}" for i in range(evidence_n)] or None
    (prediction_id,) = submit_predictions(
        run_id, analyst, "BTCUSDT", SCOPE_CONTRACT, [entry], "2026-08-25",
        evidence_ids=evidence,
    )
    write_outcome(
        prediction_id,
        run_id,
        horizon,
        status="complete" if complete else "incomplete",
        net_return=net_return if complete else None,
        legs_missing=None if complete else ["funding"],
    )


@pytest.fixture()
def base_rows() -> None:
    """Seed r1-r5 exactly as documented in the module docstring table."""
    _seed_row(run_id="r1", analyst="fundamentals", direction="up",
              prob_up=0.8, net_return=0.05, evidence_n=0)
    _seed_row(run_id="r2", analyst="fundamentals", direction="up",
              prob_up=0.6, net_return=-0.02, evidence_n=2)
    _seed_row(run_id="r3", analyst="fundamentals", direction="up",
              prob_up=0.9, net_return=0.01, evidence_n=3)
    _seed_row(run_id="r4", analyst="macro", direction="flat",
              prob_up=0.5, net_return=0.03, evidence_n=0, horizon=1,
              instrument_class="stock_perp")
    _seed_row(run_id="r5", analyst="sentiment", direction="down",
              prob_up=None, net_return=-0.04, evidence_n=5, horizon=21,
              instrument_class="perp_linear")


# --------------------------------------------------------------------------- #
# Pinned metrics
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_overall_metrics_pinned(base_rows: None):
    board = build_scoreboard()
    overall = board["overall"]
    assert overall["n"] == 5
    assert overall["directional_n"] == 4  # r1, r2, r3, r5
    assert overall["hits"] == 3           # r1, r3, r5
    assert overall["directional_accuracy"] == pytest.approx(3 / 4)
    assert overall["flat_n"] == 1         # r4 counted but not scored
    assert overall["missing_prob_up"] == 1  # r5 excluded from prob metrics
    # prob-metric set is r1-r3 only (r4 flat, r5 no prob) — same as the
    # fundamentals cell below
    assert overall["brier_score"] == pytest.approx(0.41 / 3)
    expected_ll = (math.log(1 / 0.8) + math.log(1 / 0.4) + math.log(1 / 0.9)) / 3
    assert overall["log_loss"] == pytest.approx(expected_ll)  # ~0.414932
    assert overall["calibration_error"] == pytest.approx(0.3)
    assert overall["below_v3_min_samples"] is True


@pytest.mark.unit
def test_analyst_cell_pinned(base_rows: None):
    cell = build_scoreboard()["by_analyst"]["fundamentals"]
    assert cell["n"] == 3
    assert cell["directional_n"] == 3
    assert cell["hits"] == 2
    assert cell["directional_accuracy"] == pytest.approx(2 / 3)
    assert cell["brier_score"] == pytest.approx(0.41 / 3)  # 0.136667
    expected_ll = (math.log(1 / 0.8) + math.log(1 / 0.4) + math.log(1 / 0.9)) / 3
    assert cell["log_loss"] == pytest.approx(expected_ll)  # 0.414932
    assert cell["calibration_error"] == pytest.approx(0.3)


@pytest.mark.unit
def test_ece_binning_merges_nearby_probs():
    # 0.8 (hit) and 0.82 (miss) share the [0.8, 0.9) bin:
    # acc(bin) = 0.5, conf(bin) = (0.8+0.82)/2 = 0.81, bin weight 2/2 = 1
    # -> ECE = |0.5 - 0.81| = 0.31 (a merged bin beats two perfect bins)
    _seed_row(run_id="b1", analyst="fundamentals", direction="up",
              prob_up=0.8, net_return=0.02)
    _seed_row(run_id="b2", analyst="fundamentals", direction="up",
              prob_up=0.82, net_return=-0.01)
    cell = build_scoreboard()["by_analyst"]["fundamentals"]
    assert cell["n"] == 2
    assert cell["directional_accuracy"] == pytest.approx(1 / 2)
    assert cell["brier_score"] == pytest.approx((0.2**2 + 0.82**2) / 2)  # 0.3562
    expected_ll = (math.log(1 / 0.8) + math.log(1 / 0.18)) / 2  # 0.968971
    assert cell["log_loss"] == pytest.approx(expected_ll)
    assert cell["calibration_error"] == pytest.approx(abs(0.5 - 0.81))


# --------------------------------------------------------------------------- #
# Slices, filters, markdown
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_all_slices_present(base_rows: None):
    board = build_scoreboard()
    assert set(board["by_analyst"]) == {"fundamentals", "macro", "sentiment"}
    assert set(board["by_instrument_class"]) == {
        "perp_linear", "pure_crypto_perp", "stock_perp",
    }
    assert set(board["by_horizon_days"]) == {"1", "5", "21"}
    assert set(board["by_direction"]) == {"down", "flat", "up"}
    assert set(board["by_evidence_bucket"]) == {"0", "1-2", "3+"}
    # spot-check one slice of each kind
    assert board["by_instrument_class"]["pure_crypto_perp"]["n"] == 3
    assert board["by_horizon_days"]["5"]["hits"] == 2
    down = board["by_direction"]["down"]
    assert down["directional_accuracy"] == pytest.approx(1.0)  # r5 hit
    assert down["brier_score"] is None  # r5 has no prob_up
    flat = board["by_direction"]["flat"]
    assert flat["directional_n"] == 0 and flat["directional_accuracy"] is None
    assert board["by_evidence_bucket"]["1-2"]["n"] == 1  # r2 alone


@pytest.mark.unit
def test_min_samples_display_filters_cells(base_rows: None):
    board = build_scoreboard(min_samples_display=2)
    # only cells with n >= 2 survive; overall is always present
    assert set(board["by_analyst"]) == {"fundamentals"}
    assert set(board["by_direction"]) == {"up"}
    assert set(board["by_evidence_bucket"]) == {"0", "3+"}
    assert board["overall"]["n"] == 5


@pytest.mark.unit
def test_non_complete_outcomes_excluded(base_rows: None):
    _seed_row(run_id="r6", analyst="fundamentals", direction="up",
              prob_up=0.7, net_return=0.0, complete=False)
    board = build_scoreboard()
    assert board["overall"]["n"] == 5  # the incomplete row never scores
    assert board["by_analyst"]["fundamentals"]["n"] == 3


@pytest.mark.unit
def test_markdown_renders_tables_and_v3_note(base_rows: None):
    board = build_scoreboard()
    markdown = render_scoreboard_markdown(board)
    assert markdown.startswith("# Prediction calibration scoreboard")
    assert "## By analyst" in markdown
    assert "| analyst | n | dir n | hits | accuracy | brier | log loss | ECE | note |" in markdown
    assert "| fundamentals | 3 | 3 | 2 | 66.7% |" in markdown
    # every cell here is below the V3 threshold -> annotated display-only
    assert board["overall"]["n"] < V3_MIN_SAMPLES_PER_CELL
    assert "below V3 sample threshold — display only, no weight adjustment" in markdown


@pytest.mark.unit
def test_empty_ledger_renders_empty_sections():
    board = build_scoreboard()
    assert board["overall"]["n"] == 0
    assert board["overall"]["directional_accuracy"] is None
    for key in ("by_analyst", "by_direction"):
        assert board[key] == {}
    markdown = render_scoreboard_markdown(board)
    assert "## By analyst" in markdown
    assert "_(no rows)_" in markdown


# --------------------------------------------------------------------------- #
# CLI smoke (isolated tmp results dir; the per-test ledger is empty)
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_scoreboard_cli_smoke_writes_reports(tmp_path):
    result = runner.invoke(app, ["scoreboard", "--results-dir", str(tmp_path)])
    assert result.exit_code == 0, result.output
    out_dir = tmp_path / "scoreboard"
    assert (out_dir / "scoreboard.json").is_file()
    assert (out_dir / "scoreboard.md").is_file()
    board = json.loads((out_dir / "scoreboard.json").read_text(encoding="utf-8"))
    assert board["overall"]["n"] == 0
    markdown = (out_dir / "scoreboard.md").read_text(encoding="utf-8")
    assert markdown.startswith("# Prediction calibration scoreboard")
    assert "Outcomes: 0 considered" in result.output
    # no .tmp leftovers from the atomic write
    assert list(out_dir.glob("*.tmp")) == []
