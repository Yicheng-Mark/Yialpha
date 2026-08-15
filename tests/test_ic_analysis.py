"""IC diagnostics: decay, quantile spread, turnover (2026-08-15 expansion).

Beyond level+persistence, the IC evidence chain now answers: how fast does
predictive power fade with the forward horizon (ic_decay), is the factor
monotone across quantiles or only tail-effective (quantile_spread), and how
churny is the signal to capture (factor_turnover). Pure math on synthetic
data — plus CLI integration over a CSV with extra-horizon columns.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from yiagents.backtest.ic import (
    factor_turnover,
    ic_decay,
    information_coefficient,
    quantile_spread,
)

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "prune_indicators_cli.py"
_spec = importlib.util.spec_from_file_location("prune_cli_under_test", _SCRIPT)
assert _spec is not None and _spec.loader is not None
prune_cli = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(prune_cli)


@pytest.mark.unit
class TestIcDecay:
    def test_predictive_power_fading_shows_in_curve(self):
        rng = np.random.default_rng(11)
        n = 400
        factor = rng.normal(0.0, 1.0, n)
        fwd_1d = factor + rng.normal(0.0, 0.3, n)   # strong signal
        fwd_10d = 0.2 * factor + rng.normal(0.0, 1.0, n)  # weak signal
        curve = ic_decay(factor, {"1d": fwd_1d, "10d": fwd_10d})
        assert curve["1d"] is not None and curve["10d"] is not None
        assert curve["1d"] > 0.8
        assert curve["10d"] < curve["1d"]

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="equal length"):
            ic_decay([1.0, 2.0, 3.0], {"1d": [1.0, 2.0]})

    def test_matches_information_coefficient(self):
        rng = np.random.default_rng(3)
        factor = rng.normal(size=200)
        fwd = 0.5 * factor + rng.normal(size=200)
        curve = ic_decay(factor, {"5d": fwd})
        assert curve["5d"] == pytest.approx(
            information_coefficient(factor, fwd), abs=1e-12
        )


@pytest.mark.unit
class TestQuantileSpread:
    def test_monotone_factor_is_flagged_with_positive_spread(self):
        n = 500
        factor = np.arange(n, dtype=float)
        fwd = factor / 10_000.0  # perfectly monotone payoff
        qs = quantile_spread(factor, fwd)
        assert qs is not None
        assert qs["monotonic"] is True
        assert qs["spread"] > 0
        means = qs["quantile_means"]
        assert len(means) == 5
        assert all(b > a for a, b in zip(means, means[1:], strict=False))

    def test_noise_factor_is_not_monotone(self):
        rng = np.random.default_rng(5)
        factor = rng.normal(size=1000)
        fwd = rng.normal(size=1000)  # independent of the factor
        qs = quantile_spread(factor, fwd)
        assert qs is not None
        assert qs["monotonic"] is False
        # No systematic Q5−Q1 edge: each quantile mean averages 200 draws
        # (std ≈ 0.07), so a difference up to ~0.2 is pure noise.
        assert abs(qs["spread"]) < 0.25

    def test_inverted_factor_is_monotone_with_negative_spread(self):
        n = 500
        factor = np.arange(n, dtype=float)
        fwd = -factor / 10_000.0
        qs = quantile_spread(factor, fwd)
        assert qs["monotonic"] is True
        assert qs["spread"] < 0

    def test_too_few_rows_returns_none(self):
        assert quantile_spread([1.0, 2.0], [0.01, 0.02]) is None

    def test_constant_factor_returns_none(self):
        qs = quantile_spread([5.0] * 100, np.arange(100, dtype=float) / 100)
        assert qs is None  # zero variance collapses every bucket

    def test_bad_args_raise(self):
        with pytest.raises(ValueError, match="equal length"):
            quantile_spread([1.0, 2.0], [0.1])
        with pytest.raises(ValueError, match="n_quantiles"):
            quantile_spread([1.0, 2.0], [0.1, 0.2], n_quantiles=1)


@pytest.mark.unit
class TestFactorTurnover:
    def test_monotone_factor_has_near_zero_turnover(self):
        factor = np.arange(300, dtype=float)  # ranks never reshuffle
        assert factor_turnover(factor) == pytest.approx(1.0 / 299.0)

    def test_reshuffling_factor_has_high_turnover(self):
        rng = np.random.default_rng(9)
        perm = rng.permutation(300).astype(float)
        assert factor_turnover(perm) > 0.2

    def test_monotone_beats_reshuffled(self):
        rng = np.random.default_rng(9)
        assert factor_turnover(rng.permutation(300).astype(float)) > (
            10 * factor_turnover(np.arange(300, dtype=float))
        )

    def test_constant_factor_returns_none(self):
        assert factor_turnover([3.0] * 50) is None

    def test_nan_rows_are_dropped(self):
        factor = [1.0, np.nan, 3.0, 2.0]
        assert factor_turnover(factor) is not None


@pytest.mark.unit
class TestPruneCliDiagnostics:
    def _write_csv(self, path: Path) -> None:
        rng = np.random.default_rng(21)
        n = 200
        factor = rng.normal(size=n)
        rows = {
            "date": pd.bdate_range("2025-01-01", periods=n).strftime("%Y-%m-%d"),
            "forward_return": 0.3 * factor + rng.normal(0.0, 0.5, n),
            "fwd_ret_1d": 0.8 * factor + rng.normal(0.0, 0.3, n),
            "fwd_ret_10d": 0.1 * factor + rng.normal(0.0, 1.0, n),
            "good_factor": factor,
            "noise_factor": rng.normal(size=n),
        }
        pd.DataFrame(rows).to_csv(path, index=False)

    def test_report_contains_decay_and_spread_sections(self, tmp_path, capsys):
        csv = tmp_path / "ic.csv"
        json_out = tmp_path / "ic.prune.json"
        self._write_csv(csv)
        rc = prune_cli.main([
            str(csv), "--window", "30", "--min-consecutive", "10",
            "--json-out", str(json_out),
        ])
        assert rc == 0
        out = capsys.readouterr().out
        assert "IC decay" in out
        assert "1d" in out and "10d" in out
        assert "Quantile spread & turnover" in out
        assert "Monotonic" in out and "Turnover" in out

        verdict = json.loads(json_out.read_text(encoding="utf-8"))
        gf = verdict["per_indicator"]["good_factor"]
        assert set(gf["ic_by_horizon"]) == {"1d", "10d"}
        assert gf["ic_by_horizon"]["1d"] > gf["ic_by_horizon"]["10d"]
        assert "quantile_spread" in gf and "turnover" in gf
        assert verdict["per_indicator"]["noise_factor"]["ic_by_horizon"]["1d"] is not None

    def test_single_horizon_csv_keeps_original_report_shape(self, tmp_path, capsys):
        rng = np.random.default_rng(4)
        n = 150
        pd.DataFrame({
            "date": pd.bdate_range("2025-01-01", periods=n).strftime("%Y-%m-%d"),
            "forward_return": rng.normal(0.0, 0.01, n),
            "rsi": rng.uniform(20, 80, n),
        }).to_csv(tmp_path / "plain.csv", index=False)
        rc = prune_cli.main([str(tmp_path / "plain.csv"), "--window", "30"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "IC decay" not in out  # hidden without fwd_ret_*d columns
        assert "IC Indicator Pruning Report" in out
