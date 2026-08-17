"""Tests for the ic-cycle batch (T1): package-side IC dataset export + cycle,
the ic-cycle CLI wiring, snapshot evidence hygiene, and the
indicator_ic_context advisory block.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest import mock

import pandas as pd
import pytest

from yiagents.dataflows import stockstats_utils


def _ohlcv(days: int = 60, start_close: float = 100.0) -> pd.DataFrame:
    dates = pd.date_range("2026-01-01", periods=days, freq="D")
    closes = start_close + pd.Series(range(days)) * 0.5
    return pd.DataFrame(
        {
            "Date": dates,
            "Open": closes - 0.2,
            "High": closes + 0.5,
            "Low": closes - 0.5,
            "Close": closes,
            "Volume": [1_000_000] * days,
        }
    )


@pytest.fixture()
def patched_ohlcv(monkeypatch):
    monkeypatch.setattr(stockstats_utils, "load_ohlcv", lambda t, d: _ohlcv())


# --------------------------------------------------------------------------- #
# Package-side cycle
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_export_ic_datasets_writes_csvs(tmp_path, patched_ohlcv):
    from yiagents.backtest.ic_dataset import export_ic_datasets

    written = export_ic_datasets(
        ["NVDA", "AMD"], horizon=5, output_dir=tmp_path,
        indicators=["rsi", "close_50_sma"],
    )
    assert set(written) == {"NVDA", "AMD"}
    df = pd.read_csv(written["NVDA"])
    assert list(df.columns)[:3] == ["date", "forward_return", "close_50_sma"]
    assert df["forward_return"].notna().all()  # tail dropped


@pytest.mark.unit
def test_export_rejects_unknown_indicator(tmp_path):
    from yiagents.backtest.ic_dataset import export_ic_datasets

    with pytest.raises(ValueError, match="unknown indicator"):
        export_ic_datasets(
            ["NVDA"], output_dir=tmp_path, indicators=["not_real"],
        )


@pytest.mark.unit
def test_prune_verdict_for_csv_shape(tmp_path):
    from yiagents.backtest.ic_dataset import prune_verdict_for_csv

    # A perfectly monotone indicator (close itself) predicts the forward
    # return of a trending series at rank level; rsi on a straight-line trend
    # is constant → zero variance → no IC windows. Both must be handled.
    n = 80
    closes = pd.Series(100.0 + 0.5 * pd.Series(range(n)))
    df = pd.DataFrame({
        "date": pd.date_range("2026-01-01", periods=n, freq="D"),
        "forward_return": closes.shift(-5) / closes - 1.0,
        "close_50_sma": closes,        # computable, monotone
        "flat_junk": pd.Series(1.0, index=closes.index),  # zero variance
    }).dropna()
    csv = tmp_path / "X_5d.csv"
    df.to_csv(csv, index=False)

    verdict = prune_verdict_for_csv(csv, window=30, min_observations=10)
    assert set(verdict) == {"generated_at", "params", "keep", "prune", "per_indicator"}
    assert "close_50_sma" in verdict["per_indicator"]
    entry = verdict["per_indicator"]["close_50_sma"]
    assert entry["verdict"] in ("keep", "prune")
    assert entry["finite_windows"] >= 0
    # Same shape the prune CLI's --json-out writes (one format, two producers).
    assert isinstance(entry["mean_abs_ic"], float)


@pytest.mark.unit
def test_run_ic_cycle_end_to_end(tmp_path, patched_ohlcv):
    from yiagents.backtest.ic_dataset import run_ic_cycle

    result = run_ic_cycle(
        ["NVDA"], horizon=5, output_dir=tmp_path, indicators=["rsi"],
    )
    csv_path = result["csv"]["NVDA"]
    assert csv_path.exists()
    assert Path(f"{csv_path}.prune.json").exists()
    verdict = json.loads(Path(f"{csv_path}.prune.json").read_text(encoding="utf-8"))
    assert verdict == result["verdict"]["NVDA"]

    # All tickers failing → typed failure, never an empty "success".
    with mock.patch.object(
        stockstats_utils, "load_ohlcv", side_effect=RuntimeError("net down")
    ), pytest.raises(RuntimeError, match="no dataset exported"):
        run_ic_cycle(["BAD"], horizon=5, output_dir=tmp_path / "x")


# --------------------------------------------------------------------------- #
# CLI wiring
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_ic_cycle_command_registered():
    from yiagents.cli.main import app

    names = [
        getattr(c, "name", None) or (c.callback.__name__ if c.callback else "")
        for c in app.registered_commands
    ]
    assert "ic-cycle" in names


# --------------------------------------------------------------------------- #
# snapshot record evidence hygiene
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_snapshot_record_warns_on_missing_pathlike_evidence(monkeypatch, capsys):
    from yiagents.cli import main as cli_main

    def fake_record(config, reason, evidence):
        return Path("unused.json")

    monkeypatch.setattr(
        "yiagents.config_snapshot.record_config_snapshot", fake_record
    )
    cli_main.snapshot_record(
        reason="test", evidence="does/not/exist.prune.json"
    )
    out = " ".join(capsys.readouterr().out.split())  # rich wraps at 80 cols
    assert "does not exist" in out

    # Free-text descriptions and real paths stay silent.
    (Path("real.prune.json")).write_text("{}", encoding="utf-8")
    try:
        cli_main.snapshot_record(reason="test", evidence="real.prune.json")
        cli_main.snapshot_record(reason="test", evidence="IC report from last week")
        out2 = " ".join(capsys.readouterr().out.split())
        assert "does not exist" not in out2
    finally:
        Path("real.prune.json").unlink(missing_ok=True)


# --------------------------------------------------------------------------- #
# indicator_ic_context (default off; on → advisory line from *.prune.json)
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_indicator_ic_context_render(tmp_path, monkeypatch):
    from yiagents.agents.analysts.market_analyst import _format_indicator_ic_context

    monkeypatch.chdir(tmp_path)
    assert _format_indicator_ic_context() is None  # no ic_data dir

    ic_dir = tmp_path / "ic_data"
    ic_dir.mkdir()
    (ic_dir / "NVDA_5d.prune.json").write_text(json.dumps({
        "per_indicator": {
            "rsi": {"mean_abs_ic": 0.06, "verdict": "keep"},
            "macd": {"mean_abs_ic": 0.01, "verdict": "prune"},
        }
    }), encoding="utf-8")
    (ic_dir / "AMD_5d.prune.json").write_text(json.dumps({
        "per_indicator": {"rsi": {"mean_abs_ic": 0.08, "verdict": "keep"}}
    }), encoding="utf-8")
    (ic_dir / "broken.prune.json").write_text("{not json", encoding="utf-8")

    line = _format_indicator_ic_context()
    assert line is not None
    assert "rsi |IC|=0.070" in line  # averaged across the two verdicts
    assert "macd" in line
    assert "broken" not in line  # unreadable verdict skipped


@pytest.mark.unit
def test_indicator_ic_context_config_key_default_off():
    from yiagents.default_config import _ENV_OVERRIDES, DEFAULT_CONFIG

    assert DEFAULT_CONFIG["indicator_ic_context"] is False
    assert _ENV_OVERRIDES["YIAGENTS_INDICATOR_IC_CONTEXT"] == "indicator_ic_context"
