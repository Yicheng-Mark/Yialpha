"""Tests for scripts/export_ic_dataset.py — the IC evidence-chain exporter.

The exporter turns the OHLCV cache + stockstats into the exact CSV contract
``prune_indicators_cli`` consumes. Tests pin: column shape, the tail drop
(no fabricated forward returns), skip-and-report for indicators stockstats
cannot compute, and the fail-closed rejection of unknown indicator names.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "export_ic_dataset.py"
_spec = importlib.util.spec_from_file_location("export_ic_under_test", _SCRIPT)
assert _spec is not None and _spec.loader is not None
exporter = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(exporter)

# The implementation moved into the package (2026-08-16) so `yialpha
# ic-cycle` can share it; the script is a thin CLI wrapper. build_ic_frame's
# lazy imports resolve against the real dataflow modules, so tests stub
# THOSE (not the script namespace, which no longer carries them).
from yialpha.backtest.ic_dataset import build_ic_frame  # noqa: E402
from yialpha.dataflows import feature_registry, stockstats_utils  # noqa: E402


def _ohlcv(days: int = 40, start_close: float = 100.0) -> pd.DataFrame:
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
    """Serve a deterministic OHLCV frame; capture the requested ticker."""
    served = {"ticker": None, "as_of": None}

    def fake_load(ticker, curr_date):
        served["ticker"] = ticker
        served["as_of"] = curr_date
        return _ohlcv()

    monkeypatch.setattr(stockstats_utils, "load_ohlcv", fake_load)
    return served


@pytest.mark.unit
def test_column_shape_and_tail_drop(patched_ohlcv):
    frame, skipped = build_ic_frame(
        "NVDA", horizon=5, indicators=["close_50_sma", "rsi"], as_of="2026-06-10"
    )
    # 40 rows of data, last 5 cannot have a realizable 5-row forward return.
    assert len(frame) == 35
    assert list(frame.columns) == ["date", "forward_return", "close_50_sma", "rsi"]
    assert frame["forward_return"].notna().all()
    assert patched_ohlcv["ticker"] == "NVDA"
    assert patched_ohlcv["as_of"] == "2026-06-10"
    # skip list empty: both names are computable on this frame (close_50_sma
    # yields NaN values with only 40 rows but the COLUMN exists).
    assert skipped == []


@pytest.mark.unit
def test_forward_return_math(patched_ohlcv):
    frame, _ = build_ic_frame(
        "X", horizon=1, indicators=["rsi"], as_of="2026-06-10"
    )
    # Row i (except the dropped tail) must equal close[i+1]/close[i] - 1.
    closes = pd.Series([100.0 + 0.5 * i for i in range(40)])
    expected = closes.shift(-1) / closes - 1.0
    pd.testing.assert_series_equal(
        frame["forward_return"].reset_index(drop=True),
        expected.dropna().reset_index(drop=True),
        check_names=False,
    )


@pytest.mark.unit
def test_uncomputable_indicator_is_skipped_not_zerofilled(patched_ohlcv, caplog):
    import unittest.mock as mock

    class _BrokenWrap:
        def __getitem__(self, item):
            if item == "macd":
                raise RuntimeError("stockstats exploded")
            return pd.Series([1.0] * 40)

    with mock.patch("stockstats.wrap", lambda df: _BrokenWrap()), caplog.at_level("WARNING"):
        frame, skipped = build_ic_frame(
            "X", horizon=2, indicators=["macd", "rsi"], as_of="2026-06-10"
        )
    assert skipped == ["macd"]
    assert "macd" not in frame.columns
    assert "rsi" in frame.columns
    assert any("macd" in r.message for r in caplog.records)


@pytest.mark.unit
def test_cli_rejects_unknown_indicator(tmp_path, capsys):
    with pytest.raises(SystemExit) as excinfo:
        exporter.main(
            ["NVDA", "--indicators", "not_a_real_indicator",
             "--output-dir", str(tmp_path)]
        )
    assert excinfo.value.code == 2  # argparse error exit
    assert "unknown indicator" in capsys.readouterr().err


@pytest.mark.unit
def test_cli_writes_csv_per_ticker(tmp_path, patched_ohlcv):
    rc = exporter.main(
        ["NVDA", "AMD", "--horizon", "5", "--output-dir", str(tmp_path)]
    )
    assert rc == 0
    out1 = tmp_path / "NVDA_5d.csv"
    out2 = tmp_path / "AMD_5d.csv"
    assert out1.exists() and out2.exists()
    df = pd.read_csv(out1)
    assert list(df.columns)[:2] == ["date", "forward_return"]
    # Full default battery (24 indicators after the 2026-08-15 expansion).
    assert len(df.columns) == 2 + len(exporter.INDICATOR_NAMES)


@pytest.mark.unit
def test_cli_extra_horizons_add_decay_columns(tmp_path, patched_ohlcv):
    rc = exporter.main(
        ["NVDA", "--horizon", "5", "--extra-horizons", "1,10,20",
         "--output-dir", str(tmp_path)]
    )
    assert rc == 0
    df = pd.read_csv(tmp_path / "NVDA_5d.csv")
    for col in ("fwd_ret_1d", "fwd_ret_10d", "fwd_ret_20d"):
        assert col in df.columns
    # 40 rows, primary horizon 5 -> 35 kept rows; the 20d column keeps its
    # own honest NaN tail (rows realizable at 5d but not at 20d stay, with
    # NaN cells — never fabricated).
    assert len(df) == 35
    assert df["fwd_ret_1d"].notna().all()      # 1d realizable wherever 5d is
    assert int(df["fwd_ret_20d"].notna().sum()) == 20  # 40 - 20 tail rows


@pytest.mark.unit
def test_build_ic_frame_extra_horizon_math(patched_ohlcv):
    frame, _ = build_ic_frame(
        "X", horizon=1, indicators=["rsi"], as_of="2026-06-10",
        extra_horizons=[3],
    )
    closes = pd.Series([100.0 + 0.5 * i for i in range(40)])
    expected_3d = closes.shift(-3) / closes - 1.0
    actual = frame["fwd_ret_3d"].reset_index(drop=True)
    pd.testing.assert_series_equal(
        actual, expected_3d.iloc[:39].reset_index(drop=True), check_names=False
    )


@pytest.mark.unit
def test_cli_all_indicators_skipped_is_a_failure(tmp_path, monkeypatch, capsys):
    """When every indicator is skipped the CSV has no indicator columns —
    prune_indicators_cli rejects it. The exporter must count it as a failure
    (exit 1, no CSV, no misleading 'next:' hint) instead of a success."""
    import unittest.mock as mock

    class _AllBrokenWrap:
        def __getitem__(self, item):
            raise RuntimeError("stockstats exploded")

    def _all_derived_broken(data, name):
        raise RuntimeError("derived feature exploded")

    monkeypatch.setattr(stockstats_utils, "load_ohlcv", lambda t, d: _ohlcv())
    # Break BOTH compute paths: stockstats wrap (classic indicators) and the
    # derived-feature registry (rvol_20/ewma_vol/obv/rel_vol_20 don't use
    # wrap, so a broken wrap alone no longer skips the whole battery).
    monkeypatch.setattr(feature_registry, "compute_derived", _all_derived_broken)
    with mock.patch("stockstats.wrap", lambda df: _AllBrokenWrap()):
        rc = exporter.main(["NVDA", "--output-dir", str(tmp_path)])
    assert rc == 1
    assert not (tmp_path / "NVDA_5d.csv").exists()
    out = capsys.readouterr().out
    assert "next:" not in out
