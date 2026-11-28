"""Tests for the ic-cycle batch (T1): package-side IC dataset export + cycle,
the ic-cycle CLI wiring, snapshot evidence hygiene, and the
indicator_ic_context advisory block.
"""

from __future__ import annotations

import json
import warnings
from contextlib import contextmanager
from pathlib import Path
from unittest import mock

import pandas as pd
import pytest

from yialpha.dataflows import stockstats_utils


@contextmanager
def _constant_input_warning():
    """Context for a constant Spearman input (the ``flat_junk`` column below):
    nothing may warn, with or without scipy. A constant input is refused before
    ``scipy.stats.spearmanr`` is reached (round-17 follow-up)."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        warnings.simplefilter("error", UserWarning)  # scipy's ConstantInputWarning
        yield


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
    from yialpha.backtest.ic_dataset import export_ic_datasets

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
    from yialpha.backtest.ic_dataset import export_ic_datasets

    with pytest.raises(ValueError, match="unknown indicator"):
        export_ic_datasets(
            ["NVDA"], output_dir=tmp_path, indicators=["not_real"],
        )


@pytest.mark.unit
def test_prune_verdict_for_csv_shape(tmp_path):
    from yialpha.backtest.ic_dataset import prune_verdict_for_csv

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

    with _constant_input_warning():
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
    from yialpha.backtest.ic_dataset import run_ic_cycle

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
# Venue awareness: crypto IC measured on the Binance candles it trades
# --------------------------------------------------------------------------- #


def _binance_frame(days: int = 320) -> pd.DataFrame:
    dates = pd.date_range("2025-01-01", periods=days, freq="D")
    closes = (100.0 + pd.Series(range(days)) * 0.5).values
    df = pd.DataFrame(
        {
            "Open": closes - 0.2, "High": closes + 0.5,
            "Low": closes - 0.5, "Close": closes,
            "Volume": [1_000.0] * days,
        },
        index=pd.DatetimeIndex(dates, name="Date"),
    )
    return df


@pytest.mark.unit
def test_export_crypto_perp_uses_binance_venue_and_suffix(tmp_path, monkeypatch):
    from yialpha.backtest.ic_dataset import export_ic_datasets

    calls: list[dict] = []

    def fake_frame(symbol, start, end, interval="1d", venue="binance_perp",
                   price_type="last", closed_as_of=None):
        calls.append({"symbol": symbol, "start": start, "end": end,
                      "venue": venue})
        return _binance_frame()

    monkeypatch.setattr(
        "yialpha.dataflows.binance.binance_klines_frame", fake_frame,
    )
    written = export_ic_datasets(
        ["BTCUSDT"], horizon=5, output_dir=tmp_path,
        indicators=["rsi"], asset_type="crypto_perp", as_of="2026-01-10",
    )
    assert set(written) == {"BTCUSDT"}
    assert calls and calls[0]["venue"] == "binance_perp"
    # Wide lookback so SMA-200 batteries are warm (1100 calendar days back).
    assert calls[0]["start"] == "2023-01-06"
    assert calls[0]["end"] == "2026-01-10"
    # Venue-tagged filename: a spot export of the same symbol cannot collide.
    assert written["BTCUSDT"].name == "BTCUSDT_5d_perp.csv"
    df = pd.read_csv(written["BTCUSDT"])
    assert "rsi" in df.columns


@pytest.mark.unit
def test_export_crypto_defaults_to_binance_battery(tmp_path, monkeypatch):
    from yialpha.agents.utils.binance_indicator_tools import (
        BINANCE_INDICATOR_DEFAULTS,
    )
    from yialpha.backtest.ic_dataset import export_ic_datasets

    monkeypatch.setattr(
        "yialpha.dataflows.binance.binance_klines_frame",
        lambda *a, **k: _binance_frame(),
    )
    written = export_ic_datasets(
        ["BTCUSDT"], horizon=5, output_dir=tmp_path,
        asset_type="crypto_spot",
    )
    df = pd.read_csv(written["BTCUSDT"])
    assert written["BTCUSDT"].name == "BTCUSDT_5d_spot.csv"
    assert set(BINANCE_INDICATOR_DEFAULTS) <= set(df.columns)


@pytest.mark.unit
def test_indicator_ic_context_filters_by_venue(tmp_path, monkeypatch):
    from yialpha.agents.analysts.market_analyst import (
        _format_indicator_ic_context,
    )

    ic_dir = tmp_path / "ic_data"
    ic_dir.mkdir()
    per = {"per_indicator": {"rsi": {"mean_abs_ic": 0.10}}}
    (ic_dir / "BTCUSDT_5d_perp.csv.prune.json").write_text(
        json.dumps({**per, "per_indicator": {"rsi": {"mean_abs_ic": 0.10}}}),
        encoding="utf-8",
    )
    (ic_dir / "NVDA_5d.csv.prune.json").write_text(
        json.dumps({**per, "per_indicator": {"rsi": {"mean_abs_ic": 0.01}}}),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    perp_line = _format_indicator_ic_context("crypto_perp")
    assert perp_line is not None and "rsi |IC|=0.100" in perp_line
    stock_line = _format_indicator_ic_context("stock")
    assert stock_line is not None and "rsi |IC|=0.010" in stock_line
    # Round-7: a venue with no tagged verdicts yet falls back to UNTAGGED
    # files only (legacy dirs) — the old every-file fallback blended the
    # perp verdict (0.055 average) into a spot run whose advisory line
    # claims venue purity.
    spot_line = _format_indicator_ic_context("crypto_spot")
    assert spot_line is not None and "rsi |IC|=0.010" in spot_line


# --------------------------------------------------------------------------- #
# CLI wiring
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_ic_cycle_command_registered():
    from yialpha.cli.main import app

    names = [
        getattr(c, "name", None) or (c.callback.__name__ if c.callback else "")
        for c in app.registered_commands
    ]
    assert "ic-cycle" in names


# --------------------------------------------------------------------------- #
# snapshot record evidence hygiene
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_snapshot_record_warns_on_missing_pathlike_evidence(monkeypatch, capsys, tmp_path):
    from yialpha.cli import main as cli_main

    def fake_record(config, reason, evidence):
        return Path("unused.json")

    monkeypatch.setattr(
        "yialpha.config_snapshot.record_config_snapshot", fake_record
    )
    cli_main.snapshot_record(
        reason="test", evidence="does/not/exist.prune.json"
    )
    out = " ".join(capsys.readouterr().out.split())  # rich wraps at 80 cols
    assert "does not exist" in out

    # Free-text descriptions and real paths stay silent. The real file lives in
    # the test's own directory: written to the working directory instead, two
    # suite runs in one checkout could delete each other's copy mid-test.
    monkeypatch.chdir(tmp_path)
    (tmp_path / "real.prune.json").write_text("{}", encoding="utf-8")
    cli_main.snapshot_record(reason="test", evidence="real.prune.json")
    cli_main.snapshot_record(reason="test", evidence="IC report from last week")
    out2 = " ".join(capsys.readouterr().out.split())
    assert "does not exist" not in out2


# --------------------------------------------------------------------------- #
# indicator_ic_context (default off; on → advisory line from *.prune.json)
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_indicator_ic_context_render(tmp_path, monkeypatch):
    from yialpha.agents.analysts.market_analyst import _format_indicator_ic_context

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
    from yialpha.default_config import _ENV_OVERRIDES, DEFAULT_CONFIG

    assert DEFAULT_CONFIG["indicator_ic_context"] is False
    assert _ENV_OVERRIDES["YIALPHA_INDICATOR_IC_CONTEXT"] == "indicator_ic_context"


# --------------------------------------------------------------------------- #
# round-11: fail-visible when the flag is on but nothing is loadable
# --------------------------------------------------------------------------- #
_MA_LOGGER = "yialpha.agents.analysts.market_analyst"


@pytest.mark.unit
def test_indicator_ic_context_warns_when_on_and_no_verdict(tmp_path, monkeypatch, caplog):
    """Flag on + nothing loadable -> one WARNING naming the path and remedy.

    The flag used to no-op silently (missing ic_data dir, empty dir, or
    nothing catalog-known inside it) — inconsistent with config-check's
    posture toward indicator_battery.
    """
    import logging

    from yialpha.agents.analysts.market_analyst import _format_indicator_ic_context
    from yialpha.dataflows.config import set_config

    monkeypatch.chdir(tmp_path)  # no ic_data at all
    set_config({"indicator_ic_context": True})
    try:
        with caplog.at_level(logging.WARNING, logger=_MA_LOGGER):
            assert _format_indicator_ic_context() is None
        assert "ic-cycle" in caplog.text
        assert "ic_data" in caplog.text
        caplog.clear()
        # Directory present but no loadable verdict inside: same warning.
        (tmp_path / "ic_data").mkdir()
        with caplog.at_level(logging.WARNING, logger=_MA_LOGGER):
            assert _format_indicator_ic_context() is None
        assert "ic-cycle" in caplog.text
    finally:
        set_config({"indicator_ic_context": False})


@pytest.mark.unit
def test_indicator_ic_context_no_warning_when_flag_off(tmp_path, monkeypatch, caplog):
    """Flag off: a direct lookup returns None with zero warnings."""
    import logging

    from yialpha.agents.analysts.market_analyst import _format_indicator_ic_context

    monkeypatch.chdir(tmp_path)
    # conftest/reset_config holds indicator_ic_context OFF here.
    with caplog.at_level(logging.WARNING, logger=_MA_LOGGER):
        assert _format_indicator_ic_context() is None
    assert caplog.text == ""


@pytest.mark.unit
def test_indicator_ic_context_node_gate_skips_lookup_when_flag_off(monkeypatch):
    """Flag off: the node-level gate never invokes the IC lookup at all.

    Pins the run-path half of the byte-equivalence: with the flag off the
    ic_data directory is not even read (the call-site gate predates round-11
    and is unchanged by it).
    """
    from langchain_core.messages import AIMessage, HumanMessage
    from langchain_core.runnables import Runnable

    import yialpha.agents.analysts.market_analyst as ma

    calls: list[tuple] = []
    monkeypatch.setattr(
        ma, "_format_indicator_ic_context",
        lambda *a, **k: calls.append(a) or "SHOULD NOT APPEAR",
    )

    class _CaptureLLM(Runnable):
        def invoke(self, inp, config=None, **kwargs):  # noqa: ARG002
            return AIMessage(content="R", tool_calls=[])

        def bind_tools(self, tools, **kwargs):  # noqa: ARG002
            return self

    ma.create_market_analyst(_CaptureLLM())(
        {
            "trade_date": "2026-08-01",
            "company_of_interest": "AAPL",
            "asset_type": "stock",
            "instrument_context": "CTX",
            "messages": [HumanMessage(content="analyze")],
        }
    )
    assert calls == []
