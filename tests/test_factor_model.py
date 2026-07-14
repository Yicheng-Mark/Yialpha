"""Unit tests for ``yiagents.backtest.factor_model`` + engine wiring.

Hermetic: no network, no LLM. ``_parse_french_zip`` is exercised with a
synthetic zip fixture; ``load_factor_returns`` fail-open is checked by
monkeypatching the downloader; ``factor_attribution`` OLS recovery and the
engine's default-None (byte-equivalent) path round out coverage.
"""

from __future__ import annotations

import io
import zipfile

import numpy as np
import pandas as pd
import pytest

from yiagents.backtest.factor_model import (
    FactorAttribution,
    _parse_french_zip,
    factor_attribution,
    load_factor_returns,
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _synthetic_french_zip(model: str, rows: list[tuple]) -> bytes:
    """Build a zip containing one CSV that mimics a French daily-factor file.

    Each row is ``(yyyymmdd:int, *factor_values_in_percent)``. A header line
    (with a leading comma, like the real file) and a trailing copyright footer
    are included so the parser's skip/stop logic is exercised.
    """
    cols = {"3": ["Mkt-RF", "SMB", "HML", "RF"], "5": ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"]}[model]
    lines = ["," + ",".join([""] + cols)]  # leading-comma header, like the source
    for yyyymmdd, *vals in rows:
        lines.append(f"{yyyymmdd}," + ",".join(f"{v}" for v in vals))
    lines.append("Copyright 2026 Kenneth R. French")  # footer must be dropped
    text = "\n".join(lines) + "\n"
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("F-F_dummy.CSV", text)
    return buf.getvalue()


def _factor_frame(n: int = 60, seed: int = 1) -> pd.DataFrame:
    """A clean synthetic FF3 daily frame (decimals), indexed by business days."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2024-01-01", periods=n)
    return pd.DataFrame(
        {
            "Mkt-RF": rng.normal(0.0004, 0.01, n),
            "SMB": rng.normal(0.0001, 0.008, n),
            "HML": rng.normal(-0.0001, 0.007, n),
            "RF": np.full(n, 0.0001),  # constant small risk-free
        },
        index=idx,
    )


# --------------------------------------------------------------------------- #
# _parse_french_zip
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_parse_french_zip_percent_to_decimal_and_drops_footer():
    raw = _synthetic_french_zip("3", [
        (19260701, 0.10, 0.20, -0.05, 0.01),
        (19260702, 0.15, 0.10, 0.02, 0.02),
        (19260705, -0.30, -0.10, 0.00, 0.01),
    ])
    df = _parse_french_zip(raw, "3")
    assert list(df.columns) == ["Mkt-RF", "SMB", "HML", "RF"]
    assert len(df) == 3                       # footer dropped
    assert df.index[0] == pd.Timestamp("1926-07-01")
    # percent -> decimal
    assert df["Mkt-RF"].iloc[0] == pytest.approx(0.0010)
    assert df["RF"].iloc[0] == pytest.approx(0.0001)


@pytest.mark.unit
def test_parse_french_zip_handles_missing_sentinel():
    raw = _synthetic_french_zip("3", [
        (19260701, 0.10, 0.20, -0.05, 0.01),
        (19260702, -99.99, 0.10, 0.02, 0.02),   # French "missing" sentinel
        (19260705, 0.05, 0.01, 0.00, 0.01),
    ])
    df = _parse_french_zip(raw, "3")
    # The -99.99 row is dropped (NaN after replace), not kept as -0.9999.
    assert len(df) == 2
    assert (df["Mkt-RF"] > -1.0).all()


@pytest.mark.unit
def test_parse_french_zip_unknown_model_raises():
    with pytest.raises(ValueError):
        _parse_french_zip(b"x", "7")


# --------------------------------------------------------------------------- #
# factor_attribution
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_attribution_recovers_known_alphas_and_betas():
    """No-noise construction: OLS must recover the exact alpha and betas, with
    R² ~ 1.0. RF is nonzero so this also proves RF is subtracted first."""
    fr = _factor_frame(80)
    # excess = 0.001 (per-day alpha) + 1.2*Mkt-RF + 0.5*SMB + 0.3*HML
    excess = (
        0.001
        + 1.2 * fr["Mkt-RF"]
        + 0.5 * fr["SMB"]
        + 0.3 * fr["HML"]
    )
    strategy = (excess + fr["RF"]).rename("ret")  # strategy return incl. RF
    strategy.index = fr.index

    attr = factor_attribution(strategy, fr, periods_per_year=252, model="FF3")
    assert isinstance(attr, FactorAttribution)
    assert attr.betas["Mkt-RF"] == pytest.approx(1.2, abs=1e-6)
    assert attr.betas["SMB"] == pytest.approx(0.5, abs=1e-6)
    assert attr.betas["HML"] == pytest.approx(0.3, abs=1e-6)
    assert attr.alpha_annual == pytest.approx(0.001 * 252, abs=1e-6)  # annualized
    assert attr.r_squared == pytest.approx(1.0, abs=1e-6)
    assert attr.n_obs == 80


@pytest.mark.unit
def test_attribution_returns_none_when_too_few_overlapping_obs():
    fr = _factor_frame(80)
    # strategy covers only 3 dates -> below the _MIN_OBS threshold.
    strat = pd.Series([0.001, 0.002, 0.0], index=fr.index[:3])
    assert factor_attribution(strat, fr, model="FF3") is None


@pytest.mark.unit
def test_attribution_uses_only_overlapping_dates():
    fr = _factor_frame(80)                       # 80 business days from 2024-01-01
    overlap_idx = fr.index[40:80]                # strategy overlaps only the last 40
    excess = 0.0005 + 1.0 * fr["Mkt-RF"].loc[overlap_idx]
    strat_overlap = (excess + fr["RF"].loc[overlap_idx]).rename("ret")
    # Two extra dates NOT in the factor frame (after it) -- must be ignored.
    extra_idx = pd.bdate_range(fr.index[-1] + pd.Timedelta(days=1), periods=2)
    strat = pd.concat([strat_overlap, pd.Series([0.0, 0.0], index=extra_idx)])

    attr = factor_attribution(strat, fr, model="FF3")
    assert attr is not None
    assert attr.n_obs == 40                       # only the overlapping window
    assert attr.betas["Mkt-RF"] == pytest.approx(1.0, abs=1e-6)


# --------------------------------------------------------------------------- #
# load_factor_returns fail-open
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_load_factor_returns_unknown_model_returns_none(monkeypatch, tmp_path):
    out = load_factor_returns("2024-01-01", "2024-06-01", model="7", cache_dir=tmp_path)
    assert out is None


@pytest.mark.unit
def test_load_factor_returns_fail_open_on_download_failure(monkeypatch, tmp_path):
    """When the downloader can supply nothing (no cache, network down) the
    loader returns None rather than raising — a backtest must never die on it."""
    from yiagents.backtest import factor_model as fm

    def _boom(_path, _url, ttl_days=1.0):
        return None

    monkeypatch.setattr(fm, "_cached_or_download", _boom)
    out = load_factor_returns("2024-01-01", "2024-06-01", model="3", cache_dir=tmp_path)
    assert out is None


@pytest.mark.unit
def test_load_factor_returns_serves_from_cached_zip(monkeypatch, tmp_path):
    """A fresh cached zip on disk is parsed without any network call."""
    raw = _synthetic_french_zip("3", [
        (20240101, 0.05, 0.01, 0.00, 0.01),
        (20240102, 0.06, 0.02, 0.01, 0.01),
        (20240103, 0.04, 0.00, -0.01, 0.01),
    ])
    (tmp_path / "F-F_Research_Data_Factors_daily_CSV.zip").write_bytes(raw)

    # If the cache-first path works, requests.get is never reached.
    def _no_network(*_args, **_kwargs):
        raise AssertionError("network should not be hit when a fresh cache exists")

    monkeypatch.setattr("requests.get", _no_network)
    df = load_factor_returns("2024-01-01", "2024-12-31", model="3", cache_dir=tmp_path)
    assert df is not None
    assert len(df) == 3
    assert df["Mkt-RF"].iloc[0] == pytest.approx(0.0005)  # 0.05% -> decimal


# --------------------------------------------------------------------------- #
# Engine wiring: default-None is byte-equivalent
# --------------------------------------------------------------------------- #
class _FakeGraph:
    def __init__(self, ratings):
        self._ratings = dict(ratings)

    def propagate(self, company_name, trade_date, asset_type="stock"):
        r = self._ratings.get(trade_date, "Hold")
        return {"final_trade_decision": f"**Rating**: {r}\n"}, r

    def _resolve_benchmark(self, ticker):
        return "SPY"


def _rising_prices(ticker, start, end):
    idx = pd.bdate_range(start, end)
    values = [100.0 * (1 + 0.001 * i) for i in range(len(idx))]
    return pd.Series(values, index=idx.strftime("%Y-%m-%d"), dtype=float)


@pytest.mark.unit
def test_run_backtest_default_factor_model_leaves_fields_none():
    """factor_model defaults to None -> the four factor fields stay None and the
    run is byte-equivalent to the pre-feature behaviour."""
    from yiagents.backtest.engine import run_backtest

    idx = pd.bdate_range("2024-01-01", periods=60, freq="B")
    dates = [d.strftime("%Y-%m-%d") for d in idx[::6]][:6]
    res = run_backtest(
        _FakeGraph(dict.fromkeys(dates, "Buy")), "AAPL", dates,
        holding_days=5, price_provider=_rising_prices,
    )
    m = res.metrics
    assert m.factor_model is None
    assert m.factor_alpha is None
    assert m.factor_betas is None
    assert m.factor_r_squared is None


@pytest.mark.unit
def test_run_backtest_factor_model_optin_fail_open_offline(monkeypatch):
    """Opting in but with the factor fetch failing (monkeypatched to None) must
    leave the factor fields None and NOT raise — the backtest still succeeds."""
    from yiagents.backtest import factor_model as fm
    from yiagents.backtest.engine import run_backtest

    monkeypatch.setattr(fm, "load_factor_returns", lambda *a, **k: None)

    idx = pd.bdate_range("2024-01-01", periods=60, freq="B")
    dates = [d.strftime("%Y-%m-%d") for d in idx[::6]][:6]
    res = run_backtest(
        _FakeGraph(dict.fromkeys(dates, "Buy")), "AAPL", dates,
        holding_days=5, price_provider=_rising_prices, factor_model="3",
    )
    # Fail-open: no attribution, but the run completed with normal metrics.
    assert res.metrics.factor_model is None
    assert res.metrics.total_return is not None
