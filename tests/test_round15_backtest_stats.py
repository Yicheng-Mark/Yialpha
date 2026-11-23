"""Round-15 hardening: backtest statistics modules (round-15 coverage shard).

Targets the uncovered branches of ``yialpha.backtest.factor_model``,
``metrics``, ``event_study``, ``ic`` and ``cache`` -- parser skip branches,
the ``_cached_or_download`` cache/TTL/stale matrix, the scipy-optional
fallbacks (Acklam ppf, erf CDF, numpy skew/kurtosis, rank-Spearman), metric
edge branches (DSR degenerate denominator, clamp), event-study edge skips
and the markdown renderer, IC fallback seams, and the decision-cache
dirless/disabled/mismatched-entry guards.

Hermetic: no network (``requests.get`` monkeypatched at every fetch seam),
no LLM, no sleeps. Statistical values are hand-computed or pinned against
scipy references computed in-test.
"""

from __future__ import annotations

import io
import json
import math
import os
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from yialpha.backtest import factor_model as fm, ic as ic_mod, metrics as mm
from yialpha.backtest.cache import CachedDecision, DecisionCache
from yialpha.backtest.event_study import (
    EventCAR,
    EventStudyResult,
    _fmt,
    _index_at_or_after,
    _price_returns,
    abnormal_returns,
    bootstrap_ci,
    cross_sectional_ttest,
    event_study,
    fit_market_model,
    render_event_study,
)
from yialpha.backtest.factor_model import (
    _parse_french_zip,
    factor_attribution,
    label_for,
    load_factor_returns,
)
from yialpha.backtest.metrics import (
    _deflated_sharpe_ratio,
    _norm_cdf,
    _norm_ppf,
    _sample_excess_kurtosis,
    _sample_skew,
    benchmark_comparison,
    compute_metrics,
)

_FF3_NAME = "F-F_Research_Data_Factors_daily_CSV.zip"
_FF5_NAME = "F-F_Research_Data_5_Factors_2x3_daily_CSV.zip"


def _zip_bytes(csv_text: str, member: str = "F-F_dummy.CSV") -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(member, csv_text)
    return buf.getvalue()


def _ff3_csv(rows: list[str], header: bool = True, footer: bool = True) -> str:
    lines = [",  ,Mkt-RF,SMB,HML,RF"] if header else []
    lines.extend(rows)
    if footer:
        lines.append("Copyright 2026 Kenneth R. French")
    return "\n".join(lines) + "\n"


class _FakeResponse:
    def __init__(self, content: bytes):
        self.content = content

    def raise_for_status(self) -> None:
        return None


def _no_network(*_args, **_kwargs):
    raise AssertionError("network must not be hit in this test")


# =========================================================================== #
# factor_model: parser skip/stop branches
# =========================================================================== #
@pytest.mark.unit
def test_r15_bs_parse_french_zip_skips_blank_annual_and_nonnumeric_rows():
    """Blank lines, annual-summary rows (extra tokens), non-numeric tails and
    the footer are skipped; out-of-order rows are re-sorted ascending."""
    csv_text = _ff3_csv([
        "",                                    # blank line -> skipped
        "20240103, 0.04, 0.00, -0.01, 0.01",
        "20240101, 0.05, 0.01, 0.00, 0.01",   # out of order -> sorted below
        # Averaged annual row: 8-digit date but FIVE numeric tokens (the
        # annual factor averages) -> rejected by the column-count check.
        "20241231, 12.34, 15.12, 18.25, 9.67, 12.90",
        "20240102, abc, def, ghi, jkl",       # non-numeric tail -> ValueError skip
    ])
    df = _parse_french_zip(_zip_bytes(csv_text), "3")
    assert list(df.index) == [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-03")]
    # Percent -> decimal pinned.
    assert df["Mkt-RF"].iloc[0] == pytest.approx(0.0005)
    assert df["RF"].iloc[0] == pytest.approx(0.0001)


@pytest.mark.unit
def test_r15_bs_parse_french_zip_rejects_zip_without_csv_member():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("readme.txt", "no csv here")
    with pytest.raises(ValueError, match="no .csv member"):
        _parse_french_zip(buf.getvalue(), "3")


@pytest.mark.unit
def test_r15_bs_parse_french_zip_rejects_zip_without_data_rows():
    csv_text = _ff3_csv([])  # header + footer only
    with pytest.raises(ValueError, match="no data rows"):
        _parse_french_zip(_zip_bytes(csv_text), "3")


@pytest.mark.unit
def test_r15_bs_parse_french_zip_ff5_has_five_factor_columns():
    csv_text = "\n".join([
        ",Mkt-RF,SMB,HML,RMW,CMA,RF",
        "20240101, 0.10, 0.02, -0.03, 0.01, -0.02, 0.009",
        "20240102, -0.20, -0.05, 0.10, -0.01, 0.04, 0.009",
    ]) + "\n"
    df = _parse_french_zip(_zip_bytes(csv_text), "5")
    assert list(df.columns) == ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"]
    assert df["CMA"].iloc[0] == pytest.approx(-0.0002)
    assert df["RMW"].iloc[1] == pytest.approx(-0.0001)


@pytest.mark.unit
def test_r15_bs_attribution_missing_rf_column_returns_none():
    idx = pd.bdate_range("2024-01-01", periods=30)
    fr = pd.DataFrame({"Mkt-RF": np.zeros(30), "SMB": np.zeros(30)}, index=idx)
    strat = pd.Series(0.001, index=idx)
    assert factor_attribution(strat, fr) is None


@pytest.mark.unit
def test_r15_bs_attribution_rf_only_frame_returns_none():
    idx = pd.bdate_range("2024-01-01", periods=30)
    fr = pd.DataFrame({"RF": np.full(30, 0.0001)}, index=idx)
    strat = pd.Series(0.001, index=idx)
    assert factor_attribution(strat, fr) is None


@pytest.mark.unit
def test_r15_bs_label_for_known_and_unknown_models():
    assert label_for("3") == "FF3"
    assert label_for("5") == "FF5"
    assert label_for("bogus") == "FF3"
    assert label_for(5) == "FF5"


@pytest.mark.unit
def test_r15_bs_load_factor_returns_parse_failure_fails_open(monkeypatch, tmp_path):
    monkeypatch.setattr(fm, "_cached_or_download", lambda *a, **k: b"not-a-zip")
    out = load_factor_returns("2024-01-01", "2024-12-31", model="3", cache_dir=tmp_path)
    assert out is None


@pytest.mark.unit
def test_r15_bs_load_factor_returns_uses_default_cache_dir_and_downloads(
    monkeypatch, tmp_path
):
    """cache_dir=None resolves through the disk_cache vendor layer; the
    downloaded zip is parsed, cached and sliced to the window."""
    import yialpha.dataflows.disk_cache as dc

    monkeypatch.setattr(dc, "vendor_cache_dir", lambda name: str(tmp_path / name))
    csv_text = _ff3_csv([
        "20240101, 0.05, 0.01, 0.00, 0.01",
        "20240102, 0.06, 0.02, 0.01, 0.01",
        "20240103, 0.04, 0.00, -0.01, 0.01",
    ])
    raw = _zip_bytes(csv_text)
    monkeypatch.setattr("requests.get", lambda *a, **k: _FakeResponse(raw))

    df = load_factor_returns("2024-01-02", "2024-12-31", model="3")
    assert df is not None
    assert list(df.index) == [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")]
    assert df["Mkt-RF"].iloc[0] == pytest.approx(0.0006)
    # The download landed in the vendor-resolved cache dir.
    assert (tmp_path / "factors" / _FF3_NAME).read_bytes() == raw


# =========================================================================== #
# factor_model: _cached_or_download cache/TTL/stale matrix
# =========================================================================== #
@pytest.mark.unit
def test_r15_bs_cached_or_download_serves_fresh_cache_without_network(
    monkeypatch, tmp_path
):
    path = tmp_path / _FF3_NAME
    path.write_bytes(b"cached-zip")
    monkeypatch.setattr("requests.get", _no_network)
    assert fm._cached_or_download(path, "http://x/" + _FF3_NAME) == b"cached-zip"


@pytest.mark.unit
def test_r15_bs_cached_or_download_expired_ttl_refetches_and_rewrites(
    monkeypatch, tmp_path
):
    path = tmp_path / _FF3_NAME
    path.write_bytes(b"stale-by-ttl")
    two_days_ago = time.time() - 2 * 86_400.0
    os.utime(path, (two_days_ago, two_days_ago))
    monkeypatch.setattr("requests.get", lambda *a, **k: _FakeResponse(b"fresh-zip"))
    out = fm._cached_or_download(path, "http://x/" + _FF3_NAME, ttl_days=1.0)
    assert out == b"fresh-zip"
    assert path.read_bytes() == b"fresh-zip"


@pytest.mark.unit
def test_r15_bs_cached_or_download_read_failure_falls_through_to_download(
    monkeypatch, tmp_path
):
    """An unreadable cache entry (here: the path is a directory, so read_bytes
    raises OSError on Windows) must not shadow the network; the download is
    served, and even the failing cache rewrite stays best-effort."""
    path = tmp_path / _FF3_NAME
    path.mkdir()
    monkeypatch.setattr("requests.get", lambda *a, **k: _FakeResponse(b"net-zip"))
    out = fm._cached_or_download(path, "http://x/" + _FF3_NAME)
    assert out == b"net-zip"


@pytest.mark.unit
def test_r15_bs_cached_or_download_write_failure_still_returns_raw(
    monkeypatch, tmp_path
):
    path = tmp_path / _FF3_NAME

    def _boom(_self, _data):
        raise OSError("disk full")

    monkeypatch.setattr(Path, "write_bytes", _boom)
    monkeypatch.setattr("requests.get", lambda *a, **k: _FakeResponse(b"net-zip"))
    assert fm._cached_or_download(path, "http://x/" + _FF3_NAME) == b"net-zip"
    assert not path.exists()  # nothing was written


def _failing_download(monkeypatch):
    def _down(*_args, **_kwargs):
        raise RuntimeError("network down")

    monkeypatch.setattr("requests.get", _down)


@pytest.mark.unit
def test_r15_bs_cached_or_download_failure_without_cache_returns_none(
    monkeypatch, tmp_path
):
    _failing_download(monkeypatch)
    out = fm._cached_or_download(tmp_path / _FF3_NAME, "http://x/" + _FF3_NAME)
    assert out is None


@pytest.mark.unit
def test_r15_bs_cached_or_download_failure_serves_stale_within_cap(
    monkeypatch, tmp_path
):
    import yialpha.dataflows.disk_cache as dc

    monkeypatch.setattr(dc, "max_stale_days", lambda: 30.0)
    _failing_download(monkeypatch)
    path = tmp_path / _FF3_NAME
    path.write_bytes(b"five-day-old-zip")
    five_days_ago = time.time() - 5 * 86_400.0
    os.utime(path, (five_days_ago, five_days_ago))
    assert fm._cached_or_download(path, "http://x/" + _FF3_NAME) == b"five-day-old-zip"


@pytest.mark.unit
def test_r15_bs_cached_or_download_unreadable_stale_returns_none(
    monkeypatch, tmp_path
):
    """A stale-but-within-cap entry that cannot be read (path is a directory
    on Windows) serves nothing: neither the cache nor the network can supply
    the bytes, so the caller gets None (attribution is skipped)."""
    import yialpha.dataflows.disk_cache as dc

    monkeypatch.setattr(dc, "max_stale_days", lambda: 30.0)
    _failing_download(monkeypatch)
    path = tmp_path / _FF3_NAME
    path.mkdir()
    five_days_ago = time.time() - 5 * 86_400.0
    os.utime(path, (five_days_ago, five_days_ago))
    assert fm._cached_or_download(path, "http://x/" + _FF3_NAME) is None


@pytest.mark.unit
def test_r15_bs_cached_or_download_failure_refuses_overaged_stale(
    monkeypatch, tmp_path, caplog
):
    import yialpha.dataflows.disk_cache as dc

    monkeypatch.setattr(dc, "max_stale_days", lambda: 30.0)
    _failing_download(monkeypatch)
    path = tmp_path / _FF3_NAME
    path.write_bytes(b"forty-day-old-zip")
    forty_days_ago = time.time() - 40 * 86_400.0
    os.utime(path, (forty_days_ago, forty_days_ago))
    with caplog.at_level("WARNING", logger="yialpha.backtest.factor_model"):
        out = fm._cached_or_download(path, "http://x/" + _FF3_NAME)
    assert out is None
    assert any("refusing to serve stale" in r.message for r in caplog.records)


# =========================================================================== #
# metrics: scipy-optional fallbacks and DSR edge branches
# =========================================================================== #
@pytest.mark.unit
def test_r15_bs_norm_cdf_erf_fallback_matches_scipy(monkeypatch):
    scipy_stats = pytest.importorskip("scipy.stats")
    monkeypatch.setattr(mm, "_HAVE_SCIPY", False)
    for x in (-3.5, -1.0, 0.0, 1.96, 4.2):
        assert _norm_cdf(x) == pytest.approx(float(scipy_stats.norm.cdf(x)), abs=1e-12)
    assert _norm_cdf(0.0) == pytest.approx(0.5)


@pytest.mark.unit
def test_r15_bs_norm_ppf_acklam_fallback_matches_scipy(monkeypatch):
    scipy_stats = pytest.importorskip("scipy.stats")
    monkeypatch.setattr(mm, "_HAVE_SCIPY", False)
    # Cover all three Acklam branches: lower tail, central, upper tail.
    for p in (0.001, 0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99, 0.999):
        expected = float(scipy_stats.norm.ppf(p))
        assert _norm_ppf(p) == pytest.approx(expected, rel=1e-8, abs=1e-10)
    # Degenerate probabilities saturate rather than raise.
    assert _norm_ppf(0.0) == float("-inf")
    assert _norm_ppf(1.0) == float("inf")


@pytest.mark.unit
def test_r15_bs_sample_skew_numpy_fallback_matches_scipy(monkeypatch):
    scipy_stats = pytest.importorskip("scipy.stats")
    monkeypatch.setattr(mm, "_HAVE_SCIPY", False)
    rets = np.array([0.01, -0.004, 0.02, 0.003, -0.002, 0.012, -0.007, 0.005])
    assert _sample_skew(rets) == pytest.approx(
        float(scipy_stats.skew(rets, bias=True)), abs=1e-12
    )
    # Degenerate guards: a single observation and a constant series -> 0.0.
    assert _sample_skew(np.array([0.01])) == 0.0
    assert _sample_skew(np.full(6, 0.01)) == 0.0


@pytest.mark.unit
def test_r15_bs_sample_kurtosis_numpy_fallback_matches_scipy(monkeypatch):
    scipy_stats = pytest.importorskip("scipy.stats")
    monkeypatch.setattr(mm, "_HAVE_SCIPY", False)
    rets = np.array([0.01, -0.004, 0.02, 0.003, -0.002, 0.012, -0.007, 0.005])
    assert _sample_excess_kurtosis(rets) == pytest.approx(
        float(scipy_stats.kurtosis(rets, fisher=True, bias=True)), abs=1e-12
    )
    assert _sample_excess_kurtosis(np.array([0.01])) == 0.0
    assert _sample_excess_kurtosis(np.full(6, 0.01)) == 0.0


@pytest.mark.unit
def test_r15_bs_dsr_single_observation_returns_zero():
    assert _deflated_sharpe_ratio(
        np.array([0.01]), sharpe_annualized=1.0, n_trials=1,
        periods_per_year=252, skew_override=None, kurt_override=None,
    ) == 0.0


@pytest.mark.unit
def test_r15_bs_dsr_imaginary_denominator_returns_zero():
    # returns [2.0, 0.0]: mean 1.0, population std 1.0 -> sr_per = 1.0.
    # denom_sq = 1 - 5*1 + ((-2+2)*1)/4 = -4 <= 0 -> the statistic is not
    # real-valued, so DSR degrades to 0 rather than returning a complex NaN.
    got = _deflated_sharpe_ratio(
        np.array([2.0, 0.0]), sharpe_annualized=1.0, n_trials=1,
        periods_per_year=252, skew_override=5.0, kurt_override=-2.0,
    )
    assert got == 0.0


@pytest.mark.unit
def test_r15_bs_dsr_clamps_cdf_excursions_into_unit_interval(monkeypatch):
    rets = np.array([0.01, 0.02, 0.015, 0.005])
    monkeypatch.setattr(mm, "_norm_cdf", lambda _x: -1e-16)
    assert _deflated_sharpe_ratio(
        rets, 1.0, 1, 252, skew_override=None, kurt_override=None
    ) == 0.0
    monkeypatch.setattr(mm, "_norm_cdf", lambda _x: 1.0 + 1e-12)
    assert _deflated_sharpe_ratio(
        rets, 1.0, 1, 252, skew_override=None, kurt_override=None
    ) == 1.0


@pytest.mark.unit
def test_r15_bs_compute_metrics_mismatched_benchmark_length_leaves_alpha_none():
    # A benchmark of a different length cannot be differenced -> all
    # benchmark-relative fields stay None, not garbage.
    m = compute_metrics([100.0, 101.0, 102.0], benchmark_equity=[100.0, 101.0])
    assert m.alpha_vs_buyhold is None
    assert m.information_ratio is None
    assert m.tracking_error is None


@pytest.mark.unit
def test_r15_bs_benchmark_comparison_too_short_series_returns_nones():
    # Equal lengths but a single observation -> IR and TE are both undefined.
    assert benchmark_comparison([0.01], [0.008], 252) == (None, None)


# =========================================================================== #
# event_study: pure helpers and edge skips
# =========================================================================== #
@pytest.mark.unit
def test_r15_bs_price_returns_short_series_is_empty():
    out = _price_returns(pd.Series([100.0], index=["2024-01-02"]))
    assert isinstance(out, pd.Series)
    assert out.empty


@pytest.mark.unit
def test_r15_bs_price_returns_values_and_index_alignment():
    idx = ["2024-01-02", "2024-01-03", "2024-01-04"]
    out = _price_returns(pd.Series([100.0, 110.0, 99.0], index=idx))
    assert list(out.index) == idx[1:]  # return[t] belongs to date[t]
    assert out.iloc[0] == pytest.approx(0.1)
    assert out.iloc[1] == pytest.approx(-0.1)


@pytest.mark.unit
def test_r15_bs_fit_market_model_short_or_mismatched_returns_zeros():
    assert fit_market_model(np.array([0.01]), np.array([0.01])) == (0.0, 0.0)
    assert fit_market_model(np.array([0.01, 0.02]), np.array([0.01])) == (0.0, 0.0)


@pytest.mark.unit
def test_r15_bs_cross_sectional_ttest_single_car_returns_zero_and_none_p():
    assert cross_sectional_ttest(np.array([0.03])) == (0.0, None)


@pytest.mark.unit
def test_r15_bs_cross_sectional_ttest_zero_variance_stack():
    from yialpha.backtest import event_study as es

    cars = np.array([0.02, 0.02, 0.02])  # sd(ddof=1) == 0
    t_stat, p_value = cross_sectional_ttest(cars)
    assert t_stat == 0.0
    # With scipy the degenerate stack is reported as "cannot reject" (p=1);
    # without scipy the p-value is honestly None.
    if es._HAVE_SCIPY:
        assert p_value == 1.0
    else:
        assert p_value is None


@pytest.mark.unit
def test_r15_bs_cross_sectional_ttest_without_scipy_returns_stat_only(monkeypatch):
    from yialpha.backtest import event_study as es

    monkeypatch.setattr(es, "_HAVE_SCIPY", False)
    # Hand-computed Brown & Warner t: cars [0.01, 0.03] have mean 0.02 and
    # sd(ddof=1) = 0.01*sqrt(2), so se = sd/sqrt(2) = 0.01 and t = 2.0 exactly.
    t_stat, p_value = cross_sectional_ttest(np.array([0.01, 0.03]))
    assert t_stat == pytest.approx(2.0)
    assert p_value is None


@pytest.mark.unit
def test_r15_bs_bootstrap_ci_single_car_returns_none():
    assert bootstrap_ci(np.array([0.01]), n_bootstrap=50, rng_seed=1) is None


@pytest.mark.unit
def test_r15_bs_event_study_ratings_misaligned_raises():
    idx = pd.bdate_range("2024-01-02", periods=80).strftime("%Y-%m-%d")
    asset = pd.Series(np.linspace(100, 120, 80), index=idx)
    bench = pd.Series(np.linspace(100, 115, 80), index=idx)
    with pytest.raises(ValueError, match="1:1"):
        event_study(asset, bench, [idx[40]], ratings=["Buy", "Sell"])


@pytest.mark.unit
def test_r15_bs_event_study_no_common_days_returns_empty_result():
    asset = pd.Series([100.0, 101.0], index=["2024-01-02", "2024-01-03"])
    bench = pd.Series([100.0, 101.0], index=["2024-02-05", "2024-02-06"])
    result = event_study(asset, bench, ["2024-01-03"], benchmark_name="SPY")
    assert result.n_events == 0
    assert result.events == []
    assert result.mean_car is None
    assert result.t_stat is None
    assert result.ci_low is None and result.ci_high is None
    assert result.benchmark == "SPY"


def _es_series(n: int = 80) -> tuple[pd.Series, pd.Series]:
    idx = pd.bdate_range("2024-01-02", periods=n).strftime("%Y-%m-%d")
    rng = np.random.default_rng(3)
    rets = rng.normal(0.0004, 0.01, size=n)
    asset = pd.Series(100.0 * np.cumprod(1.0 + rets), index=idx)
    bench = pd.Series(100.0 * np.cumprod(1.0 + rets), index=idx)
    return asset, bench


@pytest.mark.unit
def test_r15_bs_event_study_skips_first_bar_and_beyond_end_dates():
    """An event on the very first bar (pos 0 -> no event-day return yet) and
    one past the tape's end are skipped; a mid-tape event survives."""
    asset, bench = _es_series()
    idx = asset.index
    result = event_study(
        asset, bench, [idx[0], idx[40], "2099-12-31"],
        estimation_size=20, pre_event_gap=5, event_window=(0, 2),
    )
    assert [e.event_date for e in result.events] == [idx[40]]
    assert result.n_events == 1
    # Beta=1 / alpha=0 construction (asset == benchmark) -> CAR ~ 0.
    assert result.events[0].car == pytest.approx(0.0, abs=1e-9)


@pytest.mark.unit
def test_r15_bs_index_at_or_after_snaps_forward_or_returns_none():
    idx = pd.Index(["2024-01-02", "2024-01-03", "2024-01-05"])
    assert _index_at_or_after(idx, "2024-01-03") == 1        # exact hit
    assert _index_at_or_after(idx, "2024-01-04") == 2         # snaps forward
    assert _index_at_or_after(idx, "2024-12-31") is None      # nothing after


@pytest.mark.unit
def test_r15_bs_fmt_formats_four_decimals_and_na():
    assert _fmt(None) == "n/a"
    assert _fmt(0.0) == "0.0000"
    assert _fmt(2.5) == "2.5000"


def _es_result(events: list[EventCAR]) -> EventStudyResult:
    cars = np.array([e.car for e in events], dtype=float) if events else np.empty(0)
    return EventStudyResult(
        events=events,
        mean_car=float(cars.mean()) if cars.size else None,
        t_stat=2.5 if cars.size else None,
        p_value=0.0125 if cars.size else None,
        ci_low=0.005 if cars.size else None,
        ci_high=0.02 if cars.size else None,
        n_events=len(events),
        estimation_size=250,
        event_window=(1, 20),
        benchmark="SPY",
    )


@pytest.mark.unit
def test_r15_bs_render_event_study_full_result_markdown():
    ev = EventCAR(
        event_date="2024-06-03", car=0.0125, alpha=0.0001, beta=1.1,
        n_estimation=250, n_event=20, rating="Buy",
    )
    out = render_event_study(_es_result([ev, ev]))
    assert "## Event study (market-model abnormal returns)" in out
    assert "Benchmark: `SPY`  |  Events used: 2" in out
    assert "Estimation window: 250 returns" in out
    assert "Event window: [1, +20]" in out
    # All statistics rendered at 4 decimals.
    assert "| Mean CAR | 0.0125 |" in out
    assert "| t-statistic | 2.5000 |" in out
    assert "| p-value (two-sided) | 0.0125 |" in out
    assert "[0.0050, 0.0200]" in out
    # Interpretation guidance only appears when events were used.
    assert "moved abnormally" in out


@pytest.mark.unit
def test_r15_bs_render_event_study_empty_result_renders_na():
    out = render_event_study(_es_result([]))
    assert "Events used: 0" in out
    assert "| Mean CAR | n/a |" in out
    assert "| t-statistic | n/a |" in out
    assert "| Bootstrap 95% CI (mean CAR) | [n/a, n/a] |" in out
    assert "moved abnormally" not in out


# =========================================================================== #
# ic: rank-Spearman fallback, no-scipy paths and edge guards
# =========================================================================== #
@pytest.mark.unit
def test_r15_bs_spearman_from_ranks_hand_value_and_degenerate_inputs():
    from yialpha.backtest.ic import _spearman_from_ranks

    fr = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    rr = pd.Series([2.0, 1.0, 4.0, 3.0, 5.0])
    # Hand-computed Pearson on ranks: cov=8, both variances=10 -> 0.8.
    assert _spearman_from_ranks(fr, rr) == pytest.approx(0.8)
    assert _spearman_from_ranks(pd.Series(dtype=float), rr) is None
    assert _spearman_from_ranks(pd.Series([5.0, 5.0, 5.0]), rr) is None
    assert _spearman_from_ranks(fr, pd.Series([1.0, 1.0, 1.0])) is None


@pytest.mark.unit
def test_r15_bs_information_coefficient_empty_inputs_return_none():
    from yialpha.backtest.ic import information_coefficient

    assert information_coefficient([], []) is None


@pytest.mark.unit
def test_r15_bs_information_coefficient_without_scipy_matches_scipy(monkeypatch):
    scipy_stats = pytest.importorskip("scipy.stats")
    from yialpha.backtest.ic import information_coefficient

    monkeypatch.setattr(ic_mod, "_HAVE_SCIPY", False)
    rng = np.random.default_rng(17)
    factor = rng.normal(size=60)
    fwd = 0.6 * factor + rng.normal(scale=0.8, size=60)
    expected = float(scipy_stats.spearmanr(factor, fwd).statistic)
    assert information_coefficient(factor, fwd) == pytest.approx(expected, abs=1e-12)


@pytest.mark.unit
def test_r15_bs_information_coefficient_spearmanr_exception_falls_back(monkeypatch):
    scipy_stats = pytest.importorskip("scipy.stats")
    from yialpha.backtest.ic import information_coefficient

    def _raise(*_args, **_kwargs):
        raise RuntimeError("scipy exploded")

    monkeypatch.setattr(ic_mod, "_scipy_spearmanr", _raise)
    rng = np.random.default_rng(17)
    factor = rng.normal(size=60)
    fwd = 0.6 * factor + rng.normal(scale=0.8, size=60)
    expected = float(scipy_stats.spearmanr(factor, fwd).statistic)
    # A scipy failure must not corrupt the IC: the rank fallback kicks in.
    assert information_coefficient(factor, fwd) == pytest.approx(expected, abs=1e-12)


@pytest.mark.unit
def test_r15_bs_rolling_ic_index_taken_from_returns_series():
    from yialpha.backtest.ic import rolling_ic

    # factor is a plain list, forward_returns a Series -> the output must
    # carry the RETURNS' index (trailing-window alignment), not a RangeIndex.
    labels = ["a", "b", "c", "d", "e", "f", "g"]
    fwd = pd.Series([2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0], index=labels)
    out = rolling_ic([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], fwd, window=5)
    assert list(out.index) == labels
    # Windows shorter than _MIN_PAIRS=5 valid pairs stay NaN; every complete
    # window holds perfectly monotone pairs -> IC exactly 1.
    assert out.iloc[:4].isna().all()
    assert np.allclose(out.iloc[4:].to_numpy(), 1.0)


@pytest.mark.unit
def test_r15_bs_rolling_ic_zero_window_is_all_nan():
    from yialpha.backtest.ic import rolling_ic

    out = rolling_ic([1.0, 2.0, 3.0], [2.0, 4.0, 6.0], window=0)
    assert len(out) == 3
    assert out.isna().all()


@pytest.mark.unit
def test_r15_bs_quantile_spread_qcut_failure_returns_none(monkeypatch):
    from yialpha.backtest.ic import quantile_spread

    def _raise(*_args, **_kwargs):
        raise ValueError("bin edges not unique")

    monkeypatch.setattr(pd, "qcut", _raise)
    factor = list(map(float, range(10)))
    out = quantile_spread(factor, [0.01] * 10)
    assert out is None


@pytest.mark.unit
def test_r15_bs_factor_turnover_too_few_rows_returns_none():
    from yialpha.backtest.ic import factor_turnover

    assert factor_turnover([1.0]) is None
    assert factor_turnover([np.nan]) is None
    assert factor_turnover([np.nan, 3.0]) is None  # one finite row after drop


# =========================================================================== #
# cache: dirless/disabled/mismatched-entry guards
# =========================================================================== #
@pytest.mark.unit
def test_r15_bs_decision_cache_from_config_roundtrip(tmp_path):
    cache = DecisionCache.from_config({"data_cache_dir": str(tmp_path)})
    assert cache.cache_dir == tmp_path / "backtest_decisions"
    decision = cache.remember("T", "2024-01-02", "Buy", "Rating: Buy", run_tag="r1")
    assert cache.get("T", "2024-01-02", run_tag="r1") == decision
    # Disabled by flag: dir is derived but every access is a no-op miss.
    off = DecisionCache.from_config({"data_cache_dir": str(tmp_path)}, enabled=False)
    assert off.get("T", "2024-01-02", run_tag="r1") is None


@pytest.mark.unit
def test_r15_bs_decision_cache_dirless_get_and_put_are_noops(tmp_path):
    cache = DecisionCache(None, enabled=True)
    assert cache._key_path("T", "2024-01-02", "r1") is None
    assert cache.get("T", "2024-01-02", "r1") is None
    cache.put(CachedDecision(
        ticker="T", date="2024-01-02", run_tag="r1", rating="Buy",
        final_decision="md",
    ))
    assert list(tmp_path.glob("*.json")) == []  # nothing written anywhere


@pytest.mark.unit
def test_r15_bs_decision_cache_disabled_put_writes_nothing(tmp_path):
    cache = DecisionCache(tmp_path, enabled=False)
    cache.put(CachedDecision(
        ticker="T", date="2024-01-02", run_tag="r1", rating="Buy",
        final_decision="md",
    ))
    assert list(tmp_path.glob("*.json")) == []


@pytest.mark.unit
def test_r15_bs_decision_cache_venue_mismatched_entry_is_miss(tmp_path):
    cache = DecisionCache(tmp_path)
    path = cache._key_path("T", "2024-01-02", "r1", "stock", "")
    payload = {
        "schema_version": 5, "ticker": "T", "date": "2024-01-02",
        "run_tag": "r1", "rating": "Buy", "final_decision": "md",
        "asset_type": "crypto", "was_degraded": False, "config_signature": "",
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert cache.get("T", "2024-01-02", "r1", asset_type="stock") is None
    # Positive control: a genuinely crypto-keyed entry roundtrips fine.
    cache.remember("T", "2024-01-02", "Buy", "md", run_tag="r1", asset_type="crypto")
    got = cache.get("T", "2024-01-02", "r1", asset_type="crypto")
    assert got is not None and got.asset_type == "crypto"


@pytest.mark.unit
def test_r15_bs_decision_cache_config_mismatched_entry_is_miss(tmp_path):
    cache = DecisionCache(tmp_path)
    decision = cache.remember(
        "T", "2024-01-02", "Buy", "md", run_tag="r1", config_signature="sigA",
    )
    path = cache._key_path("T", "2024-01-02", "r1", "stock", "sigA")
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["config_signature"] = "sigB"  # tampered / legacy collision
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert decision.config_signature == "sigA"
    assert cache.get("T", "2024-01-02", "r1", config_signature="sigA") is None


# =========================================================================== #
# Cross-module statistical pins (money-path sanity, cheap to keep)
# =========================================================================== #
@pytest.mark.unit
def test_r15_bs_abnormal_returns_pure_beta_identity():
    ar = abnormal_returns(
        np.array([0.02, 0.03, 0.0]), np.array([0.01, 0.02, -0.01]),
        alpha=0.001, beta=1.0,
    )
    # asset - (alpha + beta*bench): e.g. 0.0 - (0.001 + (-0.01)) = 0.009.
    assert ar == pytest.approx(np.array([0.009, 0.009, 0.009]))


@pytest.mark.unit
def test_r15_bs_attribution_alpha_sign_contract():
    """A strategy that merely matches the factors with zero intercept must
    report ~zero annualized factor alpha (contrast with alpha_vs_buyhold,
    which would flag market beta as 'alpha')."""
    idx = pd.bdate_range("2024-01-01", periods=60)
    rng = np.random.default_rng(9)
    fr = pd.DataFrame({
        "Mkt-RF": rng.normal(0.0004, 0.01, 60),
        "SMB": rng.normal(0.0, 0.008, 60),
        "HML": rng.normal(0.0, 0.007, 60),
        "RF": np.full(60, 0.0001),
    }, index=idx)
    strat = (fr["Mkt-RF"] + fr["RF"]).rename("ret")
    attr = factor_attribution(strat, fr, model="FF3")
    assert attr is not None
    assert attr.alpha_annual == pytest.approx(0.0, abs=1e-8)
    assert attr.betas["Mkt-RF"] == pytest.approx(1.0, abs=1e-8)
    assert attr.betas["SMB"] == pytest.approx(0.0, abs=1e-8)
    assert math.isfinite(attr.r_squared)
