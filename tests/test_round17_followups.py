"""Round-17 follow-ups: fixes chosen from the round-17 ledger's open decisions.

The round-17 audit left a list of observations whose fix was a design decision
(``r17/LEDGER.md`` "待用户决定"). On 2026-11-28 the decisions were taken and the
chosen fixes landed; this file holds their regression tests. Each test names
the ledger item it closes, and each product change has at least one test that
goes red when the change is reverted.

Hermetic: no network. Every vendor is a stand-in installed on the module that
calls it.

Item ids: ``B`` data correctness, ``C`` risk / backtest, ``D`` scripts / CLI,
``E`` agents / LLM, ``F`` web / tests, ``A`` CI / repo policy.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
import warnings
from contextlib import contextmanager
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import yialpha.accuracy as acc
import yialpha.dataflows.stockstats_utils as ssu
import yialpha.dataflows.stocktwits as stocktwits
import yialpha.dataflows.y_finance as y_fin
import yialpha.dataflows.yfinance_news as ynews
from yialpha.backtest import ic as ic_mod, metrics as bt_metrics
from yialpha.dataflows import quality
from yialpha.dataflows.config import get_config, set_config
from yialpha.dataflows.errors import NoMarketDataError
from yialpha.risk import perp_ticket

DOWN = OSError("connection reset")


def _age(path: Path, seconds: float) -> Path:
    """Write nothing; set ``path``'s mtime to ``seconds`` in the past."""
    stamp = time.time() - seconds
    os.utime(path, (stamp, stamp))
    return path


def _aged_vendor_cache(module, subdir: str, name: str, payload, age_s: float) -> Path:
    """Put ``payload`` into the vendor's real cache dir (isolated per test by the
    autouse ``_isolated_vendor_cache`` fixture) with an mtime ``age_s`` old."""
    path = Path(module.vendor_cache_dir(subdir)) / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    return _age(path, age_s)


# --------------------------------------------------------------------------- #
# B3 — a past analysis date never trusts a cache written on or before it
# --------------------------------------------------------------------------- #
@pytest.fixture()
def _clean_ohlcv_memo():
    """``load_ohlcv``'s memo is process-global; start and end each test clean."""
    with ssu._OHLCV_MEMO_LOCK:
        ssu._OHLCV_MEMO.clear()
    yield
    with ssu._OHLCV_MEMO_LOCK:
        ssu._OHLCV_MEMO.clear()


def _ohlcv_frame(end: str, closes: list[float]) -> pd.DataFrame:
    n = len(closes)
    return pd.DataFrame({
        "Date": pd.bdate_range(end=end, periods=n).strftime("%Y-%m-%d"),
        "Open": [99.0] * n, "High": [106.0] * n, "Low": [98.0] * n,
        "Close": closes, "Volume": [1_000_000] * n,
    })


@pytest.mark.unit
def test_b3_historical_date_refreshes_only_a_cache_written_on_or_before_it(tmp_path):
    """A past date's rows are immutable, but only in a cache written AFTER that
    date ended (UTC). Written on the analysis day itself it may hold an
    unclosed bar; written before it, it lacks rows."""
    cache = tmp_path / "X-YFin-data.csv"
    cache.write_text("Date,Close\n2026-11-26,10.0\n", encoding="utf-8")
    today = pd.Timestamp("2026-11-28 06:00", tz="UTC")
    yesterday, older, later = pd.Timestamp("2026-11-27"), pd.Timestamp("2026-11-20"), pd.Timestamp("2026-11-24")

    def written(day: str, hour: int) -> None:
        stamp = pd.Timestamp(f"{day} {hour:02d}:00", tz="UTC").timestamp()
        os.utime(cache, (stamp, stamp))

    written("2026-11-27", 15)  # downloaded during the previous session
    assert ssu._needs_same_day_refresh(cache, yesterday, today) is True
    written("2026-11-27", 23)  # still that UTC day: the bar may not be final
    assert ssu._needs_same_day_refresh(cache, yesterday, today) is True
    written("2026-11-28", 0)  # the day is over: the bar is final
    assert ssu._needs_same_day_refresh(cache, yesterday, today) is False
    written("2026-11-24", 12)  # newer than the analysis date
    assert ssu._needs_same_day_refresh(cache, older, today) is False
    written("2026-11-20", 12)  # OLDER than the analysis date: rows are missing
    assert ssu._needs_same_day_refresh(cache, later, today) is True


@pytest.mark.unit
def test_b3_previous_session_analysis_no_longer_reads_an_intraday_snapshot(monkeypatch, _clean_ohlcv_memo):
    """Ledger S12: the bar downloaded intraday on day D stayed in the cache and
    was read as D's close by the next day's "previous session" analysis. The
    cache was never refreshed for a past date."""
    monkeypatch.setattr(ssu, "_utc_today", lambda: pd.Timestamp("2026-11-28 12:00", tz="UTC"))
    data_file = ssu._ohlcv_cache_path(get_config(), "AAPL")
    snapshot = [99.0, 100.0, 101.0, 102.0, 103.0, 101.5]
    _ohlcv_frame("2026-11-27", snapshot).to_csv(data_file, index=False)
    stamp = pd.Timestamp("2026-11-27 20:00", tz="UTC").timestamp()  # downloaded mid-session
    os.utime(data_file, (stamp, stamp))
    downloads: list[str] = []

    def final_bars(symbol, start, end, **kwargs):
        downloads.append(symbol)
        return _ohlcv_frame("2026-11-27", [99.0, 100.0, 101.0, 102.0, 103.0, 105.0])

    monkeypatch.setattr(ssu.yf, "download", final_bars)

    out = ssu.load_ohlcv("AAPL", "2026-11-27")

    assert downloads == ["AAPL"]
    assert out["Close"].iloc[-1] == 105.0  # the final bar, not the 101.5 snapshot
    # The refreshed cache was written today: the same analysis date is served from it.
    assert ssu.load_ohlcv("AAPL", "2026-11-27")["Close"].iloc[-1] == 105.0
    assert ssu.read_cached_ohlcv("AAPL", "2026-11-27")["Close"].iloc[-1] == 105.0
    assert downloads == ["AAPL"]


@pytest.mark.unit
def test_b3_a_backtest_still_downloads_each_symbol_once(monkeypatch, _clean_ohlcv_memo):
    """The new rule must not turn every backtest date into a download: a cache
    written today serves every past analysis date unchanged."""
    monkeypatch.setattr(ssu, "_utc_today", lambda: pd.Timestamp("2026-11-28 12:00", tz="UTC"))
    data_file = ssu._ohlcv_cache_path(get_config(), "AAPL")
    _ohlcv_frame("2026-11-27", [100.0] * 400).to_csv(data_file, index=False)  # mtime = now (UTC today)

    def no_network(symbol, start, end, **kwargs):
        raise AssertionError("a cache written today must serve every past date")

    monkeypatch.setattr(ssu.yf, "download", no_network)
    # 400 business days end 2026-11-27: the first of these is inside the cached window.
    for day in ("2025-06-02", "2026-06-01", "2026-11-20", "2026-11-26"):
        assert len(ssu.load_ohlcv("AAPL", day)) > 0


# --------------------------------------------------------------------------- #
# B4 — a return is scored only over finite, positive closes
# --------------------------------------------------------------------------- #
def _install_tape(monkeypatch, closes: dict[str, list[float]]) -> None:
    sessions = pd.bdate_range("2026-01-05", periods=8)

    def history(symbol, start, end):
        return pd.DataFrame({"Close": closes[symbol]}, index=pd.DatetimeIndex(sessions))

    monkeypatch.setattr(y_fin, "get_YFin_history_cached", history)
    monkeypatch.setattr(acc, "_us_equity_today", lambda: date(2026, 6, 1))


@pytest.mark.unit
@pytest.mark.parametrize(
    ("leg", "position", "bad"),
    [
        ("AAPL", 0, float("nan")),
        ("AAPL", 5, float("nan")),
        ("SPY", 0, 0.0),
        ("SPY", 5, 0.0),
        ("AAPL", 5, float("inf")),
        ("AAPL", 5, -1.0),
    ],
)
def test_b4_fetch_returns_needs_finite_positive_closes_on_both_legs(monkeypatch, leg, position, bad):
    """Ledger S7: a NaN / zero / infinite close came back as nan / -100% / inf,
    the reflection layer wrote ``+nan%`` into the memory log and retired the
    entry. Such an outcome stays pending instead."""
    closes = {"AAPL": [100.0] * 8, "SPY": [500.0] * 8}
    closes[leg][position] = bad
    _install_tape(monkeypatch, closes)

    assert acc.fetch_returns_yf("AAPL", "2026-01-05", holding_days=5) == (None, None, None)


@pytest.mark.unit
def test_b4_fetch_returns_still_scores_a_clean_tape(monkeypatch):
    _install_tape(monkeypatch, {"AAPL": [100.0 + i for i in range(8)], "SPY": [500.0] * 8})

    assert acc.fetch_returns_yf("AAPL", "2026-01-05", holding_days=5) == (
        pytest.approx(0.05), pytest.approx(0.05), 5,
    )


# --------------------------------------------------------------------------- #
# B5 — minute-scale feeds are not replayed days old
# --------------------------------------------------------------------------- #
_STW_PAYLOAD = {"messages": [{
    "created_at": "2026-01-01T00:00:00Z", "user": {"username": "u"},
    "entities": {"sentiment": {"basic": "Bullish"}}, "body": "cached chatter",
}]}


def _stocktwits_down(monkeypatch, age_s: float) -> None:
    """A cached StockTwits stream ``age_s`` old and a dead endpoint."""
    _aged_vendor_cache(stocktwits, "stocktwits", "stream_AAPL.json", _STW_PAYLOAD, age_s)

    def down(req, timeout=10.0):
        raise DOWN

    monkeypatch.setattr(stocktwits, "urlopen", down)


def _search_down(monkeypatch, query: str, age_s: float) -> None:
    """A cached global-news Search for ``query`` ``age_s`` old and a dead Search."""
    key = json.dumps({"q": query, "n": 10}, sort_keys=True).encode("utf-8")
    name = f"search_{hashlib.sha256(key).hexdigest()[:12]}.json"
    _aged_vendor_cache(ynews, "yfnews", name, [{"title": "cached headline"}], age_s)

    def boom(*args, **kwargs):
        raise DOWN

    monkeypatch.setattr(ynews.yf, "Search", boom)


@pytest.mark.unit
def test_b5_each_feed_may_be_served_stale_for_three_times_its_ttl():
    """The rule binance_square already followed (5-minute TTL, 15-minute cap),
    applied to the two feeds that still fell back to the 30-day global cap."""
    stale_minutes = stocktwits._MAX_STALE_MINUTES
    stale_days = ynews._SEARCH_MAX_STALE_DAYS
    assert stale_minutes == pytest.approx(3 * stocktwits._CACHE_TTL_DAYS * 1440)
    assert stale_days == pytest.approx(3 * ynews._SEARCH_CACHE_TTL_DAYS)


@pytest.mark.unit
def test_b5_stocktwits_replays_a_failed_fetch_for_at_most_fifteen_minutes(monkeypatch):
    set_config({"data_cache_max_stale_days": 30})  # the global cap, which must not decide here
    quality.ensure_run_context()

    _stocktwits_down(monkeypatch, age_s=10 * 60)
    out = stocktwits.fetch_stocktwits_messages("AAPL")
    assert "cached chatter" in out
    assert any(e["kind"] == quality.KIND_STALE_CACHE for e in quality.snapshot_quality())

    quality.reset_quality()
    _stocktwits_down(monkeypatch, age_s=20 * 60)
    out = stocktwits.fetch_stocktwits_messages("AAPL")
    assert out == "<stocktwits unavailable: OSError>"
    assert any(
        e["method"] == "fetch_stocktwits_messages" and e["kind"] == quality.KIND_OPTIONAL_UNAVAILABLE
        for e in quality.snapshot_quality()
    )


@pytest.mark.unit
def test_b5_global_news_replays_a_failed_fetch_for_at_most_ninety_minutes(monkeypatch):
    set_config({"data_cache_max_stale_days": 30})
    _search_down(monkeypatch, "stock market", age_s=60 * 60)
    assert ynews._cached_search_news("stock market", 10) == [{"title": "cached headline"}]

    _search_down(monkeypatch, "stock market", age_s=2 * 3600)
    with pytest.raises(NoMarketDataError, match="Yahoo unreachable"):
        ynews._cached_search_news("stock market", 10)


# --------------------------------------------------------------------------- #
# B6 — the typed Yahoo error names the instrument
# --------------------------------------------------------------------------- #
class _DownTicker:
    """``yf.Ticker`` stand-in: every data call dies at the transport level."""

    def __init__(self, *args, **kwargs):
        pass

    @property
    def info(self):
        raise DOWN

    def get_income_stmt(self, **kwargs):
        raise DOWN

    def history(self, start, end):
        raise DOWN

    def get_news(self, count):
        raise DOWN


@pytest.mark.unit
@pytest.mark.parametrize(
    ("where", "expected"),
    [
        ("ticker_info", ("brk.b", "BRK-B")),
        ("statement_frame", ("MSFT", "MSFT")),
        ("history", ("SPY", "SPY")),
    ],
)
def test_b6_yfinance_errors_name_the_symbol(monkeypatch, where, expected):
    """Five ``yf_retry`` call sites passed no symbol, so the error the model
    read was ``No usable market data for '?'``."""
    monkeypatch.setattr(y_fin.yf, "Ticker", _DownTicker)
    monkeypatch.setattr(y_fin, "cached_or_fetch", lambda base, name, fetch, **kw: fetch())
    with pytest.raises(NoMarketDataError) as excinfo:
        if where == "ticker_info":
            y_fin._cached_ticker_info("brk.b", "BRK-B")
        elif where == "statement_frame":
            y_fin._cached_yf_frame("MSFT", "x.csv", _DownTicker().get_income_stmt)
        else:
            y_fin.get_YFin_history_cached("SPY", "2026-01-01", "2026-06-11")
    assert (excinfo.value.symbol, excinfo.value.canonical) == expected
    assert "'?'" not in str(excinfo.value)


@pytest.mark.unit
def test_b6_ticker_news_error_names_the_ticker(monkeypatch):
    monkeypatch.setattr(ynews.yf, "Ticker", _DownTicker)
    with pytest.raises(NoMarketDataError) as excinfo:
        ynews.get_news_yfinance("aapl", "2025-01-01", "2025-01-10")
    assert (excinfo.value.symbol, excinfo.value.canonical) == ("aapl", "AAPL")


@pytest.mark.unit
def test_b6_global_news_error_names_the_query(monkeypatch):
    def boom(*args, **kwargs):
        raise DOWN

    monkeypatch.setattr(ynews.yf, "Search", boom)
    monkeypatch.setattr(ynews, "cached_or_fetch", lambda base, name, fetch, **kw: fetch())
    with pytest.raises(NoMarketDataError) as excinfo:
        ynews._cached_search_news("stock market today", 10)
    assert excinfo.value.symbol == "news search: stock market today"


# --------------------------------------------------------------------------- #
# C2 — an unknown risk profile is sized under the conservative one
# --------------------------------------------------------------------------- #
_PT_LOGGER = "yialpha.risk.perp_ticket"


@pytest.mark.unit
def test_c2_unknown_risk_profile_sizes_under_the_conservative_profile(caplog):
    """Ledger S13: a misspelt ``risk_profile`` took the 5.0 default of the
    conviction cap -- for a weak rating (Overweight / Underweight) looser than
    the conservative 3x, so the typo quietly raised the per-ticket leverage
    ceiling."""
    with caplog.at_level(logging.WARNING, logger=_PT_LOGGER):
        weak_lev, weak = perp_ticket.compute_leverage(0.02, 0.03, "crypto_perp", 1, profile="conservativ")
        strong_lev, strong = perp_ticket.compute_leverage(0.02, 0.03, "crypto_perp", 2, profile="Moderate")
    assert (weak["L_conv"], weak_lev) == (3.0, 3.0)
    assert (strong["L_conv"], strong_lev) == (5.0, 5.0)
    assert len(caplog.records) == 2
    assert all("risk_profile" in r.getMessage() and "conservative" in r.getMessage() for r in caplog.records)


@pytest.mark.unit
@pytest.mark.parametrize("unset", [None, ""])
def test_c2_an_unset_profile_is_the_default_and_does_not_warn(caplog, unset):
    with caplog.at_level(logging.WARNING, logger=_PT_LOGGER):
        _lev, caps = perp_ticket.compute_leverage(0.02, 0.03, "crypto_perp", 1, profile=unset)
    assert caps["L_conv"] == 3.0
    assert caplog.records == []


@pytest.mark.unit
def test_c2_a_non_string_profile_is_sized_under_the_conservative_profile(caplog):
    """A list or a number where a name belongs (a hand-edited config file)."""
    with caplog.at_level(logging.WARNING, logger=_PT_LOGGER):
        _lev, caps = perp_ticket.compute_leverage(0.02, 0.03, "crypto_perp", 1, profile=["moderate"])
    assert caps["L_conv"] == 3.0
    assert len(caplog.records) == 1


@pytest.mark.unit
@pytest.mark.parametrize(
    ("strength", "profile", "cap"),
    [
        (2, "conservative", 5.0), (2, "moderate", 10.0), (2, "aggressive", 15.0),
        (1, "conservative", 3.0), (1, "moderate", 5.0), (1, "aggressive", 8.0),
    ],
)
def test_c2_known_profiles_keep_their_caps(caplog, strength, profile, cap):
    with caplog.at_level(logging.WARNING, logger=_PT_LOGGER):
        _lev, caps = perp_ticket.compute_leverage(0.02, 0.01, "crypto_perp", strength, profile=profile)
    assert caps["L_conv"] == cap
    assert caplog.records == []


@pytest.mark.unit
def test_c2_config_check_reports_the_risk_profile(monkeypatch):
    """``yialpha config-check`` names an unknown profile as an error line and
    still exits 0 (advisory, like indicator_battery)."""
    from typer.testing import CliRunner

    from yialpha.cli.main import app

    runner = CliRunner()
    monkeypatch.setenv("YIALPHA_LLM_PROVIDER", "deepseek")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test-dummy")
    # Rich folds at the terminal width; pin it wide so the line is not broken up.
    monkeypatch.setenv("COLUMNS", "400")
    set_config({"risk_profile": "conservativ"})
    result = runner.invoke(app, ["config-check"])
    assert result.exit_code == 0, result.output
    assert "❌ risk_profile='conservativ' is not one of conservative, moderate, aggressive" in result.output
    assert "falls back to conservative and warns" in result.output

    set_config({"risk_profile": "moderate"})
    assert "✅ risk_profile: moderate" in runner.invoke(app, ["config-check"]).output
    set_config({"risk_profile": None})
    assert "risk_profile unset — conservative (default)" in runner.invoke(app, ["config-check"]).output


# --------------------------------------------------------------------------- #
# F2 — degenerate inputs are handled silently (numpy / scipy warnings removed)
# --------------------------------------------------------------------------- #
@contextmanager
def _no_warnings():
    """Any RuntimeWarning or UserWarning inside the block is an error. numpy's
    divide/invalid warnings are RuntimeWarnings; scipy's ``ConstantInputWarning``
    is a UserWarning."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        warnings.simplefilter("error", UserWarning)
        yield


@pytest.mark.unit
@pytest.mark.parametrize("with_scipy", [True, False])
def test_f2a_constant_ic_input_is_none_without_a_warning(monkeypatch, with_scipy):
    """Ledger S12: ``scipy.stats.spearmanr`` warned ("An input array is
    constant") and returned NaN, which was then normalised to None anyway."""
    monkeypatch.setattr(ic_mod, "_HAVE_SCIPY", with_scipy)
    factor = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    with _no_warnings():
        assert ic_mod.information_coefficient([7.0] * 6, factor) is None
        assert ic_mod.information_coefficient(factor, [7.0] * 6) is None
        # A non-constant input still correlates.
        assert ic_mod.information_coefficient(factor, [2.0, 4.0, 6.0, 8.0, 10.0, 12.0]) == pytest.approx(1.0)


@pytest.mark.unit
def test_f2a_constant_input_never_reaches_scipy(monkeypatch):
    """The gate sits in front of the accelerator. A warning-as-error cannot show
    that: the scipy call is wrapped in ``except Exception``, which swallows a
    warning turned into an exception as well, so the old code looked silent too
    (the revert proof of this batch found that)."""
    spearmanr = pytest.importorskip("scipy.stats").spearmanr
    asked: list[int] = []

    def spy(a, b):
        asked.append(1)
        return spearmanr(a, b)

    monkeypatch.setattr(ic_mod, "_HAVE_SCIPY", True)
    monkeypatch.setattr(ic_mod, "_scipy_spearmanr", spy)
    factor = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]

    assert ic_mod.information_coefficient([7.0] * 6, factor) is None
    assert ic_mod.information_coefficient(factor, [7.0] * 6) is None
    assert asked == []  # refused before the accelerator was consulted
    assert ic_mod.information_coefficient(factor, [2.0, 4.0, 6.0, 8.0, 10.0, 12.0]) == pytest.approx(1.0)
    assert asked == [1]


@pytest.mark.unit
def test_f2b_zero_equity_points_no_longer_raise_numpy_warnings():
    """Ledger S12: a curve through zero seeded 0/0 and x/0 arithmetic whose
    numpy warnings ran through the means, the drawdown and the scipy moments.
    The metrics themselves are unchanged."""
    with _no_warnings():
        returns = bt_metrics.returns_from_equity([0.0, 5.0, 10.0])
        wiped = bt_metrics.compute_metrics([0.0, 5.0, 10.0])
        gapped = bt_metrics.compute_metrics([100.0, 0.0, 0.0, 10.0])
    assert returns[0] == np.inf
    assert (wiped.total_return, wiped.cagr) == (-1.0, -1.0)
    assert np.isnan(gapped.deflated_sharpe) and gapped.sharpe == 0.0


# --------------------------------------------------------------------------- #
# D2 — the batch runner hands the graph the ticker it validated (stripped)
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_d2_batch_runner_strips_the_tickers_it_is_given(tmp_path):
    """Ledger S11: ``prepare_batch_run`` validated ``strip()``-ed tickers but the
    original strings went down to ``propagate()`` -- ``--tickers " AAPL "`` only
    failed when a report directory was named after it."""
    from types import SimpleNamespace

    from yialpha.batch.runner import BatchRunner

    config = {
        "results_dir": str(tmp_path / "results"),
        "data_cache_dir": str(tmp_path / "cache"),
        "batch_concurrency": False,
    }
    asked: list[tuple[str, str]] = []

    def propagate(ticker, trade_date, asset_type="stock"):
        asked.append((ticker, trade_date))
        return {"ticker": ticker}, {"rating": "Buy"}

    graph = SimpleNamespace(
        config=dict(config), propagate=propagate,
        save_reports=lambda state, ticker: tmp_path / f"{ticker}.md",
    )
    runner = BatchRunner(config, workers=1, graph_factory=lambda cfg: graph, progress=False)
    try:
        rows = runner.run([" AAPL ", "MSFT\t", "NVDA"], "2026-06-27")
    finally:
        runner.close()

    assert asked == [("AAPL", "2026-06-27"), ("MSFT", "2026-06-27"), ("NVDA", "2026-06-27")]
    assert [row["ticker"] for row in rows] == ["AAPL", "MSFT", "NVDA"]
    assert [row["error"] for row in rows] == [None, None, None]


# --------------------------------------------------------------------------- #
# F1 — the Web listings skip an entry that cannot be inspected
# --------------------------------------------------------------------------- #
def _state_log(root: Path, ticker: str, day: str) -> Path:
    d = root / ticker / "YiAlphaStrategy_logs"
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"full_states_log_{day}.json"
    path.write_text(json.dumps({
        "company_of_interest": ticker, "trade_date": day, "final_trade_decision": "**Rating**: Buy",
    }), encoding="utf-8")
    return d


@pytest.fixture()
def _web_roots(monkeypatch, tmp_path):
    from web import store

    root = tmp_path / "logs"
    monkeypatch.setattr(store, "LOGS_ROOT", root)
    monkeypatch.setattr(store, "REPORTS_ROOT", root / "reports")
    return store, root


@pytest.mark.unit
def test_f1_web_ticker_listings_skip_a_ticker_dir_that_cannot_be_stat_ed(_web_roots, monkeypatch):
    """Ledger S10: ``Path.is_dir()`` re-raises a stat error on Python <= 3.13, so
    one ticker directory that cannot be inspected (an unusable link target, an
    odd ACL) took ``/api/tickers`` and ``/api/compare`` down with a 500. The
    same guard ``list_reports`` got in round 17 now covers these listings."""
    store, root = _web_roots
    _state_log(root, "AAPL", "2026-06-10")
    _state_log(root, "ZZZ", "2026-06-10")  # would be the second ticker, by name order
    blocked = root / "ZZZ"
    real_stat = Path.stat

    def stat(self, *args, **kwargs):
        if self == blocked or blocked in self.parents:  # the directory and everything under it
            raise PermissionError(13, "Permission denied")
        return real_stat(self, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", stat)

    assert [row["ticker"] for row in store.list_tickers()] == ["AAPL"]
    assert [row["ticker"] for row in store.list_compare()["tickers"]] == ["AAPL"]
    assert store._dates_for("ZZZ") == []
    assert store._dates_for("AAPL") == ["2026-06-10"]


@pytest.mark.unit
@pytest.mark.parametrize(
    "reader",
    [
        lambda store, day: store._latest_rating("AAPL", day),
        lambda store, day: store.load_run("AAPL", day),
        lambda store, day: store.load_node_perf("AAPL", day),
    ],
    ids=["latest_rating", "load_run", "load_node_perf"],
)
def test_f1_web_file_readers_answer_none_for_a_file_that_cannot_be_stat_ed(_web_roots, monkeypatch, reader):
    store, root = _web_roots
    day = "2026-06-10"
    run_dir = _state_log(root, "AAPL", day)
    (run_dir / f"node_perf_{day}.json").write_text(json.dumps({"trade_date": day}), encoding="utf-8")
    assert reader(store, day) is not None  # everything readable first
    blocked = {run_dir / f"full_states_log_{day}.json", run_dir / f"node_perf_{day}.json"}
    real_stat = Path.stat

    def stat(self, *args, **kwargs):
        if self in blocked:
            raise PermissionError(13, "Permission denied")
        return real_stat(self, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", stat)

    assert reader(store, day) is None


# --------------------------------------------------------------------------- #
# A1 — the dependency audit runs weekly and by hand, never as a push gate
# --------------------------------------------------------------------------- #
_REPO = Path(__file__).resolve().parents[1]


@pytest.mark.unit
def test_a1_pip_audit_job_is_weekly_and_by_hand_only():
    """Item A1: ``requirements.lock`` is audited with pip-audit in the weekly
    scheduled run and on a manual dispatch. It is not a push / PR gate -- a new
    advisory is published on someone else's schedule, and gating pushes on it
    turns main red for a change nobody made."""
    text = (_REPO / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    start = text.index("\n  dependency-audit:\n")
    job = text[start:text.index("\n  test-windows:\n", start)]

    assert "if: github.event_name == 'schedule' || github.event_name == 'workflow_dispatch'" in job
    # --no-deps: without it pip-audit resolves and installs an unconstrained
    # dependency set and audits that instead of the lock.
    assert "pip-audit -r requirements.lock --no-deps" in job
    assert "- cron: \"15 3 * * 0\"" in text  # Sunday 03:15 UTC
    assert "\n  workflow_dispatch:\n" in text
