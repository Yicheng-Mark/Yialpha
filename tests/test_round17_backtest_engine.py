"""Round-17 audit tests for the backtest engine shard (``yialpha/backtest``).

Hermetic: every price, funding or benchmark series is injected; nothing here
touches the network. Each test pins ONE contract named in its docstring.
"""

from __future__ import annotations

import contextlib
import dataclasses
import logging
import re
import time
from collections.abc import Callable, Mapping, Sequence
from datetime import date
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest

from yialpha.backtest import (
    engine as eng,
    event_study as es_mod,
    factor_model as fm,
    ic_dataset as icd,
    report as rep,
)
from yialpha.backtest.engine import BacktestResult, TradeRow
from yialpha.backtest.metrics import compute_metrics
from yialpha.dataflows.utils import pinned_analysis_date


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def _fake_yfinance(monkeypatch: pytest.MonkeyPatch, hist: pd.DataFrame, calls: list[tuple]) -> None:
    """Replace ``yfinance.Ticker`` (imported lazily inside the provider body).

    ``calls`` collects ``(symbol, start, end, auto_adjust)`` for every
    ``history()`` request so a test can pin the exact arguments.
    """
    import yfinance as yf

    def _ticker(symbol: str) -> SimpleNamespace:
        def _history(*, start: str, end: str, auto_adjust: bool) -> pd.DataFrame:
            calls.append((symbol, start, end, auto_adjust))
            return hist

        return SimpleNamespace(history=_history)

    monkeypatch.setattr(yf, "Ticker", _ticker)


# ---------------------------------------------------------------------------
# engine._yfinance_price_provider
# ---------------------------------------------------------------------------
class TestYfinancePriceProvider:
    def test_r17_yfinance_provider_strips_tz_and_reindexes_to_date_strings(self, monkeypatch):
        """A tz-aware daily frame comes back as a float Series keyed by
        'YYYY-MM-DD' strings with NaN closes dropped; Yahoo receives the
        normalized symbol (``BTCUSDT`` -> ``BTC-USD``) and ``auto_adjust``."""
        hist = pd.DataFrame(
            {"Close": [100.0, np.nan, 104.0], "Open": [1.0, 2.0, 3.0]},
            index=pd.DatetimeIndex(["2024-01-02", "2024-01-03", "2024-01-04"], tz="UTC"),
        )
        calls: list[tuple] = []
        _fake_yfinance(monkeypatch, hist, calls)

        s = eng._yfinance_price_provider("BTCUSDT", "2024-01-01", "2024-01-05")

        assert calls == [("BTC-USD", "2024-01-01", "2024-01-05", True)]
        assert list(s.index) == ["2024-01-02", "2024-01-04"]
        assert all(isinstance(x, str) for x in s.index)
        assert s.tolist() == [100.0, 104.0]
        assert s.dtype == float

    def test_r17_yfinance_provider_keeps_naive_index_and_collapses_to_dates(self, monkeypatch):
        """A tz-naive index needs no localization; intraday timestamps collapse
        to their calendar date string."""
        hist = pd.DataFrame(
            {"Close": [10.0, 11.0]},
            index=pd.DatetimeIndex(["2024-03-05 09:30:00", "2024-03-06 16:00:00"]),
        )
        _fake_yfinance(monkeypatch, hist, [])

        s = eng._yfinance_price_provider("aapl", "2024-03-01", "2024-03-10")

        assert not isinstance(s.index, pd.DatetimeIndex)  # plain string index
        assert list(s.index) == ["2024-03-05", "2024-03-06"]
        assert s.tolist() == [10.0, 11.0]

    def test_r17_yfinance_provider_empty_history_returns_empty_float_series(self, monkeypatch):
        """No bars -> an empty float Series (not a KeyError, not NaN)."""
        _fake_yfinance(monkeypatch, pd.DataFrame(), [])

        s = eng._yfinance_price_provider("NOPE", "2024-01-01", "2024-01-05")

        assert s.empty
        assert s.dtype == float


# ---------------------------------------------------------------------------
# engine._index_position_at_or_after / engine._asset_return
# ---------------------------------------------------------------------------
class TestIndexPositionAtOrAfter:
    _IDX = pd.Index(["2024-01-02", "2024-01-03", "2024-01-05"])

    @pytest.mark.parametrize(
        ("target", "expected"),
        [
            ("2024-01-01", 0),  # before the first bar -> first bar
            ("2024-01-02", 0),  # exact
            ("2024-01-03", 1),  # exact
            ("2024-01-04", 2),  # gap day -> first strictly-later bar
            ("2024-01-05", 2),  # exact last bar
        ],
    )
    def test_r17_index_position_exact_else_first_strictly_later(self, target, expected):
        assert eng._index_position_at_or_after(self._IDX, target) == expected

    def test_r17_index_position_none_when_no_bar_on_or_after_target(self):
        assert eng._index_position_at_or_after(self._IDX, "2024-01-06") is None
        assert eng._index_position_at_or_after(pd.Index([]), "2024-01-02") is None

    def test_r17_index_position_duplicate_dates_return_first_occurrence(self):
        idx = pd.Index(["2024-01-02", "2024-01-02", "2024-01-03"])
        assert eng._index_position_at_or_after(idx, "2024-01-02") == 0


class TestAssetReturn:
    # 2024-01-04 is a Thursday left out on purpose: a gap INSIDE the window.
    _PRICES = pd.Series(
        [100.0, 110.0, 121.0, 133.1],
        index=["2024-01-02", "2024-01-03", "2024-01-05", "2024-01-08"],
    )

    def test_r17_asset_return_holds_from_start_bar_for_horizon_sessions(self):
        """``horizon`` counts SESSIONS forward from the entry bar (not calendar
        days): entry on the 3rd, two sessions later is the 8th."""
        assert eng._asset_return(self._PRICES, "2024-01-03", 2) == pytest.approx(133.1 / 110.0 - 1.0)
        assert eng._asset_return(self._PRICES, "2024-01-03", 1) == pytest.approx(121.0 / 110.0 - 1.0)

    def test_r17_asset_return_enters_at_first_bar_on_or_after_start_date(self):
        """A start date with no bar (weekend / holiday) enters on the next
        available bar at that bar's close -- never the bar BEFORE the date."""
        assert eng._asset_return(self._PRICES, "2024-01-04", 1) == pytest.approx(133.1 / 121.0 - 1.0)
        assert eng._asset_return(self._PRICES, "2024-01-04", 5) == pytest.approx(133.1 / 121.0 - 1.0)
        # Horizon longer than the remaining window clips to the last bar.
        assert eng._asset_return(self._PRICES, "2024-01-02", 10) == pytest.approx(133.1 / 100.0 - 1.0)

    def test_r17_asset_return_none_when_no_forward_bar_exists(self):
        assert eng._asset_return(self._PRICES, "2024-01-08", 1) is None  # last bar
        assert eng._asset_return(self._PRICES, "2024-01-09", 1) is None  # after the last bar
        assert eng._asset_return(pd.Series(dtype=float), "2024-01-02", 1) is None

    def test_r17_asset_return_none_on_zero_or_non_finite_entry_or_exit_price(self):
        bad_entry = pd.Series([0.0, 10.0], index=["2024-01-02", "2024-01-03"])
        assert eng._asset_return(bad_entry, "2024-01-02", 1) is None
        bad_exit = pd.Series([10.0, float("nan")], index=["2024-01-02", "2024-01-03"])
        assert eng._asset_return(bad_exit, "2024-01-02", 1) is None
        inf_exit = pd.Series([10.0, float("inf")], index=["2024-01-02", "2024-01-03"])
        assert eng._asset_return(inf_exit, "2024-01-02", 1) is None


# ---------------------------------------------------------------------------
# engine._split_ratio
# ---------------------------------------------------------------------------
def _record(**details) -> SimpleNamespace:
    return SimpleNamespace(details=dict(details))


class TestSplitRatio:
    def test_r17_split_ratio_details_ratio_wins_over_to_from_pair(self):
        assert eng._split_ratio(_record(ratio=2.0, **{"to": 5, "from": 1})) == 2.0
        assert eng._split_ratio(_record(ratio="4")) == 4.0
        assert eng._split_ratio(_record(ratio=0.5)) == 0.5  # reverse split

    def test_r17_split_ratio_to_over_from_when_ratio_absent(self):
        assert eng._split_ratio(_record(**{"to": 2, "from": 1})) == 2.0
        assert eng._split_ratio(_record(ratio=None, **{"to": 1, "from": 10})) == 0.1

    def test_r17_split_ratio_none_on_missing_or_unparseable_details(self):
        assert eng._split_ratio(_record()) is None
        assert eng._split_ratio(SimpleNamespace(details=None)) is None
        assert eng._split_ratio(_record(ratio="2:1")) is None
        assert eng._split_ratio(_record(**{"to": 2})) is None  # no "from"
        assert eng._split_ratio(_record(**{"to": "x", "from": 1})) is None
        assert eng._split_ratio(_record(**{"to": 2, "from": 0})) is None  # division guarded


# ---------------------------------------------------------------------------
# report._sparkline
# ---------------------------------------------------------------------------
class TestSparkline:
    def test_r17_sparkline_empty_and_flat_series(self):
        assert rep._sparkline([]) == ""
        assert rep._sparkline([7.0, 7.0, 7.0]) == "███"
        assert rep._sparkline([-1.25]) == "█"

    def test_r17_sparkline_buckets_between_min_and_max(self):
        assert rep._sparkline([0.0, 1.0, 2.0, 3.0]) == "▁▃▅█"
        assert rep._sparkline([3.0, 2.0, 1.0, 0.0]) == "█▅▃▁"
        assert rep._sparkline([100.0, 100.0, 103.0, 100.0]) == "▁▁█▁"


# ---------------------------------------------------------------------------
# engine.BacktestResult series accessors / engine._finish_open_position_episode
# ---------------------------------------------------------------------------
def _result(equity: list[float], dates: list[str], bench: list[float]) -> BacktestResult:
    return BacktestResult(
        ticker="AAPL",
        initial_capital=100.0,
        holding_days=1,
        equity=equity,
        equity_dates=dates,
        trades=[],
        benchmark_equity=bench,
        benchmark_name="SPY",
    )


class TestBacktestResultSeries:
    def test_r17_equity_series_is_float_series_on_equity_dates(self):
        s = _result([100.0, 101.0], ["2024-01-02", "2024-01-03"], [50.0, 50.5]).equity_series()
        assert list(s.index) == ["2024-01-02", "2024-01-03"]
        assert s.dtype == float
        assert s.tolist() == [100.0, 101.0]

    def test_r17_benchmark_series_shares_the_equity_date_index(self):
        """The benchmark curve is indexed by the SAME dates as the equity curve
        (not by a benchmark-only calendar), so relative return aligns bar for bar."""
        res = _result([100.0, 101.0], ["2024-01-02", "2024-01-03"], [50.0, 50.5])
        s = res.benchmark_series()
        assert list(s.index) == list(res.equity_series().index)
        assert s.tolist() == [50.0, 50.5]
        assert s.dtype == float

    def test_r17_equity_series_empty_run_gives_empty_float_series(self):
        s = _result([], [], []).equity_series()
        assert s.empty
        assert s.dtype == float


class TestFinishOpenPositionEpisode:
    def test_r17_finish_episode_stamps_net_return_on_opening_trade(self):
        trade = TradeRow(date="2024-01-02", rating="Buy", target_weight=1.0, executed_weight=1.0, price=10.0)
        assert trade.position_return is None  # untouched until the episode closes
        realized = eng._finish_open_position_episode(trade, 100.0, 105.0)
        assert realized == pytest.approx(0.05)
        assert trade.position_return == pytest.approx(0.05)

    def test_r17_finish_episode_full_loss_is_minus_one(self):
        trade = TradeRow(date="2024-01-02", rating="Buy", target_weight=1.0, executed_weight=1.0, price=10.0)
        assert eng._finish_open_position_episode(trade, 100.0, 0.0) == pytest.approx(-1.0)

    @pytest.mark.parametrize(
        ("opening_equity", "ending_equity"),
        [(None, 110.0), (0.0, 110.0), (-5.0, 110.0)],
    )
    def test_r17_finish_episode_none_when_opening_equity_missing_or_non_positive(
        self, opening_equity, ending_equity
    ):
        """No opening equity (or a non-positive one) must not fabricate a return
        (``x / 0`` -> inf) -- the episode is reported as unknown."""
        trade = TradeRow(date="2024-01-02", rating="Buy", target_weight=1.0, executed_weight=1.0, price=10.0)
        assert eng._finish_open_position_episode(trade, opening_equity, ending_equity) is None
        assert trade.position_return is None

    def test_r17_finish_episode_none_without_an_opening_trade(self):
        """A cash episode (no opening trade) has no return and stamps nothing."""
        assert eng._finish_open_position_episode(None, 100.0, 110.0) is None


# ---------------------------------------------------------------------------
# engine._binance_funding_provider
# ---------------------------------------------------------------------------
def _utc_ms(moment: str) -> int:
    """Epoch milliseconds of an ISO UTC instant (hand-computable in tests)."""
    y, m, d = (int(x) for x in moment[:10].split("-"))
    hh, mm, ss = (int(x) for x in moment[11:19].split(":"))
    epoch_day = (date(y, m, d) - date(1970, 1, 1)).days
    return ((epoch_day * 24 + hh) * 60 + mm) * 60_000 + ss * 1000


def _fake_pagination(monkeypatch: pytest.MonkeyPatch, rows: object) -> list[tuple]:
    """Stub ``yialpha.dataflows.binance._paginate_history`` and record its args."""
    from yialpha.dataflows import binance as binance_mod

    calls: list[tuple] = []

    def _fake(path, base_params, page_limit, cursor_of, start_ms, end_ms, symbol_for_error, canonical):
        calls.append((path, base_params, page_limit, cursor_of, start_ms, end_ms, symbol_for_error, canonical))
        return rows

    monkeypatch.setattr(binance_mod, "_paginate_history", _fake)
    return calls


class TestBinanceFundingProvider:
    def test_r17_funding_provider_sums_settlements_per_utc_day(self, monkeypatch):
        """Each UTC day's settlement rates are SUMMED; bucketing is by UTC date
        (the 20:00Z settlement is already 'tomorrow' on a UTC+8 box)."""
        rows = [
            {"fundingTime": _utc_ms("2024-01-02T00:00:00"), "fundingRate": "0.0001"},
            {"fundingTime": _utc_ms("2024-01-02T08:00:00"), "fundingRate": "0.0002"},
            {"fundingTime": _utc_ms("2024-01-02T20:00:00"), "fundingRate": "0.0003"},
            {"fundingTime": _utc_ms("2024-01-03T08:00:00"), "fundingRate": "-0.0005"},
        ]
        calls = _fake_pagination(monkeypatch, rows)

        s = eng._binance_funding_provider("BTC-USDT", "2024-01-02", "2024-01-04")

        assert s.to_dict() == pytest.approx({"2024-01-02": 0.0006, "2024-01-03": -0.0005})
        assert list(s.index) == ["2024-01-02", "2024-01-03"]  # first-seen (oldest-first) order
        assert s.dtype == float
        (call,) = calls
        path, params, page_limit, cursor_of, start_ms, end_ms, symbol_for_error, canonical = call
        assert path == "/fapi/v1/fundingRate"
        assert params == {"symbol": "BTCUSDT"}
        assert page_limit == 1000
        assert symbol_for_error == "BTC-USDT"
        assert canonical == "BTCUSDT"
        assert cursor_of(rows[0]) == rows[0]["fundingTime"]
        assert start_ms == 1704153600000  # 2024-01-02T00:00:00Z
        assert end_ms == 1704412799000  # 2024-01-04T23:59:59Z -- end date INCLUSIVE
        assert (start_ms, end_ms) == (_utc_ms("2024-01-02T00:00:00"), _utc_ms("2024-01-04T23:59:59"))

    def test_r17_funding_provider_clamps_window_end_to_the_pinned_analysis_date(self, monkeypatch):
        """Under a pinned PIT analysis date the request window (and therefore the
        aggregated series) cannot extend past that date."""
        rows = [
            {"fundingTime": _utc_ms("2024-01-03T00:00:00"), "fundingRate": "0.0001"},
            {"fundingTime": _utc_ms("2024-01-04T00:00:00"), "fundingRate": "0.0009"},
        ]
        calls = _fake_pagination(monkeypatch, rows)

        with pinned_analysis_date("2024-01-03"):
            eng._binance_funding_provider("BTC-USDT", "2024-01-02", "2024-01-09")

        ((_, _, _, _, start_ms, end_ms, _, _),) = calls
        assert start_ms == 1704153600000  # 2024-01-02T00:00:00Z
        assert end_ms == 1704326399000  # 2024-01-03T23:59:59Z, not the requested 01-09
        assert end_ms == _utc_ms("2024-01-03T23:59:59")

    def test_r17_funding_provider_skips_malformed_rows(self, monkeypatch):
        """Rows that are not dicts / carry no settlement time / no parseable rate
        are dropped -- they never become a 0.0 settlement."""
        rows = [
            None,
            "garbage",
            {"symbol": "BTCUSDT"},
            {"fundingTime": None, "fundingRate": "0.5"},
            {"fundingTime": _utc_ms("2024-01-02T00:00:00"), "fundingRate": None},
            {"fundingTime": _utc_ms("2024-01-02T08:00:00"), "fundingRate": "not-a-number"},
            {"fundingTime": _utc_ms("2024-01-02T16:00:00"), "fundingRate": "0.0001"},
        ]
        _fake_pagination(monkeypatch, rows)

        s = eng._binance_funding_provider("BTC-USDT", "2024-01-02", "2024-01-03")

        assert s.to_dict() == pytest.approx({"2024-01-02": 0.0001})

    @pytest.mark.parametrize("rows", [None, "oops", {"fundingTime": 1}])
    def test_r17_funding_provider_non_list_payload_gives_empty_float_series(self, monkeypatch, rows):
        _fake_pagination(monkeypatch, rows)

        s = eng._binance_funding_provider("BTC-USDT", "2024-01-02", "2024-01-03")

        assert isinstance(s, pd.Series)
        assert s.empty
        assert s.dtype == float


# ===========================================================================
# run_backtest() — a hermetic world
# ===========================================================================
def _bdays(start: str, n: int) -> list[str]:
    """``n`` consecutive business-day date strings from ``start``."""
    return [d.strftime("%Y-%m-%d") for d in pd.bdate_range(start, periods=n)]


def _closes(values: Sequence[float], index: Sequence[str]) -> pd.Series:
    """A float close-price Series over the given date-string ``index``."""
    return pd.Series([float(v) for v in values], index=list(index), dtype=float)


def _ramp(index: Sequence[str], start: float = 100.0, step: float = 0.5) -> pd.Series:
    """Closes rising by ``step`` per row: ``start + step * i``."""
    return _closes([start + step * i for i in range(len(index))], index)


class _TapeGraph:
    """Scripted ratings per decision date; records every ``propagate`` call."""

    def __init__(self, ratings: Mapping[str, str], benchmark: str = "SPY") -> None:
        self._ratings = dict(ratings)
        self._benchmark = benchmark
        self.calls: list[tuple[str, str, str]] = []

    def propagate(self, company_name: str, trade_date: str, asset_type: str = "stock"):
        self.calls.append((company_name, trade_date, asset_type))
        rating = self._ratings.get(trade_date, "Hold")
        decision = f"**Rating**: {rating}\n\nAnalysis for {trade_date}."
        return {"final_trade_decision": decision}, rating

    def _resolve_benchmark(self, ticker: str) -> str:
        return self._benchmark


def _world(**series: pd.Series) -> Callable[[str, str, str], pd.Series]:
    """Price provider serving each symbol its own series, clipped to ``[start, end]``.

    A symbol without a series gets an empty float Series (what a vendor returns
    for a name it does not list).
    """

    def provider(ticker: str, start: str, end: str) -> pd.Series:
        s = series.get(str(ticker))
        if s is None:
            return pd.Series(dtype=float)
        return s.loc[(s.index >= start) & (s.index <= end)].copy()

    return provider


def _funding(rates: Mapping[str, float]) -> Callable[[str, str, str], pd.Series]:
    """Funding provider over a fixed ``{date: summed daily rate}`` table."""

    def provider(ticker: str, start: str, end: str) -> pd.Series:
        return pd.Series(list(rates.values()), index=list(rates), dtype=float)

    return provider


#: 2024-01-02 .. 2024-02-26; ``_R17_AAPL`` closes at ``100.0 + 0.5`` per session.
_R17_DAYS = _bdays("2024-01-02", 40)
_R17_AAPL = _ramp(_R17_DAYS)
_R17_SPY = _closes([100.0] * len(_R17_DAYS), _R17_DAYS)
#: 2024-01-02, 2024-01-09, 2024-01-16.
_R17_DATES = [_R17_DAYS[i] for i in (0, 5, 10)]


def _stock(**overrides: Any) -> dict[str, Any]:
    """Keyword set for a hermetic AAPL stock run on the shared tape."""
    kwargs: dict[str, Any] = {
        "holding_days": 5,
        "price_provider": _world(AAPL=_R17_AAPL, SPY=_R17_SPY),
        **overrides,
    }
    return kwargs


def _perp(**overrides: Any) -> dict[str, Any]:
    """Keyword set for a hermetic BTCUSDT perp run on the shared tape."""
    kwargs: dict[str, Any] = {
        "asset_type": "crypto_perp",
        "compute_index_alpha": False,
        "cost_bps": 0.0,
        "holding_days": 5,
        "price_provider": _world(BTCUSDT=_R17_AAPL),
        "funding_provider": _funding(dict.fromkeys(_R17_DAYS, 0.0)),
        **overrides,
    }
    return kwargs


_ENGINE_LOGGER = "yialpha.backtest.engine"


# ---------------------------------------------------------------------------
# run_backtest() — argument gates
# ---------------------------------------------------------------------------
class TestRunBacktestArgumentGates:
    """Every argument gate fires BEFORE the graph is called.

    ``run_backtest`` bills one LLM decision per date (the production graph
    takes minutes per date), so a bad argument that is only rejected after the
    decision loop costs the whole run's token spend.
    """

    @pytest.mark.parametrize("holding_days", [0, -3])
    def test_r17_run_backtest_rejects_holding_days_below_one(self, holding_days):
        graph = _TapeGraph(dict.fromkeys(_R17_DATES, "Buy"))
        with pytest.raises(ValueError, match="holding_days must be >= 1"):
            eng.run_backtest(
                graph, "AAPL", _R17_DATES,
                **_stock(holding_days=holding_days, compute_index_alpha=False),
            )
        assert graph.calls == []

    @pytest.mark.parametrize(
        ("leverage", "shown"),
        [(0.0, "0.0"), (0.5, "0.5"), (-1.0, "-1.0"), (float("nan"), "nan"), (float("inf"), "inf")],
    )
    def test_r17_run_backtest_rejects_non_finite_or_sub_one_leverage(self, leverage, shown):
        graph = _TapeGraph(dict.fromkeys(_R17_DATES, "Buy"))
        with pytest.raises(
            ValueError, match=rf"leverage must be a finite number >= 1\.0, got {shown}$",
        ):
            eng.run_backtest(
                graph, "AAPL", _R17_DATES,
                **_stock(leverage=leverage, compute_index_alpha=False),
            )
        assert graph.calls == []

    @pytest.mark.parametrize(
        "bad", [{"cost_bps": -1.0}, {"taker_bps": -0.5}, {"slippage_bps": -2.0}],
    )
    def test_r17_run_backtest_rejects_negative_costs(self, bad):
        graph = _TapeGraph(dict.fromkeys(_R17_DATES, "Buy"))
        with pytest.raises(ValueError, match="cost_bps / taker_bps / slippage_bps must be"):
            eng.run_backtest(
                graph, "AAPL", _R17_DATES, **_stock(compute_index_alpha=False, **bad),
            )
        assert graph.calls == []

    @pytest.mark.parametrize(
        "bad",
        [
            {"cost_bps": float("nan")},
            {"taker_bps": float("nan")},
            {"slippage_bps": float("nan")},
            {"cost_bps": float("inf")},
            {"taker_bps": float("inf")},
            {"slippage_bps": float("inf")},
        ],
    )
    def test_r17_run_backtest_rejects_non_finite_costs(self, bad):
        """A NaN / inf cost is rejected by the same gate as a negative one.

        ``x < 0.0`` is False for NaN and for +inf alike, so both used to walk
        past the ``must be >= 0`` check; every fill then carried a non-finite
        fee or slippage and the WHOLE equity curve came back NaN
        (``total_return=nan`` beside a plausible ``sharpe=0.0``) with no
        exception.
        """
        graph = _TapeGraph(dict.fromkeys(_R17_DATES, "Buy"))
        with pytest.raises(ValueError, match="cost_bps / taker_bps / slippage_bps must be"):
            eng.run_backtest(
                graph, "AAPL", _R17_DATES, **_stock(compute_index_alpha=False, **bad),
            )
        assert graph.calls == []

    @pytest.mark.parametrize("periods_per_year", [0, -252])
    def test_r17_run_backtest_rejects_non_positive_periods_per_year(self, periods_per_year):
        graph = _TapeGraph(dict.fromkeys(_R17_DATES, "Buy"))
        with pytest.raises(ValueError, match="periods_per_year must be an integer >= 1"):
            eng.run_backtest(
                graph, "AAPL", _R17_DATES,
                **_stock(periods_per_year=periods_per_year, compute_index_alpha=False),
            )
        assert graph.calls == []

    @pytest.mark.parametrize(
        "periods_per_year", [252.0, 2.5, float("nan"), np.int64(365)],
        ids=["whole-float", "fraction", "nan", "numpy-int"],
    )
    def test_r17_run_backtest_rejects_non_integer_periods_per_year_before_any_decision(
        self, periods_per_year
    ):
        """``compute_metrics`` needs a positive ``int``. The engine's own gate
        only tested ``< 1``, which ``252.0`` and NaN both pass, so a float went
        through the WHOLE decision loop -- every date billed -- before
        ``compute_metrics`` raised "must be a positive integer".
        """
        graph = _TapeGraph(dict.fromkeys(_R17_DATES, "Buy"))
        with pytest.raises(ValueError, match="periods_per_year must be an integer >= 1"):
            eng.run_backtest(
                graph, "AAPL", _R17_DATES,
                **_stock(periods_per_year=periods_per_year, compute_index_alpha=False),
            )
        assert graph.calls == []

    @pytest.mark.parametrize("initial_capital", [0.0, -5.0, float("nan"), float("inf")])
    def test_r17_run_backtest_rejects_unusable_initial_capital(self, initial_capital):
        """Capital the simulation cannot start from.

        On this tape ``0`` marked a flat, empty book and reported
        ``total_return=-1.0``; ``-5`` SOLD SHORT in a long-only run (negative
        shares, equity below zero) and reported ``sharpe=68.7``; NaN / inf
        capital returned NaN metrics beside a plausible ``sharpe=0.0``.
        None of them raised.
        """
        graph = _TapeGraph(dict.fromkeys(_R17_DATES, "Buy"))
        with pytest.raises(
            ValueError,
            match=rf"initial_capital must be a finite number > 0, got {initial_capital!r}$",
        ):
            eng.run_backtest(
                graph, "AAPL", _R17_DATES,
                **_stock(initial_capital=initial_capital, compute_index_alpha=False),
            )
        assert graph.calls == []


# ---------------------------------------------------------------------------
# run_backtest() — perp funding gate
# ---------------------------------------------------------------------------
class TestRunBacktestFundingGate:
    """A perp backtest must not run as a spot simulation by accident."""

    def test_r17_run_backtest_funding_fetch_failure_fails_closed_and_chained(self):
        def unreachable(ticker: str, start: str, end: str) -> pd.Series:
            raise ConnectionError("fapi unreachable")

        graph = _TapeGraph(dict.fromkeys(_R17_DATES, "Buy"))
        with pytest.raises(
            ValueError,
            match=(
                r"requires funding-rate history for BTCUSDT; the funding fetch failed "
                r"\(ConnectionError: fapi unreachable\)\. Pass funding_provider with the "
                r"data, or use asset_type='crypto' for an explicit spot-style simulation\.$"
            ),
        ) as exc_info:
            eng.run_backtest(graph, "BTCUSDT", _R17_DATES, **_perp(funding_provider=unreachable))
        # The vendor error stays reachable for the operator (raise ... from).
        assert isinstance(exc_info.value.__cause__, ConnectionError)
        # The gate sits before the decision loop: no decision was billed.
        assert graph.calls == []

    @pytest.mark.parametrize(
        "bad_rate", [float("nan"), float("inf"), -float("inf"), None],
        ids=["nan", "inf", "-inf", "day-absent"],
    )
    def test_r17_run_backtest_unusable_funding_rate_is_missing_coverage(self, bad_rate):
        """A priced day whose summed rate is not a finite number is charged as
        zero by the funding loop -- exactly the silent zero drag an ABSENT day
        would produce -- so the fail-closed coverage gate must treat it the
        same way (``allow_funding_gaps=False`` is the default).

        NaN and +/-inf used to pass the gate untouched: the day was in the
        series, only its value was unusable, and the run reported a strategy
        that paid no funding that day.
        """
        rates = dict.fromkeys(_R17_DAYS, 0.0001)
        bad_day = _R17_DAYS[6]  # 2024-01-10, an interior charge day
        if bad_rate is None:
            del rates[bad_day]
        else:
            rates[bad_day] = bad_rate
        graph = _TapeGraph(dict.fromkeys(_R17_DATES, "Buy"))

        with pytest.raises(
            ValueError, match=rf"is missing 1 priced day\(s\) \({bad_day}\); those days would",
        ):
            eng.run_backtest(graph, "BTCUSDT", _R17_DATES, **_perp(funding_provider=_funding(rates)))

    def test_r17_run_backtest_unusable_funding_rate_with_gaps_allowed_is_zero_drag(self):
        """Opting into the approximation (``allow_funding_gaps=True``) keeps the
        run alive and charges nothing on the unusable day -- pinned exactly
        against the fully-funded twin."""
        clean = dict.fromkeys(_R17_DAYS, 0.0001)
        gappy = {**clean, _R17_DAYS[6]: float("nan")}
        funded = eng.run_backtest(
            _TapeGraph({"2024-01-02": "Buy"}), "BTCUSDT", ["2024-01-02"],
            **_perp(funding_provider=_funding(clean)),
        )
        gappy = eng.run_backtest(
            _TapeGraph({"2024-01-02": "Buy"}), "BTCUSDT", ["2024-01-02"],
            **_perp(funding_provider=_funding(gappy), allow_funding_gaps=True),
        )
        # One Buy on 2024-01-02 fills at the 2024-01-03 close (100.5): the
        # position is never rebalanced, so the only difference between the two
        # runs is the one settlement the gappy run did not pay. A day's charge
        # is taken BEFORE that day is marked, so the curves part at the 01-10
        # mark itself (equity index 6).
        shares = 100_000.0 / 100.5
        unpaid = shares * _R17_AAPL[_R17_DAYS[6]] * 0.0001
        assert gappy.equity_dates == _R17_DAYS[:11]
        assert gappy.equity[:6] == funded.equity[:6]  # through 2024-01-09
        assert gappy.equity[6:] == pytest.approx([e + unpaid for e in funded.equity[6:]])
        assert unpaid == pytest.approx(10.24875621890547)

    def test_r17_run_backtest_unusable_funding_after_the_marking_window_is_not_a_gap(self):
        """Days past ``mark_end`` sit in the forward-window buffer only: no
        charge can ever touch them, so an unusable rate there is no coverage gap."""
        rates = dict.fromkeys(_R17_DAYS, 0.0001)
        rates["2024-02-15"] = float("nan")  # mark_end is 2024-01-31
        result = eng.run_backtest(
            _TapeGraph(dict.fromkeys(_R17_DATES, "Buy")), "BTCUSDT", _R17_DATES,
            **_perp(funding_provider=_funding(rates)),
        )
        assert result.equity_dates[-1] == "2024-01-30"
        assert np.isfinite(result.equity).all()


# ---------------------------------------------------------------------------
# run_backtest() — optional execution-data sources
# ---------------------------------------------------------------------------
class TestRunBacktestOptionalSources:
    """Sources the perp simulation may use degrade LOUDLY, and only ever
    remove an add-on: the money path stays the one documented for their
    absence."""

    def test_r17_run_backtest_filters_failure_falls_back_to_fractional_fills(self, caplog):
        def exchange_info_down(ticker: str):
            raise ConnectionError("exchangeInfo down")

        graph = _TapeGraph(dict.fromkeys(_R17_DATES, "Buy"))
        with caplog.at_level(logging.WARNING, logger=_ENGINE_LOGGER):
            result = eng.run_backtest(
                graph, "BTCUSDT", _R17_DATES, filters_provider=exchange_info_down, **_perp(),
            )
        assert (
            "filters unavailable for BTCUSDT (ConnectionError: exchangeInfo down); "
            "fills stay fractional"
        ) in caplog.text
        assert result.config_summary["perp_fill_quantization"] == "fractional (no filters source)"
        # Exactly the run the engine documents for "no filters source".
        twin = eng.run_backtest(
            _TapeGraph(dict.fromkeys(_R17_DATES, "Buy")), "BTCUSDT", _R17_DATES, **_perp(),
        )
        assert result.equity == twin.equity
        assert result.benchmark_equity == twin.benchmark_equity

    @pytest.mark.parametrize("ladder_available", [True, False], ids=["leverageBracket", "default"])
    def test_r17_run_backtest_mmr_ladder_source_is_named_and_actually_used(
        self, monkeypatch, ladder_available
    ):
        """``mmr_source`` is not just a label: the liquidation level is computed
        from the ladder the label names."""
        from yialpha.dataflows import binance_brackets as bb
        from yialpha.dataflows.errors import NoMarketDataError

        asked: list[str] = []

        def leverage_brackets(symbol: str, *, force_refresh: bool = False):
            asked.append(symbol)
            if not ladder_available:
                raise NoMarketDataError("leverageBracket: no operator keys")
            return [bb.Bracket(0.0, 0.02)]  # one tier, MMR 2 %

        monkeypatch.setattr(bb, "get_leverage_brackets", leverage_brackets)
        days = _bdays("2024-01-02", 8)
        # 3x long filled at 100 on bar 1; the close-only trigger fires once the
        # close reaches 65 (below both candidate levels).
        tape = _closes([100.0, 100.0, 70.0, 65.0, 65.0, 65.0, 65.0, 65.0], days)
        result = eng.run_backtest(
            _TapeGraph({days[0]: "Buy"}), "BTCUSDT", [days[0]], leverage=3.0,
            **_perp(
                price_provider=_world(BTCUSDT=tape),
                funding_provider=_funding(dict.fromkeys(days, 0.0)),
            ),
        )
        # The ladder is read at the trigger bar's notional (3,000 x 65 = 195,000):
        # tier 2 of the documented default, the single tier of the fetched one.
        mmr = 0.02 if ladder_available else 0.006
        liq_price = 100.0 * (1.0 - 1.0 / 3.0 + mmr)
        source = "leverageBracket" if ladder_available else "default-ladder"
        (event,) = result.config_summary["perp_liquidations"]
        assert event["date"] == days[3]
        assert event["liquidation_price"] == pytest.approx(liq_price)
        assert f"isolated-margin liquidation vs the {source} MMR ladder" in (
            result.config_summary["perp_model_note"]
        )
        assert asked == ["BTCUSDT"]
        # cash = 100_000 - 3_000 * 100 + 3_000 * liq_price
        assert result.equity[-1] == pytest.approx(-200_000.0 + 3_000.0 * liq_price)

    def test_r17_run_backtest_injected_extremes_failure_runs_triggers_on_closes(
        self, caplog
    ):
        days = _bdays("2024-01-02", 5)
        closes = [100.0, 100.0, 96.0, 97.0, 98.0]
        tape = _closes(closes, days)
        # 2024-01-04 wicks to 94 (through the 95 stop) but closes at 96.
        lows = _closes([100.0, 100.0, 94.0, 97.0, 98.0], days)
        highs = _closes([100.0, 100.0, 96.5, 97.5, 98.5], days)

        def armed(rating: str, date: str, ctx: dict) -> float:
            ctx["risk_decision"] = SimpleNamespace(stop_loss=95.0, action=None)
            return 1.0

        def extremes_down(ticker: str, start: str, end: str):
            raise ConnectionError("klines down")

        def run(extremes: Callable | None) -> BacktestResult:
            with caplog.at_level(logging.WARNING, logger=_ENGINE_LOGGER):
                return eng.run_backtest(
                    _TapeGraph({days[0]: "Buy"}), "BTCUSDT", [days[0]], weight_fn=armed,
                    extremes_provider=extremes,
                    **_perp(
                        price_provider=_world(BTCUSDT=tape),
                        funding_provider=_funding(dict.fromkeys(days, 0.0)),
                    ),
                )

        blind = run(extremes_down)
        assert (
            "bar extremes unavailable for BTCUSDT (ConnectionError: klines down); "
            "liquidation/stop checks run on closes only"
        ) in caplog.text
        assert "perp_stop_triggers" not in blind.config_summary
        assert blind.metrics.stop_trigger_count == 0
        assert blind.equity[-1] == pytest.approx(98_000.0)  # 1,000 long marked at 98

        sighted = run(lambda ticker, start, end: (lows, highs))
        assert sighted.config_summary["perp_stop_triggers"] == [
            {
                "date": days[2],
                "side": "long",
                "shares": pytest.approx(1_000.0),
                "entry": 100.0,
                "stop_price": 95.0,
                "exit_price": 95.0,
                "fee": 0.0,
            }
        ]
        assert sighted.equity[-1] == pytest.approx(95_000.0)


# ---------------------------------------------------------------------------
# run_backtest() — index benchmark leg
# ---------------------------------------------------------------------------
class TestRunBacktestIndexLeg:
    @staticmethod
    def _provider(spy_raises: BaseException | None, asked: list[str]):
        def provider(ticker: str, start: str, end: str) -> pd.Series:
            asked.append(ticker)
            if ticker == "SPY":
                if spy_raises is not None:
                    raise spy_raises
                return _R17_SPY.loc[(_R17_SPY.index >= start) & (_R17_SPY.index <= end)].copy()
            return _R17_AAPL.loc[(_R17_AAPL.index >= start) & (_R17_AAPL.index <= end)].copy()

        return provider

    def test_r17_run_backtest_index_leg_failure_is_advisory(self, caplog):
        asked: list[str] = []
        provider = self._provider(ConnectionError("vendor down"), asked)
        graph = _TapeGraph(dict.fromkeys(_R17_DATES, "Buy"), benchmark="SPY")
        with caplog.at_level(logging.WARNING, logger=_ENGINE_LOGGER):
            result = eng.run_backtest(graph, "AAPL", _R17_DATES, **_stock(price_provider=provider))
        assert "Could not load index benchmark SPY: vendor down" in caplog.text
        # No benchmark -> no alpha, but the asset's own forward return survives
        # and the run is complete.
        assert result.config_summary["index_benchmark"] == "SPY"
        # Executions at 100.5 / 103.0 / 105.5, five sessions on: 103.0 / 105.5 / 108.0.
        assert [t.raw_return for t in result.trades] == pytest.approx(
            [103.0 / 100.5 - 1.0, 105.5 / 103.0 - 1.0, 108.0 / 105.5 - 1.0]
        )
        assert [t.alpha_vs_index for t in result.trades] == [None, None, None]
        # The index leg is advisory: with it gone the money path is the run
        # that never asked for it.
        twin = eng.run_backtest(
            _TapeGraph(dict.fromkeys(_R17_DATES, "Buy"), benchmark="SPY"), "AAPL", _R17_DATES,
            **_stock(price_provider=provider, compute_index_alpha=False),
        )
        assert result.equity == twin.equity
        assert result.benchmark_equity == twin.benchmark_equity
        assert result.metrics.total_return == twin.metrics.total_return

    def test_r17_run_backtest_alpha_without_index_leg_fetches_nothing(self):
        asked: list[str] = []
        provider = self._provider(None, asked)
        result = eng.run_backtest(
            _TapeGraph(dict.fromkeys(_R17_DATES, "Buy")), "AAPL", _R17_DATES,
            **_stock(price_provider=provider, compute_index_alpha=False),
        )
        assert "SPY" not in asked
        assert result.config_summary["index_benchmark"] == ""
        assert [t.alpha_vs_index for t in result.trades] == [None, None, None]

    @pytest.mark.parametrize("interrupt", [KeyboardInterrupt, SystemExit])
    def test_r17_run_backtest_index_leg_does_not_swallow_interrupts(self, interrupt):
        """Advisory means ``except Exception``: an operator interrupt must still
        stop the backtest instead of being logged away as a missing benchmark."""
        provider = self._provider(interrupt("operator stop"), [])
        with pytest.raises(interrupt, match="operator stop"):
            eng.run_backtest(
                _TapeGraph(dict.fromkeys(_R17_DATES, "Buy")), "AAPL", _R17_DATES,
                **_stock(price_provider=provider),
            )


# ---------------------------------------------------------------------------
# run_backtest() — marking-window gates
# ---------------------------------------------------------------------------
class TestRunBacktestMarkingWindow:
    @pytest.mark.parametrize(
        ("value", "shown"),
        [(0.0, "0.0"), (-1.0, "-1.0"), (float("nan"), "nan"), (float("inf"), "inf")],
    )
    def test_r17_run_backtest_rejects_unusable_price_inside_the_window(self, value, shown):
        """A close the simulator cannot mark or fill against (zero, negative,
        NaN, inf) stops the backtest instead of entering the equity curve."""
        tape = _R17_AAPL.copy()
        tape[_R17_DAYS[6]] = value
        with pytest.raises(
            ValueError, match=rf"Invalid execution price for AAPL on {_R17_DAYS[6]}: {shown}$",
        ):
            eng.run_backtest(
                _TapeGraph(dict.fromkeys(_R17_DATES, "Buy")), "AAPL", _R17_DATES,
                **_stock(price_provider=_world(AAPL=tape), compute_index_alpha=False),
            )

    def test_r17_run_backtest_price_gate_bounds_the_marking_window_only(self):
        """Bars the simulator never marks on are not execution prices: one
        before the first decision and ones after ``mark_end`` (kept only so
        the last decisions' forward windows are complete) may be unusable."""
        late = _R17_AAPL.copy()
        late["2024-02-15"] = float("nan")  # mark_end is 2024-01-31
        result = eng.run_backtest(
            _TapeGraph(dict.fromkeys(_R17_DATES, "Buy")), "AAPL", _R17_DATES,
            **_stock(price_provider=_world(AAPL=late), compute_index_alpha=False),
        )
        assert result.equity_dates == _R17_DAYS[:21]
        assert np.isfinite(result.equity).all()

        early = _R17_AAPL.copy()
        early[_R17_DAYS[2]] = float("inf")  # three sessions before the first decision
        result = eng.run_backtest(
            _TapeGraph({_R17_DAYS[5]: "Buy"}), "AAPL", [_R17_DAYS[5]],
            **_stock(price_provider=_world(AAPL=early), compute_index_alpha=False),
        )
        # mark_end = 2024-01-09 + 15 days: bars 01-09 .. 01-23.
        assert result.equity_dates == _R17_DAYS[5:16]
        assert result.equity[0] == 100_000.0  # cash until the 01-10 fill

    def test_r17_run_backtest_single_marked_bar_fails_closed(self):
        with pytest.raises(
            ValueError,
            match="Backtest produced fewer than 2 equity points for AAPL; need a wider date window.",
        ):
            eng.run_backtest(
                _TapeGraph({"2024-01-02": "Buy"}), "AAPL", ["2024-01-02"],
                **_stock(price_provider=_world(AAPL=_closes([100.0], ["2024-01-02"]))),
            )

    def test_r17_run_backtest_first_decision_after_the_last_bar_fails_closed(self):
        """Prices exist (so the 'no price data' gate passes) but every bar predates
        the decision date: nothing is ever marked."""
        days = _bdays("2024-01-02", 3)
        early_tape = _closes([100.0, 101.0, 102.0], days)

        def window_agnostic(ticker: str, start: str, end: str) -> pd.Series:
            return early_tape.copy()

        with pytest.raises(
            ValueError, match="Backtest produced fewer than 2 equity points for AAPL",
        ):
            eng.run_backtest(
                _TapeGraph({"2024-01-08": "Buy"}), "AAPL", ["2024-01-08"],
                **_stock(price_provider=window_agnostic),
            )

    def test_r17_run_backtest_exactly_two_marked_bars_run(self):
        days = _bdays("2024-01-02", 2)
        result = eng.run_backtest(
            _TapeGraph({days[0]: "Buy"}), "AAPL", [days[0]],
            **_stock(price_provider=_world(AAPL=_closes([100.0, 101.0], days))),
        )
        assert result.equity_dates == days
        # Cash on the signal bar; the fill lands at the 01-03 close itself.
        assert result.equity == [100_000.0, 100_000.0]
        assert result.metrics.num_periods == 1
        assert result.trades[0].execution_date == days[1]

    def test_r17_run_backtest_no_price_data_fails_closed_before_funding_or_decisions(self):
        funding_calls: list[tuple[str, str, str]] = []

        def funding(ticker: str, start: str, end: str) -> pd.Series:
            funding_calls.append((ticker, start, end))
            return _funding(dict.fromkeys(_R17_DAYS, 0.0))(ticker, start, end)

        graph = _TapeGraph(dict.fromkeys(_R17_DATES, "Buy"))
        with pytest.raises(
            ValueError,
            match=(
                "No price data for BTCUSDT between 2024-01-02 and 2024-02-02; "
                "cannot mark the backtest to market"
            ),
        ):
            eng.run_backtest(
                graph, "BTCUSDT", _R17_DATES,
                **_perp(price_provider=_world(), funding_provider=funding),
            )
        assert graph.calls == []
        assert funding_calls == []


class TestRunBacktestProgressLog:
    def test_r17_run_backtest_progress_logs_one_line_per_executed_decision(self, caplog):
        graph = _TapeGraph(dict.fromkeys(_R17_DATES, "Buy"))
        with caplog.at_level(logging.INFO, logger=_ENGINE_LOGGER):
            eng.run_backtest(
                graph, "AAPL", _R17_DATES, progress=True,
                **_stock(compute_index_alpha=False),
            )
        lines = [r.getMessage() for r in caplog.records if r.levelno == logging.INFO]
        # 1,000-ish shares bought at the 01-03 close (100.5) and then carried:
        # the later Buy decisions are no-order holds at 100 % weight.
        assert lines == [
            "2024-01-02 signal 2024-01-03 AAPL: rating=Buy weight=1.00 "
            "notional=100000.00 equity=100000",
            "2024-01-09 signal 2024-01-10 AAPL: rating=Buy weight=1.00 "
            "notional=0.00 equity=102488",
            "2024-01-16 signal 2024-01-17 AAPL: rating=Buy weight=1.00 "
            "notional=0.00 equity=104975",
        ]

    def test_r17_run_backtest_progress_off_logs_nothing(self, caplog):
        with caplog.at_level(logging.INFO, logger=_ENGINE_LOGGER):
            eng.run_backtest(
                _TapeGraph(dict.fromkeys(_R17_DATES, "Buy")), "AAPL", _R17_DATES,
                **_stock(compute_index_alpha=False),
            )
        assert not [r.getMessage() for r in caplog.records if " signal " in r.getMessage()]


# ---------------------------------------------------------------------------
# run_backtest() — GTC stop across a direction flip
# ---------------------------------------------------------------------------
def _stop_tape(closes: Sequence[float]) -> tuple[list[str], pd.Series]:
    days = _bdays("2024-01-02", len(closes))
    return days, _closes(closes, days)


class TestRunBacktestStopAcrossFlip:
    """A resting stop belongs to the position direction it was armed for.

    Flipping the position on a decision that supplies no stop must disarm the
    old level: a long's stop sits BELOW the entry, so left resting on the new
    short it is a profit level the very next bar satisfies.
    """

    def test_r17_run_backtest_long_stop_disarmed_when_the_next_decision_flips_short(self):
        days, tape = _stop_tape([100.0, 100.0, 99.0, 97.0, 96.0, 97.0, 98.0, 99.0])

        def sizer(rating: str, date: str, ctx: dict) -> float:
            if date == days[0]:
                ctx["risk_decision"] = SimpleNamespace(stop_loss=95.0, action=None)
                return 1.0
            return -1.0  # flip to short, no stop supplied

        result = eng.run_backtest(
            _TapeGraph({days[0]: "Buy", days[3]: "Buy"}), "BTCUSDT", [days[0], days[3]],
            weight_fn=sizer, allow_short=True,
            **_perp(
                price_provider=_world(BTCUSDT=tape),
                funding_provider=_funding(dict.fromkeys(days, 0.0)),
            ),
        )
        first, second = result.trades
        assert (first.execution_date, first.stop_loss) == (days[1], 95.0)
        assert (second.execution_date, second.stop_loss) == (days[4], None)
        assert second.executed_weight == pytest.approx(-1.0)
        assert "perp_stop_triggers" not in result.config_summary
        # Long leg 100 -> 96; short leg 96 -> 99 (1,000 contracts each, no costs).
        assert first.position_return == pytest.approx(96_000.0 / 100_000.0 - 1.0)
        assert second.position_return == pytest.approx(93_000.0 / 96_000.0 - 1.0)
        assert result.metrics.num_trades == 2
        assert result.equity[-1] == pytest.approx(93_000.0)

    def test_r17_run_backtest_short_stop_disarmed_when_the_next_decision_flips_long(self):
        days, tape = _stop_tape([100.0, 100.0, 101.0, 103.0, 104.0, 103.0, 102.0, 103.0])

        def sizer(rating: str, date: str, ctx: dict) -> float:
            if date == days[0]:
                ctx["risk_decision"] = SimpleNamespace(stop_loss=105.0, action=None)
                return -1.0
            return 1.0  # flip to long, no stop supplied

        result = eng.run_backtest(
            _TapeGraph({days[0]: "Sell", days[3]: "Sell"}), "BTCUSDT", [days[0], days[3]],
            weight_fn=sizer, allow_short=True,
            **_perp(
                price_provider=_world(BTCUSDT=tape),
                funding_provider=_funding(dict.fromkeys(days, 0.0)),
            ),
        )
        first, second = result.trades
        assert (first.execution_date, first.stop_loss) == (days[1], 105.0)
        assert (second.execution_date, second.stop_loss) == (days[4], None)
        assert "perp_stop_triggers" not in result.config_summary
        # Short leg 100 -> 104 (the stop at 105 was never touched: closes only).
        assert first.position_return == pytest.approx(96_000.0 / 100_000.0 - 1.0)
        # Long leg: 200,000 / 104 contracts bought at 104, marked at 103.
        assert second.position_return == pytest.approx(103.0 / 104.0 - 1.0)
        assert result.equity[-1] == pytest.approx((200_000.0 / 104.0 - 1_000.0) * 103.0)

    def test_r17_run_backtest_stop_keeps_resting_across_a_same_direction_resize(self):
        """GTC: a later decision that resizes the position the same way and
        supplies no stop leaves the resting level in force."""
        days, tape = _stop_tape([100.0, 100.0, 99.0, 99.0, 99.0, 94.0, 94.0, 94.0])

        def sizer(rating: str, date: str, ctx: dict) -> float:
            if date == days[0]:
                ctx["risk_decision"] = SimpleNamespace(stop_loss=95.0, action=None)
                return 1.0
            return 0.5  # trim to half, no stop supplied

        result = eng.run_backtest(
            _TapeGraph({days[0]: "Buy", days[3]: "Buy"}), "BTCUSDT", [days[0], days[3]],
            weight_fn=sizer,
            **_perp(
                price_provider=_world(BTCUSDT=tape),
                funding_provider=_funding(dict.fromkeys(days, 0.0)),
            ),
        )
        (trigger,) = result.config_summary["perp_stop_triggers"]
        assert trigger["date"] == days[5]
        assert trigger["stop_price"] == 95.0
        assert trigger["shares"] == pytest.approx(500.0)  # the trimmed position
        assert trigger["exit_price"] == 95.0
        # 49,500 cash banked by the trim + the 500 remaining contracts at 95.
        assert result.equity[-1] == pytest.approx(49_500.0 + 500.0 * 95.0)


# ---------------------------------------------------------------------------
# run_backtest() — the sizing seam
# ---------------------------------------------------------------------------
class TestRunBacktestSizingSeam:
    def test_r17_run_backtest_non_finite_weight_is_rejected_not_poisoning_equity(self):
        """``weight_fn`` may return ``None`` (hold) or a real number.

        ``np.clip`` passes NaN through, so a NaN target used to flow into the
        share count, the cash leg and every later equity point: the run came
        back without an exception carrying ``total_return=nan``, ``alpha=nan``
        and -- because a NaN std fails the ``std >= floor`` comparison -- a
        plausible ``sharpe=0.0``.
        """

        def broken_sizing(rating: str, date: str, ctx: dict) -> float:
            return float("nan")  # e.g. a volatility target divided by a zero vol

        with pytest.raises(
            ValueError, match=r"weight_fn returned nan for AAPL on signal 2024-01-02",
        ):
            eng.run_backtest(
                _TapeGraph(dict.fromkeys(_R17_DATES, "Buy")), "AAPL", _R17_DATES,
                weight_fn=broken_sizing, **_stock(compute_index_alpha=False),
            )

    @pytest.mark.parametrize(("requested", "weight"), [(2.5, 1.0), (float("inf"), 1.0), (-0.3, 0.0)])
    def test_r17_run_backtest_real_weights_are_clipped_to_the_long_band(
        self, requested, weight
    ):
        """Control case for the gate above: +/-inf is a real number the clip
        handles, and out-of-band weights clamp instead of raising."""
        result = eng.run_backtest(
            _TapeGraph(dict.fromkeys(_R17_DATES, "Buy")), "AAPL", _R17_DATES,
            weight_fn=lambda rating, date, ctx: requested, **_stock(compute_index_alpha=False),
        )
        assert all(t.executed_weight == pytest.approx(weight) for t in result.trades)


# ===========================================================================
# _attribute_factors — Fama-French attribution
# ===========================================================================
_FF_ALPHA = 0.001  # per-session intercept of the synthetic strategies below
_FF_LOADINGS = {"Mkt-RF": 1.5, "SMB": 0.5, "HML": 0.0}
_FF5_LOADINGS = {**_FF_LOADINGS, "RMW": 0.25, "CMA": -0.1}


def _ff_factors(index: Sequence[str], five_factor: bool = False) -> pd.DataFrame:
    """Daily French factors in decimals over ``index``.

    Every factor moves day by day (``RF`` included) so the design matrix has
    full rank and an exact fit has exactly one solution.
    """
    rng = np.random.default_rng(7)
    n = len(index)
    columns = {
        "Mkt-RF": rng.normal(0.0004, 0.008, n),
        "SMB": rng.normal(0.0001, 0.005, n),
        "HML": rng.normal(0.0, 0.005, n),
    }
    if five_factor:
        columns["RMW"] = rng.normal(0.0, 0.004, n)
        columns["CMA"] = rng.normal(0.0, 0.004, n)
    columns["RF"] = 0.00004 + 0.00001 * np.arange(n)
    return pd.DataFrame(columns, index=pd.to_datetime(list(index)))


def _ff_excess(row: pd.Series, loadings: Mapping[str, float]) -> float:
    """Strategy excess return built from one factor row: ``alpha + loadings . f``."""
    return _FF_ALPHA + sum(float(row[name]) * weight for name, weight in loadings.items())


def _ff_strategy_tape(days: Sequence[str], factors: pd.DataFrame, loadings) -> pd.Series:
    """Closes whose return on every factor date is ``RF + alpha + loadings . f``."""
    closes = [100.0]
    for day in days[1:]:
        stamp = pd.Timestamp(day)
        gross = float(factors.loc[stamp, "RF"]) + _ff_excess(factors.loc[stamp], loadings)
        closes.append(closes[-1] * (1.0 + gross))
    return _closes(closes, days)


def _patch_factor_loader(monkeypatch: pytest.MonkeyPatch, behaviour) -> list[tuple]:
    """Replace ``load_factor_returns``; ``behaviour`` is a frame, ``None`` or an
    exception to raise. Returns the call log ``(start, end, model)``."""

    calls: list[tuple] = []

    def load(start: str, end: str, model: str = "3", **_kwargs):
        calls.append((start, end, model))
        if isinstance(behaviour, BaseException):
            raise behaviour
        return behaviour

    monkeypatch.setattr(fm, "load_factor_returns", load)
    return calls


#: The metric fields attribution owns; every other field must survive it untouched.
_FACTOR_FIELDS = ("factor_model", "factor_alpha", "factor_betas", "factor_r_squared")


def _factor_fields(metrics) -> tuple:
    return tuple(getattr(metrics, name) for name in _FACTOR_FIELDS)


class TestAttributeFactors:
    def test_r17_factor_attribution_stamps_the_exact_ols_fit_in_place(self, monkeypatch):

        days = _bdays("2024-01-02", 40)
        factors = _ff_factors(days[1:])
        returns = [float(row["RF"]) + _ff_excess(row, _FF_LOADINGS) for _, row in factors.iterrows()]
        equity = [100_000.0]
        for gross in returns:
            equity.append(equity[-1] * (1.0 + gross))
        reference = compute_metrics(equity)
        metrics = compute_metrics(equity)

        calls = _patch_factor_loader(monkeypatch, factors)
        eng._attribute_factors(metrics, equity, days, days[0], days[-1], "3", 252)

        # alpha_annual = per-session intercept x (joined obs per year over the
        # join's own span), NOT x the caller's 252.
        span_days = (pd.Timestamp(days[-1]) - pd.Timestamp(days[1])).days
        joined_ppy = len(returns) / (span_days / 365.25)
        assert metrics.factor_model == "FF3"
        assert metrics.factor_betas == pytest.approx(_FF_LOADINGS, abs=1e-9)
        assert metrics.factor_r_squared == pytest.approx(1.0, abs=1e-9)
        assert metrics.factor_alpha == pytest.approx(_FF_ALPHA * joined_ppy, rel=1e-9)
        assert span_days == 54  # 2024-01-03 .. 2024-02-26
        assert joined_ppy == pytest.approx(39 / (54 / 365.25))
        # The window handed to the loader is the run's fetch window verbatim.
        assert calls == [(days[0], days[-1], "3")]
        # Stamped in place, and nothing but the four factor fields moved.
        moved = {k: v for k, v in vars(metrics).items() if k not in _FACTOR_FIELDS}
        assert moved == {k: v for k, v in vars(reference).items() if k not in _FACTOR_FIELDS}

    @pytest.mark.parametrize("behaviour", [None, pd.DataFrame()], ids=["none", "empty-frame"])
    def test_r17_factor_attribution_without_factor_data_stamps_nothing(self, monkeypatch, behaviour):

        metrics = compute_metrics([100_000.0, 101_000.0, 102_000.0])
        _patch_factor_loader(monkeypatch, behaviour)
        eng._attribute_factors(
            metrics, [100_000.0, 101_000.0, 102_000.0], _R17_DAYS[:3], _R17_DAYS[0], _R17_DAYS[-1],
            "3", 252,
        )
        assert _factor_fields(metrics) == (None, None, None, None)

    def test_r17_factor_attribution_with_too_few_aligned_days_stamps_nothing(self, monkeypatch):
        """``factor_attribution`` returns ``None`` below ``_MIN_OBS``: the run
        must then carry NO attribution, not a model label around a None alpha."""

        days = _bdays("2024-01-02", 40)
        metrics = compute_metrics([100_000.0 * 1.001 ** i for i in range(40)])
        _patch_factor_loader(monkeypatch, _ff_factors(days[30:33]))  # 3 aligned days
        eng._attribute_factors(
            metrics, [100_000.0 * 1.001 ** i for i in range(40)], days, days[0], days[-1], "3", 252,
        )
        assert _factor_fields(metrics) == (None, None, None, None)

    def test_r17_factor_attribution_on_a_single_return_stamps_nothing(self, monkeypatch):
        """Two equity points give ONE return: too thin for any fit, so the
        loader is consulted but no model label is stamped."""
        metrics = compute_metrics([100_000.0, 101_000.0])
        calls = _patch_factor_loader(monkeypatch, _ff_factors(_R17_DAYS[1:3]))
        eng._attribute_factors(
            metrics, [100_000.0, 101_000.0], _R17_DAYS[:2], _R17_DAYS[0], _R17_DAYS[1], "3", 252,
        )
        assert calls == [(_R17_DAYS[0], _R17_DAYS[1], "3")]
        assert _factor_fields(metrics) == (None, None, None, None)


class TestRunBacktestFactorAttribution:
    """``factor_model=...`` opts a run into attribution: pure post-processing
    of the equity curve that fills four metric fields and changes nothing else."""

    @pytest.mark.parametrize(
        ("model", "label", "loadings"),
        [("3", "FF3", _FF_LOADINGS), ("5", "FF5", _FF5_LOADINGS)],
    )
    def test_r17_run_backtest_factor_attribution_recovers_the_exact_fit(
        self, monkeypatch, model, label, loadings
    ):
        days = _bdays("2024-01-02", 40)
        factors = _ff_factors(days[1:], five_factor=model == "5")
        tape = _ff_strategy_tape(days, factors, loadings)
        # The strategy's return on the 01-03 bar is 0 (it buys at that close),
        # so a factor file that simply lacks that one date lines up exactly
        # with every return the strategy actually earned.
        calls = _patch_factor_loader(monkeypatch, factors.loc[pd.Timestamp(days[2]):])
        graph = _TapeGraph({days[0]: "Buy"})
        result = eng.run_backtest(
            graph, "AAPL", [days[0]], factor_model=model,
            **_stock(
                holding_days=20,
                price_provider=_world(AAPL=tape, SPY=tape),
                compute_index_alpha=False,
            ),
        )
        # holding_days=20 -> fetch end = last decision + ceil(20*7/5)+10 days.
        assert calls == [(days[0], "2024-02-09", model)]
        joined = days[2:22]
        span_days = (pd.Timestamp(joined[-1]) - pd.Timestamp(joined[0])).days
        joined_ppy = len(joined) / (span_days / 365.25)
        m = result.metrics
        assert m.factor_model == label
        assert m.factor_betas == pytest.approx(loadings, abs=1e-9)
        assert m.factor_r_squared == pytest.approx(1.0, abs=1e-9)
        assert m.factor_alpha == pytest.approx(_FF_ALPHA * joined_ppy, rel=1e-9)

    def test_r17_run_backtest_factor_attribution_leaves_every_other_metric_unchanged(
        self, monkeypatch
    ):
        days = _bdays("2024-01-02", 40)
        factors = _ff_factors(days[1:])
        tape = _ff_strategy_tape(days, factors, _FF_LOADINGS)
        calls = _patch_factor_loader(monkeypatch, factors.loc[pd.Timestamp(days[2]):])
        kwargs = _stock(price_provider=_world(AAPL=tape, SPY=tape), compute_index_alpha=False)
        attributed = eng.run_backtest(
            _TapeGraph(dict.fromkeys(_R17_DATES, "Buy")), "AAPL", _R17_DATES,
            factor_model="3", **kwargs,
        )
        plain = eng.run_backtest(
            _TapeGraph(dict.fromkeys(_R17_DATES, "Buy")), "AAPL", _R17_DATES, **kwargs,
        )
        # Three decisions, the last on 2024-01-16: fetch end = + ceil(5*7/5)+10 days.
        assert calls == [(days[0], "2024-02-02", "3")]
        assert attributed.equity == plain.equity
        moved = {k: v for k, v in vars(attributed.metrics).items() if k not in _FACTOR_FIELDS}
        assert moved == {k: v for k, v in vars(plain.metrics).items() if k not in _FACTOR_FIELDS}
        assert _factor_fields(plain.metrics) == (None, None, None, None)
        assert attributed.metrics.factor_model == "FF3"

    def test_r17_run_backtest_factor_loader_failure_is_advisory_and_logged(
        self, monkeypatch, caplog
    ):
        _patch_factor_loader(monkeypatch, OSError("factor zip unreadable"))
        kwargs = _stock(compute_index_alpha=False)
        with caplog.at_level(logging.WARNING, logger=_ENGINE_LOGGER):
            result = eng.run_backtest(
                _TapeGraph(dict.fromkeys(_R17_DATES, "Buy")), "AAPL", _R17_DATES,
                factor_model="3", **kwargs,
            )
        assert (
            "factor attribution failed for 2024-01-02..2024-02-02 (3): factor zip unreadable"
        ) in caplog.text
        plain = eng.run_backtest(
            _TapeGraph(dict.fromkeys(_R17_DATES, "Buy")), "AAPL", _R17_DATES, **kwargs
        )
        assert result.equity == plain.equity
        assert vars(result.metrics) == vars(plain.metrics)


# ===========================================================================
# _run_event_study — market-model event study
# ===========================================================================
_EVENT_LOGGER = "yialpha.backtest.event_study"


def _trade(signal_date: str, execution_date: str, rating: str = "Buy") -> TradeRow:
    return TradeRow(
        date=signal_date, rating=rating, target_weight=1.0, executed_weight=1.0,
        price=100.0, execution_date=execution_date,
    )


def _event_world() -> tuple[list[str], pd.Series, pd.DataFrame]:
    """A two-event world for ``_run_event_study``'s window bookkeeping."""
    trades = [_trade("2024-06-03", "2024-06-04"), _trade("2024-06-10", "2024-06-11", "Sell")]
    asked: list[tuple[str, str, str, str]] = []

    def asset(ticker: str, start: str, end: str) -> pd.Series:
        asked.append(("asset", ticker, start, end))
        return _ramp(_bdays("2023-01-02", 500))

    def bench(ticker: str, start: str, end: str) -> pd.Series:
        asked.append(("bench", ticker, start, end))
        return _closes([100.0] * 500, _bdays("2023-01-02", 500))

    return trades, asked, (asset, bench)


class TestRunEventStudyHelper:
    def test_r17_event_study_without_trades_touches_no_provider(self):
        def forbidden(*args: Any, **kwargs: Any) -> None:
            raise AssertionError("no price may be pulled when there is no event")

        metrics = SimpleNamespace()
        eng._run_event_study(
            metrics, None, "BTCUSDT", forbidden, [], index_name="SPY",
            end_date="2024-07-01", bench_provider=forbidden,
        )
        assert vars(metrics) == {}

    def test_r17_event_study_empty_benchmark_leaves_metrics_untouched(self):
        trades, asked, (asset, _full_bench) = _event_world()

        def no_benchmark(ticker: str, start: str, end: str) -> pd.Series:
            asked.append(("bench", ticker, start, end))
            return pd.Series(dtype=float)

        metrics = SimpleNamespace()
        eng._run_event_study(
            metrics, None, "BTCUSDT", asset, trades, index_name="SPY",
            end_date="2024-07-01", bench_provider=no_benchmark,
        )
        assert vars(metrics) == {}
        # Both legs are pulled over the SAME wide window, which starts 400
        # calendar days before the FIRST EVENT'S EXECUTION bar (2024-06-04) --
        # not before the signal bar and not before the last event.
        wide_start = "2023-05-01"
        assert asked == [
            ("asset", "BTCUSDT", wide_start, "2024-07-01"),
            ("bench", "SPY", wide_start, "2024-07-01"),
        ]

    @pytest.mark.parametrize("broken", ["asset-pull", "event-study"])
    def test_r17_event_study_failure_is_advisory_and_logged(self, monkeypatch, caplog, broken):

        def blew_up(*args: Any, **kwargs: Any) -> None:
            raise RuntimeError("lstsq blew up")

        trades, _asked, (asset, bench) = _event_world()
        if broken == "event-study":
            monkeypatch.setattr(es_mod, "event_study", blew_up)
        metrics = SimpleNamespace()
        with caplog.at_level(logging.WARNING, logger=_ENGINE_LOGGER):
            eng._run_event_study(
                metrics, None, "BTCUSDT", blew_up if broken == "asset-pull" else asset, trades,
                index_name="SPY", end_date="2024-07-01", bench_provider=bench,
            )
        assert "event study failed for BTCUSDT: lstsq blew up" in caplog.text
        assert vars(metrics) == {}


class TestRunBacktestEventStudy:
    def test_r17_run_backtest_event_study_window_follows_holding_days_and_anchors_on_execution(
        self, monkeypatch
    ):
        """The CAR window the engine asks for is ``(0, holding_days)`` -- the
        module's own ``(0, 20)`` default would measure a 21-session drift for
        a 3-session strategy -- and its events are the EXECUTION bars, the
        first bar the decision could actually be filled on."""

        seen: dict[str, Any] = {}

        def spy(asset_prices, benchmark_prices, event_dates, **kwargs):
            seen.update(kwargs, event_dates=list(event_dates))
            return SimpleNamespace(n_events=0)

        monkeypatch.setattr(es_mod, "event_study", spy)
        result = eng.run_backtest(
            _TapeGraph(dict.fromkeys(_R17_DATES, "Buy")), "AAPL", _R17_DATES,
            event_study=True, **_stock(holding_days=3, compute_index_alpha=False),
        )
        executions = [t.execution_date for t in result.trades]
        assert executions == ["2024-01-03", "2024-01-10", "2024-01-17"]
        assert seen["event_dates"] == executions  # never the signal bars
        assert seen["event_window"] == (0, 3)
        assert seen["ratings"] == ["Buy", "Buy", "Buy"]
        # The benchmark is resolved for the study although index alpha was off.
        assert seen["benchmark_name"] == "SPY"
        assert seen["rng_seed"] == eng._event_study_seed("AAPL", executions)  # replay-stable
        # Nothing decidable -> nothing stamped.
        assert result.metrics.event_study_n == 0
        assert result.metrics.event_study_window is None

    @pytest.mark.parametrize("holding_days", [3, 5])
    def test_r17_run_backtest_event_study_reports_the_holding_day_car_window(self, holding_days):
        """End to end with the real study: the disclosed window is
        ``(0, holding_days)`` and the mean CAR is exactly what that window
        gives. The module's end offset is INCLUSIVE, so the CAR integrates
        ``holding_days + 1`` returns: the execution bar's own (signal close ->
        execution close, already complete at the fill) plus the ``holding_days``
        the strategy sits through. The model is fitted on the 250 returns
        ending 10 sessions before the event (hand-built here, not via the engine)."""
        long_days = _bdays("2023-01-02", 400)
        closes = [100.0 + 0.1 * i for i in range(len(long_days))]
        world = _world(
            AAPL=_closes(closes, long_days),
            SPY=_closes([100.0] * len(long_days), long_days),
        )
        result = eng.run_backtest(
            _TapeGraph(dict.fromkeys(_R17_DATES, "Buy")), "AAPL", _R17_DATES,
            event_study=True,
            **_stock(holding_days=holding_days, price_provider=world, compute_index_alpha=False),
        )
        m = result.metrics
        assert (m.event_study_window, m.event_study_n, m.event_study_benchmark) == (
            (0, holding_days), 3, "SPY",
        )
        rets = [closes[t] / closes[t - 1] - 1.0 for t in range(1, len(closes))]
        expected = []
        for execution_date in [t.execution_date for t in result.trades]:
            ret_idx = long_days.index(execution_date) - 1  # the execution bar's own return
            estimation = rets[ret_idx - 260 : ret_idx - 10]
            window = rets[ret_idx : ret_idx + holding_days + 1]
            assert (len(estimation), len(window)) == (250, holding_days + 1)
            # A flat benchmark leaves beta unidentified at 0 and alpha at the mean.
            expected.append(sum(r - sum(estimation) / 250 for r in window))
        assert m.event_study_mean_car == pytest.approx(sum(expected) / 3, abs=1e-15)
        assert f"CAR window: [0, +{holding_days}]" in rep.render_backtest_report(result)


# ---------------------------------------------------------------------------
# event_study.event_study — what the estimation window may see
# ---------------------------------------------------------------------------
#: Price index of the event bar in ``_market_model_world`` (return index 279).
_EVENT_POS = 280


def _divide_by_zero(expected: bool):
    """A zero close divides to inf and numpy says so (RuntimeWarning); every
    other unusable close fails silently."""
    if expected:
        return pytest.warns(RuntimeWarning, match="divide by zero")
    return contextlib.nullcontext()


def _market_model_world(
    n: int = 320, event_shock: float = 0.0, bench_shock: float = 0.0
) -> tuple[list[str], pd.Series, pd.Series]:
    """Asset = ``0.05 % + 1.2 x benchmark`` every day; benchmark alternates.

    A shock is added to ONE return -- the event bar's -- so the level shifts
    for good (a single shocked close would just spike and revert).
    """
    days = _bdays("2023-01-02", n)
    bench_ret = np.array([0.004 if i % 2 else -0.002 for i in range(n)])
    asset_ret = 0.0005 + 1.2 * bench_ret
    bench_ret[_EVENT_POS] += bench_shock
    asset_ret[_EVENT_POS] += event_shock
    return (
        days,
        _closes(50.0 * np.cumprod(1.0 + asset_ret), days),
        _closes(100.0 * np.cumprod(1.0 + bench_ret), days),
    )


class TestEventStudyEstimationWindow:
    @pytest.mark.parametrize(
        ("event_shock", "bench_shock", "expected_car"),
        [(0.05, 0.0, 0.05), (0.10, 0.0, 0.10), (0.0, 0.20, -0.24), (0.0, 0.0, 0.0)],
    )
    def test_r17_event_study_market_model_never_sees_the_event_window(
        self, event_shock, bench_shock, expected_car
    ):
        """The market model is fitted on the 250 returns ending 10 sessions
        BEFORE the event: whatever the asset or the benchmark does on the event
        bar moves the CAR and nothing else. (If the event day leaked into the
        fit, beta would absorb the shock and differ between these cases.)"""

        days, asset, bench = _market_model_world(event_shock=event_shock, bench_shock=bench_shock)
        result = es_mod.event_study(
            asset, bench, [days[_EVENT_POS]], event_window=(0, 3), n_bootstrap=200,
        )
        (event,) = result.events
        assert event.event_date == days[_EVENT_POS]
        assert (event.n_estimation, event.n_event) == (250, 4)
        assert (event.alpha, event.beta) == pytest.approx((0.0005, 1.2), abs=1e-9)
        assert event.car == pytest.approx(expected_car, abs=1e-12)
        assert result.mean_car == pytest.approx(expected_car, abs=1e-12)

    @pytest.mark.parametrize(
        ("bad_close", "bad_pos"),
        [
            (float("nan"), 150),
            (float("inf"), 150),
            (0.0, 150),
            (float("nan"), _EVENT_POS + 1),
            (0.0, _EVENT_POS + 2),
        ],
        ids=["nan-in-estimation", "inf-in-estimation", "zero-in-estimation",
             "nan-in-event-window", "zero-in-event-window"],
    )
    def test_r17_event_study_skips_an_event_with_an_unusable_return_in_its_windows(
        self, caplog, bad_close, bad_pos
    ):
        """A NaN / inf close, or a zero one (the return divides to inf), makes
        the fit or the CAR NaN. That event cannot be decided: it must be
        skipped like one whose windows run off the tape -- with a warning.

        It used to be COUNTED (``n_events=1``) carrying ``car=nan``, so the
        aggregate came back ``mean_car=nan`` with every statistic gone.
        """

        days, asset, bench = _market_model_world()
        asset.iloc[bad_pos] = bad_close
        with (
            caplog.at_level(logging.WARNING, logger=_EVENT_LOGGER),
            _divide_by_zero(bad_close == 0.0),
        ):
            result = es_mod.event_study(asset, bench, [days[_EVENT_POS]], event_window=(0, 3))
        assert result.events == []
        assert (result.n_events, result.mean_car, result.t_stat) == (0, None, None)
        assert (result.ci_low, result.ci_high, result.p_value) == (None, None, None)
        assert result.event_window == (0, 3)  # the requested window is still disclosed
        assert f"event {days[_EVENT_POS]} skipped" in caplog.text

    def test_r17_event_study_one_unusable_event_does_not_sink_the_others(self, caplog):

        # The second event sits 300 bars later: close enough to share the tape,
        # far enough that its estimation window clears the first event's window.
        days, asset, bench = _market_model_world(n=700)
        second = _EVENT_POS + 300
        # A zero close -> a -1 / inf return pair inside the SECOND event's
        # estimation window only.
        asset.iloc[_EVENT_POS + 120] = 0.0
        with (
            caplog.at_level(logging.WARNING, logger=_EVENT_LOGGER),
            _divide_by_zero(True),
        ):
            result = es_mod.event_study(
                asset, bench, [days[_EVENT_POS], days[second]], event_window=(0, 3),
            )
        assert [e.event_date for e in result.events] == [days[_EVENT_POS]]
        assert result.n_events == 1
        assert result.events[0].car == pytest.approx(0.0, abs=1e-12)
        assert f"event {days[second]} skipped" in caplog.text
        assert f"event {days[_EVENT_POS]} skipped" not in caplog.text

    def test_r17_run_backtest_event_study_leaves_defaults_when_no_event_is_decidable(self):
        """Through ``run_backtest``: the event study's wide pull reaches ~400
        days back, far outside the marking window the price gate protects, so
        an unusable close there arrives as a NaN return."""
        long_days = _bdays("2023-01-02", 400)
        asset = _ramp(long_days, step=0.1)
        asset.iloc[100] = float("nan")  # 160 sessions before the first event
        world = _world(AAPL=asset, SPY=_closes([100.0] * len(long_days), long_days))
        result = eng.run_backtest(
            _TapeGraph(dict.fromkeys(_R17_DATES, "Buy")), "AAPL", _R17_DATES,
            event_study=True,
            **_stock(holding_days=3, price_provider=world, compute_index_alpha=False),
        )
        m = result.metrics
        assert m.event_study_n == 0
        assert (m.event_study_mean_car, m.event_study_t_stat, m.event_study_p_value) == (
            None, None, None,
        )
        assert (m.event_study_ci, m.event_study_benchmark, m.event_study_window) == (
            None, None, None,
        )
        # The run itself is untouched.
        assert np.isfinite(result.equity).all()


# ===========================================================================
# report — disclosure lines and optional sections
# ===========================================================================
def _disclosure_world():
    """A three-bar stock world whose middle decision fails in the graph and
    whose last decision has no bar left to execute on."""
    days = _bdays("2024-01-02", 3)

    class _FlakyGraph(_TapeGraph):
        def propagate(self, company_name, trade_date, asset_type="stock"):
            if trade_date == days[1]:
                self.calls.append((company_name, trade_date, asset_type))
                raise RuntimeError("node timeout")
            return super().propagate(company_name, trade_date, asset_type)

    dates = [days[0], days[1], _R17_DAYS[5]]
    graph = _FlakyGraph(dict.fromkeys(dates, "Buy"))
    kwargs = _stock(
        price_provider=_world(AAPL=_closes([100.0, 101.0, 102.0], days)),
        compute_index_alpha=False,
    )
    return graph, dates, kwargs


_DEGRADED_LINE = (
    "- ⚠️ Degraded decisions: 1 (propagate failed or rating unparseable — forced to Hold)"
)
_UNEXECUTED_LINE = "- WARNING: Unexecuted decisions: 1 (no strictly-later price bar was available)"


class TestRenderBacktestReportDisclosures:
    def test_r17_report_discloses_degraded_and_unexecuted_decisions_of_a_real_run(self):
        graph, dates, kwargs = _disclosure_world()
        result = eng.run_backtest(graph, "AAPL", dates, **kwargs)
        assert (result.degraded_decision_count, result.unexecuted_decision_count) == (1, 1)
        lines = rep.render_backtest_report(result).splitlines()
        at = lines.index(_DEGRADED_LINE)
        # The disclosure block: cache, degraded, unexecuted, then the blank
        # line that ends it (no risk warnings in this run).
        assert lines[at - 1 : at + 3] == [
            "- Cache: 0 hits / 3 misses",
            _DEGRADED_LINE,
            _UNEXECUTED_LINE,
            "",
        ]

    def test_r17_report_without_those_events_has_no_disclosure_lines(self):
        result = eng.run_backtest(
            _TapeGraph(dict.fromkeys(_R17_DATES, "Buy")), "AAPL", _R17_DATES,
            **_stock(compute_index_alpha=False),
        )
        lines = rep.render_backtest_report(result).splitlines()
        assert not [line for line in lines if "Degraded decisions" in line]
        assert not [line for line in lines if "Unexecuted decisions" in line]
        assert not [line for line in lines if "Session calendar" in line]

    def test_r17_report_renders_the_session_calendar_assumption_after_the_decisions_line(self):
        result = eng.run_backtest(
            _TapeGraph(dict.fromkeys(_R17_DATES, "Buy")), "AAPL", _R17_DATES,
            **_stock(compute_index_alpha=False),
        )
        assumption = "365 sessions/year — klines-verified 24/7 calendar"
        stamped = dataclasses.replace(
            result,
            config_summary={**result.config_summary, "session_calendar_assumption": assumption},
        )
        lines = rep.render_backtest_report(stamped).splitlines()
        at = lines.index(f"- ⚠️ Session calendar: {assumption}")
        assert lines[at - 1].startswith("- Decisions: 3")
        assert lines[at + 1].startswith("- Cache: ")


class TestPerpSection:
    def test_r17_perp_section_of_a_plain_perp_run_states_zero_stop_exits(self):
        """Stop simulation is ON by default for perps, so "0" is a finding, not
        a missing row: the report must say it."""
        result = eng.run_backtest(
            _TapeGraph(dict.fromkeys(_R17_DATES, "Buy")), "BTCUSDT", _R17_DATES, **_perp(),
        )
        lines = rep._perp_section(result)
        assert lines[:7] == [
            "",
            "## Perp simulation",
            "",
            "- Leverage: 1.0x  |  Side: long-only  |  "
            "Fill quantization: fractional (no filters source)",
            "- Funding over the window: +0.00 USDT (net paid (drag))",
            "- Stop-trigger exits: 0",
            "- Liquidation events: 0",
        ]
        assert len(lines) == 8 and lines[7].startswith("- Model: USDT-M perp simulation")

    def test_r17_perp_section_lists_trigger_source_funding_and_liquidations(self):
        base = eng.run_backtest(
            _TapeGraph(dict.fromkeys(_R17_DATES, "Buy")), "BTCUSDT", _R17_DATES, **_perp(),
        )
        result = dataclasses.replace(
            base,
            metrics=dataclasses.replace(
                base.metrics, funding_drag_annualized=-0.0312, stop_trigger_count=2,
            ),
            config_summary={
                **base.config_summary,
                "perp_leverage": 5,
                "perp_short": True,
                "perp_liq_price_source": "last (mark unavailable)",
                "perp_funding_paid_total": -12.5,
                "perp_stop_triggers": [{"date": "2024-01-10"}, {"date": "2024-01-20"}],
                "perp_liquidations": [
                    {
                        "date": "2024-01-12", "side": "long", "shares": 2.0,
                        "entry": 0.0000123456, "liquidation_price": 0.0000041,
                        "exit_price": 0.0000041, "fee": 3.2,
                    }
                ],
                "perp_model_note": "NOTE",
            },
        )
        assert rep._perp_section(result) == [
            "",
            "## Perp simulation",
            "",
            "- Leverage: 5x  |  Side: long + short  |  Fill quantization: "
            "fractional (no filters source)",
            "- Liquidation trigger source: last (mark unavailable)",
            # Negative = net received; the annualized figure keeps the sign of
            # the signed total, so a short-biased run reads "-x%/yr".
            "- Funding over the window: -12.50 USDT (net received)  |  "
            "~-3.12%/yr of initial capital",
            "- Stop-trigger exits: 2",
            "- ⚠️ Liquidation events: 1",
            "",
            "| Date | Side | Entry | Liquidation price | Fee |",
            "|---|---|---:|---:|---:|",
            "| 2024-01-12 | long | 1.235e-05 | 4.1e-06 | 3.200 |",
            "- Model: NOTE",
        ]

    def test_r17_perp_section_omits_the_stop_row_when_nothing_is_simulated(self):
        base = eng.run_backtest(
            _TapeGraph(dict.fromkeys(_R17_DATES, "Buy")), "BTCUSDT", _R17_DATES,
            **_perp(simulate_stop_triggers=False),
        )
        assert base.config_summary["perp_stop_simulation"] is False
        lines = rep._perp_section(base)
        assert not [line for line in lines if "Stop-trigger" in line]
        assert "stop-losses advisory only (not simulated)" in lines[-1]

    def test_r17_perp_section_is_empty_for_non_perp_results(self):
        plain = _result([100.0, 101.0], ["2024-01-02", "2024-01-03"], [50.0])
        assert rep._perp_section(plain) == []
        assert "Perp simulation" not in rep.render_backtest_report(plain)


class TestRenderFactorSection:
    def test_r17_report_renders_the_factor_section_of_an_attributed_run(self, monkeypatch):
        days = _bdays("2024-01-02", 40)
        factors = _ff_factors(days[1:])
        tape = _ff_strategy_tape(days, factors, _FF_LOADINGS)
        _patch_factor_loader(monkeypatch, factors.loc[pd.Timestamp(days[2]):])
        result = eng.run_backtest(
            _TapeGraph({days[0]: "Buy"}), "AAPL", [days[0]], factor_model="3",
            **_stock(
                holding_days=20,
                price_provider=_world(AAPL=tape, SPY=tape),
                compute_index_alpha=False,
            ),
        )
        alpha = result.metrics.factor_alpha
        lines = rep.render_backtest_report(result).splitlines()
        at = lines.index("## Factor attribution")
        assert lines[at : at + 6] == [
            "## Factor attribution",
            "",
            f"- Model: **FF3**  |  R²: 1.000  |  Factor alpha (ann.): {alpha * 100:.2f}%",
            "",
            "| Factor | Beta |",
            "|---|---:|",
        ]
        # One row per factor, sorted by name, three decimals with a sign. The
        # HML loading is exactly 0 in the data, so its sign is float noise.
        assert lines[at + 7] == "| Mkt-RF | +1.500 |"
        assert lines[at + 8] == "| SMB | +0.500 |"
        assert re.fullmatch(r"\| HML \| [+-]0\.000 \|", lines[at + 6])
        assert lines[at + 9] == ""

    def test_r17_report_has_no_factor_section_without_attribution(self):
        md = rep.render_backtest_report(
            eng.run_backtest(
                _TapeGraph(dict.fromkeys(_R17_DATES, "Buy")), "AAPL", _R17_DATES,
                **_stock(compute_index_alpha=False),
            )
        )
        assert "Factor attribution" not in md


class TestMultiRunReport:
    def test_r17_multi_run_report_without_results_says_so(self):
        assert rep.render_multi_run_report([]) == "# Multi-run report\n\nNo runs.\n"

    @pytest.mark.parametrize("n_runs", [0, -1])
    def test_r17_multi_run_rejects_a_non_positive_run_count_without_calling_the_factory(
        self, n_runs
    ):
        calls: list[int] = []

        def factory(i: int) -> BacktestResult:
            calls.append(i)
            raise AssertionError("the factory must not run")

        with pytest.raises(ValueError, match="n_runs must be >= 1"):
            rep.multi_run(factory, n_runs)
        assert calls == []


# ===========================================================================
# ic_dataset — export gates and prune verdict
# ===========================================================================
def _ohlcv(days: int = 12) -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-02", periods=days)
    closes = [100.0 + 0.5 * i for i in range(days)]
    return pd.DataFrame(
        {
            "Date": dates,
            "Open": closes,
            "High": [c + 0.5 for c in closes],
            "Low": [c - 0.5 for c in closes],
            "Close": closes,
            "Volume": [1_000.0] * days,
        }
    )


@pytest.fixture()
def _ohlcv_cache(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, str]]:
    """Serve a 12-row OHLCV frame from ``load_ohlcv``; record what was asked."""
    from yialpha.dataflows import stockstats_utils

    served: list[tuple[str, str]] = []

    def load(ticker: str, curr_date: str) -> pd.DataFrame:
        served.append((ticker, curr_date))
        return _ohlcv()

    monkeypatch.setattr(stockstats_utils, "load_ohlcv", load)
    return served


_IC_LOGGER = "yialpha.backtest.ic_dataset"


class TestBuildIcFrameGates:
    @pytest.mark.parametrize(
        ("extra", "shown"), [([0], 0), ([-3], -3), ([5, 0], 0), ([2, -1, 3], -1)],
    )
    def test_r17_build_ic_frame_rejects_non_positive_extra_horizons(
        self, _ohlcv_cache, extra, shown
    ):
        with pytest.raises(ValueError, match=rf"extra horizons must be positive; got {shown}$"):
            icd.build_ic_frame(
                "NVDA", horizon=5, indicators=[], as_of="2024-01-31", extra_horizons=extra,
            )

    def test_r17_build_ic_frame_longer_extra_horizon_is_nan_not_fabricated_in_the_tail(
        self, _ohlcv_cache
    ):
        """The tail drop keys on the MAIN horizon. A longer extra horizon has
        no realizable return on the rows between the two horizons: those cells
        stay empty instead of borrowing a shorter look-ahead."""
        frame, skipped = icd.build_ic_frame(
            "NVDA", horizon=2, indicators=["rsi"], as_of="2024-01-31", extra_horizons=[4],
        )
        closes = _ohlcv()["Close"]
        assert skipped == []
        assert _ohlcv_cache == [("NVDA", "2024-01-31")]
        # 12 rows, the last 2 dropped for horizon 2; rows 0-7 have a 4-row
        # forward return, rows 8-9 (between the two horizons) do not.
        assert frame["date"].tolist() == _bdays("2024-01-02", 10)
        assert frame["fwd_ret_4d"].notna().tolist() == [True] * 8 + [False] * 2
        expected = (closes.shift(-4) / closes - 1.0)[:8]
        assert frame["fwd_ret_4d"].dropna().tolist() == pytest.approx(expected.tolist())
        assert frame["forward_return"].tolist() == pytest.approx(
            (closes.shift(-2) / closes - 1.0)[:10].tolist()
        )


class TestExportIcDatasetsGates:
    @pytest.mark.parametrize("horizon", [0, -1])
    def test_r17_export_rejects_non_positive_horizon_before_touching_disk(
        self, tmp_path, horizon
    ):
        out = tmp_path / "ic_data"
        with pytest.raises(ValueError, match="horizon must be a positive number of trading days"):
            icd.export_ic_datasets(
                ["AAPL"], horizon=horizon, indicators=["rsi"], as_of="2024-01-31", output_dir=out,
            )
        assert not out.exists()

    def test_r17_export_skips_a_ticker_whose_history_is_shorter_than_the_horizon(
        self, tmp_path, monkeypatch, caplog
    ):
        """Four rows of history, a five-row horizon: no row has a realizable
        forward return, so no CSV is written and the ticker is left out."""
        from yialpha.dataflows import stockstats_utils

        monkeypatch.setattr(stockstats_utils, "load_ohlcv", lambda ticker, curr_date: _ohlcv(4))
        with caplog.at_level(logging.ERROR, logger=_IC_LOGGER):
            written = icd.export_ic_datasets(
                ["AAPL"], horizon=5, indicators=["rsi"], as_of="2024-01-31", output_dir=tmp_path,
            )
        assert written == {}
        assert list(tmp_path.iterdir()) == []
        assert "no usable rows for AAPL (empty frame after tail drop)" in caplog.text


class TestPruneVerdictForCsv:
    def test_r17_prune_verdict_rejects_a_csv_without_the_required_columns(self, tmp_path):
        bad = tmp_path / "NVDA_5d.csv"
        bad.write_text("date,rsi\n2024-01-02,50\n", encoding="utf-8")
        with pytest.raises(ValueError, match="missing date/forward_return columns") as exc_info:
            icd.prune_verdict_for_csv(bad)
        assert str(bad) in str(exc_info.value)

    def test_r17_prune_verdict_rejects_a_csv_without_indicator_columns(self, tmp_path):
        """``fwd_ret_*`` columns are extra forward returns, never indicators."""
        bad = tmp_path / "NVDA_5d.csv"
        bad.write_text("date,forward_return,fwd_ret_10d\n2024-01-02,0.01,0.02\n", encoding="utf-8")
        with pytest.raises(ValueError, match="no indicator columns") as exc_info:
            icd.prune_verdict_for_csv(bad)
        assert str(bad) in str(exc_info.value)


# ===========================================================================
# factor_model — stale-cache fallback
# ===========================================================================
class _VanishingCacheFile:
    """Stands in for the cache path: old enough to need a download, and gone by
    the time its age is read again for the stale-serve decision."""

    def __init__(self) -> None:
        self.stats = 0

    def exists(self) -> bool:
        return True

    def stat(self) -> SimpleNamespace:
        self.stats += 1
        if self.stats == 1:
            return SimpleNamespace(st_mtime=time.time() - 5 * 86_400.0)
        raise OSError("cache file vanished")

    def read_bytes(self) -> bytes:
        raise AssertionError("a stale cache must not be read while the download is tried")


class TestFactorCacheFallback:
    def test_r17_factor_cache_vanishing_between_probes_returns_none(self, monkeypatch, caplog):
        import requests

        from yialpha.dataflows import disk_cache

        monkeypatch.setattr(disk_cache, "max_stale_days", lambda: 7.0)

        def offline(url: str, timeout: int):
            raise ConnectionError("offline")

        monkeypatch.setattr(requests, "get", offline)
        path = _VanishingCacheFile()
        with caplog.at_level(logging.WARNING, logger="yialpha.backtest.factor_model"):
            raw = fm._cached_or_download(path, "https://example.invalid/factors.zip")
        assert raw is None
        assert path.stats == 2
        assert "is inf days old (over the stale-serve cap); refusing to serve stale" in caplog.text


# ---------------------------------------------------------------------------
# run_backtest() — interrupts and the event study's benchmark leg
# ---------------------------------------------------------------------------
class TestRunBacktestInterrupts:
    @pytest.mark.parametrize("interrupt", [KeyboardInterrupt, SystemExit])
    def test_r17_run_backtest_does_not_turn_an_interrupt_into_a_hold(self, interrupt):
        """A failing ``propagate`` degrades to Hold; an operator interrupt is
        not a failing node and must stop the backtest."""

        class _Interrupted(_TapeGraph):
            def propagate(self, company_name, trade_date, asset_type="stock"):
                raise interrupt("operator stop")

        with pytest.raises(interrupt, match="operator stop"):
            eng.run_backtest(
                _Interrupted({}), "AAPL", _R17_DATES, **_stock(compute_index_alpha=False),
            )


class TestRunBacktestEventStudyBenchmarkLeg:
    @pytest.mark.parametrize(
        ("benchmark", "served_by_the_perp_leg"),
        [("BTCUSDT", True), ("SPY", False)],
    )
    def test_r17_run_backtest_event_study_benchmark_leg_follows_the_index_leg_provider(
        self, monkeypatch, benchmark, served_by_the_perp_leg
    ):
        """On a run whose price provider was swapped for Binance (the default
        provider on a perp), a Binance-shaped benchmark is pulled from the
        run's own provider -- yfinance would quietly hand back the Yahoo SPOT
        pair -- and every other name keeps the yfinance leg."""
        asked: list[tuple[str, str, str]] = []

        def perp_prices(ticker: str, start: str, end: str) -> pd.Series:
            asked.append((ticker, start, end))
            return _R17_AAPL.loc[(_R17_AAPL.index >= start) & (_R17_AAPL.index <= end)].copy()

        monkeypatch.setattr(eng, "_binance_perp_price_provider", lambda price_type="last": perp_prices)
        seen: dict[str, Any] = {}

        def spy(metrics, graph, ticker, price_provider, trades, index_name, end_date, **kwargs):
            seen.update(asset_leg=price_provider, bench_leg=kwargs["bench_provider"])

        monkeypatch.setattr(eng, "_run_event_study", spy)
        # No price_provider: the engine swaps its yfinance default for Binance.
        result = eng.run_backtest(
            _TapeGraph(dict.fromkeys(_R17_DATES, "Buy"), benchmark=benchmark), "BTCUSDT", _R17_DATES,
            event_study=True, simulate_stop_triggers=False,
            asset_type="crypto_perp", compute_index_alpha=False, cost_bps=0.0, holding_days=5,
            funding_provider=_funding(dict.fromkeys(_R17_DAYS, 0.0)),
        )
        assert result.config_summary["perp_price_source"] == "binance perp klines"
        assert seen["asset_leg"] is perp_prices
        assert (seen["bench_leg"] is perp_prices) is served_by_the_perp_leg
        if not served_by_the_perp_leg:
            assert seen["bench_leg"] is eng._yfinance_price_provider
        # Only the asset itself was ever priced on the Binance leg here.
        assert {ticker for ticker, _start, _end in asked} == {"BTCUSDT"}
