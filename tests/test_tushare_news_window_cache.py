"""Tushare news fixes (B8): dynamic query window + short-TTL cache, the
cached pro handle, and the permission-error taxonomy.

The old implementation hardcoded ``start_date="20240101`` — a backtest for a
pre-2024 date queried a window that could not contain it and a live run pulled
the entire table every call — and re-ran ``ts.set_token`` + ``ts.pro_api()``
on every vendor call, and classified a permission denial (forbidden) as a
rate limit (transient).
"""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import pytest

from yialpha.dataflows import tushare_vendor as tv
from yialpha.dataflows.errors import (
    NoMarketDataError,
    VendorError,
    VendorNotConfiguredError,
    VendorRateLimitError,
)


def _news_df(as_of: date | None = None, days: int = 3,
             keyword: str = "600519") -> pd.DataFrame:
    """Headlines dated inside the [as_of - days, as_of] window."""
    as_of = as_of or date.today()
    rows = []
    for i in range(days):
        d = as_of - timedelta(days=i + 1)
        rows.append({
            "title": f"{keyword} headline {i}",
            "datetime": f"{d.isoformat()} 10:00:00",
            "src": "sina",
            "content": f"{keyword} content {i}",
        })
    return pd.DataFrame(rows)


class _Pro:
    """Fake pro_api handle that records the kwargs of each news call."""

    def __init__(self, df: pd.DataFrame | Exception | None = None):
        self._df = df
        self.news_calls: list[dict] = []

    def news(self, **kw):
        self.news_calls.append(kw)
        if isinstance(self._df, Exception):
            raise self._df
        return self._df


@pytest.fixture(autouse=True)
def _token(monkeypatch):
    monkeypatch.setenv("TUSHARE_TOKEN", "test-token")


@pytest.fixture(autouse=True)
def _isolated_vendor_cache(monkeypatch, tmp_path):
    """Route the news disk cache into tmp_path — the REAL vendor_cache_dir
    writes into the user's ~/.yialpha cache, so an unisolated run both
    pollutes it and becomes order-dependent across test invocations."""

    def _cache_dir(name: str) -> str:
        d = tmp_path / name
        d.mkdir(parents=True, exist_ok=True)
        return str(d)

    monkeypatch.setattr(tv, "vendor_cache_dir", _cache_dir)


# --------------------------------------------------------------------------- #
# Dynamic window + cache
# --------------------------------------------------------------------------- #

@pytest.mark.unit
def test_news_query_window_derived_from_curr_date(monkeypatch):
    """A pre-2024 backtest must query ITS OWN window (old code hardcoded
    20240101, guaranteeing empty results for earlier dates)."""
    pro = _Pro(_news_df())
    monkeypatch.setattr(tv, "_require_tushare", lambda: pro)
    tv.get_a_share_news_native("600519.SS", "2020-03-10", 14)
    kw = pro.news_calls[0]
    assert kw["start_date"] == "20200225"   # 2020-03-10 minus 14 days
    assert kw["end_date"] == "20200310"


@pytest.mark.unit
def test_news_window_capped_for_runaway_lookback(monkeypatch):
    """A huge look_back_days cannot pull months of rows: the API window is
    clamped to _NEWS_WINDOW_CAP_DAYS."""
    pro = _Pro(_news_df())
    monkeypatch.setattr(tv, "_require_tushare", lambda: pro)
    tv.get_a_share_news_native("600519.SS", "2020-06-30", 3650)
    kw = pro.news_calls[0]
    expected_start = (
        date(2020, 6, 30) - timedelta(days=tv._NEWS_WINDOW_CAP_DAYS)
    ).strftime("%Y%m%d")
    assert kw["start_date"] == expected_start


@pytest.mark.unit
def test_news_calls_share_one_cached_window(monkeypatch):
    """Same window -> one pro call; the disk cache serves repeats (TTL is
    minutes, matching the other news vendors)."""
    # Headlines dated INSIDE the queried window (2020-02-24..2020-03-10) so
    # the client-side window filter keeps them.
    pro = _Pro(_news_df(date(2020, 3, 9)))
    monkeypatch.setattr(tv, "_require_tushare", lambda: pro)
    first = tv.get_a_share_news_native("600519.SS", "2020-03-10", 14)
    second = tv.get_a_share_news_native("600519.SS", "2020-03-10", 14)
    assert len(pro.news_calls) == 1
    assert first == second
    assert "600519 headline" in first


@pytest.mark.unit
def test_news_fetch_errors_propagate_uncached(monkeypatch):
    """A typed vendor error from the fetch is NOT swallowed or cached: the
    router must see it (and a retry re-fetches)."""
    pro = _Pro(VendorRateLimitError("每分钟访问次数超限"))
    monkeypatch.setattr(tv, "_require_tushare", lambda: pro)
    with pytest.raises(VendorRateLimitError):
        tv.get_a_share_news_native("600519.SS", "2020-03-10", 14)
    with pytest.raises(VendorRateLimitError):
        tv.get_a_share_news_native("600519.SS", "2020-03-10", 14)
    assert len(pro.news_calls) == 2  # nothing poisoned the cache


@pytest.mark.unit
def test_news_empty_frame_is_honest_empty(monkeypatch):
    pro = _Pro(pd.DataFrame())
    monkeypatch.setattr(tv, "_require_tushare", lambda: pro)
    out = tv.get_a_share_news_native("600519.SS", "2020-03-10", 14)
    assert "No news items" in out


@pytest.mark.unit
def test_news_cache_ttl_is_minutes():
    expected_ttl_days = 30.0 / (24.0 * 60.0)
    assert pytest.approx(expected_ttl_days) == tv._NEWS_CACHE_TTL_DAYS


# --------------------------------------------------------------------------- #
# Cached pro handle (module-level, keyed by token)
# --------------------------------------------------------------------------- #

@pytest.mark.unit
def test_pro_handle_is_cached_per_token(monkeypatch):
    """set_token + pro_api run ONCE per token; repeat vendor calls reuse the
    handle, and switching the token builds a fresh one."""
    built = []

    class _FakeTs:
        @staticmethod
        def set_token(token):
            built.append(("set_token", token))

        @staticmethod
        def pro_api():
            built.append(("pro_api", None))
            return _Pro(_news_df())

    import sys
    monkeypatch.setitem(sys.modules, "tushare", _FakeTs)

    tv._PRO_HANDLE_CACHE.clear()
    try:
        first = tv._require_tushare()
        second = tv._require_tushare()
        assert first is second
        assert built == [("set_token", "test-token"), ("pro_api", None)]

        monkeypatch.setenv("TUSHARE_TOKEN", "other-token")
        third = tv._require_tushare()
        assert third is not first
        assert built == [
            ("set_token", "test-token"), ("pro_api", None),
            ("set_token", "other-token"), ("pro_api", None),
        ]
    finally:
        tv._PRO_HANDLE_CACHE.clear()


@pytest.mark.unit
def test_missing_token_still_not_configured(monkeypatch):
    monkeypatch.delenv("TUSHARE_TOKEN", raising=False)
    from yialpha.dataflows import config as cfgmod
    orig = cfgmod.get_config()
    try:
        cfgmod.set_config({**orig, "tushare_token": None})
        with pytest.raises(VendorNotConfiguredError):
            tv._require_tushare()
    finally:
        cfgmod.set_config(orig)


# --------------------------------------------------------------------------- #
# Error taxonomy: permission != rate limit != no-data
# --------------------------------------------------------------------------- #

@pytest.mark.unit
@pytest.mark.parametrize("msg", [
    "抱歉，您没有访问该接口的权限",
    "Permission denied: news api requires higher tier",
    "api forbidden (403)",
])
def test_permission_error_is_vendor_error_not_rate_limit(msg):
    """A permission denial is forbidden, not transient: it must surface as the
    VendorError base — neither the router's skip-and-retry rate-limit path nor
    a ticker-blaming NoMarketDataError."""
    pro = _Pro(RuntimeError(msg))
    with pytest.raises(VendorError) as ei:
        tv._query(pro, "news", ts_code="600519.SH")
    assert not isinstance(ei.value, VendorRateLimitError)
    assert not isinstance(ei.value, NoMarketDataError)
    assert "permission" in str(ei.value).lower()


@pytest.mark.unit
def test_rate_limit_still_classified_as_rate_limit():
    pro = _Pro(RuntimeError("每分钟访问次数超限"))
    with pytest.raises(VendorRateLimitError):
        tv._query(pro, "news", ts_code="600519.SH")


@pytest.mark.unit
def test_other_api_failure_still_no_market_data():
    pro = _Pro(RuntimeError("unknown param foo"))
    with pytest.raises(NoMarketDataError):
        tv._query(pro, "news", ts_code="600519.SH")
