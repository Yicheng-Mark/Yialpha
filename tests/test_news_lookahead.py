"""yfinance news must not leak future-dated (or undated, in a backtest) articles
into a historical window.

Regressions for #992 (flat articles bypassed the date filter), #1007 (global
news injected future articles), #993 (empty-after-filter returned a blank body),
plus the workflow-B news PIT clamps (B9):

* ``get_news_yfinance`` / ``get_global_news_yfinance`` clamp the LLM-supplied
  window to the run's PINNED analysis date, so a backtest prompt can never
  receive headlines published after its decision date.
* ``_in_news_window`` compares aware pub timestamps on a single calendar
  standard (naive-UTC) — the old ``replace(tzinfo=None)`` compared the
  original offset's wall clock, skewing windows by up to the UTC offset.
* The Alpha Vantage news vendor applies the same pinned-date clamp to its
  ``time_to`` query parameter.
"""
from datetime import datetime, timezone

import pytest

import yiagents.dataflows.alpha_vantage_news as av_news
import yiagents.dataflows.yfinance_news as ynews
from yiagents.dataflows.utils import set_analysis_date


@pytest.fixture(autouse=True)
def _isolated_search_cache(tmp_path, monkeypatch):
    """Route the global-news Search disk cache into a per-test tmp dir.

    ``_cached_search_news`` serves repeats from disk, which would otherwise
    bypass the mocked ``yf.Search`` of a later test (and pollute the real
    user cache).
    """
    def _cache_dir(name):
        d = tmp_path / name
        d.mkdir(parents=True, exist_ok=True)
        return str(d)

    monkeypatch.setattr(ynews, "vendor_cache_dir", _cache_dir)


def _epoch(date_str):
    # UTC-derived epoch: the flat-article parser converts epochs as UTC (the
    # 2026-08-16 fix), so the synthetic timestamps must be UTC too — a
    # host-local ``time.mktime`` would make every assertion below depend on
    # the machine's timezone.
    dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return int(dt.timestamp())


@pytest.mark.unit
def test_flat_article_publish_time_is_parsed():
    # #992: flat articles now carry a pub_date (was always None -> unfilterable).
    data = ynews._extract_article_data(
        {"title": "X", "publisher": "P", "link": "l", "providerPublishTime": _epoch("2025-05-09")}
    )
    assert data["pub_date"] is not None
    assert data["pub_date"].strftime("%Y-%m-%d") == "2025-05-09"


@pytest.mark.unit
def test_flat_article_epoch_parsed_as_utc_not_host_wall_clock():
    """The epoch must land on the UTC calendar, not the host's wall clock.

    2025-05-10 00:30 UTC reads 08:30 on a UTC+8 host and 2025-05-09 19:30 on
    a UTC-5 host — ``fromtimestamp(ts)`` without tz made the parsed DATE
    follow the host, skewing window inclusion near midnight by the offset.
    """
    ts = int(datetime(2025, 5, 10, 0, 30, tzinfo=timezone.utc).timestamp())
    data = ynews._extract_article_data(
        {"title": "X", "publisher": "P", "link": "l", "providerPublishTime": ts}
    )
    assert data["pub_date"] == datetime(2025, 5, 10, 0, 30, tzinfo=timezone.utc)
    assert data["pub_date"].strftime("%Y-%m-%d") == "2025-05-10"


@pytest.mark.unit
def test_window_excludes_future_and_undated_in_backtest():
    start = datetime(2025, 5, 1)
    end = datetime(2025, 5, 9)  # historical window (well in the past)
    inside = datetime(2025, 5, 5)
    future = datetime(2025, 6, 1)
    assert ynews._in_news_window(inside, start, end) is True
    assert ynews._in_news_window(future, start, end) is False     # look-ahead blocked
    assert ynews._in_news_window(None, start, end) is False        # undated -> excluded in backtest


@pytest.mark.unit
def test_window_keeps_undated_in_live_window():
    # Live window (reaches today): undated articles can't be "future", so keep them.
    start = datetime.now()
    end = datetime.now()
    assert ynews._in_news_window(None, start, end) is True


@pytest.mark.unit
def test_global_news_future_flat_article_excluded(monkeypatch):
    # #1007: a flat, future-dated global article must not appear in a historical run.
    future_article = {"title": "FUTURE EVENT", "publisher": "P", "link": "l",
                      "providerPublishTime": _epoch("2025-06-01")}
    past_article = {"title": "PAST EVENT", "publisher": "P", "link": "l",
                    "providerPublishTime": _epoch("2025-05-05")}

    class FakeSearch:
        def __init__(self, *a, **k):
            self.news = [future_article, past_article]

    monkeypatch.setattr(ynews.yf, "Search", FakeSearch)
    out = ynews.get_global_news_yfinance("2025-05-09", look_back_days=7, limit=10)
    assert "PAST EVENT" in out
    assert "FUTURE EVENT" not in out  # #1007


@pytest.mark.unit
def test_global_news_empty_after_filter_is_informative(monkeypatch):
    # #993: everything filtered out -> a clear message, not a blank-bodied report.
    only_future = {"title": "FUTURE", "publisher": "P", "link": "l",
                   "providerPublishTime": _epoch("2025-06-01")}

    class FakeSearch:
        def __init__(self, *a, **k):
            self.news = [only_future]

    monkeypatch.setattr(ynews.yf, "Search", FakeSearch)
    out = ynews.get_global_news_yfinance("2025-05-09", look_back_days=7, limit=10)
    assert "No global news found" in out
    assert "###" not in out  # no empty article body


# --------------------------------------------------------------------------- #
# B9 — pinned-analysis-date clamps on the news tools
# --------------------------------------------------------------------------- #

def _content_article(title: str, pub_iso: str) -> dict:
    """A nested-``content`` yfinance news article (the current get_news shape)."""
    return {"content": {
        "title": title, "summary": "", "provider": {"displayName": "P"},
        "canonicalUrl": {"url": "l"}, "pubDate": pub_iso,
    }}


@pytest.mark.unit
def test_news_end_date_clamped_to_pinned_analysis_date(monkeypatch):
    """The stock-news tool carries no analysis date; with one pinned, an
    LLM-supplied end_date beyond it must be clamped so future headlines never
    enter the backtest prompt (live mode: no-op)."""
    set_analysis_date("2025-05-09")
    try:
        inside = _content_article("PAST EVENT", "2025-05-05T10:00:00+00:00")
        future = _content_article("FUTURE EVENT", "2025-05-20T10:00:00+00:00")

        class FakeTicker:
            def __init__(self, symbol):
                pass

            def get_news(self, count):
                return [inside, future]

        monkeypatch.setattr(ynews.yf, "Ticker", FakeTicker)
        monkeypatch.setattr(ynews, "yf_retry", lambda fn, **kw: fn())
        out = ynews.get_news_yfinance("AAPL", "2025-05-01", "2030-01-01")
        assert "PAST EVENT" in out
        assert "FUTURE EVENT" not in out  # clamped to 2025-05-09, then filtered
    finally:
        set_analysis_date(None)


@pytest.mark.unit
def test_global_news_curr_date_clamped_to_pinned_analysis_date(monkeypatch):
    """Same clamp for the global-news Search path: a curr_date past the pinned
    analysis date cannot widen the window into the future."""
    set_analysis_date("2025-05-09")
    try:
        past_article = {"title": "PAST EVENT", "publisher": "P", "link": "l",
                        "providerPublishTime": _epoch("2025-05-05")}
        future_article = {"title": "FUTURE EVENT", "publisher": "P", "link": "l",
                          "providerPublishTime": _epoch("2025-06-01")}

        class FakeSearch:
            def __init__(self, *a, **k):
                self.news = [past_article, future_article]

        monkeypatch.setattr(ynews.yf, "Search", FakeSearch)
        out = ynews.get_global_news_yfinance("2030-01-01", look_back_days=7, limit=10)
        assert "PAST EVENT" in out
        assert "FUTURE EVENT" not in out
    finally:
        set_analysis_date(None)


@pytest.mark.unit
def test_in_news_window_compares_aware_timestamps_in_utc():
    """An aware pub timestamp must be compared on UTC semantics, not its
    original offset's wall clock (the old replace(tzinfo=None) skew)."""
    start, end = datetime(2025, 5, 1), datetime(2025, 5, 9)
    # 2025-05-10T01:30+08:00 == 2025-05-09 17:30 UTC -> INSIDE a window
    # ending 2025-05-09; the wall-clock comparison read it as May 10th.
    plus8 = datetime.fromisoformat("2025-05-10T01:30:00+08:00")
    assert ynews._in_news_window(plus8, start, end) is True
    # Mirror case: 2025-05-09T23:30-08:00 == 2025-05-10 07:30 UTC -> OUTSIDE
    # (the wall-clock comparison read it as May 9th, inside).
    minus8 = datetime.fromisoformat("2025-05-09T23:30:00-08:00")
    assert ynews._in_news_window(minus8, start, end) is False


@pytest.mark.unit
def test_av_news_time_to_clamped_to_pinned_analysis_date(monkeypatch):
    """The Alpha Vantage news vendor applies the same pinned-date clamp to its
    ``time_to`` query parameter (live mode: unchanged)."""
    captured: dict[str, str] = {}
    monkeypatch.setattr(
        av_news, "_make_api_request",
        lambda fn, params: captured.update(params) or {},
    )
    set_analysis_date("2025-05-09")
    try:
        av_news.get_news("AAPL", "2025-05-01", "2030-01-01")
        assert captured["time_to"] == "20250509T0000"  # clamped, not 2030
    finally:
        set_analysis_date(None)

    av_news.get_news("AAPL", "2025-05-01", "2030-01-01")
    assert captured["time_to"] == "20300101T0000"  # live: no-op pass-through
