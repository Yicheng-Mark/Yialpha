"""Round-13 adversarial-remediation regressions (2026-11-22).

Six confirmed findings from the round-13 adversarial sweep, each pinned here
so its exact failure mode cannot silently return:

1. [P1] ``y_finance._get_stock_stats_bulk`` zipped the derived series against
   the full frame with ``strict=True``, but the vol estimators clean their
   input first (``vol_estimators._clean_ohlcv`` dropna's OHLC and drops
   non-positive rows), so a single Low=0 bad tick or a leading NaN High made
   the derived series SHORTER than the frame and the ValueError collapsed the
   WHOLE rvol_20/ewma_vol window into NO_DATA over otherwise-good rows.
2. [P2] ``fundamentals_overview`` and ``alpha_vantage_fundamentals`` judged
   the overview leak gate on an UNCLAMPED curr_date — an LLM echoing a cached
   header's "today" into a run pinned to a past analysis date passed as live
   mode and served today's snapshot into the historical run (the round-8/
   round-11 clamps existed everywhere else around these gates).
3. [P3] Five A-share entry points (BaoStock OHLC/valuation, Tushare
   valuation/news, Eastmoney margin) missed the analysis-date clamp the
   round-11 statement entries already had.
4. [P3] BaoStock's same-day/finality cache TTL anchored on the HOST clock
   instead of the exchange's Beijing calendar (the stockstats_utils._utc_today
   trap, A-share edition).
5. [P3] Tushare's empty news frame was persisted as ``b""`` for the
   ~30-minute news TTL, blocking later retries from seeing headlines
   published in between.
6. [P3] Eastmoney's pagination terminated on the FILTERED row count, so a
   full 400-row page carrying one empty-DATE row was misread as the report's
   last page and older history was silently truncated.

Hermetic: every vendor seam (load_ohlcv, pro_api, BaoStock session/cache,
Eastmoney HTTP) is monkeypatched; no network, no tokens.
"""

from __future__ import annotations

import json
import unittest
from datetime import date, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

import yialpha.dataflows.alpha_vantage_fundamentals as avf
import yialpha.dataflows.baostock_vendor as bsv
import yialpha.dataflows.eastmoney as em
import yialpha.dataflows.fundamentals_overview as fo
import yialpha.dataflows.tushare_vendor as tv
import yialpha.dataflows.y_finance as y_finance
from yialpha.dataflows.errors import NoMarketDataError
from yialpha.dataflows.utils import set_analysis_date

_TODAY = date.today().isoformat()


# --------------------------------------------------------------------------- #
# 1. [P1] y_finance — derived-series alignment after estimator row cleaning
# --------------------------------------------------------------------------- #
def _dirty_ohlcv(n: int = 30) -> pd.DataFrame:
    """A load_ohlcv-shaped frame carrying the two row classes that survive
    ``_clean_dataframe`` but are dropped by ``vol_estimators._clean_ohlcv``:
    a LEADING NaN High (forward-only gap fill cannot repair leading gaps) and
    a Low=0 bad tick (non-positive prices poison log-ratio windows)."""
    closes = [100.0 + (i % 5) for i in range(n)]
    frame = pd.DataFrame({
        "Date": pd.bdate_range("2026-07-01", periods=n),
        "Open": [c - 0.5 for c in closes],
        "High": [c + 1.0 for c in closes],
        "Low": [c - 1.0 for c in closes],
        "Close": closes,
        "Volume": [1000.0] * n,
    })
    frame.loc[0, "High"] = float("nan")  # leading NaN High (bdate row 2026-07-01)
    frame.loc[5, "Low"] = 0.0            # bad tick (bdate row 2026-07-08)
    return frame


def _window_lines(report: str) -> dict[str, str]:
    """Parse an indicator-window report into {date: rendered_value}."""
    return {
        ln.split(": ", 1)[0]: ln.split(": ", 1)[1]
        for ln in report.splitlines()
        if ln[:4].isdigit() and ": " in ln
    }


@pytest.mark.unit
@pytest.mark.parametrize("indicator", ["rvol_20", "ewma_vol"])
def test_indicators_window_survives_dirty_ohlcv_rows(monkeypatch, indicator):
    """A frame with cleanable bad rows must NOT crash the window into
    NO_DATA — the dropped days render 'N/A' and the rest keep their values
    (pre-fix: ValueError from the strict zip, whole window unavailable)."""
    frame = _dirty_ohlcv()
    monkeypatch.setattr(y_finance, "load_ohlcv", lambda s, d: frame.copy())
    out = y_finance.get_stock_stats_indicators_window(
        "AAPL", indicator, "2026-08-11", 45)
    assert f"## {indicator} values" in out
    lines = _window_lines(out)
    # The two dropped trading days render an honest 'N/A' VALUE — distinct
    # from a non-trading day's "N/A: Not a trading day" filler.
    assert lines["2026-07-01"] == "N/A"
    assert lines["2026-07-08"] == "N/A"
    # The last (good) trading day still carries a numeric reading.
    assert lines["2026-08-11"] != "N/A"


@pytest.mark.unit
def test_dirty_ohlcv_good_rows_match_clean_estimator(monkeypatch):
    """好行数值不变: the surviving rows must carry EXACTLY the values the
    estimator yields on the pre-cleaned frame — the alignment repair may not
    shift, refit, or otherwise alter good rows' numbers."""
    frame = _dirty_ohlcv()
    dropped = [0, 5]  # what vol_estimators._clean_ohlcv will remove
    clean = frame.drop(index=dropped)
    expected = y_finance.compute_derived(clean, "ewma_vol", periods_per_year=252.0)

    monkeypatch.setattr(y_finance, "load_ohlcv", lambda s, d: frame.copy())
    got = y_finance._get_stock_stats_bulk("AAPL", "ewma_vol", "2026-08-11")

    for idx in clean.index:
        date_str = frame.loc[idx, "Date"].strftime("%Y-%m-%d")
        exp = expected.loc[idx]
        assert got[date_str] == ("N/A" if pd.isna(exp) else str(float(exp))), date_str
    # The dropped days themselves are present-but-N/A, not absent.
    assert got["2026-07-01"] == "N/A"
    assert got["2026-07-08"] == "N/A"
    assert len(got) == len(frame)


@pytest.mark.unit
def test_clean_frame_path_is_byte_equivalent(monkeypatch):
    """A well-formed frame (no rows dropped) keeps the exact pre-round-13
    mapping — the reindex branch is a no-op when the derived index already
    equals the frame index."""
    frame = _dirty_ohlcv().drop(index=[0, 5]).reset_index(drop=True)
    expected = y_finance.compute_derived(frame, "ewma_vol")

    monkeypatch.setattr(y_finance, "load_ohlcv", lambda s, d: frame.copy())
    got = y_finance._get_stock_stats_bulk("AAPL", "ewma_vol", "2026-08-11")

    # The literal pre-fix construction (dates zipped against the series).
    pre_fix = {
        d.strftime("%Y-%m-%d"): ("N/A" if pd.isna(v) else str(float(v)))
        for d, v in zip(frame["Date"], expected, strict=True)
    }
    assert got == pre_fix


@pytest.mark.unit
def test_all_rows_dropped_renders_all_na(monkeypatch):
    """Pathological frame where EVERY row fails the cleaner: an honest
    all-N/A wall instead of a ValueError (the empty estimator series used to
    fail the strict zip like any other length mismatch)."""
    frame = _dirty_ohlcv()
    frame["Low"] = 0.0  # every row non-positive -> empty cleaned frame
    monkeypatch.setattr(y_finance, "load_ohlcv", lambda s, d: frame.copy())
    out = y_finance.get_stock_stats_indicators_window(
        "AAPL", "ewma_vol", "2026-08-11", 45)
    assert "## ewma_vol values" in out
    assert "2026-08-11: N/A\n" in out


# --------------------------------------------------------------------------- #
# 2. [P2] overview leak gates — clamp before overview_would_leak_future
# --------------------------------------------------------------------------- #
_SEC_OVERVIEW = (
    "# Company Fundamentals for MU\n"
    "# Source: SEC EDGAR XBRL\n"
    "Name: Micron Technology Inc.\n"
    "Total Assets: 130,000,000,000  (period ending 2026-06-05)\n"
)


@pytest.mark.unit
def test_aggregate_overview_inflated_curr_date_clamped_to_pinned(monkeypatch):
    """Round-13: an LLM-echoed 'today' curr_date in a run pinned to a past
    analysis date must be clamped BEFORE the Yahoo leak gate judges it — the
    aggregate degrades to SEC-only exactly like the direct past-date call
    (pre-fix the gate saw live mode and appended today's snapshot)."""
    monkeypatch.setattr(fo, "_sec_fundamentals", lambda t, c: _SEC_OVERVIEW)

    def no_yahoo(t, c):
        raise AssertionError("Yahoo snapshot fetched on a clamped-past run")

    monkeypatch.setattr(fo, "_cached_ticker_info", no_yahoo)
    set_analysis_date("2026-01-15")
    try:
        clamped = fo.get_fundamentals("MU", _TODAY)   # the inflated echo
        direct = fo.get_fundamentals("MU", "2026-01-15")
    finally:
        set_analysis_date(None)
    assert clamped == direct
    assert "point-in-time guard" in clamped   # Yahoo half gated, not served
    assert "Market Cap" not in clamped


@pytest.mark.unit
def test_av_overview_inflated_curr_date_refused_after_clamp(monkeypatch):
    """Same clamp for the Alpha Vantage OVERVIEW gate: the echoed 'today'
    must not reach the endpoint once the run is pinned to a past date."""
    def boom(_fn, _params):
        raise AssertionError("OVERVIEW endpoint reached on a clamped-past run")

    monkeypatch.setattr(avf, "_make_api_request", boom)
    set_analysis_date("2026-01-15")
    try:
        with pytest.raises(NoMarketDataError, match="point-in-time"):
            avf.get_fundamentals("AAPL", curr_date=_TODAY)
    finally:
        set_analysis_date(None)


@pytest.mark.unit
def test_av_overview_live_mode_still_serves(monkeypatch):
    """Live mode (no pinned analysis date): today's curr_date passes the gate
    unchanged — the round-13 clamp is a no-op there, byte-equivalent."""
    monkeypatch.setattr(
        avf, "_make_api_request", lambda _fn, _params: {"Symbol": "AAPL"})
    assert avf.get_fundamentals("AAPL", curr_date=_TODAY) == {"Symbol": "AAPL"}


# --------------------------------------------------------------------------- #
# 3. [P3] five A-share entry points — the analysis-date clamp
# --------------------------------------------------------------------------- #
def _bs_row(d: str, close: str = "100.0", pe: str = "25.0") -> dict:
    return {
        "date": d, "code": "sh.600519", "open": "99.0", "high": "101.0",
        "low": "98.5", "close": close, "preclose": "99.5", "volume": "1000000",
        "amount": "100000000", "turn": "1.0", "pctChg": "0.5", "peTTM": pe,
        "pbMRQ": "3.0", "psTTM": "5.0", "pcfNcfTTM": "12.0", "isST": "0",
    }


def _tushare_basic_df() -> pd.DataFrame:
    return pd.DataFrame([
        {"trade_date": "20240614", "pe_ttm": 27.0, "pb": 8.1,
         "total_mv": 1.64e12, "circ_mv": 1.64e12, "turnover_rate": 0.2,
         "free_share": 0.0},
        {"trade_date": "20241220", "pe_ttm": 60.0, "pb": 18.0,   # future row
         "total_mv": 3.0e12, "circ_mv": 3.0e12, "turnover_rate": 0.5,
         "free_share": 0.0},
    ])


def _tushare_news_df() -> pd.DataFrame:
    return pd.DataFrame([
        {"title": "茅台发布年报", "datetime": "2024-06-14 18:00:00",
         "src": "sina", "content": "600519 净利润增长"},
        {"title": "茅台未来展望", "datetime": "2024-12-20 08:00:00",   # future
         "src": "eastmoney", "content": "600519"},
    ])


def _em_margin_json(rows: list[dict]) -> bytes:
    return json.dumps(
        {"version": "v1", "result": {"pages": 1, "data": rows},
         "success": True, "message": "ok", "code": 200}).encode("utf-8")


def _em_margin_row(d: str) -> dict:
    return {"DATE": f"{d} 00:00:00", "SCODE": "600519", "SECNAME": "Moutai",
            "RZYE": 1.0e10, "RQYE": 2.0e9, "RZRQYE": 1.2e10, "RZMRE": 5.0e8,
            "RQMCL": 100000, "RQYL": 1_000_000, "RZJME": 0.0, "RZYEZB": 0.5}


@pytest.mark.unit
@pytest.mark.parametrize("vendor", [
    "baostock_ohlc", "baostock_valuation",
    "tushare_valuation", "tushare_news",
    "eastmoney_margin",
])
def test_entry_clamps_inflated_curr_date_to_pinned_analysis_date(
    monkeypatch, tmp_path, vendor,
):
    """Round-13 (closes the round-11 gap on the daily-series/news entries):
    a curr_date beyond the run's pinned analysis date (an LLM echoing a
    cached header's 'today') must not return future rows — the clamped call
    is byte-identical to passing the pinned date itself."""
    pinned = "2024-06-15"
    future = "2024-12-20"

    if vendor in ("baostock_ohlc", "baostock_valuation"):
        rows = [_bs_row("2024-06-10"), _bs_row("2024-06-14"),
                _bs_row(future, close="200.0", pe="60.0")]
        monkeypatch.setattr(
            bsv, "_cached_daily", lambda code, curr_date=None: list(rows))
        fn = (bsv.get_a_share_ohlc_native if vendor == "baostock_ohlc"
              else bsv.get_a_share_fundamentals_native)

        def call(curr: str) -> str:
            return fn("600519.SS", curr, 3650)
    elif vendor == "tushare_valuation":
        pro = SimpleNamespace(daily_basic=lambda **kw: _tushare_basic_df())
        monkeypatch.setattr(tv, "_require_tushare", lambda: pro)

        def call(curr: str) -> str:
            return tv.get_a_share_fundamentals_native("600519.SS", curr, 3650)
    elif vendor == "tushare_news":
        pro = SimpleNamespace(news=lambda **kw: _tushare_news_df())
        monkeypatch.setattr(tv, "_require_tushare", lambda: pro)
        # Passthrough cache so the (clamped) window's fetch always runs and
        # the client-side PIT filter is the only gate under test.
        monkeypatch.setattr(tv, "cached_or_fetch",
                            lambda base, name, fetch, **kw: fetch())

        def call(curr: str) -> str:
            return tv.get_a_share_news_native("600519.SS", curr, 3650)
    else:  # eastmoney_margin
        monkeypatch.setattr(em, "_cache_dir", lambda: str(tmp_path))
        payload = _em_margin_json([
            _em_margin_row("2024-06-10"), _em_margin_row("2024-06-15"),
            _em_margin_row(future),                      # future row
        ])
        monkeypatch.setattr(
            em, "_cached_or_fetch",
            lambda _path, _url, params=None, ttl_days=1.0: payload)

        def call(curr: str) -> str:
            return em.get_margin_trading("600519.SS", curr, 3650)

    set_analysis_date(pinned)
    try:
        clamped = call(_TODAY)      # the inflated echo (far beyond pinned)
        direct = call(pinned)
    finally:
        set_analysis_date(None)
    assert clamped == direct
    assert future not in clamped   # no future row / headline leaks through


# --------------------------------------------------------------------------- #
# 4. [P3] BaoStock same-day/finality TTL — Beijing calendar anchor
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_cn_today_is_anchored_on_beijing():
    """The TTL anchor is the exchange's own calendar (Asia/Shanghai), not the
    host clock — same class of fix as stockstats_utils._utc_today."""
    assert bsv._cn_today() == datetime.now(ZoneInfo("Asia/Shanghai")).date()


@pytest.mark.unit
@pytest.mark.parametrize("skew_days", [-1, 1])
def test_same_day_ttl_anchors_on_beijing_not_host(monkeypatch, skew_days):
    """Frozen at both directions of a host/Beijing calendar skew: the
    same-day decision follows the BEIJING date in both cases (pre-fix the
    host clock decided — freezing the exchange's current day at 1-day TTL on
    one skew and refreshing a finished day on the other)."""
    host_today = date.today()
    beijing_today = host_today + timedelta(days=skew_days)
    assert beijing_today != host_today  # the skew is the point
    monkeypatch.setattr(bsv, "_cn_today", lambda: beijing_today)

    # The Beijing calendar date is same-day regardless of the host clock.
    assert bsv._daily_cache_ttl_days(beijing_today.isoformat()) == pytest.approx(
        bsv._SAME_DAY_REFRESH_S / 86_400.0)
    # The HOST's today is NOT same-day while the calendars are skewed.
    assert bsv._daily_cache_ttl_days(host_today.isoformat()) == pytest.approx(
        bsv._CACHE_TTL_S / 86_400.0)
    # Live mode (no curr_date) follows the Beijing anchor too.
    assert bsv._daily_cache_ttl_days(None) == pytest.approx(
        bsv._SAME_DAY_REFRESH_S / 86_400.0)

    # Finality gate (终局判定) keys on the Beijing date as well: a cached bar
    # dated the Beijing today finalizes the entry even while the host
    # calendar disagrees (pre-fix it looked for the HOST today's bar).
    with_bar = json.dumps(
        [{"date": beijing_today.isoformat(), "close": "5.92"}]).encode()
    assert bsv._daily_cache_ttl_days(beijing_today.isoformat(), with_bar) == \
        pytest.approx(bsv._CACHE_TTL_S / 86_400.0)
    # ...and a bar dated only the HOST today does not finalize the entry.
    host_bar = json.dumps(
        [{"date": host_today.isoformat(), "close": "5.92"}]).encode()
    assert bsv._daily_cache_ttl_days(beijing_today.isoformat(), host_bar) == \
        pytest.approx(bsv._SAME_DAY_REFRESH_S / 86_400.0)


# --------------------------------------------------------------------------- #
# 5. [P3] Tushare news — an empty frame must not poison the 30-minute cache
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_tushare_news_empty_result_writes_no_cache_file(monkeypatch):
    """Round-13: an empty news frame must not be persisted — pre-fix it
    round-tripped as b"" for the ~30-minute TTL, so headlines published in
    the following minutes stayed invisible (and the retry never re-queried)."""
    calls = {"n": 0}

    def _news(**kw):
        calls["n"] += 1
        return pd.DataFrame()

    monkeypatch.setattr(tv, "_require_tushare", lambda: SimpleNamespace(news=_news))

    first = tv.get_a_share_news_native("600519.SS", "2020-03-10", 14)
    assert "No news items" in first
    cache_dir = Path(tv.vendor_cache_dir("tushare"))
    assert not [p for p in cache_dir.iterdir() if p.is_file()]  # nothing written

    # Nothing poisoned the cache: the retry re-queries the API.
    second = tv.get_a_share_news_native("600519.SS", "2020-03-10", 14)
    assert "No news items" in second
    assert calls["n"] == 2


@pytest.mark.unit
def test_tushare_news_existing_cache_not_overwritten_by_empty(monkeypatch):
    """A primed news cache must survive an empty re-fetch: with the TTL
    expired (forced to 0 here) the empty marker serves the stale bytes
    (old headlines beat no headlines) and the cache file keeps the real CSV —
    never a b"" clobber."""
    state = {"empty": False}

    def _news(**kw):
        return pd.DataFrame() if state["empty"] else _tushare_news_df()

    monkeypatch.setattr(tv, "_require_tushare", lambda: SimpleNamespace(news=_news))
    monkeypatch.setattr(tv, "_NEWS_CACHE_TTL_DAYS", 0.0)  # every call refetches

    first = tv.get_a_share_news_native("600519.SS", "2024-06-15", 14)
    assert "茅台发布年报" in first                       # primes the cache

    cache_dir = Path(tv.vendor_cache_dir("tushare"))
    cached = [p for p in cache_dir.iterdir() if p.is_file()]
    assert len(cached) == 1
    assert "茅台发布年报".encode() in cached[0].read_bytes()

    state["empty"] = True
    second = tv.get_a_share_news_native("600519.SS", "2024-06-15", 14)
    assert "茅台发布年报" in second                       # stale-serve beats empty
    assert "茅台发布年报".encode() in cached[0].read_bytes()  # file not clobbered


# --------------------------------------------------------------------------- #
# 6. [P3] Eastmoney pagination — terminate on the API's RAW row count
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_margin_full_page_with_invalid_date_row_keeps_paginating(
    monkeypatch, tmp_path,
):
    """Round-13: a FULL 400-row API page carrying one empty-DATE row parsed
    to 399 rows and was misread as the report's last page — older history
    silently truncated out of long look-back windows. Termination must count
    the API's raw rows, so pagination continues to page 2 here."""
    newest = date(2024, 6, 20)

    def _daily_rows(count: int, start: date) -> list[dict]:
        return [_em_margin_row((start - timedelta(days=i)).isoformat())
                for i in range(count)]

    page1 = _daily_rows(400, newest)
    page1[7]["DATE"] = ""                 # one invalid row; raw count still 400
    page2 = _daily_rows(400, newest - timedelta(days=400))
    pages = {"1": page1, "2": page2}
    fetched: list[str] = []

    monkeypatch.setattr(em, "_cache_dir", lambda: str(tmp_path))

    def fake_fetch(_path, _url, params=None, ttl_days=1.0):
        fetched.append(params["pageNumber"])
        return _em_margin_json(pages[params["pageNumber"]])

    monkeypatch.setattr(em, "_cached_or_fetch", fake_fetch)

    # look_back 700d: page 1's oldest row (~2023-05-17) does not yet cover the
    # lower bound (2024-06-20 - 700d), so pagination must reach page 2, whose
    # oldest row (~2022-04-13) does.
    out = em.get_margin_trading("600519.SS", "2024-06-20", 700)
    assert fetched == ["1", "2"]          # page 2 reached despite the 399 parse
    page2_only = (newest - timedelta(days=400)).isoformat()
    assert page2_only in out              # page-2 history is in the window


@pytest.mark.unit
def test_margin_still_terminates_on_genuinely_short_page(monkeypatch, tmp_path):
    """The raw-count rule must not over-paginate: a full VALID page followed
    by a genuinely short page stops exactly there (no page-3 fan-out)."""
    newest = date(2024, 6, 20)

    def _daily_rows(count: int, start: date) -> list[dict]:
        return [_em_margin_row((start - timedelta(days=i)).isoformat())
                for i in range(count)]

    pages = {"1": _daily_rows(400, newest),
             "2": _daily_rows(10, newest - timedelta(days=400))}
    fetched: list[str] = []

    monkeypatch.setattr(em, "_cache_dir", lambda: str(tmp_path))

    def fake_fetch(_path, _url, params=None, ttl_days=1.0):
        fetched.append(params["pageNumber"])
        return _em_margin_json(pages[params["pageNumber"]])

    monkeypatch.setattr(em, "_cached_or_fetch", fake_fetch)
    em.get_margin_trading("600519.SS", "2024-06-20", 700)
    assert fetched == ["1", "2"]          # 10 raw rows < 400 -> last page


if __name__ == "__main__":
    unittest.main()
