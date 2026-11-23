"""Round-14 type-fix regressions for ``sec_edgar`` and ``coingecko_vendor``.

Two behaviors locked here were exposed by mypy's union analysis:

1. ``sec_edgar.get_fundamentals``: ``_pick_period_record`` genuinely returns
   None when no record's ``r.get("end")`` equals the requested ``end``. The
   ``_latest`` closure computes ``end = max(r.get("end", "") for r in recs)``,
   so when every PIT-surviving record LACKS an "end" key the max is the ""
   default while ``r.get("end")`` is None — ``None == ""`` matches nothing,
   and the pre-fix ``chosen["val"]`` crashed with TypeError. The fix degrades
   to the established ``(None, "")`` -> "n/a" row instead.
2. ``coingecko_vendor.get_crypto_token_fundamentals``: the historical branch
   now passes the VALIDATED ``str(curr_date)[:10]`` slice to
   ``_render_historical`` (which only ever consumed that same slice), so a
   curr_date longer than yyyy-mm-dd (e.g. with a time suffix) must render the
   identical as-of output as before the refactor.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest

from yialpha.dataflows import coingecko_vendor as cg, sec_edgar

# --------------------------------------------------------------------------- #
# 1. sec_edgar: fundamentals snapshot survives records without an "end" key
# --------------------------------------------------------------------------- #
_TICKERS_JSON = json.dumps({
    "0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."},
}).encode("utf-8")

#: Every fact has a val but NO "end" key — the exact payload shape under
#: which the pre-fix code raised TypeError('NoneType' object is not
#: subscriptable) at the ``chosen["val"]`` line.
_FACTS_NO_END_JSON = json.dumps({
    "cik": 320193,
    "entityName": "Apple Inc.",
    "facts": {"us-gaap": {
        "Assets": {"units": {"USD": [
            {"val": 36498000000, "fy": 2024, "fp": "FY",
             "form": "10-K", "filed": "2024-11-01"},
        ]}},
    }},
}).encode("utf-8")


def _patch_fetch_no_end(monkeypatch, tmp_path):
    """Point the vendor at the end-less synthetic payload; no network."""
    monkeypatch.setattr(sec_edgar, "_cache_dir", lambda: str(tmp_path))

    def fake_cached_or_fetch(_path, url, ttl_days):
        if url.endswith("company_tickers.json"):
            return _TICKERS_JSON
        if "companyfacts" in url:
            return _FACTS_NO_END_JSON
        raise AssertionError(f"unexpected url: {url}")

    monkeypatch.setattr(sec_edgar, "_cached_or_fetch", fake_cached_or_fetch)


@pytest.mark.unit
def test_pick_period_record_returns_none_when_records_lack_end_key():
    """Mechanism check: records without an "end" key can never match the ""
    default that max(r.get("end", "")) produces — so None IS reachable."""
    assert sec_edgar._pick_period_record(
        [{"val": 1, "filed": "2024-01-01"}], "", 90
    ) is None


@pytest.mark.unit
def test_fundamentals_degrades_to_na_when_records_lack_end_key(
    monkeypatch, tmp_path,
):
    """get_fundamentals must render "n/a" rows, not crash, when the only
    public records carry a val but no period-end key (was: TypeError at
    ``chosen["val"]`` because _pick_period_record returned None)."""
    _patch_fetch_no_end(monkeypatch, tmp_path)
    out = sec_edgar.get_fundamentals("AAPL")  # live mode: every fact public
    assert "Total Assets: n/a" in out
    assert "Total Equity: n/a" in out
    # The snapshot still carries the header/identity lines it always did.
    assert out.startswith("# Company Fundamentals for AAPL")
    assert "Name: Apple Inc." in out


# --------------------------------------------------------------------------- #
# 2. coingecko_vendor: historical branch consumes the validated day slice
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_historical_datetime_suffix_curr_date_renders_as_of_day(monkeypatch):
    """A curr_date with a time suffix ("2026-06-28T12:34:56") must produce the
    SAME historical render as the bare day: the vendor slices to [:10], and
    that validated slice — not the raw str|None — is what flows into
    _render_historical (regression lock for the type fix)."""
    from yialpha.dataflows.disk_cache import MinIntervalThrottle

    monkeypatch.setattr(cg, "_throttler", MinIntervalThrottle(0.0))

    def fake(url, params=None):
        if "/search" in url:
            return {"coins": [
                {"id": "bitcoin", "symbol": "btc", "market_cap_rank": 1},
            ]}
        ts = int(datetime(2026, 6, 27, 12, tzinfo=UTC).timestamp() * 1000)
        return {
            "prices": [[ts, 100.0]],
            "market_caps": [[ts, 2.0e12]],
            "total_volumes": [[ts, 5.0e8]],
        }

    monkeypatch.setattr(cg, "_http_get_json", fake)
    out = cg.get_crypto_token_fundamentals(
        "BTCUSDT", curr_date="2026-06-28T12:34:56", look_back_days=30,
    )
    # Header/window anchor on the yyyy-mm-dd slice, never the full timestamp.
    assert "point-in-time series as of 2026-06-28" in out
    assert "2026-06-28T12:34:56" not in out
    assert "| 2026-06-27 |" in out  # the row survives the window filter
