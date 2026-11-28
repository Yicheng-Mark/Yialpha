"""Round-16 regression + coverage tests for the ``vendors_equity`` shard.

Owned modules: ``y_finance`` / ``yfinance_news`` / ``sec_edgar`` /
``sec_ownership`` / ``alpha_vantage*`` / ``fred`` / ``reddit`` / ``stocktwits`` /
``tavily`` / ``fundamentals_overview``.

Hermetic: every vendor call is mocked at the project's own seam (patched in
the namespace where the name is *used*), no sleeps, dates frozen by
``curr_date`` injection.
"""

from __future__ import annotations

import io
import json
import zipfile

import pytest
import requests

from yialpha.dataflows import sec_edgar, sec_ownership
from yialpha.dataflows.errors import NoMarketDataError, VendorRateLimitError
from yialpha.dataflows.sec_edgar import SecNoFileError

# --------------------------------------------------------------------------- #
# Shared synthetic payloads
# --------------------------------------------------------------------------- #
_R16_TICKERS = json.dumps({
    "0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."},
}).encode("utf-8")

# Two quarterly Assets facts; the second is filed AFTER the first one's
# filing date, so a leaky as-of gate shows the "2024-06-29" column.
_R16_FACTS = json.dumps({
    "cik": 320193,
    "entityName": "Apple Inc.",
    "facts": {"us-gaap": {"Assets": {"units": {"USD": [
        {"end": "2024-03-30", "val": 34700000000, "fy": 2024, "fp": "Q2",
         "form": "10-Q", "filed": "2024-05-02"},
        {"end": "2024-06-29", "val": 35300000000, "fy": 2024, "fp": "Q3",
         "form": "10-Q", "filed": "2024-08-01"},
    ]}}}},
}).encode("utf-8")

# Forms an LLM is documented to emit instead of YYYY-MM-DD. Each one denotes
# 2024-05-03 (or is not a date at all) — none may make a later filing visible.
# ("2024-5-3" is NOT in this list: strptime accepts a non-zero-padded month/day,
# so the project treats it as the valid date 2024-05-03 — see the
# non-padded test below, where it must still not be compared lexically.)
_MALFORMED_DATES = ["2024/05/03", "20240503", "2024-13-45", "garbage",
                    "05/03/2024"]


def _r16_patch_sec(monkeypatch, facts: bytes = _R16_FACTS):
    """Serve synthetic SEC payloads; any other URL is a hard failure."""
    def fetch(_path, url, ttl_days):
        if url.endswith("company_tickers.json"):
            return _R16_TICKERS
        assert "companyfacts" in url, f"unexpected SEC url: {url}"
        return facts

    monkeypatch.setattr(sec_edgar, "_cached_or_fetch", fetch)
    monkeypatch.setattr(sec_edgar, "_throttle", lambda: None)   # no real sleeps


def _r16_zip(member: str, content: bytes) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(member, content)
    return buf.getvalue()


def _r16_patch_ftd(monkeypatch, tmp_path, content: bytes, ym="202405", half="b"):
    """Serve one synthetic cnsfails ZIP; 404 every other URL; record URLs."""
    monkeypatch.setattr(sec_ownership, "_cache_dir", lambda: str(tmp_path))
    requested: list[str] = []
    key = f"cnsfails{ym}{half}.zip"
    payload = _r16_zip(f"cnsfails{ym}{half}", content)

    def fetch(_path, url, ttl_days):
        requested.append(url)
        if key in url:
            return payload
        raise SecNoFileError(url, detail="404")

    monkeypatch.setattr(sec_ownership, "_cached_or_fetch", fetch)
    return requested


# --------------------------------------------------------------------------- #
# sec_edgar: PIT gate on malformed dates
# --------------------------------------------------------------------------- #
@pytest.mark.unit
@pytest.mark.parametrize("bad_date", _MALFORMED_DATES)
def test_r16_sec_period_public_malformed_curr_date_fails_closed(bad_date):
    """A curr_date that is not YYYY-MM-DD must make nothing visible: the old
    lexicographic compare ranked "2024-11-15" <= "2024/05/03" ('-' < '/') and
    so exposed filings filed months AFTER the requested as-of date."""
    rec = {"end": "2024-09-28", "filed": "2024-11-15", "val": 1}
    assert sec_edgar._period_public(rec, bad_date) is False


@pytest.mark.unit
@pytest.mark.parametrize(
    "filed,expected",
    [
        ("2024-05-10", False),       # filed after the nominal 2024-05-03
        ("2024-05-03", True),
        ("2024-04-30", True),
    ],
)
def test_r16_sec_period_public_non_padded_date_is_parsed_not_compared_lexically(
    filed, expected
):
    """A non-zero-padded as-of date is the real date 2024-05-03. The old
    string compare ranked "2024-05-10" <= "2024-5-3" ('0' < '5'), exposing
    every filing through 2024-12-31 as of early May."""
    rec = {"end": "2024-03-30", "filed": filed, "val": 1}
    assert sec_edgar._period_public(rec, "2024-5-3") is expected


@pytest.mark.unit
@pytest.mark.parametrize(
    "curr_date,expected",
    [
        ("2024-05-01", False),       # day before the filing
        ("2024-05-02", True),        # filed on the as-of date: visible
        ("2024-05-03", True),
        ("2024-05-02 21:30:00", True),   # datetime label: date part only
        ("2024-05-02T21:30:00+00:00", True),
    ],
)
def test_r16_sec_period_public_boundary_unchanged_for_valid_dates(curr_date, expected):
    """Valid ISO dates keep the ground-truth gate (filed <= curr_date); only
    the first ten characters matter, so datetime labels behave like dates."""
    rec = {"end": "2024-03-30", "filed": "2024-05-02", "val": 1}
    assert sec_edgar._period_public(rec, curr_date) is expected


@pytest.mark.unit
def test_r16_sec_period_public_malformed_filed_falls_back_to_heuristic():
    """A ``filed`` value that is not a date cannot be compared, so the gate
    falls back to the 45-day ``is_filing_public`` heuristic on ``end``:
    2024-03-30 + 45 = 2024-05-14, visible ON that day (not the day before)."""
    rec = {"end": "2024-03-30", "filed": "02/05/2024", "val": 1}
    assert sec_edgar._period_public(rec, "2024-05-15") is True
    assert sec_edgar._period_public(rec, "2024-05-14") is True
    assert sec_edgar._period_public(rec, "2024-05-13") is False


@pytest.mark.unit
@pytest.mark.parametrize("bad_date", ["2024/05/03", "garbage"])
def test_r16_sec_statement_malformed_curr_date_never_renders_a_filing(
    monkeypatch, bad_date
):
    """Public entries: a malformed as-of date degrades to the typed
    NoMarketDataError (router NO_DATA sentinel) instead of rendering the
    filings that were filed after the nominal date."""
    _r16_patch_sec(monkeypatch)
    for entry in (
        lambda d: sec_edgar.get_balance_sheet("AAPL", "quarterly", d),
        lambda d: sec_edgar.get_income_statement("AAPL", "quarterly", d),
        lambda d: sec_edgar.get_cashflow("AAPL", "quarterly", d),
    ):
        with pytest.raises(NoMarketDataError, match="no SEC quarterly filings public"):
            entry(bad_date)
    # The snapshot entry renders "n/a" rows rather than raising.
    out = sec_edgar.get_fundamentals("AAPL", bad_date)
    assert "Total Assets: n/a" in out
    assert "(period ending" not in out


@pytest.mark.unit
def test_r16_sec_statement_valid_curr_date_shows_only_filed_records(monkeypatch):
    """Control for the malformed-date tests: at 2024-05-03 the 10-Q filed
    2024-05-02 is the only visible period; the one filed 2024-08-01 is not."""
    _r16_patch_sec(monkeypatch)
    out = sec_edgar.get_balance_sheet("AAPL", "quarterly", "2024-05-03")
    assert ",2024-03-30" in out
    assert "34700000000" in out
    assert "2024-06-29" not in out
    assert "35300000000" not in out


# --------------------------------------------------------------------------- #
# sec_edgar: transport failure vs. genuine 404
# --------------------------------------------------------------------------- #
class _Resp:
    def __init__(self, status: int, content: bytes = b""):
        self.status_code = status
        self.content = content


@pytest.mark.unit
@pytest.mark.parametrize(
    "status",
    [400, 401, 408, 410, 418, 451, 500, 502, 503, 504],
)
def test_r16_sec_get_non_404_errors_are_never_a_true_missing_artifact(
    monkeypatch, status
):
    """Only HTTP 404 may be read as "the artifact genuinely does not exist".
    Every other error status maps to a plain NoMarketDataError that is NOT a
    SecNoFileError, so no caller may skip it as a missing file."""
    monkeypatch.setattr(sec_edgar, "_throttle", lambda: None)
    monkeypatch.setattr(requests, "get", lambda *a, **k: _Resp(status))
    with pytest.raises(NoMarketDataError) as excinfo:
        sec_edgar._sec_get("https://data.sec.gov/api/xbrl/x")
    assert not isinstance(excinfo.value, SecNoFileError)
    assert f"SEC HTTP {status}" in str(excinfo.value)


@pytest.mark.unit
@pytest.mark.parametrize("status", [429, 403])
def test_r16_sec_get_throttle_statuses_escape_no_data_handlers(
    monkeypatch, status
):
    """429/403 raise VendorRateLimitError, which is NOT a NoMarketDataError:
    the ``except NoMarketDataError`` skip sites in sec_ownership must not
    swallow a throttle as a skipped filing."""
    monkeypatch.setattr(sec_edgar, "_throttle", lambda: None)
    monkeypatch.setattr(requests, "get", lambda *a, **k: _Resp(status))
    with pytest.raises(VendorRateLimitError) as excinfo:
        sec_edgar._sec_get("https://data.sec.gov/api/xbrl/x")
    assert not isinstance(excinfo.value, NoMarketDataError)
    assert str(status) in str(excinfo.value)


@pytest.mark.unit
def test_r16_sec_get_404_is_the_only_true_missing_signal(monkeypatch):
    """404 raises SecNoFileError (a NoMarketDataError subclass, so older
    ``except`` sites keep working) carrying the URL."""
    monkeypatch.setattr(sec_edgar, "_throttle", lambda: None)
    monkeypatch.setattr(requests, "get", lambda *a, **k: _Resp(404))
    with pytest.raises(SecNoFileError) as excinfo:
        sec_edgar._sec_get("https://data.sec.gov/api/xbrl/companyfacts/CIK0000000000.json")
    assert isinstance(excinfo.value, NoMarketDataError)
    assert "CIK0000000000.json" in str(excinfo.value)


@pytest.mark.unit
@pytest.mark.parametrize(
    "exc",
    [
        requests.exceptions.Timeout("read timed out"),
        requests.exceptions.ProxyError("proxy refused"),
        requests.exceptions.SSLError("bad handshake"),
        requests.exceptions.ConnectionError("name resolution failed"),
        requests.exceptions.ChunkedEncodingError("truncated"),
    ],
)
def test_r16_sec_get_transport_failure_is_not_a_404(monkeypatch, exc):
    """A transport error is a plain NoMarketDataError — never SecNoFileError —
    so an outage cannot be mistaken for "the filing is not there"."""
    monkeypatch.setattr(sec_edgar, "_throttle", lambda: None)

    def _boom(*a, **k):
        raise exc

    monkeypatch.setattr(requests, "get", _boom)
    with pytest.raises(NoMarketDataError) as excinfo:
        sec_edgar._sec_get("https://data.sec.gov/files/company_tickers.json")
    assert not isinstance(excinfo.value, SecNoFileError)
    assert "SEC request failed" in str(excinfo.value)


@pytest.mark.unit
def test_r16_sec_get_sends_ua_and_proxy_and_returns_bytes(monkeypatch):
    """The SEC fair-access User-Agent and the configured proxies ride every
    request; a 200 returns the raw body bytes."""
    monkeypatch.setattr(sec_edgar, "_throttle", lambda: None)
    seen = {}

    def _get(url, headers=None, proxies=None, timeout=None):
        seen.update(url=url, headers=headers, proxies=proxies, timeout=timeout)
        return _Resp(200, b'{"ok": 1}')

    monkeypatch.setattr(requests, "get", _get)
    assert sec_edgar._sec_get("https://data.sec.gov/x") == b'{"ok": 1}'
    assert seen["url"] == "https://data.sec.gov/x"
    assert seen["headers"]["User-Agent"] == sec_edgar._user_agent()
    assert seen["timeout"] == sec_edgar._TIMEOUT
    assert "proxies" in seen


# --------------------------------------------------------------------------- #
# sec_ownership: FTD malformed-row handling
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_r16_sec_ftd_row_with_unparseable_quantity_does_not_crash(
    monkeypatch, tmp_path
):
    """A ticker row whose QUANTITY (FAILS) cell is blank still renders (the
    source row exists for that settlement date); the peak line reports the
    quantity as n/a instead of raising TypeError on ``None:,.0f``."""
    content = (b"SETTLEMENT DATE|CUSIP|SYMBOL|QUANTITY (FAILS)|DESCRIPTION|PRICE\n"
               b"20240510|037833100|AAPL||APPLE INC|195.20\n")
    _r16_patch_ftd(monkeypatch, tmp_path, content)
    out = sec_ownership.get_ftd_data("AAPL", "2024-06-20", 45)
    assert "1 fail-day(s)" in out
    assert "20240510" in out
    assert "Peak fail-day: 20240510 (n/a fails @ $195.20)" in out


@pytest.mark.unit
def test_r16_sec_ftd_mixed_rows_peak_ignores_unparseable_quantity(
    monkeypatch, tmp_path
):
    """With one healthy and one blank-quantity row the peak is the healthy
    row's quantity; the blank row is still counted as a fail-day."""
    content = (b"SETTLEMENT DATE|CUSIP|SYMBOL|QUANTITY (FAILS)|DESCRIPTION|PRICE\n"
               b"20240510|037833100|AAPL|1234567|APPLE INC|195.20\n"
               b"20240513|037833100|AAPL||APPLE INC|195.20\n")
    _r16_patch_ftd(monkeypatch, tmp_path, content)
    out = sec_ownership.get_ftd_data("AAPL", "2024-06-20", 45)
    assert "2 fail-day(s)" in out
    assert "Peak fail-day: 20240510 (1,234,567 fails @ $195.20)" in out
