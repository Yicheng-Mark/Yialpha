"""Round-15 hardening tests for ``sec_ownership`` / ``sec_edgar`` / ``tavily``.

Targets the fresh-coverage gaps that the existing suites leave open:

* ``sec_ownership`` — Form 4 ``files[]`` archive backfill (segment scans,
  unnamed/404/unparseable segments, budget break, disclosure note), the
  25-filing fetch budget, per-filing degrade predicates, FTD look-back cap /
  year rollover / empty-ZIP / symbol-less rows, 13F helper edges, and the
  ``_extract_cusip`` PIT record gates.
* ``sec_edgar`` — ``_sec_get`` 5xx/success mapping, unparsable payload
  degrades, val-less concept records, the no-filed heuristic fallback,
  ``_fmt`` unit kinds, and ``get_cashflow`` (previously never called).
* ``tavily`` — lazy per-context state init (counters / key cursor / dead-key
  set) in a brand-new ``contextvars.Context``, all-keys-dead ``_pick_key``,
  timeout env parsing, ``days`` -> news topic, non-JSON body, published_date.

PIT publication-lag gates are pinned at their exact boundary: an FTD
half-month file (default ``ftd_pub_lag_days=10``) and a 13F bulk dataset
(default ``sec_13f_pub_lag_days=45``) become visible the day
``half/period end + lag == curr_date`` and are invisible the day before
(never even requested).

Hermetic: no network, no LLM, no sleeps. The SEC seam is patched at
``_cached_or_fetch`` / ``_sec_get`` / ``_fetch_form4_xml`` and the Tavily
client at ``requests.post`` (fresh ``contextvars.Context()`` objects never
touch the shared run state).
"""

from __future__ import annotations

import contextvars
import io
import json
import os
import zipfile
from datetime import date
from unittest import mock

import pytest
import requests

from yialpha.dataflows import quality, sec_edgar, sec_ownership, tavily
from yialpha.dataflows.errors import NoMarketDataError
from yialpha.dataflows.sec_edgar import SecNoFileError

# --------------------------------------------------------------------------- #
# sec_ownership shared fixtures / builders
# --------------------------------------------------------------------------- #
CIK = 320193


def _r15_form4_xml(owner="R15 INSIDER", title=None, code="P", shares="1000",
                   price="10.00", tdate="2024-05-01", post="5000"):
    """Minimal parseable Form 4 XML (one non-derivative transaction).

    ``title=None`` omits the officerTitle element entirely (missing-element
    branch of ``_txt``)."""
    title_el = f"<officerTitle>{title}</officerTitle>" if title else ""
    return (
        "<ownershipDocument><reportingOwner>"
        f"<reportingOwnerId><rptOwnerName>{owner}</rptOwnerName></reportingOwnerId>"
        f"<reportingOwnerRelationship>{title_el}</reportingOwnerRelationship>"
        "</reportingOwner><nonDerivativeTable><nonDerivativeTransaction>"
        f"<transactionDate><value>{tdate}</value></transactionDate>"
        f"<transactionCoding><transactionCode>{code}</transactionCode></transactionCoding>"
        "<transactionAmounts>"
        f"<transactionShares><value>{shares}</value></transactionShares>"
        f"<transactionPricePerShare><value>{price}</value></transactionPricePerShare>"
        "</transactionAmounts>"
        "<postTransactionAmounts>"
        f"<sharesOwnedFollowingTransaction><value>{post}</value>"
        "</sharesOwnedFollowingTransaction>"
        "</postTransactionAmounts>"
        "</nonDerivativeTransaction></nonDerivativeTable></ownershipDocument>"
    ).encode()


def _r15_subs(recent: dict, files: list | None = None) -> bytes:
    return json.dumps({
        "cik": CIK,
        "filings": {"recent": recent, "files": files or []},
    }).encode()


def _r15_head(fillers: int, form4s: int, filing_date="2024-05-01") -> dict:
    """A submissions ``recent`` head: ``form4s`` in-window Form 4s first, then
    ``fillers`` 8-K fillers (to reach the ~1000 cap when needed)."""
    n4, n8 = form4s, fillers
    return {
        "form": ["4"] * n4 + ["8-K"] * n8,
        "accessionNumber": [f"0000320193-24-{i:06d}" for i in range(n4 + n8)],
        "filingDate": [filing_date] * (n4 + n8),
        "primaryDocument": [f"f{i:03d}.xml" for i in range(n4 + n8)],
    }


def _r15_patch_form4(monkeypatch, tmp_path, subs: bytes, xml_by_doc=None,
                     form4_fetch=None):
    """Serve ``subs`` for the head submissions URL; route Form 4 XMLs through
    ``xml_by_doc`` (by doc substring) or a callable ``form4_fetch``."""
    monkeypatch.setattr(sec_ownership, "_cache_dir", lambda: str(tmp_path))

    def fetch(_path, url, ttl_days):
        assert "submissions" in url, f"unexpected url: {url}"
        return subs

    monkeypatch.setattr(sec_ownership, "_cached_or_fetch", fetch)

    if form4_fetch is not None:
        monkeypatch.setattr(sec_ownership, "_fetch_form4_xml", form4_fetch)
    else:
        def _xml(cik, accession, doc):
            for key, payload in (xml_by_doc or {}).items():
                if key in doc:
                    return payload
            raise AssertionError(f"unexpected form4 doc: {doc}")

        monkeypatch.setattr(sec_ownership, "_fetch_form4_xml", _xml)


def _r15_ftd_zip(member: str, content: bytes) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(member, content)
    return buf.getvalue()


_R15_FTD_PIPE = (
    b"SETTLEMENT DATE|CUSIP|SYMBOL|QUANTITY (FAILS)|DESCRIPTION|PRICE\n"
    b"20240510|037833100|AAPL|1234567|APPLE INC|195.20\n"
    b"20240512|594918104|MSFT|999999|MICROSOFT CORP|420.00\n"
)


def _r15_patch_ftd(monkeypatch, tmp_path, content=_R15_FTD_PIPE,
                   ym="202405", half="b", as_zip=True):
    """Serve one synthetic cnsfails ZIP; 404 every other URL; record URLs."""
    monkeypatch.setattr(sec_ownership, "_cache_dir", lambda: str(tmp_path))
    requested: list[str] = []
    key = f"cnsfails{ym}{half}.zip"
    payload = _r15_ftd_zip(f"cnsfails{ym}{half}", content) if as_zip else content

    def fetch(_path, url, ttl_days):
        requested.append(url)
        if key in url:
            return payload
        raise SecNoFileError(url, detail="404")

    monkeypatch.setattr(sec_ownership, "_cached_or_fetch", fetch)
    return requested


# --------------------------------------------------------------------------- #
# sec_ownership: publication-lag config parsing
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_r15_pub_lag_garbage_config_falls_back_to_defaults(monkeypatch):
    """A non-int pub-lag value (like a negative one) falls back to the
    conservative default instead of guessing a shorter lag."""
    from yialpha.dataflows import config as cfgmod

    orig = cfgmod.get_config()
    try:
        cfgmod.set_config({**orig, "ftd_pub_lag_days": "ten",
                           "sec_13f_pub_lag_days": "many"})
        assert sec_ownership._ftd_pub_lag_days() == 10
        assert sec_ownership._sec_13f_pub_lag_days() == 45
    finally:
        cfgmod.set_config(orig)


# --------------------------------------------------------------------------- #
# sec_ownership: submissions fetch degrade
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_r15_fetch_submissions_bad_json_degrades(monkeypatch, tmp_path):
    monkeypatch.setattr(sec_ownership, "_cache_dir", lambda: str(tmp_path))
    monkeypatch.setattr(sec_ownership, "_cached_or_fetch",
                        lambda path, url, ttl_days: b"{not-json")
    with pytest.raises(NoMarketDataError, match="could not parse submissions"):
        sec_ownership._fetch_submissions(CIK)


@pytest.mark.unit
def test_r15_fetch_submissions_segment_bad_json_degrades(monkeypatch, tmp_path):
    monkeypatch.setattr(sec_ownership, "_cache_dir", lambda: str(tmp_path))
    monkeypatch.setattr(sec_ownership, "_cached_or_fetch",
                        lambda path, url, ttl_days: b"{not-json")
    with pytest.raises(NoMarketDataError,
                       match="could not parse submissions segment"):
        sec_ownership._fetch_submissions_segment(CIK, "CIK0000320193-submissions-001.json")


# --------------------------------------------------------------------------- #
# sec_ownership: Form 4 parse guards + per-filing degrade
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_r15_parse_form4_refuses_dtd_and_malformed_xml():
    dtd = (b"<?xml version='1.0'?><!DOCTYPE ownershipDocument ["
           b"<!ENTITY x 'y'>]><ownershipDocument/>")
    with pytest.raises(NoMarketDataError, match="could not parse Form 4 XML"):
        sec_ownership._parse_form4(dtd)
    with pytest.raises(NoMarketDataError, match="could not parse Form 4 XML"):
        sec_ownership._parse_form4(b"<not-xml <<<")


@pytest.mark.unit
def test_r15_form4_unparseable_xml_filing_skipped_not_fatal(
        monkeypatch, tmp_path):
    """One malformed-XML filing is skipped with the note; the healthy filing
    still renders (a single bad document must not fail the whole tool)."""
    subs = _r15_subs({
        "form": ["4", "4"],
        "accessionNumber": ["0000320193-24-000001", "0000320193-24-000002"],
        "filingDate": ["2024-05-01", "2024-05-02"],
        "primaryDocument": ["bad.xml", "good.xml"],
    })
    _r15_patch_form4(
        monkeypatch, tmp_path, subs,
        xml_by_doc={"bad.xml": b"<broken <<<", "good.xml": _r15_form4_xml()},
    )
    out = sec_ownership.get_form4_insider_trading("AAPL", "2024-06-30", 180)
    assert "R15 INSIDER" in out
    assert "1 filing(s) skipped" in out


@pytest.mark.unit
def test_r15_form4_dtd_filing_skipped_not_fatal(monkeypatch, tmp_path):
    """A DTD/ENTITY-refused filing degrades like any other unparseable filing
    (skip + note) — the refusal detail must not take the whole tool down."""
    dtd = (b"<?xml version='1.0'?><!DOCTYPE ownershipDocument ["
           b"<!ENTITY x 'y'>]><ownershipDocument/>")
    subs = _r15_subs({
        "form": ["4", "4"],
        "accessionNumber": ["0000320193-24-000001", "0000320193-24-000002"],
        "filingDate": ["2024-05-01", "2024-05-02"],
        "primaryDocument": ["dtd.xml", "good.xml"],
    })
    _r15_patch_form4(
        monkeypatch, tmp_path, subs,
        xml_by_doc={"dtd.xml": dtd, "good.xml": _r15_form4_xml()},
    )
    out = sec_ownership.get_form4_insider_trading("AAPL", "2024-06-30", 180)
    assert "R15 INSIDER" in out
    assert "1 filing(s) skipped" in out


@pytest.mark.unit
def test_r15_form4_xml_transport_error_in_scan_raises(monkeypatch, tmp_path):
    """A transport failure on ONE Form 4 XML (not a 404, not unparseable)
    fails the tool — skipping it would fabricate a clean window."""
    subs = _r15_subs({
        "form": ["4"],
        "accessionNumber": ["0000320193-24-000001"],
        "filingDate": ["2024-05-01"],
        "primaryDocument": ["f.xml"],
    })

    def dead_fetch(cik, accession, doc):
        raise NoMarketDataError("xml", detail="SEC request failed: proxy refused")

    _r15_patch_form4(monkeypatch, tmp_path, subs, form4_fetch=dead_fetch)
    with pytest.raises(NoMarketDataError, match="proxy refused"):
        sec_ownership.get_form4_insider_trading("AAPL", "2024-06-30", 180)


@pytest.mark.unit
def test_r15_form4_empty_window_with_skipped_note(monkeypatch, tmp_path):
    """No renderable rows AND skipped filings -> the empty verdict carries the
    skip note (never a silently clean window)."""
    subs = _r15_subs({
        "form": ["4"],
        "accessionNumber": ["0000320193-24-000001"],
        "filingDate": ["2024-05-01"],
        "primaryDocument": ["gone.xml"],
    })

    def gone(cik, accession, doc):
        raise SecNoFileError(doc, detail="SEC returned 404")

    _r15_patch_form4(monkeypatch, tmp_path, subs, form4_fetch=gone)
    out = sec_ownership.get_form4_insider_trading("AAPL", "2024-06-30", 180)
    assert "No Form 4 transactions" in out
    assert "1 filing(s) in the window could not be retrieved" in out


@pytest.mark.unit
def test_r15_form4_missing_title_renders(monkeypatch, tmp_path):
    """A Form 4 with no officerTitle element parses to an empty title and the
    trade still renders."""
    subs = _r15_subs({
        "form": ["4"],
        "accessionNumber": ["0000320193-24-000001"],
        "filingDate": ["2024-05-01"],
        "primaryDocument": ["notitle.xml"],
    })
    _r15_patch_form4(
        monkeypatch, tmp_path, subs,
        xml_by_doc={"notitle.xml": _r15_form4_xml(owner="R15 NOTITLE")},
    )
    out = sec_ownership.get_form4_insider_trading("AAPL", "2024-06-30", 180)
    assert "R15 NOTITLE" in out


@pytest.mark.unit
def test_r15_form4_summary_skips_nonnumeric_amounts(monkeypatch, tmp_path):
    """Rows with non-numeric shares/price render in the table but are excluded
    from the dollar-flow summary (never crash, never counted as 0-dollar)."""
    subs = _r15_subs({
        "form": ["4", "4"],
        "accessionNumber": ["0000320193-24-000001", "0000320193-24-000002"],
        "filingDate": ["2024-05-01", "2024-05-02"],
        "primaryDocument": ["buy.xml", "sell.xml"],
    })
    _r15_patch_form4(
        monkeypatch, tmp_path, subs,
        xml_by_doc={
            "buy.xml": _r15_form4_xml(owner="R15 BUYER", code="P",
                                      shares="10,000", price="180.50"),
            "sell.xml": _r15_form4_xml(owner="R15 SELLER", code="S",
                                       shares="5000", price="n/a"),
        },
    )
    out = sec_ownership.get_form4_insider_trading("AAPL", "2024-06-30", 180)
    assert "R15 BUYER" in out and "R15 SELLER" in out   # both rows render
    assert "1 buy(s) $1,805,000" in out                 # only the parsed buy
    assert "0 sell(s) $0" in out                        # n/a price not counted
    assert "net buyer" in out
    # _num degrades on garbage and on non-strings alike
    assert sec_ownership._num("n/a") is None
    assert sec_ownership._num(None) is None
    assert sec_ownership._num("1,234.5") == 1234.5


# --------------------------------------------------------------------------- #
# sec_ownership: _scan_form4_segment defensive branches (direct)
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_r15_scan_form4_segment_skips_non_form4_and_short_arrays(monkeypatch):
    """Non-Form-4 entries are skipped; an index past the parallel arrays stops
    the scan instead of raising IndexError."""
    monkeypatch.setattr(sec_ownership, "_fetch_form4_xml",
                        lambda cik, acc, doc: _r15_form4_xml())
    rows: list[dict] = []
    skipped: list[str] = []
    fetched = sec_ownership._scan_form4_segment(
        CIK,
        {"form": ["8-K", "4"],
         "accessionNumber": ["0000320193-24-000001", "0000320193-24-000002"],
         "filingDate": ["2024-05-01", "2024-05-02"],
         "primaryDocument": ["x.xml", "f.xml"]},
        [0, 1, 7],                       # 7 >= len(forms) -> stop
        date(2024, 1, 1), date(2024, 6, 30), "2024-06-30",
        rows, skipped, 0,
    )
    assert fetched == 1
    assert len(rows) == 1                # only the form == "4" entry
    assert rows[0]["owner"] == "R15 INSIDER"


# --------------------------------------------------------------------------- #
# sec_ownership: fetch budget (_MAX_FORM4 = 25)
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_r15_form4_fetch_budget_caps_at_25(monkeypatch, tmp_path):
    subs = _r15_subs(_r15_head(fillers=0, form4s=26))
    calls = {"n": 0}

    def counting_fetch(cik, accession, doc):
        calls["n"] += 1
        return _r15_form4_xml(owner=f"OWNER {doc}")

    _r15_patch_form4(monkeypatch, tmp_path, subs, form4_fetch=counting_fetch)
    out = sec_ownership.get_form4_insider_trading("AAPL", "2024-06-30", 180)
    assert "# 25 non-derivative trade(s) across 25 filing(s)" in out
    assert calls["n"] == 25              # the 26th filing was never fetched


@pytest.mark.unit
def test_r15_form4_head_fills_budget_discloses_archive_gap(monkeypatch, tmp_path):
    """Head at the ~1000 cap AND the budget spent by the head alone: the
    archive was never scanned, and the output must say so."""
    subs = _r15_subs(_r15_head(fillers=974, form4s=26))
    _r15_patch_form4(
        monkeypatch, tmp_path, subs,
        form4_fetch=lambda cik, acc, doc: _r15_form4_xml(),
    )
    out = sec_ownership.get_form4_insider_trading("AAPL", "2024-06-30", 180)
    assert "files[] archive was not fully scanned" in out
    assert "# 25 non-derivative trade(s) across 25 filing(s)" in out


# --------------------------------------------------------------------------- #
# sec_ownership: files[] archive backfill
# --------------------------------------------------------------------------- #
_SEG1 = "CIK0000320193-submissions-001.json"
_SEG2 = "CIK0000320193-submissions-002.json"


def _r15_patch_archive(monkeypatch, tmp_path, files, seg_payloads, subs=None):
    """Head with 1000 non-Form-4 filings + ``files[]`` segments served from
    ``seg_payloads`` (name -> bytes); missing segments 404 unless listed in
    ``raise_for``. Returns the requested-URL recorder."""
    monkeypatch.setattr(sec_ownership, "_cache_dir", lambda: str(tmp_path))
    requested: list[str] = []
    head = subs if subs is not None else _r15_subs(
        _r15_head(fillers=1000, form4s=0), files=files)

    def fetch(_path, url, ttl_days):
        requested.append(url)
        if url.endswith(f"CIK{CIK:010d}.json"):
            return head
        for name, payload in seg_payloads.items():
            if url.endswith(name):
                return payload
        raise SecNoFileError(url, detail="404")

    monkeypatch.setattr(sec_ownership, "_cached_or_fetch", fetch)
    monkeypatch.setattr(
        sec_ownership, "_fetch_form4_xml",
        lambda cik, acc, doc: _r15_form4_xml(owner=doc.replace(".xml", "")),
    )
    return requested


@pytest.mark.unit
def test_r15_form4_archive_backfill_scans_segments_newest_first(
        monkeypatch, tmp_path):
    """Older in-window Form 4s live in the immutable files[] segments when the
    head is at its cap — both segments are scanned (newest segment first) and
    render without the disclosure note (backfill completed)."""
    seg1 = _r15_subs(_r15_head(fillers=0, form4s=1, filing_date="2024-04-01"))
    seg2 = _r15_subs(_r15_head(fillers=0, form4s=1, filing_date="2024-05-01"))
    requested = _r15_patch_archive(
        monkeypatch, tmp_path,
        files=[{"name": _SEG1}, {"name": _SEG2}],
        seg_payloads={_SEG1: seg1, _SEG2: seg2},
    )
    out = sec_ownership.get_form4_insider_trading("AAPL", "2024-06-30", 180)
    assert "f000" in out                 # flattened owner from each segment doc
    assert "# 2 non-derivative trade(s) across 2 filing(s)" in out
    assert "files[] archive was not fully scanned" not in out
    # Segments are pulled newest-first (reversed files[] order).
    assert requested[1].endswith(_SEG2)
    assert requested[2].endswith(_SEG1)


@pytest.mark.unit
def test_r15_form4_archive_budget_break_discloses_gap(monkeypatch, tmp_path):
    """Budget exhausted mid-backfill leaves older segments unscanned — the
    disclosure note must fire (its text lists "fetch budget reached")."""
    seg1 = _r15_subs(_r15_head(fillers=0, form4s=1, filing_date="2024-04-01"))
    seg2 = _r15_subs(_r15_head(fillers=0, form4s=25, filing_date="2024-05-01"))
    requested = _r15_patch_archive(
        monkeypatch, tmp_path,
        files=[{"name": _SEG1}, {"name": _SEG2}],
        seg_payloads={_SEG1: seg1, _SEG2: seg2},
    )
    out = sec_ownership.get_form4_insider_trading("AAPL", "2024-06-30", 180)
    assert "# 25 non-derivative trade(s) across 25 filing(s)" in out
    assert "files[] archive was not fully scanned" in out
    assert not any(u.endswith(_SEG1) for u in requested)  # never scanned


@pytest.mark.unit
def test_r15_form4_head_at_cap_without_segments_discloses_gap(
        monkeypatch, tmp_path):
    """Head at the ~1000 cap but NO files[] segments listed: the archive
    cannot be scanned at all — the disclosure note fires, and with no Form 4s
    the empty verdict stays honest."""
    _r15_patch_archive(monkeypatch, tmp_path, files=[], seg_payloads={})
    out = sec_ownership.get_form4_insider_trading("AAPL", "2024-06-30", 180)
    assert "No Form 4 transactions" in out
    assert "files[] archive was not fully scanned" in out


@pytest.mark.unit
def test_r15_form4_archive_unnamed_segment_marks_gap(monkeypatch, tmp_path):
    seg9 = "CIK0000320193-submissions-009.json"
    _r15_patch_archive(
        monkeypatch, tmp_path,
        files=[{"name": ""}, {"name": seg9}],
        seg_payloads={seg9: _r15_subs(_r15_head(fillers=0, form4s=1))},
    )
    out = sec_ownership.get_form4_insider_trading("AAPL", "2024-06-30", 180)
    assert "f000" in out                 # the named segment still renders
    assert "files[] archive was not fully scanned" in out


@pytest.mark.unit
def test_r15_form4_archive_404_segment_marks_gap(monkeypatch, tmp_path):
    """A genuinely-missing archive segment (404) degrades to the note; the
    healthy segment still renders."""
    seg2 = _r15_subs(_r15_head(fillers=0, form4s=1))
    requested = _r15_patch_archive(
        monkeypatch, tmp_path,
        files=[{"name": _SEG1}, {"name": _SEG2}],
        seg_payloads={_SEG2: seg2},       # _SEG1 -> 404
    )
    out = sec_ownership.get_form4_insider_trading("AAPL", "2024-06-30", 180)
    assert "f000" in out
    assert "files[] archive was not fully scanned" in out
    assert any(u.endswith(_SEG1) for u in requested)


@pytest.mark.unit
def test_r15_form4_archive_unparsable_segment_marks_gap(monkeypatch, tmp_path):
    """An unparseable segment JSON is a per-segment degrade (note), not a tool
    failure; rows may legitimately be empty then."""
    _r15_patch_archive(
        monkeypatch, tmp_path,
        files=[{"name": _SEG1}],
        seg_payloads={_SEG1: b"{not-json"},
    )
    out = sec_ownership.get_form4_insider_trading("AAPL", "2024-06-30", 180)
    assert "No Form 4 transactions" in out
    assert "files[] archive was not fully scanned" in out


@pytest.mark.unit
def test_r15_form4_archive_transport_error_raises(monkeypatch, tmp_path):
    """A transport failure fetching a segment fails the tool — the archive
    being unreachable is not the same as 'no insider activity'."""
    monkeypatch.setattr(sec_ownership, "_cache_dir", lambda: str(tmp_path))
    head = _r15_subs(_r15_head(fillers=1000, form4s=0), files=[{"name": _SEG1}])
    requested: list[str] = []

    def dead(_path, url, ttl_days):
        requested.append(url)
        if url.endswith(f"CIK{CIK:010d}.json"):
            return head
        raise NoMarketDataError(url, detail="SEC request failed: timeout")

    monkeypatch.setattr(sec_ownership, "_cached_or_fetch", dead)
    with pytest.raises(NoMarketDataError, match="timeout"):
        sec_ownership.get_form4_insider_trading("AAPL", "2024-06-30", 180)


# --------------------------------------------------------------------------- #
# sec_ownership: FTD edges
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_r15_ftd_pub_lag_boundary_visible_exactly_at_cutoff(
        monkeypatch, tmp_path):
    """Default 10-day lag: the 202405b file (half end 2024-05-31) is visible
    the day 5/31 + 10 == curr_date (2024-06-10) and invisible the day before
    (2024-06-09) — never even requested then."""
    requested = _r15_patch_ftd(monkeypatch, tmp_path)
    out_open = sec_ownership.get_ftd_data("AAPL", "2024-06-10", 45)
    assert any("cnsfails202405b.zip" in u for u in requested)
    assert "1234567" in out_open
    requested.clear()
    out_before = sec_ownership.get_ftd_data("AAPL", "2024-06-09", 45)
    assert not any("cnsfails202405b.zip" in u for u in requested)
    assert "No fails-to-deliver reported" in out_before


@pytest.mark.unit
def test_r15_ftd_window_rolls_over_year_end(monkeypatch, tmp_path):
    """A look-back window spanning December -> January enumerates both years'
    files; the not-yet-public January file is never requested."""
    content = (b"SETTLEMENT DATE|CUSIP|SYMBOL|QUANTITY (FAILS)|DESCRIPTION|PRICE\n"
               b"20231220|037833100|AAPL|55555|APPLE INC|190.00\n")
    requested = _r15_patch_ftd(monkeypatch, tmp_path, content=content,
                               ym="202312", half="b")
    out = sec_ownership.get_ftd_data("AAPL", "2024-01-20", 60)
    assert any("cnsfails202312b.zip" in u for u in requested)
    assert not any("cnsfails202401" in u for u in requested)
    assert "55555" in out


@pytest.mark.unit
def test_r15_ftd_lookback_capped_at_730(monkeypatch, tmp_path, caplog):
    """An LLM-supplied huge look-back is capped to 730 days with a warning."""
    import logging

    _r15_patch_ftd(monkeypatch, tmp_path)
    with caplog.at_level(logging.WARNING,
                         logger="yialpha.dataflows.sec_ownership"):
        out = sec_ownership.get_ftd_data("AAPL", "2024-06-20", 5000)
    assert "last 730 days" in out
    assert any("capped to 730" in r.message for r in caplog.records)


@pytest.mark.unit
def test_r15_ftd_malformed_row_date_skipped(monkeypatch, tmp_path):
    """Rows whose settlement date cannot be parsed are dropped; healthy rows
    in the same file still render."""
    content = (b"SETTLEMENT DATE|CUSIP|SYMBOL|QUANTITY (FAILS)|DESCRIPTION|PRICE\n"
               b"20240510|037833100|AAPL|1234567|APPLE INC|195.20\n"
               b"202405XX|037833100|AAPL|7777777|APPLE INC|195.20\n"
               b"2024|037833100|AAPL|8888888|APPLE INC|195.20\n")
    _r15_patch_ftd(monkeypatch, tmp_path, content=content)
    out = sec_ownership.get_ftd_data("AAPL", "2024-06-20", 45)
    assert "1234567" in out
    assert "7777777" not in out
    assert "8888888" not in out
    assert "1 fail-day(s)" in out


@pytest.mark.unit
def test_r15_ftd_issuer_column_mapped():
    cols = sec_ownership._ftd_column_map(
        "SETTLEMENT DATE|CUSIP|SYMBOL|QUANTITY (FAILS)|ISSUER NAME|PRICE")
    assert cols == {"date": 0, "cusip": 1, "symbol": 2, "fails": 3,
                    "issuer": 4, "price": 5}


@pytest.mark.unit
def test_r15_ftd_empty_zip_raises():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w"):
        pass                            # a ZIP with zero members
    with pytest.raises(NoMarketDataError, match="no members"):
        sec_ownership._unzip_ftd_member(buf.getvalue())


@pytest.mark.unit
def test_r15_ftd_parse_empty_and_symbolless_content():
    assert sec_ownership._parse_ftd_text(b"", "AAPL") == []
    assert sec_ownership._parse_ftd_text(b"   \n \n", "AAPL") == []
    # Header without a symbol column: no row can match the ticker.
    assert sec_ownership._parse_ftd_text(
        b"DATE|CUSIP|FAILS|PRICE\n20240101|037833100|100|1.0", "AAPL") == []
    # Row shorter than the symbol column index: skipped, not an IndexError.
    assert sec_ownership._parse_ftd_text(
        b"SETTLEMENT DATE|CUSIP|SYMBOL|FAILS|PRICE\n20240101", "AAPL") == []


# --------------------------------------------------------------------------- #
# sec_ownership: 13F helpers + CUSIP PIT gates
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_r15_13f_report_quarter_aug_and_nov():
    assert sec_ownership._13f_report_quarter(date(2024, 8, 31)) == (2024, 2)
    assert sec_ownership._13f_report_quarter(date(2024, 11, 30)) == (2024, 3)


@pytest.mark.unit
def test_r15_13f_zip_filename_aug_nov_and_pre2024_scheme():
    assert sec_ownership._13f_zip_filename(date(2024, 8, 31)) == \
        "01jun2024-31aug2024_form13f.zip"
    assert sec_ownership._13f_zip_filename(date(2024, 11, 30)) == \
        "01sep2024-30nov2024_form13f.zip"
    assert sec_ownership._13f_zip_filename(date(2023, 11, 30)) == \
        "2023q3_form13f.zip"
    assert sec_ownership._13f_zip_filename(date(2021, 2, 26)) == \
        "2020q4_form13f.zip"


@pytest.mark.unit
def test_r15_13f_tsv_helper_edges():
    assert sec_ownership._tsv_header_lines(b"") == ([], [])
    assert sec_ownership._idx(["NAME_OF_ISSUER", "CUSIP"], "filing_date") is None
    assert sec_ownership._cell(["x"], 3) == ""
    assert sec_ownership._cell(["x"], None) == ""
    assert sec_ownership._cell(["x", " y "], 1) == "y"


@pytest.mark.unit
def test_r15_13f_corrupt_zip_raises():
    with pytest.raises(NoMarketDataError, match="could not open 13F ZIP"):
        sec_ownership._parse_13f_tsv(b"not a zip at all")


def _r15_13f_zip_bytes(members: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data in members.items():
            zf.writestr(name, data)
    return buf.getvalue()


_R13F_COVER = ("ACCESSION_NUMBER\tFILER_CIK\tFILING_DATE\tFILING_MANAGER_NAME\n"
               "R15A\t111\t2024-05-15\tR15 CAPITAL")
_R13F_HOLDING = ("ACCESSION_NUMBER\tNAME_OF_ISSUER\tCUSIP\tVALUE\t"
                 "SSH_PRNAMT\tSSH_PRNAMT_TYPE\tPUT_CALL\n"
                 "R15A\tAPPLE INC\t037833100\t1000\t100\tSH\t")


@pytest.mark.unit
def test_r15_13f_zip_member_selection():
    # A non-TSV member is ignored; cover+holding parse fine alongside it.
    good = _r15_13f_zip_bytes({
        "readme.xml": b"<readme/>",
        "filing_cover.tsv": _R13F_COVER.encode(),
        "filing_holding.tsv": _R13F_HOLDING.encode(),
    })
    cover, holding = sec_ownership._parse_13f_tsv(good)
    assert cover[0]["manager"] == "R15 CAPITAL"
    assert holding[0]["cusip"] == "037833100"
    # Missing the holding member -> typed degrade, not a KeyError.
    only_cover = _r15_13f_zip_bytes({"filing_cover.tsv":
                                     _R13F_COVER.encode()})
    with pytest.raises(NoMarketDataError, match="missing cover/holding"):
        sec_ownership._parse_13f_tsv(only_cover)


def _r15_cusip_facts(records):
    return {"dei": {"EntityCusip": {"units": {"NONE": records}}}}


@pytest.mark.unit
def test_r15_extract_cusip_skips_valless_and_malformed_filed():
    facts = _r15_cusip_facts([
        {"end": "2024-03-31", "filed": "2024-04-26", "val": ""},        # no val
        {"end": "2024-03-31", "filed": "not-a-date", "val": "111111111"},  # PIT
        {"end": "2024-03-31", "filed": "2024-04-26", "val": "037833100"},
    ])
    # The val-less and malformed-filed records are dropped (fail-closed);
    # the healthy record survives and wins.
    assert sec_ownership._extract_cusip(facts, "2024-06-15") == "037833100"
    # Only-malformed-filed facts leave nothing PIT-visible -> typed degrade.
    with pytest.raises(NoMarketDataError, match="no PIT-visible EntityCusip"):
        sec_ownership._extract_cusip(_r15_cusip_facts(
            [{"end": "2024-03-31", "filed": "garbage", "val": "037833100"}]),
            "2024-06-15")


@pytest.mark.unit
def test_r15_extract_cusip_malformed_or_missing_end_survives():
    """The end date only orders record selection — an unparseable or missing
    end degrades to date.min (the filed-at gate alone decides PIT)."""
    assert sec_ownership._extract_cusip(_r15_cusip_facts(
        [{"end": "garbage-date", "filed": "2024-04-26", "val": "037833100"}]),
        "2024-06-15") == "037833100"
    assert sec_ownership._extract_cusip(_r15_cusip_facts(
        [{"filed": "2024-04-26", "val": "037833100"}]), "2024-06-15") == \
        "037833100"


@pytest.mark.unit
def test_r15_extract_cusip_no_pit_visible_record_raises():
    # Every record's end postdates curr_date -> nothing PIT-visible.
    facts = _r15_cusip_facts([
        {"end": "2030-01-01", "filed": "2024-01-01", "val": "037833100"},
    ])
    with pytest.raises(NoMarketDataError, match="no PIT-visible EntityCusip"):
        sec_ownership._extract_cusip(facts, "2024-06-15")
    # Same for every record filed after curr_date.
    facts2 = _r15_cusip_facts([
        {"end": "2024-03-31", "filed": "2024-12-01", "val": "037833100"},
    ])
    with pytest.raises(NoMarketDataError, match="no PIT-visible EntityCusip"):
        sec_ownership._extract_cusip(facts2, "2024-06-15")


@pytest.mark.unit
def test_r15_extract_cusip_too_short_raises():
    facts = _r15_cusip_facts([
        {"end": "2024-03-31", "filed": "2024-04-26", "val": "123"},
    ])
    with pytest.raises(NoMarketDataError, match="too short"):
        sec_ownership._extract_cusip(facts, "2024-06-15")


# --------------------------------------------------------------------------- #
# sec_ownership: 13F tool-level edges
# --------------------------------------------------------------------------- #
def _r15_patch_13f(monkeypatch, tmp_path, zip_bytes=None):
    monkeypatch.setattr(sec_ownership, "_cache_dir", lambda: str(tmp_path))
    monkeypatch.setattr(sec_ownership, "_cik_for_ticker", lambda t: CIK)
    monkeypatch.setattr(
        sec_ownership, "_fetch_company_facts",
        lambda cik: _r15_cusip_facts([
            {"end": "2024-03-31", "filed": "2024-04-26", "val": "037833100"},
        ]),
    )
    requested: list[str] = []

    def fetch(_path, url, ttl_days):
        requested.append(url)
        if "form13f.zip" in url:
            if zip_bytes is None:
                return _r15_13f_zip_bytes({
                    "filing_cover.tsv": _R13F_COVER.encode(),
                    "filing_holding.tsv": _R13F_HOLDING.encode(),
                })
            return zip_bytes
        raise NoMarketDataError(url, detail="404")

    monkeypatch.setattr(sec_ownership, "_cached_or_fetch", fetch)
    return requested


@pytest.mark.unit
def test_r15_13f_pub_lag_boundary_visible_exactly_at_cutoff(
        monkeypatch, tmp_path):
    """Default 45-day lag: the Mar-May 2024 dataset (period end 2024-05-31)
    is visible the day 5/31 + 45 == curr_date (2024-07-15); on 2024-07-14 the
    Dec-Feb dataset is the latest visible one."""
    requested = _r15_patch_13f(monkeypatch, tmp_path)
    sec_ownership.get_institutional_holdings("AAPL", "2024-07-15", 180)
    assert any("01mar2024-31may2024_form13f.zip" in u for u in requested)
    requested.clear()
    sec_ownership.get_institutional_holdings("AAPL", "2024-07-14", 180)
    assert not any("01mar2024" in u for u in requested)
    assert any("01dec2023-29feb2024_form13f.zip" in u for u in requested)


@pytest.mark.unit
def test_r15_13f_zip_transport_error_propagates(monkeypatch, tmp_path):
    """A transport failure fetching the visible dataset propagates instead of
    fabricating an empty holders table."""
    monkeypatch.setattr(sec_ownership, "_cache_dir", lambda: str(tmp_path))
    monkeypatch.setattr(sec_ownership, "_cik_for_ticker", lambda t: CIK)
    monkeypatch.setattr(
        sec_ownership, "_fetch_company_facts",
        lambda cik: _r15_cusip_facts([
            {"end": "2024-03-31", "filed": "2024-04-26", "val": "037833100"},
        ]),
    )
    monkeypatch.setattr(
        sec_ownership, "_cached_or_fetch",
        lambda path, url, ttl_days: (_ for _ in ()).throw(
            NoMarketDataError(url, detail="SEC request failed: timeout")),
    )
    with pytest.raises(NoMarketDataError, match="timeout"):
        sec_ownership.get_institutional_holdings("AAPL", "2024-07-20", 180)


@pytest.mark.unit
def test_r15_13f_nonnumeric_value_row_excluded(monkeypatch, tmp_path):
    """A holding row whose VALUE cannot be parsed is excluded from the
    aggregation; the healthy row still aggregates."""
    holding = ("ACCESSION_NUMBER\tNAME_OF_ISSUER\tCUSIP\tVALUE\t"
               "SSH_PRNAMT\tSSH_PRNAMT_TYPE\tPUT_CALL\n"
               "R15A\tAPPLE INC\t037833100\tn/a\t999\tSH\t\n"
               "R15A\tAPPLE INC\t037833100\t1000\t100\tSH\t")
    zip_bytes = _r15_13f_zip_bytes({
        "filing_cover.tsv": _R13F_COVER.encode(),
        "filing_holding.tsv": holding.encode(),
    })
    _r15_patch_13f(monkeypatch, tmp_path, zip_bytes=zip_bytes)
    out = sec_ownership.get_institutional_holdings("AAPL", "2024-07-20", 180)
    assert "R15 CAPITAL" in out
    assert "999" not in out              # the unparseable row's shares never show
    assert "1.00" in out                 # 1000 ($000s) * 1000 = $1.00M


# --------------------------------------------------------------------------- #
# sec_edgar
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_r15_sec_cache_dir_is_sec_vendor_dir(monkeypatch, tmp_path):
    from yialpha.dataflows import config as cfgmod

    orig = cfgmod.get_config()
    try:
        cfgmod.set_config({**orig, "data_cache_dir": str(tmp_path)})
        path = sec_edgar._cache_dir()
    finally:
        cfgmod.set_config(orig)
    assert os.path.basename(path) == "sec"
    assert os.path.isdir(path)


class _R15Resp:
    def __init__(self, status, content=b""):
        self.status_code = status
        self.content = content


@pytest.mark.unit
def test_r15_sec_get_5xx_and_success_mapping(monkeypatch):
    monkeypatch.setattr("requests.get", lambda *a, **k: _R15Resp(503))
    with pytest.raises(NoMarketDataError, match="SEC HTTP 503"):
        sec_edgar._sec_get("https://data.sec.gov/x")
    monkeypatch.setattr("requests.get",
                        lambda *a, **k: _R15Resp(200, b"payload-bytes"))
    assert sec_edgar._sec_get("https://data.sec.gov/y") == b"payload-bytes"


@pytest.mark.unit
def test_r15_sec_unparsable_payloads_degrade(monkeypatch, tmp_path):
    monkeypatch.setattr(sec_edgar, "_cache_dir", lambda: str(tmp_path))
    monkeypatch.setattr(sec_edgar, "_cached_or_fetch",
                        lambda path, url, ttl_days: b"{not-json")
    with pytest.raises(NoMarketDataError,
                       match="could not parse SEC ticker table"):
        sec_edgar._cik_for_ticker("AAPL")
    with pytest.raises(NoMarketDataError, match="could not parse companyfacts"):
        sec_edgar._fetch_company_facts(CIK)


@pytest.mark.unit
def test_r15_concept_records_skips_records_without_val():
    facts = {"facts": {"us-gaap": {"Assets": {"units": {"USD": [
        {"end": "2024-06-30", "val": None},          # explicit None
        {"end": "2024-03-31"},                        # no val key at all
        {"end": "2024-06-30", "val": 100},
    ]}}}}}
    recs = sec_edgar._concept_records(facts, ["Assets"])
    assert recs == [{"end": "2024-06-30", "val": 100}]


@pytest.mark.unit
def test_r15_period_public_missing_filed_falls_back_to_heuristic():
    rec = {"end": "2024-06-30"}        # no filed date at all
    # 45-day filing lag heuristic: 6/30 + 45 = 8/14.
    assert sec_edgar._period_public(rec, "2024-08-15") is True
    assert sec_edgar._period_public(rec, "2024-08-14") is True
    assert sec_edgar._period_public(rec, "2024-08-13") is False


@pytest.mark.unit
def test_r15_fmt_unit_kinds_and_invalid_values():
    assert sec_edgar._fmt(6.5137, "per_share") == "6.5137"
    assert sec_edgar._fmt(1234.6, "shares") == "1235"
    assert sec_edgar._fmt(9876.4, "money") == "9876"
    assert sec_edgar._fmt("n/a", "money") == ""
    assert sec_edgar._fmt(None, "shares") == ""


@pytest.mark.unit
def test_r15_get_cashflow_renders_pit_filtered_statement(monkeypatch, tmp_path):
    facts = json.dumps({"facts": {"us-gaap": {
        "NetCashProvidedByUsedInOperatingActivities": {"units": {"USD": [
            {"start": "2023-09-25", "end": "2024-09-28", "val": 118254,
             "fy": 2024, "fp": "FY", "form": "10-K", "filed": "2024-11-01"},
            {"start": "2023-12-31", "end": "2024-03-30", "val": 28815,
             "fy": 2024, "fp": "Q2", "form": "10-Q", "filed": "2024-05-02"},
            {"start": "2024-03-31", "end": "2024-06-29", "val": 28902,
             "fy": 2024, "fp": "Q3", "form": "10-Q", "filed": "2024-08-01"},
        ]}},
    }}}).encode()
    monkeypatch.setattr(sec_edgar, "_cache_dir", lambda: str(tmp_path))
    tickers = json.dumps({"0": {"cik_str": CIK, "ticker": "AAPL"}}).encode()
    monkeypatch.setattr(
        sec_edgar, "_cached_or_fetch",
        lambda path, url, ttl_days: facts if "companyfacts" in url else tickers,
    )
    out = sec_edgar.get_cashflow("AAPL", "quarterly", "2024-07-01")
    assert out.startswith("# Cash Flow Statement for AAPL (quarterly)")
    op_line = next(line for line in out.splitlines()
                   if line.startswith("Operating CF,"))
    assert "28815" in op_line           # Q2 (filed 5/2) is the value shown
    assert "2024-06-29" not in out      # Q3 filed 8/1 > 7/1 -> invisible
    assert "2024-03-30" in out


# --------------------------------------------------------------------------- #
# tavily
# --------------------------------------------------------------------------- #
_R15_KEY_ENV = "TAVILY" + "_API_KEY"
_R15_KEYS_ENV = "TAVILY_API" + "_KEYS"
_R15_MAX_ENV = "YIALPHA_TAVILY_MAX_CALLS" + "_PER_RUN"
_R15_SPLIT_ENV = "YIALPHA_TAVILY_BUDGET" + "_SPLIT"
_R15_TIMEOUT_ENV = "YIALPHA_TAVILY_TIMEOUT" + "_S"
_R15_FAKE_KEY = "tvly-dev-" + "r15-fake"


class _R15TavilyResponse:
    def __init__(self, payload=None, status=200):
        self._payload = payload
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.exceptions.HTTPError(
                f"{self.status_code} error", response=self)

    def json(self):
        if self._payload is None:
            raise ValueError("no json")
        return self._payload


def _r15_fresh(monkeypatch, *, key=_R15_FAKE_KEY, keys=None):
    quality.ensure_run_context()
    tavily.reset_run_budget()
    if keys is None:
        monkeypatch.delenv(_R15_KEYS_ENV, raising=False)
        if key is None:
            monkeypatch.delenv(_R15_KEY_ENV, raising=False)
        else:
            monkeypatch.setenv(_R15_KEY_ENV, key)
    else:
        monkeypatch.delenv(_R15_KEY_ENV, raising=False)
        monkeypatch.setenv(_R15_KEYS_ENV, ",".join(keys))
    monkeypatch.delenv(_R15_MAX_ENV, raising=False)
    monkeypatch.delenv(_R15_SPLIT_ENV, raising=False)
    monkeypatch.delenv(_R15_TIMEOUT_ENV, raising=False)


def _r15_mock_post(fake):
    return mock.patch.object(tavily.requests, "post", mock.Mock(wraps=fake))


@pytest.mark.unit
def test_r15_tavily_fresh_context_lazily_binds_counters_and_cursor(monkeypatch):
    """A brand-new execution context (a ToolNode worker that never inherited
    the run-root bindings) lazily binds the counter and cursor boxes in place
    — the mutation contract that keeps usage visible to the run root."""
    _r15_fresh(monkeypatch, keys=["tvly-dev-r15-a", "tvly-dev-r15-b"])
    seen: list[str] = []

    def _post(url, json=None, headers=None, timeout=None):  # noqa: A002
        seen.append(headers["Authorization"])
        return _R15TavilyResponse({"results": []})

    ctx = contextvars.Context()
    with _r15_mock_post(_post):
        out = ctx.run(tavily.get_web_search, "q")
    assert "WEB_SEARCH_UNAVAILABLE" not in out
    assert len(seen) == 1                # the round-robin cursor was bound
    assert ctx.run(tavily.run_usage)["news"] == 1


@pytest.mark.unit
def test_r15_tavily_fresh_context_binds_dead_key_set_and_rotates(monkeypatch):
    _r15_fresh(monkeypatch, keys=["tvly-dev-r15-bad", "tvly-dev-r15-good"])
    posts = {"n": 0}

    def _post(url, json=None, headers=None, timeout=None):  # noqa: A002
        posts["n"] += 1
        if "bad" in headers["Authorization"]:
            return _R15TavilyResponse(status=401)
        return _R15TavilyResponse({"results": []})

    ctx = contextvars.Context()
    with _r15_mock_post(_post):
        out = ctx.run(tavily.get_web_search, "q")
    assert "WEB_SEARCH_UNAVAILABLE" not in out
    assert posts["n"] == 2               # bad key once, then the good key


@pytest.mark.unit
def test_r15_tavily_pick_key_all_dead_returns_none(monkeypatch):
    _r15_fresh(monkeypatch, keys=["tvly-dev-r15-a", "tvly-dev-r15-b"])
    tavily._mark_key_dead(0, 2, 401)
    tavily._mark_key_dead(1, 2, 429)
    assert tavily._pick_key(["tvly-dev-r15-a", "tvly-dev-r15-b"]) is None
    # The rotations stay visible in the ledger, never silent.
    assert any("HTTP 401" in e["detail"] for e in quality.snapshot_quality())
    assert any("HTTP 429" in e["detail"] for e in quality.snapshot_quality())


@pytest.mark.unit
def test_r15_tavily_timeout_env_floor_and_garbage(monkeypatch):
    captured: dict[str, object] = {}

    def _post(url, json=None, headers=None, timeout=None):  # noqa: A002
        captured["timeout"] = timeout
        return _R15TavilyResponse({"results": []})

    _r15_fresh(monkeypatch)
    monkeypatch.setenv(_R15_TIMEOUT_ENV, "0.2")
    with _r15_mock_post(_post):
        tavily.get_web_search("q")
    assert captured["timeout"] == 1.0    # floored to the 1s minimum
    monkeypatch.setenv(_R15_TIMEOUT_ENV, "not-a-number")
    with _r15_mock_post(_post):
        tavily.get_web_search("q2")
    assert captured["timeout"] == tavily.DEFAULT_TIMEOUT_S


@pytest.mark.unit
def test_r15_tavily_days_param_rides_news_topic(monkeypatch):
    captured: dict[str, object] = {}

    def _post(url, json=None, headers=None, timeout=None):  # noqa: A002
        captured["payload"] = json
        return _R15TavilyResponse({"results": []})

    _r15_fresh(monkeypatch)
    with _r15_mock_post(_post):
        tavily.get_web_search("q", days=7)
    assert captured["payload"]["topic"] == "news"
    assert captured["payload"]["days"] == 7
    # A non-positive recency bound is ignored: plain query, no news topic.
    with _r15_mock_post(_post):
        tavily.get_web_search("q2", days=0)
    assert "topic" not in captured["payload"]


@pytest.mark.unit
def test_r15_tavily_non_json_body_sentinel(monkeypatch):
    _r15_fresh(monkeypatch)
    with _r15_mock_post(lambda *a, **k: _R15TavilyResponse(payload=None)):
        result = tavily.get_web_search("q")
    assert "WEB_SEARCH_UNAVAILABLE" in result
    assert "non-JSON body" in result
    assert any("non-JSON" in e["detail"] for e in quality.snapshot_quality())


@pytest.mark.unit
def test_r15_tavily_published_date_rendered(monkeypatch):
    payload = {"results": [{
        "title": "R15 news", "url": "https://example.com/r15",
        "content": "Something happened.",
        "published_date": "2026-11-01",
    }]}
    _r15_fresh(monkeypatch)
    with _r15_mock_post(lambda *a, **k: _R15TavilyResponse(payload)):
        result = tavily.get_web_search("q")
    assert "Published: 2026-11-01" in result
    assert quality.snapshot_quality() == []
