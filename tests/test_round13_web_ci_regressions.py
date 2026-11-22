"""Round-13 regression pins: report-dir glob prefix collision.

``web/store.list_reports`` and ``scripts/run_robust``'s
``_complete_mtime`` / ``_find_new_report`` all glob ``{ticker}_*`` under the
reports root. A glob ``*`` also matches underscores, so the glob for ``FOO``
also enumerated ``FOO_BAR_20260919_123456`` — a longer ticker sharing the
prefix. The fix (round-13) pins the exact write-side shape
``<TICKER>_<YYYYMMDD>_<HHMMSS>`` (``trading_graph.save_reports``) with a
``re.fullmatch`` filter. These tests钉住 that behavior:

* ``FOO`` must NOT list/see ``FOO_BAR``'s dirs;
* ``FOO_BAR`` must still list its own.

The run_robust script is loaded as an isolated module (importlib) so its
``__main__`` guard never fires — same pattern as test_run_robust_watchdog.
"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest

from web import store

_SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "run_robust.py"
_spec = importlib.util.spec_from_file_location("run_robust_round13", _SCRIPT_PATH)
assert _spec is not None and _spec.loader is not None
rr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rr)


def _make_report_dir(root: Path, name: str, *, complete: bool = True, mtime: float | None = None) -> Path:
    d = root / name
    d.mkdir(parents=True)
    if complete:
        cr = d / "complete_report.md"
        cr.write_text("# report\n", encoding="utf-8")
        if mtime is not None:
            os.utime(cr, (mtime, mtime))
    return d


@pytest.fixture
def reports_root(tmp_path, monkeypatch):
    root = tmp_path / "reports"
    root.mkdir()
    monkeypatch.setattr(store, "REPORTS_ROOT", root)
    return root


# --------------------------- web.store.list_reports ---------------------------


@pytest.mark.unit
def test_list_reports_prefix_ticker_not_listed(reports_root):
    """FOO must not enumerate FOO_BAR's report dirs (glob prefix collision)."""
    _make_report_dir(reports_root, "FOO_20260919_120000")
    _make_report_dir(reports_root, "FOO_BAR_20260919_130000")

    listed = [r["dir"] for r in store.list_reports("FOO")]
    assert listed == ["FOO_20260919_120000"]


@pytest.mark.unit
def test_list_reports_longer_ticker_lists_own(reports_root):
    """FOO_BAR's own dirs are still reachable under its full name."""
    _make_report_dir(reports_root, "FOO_20260919_120000")
    _make_report_dir(reports_root, "FOO_BAR_20260919_130000")

    listed = [r["dir"] for r in store.list_reports("FOO_BAR")]
    assert listed == ["FOO_BAR_20260919_130000"]


@pytest.mark.unit
def test_list_reports_non_stamp_suffix_excluded(reports_root):
    """Only the exact ``<TICKER>_<YYYYMMDD>_<HHMMSS>`` shape passes the gate."""
    _make_report_dir(reports_root, "FOO_20260919_120000")   # valid stamp
    _make_report_dir(reports_root, "FOO_20260919")          # no time part
    _make_report_dir(reports_root, "FOO_20260919_1200")     # short time
    _make_report_dir(reports_root, "FOO_backup_20260919")   # junk suffix
    _make_report_dir(reports_root, "FOO_20261319_120000")   # parses as stamp shape

    listed = [r["dir"] for r in store.list_reports("FOO")]
    # Shape check only (the date's validity is the write side's business).
    assert "FOO_20260919_120000" in listed
    assert "FOO_20261319_120000" in listed  # \d{8}_\d{6} matches digits regardless
    assert "FOO_20260919" not in listed
    assert "FOO_20260919_1200" not in listed
    assert "FOO_backup_20260919" not in listed


@pytest.mark.unit
def test_list_reports_dot_ticker_roundtrip(reports_root):
    """The server charset's flagship case (0700.HK) must round-trip through
    the fullmatch filter — the round-13 viewer-gate fix depends on dirs like
    this being both creatable and listable."""
    _make_report_dir(reports_root, "0700.HK_20260919_123456")

    listed = [r["dir"] for r in store.list_reports("0700.HK")]
    assert listed == ["0700.HK_20260919_123456"]


# ------------------------- scripts/run_robust glob paths ----------------------


@pytest.mark.unit
def test_complete_mtime_ignores_prefix_ticker(tmp_path):
    """_complete_mtime(FOO) must not see FOO_BAR's (newer) complete_report."""
    root = tmp_path / "reports"
    root.mkdir()
    _make_report_dir(root, "FOO_20260919_120000", mtime=1_000.0)
    _make_report_dir(root, "FOO_BAR_20260919_130000", mtime=2_000.0)

    assert rr._complete_mtime(root, "FOO") == 1_000.0
    assert rr._complete_mtime(root, "FOO_BAR") == 2_000.0


@pytest.mark.unit
def test_find_new_report_ignores_prefix_ticker(tmp_path):
    """A brand-new FOO_BAR report must not count as "FOO produced a report" —
    under the old glob, robust-runner retries for FOO could succeed on
    FOO_BAR's fresh output (and vice versa miss a real one)."""
    root = tmp_path / "reports"
    root.mkdir()
    foo_cr = _make_report_dir(root, "FOO_20260919_120000", mtime=1_000.0) / "complete_report.md"
    foobar_cr = _make_report_dir(root, "FOO_BAR_20260919_130000", mtime=2_000.0) / "complete_report.md"

    # Snapshot taken between the two writes: FOO_BAR's newer file is invisible
    # to FOO's search, and FOO's older file is invisible to FOO_BAR's search.
    assert rr._find_new_report(root, "FOO", pre_mtime=1_500.0) is None
    assert rr._find_new_report(root, "FOO_BAR", pre_mtime=1_500.0) == foobar_cr

    # Lower the bar and each ticker finds only its own file.
    assert rr._find_new_report(root, "FOO", pre_mtime=500.0) == foo_cr


@pytest.mark.unit
def test_find_new_report_ignores_non_stamp_dirs(tmp_path):
    """Dirs that merely share the prefix (no stamp shape) never qualify."""
    root = tmp_path / "reports"
    root.mkdir()
    _make_report_dir(root, "FOO_manual_backup", mtime=9_999.0)

    assert rr._find_new_report(root, "FOO", pre_mtime=0.0) is None
    assert rr._complete_mtime(root, "FOO") == 0.0
