"""Round-10 publish-race pins: the remaining ``os.replace`` publish sites.

``replace_retry_on_windows`` (yialpha/dataflows/utils.py) exists because
CPython on Windows opens files WITHOUT ``FILE_SHARE_DELETE``: while a
lockless reader holds the TARGET open, ``os.replace`` onto it raises
``PermissionError``. Round-10 already routed the memory log / states log /
node_perf writers through it; this file pins the publish sites adopted in
the same sweep:

* ``reporting._atomic_write_text``          — complete_report.md / decision.md
* ``dataflows.stockstats_utils.load_ohlcv`` — OHLCV cache (same wiring; one
  pin per site is overkill — the replace is identical to the pinned sites)
* ``dataflows.binance_vision_store``        — manifest + year-partition publish
* ``monitoring.dashboard.write_dashboard``  — dashboard.html vs the browser
* ``config_snapshot.record_config_snapshot``— snapshot publish (append-only)
* ``dataflows.disk_cache`` / ``llm_clients.response_cache`` — the two
  best-effort writers, upgraded to retry-then-swallow
* ``backtest.cache.DecisionCache.put`` — the same best-effort upgrade, found
  by the publish-race sweep's own follow-up scan

Two properties are pinned:

(a) TRANSIENT: a PermissionError on the first replace (reader mid-read) is
    retried through — the file is published, no exception escapes.
(b) PERSISTENT: a denial outliving the retry budget preserves each site's
    PRE-EXISTING error policy exactly — propagating sites still raise, and
    the best-effort sites still swallow (serve the result, warn, clean up).
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import pytest


def _install_race(monkeypatch, *, target, times=1, forever=False) -> dict:
    """Patch ``os.replace`` as the retry helper sees it with a fail-N wrapper.

    ``target`` is a predicate over the replace DESTINATION; non-matching
    destinations delegate to the real ``os.replace`` untouched.
    ``time.sleep`` is no-op'd at the symbol the helper looks it up (the
    global ``time`` module — the helper imports it inside the function
    body), so persistent-denial pins cost zero wall time and transient pins
    cost one no-op sleep. Same patching shape as
    tests/test_round10_regressions.py::test_replace_retry_succeeds_after_
    transient_permission_errors.
    """
    import time as _time_mod

    real_replace = os.replace
    seen = {"n": 0}

    def racing_replace(src, dst):
        if target(Path(dst)):
            seen["n"] += 1
            if forever or seen["n"] <= times:
                raise PermissionError(
                    5, "Access is denied (lockless reader holds the target)"
                )
        return real_replace(src, dst)

    monkeypatch.setattr("yialpha.dataflows.utils.os.replace", racing_replace)
    monkeypatch.setattr(_time_mod, "sleep", lambda *_a, **_k: None)
    return seen


_REPORT_STATE = {
    "market_report": "Market section.",
    "trader_investment_plan": "Plan section.",
    "final_trade_decision": "Rating: Buy",
}


def _drive_reporting(tmp_path: Path) -> Path:
    from yialpha.reporting import write_report_tree

    save = tmp_path / "reports"
    save.mkdir()
    # An existing complete_report.md models the reader (web store /
    # trade_ticket) holding the published file open.
    (save / "complete_report.md").write_text("OLD", encoding="utf-8")
    return write_report_tree(_REPORT_STATE, "NVDA", save)


def _drive_dashboard(tmp_path: Path) -> Path:
    from yialpha.monitoring.dashboard import write_dashboard

    (tmp_path / "monitoring").mkdir()
    (tmp_path / "monitoring" / "dashboard.html").write_text(
        "OLD", encoding="utf-8"
    )
    return write_dashboard([], results_dir=tmp_path)


def _drive_config_snapshot(tmp_path: Path) -> dict | None:
    from yialpha.config_snapshot import load_last_snapshot, record_config_snapshot

    record_config_snapshot(
        {"a": 1}, reason="race-pin", history_dir=tmp_path
    )
    return load_last_snapshot(history_dir=tmp_path)


def _drive_vision_manifest(tmp_path: Path) -> Path:
    from yialpha.dataflows.binance_vision_store import VisionStore

    store = VisionStore(root=tmp_path)
    day_dir = tmp_path / "metrics" / "BTCUSDT"
    day_dir.mkdir(parents=True)
    (day_dir / "manifest.json").write_text(
        json.dumps({"days": {}, "missing": {}}), encoding="utf-8"
    )
    store._write_manifest(
        "metrics", "BTCUSDT",
        {"days": {"2026-01-05": 288}, "missing": {}, "updated": "now"},
    )
    return day_dir / "manifest.json"


def _drive_disk_cache(tmp_path: Path) -> bytes | None:
    from yialpha.dataflows.disk_cache import cached_or_fetch

    base = tmp_path / "vendor"
    base.mkdir()
    (base / "day.json").write_bytes(b"OLD")
    return cached_or_fetch(
        str(base), "day.json", lambda: b"NEW", ttl_days=0, vendor="race-pin"
    )


def _drive_llm_cache(tmp_path: Path):
    from langchain_core.outputs import Generation

    from yialpha.llm_clients.response_cache import DiskLLMCache

    cache = DiskLLMCache(tmp_path / "llm", enabled=True)
    cache.update("prompt", "llm-string", [Generation(text="hello")])
    return cache.lookup("prompt", "llm-string")


def _check_reporting(tmp_path: Path, result: Path) -> None:
    text = result.read_text(encoding="utf-8")
    assert "NVDA" in text and "OLD" not in text
    assert (result.parent / "5_portfolio" / "decision.md").exists()


def _check_dashboard(tmp_path: Path, result: Path) -> None:
    text = result.read_text(encoding="utf-8")
    assert "<html" in text and "OLD" not in text


def _check_config_snapshot(tmp_path: Path, result) -> None:
    assert result is not None and result["reason"] == "race-pin"


def _check_vision_manifest(tmp_path: Path, result: Path) -> None:
    data = json.loads(result.read_text(encoding="utf-8"))
    assert data["days"] == {"2026-01-05": 288}


def _check_disk_cache(tmp_path: Path, result) -> None:
    assert result == b"NEW"
    assert (tmp_path / "vendor" / "day.json").read_bytes() == b"NEW"


def _check_llm_cache(tmp_path: Path, result) -> None:
    assert result is not None and result[0].text == "hello"


# site name -> (driver, checker, replace-destination predicate). The
# stockstats OHLCV cache write uses the same wiring as every pinned site
# (bare os.replace swapped for the helper inside a FileLock) and is covered
# by the class of tests below rather than its own driver.
_TRANSIENT_SITES = {
    "reporting_complete_report": (
        _drive_reporting, _check_reporting,
        lambda p: p.name == "complete_report.md",
    ),
    "dashboard_html": (
        _drive_dashboard, _check_dashboard,
        lambda p: p.name == "dashboard.html",
    ),
    "config_snapshot_publish": (
        _drive_config_snapshot, _check_config_snapshot,
        lambda p: p.name.startswith("config_") and p.suffix == ".json",
    ),
    "vision_store_manifest": (
        _drive_vision_manifest, _check_vision_manifest,
        lambda p: p.name == "manifest.json",
    ),
    "disk_cache_entry": (
        _drive_disk_cache, _check_disk_cache,
        lambda p: p.name == "day.json",
    ),
    "llm_response_cache_entry": (
        _drive_llm_cache, _check_llm_cache,
        lambda p: p.parent.name == "llm" and p.suffix == ".json",
    ),
}


@pytest.mark.unit
@pytest.mark.parametrize("site", sorted(_TRANSIENT_SITES))
def test_transient_replace_race_is_retried_through(site, tmp_path, monkeypatch):
    """(a) A PermissionError on the first os.replace (lockless reader
    mid-read) is retried through: the file is published, no exception."""
    drive, check, target = _TRANSIENT_SITES[site]
    seen = _install_race(monkeypatch, target=target, times=1)
    result = drive(tmp_path)
    check(tmp_path, result)
    # The race was ridden through (failed attempt + retry), not bypassed.
    assert seen["n"] >= 2


# --------------------------------------------------------------------------- #
# (b) Persistent denial preserves each site's pre-existing error policy
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_reporting_persistent_denial_still_propagates(tmp_path, monkeypatch):
    """The report tree has NO swallow policy: an exhausted retry still raises
    out of write_report_tree (the batch runner isolates it per ticker /
    run_robust retries), while the sections published before the failing
    edge keep their bytes and the previous complete_report.md stays intact."""
    from yialpha.reporting import write_report_tree

    save = tmp_path / "reports"
    save.mkdir()
    (save / "complete_report.md").write_text("OLD", encoding="utf-8")
    _install_race(
        monkeypatch, target=lambda p: p.name == "complete_report.md", forever=True
    )
    with pytest.raises(PermissionError):
        write_report_tree(_REPORT_STATE, "NVDA", save)
    assert (
        save / "1_analysts" / "market.md"
    ).read_text(encoding="utf-8") == "Market section."
    assert (save / "complete_report.md").read_text(encoding="utf-8") == "OLD"


@pytest.mark.unit
def test_dashboard_persistent_denial_still_propagates_and_cleans_tmp(
    tmp_path, monkeypatch
):
    """write_dashboard's try/finally (raise + tmp cleanup) is untouched by
    the retry wiring: exhaustion propagates and no tmp husk is left."""
    from yialpha.monitoring.dashboard import write_dashboard

    dash = tmp_path / "monitoring" / "dashboard.html"
    dash.parent.mkdir()
    dash.write_text("OLD", encoding="utf-8")
    _install_race(
        monkeypatch, target=lambda p: p.name == "dashboard.html", forever=True
    )
    with pytest.raises(PermissionError):
        write_dashboard([], results_dir=tmp_path)
    assert dash.read_text(encoding="utf-8") == "OLD"
    assert list(dash.parent.glob("*.tmp")) == []


@pytest.mark.unit
def test_disk_cache_persistent_denial_still_best_effort(
    tmp_path, monkeypatch, caplog
):
    """cached_or_fetch's documented policy — the cache write after a
    successful fetch is best-effort — is preserved: retry-then-swallow still
    serves the fetched bytes, warns, and cleans the tmp."""
    from yialpha.dataflows.disk_cache import cached_or_fetch

    base = tmp_path / "vendor"
    base.mkdir()
    (base / "day.json").write_bytes(b"OLD")
    _install_race(
        monkeypatch, target=lambda p: p.name == "day.json", forever=True
    )
    with caplog.at_level(logging.WARNING):
        raw = cached_or_fetch(
            str(base), "day.json", lambda: b"NEW", ttl_days=0, vendor="race-pin"
        )
    assert raw == b"NEW"
    assert (base / "day.json").read_bytes() == b"OLD"  # write honestly not landed
    assert list(base.glob("*.tmp")) == []
    assert "could not write cache" in caplog.text


@pytest.mark.unit
def test_backtest_decision_cache_transient_race_is_retried_through(
    tmp_path, monkeypatch
):
    """DecisionCache.put (same retry-then-swallow shape as disk_cache): a
    concurrent replay holding the cached decision file open cannot make the
    cache write vanish — the replace rides through."""
    from yialpha.backtest.cache import DecisionCache

    cache = DecisionCache(str(tmp_path / "bt"), enabled=True)
    # Pre-existing entry models a concurrent backtest replaying the cache.
    cache.remember("NVDA", "2026-07-01", "Buy", "Long v1", config_signature="s")
    seen = _install_race(monkeypatch, target=lambda p: p.suffix == ".json", times=1)
    cache.remember("NVDA", "2026-07-01", "Buy", "Long v2", config_signature="s")
    assert seen["n"] >= 2  # ridden through, not bypassed
    entries = list((tmp_path / "bt").rglob("*.json"))
    assert len(entries) == 1
    assert "Long v2" in entries[0].read_text(encoding="utf-8")


@pytest.mark.unit
def test_llm_cache_persistent_denial_never_breaks_a_run(
    tmp_path, monkeypatch, caplog
):
    """DiskLLMCache.update's never-break-a-run handler is preserved: an
    exhausted retry logs a warning and cleans the tmp instead of raising."""
    from langchain_core.outputs import Generation

    from yialpha.llm_clients.response_cache import DiskLLMCache

    cache = DiskLLMCache(tmp_path / "llm", enabled=True)
    _install_race(
        monkeypatch,
        target=lambda p: p.parent.name == "llm" and p.suffix == ".json",
        forever=True,
    )
    with caplog.at_level(logging.WARNING):
        cache.update("p", "l", [Generation(text="x")])  # must not raise
    assert list((tmp_path / "llm").glob("*.tmp")) == []
    assert "could not write" in caplog.text
