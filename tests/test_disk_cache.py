"""Unit tests for the shared on-disk cache helpers (dataflows/disk_cache).

Covers every branch of ``cached_or_fetch`` (fresh hit, miss+fetch+write,
stale-on-failure, fail_open, unwritable cache dir), the filename allowlist,
the containment check, and the throttle spacing.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

from yiagents.dataflows import disk_cache as dc


def _fetch_ok(payload: bytes = b"data"):
    calls = {"n": 0}

    def fetch() -> bytes:
        calls["n"] += 1
        return payload

    return fetch, calls


# --------------------------------------------------------------------------- #
# safe_cache_component
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_safe_cache_component_flattens_disallowed_chars():
    # EDGAR primaryDocument values carry rendering prefixes with a path
    # separator; the flattened result must be a valid single component.
    flattened = dc.safe_cache_component("xslF345X05/wk-form4_20250130.xml")
    assert flattened == "xslF345X05_wk-form4_20250130.xml"
    dc.sanitize_cache_filename(flattened)  # must not raise
    assert dc.safe_cache_component("a b:c%") == "a_b_c_"


# --------------------------------------------------------------------------- #
# cached_or_fetch
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_fresh_cache_served_without_fetch(tmp_path):
    fetch, calls = _fetch_ok()
    (tmp_path / "f.json").write_bytes(b"cached")
    # mtime is now -> fresh for any positive TTL
    out = dc.cached_or_fetch(str(tmp_path), "f.json", fetch, ttl_days=1.0, vendor="t")
    assert out == b"cached"
    assert calls["n"] == 0


@pytest.mark.unit
def test_miss_fetches_writes_and_returns(tmp_path):
    fetch, calls = _fetch_ok(b"fresh-bytes")
    out = dc.cached_or_fetch(str(tmp_path), "f.json", fetch, ttl_days=1.0, vendor="t")
    assert out == b"fresh-bytes"
    assert calls["n"] == 1
    assert (tmp_path / "f.json").read_bytes() == b"fresh-bytes"


@pytest.mark.unit
def test_expired_cache_refetches(tmp_path):
    path = tmp_path / "f.json"
    path.write_bytes(b"old")
    # Backdate the cache file beyond the TTL.
    old_mtime = time.time() - 10 * 86_400
    import os

    os.utime(path, (old_mtime, old_mtime))
    fetch, calls = _fetch_ok(b"new")
    out = dc.cached_or_fetch(str(tmp_path), "f.json", fetch, ttl_days=1.0, vendor="t")
    assert out == b"new"
    assert calls["n"] == 1


@pytest.mark.unit
def test_stale_served_when_fetch_fails(tmp_path, caplog):
    path = tmp_path / "f.json"
    path.write_bytes(b"stale-data")
    old_mtime = time.time() - 3 * 86_400
    import os

    os.utime(path, (old_mtime, old_mtime))

    def boom() -> bytes:
        raise RuntimeError("network down")

    with caplog.at_level("WARNING"):
        out = dc.cached_or_fetch(
            str(tmp_path), "f.json", boom, ttl_days=1.0, vendor="vendorx"
        )
    assert out == b"stale-data"
    assert any(
        "vendorx" in r.message and "STALE" in r.message and "3.0 days" in r.message
        for r in caplog.records
    )


@pytest.mark.unit
def test_failure_without_cache_raises_by_default(tmp_path):
    def boom() -> bytes:
        raise RuntimeError("network down")

    with pytest.raises(RuntimeError, match="network down"):
        dc.cached_or_fetch(str(tmp_path), "f.json", boom, ttl_days=1.0, vendor="t")


@pytest.mark.unit
def test_stale_over_cap_refused_fail_closed(tmp_path, caplog):
    """A cache older than data_cache_max_stale_days must NOT be served when
    the vendor fails — an arbitrarily old cache is worse than an honest error."""
    from yiagents.dataflows import config as cfgmod, quality

    path = tmp_path / "f.json"
    path.write_bytes(b"ancient")
    old_mtime = time.time() - 40 * 86_400
    os.utime(path, (old_mtime, old_mtime))

    orig = cfgmod.get_config()
    try:
        cfgmod.set_config({**orig, "data_cache_max_stale_days": 30})
        quality.ensure_run_context()
        with caplog.at_level("WARNING"), pytest.raises(RuntimeError, match="network down"):
            dc.cached_or_fetch(
                str(tmp_path), "f.json", boom_fn, ttl_days=1.0, vendor="vendorx"
            )
        assert any("refusing to serve stale" in r.message for r in caplog.records)
        # Refused serve records no stale sentinel — the raise propagates to the
        # vendor's typed-error path instead.
        assert quality.summarize_quality(quality.snapshot_quality())[
            "stale_cache_count"
        ] == 0
    finally:
        cfgmod.set_config(orig)
        quality.reset_quality()


def boom_fn() -> bytes:
    raise RuntimeError("network down")


@pytest.mark.unit
def test_stale_within_cap_served_with_sentinel(tmp_path, caplog):
    from yiagents.dataflows import config as cfgmod, quality

    path = tmp_path / "f.json"
    path.write_bytes(b"stale-data")
    old_mtime = time.time() - 3 * 86_400
    os.utime(path, (old_mtime, old_mtime))

    orig = cfgmod.get_config()
    try:
        cfgmod.set_config({**orig, "data_cache_max_stale_days": 30})
        quality.ensure_run_context()
        with caplog.at_level("WARNING"):
            out = dc.cached_or_fetch(
                str(tmp_path), "f.json", boom_fn, ttl_days=1.0, vendor="vendorx"
            )
        assert out == b"stale-data"
        events = quality.snapshot_quality()
        assert len(events) == 1
        assert events[0]["kind"] == quality.KIND_STALE_CACHE
        assert events[0]["method"] == "vendorx/f.json"
        assert "3.0 days" in events[0]["detail"]
    finally:
        cfgmod.set_config(orig)
        quality.reset_quality()


@pytest.mark.unit
def test_cap_zero_never_serves_stale(tmp_path):
    from yiagents.dataflows import config as cfgmod

    path = tmp_path / "f.json"
    path.write_bytes(b"stale-data")
    # 3-day-old mtime so the TTL freshness check misses, but cap 0 means
    # "never serve stale" regardless of age.
    old_mtime = time.time() - 3 * 86_400
    os.utime(path, (old_mtime, old_mtime))
    orig = cfgmod.get_config()
    try:
        cfgmod.set_config({**orig, "data_cache_max_stale_days": 0})
        with pytest.raises(RuntimeError, match="network down"):
            dc.cached_or_fetch(
                str(tmp_path), "f.json", boom_fn, ttl_days=0.001, vendor="t"
            )
    finally:
        cfgmod.set_config(orig)


@pytest.mark.unit
def test_failure_without_cache_returns_none_when_fail_open(tmp_path, caplog):
    def boom() -> bytes:
        raise RuntimeError("network down")

    with caplog.at_level("WARNING"):
        out = dc.cached_or_fetch(
            str(tmp_path), "f.json", boom, ttl_days=1.0, vendor="t", fail_open=True
        )
    assert out is None
    assert any("no cache exists" in r.message for r in caplog.records)


@pytest.mark.unit
def test_unwritable_cache_target_still_returns_fetched_bytes(tmp_path, caplog):
    # base_dir pointing AT a regular file makes the joined cache path
    # unwritable (NotADirectoryError, an OSError) — the fetch already
    # succeeded, so the write failure must only log and never propagate.
    blocker = tmp_path / "blocker"
    blocker.write_bytes(b"x")
    fetch, _ = _fetch_ok(b"payload")
    with caplog.at_level("WARNING"):
        out = dc.cached_or_fetch(
            str(blocker), "sub.json", fetch, ttl_days=1.0, vendor="t"
        )
    assert out == b"payload"
    assert any("could not write cache" in r.message for r in caplog.records)


# --------------------------------------------------------------------------- #
# filename allowlist + containment
# --------------------------------------------------------------------------- #
@pytest.mark.unit
@pytest.mark.parametrize(
    "bad",
    ["../escape", "..\\escape", "a/b", "a\\b", "", ".", "..", "C:/x", "a b\tc"],
)
def test_unsafe_filenames_rejected(tmp_path, bad):
    with pytest.raises(ValueError):
        dc.cached_or_fetch(
            str(tmp_path), bad, lambda: b"x", ttl_days=1.0, vendor="t"
        )


@pytest.mark.unit
def test_containment_and_sanitize_helpers(tmp_path):
    base = tmp_path / "base"
    base.mkdir()
    # The realpath containment check keeps a path inside its base...
    assert dc.cache_file_path(str(base), "daily_sh.600519.json") == base / "daily_sh.600519.json"
    # ...and the allowlist accepts exactly the vendor filename shapes in use.
    assert dc.sanitize_cache_filename("daily_sh.600519.json") == "daily_sh.600519.json"
    assert dc.sanitize_cache_filename("factors_5.zip") == "factors_5.zip"
    assert dc.sanitize_cache_filename("13f_2024q1.zip") == "13f_2024q1.zip"


@pytest.mark.unit
def test_cache_file_path_roundtrip(tmp_path):
    p = dc.cache_file_path(str(tmp_path), "ok-name_1.json")
    assert p == tmp_path / "ok-name_1.json"
    assert str(p).startswith(str(tmp_path))


# --------------------------------------------------------------------------- #
# vendor_cache_dir
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_vendor_cache_dir_creates_subdir(tmp_path, monkeypatch):
    from yiagents.dataflows import config as cfgmod

    orig = cfgmod.get_config()
    try:
        cfgmod.set_config({**orig, "data_cache_dir": str(tmp_path)})
        out = dc.vendor_cache_dir("unittest")
        assert out == str(tmp_path / "unittest")
        assert Path(out).is_dir()
    finally:
        cfgmod.set_config(orig)


# --------------------------------------------------------------------------- #
# MinIntervalThrottle
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_throttle_spaces_calls():
    t = dc.MinIntervalThrottle(0.05)
    start = time.monotonic()
    t.wait()
    t.wait()
    elapsed = time.monotonic() - start
    assert elapsed >= 0.045  # second call slept for the remaining interval


@pytest.mark.unit
def test_throttle_no_sleep_when_spaced():
    t = dc.MinIntervalThrottle(0.2)
    t._last = time.monotonic() - 5.0  # last call long ago
    start = time.monotonic()
    t.wait()
    assert time.monotonic() - start < 0.1


# --------------------------------------------------------------------------- #
# json round-trip shape used by baostock
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_json_roundtrip_through_bytes_cache(tmp_path):
    rows = [{"date": "2024-01-02", "close": "10.5"}, {"date": "2024-01-03", "close": "10.6"}]

    def fetch() -> bytes:
        return json.dumps(rows, ensure_ascii=False).encode("utf-8")

    raw = dc.cached_or_fetch(str(tmp_path), "daily_x.json", fetch, ttl_days=1.0, vendor="t")
    assert json.loads(raw) == rows
