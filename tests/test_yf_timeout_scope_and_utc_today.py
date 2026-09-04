"""Scoped yfinance socket timeout + UTC "today" boundary tests.

Covers two 2026-09 fixes in dataflows:

* ``_scoped_yf_socket_timeout``: the process-default socket timeout is a
  GLOBAL — per-call save/restore across threads let an interleaved exit
  permanently pin the YF timeout as the process default. The fix is a
  depth-counted window (same pattern as ``timeout_shim``): only the outermost
  entrant saves and only the outermost leaver restores.
* "today" for the yfinance OHLCV cache freshness rule and the y_finance
  history-window TTL is anchored to UTC, not the host-local clock: on a host
  whose local date runs ahead of UTC (e.g. Asia/Shanghai past midnight) a
  still-open US session day was misclassified as immutable history, so a
  cached window never picked up that day's close.
"""

from __future__ import annotations

import socket
import threading
import time
from datetime import UTC, datetime

import pandas as pd
import pytest

from yialpha.dataflows import stockstats_utils as ssu, y_finance as yfm


@pytest.fixture(autouse=True)
def _isolated_socket_default():
    """Snapshot/restore the process socket default and the patch depth."""
    prev = socket.getdefaulttimeout()
    yield
    socket.setdefaulttimeout(prev)
    with ssu._yf_timeout_state_lock:
        ssu._yf_timeout_depth = 0
        ssu._yf_timeout_prev = None


@pytest.mark.unit
class TestScopedYfSocketTimeout:
    def test_interleaved_windows_do_not_pollute_process_default(self, monkeypatch):
        """Two threads enter/exit staggered: A's exit must NOT restore the
        original default while B's window is still active, and B's exit must
        restore it exactly once.

        The old per-call save/restore ended with the process default pinned to
        the YF timeout forever (last restorer wins) under this exact ordering.
        """
        monkeypatch.setattr(ssu, "YF_HTTP_TIMEOUT", 30.0)
        socket.setdefaulttimeout(None)

        a_enter, b_enter, a_exit = (
            threading.Event(), threading.Event(), threading.Event(),
        )
        seen: dict[str, float | None] = {}

        def worker_a():
            with ssu._scoped_yf_socket_timeout():
                seen["a_inside"] = socket.getdefaulttimeout()
                a_enter.set()
                assert a_exit.wait(5)
            seen["a_after"] = socket.getdefaulttimeout()

        def worker_b():
            assert a_enter.wait(5)
            cm = ssu._scoped_yf_socket_timeout()
            cm.__enter__()
            seen["b_inside"] = socket.getdefaulttimeout()
            b_enter.set()
            assert a_exit.wait(5)  # A leaves while B is still inside
            seen["b_after_a_exit"] = socket.getdefaulttimeout()
            cm.__exit__(None, None, None)
            seen["b_after_own_exit"] = socket.getdefaulttimeout()

        ta = threading.Thread(target=worker_a)
        tb = threading.Thread(target=worker_b)
        ta.start()
        tb.start()
        assert b_enter.wait(5)
        a_exit.set()  # A's window closes first
        ta.join(5)
        tb.join(5)

        assert seen["a_inside"] == 30.0
        assert seen["b_inside"] == 30.0
        # A's exit must not have restored the pre-existing default early.
        assert seen["b_after_a_exit"] == 30.0

        # Only the OUTERMOST exit restores — and to the true original (None).
        # Other tests' lingering worker threads may legitimately hold a window
        # of their own, so wait for the process-wide depth to drain before
        # pinning the restored value: with the old per-call save/restore the
        # default was pinned at 30.0 forever no matter how long we waited.
        deadline = time.monotonic() + 5
        while ssu._yf_timeout_depth > 0 and time.monotonic() < deadline:
            time.sleep(0.005)
        assert ssu._yf_timeout_depth == 0
        assert socket.getdefaulttimeout() is None

    def test_nested_contexts_count_depth_and_restore_nonzero_default(
        self, monkeypatch,
    ):
        monkeypatch.setattr(ssu, "YF_HTTP_TIMEOUT", 30.0)
        socket.setdefaulttimeout(7.0)  # a non-None pre-existing default
        with ssu._scoped_yf_socket_timeout():
            assert socket.getdefaulttimeout() == 30.0
            with ssu._scoped_yf_socket_timeout():
                assert socket.getdefaulttimeout() == 30.0
                assert ssu._yf_timeout_depth == 2
            # Inner exit must NOT restore 7.0 — outer window still active.
            assert socket.getdefaulttimeout() == 30.0
            assert ssu._yf_timeout_depth == 1
        assert socket.getdefaulttimeout() == 7.0
        assert ssu._yf_timeout_depth == 0

    def test_disabled_timeout_yields_without_touching_default(self, monkeypatch):
        monkeypatch.setattr(ssu, "YF_HTTP_TIMEOUT", None)
        socket.setdefaulttimeout(9.0)
        with ssu._scoped_yf_socket_timeout():
            assert socket.getdefaulttimeout() == 9.0
        assert socket.getdefaulttimeout() == 9.0
        assert ssu._yf_timeout_depth == 0


@pytest.mark.unit
class TestUtcTodayBoundaries:
    def test_utc_today_is_tz_aware_utc(self):
        now = ssu._utc_today()
        assert now.tzinfo is not None
        assert now.utcoffset() == pd.Timedelta(0)
        assert now.date() == datetime.now(UTC).date()

    def test_ohlcv_cache_window_ignores_host_local_today(self, monkeypatch):
        """Simulate a host clock running ahead of UTC (Asia/Shanghai past
        midnight): ``pd.Timestamp.today()`` says 2026-09-05 while the UTC date
        is 2026-09-04. The download window must derive from the UTC anchor —
        end = UTC-today + 1d = 09-05, NOT local-today + 1d = 09-06.
        """
        fixed_utc = pd.Timestamp("2026-09-04 19:00", tz="UTC")

        def _poisoned_local_today(cls):
            return pd.Timestamp("2026-09-05 03:00")  # naive "local" 8h ahead

        monkeypatch.setattr(pd.Timestamp, "today", classmethod(_poisoned_local_today))
        monkeypatch.setattr(ssu, "_utc_today", lambda: fixed_utc)

        start, end = ssu._ohlcv_cache_window()
        assert end == "2026-09-05"
        assert start == "2021-09-04"

    def test_same_day_refresh_classifies_by_utc_date(self, monkeypatch, tmp_path):
        """A cache file missing the requested UTC day must refresh; a
        historical day must never refresh — regardless of what the (poisoned)
        host-local clock claims today is."""
        import os
        import time as _time

        data_file = tmp_path / "X-YFin-data.csv"
        data_file.write_text("Date,Close\n2026-09-01,10.0\n")
        old = _time.time() - 10_000  # mtime far past the 15-min refresh TTL
        os.utime(data_file, (old, old))

        def _poisoned_local_today(cls):
            return pd.Timestamp("2026-09-07 03:00")  # "local" 3 days ahead of UTC

        monkeypatch.setattr(pd.Timestamp, "today", classmethod(_poisoned_local_today))

        utc_today = pd.Timestamp("2026-09-04 19:00", tz="UTC")
        # Same UTC day as the anchor: refresh rule applies -> stale file refreshes.
        assert ssu._needs_same_day_refresh(
            data_file, pd.Timestamp("2026-09-04"), utc_today
        ) is True
        # Strictly historical (even "yesterday" UTC): immutable, no refresh.
        assert ssu._needs_same_day_refresh(
            data_file, pd.Timestamp("2026-09-03"), utc_today
        ) is False

    def test_load_ohlcv_and_read_cache_take_today_from_utc_seam(
        self, monkeypatch, tmp_path,
    ):
        """Pin that load_ohlcv / read_cached_ohlcv resolve 'today' through
        _utc_today, never the host-local pd.Timestamp.today() (poisoned here
        to explode if the old direct call comes back)."""

        def _poisoned_local_today(cls):
            raise AssertionError("pd.Timestamp.today() must not be used for 'today'")

        monkeypatch.setattr(pd.Timestamp, "today", classmethod(_poisoned_local_today))
        monkeypatch.setattr(
            ssu, "_utc_today", lambda: pd.Timestamp("2026-01-01", tz="UTC"),
        )
        monkeypatch.setattr(
            ssu, "_ohlcv_cache_path", lambda config, safe: str(tmp_path / "a.csv"),
        )
        # read_cached_ohlcv: missing file -> honest None (today resolved fine).
        assert ssu.read_cached_ohlcv("AAPL", "2025-12-31") is None

        # load_ohlcv: missing cache -> download path; stop the trace there.
        def _stop(*args, **kwargs):
            raise AssertionError("stop-before-network")

        monkeypatch.setattr(ssu, "yf_retry", _stop)
        with pytest.raises(AssertionError, match="stop-before-network"):
            ssu.load_ohlcv("AAPL", "2025-12-31")


@pytest.mark.unit
class TestHistoryCacheTtlUtc:
    @staticmethod
    def _fake_datetime(fixed_utc: datetime, fixed_local: datetime):
        class _FakeDatetime(datetime):
            @classmethod
            def now(cls, tz=None):  # noqa: ANN001
                return fixed_utc if tz is not None else fixed_local

        return _FakeDatetime

    def test_window_reaching_utc_today_gets_short_ttl(self, monkeypatch):
        """UTC date 09-04, host-local (poisoned) date 09-05. A window ending on
        the UTC-current day — where the US session can still be open — must
        keep the short refresh TTL; the old local-clock comparison classified
        it as immutable history and pinned it in cache for 24h."""
        fake = self._fake_datetime(
            fixed_utc=datetime(2026, 9, 4, 19, 0, tzinfo=UTC),
            fixed_local=datetime(2026, 9, 5, 3, 0),
        )
        monkeypatch.setattr(yfm, "datetime", fake)

        short = yfm.OHLCV_CACHE_TTL_SECONDS / 86400.0
        assert yfm._history_cache_ttl_days("2026-09-04") == short  # UTC today
        assert yfm._history_cache_ttl_days("2026-09-05") == short  # future
        assert yfm._history_cache_ttl_days("2026-09-03") == 1.0    # fully past
        assert yfm._history_cache_ttl_days("not-a-date") == 1.0
