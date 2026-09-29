"""Parsed multi-year store for the data.binance.vision archives (PR6).

The raw-zip disk cache in :mod:`yialpha.dataflows.binance_vision` already
persists every downloaded day forever, but every query re-parsed the whole
window, a per-call file cap refused multi-year windows even when fully
cached, and the only semantic QA was a missing-day count. This module is
the repository layer on top:

* **Sync** — incremental, budgeted per call: days already in the store are
  never re-downloaded, NEW days download most-recent-first (the
  decision-adjacent days land first) until the per-call budget is spent;
  coverage deepens across successive queries. A 404 day is remembered (with
  a last-tried timestamp, retried at most once per day — unpublished days
  can appear later), so a pre-listing hole costs one attempt, not one per
  query.
* **Storage** — one ``data-<YYYY>.csv.gz`` partition per (dataset, symbol,
  year) plus a ``manifest.json`` with per-day row counts and the missing-day
  map. CSV.gz keeps the zero-new-dependency contract (pandas already
  round-trips it); a year of 5m metrics is ~100k rows, and multi-year loads
  touch a handful of files. Writes are lock-guarded (threads AND processes,
  via the shared file lock) and atomic (tmp + ``os.replace``).
* **QA** — the manifest makes the semantic checks the checksum cannot do:
  interior day holes inside the synced range, and low-row days (< 25% of
  the median day's rows — a truncated or partially-published file). The
  query path surfaces both in the tool header so a gap never reads as a
  calm continuous series.

The store never invents coverage: a window is served from synced days
only, and every absence (404, budget-unsynced, low-row) is disclosed.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd

from ..batch.locks import FileLock
from .disk_cache import sanitize_cache_filename, vendor_cache_dir

logger = logging.getLogger(__name__)

_STORE_CACHE = "binance_vision_store"

#: A remembered 404 is re-tried at most this often (an unpublished day can
#: appear later; retrying every query would burn the budget on holes).
_MISSING_RETRY_HOURS = 24.0


@dataclass
class SyncReport:
    """What one incremental sync did (all counts are window days)."""

    budget: int = 0
    downloaded: int = 0          # new days parsed into the store
    already_synced: int = 0      # window days already present
    missing: int = 0             # 404 days (recorded in the manifest)
    unsynced: int = 0            # days skipped because the budget ran out
    missing_days: list[str] = field(default_factory=list)
    window_days: int = 0

    @property
    def synced_days_count(self) -> int:
        return self.already_synced + self.downloaded


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _day_list(start_dt: datetime, end_dt: datetime) -> list[str]:
    days: list[str] = []
    day = start_dt.date()
    last = end_dt.date()
    while day <= last:
        days.append(day.isoformat())
        day += timedelta(days=1)
    return days


class VisionStore:
    """Year-partitioned parsed archive store under ``<data_cache_dir>``."""

    def __init__(self, root: str | Path | None = None):
        self.root = Path(root) if root is not None else Path(
            vendor_cache_dir(_STORE_CACHE)
        )

    # ---- paths -------------------------------------------------------------

    def _dir(self, dataset: str, symbol: str) -> Path:
        return self.root / sanitize_cache_filename(dataset) / sanitize_cache_filename(symbol)

    def _manifest_path(self, dataset: str, symbol: str) -> Path:
        return self._dir(dataset, symbol) / "manifest.json"

    def _year_path(self, dataset: str, symbol: str, year: int) -> Path:
        return self._dir(dataset, symbol) / f"data-{year}.csv.gz"

    # ---- manifest ----------------------------------------------------------

    def _read_manifest(self, dataset: str, symbol: str) -> dict:
        try:
            raw = self._manifest_path(dataset, symbol).read_text(encoding="utf-8")
            data = json.loads(raw)
            if not isinstance(data, dict):
                raise ValueError("manifest is not an object")
            days = data.get("days")
            missing = data.get("missing")
            if not isinstance(days, dict) or not isinstance(missing, dict):
                raise ValueError("manifest shape unexpected")
            return {"days": days, "missing": missing, "updated": data.get("updated")}
        except (OSError, ValueError, json.JSONDecodeError):
            # Missing (first use) or corrupt (crashed write): start fresh.
            # Year partitions survive, so already-synced days are re-derived
            # lazily by the next sync re-downloading from the ZIP cache
            # (bytes-level cache is untouched) — correctness over cleverness.
            return {"days": {}, "missing": {}, "updated": None}

    def _write_manifest(self, dataset: str, symbol: str, manifest: dict) -> None:
        path = self._manifest_path(dataset, symbol)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(
            json.dumps(manifest, sort_keys=True), encoding="utf-8"
        )
        os.replace(tmp, path)

    # ---- frame normalization (same semantics as the pre-store loader) -----

    @staticmethod
    def _normalize(frame: pd.DataFrame, what: str) -> pd.DataFrame:
        """Add a normalized UTC ``_ts`` (stored ISO), dedupe on the key.

        Mirrors the loader's contract: metrics dedupes on the time column,
        bookDepth on (time column, percentage) — deduping bookDepth on time
        alone would silently drop every band but the last."""
        from .errors import NoMarketDataError

        tcol = next(
            (c for c in ("create_time", "creation_time", "timestamp") if c in frame.columns),
            None,
        )
        if tcol is None:
            raise NoMarketDataError(
                what, what,
                f"{what} archive CSV has no time column (columns: {list(frame.columns)})",
            )
        ts = pd.to_datetime(frame[tcol], utc=True, errors="coerce")
        df = frame.assign(_ts=ts).dropna(subset=["_ts"])
        dedupe_keys = [tcol] + (
            ["percentage"] if "percentage" in df.columns else []
        )
        df = df.sort_values("_ts").drop_duplicates(subset=dedupe_keys, keep="last")
        df = df.copy()
        df["_ts"] = df["_ts"].dt.strftime("%Y-%m-%d %H:%M:%S")
        return df.reset_index(drop=True)

    # ---- sync --------------------------------------------------------------

    def sync(
        self,
        dataset: str,
        symbol: str,
        start_dt: datetime,
        end_dt: datetime,
        *,
        budget: int,
    ) -> SyncReport:
        """Incrementally extend the store over ``[start_dt, end_dt]``.

        New days download MOST-RECENT-FIRST (the decision-adjacent days
        land before deep history) until ``budget`` DOWNLOAD ATTEMPTS are
        spent — a 404 miss is a real network round trip and counts against
        the budget exactly like a download (without this, a symbol whose
        history is mostly missing would make an unbounded number of free
        404 requests per run); days already in the store cost nothing.
        A plain 404 (and a missing CHECKSUM sidecar, which classifies the
        day as un-verifiable rather than aborting the whole incremental
        sync) is recorded and retried at most once per day; all other
        transport/checksum failures propagate (fail-closed — the caller is
        the typed-error tool path). The whole read-modify-write runs under
        a per-(dataset, symbol) file lock so concurrent workers converge
        instead of racing the manifest.
        """
        from . import binance_vision as bnv  # lazy: binance_vision imports this module

        report = SyncReport(budget=budget)
        window = _day_list(start_dt, end_dt)
        report.window_days = len(window)

        directory = self._dir(dataset, symbol)
        directory.mkdir(parents=True, exist_ok=True)
        lock = FileLock(str(directory / "sync.lock"))
        with lock:
            manifest = self._read_manifest(dataset, symbol)
            # Round-9 self-review R1: claims whose year partition has
            # vanished (the load error's "delete it" remediation, or manual
            # cleanup) would never re-sync — pending only considers days
            # ABSENT from the manifest, so the account would keep claiming
            # coverage nothing can serve. Re-open claims with no backing
            # partition here, BEFORE pending is computed, so this call
            # re-downloads them (byte-level zip cache: zero network cost).
            for year in sorted({int(d[:4]) for d in manifest["days"]}):
                if not self._year_path(dataset, symbol, year).exists():
                    orphaned = [
                        d for d in manifest["days"]
                        if d.startswith(f"{year}-")
                    ]
                    for d in orphaned:
                        del manifest["days"][d]
                    logger.warning(
                        "vision store: manifest claims %d day(s) for %d but "
                        "partition %s is gone; re-opening them for re-sync",
                        len(orphaned), year,
                        self._year_path(dataset, symbol, year).name,
                    )
            # Pre-sync coverage: days already in the store BEFORE this call
            # (counted here — after the loop they would include this call's
            # downloads and double-report coverage).
            report.already_synced = sum(
                1 for d in window if d in manifest["days"]
            )
            retry_cutoff = (
                datetime.now(UTC) - timedelta(hours=_MISSING_RETRY_HOURS)
            ).isoformat()

            def _due(day: str) -> bool:
                last_try = manifest["missing"].get(day)
                return last_try is None or last_try < retry_cutoff

            pending = [
                d for d in reversed(window)
                if d not in manifest["days"] and _due(d)
            ]
            buckets: dict[int, list[pd.DataFrame]] = {}
            new_days: dict[int, list[str]] = {}
            touched_years: set[int] = set()
            for day in pending:
                if report.downloaded + report.missing >= budget:
                    break
                url = bnv._archive_url(dataset, symbol, day)
                fname = url.rsplit("/", 1)[-1]
                try:
                    raw = bnv._load_zip(url)
                    frame = self._normalize(bnv._zip_csv_dataframe(raw, fname), fname)
                except (bnv._ArchiveMissingError, bnv._ChecksumMissingError):
                    # A 404 archive or a missing CHECKSUM sidecar: the day
                    # is unverifiable either way — record it as missing
                    # (24h retry backoff) and keep syncing the rest instead
                    # of aborting the loop. A checksum MISMATCH stays a
                    # fail-closed _ChecksumError (never caught here), and
                    # the day is never served either way: only
                    # checksum-verified frames enter manifest["days"].
                    report.missing += 1
                    report.missing_days.append(day)
                    manifest["missing"][day] = _now_iso()
                    continue
                year = int(day[:4])
                buckets.setdefault(year, []).append(frame)
                new_days.setdefault(year, []).append(day)
                touched_years.add(year)
                manifest["days"][day] = int(len(frame))
                manifest["missing"].pop(day, None)
                report.downloaded += 1
            report.unsynced = (
                len(pending) - report.downloaded - report.missing
                if len(pending) > report.downloaded + report.missing
                else 0
            )
            for year in sorted(touched_years):
                self._merge_year(
                    dataset, symbol, year, buckets[year], manifest,
                    new_days[year],
                )
            # Round-9 self-review R4: a merge that found an unreadable
            # partition drops that year's OLD claims — recompute the
            # already-synced count from the post-merge manifest so the
            # report never claims coverage this very call removed.
            report.already_synced = max(
                0,
                sum(1 for d in window if d in manifest["days"])
                - report.downloaded,
            )
            manifest["updated"] = _now_iso()
            self._write_manifest(dataset, symbol, manifest)

        # Days recorded missing on a PREVIOUS call count toward coverage
        # accounting nowhere — they stay disclosed via qa()/report.
        return report

    def _merge_year(
        self,
        dataset: str,
        symbol: str,
        year: int,
        new_frames: list[pd.DataFrame],
        manifest: dict,
        merged_days: list[str],
    ) -> None:
        """Merge parsed day frames into the year partition (idempotent)."""
        path = self._year_path(dataset, symbol, year)
        frames = list(new_frames)
        if path.exists():
            try:
                frames.append(pd.read_csv(path, compression="gzip"))
            except (OSError, ValueError, EOFError) as exc:  # noqa: BLE001 — rebuild over crash
                # Round-9 S3: rebuilding from this call's frames alone while
                # manifest["days"] keeps claiming the old rows would serve
                # silently truncated deep history with zero disclosure (qa
                # is manifest-driven). Drop the year's OLD manifest days —
                # this call's freshly parsed days (merged_days) survive, the
                # rest become unsynced and re-download from the byte-level
                # zip cache on later syncs, and every absence flows through
                # the normal disclosure chain. EOFError joins the catch: a
                # byte-truncated gzip previously escaped uncaught and bricked
                # every later sync against the same dead read; it now takes
                # this same rebuild-from-cache repair path.
                keep = set(merged_days)
                dropped = [
                    d for d in manifest["days"]
                    if d.startswith(f"{year}-") and d not in keep
                ]
                for d in dropped:
                    del manifest["days"][d]
                logger.warning(
                    "vision store: unreadable partition %s (%s); rebuilding "
                    "from fresh downloads (%d manifest day(s) re-opened)",
                    path, exc, len(dropped),
                )
                path.unlink(missing_ok=True)
        if not frames:
            return
        merged = pd.concat(frames, ignore_index=True)
        tcol = next(
            (c for c in ("create_time", "creation_time", "timestamp") if c in merged.columns),
            "_ts",
        )
        dedupe_keys = [tcol] + (
            ["percentage"] if "percentage" in merged.columns else []
        )
        merged = (
            merged.assign(_ts=pd.to_datetime(merged["_ts"], utc=True, errors="coerce"))
            .dropna(subset=["_ts"])
            .sort_values("_ts")
            .drop_duplicates(subset=dedupe_keys, keep="last")
        )
        merged["_ts"] = merged["_ts"].dt.strftime("%Y-%m-%d %H:%M:%S")
        tmp = path.with_suffix(".csv.gz.tmp")
        merged.to_csv(tmp, index=False, compression="gzip")
        os.replace(tmp, path)

    # ---- load / QA ---------------------------------------------------------

    def load(
        self, dataset: str, symbol: str, start_dt: datetime, end_dt: datetime,
    ) -> pd.DataFrame:
        """Window frame from the year partitions (``_ts`` tz-aware, sorted).

        Empty (no columns) when nothing is synced — the caller raises the
        typed no-data error with the sync report's reasons."""
        from .errors import NoMarketDataError

        frames: list[pd.DataFrame] = []
        for year in range(start_dt.year, end_dt.year + 1):
            path = self._year_path(dataset, symbol, year)
            if not path.exists():
                continue
            try:
                frames.append(pd.read_csv(path, compression="gzip"))
            except (OSError, ValueError, EOFError) as exc:
                # Round-9 S3: this used to skip the year with a warning while
                # the manifest (and qa/headers driven by it) still counted
                # the partition's days — multi-year windows silently served
                # less than the account claimed, and the hole never repaired
                # (pending was empty, so sync never re-merged). Fail closed
                # instead: the typed error names the partition; deleting it
                # (or the next sync, which rebuilds from the byte-level zip
                # cache) restores coverage honestly.
                raise NoMarketDataError(
                    symbol, symbol,
                    f"vision store partition {path} unreadable "
                    f"({type(exc).__name__}: {exc}); delete it — the next "
                    "sync re-opens its manifest claims and rebuilds from "
                    "the cached archives",
                ) from exc
        if not frames:
            return pd.DataFrame()
        df = pd.concat(frames, ignore_index=True)
        df["_ts"] = pd.to_datetime(df["_ts"], utc=True, errors="coerce")
        df = df.dropna(subset=["_ts"])
        df = df[(df["_ts"] >= start_dt) & (df["_ts"] <= end_dt)]
        return df.sort_values("_ts").reset_index(drop=True)

    def qa(
        self,
        dataset: str,
        symbol: str,
        start_dt: datetime,
        end_dt: datetime,
    ) -> dict:
        """Semantic QA over the SYNCED range (checksums prove bytes, not
        continuity): interior day holes and low-row days.

        Returns ``{present_days, interior_missing_days, low_row_days,
        first_day, last_day, known_missing_days}`` — days outside the synced
        range are the sync report's business, not holes.
        ``known_missing_days`` lists the WINDOW EDGE days the manifest has
        already recorded as 404-missing: the sync's 24h retry backoff
        excludes them from every later call's attempt set, so without this
        key a backoff-suppressed edge day would disclose once on the call
        that 404'd and then silently vanish from the header (round-7)."""
        manifest = self._read_manifest(dataset, symbol)
        window = _day_list(start_dt, end_dt)
        present = [d for d in window if d in manifest["days"]]
        out: dict = {
            "present_days": len(present),
            "interior_missing_days": [],
            "low_row_days": [],
            "first_day": present[0] if present else None,
            "last_day": present[-1] if present else None,
            # Computed BEFORE the empty-present early return below: an edge
            # day that 404'd is exactly the case where present may be short.
            "known_missing_days": [
                d
                for d in dict.fromkeys((window[0], window[-1]))
                if d not in manifest["days"] and d in manifest["missing"]
            ],
        }
        if not present:
            return out
        # Interior holes: missing days BETWEEN the first and last synced day
        # (edges before the first synced day are simply unsynced history).
        first_idx = window.index(present[0])
        last_idx = window.index(present[-1])
        synced_set = set(present)
        out["interior_missing_days"] = [
            d for d in window[first_idx + 1:last_idx] if d not in synced_set
        ]
        # Low-row days: < 25% of the median synced day's rows (a truncated or
        # partially-published file). Only meaningful with a week of context.
        # The threshold stays a FLOAT — an int floor would round small
        # medians to zero and never fire.
        if len(present) >= 7:
            counts = [int(manifest["days"][d]) for d in present]
            median = sorted(counts)[len(counts) // 2]
            if median > 0:
                floor = median * 0.25
                out["low_row_days"] = [
                    d for d in present if int(manifest["days"][d]) < floor
                ]
        return out
