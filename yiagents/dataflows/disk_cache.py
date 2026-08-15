"""Shared on-disk caching / throttling helpers for data vendors.

Before this module existed, the same TTL-cache-read/fetch/stale-fallback loop
was copy-pasted per vendor (eastmoney, sec_edgar, baostock) plus a fourth
variant in ``backtest/factor_model.py`` — four implementations of one contract
that had already drifted (different warning formats, one bypassing the config
layer to read ``YIAGENTS_CACHE_DIR`` directly). This module is the single
implementation of that contract:

* **fresh** cache (mtime within ``ttl_days``) is served without a network call;
* on a miss the caller's ``fetch`` runs and the result is written best-effort;
* on a fetch failure a **stale** cache is served with a WARNING that includes
  the cache age, but only up to ``YIAGENTS_DATA_CACHE_MAX_STALE_DAYS`` (default
  30) — beyond the cap the failure re-raises instead (an arbitrarily old cache
  is worse than an honest error). Every stale serve also records a
  ``stale_cache`` data-quality sentinel, because the call "succeeds" and the
  router would otherwise never see the degradation;
* if no cache exists either, the failure either re-raises (fail-closed, the
  vendor turns it into its typed error) or returns ``None`` (``fail_open=True``,
  for advisory data whose absence must not abort a run).

Also hosts :class:`MinIntervalThrottle`, the thread-safe minimum spacing that
eastmoney/sec_edgar each implemented privately, and :func:`vendor_cache_dir`,
the single resolver for ``<data_cache_dir>/<vendor>/`` (config-aware, with the
``~/.yiagents/cache`` fallback in exactly one place).

Security: every cache filename passes :func:`sanitize_cache_filename` (an
allowlist — only ``[A-Za-z0-9._-]`` characters, and never ``.``/``..``), and the
joined path is re-checked with ``os.path.commonpath`` against the normalized
``base_dir``. A ticker-derived filename containing ``..`` or a path separator
can therefore never resolve outside the cache directory.
"""

from __future__ import annotations

import logging
import os
import re
import threading
import time
from collections.abc import Callable
from pathlib import Path

from . import quality
from .config import get_config

logger = logging.getLogger(__name__)

#: Fallback cache root when the config carries no ``data_cache_dir``. Kept in
#: one place; every vendor cache resolves through :func:`cache_base_dir`.
_DEFAULT_CACHE_BASE = os.path.join(os.path.expanduser("~"), ".yiagents", "cache")

#: A cache filename must be one plain component of these characters only.
#: Anything containing a separator, ``..``, a drive letter, or shell/percent
#: metacharacters is rejected before it is ever joined onto the cache dir.
_SAFE_FILENAME_RE = re.compile(r"^[A-Za-z0-9._\-()]+$")


def cache_base_dir() -> str:
    """Resolve the shared on-disk cache root (config-aware)."""
    cfg = get_config()
    return cfg.get("data_cache_dir") or _DEFAULT_CACHE_BASE


def vendor_cache_dir(name: str) -> str:
    """Return ``<cache-root>/<name>``, creating it if missing."""
    path = os.path.join(cache_base_dir(), name)
    os.makedirs(path, exist_ok=True)
    return path


def sanitize_cache_filename(filename: str) -> str:
    """Validate a cache filename against a strict allowlist and return it.

    Cache filenames embed ticker-derived strings (e.g. ``daily_sh.600519.json``).
    This allowlist (mirrors ``safe_ticker_component``'s contract) guarantees
    the value is a single plain path component: no separators, no ``..``, no
    absolute/drive forms. Raises ``ValueError`` on anything else.
    """
    if (
        not filename
        or filename in (".", "..")
        or not _SAFE_FILENAME_RE.fullmatch(filename)
    ):
        raise ValueError(
            f"unsafe cache filename {filename!r}: must match "
            f"{_SAFE_FILENAME_RE.pattern} (single path component)"
        )
    return filename


def cache_file_path(base_dir: str, filename: str) -> Path:
    """Build the on-disk cache path from sanitized parts.

    The filename is allowlist-validated (:func:`sanitize_cache_filename`) and
    the joined result is containment-checked against the normalized
    ``base_dir`` (symlinks resolved), so the returned path provably stays
    inside the cache directory.
    """
    clean = sanitize_cache_filename(filename)
    real_path = os.path.realpath(os.path.join(base_dir, clean))
    real_base = os.path.realpath(base_dir)
    try:
        common = os.path.commonpath([real_path, real_base])
    except ValueError:  # different drives (Windows) — definitely an escape
        common = ""
    if common != real_base:
        raise ValueError(
            f"cache path {filename!r} resolves outside the cache directory "
            f"{real_base!r}"
        )
    return Path(base_dir) / clean


#: Complement of the allowlist above, for normalizing (not validating).
_UNSAFE_FILENAME_RE = re.compile(r"[^A-Za-z0-9._\-()]")


def safe_cache_component(value: str) -> str:
    """Flatten any string into a single safe cache-filename component.

    Unlike :func:`sanitize_cache_filename`, which *validates* and raises,
    this *normalizes*: every character outside the allowlist becomes ``_``.
    Use it for filename parts derived from external identifiers — e.g. EDGAR
    ``primaryDocument`` values carry rendering prefixes with a path separator
    (``xslF345X05/wk-form4_20250130.xml``) — so the cache key can never fail
    the data call or silently split into a bogus subdirectory.
    """
    return _UNSAFE_FILENAME_RE.sub("_", value)


class MinIntervalThrottle:
    """Thread-safe minimum spacing between consecutive calls.

    Each vendor holds one instance; :meth:`wait` blocks (holding the instance
    lock, so callers queue up) until ``min_interval`` has elapsed since the
    last released call. Uses a monotonic clock, so wall-clock adjustments
    (NTP steps, DST) cannot create negative or huge sleeps.
    """

    def __init__(self, min_interval: float) -> None:
        self._min_interval = float(min_interval)
        self._lock = threading.Lock()
        self._last = 0.0

    def wait(self) -> None:
        with self._lock:
            elapsed = time.monotonic() - self._last
            if elapsed < self._min_interval:
                time.sleep(self._min_interval - elapsed)
            self._last = time.monotonic()


def _mtime_age_days(path: Path) -> float:
    try:
        return (time.time() - path.stat().st_mtime) / 86_400.0
    except OSError:
        return float("inf")


def max_stale_days() -> float:
    """Configured cap (days) beyond which a stale cache is refused on failure.

    ``YIAGENTS_DATA_CACHE_MAX_STALE_DAYS`` (default 30). ``0`` is valid and
    means "never serve stale" (fully fail-closed); negative/garbage values
    fall back to the default rather than silently disabling the cap.
    """
    try:
        n = float(get_config().get("data_cache_max_stale_days", 30))
    except (TypeError, ValueError):
        return 30.0
    return n if n >= 0 else 30.0


def cached_or_fetch(
    base_dir: str,
    filename: str,
    fetch: Callable[[], bytes],
    *,
    ttl_days: float,
    vendor: str,
    fail_open: bool = False,
) -> bytes | None:
    """Serve ``base_dir/filename`` from a fresh cache, else fetch and cache.

    The on-disk path is built exclusively inside this function via
    :func:`cache_file_path` (allowlisted single-component filename,
    containment-checked against the normalized ``base_dir``) — callers never
    hand in a raw filesystem path.

    Failure policy (identical for every vendor — see module docstring):

    * fetch fails + a stale cache exists -> WARNING (with age) + serve stale;
    * fetch fails + no cache -> re-raise, or return ``None`` when
      ``fail_open=True`` (advisory data only);
    * the cache write after a successful fetch is best-effort (an unwritable
      cache directory must never fail a data call that already succeeded).

    ``ttl_days`` may be fractional (e.g. ``1/24`` for one hour).
    """
    # Sanitized + containment-checked inside cache_file_path; the validated
    # Path is the only object ever opened for this cache entry.
    validated = cache_file_path(base_dir, filename)

    stale: bytes | None = None
    stale_mtime: float | None = None
    try:
        stale = validated.read_bytes()
        # Capture the mtime while the bytes are in hand: a separate stat after
        # the read races with deletion (TOCTOU) and would discard fresh bytes
        # we already hold.
        stale_mtime = validated.stat().st_mtime
    except OSError:
        stale = None

    if stale is not None and stale_mtime is not None and (
        time.time() - stale_mtime
    ) < ttl_days * 86_400.0:
        return stale

    try:
        raw = fetch()
    except Exception:
        if stale is not None:
            # Age is computed from the captured mtime when possible; a file
            # that vanished since the read counts as infinitely old.
            age = (
                (time.time() - stale_mtime) / 86_400.0
                if stale_mtime is not None
                else _mtime_age_days(validated)
            )
            cap = max_stale_days()
            if age > cap:
                # Fail-closed: an arbitrarily old cache is worse than an
                # honest error — the vendor layer turns this into its typed
                # error and the router records a sentinel.
                logger.warning(
                    "%s: fetch failed and cache %s is %.1f days old (over the "
                    "%.0f-day cap); refusing to serve stale data",
                    vendor, validated, age, cap,
                )
                if fail_open:
                    return None
                raise
            logger.warning(
                "%s: fetch failed; serving STALE cache for %s (age %.1f days)",
                vendor, validated, age,
            )
            # The serve "succeeds", so the router would never see the failure
            # — record it as data-quality evidence instead so the run's
            # report/DEGRADED verdict can reflect it.
            quality.record_sentinel(
                f"{vendor}/{filename}",
                quality.KIND_STALE_CACHE,
                f"served stale cache (age {age:.1f} days) after fetch failure",
            )
            return stale
        if fail_open:
            logger.warning(
                "%s: fetch failed for %s and no cache exists", vendor, validated
            )
            return None
        raise

    try:
        validated.write_bytes(raw)
    except OSError as exc:  # noqa: BLE001 -- caching is best-effort
        logger.warning("%s: could not write cache %s: %s", vendor, validated, exc)
    return raw
