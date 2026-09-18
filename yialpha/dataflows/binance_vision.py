"""Binance public archive datasets (data.binance.vision) — deep-history perp data.

The REST ``/futures/data/*`` family (openInterestHist, the long/short ratios,
taker volume) retains only the last ~30 days, so every positioning feature in
this system is capped at a 30-day window and a daily grain. Binance however
publishes the SAME series (plus order-book depth) as free, immutable,
checksummed CSV archives under ``https://data.binance.vision``, going back
years. Schema and naming below were VERIFIED against the live S3 bucket on
2026-08-17 (download + unzip + header read, not doc copying — the repo docs'
underscore names and bookDepth columns do not match reality):

  - ``data/futures/um/daily/metrics/{SYM}/{SYM}-metrics-{YYYY-MM-DD}.zip``
    CSV columns (8): create_time, symbol, sum_open_interest,
    sum_open_interest_value, count_toptrader_long_short_ratio,
    sum_toptrader_long_short_ratio, count_long_short_ratio,
    sum_taker_long_short_vol_ratio. 5-minute grain (288 rows/day). The
    count_/sum_ prefixes are legacy naming residue, NOT counts/sums — every
    ratio column carries that 5m snapshot's VALUE (cross-checked against the
    live REST series on identical timestamps 2026-08-18: count_toptrader… =
    top-trader ACCOUNT ratio, sum_toptrader… = top-trader POSITION ratio,
    count_long_short… = GLOBAL account ratio), so the global long/short
    ratio IS servable from the archives.
  - ``data/futures/um/daily/bookDepth/{SYM}/{SYM}-bookDepth-{YYYY-MM-DD}.zip``
    CSV columns (4): timestamp, percentage, depth, notional. ~30-second grain
    (28.5k-34.5k rows/day). ``percentage`` is SIGNED: negative = bid-side
    band, positive = ask-side band; bands observed at ±0.2/1/2/3/4/5%.
    ``depth`` is base-asset units, ``notional`` is USD.

  IMPORTANT: unlike klines, these two datasets have NO monthly aggregates
  (S3 listing under ``monthly/metrics`` / ``monthly/bookDepth`` returns zero
  keys) — a window costs ONE request per NEW day, cached forever after. The
  parsed multi-year STORE (:mod:`yialpha.dataflows.binance_vision_store`,
  PR6) bounds new downloads to ~13 months per call while serving synced
  history without limit.

Each ``.zip`` carries a sibling ``.CHECKSUM`` (``<sha256>  <filename>``,
sha256sum format); every download here is verified fail-closed — a
mismatched or missing checksum NEVER serves data.

Outbound security: every request URL is validated against a fixed
scheme+host allowlist (https://data.binance.vision only — the host is a
module constant, never caller- or config-supplied), passes the shared
literal private/loopback IP guard, and redirects are refused, so a crafted
input can never turn an archive fetch into a probe of internal endpoints.

PIT safety is structural: an archive file for day D only contains rows from
D, so a window clamped to ``[start, min(end, PIT-end, yesterday-UTC)]``
can never leak future rows. Publication lags ~1 day, so the live current day
is simply not requestable from the archives (the REST tools cover the present).

Design constraints (mirrors :mod:`yialpha.dataflows.binance`):
  - side-effect-free import; proxies read at call time via ``proxy_map()``;
  - raises the typed errors from :mod:`yialpha.dataflows.errors` so the
    router degrades an optional-category miss to a sentinel;
  - no new dependencies (``requests`` + stdlib ``zipfile``/``hashlib``);
  - downloaded zips are cached under ``<data_cache_dir>/binance_vision/``
    through the shared :mod:`yialpha.dataflows.disk_cache` contract —
    archives are immutable, so the cache TTL is effectively unlimited.
"""

from __future__ import annotations

import hashlib
import io
import logging
import zipfile
from datetime import UTC, datetime, timedelta
from urllib.parse import urlsplit

import pandas as pd
import requests

from .binance import _validate_outbound_url
from .config import get_config
from .disk_cache import cached_or_fetch, vendor_cache_dir
from .errors import NoMarketDataError, VendorRateLimitError
from .symbol_utils import normalize_symbol_for_venue
from .utils import current_pit_end, proxy_map

logger = logging.getLogger(__name__)

_VISION_BASE = "https://data.binance.vision"
_VISION_HOST = "data.binance.vision"
# Binaries can be multi-MB zips — generous-but-bounded read phase.
_TIMEOUT = (5, 60)
# Cache namespace under <data_cache_dir>/; filename = the archive zip name.
_VENDOR_CACHE = "binance_vision"
# Archived files are immutable once published — a cached copy never goes
# stale, so the TTL only needs to outlive any plausible process lifetime.
_ARCHIVE_TTL_DAYS = 3650.0
# Per-call SYNC budget for the parsed store (PR6): NEW days are attempted
# most-recent-first (downloads AND 404 misses — a miss is a real network
# round trip and must not be free) until this many attempts are spent;
# already-synced days are read without limit, so multi-year windows work
# across successive queries (coverage deepens incrementally instead of
# being refused). 400 files ≈ 13 months of new history per call — the same
# bound the old hard cap enforced, now a rate bound instead of a coverage
# ceiling.
_SYNC_BUDGET_FILES = 400
_MAX_OUTPUT_ROWS = 2000
_MAX_RAW_ROWS = 20000

# Resample targets for the LLM-facing output.
_METRICS_INTERVALS = ("5m", "1h", "4h", "1d")
_DEPTH_INTERVALS = ("5m", "1h", "1d")


class _ArchiveMissingError(Exception):
    """The requested archive file does not exist (HTTP 404)."""


class _ArchiveHttpError(Exception):
    """Non-404 HTTP failure from data.binance.vision."""


class _ChecksumError(Exception):
    """Checksum missing or mismatched — the archive cannot be trusted."""


class _ChecksumMissingError(_ChecksumError):
    """The archive EXISTS but its CHECKSUM sidecar 404s.

    Distinct from a mismatch/unparseable checksum (corruption signals, which
    stay fail-closed): the sync loop treats a missing sidecar like a missing
    day — record, skip, retry on the 24h backoff — instead of aborting the
    whole incremental sync. The day is still never served: only
    checksum-verified frames enter the store.
    """


def _validate_vision_url(url: str) -> None:
    """Refuse anything that is not a plain https URL on the archive host.

    The host is a fixed module constant (never caller- or config-supplied),
    so this allowlist plus the shared literal private/loopback IP guard makes
    the outbound target provably the public Binance archive. Redirects are
    disabled at request time so a response cannot pivot the fetch elsewhere.
    """
    try:
        parts = urlsplit(url)
    except ValueError as exc:
        raise ValueError(f"unparseable outbound URL: {url!r}") from exc
    if parts.scheme != "https" or (parts.hostname or "").lower() != _VISION_HOST:
        raise ValueError(
            f"refusing non-archive outbound URL {url!r}: only "
            f"https://{_VISION_HOST} is allowed"
        )
    _validate_outbound_url(url)


def _fetch(url: str) -> bytes:
    """GET one data.binance.vision object, mapping statuses to typed errors.

    Raises :class:`_ArchiveMissingError` on 404 (missing dataset/symbol/date),
    :class:`VendorRateLimitError` on 429/418, :class:`_ArchiveHttpError` on any
    other non-200 (redirects included — they are refused); transport
    exceptions propagate for the caller to convert.
    """
    _validate_vision_url(url)
    resp = requests.get(
        url, proxies=proxy_map(), timeout=_TIMEOUT, allow_redirects=False,
    )
    if resp.status_code in (429, 418):
        raise VendorRateLimitError(
            f"data.binance.vision rate-limited (HTTP {resp.status_code})"
        )
    if resp.status_code == 404:
        raise _ArchiveMissingError(url.rsplit("/", 1)[-1])
    if resp.status_code != 200:
        snippet = (resp.text or "").strip()[:200]
        raise _ArchiveHttpError(f"HTTP {resp.status_code}: {snippet}")
    return resp.content


def _expected_sha256(checksum_body: bytes, filename: str) -> str:
    """Parse ``<sha256>  <filename>`` (``sha256sum`` format) fail-closed.

    The first whitespace-separated token of the first line must be 64 hex
    chars. Anything else means the checksum cannot be verified, which refuses
    the archive rather than trusting unverified bytes.
    """
    try:
        first_line = checksum_body.decode("ascii", errors="strict").strip().splitlines()[0]
    except (UnicodeDecodeError, IndexError) as exc:
        raise _ChecksumError(f"unparseable CHECKSUM for {filename}") from exc
    parts = first_line.split()
    token = parts[0].lower() if parts else ""
    if len(token) != 64 or any(c not in "0123456789abcdef" for c in token):
        raise _ChecksumError(f"malformed CHECKSUM for {filename}")
    return token


def _fetch_verified_zip(url: str) -> bytes:
    """Download one archive zip and verify its sha256 CHECKSUM.

    A mismatched OR missing CHECKSUM raises :class:`_ChecksumError` — the
    bytes are never returned, let alone cached or served. A cached file was
    verified when it was first written (archives are immutable), so the
    checksum cost is paid once per file per installation.
    """
    filename = url.rsplit("/", 1)[-1]
    raw = _fetch(url)
    try:
        checksum_body = _fetch(f"{url}.CHECKSUM")
    except _ArchiveMissingError as exc:
        raise _ChecksumMissingError(
            f"CHECKSUM file missing for {filename}"
        ) from exc
    expected = _expected_sha256(checksum_body, filename)
    actual = hashlib.sha256(raw).hexdigest()
    if expected != actual:
        raise _ChecksumError(
            f"sha256 mismatch for {filename}: expected {expected}, got {actual}"
        )
    return raw


def _load_zip(url: str) -> bytes:
    """Serve one archive zip from the shared disk cache, else download+verify.

    ``cached_or_fetch`` gives the repo-wide failure contract: a fetch failure
    falls back to a stale cached copy (recorded as data-quality evidence) and
    otherwise re-raises — here that is the 404 signal or a typed
    checksum/rate-limit failure.
    """
    filename = url.rsplit("/", 1)[-1]
    raw = cached_or_fetch(
        vendor_cache_dir(_VENDOR_CACHE),
        filename,
        lambda: _fetch_verified_zip(url),
        ttl_days=_ARCHIVE_TTL_DAYS,
        vendor=_VENDOR_CACHE,
    )
    if raw is None:  # fail_open is never set here — defensive, unreachable
        raise _ArchiveHttpError(f"{filename}: cache returned no bytes")
    return raw


def _zip_csv_dataframe(raw: bytes, what: str) -> pd.DataFrame:
    """Extract the single CSV member of one archive zip into a DataFrame."""
    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as zf:
            members = [n for n in zf.namelist() if n.lower().endswith(".csv")]
            if not members:
                raise _ArchiveHttpError(f"{what}: zip contains no CSV member")
            text = zf.read(members[0]).decode("utf-8-sig", errors="strict")
    except (zipfile.BadZipFile, UnicodeDecodeError) as exc:
        raise _ArchiveHttpError(f"{what}: not a readable zip/CSV ({exc})") from exc
    return pd.read_csv(io.StringIO(text))


# ---- window / file-plan resolution ------------------------------------------


def _last_published_day(now: datetime | None = None) -> datetime:
    """Start of the most recent UTC day whose archive is plausibly published.

    Daily files appear with ~1 day lag (observed: the 2026-08-11 file landed
    2026-08-12 06:32 UTC), so the live current day is never requestable from
    the archive (the REST tools cover the present).
    """
    now = now or datetime.now(UTC)
    return (now - timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)


def _resolve_window(
    symbol: str, canonical: str, start_date: str, end_date: str,
) -> tuple[datetime, datetime, str]:
    """Parse + PIT-clamp the window; cap the end at the last published day.

    Returns ``(start_dt, end_dt, note)`` — ``note`` is "" or an availability
    line for the header when the effective end is below the requested end
    (PIT clamp or publication lag).
    """
    start_dt = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=UTC)
    end_date = current_pit_end(end_date) or end_date
    requested_end = datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=UTC)
    end_dt = requested_end.replace(hour=23, minute=59, second=59)
    last_pub = _last_published_day().replace(hour=23, minute=59, second=59)
    note = ""
    if end_dt > last_pub:
        note = (
            f"# window end {requested_end.date()} exceeds available archives; "
            f"serving through {last_pub.date()} (publication lags ~1 day)\n"
        )
        end_dt = last_pub
    if start_dt > end_dt:
        raise NoMarketDataError(
            symbol, canonical,
            f"no published archive days in [{start_date}, {end_date}] "
            f"(PIT clamp / publication lag); archives cover through "
            f"{end_dt.date()}",
        )
    return start_dt, end_dt, note


def _archive_url(dataset: str, symbol: str, day: str) -> str:
    """One daily archive object URL (dash naming, verified 2026-08-17)."""
    fname = f"{symbol}-{dataset}-{day}.zip"
    return f"{_VISION_BASE}/data/futures/um/daily/{dataset}/{symbol}/{fname}"


def _archive_plan(
    dataset: str, symbol: str, start_dt: datetime, end_dt: datetime,
) -> list[tuple[str, str]]:
    """File plan for the window: one daily zip per UTC day, inclusive.

    These datasets publish daily files only — there are no monthly
    aggregates to prefer (verified via S3 listing; klines datasets do have
    them, but klines are served via the REST path elsewhere in this repo).
    """
    plan: list[tuple[str, str]] = []
    day = start_dt.date()
    last = end_dt.date()
    while day <= last:
        ds = day.isoformat()
        plan.append((_archive_url(dataset, symbol, ds), f"{symbol}-{dataset}-{ds}.zip"))
        day += timedelta(days=1)
    return plan


def _load_window_from_store(
    dataset: str, symbol: str,
    start_dt: datetime, end_dt: datetime,
    symbol_for_error: str, canonical: str,
) -> tuple[pd.DataFrame, object, dict]:
    """Sync + load the window through the parsed multi-year store (PR6).

    Returns ``(df, sync_report, qa)``. The store makes coverage
    INCREMENTAL: days already synced are read without limit (multi-year
    windows work), new days download most-recent-first until the per-call
    budget is spent, 404 days are remembered (retried at most once per day),
    and the semantic QA (interior holes, low-row days) rides the header so
    a gap never reads as a calm continuous series. Raises the typed no-data
    error when nothing is available for the window.
    """
    from .binance_vision_store import VisionStore

    store = VisionStore()
    report = store.sync(
        dataset, symbol, start_dt, end_dt, budget=_SYNC_BUDGET_FILES,
    )
    df = store.load(dataset, symbol, start_dt, end_dt)
    qa = store.qa(dataset, symbol, start_dt, end_dt)
    if df.empty:
        raise NoMarketDataError(
            symbol_for_error, canonical,
            f"no {dataset} archive data found for the window — the dataset "
            f"is not published for this symbol or the dates precede its "
            f"listing ({report.missing} day(s) 404'd, "
            f"{report.unsynced} unsynced within the per-call budget)",
        )
    return df, report, qa


# ---- output shaping -----------------------------------------------------------

# metrics column -> (output name, aggregation). Levels (open interest) take the
# bucket CLOSE (matches openInterestHist's daily snapshot semantics); ratios
# take the bucket MEAN (they are intensities, not levels). The count_/sum_
# prefixes are legacy naming residue — each ratio column carries the 5m
# snapshot's VALUE (REST-verified 2026-08-18): count_toptrader… = top-trader
# ACCOUNT ratio, sum_toptrader… = top-trader POSITION ratio, count_long_short…
# = GLOBAL account ratio. The sum_long_short_ratio spelling stays as a
# fallback for hypothetical older/future files; real ones carry count_ only
# (see the both-present precedence in _shape_metrics).
_METRICS_AGGS: dict[str, tuple[str, str]] = {
    "sum_open_interest": ("open_interest", "last"),
    "sum_open_interest_value": ("open_interest_value", "last"),
    "count_toptrader_long_short_ratio": (
        "top_trader_account_long_short_ratio", "mean",
    ),
    "sum_toptrader_long_short_ratio": ("top_trader_long_short_ratio", "mean"),
    "count_long_short_ratio": ("global_long_short_ratio", "mean"),
    "sum_long_short_ratio": ("global_long_short_ratio", "mean"),
    "sum_taker_long_short_vol_ratio": ("taker_buy_sell_ratio", "mean"),
}


def _format_ts(idx: pd.Series, interval: str) -> pd.Series:  # type: ignore[type-arg]
    fmt = "%Y-%m-%d" if interval == "1d" else "%Y-%m-%d %H:%M"
    return idx.dt.strftime(fmt)


def _shape_metrics(
    df: pd.DataFrame, interval: str,
    symbol_for_error: str, canonical: str,
) -> pd.DataFrame:
    # count_/sum_ spellings both map to global_long_short_ratio; real files
    # carry only count_, and when a (hypothetical) file carries both the
    # historical sum_ mapping wins — the output must never carry two columns
    # of the same name.
    if {"count_long_short_ratio", "sum_long_short_ratio"} <= set(df.columns):
        df = df.drop(columns=["count_long_short_ratio"])
    keep = [c for c in _METRICS_AGGS if c in df.columns]
    if not keep:
        raise NoMarketDataError(
            symbol_for_error, canonical,
            f"metrics archive CSV carries none of the expected columns "
            f"(columns: {list(df.columns)})",
        )
    out = df[keep + ["_ts"]].copy()
    for c in keep:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.dropna(subset=keep, how="all")
    if interval == "5m":
        if len(out) > _MAX_RAW_ROWS:
            raise NoMarketDataError(
                symbol_for_error, canonical,
                f"{len(out)} raw 5m rows exceed the {_MAX_RAW_ROWS}-row output "
                f"cap — request a coarser interval (1h/4h/1d) or a narrower window",
            )
        shaped = out.set_index("_ts").rename(
            columns={c: _METRICS_AGGS[c][0] for c in keep}
        )
    else:
        agg = {c: _METRICS_AGGS[c][1] for c in keep}
        shaped = (
            out.set_index("_ts")
            .resample(interval)
            .agg(agg)
            .rename(columns={c: _METRICS_AGGS[c][0] for c in keep})
            .dropna(how="all")
        )
    shaped = shaped.reset_index()
    shaped["time"] = _format_ts(shaped["_ts"], interval)
    return shaped.drop(columns=["_ts"])


def _shape_book_depth(
    df: pd.DataFrame, interval: str,
    symbol_for_error: str, canonical: str,
) -> pd.DataFrame:
    """Shape the verified bookDepth schema: timestamp, percentage, depth,
    notional — SIGNED percentage (negative = bid side, positive = ask side),
    ~30s source grain resampled to the requested interval per band."""
    if "percentage" not in df.columns:
        raise NoMarketDataError(
            symbol_for_error, canonical,
            f"bookDepth archive CSV has no percentage column "
            f"(columns: {list(df.columns)})",
        )
    keep = [c for c in ("depth", "notional") if c in df.columns]
    if not keep:
        raise NoMarketDataError(
            symbol_for_error, canonical,
            f"bookDepth archive CSV carries neither depth nor notional "
            f"(columns: {list(df.columns)})",
        )
    out = df[keep + ["percentage", "_ts"]].copy()
    for c in keep:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out["percentage"] = pd.to_numeric(out["percentage"], errors="coerce")
    out = out.dropna(subset=["percentage"])
    shaped = (
        out.groupby([pd.Grouper(key="_ts", freq=interval), "percentage"])[keep]
        .mean()
        .reset_index()
        .sort_values(["_ts", "percentage"])
    )
    shaped = shaped.reset_index(drop=True)
    shaped["time"] = _format_ts(shaped["_ts"], interval)
    return shaped.drop(columns=["_ts"])


def _enforce_output_cap(
    shaped: pd.DataFrame, symbol_for_error: str, canonical: str,
) -> pd.DataFrame:
    if len(shaped) > _MAX_OUTPUT_ROWS:
        raise NoMarketDataError(
            symbol_for_error, canonical,
            f"{len(shaped)} output rows exceed the {_MAX_OUTPUT_ROWS}-row cap — "
            f"narrow the window, request a coarser interval, or use the "
            f"daily summary mode (summary=true)",
        )
    return shaped


def _coverage_notes(report, qa: dict, start_dt=None, end_dt=None) -> str:
    """Header notes from the sync report + semantic QA — every absence is
    disclosed: interior holes, budget-unsynced days, low-row days, and a
    missing EDGE day (the requested start/end day itself not yet published
    — reachable for the end day in the ~06:30 UTC pre-publication window
    where ``_resolve_window``'s "exceeds available archives" note does not
    fire because the day is not past the publication cap, only 404ing)."""
    notes = ""
    interior = qa.get("interior_missing_days") or []
    if interior:
        notes += (
            f"# ⚠ {len(interior)} archive day(s) missing inside the window "
            "(pre-listing / unpublished days)\n"
        )
    if report is not None and getattr(report, "missing_days", None):
        missing = set(report.missing_days)
        edges = []
        if start_dt is not None and start_dt.strftime("%Y-%m-%d") in missing:
            edges.append(f"start {start_dt.date()}")
        if end_dt is not None and end_dt.strftime("%Y-%m-%d") in missing:
            edges.append(f"end {end_dt.date()}")
        if edges:
            notes += (
                f"# ⚠ window {' and '.join(edges)} day(s) not published yet "
                "(archive publication lags ~1 day; the REST tools cover the "
                "present)\n"
            )
    if report is not None and getattr(report, "unsynced", 0) > 0:
        notes += (
            f"# ⚠ store covers {report.synced_days_count} of "
            f"{report.window_days} window days (per-call sync budget "
            f"{report.budget} files; coverage deepens incrementally — "
            "re-query to extend history)\n"
        )
    low = qa.get("low_row_days") or []
    if low:
        notes += (
            f"# ⚠ {len(low)} synced day(s) carry far fewer rows than the "
            "median day (truncated / partially published file(s))\n"
        )
    return notes


def _stats_table(shaped: pd.DataFrame, cols: list[str]) -> str:
    """Per-column distribution stats over the FULL window (count/mean/
    p10/median/p90/min/max/first/last) — the summary the LLM reads instead
    of thousands of raw rows."""
    rows = [
        "stat," + ",".join(cols),
    ]
    series = {c: pd.to_numeric(shaped[c], errors="coerce") for c in cols}

    def _fmt(v: float) -> str:
        return "" if pd.isna(v) else f"{v:.6g}"

    defs = (
        ("count", lambda s: s.count()),
        ("mean", lambda s: s.mean()),
        ("p10", lambda s: s.quantile(0.10)),
        ("median", lambda s: s.quantile(0.50)),
        ("p90", lambda s: s.quantile(0.90)),
        ("min", lambda s: s.min()),
        ("max", lambda s: s.max()),
        ("first", lambda s: s.iloc[0] if len(s) else float("nan")),
        ("last", lambda s: s.iloc[-1] if len(s) else float("nan")),
    )
    for label, fn in defs:
        rows.append(label + "," + ",".join(_fmt(fn(series[c])) for c in cols))
    return "\n".join(rows)


def _render_metrics_summary(shaped: pd.DataFrame, tail_rows: int = 30) -> str:
    """Distribution summary + recent tail for the daily metrics frame."""
    cols = [c for c in shaped.columns if c != "time"]
    tail = shaped.tail(tail_rows)
    return (
        "## Distribution over the full window (daily aggregates)\n"
        + _stats_table(shaped, cols)
        + f"\n\n## Recent tail (last {len(tail)} day(s))\n"
        + tail.to_csv(index=False)
    )


_DEPTH_TAIL_DAYS = 14


def _render_depth_summary(shaped: pd.DataFrame) -> str:
    """Per-band depth summary + liquidity-thin streak + recent tail.

    The audit's liquidation-cascade question — "how long has the tight
    band been thin?" — needs a streak, not a mean: the longest run of
    consecutive days where the tightest band's best side fell below its
    own p25."""

    stats_rows = ["percentage,days,mean_notional,p10_notional,min_notional,mean_depth"]
    for pct, grp in shaped.groupby("percentage"):
        notional = pd.to_numeric(grp["notional"], errors="coerce").dropna()
        depth = pd.to_numeric(grp["depth"], errors="coerce").dropna()
        if notional.empty:
            continue
        stats_rows.append(
            f"{pct:.1f},{len(notional)},{notional.mean():.6g},"
            f"{notional.quantile(0.10):.6g},{notional.min():.6g},"
            f"{(depth.mean() if not depth.empty else float('nan')):.6g}"
        )

    collapse_note = ""
    if "percentage" in shaped.columns and not shaped.empty:
        tightest = min(abs(float(p)) for p in shaped["percentage"].unique())
        band = shaped[shaped["percentage"].abs() == tightest].copy()
        if not band.empty:
            per_day = band.groupby("time")["notional"].apply(
                lambda s: pd.to_numeric(s, errors="coerce").max()
            ).dropna()
            if len(per_day) >= 5:
                threshold = per_day.quantile(0.25)
                thin = (per_day < threshold).to_numpy()
                streak = best = 0
                for flag in thin:
                    streak = streak + 1 if flag else 0
                    best = max(best, streak)
                if best > 0:
                    collapse_note = (
                        f"\n## Liquidity-thin streak\n"
                        f"Longest run of consecutive days with the ±{tightest:g}% "
                        f"band's best-side notional below its own p25: "
                        f"{best} day(s) (window p25 = "
                        f"{threshold:.6g}).\n"
                    )

    tail = shaped.tail(_DEPTH_TAIL_DAYS * max(1, shaped["percentage"].nunique()))
    return (
        "## Per-band distribution over the full window (daily means)\n"
        + "\n".join(stats_rows)
        + collapse_note
        + f"\n## Recent tail (last {_DEPTH_TAIL_DAYS} day(s))\n"
        + tail.to_csv(index=False)
    )


def _vision_header(
    title: str, label: str, start_dt: datetime, end_dt: datetime,
    note: str, interval: str, rows: int, semantics: str = "",
    rows_note: str | None = None,
) -> str:
    header = (
        f"# {title} for {label} from {start_dt.date()} to {end_dt.date()}\n"
        f"# source: data.binance.vision public archives (sha256-verified, deep "
        f"history beyond the 30-day REST retention; one cached file per day)\n"
    )
    if note:
        header += note
    header += f"# interval: {interval}\n"
    # rows_note overrides the plain record count (summary mode reports what
    # the numbers summarize instead of a raw row count).
    header += rows_note or f"# Total records: {rows}\n"
    header += f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    if semantics:
        header += semantics
    header += "\n"
    return header


def _label(symbol: str, canonical: str) -> str:
    return canonical if canonical == symbol.upper() else f"{canonical} (from {symbol})"


def get_binance_vision_metrics(
    symbol: str,
    start_date: str,
    end_date: str,
    interval: str = "1d",
    summary: bool | None = None,
) -> str:
    """Deep-history derivative metrics for a Binance USDT-M perp (archived).

    Serves the data.binance.vision ``metrics`` dataset — 5-minute open
    interest, the top-trader ACCOUNT and POSITION long/short ratios, the
    GLOBAL account long/short ratio and the taker buy/sell volume ratio —
    back YEARS (BTCUSDT coverage starts 2020-09), where the REST endpoints
    retain only 30 days. The archive's count_/sum_ column prefixes are
    legacy naming for the VALUE (REST-verified on identical timestamps), so
    this carries the same three ratio series as the REST
    ``get_binance_long_short_ratio`` tool, whose 30-day retention it extends
    by years. This is the deep-history positioning pillar: regime analysis
    across funding cycles, IC work on positioning features, and PIT-correct
    positioning context for historical replay dates.

    Multi-year queries run through the parsed local STORE (PR6): synced
    days are read without limit, new days download most-recent-first under
    a per-call budget (~400 files ≈ 13 months of NEW history per call) and
    coverage deepens across successive queries — every hole (404 day,
    unsynced day, low-row day) is disclosed in the header.

    ``interval`` resamples the 5m source: ``"1d"`` (default) reports the
    day-close open interest and day-mean ratios; ``"5m"`` returns raw rows.
    At the daily grain, ``summary`` (default on, config
    ``binance_vision_summary``) renders a distribution table over the FULL
    window (count/mean/p10/median/p90/min/max/first/last per column) plus a
    recent 30-day tail — instead of thousands of raw rows; pass
    ``summary=False`` for the raw daily CSV (subject to the output cap).
    Every zip is sha256-verified before use. The window is PIT-clamped to
    the run's analysis date and capped at the last published archive day
    (publication lags ~1 day).
    """
    if interval not in _METRICS_INTERVALS:
        raise ValueError(
            f"interval must be one of {_METRICS_INTERVALS}, got {interval!r}"
        )
    canonical = normalize_symbol_for_venue(symbol, "binance_perp")
    start_dt, end_dt, note = _resolve_window(symbol, canonical, start_date, end_date)
    try:
        df, report, qa = _load_window_from_store(
            "metrics", canonical, start_dt, end_dt, symbol, canonical,
        )
        shaped = _shape_metrics(df, interval, symbol, canonical)
    except (_ArchiveHttpError, _ChecksumError, requests.RequestException) as exc:
        raise NoMarketDataError(
            symbol, canonical, f"data.binance.vision metrics unavailable: {exc}",
        ) from exc
    note += _coverage_notes(report, qa, start_dt, end_dt)
    if shaped.empty:
        raise NoMarketDataError(
            symbol, canonical,
            f"no metrics rows in [{start_dt.date()}, {end_dt.date()}]",
        )
    use_summary = (
        summary if summary is not None
        else get_config().get("binance_vision_summary", True)
    ) and interval == "1d"
    if use_summary:
        body = _render_metrics_summary(shaped)
        rows_note = (
            f"# Total records summarized: {len(shaped)} daily rows "
            f"({qa.get('first_day')} → {qa.get('last_day')} synced)\n"
            "# summary mode: distribution over the full window + recent "
            "tail (config binance_vision_summary=false serves the raw "
            "daily CSV)\n"
        )
    else:
        shaped = _enforce_output_cap(shaped, symbol, canonical)
        body = shaped.to_csv(index=False)
        rows_note = None
    header = _vision_header(
        "Perp USDT-M deep-history metrics (OI/long-short/taker)",
        _label(symbol, canonical), start_dt, end_dt, note, interval,
        len(shaped),
        semantics=(
            "# open_interest: base-asset units (day-close for 1d); "
            "open_interest_value: USDT; top_trader_account_long_short_ratio / "
            "top_trader_long_short_ratio > 1 = top traders' accounts / "
            "positions long-dominated; global_long_short_ratio > 1 = all "
            "traders' accounts long-dominated (top-vs-global divergence is a "
            "contrary signal); taker_buy_sell_ratio > 1 = taker buy pressure.\n"
        ),
        rows_note=rows_note,
    )
    return header + body


def get_binance_vision_book_depth(
    symbol: str,
    start_date: str,
    end_date: str,
    interval: str = "1d",
    summary: bool | None = None,
) -> str:
    """Historical order-book depth for a Binance USDT-M perp (archived).

    Serves the data.binance.vision ``bookDepth`` dataset — resting depth
    within SIGNED ±``percentage`` price bands of mid (negative = bid side,
    positive = ask side; bands at ±0.2/1/2/3/4/5%), ~30-second source grain,
    back years. Depth persistence is the liquidation-cascade context — a
    thin book into a falling price means slippage amplifies forced selling;
    a thick book absorbing a dump signals real demand.

    Multi-year queries run through the parsed local STORE (PR6): synced
    days are read without limit (the raw output cap no longer bounds
    history), new days download most-recent-first under a per-call budget,
    and every hole is disclosed. At the daily grain ``summary`` (default
    on, config ``binance_vision_summary``) renders per-band distributions
    (mean/p10/min notional per signed band), the longest liquidity-thin
    streak (consecutive days with the tightest band below its own p25) and
    a recent 14-day tail; pass ``summary=False`` for the raw per-band daily
    CSV (subject to the output cap). PIT-clamped, sha256-verified, same
    failure contract as the metrics tool.
    """
    if interval not in _DEPTH_INTERVALS:
        raise ValueError(
            f"interval must be one of {_DEPTH_INTERVALS}, got {interval!r}"
        )
    canonical = normalize_symbol_for_venue(symbol, "binance_perp")
    start_dt, end_dt, note = _resolve_window(symbol, canonical, start_date, end_date)
    try:
        df, report, qa = _load_window_from_store(
            "bookDepth", canonical, start_dt, end_dt, symbol, canonical,
        )
        shaped = _shape_book_depth(df, interval, symbol, canonical)
    except (_ArchiveHttpError, _ChecksumError, requests.RequestException) as exc:
        raise NoMarketDataError(
            symbol, canonical, f"data.binance.vision bookDepth unavailable: {exc}",
        ) from exc
    note += _coverage_notes(report, qa, start_dt, end_dt)
    if shaped.empty:
        raise NoMarketDataError(
            symbol, canonical,
            f"no bookDepth rows in [{start_dt.date()}, {end_dt.date()}]",
        )
    use_summary = (
        summary if summary is not None
        else get_config().get("binance_vision_summary", True)
    ) and interval == "1d"
    if use_summary:
        body = _render_depth_summary(shaped)
        rows_note = (
            f"# Total records summarized: {len(shaped)} daily band rows "
            f"({qa.get('first_day')} → {qa.get('last_day')} synced)\n"
            "# summary mode: per-band distribution + liquidity streak + "
            "recent tail (config binance_vision_summary=false serves the "
            "raw per-band CSV)\n"
        )
    else:
        shaped = _enforce_output_cap(shaped, symbol, canonical)
        body = shaped.to_csv(index=False)
        rows_note = None
    header = _vision_header(
        "Perp USDT-M order-book depth history",
        _label(symbol, canonical), start_dt, end_dt, note, interval, len(shaped),
        semantics=(
            "# percentage is SIGNED (negative = bid side, positive = ask side); "
            "depth = base-asset units, notional = USD; thin depth into a "
            "falling price = slippage / liquidation-cascade risk.\n"
        ),
        rows_note=rows_note,
    )
    return header + body
