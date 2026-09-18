"""Rating↔outcome verification: did the PM's ratings actually play out?

The pipeline archives every completed run as
``<results_dir>/<TICKER>/YiAlphaStrategy_logs/full_states_log_<date>.json``
with the PM's rating (``pm_rating`` / the decision markdown) and — since the
T0-5 batch — the decision-time close (``price_at_decision``). This module is
the assembly that was missing: it scans those logs, fetches the PIT forward
return for each (ticker, date, rating), and computes the aggregates that turn
"the report looks professional" into a measurable hit rate.

Design notes:

* **PIT honesty.** Forward returns are computed from the vendor's dated daily
  closes only — baseline is the close at/after the decision date, endpoint is
  the close ``holding_days`` sessions later. A decision whose horizon has not
  fully elapsed is reported as ``pending`` and excluded from the aggregates
  (scoring partial horizons would bias every recent decision toward whatever
  the market did in the interim).
* **Direction semantics.** Buy/Overweight are directional-up calls,
  Sell/Underweight directional-down; the hit rate covers directional calls
  only. Hold entries are reported separately with their mean forward return
  (the opportunity-cost view) and never counted as hits or misses.
* **Asset routing.** ``asset_type`` from the log selects the price source:
  ``crypto_perp``/``crypto_spot`` use the Binance klines frame (24/7 daily
  bars, mark-price venue), everything else the yfinance cached history. Logs
  predating ``asset_type`` get a conservative heuristic: a ``USDT``-suffixed
  ticker is priced as a Binance perp, everything else as spot equity — the
  report marks these so a misrouted legacy record is visible, not silent.
* **Old logs without ``price_at_decision``** are still scorable: the baseline
  is simply the vendor's close on the decision date (PIT-safe by
  construction), and the archived price, when present, is reported for
  cross-checking.

The heavy lifting runs in the CLI (``yialpha verify-history``) which writes
``<results_dir>/accuracy/accuracy_report.{json,md}``; the web layer serves
that artifact read-only (``/api/accuracy``) rather than doing vendor fetches
inside a request handler.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from statistics import fmean
from typing import Any

import pandas as pd

from yialpha.agents.utils.rating import parse_rating

logger = logging.getLogger(__name__)

#: Directional (position-taking) ratings — the only ones a hit rate covers.
_BULLISH = {"buy", "overweight"}
_BEARISH = {"sell", "underweight"}

_CRYPTO_ASSET_TYPES = {"crypto_perp", "crypto_spot"}


@dataclass(frozen=True)
class DecisionRecord:
    """One archived decision extracted from a full_states_log file."""

    ticker: str
    trade_date: str
    rating: str  # canonical Title-cased 5-tier rating
    asset_type: str
    price_at_decision: float | None
    price_basis: str  # logged basis, "legacy" when the log predates T0-5
    asset_source: str  # "logged" | "inferred" — inference is a heuristic
    log_path: str


# --------------------------------------------------------------------------- #
# History scan
# --------------------------------------------------------------------------- #
def scan_history(results_dir: str | Path | None = None) -> list[DecisionRecord]:
    """Extract one DecisionRecord per readable full_states_log file.

    Unreadable JSON files are skipped with a warning (a half-written legacy
    log must not sink the whole scan); rating falls back through
    ``pm_rating`` → decision-markdown ``parse_rating`` (its own default of
    Hold matches the pipeline's signal processor).
    """
    root = Path(results_dir) if results_dir else _default_results_dir()
    records: list[DecisionRecord] = []
    if not root.is_dir():
        return records
    for log_path in sorted(root.glob("*/*/full_states_log_*.json")):
        try:
            data = json.loads(log_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Skipping unreadable log %s: %s", log_path, exc)
            continue
        ticker = data.get("company_of_interest") or log_path.parent.parent.name
        trade_date = str(data.get("trade_date") or "")[:10]
        if not ticker or not trade_date:
            continue
        rating = _normalize_rating(data)
        asset_type, source = _resolve_asset_type(
            str(ticker), data.get("asset_type")
        )
        price = data.get("price_at_decision")
        basis = data.get("price_at_decision_basis") or "legacy"
        records.append(
            DecisionRecord(
                ticker=str(ticker),
                trade_date=trade_date,
                rating=rating,
                asset_type=asset_type,
                price_at_decision=float(price) if isinstance(price, (int, float)) else None,
                price_basis=str(basis),
                asset_source=source,
                log_path=str(log_path),
            )
        )
    records.sort(key=lambda r: (r.ticker, r.trade_date))
    return records


def _default_results_dir() -> Path:
    from yialpha.default_config import DEFAULT_CONFIG

    return Path(DEFAULT_CONFIG["results_dir"])


def _normalize_rating(data: dict[str, Any]) -> str:
    """Prefer the structured pm_rating; fall back to markdown parsing."""
    pm = str(data.get("pm_rating") or "").strip()
    if pm:
        parsed = parse_rating(pm, warn_on_default=False)
        if parsed.lower() != "hold" or "hold" in pm.lower():
            return parsed
    return parse_rating(str(data.get("final_trade_decision") or ""))


def _resolve_asset_type(ticker: str, logged: Any) -> tuple[str, str]:
    """Map the log's asset_type onto a pricing venue.

    ``crypto`` (the umbrella value, still produced by auto-detect) prices as
    crypto_spot — matching what the LIVE pipeline did for those runs
    (routing maps the umbrella to crypto_spot and the graph prices it on the
    Yahoo spot series; scoring those decisions on Binance PERP klines would
    fold basis/funding divergence into outcomes the call never traded —
    round-3 alignment with yialpha.graph.routing). A missing asset_type
    (pre-T0-5 log) is inferred from the ticker: ``USDT``-suffixed → Binance
    perp, else spot equity. Inference is reported in the record so a
    misrouted legacy record is visible.
    """
    logged_str = str(logged or "").strip().lower()
    if logged_str in _CRYPTO_ASSET_TYPES:
        return logged_str, "logged"
    if logged_str == "crypto":
        return "crypto_spot", "logged"
    if logged_str == "stock" or not logged_str:
        inferred = (
            "crypto_perp"
            if ticker.upper().replace("-", "").endswith("USDT")
            else "stock"
        )
        return inferred, "logged" if logged_str == "stock" else "inferred"
    return "stock", "logged"


# --------------------------------------------------------------------------- #
# Forward returns
# --------------------------------------------------------------------------- #
def _crypto_daily_frame(
    ticker: str,
    asset_type: str,
    start: str,
    end: str,
    cutoff: datetime | None = None,
) -> pd.DataFrame:
    """Daily Binance klines for a crypto asset, pinned to CLOSED bars.

    The one crypto fetch seam in this module: ``closed_as_of`` drops the
    current UTC day's still-forming bar (crypto trades 24/7, so on the
    horizon's boundary day the frame would otherwise carry today's intraday
    partial close and a near-zero return's hit/miss label could flip by the
    day's close while the report already recorded it as final). A ``cutoff``
    additionally caps visibility at that day's UTC close — replaying
    resolution as of an older date must not see bars closed after it. Same
    seam outcome_compute._perp_close_series uses; UTC-everywhere (a naive
    cutoff would shift the boundary by the host's offset).
    """
    from yialpha.dataflows.binance import _now_ms, binance_klines_frame

    closed_ms = _now_ms()
    if cutoff is not None:
        closed_ms = min(
            closed_ms,
            int(
                (cutoff + timedelta(days=1)).replace(tzinfo=UTC).timestamp() * 1000
            ) - 1,
        )
    return binance_klines_frame(
        ticker, start, end, interval="1d",
        venue=f"binance_{asset_type.split('_', 1)[1]}",
        closed_as_of=closed_ms,
    )


def _dated_close(
    ticker: str, asset_type: str, start: str, end: str
) -> pd.Series:
    """Dated close series for the asset's native venue (empty on failure)."""
    try:
        if asset_type in _CRYPTO_ASSET_TYPES:
            frame = _crypto_daily_frame(ticker, asset_type, start, end)
            series = frame["Close"].astype(float)
            series.index = pd.to_datetime(series.index).date
            return series
        from yialpha.dataflows.symbol_utils import normalize_symbol
        from yialpha.dataflows.y_finance import get_YFin_history_cached

        frame = get_YFin_history_cached(normalize_symbol(ticker), start, end)
        series = frame["Close"].astype(float)
        series.index = pd.to_datetime(series.index).date
        return series
    except Exception as exc:  # noqa: BLE001 -- one record must not sink the scan
        logger.warning("Forward-price fetch failed for %s (%s): %s",
                       ticker, asset_type, exc)
        return pd.Series(dtype=float)


def forward_outcome(
    record: DecisionRecord, holding_days: int = 5
) -> dict[str, Any] | None:
    """Forward return for one record, or ``None`` when not yet scoreable.

    ``None`` means the horizon has not fully elapsed (pending) or the vendor
    has no data (delisted/gap) — both are reported as excluded, never scored.
    """
    try:
        start_dt = datetime.strptime(record.trade_date, "%Y-%m-%d")
    except ValueError:
        return None
    end = (start_dt + timedelta(days=holding_days * 2 + 10)).strftime("%Y-%m-%d")
    closes = _dated_close(record.ticker, record.asset_type, record.trade_date, end)
    if len(closes) <= holding_days:
        return None
    baseline = float(closes.iloc[0])
    endpoint = float(closes.iloc[holding_days])
    if baseline <= 0:
        return None
    return {
        "raw_return": (endpoint - baseline) / baseline,
        "baseline_close": baseline,
        "end_close": endpoint,
        "days": holding_days,
    }


def fetch_returns_yf(
    ticker: str,
    trade_date: str,
    benchmark: str = "SPY",
    holding_days: int = 5,
    as_of_date: str | None = None,
    asset_type: str | None = None,
) -> tuple[float | None, float | None, int | None]:
    """Raw + alpha return over ``holding_days`` from ``trade_date``.

    Same semantics as ``YiAlphaGraph._fetch_returns`` (which delegates here):
    weekend-buffered window, ``as_of_date`` PIT cutoff with exclusive-end
    trimming, and ``(None, None, None)`` when either leg has no usable rows.
    Used by the memory-log resolution path so the graph and the standalone
    ``memory-resolve`` CLI cannot drift apart.

    ``asset_type`` routes the ASSET leg: a crypto_perp/crypto_spot decision
    is priced on the venue it actually traded (Binance klines, forming bar
    dropped via ``closed_as_of`` — the same seam ``_dated_close`` uses),
    because ``normalize_symbol`` maps BTCUSDT to the Yahoo SPOT symbol and a
    perp entry's basis/funding divergence would otherwise fold silently into
    the "did the call play out" number; a tokenized-stock perp (MUUSDT) is
    not on Yahoo at all and would never resolve. ``None`` (equities, legacy
    callers) keeps the yfinance path byte-identical. The BENCHMARK leg is
    always yfinance — alpha is measured against the index either way.
    """
    try:
        start = datetime.strptime(trade_date, "%Y-%m-%d")
        end = start + timedelta(days=holding_days + 7)  # weekend/holiday buffer
        cutoff = None
        if as_of_date is not None:
            cutoff = datetime.strptime(str(as_of_date)[:10], "%Y-%m-%d")
            if cutoff <= start:
                return None, None, None
            end = min(end, cutoff + timedelta(days=1))
        end_str = end.strftime("%Y-%m-%d")

        from yialpha.dataflows.y_finance import get_YFin_history_cached

        if asset_type in _CRYPTO_ASSET_TYPES:
            stock = _crypto_daily_frame(ticker, asset_type, trade_date, end_str, cutoff)
        else:
            from yialpha.dataflows.symbol_utils import normalize_symbol

            stock = get_YFin_history_cached(normalize_symbol(ticker), trade_date, end_str)
        bench = get_YFin_history_cached(benchmark, trade_date, end_str)

        if cutoff is not None:
            # PIT belt-and-suspenders: trim dated frames again on the client.
            def _through_cutoff(frame: pd.DataFrame) -> pd.DataFrame:
                try:
                    return frame[frame.index.date <= cutoff.date()]  # type: ignore[attr-defined]
                except (AttributeError, TypeError):
                    return frame

            stock = _through_cutoff(stock)
            bench = _through_cutoff(bench)

        if len(stock) < 2 or len(bench) < 2:
            return None, None, None

        if cutoff is not None:
            # A reflection labelled as a five-session outcome must not come
            # from a partial horizon.
            if len(stock) <= holding_days or len(bench) <= holding_days:
                return None, None, None
            actual_days = holding_days
        else:
            actual_days = min(holding_days, len(stock) - 1, len(bench) - 1)
        raw = float(
            (stock["Close"].iloc[actual_days] - stock["Close"].iloc[0])
            / stock["Close"].iloc[0]
        )
        bench_ret = float(
            (bench["Close"].iloc[actual_days] - bench["Close"].iloc[0])
            / bench["Close"].iloc[0]
        )
        return raw, raw - bench_ret, actual_days
    except Exception as exc:  # noqa: BLE001 -- outcome resolution is best-effort
        logger.warning(
            "Could not resolve outcome for %s on %s vs %s: %s",
            ticker, trade_date, benchmark, exc,
        )
        return None, None, None


# --------------------------------------------------------------------------- #
# Aggregation + reports
# --------------------------------------------------------------------------- #
def _outcome_label(rating: str, raw_return: float | None) -> str:
    if raw_return is None:
        return "pending"
    low = rating.lower()
    if low in _BULLISH:
        return "hit" if raw_return > 0 else "miss"
    if low in _BEARISH:
        return "hit" if raw_return < 0 else "miss"
    return "neutral"  # Hold: no directional claim to score


def build_accuracy_report(
    records: list[DecisionRecord], holding_days: int = 5
) -> dict[str, Any]:
    """Aggregate forward outcomes across all scanned records."""
    rows: list[dict[str, Any]] = []
    by_rating: dict[str, dict[str, Any]] = {}
    by_ticker: dict[str, dict[str, Any]] = {}
    directional = {"n": 0, "hits": 0}
    hold_returns: list[float] = []
    pending = 0

    for record in records:
        outcome = forward_outcome(record, holding_days=holding_days)
        raw = outcome["raw_return"] if outcome else None
        label = _outcome_label(record.rating, raw)
        if label == "pending":
            pending += 1
        row = {
            **asdict(record),
            "outcome": label,
            "forward_return": raw,
        }
        if outcome is not None:
            row["baseline_close"] = outcome.get("baseline_close")
        rows.append(row)

        if raw is not None:
            _bucket(by_rating, record.rating)["returns"].append(raw)
            bucket_t = _bucket(by_ticker, record.ticker)
            bucket_t["returns"].append(raw)
            if label in ("hit", "miss"):
                directional["n"] += 1
                directional["hits"] += 1 if label == "hit" else 0
                _bucket(by_rating, record.rating)["directional"] += 1
                _bucket(by_rating, record.rating)["hits"] += (
                    1 if label == "hit" else 0
                )
                bucket_t["directional"] += 1
                bucket_t["hits"] += 1 if label == "hit" else 0
            elif label == "neutral":
                hold_returns.append(raw)

    def _finalize(buckets: dict[str, dict[str, Any]]) -> dict[str, Any]:
        finalized = {}
        for key, bucket in sorted(buckets.items()):
            rets = bucket["returns"]
            finalized[key] = {
                "n": len(rets),
                "mean_return": fmean(rets) if rets else None,
                "directional_n": bucket["directional"],
                "hits": bucket["hits"],
                "hit_rate": (
                    bucket["hits"] / bucket["directional"]
                    if bucket["directional"]
                    else None
                ),
            }
        return finalized

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "holding_days": holding_days,
        "total_runs": len(records),
        "scored": len(records) - pending,
        "pending": pending,
        "direction": {
            **directional,
            "hit_rate": (
                directional["hits"] / directional["n"]
                if directional["n"] else None
            ),
        },
        "hold_mean_return": fmean(hold_returns) if hold_returns else None,
        "hold_n": len(hold_returns),
        "by_rating": _finalize(by_rating),
        "by_ticker": _finalize(by_ticker),
        "records": rows,
    }


def _bucket(store: dict[str, dict[str, Any]], key: str) -> dict[str, Any]:
    if key not in store:
        store[key] = {
            "returns": [], "directional": 0, "hits": 0,
        }
    return store[key]


def render_markdown(report: dict[str, Any]) -> str:
    """Human-facing accuracy report (tables always carry sample sizes)."""
    d = report["direction"]
    lines = [
        "# Rating accuracy report",
        "",
        f"- Generated: {report['generated_at']}  ",
        f"- Horizon: {report['holding_days']} sessions  ",
        f"- Runs scanned: {report['total_runs']} "
        f"(scored {report['scored']}, pending {report['pending']})",
        "",
        "## Directional hit rate (Buy/Overweight/Sell/Underweight)",
        "",
        f"- **{d['hits']}/{d['n']} correct ({d['hit_rate']:.1%})**"
        if d["hit_rate"] is not None
        else "- No fully-elapsed directional decisions yet.",
    ]
    if report["hold_n"]:
        lines.append(
            f"- Hold entries: n={report['hold_n']}, mean forward return "
            f"{report['hold_mean_return']:+.2%} (opportunity-cost view, "
            "not scored for direction)"
        )

    lines += ["", "## By rating", "", "| Rating | n | mean fwd ret | directional n | hit rate |",
              "|---|---:|---:|---:|---:|"]
    for rating, bucket in report["by_rating"].items():
        mean = (
            f"{bucket['mean_return']:+.2%}"
            if bucket["mean_return"] is not None else "—"
        )
        rate = (
            f"{bucket['hit_rate']:.1%}" if bucket["hit_rate"] is not None else "—"
        )
        lines.append(
            f"| {rating} | {bucket['n']} | {mean} | "
            f"{bucket['directional_n']} | {rate} |"
        )

    lines += ["", "## By ticker", "", "| Ticker | n | directional n | hit rate | mean fwd ret |",
              "|---|---:|---:|---:|---:|"]
    for ticker, bucket in report["by_ticker"].items():
        mean = (
            f"{bucket['mean_return']:+.2%}"
            if bucket["mean_return"] is not None else "—"
        )
        rate = (
            f"{bucket['hit_rate']:.1%}" if bucket["hit_rate"] is not None else "—"
        )
        lines.append(
            f"| {ticker} | {bucket['n']} | {bucket['directional_n']} | "
            f"{rate} | {mean} |"
        )

    inferred = [
        r for r in report["records"] if r.get("asset_source") == "inferred"
    ]
    if inferred:
        lines += [
            "", "> ⚠ Legacy-log note: records without a logged asset_type were "
            "inferred from the ticker (USDT suffix → Binance perp). "
            f"{len(inferred)} record(s) affected.",
        ]
    return "\n".join(lines) + "\n"


def verify_history(
    results_dir: str | Path | None = None, holding_days: int = 5
) -> tuple[dict[str, Any], Path, Path]:
    """Scan → score → write ``accuracy_report.{json,md}``; returns the report."""
    records = scan_history(results_dir)
    report = build_accuracy_report(records, holding_days=holding_days)

    root = Path(results_dir) if results_dir else _default_results_dir()
    out_dir = root / "accuracy"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "accuracy_report.json"
    md_path = out_dir / "accuracy_report.md"
    json_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    md_path.write_text(render_markdown(report), encoding="utf-8")
    return report, json_path, md_path
