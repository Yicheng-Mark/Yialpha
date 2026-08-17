"""IC dataset export + cycle orchestration (package-side).

Extracted from ``scripts/export_ic_dataset.py`` (2026-08-16, T1 batch) so the
new ``yiagents ic-cycle`` command can drive the export programmatically — the
script is not part of the wheel (``pyproject`` ships ``yiagents*`` only), so
importing it from the CLI would break installed deployments. The script now
delegates here; its CLI contract is unchanged.

Fail-closed contract (never silently wrong data) — unchanged from the script:

- Unknown indicator names are rejected upfront (validated against the market
  analyst's ``INDICATOR_NAMES``).
- The last ``horizon`` rows have no realizable forward return and are DROPPED.
- An indicator that fails to compute is skipped with a WARNING — never
  zero-filled.
- Data comes from ``load_ohlcv`` (per-symbol cache, PIT-filtered to as-of).

The cycle half (``run_ic_cycle``) additionally computes the prune verdict per
ticker using the same ``yiagents.backtest.ic`` math ``prune_indicators_cli``
runs, writes ``<csv>.prune.json``, and returns the verdicts for display.
It NEVER edits the live config — the human review step is the contract.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from yiagents.backtest.ic import (
    consecutive_below_threshold,
    prune_indicators,
    rolling_ic,
)

logger = logging.getLogger(__name__)


def build_ic_frame(
    ticker: str,
    horizon: int,
    indicators: list[str],
    as_of: str,
    extra_horizons: Sequence[int] = (),
) -> tuple[pd.DataFrame, list[str]]:
    """Compute the IC table for one ticker.

    Returns ``(frame, skipped_indicators)``. The frame has ``date``,
    ``forward_return`` (Close.shift(-horizon)/Close - 1) and one column per
    computable indicator; tail rows without a realizable forward return are
    dropped.
    """
    from yiagents.dataflows.feature_registry import compute_derived
    from yiagents.dataflows.stockstats_utils import (
        compute_indicator,
        load_ohlcv,
    )

    data = load_ohlcv(ticker, as_of)

    from stockstats import wrap

    wrapped = wrap(data.copy())
    closes = pd.to_numeric(data["Close"], errors="coerce")

    out = pd.DataFrame(
        {"date": pd.to_datetime(data["Date"]).dt.strftime("%Y-%m-%d")}
    )
    # Forward return over the NEXT `horizon` trading rows. The final
    # `horizon` rows get NaN and are dropped below — their "future" has not
    # happened yet, and fabricating it would poison the IC tail windows.
    out["forward_return"] = closes.shift(-horizon) / closes - 1.0

    for h in sorted({int(h) for h in extra_horizons if int(h) != horizon}):
        if h <= 0:
            raise ValueError(f"extra horizons must be positive; got {h}")
        out[f"fwd_ret_{h}d"] = closes.shift(-h) / closes - 1.0

    skipped: list[str] = []
    for ind in indicators:
        # Derived features (vol estimators, OBV, ...) compute on the raw
        # frame; stockstats names on the wrapped one. Same skip-and-report
        # contract either way — never zero-fill.
        try:
            derived = compute_derived(data, ind)
            if derived is not None:
                out[ind] = pd.to_numeric(derived, errors="coerce")
                continue
            # compute_indicator (not a bare wrapped[ind]) so the vendor-scale
            # fixes (e.g. mfi 0-1 -> 0-100) apply here too.
            series = compute_indicator(wrapped, ind)
        except Exception as exc:  # noqa: BLE001 -- skip-and-report, never zero-fill
            logger.warning("indicator %s failed to compute for %s: %s", ind, ticker, exc)
            skipped.append(ind)
            continue
        out[ind] = pd.to_numeric(series, errors="coerce")

    out = out.dropna(subset=["forward_return"]).reset_index(drop=True)
    return out, skipped


def export_ic_datasets(
    tickers: Sequence[str],
    horizon: int = 5,
    indicators: Sequence[str] | None = None,
    as_of: str | None = None,
    output_dir: str | Path = "ic_data",
    extra_horizons: Sequence[int] = (),
) -> dict[str, Path]:
    """Export one ``<TICKER>_<horizon>d.csv`` per ticker; returns the paths.

    Tickers whose export fails (no data, all indicators skipped) are logged
    and omitted from the result — the caller decides whether an empty result
    is fatal (ic-cycle treats it as a typed failure).
    """
    from yiagents.agents.analysts.market_analyst import INDICATOR_NAMES

    names = sorted(indicators) if indicators else sorted(INDICATOR_NAMES)
    unknown = [n for n in names if n not in INDICATOR_NAMES]
    if unknown:
        raise ValueError(
            f"unknown indicator name(s) {unknown}; valid names are pinned by "
            f"INDICATOR_NAMES: {sorted(INDICATOR_NAMES)}"
        )
    if horizon <= 0:
        raise ValueError("horizon must be a positive number of trading days")

    cutoff = as_of or pd.Timestamp.today().strftime("%Y-%m-%d")
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    written: dict[str, Path] = {}
    for ticker in tickers:
        try:
            frame, skipped = build_ic_frame(
                ticker, horizon, names, cutoff, extra_horizons=extra_horizons,
            )
        except Exception as exc:  # noqa: BLE001 -- one bad ticker must not kill the batch
            logger.error("export failed for %s: %s", ticker, exc)
            continue
        if frame.empty:
            logger.error("no usable rows for %s (empty frame after tail drop)", ticker)
            continue
        n_extra = sum(1 for c in frame.columns if c.startswith("fwd_ret_"))
        n_indicator_cols = len(frame.columns) - 2 - n_extra
        if n_indicator_cols == 0:
            logger.error(
                "all indicators skipped for %s (%s); not writing an "
                "indicator-less CSV", ticker, ", ".join(skipped) or "?",
            )
            continue
        out_path = out_dir / f"{ticker.replace('.', '_')}_{horizon}d.csv"
        frame.to_csv(out_path, index=False)
        written[ticker] = out_path
    return written


def prune_verdict_for_csv(
    csv_path: str | Path,
    *,
    window: int = 60,
    min_abs_ic: float = 0.03,
    min_consecutive: int = 30,
    min_observations: int = 30,
) -> dict[str, Any]:
    """Compute the keep/prune verdict for one exported CSV.

    Same math as ``scripts/prune_indicators_cli.py`` (rolling IC →
    ``prune_indicators`` → per-indicator stats), producing the identical
    ``--json-out`` shape so downstream consumers (``indicator_ic_context``,
    ic-cycle reports, tests) read one format.
    """
    df = pd.read_csv(csv_path)
    if "forward_return" not in df.columns or "date" not in df.columns:
        raise ValueError(f"{csv_path}: missing date/forward_return columns")
    reserved = {"date", "forward_return"}
    indicators = {
        c: df[c] for c in df.columns
        if c not in reserved and not c.startswith("fwd_ret_")
    }
    if not indicators:
        raise ValueError(f"{csv_path}: no indicator columns")

    ic_by_indicator = {
        name: rolling_ic(values, df["forward_return"], window=window)
        for name, values in indicators.items()
    }
    result = prune_indicators(
        ic_by_indicator,
        min_abs_ic=min_abs_ic,
        min_consecutive_days=min_consecutive,
        min_observation_windows=min_observations,
    )

    per_indicator: dict[str, dict[str, Any]] = {}
    for name, series in ic_by_indicator.items():
        finite = series.dropna()
        per_indicator[name] = {
            "verdict": "prune" if name in set(result["prune"]) else "keep",
            "mean_abs_ic": (
                float(finite.abs().mean()) if not finite.empty else None
            ),
            "finite_windows": int(len(finite)),
            "longest_low_run": int(
                consecutive_below_threshold(
                    series, threshold=min_abs_ic, min_consecutive=min_consecutive,
                )
            ),
        }
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "params": {
            "window": window,
            "min_abs_ic": min_abs_ic,
            "min_consecutive": min_consecutive,
            "min_observations": min_observations,
        },
        "keep": list(result["keep"]),
        "prune": list(result["prune"]),
        "per_indicator": per_indicator,
    }


def run_ic_cycle(
    tickers: Sequence[str],
    horizon: int = 5,
    window: int = 60,
    min_abs_ic: float = 0.03,
    min_consecutive: int = 30,
    min_observations: int = 30,
    output_dir: str | Path = "ic_data",
    as_of: str | None = None,
    indicators: Sequence[str] | None = None,
) -> dict[str, Any]:
    """export → prune verdict per ticker → ``<csv>.prune.json`` on disk.

    Returns ``{"csv": {ticker: path}, "verdict": {ticker: verdict_dict}}``.
    Applying any prune to the live ``indicator_battery`` stays a HUMAN step
    (fail-closed); this function never edits config.
    """
    csvs = export_ic_datasets(
        tickers, horizon=horizon, as_of=as_of, output_dir=output_dir,
        indicators=indicators,
    )
    if not csvs:
        raise RuntimeError(
            "ic-cycle: no dataset exported (all tickers failed — check the "
            "OHLCV cache / as-of date)"
        )
    verdicts: dict[str, Any] = {}
    for ticker, csv_path in csvs.items():
        verdict = prune_verdict_for_csv(
            csv_path,
            window=window,
            min_abs_ic=min_abs_ic,
            min_consecutive=min_consecutive,
            min_observations=min_observations,
        )
        json_path = Path(f"{csv_path}.prune.json")
        json_path.write_text(
            json.dumps(verdict, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        verdicts[ticker] = verdict
    return {"csv": csvs, "verdict": verdicts}
