#!/usr/bin/env python
"""Build the IC dataset that scripts/prune_indicators_cli.py consumes.

This closes the most upstream gap in the self-improvement loop: the IC
pruning rule existed, but the ``date, forward_return, <indicator>...`` table
it needs had to be assembled by hand. This exporter derives it straight from
the OHLCV cache + stockstats, so the evidence chain is now fully mechanical:

    python scripts/export_ic_dataset.py NVDA AMD --horizon 5 \
        && python scripts/prune_indicators_cli.py ic_data/NVDA_5d.csv --json-out ic_data/NVDA_5d.prune.json

Fail-closed contract (never silently wrong data):
- Unknown indicator names are rejected upfront (validated against the market
  analyst's INDICATOR_NAMES — the same battery the LLM selects from).
- The last ``horizon`` rows have no realizable forward return and are DROPPED
  (a fake forward_return would corrupt every IC window touching the tail).
- An indicator that fails to compute is skipped with a WARNING and listed in
  the run summary — never filled with zeros.
- Data comes from load_ohlcv (per-symbol cache, PIT-filtered to --as-of).

Output: one CSV per ticker, ``ic_data/<TICKER>_<horizon>d.csv`` (override
with --output-dir), columns ``date, forward_return, <indicator>...``.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("export_ic_dataset")

# Allow running as `python scripts/export_ic_dataset.py` without an editable
# install.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from yiagents.agents.analysts.market_analyst import INDICATOR_NAMES  # noqa: E402
from yiagents.dataflows.stockstats_utils import load_ohlcv  # noqa: E402


def build_ic_frame(
    ticker: str,
    horizon: int,
    indicators: list[str],
    as_of: str,
) -> tuple[pd.DataFrame, list[str]]:
    """Compute the IC table for one ticker.

    Returns ``(frame, skipped_indicators)``. The frame has ``date``,
    ``forward_return`` (Close.shift(-horizon)/Close - 1) and one column per
    computable indicator; tail rows without a realizable forward return are
    dropped.
    """
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

    skipped: list[str] = []
    for ind in indicators:
        try:
            series = wrapped[ind]
        except Exception as exc:  # noqa: BLE001 -- skip-and-report, never zero-fill
            logger.warning("indicator %s failed to compute for %s: %s", ind, ticker, exc)
            skipped.append(ind)
            continue
        out[ind] = pd.to_numeric(series, errors="coerce")

    out = out.dropna(subset=["forward_return"]).reset_index(drop=True)
    return out, skipped


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Export the IC dataset (date, forward_return, indicators) "
        "for the self-improvement pruning loop.",
    )
    parser.add_argument("tickers", nargs="+", help="Ticker symbols (e.g. NVDA AMD)")
    parser.add_argument(
        "--horizon", type=int, default=5,
        help="Forward-return horizon in trading rows (default: 5)",
    )
    parser.add_argument(
        "--indicators", nargs="*", default=None,
        help="Indicator names (default: the market analyst's full battery). "
        "Valid names are pinned by INDICATOR_NAMES in market_analyst.py.",
    )
    parser.add_argument(
        "--as-of", default=pd.Timestamp.today().strftime("%Y-%m-%d"),
        help="PIT cutoff date YYYY-MM-DD (default: today). Rows after this "
        "date are excluded by load_ohlcv's look-ahead guard.",
    )
    parser.add_argument(
        "--output-dir", default="ic_data",
        help="Directory for the per-ticker CSVs (default: ./ic_data)",
    )
    args = parser.parse_args(argv)

    if args.horizon <= 0:
        parser.error("--horizon must be a positive number of trading days")

    indicators = args.indicators or sorted(INDICATOR_NAMES)
    unknown = [n for n in indicators if n not in INDICATOR_NAMES]
    if unknown:
        parser.error(
            f"unknown indicator name(s) {unknown}; valid names are pinned by "
            f"INDICATOR_NAMES: {sorted(INDICATOR_NAMES)}"
        )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    failures = 0
    for ticker in args.tickers:
        try:
            frame, skipped = build_ic_frame(ticker, args.horizon, indicators, args.as_of)
        except Exception as exc:  # noqa: BLE001 -- one bad ticker must not kill the batch
            logger.error("export failed for %s: %s", ticker, exc)
            failures += 1
            continue

        if frame.empty:
            logger.error("no usable rows for %s (empty frame after tail drop)", ticker)
            failures += 1
            continue

        if len(frame.columns) == 2:
            # date + forward_return only: every indicator was skipped. The
            # prune CLI rejects this CSV ("at least one indicator column"), so
            # writing it with a success verdict and a "next:" hint would send
            # the operator into a guaranteed failure.
            logger.error(
                "all indicators skipped for %s (%s); not writing an "
                "indicator-less CSV", ticker, ", ".join(skipped) or "?",
            )
            failures += 1
            continue

        out_path = out_dir / f"{ticker.replace('.', '_')}_{args.horizon}d.csv"
        frame.to_csv(out_path, index=False)
        print(
            f"[{ticker}] wrote {out_path}: {len(frame)} rows, "
            f"{len(frame.columns) - 2} indicator column(s)"
            + (f" (skipped: {', '.join(skipped)})" if skipped else "")
        )
        print(
            f"[{ticker}] next: python scripts/prune_indicators_cli.py {out_path} "
            f"--window 60 --json-out {out_path}.prune.json"
        )

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
