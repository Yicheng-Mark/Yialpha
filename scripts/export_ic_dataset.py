#!/usr/bin/env python
"""Build the IC dataset that scripts/prune_indicators_cli.py consumes.

This closes the most upstream gap in the self-improvement loop: the IC
pruning rule existed, but the ``date, forward_return, <indicator>...`` table
it needs had to be assembled by hand. This exporter derives it straight from
the OHLCV cache + stockstats, so the evidence chain is now fully mechanical:

    python scripts/export_ic_dataset.py NVDA AMD --horizon 5 \
        && python scripts/prune_indicators_cli.py ic_data/NVDA_5d.csv --json-out ic_data/NVDA_5d.prune.json

Fail-closed contract (never silently wrong data) — enforced by the package
implementation this delegates to (:mod:`yialpha.backtest.ic_dataset`,
extracted 2026-08-16 so ``yialpha ic-cycle`` can share it):

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

from yialpha.agents.analysts.market_analyst import INDICATOR_NAMES  # noqa: E402
from yialpha.backtest.ic_dataset import export_ic_datasets  # noqa: E402


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
        "--extra-horizons", type=str, default="",
        help="Comma-separated additional forward horizons emitted as "
        "fwd_ret_<h>d columns for IC-decay analysis, e.g. '1,10,20' "
        "(default: none). The primary --horizon column stays 'forward_return'.",
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

    try:
        extra = [int(x) for x in args.extra_horizons.split(",") if x.strip()]
    except ValueError:
        parser.error(f"--extra-horizons must be comma-separated integers; got {args.extra_horizons!r}")
    bad = [h for h in extra if h <= 0]
    if bad:
        parser.error(f"--extra-horizons must be positive; got {bad}")

    indicators = args.indicators or sorted(INDICATOR_NAMES)
    unknown = [n for n in indicators if n not in INDICATOR_NAMES]
    if unknown:
        parser.error(
            f"unknown indicator name(s) {unknown}; valid names are pinned by "
            f"INDICATOR_NAMES: {sorted(INDICATOR_NAMES)}"
        )

    # Package-side implementation (shared with `yialpha ic-cycle`); this
    # wrapper keeps the script's argv contract and per-ticker "next:" hints.
    written = export_ic_datasets(
        args.tickers,
        horizon=args.horizon,
        indicators=args.indicators,
        as_of=args.as_of,
        output_dir=args.output_dir,
        extra_horizons=extra,
    )

    for ticker, out_path in written.items():
        with open(out_path, encoding="utf-8") as fh:
            frame_rows = sum(1 for _ in fh) - 1
        print(
            f"[{ticker}] wrote {out_path}: {frame_rows} rows "
            "(see WARNINGs above for any skipped indicators)"
        )
        print(
            f"[{ticker}] next: python scripts/prune_indicators_cli.py {out_path} "
            f"--window 60 --json-out {out_path}.prune.json"
        )

    return 0 if len(written) == len(args.tickers) else 1


if __name__ == "__main__":
    raise SystemExit(main())
