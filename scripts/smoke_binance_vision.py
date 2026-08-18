"""Real-network smoke for the perp accuracy expansion (2026-08-17).

Loads .env (SOCKS5 proxy — searched upward from this file), then exercises
the three new real paths:

  1. data.binance.vision daily metrics archives (sha256-verified + cached,
     one request per day — no monthly aggregates exist for these datasets)
  2. data.binance.vision bookDepth archive (one recent closed day)
  3. /fapi/v1/depth snapshot + an intraday-period /futures/data call

Run:  python scripts/smoke_binance_vision.py
"""

from __future__ import annotations

import io
import os
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

# Load proxy env from .env exactly like the run entrypoints do (search upward
# so this works from scripts/ as well as the repo root).
for env_path in (
    Path(__file__).resolve().parent / ".env",
    Path(__file__).resolve().parent.parent / ".env",
):
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k, v = k.strip(), v.strip().strip('"').strip("'")
            if k in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY") and v:
                os.environ.setdefault(k, v)
        break

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from yiagents.dataflows.binance import (  # noqa: E402
    get_binance_depth_snapshot,
    get_binance_open_interest,
)
from yiagents.dataflows.binance_vision import (  # noqa: E402
    get_binance_vision_book_depth,
    get_binance_vision_metrics,
)


def _csv_rows(out: str) -> pd.DataFrame:
    return pd.read_csv(io.StringIO(out.split("\n\n", 1)[1]))


print("== 1. vision metrics: BTCUSDT 2024-01 (31 daily archives, 1d resample) ==")
m = get_binance_vision_metrics("BTCUSDT", "2024-01-01", "2024-01-31")
df = _csv_rows(m)
print(df.head(3).to_string(index=False))
print(f"... rows={len(df)}  cols={list(df.columns)}")
assert len(df) == 31, f"expected 31 daily rows, got {len(df)}"
assert df["open_interest"].notna().all()

print("\n== 2. vision metrics again (cache hit — no re-download) ==")
m2 = get_binance_vision_metrics("BTCUSDT", "2024-01-01", "2024-01-31")
assert len(_csv_rows(m2)) == 31

print("\n== 3. vision metrics 4h resample spot check ==")
m4 = get_binance_vision_metrics("BTCUSDT", "2024-01-10", "2024-01-12", interval="4h")
df4 = _csv_rows(m4)
print(df4.head(3).to_string(index=False))
print(f"... rows={len(df4)} (3 days x 6 buckets = 18 expected)")
assert len(df4) == 18

print("\n== 4. vision bookDepth: BTCUSDT one recent closed day ==")
day = (datetime.now(UTC) - timedelta(days=3)).strftime("%Y-%m-%d")
d = get_binance_vision_book_depth("BTCUSDT", day, day)
dfd = _csv_rows(d)
print(dfd.to_string(index=False))
print(f"... rows={len(dfd)}  bands={sorted(dfd['percentage'].unique())}")
assert set(dfd["percentage"]) <= {
    -5.0, -4.0, -3.0, -2.0, -1.0, -0.2, 0.2, 1.0, 2.0, 3.0, 4.0, 5.0,
}

print("\n== 5. REST: open interest period=1h (intraday passthrough) ==")
oi = get_binance_open_interest("BTCUSDT", 2, period="1h")
doi = _csv_rows(oi)
print(doi.tail(3).to_string(index=False))
print(f"... rows={len(doi)}; last window times carry HH:MM")
assert doi["time"].str.contains(":").any()

print("\n== 6. REST: live depth snapshot ==")
ds = get_binance_depth_snapshot("BTCUSDT", 20)
dds = _csv_rows(ds)
print("\n".join(ds.splitlines()[:5]))
print(f"... ladder rows={len(dds)}")
assert len(dds) == 20

print("\nSMOKE OK")
