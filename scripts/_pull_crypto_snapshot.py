"""One-shot real-data snapshot for ETH/SOL/XRP/HYPE USDT perps.

Pulls klines + funding + OI + long/short + taker buy/sell + basis via the
project's verified Binance dataflows (SOCKS5 proxy from .env). Prints a compact
decision-grade summary per coin. Analysis-only, read-only, no key.
"""
from __future__ import annotations

import io
import sys
import traceback
from datetime import datetime, timedelta, timezone

import pandas as pd

_TODAY = datetime.now(timezone.utc).strftime("%Y-%m-%d")
_START = (datetime.now(timezone.utc) - timedelta(days=130)).strftime("%Y-%m-%d")

from yiagents.dataflows.binance import (
    get_binance_klines,
    get_binance_funding_rate,
    get_binance_open_interest,
    get_binance_long_short_ratio,
    get_binance_taker_buy_sell,
    get_binance_basis,
)

SYMBOLS = ["ETHUSDT", "SOLUSDT", "XRPUSDT", "HYPEUSDT"]


def _csv(fn, *a, **k):
    try:
        s = fn(*a, **k)
        if not s or not s.strip():
            return None
        # Strip the vendor header block (lines starting with '#') + blank lines
        # so only the CSV body remains.
        body = "\n".join(
            ln for ln in s.splitlines()
            if ln.strip() and not ln.lstrip().startswith("#")
        )
        return pd.read_csv(io.StringIO(body))
    except Exception as e:
        return f"ERR: {type(e).__name__}: {e}"


def summarize(sym: str) -> None:
    print("=" * 78)
    print(f"# {sym}")
    print("=" * 78)

    kl = _csv(get_binance_klines, sym, _START, _TODAY)
    if isinstance(kl, str):
        print(f"  KLINES FAIL -> {kl}")
        print("  (symbol likely not listed on Binance USDT-M perp; skipping)\n")
        return
    if kl is None or len(kl) == 0:
        print("  KLINES EMPTY\n")
        return

    # normalize column names (klines CSV has lowercase header from df.to_csv)
    kl = kl.rename(columns={c: c.lower() for c in kl.columns})
    close_col = "close" if "close" in kl.columns else kl.columns[4]
    kl[close_col] = pd.to_numeric(kl[close_col], errors="coerce")
    kl = kl.dropna(subset=[close_col]).reset_index(drop=True)
    px = kl[close_col]
    last = float(px.iloc[-1])

    def ret(days):
        if len(px) <= days:
            return float("nan")
        return (last / float(px.iloc[-1 - days]) - 1.0) * 100.0

    # daily log returns -> annualized vol (30d window)
    import numpy as np
    logret = np.log(px / px.shift(1)).dropna().tail(30)
    vol30_ann = float(logret.std(ddof=0) * (365 ** 0.5) * 100) if len(logret) > 1 else float("nan")
    # 30d realized range / drawdown
    tail30 = px.tail(30)
    hi30, lo30 = float(tail30.max()), float(tail30.min())
    dd30 = (last / hi30 - 1) * 100 if hi30 > 0 else float("nan")

    print(f"  last price       : {last:,.6g}")
    print(f"  24h / 7d / 30d   : {ret(1):+.2f}%  {ret(7):+.2f}%  {ret(30):+.2f}%")
    print(f"  60d / 90d / 120d : {ret(60):+.2f}%  {ret(90):+.2f}%  {ret(120):+.2f}%")
    print(f"  30d vol (ann.)   : {vol30_ann:.1f}%   |  30d range: {lo30:,.6g} - {hi30:,.6g}")
    print(f"  30d drawdown now : {dd30:+.1f}% from 30d high")

    def _col(df, *keys, contains=None):
        c = [x.lower() for x in df.columns]
        for k in keys:
            if k in c:
                return df.columns[c.index(k)]
        if contains:
            for i, x in enumerate(c):
                if contains in x:
                    return df.columns[i]
        return None

    # funding
    fr = _csv(get_binance_funding_rate, sym, _START, _TODAY)
    if isinstance(fr, pd.DataFrame) and len(fr):
        fr = fr.rename(columns={c: c.lower() for c in fr.columns})
        rc = _col(fr, "fundingrate", "funding_rate", contains="rate")
        if rc and rc != _col(fr, contains="time"):
            fr[rc] = pd.to_numeric(fr[rc], errors="coerce")
            series = fr[rc].dropna()
            if len(series):
                fr7 = series.tail(21)
                fr30 = series.tail(90)
                print(f"  funding last     : {float(series.iloc[-1])*100:+.4f}%")
                print(f"  funding avg 7d   : {fr7.mean()*100*3:+.3f}%/day  | 30d: {fr30.mean()*100*3:+.3f}%/day")
                print(f"  ~cost to hold L 30d: {fr30.sum()*100:+.2f}% (long pays+ / short pays-)")
    else:
        print(f"  funding: {fr}")

    # open interest
    oi = _csv(get_binance_open_interest, sym, 14)
    if isinstance(oi, pd.DataFrame) and len(oi):
        oi = oi.rename(columns={c: c.lower() for c in oi.columns})
        oc = _col(oi, "sum_open_interest_value", "openinterestvalue", contains="value")
        if oc:
            oi[oc] = pd.to_numeric(oi[oc], errors="coerce")
            s = oi[oc].dropna()
            if len(s):
                cur_oi, prev_oi = float(s.iloc[-1]), float(s.iloc[0])
                chg = (cur_oi / prev_oi - 1) * 100 if prev_oi else float("nan")
                print(f"  open interest    : ${cur_oi/1e6:,.1f}M  (14d {chg:+.1f}%)")
    else:
        print(f"  OI: {oi}")

    # long/short ratio
    lsr = _csv(get_binance_long_short_ratio, sym, 7)
    if isinstance(lsr, pd.DataFrame) and len(lsr):
        lsr = lsr.rename(columns={c: c.lower() for c in lsr.columns})
        lc = _col(lsr, "longshortratio", "long_short_ratio", contains="ratio")
        if lc:
            lsr[lc] = pd.to_numeric(lsr[lc], errors="coerce")
            s = lsr[lc].dropna()
            if len(s):
                print(f"  L/S ratio now    : {float(s.iloc[-1]):.3f}  (7d avg {s.mean():.3f})")
    else:
        print(f"  L/S: {lsr}")

    # taker buy/sell
    tbs = _csv(get_binance_taker_buy_sell, sym, 7)
    if isinstance(tbs, pd.DataFrame) and len(tbs):
        tbs = tbs.rename(columns={c: c.lower() for c in tbs.columns})
        bv = _col(tbs, "buyvol", "buy_vol", "buyvolume", contains="buy")
        sv = _col(tbs, "sellvol", "sell_vol", "sellvolume", contains="sell")
        if bv and sv and bv != sv:
            tbs[bv] = pd.to_numeric(tbs[bv], errors="coerce")
            tbs[sv] = pd.to_numeric(tbs[sv], errors="coerce")
            rt = tbs[[bv, sv]].dropna()
            if len(rt):
                ratio = rt[bv].sum() / rt[sv].sum()
                print(f"  taker buy/sell   : {ratio:.3f} (>1 = net taker buying over 7d)")
    else:
        print(f"  taker: {tbs}")

    # basis (perp vs quarterly)
    bs = _csv(get_binance_basis, sym, 7)
    if isinstance(bs, pd.DataFrame) and len(bs):
        bs = bs.rename(columns={c: c.lower() for c in bs.columns})
        bc = _col(bs, "basisrate", "basis_rate", contains="rate")
        if bc:
            bs[bc] = pd.to_numeric(bs[bc], errors="coerce")
            s = bs[bc].dropna()
            if len(s):
                print(f"  basis rate last  : {float(s.iloc[-1]):+.3f}%")
    else:
        print(f"  basis: {bs}")

    print()


def main():
    for s in SYMBOLS:
        try:
            summarize(s)
        except Exception:
            print(f"{s}: top-level error")
            traceback.print_exc()
            print()


if __name__ == "__main__":
    main()
