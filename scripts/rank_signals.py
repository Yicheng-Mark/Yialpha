#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
rank_signals.py — YiAgents 多标的信心排名。

扫描一批 ticker 的最新分析报告，按 做多 / 做空 / 观望 分组，各自按「综合信心分」排名，
一眼看到最强多头 / 最强空头。只读报告产物，零触碰 agent/图/数据流（守「增强层零影响」铁律）。

信心分（0–100，代理分，非校准概率）= 评级 55% + 情绪 25% + 盈亏比 20%（缺失分量重归一化）。
评级用 trade_ticket.parse_pm_decision 解析（中英文都认）——修复框架 parse_rating 不认中文评级
导致 BTCUSDT「卖出」被错读成 Hold 的坑。

用法：
    python scripts/rank_signals.py --tickers BTCUSDT 0700.HK AAPL 600519.SS
    python scripts/rank_signals.py --tickers BTCUSDT ETHUSDT SOLUSDT --top 5 --capital 10000
    python scripts/rank_signals.py                       # 不传则全部 ticker 各取最新
    python scripts/rank_signals.py --tickers ... --json
    python scripts/rank_signals.py --tickers ... --since 2026-07-01
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

# Windows 控制台是 GBK(cp936)，打印中文/符号会 UnicodeEncodeError —— 强制 UTF-8
try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except Exception:
    pass

# 复用 trade_ticket.py 的全部解析（scripts/ 不是包，sys.path 注入）
sys.path.insert(0, str(Path(__file__).resolve().parent))
from trade_ticket import (  # noqa: E402
    _results_dir, parse_pm_decision, parse_trader, parse_market,
    decide_direction, detect_asset_type, resolve_levels,
)

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

#: 方向信心（长短镜像：Sell 对空头 = Buy 对多头）
LONG_CONV = {"Buy": 0.90, "Overweight": 0.72}
SHORT_CONV = {"Sell": 0.90, "Underweight": 0.72}

#: 综合信心分权重（rating 恒在，缺分量重归一化）
WEIGHTS = {"rating": 0.55, "sent": 0.25, "rr": 0.20}

#: 盈亏比归一化上限（rr/RR_CAP clip 到 1.0）
RR_CAP = 5.0

#: 回撤熔断体制（框架 Kelly×Breaker×CVaR 已拒部署）
BLOCKED_REGIMES = {"no_new", "hard_stop"}

#: 资产类型中文名
ASSET_CN = {
    "crypto_perp": "加密永续", "crypto_spot": "加密现货",
    "us_stock": "美股", "hk_stock": "港股", "cn_stock": "A股",
}

_DIR_STAMP_RE = re.compile(r"_(\d{8})_(\d{6})$")


# ---------------------------------------------------------------------------
# 枚举 / 解析
# ---------------------------------------------------------------------------

def latest_for_ticker(ticker: str) -> Path | None:
    """该 ticker 现存报告中 complete_report.md mtime 最新的目录（无则 None）。"""
    root = _results_dir()
    if not root.is_dir():
        return None
    best, best_mt = None, 0.0
    for d in root.glob(f"{ticker}_*"):  # 点号 ticker（600519.SS/0700.HK）原样匹配
        cr = d / "complete_report.md"
        if not cr.is_file():
            continue  # 跳半截 / 中断 run
        try:
            mt = cr.stat().st_mtime
        except OSError:
            continue
        if mt > best_mt:
            best, best_mt = d, mt
    return best


def report_date(report_dir: Path) -> str | None:
    """从目录名 <TICKER>_<YYYYMMDD>_<HHMMSS> 取日期，返回 'MM-DD'。"""
    m = _DIR_STAMP_RE.search(report_dir.name)
    if not m:
        return None
    ymd = m.group(1)
    return f"{ymd[4:6]}-{ymd[6:8]}" if len(ymd) == 8 else ymd


def parse_sentiment_score(md: str) -> float | None:
    """1_analysts/sentiment.md 的 Overall Sentiment Score (0–10)。0=最空,10=最牛。"""
    if not md:
        return None
    m = re.search(r"\*\*Overall Sentiment:\*\*\s*\*\*[^*]+\*\*\s*\(Score:\s*([\d.]+)\s*/\s*10\)",
                  md, re.I)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            return None
    # 中文回退：**整体情绪**：xxx（评分 N/10）
    m = re.search(r"整体情绪[^\d(]{0,12}（?评分[:：]?\s*([\d.]+)\s*/\s*10", md)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            return None
    return None


def resolve_target(parsed_pm, entry, direction) -> float | None:
    """
    止盈目标（仅用 PM price_target 算盈亏比）。结构位（支撑/阻力）正则噪声大
    （会出现 59× 止损距离这类垃圾值），故弃用；无 PM 目标价 → 返回 None，信心分重归一化。
    """
    pt = parsed_pm.get("price_target")
    if pt and entry:
        # 目标价必须在方向上有利（多头 pt>entry，空头 pt<entry），否则不可用
        if (direction == "long" and pt > entry) or (direction == "short" and pt < entry):
            return pt
    return None


# ---------------------------------------------------------------------------
# 信心分
# ---------------------------------------------------------------------------

def conviction(direction, rating, sentiment_score, rr):
    """
    返回 (score_0_100, {c_rating, c_sent, c_rr})。rating 恒在 → 不会除零。
    """
    conv_map = LONG_CONV if direction == "long" else SHORT_CONV
    c_rating = conv_map.get(rating)
    if c_rating is None:
        return None, {}

    c_sent = None
    if sentiment_score is not None:
        c_sent = (sentiment_score / 10.0) if direction == "long" else ((10.0 - sentiment_score) / 10.0)
        c_sent = max(0.0, min(1.0, c_sent))

    c_rr = None
    if rr and rr > 0:
        c_rr = min(rr / RR_CAP, 1.0)

    comps = {"rating": c_rating, "sent": c_sent, "rr": c_rr}
    present = {k: v for k, v in comps.items() if v is not None}
    tot = sum(WEIGHTS[k] for k in present)
    score = round(100.0 * sum(WEIGHTS[k] * present[k] for k in present) / tot) if tot else 0
    return score, comps


def analyze_ticker(ticker: str) -> dict | None:
    """读该 ticker 最新报告，解析 + 算信心分。无报告返回 None。"""
    rd = latest_for_ticker(ticker)
    if rd is None:
        return {"ticker": ticker, "status": "no_report"}

    def read(rel):
        p = rd / rel
        return p.read_text(encoding="utf-8", errors="ignore") if p.exists() else ""

    parsed_pm = parse_pm_decision(read("5_portfolio/decision.md"))
    parsed_trader = parse_trader(read("3_trading/trader.md"))
    parsed_market = parse_market(read("1_analysts/market.md"))
    sent_score = parse_sentiment_score(read("1_analysts/sentiment.md"))

    rating = parsed_pm.get("rating")
    action = parsed_trader.get("action")
    direction, _ = decide_direction(rating, action)
    entry, stop, _atr = resolve_levels(parsed_trader, parsed_pm, parsed_market, direction)
    target = resolve_target(parsed_pm, entry, direction)

    rr = None
    if entry and stop and target and abs(entry - stop) > 0:
        rr = abs(target - entry) / abs(entry - stop)

    score, comps = (conviction(direction, rating, sent_score, rr)
                    if direction in ("long", "short") else (None, {}))

    regime = parsed_pm.get("ovl_drawdown_regime")
    return {
        "ticker": ticker,
        "status": "ok",
        "report_dir": str(rd),
        "report_date": report_date(rd),
        "asset_type": detect_asset_type(ticker),
        "rating": rating,
        "trader_action": action,
        "direction": direction,
        "conviction": score,
        "comps": comps,
        "sentiment_score": sent_score,
        "rr": round(rr, 2) if rr is not None else None,
        "entry": entry,
        "stop": stop,
        "target": target,
        "price_target_pm": parsed_pm.get("price_target"),
        "drawdown_regime": regime,
        "blocked": regime in BLOCKED_REGIMES,
    }


# ---------------------------------------------------------------------------
# 渲染
# ---------------------------------------------------------------------------

def _m(x):
    if x is None:
        return "—"
    if abs(x) >= 1000:
        return f"{x:,.0f}"
    return f"{x:,.2f}".rstrip("0").rstrip(".")


def _f2(x):
    return "—" if x is None else f"{x:.2f}"


def _asset_cn(t):
    return ASSET_CN.get(t, t)


DIR_BADGE = {"long": "🟢 做多", "short": "🔴 做空", "hold": "⚪ 观望"}


def render_rank(rows: list[dict], tickers: list[str], top: int, capital=None) -> str:
    ok = [r for r in rows if r.get("status") == "ok"]
    missing = [r for r in rows if r.get("status") != "ok"]

    longs = sorted([r for r in ok if r["direction"] == "long"],
                   key=lambda r: (r["conviction"] or 0), reverse=True)
    shorts = sorted([r for r in ok if r["direction"] == "short"],
                    key=lambda r: (r["conviction"] or 0), reverse=True)
    holds = [r for r in ok if r["direction"] == "hold"]

    lines = []
    lines.append(f"# 多标的信心排名 · {len(ok)} 个 ticker"
                 + (f"（展示每类前 {top}）" if top else ""))
    lines.append(f"> 综合信心分 = 评级 55% + 情绪 25% + 盈亏比 20%（缺失分量重归一化）。"
                 "信心分是**代理分，非校准概率**。评级中/英文均识别。"
                 "⚠️ = 回撤熔断（框架已拒部署，不进 #1 推荐）。")
    lines.append("")

    def pick_top(group):
        """#1 候选：跳过被熔断的。"""
        for r in group:
            if not r.get("blocked"):
                return r
        return None

    top_long = pick_top(longs)
    top_short = pick_top(shorts)

    def group_table(title, group, emoji):
        lines.append(f"## {emoji} {title}（{len(group)}）")
        if not group:
            lines.append("_无_")
            lines.append("")
            return
        lines.append("| # | Ticker | 评级 | 信心分 | 情绪/10 | 盈亏比 | 入场 | 止损 | 报告日期 | 资产 | |")
        lines.append("|---|---|---|---:|---:|---:|---|---|---|---|---|")
        for i, r in enumerate(group[:top] if top else group, 1):
            flag = " ⚠️熔断" if r.get("blocked") else ""
            star = ""
            if r is top_long:
                star = " ← **最强多头**"
            elif r is top_short:
                star = " ← **最强空头**"
            lines.append(
                f"| {i} | `{r['ticker']}` | {r['rating']} | **{r['conviction']}** | "
                f"{r['sentiment_score'] if r['sentiment_score'] is not None else '—'} | "
                f"{r['rr'] if r['rr'] is not None else '—'} | "
                f"{_m(r['entry'])} | {_m(r['stop'])} | {r['report_date'] or '—'} | "
                f"{_asset_cn(r['asset_type'])}{flag} |{star} |"
            )
        lines.append("")

    group_table("做多排名", longs, "🟢")
    group_table("做空排名", shorts, "🔴")

    if holds:
        lines.append(f"## ⚪ 观望（{len(holds)}，不建议进场）")
        lines.append(", ".join(f"`{r['ticker']}`" for r in holds))
        lines.append("")

    if missing:
        lines.append("## ⚠️ 无报告")
        lines.append("没找到最新完成报告（可能没分析过，或目录无 complete_report.md）："
                     + ", ".join(f"`{r['ticker']}`" for r in missing))
        lines.append("")

    # 信心分明细（#1 多头 + #1 空头）
    detail = [r for r in (top_long, top_short) if r]
    if detail:
        lines.append("## 信心分明细（最强多头 / 最强空头）")
        lines.append("| Ticker | 方向 | 评级分 | 情绪分 | 盈亏比分 | 加权后 |")
        lines.append("|---|---|---:|---:|---:|---:|")
        for r in detail:
            c = r["comps"]
            lines.append(
                f"| `{r['ticker']}` | {DIR_BADGE[r['direction']]} | "
                f"{_f2(c.get('rating'))} | {_f2(c.get('sent'))} | "
                f"{_f2(c.get('rr'))} | **{r['conviction']}** |"
            )
        lines.append("")

    # 下一步 handoff
    lines.append("## 下一步（选一个出执行单）")
    if top_long:
        lines.append(f"- 最强多头 `{top_long['ticker']}` → "
                     f"`python scripts/trade_ticket.py --ticker {top_long['ticker']} --capital <你的资金>`")
    if top_short:
        lines.append(f"- 最强空头 `{top_short['ticker']}` → "
                     f"`python scripts/trade_ticket.py --ticker {top_short['ticker']} --capital <你的资金>`")
    if not top_long and not top_short:
        lines.append("- 本组无可推荐标的（全为观望 / 熔断 / 无报告）。")
    # A 股做空通道提示
    if top_short and top_short["asset_type"] == "cn_stock":
        lines.append("- ⚠️ 最强空头是 A 股：做空通道受限（融券难/利率高），建议反向 ETF 或认沽期权替代。")
    lines.append("")
    lines.append("---")
    lines.append("*免责声明：信心分为代理分（评级+情绪+盈亏比加权），基于 LLM 分析结论与历史公式，"
                 "非校准概率，不构成投资建议。杠杆交易有爆仓归零风险。*")

    # 可选：对 #1 多/空追加完整执行单
    if capital and (top_long or top_short):
        for r in [x for x in (top_long, top_short) if x]:
            lines.append("")
            lines.append(f"---\n# 附：`{r['ticker']}` 执行单（capital={capital}, moderate 档）")
            try:
                lines.append(_render_ticket_md(Path(r["report_dir"]), capital))
            except Exception as e:
                lines.append(f"_(生成执行单失败：{e})_")

    return "\n".join(lines)


def _render_ticket_md(report_dir: Path, capital: float) -> str:
    """调 trade_ticket.build_ticket + render_ticket 出完整执行单 markdown。"""
    from trade_ticket import build_ticket as _bt, render_ticket as _rt
    return _rt(_bt(report_dir, capital, "moderate"))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="YiAgents 多标的信心排名（做多/做空/观望分组）")
    ap.add_argument("--tickers", nargs="+", help="要排名的 ticker 列表（不传则全部 ticker 各取最新）")
    ap.add_argument("--top", type=int, default=10, help="每组展示前 N（默认 10；0=全部）")
    ap.add_argument("--since", help="只排报告日期 ≥ 此日的（YYYY-MM-DD）")
    ap.add_argument("--capital", type=float, help="给了则对 #1 多头/#1 空头追加完整执行单")
    ap.add_argument("--json", action="store_true", help="输出 JSON")
    args = ap.parse_args()

    # 决定 ticker 列表：带点号的（600519.SS/0700.HK）原样，纯字母（btcusdt/aapl）大写
    if args.tickers:
        tickers = [t if "." in t else t.upper() for t in args.tickers]
    else:
        root = _results_dir()
        if not root.is_dir():
            print(f"✗ 报告根目录不存在：{root}", file=sys.stderr)
            sys.exit(2)
        seen = set()
        tickers = []
        for d in root.iterdir():
            m = _DIR_STAMP_RE.search(d.name)
            if m and (d / "complete_report.md").is_file():
                tk = d.name[: m.start()]
                if tk not in seen:
                    seen.add(tk)
                    tickers.append(tk)

    # since 过滤
    since = args.since

    rows = []
    for tk in tickers:
        r = analyze_ticker(tk)
        if r.get("status") != "ok":
            rows.append(r)
            continue
        if since and r.get("report_date"):
            # report_date 是 MM-DD；补当前年份比对
            ymd = None
            m = _DIR_STAMP_RE.search(Path(r["report_dir"]).name)
            if m:
                ymd = m.group(1)[:8]  # YYYYMMDD
            if ymd and f"{ymd[0:4]}-{ymd[4:6]}-{ymd[6:8]}" < since:
                continue
        rows.append(r)

    if args.json:
        print(json.dumps(rows, ensure_ascii=False, indent=2, default=str))
    else:
        print(render_rank(rows, tickers, args.top, args.capital))


if __name__ == "__main__":
    main()
