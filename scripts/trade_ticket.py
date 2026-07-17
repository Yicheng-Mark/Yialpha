#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
trade_ticket.py — YiAgents 分析完成后的「交执行单」生成器。

只读取已生成的报告产物（~/.yiagents/logs/reports/<TICKER>_<ts>/），不触碰任何
agent / 图 / 数据流，零影响（符合「增强层零影响」铁律）。给定用户资金 + 风险偏好，
依据固定分数法（risk-of-ruin）+ ATR 止损 + R 倍数止盈 + 爆仓安全杠杆，输出一张
可执行的交易单：方向 / 入场 / 止损 / 止盈 / 仓位 / 杠杆 / 保证金 / 爆仓价。

用法：
    python scripts/trade_ticket.py                       # 自动取最新报告
    python scripts/trade_ticket.py --ticker BTCUSDT      # 取该 ticker 最新报告
    python scripts/trade_ticket.py --report-dir <path>   # 指定报告目录
    python scripts/trade_ticket.py --capital 10000       # 资金（必填，或交互询问）
    python scripts/trade_ticket.py --capital 10000 --risk-profile moderate
    python scripts/trade_ticket.py --capital 10000 --json

风险偏好：conservative(单笔风险 1.0%) / moderate(1.5%, 默认) / aggressive(2.5%)

免责声明：本脚本为研究框架的后处理工具，输出基于历史回测公式与 LLM 推理结论，不构成
任何投资建议。杠杆交易有爆仓归零风险。
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

# ---------------------------------------------------------------------------
# 资产类型 / 方向 / 评级 常量
# ---------------------------------------------------------------------------

#: 各资产类型的「硬顶杠杆」（监管 / sane ceiling）
HARD_CEILING = {
    "crypto_perp": 20.0,   # 永续合约 sane 上限（币安零售档）
    "crypto_spot": 1.0,    # 现货无杠杆
    "us_stock": 5.0,       # 美股 CFD / 融资实际档（Reg-T 隔夜 2×，日内高些）
    "hk_stock": 5.0,       # 港股融资融券
    "cn_stock": 1.5,       # A 股两融维持担保比约束；做空通道极受限
}

#: 方向强度（|值| 决定信心杠杆上限）：Buy/Sell 强方向=2，Overweight/Underweight 倾斜=1，Hold=0
RATING_STRENGTH = {
    "Buy": 2, "Overweight": 1, "Hold": 0, "Underweight": -1, "Sell": -2,
}

#: 信心杠杆上限（按风险偏好 × 方向强度）
CONV_CAP = {
    2: {"conservative": 5.0, "moderate": 10.0, "aggressive": 15.0},
    1: {"conservative": 3.0, "moderate": 5.0, "aggressive": 8.0},
}

#: 波动率杠杆系数 K：L_vol = K / ATR%
VOL_K = {"crypto_perp": 0.30, "crypto_spot": 0.20, "us_stock": 0.20,
         "hk_stock": 0.20, "cn_stock": 0.15}

#: 单笔风险预算（占资金比例）
RISK_FRAC = {"conservative": 0.010, "moderate": 0.015, "aggressive": 0.025}

#: ATR 止损乘数（与框架 risk/atr_stop.py 默认 mult=2.0 对齐）
ATR_STOP_MULT = 2.0

#: 框架的评级→信心映射（仅作透明展示，不直接驱动杠杆；杠杆用方向强度）
RATING_TO_CONFIDENCE = {
    "Buy": 0.90, "Overweight": 0.72, "Hold": 0.55, "Underweight": 0.35, "Sell": 0.12,
}

#: 爆仓安全倍数：爆仓距离必须 ≥ 爆仓安全倍数 × 止损距离（让止损先于爆仓触发）
LIQ_SAFETY = 2.0


# ---------------------------------------------------------------------------
# 报告定位
# ---------------------------------------------------------------------------

def _results_dir() -> Path:
    """复用框架的 results_dir 解析（YIAGENTS_RESULTS_DIR / ~/.yiagents/logs）。"""
    home = os.path.join(os.path.expanduser("~"), ".yiagents")
    return Path(os.getenv("YIAGENTS_RESULTS_DIR", os.path.join(home, "logs"))) / "reports"


def find_report(ticker: str | None = None, report_dir: str | None = None) -> Path:
    """定位报告目录。优先 report_dir > 按 ticker 最新 > 全局最新。"""
    if report_dir:
        p = Path(report_dir)
        if not p.is_dir():
            raise FileNotFoundError(f"报告目录不存在: {p}")
        return p

    root = _results_dir()
    if not root.is_dir():
        raise FileNotFoundError(
            f"未找到报告根目录 {root}。用 --report-dir 指定，或先跑一次分析。"
        )

    dirs = sorted([d for d in root.iterdir() if d.is_dir()],
                  key=lambda d: d.stat().st_mtime, reverse=True)
    if ticker:
        tk = ticker.upper()
        for d in dirs:
            if d.name.upper().startswith(tk + "_") or d.name.upper() == tk:
                return d
        raise FileNotFoundError(f"在 {root} 下未找到 ticker={ticker} 的报告。")
    if not dirs:
        raise FileNotFoundError(f"{root} 下没有任何报告。")
    return dirs[0]


def detect_asset_type(ticker: str) -> str:
    """从 ticker 后缀推断资产类型（与报告 state.asset_type 一致）。"""
    t = ticker.upper()
    if t.endswith("USDT") or t.endswith("USDC") or t.endswith("BUSD"):
        # 现货 vs 永续：YiAgents 的加密报告目录一律 *USDT，含 perp / spot；
        # 杠杆语义下两者都按 perp 处理（spot 无杠杆会被 HARD_CEILING 兜住）
        return "crypto_perp"
    if t.endswith(".SS") or t.endswith(".SZ"):
        return "cn_stock"
    if t.endswith(".HK"):
        return "hk_stock"
    return "us_stock"


# ---------------------------------------------------------------------------
# 报告解析（多源优先级）
# ---------------------------------------------------------------------------

def _fnum(s) -> float | None:
    """从字符串里抠出第一个数字（含千分位/逗号/负号/小数）。"""
    if s is None:
        return None
    s = str(s)
    # 先抓"数字串（允许内嵌逗号）"，再去掉逗号转 float。
    # 用 [\d,]* 兼容 "1,234.56" / "65,000.0" / "65000.0"，且不会因 {1,3} 截断无逗号大数。
    m = re.search(r"-?\d[\d,]*\.?\d*", s)
    if not m:
        return None
    try:
        return float(m.group(0).replace(",", ""))
    except ValueError:
        return None


def parse_trader(md: str) -> dict:
    """解析 3_trading/trader.md（3 档 Action + Entry/Stop）。"""
    out = {"action": None, "entry": None, "stop": None, "sizing": None}

    m = re.search(r"\*\*Action\*\*\s*:\s*(\w+)", md, re.I)
    if m:
        out["action"] = m.group(1).strip().capitalize()

    # FINAL TRANSACTION PROPOSAL: **BUY** 作为 action 的权威兜底
    mfp = re.search(r"FINAL TRANSACTION PROPOSAL:\s*\*\*(\w+)\*\*", md, re.I)
    if mfp and not out["action"]:
        out["action"] = mfp.group(1).strip().capitalize()

    m = re.search(r"\*\*Entry Price\*\*\s*:\s*([^\n]+)", md, re.I)
    out["entry"] = _fnum(m.group(1)) if m else None

    m = re.search(r"\*\*Stop Loss\*\*\s*:\s*([^\n]+)", md, re.I)
    out["stop"] = _fnum(m.group(1)) if m else None

    m = re.search(r"\*\*Position Sizing\*\*\s*:\s*([^\n]+)", md, re.I)
    out["sizing"] = m.group(1).strip() if m else None
    return out


def parse_pm_decision(md: str) -> dict:
    """解析 5_portfolio/decision.md：5 档评级 + 量化风控覆盖层（若存在）。"""
    out = {
        "rating": None, "price_target": None,
        # 量化风控覆盖层字段（risk_enabled=True 时追加；ATR 止损 / 爆仓体制 / 目标仓位）
        "ovl_action": None, "ovl_target_weight": None, "ovl_stop": None,
        "ovl_entry_ref": None, "ovl_drawdown_regime": None,
    }

    # 评级：**Rating**: X（5 档权威）
    m = re.search(r"\*\*Rating\*\*\s*:\s*:?-?\s*\**\s*(Buy|Overweight|Hold|Underweight|Sell)",
                  md, re.I)
    if m:
        out["rating"] = m.group(1).strip().capitalize()
    # 中文报告：**评级**：买入/增持/持有/减持/卖出
    if not out["rating"]:
        cn = {"买入": "Buy", "增持": "Overweight", "持有": "Hold",
              "减持": "Underweight", "卖出": "Sell"}
        m = re.search(r"\*\*评级\*\*\s*[:：]\s*(买入|增持|持有|减持|卖出)", md)
        if m:
            out["rating"] = cn[m.group(1)]

    m = re.search(r"\*\*Price Target\*\*\s*:\s*([^\n]+)", md, re.I)
    out["price_target"] = _fnum(m.group(1)) if m else None

    # 量化风控覆盖层（追加段，可能在 decision.md 或仅 in final_trade_decision）
    def ovl(field_pat):
        m = re.search(r"\*\*" + field_pat + r"\*\*\s*:\s*([^\n]+)", md, re.I)
        return m.group(1).strip() if m else None

    out["ovl_action"] = ovl("Action")  # enter|add|hold|reduce|exit|blocked
    tw = ovl(r"Target Weight")
    if tw:
        out["ovl_target_weight"] = _fnum(tw)  # 百分比数值
    out["ovl_stop"] = _fnum(ovl(r"Stop Loss")) if ovl(r"Stop Loss") else None
    out["ovl_entry_ref"] = _fnum(ovl(r"Entry Reference")) if ovl(r"Entry Reference") else None
    dr = ovl(r"Drawdown Regime")
    if dr:
        mm = re.search(r"(normal|caution|no_new|hard_stop)", dr, re.I)
        out["ovl_drawdown_regime"] = mm.group(1).lower() if mm else None
    return out


def parse_market(md: str) -> dict:
    """解析 1_analysts/market.md：当前价、ATR、支撑/阻力（尽力提取）。"""
    out = {"current_price": None, "atr": None, "supports": [], "resistances": []}
    if not md:
        return out

    # ATR：多种写法 —— "ATR(14): $7.29" / "ATR at $7.29" / "ATR为 7.29"
    for pat in [r"ATR\(?\d*\)?\s*(?:\(14\))?\s*[:：]\s*\$?\s*([\d,]+\.?\d*)",
                r"ATR\s+(?:at|is|为)?\s*\$?\s*([\d,]+\.?\d*)",
                r"14[日天]ATR[^\d]{0,6}([\d,]+\.?\d*)"]:
        m = re.search(pat, md, re.I)
        if m:
            out["atr"] = _fnum(m.group(1))
            break

    # 当前价 / 收盘价
    for pat in [r"(?:当前价|收盘价|最新价|Current Price|Close)\s*[:：]?\s*\$?\s*([\d,]+\.?\d*)",
                r"价格[：:]\s*\$?\s*([\d,]+\.?\d*)"]:
        m = re.search(pat, md, re.I)
        if m:
            out["current_price"] = _fnum(m.group(1))
            break

    # 支撑 / 阻力（中英，取数值；best-effort，失败不致命）
    for m in re.finditer(r"(?:支撑|Support)[^$\d]{0,12}\$?\s*([\d,]+\.?\d*)", md, re.I):
        v = _fnum(m.group(1))
        if v:
            out["supports"].append(v)
    for m in re.finditer(r"(?:阻力|压力|Resistance)[^$\d]{0,12}\$?\s*([\d,]+\.?\d*)", md, re.I):
        v = _fnum(m.group(1))
        if v:
            out["resistances"].append(v)
    return out


# ---------------------------------------------------------------------------
# 核心：方向判定 + 仓位/杠杆/TP/SL 计算
# ---------------------------------------------------------------------------

def decide_direction(rating: str | None, action: str | None) -> tuple[str, str]:
    """
    返回 (方向, 理由)。方向 = long / short / hold。
    PM 评级（5 档）优先；缺失时回退 Trader Action（3 档）。
    """
    if rating in ("Buy", "Overweight"):
        return "long", f"PM 评级 {rating} → 做多"
    if rating in ("Sell", "Underweight"):
        return "short", f"PM 评级 {rating} → 做空"
    if rating == "Hold":
        return "hold", "PM 评级 Hold → 观望，不建议进场"
    # 评级缺失，回退 Trader action
    if action == "Buy":
        return "long", "Trader Action Buy（PM 评级缺失）→ 做多"
    if action == "Sell":
        return "short", "Trader Action Sell（PM 评级缺失）→ 做空"
    return "hold", "无明确方向信号 → 观望"


def resolve_levels(parsed_trader, parsed_pm, parsed_market, direction):
    """
    多源解析入场价 / 止损价 / ATR。优先级：
      Entry  : trader.Entry Price（计划入场区）> overlay.Entry Reference（last close）> market.当前价
      Stop   : overlay.Stop Loss（ATR 止损）> trader.Stop Loss > 由 ATR 推导
      ATR    : market.ATR > 由 stop 推导（stop ≈ entry ∓ 2·ATR）
    """
    entry = parsed_trader["entry"] or parsed_pm["ovl_entry_ref"] or parsed_market["current_price"]
    stop = parsed_pm["ovl_stop"] or parsed_trader["stop"]
    atr = parsed_market["atr"]

    # 由 stop 反推 ATR（若 stop 是 ATR 止损：stop = entry ∓ mult·ATR）
    if atr is None and entry and stop:
        atr = abs(entry - stop) / ATR_STOP_MULT

    # 仍无 stop：用 ATR 构造
    if stop is None and entry and atr:
        stop = entry - ATR_STOP_MULT * atr if direction == "long" else entry + ATR_STOP_MULT * atr

    return entry, stop, atr


def compute_leverage(stop_dist, atr_pct, asset_type, strength, profile):
    """
    杠杆 = min(四个上限)。返回 (L, 各上限明细)。
      L_liq  = 1 / (爆仓安全倍数 × stop_dist)   —— 爆仓距离 ≥ 安全倍数 × 止损距离
      L_vol  = K / atr_pct                       —— 高波动降杠杆
      L_conv = 信心上限（方向强度 × 风险偏好）
      L_hard = 资产类型硬顶
    """
    hard = HARD_CEILING.get(asset_type, 5.0)

    if stop_dist and stop_dist > 0:
        l_liq = 1.0 / (LIQ_SAFETY * stop_dist)
    else:
        l_liq = hard
    if atr_pct and atr_pct > 0:
        l_vol = VOL_K.get(asset_type, 0.20) / atr_pct
    else:
        l_vol = hard
    l_conv = CONV_CAP.get(abs(strength), {}).get(profile, 5.0) if strength else 0.0

    L = max(1.0, min(l_liq, l_vol, l_conv, hard))
    return L, {"L_liq": l_liq, "L_vol": l_vol, "L_conv": l_conv, "L_hard": hard}


def liquidation_price(entry, L, direction, asset_type):
    """隔离保证金爆仓价估算（忽略维持保证金/费率，保守略近）。None 表示不适用。"""
    if asset_type == "crypto_spot" or L <= 1.0:
        return None
    if direction == "long":
        return entry * (1.0 - 1.0 / L)
    if direction == "short":
        return entry * (1.0 + 1.0 / L)
    return None


def take_profits(entry, stop, direction):
    """
    R 倍数止盈（R = |entry−stop|）：TP1 = 1.5R, TP2 = 3R, TP3 = 5R。
    纯 R 倍数，结构位另作参考提示（见 render），不硬钳 —— 突破阻力后常有 runner，
    硬钳到阻力位会把三档止盈压成同一个值，反而失去分批意义。
    """
    if entry is None or stop is None:
        return []
    R = abs(entry - stop)
    if R <= 0:
        return []
    out = []
    for mult in (1.5, 3.0, 5.0):
        tp = entry + mult * R if direction == "long" else entry - mult * R
        out.append(round(tp, 6))
    return out


def build_ticket(report_dir: Path, capital: float, profile: str) -> dict:
    """读报告 + 算交易单。返回结构化 dict。"""
    def read(rel):
        p = report_dir / rel
        return p.read_text(encoding="utf-8", errors="ignore") if p.exists() else ""

    trader_md = read("3_trading/trader.md")
    pm_md = read("5_portfolio/decision.md")
    market_md = read("1_analysts/market.md")

    parsed_trader = parse_trader(trader_md)
    parsed_pm = parse_pm_decision(pm_md)
    parsed_market = parse_market(market_md)

    # ticker 从目录名取
    ticker = report_dir.name.split("_2")[0] if "_2" in report_dir.name else report_dir.name
    # 更稳妥：剥掉末尾 _YYYYMMDD_HHMMSS
    ticker = re.sub(r"_\d{8}_\d{6}$", "", report_dir.name)

    asset_type = detect_asset_type(ticker)
    rating = parsed_pm["rating"]
    action = parsed_trader["action"]
    direction, dir_reason = decide_direction(rating, action)

    entry, stop, atr = resolve_levels(parsed_trader, parsed_pm, parsed_market, direction)

    # 熔断器护栏：回撤体制 no_new / hard_stop → 拒绝进场
    regime = parsed_pm["ovl_drawdown_regime"]
    blocked = regime in ("no_new", "hard_stop")

    ticket = {
        "ticker": ticker,
        "asset_type": asset_type,
        "report_dir": str(report_dir),
        "rating": rating,
        "trader_action": action,
        "direction": direction,
        "direction_reason": dir_reason,
        "entry": entry,
        "stop": stop,
        "atr": atr,
        "current_price": parsed_market["current_price"],
        "price_target_pm": parsed_pm["price_target"],
        "supports": parsed_market["supports"][:4],
        "resistances": parsed_market["resistances"][:4],
        "drawdown_regime": regime,
        "overlay_target_weight_pct": parsed_pm["ovl_target_weight"],
        "framework_positioning": parsed_trader["sizing"],
        "capital": capital,
        "risk_profile": profile,
        "blocked_by_breaker": blocked,
    }

    if direction == "hold" or blocked or entry is None or stop is None:
        ticket["status"] = ("no_trade_breaker" if blocked else
                            "no_trade_hold" if direction == "hold" else "missing_levels")
        return ticket

    # ---- 计算仓位 / 杠杆 / TP ----
    stop_dist = abs(entry - stop) / entry            # 止损百分比距离
    R_dollar = abs(entry - stop)                      # 每单位风险（1R）
    atr_pct = (atr / entry) if (atr and entry) else None
    risk_dollar = capital * RISK_FRAC[profile]        # 单笔风险预算
    notional = risk_dollar / stop_dist                # 名义仓位（核心：打止损恰好亏 risk_dollar）
    quantity = notional / entry

    strength = RATING_STRENGTH.get(rating, 1 if direction != "hold" else 0)
    L, caps = compute_leverage(stop_dist, atr_pct, asset_type, strength, profile)
    margin = notional / L
    margin_pct = margin / capital

    # 保证金占比护栏：>50% 提示过度集中
    margin_warn = margin_pct > 0.50

    liq = liquidation_price(entry, L, direction, asset_type)
    liq_dist = (abs(entry - liq) / entry) if liq else None

    tps = take_profits(entry, stop, direction)

    ticket.update({
        "status": "ok",
        "stop_dist_pct": stop_dist,
        "R_dollar": R_dollar,
        "atr_pct": atr_pct,
        "risk_dollar": risk_dollar,
        "risk_frac": RISK_FRAC[profile],
        "notional": notional,
        "quantity": quantity,
        "leverage": L,
        "leverage_caps": caps,
        "margin": margin,
        "margin_pct_of_capital": margin_pct,
        "margin_warning": margin_warn,
        "liquidation_price": liq,
        "liquidation_dist_pct": liq_dist,
        "take_profits": tps,
        "conviction_score": RATING_TO_CONFIDENCE.get(rating),
    })
    return ticket


# ---------------------------------------------------------------------------
# 渲染
# ---------------------------------------------------------------------------

def _money(x, unit=""):
    if x is None:
        return "—"
    if abs(x) >= 1000:
        s = f"{x:,.2f}"
    else:
        s = f"{x:,.4f}".rstrip("0").rstrip(".")
    return f"{unit}{s}"


def _pct(x):
    return "—" if x is None else f"{x*100:.2f}%"


def _mult(x):
    return "—" if x is None else f"{x:.1f}×"


DIR_CN = {"long": "做多 🟢", "short": "做空 🔴", "hold": "观望 ⚪"}


def render_ticket(t: dict) -> str:
    lines = []
    lines.append(f"# 交易执行单 · {t['ticker']}  ({t['asset_type']})")
    lines.append(f"> 报告：`{t['report_dir']}`")
    lines.append(f"> 评级 `{t['rating']}` · Trader Action `{t['trader_action']}` · "
                 f"资金 {_money(t['capital'])} · 风险偏好 `{t['risk_profile']}`")
    lines.append("")

    # ---- 不进场情形 ----
    if t["status"] in ("no_trade_hold", "no_trade_breaker", "missing_levels"):
        if t["status"] == "no_trade_breaker":
            lines.append(f"## ⛔ 不建议进场 —— 风控熔断器触发")
            lines.append(f"回撤体制 = **{t['drawdown_regime']}**（no_new/hard_stop）。"
                         "框架的 Kelly×Breaker×CVaR 层已主动拒绝部署新资金。"
                         "此时硬上杠杆等于在回撤中接刀，应空仓等体制回到 normal/caution。")
        elif t["status"] == "no_trade_hold":
            lines.append(f"## ⚪ 本笔观望，不建议进场")
            lines.append(f"{t['direction_reason']}。")
            lines.append("Hold 评级意味着多空力量均衡，强行套杠杆没有统计优势。"
                         "若你非要进场，至少等评级转为 Overweight/Underweight，"
                         "或价格突破关键位后再用本单的条件单逻辑。")
        else:
            lines.append("## ⚠️ 无法生成交易单")
            lines.append("报告里没有可解析的入场价/止损价。请在 trader.md / decision.md "
                         "里补全 **Entry Price** / **Stop Loss**，或用 --report-dir 指定含"
                         "量化风控覆盖层的报告（risk_enabled=True 才会带 ATR 止损）。")
        lines.append("")
        lines.append(_disclaimer())
        return "\n".join(lines)

    # ---- 正常交易单 ----
    d = t["direction"]
    lines.append(f"## ① 方向：**{DIR_CN[d]}**")
    lines.append(f"{t['direction_reason']}")
    if t["conviction_score"] is not None:
        lines.append(f"_框架评级信心映射：{t['conviction_score']:.2f}"
                     f"（仅作透明展示，杠杆由方向强度+波动率+爆仓安全共同决定）_")
    lines.append("")

    lines.append("## ② 入场 / 止损 / 止盈")
    lines.append("| 项 | 值 | 说明 |")
    lines.append("|---|---|---|")
    lines.append(f"| 入场价 E | **{_money(t['entry'])}** | 计划建仓区 |")
    lines.append(f"| 止损价 S | **{_money(t['stop'])}** | 距入场 {_pct(t['stop_dist_pct'])}（1R = {_money(t['R_dollar'])}） |")
    for i, tp in enumerate(t["take_profits"], 1):
        tag = ["1.5R（首笔）", "3R（主仓）", "5R（尾仓）"][i - 1] if i <= 3 else ""
        lines.append(f"| 止盈 TP{i} | **{_money(tp)}** | {tag} |")
    # 结构位参考（不钳 TP，只提示可能的遇阻/支撑）
    if d == "long" and t["resistances"]:
        r = min(v for v in t["resistances"] if v > t["entry"])
        lines.append(f"| 结构参考·最近阻力 | {_money(r)} | 注意 TP 可能在此之前遇阻（突破则看下一档） |")
    if d == "short" and t["supports"]:
        s = max(v for v in t["supports"] if v < t["entry"])
        lines.append(f"| 结构参考·最近支撑 | {_money(s)} | 注意 TP 可能在此之前企稳（跌破则看下一档） |")
    if t["price_target_pm"]:
        lines.append(f"| PM 目标价 | {_money(t['price_target_pm'])} | 组合经理给的参考 |")
    if t["liquidation_price"] is not None:
        ratio = (t["liquidation_dist_pct"] / t["stop_dist_pct"]) if t["stop_dist_pct"] else None
        rstr = f"，{ratio:.1f}× 止损距离（≥{LIQ_SAFETY:.0f}× 才安全）" if ratio else ""
        lines.append(f"| 爆仓价(估) | {_money(t['liquidation_price'])} | 距入场 {_pct(t['liquidation_dist_pct'])}{rstr} |")
    lines.append("")

    lines.append("## ③ 仓位 / 杠杆 / 保证金")
    lines.append("| 项 | 值 | 说明 |")
    lines.append("|---|---|---|")
    lines.append(f"| 单笔最大亏损 | **{_money(t['risk_dollar'])}** | 资金的 {_pct(t['risk_frac'])}（{t['risk_profile']} 档：打止损恰好亏这么多） |")
    lines.append(f"| 名义仓位 | **{_money(t['notional'])}** | = 风险预算 ÷ 止损距离；占资金 {_pct(t['notional']/t['capital'])} |")
    lines.append(f"| 数量 | {_money(t['quantity'])} {_share_unit(t)} | 名义 ÷ 入场价 |")
    lines.append(f"| **建议杠杆** | **{_mult(t['leverage'])}** | 见下方四上限 |")
    lines.append(f"| 保证金(占用) | **{_money(t['margin'])}** | 占资金 {_pct(t['margin_pct_of_capital'])}"
                 + (" ⚠️ 过度集中，建议降风险预算" if t["margin_warning"] else "") + " |")
    lines.append("")

    lines.append("### 杠杆是怎么来的（取四个上限的最小值）")
    c = t["leverage_caps"]
    rows = [
        ("爆仓安全上限 L_liq", c["L_liq"], f"1 ÷ ({LIQ_SAFETY:.0f} × 止损距离 {_pct(t['stop_dist_pct'])})，保证爆仓远在止损之外"),
        ("波动率上限 L_vol", c["L_vol"], f"K ÷ ATR% = {VOL_K.get(t['asset_type'])} ÷ {_pct(t['atr_pct'])}，高波动降杠杆"),
        ("信心上限 L_conv", c["L_conv"], f"方向强度 {t['rating']} × {t['risk_profile']} 档"),
        ("资产硬顶 L_hard", c["L_hard"], f"{t['asset_type']} 监管/sane 上限"),
    ]
    lines.append("| 上限 | 值 | 依据 |")
    lines.append("|---|---|---|")
    for name, val, why in rows:
        flag = "  ← **生效**" if abs(val - t["leverage"]) < 1e-9 else ""
        lines.append(f"| {name} | {_mult(val)} | {why}{flag} |")
    lines.append("")

    lines.append("## ④ 执行建议")
    lines.append(_execution_plan(t))
    lines.append("")

    # 框架自带的仓位建议作为交叉校验
    if t["framework_positioning"] or t["overlay_target_weight_pct"]:
        lines.append("## ⑤ 与框架自带仓位建议的交叉校验")
        if t["overlay_target_weight_pct"] is not None:
            lines.append(f"- 框架量化风控层目标仓位：**{t['overlay_target_weight_pct']:.1f}%** 净值"
                         f"（Kelly×Breaker×CVaR；本单名义占资金 {_pct(t['notional']/t['capital'])}，含杠杆敞口）")
        if t["framework_positioning"]:
            lines.append(f"- Trader 原始仓位描述：{t['framework_positioning']}")
        if t["drawdown_regime"]:
            lines.append(f"- 回撤体制：**{t['drawdown_regime']}**")
        lines.append("")

    lines.append(_disclaimer())
    return "\n".join(lines)


def _share_unit(t):
    if t["asset_type"].startswith("crypto"):
        # BTCUSDT → BTC（展示基础币种）
        return re.sub(r"(USDT|USDC|BUSD)$", "", t["ticker"].upper())
    return "股"


def _execution_plan(t):
    d = t["direction"]
    tps = t["take_profits"]
    parts = [
        f"分批进场：首批用 {t['margin']:.0f} 保证金开 {_mult(t['leverage'])} 杠杆头寸；"
        f"若留有补仓预算，分 2–3 笔在入场区附近摊平。",
        f"止损铁律：价格触及 **{_money(t['stop'])}** 必须无条件平仓，不挪止损。"
        f"单笔最大亏损 {_money(t['risk_dollar'])}（资金的 {_pct(t['risk_frac'])}）。",
    ]
    if len(tps) >= 2:
        parts.append(
            f"分批止盈：到 TP1 {_money(tps[0])} 平 1/3 锁利并把止损上移到成本；"
            f"到 TP2 {_money(tps[1])} 再平 1/3；余仓用 "
            + ("价格回到成本下方" if d == "long" else "价格回到成本上方")
            + f" 的移动止损（或跌破/突破 10 周线）离场。"
        )
    if t["asset_type"].startswith("crypto"):
        parts.append("永续合约：留意资金费率（多头为正时持续扣费），隔夜成本会侵蚀收益；"
                     "设止损时给盘口噪音留 0.3–0.5% 缓冲，避免插针打穿。")
    if t["asset_type"] == "cn_stock" and d == "short":
        parts.append("⚠️ A 股做空通道受限：普通散户融券标的少、利率高，实务上可用反向 ETF / "
                     "认沽期权替代裸做空；两融维持担保比别低于 1.5。")
    if t["asset_type"] == "hk_stock" and d == "short":
        parts.append("港股做空需融券，注意融券费率与可借券额度；流动性差的小盘股慎做空。")
    return "\n".join(f"- {p}" for p in parts)


def _disclaimer():
    return ("---\n*免责声明：本单由 `scripts/trade_ticket.py` 依据固定分数法 + ATR 止损 + "
            "R 倍数止盈 + 爆仓安全杠杆公式生成，输入为 YiAgents 的 LLM 分析结论。LLM 输出"
            "具非确定性，公式基于历史假设，杠杆交易有爆仓归零风险。不构成投资建议。*")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="YiAgents 分析完成后生成交执行单（方向+杠杆+止盈止损+仓位）")
    ap.add_argument("--ticker", help="按 ticker 取最新报告")
    ap.add_argument("--report-dir", help="直接指定报告目录")
    ap.add_argument("--capital", type=float, help="本笔可动用资金（必填）")
    ap.add_argument("--risk-profile", choices=["conservative", "moderate", "aggressive"],
                    default="moderate", help="风险偏好（默认 moderate）")
    ap.add_argument("--json", action="store_true", help="输出 JSON（便于机器读取）")
    args = ap.parse_args()

    capital = args.capital
    if capital is None:
        # 交互询问（SKILL.md 也可以让 Claude 先问好再传进来）
        try:
            capital = float(input("本笔可动用资金（金额）: ").strip().replace(",", ""))
        except (ValueError, EOFError):
            print("✗ 必须提供 --capital", file=sys.stderr)
            sys.exit(2)
    if capital <= 0:
        print("✗ 资金必须 > 0", file=sys.stderr)
        sys.exit(2)

    report_dir = find_report(ticker=args.ticker, report_dir=args.report_dir)
    ticket = build_ticket(report_dir, capital, args.risk_profile)

    if args.json:
        print(json.dumps(ticket, ensure_ascii=False, indent=2, default=str))
    else:
        print(render_ticket(ticket))


if __name__ == "__main__":
    main()
