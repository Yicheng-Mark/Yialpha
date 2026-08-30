"""Deterministic perp ticket math: suggested leverage / liquidation / targets.

Extracted verbatim from ``scripts/trade_ticket.py`` (2026-08-17) so the
runtime decision chain — the risk overlay on a crypto_perp run — computes its
advisory leverage and liquidation-price fields with the SAME tested formulas
the post-hoc ticket script uses, instead of a second divergent
implementation. The script now imports from here; behaviour is unchanged.

Design contract: pure functions, no I/O, no LLM. Leverage is the minimum of
four caps so a stop-distance, a volatility budget, a conviction tier and a
hard asset ceiling ALL have to agree before leverage is granted.
"""
from __future__ import annotations

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

#: ATR 止损乘数（与框架 risk/atr_stop.py 默认 mult=2.0 对齐）
ATR_STOP_MULT = 2.0

#: 爆仓安全倍数：爆仓距离必须 ≥ 爆仓安全倍数 × 止损距离（让止损先于爆仓触发）
LIQ_SAFETY = 2.0

#: Risk-overlay 默认风险偏好（与脚本默认一致；保守档）
DEFAULT_PROFILE = "conservative"


def compute_leverage(
    stop_dist: float | None,
    atr_pct: float | None,
    asset_type: str,
    strength: int,
    profile: str = DEFAULT_PROFILE,
) -> tuple[float, dict[str, float]]:
    """杠杆 = min(四个上限)。返回 (L, 各上限明细)。

      L_liq  = 1 / (爆仓安全倍数 × stop_dist)   —— 爆仓距离 ≥ 安全倍数 × 止损距离
      L_vol  = K / atr_pct                       —— 高波动降杠杆
      L_conv = 信心上限（方向强度 × 风险偏好）
      L_hard = 资产类型硬顶
    """
    hard = HARD_CEILING.get(asset_type, 5.0)

    l_liq = 1.0 / (LIQ_SAFETY * stop_dist) if stop_dist and stop_dist > 0 else hard
    l_vol = VOL_K.get(asset_type, 0.20) / atr_pct if atr_pct and atr_pct > 0 else hard
    l_conv = CONV_CAP.get(abs(strength), {}).get(profile, 5.0) if strength else 0.0

    L = max(1.0, min(l_liq, l_vol, l_conv, hard))
    return L, {"L_liq": l_liq, "L_vol": l_vol, "L_conv": l_conv, "L_hard": hard}


def liquidation_price(
    entry: float, L: float, direction: str, asset_type: str,
) -> float | None:
    """隔离保证金爆仓价估算（忽略维持保证金/费率，保守略近）。None 表示不适用。"""
    if asset_type == "crypto_spot" or L <= 1.0:
        return None
    if direction == "long":
        return entry * (1.0 - 1.0 / L)
    if direction == "short":
        return entry * (1.0 + 1.0 / L)
    return None


def take_profits(
    entry: float | None, stop: float | None, direction: str,
) -> list[float]:
    """R 倍数止盈（R = |entry−stop|）：TP1 = 1.5R, TP2 = 3R, TP3 = 5R。

    纯 R 倍数，结构位另作参考，不硬钳 —— 突破阻力后常有 runner，硬钳到
    阻力位会把三档止盈压成同一个值，反而失去分批意义。
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


def perp_ticket_numbers(
    entry: float | None,
    atr: float | None,
    rating: str,
    stop: float | None,
    weight: float,
) -> tuple[float, dict[str, float], float | None, float] | None:
    """Shared numeric core of the perp ticket: leverage, cap detail, estimated
    liquidation price, and the stop actually used.

    Faithful extraction of the overlay renderer's math (V2.0) so the runtime
    :class:`~yialpha.tickets.ExecutionTicket` and the markdown advisory can
    never drift apart. Returns None when the inputs cannot support a ticket —
    non-positive price/ATR, or a flat (zero) weight. Direction is the SIGN of
    ``weight`` (long for positive, short for negative), mirroring the overlay;
    a SHORT always re-derives its stop from ATR (the ATR stop module
    implements the long form only), and a LONG without an explicit stop gets
    one constructed the same way.
    """
    if entry is None or entry <= 0.0 or atr is None or atr <= 0.0:
        return None
    if weight == 0.0:
        return None
    direction = "long" if weight > 0.0 else "short"
    strength = RATING_STRENGTH.get(rating, 0)
    stop_used = stop
    if direction == "short" or stop_used is None:
        stop_used = (
            entry - ATR_STOP_MULT * atr
            if direction == "long"
            else entry + ATR_STOP_MULT * atr
        )
    stop_dist = abs(entry - stop_used) / entry
    atr_pct = atr / entry
    lev, detail = compute_leverage(stop_dist, atr_pct, "crypto_perp", strength)
    liq = liquidation_price(entry, lev, direction, "crypto_perp")
    return lev, detail, liq, stop_used
