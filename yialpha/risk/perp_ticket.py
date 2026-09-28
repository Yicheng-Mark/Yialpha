"""Deterministic perp ticket math: suggested leverage / liquidation / targets.

Extracted verbatim from ``scripts/trade_ticket.py`` (2026-08-17) so the
runtime decision chain — the risk overlay on a crypto_perp run — computes its
advisory leverage and liquidation-price fields with the SAME tested formulas
the post-hoc ticket script uses, instead of a second divergent
implementation. The script now imports from here; behaviour is unchanged.

Design contract: pure functions, no I/O, no LLM. Leverage is the minimum of
four caps so a stop-distance, a volatility budget, a conviction tier and a
hard asset ceiling ALL have to agree before leverage is granted.

Liquidation uses the same MMR-aware trigger the backtest engine simulates
(``backtest/engine.py``): entry × (1 − 1/L + MMR + fee) for longs, mirrored
for shorts. The MMR ladder is the STATIC default bracket table (no network —
the signed ``/fapi/v1/leverageBracket`` endpoint needs operator keys the
advisory path must not require), evaluated at an ASSUMED notional
(:data:`ASSUMED_NOTIONAL_USDT` — the ticket is pre-trade advice, no position
exists yet); callers that know a real position notional pass it explicitly.
The ticket discloses both assumptions.
"""
from __future__ import annotations

from ..dataflows.binance import _sig_digits
from ..dataflows.binance_brackets import (
    DEFAULT_USDT_M_BRACKETS,
    mmr_for_notional,
)

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

#: ATR 止损乘数（与框架 risk/atr_stop.py 默认 mult=2.0 对齐）。仅作缺省值：
#: 调用方（risk manager 的 ``atr_stop_mult`` 配置）可传自己的乘数，票据
#: 数字与 overlay 实际挂出的止损必须同一乘数（round-6 修：此前 SHORT 一律
#: 硬编码 2·ATR 重建止损，配置 3·ATR 时报告与挂单差一整个 ATR）。
ATR_STOP_MULT = 2.0

#: 爆仓安全倍数：爆仓距离必须 ≥ 爆仓安全倍数 × 止损距离（让止损先于爆仓触发）
LIQ_SAFETY = 2.0

#: Risk-overlay 默认风险偏好（保守档；config ``risk_profile`` 可覆盖）。
#: 注意与票据脚本**不共用默认**：scripts/trade_ticket.py 的 ``--risk-profile``
#: 默认 moderate——那是下单时点的用户选择，这是报告时点的咨询默认，两个
#: 旋钮故意分开（round-6 修正：旧注释声称「与脚本默认一致」，失实）。
DEFAULT_PROFILE = "conservative"

#: 爆仓触发里的费用近似（taker 费 + 清算费缓冲，bps），与回测引擎的
#: fee_rate 处理对齐。作为假设披露在票据里。
LIQ_FEE_RATE_EST = 0.0005

#: 票据时点没有真实仓位（盘前建议），MMR 只能按一个披露的假设名义本金
#: 定档。绝不许用 entry 价格冒充 notional（BTC 60000 的价格会被当成
#: 60,000 USDT 名义本金落到第二档，而 4 美元的股票又落到第一档——
#: 两个都是巧合而非仓位事实）。50,000 USDT 取零售咨询仓位的保守档界；
#: 实际档位随名义本金上升，票据披露该假设。
ASSUMED_NOTIONAL_USDT = 50_000.0

#: 票据必须显式声明的止损触发口径：Binance 条件单默认按 CONTRACT_PRICE
#: （最新成交价）触发，而爆仓本身按 MARK price 判定——两者可以偏离。
#: 靠近爆仓区的止损应显式下单为 MARK_PRICE，而不是靠默认值碰巧。
STOP_TRIGGER_BASIS = (
    "CONTRACT_PRICE (Binance conditional-order default; liquidation itself "
    "is judged on MARK price — the two can diverge)"
)


def rating_strength(rating: str | None) -> int:
    """Case-insensitive :data:`RATING_STRENGTH` lookup（"buy"/"BUY" → 2）。

    enforced 组合路径转发的是 side 派生的小写 proxy（"buy"/"sell"/"hold"），
    而强度表是首字母大写。round-6 修：此前 miss 一律读 0 → L_conv=0 →
    杠杆恒 1.0x、爆仓 advisory 整体消失（每张 enforced perp 票据）。
    """
    if not isinstance(rating, str):
        return 0
    return RATING_STRENGTH.get(rating.strip().capitalize(), 0)


def compute_leverage(
    stop_dist: float | None,
    atr_pct: float | None,
    asset_type: str,
    strength: int,
    profile: str = DEFAULT_PROFILE,
    bracket_max_leverage: float | None = None,
) -> tuple[float, dict[str, float]]:
    """杠杆 = min(各上限)。返回 (L, 各上限明细)。

      L_liq  = 1 / (爆仓安全倍数 × stop_dist)   —— 爆仓距离 ≥ 安全倍数 × 止损距离
      L_vol  = K / atr_pct                       —— 高波动降杠杆
      L_conv = 信心上限（方向强度 × 风险偏好）
      L_hard = 资产类型硬顶
      L_bracket（可选）= 交易所 leverageBracket 该名义档的 initialLeverage 上限
      —— round-6 新增：低 maxLeverage 合约（< 硬顶 20x）此前会收到交易所
      直接拒单的建议杠杆。None（无签名端点/无该字段）时不参与，行为与旧版一致。
    """
    hard = HARD_CEILING.get(asset_type, 5.0)

    l_liq = 1.0 / (LIQ_SAFETY * stop_dist) if stop_dist and stop_dist > 0 else hard
    l_vol = VOL_K.get(asset_type, 0.20) / atr_pct if atr_pct and atr_pct > 0 else hard
    l_conv = CONV_CAP.get(abs(strength), {}).get(profile, 5.0) if strength else 0.0

    caps: dict[str, float] = {"L_liq": l_liq, "L_vol": l_vol, "L_conv": l_conv, "L_hard": hard}
    if bracket_max_leverage is not None and bracket_max_leverage > 0:
        caps["L_bracket"] = float(bracket_max_leverage)

    L = max(1.0, min(min(caps.values()), hard))
    return L, caps


def liquidation_price(
    entry: float, L: float, direction: str, asset_type: str,
    notional: float | None = None,
) -> float | None:
    """隔离保证金爆仓价估算（含维持保证金与费用）。None 表示不适用。

    触发价 = entry × (1 − 1/L + MMR + fee)（多头；空头镜像）——与回测引擎
    （backtest/engine.py）同一公式，咨询票据与模拟权益曲线不会给出两个
    不同的爆仓价。MMR 取静态默认 USDT-M 档位表：``notional`` 显式传入时
    按真实名义本金定档；缺省按 :data:`ASSUMED_NOTIONAL_USDT`（票据是盘前
    建议，无真实仓位）。历史版本用 entry 价格冒充 notional 定档（BTC 60000
    价被当成 60,000 USDT 名义本金），已改为显式假设并在票据中披露。
    旧公式 entry × (1 − 1/L) 忽略 MMR/费用，估出的爆仓价比实际更远，
    系统性低估爆仓风险。
    """
    if asset_type == "crypto_spot" or L <= 1.0:
        return None
    if direction not in ("long", "short"):
        return None
    mmr = mmr_for_notional(
        DEFAULT_USDT_M_BRACKETS,
        ASSUMED_NOTIONAL_USDT if notional is None else notional,
    )
    if direction == "long":
        return entry * (1.0 - 1.0 / L + mmr + LIQ_FEE_RATE_EST)
    return entry * (1.0 + 1.0 / L - mmr - LIQ_FEE_RATE_EST)


def take_profits(
    entry: float | None, stop: float | None, direction: str,
) -> list[float]:
    """R 倍数止盈（R = |entry−stop|）：TP1 = 1.5R, TP2 = 3R, TP3 = 5R。

    纯 R 倍数，结构位另作参考，不硬钳 —— 突破阻力后常有 runner，硬钳到
    阻力位会把三档止盈压成同一个值，反而失去分批意义。舍入用**有效数字**
    （数据层 ``_sig_digits`` 约定、位数放宽到 8）而非小数位：USDT-M 上有
    ~1e-5 价位的合约（PEPEUSDT 等），``round(x, 6)`` 会把 1.3e-5 档位削到
    只剩一位有效数字；8 位在 1e5 价位（BTC）仍保留到分级价位（0.1 价步），
    两端都不损失。
    """
    if entry is None or stop is None:
        return []
    R = abs(entry - stop)
    if R <= 0:
        return []
    out = []
    for mult in (1.5, 3.0, 5.0):
        tp = entry + mult * R if direction == "long" else entry - mult * R
        out.append(_sig_digits(tp, digits=8))
    return out


def perp_ticket_numbers(
    entry: float | None,
    atr: float | None,
    rating: str,
    stop: float | None,
    weight: float,
    notional: float | None = None,
    *,
    profile: str = DEFAULT_PROFILE,
    atr_mult: float | None = None,
    bracket_max_leverage: float | None = None,
) -> tuple[float, dict[str, float | str], float | None, float] | None:
    """Shared numeric core of the perp ticket: leverage, cap detail, estimated
    liquidation price, and the stop actually used.

    Faithful extraction of the overlay renderer's math (V2.0) so the runtime
    :class:`~yialpha.tickets.ExecutionTicket` and the markdown advisory can
    never drift apart. Returns None when the inputs cannot support a ticket —
    non-positive price/ATR, or a flat (zero) weight. Direction is the SIGN of
    ``weight`` (long for positive, short for negative), mirroring the overlay.
    Round-8: an explicit ``stop`` is honoured on BOTH directions when it is
    finite and direction-correct (long: below entry; short: above entry);
    anything missing, non-finite, or on the wrong side is rebuilt as
    ``mult * ATR`` offset against the position. ``notional`` (USDT) tiers the
    liquidation
    MMR bracket at a REAL position notional when the caller knows one;
    otherwise the disclosed :data:`ASSUMED_NOTIONAL_USDT` applies. The detail
    dict carries the leverage caps as floats plus, when a liquidation
    estimate exists, the liq assumption fields (floats + one human-readable
    basis note).

    Keyword overrides (round-6): ``profile`` threads the run's
    ``risk_profile`` config (default :data:`DEFAULT_PROFILE`); ``atr_mult``
    makes the constructed/SHORT stop use the SAME multiplier the risk manager
    configured (default :data:`ATR_STOP_MULT` — previously hardcoded, a
    configured 3·ATR run rendered advice off the actual stop by a whole ATR);
    ``bracket_max_leverage`` adds the exchange's initialLeverage cap when the
    caller could fetch it (see :func:`advisory_leverage_cap`).
    """
    if entry is None or entry <= 0.0 or atr is None or atr <= 0.0:
        return None
    if weight == 0.0:
        return None
    direction = "long" if weight > 0.0 else "short"
    strength = rating_strength(rating)
    mult = ATR_STOP_MULT if atr_mult is None else float(atr_mult)
    stop_used = stop
    if stop_used is not None and not (0.0 < stop_used < float("inf")):
        # Non-positive/NaN/inf stop: unusable, fall through to the rebuild.
        stop_used = None
    # Round-8: a caller-supplied stop must be direction-correct to be
    # honoured — SHORT used to discard ANY supplied stop and silently
    # rebuild mult·ATR, so a future structural short stop would advise
    # leverage/TP levels off a stop the ticket never placed. (Rejection
    # here mirrors the round-6 LONG side-guard.)
    if stop_used is not None and (
        (direction == "long" and stop_used >= entry)
        or (direction == "short" and stop_used <= entry)
    ):
        stop_used = None
    if stop_used is None:
        stop_used = (
            entry - mult * atr
            if direction == "long"
            else entry + mult * atr
        )
    stop_dist = abs(entry - stop_used) / entry
    atr_pct = atr / entry
    lev, lev_caps = compute_leverage(
        stop_dist, atr_pct, "crypto_perp", strength,
        profile=profile, bracket_max_leverage=bracket_max_leverage,
    )
    detail: dict[str, float | str] = dict(lev_caps)
    liq = liquidation_price(entry, lev, direction, "crypto_perp", notional)
    if liq is not None:
        # Assumption disclosure rides the existing detail dict so both
        # renderers (overlay + ExecutionTicket) can state them without a
        # return-shape change.
        detail["liq_mmr"] = mmr_for_notional(
            DEFAULT_USDT_M_BRACKETS,
            ASSUMED_NOTIONAL_USDT if notional is None else notional,
        )
        detail["liq_fee_rate"] = LIQ_FEE_RATE_EST
        detail["liq_mmr_basis"] = (
            "default bracket ladder @ "
            + (
                f"assumed {ASSUMED_NOTIONAL_USDT:,.0f} USDT notional "
                "(real tier rises with size)"
                if notional is None
                else f"notional {notional:,.0f} USDT"
            )
        )
    return lev, detail, liq, stop_used


def advisory_leverage_cap(symbol: str, notional: float | None = None) -> float | None:
    """Fail-soft 交易所 initialLeverage 上限（供票据/overlay advisory）。

    需要签名 ``/fapi/v1/leverageBracket`` 端点（操作者的
    ``BINANCE_API_KEY``/``BINANCE_API_SECRET``）。无 key、传输失败、或该
    合约的档位表不带 initialLeverage（无签名默认近似表就不带）时返回
    None——调用方保持旧的静态硬顶，行为与今天字节一致。取档口径与 MMR
    一致：按 ``notional``（缺省 :data:`ASSUMED_NOTIONAL_USDT`）所在的
    名义档读该档的 initialLeverage。
    """
    try:
        from ..dataflows.binance_brackets import (
            bracket_max_leverage_for,
            get_leverage_brackets,
        )

        brackets = get_leverage_brackets(symbol)
        return bracket_max_leverage_for(
            brackets, ASSUMED_NOTIONAL_USDT if notional is None else notional
        )
    except Exception:  # noqa: BLE001 — advisory-only, must never break a render
        return None
