"""Dedicated tests for scripts/trade_ticket.py (the post-analysis ticket CLI).

Before this file the script had zero dedicated coverage — only three pins
inside test_perp_bugfix_regressions.py (``_fnum`` exponent parsing, ``_money``
micro prices, the breakout-resistance render guard), which are NOT duplicated
here. This file owns the script's own logic end to end:

* ``decide_direction`` — the 5-tier rating matrix + Trader-action fallback;
* ``resolve_levels`` — the entry/stop/ATR multi-source priority chain;
* ``parse_trader`` / ``parse_pm_decision`` — markdown extraction, including
  the 2026-09-19 ``ovl_is_perp`` spot/perp discriminator;
* ``build_ticket`` — report-directory consumption through tmp_path report
  trees: perp overlay / spot overlay / bare (no overlay) / hold, plus the
  breaker and missing-levels early returns;
* ``detect_asset_type`` — the ticker-suffix table (incl. the documented
  "crypto suffix defaults to perp" rule).

Hermetic: every input is a markdown string or a tmp_path report tree; no
network, no LLM, no ``~/.yialpha`` access.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_tt_spec = importlib.util.spec_from_file_location(
    "trade_ticket_dedicated_test",
    Path(__file__).resolve().parents[1] / "scripts" / "trade_ticket.py",
)
tt = importlib.util.module_from_spec(_tt_spec)
_tt_spec.loader.exec_module(tt)


# --------------------------------------------------------------------------- #
# detect_asset_type: ticker-suffix table
# --------------------------------------------------------------------------- #


@pytest.mark.unit
@pytest.mark.parametrize(
    ("ticker", "expected"),
    [
        ("BTCUSDT", "crypto_perp"),
        ("PEPEUSDC", "crypto_perp"),
        ("CAKEBUSD", "crypto_perp"),
        ("600519.SS", "cn_stock"),
        ("0700.HK", "hk_stock"),
        ("AAPL", "us_stock"),
    ],
)
def test_detect_asset_type_by_suffix(ticker, expected):
    assert tt.detect_asset_type(ticker) == expected


# --------------------------------------------------------------------------- #
# decide_direction: 5-tier matrix + Trader-action fallback
# --------------------------------------------------------------------------- #


@pytest.mark.unit
@pytest.mark.parametrize(
    ("rating", "action", "expected"),
    [
        # The 5 PM tiers drive the direction on their own.
        ("Buy", None, "long"),
        ("Overweight", None, "long"),
        ("Hold", None, "hold"),
        ("Underweight", None, "short"),
        ("Sell", None, "short"),
        # Rating missing -> Trader action fallback.
        (None, "Buy", "long"),
        (None, "Sell", "short"),
        (None, "Hold", "hold"),  # the trader's neutral wording is no direction
        # Both missing -> flat.
        (None, None, "hold"),
        # An explicit Hold rating outranks the trader's Buy (PM is authoritative).
        ("Hold", "Buy", "hold"),
    ],
)
def test_decide_direction_matrix(rating, action, expected):
    direction, reason = tt.decide_direction(rating, action)
    assert direction == expected
    assert reason  # the ticket always carries a human-readable reason


# --------------------------------------------------------------------------- #
# resolve_levels: entry/stop/ATR priority chain
# --------------------------------------------------------------------------- #


def _trader(entry=None, stop=None):
    return {"action": None, "entry": entry, "stop": stop, "sizing": None}


def _pm(ovl_entry_ref=None, ovl_stop=None):
    return {
        "rating": None, "price_target": None,
        "ovl_action": None, "ovl_target_weight": None,
        "ovl_stop": ovl_stop, "ovl_entry_ref": ovl_entry_ref,
        "ovl_drawdown_regime": None, "ovl_is_perp": False,
    }


def _market(current_price=None, atr=None, supports=(), resistances=()):
    return {
        "current_price": current_price, "atr": atr,
        "supports": list(supports), "resistances": list(resistances),
    }


@pytest.mark.unit
def test_resolve_levels_entry_priority_trader_overlay_market():
    # trader.entry (the planned entry zone) wins ...
    entry, _, _ = tt.resolve_levels(
        _trader(entry=100.0), _pm(ovl_entry_ref=90.0),
        _market(current_price=80.0), "long",
    )
    assert entry == 100.0
    # ... then the overlay's Entry Reference (last close) ...
    entry, _, _ = tt.resolve_levels(
        _trader(entry=None), _pm(ovl_entry_ref=90.0),
        _market(current_price=80.0), "long",
    )
    assert entry == 90.0
    # ... then the market report's current price.
    entry, _, _ = tt.resolve_levels(
        _trader(entry=None), _pm(ovl_entry_ref=None),
        _market(current_price=80.0), "long",
    )
    assert entry == 80.0
    # Nothing anywhere -> all three stay None (build_ticket reports missing).
    entry, stop, atr = tt.resolve_levels(_trader(), _pm(), _market(), "long")
    assert (entry, stop, atr) == (None, None, None)


@pytest.mark.unit
def test_resolve_levels_stop_prefers_overlay_over_trader():
    _, stop, _ = tt.resolve_levels(
        _trader(stop=90.0), _pm(ovl_stop=95.0), _market(), "long",
    )
    assert stop == 95.0
    _, stop, _ = tt.resolve_levels(
        _trader(stop=90.0), _pm(ovl_stop=None), _market(), "long",
    )
    assert stop == 90.0


@pytest.mark.unit
def test_resolve_levels_market_atr_wins_over_derivation():
    # The market's own ATR is used verbatim; this entry/stop pair would
    # otherwise derive abs(100-92)/2 = 4.0 — the market value must not be
    # recomputed from the stop.
    _, _, atr = tt.resolve_levels(
        _trader(entry=100.0, stop=92.0), _pm(), _market(atr=7.0), "long",
    )
    assert atr == 7.0


@pytest.mark.unit
def test_resolve_levels_atr_derived_from_stop_when_market_silent():
    _, _, atr = tt.resolve_levels(
        _trader(entry=100.0, stop=92.0), _pm(), _market(), "long",
    )
    assert atr == pytest.approx(abs(100.0 - 92.0) / tt.ATR_STOP_MULT)


@pytest.mark.unit
def test_resolve_levels_stop_constructed_from_atr_per_direction():
    # No stop anywhere -> constructed from ATR: below entry for a long,
    # mirrored above entry for a short.
    entry_l, stop_l, _ = tt.resolve_levels(
        _trader(entry=100.0), _pm(), _market(atr=4.0), "long",
    )
    assert stop_l == pytest.approx(100.0 - tt.ATR_STOP_MULT * 4.0)  # 92
    _, stop_s, _ = tt.resolve_levels(
        _trader(entry=100.0), _pm(), _market(atr=4.0), "short",
    )
    assert stop_s == pytest.approx(100.0 + tt.ATR_STOP_MULT * 4.0)  # 108
    assert entry_l == 100.0


# --------------------------------------------------------------------------- #
# parse_trader
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_parse_trader_action_bullet_normalizes_case():
    assert tt.parse_trader("**Action**: buy")["action"] == "Buy"
    assert tt.parse_trader("**ACTION**: SELL")["action"] == "Sell"


@pytest.mark.unit
def test_parse_trader_final_proposal_fallback():
    md = "Weighing both sides...\n\nFINAL TRANSACTION PROPOSAL: **BUY**\n"
    assert tt.parse_trader(md)["action"] == "Buy"


@pytest.mark.unit
def test_parse_trader_action_bullet_outranks_final_proposal():
    md = "**Action**: Sell\n\nFINAL TRANSACTION PROPOSAL: **BUY**\n"
    assert tt.parse_trader(md)["action"] == "Sell"


@pytest.mark.unit
def test_parse_trader_levels_and_sizing():
    md = (
        "**Action**: Buy\n"
        "**Entry Price**: $64,200\n"
        "**Stop Loss**: 62,100.5\n"
        "**Position Sizing**: 8% of portfolio, scaled by volatility\n"
    )
    out = tt.parse_trader(md)
    assert out["entry"] == pytest.approx(64_200.0)
    assert out["stop"] == pytest.approx(62_100.5)
    assert out["sizing"] == "8% of portfolio, scaled by volatility"


@pytest.mark.unit
def test_parse_trader_empty_markdown_all_none():
    assert tt.parse_trader("") == {
        "action": None, "entry": None, "stop": None, "sizing": None,
    }


# --------------------------------------------------------------------------- #
# parse_pm_decision
# --------------------------------------------------------------------------- #


@pytest.mark.unit
@pytest.mark.parametrize("tier", ["Buy", "Overweight", "Hold", "Underweight", "Sell"])
def test_parse_pm_decision_english_rating(tier):
    out = tt.parse_pm_decision(f"**Rating**: {tier}\n\nThesis.")
    assert out["rating"] == tier
    # The regex is case-insensitive and capitalize() renormalizes.
    assert tt.parse_pm_decision("**Rating**: SELL")["rating"] == "Sell"
    # Bolded value form (``**Rating**: **Buy**``) also parses.
    assert tt.parse_pm_decision("**Rating**: **Buy**")["rating"] == "Buy"


@pytest.mark.unit
@pytest.mark.parametrize(
    ("cn", "en"),
    [
        ("买入", "Buy"),
        ("增持", "Overweight"),
        ("持有", "Hold"),
        ("减持", "Underweight"),
        ("卖出", "Sell"),
    ],
)
def test_parse_pm_decision_chinese_rating(cn, en):
    out = tt.parse_pm_decision(f"**评级**：{cn}\n\n看多逻辑。")
    assert out["rating"] == en


@pytest.mark.unit
def test_parse_pm_decision_price_target_and_overlay_fields():
    md = (
        "**Rating**: Overweight\n"
        "**Price Target**: $70,000\n\n"
        "### Quantitative Risk Overlay\n"
        "- **Action**: enter\n"
        "- **Target Weight**: 8.0% (960)\n"
        "- **Stop Loss**: 62,000.0\n"
        "- **Entry Reference**: 64,000.0\n"
        "- **Drawdown Regime**: caution\n"
    )
    out = tt.parse_pm_decision(md)
    assert out["rating"] == "Overweight"
    assert out["price_target"] == pytest.approx(70_000.0)
    assert out["ovl_action"] == "enter"
    assert out["ovl_target_weight"] == pytest.approx(8.0)
    assert out["ovl_stop"] == pytest.approx(62_000.0)
    assert out["ovl_entry_ref"] == pytest.approx(64_000.0)
    assert out["ovl_drawdown_regime"] == "caution"
    # An overlay WITHOUT perp-only bullets is spot evidence, not perp.
    assert out["ovl_is_perp"] is False


@pytest.mark.unit
def test_parse_pm_decision_perp_discriminator_bullets():
    base = (
        "### Quantitative Risk Overlay\n"
        "- **Action**: enter\n"
        "- **Target Weight**: 8.0% (960)\n"
        "- **Stop Loss**: 62,000.0\n"
        "- **Entry Reference**: 64,000.0\n"
    )
    # Either perp-only bullet marks the report as a perp run...
    with_lev = base + "- **Suggested Leverage**: ≤ 4.0x\n"
    assert tt.parse_pm_decision(with_lev)["ovl_is_perp"] is True
    with_liq = base + "- **Est. Liquidation Price**: 52,000.0\n"
    assert tt.parse_pm_decision(with_liq)["ovl_is_perp"] is True
    both = base + "- **Suggested Leverage**: ≤ 4.0x\n- **Est. Liquidation Price**: 52,000.0\n"
    assert tt.parse_pm_decision(both)["ovl_is_perp"] is True
    # ...while the overlay with neither stays non-perp.
    assert tt.parse_pm_decision(base)["ovl_is_perp"] is False


@pytest.mark.unit
def test_parse_pm_decision_missing_rating_stays_none():
    out = tt.parse_pm_decision("Free-text decision without any rating line.")
    assert out["rating"] is None
    assert out["ovl_is_perp"] is False


# --------------------------------------------------------------------------- #
# build_ticket: end-to-end report trees under tmp_path
# --------------------------------------------------------------------------- #

TRADER_MD = (
    "## Trader Plan\n\n"
    "**Action**: Buy\n\n"
    "**Entry Price**: $64,200\n"
    "**Stop Loss**: $62,100\n"
    "**Position Sizing**: 8% of portfolio\n\n"
    "FINAL TRANSACTION PROPOSAL: **BUY**\n"
)

MARKET_MD = (
    "## BTCUSDT Market Snapshot\n\n"
    "Current Price: $64,000\n"
    "ATR(14): $1,050\n"
    "Support: 62,000\n"
    "Resistance: 66,000\n"
)

#: The overlay as a crypto_perp run renders it (leverage + liquidation bullets).
PERP_OVERLAY = (
    "### Quantitative Risk Overlay\n"
    "- **Action**: enter\n"
    "- **Target Weight**: 8.0% (960)\n"
    "- **Stop Loss**: 62,000.0\n"
    "- **Entry Reference**: 64,000.0\n"
    "- **Drawdown Regime**: normal\n"
    "- **Suggested Leverage**: ≤ 4.0x\n"
    "- **Est. Liquidation Price**: 52,000.0\n"
    "- **Funding (7d)**: +0.021% (longs pay)\n"
)

#: Same overlay on a crypto_spot run: perp-only bullets absent.
SPOT_OVERLAY = (
    "### Quantitative Risk Overlay\n"
    "- **Action**: enter\n"
    "- **Target Weight**: 8.0% (960)\n"
    "- **Stop Loss**: 62,000.0\n"
    "- **Entry Reference**: 64,000.0\n"
    "- **Drawdown Regime**: normal\n"
)


def _write_report(
    root: Path,
    decision_md: str,
    trader_md: str = TRADER_MD,
    market_md: str = MARKET_MD,
    name: str = "BTCUSDT_20260919_120000",
) -> Path:
    """Lay out a minimal report tree shaped like ~/.yialpha/logs/reports/<T>_<ts>."""
    report_dir = root / name
    (report_dir / "3_trading").mkdir(parents=True)
    (report_dir / "5_portfolio").mkdir(parents=True)
    (report_dir / "1_analysts").mkdir(parents=True)
    (report_dir / "3_trading" / "trader.md").write_text(trader_md, encoding="utf-8")
    (report_dir / "5_portfolio" / "decision.md").write_text(decision_md, encoding="utf-8")
    (report_dir / "1_analysts" / "market.md").write_text(market_md, encoding="utf-8")
    return report_dir


@pytest.mark.unit
def test_build_ticket_perp_report_keeps_perp_semantics(tmp_path):
    decision = "**Rating**: Buy\n\nMomentum continuation.\n\n" + PERP_OVERLAY
    report_dir = _write_report(tmp_path, decision)

    t = tt.build_ticket(report_dir, capital=10_000.0, profile="moderate")

    # Ticker is the directory name minus the _YYYYMMDD_HHMMSS suffix.
    assert t["ticker"] == "BTCUSDT"
    # Perp bullets present -> the suffix-derived perp default stands.
    assert t["asset_type"] == "crypto_perp"
    assert t["status"] == "ok"
    assert t["direction"] == "long"
    # Priority chain through real files: trader entry / overlay stop / market ATR.
    assert t["entry"] == pytest.approx(64_200.0)
    assert t["stop"] == pytest.approx(62_000.0)
    assert t["atr"] == pytest.approx(1_050.0)
    assert t["supports"] == [62_000.0]
    assert t["resistances"] == [66_000.0]
    # Fixed-fraction sizing: risk budget = 1.5% of 10k over a 2200-wide stop.
    assert t["risk_dollar"] == pytest.approx(150.0)
    assert t["notional"] == pytest.approx(150.0 / (2_200.0 / 64_200.0))
    # Leverage > 1 is possible on perp: conviction cap (|Buy|=2, moderate) = 10x
    # binds below L_liq (14.6x) and L_vol (18.3x).
    assert t["leverage"] == pytest.approx(10.0)
    # MMR-aware long liquidation at the disclosed assumed-50k bracket
    # (MMR 0.50% + 5bps fee estimate).
    assert t["liquidation_price"] == pytest.approx(
        64_200.0 * (1.0 - 1.0 / 10.0 + 0.005 + 0.0005)
    )
    assert t["take_profits"] == pytest.approx([67_500.0, 70_800.0, 75_200.0])
    # The rendered perp ticket carries the funding note, the liquidation row,
    # and the nearest-resistance reference above entry.
    out = tt.render_ticket(t)
    assert "永续合约" in out
    assert "爆仓价(估)" in out
    assert "结构参考·最近阻力" in out


@pytest.mark.unit
def test_build_ticket_spot_report_flips_to_spot_semantics(tmp_path):
    decision = "**Rating**: Buy\n\nCash-market only view.\n\n" + SPOT_OVERLAY
    report_dir = _write_report(tmp_path, decision)

    t = tt.build_ticket(report_dir, capital=10_000.0, profile="moderate")

    # Overlay rendered but WITHOUT the perp-only bullets -> ironclad spot
    # evidence; the discriminator flips the suffix-derived default.
    assert t["asset_type"] == "crypto_spot"
    assert t["status"] == "ok"
    # Spot: HARD_CEILING caps leverage at 1.0 ...
    assert t["leverage"] == pytest.approx(1.0)
    # ... so liquidation_price() returns None and margin == notional.
    assert t["liquidation_price"] is None
    assert t["margin"] == pytest.approx(t["notional"])
    # Rendered spot advice: no perpetual funding note, no liquidation row.
    out = tt.render_ticket(t)
    assert "(crypto_spot)" in out
    assert "永续合约" not in out
    assert "爆仓价" not in out


@pytest.mark.unit
def test_build_ticket_bare_report_keeps_perp_default(tmp_path):
    # No overlay at all (risk_enabled off / legacy report): the discriminator
    # has NO evidence, so the documented perp default must stand.
    decision = "**Rating**: Buy\n\nNo overlay was rendered for this run.\n"
    report_dir = _write_report(tmp_path, decision)

    t = tt.build_ticket(report_dir, capital=10_000.0, profile="moderate")

    assert t["asset_type"] == "crypto_perp"
    assert t["status"] == "ok"
    assert t["direction"] == "long"
    assert t["drawdown_regime"] is None
    assert t["overlay_target_weight_pct"] is None
    # Without the overlay: trader's own stop (62,100), not an ATR-stop one.
    assert t["entry"] == pytest.approx(64_200.0)
    assert t["stop"] == pytest.approx(62_100.0)


@pytest.mark.unit
def test_build_ticket_hold_rating_is_no_trade(tmp_path):
    decision = "**Rating**: Hold\n\nBalanced two-way flow.\n\n" + PERP_OVERLAY
    report_dir = _write_report(tmp_path, decision)

    t = tt.build_ticket(report_dir, capital=10_000.0, profile="moderate")

    assert t["direction"] == "hold"
    assert t["status"] == "no_trade_hold"
    # The early return carries no sizing block at all.
    assert "leverage" not in t and "notional" not in t
    out = tt.render_ticket(t)
    assert "本笔观望" in out


@pytest.mark.unit
def test_build_ticket_breaker_regime_blocks_entry(tmp_path):
    decision = "**Rating**: Buy\n\nThesis.\n\n" + PERP_OVERLAY.replace(
        "- **Drawdown Regime**: normal", "- **Drawdown Regime**: no_new"
    )
    report_dir = _write_report(tmp_path, decision)

    t = tt.build_ticket(report_dir, capital=10_000.0, profile="moderate")

    # Direction and levels are all fine — the breaker alone refuses entry.
    assert t["direction"] == "long"
    assert t["entry"] == pytest.approx(64_200.0)
    assert t["blocked_by_breaker"] is True
    assert t["status"] == "no_trade_breaker"
    assert "leverage" not in t
    out = tt.render_ticket(t)
    assert "熔断器" in out


@pytest.mark.unit
def test_build_ticket_missing_levels_reports_missing_status(tmp_path):
    decision = "**Rating**: Sell\n\nNo overlay, no numeric levels.\n"
    trader = "## Trader Plan\n\n**Action**: Sell\n\nNarrative only, no numbers.\n"
    market = "Analyst narrative without any parseable price field.\n"
    report_dir = _write_report(tmp_path, decision, trader_md=trader, market_md=market)

    t = tt.build_ticket(report_dir, capital=10_000.0, profile="moderate")

    assert t["direction"] == "short"
    assert t["entry"] is None and t["stop"] is None
    assert t["status"] == "missing_levels"


@pytest.mark.unit
def test_build_ticket_stop_equals_entry_degenerates_instead_of_crashing(tmp_path):
    """stop == entry -> stop_dist = 0 -> notional would divide by zero.

    The render path got an empty-sequence guard in the round-4 sweep; this
    is the compute-path sibling (round 5): a trader Entry Price and an
    overlay Stop Loss pinning the same number must degrade to an honest
    no-ticket status, not raise.
    """
    overlay = PERP_OVERLAY.replace("- **Stop Loss**: 62,000.0",
                                   "- **Stop Loss**: 64,200.0")
    decision = "**Rating**: Buy\n\nThesis.\n\n" + overlay
    report_dir = _write_report(tmp_path, decision)

    t = tt.build_ticket(report_dir, capital=10_000.0, profile="moderate")

    assert t["entry"] == pytest.approx(64_200.0)
    assert t["stop"] == pytest.approx(64_200.0)
    assert t["status"] == "degenerate_levels"
    assert "leverage" not in t and "notional" not in t
    out = tt.render_ticket(t)
    assert "价位退化" in out


@pytest.mark.unit
def test_degenerate_perp_report_stays_perp_not_misread_as_spot(tmp_path):
    """Adversarial-review tightening: a perp overlay renders NO perp bullets
    when close/ATR were unavailable -- exactly the signature the spot
    discriminator keys on. Such a report must NOT flip to crypto_spot
    (a spot ticket for a perp position); the flip requires a HEALTHY
    overlay (entry + stop present, which a healthy perp overlay always
    accompanies with its perp bullets).
    """
    degraded_overlay = (
        "### Quantitative Risk Overlay\n"
        "- **Action**: enter\n"
        "- **Target Weight**: 8.0% (960)\n"
        "- **Warning**: Stop-loss not set: price/ATR data unavailable\n"
        "- **Drawdown Regime**: normal\n"
    )
    decision = "**Rating**: Buy\n\nThesis.\n\n" + degraded_overlay
    trader = "## Trader Plan\n\n**Action**: Buy\n\nNarrative only, no levels.\n"
    market = "Analyst narrative without any parseable price field.\n"
    report_dir = _write_report(
        tmp_path, decision, trader_md=trader, market_md=market
    )

    t = tt.build_ticket(report_dir, capital=10_000.0, profile="moderate")

    assert t["asset_type"] == "crypto_perp"
    # No levels anywhere (degraded overlay, narrative trader) -> the honest
    # no-ticket status, never a spot-semantics ticket.
    assert t["status"] == "missing_levels"
