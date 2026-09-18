"""tests for scripts/rank_signals.py — multi-ticker conviction ranking.

rank_signals scans YiAlpha report dirs and groups tickers by direction with a
composite conviction score (rating 55% + sentiment 25% + risk-reward 20%,
renormalized over the legs that parsed). These tests pin:

* the hand-computable 55/25/20 weighting (full components and the
  renormalization when sentiment / risk-reward legs are missing);
* direction grouping (long / short / hold) and strongest-long / strongest-short
  identification, including the blocked-regime (#1-candidate) skip;
* mixed asset types ranked in one table, with exponent-notation overlay
  prices ("1.14e-05", the %.6g render of micro-price contracts) parsed at the
  right magnitude — not truncated at the "e";
* hold / no-signal reports landing in the hold group, and missing /
  incomplete report dirs degrading to the honest no_report status.

All fixtures are hermetic: report trees are built under tmp_path and the
reports root is redirected via YIALPHA_RESULTS_DIR (the env seam
trade_ticket._results_dir reads at call time).
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import pytest

# scripts/ is not a package; rank_signals itself injects it for trade_ticket,
# and the test does the same before the first scripts import.
_SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import rank_signals as rs  # noqa: E402, I001 — must follow the sys.path insert


# ---------------------------------------------------------------------------
# fixtures / report-tree builders
# ---------------------------------------------------------------------------

@pytest.fixture()
def reports_root(tmp_path, monkeypatch):
    """Redirect the reports root to tmp_path/reports (hermetic)."""
    root = tmp_path / "reports"
    root.mkdir()
    monkeypatch.setenv("YIALPHA_RESULTS_DIR", str(tmp_path))
    return root


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def make_report(
    root: Path,
    ticker: str,
    stamp: str = "20260918_120000",
    *,
    rating: str | None = None,
    cn_rating: str | None = None,
    pm_target: str | None = None,
    ovl_stop: str | None = None,
    ovl_regime: str | None = None,
    action: str | None = None,
    entry: str | None = None,
    stop: str | None = None,
    sentiment: str | None = None,
    market_price: str | None = None,
    complete: bool = True,
    complete_mtime: float | None = None,
) -> Path:
    """Build one fake report dir with exactly the files/fields the script reads.

    ``pm_target`` / ``entry`` / ``stop`` / ``ovl_stop`` are RAW strings so a
    test can feed "$120.00", "100.00" or "1.14e-05" verbatim.
    """
    d = root / f"{ticker}_{stamp}"
    decision = ["# Portfolio Decision"]
    if rating:
        decision.append(f"**Rating**: {rating}")
    if cn_rating:
        decision.append(f"**评级**：{cn_rating}")
    if pm_target is not None:
        decision.append(f"**Price Target**: {pm_target}")
    if ovl_stop is not None:
        # The perp risk overlay renders prices with %.6g — micro-price
        # contracts come out as "1.14e-05" in this exact bullet shape.
        decision.append(f"- **Stop Loss**: {ovl_stop}")
    if ovl_regime:
        decision.append(f"- **Drawdown Regime**: {ovl_regime}")
    _write(d / "5_portfolio" / "decision.md", "\n".join(decision) + "\n")

    trader = []
    if action:
        trader.append(f"**Action**: {action}")
    if entry is not None:
        trader.append(f"**Entry Price**: {entry}")
    if stop is not None:
        trader.append(f"**Stop Loss**: {stop}")
    _write(d / "3_trading" / "trader.md", "\n".join(trader) + "\n")

    _write(
        d / "1_analysts" / "market.md",
        f"Current Price: {market_price}\n" if market_price else "",
    )
    _write(
        d / "1_analysts" / "sentiment.md",
        f"**Overall Sentiment:** **Bullish** (Score: {sentiment}/10)\n"
        if sentiment is not None
        else "",
    )
    if complete:
        cr = d / "complete_report.md"
        cr.write_text("done\n", encoding="utf-8")
        if complete_mtime is not None:
            os.utime(cr, (complete_mtime, complete_mtime))
    return d


def _group_tickers(rendered: str, title: str) -> list[str]:
    """Tickers of one group table's data rows, in ranked order."""
    lines = rendered.splitlines()
    start = next(i for i, ln in enumerate(lines) if title in ln)
    end = next(
        (i for i, ln in enumerate(lines[start + 1:], start + 1)
         if ln.startswith("## ")),
        len(lines),
    )
    return [
        m.group(1)
        for ln in lines[start:end]
        if (m := re.match(r"^\| \d+ \| `([^`]+)`", ln))
    ]


# ---------------------------------------------------------------------------
# conviction() — the 55/25/20 composite, hand-computed
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_conviction_full_weighting_all_three_legs():
    # Buy=0.90, sent 8.2/10 -> 0.82, rr 2.5 capped at /5 -> 0.50:
    # 0.55*0.90 + 0.25*0.82 + 0.20*0.50 = 0.80 -> 80/100.
    score, comps = rs.conviction("long", "Buy", 8.2, 2.5)
    assert score == 80
    assert comps == pytest.approx({"rating": 0.90, "sent": 0.82, "rr": 0.50})


@pytest.mark.unit
def test_conviction_renormalizes_missing_legs():
    # rr missing: 0.55+0.25 weights only ->
    # (0.55*0.72 + 0.25*0.94) / 0.80 = 78.875 -> 79. This is how an
    # Overweight + strong sentiment, no-target report can outrank a Buy.
    score, comps = rs.conviction("long", "Overweight", 9.4, None)
    assert score == 79
    assert comps["rr"] is None
    # rating only: 0.55*0.90 / 0.55 = 90.
    assert rs.conviction("long", "Buy", None, None)[0] == 90
    # sentiment only missing leg on the low side: (0.495 + 0.12)/0.8 = 76.875.
    assert rs.conviction("long", "Buy", 4.8, None)[0] == 77


@pytest.mark.unit
def test_conviction_short_mirror_clamps_and_unknown_rating():
    # Short mirrors sentiment: score 1/10 is maximally bearish -> 0.90.
    score, comps = rs.conviction("short", "Sell", 1.0, None)
    assert score == 90
    assert comps["sent"] == pytest.approx(0.90)
    # Full short composite: Sell 2.4/10 -> 0.76, rr 1.4 -> 0.28:
    # 0.55*0.90 + 0.25*0.76 + 0.20*0.28 = 0.741 -> 74.
    score2, comps2 = rs.conviction("short", "Sell", 2.4, 1.4)
    assert score2 == 74
    assert comps2["sent"] == pytest.approx(0.76)
    assert comps2["rr"] == pytest.approx(0.28)
    # rr is capped at RR_CAP: 8/5 clips to 1.0.
    assert rs.conviction("long", "Buy", 8.0, 8.0)[1]["rr"] == pytest.approx(1.0)
    # sentiment clamps into [0, 1] even if a report overflows the 0-10 scale.
    assert rs.conviction("long", "Buy", 12.0, None)[1]["sent"] == pytest.approx(1.0)
    # A rating outside the direction's map (e.g. Hold) yields no score at all.
    assert rs.conviction("hold", "Hold", 5.0, 2.0) == (None, {})


# ---------------------------------------------------------------------------
# small parsers
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_parse_sentiment_score_english_chinese_and_garbage():
    en = "**Overall Sentiment:** **Bullish** (Score: 8.2/10)\n"
    assert rs.parse_sentiment_score(en) == pytest.approx(8.2)
    cn = "**整体情绪**：看涨（评分 7.5/10）\n"
    assert rs.parse_sentiment_score(cn) == pytest.approx(7.5)
    assert rs.parse_sentiment_score("nothing numeric here") is None
    assert rs.parse_sentiment_score("") is None


@pytest.mark.unit
def test_report_date_from_dir_stamp():
    assert rs.report_date(Path("AAA_20260918_120000")) == "09-18"
    # A dir without the _YYYYMMDD_HHMMSS suffix degrades to None (rendered "—").
    assert rs.report_date(Path("AAA_manual_copy")) is None


# ---------------------------------------------------------------------------
# analyze_ticker / render_rank — end to end over a fake report tree
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_latest_complete_report_wins_by_mtime(reports_root):
    # Oldest complete report says Buy...
    make_report(reports_root, "MT", "20260917_090000", rating="Buy",
                complete_mtime=1_000_000_000)
    # ...newest complete says Sell...
    make_report(reports_root, "MT", "20260918_090000", rating="Sell",
                complete_mtime=2_000_000_000)
    # ...and an even newer but INCOMPLETE run (no complete_report.md) must be
    # skipped: latest_for_ticker only ranks dirs whose run finished.
    make_report(reports_root, "MT", "20260919_090000", rating="Hold",
                complete=False)
    row = rs.analyze_ticker("MT")
    assert row["status"] == "ok"
    assert row["rating"] == "Sell"
    assert row["report_dir"].endswith("MT_20260918_090000")
    assert row["report_date"] == "09-18"


@pytest.mark.unit
def test_ranking_order_matches_hand_computed_scores(reports_root):
    # Three longs with hand-computed scores 80 / 79 / 77 (see conviction
    # tests) and two shorts 90 / 74.
    make_report(reports_root, "AAA", rating="Buy", pm_target="120.00",
                entry="100.00", stop="92.00", sentiment=8.2, market_price="100.00")
    make_report(reports_root, "BBB", rating="Overweight",
                entry="100.00", stop="90.00", sentiment=9.4)
    make_report(reports_root, "CCC", rating="Buy",
                entry="50.00", stop="48.00", sentiment=4.8)
    make_report(reports_root, "SHRT", rating="Sell", pm_target="43.00",
                entry="50.00", stop="55.00", sentiment=2.4)
    make_report(reports_root, "DWSH", rating="Sell",
                entry="30.00", stop="32.00", sentiment=1.0)

    tickers = ["AAA", "BBB", "CCC", "SHRT", "DWSH"]
    rows = {t: rs.analyze_ticker(t) for t in tickers}
    scores = {t: r["conviction"] for t, r in rows.items()}
    assert scores == {"AAA": 80, "BBB": 79, "CCC": 77, "DWSH": 90, "SHRT": 74}
    # The risk-reward leg really came from entry/stop/target arithmetic.
    assert rows["AAA"]["rr"] == pytest.approx(2.5)
    assert rows["SHRT"]["rr"] == pytest.approx(1.4)
    assert rows["BBB"]["rr"] is None  # no PM target -> leg absent, renormalized

    rendered = rs.render_rank(list(rows.values()), tickers, top=10)
    assert _group_tickers(rendered, "做多排名") == ["AAA", "BBB", "CCC"]
    assert _group_tickers(rendered, "做空排名") == ["DWSH", "SHRT"]
    assert "做多排名（3）" in rendered
    assert "做空排名（2）" in rendered
    # No hold reports -> no hold section.
    assert "不建议进场" not in rendered

    # Strongest-long / strongest-short stars sit on the #1 rows.
    lines = rendered.splitlines()
    aaa = next(ln for ln in lines if ln.startswith("| 1 | `AAA`"))
    dwsh = next(ln for ln in lines if ln.startswith("| 1 | `DWSH`"))
    assert "最强多头" in aaa
    assert "最强空头" in dwsh
    # The detail table spells out the component scores for the two winners:
    # AAA = rating 0.90 / sent 0.82 / rr 0.50 -> weighted 80.
    detail = next(ln for ln in lines if ln.startswith("| `AAA` |"))
    assert "0.90 | 0.82 | 0.50 | **80**" in detail


@pytest.mark.unit
def test_direction_groups_hold_no_signal_and_missing(reports_root):
    make_report(reports_root, "LONGLIVE", rating="Buy", pm_target="120.00",
                entry="100.00", stop="92.00", sentiment=8.2)
    # Chinese rating 卖出 must parse to Sell -> short (the regression the
    # script's docstring calls out: framework parse_rating misread it as Hold).
    make_report(reports_root, "CNSELL", cn_rating="卖出",
                entry="30.00", stop="32.00")
    make_report(reports_root, "HOLD1", rating="Hold")
    # A completely empty report (no rating, no action) is a no-signal hold.
    make_report(reports_root, "EMPTY1")

    tickers = ["LONGLIVE", "CNSELL", "HOLD1", "EMPTY1", "GHOST"]
    rows = {t: rs.analyze_ticker(t) for t in tickers}

    assert rows["LONGLIVE"]["direction"] == "long"
    assert rows["CNSELL"]["direction"] == "short"
    assert rows["CNSELL"]["rating"] == "Sell"
    assert rows["HOLD1"]["direction"] == "hold"
    assert rows["HOLD1"]["conviction"] is None
    assert rows["EMPTY1"]["direction"] == "hold"
    assert rows["EMPTY1"]["conviction"] is None
    # No report dirs at all -> honest no_report status, not a crash.
    assert rows["GHOST"] == {"ticker": "GHOST", "status": "no_report"}

    rendered = rs.render_rank(list(rows.values()), tickers, top=10)
    assert _group_tickers(rendered, "做多排名") == ["LONGLIVE"]
    assert _group_tickers(rendered, "做空排名") == ["CNSELL"]
    # Both hold flavors land in the hold group's comma line.
    assert "观望（2，不建议进场）" in rendered
    hold_line = next(
        ln for ln in rendered.splitlines() if "`HOLD1`" in ln and "`EMPTY1`" in ln
    )
    assert hold_line.startswith("`HOLD1`")
    # ...and the missing ticker is disclosed, not silently dropped.
    assert "无报告" in rendered
    assert "`GHOST`" in rendered


@pytest.mark.unit
def test_blocked_regime_is_ranked_but_never_the_top_pick(reports_root):
    # BLOCKED scores higher (84) than FREE (80) but carries Drawdown Regime
    # no_new -> the #1-candidate picker must skip it.
    make_report(reports_root, "BLOCKED", rating="Buy", pm_target="120.00",
                entry="100.00", stop="92.00", sentiment=9.8,
                ovl_regime="no_new")
    make_report(reports_root, "FREE", rating="Buy", pm_target="120.00",
                entry="100.00", stop="92.00", sentiment=8.2)

    rows = {t: rs.analyze_ticker(t) for t in ("BLOCKED", "FREE")}
    assert rows["BLOCKED"]["conviction"] == 84
    assert rows["BLOCKED"]["blocked"] is True
    assert rows["FREE"]["blocked"] is False

    rendered = rs.render_rank(list(rows.values()), ["BLOCKED", "FREE"], top=10)
    # Ranked order is by score: BLOCKED is still listed first...
    assert _group_tickers(rendered, "做多排名") == ["BLOCKED", "FREE"]
    lines = rendered.splitlines()
    blocked_row = next(ln for ln in lines if ln.startswith("| 1 | `BLOCKED`"))
    free_row = next(ln for ln in lines if ln.startswith("| 2 | `FREE`"))
    assert "熔断" in blocked_row
    assert "最强多头" not in blocked_row
    assert "最强多头" in free_row
    # The handoff section must recommend the unblocked one.
    assert "最强多头 `FREE`" in rendered


@pytest.mark.unit
def test_mixed_asset_types_rank_comparably_and_exponent_prices_parse(reports_root):
    # PEPEUSDT-class micro-price contract: the overlay's %.6g render puts the
    # stop in decision.md as "1.14e-05"; entry/target ride along in the same
    # notation. rr = |1.5e-5 - 1.2e-5| / |1.2e-5 - 1.14e-5| = 5.0 (c_rr 1.0):
    # 0.55*0.90 + 0.20*1.0 over 0.75 = 92.67 -> 93.
    make_report(reports_root, "PEPEUSDT", rating="Buy", pm_target="1.5e-05",
                ovl_stop="1.14e-05", entry="1.2e-05")
    # A plain US stock next to it (score 80, see the weighting test).
    make_report(reports_root, "AAPL", rating="Buy", pm_target="120.00",
                entry="100.00", stop="92.00", sentiment=8.2)

    pepe = rs.analyze_ticker("PEPEUSDT")
    aapl = rs.analyze_ticker("AAPL")
    assert pepe["asset_type"] == "crypto_perp"
    assert aapl["asset_type"] == "us_stock"
    # The exponent must survive parsing: 1.14e-05, not 1.14 (a truncation at
    # the "e" would leave the absolute levels five orders of magnitude off —
    # the rr ratio alone is scale-invariant and cannot catch it).
    assert pepe["entry"] == pytest.approx(1.2e-05)
    assert pepe["stop"] == pytest.approx(1.14e-05)
    assert pepe["rr"] == pytest.approx(5.0)
    assert pepe["conviction"] == 93
    assert aapl["conviction"] == 80

    rendered = rs.render_rank([pepe, aapl], ["PEPEUSDT", "AAPL"], top=10)
    assert _group_tickers(rendered, "做多排名") == ["PEPEUSDT", "AAPL"]
    # Both asset flavors share the same ranked table.
    assert "加密永续" in rendered
    assert "美股" in rendered


@pytest.mark.unit
def test_missing_and_incomplete_reports_degrade_honestly(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("YIALPHA_RESULTS_DIR", str(tmp_path))
    # Root with one INCOMPLETE dir only (run crashed before complete_report).
    root = tmp_path / "reports"
    root.mkdir()
    make_report(root, "HALFWAY", complete=False)
    assert rs.analyze_ticker("HALFWAY") == {
        "ticker": "HALFWAY", "status": "no_report",
    }
    # Missing reports root entirely: same honest status, no exception.
    monkeypatch.setenv("YIALPHA_RESULTS_DIR", str(tmp_path / "nowhere"))
    assert rs.analyze_ticker("ANY")["status"] == "no_report"
    # And the CLI's discovery mode exits 2 when the root does not exist.
    monkeypatch.setattr(sys, "argv", ["rank_signals.py"])
    with pytest.raises(SystemExit) as exc:
        rs.main()
    assert exc.value.code == 2
    assert "报告根目录不存在" in capsys.readouterr().err


@pytest.mark.unit
def test_main_json_output_and_discovery(reports_root, monkeypatch, capsys):
    make_report(reports_root, "AAA", rating="Buy", pm_target="120.00",
                entry="100.00", stop="92.00", sentiment=8.2)
    make_report(reports_root, "DWSH", rating="Sell",
                entry="30.00", stop="32.00", sentiment=1.0)

    # Explicit --tickers --json: machine-readable rows with the scores.
    monkeypatch.setattr(
        sys, "argv", ["rank_signals.py", "--tickers", "aaa", "DWSH", "--json"]
    )
    rs.main()
    rows = json.loads(capsys.readouterr().out)
    by_ticker = {r["ticker"]: r for r in rows}
    assert set(by_ticker) == {"AAA", "DWSH"}  # bare ticker got upper-cased
    assert by_ticker["AAA"]["conviction"] == 80
    assert by_ticker["DWSH"]["direction"] == "short"

    # No --tickers: discovery lists every ticker that has a complete report.
    monkeypatch.setattr(sys, "argv", ["rank_signals.py", "--json"])
    rs.main()
    discovered = json.loads(capsys.readouterr().out)
    assert {r["ticker"] for r in discovered} == {"AAA", "DWSH"}

    # --since newer than the report date filters the row out entirely.
    monkeypatch.setattr(
        sys, "argv",
        ["rank_signals.py", "--tickers", "AAA", "--json", "--since", "2026-09-19"],
    )
    rs.main()
    assert json.loads(capsys.readouterr().out) == []
