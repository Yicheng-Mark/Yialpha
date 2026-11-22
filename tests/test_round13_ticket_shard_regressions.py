"""Round-13 closing sweep: discriminating pins for the 8 confirmed fixes.

Each test below was walked against the PRE-FIX behavior and must fail there
(that is what makes it a regression pin, not a smoke test):

* [P2] trade_ticket.build_ticket now tiers the liquidation MMR bracket at the
  REAL ticket notional (the same notional advisory_leverage_cap tiers on),
  discloses the MMR basis on the rendered row (overlay liq_mmr_basis wording),
  and REFUSES to render when the real-tier MMR (+fee buffer) >= 1/L flips the
  formula to the wrong side of entry. Three branches pinned: real tier !=
  assumed tier (large notional), refusal (long + short), first-tier small
  notional.
* [P3] the leverage-source table renders the round-12 L_bracket cap row and a
  count-aware title (四/五个上限) — the caps dict carried L_bracket but the
  table hard-coded four rows.
* [P3] --capital gates reject NaN/Inf (trade_ticket.main and the rank_signals
  appendix section): `capital <= 0` passes NaN (any NaN comparison is False)
  and `if capital` passes NaN because NaN is truthy.
* [P3] rank_signals rejects negative --top (group[:top] tail-slices rows off).
* [P3] --since compares parsed DATES, not strings: a non-zero-padded --since
  killed every row bytewise ("2026-09-18" < "2026-9-1" at char '0'<'9'), and
  unstamped report dirs bypassed the filter entirely.
* [P3] ic_dataset validates the ticker against a filename whitelist before
  building the CSV path (path-hostile tickers used to reach ``Path / stem``).
* [P3] analyze_window validates ticker (whitelist) and dates
  (is_canonical_ymd) before interpolating them into output filenames.

Hermetic: report trees under tmp_path, YIALPHA_RESULTS_DIR redirected,
OHLCV stubbed, graphs faked — no network, no LLM, no ~/.yialpha.
"""

from __future__ import annotations

import importlib.util
import json
import logging
import sys
from pathlib import Path

import pandas as pd
import pytest

_PROJECT = Path(__file__).resolve().parents[1]
_SCRIPTS = _PROJECT / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import rank_signals as rs  # noqa: E402, I001 — must follow the sys.path insert
import trade_ticket as tt  # noqa: E402, I001

from yialpha.backtest.ic_dataset import export_ic_datasets  # noqa: E402
from yialpha.dataflows import stockstats_utils  # noqa: E402


def _load_script(name: str):
    """Import a scripts/ entry point under a private round-13 module name."""
    spec = importlib.util.spec_from_file_location(
        f"round13_{name}", _SCRIPTS / f"{name}.py"
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------- #
# report-tree builders (same shapes the dedicated test files use)
# --------------------------------------------------------------------------- #

def _ticket_report(
    tmp_path: Path,
    *,
    rating: str = "Buy",
    entry: str = "100.0",
    stop: str | None = None,
    atr: str = "1.0",
    name: str = "BTCUSDT_20260919_120000",
) -> Path:
    """Bare report tree (no overlay): the suffix-derived crypto_perp default
    stands and the trader's own levels are the resolved ones — both behaviors
    already pinned by tests/test_trade_ticket.py."""
    if stop is None:
        stop = "98.0" if rating in ("Buy", "Overweight") else "102.0"
    d = tmp_path / name
    (d / "3_trading").mkdir(parents=True)
    (d / "5_portfolio").mkdir(parents=True)
    (d / "1_analysts").mkdir(parents=True)
    (d / "3_trading" / "trader.md").write_text(
        f"**Action**: {rating}\n"
        f"**Entry Price**: ${entry}\n"
        f"**Stop Loss**: {stop}\n",
        encoding="utf-8",
    )
    (d / "5_portfolio" / "decision.md").write_text(
        f"**Rating**: {rating}\n\nThesis.\n", encoding="utf-8"
    )
    (d / "1_analysts" / "market.md").write_text(
        f"Current Price: {entry}\nATR(14): {atr}\n", encoding="utf-8"
    )
    return d


def _rank_report(
    root: Path, ticker: str, stamp: str, *, rating: str = "Buy",
) -> Path:
    """One complete rank_signals report dir (the files analyze_ticker reads)."""
    d = root / f"{ticker}_{stamp}"
    d.mkdir(parents=True)
    (d / "5_portfolio").mkdir()
    (d / "3_trading").mkdir()
    (d / "5_portfolio" / "decision.md").write_text(
        f"**Rating**: {rating}\n\nThesis.\n", encoding="utf-8"
    )
    (d / "3_trading" / "trader.md").write_text(
        f"**Action**: {rating}\n**Entry Price**: $100.0\n**Stop Loss**: $92.0\n",
        encoding="utf-8",
    )
    (d / "complete_report.md").write_text("done\n", encoding="utf-8")
    return d


def _ohlcv(days: int = 40) -> pd.DataFrame:
    dates = pd.date_range("2026-01-01", periods=days, freq="D")
    closes = 100.0 + pd.Series(range(days)) * 0.5
    return pd.DataFrame(
        {
            "Date": dates,
            "Open": closes - 0.2,
            "High": closes + 0.5,
            "Low": closes - 0.5,
            "Close": closes,
            "Volume": [1_000_000] * days,
        }
    )


# --------------------------------------------------------------------------- #
# [P2] liquidation_price tiers at the REAL notional — three branches
# --------------------------------------------------------------------------- #

@pytest.mark.unit
def test_liq_real_notional_tier_differs_from_assumed_tier(tmp_path):
    # capital 1M, moderate (1.5% risk), stop_dist 2% -> notional 750,000 USDT
    # -> DEFAULT bracket tier MMR 0.80% (not the assumed-50k 0.50%).
    # L: min(L_liq 25, L_vol 30, L_conv 10, hard 20) = 10.
    # liq = 100 x (1 - 1/10 + 0.008 + 0.0005) = 90.85. Pre-fix (assumed 50k,
    # MMR 0.50%) this rendered 90.55 — a systematically farther liquidation.
    rd = _ticket_report(tmp_path)
    t = tt.build_ticket(rd, capital=1_000_000.0, profile="moderate")

    assert t["notional"] == pytest.approx(750_000.0)
    assert t["leverage"] == pytest.approx(10.0)
    assert t["liquidation_mmr"] == pytest.approx(0.008)
    assert t["liquidation_price"] == pytest.approx(90.85)
    assert t["liquidation_price"] != pytest.approx(100.0 * 0.9055)
    assert t["liquidation_mmr_basis"] == (
        "default bracket ladder @ notional 750,000 USDT"
    )
    # The rendered row discloses the MMR source (overlay liq_mmr_basis wording).
    out = tt.render_ticket(t)
    assert "MMR 0.80%" in out
    assert "750,000 USDT" in out


@pytest.mark.unit
def test_liq_small_notional_uses_first_tier(tmp_path):
    # capital 10k -> notional 7,500 USDT (< 50k first-tier floor) -> MMR
    # 0.40%: liq = 100 x (1 - 0.1 + 0.004 + 0.0005) = 90.45. Pre-fix this was
    # the assumed-50k 0.50% -> 90.55 — the same semantic move the updated pin
    # in test_trade_ticket.py documents.
    rd = _ticket_report(tmp_path)
    t = tt.build_ticket(rd, capital=10_000.0, profile="moderate")

    assert t["notional"] == pytest.approx(7_500.0)
    assert t["liquidation_mmr"] == pytest.approx(0.004)
    assert t["liquidation_price"] == pytest.approx(90.45)
    out = tt.render_ticket(t)
    assert "MMR 0.40%" in out
    assert "notional 7,500 USDT" in out


@pytest.mark.unit
def test_liq_mmr_inversion_refuses_render_long(tmp_path):
    # capital 20M aggressive (2.5% risk), stop_dist 2% -> notional 25,000,000
    # USDT -> MMR 10%; L = min(25, 30, L_conv 15, hard 20) = 15, so
    # MMR + fee (0.1005) >= 1/L (0.0667): liq flips to 103.38 — ABOVE the
    # long entry. The inverted number must never be rendered; the row states
    # the refusal instead. Pre-fix (assumed-50k MMR 0.50%) this printed a
    # plausible-looking 93.88 liquidation for a position whose real tier had
    # already inverted the formula.
    rd = _ticket_report(tmp_path, rating="Buy")
    t = tt.build_ticket(rd, capital=20_000_000.0, profile="aggressive")

    assert t["notional"] == pytest.approx(25_000_000.0)
    assert t["leverage"] == pytest.approx(15.0)
    assert t["liquidation_price"] is None
    assert t["liquidation_dist_pct"] is None
    reason = t["liquidation_refusal_reason"]
    assert reason and "25,000,000" in reason and "倒挂" in reason
    out = tt.render_ticket(t)
    assert "拒绝渲染" in out
    assert "103.3" not in out  # the inverted value never reaches the page


@pytest.mark.unit
def test_liq_mmr_inversion_refuses_render_short(tmp_path):
    # Mirrored short: liq = 100 x (1 + 1/15 - 0.1 - 0.0005) = 96.62 — BELOW
    # the short entry (wrong side), same refusal. Pre-fix, the assumed-50k
    # MMR printed a plausible 106.12 (correct side, wrong tier) for a position
    # whose real tier had already inverted the formula.
    rd = _ticket_report(tmp_path, rating="Sell", stop="102.0")
    t = tt.build_ticket(rd, capital=20_000_000.0, profile="aggressive")

    assert t["direction"] == "short"
    assert t["liquidation_price"] is None
    assert t["liquidation_refusal_reason"]
    out = tt.render_ticket(t)
    assert "拒绝渲染" in out
    assert "96.6" not in out  # the inverted value never reaches the page


# --------------------------------------------------------------------------- #
# [P3] leverage-source table renders the round-12 L_bracket cap
# --------------------------------------------------------------------------- #

@pytest.mark.unit
def test_leverage_table_renders_bracket_cap_row(tmp_path, monkeypatch):
    # caps["L_bracket"] existed since round-12 (compute_leverage adds it when
    # the caller passes a bracket cap) but the rendered table hard-coded four
    # rows — a bracket-bound ticket never showed the cap that binds it.
    rd = _ticket_report(tmp_path)
    monkeypatch.setattr(tt, "advisory_leverage_cap", lambda ticker, notional: 5.0)
    t = tt.build_ticket(rd, capital=1_000_000.0, profile="moderate")

    assert t["leverage_caps"]["L_bracket"] == pytest.approx(5.0)
    assert t["leverage"] == pytest.approx(5.0)
    out = tt.render_ticket(t)
    assert "L_bracket" in out
    assert "initialLeverage" in out
    assert "取五个上限的最小值" in out
    assert "见下方五上限" in out

    # Without a bracket cap the four-row table (and wording) is unchanged.
    monkeypatch.setattr(tt, "advisory_leverage_cap", lambda ticker, notional: None)
    t4 = tt.build_ticket(rd, capital=1_000_000.0, profile="moderate")
    out4 = tt.render_ticket(t4)
    assert "L_bracket" not in out4
    assert "取四个上限的最小值" in out4
    assert "见下方四上限" in out4


# --------------------------------------------------------------------------- #
# [P3] trade_ticket --capital gate: NaN/Inf must not pass
# --------------------------------------------------------------------------- #

@pytest.mark.unit
@pytest.mark.parametrize("bad", ["nan", "inf", "-100", "0"])
def test_ticket_capital_gate_rejects_nan_inf_nonpositive(tmp_path, monkeypatch, capsys, bad):
    # `capital <= 0` passes NaN (every NaN comparison is False) and `> 0`
    # passes inf — either flowed straight into the sizing formulas. The gate
    # must reject both before any ticket math runs.
    rd = _ticket_report(tmp_path)
    monkeypatch.setattr(
        sys, "argv", ["trade_ticket.py", "--report-dir", str(rd), "--capital", bad]
    )
    with pytest.raises(SystemExit) as exc:
        tt.main()
    assert exc.value.code == 2
    assert "资金必须" in capsys.readouterr().err


# --------------------------------------------------------------------------- #
# [P3] rank_signals: --capital gate on the appendix ticket section
# --------------------------------------------------------------------------- #

@pytest.mark.unit
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -100.0, 0.0])
def test_rank_signals_capital_gate_skips_ticket_section(tmp_path, monkeypatch, capsys, bad):
    # render_rank's `if capital and ...` lets NaN through (NaN is truthy) and
    # negative capital built a negative-notional ticket. The gate must skip
    # the appendix with a stderr note while the ranking itself still renders.
    root = tmp_path / "reports"
    root.mkdir()
    monkeypatch.setenv("YIALPHA_RESULTS_DIR", str(tmp_path))
    _rank_report(root, "AAA", "20260918_120000")

    rows = [rs.analyze_ticker("AAA")]
    out = rs.render_rank(rows, ["AAA"], top=10, capital=bad)
    err = capsys.readouterr().err
    assert "--capital" in err and "非法" in err
    assert "# 附：" not in out  # no half-fabricated ticket appendix
    assert "AAA" in out        # the ranking table itself survived


# --------------------------------------------------------------------------- #
# [P3] rank_signals: negative --top is a usage error
# --------------------------------------------------------------------------- #

@pytest.mark.unit
def test_rank_signals_negative_top_is_parser_error(tmp_path, monkeypatch, capsys):
    # group[:top] with a negative top tail-slices ranked rows off silently.
    root = tmp_path / "reports"
    root.mkdir()
    monkeypatch.setenv("YIALPHA_RESULTS_DIR", str(tmp_path))
    _rank_report(root, "AAA", "20260918_120000")

    monkeypatch.setattr(
        sys, "argv", ["rank_signals.py", "--tickers", "AAA", "--top", "-3"]
    )
    with pytest.raises(SystemExit) as exc:
        rs.main()
    assert exc.value.code == 2  # argparse error, not a silently-truncated table
    assert "--top" in capsys.readouterr().err


# --------------------------------------------------------------------------- #
# [P3] rank_signals: --since compares dates, not strings
# --------------------------------------------------------------------------- #

@pytest.mark.unit
def test_since_nonpadded_date_no_longer_kills_every_row(tmp_path, monkeypatch, capsys):
    # "2026-09-18" < "2026-9-1" holds BYTewise (char 5: '0' < '9'), so a
    # non-zero-padded --since filtered out every report. Parsed as dates,
    # Sept 18 >= Sept 1 keeps the row.
    root = tmp_path / "reports"
    root.mkdir()
    monkeypatch.setenv("YIALPHA_RESULTS_DIR", str(tmp_path))
    _rank_report(root, "AAA", "20260918_120000")

    def _rows(since: str) -> list:
        monkeypatch.setattr(
            sys, "argv", ["rank_signals.py", "--tickers", "AAA", "--json",
                          "--since", since]
        )
        rs.main()
        return json.loads(capsys.readouterr().out)

    assert len(_rows("2026-9-1")) == 1   # the pin: pre-fix this returned []
    assert len(_rows("2026-09-01")) == 1  # padded control, same semantics
    assert _rows("2026-09-19") == []      # the filter itself still filters


@pytest.mark.unit
def test_since_excludes_undated_report_with_stderr_note(tmp_path, monkeypatch, capsys):
    # A report dir without the _YYYYMMDD_HHMMSS stamp used to bypass the
    # --since filter entirely (undated = unconditionally included). A report
    # whose date cannot be determined must be excluded and said so on stderr.
    root = tmp_path / "reports"
    root.mkdir()
    monkeypatch.setenv("YIALPHA_RESULTS_DIR", str(tmp_path))
    _rank_report(root, "AAA", "manual_copy")  # stamp-less but complete

    monkeypatch.setattr(
        sys, "argv", ["rank_signals.py", "--tickers", "AAA", "--json",
                      "--since", "2026-01-01"]
    )
    rs.main()
    captured = capsys.readouterr()
    assert json.loads(captured.out) == []
    assert "无法判定" in captured.err
    assert "AAA" in captured.err


@pytest.mark.unit
def test_since_malformed_is_parser_error(tmp_path, monkeypatch):
    monkeypatch.setenv("YIALPHA_RESULTS_DIR", str(tmp_path))
    monkeypatch.setattr(
        sys, "argv", ["rank_signals.py", "--tickers", "AAA", "--since", "not-a-date"]
    )
    with pytest.raises(SystemExit) as exc:
        rs.main()
    assert exc.value.code == 2


# --------------------------------------------------------------------------- #
# [P3] ic_dataset: ticker whitelist for the CSV filename
# --------------------------------------------------------------------------- #

@pytest.fixture()
def patched_ohlcv(monkeypatch):
    monkeypatch.setattr(stockstats_utils, "load_ohlcv", lambda t, d: _ohlcv())


@pytest.mark.unit
def test_ic_dataset_path_hostile_ticker_skipped_not_crashed(tmp_path, monkeypatch, caplog):
    # "a/b" used to flow into out_dir / "a/b_5d.csv": a path separator in the
    # stem writing outside the output dir (or an uncontrolled FileNotFoundError
    # from to_csv). The whitelist rejects it inside the per-ticker try, so the
    # batch skips it with an error naming the ticker.
    monkeypatch.setattr(stockstats_utils, "load_ohlcv", lambda t, d: _ohlcv())
    with caplog.at_level(logging.ERROR, logger="yialpha.backtest.ic_dataset"):
        written = export_ic_datasets(
            ["a/b"], horizon=5, indicators=["rsi"], as_of="2026-06-10",
            output_dir=str(tmp_path),
        )
    assert written == {}
    joined = " | ".join(r.message for r in caplog.records)
    assert "export failed for a/b" in joined
    assert "whitelist" in joined
    assert list(tmp_path.rglob("*.csv")) == []


@pytest.mark.unit
def test_ic_dataset_dot_ticker_still_exports(tmp_path, monkeypatch):
    # The whitelist must not reject legitimate dotted tickers: 600519.SS
    # becomes 600519_SS_<h>d.csv exactly as before.
    monkeypatch.setattr(stockstats_utils, "load_ohlcv", lambda t, d: _ohlcv())
    written = export_ic_datasets(
        ["600519.SS"], horizon=1, indicators=["rsi"], as_of="2026-06-10",
        output_dir=str(tmp_path),
    )
    assert written == {"600519.SS": tmp_path / "600519_SS_1d.csv"}
    assert written["600519.SS"].exists()


@pytest.mark.unit
def test_export_cli_bad_ticker_exits_one(tmp_path, monkeypatch, patched_ohlcv, capsys):
    # The script's per-ticker except converts the ValueError into skip +
    # exit 1 (len(written) < len(unique_tickers)) — fail-closed, no traceback.
    exporter = _load_script("export_ic_dataset")
    rc = exporter.main(["a/b", "--horizon", "5", "--output-dir", str(tmp_path)])
    assert rc == 1
    assert "next:" not in capsys.readouterr().out
    assert list(tmp_path.rglob("*.csv")) == []


# --------------------------------------------------------------------------- #
# [P3] analyze_window: filename-interpolated ticker + dates
# --------------------------------------------------------------------------- #

class _FakeGraph:
    """Stands in for YiAlphaGraph so the valid-arg cases never build an LLM
    graph — they only prove the validation gate let them through."""

    def __init__(self, debug=False, config=None):
        pass

    def propagate(self, ticker, date, portfolio_state=None):
        return {"final_trade_decision": "ok"}, "Buy"


@pytest.mark.unit
def test_analyze_window_rejects_bad_ticker_and_dates(tmp_path, monkeypatch, capsys):
    window = _load_script("analyze_window")

    monkeypatch.setattr(
        sys, "argv", ["analyze_window.py", "--ticker", "win/evil"]
    )
    assert window.main() == 2
    err = capsys.readouterr().err
    assert "--ticker" in err and "win/evil" in err

    monkeypatch.setattr(
        sys, "argv",
        ["analyze_window.py", "--ticker", "NVDA",
         "--dates", "2026-05-14", "2026-5-29"],
    )
    assert window.main() == 2
    assert "YYYY-MM-DD" in capsys.readouterr().err

    # Valid args pass the gate; the faked graph runs the (single-day) window
    # and the dump lands exactly where the whitelist says it may.
    monkeypatch.setattr(window, "YiAlphaGraph", _FakeGraph)
    monkeypatch.setattr(
        sys, "argv",
        ["analyze_window.py", "--ticker", "NVDA", "--dates", "2026-05-14",
         "--out", str(tmp_path)],
    )
    assert window.main() == 0
    assert (tmp_path / "window_NVDA.json").exists()
