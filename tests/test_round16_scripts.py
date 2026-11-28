"""Round-16 tests for the ``scripts/`` shard: audit regressions + coverage gaps.

Every test is hermetic (no network, no LLM, no real subprocess) and enters
through the public function / CLI entry point of the script under test.
Scripts are not a package, so each is loaded as an isolated module via
``importlib`` — the same convention as tests/test_run_robust_watchdog.py,
tests/test_export_ic_dataset.py and tests/test_run_analyst_parallel_ab.py.
"""

from __future__ import annotations

import importlib.util
import io
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"


# ---------------------------------------------------------------------------
# loaders
# ---------------------------------------------------------------------------


def _load(name: str, filename: str):
    """Load ``scripts/<filename>`` as an isolated module (``__main__`` never fires)."""
    spec = importlib.util.spec_from_file_location(name, SCRIPTS_DIR / filename)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _exec_as_main(monkeypatch, filename: str, argv: list[str]) -> int | str | None:
    """Run ``scripts/<filename>`` with ``__name__ == "__main__"``; return its exit code.

    ``spec_from_file_location`` refuses a ``__main__`` module name, so the
    guard branch is exercised by exec-ing the compiled source in a fresh
    namespace whose ``__name__`` is ``"__main__"``.
    """
    path = SCRIPTS_DIR / filename
    code = compile(path.read_text(encoding="utf-8"), str(path), "exec")
    monkeypatch.setattr(sys, "argv", [filename, *argv])
    namespace = {"__name__": "__main__", "__file__": str(path)}
    with pytest.raises(SystemExit) as excinfo:
        exec(code, namespace)  # noqa: S102 — the script under test IS the code
    return excinfo.value.code


# ===========================================================================
# scripts/_bootstrap.py
# ===========================================================================


class _FakeStream:
    """Stream double recording ``reconfigure`` kwargs; optionally raising."""

    def __init__(self, exc: Exception | None = None) -> None:
        self.calls: list[dict] = []
        self._exc = exc

    def reconfigure(self, **kwargs) -> None:
        self.calls.append(kwargs)
        if self._exc is not None:
            raise self._exc


@pytest.mark.unit
def test_r16_bootstrap_inserts_project_root_at_front_exactly_once(monkeypatch):
    """bootstrap() puts the project root at sys.path[0] and is idempotent."""
    boot = _load("r16_bootstrap", "_bootstrap.py")
    root = str(PROJECT_ROOT)
    without_root = [p for p in sys.path if p != root]
    monkeypatch.setattr(sys, "path", list(without_root))

    boot.bootstrap()
    assert sys.path[0] == root

    boot.bootstrap()
    assert sys.path.count(root) == 1
    assert sys.path[1:] == without_root


@pytest.mark.unit
def test_r16_bootstrap_keeps_existing_root_entry_untouched(monkeypatch):
    """An already-present project root is neither moved nor duplicated."""
    boot = _load("r16_bootstrap", "_bootstrap.py")
    root = str(PROJECT_ROOT)
    existing = ["/some/site-packages", root, "/some/scripts"]
    monkeypatch.setattr(sys, "path", list(existing))

    boot.bootstrap()

    assert sys.path == existing


@pytest.mark.unit
def test_r16_bootstrap_reconfigures_both_streams_utf8_replace(monkeypatch):
    """Both streams are switched to utf-8/replace (Windows GBK console)."""
    boot = _load("r16_bootstrap", "_bootstrap.py")
    out, err = _FakeStream(), _FakeStream()
    monkeypatch.setattr(sys, "stdout", out)
    monkeypatch.setattr(sys, "stderr", err)

    boot.bootstrap()

    assert out.calls == [{"encoding": "utf-8", "errors": "replace"}]
    assert err.calls == [{"encoding": "utf-8", "errors": "replace"}]


@pytest.mark.unit
@pytest.mark.parametrize(
    "broken",
    [
        pytest.param(io.StringIO(), id="no-reconfigure-attr"),
        pytest.param(SimpleNamespace(reconfigure=None), id="reconfigure-not-callable"),
    ],
)
def test_r16_bootstrap_tolerates_stream_without_reconfigure(monkeypatch, broken):
    """Captured pytest output has no ``reconfigure``; bootstrap must not raise."""
    boot = _load("r16_bootstrap", "_bootstrap.py")
    err = _FakeStream()
    monkeypatch.setattr(sys, "stdout", broken)
    monkeypatch.setattr(sys, "stderr", err)

    boot.bootstrap()

    assert err.calls == [{"encoding": "utf-8", "errors": "replace"}]


@pytest.mark.unit
@pytest.mark.parametrize("exc", [AttributeError("detached"), ValueError("no encoding")])
def test_r16_bootstrap_stream_error_does_not_block_the_other_stream(monkeypatch, exc):
    """A failing stdout reconfigure is suppressed and stderr is still fixed."""
    boot = _load("r16_bootstrap", "_bootstrap.py")
    out, err = _FakeStream(exc), _FakeStream()
    monkeypatch.setattr(sys, "stdout", out)
    monkeypatch.setattr(sys, "stderr", err)

    boot.bootstrap()

    assert out.calls == [{"encoding": "utf-8", "errors": "replace"}]
    assert err.calls == [{"encoding": "utf-8", "errors": "replace"}]


# ===========================================================================
# scripts/analyze_window.py
# ===========================================================================


@pytest.fixture(scope="module")
def aw():
    return _load("r16_analyze_window", "analyze_window.py")


def _ok_record(date: str, rating: str = "Buy") -> dict:
    return {
        "date": date,
        "rating": rating,
        "price": 100.0,
        "overlay": {"entry": 100.0, "action": "open"},
        "decision_excerpt": "excerpt",
        "full_decision": "full",
    }


def _set_argv(monkeypatch, *argv: str) -> None:
    monkeypatch.setattr(sys, "argv", ["analyze_window.py", *argv])


@pytest.mark.unit
@pytest.mark.parametrize(
    "ticker",
    ["../evil", "NV DA", "NVDA/", "NVDA|x", ""],
)
def test_r16_analyze_window_rejects_unsafe_ticker_before_any_run(aw, tmp_path, monkeypatch, capsys, ticker):
    """A ticker outside [A-Za-z0-9._-] exits 2 before any propagate or mkdir."""
    calls: list[tuple] = []
    monkeypatch.setattr(aw, "run_one", lambda *a, **k: calls.append((a, k)))
    out = tmp_path / "out"
    _set_argv(monkeypatch, "--ticker", ticker, "--out", str(out))

    rc = aw.main()

    assert rc == 2
    assert calls == []
    assert not out.exists()
    assert "--ticker" in capsys.readouterr().err


@pytest.mark.unit
@pytest.mark.parametrize(
    "dates",
    [
        pytest.param(["2026-5-14"], id="not-zero-padded"),
        pytest.param(["2026-05-14", "2026-13-01"], id="impossible-month-after-valid"),
        pytest.param(["2026/05/14"], id="wrong-separator"),
    ],
)
def test_r16_analyze_window_rejects_non_canonical_dates(aw, tmp_path, monkeypatch, capsys, dates):
    """Every --dates entry must be a canonical zero-padded YYYY-MM-DD (exit 2)."""
    calls: list[tuple] = []
    monkeypatch.setattr(aw, "run_one", lambda *a, **k: calls.append((a, k)))
    out = tmp_path / "out"
    _set_argv(monkeypatch, "--ticker", "NVDA", "--out", str(out), "--dates", *dates)

    rc = aw.main()

    assert rc == 2
    assert calls == []
    assert not out.exists()
    assert "YYYY-MM-DD" in capsys.readouterr().err


@pytest.mark.unit
def test_r16_analyze_window_per_day_failure_isolated_incremental_dump_exit_1(
    aw, tmp_path, monkeypatch, capsys
):
    """A failing day becomes an ERROR row, later days still run, rc is 1.

    Also pins the incremental contract: the dump is rewritten after EVERY day,
    so a later day hanging cannot lose the earlier results.
    """
    out = tmp_path / "out"
    dump = out / "window_NVDA.json"
    seen: dict[str, list[str] | None] = {}

    def fake_run_one(ticker, date, equity, risk):
        seen[date] = [r["date"] for r in json.loads(dump.read_text(encoding="utf-8"))] if dump.exists() else None
        if date == "2026-05-29":
            raise RuntimeError("LLM exploded")
        return _ok_record(date)

    monkeypatch.setattr(aw, "run_one", fake_run_one)
    _set_argv(
        monkeypatch, "--ticker", "NVDA", "--out", str(out),
        "--dates", "2026-05-14", "2026-05-29", "2026-06-10",
    )

    rc = aw.main()

    assert rc == 1
    rows = json.loads(dump.read_text(encoding="utf-8"))
    assert [r["date"] for r in rows] == ["2026-05-14", "2026-05-29", "2026-06-10"]
    assert [r["rating"] for r in rows] == ["Buy", "ERROR", "Buy"]
    assert rows[1]["overlay"] == {}
    assert "LLM exploded" in rows[1]["decision_excerpt"]
    # day 1 ran on an empty dump, day 2 saw 1 persisted row, day 3 saw 2.
    assert seen == {
        "2026-05-14": None,
        "2026-05-29": ["2026-05-14"],
        "2026-06-10": ["2026-05-14", "2026-05-29"],
    }
    err = capsys.readouterr().err
    assert "1/3 天失败" in err
    assert "LLM exploded" in err


@pytest.mark.unit
def test_r16_analyze_window_all_days_ok_exit_0_and_dump_content(aw, tmp_path, monkeypatch, capsys):
    """All-good run exits 0 and the dump holds exactly what run_one returned."""
    out = tmp_path / "out"
    monkeypatch.setattr(aw, "run_one", lambda ticker, date, equity, risk: _ok_record(date, "Sell"))
    _set_argv(monkeypatch, "--ticker", "MSFT", "--out", str(out), "--dates", "2026-05-14", "2026-06-10")

    rc = aw.main()

    assert rc == 0
    dump = out / "window_MSFT.json"
    rows = json.loads(dump.read_text(encoding="utf-8"))
    assert rows == [_ok_record("2026-05-14", "Sell"), _ok_record("2026-06-10", "Sell")]
    captured = capsys.readouterr()
    assert "全部完成" in captured.out
    assert captured.err == ""


@pytest.mark.unit
@pytest.mark.parametrize(
    ("flags", "expected_risk"),
    [pytest.param([], True, id="default-on"), pytest.param(["--no-risk"], False, id="no-risk")],
)
def test_r16_analyze_window_forwards_ticker_equity_and_risk_flag(aw, tmp_path, monkeypatch, flags, expected_risk):
    """run_one receives the parsed --equity and the --risk / --no-risk state."""
    aw_module = aw
    got: list[tuple] = []

    def fake_run_one(ticker, date, equity, risk):
        got.append((ticker, date, equity, risk))
        return _ok_record(date)

    monkeypatch.setattr(aw_module, "run_one", fake_run_one)
    _set_argv(
        monkeypatch, "--ticker", "NVDA", "--out", str(tmp_path / "out"),
        "--dates", "2026-05-14", "--equity", "250000", *flags,
    )

    rc = aw_module.main()

    assert rc == 0
    assert got == [("NVDA", "2026-05-14", 250000.0, expected_risk)]


@pytest.mark.unit
def test_r16_analyze_window_main_guard_propagates_exit_code(tmp_path, monkeypatch):
    """``python scripts/analyze_window.py`` exits with main()'s code (2 on bad ticker)."""
    rc = _exec_as_main(
        monkeypatch, "analyze_window.py",
        ["--ticker", "bad ticker", "--out", str(tmp_path / "out")],
    )

    assert rc == 2
    assert not (tmp_path / "out").exists()


# ===========================================================================
# scripts/export_ic_dataset.py
# ===========================================================================


@pytest.fixture(scope="module")
def exporter():
    return _load("r16_export_ic_dataset", "export_ic_dataset.py")


@pytest.fixture()
def fake_export(exporter, monkeypatch):
    """Replace the package exporter with a recorder; ``result`` drives the return."""
    state = SimpleNamespace(calls=[], result={})

    def _export(tickers, **kwargs):
        state.calls.append((list(tickers), kwargs))
        return state.result

    monkeypatch.setattr(exporter, "export_ic_datasets", _export)
    return state


@pytest.mark.unit
@pytest.mark.parametrize("horizon", ["0", "-3"])
def test_r16_export_ic_horizon_must_be_positive(exporter, fake_export, capsys, horizon):
    """--horizon <= 0 is a usage error (exit 2) raised before any export."""
    with pytest.raises(SystemExit) as excinfo:
        exporter.main(["NVDA", "--horizon", horizon, "--output-dir", "ic_data"])

    assert excinfo.value.code == 2
    assert fake_export.calls == []
    assert "--horizon must be a positive number" in capsys.readouterr().err


@pytest.mark.unit
@pytest.mark.parametrize(
    ("extra", "fragment"),
    [
        pytest.param("1,x", "comma-separated integers", id="non-integer"),
        pytest.param("1,-2", "must be positive", id="negative"),
        pytest.param("1,0", "must be positive", id="zero"),
    ],
)
def test_r16_export_ic_extra_horizons_rejected(exporter, fake_export, capsys, extra, fragment):
    """--extra-horizons must be positive integers; anything else is exit 2."""
    with pytest.raises(SystemExit) as excinfo:
        exporter.main(["NVDA", "--extra-horizons", extra, "--output-dir", "ic_data"])

    assert excinfo.value.code == 2
    assert fake_export.calls == []
    assert fragment in capsys.readouterr().err


@pytest.mark.unit
def test_r16_export_ic_unknown_indicator_rejected_before_export(exporter, fake_export, capsys):
    """Unknown indicator names are refused upfront (never a silently empty column)."""
    with pytest.raises(SystemExit) as excinfo:
        exporter.main(["NVDA", "--indicators", "rsi", "not_a_real_indicator", "--output-dir", "ic_data"])

    assert excinfo.value.code == 2
    assert fake_export.calls == []
    assert "unknown indicator name(s) ['not_a_real_indicator']" in capsys.readouterr().err


@pytest.mark.unit
def test_r16_export_ic_forwards_parsed_args_and_all_written_exit_0(exporter, fake_export, capsys, tmp_path):
    """Parsed argv reaches export_ic_datasets verbatim; full export exits 0."""
    csv = tmp_path / "NVDA_5d.csv"
    csv.write_text("date,forward_return,rsi\n2026-01-01,0.01,50\n2026-01-02,-0.02,51\n", encoding="utf-8")
    fake_export.result = {"NVDA": csv}

    rc = exporter.main(
        ["NVDA", "--horizon", "10", "--extra-horizons", "1, 20", "--indicators", "rsi",
         "--as-of", "2026-06-30", "--output-dir", str(tmp_path)],
    )

    assert rc == 0
    assert fake_export.calls == [
        (
            ["NVDA"],
            {
                "horizon": 10,
                "indicators": ["rsi"],
                "as_of": "2026-06-30",
                "output_dir": str(tmp_path),
                "extra_horizons": [1, 20],
            },
        )
    ]
    out = capsys.readouterr().out
    assert f"[NVDA] wrote {csv}: 2 rows" in out
    assert f"next: python scripts/prune_indicators_cli.py {csv} --window 60 --json-out {csv}.prune.json" in out


@pytest.mark.unit
def test_r16_export_ic_defaults_full_battery_and_empty_extra_horizons(exporter, fake_export, tmp_path):
    """No --indicators means the full battery (``None`` passthrough); blank extras -> []."""
    fake_export.result = {"NVDA": tmp_path / "a.csv"}
    (tmp_path / "a.csv").write_text("date,forward_return\n", encoding="utf-8")

    rc = exporter.main(["NVDA", "--output-dir", str(tmp_path), "--as-of", "2026-06-30"])

    assert rc == 0
    _tickers, kwargs = fake_export.calls[0]
    assert kwargs["indicators"] is None
    assert kwargs["extra_horizons"] == []
    assert kwargs["horizon"] == 5


@pytest.mark.unit
def test_r16_export_ic_duplicate_tickers_collapsed_and_not_a_failure(exporter, fake_export, capsys, tmp_path):
    """``NVDA NVDA`` requests one CSV and must count as a full success (rc 0)."""
    csv = tmp_path / "NVDA_5d.csv"
    csv.write_text("date,forward_return\n2026-01-01,0.01\n", encoding="utf-8")
    amd = tmp_path / "AMD_5d.csv"
    amd.write_text("date,forward_return\n2026-01-01,0.02\n", encoding="utf-8")
    fake_export.result = {"NVDA": csv, "AMD": amd}

    rc = exporter.main(["NVDA", "NVDA", "AMD", "AMD", "--output-dir", str(tmp_path), "--as-of", "2026-06-30"])

    assert rc == 0
    assert fake_export.calls[0][0] == ["NVDA", "AMD"]
    assert "duplicate tickers in argv collapsed to ['NVDA', 'AMD']" in capsys.readouterr().err


@pytest.mark.unit
def test_r16_export_ic_partial_export_is_exit_1(exporter, fake_export, capsys, tmp_path):
    """A ticker the package skipped (no data) makes the run a failure (rc 1)."""
    csv = tmp_path / "NVDA_5d.csv"
    csv.write_text("date,forward_return\n2026-01-01,0.01\n", encoding="utf-8")
    fake_export.result = {"NVDA": csv}

    rc = exporter.main(["NVDA", "AMD", "--output-dir", str(tmp_path), "--as-of", "2026-06-30"])

    assert rc == 1
    assert "[NVDA]" in capsys.readouterr().out


@pytest.mark.unit
@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        pytest.param(["NVDA", "--horizon", "0"], 2, id="usage-error"),
        pytest.param([], 2, id="missing-tickers"),
    ],
)
def test_r16_export_ic_main_guard_propagates_exit_code(monkeypatch, argv, expected):
    """``python scripts/export_ic_dataset.py`` exits with main()'s / argparse's code."""
    rc = _exec_as_main(monkeypatch, "export_ic_dataset.py", argv)

    assert rc == expected


# ===========================================================================
# scripts/trade_ticket.py
# ===========================================================================


@pytest.fixture(scope="module")
def tt():
    return _load("r16_trade_ticket", "trade_ticket.py")


@pytest.fixture()
def bracket_cap(tt, monkeypatch):
    """Pin the signed-endpoint advisory cap (``value``); records each lookup."""
    holder = SimpleNamespace(value=None, calls=[])

    def fake(symbol, notional=None):
        holder.calls.append((symbol, notional))
        if isinstance(holder.value, Exception):
            raise holder.value
        return holder.value

    monkeypatch.setattr(tt, "advisory_leverage_cap", fake)
    return holder


@pytest.fixture()
def reports(tmp_path, monkeypatch):
    """Hermetic reports root: <tmp>/reports (trade_ticket reads YIALPHA_RESULTS_DIR)."""
    monkeypatch.setenv("YIALPHA_RESULTS_DIR", str(tmp_path))
    root = tmp_path / "reports"
    root.mkdir()
    return root


def _write_report(
    root,
    name="NVDA_20260610_120000",
    rating="Buy",
    action="Buy",
    entry=100.0,
    stop=95.0,
    atr=2.0,
    *,
    decision_extra="",
    market_extra="",
    overlay="",
    complete=True,
) -> Path:
    """Lay out a minimal report tree shaped like ``reports/<TICKER>_<ts>/``."""
    report = root / name
    for sub in ("1_analysts", "3_trading", "5_portfolio"):
        (report / sub).mkdir(parents=True)
    if action is None:
        trader_md = "## Trader Plan\n"
    else:
        stop_line = "" if stop is None else f"**Stop Loss**: ${stop:g}\n"
        trader_md = (
            f"## Trader Plan\n**Action**: {action}\n"
            f"**Entry Price**: ${entry:g}\n{stop_line}"
            "**Position Sizing**: 10% of portfolio\n"
        )
    market_md = "## Market\n"
    if atr is not None:
        market_md += f"ATR(14): ${atr:g}\n"
    market_md += market_extra
    decision_md = "" if rating is None else f"**Rating**: {rating}\n"
    decision_md += decision_extra + overlay
    (report / "3_trading" / "trader.md").write_text(trader_md, encoding="utf-8")
    (report / "1_analysts" / "market.md").write_text(market_md, encoding="utf-8")
    (report / "5_portfolio" / "decision.md").write_text(decision_md, encoding="utf-8")
    if complete:
        (report / "complete_report.md").write_text("done\n", encoding="utf-8")
    return report


def _set_mtime(path: Path, mtime: float) -> None:
    os.utime(path, (mtime, mtime))


# ---- find_report ----------------------------------------------------------


@pytest.mark.unit
def test_r16_trade_ticket_find_report_missing_explicit_dir_raises(tt, tmp_path):
    """--report-dir pointing at a non-directory is a FileNotFoundError naming it."""
    with pytest.raises(FileNotFoundError, match="报告目录不存在"):
        tt.find_report(None, str(tmp_path / "nope"))


@pytest.mark.unit
def test_r16_trade_ticket_find_report_missing_root_points_at_report_dir(tt, tmp_path, monkeypatch):
    """No reports root at all → error tells the user how to proceed."""
    monkeypatch.setenv("YIALPHA_RESULTS_DIR", str(tmp_path / "absent"))
    with pytest.raises(FileNotFoundError, match="--report-dir"):
        tt.find_report()


@pytest.mark.unit
def test_r16_trade_ticket_find_report_unknown_ticker_names_ticker_and_root(tt, reports):
    """A ticker with no complete report raises (never falls back to another ticker)."""
    _write_report(reports, name="NVDA_20260610_120000")
    with pytest.raises(FileNotFoundError, match=r"ticker=MSFT"):
        tt.find_report("MSFT")


@pytest.mark.unit
def test_r16_trade_ticket_find_report_only_incomplete_runs_refuses(tt, reports, tmp_path):
    """Every run missing complete_report.md → refuse instead of ticketing a crash."""
    _write_report(reports, name="NVDA_20260610_120000", complete=False)
    _write_report(reports, name="NVDA_20260611_120000", complete=False)
    with pytest.raises(FileNotFoundError, match="complete_report.md"):
        tt.find_report()
    with pytest.raises(FileNotFoundError, match="ticker=NVDA"):
        tt.find_report("NVDA")
    # An explicit --report-dir is honoured verbatim, complete or not.
    explicit = tmp_path / "elsewhere"
    explicit.mkdir()
    assert tt.find_report(None, str(explicit)) == explicit


@pytest.mark.unit
@pytest.mark.parametrize("ticker", ["brk.b", "BRK.B"])
def test_r16_trade_ticket_find_report_latest_complete_wins_for_ticker(
    tt, reports, ticker
):
    """Ticker lookup is case-insensitive and returns the NEWEST complete run."""
    first = _write_report(reports, name="BRK.B_20260610_120000")
    second = _write_report(reports, name="BRK.B_20260612_120000")
    newer_incomplete = _write_report(reports, name="BRK.B_20260613_120000", complete=False)
    base = 1_800_000_000.0
    for path, offset in ((first, 0), (second, 10_000), (newer_incomplete, 20_000)):
        _set_mtime(path, base + offset)

    assert tt.find_report(ticker) == second
    assert tt.find_report() == second


@pytest.mark.unit
def test_r16_trade_ticket_find_report_ticker_does_not_prefix_match_other_tickers(tt, reports):
    """``BTC`` must not match ``BTCUSDT_...`` (exact name or ``<T>_`` prefix only)."""
    btcusdt = _write_report(reports, name="BTCUSDT_20260610_120000")
    with pytest.raises(FileNotFoundError):
        tt.find_report("BTC")
    assert tt.find_report("BTCUSDT") == btcusdt


# ---- _fnum -----------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        pytest.param(None, None, id="none"),
        pytest.param("", None, id="empty"),
        pytest.param("n/a", None, id="no-digits"),
        pytest.param("$64,200", 64200.0, id="thousands"),
        pytest.param("1.14e-05", 1.14e-05, id="exponent"),
        pytest.param("-3.5%", -3.5, id="negative-percent"),
        pytest.param("around 12. later", 12.0, id="first-number-wins"),
        pytest.param("Target: 1,234,567.89 USDT", 1234567.89, id="two-groups"),
    ],
)
def test_r16_trade_ticket_fnum_first_number_matrix(tt, text, expected):
    """_fnum extracts the first number (commas / sign / exponent) or None."""
    out = tt._fnum(text)
    if expected is None:
        assert out is None
    else:
        assert out == pytest.approx(expected)


# ---- build_ticket: sizing / leverage / margin arithmetic -------------------


@pytest.mark.unit
def test_r16_trade_ticket_long_stock_ticket_is_hand_computable(tt, bracket_cap, reports):
    """NVDA long: every money field equals the closed-form value (1.5% / 5% stop)."""
    report = _write_report(
        reports, decision_extra="**Price Target**: $120\n",
        market_extra="Support: 90\nResistance: 110\n",
    )

    t = tt.build_ticket(report, 10_000.0, "moderate")

    assert t["status"] == "ok"
    assert (t["ticker"], t["asset_type"], t["direction"]) == ("NVDA", "us_stock", "long")
    assert t["stop_dist_pct"] == pytest.approx(0.05)
    assert t["R_dollar"] == pytest.approx(5.0)
    assert t["atr_pct"] == pytest.approx(0.02)
    assert t["risk_dollar"] == pytest.approx(150.0)
    assert t["notional"] == pytest.approx(3000.0)
    assert t["quantity"] == pytest.approx(30.0)
    assert t["leverage_caps"] == {
        "L_liq": pytest.approx(10.0), "L_vol": pytest.approx(10.0),
        "L_conv": pytest.approx(10.0), "L_hard": 5.0,
    }
    assert t["leverage"] == 5.0
    assert t["margin"] == pytest.approx(600.0)
    assert t["margin_pct_of_capital"] == pytest.approx(0.06)
    assert t["margin_warning"] is False
    assert t["take_profits"] == pytest.approx([107.5, 115.0, 125.0])
    assert t["liquidation_price"] is None  # Reg-T equity: no perp liq formula
    assert t["conviction_score"] == pytest.approx(0.90)
    assert t["price_target_pm"] == pytest.approx(120.0)
    # us_stock tickets never consult the exchange bracket endpoint.
    assert bracket_cap.calls == []


@pytest.mark.unit
def test_r16_trade_ticket_short_stock_ticket_mirrors_levels(tt, bracket_cap, reports):
    """Sell short: stop above entry, TPs below entry, same risk budget."""
    report = _write_report(reports, rating="Sell", action="Sell", stop=104.0)

    t = tt.build_ticket(report, 10_000.0, "moderate")

    assert t["status"] == "ok"
    assert t["direction"] == "short"
    assert t["stop_dist_pct"] == pytest.approx(0.04)
    assert t["notional"] == pytest.approx(3750.0)  # 150 / 4%
    assert t["quantity"] == pytest.approx(37.5)
    assert t["leverage_caps"]["L_liq"] == pytest.approx(12.5)  # 1 / (2 x 4%)
    assert t["leverage"] == 5.0
    assert t["margin"] == pytest.approx(750.0)
    assert t["take_profits"] == pytest.approx([94.0, 88.0, 80.0])
    assert all(tp < t["entry"] for tp in t["take_profits"])


@pytest.mark.unit
@pytest.mark.parametrize("capital", [100.0, 10_000.0, 1_000_000.0])
@pytest.mark.parametrize("profile", ["conservative", "moderate", "aggressive"])
@pytest.mark.parametrize(
    ("rating", "action", "entry", "stop"),
    [
        pytest.param("Buy", "Buy", 100.0, 95.0, id="long-5pct"),
        pytest.param("Sell", "Sell", 100.0, 104.0, id="short-4pct"),
        pytest.param("Overweight", "Buy", 250.5, 243.0, id="tilt-long-odd-price"),
        pytest.param("Underweight", "Sell", 7.5, 7.92, id="tilt-short-micro-price"),
    ],
)
def test_r16_trade_ticket_risk_at_stop_equals_budget_matrix(
    tt, bracket_cap, reports, capital, profile, rating, action, entry, stop
):
    """Pins the sizing contract for every capital x profile x direction cell.

    Loss at the stop is exactly ``capital * RISK_FRAC[profile]``; margin x
    leverage reconciles to notional; leverage is the bounded min of its caps.
    """
    report = _write_report(
        reports, rating=rating, action=action, entry=entry, stop=stop, atr=entry * 0.02,
    )

    t = tt.build_ticket(report, capital, profile)

    assert t["status"] == "ok"
    budget = capital * tt.RISK_FRAC[profile]
    assert t["risk_frac"] == tt.RISK_FRAC[profile]
    assert t["risk_dollar"] == pytest.approx(budget)
    assert t["quantity"] * t["R_dollar"] == pytest.approx(budget)
    assert t["notional"] * t["stop_dist_pct"] == pytest.approx(budget)
    assert t["notional"] == pytest.approx(t["quantity"] * t["entry"])
    assert t["margin"] * t["leverage"] == pytest.approx(t["notional"])
    assert t["margin_pct_of_capital"] == pytest.approx(t["margin"] / capital)
    assert t["margin_warning"] == (t["margin_pct_of_capital"] > 0.50)
    expected_l = max(1.0, min(min(t["leverage_caps"].values()), 5.0))
    assert t["leverage"] == pytest.approx(expected_l)
    expected_direction = "long" if rating in ("Buy", "Overweight") else "short"
    assert t["direction"] == expected_direction
    sign = 1.0 if expected_direction == "long" else -1.0
    assert t["take_profits"] == pytest.approx(
        [entry + sign * m * t["R_dollar"] for m in (1.5, 3.0, 5.0)]
    )
    assert bracket_cap.calls == []  # us_stock: no exchange lookup


@pytest.mark.unit
@pytest.mark.parametrize(
    ("rating", "action", "expected_stop"),
    [("Buy", "Buy", 96.0), ("Sell", "Sell", 104.0)],
)
def test_r16_trade_ticket_atr_only_report_builds_stop_at_two_atr(
    tt, bracket_cap, reports, rating, action, expected_stop
):
    """No stop anywhere → stop = entry ∓ 2*ATR (ATR_STOP_MULT), on the right side."""
    report = _write_report(reports, rating=rating, action=action, stop=None, atr=2.0)

    t = tt.build_ticket(report, 10_000.0, "moderate")

    assert t["status"] == "ok"
    assert t["stop"] == pytest.approx(expected_stop)
    assert t["atr"] == pytest.approx(2.0)
    assert t["stop_dist_pct"] == pytest.approx(0.04)


@pytest.mark.unit
def test_r16_trade_ticket_stop_only_report_back_solves_atr(tt, bracket_cap, reports):
    """ATR missing but entry+stop present → ATR = |entry-stop| / ATR_STOP_MULT."""
    report = _write_report(reports, stop=95.0, atr=None)

    t = tt.build_ticket(report, 10_000.0, "moderate")

    assert t["atr"] == pytest.approx(2.5)
    assert t["atr_pct"] == pytest.approx(0.025)
    assert t["leverage_caps"]["L_vol"] == pytest.approx(0.20 / 0.025)


@pytest.mark.unit
def test_r16_trade_ticket_trader_action_fallback_without_pm_rating(tt, bracket_cap, reports):
    """No PM rating → Trader action decides, with the weaker (strength 1) cap."""
    report = _write_report(reports, rating=None, action="Sell", stop=104.0)

    t = tt.build_ticket(report, 10_000.0, "moderate")

    assert t["direction"] == "short"
    assert "Trader Action Sell" in t["direction_reason"]
    assert t["leverage_caps"]["L_conv"] == pytest.approx(5.0)  # moderate x strength 1
    assert t["conviction_score"] is None


# ---- build_ticket: non-positive stop must refuse, never a ticket ------------


def _non_positive_stop_cases():
    return [
        pytest.param(
            {"stop": 0.0, "atr": None},
            id="explicit-zero-stop",
        ),
        pytest.param(
            {"stop": None, "atr": 80.0},
            id="oversized-atr-drives-derived-stop-negative",
        ),
        pytest.param(
            {"stop": -40.0, "atr": None},
            id="explicit-negative-stop",
        ),
    ]


@pytest.mark.unit
@pytest.mark.parametrize("kwargs", _non_positive_stop_cases())
@pytest.mark.parametrize(("rating", "action"), [("Buy", "Buy"), ("Overweight", "Buy")])
def test_r16_trade_ticket_non_positive_stop_is_degenerate_not_a_ticket(
    tt, bracket_cap, reports, kwargs, rating, action
):
    """Regression: a long stop at/below $0 must be ``degenerate_levels``.

    The guard only rejected a non-positive ENTRY. A stop of 0 (parsed from
    "Stop Loss: $0") or a negative one (2*ATR > entry, or "-40" in the
    report) sailed through as stop_dist >= 100% and produced an ``ok``
    ticket built on a price that cannot exist.
    """
    report = _write_report(reports, rating=rating, action=action, **kwargs)

    t = tt.build_ticket(report, 10_000.0, "moderate")

    assert t["direction"] == "long"
    assert t["status"] == "degenerate_levels"
    assert t["stop"] <= 0
    assert "notional" not in t
    assert "leverage" not in t
    assert bracket_cap.calls == []
    assert "价位退化" in tt.render_ticket(t)


# ---- build_ticket: perp leverage / liquidation ----------------------------


def _perp_overlay(stop, spot=False):
    """The overlay as a run renders it for a stop at ``stop`` (perp bullets unless spot)."""
    head = (
        "### Quantitative Risk Overlay\n"
        "- **Action**: enter\n"
        "- **Target Weight**: 8.0%\n"
        f"- **Stop Loss**: {stop:.1f}\n"
        "- **Entry Reference**: 64000.0\n"
        "- **Drawdown Regime**: normal\n"
    )
    if spot:
        return head
    return head + (
        "- **Suggested Leverage**: ≤ 4.0x\n"
        "- **Est. Liquidation Price**: 51488.0\n"
    )


def _perp_report(root, rating="Buy", action="Buy", stop=62400.0, spot=False):
    """BTCUSDT report whose trader and overlay levels agree at ``stop``."""
    return _write_report(
        root, name="BTCUSDT_20260919_120000", rating=rating, action=action,
        entry=64000.0, stop=stop, atr=800.0, overlay=_perp_overlay(stop, spot=spot),
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("rating", "action", "sign", "mmr_sign"),
    [("Buy", "Buy", -1.0, 1.0), ("Sell", "Sell", 1.0, -1.0)],
)
def test_r16_trade_ticket_perp_ticket_hand_computed_with_liquidation(
    tt, bracket_cap, reports, rating, action, sign, mmr_sign
):
    """BTCUSDT 2.5% stop: L=10x, margin=600, liq = entry*(1 - 1/L + MMR + fee) mirrored."""
    stop = 62400.0 if sign < 0 else 65600.0
    report = _perp_report(reports, rating=rating, action=action, stop=stop)

    t = tt.build_ticket(report, 10_000.0, "moderate")

    assert t["status"] == "ok"
    assert t["asset_type"] == "crypto_perp"
    assert t["stop_dist_pct"] == pytest.approx(0.025)
    assert t["notional"] == pytest.approx(6000.0)
    assert t["quantity"] == pytest.approx(6000.0 / 64000.0)
    assert t["leverage_caps"]["L_liq"] == pytest.approx(20.0)
    assert t["leverage_caps"]["L_vol"] == pytest.approx(24.0)  # 0.30 / 1.25%
    assert t["leverage"] == 10.0
    assert t["margin"] == pytest.approx(600.0)
    assert t["liquidation_mmr"] == pytest.approx(0.004)  # first default tier
    assert t["liquidation_mmr_basis"] == "default bracket ladder @ notional 6,000 USDT"
    assert t["liquidation_refusal_reason"] is None
    expected_liq = 64000.0 * (1.0 + sign * 0.1 + mmr_sign * (0.004 + 0.0005))
    assert t["liquidation_price"] == pytest.approx(expected_liq)
    # Liquidation must sit BEYOND the stop, otherwise the stop can never fire.
    assert (t["stop"] - t["liquidation_price"]) * (-sign) > 0
    assert t["liquidation_dist_pct"] == pytest.approx(
        abs(64000.0 - expected_liq) / 64000.0
    )
    # The bracket lookup, when it happens, is tiered on the REAL notional.
    assert bracket_cap.calls == [("BTCUSDT", pytest.approx(6000.0))]


@pytest.mark.unit
@pytest.mark.parametrize(
    ("capital", "expected_mmr", "basis_notional"),
    [
        pytest.param(10_000.0, 0.004, "6,000", id="tier-1"),
        pytest.param(5_000_000.0, 0.0125, "3,000,000", id="tier-7"),
        pytest.param(10_000_000.0, 0.05, "6,000,000", id="tier-8"),
    ],
)
def test_r16_trade_ticket_perp_liquidation_mmr_tiers_on_real_notional(
    tt, bracket_cap, reports, capital, expected_mmr, basis_notional
):
    """MMR for the liq price is read off the ticket's own notional, not 50k/entry."""
    report = _perp_report(reports)

    t = tt.build_ticket(report, capital, "moderate")

    assert t["status"] == "ok"
    assert t["liquidation_mmr"] == pytest.approx(expected_mmr)
    assert t["liquidation_mmr_basis"].endswith(f"notional {basis_notional} USDT")
    expected_liq = 64000.0 * (1.0 - 0.1 + expected_mmr + 0.0005)
    assert t["liquidation_price"] == pytest.approx(expected_liq)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("rating", "action", "stop", "fragment"),
    [
        pytest.param("Buy", "Buy", 62400.0, "爆仓价落在入场价上方", id="long-inverted"),
        pytest.param("Sell", "Sell", 65600.0, "爆仓价落在入场价下方", id="short-inverted"),
    ],
)
def test_r16_trade_ticket_perp_liquidation_inversion_is_refused(
    tt, bracket_cap, reports, capsys, rating, action, stop, fragment
):
    """MMR+fee >= 1/L flips the formula to the wrong side: refuse, never render it."""
    report = _perp_report(reports, rating=rating, action=action, stop=stop)

    t = tt.build_ticket(report, 100_000_000.0, "moderate")  # notional 60M → MMR 12.5%

    assert t["status"] == "ok"
    assert t["liquidation_price"] is None
    assert t["liquidation_dist_pct"] is None
    assert t["liquidation_mmr"] == pytest.approx(0.125)
    assert fragment in t["liquidation_refusal_reason"]
    assert "公式倒挂" in t["liquidation_refusal_reason"]
    rendered = tt.render_ticket(t)
    assert "| 爆仓价(估) | **拒绝渲染** |" in rendered
    assert "距入场" not in rendered.split("爆仓价(估)")[1].split("\n")[0]


@pytest.mark.unit
def test_r16_trade_ticket_perp_bracket_cap_binds_leverage_and_margin(tt, bracket_cap, reports):
    """A 5x exchange tier caps L at 5x and is looked up on the real notional."""
    bracket_cap.value = 5.0
    report = _perp_report(reports)

    t = tt.build_ticket(report, 10_000.0, "moderate")

    assert t["leverage"] == 5.0
    assert t["leverage_caps"]["L_bracket"] == 5.0
    assert t["margin"] == pytest.approx(1200.0)
    assert bracket_cap.calls == [("BTCUSDT", pytest.approx(6000.0))]


@pytest.mark.unit
@pytest.mark.parametrize("exc", [RuntimeError("no key"), TimeoutError("socket"), ValueError("bad tier")])
def test_r16_trade_ticket_perp_bracket_lookup_failure_falls_back_to_static_ceiling(
    tt, bracket_cap, reports, exc
):
    """The advisory cap is fail-soft: any lookup error leaves the old 20x ceiling."""
    bracket_cap.value = exc
    report = _perp_report(reports)

    t = tt.build_ticket(report, 10_000.0, "moderate")

    assert t["status"] == "ok"
    assert "L_bracket" not in t["leverage_caps"]
    assert t["leverage"] == 10.0
    assert tt.render_ticket(t)


@pytest.mark.unit
def test_r16_trade_ticket_perp_loose_bracket_cap_does_not_bind(tt, bracket_cap, reports):
    """A bracket cap above the other caps is recorded but does not move L."""
    bracket_cap.value = 125.0
    report = _perp_report(reports)

    t = tt.build_ticket(report, 10_000.0, "moderate")

    assert t["leverage"] == 10.0
    assert t["leverage_caps"]["L_bracket"] == 125.0


@pytest.mark.unit
def test_r16_trade_ticket_spot_ticket_is_unlevered(tt, bracket_cap, reports):
    """A healthy overlay without perp bullets is crypto_spot: L=1, no liq row."""
    report = _perp_report(reports, spot=True)

    t = tt.build_ticket(report, 10_000.0, "moderate")

    assert t["asset_type"] == "crypto_spot"
    assert t["leverage"] == 1.0
    assert t["leverage_caps"]["L_hard"] == 1.0
    assert t["liquidation_price"] is None
    assert t["margin"] == pytest.approx(t["notional"])
    assert bracket_cap.calls == []
    rendered = tt.render_ticket(t)
    assert "永续合约：留意资金费率" not in rendered
    assert f"| 数量 | {tt._money(t['quantity'])} BTC |" in rendered


@pytest.mark.unit
def test_r16_trade_ticket_tight_stop_flags_margin_concentration(tt, bracket_cap, reports):
    """A 0.05% stop sizes a 20x-capital notional: margin > 50% raises the warning."""
    report = _write_report(reports, stop=99.95, atr=None)

    t = tt.build_ticket(report, 10_000.0, "moderate")

    assert t["status"] == "ok"
    assert t["notional"] == pytest.approx(300_000.0)
    assert t["margin_pct_of_capital"] == pytest.approx(6.0)
    assert t["margin_warning"] is True
    assert "⚠️ 过度集中，建议降风险预算" in tt.render_ticket(t)


# ---- render ----------------------------------------------------------------


@pytest.mark.unit
def test_r16_trade_ticket_render_missing_levels_explains_what_to_supply(tt, bracket_cap, reports):
    """missing_levels names the fields to add and the --report-dir escape hatch."""
    report = _write_report(reports, stop=None, atr=None)

    t = tt.build_ticket(report, 10_000.0, "moderate")

    assert t["status"] == "missing_levels"
    rendered = tt.render_ticket(t)
    assert "## ⚠️ 无法生成交易单" in rendered
    assert "**Entry Price** / **Stop Loss**" in rendered
    assert "--report-dir" in rendered
    assert "免责声明" in rendered
    assert "## ① 方向" not in rendered


@pytest.mark.unit
@pytest.mark.parametrize(
    ("rating", "action", "stop"),
    [("Sell", "Sell", 95.0), ("Underweight", "Sell", 95.0), ("Buy", "Buy", 105.0)],
)
def test_r16_trade_ticket_render_wrong_side_stop_shows_the_specific_reason(
    tt, bracket_cap, reports, rating, action, stop
):
    """A stop on the losing side is refused with a reason naming both prices."""
    report = _write_report(reports, rating=rating, action=action, stop=stop)

    t = tt.build_ticket(report, 10_000.0, "moderate")

    assert t["status"] == "degenerate_levels"
    assert f"stop {stop:g} on wrong side of entry 100" in t["status_reason"]
    rendered = tt.render_ticket(t)
    assert "方向与位价侧向矛盾" in rendered
    assert t["status_reason"] in rendered
    assert "止损距离为零" not in rendered


@pytest.mark.unit
def test_r16_trade_ticket_render_structural_reference_rows(tt, bracket_cap, reports):
    """Long: nearest resistance ABOVE entry; short: nearest support BELOW entry."""
    levels = "Support: 90\nSupport: 80\nResistance: 110\nResistance: 120\n"
    long_report = _write_report(
        reports, name="NVDA_20260610_120000", market_extra=levels, decision_extra="**Price Target**: $120\n",
    )
    short_report = _write_report(
        reports, name="NVDA_20260611_120000", rating="Sell", action="Sell", stop=104.0, market_extra=levels,
    )

    long_md = tt.render_ticket(tt.build_ticket(long_report, 10_000.0, "moderate"))
    short_md = tt.render_ticket(tt.build_ticket(short_report, 10_000.0, "moderate"))

    assert "| 结构参考·最近阻力 | 110 | 注意 TP 可能在此之前遇阻" in long_md
    assert "结构参考·最近支撑" not in long_md
    assert "| 结构参考·最近支撑 | 90 | 注意 TP 可能在此之前企稳" in short_md
    assert "结构参考·最近阻力" not in short_md
    assert "| PM 目标价 | 120 | 组合经理给的参考 |" in long_md
    assert "PM 目标价" not in short_md


@pytest.mark.unit
def test_r16_trade_ticket_render_no_structural_row_when_all_supports_above_entry(tt, bracket_cap, reports):
    """Short breakdown narrative: every parsed support sits above entry → no row, no crash."""
    report = _write_report(
        reports, rating="Sell", action="Sell", stop=104.0, market_extra="Support: 130\nSupport: 150\n",
    )

    md = tt.render_ticket(tt.build_ticket(report, 10_000.0, "moderate"))

    assert "结构参考" not in md
    assert "## ④ 执行建议" in md


@pytest.mark.unit
@pytest.mark.parametrize(
    ("cap_value", "heading", "has_row"),
    [
        pytest.param(None, "取四个上限的最小值", False, id="static-ceiling"),
        pytest.param(5.0, "取五个上限的最小值", True, id="bracket-present"),
    ],
)
def test_r16_trade_ticket_render_leverage_cap_table_marks_binding_cap(
    tt, bracket_cap, reports, cap_value, heading, has_row
):
    """The cap table lists every cap and flags exactly the binding one."""
    bracket_cap.value = cap_value
    report = _perp_report(reports)

    t = tt.build_ticket(report, 10_000.0, "moderate")
    md = tt.render_ticket(t)

    assert f"### 杠杆是怎么来的（{heading}）" in md
    assert ("| 交易所档位上限 L_bracket |" in md) is has_row
    binding = [line for line in md.splitlines() if line.endswith("← **生效** |")]
    assert len(binding) == 1
    binding_value = float(binding[0].split("|")[2].strip().rstrip("×"))
    assert binding_value == pytest.approx(t["leverage"])
    for row in (
        "| 爆仓安全上限 L_liq |", "| 波动率上限 L_vol |",
        "| 信心上限 L_conv |", "| 资产硬顶 L_hard |",
    ):
        assert row in md


@pytest.mark.unit
@pytest.mark.parametrize(
    ("value", "expected"),
    [
        pytest.param(None, "—", id="none-is-dash"),
        pytest.param(1234.567, "1,234.57", id="large-two-decimals"),
        pytest.param(150.0, "150", id="trailing-zeroes-stripped"),
        pytest.param(0.0000114, "1.14e-05", id="micro-price-scientific"),
        pytest.param(1.14e-05, "1.14e-05", id="micro-price-scientific-input"),
    ],
)
def test_r16_trade_ticket_money_format_matrix(tt, value, expected):
    """_money: dash for None, 2dp above 1000, 4dp stripped below, sci below 1e-3."""
    assert tt._money(value) == expected


@pytest.mark.unit
@pytest.mark.parametrize(
    ("ticker", "asset_type", "expected"),
    [
        pytest.param("NVDA", "us_stock", "股", id="stock-shares"),
        pytest.param("BTCUSDT", "crypto_perp", "BTC", id="perp-base"),
        pytest.param("ETHUSDC", "crypto_spot", "ETH", id="spot-base"),
        pytest.param("PEPEUSDT", "crypto_perp", "PEPE", id="perp-base-2"),
    ],
)
def test_r16_trade_ticket_share_unit_matrix(tt, ticker, asset_type, expected):
    """Crypto tickets quote the base asset, equities quote shares."""
    assert tt._share_unit({"ticker": ticker, "asset_type": asset_type}) == expected


@pytest.mark.unit
@pytest.mark.parametrize(
    ("direction", "move", "trail"),
    [
        pytest.param("long", "把止损上移到成本", "价格回到成本下方", id="long"),
        pytest.param("short", "把止损下移到成本", "价格回到成本上方", id="short"),
    ],
)
def test_r16_trade_ticket_execution_plan_stop_to_breakeven_matches_direction(
    tt, bracket_cap, reports, direction, move, trail
):
    """Regression: the TP1 instruction must move the stop TOWARD cost for the side.

    It read "把止损上移到成本" for both sides. On a short, cost is ABOVE
    price after a favourable move, so "move the stop up to cost" is the
    instruction that turns a winner into a loser.
    """
    rating, action, stop = ("Buy", "Buy", 95.0) if direction == "long" else ("Sell", "Sell", 104.0)
    report = _write_report(reports, rating=rating, action=action, stop=stop)

    plan = tt._execution_plan(tt.build_ticket(report, 10_000.0, "moderate"))

    assert move in plan
    assert trail in plan
    opposite = "下移" if direction == "long" else "上移"
    assert f"把止损{opposite}到成本" not in plan


@pytest.mark.unit
def test_r16_trade_ticket_execution_plan_quotes_margin_stop_and_risk(tt, bracket_cap, reports):
    """The plan states the margin to post, the leverage, the stop and the budget."""
    report = _write_report(reports, stop=95.0)

    t = tt.build_ticket(report, 10_000.0, "moderate")
    plan = tt._execution_plan(t)

    assert "首批用 600 保证金开 5.0× 杠杆头寸" in plan
    assert "价格触及 **95** 必须无条件平仓" in plan
    assert "单笔最大亏损 150（资金的 1.50%）" in plan
    assert "到 TP1 107.5 平 1/3" in plan
    assert "到 TP2 115 再平 1/3" in plan
    assert all(line.startswith("- ") for line in plan.splitlines())


# ---- CLI -------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize("capital", ["0", "-5", "nan", "inf", "1e-999"])
def test_r16_trade_ticket_cli_rejects_non_positive_or_non_finite_capital(
    tt, monkeypatch, capsys, capital
):
    """--capital must be finite and > 0; nothing is looked up or built."""
    monkeypatch.setattr(sys, "argv", ["trade_ticket.py", "--capital", capital])
    built = []
    monkeypatch.setattr(tt, "build_ticket", lambda *a, **k: built.append(a))

    with pytest.raises(SystemExit) as excinfo:
        tt.main()

    assert excinfo.value.code == 2
    assert built == []
    assert "✗ 资金必须 > 0 且为有限数" in capsys.readouterr().err


@pytest.mark.unit
@pytest.mark.parametrize(
    ("typed", "expected"),
    [
        pytest.param("1,000", 1000.0, id="thousands-separator"),
        pytest.param(" 250000.5 ", 250000.5, id="whitespace"),
        pytest.param("1e4", 10000.0, id="scientific"),
    ],
)
def test_r16_trade_ticket_cli_interactive_capital_is_parsed(
    tt, bracket_cap, reports, monkeypatch, capsys, typed, expected
):
    """No --capital → prompt once; the typed amount reaches the ticket."""
    report = _write_report(reports)
    monkeypatch.setattr("builtins.input", lambda _prompt: typed)
    monkeypatch.setattr(sys, "argv", ["trade_ticket.py", "--json", "--report-dir", str(report)])

    tt.main()

    ticket = json.loads(capsys.readouterr().out)
    assert ticket["capital"] == pytest.approx(expected)
    assert ticket["risk_profile"] == "moderate"
    assert ticket["risk_dollar"] == pytest.approx(expected * 0.015)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("typed_or_exc", "fragment"),
    [
        pytest.param("abc", "✗ 必须提供 --capital", id="non-numeric"),
        pytest.param(EOFError(), "✗ 必须提供 --capital", id="eof"),
        pytest.param("", "✗ 必须提供 --capital", id="blank"),
        pytest.param("nan", "✗ 资金必须 > 0 且为有限数", id="nan-typed"),
        pytest.param("-1", "✗ 资金必须 > 0 且为有限数", id="negative-typed"),
    ],
)
def test_r16_trade_ticket_cli_interactive_bad_input_exits_2(
    tt, monkeypatch, capsys, typed_or_exc, fragment
):
    """An unusable interactive answer is exit 2 with a message, never a NaN ticket."""
    def fake_input(_prompt):
        if isinstance(typed_or_exc, Exception):
            raise typed_or_exc
        return typed_or_exc

    monkeypatch.setattr("builtins.input", fake_input)
    monkeypatch.setattr(sys, "argv", ["trade_ticket.py"])

    with pytest.raises(SystemExit) as excinfo:
        tt.main()

    assert excinfo.value.code == 2
    assert fragment in capsys.readouterr().err


@pytest.mark.unit
def test_r16_trade_ticket_cli_json_output_is_machine_readable(
    tt, bracket_cap, reports, monkeypatch, capsys
):
    """--json prints one parseable document holding the whole ticket."""
    _write_report(reports, name="NVDA_20260610_120000")

    monkeypatch.setattr(sys, "argv", [
        "trade_ticket.py", "--json", "--ticker", "NVDA", "--capital", "10000", "--risk-profile", "aggressive",
    ])
    tt.main()
    ticket = json.loads(capsys.readouterr().out)

    assert ticket["status"] == "ok"
    assert ticket["ticker"] == "NVDA"
    assert ticket["capital"] == 10000.0
    assert ticket["risk_profile"] == "aggressive"
    assert ticket["risk_dollar"] == pytest.approx(250.0)
    assert ticket["notional"] == pytest.approx(5000.0)
    assert set(ticket) >= {"entry", "stop", "quantity", "leverage", "margin", "take_profits", "leverage_caps"}


@pytest.mark.unit
def test_r16_trade_ticket_cli_markdown_output_is_the_default(
    tt, bracket_cap, reports, monkeypatch, capsys
):
    """Without --json the CLI renders the human ticket for the newest report."""
    _write_report(reports, name="MSFT_20260609_120000")
    newest = _write_report(reports, name="MSFT_20260612_120000")
    _set_mtime(reports / "MSFT_20260609_120000", 1_800_000_000.0)
    _set_mtime(newest, 1_800_010_000.0)
    monkeypatch.setattr(sys, "argv", ["trade_ticket.py", "--capital", "10000", "--risk-profile", "conservative"])

    tt.main()

    out = capsys.readouterr().out
    assert out.startswith("# 交易执行单 · MSFT  (us_stock)")
    assert f"> 报告：`{newest}`" in out
    assert "资金 10,000.00 · 风险偏好 `conservative`" in out
    assert "| 单笔最大亏损 | **100** | 资金的 1.00%（conservative 档：打止损恰好亏这么多） |" in out
    with pytest.raises(json.JSONDecodeError):
        json.loads(out)


@pytest.mark.unit
def test_r16_trade_ticket_cli_missing_report_propagates_not_found(tt, monkeypatch, tmp_path):
    """A missing --report-dir surfaces as FileNotFoundError (non-zero exit via traceback)."""
    monkeypatch.setattr(sys, "argv", [
        "trade_ticket.py", "--capital", "10000", "--report-dir", str(tmp_path / "absent"),
    ])
    with pytest.raises(FileNotFoundError, match="报告目录不存在"):
        tt.main()


@pytest.mark.unit
@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        pytest.param(["--capital", "0"], 2, id="bad-capital"),
        pytest.param(["--capital", "abc"], 2, id="non-numeric-capital"),
    ],
)
def test_r16_trade_ticket_main_guard_exit_code(monkeypatch, argv, expected):
    """``python scripts/trade_ticket.py`` exits 2 on an invalid --capital."""
    assert _exec_as_main(monkeypatch, "trade_ticket.py", argv) == expected
