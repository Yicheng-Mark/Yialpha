"""Round-17 audit tests for the ``scripts_e2e`` shard (every entry script in ``scripts/``).

The scripts are thin wrappers over the package, so each test drives the real
``main()`` / helper in process with the package seam (batch runner, graph
builder, subprocess launcher, clock, output roots) replaced by a spy. Every
test pins something exact: the parsed namespace, the exit code, the emitted
lines, the arguments handed to the package.

Hermetic: no network, no LLM call, no analysis subprocess, no sleeping. Files a
script writes go to ``tmp_path``.
"""

from __future__ import annotations

import argparse
import importlib.util
import io
import json
import math
import os
import random
import re
import subprocess
import sys
import threading
from collections.abc import MutableMapping
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from yialpha import default_config as _default_config_module
from yialpha.batch import runner as batch_runner
from yialpha.dataflows.errors import UnsupportedMarketError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"


class _LiveDefaultConfig(MutableMapping):
    """``DEFAULT_CONFIG`` resolved on every use, not bound at collection time.

    ``tests/test_env_overrides.py`` and ``tests/test_web_store_results_dir.py`` reload
    ``yialpha.default_config`` and leave a NEW dict behind, built while some of this
    machine's ``.env`` variables were unset. A name bound to the old dict at collection
    time then disagrees with every module imported afterwards -- which showed up as 20
    order-dependent failures in the full suite on the dev box (``llm_max_retries`` 6 vs 2)
    and nowhere else.
    """

    @property
    def _live(self) -> dict:
        return _default_config_module.DEFAULT_CONFIG

    def __getitem__(self, key):
        return self._live[key]

    def __setitem__(self, key, value):
        self._live[key] = value

    def __delitem__(self, key):
        del self._live[key]

    def __iter__(self):
        return iter(self._live)

    def __len__(self) -> int:
        return len(self._live)

    def __eq__(self, other) -> bool:
        return self._live == other

    def __ne__(self, other) -> bool:
        return self._live != other

    __hash__ = None  # type: ignore[assignment]

    def __repr__(self) -> str:
        return f"_LiveDefaultConfig({self._live!r})"

    def copy(self) -> dict:
        return dict(self._live)


DEFAULT_CONFIG = _LiveDefaultConfig()

# Glyphs the scripts print, written as escapes: the first two carry a variation
# selector that is invisible in an editor.
_PLAY = "▶️"
_INFO = "ℹ️"
_OK = "✅"
_FAIL = "❌"


# --------------------------------------------------------------------------- #
# Loading a script
# --------------------------------------------------------------------------- #


def _load(name: str, filename: str, on_executed=None):
    """Load ``scripts/<filename>`` as an isolated module (``__main__`` never fires).

    ``sys.path`` is restored afterwards: the scripts insert their own directory
    to import ``_bootstrap``; that entry must not leak into later tests.
    ``on_executed`` sees ``sys.path`` as the module left it, before the restore.
    """
    spec = importlib.util.spec_from_file_location(name, SCRIPTS_DIR / filename)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    saved_path = list(sys.path)
    try:
        spec.loader.exec_module(mod)
        if on_executed is not None:
            on_executed(list(sys.path))
    finally:
        sys.path[:] = saved_path
    return mod


def _set_argv(monkeypatch, program: str, argv: list[str]) -> None:
    monkeypatch.setattr(sys, "argv", [program, *argv])


def _script_wall_clock(monkeypatch, mod, *seconds: float) -> None:
    """``datetime.now()`` in the script returns fixed instants, in call order.

    Each value is an offset in seconds from one fixed instant; the last value
    repeats once the list is exhausted.
    """
    base = datetime(2026, 6, 27, 9, 30, 0)
    steps = iter(seconds)

    class _Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return base + timedelta(seconds=next(steps, seconds[-1]))

    monkeypatch.setattr(mod, "datetime", _Clock)


# =========================================================================== #
# scripts/run_batch.py
# =========================================================================== #


@pytest.fixture(scope="module")
def rb_mod():
    return _load("r17_run_batch", "run_batch.py")


def _parse(mod, monkeypatch, *argv: str) -> argparse.Namespace:
    _set_argv(monkeypatch, "run_batch.py", list(argv))
    return mod._parse_args()


class TestRunBatchParseArgs:
    def test_r17_run_batch_parse_defaults(self, rb_mod, monkeypatch):
        args = _parse(rb_mod, monkeypatch, "--tickers", "AAPL", "NVDA", "--date", "2026-06-27")

        assert vars(args) == {
            "tickers": ["AAPL", "NVDA"],
            "date": "2026-06-27",
            "asset_type": "auto",
            "workers": None,
            "no_progress": False,
            "out": None,
        }

    def test_r17_run_batch_parse_every_flag(self, rb_mod, monkeypatch, tmp_path):
        out = tmp_path / "elsewhere"

        args = _parse(
            rb_mod, monkeypatch,
            "--tickers", "BTCUSDT", "ETHUSDT", "--date", "2026-06-27",
            "--asset-type", "crypto_perp", "--workers", "3", "--no-progress",
            "--out", str(out),
        )

        assert vars(args) == {
            "tickers": ["BTCUSDT", "ETHUSDT"],
            "date": "2026-06-27",
            "asset_type": "crypto_perp",
            "workers": 3,
            "no_progress": True,
            "out": str(out),
        }

    @pytest.mark.parametrize(
        "argv",
        [
            pytest.param([], id="nothing"),
            pytest.param(["--date", "2026-06-27"], id="no-tickers"),
            pytest.param(["--tickers", "AAPL"], id="no-date"),
        ],
    )
    def test_r17_run_batch_parse_missing_required_flag_exits_2(self, rb_mod, monkeypatch, capsys, argv):
        _set_argv(monkeypatch, "run_batch.py", argv)

        with pytest.raises(SystemExit) as excinfo:
            rb_mod._parse_args()

        assert excinfo.value.code == 2
        err = capsys.readouterr().err
        for flag in ("--tickers", "--date"):
            assert flag in err

    def test_r17_run_batch_parse_unknown_flag_exits_2(self, rb_mod, monkeypatch, capsys):
        _set_argv(monkeypatch, "run_batch.py", ["--tickers", "AAPL", "--date", "2026-06-27", "--bogus", "MSFT"])

        with pytest.raises(SystemExit) as excinfo:
            rb_mod._parse_args()

        assert excinfo.value.code == 2
        assert "--bogus" in capsys.readouterr().err

    def test_r17_run_batch_parse_abbreviated_flag_is_accepted_and_the_last_list_wins(self, rb_mod, monkeypatch):
        """argparse abbreviations are on (the default): ``--ticker`` IS ``--tickers`` here.

        Pins the consequence rather than a typo guard: with two spellings of the
        same option on one command line the earlier ticker list is dropped
        without a word. Standard argparse semantics, shared by every script.
        """
        args = _parse(
            rb_mod, monkeypatch, "--tickers", "AAPL", "--ticker", "NVDA", "MSFT",
            "--date", "2026-06-27", "--worker", "2", "--no", "--o", "x",
        )

        assert vars(args) == {
            "tickers": ["NVDA", "MSFT"],
            "date": "2026-06-27",
            "asset_type": "auto",
            "workers": 2,
            "no_progress": True,
            "out": "x",
        }

    @pytest.mark.parametrize(
        "value", ["auto", "stock", "crypto", "crypto_spot", "crypto_perp"]
    )
    def test_r17_run_batch_parse_accepts_every_documented_asset_type(self, rb_mod, monkeypatch, value):
        args = _parse(
            rb_mod, monkeypatch, "--tickers", "AAPL", "--date", "2026-06-27", "--asset-type", value
        )

        assert args.asset_type == value

    @pytest.mark.parametrize("value", ["perp", "STOCK", "equity", ""])
    def test_r17_run_batch_parse_rejects_undocumented_asset_type(self, rb_mod, monkeypatch, capsys, value):
        _set_argv(
            monkeypatch, "run_batch.py",
            ["--tickers", "AAPL", "--date", "2026-06-27", "--asset-type", value],
        )

        with pytest.raises(SystemExit) as excinfo:
            rb_mod._parse_args()

        assert excinfo.value.code == 2
        err = capsys.readouterr().err
        assert "--asset-type" in err
        assert repr(value) in err

    @pytest.mark.parametrize("value", ["abc", "2.5", "", "1e3", "3 workers"])
    def test_r17_run_batch_parse_rejects_non_integer_workers(self, rb_mod, monkeypatch, capsys, value):
        _set_argv(
            monkeypatch, "run_batch.py",
            ["--tickers", "AAPL", "--date", "2026-06-27", "--workers", value],
        )

        with pytest.raises(SystemExit) as excinfo:
            rb_mod._parse_args()

        assert excinfo.value.code == 2
        err = capsys.readouterr().err
        assert "--workers" in err
        assert repr(value) in err

    @pytest.mark.parametrize("value", ["0", "-3", "+5", "1", "10000"])
    def test_r17_run_batch_parse_workers_sign_is_left_to_main(self, rb_mod, monkeypatch, value):
        """argparse has no bounds check: ``0`` and negatives parse, ``main()`` rejects them."""
        args = _parse(
            rb_mod, monkeypatch, "--tickers", "AAPL", "--date", "2026-06-27", "--workers", value
        )

        assert args.workers == int(value)

    @pytest.mark.parametrize("value", ["2026-9-1", "20260627", "2026-02-30", "yesterday", ""])
    def test_r17_run_batch_parse_date_is_not_validated_here(self, rb_mod, monkeypatch, value):
        args = _parse(rb_mod, monkeypatch, "--tickers", "AAPL", "--date", value)

        assert args.date == value

    def test_r17_run_batch_parse_comma_separated_tickers_stay_one_token(self, rb_mod, monkeypatch):
        """The documented separator is whitespace; ``AAPL,MSFT`` is one (invalid) ticker."""
        args = _parse(rb_mod, monkeypatch, "--tickers", "AAPL,MSFT", "NVDA", "--date", "2026-06-27")

        assert args.tickers == ["AAPL,MSFT", "NVDA"]

    def test_r17_run_batch_parse_keeps_order_case_and_duplicates(self, rb_mod, monkeypatch):
        args = _parse(
            rb_mod, monkeypatch, "--tickers", "msft", "AAPL", "msft", "brk.b", "--date", "2026-06-27"
        )

        assert args.tickers == ["msft", "AAPL", "msft", "brk.b"]

    def test_r17_run_batch_parse_repeated_tickers_flag_keeps_only_the_last_list(self, rb_mod, monkeypatch):
        args = _parse(
            rb_mod, monkeypatch, "--tickers", "AAPL", "MSFT", "--date", "2026-06-27",
            "--tickers", "NVDA", "INTC",
        )

        assert args.tickers == ["NVDA", "INTC"]

    def test_r17_run_batch_help_documents_exactly_the_parsed_options(self, rb_mod, monkeypatch, capsys):
        # argparse wraps at the terminal width; pin it so narrow consoles do not fold the help.
        monkeypatch.setenv("COLUMNS", "160")
        _set_argv(monkeypatch, "run_batch.py", ["--help"])

        with pytest.raises(SystemExit) as excinfo:
            rb_mod._parse_args()

        assert excinfo.value.code == 0
        out = capsys.readouterr().out
        # One entry per option, in the order they are declared.
        entries = re.findall(r"^  (--[a-z-]+)\b", out, flags=re.M)
        assert entries == ["--tickers", "--date", "--asset-type", "--workers", "--no-progress", "--out"]
        assert re.search(r"^  -h, --help ", out, flags=re.M)
        for text in (
            "代码列表，如 AAPL NVDA MSFT",
            "分析日期 YYYY-MM-DD",
            "{auto,stock,crypto,crypto_spot,crypto_perp}",
            "不传则用 YIALPHA_BATCH_WORKERS，默认 3",
            "关闭进度条",
            "覆盖 results_dir",
        ):
            assert text in out


# --------------------------------------------------------------------------- #
# scripts/run_batch.py — main()
# --------------------------------------------------------------------------- #

_DAY = "2026-06-27"
_POLICY = (
    "only US-listed stocks/ETFs (e.g. AAPL, BRK.B, SPY) and crypto (e.g. BTC-USD, BTCUSDT) "
    "are supported; A-shares, HK and other exchange listings, forex, commodities and index "
    "CFDs were removed"
)


class _FakeRunner:
    """Context-manager double for one ``BatchRunner(...)`` construction."""

    def __init__(self, rows, config, workers, selected_analysts, progress):
        self._rows = rows
        self.config = dict(config)
        self.workers = workers
        self.selected_analysts = tuple(selected_analysts)
        self.progress = progress
        self.run_calls: list[tuple] = []
        self.events: list[object] = []

    def __enter__(self):
        self.events.append("enter")
        return self

    def __exit__(self, *exc):
        self.events.append(("exit", exc))

    def run(self, tickers, trade_date, asset_type="stock"):
        self.events.append("run")
        self.run_calls.append((list(tickers), trade_date, asset_type))
        return [dict(row) for row in self._rows]


class _RunnerSpy:
    """``BatchRunner`` replacement: builds one recording fake per construction."""

    def __init__(self, rows):
        self.rows = rows
        self.built: list[_FakeRunner] = []

    def __call__(self, config, *, workers=None, selected_analysts=(), progress=True):
        fake = _FakeRunner(self.rows, config, workers, selected_analysts, progress)
        self.built.append(fake)
        return fake


@pytest.fixture
def rb(rb_mod, monkeypatch):
    """``run_batch.main()`` with the package runner and analyst selector spied.

    ``harness.rows`` is what ``runner.run()`` returns (fill it before calling
    ``main()``); the wall clock is scripted so the summary line is deterministic.
    """
    harness = SimpleNamespace(mod=rb_mod, rows=[], selected_calls=[], selected=("market", "news"))
    spy = _RunnerSpy(harness.rows)

    def select_analysts(asset_type, tickers):
        harness.selected_calls.append((asset_type, list(tickers)))
        return harness.selected

    monkeypatch.setattr(rb_mod, "BatchRunner", spy)
    monkeypatch.setattr(batch_runner, "batch_selected_analysts", select_analysts)
    _script_wall_clock(monkeypatch, rb_mod, 0, 12.34)
    harness.spy = spy
    return harness


def _row(ticker, *, error=None, elapsed=1.5, report=None):
    """One ``BatchRunner.run()`` result row (only the keys ``main()`` reads)."""
    return {"ticker": ticker, "state": None, "signal": None, "report_path": report,
            "elapsed": elapsed, "error": error}


def _batch_main(harness, monkeypatch, *argv):
    _set_argv(monkeypatch, "run_batch.py", list(argv))
    return harness.mod.main()


def _runner(harness):
    (runner,) = harness.spy.built
    return runner


def _ok_row(ticker, elapsed, report):
    """One summary row for a successful ticker: 12-wide ticker, glyph, 8-wide elapsed."""
    return ticker.ljust(12) + " " + _OK.ljust(6) + " " + elapsed.rjust(8) + "  " + str(report)


def _fail_row(ticker, text):
    return ticker.ljust(12) + " " + _FAIL.ljust(6) + " " + "-".rjust(8) + "  " + text


def _table(*rows):
    return [
        "ticker".ljust(12) + " " + "状态".ljust(6) + " " + "墙钟".rjust(8) + "  " + "详情",
        *rows,
    ]


class TestRunBatchMain:
    def test_r17_run_batch_main_mixed_outcome_table_and_runner_arguments(
        self, rb, monkeypatch, capsys, tmp_path
    ):
        report_a = tmp_path / "reports" / "AAPL_1" / "complete_report.md"
        report_b = tmp_path / "reports" / "NVDA_1" / "complete_report.md"
        rb.rows.extend([
            _row("AAPL", elapsed=1.46, report=report_a),
            _row("NVDA", elapsed=0.04, report=report_b),
            _row("MSFT", error=RuntimeError("boom")),
        ])
        out_dir = tmp_path / "elsewhere"

        rc = _batch_main(
            rb, monkeypatch,
            "--tickers", "AAPL", "NVDA", "MSFT", "--date", _DAY,
            "--asset-type", "stock", "--workers", "2", "--no-progress", "--out", str(out_dir),
        )

        assert rc == 1
        runner = _runner(rb)
        # The config is the package's assembly plus exactly the two overrides main() asks for.
        assert runner.config == {
            **DEFAULT_CONFIG, "batch_concurrency": True, "results_dir": str(out_dir),
        }
        assert (runner.workers, runner.progress, runner.selected_analysts) == (2, False, ("market", "news"))
        assert runner.run_calls == [(["AAPL", "NVDA", "MSFT"], _DAY, "stock")]
        assert runner.events == ["enter", "run", ("exit", (None, None, None))]
        assert rb.selected_calls == [("stock", ["AAPL", "NVDA", "MSFT"])]
        assert capsys.readouterr().out.splitlines() == [
            f"{_PLAY}  批量分析 3 个 ticker | date={_DAY} | type=stock (K=2)",
            "",
            "=== 批量完成：2/3 成功，墙钟 12.3s ===",
            *_table(
                _ok_row("AAPL", "1.5s", report_a),
                _ok_row("NVDA", "0.0s", report_b),
                _fail_row("MSFT", "RuntimeError: boom"),
            ),
        ]

    def test_r17_run_batch_main_all_ok_exit_0(self, rb, monkeypatch, capsys, tmp_path):
        rb.rows.extend([
            _row("AAPL", elapsed=61.0, report=tmp_path / "AAPL_1"),
            _row("NVDA", elapsed=5.05, report=tmp_path / "NVDA_1"),
        ])

        rc = _batch_main(
            rb, monkeypatch, "--tickers", "AAPL", "NVDA", "--date", _DAY, "--workers", "3"
        )

        assert rc == 0
        runner = _runner(rb)
        assert (runner.workers, runner.progress) == (3, True)
        assert runner.config["results_dir"] == DEFAULT_CONFIG["results_dir"]
        out = capsys.readouterr().out.splitlines()
        assert out[:4] == [
            f"{_INFO}  资产类别自动判定：stock（按 AAPL）",
            f"{_PLAY}  批量分析 2 个 ticker | date={_DAY} | type=stock (K=3)",
            "",
            "=== 批量完成：2/2 成功，墙钟 12.3s ===",
        ]
        assert out[4:] == _table(_ok_row("AAPL", "61.0s", tmp_path / "AAPL_1"),
                                 _ok_row("NVDA", "5.0s", tmp_path / "NVDA_1"))

    def test_r17_run_batch_main_every_ticker_failing_exit_1(self, rb, monkeypatch, capsys):
        rb.rows.extend([
            _row("SNDK", error=TimeoutError("watchdog")),
            _row("INTC", error=batch_runner.BatchInputError("bad")),
        ])

        rc = _batch_main(rb, monkeypatch, "--tickers", "SNDK", "INTC", "--date", _DAY)

        assert rc == 1
        out = capsys.readouterr().out.splitlines()
        assert out[3] == "=== 批量完成：0/2 成功，墙钟 12.3s ==="
        assert out[4:] == _table(
            _fail_row("SNDK", "TimeoutError: watchdog"),
            _fail_row("INTC", "BatchInputError: bad"),
        )

    @pytest.mark.parametrize(
        ("first", "resolved", "rest"),
        [
            pytest.param("AAPL", "stock", "MSFT", id="stock"),
            pytest.param("btc-usd", "crypto", "ETH-USD", id="legacy-crypto-dashed"),
            pytest.param("1000PEPEUSDT", "crypto", "ETH-USD", id="compact-alt"),
        ],
    )
    def test_r17_run_batch_main_auto_asset_type_announces_the_detection(
        self, rb, monkeypatch, capsys, first, resolved, rest
    ):
        tickers = [first, rest]
        rb.rows.extend([_row(t) for t in tickers])

        rc = _batch_main(rb, monkeypatch, "--tickers", *tickers, "--date", _DAY)

        assert rc == 0
        runner = _runner(rb)
        assert runner.run_calls == [(tickers, _DAY, resolved)]
        assert rb.selected_calls == [(resolved, tickers)]
        # No --workers: no K hint, and the worker count is left to the config.
        assert runner.workers is None
        out = capsys.readouterr().out.splitlines()
        # The announcement names the ticker as typed, not the resolved symbol.
        assert out[:2] == [
            f"{_INFO}  资产类别自动判定：{resolved}（按 {first}）",
            f"{_PLAY}  批量分析 2 个 ticker | date={_DAY} | type={resolved}",
        ]

    @pytest.mark.parametrize("asset_type", ["stock", "crypto", "crypto_spot", "crypto_perp"])
    def test_r17_run_batch_main_explicit_asset_type_skips_detection(
        self, rb, monkeypatch, capsys, asset_type
    ):
        rb.rows.extend([_row("BTCUSDT"), _row("ETHUSDT")])

        rc = _batch_main(
            rb, monkeypatch,
            "--tickers", "BTCUSDT", "ETHUSDT", "--date", _DAY, "--asset-type", asset_type,
        )

        assert rc == 0
        runner = _runner(rb)
        assert runner.run_calls == [(["BTCUSDT", "ETHUSDT"], _DAY, asset_type)]
        assert rb.selected_calls == [(asset_type, ["BTCUSDT", "ETHUSDT"])]
        out = capsys.readouterr().out.splitlines()
        assert out[0] == f"{_PLAY}  批量分析 2 个 ticker | date={_DAY} | type={asset_type}"
        assert _INFO not in "\n".join(out)

    @pytest.mark.parametrize(
        ("flag", "workers", "concurrent", "hint"),
        [
            pytest.param([], None, None, "", id="omitted-honours-config"),
            pytest.param(["--workers", "1"], 1, False, " (K=1)", id="one-forces-serial"),
            pytest.param(["--workers", "2"], 2, True, " (K=2)", id="two-forces-pool"),
        ],
    )
    def test_r17_run_batch_main_workers_override(
        self, rb, monkeypatch, capsys, flag, workers, concurrent, hint
    ):
        rb.rows.append(_row("AAPL"))

        rc = _batch_main(rb, monkeypatch, "--tickers", "AAPL", "--date", _DAY, *flag)

        assert rc == 0
        runner = _runner(rb)
        assert runner.workers == workers
        if concurrent is None:
            assert runner.config == DEFAULT_CONFIG
        else:
            assert runner.config == {**DEFAULT_CONFIG, "batch_concurrency": concurrent}
        assert capsys.readouterr().out.splitlines() == [
            f"{_INFO}  资产类别自动判定：stock（按 AAPL）",
            f"{_PLAY}  批量分析 1 个 ticker | date={_DAY} | type=stock{hint}",
            "",
            "=== 批量完成：1/1 成功，墙钟 12.3s ===",
            *_table(_ok_row("AAPL", "1.5s", "")),
        ]

    def test_r17_run_batch_main_progress_bar_on_by_default(self, rb, monkeypatch):
        rb.rows.append(_row("AAPL"))

        rc = _batch_main(rb, monkeypatch, "--tickers", "AAPL", "--date", _DAY)

        assert rc == 0
        assert _runner(rb).progress is True

    def test_r17_run_batch_main_forwards_tickers_exactly_as_typed(self, rb, monkeypatch, capsys, tmp_path):
        """No script-side split, case-folding, de-duplication or reordering.

        The runner's rows are printed in the order the runner returns them; the
        headline counts the command line, so it can exceed the row count.
        """
        tickers = ["msft", "AAPL", "msft", "brk.b"]
        reports = {t: tmp_path / f"{t}_1" for t in ("msft", "AAPL", "brk.b")}
        rb.rows.extend([_row(t, report=reports[t]) for t in reports])

        rc = _batch_main(rb, monkeypatch, "--tickers", *tickers, "--date", _DAY)

        assert rc == 0
        runner = _runner(rb)
        assert runner.run_calls == [(tickers, _DAY, "stock")]
        assert rb.selected_calls == [("stock", tickers)]
        assert capsys.readouterr().out.splitlines() == [
            f"{_INFO}  资产类别自动判定：stock（按 msft）",
            f"{_PLAY}  批量分析 4 个 ticker | date={_DAY} | type=stock",
            "",
            "=== 批量完成：3/3 成功，墙钟 12.3s ===",
            *_table(*(_ok_row(t, "1.5s", reports[t]) for t in reports)),
        ]

    def test_r17_run_batch_main_wall_clock_spans_the_runner_context(self, rb, monkeypatch, capsys):
        """Elapsed = second ``datetime.now()`` minus first, printed to 0.1 s."""
        rb.rows.append(_row("AAPL"))
        _script_wall_clock(monkeypatch, rb.mod, 0, 3_661.04)

        rc = _batch_main(rb, monkeypatch, "--tickers", "AAPL", "--date", _DAY)

        assert rc == 0
        assert capsys.readouterr().out.splitlines()[3] == "=== 批量完成：1/1 成功，墙钟 3661.0s ==="

    @pytest.mark.parametrize(
        ("argv", "message"),
        [
            pytest.param(
                ["--tickers", "AAPL", "--date", "2026-9-1"],
                "Bad date (need YYYY-MM-DD): 2026-9-1", id="date-not-zero-padded",
            ),
            pytest.param(
                ["--tickers", "AAPL", "--date", "2026-02-30"],
                "Bad date (need YYYY-MM-DD): 2026-02-30", id="date-impossible-day",
            ),
            pytest.param(
                ["--tickers", "AAPL", "--date", "20260627"],
                "Bad date (need YYYY-MM-DD): 20260627", id="date-compact",
            ),
            pytest.param(
                ["--tickers", "AAPL,MSFT", "--date", _DAY],
                "Invalid ticker(s): ['AAPL,MSFT'] (letters/digits and ._-^=)", id="comma-joined",
            ),
            pytest.param(
                ["--tickers", "AAPL/MSFT", "--date", _DAY],
                "Invalid ticker(s): ['AAPL/MSFT'] (letters/digits and ._-^=)", id="slash-joined",
            ),
            pytest.param(
                ["--tickers", "..", "--date", _DAY],
                "Invalid ticker(s): ['..'] (letters/digits and ._-^=)", id="dot-dot",
            ),
            pytest.param(
                ["--tickers", "A" * 33, "--date", _DAY],
                f"Invalid ticker(s): ['{'A' * 33}'] (letters/digits and ._-^=)", id="too-long",
            ),
            pytest.param(
                ["--tickers", "", "--date", _DAY],
                f"Unsupported market for '': {_POLICY}", id="empty-ticker",
            ),
            pytest.param(
                ["--tickers", "0700.HK", "--date", _DAY],
                f"Unsupported market for '0700.HK': {_POLICY}", id="hong-kong",
            ),
            pytest.param(
                ["--tickers", "7203.T", "--date", _DAY],
                f"Unsupported market for '7203.T': {_POLICY}", id="tokyo",
            ),
            pytest.param(
                ["--tickers", "EURUSD", "--date", _DAY],
                "Unsupported market for 'EURUSD': spot forex pairs are no longer supported — "
                + _POLICY, id="forex",
            ),
            pytest.param(
                ["--tickers", "AAPL", "BTC-USD", "--date", _DAY],
                "Mixed asset classes in one batch: AAPL is stock, but ['BTC-USD'] differ. "
                "Use one class per batch.", id="mixed-stock-first",
            ),
            pytest.param(
                ["--tickers", "BTC-USD", "AAPL", "ETH-USD", "--date", _DAY],
                "Mixed asset classes in one batch: BTC-USD is crypto, but ['AAPL'] differ. "
                "Use one class per batch.", id="mixed-crypto-first",
            ),
            pytest.param(
                ["--tickers", "AAPL", "--date", _DAY, "--workers", "0"],
                "workers must be at least 1", id="workers-zero",
            ),
            pytest.param(
                ["--tickers", "AAPL", "--date", _DAY, "--workers", "-3"],
                "workers must be at least 1", id="workers-negative",
            ),
        ],
    )
    def test_r17_run_batch_main_bad_arguments_exit_2_before_any_runner(
        self, rb, monkeypatch, capsys, argv, message
    ):
        rc = _batch_main(rb, monkeypatch, *argv)

        assert rc == 2
        assert rb.spy.built == []
        assert rb.selected_calls == []
        captured = capsys.readouterr()
        assert captured.out == f"{_FAIL} {message}\n"
        assert captured.err == ""

    def test_r17_run_batch_main_explicit_asset_type_skips_the_one_class_check(self, rb, monkeypatch, capsys):
        """``--asset-type`` turns the mixed-batch guard off: the package takes the word.

        Pins the contract shared with ``yialpha batch`` — the script adds no
        check of its own, so a mixed list under ``stock`` (or a stock under
        ``crypto``) reaches the runner as asked.
        """
        rb.rows.extend([_row("AAPL"), _row("BTC-USD")])

        rc = _batch_main(
            rb, monkeypatch, "--tickers", "AAPL", "BTC-USD", "--date", _DAY, "--asset-type", "stock",
        )

        assert rc == 0
        assert _runner(rb).run_calls == [(["AAPL", "BTC-USD"], _DAY, "stock")]
        assert capsys.readouterr().out.splitlines()[:2] == [
            f"{_PLAY}  批量分析 2 个 ticker | date={_DAY} | type=stock",
            "",
        ]

    def test_r17_run_batch_main_runner_exception_is_not_swallowed(self, rb, monkeypatch, capsys):
        """A crash outside the per-ticker error rows propagates: no summary, no exit code."""
        def explode(*args, **kwargs):
            raise ValueError("uniformity guard")

        monkeypatch.setattr(_FakeRunner, "run", explode)

        with pytest.raises(ValueError, match="uniformity guard"):
            _batch_main(rb, monkeypatch, "--tickers", "AAPL", "--date", _DAY)

        events = _runner(rb).events
        assert events[0] == "enter"
        kind, exc = events[1]
        assert kind == "exit"
        assert exc[0] is ValueError and isinstance(exc[1], ValueError) and exc[2] is not None
        assert "批量完成" not in capsys.readouterr().out


def test_r17_run_batch_main_guard_exits_with_main_code(rb_mod, monkeypatch, tmp_path, capsys):
    """``python scripts/run_batch.py`` exits with ``main()``'s code on every path."""
    path = SCRIPTS_DIR / "run_batch.py"
    code = compile(path.read_text(encoding="utf-8"), str(path), "exec")
    # The script's own shim inserts ``scripts/`` into sys.path; undo that afterwards.
    monkeypatch.setattr(sys, "path", list(sys.path))

    def run_as_script(argv):
        _set_argv(monkeypatch, "run_batch.py", argv)
        with pytest.raises(SystemExit) as excinfo:
            exec(code, {"__name__": "__main__", "__file__": str(path)})  # noqa: S102
        return excinfo.value.code, capsys.readouterr().out.splitlines()

    # Bad arguments are rejected before any runner exists.
    outcome = run_as_script(["--tickers", "AAPL", "--date", "2026-9-1"])
    assert outcome == (2, [f"{_FAIL} Bad date (need YYYY-MM-DD): 2026-9-1"])

    # A failing ticker is exit 1; the real analyst selection reaches the runner.
    spy = _RunnerSpy([
        _row("AAPL", report=tmp_path / "AAPL_1"), _row("NVDA", error=RuntimeError("boom")),
    ])
    monkeypatch.setattr(batch_runner, "BatchRunner", spy)
    rc, out = run_as_script(["--tickers", "AAPL", "NVDA", "--date", _DAY, "--no-progress"])
    assert rc == 1
    assert out[-3:] == _table(
        _ok_row("AAPL", "1.5s", tmp_path / "AAPL_1"), _fail_row("NVDA", "RuntimeError: boom")
    )
    assert [(fake.workers, fake.progress, fake.selected_analysts) for fake in spy.built] == [
        (None, False, ("market", "social", "news", "fundamentals"))
    ]

    spy.rows = [_row("NVDA", report=tmp_path / "NVDA_1")]
    rc, out = run_as_script(["--tickers", "NVDA", "--date", _DAY])
    assert rc == 0
    assert out[-1] == _ok_row("NVDA", "1.5s", tmp_path / "NVDA_1")


class TestRunBatchMainObservations:
    """Current behaviour at the script / package seam, pinned as is (reported, not fixed)."""

    def test_r17_run_batch_main_padded_ticker_passes_validation_and_reaches_the_runner(
        self, rb, monkeypatch, capsys
    ):
        """``--tickers " AAPL "`` is accepted and forwarded with its spaces.

        The package validates the stripped value but hands the raw string on:
        ``YiAlphaGraph.propagate`` is entered with ``' AAPL '`` as typed, and
        the report directory name comes from ``safe_ticker_component``, which
        rejects it. The cleaning belongs to ``prepare_batch_run``, shared with
        ``yialpha batch``.
        """
        rb.rows.append(_row(" AAPL ", error=ValueError("ticker contains characters")))

        rc = _batch_main(rb, monkeypatch, "--tickers", " AAPL ", "--date", _DAY)

        assert rc == 1
        assert _runner(rb).run_calls == [([" AAPL "], _DAY, "stock")]
        out = capsys.readouterr().out.splitlines()
        assert out[0] == f"{_INFO}  资产类别自动判定：stock（按  AAPL ）"
        assert out[-1] == _fail_row(" AAPL ", "ValueError: ticker contains characters")

    @pytest.mark.parametrize(
        ("out", "results_dir"),
        [
            pytest.param("", "default", id="empty-is-ignored"),
            pytest.param("runs/here", "runs/here", id="relative-stays-relative"),
        ],
    )
    def test_r17_run_batch_main_out_verbatim(self, rb, monkeypatch, out, results_dir):
        """``--out`` is passed to the package untouched; an empty value falls back silently."""
        rb.rows.append(_row("AAPL"))

        rc = _batch_main(rb, monkeypatch, "--tickers", "AAPL", "--date", _DAY, "--out", out)

        assert rc == 0
        config = _runner(rb).config
        if results_dir == "default":
            assert config == DEFAULT_CONFIG
        else:
            assert config == {**DEFAULT_CONFIG, "results_dir": results_dir}


# =========================================================================== #
# scripts/run_baseline.py
# =========================================================================== #


@pytest.fixture
def rbl():
    """A fresh ``run_baseline.py`` module (the other tests import it as ``scripts.run_baseline``)."""
    return _load("r17_run_baseline", "run_baseline.py")


# --------------------------------------------------------------------------- #
# _rebalance_dates
# --------------------------------------------------------------------------- #


class TestRebalanceDates:
    def test_r17_rebalance_dates_picks_from_the_end_step_business_days_apart(self, rbl, capsys):
        """Six dates, the last one the final business day, each 10 business days earlier."""
        dates = rbl._rebalance_dates("2026-01-05", "2026-06-30", 10, 6)

        assert dates == ["2026-04-21", "2026-05-05", "2026-05-19", "2026-06-02", "2026-06-16", "2026-06-30"]
        captured = capsys.readouterr()
        assert (captured.out, captured.err) == ("", "")

    def test_r17_rebalance_dates_single_date_is_the_last_business_day(self, rbl, capsys):
        assert rbl._rebalance_dates("2026-01-05", "2026-06-30", 10, 1) == ["2026-06-30"]
        assert rbl._rebalance_dates("2026-01-05", "2026-06-27", 10, 1) == ["2026-06-26"]  # Saturday end
        assert capsys.readouterr().err == ""

    def test_r17_rebalance_dates_exact_fit_needs_no_warning(self, rbl, capsys):
        dates = rbl._rebalance_dates("2026-01-05", "2026-06-30", 3, 4)

        assert dates == ["2026-06-17", "2026-06-22", "2026-06-25", "2026-06-30"]
        assert capsys.readouterr().err == ""

    def test_r17_rebalance_dates_warns_when_fewer_than_n_fit(self, rbl, capsys):
        """Round-10 pin: 22 business days cannot hold 6 dates 10 days apart."""
        dates = rbl._rebalance_dates("2026-06-01", "2026-06-30", 10, 6)

        assert dates == ["2026-06-02", "2026-06-16", "2026-06-30"]
        assert capsys.readouterr().err == (
            "⚠ rebalance calendar shrunk: requested 6 dates ~10 business days apart over "
            "2026-06-01..2026-06-30 (22 business days) but only 3 fit; "
            "widening the window or lowering --step/--rebalance\n"
        )

    def test_r17_rebalance_dates_warns_when_the_window_has_fewer_than_n_business_days(self, rbl, capsys):
        """A window with 5 business days cannot yield 6 rebalance dates either.

        Round-17: this branch returned every business day in the window
        (consecutive days, no longer ``step`` apart) without a word, while the
        neighbouring branch above announces the same shrunken calendar.
        """
        dates = rbl._rebalance_dates("2026-01-05", "2026-01-09", 10, 6)

        assert dates == ["2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08", "2026-01-09"]
        assert capsys.readouterr().err == (
            "⚠ rebalance calendar shrunk: requested 6 dates ~10 business days apart over "
            "2026-01-05..2026-01-09 (5 business days) but only 5 fit; "
            "widening the window or lowering --step/--rebalance\n"
        )

    @pytest.mark.parametrize(
        ("start", "end"),
        [
            pytest.param("2026-06-30", "2026-06-01", id="start-after-end"),
            pytest.param("2026-01-03", "2026-01-04", id="weekend-only"),
        ],
    )
    def test_r17_rebalance_dates_empty_window_warns_too(self, rbl, capsys, start, end):
        """No business day in the window: no dates, and the same shrink warning.

        Round-17: this was the extreme of the unannounced branch (``0 fit``).
        The calendar is still returned empty; the engine rejects it per ticker.
        """
        assert rbl._rebalance_dates(start, end, 10, 6) == []
        assert capsys.readouterr().err == (
            f"⚠ rebalance calendar shrunk: requested 6 dates ~10 business days apart over "
            f"{start}..{end} (0 business days) but only 0 fit; "
            f"widening the window or lowering --step/--rebalance\n"
        )

    @pytest.mark.parametrize(
        ("start", "end"),
        [
            pytest.param("2026-1-5", "2026-3-5", id="not-zero-padded"),
            pytest.param("20260105", "20260305", id="compact"),
        ],
    )
    def test_r17_rebalance_dates_tolerant_input_canonical_output(self, rbl, capsys, start, end):
        """pandas parses the window; what leaves the function is ``YYYY-MM-DD`` either way."""
        dates = rbl._rebalance_dates(start, end, 5, 4)

        assert dates == ["2026-02-12", "2026-02-19", "2026-02-26", "2026-03-05"]
        assert capsys.readouterr().err == ""

    @pytest.mark.parametrize(
        ("start", "end"),
        [
            pytest.param("garbage", "2026-06-30", id="start-unparseable"),
            pytest.param("2026-01-05", "garbage", id="end-unparseable"),
            pytest.param("2026-02-30", "2026-03-05", id="impossible-day"),
        ],
    )
    def test_r17_rebalance_dates_unparseable_window_raises(self, rbl, start, end):
        with pytest.raises(ValueError):
            rbl._rebalance_dates(start, end, 10, 6)

    @pytest.mark.parametrize("step", [0, -5])
    def test_r17_rebalance_dates_rejects_step_below_one(self, rbl, step):
        with pytest.raises(SystemExit) as excinfo:
            rbl._rebalance_dates("2026-01-05", "2026-06-30", step, 6)

        assert str(excinfo.value) == (
            f"--step must be >= 1 (got {step}); step 0 would repeat the same rebalance date 6 times"
        )

    @pytest.mark.parametrize("n", [0, -1])
    def test_r17_rebalance_dates_non_positive_n_yields_nothing(self, rbl, capsys, n):
        """Direct call only: ``main()`` rejects ``--rebalance < 1`` before it gets here."""
        assert rbl._rebalance_dates("2026-01-05", "2026-06-30", 10, n) == []
        assert capsys.readouterr().err == ""


# --------------------------------------------------------------------------- #
# _build_graph
# --------------------------------------------------------------------------- #


@pytest.fixture
def graph_log(rbl, monkeypatch):
    """``YiAlphaGraph`` in the loaded script becomes a recorder of its constructor arguments."""
    log = []

    class _Graph:
        def __init__(self, selected_analysts, debug=False, config=None):
            log.append({
                "selected_analysts": tuple(selected_analysts),
                "debug": debug,
                "config": dict(config or {}),
            })
            self.config = config
            self.perf_tracker = None

    monkeypatch.setattr(rbl, "YiAlphaGraph", _Graph)
    return log


class TestBuildGraph:
    def test_r17_build_graph_defaults(self, rbl, graph_log):
        rbl._build_graph()

        assert graph_log == [{
            "selected_analysts": ("market", "social", "news", "fundamentals"),
            "debug": False,
            "config": {**DEFAULT_CONFIG, "memory_enabled": False},
        }]

    @pytest.mark.parametrize("risk_enabled", [True, False, None])
    def test_r17_build_graph_risk_flag(self, rbl, graph_log, risk_enabled):
        rbl._build_graph(risk_enabled=risk_enabled)

        config = graph_log[0]["config"]
        if risk_enabled is None:
            assert config == {**DEFAULT_CONFIG, "memory_enabled": False}
        else:
            assert config == {**DEFAULT_CONFIG, "memory_enabled": False, "risk_enabled": risk_enabled}

    def test_r17_build_graph_telemetry_and_debug_flags(self, rbl, graph_log):
        rbl._build_graph(debug=True, node_perf_telemetry=True)

        assert graph_log == [{
            "selected_analysts": ("market", "social", "news", "fundamentals"),
            "debug": True,
            "config": {**DEFAULT_CONFIG, "memory_enabled": False, "node_perf_telemetry": True},
        }]

    def test_r17_build_graph_memory_off_even_when_the_default_is_on(self, rbl, graph_log, monkeypatch):
        monkeypatch.setitem(DEFAULT_CONFIG, "memory_enabled", True)

        rbl._build_graph()

        assert graph_log[0]["config"]["memory_enabled"] is False
        assert DEFAULT_CONFIG["memory_enabled"] is True  # the default itself is left alone

    def test_r17_build_graph_does_not_mutate_the_default_config(self, rbl, graph_log):
        before = dict(DEFAULT_CONFIG)

        rbl._build_graph(risk_enabled=True, node_perf_telemetry=True)

        assert before == DEFAULT_CONFIG

    def test_r17_build_graph_stock_ticker_keeps_the_four_analysts(self, rbl, graph_log):
        rbl._build_graph(tickers=("AAPL", "MSFT"))

        assert graph_log[0]["selected_analysts"] == ("market", "social", "news", "fundamentals")

    def test_r17_build_graph_routes_tickers_through_the_shared_analyst_filter(
        self, rbl, graph_log, monkeypatch
    ):
        """With tickers the analyst set is whatever ``batch_selected_analysts`` says."""
        calls = []

        def select(asset_type, tickers):
            calls.append((asset_type, list(tickers)))
            return ("market", "positioning")

        monkeypatch.setattr(batch_runner, "batch_selected_analysts", select)

        rbl._build_graph(asset_type="crypto_perp", tickers=("BTCUSDT", "ETHUSDT"))

        assert calls == [("crypto_perp", ["BTCUSDT", "ETHUSDT"])]
        assert graph_log[0]["selected_analysts"] == ("market", "positioning")

    @pytest.mark.parametrize("tickers", [(), [], None])
    def test_r17_build_graph_empty_tickers_skip_the_filter(self, rbl, graph_log, monkeypatch, tickers):
        def never(*args):
            raise AssertionError("batch_selected_analysts must not be called")

        monkeypatch.setattr(batch_runner, "batch_selected_analysts", never)

        rbl._build_graph(asset_type="crypto_perp", tickers=tickers)

        assert graph_log[0]["selected_analysts"] == ("market", "social", "news", "fundamentals")


# --------------------------------------------------------------------------- #
# _print_perf_table
# --------------------------------------------------------------------------- #


def _perf_row(name, wall, pct, calls, tok_in, tok_out, tok_reason):
    """One ``--profile`` row: two-space indent, seven space-separated columns."""
    return "  " + " ".join(
        [name.ljust(28), wall, pct, calls, tok_in, tok_out, tok_reason]
    )


_PERF_HEADER = "  " + " ".join([
    "node".ljust(28),
    "wall(s)".rjust(9),
    "%".rjust(6),
    "calls".rjust(6),
    "tok_in".rjust(9),
    "tok_out".rjust(9),
    "tok_reason".rjust(11),
])
_PERF_TITLE = "=== 节点墙钟占比（--profile，仅观测） ==="


def _perf_total(wall, tok_in, tok_out, tok_reason):
    """The ``--profile`` totals row: no share column, no call count."""
    return "  " + "TOTAL".ljust(28) + " " + wall + " " * 13 + tok_in + " " + tok_out + " " + tok_reason


def _fed_tracker(*samples):
    """A real ``NodePerfTracker`` fed ``(node, wall_s, calls, in, out, reasoning)`` samples.

    The wall time is recorded in one sample and the remaining calls as
    zero-length ones, so the node's total is exactly ``wall_s``.
    """
    from yialpha.graph.perf_telemetry import NodePerfTracker

    tracker = NodePerfTracker()
    for node, wall, calls, tok_in, tok_out, tok_reason in samples:
        tracker.record(node, wall)
        for _ in range(calls - 1):
            tracker.record(node, 0.0)
        if tok_in or tok_out or tok_reason:
            tracker.record_tokens(
                node, input_tokens=tok_in, output_tokens=tok_out, reasoning_tokens=tok_reason
            )
    return tracker


class TestPrintPerfTable:
    def test_r17_perf_table_empty_tracker(self, rbl, capsys):
        rbl._print_perf_table(_fed_tracker())

        assert capsys.readouterr().out == "  (perf: 无节点耗时数据)\n"

    def test_r17_perf_table_sorted_by_wall_time_with_totals(self, rbl, capsys):
        tracker = _fed_tracker(
            ("Trader", 2.5, 1, 1500, 50, 0),
            ("Market Analyst", 12.5, 2, 3000, 400, 100),
        )

        rbl._print_perf_table(tracker)

        assert capsys.readouterr().out.splitlines() == [
            "",
            _PERF_TITLE,
            _PERF_HEADER,
            _perf_row("Market Analyst", "    12.50", " 83.3%", "     2", "     3000", "      400", "        100"),
            _perf_row("Trader", "     2.50", " 16.7%", "     1", "     1500", "       50", "          0"),
            _perf_total("    15.00", "     4500", "      450", "        100"),
        ]

    def test_r17_perf_table_zero_total_does_not_divide_by_zero(self, rbl, capsys):
        """Tokens recorded outside any timed node: wall total 0, every share 0.0 %."""
        from yialpha.graph.perf_telemetry import UNATTRIBUTED

        tracker = _fed_tracker()
        tracker.record_tokens(None, input_tokens=500, output_tokens=20)

        rbl._print_perf_table(tracker)

        assert capsys.readouterr().out.splitlines() == [
            "",
            _PERF_TITLE,
            _PERF_HEADER,
            _perf_row(UNATTRIBUTED, "     0.00", "  0.0%", "     0", "      500", "       20", "          0"),
            _perf_total("     0.00", "      500", "       20", "          0"),
        ]

    def test_r17_perf_table_equal_wall_times_keep_recording_order(self, rbl, capsys):
        tracker = _fed_tracker(("Bull", 4.0, 1, 0, 0, 0), ("Bear", 4.0, 1, 0, 0, 0), ("RM", 4.0, 1, 0, 0, 0))

        rbl._print_perf_table(tracker)

        out = capsys.readouterr().out.splitlines()
        assert [line[2:30].strip() for line in out[3:]] == ["Bull", "Bear", "RM", "TOTAL"]
        assert [line[31:40].strip() for line in out[3:6]] == ["4.00", "4.00", "4.00"]

    def test_r17_perf_table_long_names_and_big_numbers_keep_their_columns(self, rbl, capsys):
        tracker = _fed_tracker(
            ("x" * 40, 12345.678, 3, 123456789012, 987654321, 12345678901),
            ("Portfolio Manager", 0.004, 1, 0, 0, 0),
        )

        rbl._print_perf_table(tracker)

        out = capsys.readouterr().out.splitlines()
        assert out[3] == _perf_row(
            "x" * 40, " 12345.68", "100.0%", "     3", "123456789012", "987654321", "12345678901"
        )
        assert out[4] == _perf_row(
            "Portfolio Manager", "     0.00", "  0.0%", "     1", "        0", "        0", "          0"
        )
        assert out[5] == _perf_total(" 12345.68", "123456789012", "987654321", "12345678901")


# --------------------------------------------------------------------------- #
# _map_tickers
# --------------------------------------------------------------------------- #


@pytest.fixture
def built_graphs(rbl, monkeypatch):
    """``_build_graph`` in the loaded script becomes a recorder of its calls."""
    built = []

    def build(**kwargs):
        graph = SimpleNamespace(number=len(built), kwargs=kwargs)
        built.append(graph)
        return graph

    monkeypatch.setattr(rbl, "_build_graph", build)
    return built


def _counting_task(calls, **extra):
    """``task(ticker, graph)`` that records its calls and returns ``{"ticker", "n"}``.

    ``calls`` and ``n`` are taken under one lock: with a pool, two workers can be
    inside the task at the same time.
    """
    lock = threading.Lock()

    def task(ticker, graph):
        with lock:
            calls.append((ticker, graph.number))
            return {"ticker": ticker, "n": len(calls), **extra}

    return task


class TestMapTickers:
    @pytest.mark.parametrize(
        ("workers", "risk_enabled", "asset_type"),
        [
            pytest.param(1, None, "stock", id="serial-defaults"),
            pytest.param(1, False, "crypto_perp", id="serial-risk-off"),
            pytest.param(2, True, "crypto", id="pool-risk-on"),
        ],
    )
    def test_r17_map_tickers_builds_the_graphs_once_and_keeps_input_order(
        self, rbl, built_graphs, workers, risk_enabled, asset_type
    ):
        calls = []
        tickers = ["MSFT", "AAPL", "brk.b"]

        out = rbl._map_tickers(
            tickers, _counting_task(calls), workers, risk_enabled=risk_enabled,
            asset_type=asset_type,
        )

        assert len(built_graphs) == max(1, workers)
        for graph in built_graphs:
            assert graph.kwargs == {
                "debug": False, "risk_enabled": risk_enabled,
                "asset_type": asset_type, "tickers": tickers,
            }
        assert [row["ticker"] for row in out] == tickers  # the contract: results in input order
        if workers == 1:
            assert [t for t, _ in calls] == tickers
            assert [row["n"] for row in out] == [1, 2, 3]
            assert [g for _, g in calls] == [0, 0, 0]
        else:
            # With a pool, which worker thread picks which ticker is the
            # scheduler's choice (on a loaded Linux box this ran the third ticker
            # before the second in about one run in five); only the results are
            # ordered.
            assert sorted(t for t, _ in calls) == sorted(tickers)
            assert sorted(row["n"] for row in out) == [1, 2, 3]
            assert {g for _, g in calls} <= {graph.number for graph in built_graphs}

    @pytest.mark.parametrize("workers", [1, 0, -3])
    def test_r17_map_tickers_workers_below_one_take_the_serial_path(self, rbl, built_graphs, workers):
        calls = []

        out = rbl._map_tickers(["AAPL", "NVDA"], _counting_task(calls), workers)

        assert [graph.number for graph in built_graphs] == [0]
        assert [g for _, g in calls] == [0, 0]
        assert [row["ticker"] for row in out] == ["AAPL", "NVDA"]

    def test_r17_map_tickers_pool_borrows_a_graph_per_in_flight_ticker(self, rbl, built_graphs):
        """With K=2 both tickers are in flight at once, each on its own graph."""
        gate = threading.Barrier(2, timeout=30)
        using: dict[str, int] = {}
        lock = threading.Lock()

        def task(ticker, graph):
            with lock:
                using[ticker] = graph.number
            gate.wait()  # releases only when the other ticker is in flight too
            return ticker

        out = rbl._map_tickers(["AAPL", "MSFT"], task, 2)

        assert out == ["AAPL", "MSFT"]
        assert len(built_graphs) == 2
        assert sorted(using.values()) == [0, 1]

    def test_r17_map_tickers_pool_built_for_workers_even_with_one_ticker(self, rbl, built_graphs):
        calls = []

        out = rbl._map_tickers(["AAPL"], _counting_task(calls), 3)

        assert [graph.number for graph in built_graphs] == [0, 1, 2]
        assert out == [{"ticker": "AAPL", "n": 1}]
        assert calls[0][1] in {0, 1, 2}

    @pytest.mark.parametrize("workers", [1, 2])
    def test_r17_map_tickers_one_failing_ticker_does_not_abort_the_others(
        self, rbl, built_graphs, capsys, workers
    ):
        calls = []

        def task(ticker, graph):
            calls.append(ticker)
            if ticker == "MSFT":
                raise RuntimeError("vendor down")
            return [ticker]

        out = rbl._map_tickers(["AAPL", "MSFT", "NVDA"], task, workers)

        assert out == [["AAPL"], None, ["NVDA"]]
        # Each ticker ran exactly once. With a pool the start order is the
        # scheduler's choice, so AAPL is not necessarily ``calls[0]``.
        assert sorted(calls) == ["AAPL", "MSFT", "NVDA"]
        assert capsys.readouterr().err == "[MSFT] ❌ ticker failed, skipping: RuntimeError('vendor down')\n"

    def test_r17_map_tickers_failure_line_carries_the_exception_repr(self, rbl, built_graphs, capsys):
        class _Odd(Exception):
            def __repr__(self):
                return "<odd repr>"

        def task(ticker, graph):
            raise _Odd("message")

        out = rbl._map_tickers(["AAPL"], task, 1)

        assert out == [None]
        assert capsys.readouterr().err == "[AAPL] ❌ ticker failed, skipping: <odd repr>\n"

    @pytest.mark.parametrize("returned", [None, [], 0, "", {"ok": True}])
    @pytest.mark.parametrize("workers", [1, 2])
    def test_r17_map_tickers_falsy_results_are_returned_as_is(
        self, rbl, built_graphs, capsys, returned, workers
    ):
        """Only a raised task becomes the ``None`` placeholder."""
        out = rbl._map_tickers(["AAPL", "NVDA"], lambda t, g: returned, workers)

        assert out == [returned, returned]
        assert capsys.readouterr().err == ""

    @pytest.mark.parametrize("workers", [1, 2])
    def test_r17_map_tickers_keyboard_interrupt_is_not_isolated(self, rbl, built_graphs, workers):
        def task(ticker, graph):
            raise KeyboardInterrupt

        with pytest.raises(KeyboardInterrupt):
            rbl._map_tickers(["AAPL"], task, workers)

    @pytest.mark.parametrize("workers", [1, 2])
    @pytest.mark.parametrize(
        ("tickers", "kept", "dropped"),
        [
            pytest.param(["AAPL", "MSFT", "AAPL"], ["AAPL", "MSFT"], ["AAPL"], id="exact-duplicate"),
            pytest.param(["AAPL", "aapl"], ["AAPL"], ["aapl"], id="case-variant"),
            pytest.param(["MSFT", " MSFT ", "msft"], ["MSFT"], [" MSFT ", "msft"], id="padded-and-lower"),
        ],
    )
    def test_r17_map_tickers_duplicate_tickers_run_once_in_both_paths(
        self, rbl, built_graphs, capsys, workers, tickers, kept, dropped
    ):
        """Round-17: a ticker named twice must be analyzed once, and the drop must be said.

        The pool path keyed its results by ticker, so ``["AAPL", "MSFT",
        "AAPL"]`` ran AAPL twice (two full LLM backtests racing on the same
        ticker's state logs and cache) and handed the caller the second
        AAPL result twice while the first was discarded -- the serial path
        returned two distinct results. The reports then counted one run
        twice. Same rule as ``BatchRunner._dedup``.
        """
        calls = []

        out = rbl._map_tickers(tickers, _counting_task(calls), workers)

        if workers == 1:
            assert [t for t, _ in calls] == kept
            assert out == [{"ticker": t, "n": i + 1} for i, t in enumerate(kept)]
        else:
            # With a pool the start order is the scheduler's choice (see
            # ``..._keeps_input_order``); what must hold is one run per kept
            # ticker and results in input order.
            assert sorted(t for t, _ in calls) == sorted(kept)
            assert [row["ticker"] for row in out] == kept
            assert sorted(row["n"] for row in out) == list(range(1, len(kept) + 1))
        assert capsys.readouterr().err == (
            f"⚠ removed {len(tickers) - len(kept)} duplicate ticker(s) from the batch: "
            f"{dropped} (each ticker is run once)\n"
        )
        assert built_graphs[0].kwargs["tickers"] == kept


# --------------------------------------------------------------------------- #
# preflight()
# --------------------------------------------------------------------------- #


def _health_spy(monkeypatch, reports):
    """Replace the shared health check (lazy-imported from its source module) by a spy."""
    import yialpha.monitoring.preflight as preflight_module

    calls = []

    def run_health(ticker):
        calls.append(ticker)
        return reports

    monkeypatch.setattr(preflight_module, "run_health", run_health)
    return calls


_ALL_GREEN = {
    "ok": True,
    "checks": [
        {"name": "dep socks", "ok": True, "hint": ""},
        {"name": "DEEPSEEK_API_KEY set", "ok": True, "hint": ""},
        {"name": "proxy 127.0.0.1:8086 reachable", "ok": True, "hint": ""},
    ],
}


class TestPreflight:
    def test_r17_preflight_all_green_exit_0(self, rbl, monkeypatch, capsys):
        calls = _health_spy(monkeypatch, _ALL_GREEN)

        rc = rbl.preflight("AAPL")

        assert rc == 0
        assert calls == ["AAPL"]
        assert capsys.readouterr().out.splitlines() == [
            "",
            "=== 起飞检查（preflight）：AAPL ===",
            f"  {_OK} dep socks",
            f"  {_OK} DEEPSEEK_API_KEY set",
            f"  {_OK} proxy 127.0.0.1:8086 reachable",
            "",
            f"  → {_OK} 全部通过，可以跑 --smoke",
        ]

    @pytest.mark.parametrize("ticker", ["0700.HK", "btc-usd", "SPY"])
    def test_r17_preflight_passes_the_ticker_through_ungated(self, rbl, monkeypatch, capsys, ticker):
        """No market or charset gate here: the probe decides what it can check."""
        calls = _health_spy(monkeypatch, _ALL_GREEN)

        rc = rbl.preflight(ticker)

        assert rc == 0
        assert calls == [ticker]
        assert capsys.readouterr().out.splitlines()[1] == f"=== 起飞检查（preflight）：{ticker} ==="

    def test_r17_preflight_one_red_exit_2_and_hints_only_on_red_rows(self, rbl, monkeypatch, capsys):
        _health_spy(monkeypatch, {
            "ok": False,
            "checks": [
                {"name": "dep socks", "ok": True, "hint": "never shown on a green row"},
                {"name": "proxy 127.0.0.1:8086 reachable", "ok": False, "hint": "start the VPN"},
                {"name": "yfinance SPY 5d", "ok": False},  # no hint key at all
                {"name": "DeepSeek /v1/models", "ok": False, "hint": ""},  # empty hint
            ],
        })

        rc = rbl.preflight("SPY")

        assert rc == 2
        assert capsys.readouterr().out.splitlines() == [
            "",
            "=== 起飞检查（preflight）：SPY ===",
            f"  {_OK} dep socks",
            f"  {_FAIL} proxy 127.0.0.1:8086 reachable  → start the VPN",
            f"  {_FAIL} yfinance SPY 5d",
            f"  {_FAIL} DeepSeek /v1/models",
            "",
            f"  → {_FAIL} 有未通过项，先修上面 ❌ 再跑 --smoke",
        ]

    def test_r17_preflight_verdict_follows_the_report_flag_not_the_rows(self, rbl, monkeypatch, capsys):
        """``ok`` is taken as given: one red row under ``ok: True`` still exits 0."""
        _health_spy(monkeypatch, {"ok": True, "checks": [{"name": "odd", "ok": False, "hint": "x"}]})

        rc = rbl.preflight("SPY")

        assert rc == 0
        assert capsys.readouterr().out.splitlines()[-3:] == [
            f"  {_FAIL} odd  → x",
            "",
            f"  → {_OK} 全部通过，可以跑 --smoke",
        ]

    def test_r17_preflight_no_checks_at_all_is_green(self, rbl, monkeypatch, capsys):
        _health_spy(monkeypatch, {"ok": True, "checks": []})

        rc = rbl.preflight("SPY")

        assert rc == 0
        assert capsys.readouterr().out.splitlines() == [
            "",
            "=== 起飞检查（preflight）：SPY ===",
            "",
            f"  → {_OK} 全部通过，可以跑 --smoke",
        ]

    def test_r17_preflight_health_check_crash_propagates(self, rbl, monkeypatch, capsys):
        import yialpha.monitoring.preflight as preflight_module

        def boom(ticker):
            raise OSError("probe exploded")

        monkeypatch.setattr(preflight_module, "run_health", boom)

        with pytest.raises(OSError, match="probe exploded"):
            rbl.preflight("SPY")

        assert capsys.readouterr().out.splitlines() == ["", "=== 起飞检查（preflight）：SPY ==="]


# --------------------------------------------------------------------------- #
# smoke()
# --------------------------------------------------------------------------- #


@pytest.fixture
def smoke_stage(rbl, monkeypatch):
    """``YiAlphaGraph`` double for ``smoke()``: records the build, replays a scripted decision."""
    stage = SimpleNamespace(
        state={"final_trade_decision": "**Rating**: Buy\n\nThesis: demand."},
        rating="Buy",
        tracker=None,
        build_error=None,
        propagate_error=None,
        built=[],
        propagate_calls=[],
    )

    class _Graph:
        def __init__(self, selected_analysts, debug=False, config=None):
            if stage.build_error is not None:
                raise stage.build_error
            stage.built.append({
                "selected_analysts": tuple(selected_analysts),
                "debug": debug,
                "config": dict(config or {}),
            })
            self.perf_tracker = stage.tracker

        def propagate(self, ticker, date, asset_type="stock"):
            stage.propagate_calls.append((ticker, date, asset_type))
            if stage.propagate_error is not None:
                raise stage.propagate_error
            return stage.state, stage.rating

    monkeypatch.setattr(rbl, "YiAlphaGraph", _Graph)
    return stage


_SMOKE_HINTS = "排查：1) 代理是否通  2) DEEPSEEK_API_KEY 是否有效  3) yfinance 能否拉到该票该日数据"
_SMOKE_OK = f"{_OK} 链路打通，LLM/网络/key 都正常。可以进 --baseline。"


def _smoke_header(ticker, date, profile, asset_type):
    return f"=== 冒烟测试：{ticker} @ {date}{' + --profile' if profile else ''} [{asset_type}] ==="


class TestSmoke:
    def test_r17_smoke_success_exit_0_and_graph_arguments(self, rbl, smoke_stage, capsys):
        rc = rbl.smoke("AAPL", "2026-03-15")

        assert rc == 0
        assert smoke_stage.built == [{
            "selected_analysts": ("market", "social", "news", "fundamentals"),
            "debug": False,
            "config": {**DEFAULT_CONFIG, "memory_enabled": False},
        }]
        assert smoke_stage.propagate_calls == [("AAPL", "2026-03-15", "stock")]
        assert capsys.readouterr().out.splitlines() == [
            "",
            _smoke_header("AAPL", "2026-03-15", False, "stock"),
            "评级: Buy",
            "决策摘要（前 400 字）:",
            "**Rating**: Buy",
            "",
            "Thesis: demand.",
            "",
            _SMOKE_OK,
        ]

    @pytest.mark.parametrize("asset_type", ["stock", "crypto", "crypto_spot", "crypto_perp"])
    def test_r17_smoke_threads_asset_type_to_header_graph_and_propagate(
        self, rbl, smoke_stage, monkeypatch, capsys, asset_type
    ):
        """Round-5 pin, seen end to end: the perp smoke route must not run the stock pipeline."""
        selected = []

        def select(at, tickers):
            selected.append((at, list(tickers)))
            return ("market", "sentinel")

        monkeypatch.setattr(batch_runner, "batch_selected_analysts", select)

        rc = rbl.smoke("BTCUSDT", "2026-03-15", asset_type=asset_type)

        assert rc == 0
        assert selected == [(asset_type, ["BTCUSDT"])]
        assert smoke_stage.built[0]["selected_analysts"] == ("market", "sentinel")
        assert smoke_stage.propagate_calls == [("BTCUSDT", "2026-03-15", asset_type)]
        assert capsys.readouterr().out.splitlines()[1] == _smoke_header("BTCUSDT", "2026-03-15", False, asset_type)

    def test_r17_smoke_profile_turns_telemetry_on_and_prints_the_table(self, rbl, smoke_stage, capsys):
        smoke_stage.tracker = _fed_tracker(("Market Analyst", 3.0, 1, 100, 10, 0))

        rc = rbl.smoke("AAPL", "2026-03-15", profile=True)

        assert rc == 0
        assert smoke_stage.built[0]["config"] == {
            **DEFAULT_CONFIG, "memory_enabled": False, "node_perf_telemetry": True,
        }
        assert capsys.readouterr().out.splitlines() == [
            "",
            _smoke_header("AAPL", "2026-03-15", True, "stock"),
            "评级: Buy",
            "决策摘要（前 400 字）:",
            "**Rating**: Buy",
            "",
            "Thesis: demand.",
            "",
            _PERF_TITLE,
            _PERF_HEADER,
            _perf_row("Market Analyst", "     3.00", "100.0%", "     1", "      100", "       10", "          0"),
            _perf_total("     3.00", "      100", "       10", "          0"),
            "",
            _SMOKE_OK,
        ]

    def test_r17_smoke_profile_without_a_tracker_prints_no_table(self, rbl, smoke_stage, capsys):
        """``perf_tracker`` stays ``None`` when the graph was not built with telemetry."""
        rc = rbl.smoke("AAPL", "2026-03-15", profile=True)

        assert rc == 0
        assert smoke_stage.built[0]["config"]["node_perf_telemetry"] is True
        out = capsys.readouterr().out.splitlines()
        assert out[1] == _smoke_header("AAPL", "2026-03-15", True, "stock")
        assert _PERF_TITLE not in out

    def test_r17_smoke_tracker_present_but_profile_off_prints_no_table(self, rbl, smoke_stage, capsys):
        smoke_stage.tracker = _fed_tracker(("Market Analyst", 3.0, 1, 100, 10, 0))

        rc = rbl.smoke("AAPL", "2026-03-15", profile=False)

        assert rc == 0
        assert smoke_stage.built[0]["config"] == {**DEFAULT_CONFIG, "memory_enabled": False}
        assert _PERF_TITLE not in capsys.readouterr().out.splitlines()

    @pytest.mark.parametrize(
        ("decision", "printed"),
        [
            pytest.param("x" * 400, "x" * 400, id="exactly-400-kept"),
            pytest.param("y" * 401, "y" * 400, id="401-cut-at-400"),
            pytest.param("", "", id="empty"),
        ],
    )
    def test_r17_smoke_decision_is_cut_at_400_characters(self, rbl, smoke_stage, capsys, decision, printed):
        smoke_stage.state = {"final_trade_decision": decision}

        rc = rbl.smoke("AAPL", "2026-03-15")

        assert rc == 0
        out = capsys.readouterr().out.splitlines()
        assert out[2:5] == ["评级: Buy", "决策摘要（前 400 字）:", printed]

    @pytest.mark.parametrize("state", [None, {}, {"final_trade_decision": None}, {"other": 1}])
    def test_r17_smoke_missing_decision_prints_an_empty_excerpt(self, rbl, smoke_stage, capsys, state):
        smoke_stage.state = state

        rc = rbl.smoke("AAPL", "2026-03-15")

        assert rc == 0
        assert capsys.readouterr().out.splitlines()[2:5] == ["评级: Buy", "决策摘要（前 400 字）:", ""]

    def test_r17_smoke_rating_is_printed_as_is(self, rbl, smoke_stage, capsys):
        smoke_stage.rating = None

        rc = rbl.smoke("AAPL", "2026-03-15")

        assert rc == 0
        assert capsys.readouterr().out.splitlines()[2] == "评级: None"

    @pytest.mark.parametrize(
        "exc",
        [
            pytest.param(RuntimeError("no price data"), id="runtime-error"),
            pytest.param(UnsupportedMarketError("0700.HK", "only US-listed ..."), id="unsupported-market"),
            pytest.param(ValueError("time data '2026-13-45' ..."), id="value-error"),
        ],
    )
    def test_r17_smoke_propagate_failure_exit_2_with_hints(self, rbl, smoke_stage, capsys, exc):
        smoke_stage.propagate_error = exc

        rc = rbl.smoke("0700.HK", "2026-03-15")

        assert rc == 2
        captured = capsys.readouterr()
        assert captured.out.splitlines() == ["", _smoke_header("0700.HK", "2026-03-15", False, "stock")]
        assert captured.err.splitlines() == [
            "",
            f"{_FAIL} 冒烟失败：{exc}",
            _SMOKE_HINTS,
        ]

    def test_r17_smoke_graph_build_failure_exit_2_without_calling_propagate(self, rbl, smoke_stage, capsys):
        smoke_stage.build_error = RuntimeError("no api key")

        rc = rbl.smoke("AAPL", "2026-03-15", profile=True)

        assert rc == 2
        assert smoke_stage.propagate_calls == []
        captured = capsys.readouterr()
        assert captured.out.splitlines() == ["", _smoke_header("AAPL", "2026-03-15", True, "stock")]
        assert captured.err.splitlines() == ["", f"{_FAIL} 冒烟失败：no api key", _SMOKE_HINTS]

    def test_r17_smoke_keyboard_interrupt_is_not_caught(self, rbl, smoke_stage):
        smoke_stage.propagate_error = KeyboardInterrupt()

        with pytest.raises(KeyboardInterrupt):
            rbl.smoke("AAPL", "2026-03-15")


# --------------------------------------------------------------------------- #
# baseline_backtest() / full_ab()
# --------------------------------------------------------------------------- #


def _metrics(**overrides):
    base = {
        "total_return": 0.10,
        "sharpe": 1.0,
        "max_drawdown": -0.05,
        "alpha_vs_buyhold": 0.03,
        "deflated_sharpe": 0.6,
    }
    return SimpleNamespace(**{**base, **overrides})


@pytest.fixture
def ab(rbl, monkeypatch, tmp_path):
    """Everything the two backtest modes call below ``_map_tickers`` becomes a recorder.

    ``_map_tickers`` itself stays real (pool, isolation and de-duplication are
    its own tests); ``_build_graph`` returns numbered stand-ins.
    """
    stage = SimpleNamespace(
        out=tmp_path / "bt_out",
        dates=["2024-01-02", "2024-01-16"],
        metrics={},      # ticker -> metrics overrides
        failing=set(),   # tickers whose backtest raises
        verdicts={},     # ticker -> verdict overrides
        backtests=[], graphs=[], managers=[], weight_fns=[], gates=[], reports=[], dashboards=[],
    )
    stage.report_path = tmp_path / "backtest_report.md"

    def build(**kwargs):
        graph = SimpleNamespace(number=len(stage.graphs), kwargs=kwargs)
        stage.graphs.append(graph)
        return graph

    def fake_backtest(graph, ticker, dates, **kwargs):
        call = {
            "seq": len(stage.backtests), "graph": graph.number, "ticker": ticker,
            "dates": list(dates), **kwargs,
        }
        stage.backtests.append(call)
        if ticker in stage.failing:
            raise ValueError(f"no price data for {ticker}")
        return SimpleNamespace(
            ticker=ticker, metrics=_metrics(**stage.metrics.get(ticker, {})),
            run_tag=kwargs["run_tag"], seq=call["seq"],
        )

    def fake_from_config(config):
        manager = SimpleNamespace(number=len(stage.managers), config=dict(config))
        stage.managers.append(manager)
        return manager

    def fake_weight_fn(manager, ticker):
        def weight(rating, date, ctx, _who=(manager.number, ticker)):
            return 0.05

        stage.weight_fns.append((manager.number, ticker, weight))
        return weight

    def fake_gate(baselines, improved):
        ticker = baselines[0].ticker
        verdict = SimpleNamespace(
            **{
                "passes": True, "mean_dsr": 0.6, "beats_buyhold": True,
                "recommendation": "ok", **stage.verdicts.get(ticker, {}),
            }
        )
        verdict.render = lambda _t=ticker: f"# gate {_t}"
        stage.gates.append((ticker, [b.seq for b in baselines], [i.seq for i in improved]))
        return verdict

    def fake_report(results, results_dir):
        stage.reports.append((list(results), results_dir))
        return stage.report_path

    def fake_dashboard(results, results_dir, kill_switch=True):
        stage.dashboards.append((list(results), results_dir, kill_switch))

    monkeypatch.setattr(rbl, "_build_graph", build)
    monkeypatch.setattr(rbl, "_rebalance_dates", lambda *args: list(stage.dates))
    monkeypatch.setattr(rbl, "run_backtest", fake_backtest)
    monkeypatch.setattr(rbl, "write_report", fake_report)
    monkeypatch.setattr(rbl, "write_dashboard", fake_dashboard)
    monkeypatch.setattr(rbl.RiskManager, "from_config", fake_from_config)
    monkeypatch.setattr(rbl, "build_backtest_weight_fn", fake_weight_fn)
    monkeypatch.setattr(rbl, "evaluate_gate", fake_gate)
    return stage


_BASE_LINE = "总收益 {total_return:.2%} | Sharpe {sharpe:.2f} | MDD {max_drawdown:.2%} | vs B&H alpha (ann.) {alpha_vs_buyhold:.2%}"
_BASE_OK = _BASE_LINE.format(total_return=0.10, sharpe=1.0, max_drawdown=-0.05, alpha_vs_buyhold=0.03)


class TestBaselineBacktest:
    def test_r17_baseline_backtest_all_ok_report_dashboard_and_engine_calls(self, ab, rbl, capsys):
        ok = rbl.baseline_backtest(
            ["AAPL", "NVDA"], "2024-01-01", "2024-02-01", 10, 2, 5, 1.5, 2, str(ab.out)
        )

        assert ok is True
        assert [g.kwargs for g in ab.graphs] == [{
            "debug": False, "risk_enabled": False, "asset_type": "stock",
            "tickers": ["AAPL", "NVDA"],
        }]
        expected_calls = [
            {"seq": i, "graph": 0, "ticker": t, "dates": ab.dates, "holding_days": 5, "cost_bps": 1.5,
             "run_tag": f"base_r{r}", "factor_model": "3", "asset_type": "stock"}
            for i, (t, r) in enumerate((t, r) for t in ("AAPL", "NVDA") for r in (0, 1))
        ]
        assert ab.backtests == expected_calls
        assert len(ab.reports) == 1 and len(ab.reports[0][0]) == 4
        assert [(r.ticker, r.run_tag) for r in ab.reports[0][0]] == [
            ("AAPL", "base_r0"), ("AAPL", "base_r1"), ("NVDA", "base_r0"), ("NVDA", "base_r1"),
        ]
        assert ab.reports[0][1] == str(ab.out)
        assert len(ab.dashboards) == 1
        assert [r.seq for r in ab.dashboards[0][0]] == [0, 1, 2, 3]
        assert ab.dashboards[0][1] == str(ab.out)
        captured = capsys.readouterr()
        assert captured.err == ""
        assert captured.out.splitlines() == [
            "",
            "=== 基线回测：['AAPL', 'NVDA'] | 2024-01-01→2024-02-01 | 2×再平衡 ×2 次 ===",
            "  再平衡日期 ['2024-01-02', '2024-01-16']",
            "    [AAPL] run 0 ...",
            f"      [AAPL] {_BASE_OK}",
            "    [AAPL] run 1 ...",
            f"      [AAPL] {_BASE_OK}",
            "    [NVDA] run 0 ...",
            f"      [NVDA] {_BASE_OK}",
            "    [NVDA] run 1 ...",
            f"      [NVDA] {_BASE_OK}",
            "",
            f"📊 基线报告: {ab.report_path}",
            "📊 仪表盘(用浏览器打开): "
            + str(Path(str(ab.out)) / "monitoring" / "dashboard.html"),
        ]

    def test_r17_baseline_backtest_workers_two_build_a_pool_and_keep_report_order(
        self, ab, rbl, capsys
    ):
        ok = rbl.baseline_backtest(
            ["AAPL", "NVDA", "MSFT"], "2024-01-01", "2024-02-01", 10, 2, 5, 5.0, 1, str(ab.out),
            workers=2,
        )

        assert ok is True
        assert [g.number for g in ab.graphs] == [0, 1]
        assert {c["graph"] for c in ab.backtests} <= {0, 1}
        assert [(r.ticker, r.run_tag) for r in ab.reports[0][0]] == [
            ("AAPL", "base_r0"), ("NVDA", "base_r0"), ("MSFT", "base_r0"),
        ]

    @pytest.mark.parametrize(
        ("perp_kwargs", "expected"),
        [
            pytest.param(None, {}, id="none-is-empty"),
            pytest.param({}, {}, id="empty"),
            pytest.param(
                {"leverage": 3.0, "allow_short": True, "slippage_bps": 2.5},
                {"leverage": 3.0, "allow_short": True, "slippage_bps": 2.5}, id="perp-knobs",
            ),
        ],
    )
    def test_r17_baseline_backtest_perp_kwargs_reach_every_engine_call(
        self, ab, rbl, perp_kwargs, expected
    ):
        ok = rbl.baseline_backtest(
            ["BTCUSDT"], "2024-01-01", "2024-02-01", 10, 2, 5, 5.0, 2, str(ab.out),
            asset_type="crypto_perp", perp_kwargs=perp_kwargs,
        )

        assert ok is True
        assert ab.graphs[0].kwargs["asset_type"] == "crypto_perp"
        assert len(ab.backtests) == 2
        for call in ab.backtests:
            assert call["asset_type"] == "crypto_perp"
            assert {k: call[k] for k in expected} == expected
            assert set(call) == {
                "seq", "graph", "ticker", "dates", "holding_days", "cost_bps", "run_tag",
                "factor_model", "asset_type", *expected,
            }

    @pytest.mark.parametrize(
        ("overrides", "line"),
        [
            pytest.param(
                {"total_return": -0.1234, "sharpe": -0.5, "max_drawdown": -0.4, "alpha_vs_buyhold": -0.02},
                "总收益 -12.34% | Sharpe -0.50 | MDD -40.00% | vs B&H alpha (ann.) -2.00%",
                id="negative",
            ),
            pytest.param(
                {"total_return": float("nan"), "sharpe": float("inf"), "max_drawdown": float("nan"),
                 "alpha_vs_buyhold": 0.0},
                "总收益 nan% | Sharpe inf | MDD nan% | vs B&H alpha (ann.) 0.00%",
                id="non-finite-is-printed-not-raised",
            ),
        ],
    )
    def test_r17_baseline_backtest_metric_line_formatting(self, ab, rbl, capsys, overrides, line):
        ab.metrics["AAPL"] = overrides

        ok = rbl.baseline_backtest(["AAPL"], "2024-01-01", "2024-02-01", 10, 1, 5, 5.0, 1, str(ab.out))

        assert ok is True
        assert capsys.readouterr().out.splitlines()[4] == f"      [AAPL] {line}"

    def test_r17_baseline_backtest_one_ticker_fails_report_still_written_and_false_returned(
        self, ab, rbl, capsys
    ):
        ab.failing = {"NVDA"}

        ok = rbl.baseline_backtest(
            ["AAPL", "NVDA", "MSFT"], "2024-01-01", "2024-02-01", 10, 1, 5, 5.0, 1, str(ab.out)
        )

        assert ok is False
        assert [r.ticker for r in ab.reports[0][0]] == ["AAPL", "MSFT"]
        assert len(ab.dashboards) == 1
        captured = capsys.readouterr()
        assert captured.err.splitlines() == [
            "[NVDA] ❌ ticker failed, skipping: ValueError('no price data for NVDA')",
            "⚠️ 1/3 ticker 失败（已跳过，不影响其余报告）。",
        ]
        assert f"📊 基线报告: {ab.report_path}" in captured.out
        assert "NVDA] run 0" in captured.out  # the header line was printed before the failure

    def test_r17_baseline_backtest_every_ticker_fails_writes_nothing(self, ab, rbl, capsys):
        ab.failing = {"AAPL", "NVDA"}

        ok = rbl.baseline_backtest(
            ["AAPL", "NVDA"], "2024-01-01", "2024-02-01", 10, 1, 5, 5.0, 1, str(ab.out)
        )

        assert ok is False
        assert ab.reports == [] and ab.dashboards == []
        assert not ab.out.exists()
        captured = capsys.readouterr()
        assert "📊" not in captured.out
        assert captured.err.splitlines()[-1] == "❌ 基线回测：所有 ticker 均失败，无报告可写。"

    def test_r17_baseline_backtest_duplicate_tickers_are_dropped_before_any_engine_call(
        self, ab, rbl, capsys
    ):
        ok = rbl.baseline_backtest(
            ["AAPL", "NVDA", "AAPL"], "2024-01-01", "2024-02-01", 10, 1, 5, 5.0, 2, str(ab.out)
        )

        assert ok is True
        assert [c["ticker"] for c in ab.backtests] == ["AAPL", "AAPL", "NVDA", "NVDA"]
        assert [r.ticker for r in ab.reports[0][0]] == ["AAPL", "AAPL", "NVDA", "NVDA"]
        assert capsys.readouterr().err.splitlines() == [
            "⚠ removed 1 duplicate ticker(s) from the batch: ['AAPL'] (each ticker is run once)"
        ]


_AB_OK_BASE = "配对 run0 基线: 总收益 10.00% | Sharpe 1.00 | MDD -5.00%"
_AB_OK_RISK = "风控 run0: 总收益 10.00% | Sharpe 1.00 | MDD -5.00% | DSR 0.60"


class TestFullAb:
    def test_r17_full_ab_pass_writes_gate_report_and_dashboard(self, ab, rbl, capsys):
        ok = rbl.full_ab(["AAPL"], "2024-01-01", "2024-02-01", 10, 2, 5, 1.5, 1, str(ab.out))

        assert ok is True
        assert (ab.out / "gate_AAPL.md").read_text(encoding="utf-8") == "# gate AAPL"
        assert ab.gates == [("AAPL", [0], [1])]
        assert [g.kwargs for g in ab.graphs] == [{
            "debug": False, "risk_enabled": False, "asset_type": "stock", "tickers": ["AAPL"],
        }]
        assert len(ab.reports) == 1
        assert [(r.ticker, r.run_tag, r.seq) for r in ab.reports[0][0]] == [
            ("AAPL", "ab_pair_AAPL_r0", 0), ("AAPL", "ab_pair_AAPL_r0", 1),
        ]
        assert ab.reports[0][1] == str(ab.out)
        assert ab.dashboards == [([ab.reports[0][0][0]], str(ab.out), False)]
        assert ab.dashboards[0][0][0].run_tag == "ab_pair_AAPL_r0"
        captured = capsys.readouterr()
        assert captured.err == ""
        assert captured.out.splitlines() == [
            "",
            "=== 完整 A/B：基线 vs Phase-1 风控 | ['AAPL'] ===",
            "  再平衡日期 ['2024-01-02', '2024-01-16']",
            "",
            "  [AAPL] 开始",
            f"    [AAPL] {_AB_OK_BASE}",
            f"    [AAPL] {_AB_OK_RISK}",
            "",
            f"  [AAPL] 闸门判定: {_OK} PASS | DSR 0.60 | 跑赢B&H True",
            "    建议: ok...",
            "",
            f"📊 报告/仪表盘/闸门判定 都写到: {ab.out}",
        ]

    def test_r17_full_ab_each_run_pairs_one_tape_with_a_fresh_risk_manager(self, ab, rbl):
        ok = rbl.full_ab(
            ["AAPL", "MSFT"], "2024-01-01", "2024-02-01", 10, 2, 3, 5.0, 2, str(ab.out)
        )

        assert ok is True
        assert len(ab.backtests) == 8
        base, risk = ab.backtests[0::2], ab.backtests[1::2]
        # one ticker at a time; inside a ticker each baseline leg is followed by its replay
        assert [(c["ticker"], c["run_tag"]) for c in base] == [
            ("AAPL", "ab_pair_AAPL_r0"), ("AAPL", "ab_pair_AAPL_r1"),
            ("MSFT", "ab_pair_MSFT_r0"), ("MSFT", "ab_pair_MSFT_r1"),
        ]
        # same ticker+run share one run_tag and one cache object; weights only on the second leg
        for first, second in zip(base, risk, strict=True):
            assert first["run_tag"] == second["run_tag"]
            assert first["cache"] is second["cache"]
            assert "weight_fn" not in first and callable(second["weight_fn"])
            assert first["n_trials"] == second["n_trials"] == 2
            assert first["factor_model"] == second["factor_model"] == "3"
            assert (first["holding_days"], first["cost_bps"], first["dates"]) == (
                3, 5.0, ab.dates,
            )
            assert second["holding_days"] == 3 and second["cost_bps"] == 5.0
        caches = {id(c["cache"]) for c in ab.backtests}
        assert len(caches) == 1
        cache = ab.backtests[0]["cache"]
        assert cache.enabled is True and cache.cache_dir is not None
        assert not cache.cache_dir.exists()  # the decision tape dies with the run
        assert [m.number for m in ab.managers] == list(range(4))
        assert all(
            m.config == {**DEFAULT_CONFIG, "risk_enabled": True} for m in ab.managers
        )
        assert len({id(m) for m in ab.managers}) == 4
        assert [(n, t) for n, t, _ in ab.weight_fns] == [
            (0, "AAPL"), (1, "AAPL"), (2, "MSFT"), (3, "MSFT"),
        ]
        assert [g[0] for g in ab.gates] == ["AAPL", "MSFT"]
        assert [len(b) for _, b, _ in ab.gates] == [2, 2]
        assert all(len(i) == 2 for _, _, i in ab.gates)

    def test_r17_full_ab_no_graph_level_risk_on_either_leg(self, ab, rbl):
        rbl.full_ab(["AAPL"], "2024-01-01", "2024-02-01", 10, 1, 5, 5.0, 1, str(ab.out))

        assert [g.kwargs["risk_enabled"] for g in ab.graphs] == [False]

    def test_r17_full_ab_one_ticker_fails_the_others_are_still_gated(self, ab, rbl, capsys):
        ab.failing = {"MSFT"}

        ok = rbl.full_ab(
            ["AAPL", "MSFT", "NVDA"], "2024-01-01", "2024-02-01", 10, 1, 5, 5.0, 1, str(ab.out)
        )

        assert ok is False
        assert sorted(p.name for p in ab.out.iterdir()) == ["gate_AAPL.md", "gate_NVDA.md"]
        assert [g[0] for g in ab.gates] == ["AAPL", "NVDA"]
        assert [r.ticker for r in ab.reports[0][0]] == ["AAPL", "AAPL", "NVDA", "NVDA"]
        assert [r.ticker for r in ab.dashboards[0][0]] == ["AAPL", "NVDA"]
        captured = capsys.readouterr()
        assert captured.err.splitlines() == [
            "[MSFT] ❌ ticker failed, skipping: ValueError('no price data for MSFT')"
        ]
        assert "闸门判定" not in "\n".join(
            line for line in captured.out.splitlines() if "MSFT" in line
        )

    def test_r17_full_ab_gate_fail_still_writes_the_gate_file_and_returns_false(self, ab, rbl, capsys):
        ab.verdicts["AAPL"] = {"passes": False, "mean_dsr": 0.1, "beats_buyhold": False}

        ok = rbl.full_ab(["AAPL"], "2024-01-01", "2024-02-01", 10, 1, 5, 5.0, 1, str(ab.out))

        assert ok is False
        assert (ab.out / "gate_AAPL.md").read_text(encoding="utf-8") == "# gate AAPL"
        assert len(ab.reports) == 1
        assert capsys.readouterr().out.splitlines()[7:10] == [
            "",
            f"  [AAPL] 闸门判定: {_FAIL} FAIL | DSR 0.10 | 跑赢B&H False",
            "    建议: ok...",
        ]

    def test_r17_full_ab_one_pass_one_fail_overall_false(self, ab, rbl):
        ab.verdicts["MSFT"] = {"passes": False}

        ok = rbl.full_ab(["AAPL", "MSFT"], "2024-01-01", "2024-02-01", 10, 1, 5, 5.0, 1, str(ab.out))

        assert ok is False
        assert (ab.out / "gate_AAPL.md").exists() and (ab.out / "gate_MSFT.md").exists()

    @pytest.mark.parametrize(
        ("length", "printed"),
        [(159, 159), (160, 160), (161, 160), (900, 160)],
    )
    def test_r17_full_ab_recommendation_is_cut_at_160_characters(
        self, ab, rbl, capsys, length, printed
    ):
        ab.verdicts["AAPL"] = {"recommendation": "r" * length}

        rbl.full_ab(["AAPL"], "2024-01-01", "2024-02-01", 10, 1, 5, 5.0, 1, str(ab.out))

        line = capsys.readouterr().out.splitlines()[9]
        assert line == "    建议: " + "r" * printed + "..."

    def test_r17_full_ab_every_ticker_fails_no_report_empty_dashboard(
        self, ab, rbl, capsys
    ):
        """Pins the current tail: the dashboard is rewritten with no rows and the
        closing line still says everything was written."""
        ab.failing = {"AAPL", "NVDA"}

        ok = rbl.full_ab(["AAPL", "NVDA"], "2024-01-01", "2024-02-01", 10, 1, 5, 5.0, 1, str(ab.out))

        assert ok is False
        assert ab.reports == []
        assert ab.dashboards == [([], str(ab.out), False)]
        assert ab.gates == []
        cache = ab.backtests[0]["cache"]
        assert not cache.cache_dir.exists()
        captured = capsys.readouterr()
        assert captured.out.splitlines()[-2:] == ["", f"📊 报告/仪表盘/闸门判定 都写到: {ab.out}"]
        assert "❌" not in captured.out

    def test_r17_full_ab_gate_order_follows_the_ticker_list(self, ab, rbl):
        rbl.full_ab(
            ["NVDA", "AAPL", "MSFT"], "2024-01-01", "2024-02-01", 10, 1, 5, 5.0, 1, str(ab.out),
            workers=3,
        )

        assert [g[0] for g in ab.gates] == ["NVDA", "AAPL", "MSFT"]
        assert [r.ticker for r in ab.reports[0][0]] == ["NVDA", "NVDA", "AAPL", "AAPL", "MSFT", "MSFT"]

    @pytest.mark.parametrize(
        ("perp_kwargs", "expected"),
        [
            pytest.param(None, {}, id="none"),
            pytest.param({"leverage": 2.0, "allow_short": True}, {"leverage": 2.0, "allow_short": True}),
        ],
    )
    def test_r17_full_ab_perp_kwargs_reach_both_legs(self, ab, rbl, perp_kwargs, expected):
        ok = rbl.full_ab(
            ["BTCUSDT"], "2024-01-01", "2024-02-01", 10, 1, 5, 5.0, 2, str(ab.out),
            asset_type="crypto_perp", perp_kwargs=perp_kwargs,
        )

        assert ok is True
        assert ab.graphs[0].kwargs["asset_type"] == "crypto_perp"
        assert len(ab.backtests) == 4
        for call in ab.backtests:
            assert call["asset_type"] == "crypto_perp"
            assert {k: call[k] for k in expected} == expected


# --------------------------------------------------------------------------- #
# main()
# --------------------------------------------------------------------------- #


@pytest.fixture
def main_stage(rbl, monkeypatch):
    """The four mode functions become recorders returning scripted results."""
    stage = SimpleNamespace(calls=[], results={"preflight": 0, "smoke": 0, "baseline_backtest": True, "full_ab": True})

    def recorder(name):
        def record(*args, _name=name, **kwargs):
            stage.calls.append((_name, args, kwargs))
            return stage.results[_name]
        return record

    for name in ("preflight", "smoke", "baseline_backtest", "full_ab"):
        monkeypatch.setattr(rbl, name, recorder(name))
    _script_wall_clock(monkeypatch, rbl, 0)  # 2026-06-27 09:30:00 for every now()
    return stage


def _run_main(rbl, monkeypatch, capsys, *argv):
    """``main()`` always exits; return ``(code, stdout, stderr)``."""
    _set_argv(monkeypatch, "run_baseline.py", list(argv))

    with pytest.raises(SystemExit) as excinfo:
        rbl.main()

    captured = capsys.readouterr()
    return excinfo.value.code, captured.out, captured.err


def _last_usage_line(err):
    """argparse's ``prog: error: ...`` line (the last line of its usage message)."""
    return err.splitlines()[-1]


def _mode_call(stage, name):
    (call,) = [c for c in stage.calls if c[0] == name]
    return call


class TestRunBaselineMainModes:
    @pytest.mark.parametrize(
        "argv",
        [
            pytest.param([], id="no-mode"),
            pytest.param(["--smoke", "--baseline"], id="two-modes"),
            pytest.param(["--preflight", "--full"], id="another-pair"),
            pytest.param(["--ticker", "AAPL"], id="only-options"),
        ],
    )
    def test_r17_run_baseline_main_needs_exactly_one_mode(self, rbl, main_stage, monkeypatch, capsys, argv):
        code, out, err = _run_main(rbl, monkeypatch, capsys, *argv)

        assert code == 2
        assert out == ""
        assert "--preflight" in err and "--smoke" in err and "--baseline" in err and "--full" in err
        assert main_stage.calls == []

    @pytest.mark.parametrize("code", [0, 2, 7])
    def test_r17_run_baseline_main_preflight_dispatch(self, rbl, main_stage, monkeypatch, capsys, code):
        main_stage.results["preflight"] = code

        got, out, err = _run_main(rbl, monkeypatch, capsys, "--preflight")

        assert got == code
        assert _mode_call(main_stage, "preflight")[1:] == (("AAPL",), {})
        assert (out, err) == ("", "")

    def test_r17_run_baseline_main_preflight_ignores_the_ticker_list_but_still_gates_it(
        self, rbl, main_stage, monkeypatch, capsys
    ):
        got, out, err = _run_main(
            rbl, monkeypatch, capsys, "--preflight", "--ticker", "MSFT", "--tickers", "NVDA", "AAPL"
        )

        assert got == 0
        assert _mode_call(main_stage, "preflight")[1] == ("MSFT",)
        assert (out, err) == ("", "")

    @pytest.mark.parametrize("asset_type", ["crypto", "crypto_spot", "crypto_perp"])
    def test_r17_run_baseline_main_preflight_with_non_stock_asset_type_warns(
        self, rbl, main_stage, monkeypatch, capsys, asset_type
    ):
        got, out, err = _run_main(
            rbl, monkeypatch, capsys, "--preflight", "--asset-type", asset_type
        )

        assert got == 0
        assert err.splitlines() == [
            f"⚠ --preflight 不区分 asset-type（{asset_type} 仅对 --smoke/--baseline/--full 生效）"
        ]
        perp_line = (
            "  perp 模式: leverage=1.0 short=False slippage=0.0bps bnb9折=False 止损模拟=开 强平口径=mark\n"
        )
        assert out == (perp_line if asset_type == "crypto_perp" else "")

    @pytest.mark.parametrize("code", [0, 2])
    def test_r17_run_baseline_main_smoke_dispatch(self, rbl, main_stage, monkeypatch, capsys, code):
        main_stage.results["smoke"] = code

        got, out, err = _run_main(
            rbl, monkeypatch, capsys,
            "--smoke", "--ticker", "btc-usd", "--date", "2026-03-15", "--profile",
            "--asset-type", "crypto_perp", "--workers", "4",
        )

        assert got == code
        assert main_stage.calls == [
            ("smoke", ("btc-usd", "2026-03-15"), {"profile": True, "asset_type": "crypto_perp"})
        ]
        assert out == (
            "  perp 模式: leverage=1.0 short=False slippage=0.0bps bnb9折=False 止损模拟=开 强平口径=mark\n"
        )
        assert err == ""

    def test_r17_run_baseline_main_smoke_default_arguments(self, rbl, main_stage, monkeypatch, capsys):
        _run_main(rbl, monkeypatch, capsys, "--smoke")

        assert main_stage.calls == [("smoke", ("AAPL", "2026-03-15"), {"profile": False, "asset_type": "stock"})]

    @pytest.mark.parametrize(("ok", "code"), [(True, 0), (False, 1)])
    def test_r17_run_baseline_main_baseline_dispatch_defaults(self, rbl, main_stage, monkeypatch, capsys, ok, code):
        main_stage.results["baseline_backtest"] = ok

        got, out, err = _run_main(rbl, monkeypatch, capsys, "--baseline")

        assert got == code
        assert main_stage.calls == [(
            "baseline_backtest",
            (["AAPL", "NVDA"], "2025-10-30", "2026-04-28", 10, 6, 5, 5.0, 2, "backtest_output"),
            {"workers": 1, "asset_type": "stock", "perp_kwargs": {}},
        )]
        assert (out, err) == ("", "")

    @pytest.mark.parametrize(("ok", "code"), [(True, 0), (False, 1)])
    def test_r17_run_baseline_main_full_dispatch_every_flag(self, rbl, main_stage, monkeypatch, capsys, ok, code):
        main_stage.results["full_ab"] = ok

        got, out, err = _run_main(
            rbl, monkeypatch, capsys,
            "--full", "--tickers", "BTCUSDT", "ETHUSDT", "--start", "2026-01-05", "--end", "2026-06-30",
            "--step", "7", "--rebalance", "4", "--holding-days", "3", "--cost-bps", "2.5",
            "--runs", "3", "--asset-type", "crypto_perp", "--leverage", "3", "--allow-short",
            "--slippage-bps", "2.5", "--bnb-discount", "--no-simulate-stops",
            "--liq-price-type", "last", "--workers", "2", "--out", "runs/ab",
            "--ticker", "IGNORED", "--date", "IGNORED", "--profile",
        )

        assert got == code
        assert main_stage.calls == [(
            "full_ab",
            (["BTCUSDT", "ETHUSDT"], "2026-01-05", "2026-06-30", 7, 4, 3, 2.5, 3, "runs/ab"),
            {
                "workers": 2, "asset_type": "crypto_perp",
                "perp_kwargs": {
                    "leverage": 3.0, "allow_short": True, "slippage_bps": 2.5, "bnb_discount": True,
                    "simulate_stop_triggers": False, "liquidation_price_type": "last",
                },
            },
        )]
        assert out == (
            "  perp 模式: leverage=3.0 short=True slippage=2.5bps bnb9折=True 止损模拟=关 强平口径=last\n"
        )
        assert err == ""


class TestRunBaselineMainWindow:
    @pytest.mark.parametrize(
        ("argv", "start", "end"),
        [
            pytest.param([], "2025-10-30", "2026-04-28", id="both-default-today-minus-60-and-240"),
            pytest.param(["--start", "2026-01-05"], "2026-01-05", "2026-04-28", id="start-only"),
            pytest.param(["--end", "2026-06-30"], "2026-01-01", "2026-06-30", id="end-only-minus-180"),
            pytest.param(
                ["--start", "2026-1-5", "--end", "2026/03/05"], "2026-1-5", "2026/03/05", id="verbatim"
            ),
            pytest.param(["--start", "2026-06-30", "--end", "2026-01-01"], "2026-06-30", "2026-01-01", id="inverted"),
        ],
    )
    def test_r17_run_baseline_main_window(self, rbl, main_stage, monkeypatch, capsys, argv, start, end):
        _run_main(rbl, monkeypatch, capsys, "--baseline", *argv)

        assert _mode_call(main_stage, "baseline_backtest")[1][1:3] == (start, end)

    def test_r17_run_baseline_main_window_defaults_follow_the_clock(self, rbl, main_stage, monkeypatch, capsys):
        _script_wall_clock(monkeypatch, rbl, 0)
        _run_main(rbl, monkeypatch, capsys, "--baseline")
        _script_wall_clock(monkeypatch, rbl, 86_400)  # one day later
        _run_main(rbl, monkeypatch, capsys, "--baseline")

        starts = [c[1][1:3] for c in main_stage.calls if c[0] == "baseline_backtest"]
        assert starts == [("2025-10-30", "2026-04-28"), ("2025-10-31", "2026-04-29")]

    @pytest.mark.parametrize("mode", ["--smoke", "--preflight"])
    def test_r17_run_baseline_main_window_is_computed_even_where_it_is_unused(
        self, rbl, main_stage, monkeypatch, capsys, mode
    ):
        """A broken ``--end`` therefore kills a smoke run with a bare traceback."""
        _set_argv(monkeypatch, "run_baseline.py", [mode, "--end", "garbage"])

        with pytest.raises(ValueError, match="time data 'garbage'"):
            rbl.main()

        assert main_stage.calls == []

    @pytest.mark.parametrize("mode", ["--baseline", "--full"])
    def test_r17_run_baseline_main_bad_end_with_defaulted_start_is_a_bare_traceback(
        self, rbl, main_stage, monkeypatch, capsys, mode
    ):
        """The default ``--start`` is derived from ``--end``, so a broken end stops here."""
        _set_argv(monkeypatch, "run_baseline.py", [mode, "--end", "garbage", "--tickers", "AAPL"])

        with pytest.raises(ValueError, match="time data 'garbage'"):
            rbl.main()

        assert main_stage.calls == []

    @pytest.mark.parametrize("mode", ["--baseline", "--full"])
    def test_r17_run_baseline_main_start_is_forwarded_unvalidated(
        self, rbl, main_stage, monkeypatch, capsys, mode
    ):
        """Pins the current seam: with both ends given nothing parses them here; the
        mode function receives the raw strings (its ``_rebalance_dates`` raises)."""
        got, _, _ = _run_main(
            rbl, monkeypatch, capsys, mode, "--start", "garbage", "--end", "2026-06-30"
        )

        assert got == 0
        name = "baseline_backtest" if mode == "--baseline" else "full_ab"
        assert _mode_call(main_stage, name)[1][1:3] == ("garbage", "2026-06-30")


class TestRunBaselineMainPerpKnobs:
    @pytest.mark.parametrize(
        ("asset_type", "expected"),
        [
            pytest.param("stock", {}, id="stock"),
            pytest.param("crypto", {}, id="legacy-crypto"),
            pytest.param("crypto_spot", {}, id="spot"),
            pytest.param(
                "crypto_perp",
                {
                    "leverage": 3.0, "allow_short": True, "slippage_bps": 2.5, "bnb_discount": True,
                    "simulate_stop_triggers": False, "liquidation_price_type": "last",
                },
                id="perp-all-knobs",
            ),
        ],
    )
    def test_r17_run_baseline_main_perp_knobs_need_the_perp_asset_type(
        self, rbl, main_stage, monkeypatch, capsys, asset_type, expected
    ):
        got, out, err = _run_main(
            rbl, monkeypatch, capsys, "--baseline", "--asset-type", asset_type, "--leverage", "3",
            "--allow-short", "--slippage-bps", "2.5", "--bnb-discount", "--no-simulate-stops",
            "--liq-price-type", "last",
        )

        assert got == 0
        call = _mode_call(main_stage, "baseline_backtest")
        assert call[2]["perp_kwargs"] == expected
        assert call[2]["asset_type"] == asset_type
        perp_line = (
            "  perp 模式: leverage=3.0 short=True slippage=2.5bps bnb9折=True 止损模拟=关 强平口径=last\n"
        )
        assert out == (perp_line if asset_type == "crypto_perp" else "")
        assert err == ""

    @pytest.mark.parametrize(
        ("flag", "key", "value", "printed"),
        [
            pytest.param(["--leverage", "0.5"], "leverage", 0.5, "leverage=0.5", id="fractional-leverage"),
            pytest.param(["--leverage", "nan"], "leverage", float("nan"), "leverage=nan", id="nan-leverage"),
            pytest.param(["--slippage-bps", "0.5"], "slippage_bps", 0.5, "slippage=0.5bps", id="slippage"),
            pytest.param(["--bnb-discount"], "bnb_discount", True, "bnb9折=True", id="bnb"),
            pytest.param(["--no-simulate-stops"], "simulate_stop_triggers", False, "止损模拟=关", id="stops"),
            pytest.param(["--liq-price-type", "last"], "liquidation_price_type", "last", "强平口径=last", id="liq"),
        ],
    )
    def test_r17_run_baseline_main_each_perp_knob(
        self, rbl, main_stage, monkeypatch, capsys, flag, key, value, printed
    ):
        got, out, _ = _run_main(
            rbl, monkeypatch, capsys, "--baseline", "--asset-type", "crypto_perp", *flag
        )

        assert got == 0
        perp_kwargs = _mode_call(main_stage, "baseline_backtest")[2]["perp_kwargs"]
        assert list(perp_kwargs) == [key]
        if isinstance(value, float) and math.isnan(value):
            assert math.isnan(perp_kwargs[key])
        else:
            assert perp_kwargs[key] == value
        (line,) = out.splitlines()
        rest = {
            "leverage": "leverage=1.0", "slippage_bps": "slippage=0.0bps", "bnb_discount": "bnb9折=False",
            "simulate_stop_triggers": "止损模拟=开", "liquidation_price_type": "强平口径=mark",
        }
        for name, default in rest.items():
            assert (default in line) == (name != key)
        assert printed in line and "short=False" in line

    def test_r17_run_baseline_main_perp_defaults_add_nothing(self, rbl, main_stage, monkeypatch, capsys):
        got, out, _ = _run_main(
            rbl, monkeypatch, capsys, "--baseline", "--asset-type", "crypto_perp",
            "--leverage", "1.0", "--slippage-bps", "0", "--liq-price-type", "mark",
        )

        assert got == 0
        assert _mode_call(main_stage, "baseline_backtest")[2]["perp_kwargs"] == {}
        assert out == (
            "  perp 模式: leverage=1.0 short=False slippage=0.0bps bnb9折=False 止损模拟=开 强平口径=mark\n"
        )


class TestRunBaselineMainGuards:
    @pytest.mark.parametrize(
        ("flag", "message"),
        [
            pytest.param(["--runs", "0"], "--runs must be >= 1 (got 0)"),
            pytest.param(["--runs", "-2"], "--runs must be >= 1 (got -2)"),
            pytest.param(["--rebalance", "0"], "--rebalance must be >= 1 (got 0)"),
            pytest.param(["--holding-days", "0"], "--holding-days must be >= 1 (got 0)"),
            pytest.param(["--workers", "0"], "--workers must be >= 1 (got 0)"),
            pytest.param(["--workers", "-1"], "--workers must be >= 1 (got -1)"),
        ],
    )
    @pytest.mark.parametrize("mode", ["--preflight", "--smoke", "--baseline", "--full"])
    def test_r17_run_baseline_main_numeric_guards_apply_to_every_mode(
        self, rbl, main_stage, monkeypatch, capsys, flag, message, mode
    ):
        got, out, err = _run_main(rbl, monkeypatch, capsys, mode, *flag)

        assert got == 2
        assert _last_usage_line(err) == f"run_baseline.py: error: {message}"
        assert out == ""
        assert main_stage.calls == []

    @pytest.mark.parametrize(
        ("flag", "message"),
        [
            pytest.param(
                ["--asset-type", "perp"],
                # only the prefix: Python 3.13 quotes the "(choose from ...)" list, 3.12 does not
                "argument --asset-type: invalid choice: 'perp'",
            ),
            pytest.param(
                ["--liq-price-type", "mid"],
                "argument --liq-price-type: invalid choice: 'mid'",
            ),
            pytest.param(["--step", "ten"], "argument --step: invalid int value: 'ten'"),
            pytest.param(["--cost-bps", "5bp"], "argument --cost-bps: invalid float value: '5bp'"),
            pytest.param(["--leverage", "x"], "argument --leverage: invalid float value: 'x'"),
            pytest.param(["--bogus"], "unrecognized arguments: --bogus"),
        ],
    )
    def test_r17_run_baseline_main_argparse_rejections(self, rbl, main_stage, monkeypatch, capsys, flag, message):
        got, out, err = _run_main(rbl, monkeypatch, capsys, "--smoke", *flag)

        assert got == 2
        # prefix only: the "(choose from ...)" suffix differs between Python versions
        assert _last_usage_line(err).startswith(f"run_baseline.py: error: {message}")
        assert main_stage.calls == []

    @pytest.mark.parametrize("ticker", ["0700.HK", "7203.T", "EURUSD", "XAUUSD", ""])
    @pytest.mark.parametrize("mode", ["--smoke", "--baseline", "--full", "--preflight"])
    def test_r17_run_baseline_main_market_gate_on_the_ticker_list(
        self, rbl, main_stage, monkeypatch, capsys, ticker, mode
    ):
        """``sys.exit(<message>)`` before any mode runs.

        The interpreter turns that into status 1 (not the usage-error 2 the
        checks above use); inside the test the message is the exception's code.
        """
        got, out, err = _run_main(
            rbl, monkeypatch, capsys, mode, "--tickers", "AAPL", ticker, "--ticker", "AAPL"
        )

        assert (out, err) == ("", "")
        assert main_stage.calls == []
        assert isinstance(got, str) and got.startswith(
            f"✗ --tickers 含不支持的市场的标的 {ticker!r}（仅支持美股与加密货币，如 AAPL / BTCUSDT）: "
        )

    def test_r17_run_baseline_main_market_gate_names_the_first_offender_only(
        self, rbl, main_stage, monkeypatch, capsys
    ):
        got, _, _ = _run_main(rbl, monkeypatch, capsys, "--full", "--tickers", "0700.HK", "7203.T")

        assert got.startswith("✗ --tickers 含不支持的市场的标的 '0700.HK'")

    @pytest.mark.parametrize("ticker", ["AAPL", "aapl", "BRK.B", "SPY", "BTC-USD", "BTCUSDT", "MUUSDT"])
    def test_r17_run_baseline_main_market_gate_passes_us_and_crypto(
        self, rbl, main_stage, monkeypatch, capsys, ticker
    ):
        got, out, err = _run_main(rbl, monkeypatch, capsys, "--full", "--tickers", ticker)

        assert (got, out, err) == (0, "", "")
        assert _mode_call(main_stage, "full_ab")[1][0] == [ticker]

    def test_r17_run_baseline_main_market_gate_leaves_the_smoke_ticker_to_the_graph(
        self, rbl, main_stage, monkeypatch, capsys
    ):
        """``--ticker`` (smoke / preflight) is not gated here: it reaches the mode as typed."""
        got, out, err = _run_main(rbl, monkeypatch, capsys, "--smoke", "--ticker", "0700.HK")

        assert (got, out, err) == (0, "", "")
        assert _mode_call(main_stage, "smoke")[1] == ("0700.HK", "2026-03-15")

    @pytest.mark.parametrize(
        "date",
        [
            pytest.param("2026-3-15", id="not-zero-padded"),
            pytest.param("2026-13-45", id="impossible"),
            pytest.param("20260315", id="compact"),
            pytest.param("2026-03-15 09:30:00", id="with-time"),
            pytest.param("", id="empty"),
            pytest.param("garbage", id="garbage"),
        ],
    )
    def test_r17_run_baseline_main_smoke_rejects_a_non_canonical_date_before_the_graph(
        self, rbl, main_stage, monkeypatch, capsys, date
    ):
        """Round-17: ``--date`` went to ``smoke()`` untouched.

        The graph does not validate it either, so an impossible date paid for
        the analysts before the data-vacuum gate stopped the run, and a
        non-canonical one wrote ``full_states_log_<as typed>.json`` where the
        web store and ``verify-history`` never look (round-13, same defect in
        the batch entry). Same predicate as ``run_robust --date`` and the web
        endpoint.
        """
        got, out, err = _run_main(
            rbl, monkeypatch, capsys, "--smoke", "--ticker", "AAPL", "--date", date, "--profile"
        )

        assert got == 2
        assert _last_usage_line(err) == f"run_baseline.py: error: --date must be YYYY-MM-DD (got {date!r})"
        assert main_stage.calls == []

    def test_r17_run_baseline_main_smoke_accepts_a_canonical_date(self, rbl, main_stage, monkeypatch, capsys):
        got, _, _ = _run_main(
            rbl, monkeypatch, capsys, "--smoke", "--date", "2024-02-29", "--asset-type", "crypto_perp"
        )

        assert got == 0
        assert _mode_call(main_stage, "smoke")[1] == ("AAPL", "2024-02-29")

    @pytest.mark.parametrize("mode", ["--baseline", "--full", "--preflight"])
    def test_r17_run_baseline_main_date_is_not_looked_at_in_the_other_modes(
        self, rbl, main_stage, monkeypatch, capsys, mode
    ):
        got, _, _ = _run_main(rbl, monkeypatch, capsys, mode, "--date", "garbage")

        assert got == 0
        assert len(main_stage.calls) == 1


def test_r17_run_baseline_main_guard_runs_main(rbl, main_stage, monkeypatch, capsys):
    """``python scripts/run_baseline.py`` runs ``main()`` and exits with its status."""
    path = SCRIPTS_DIR / "run_baseline.py"
    code = compile(path.read_text(encoding="utf-8"), str(path), "exec")
    monkeypatch.setattr(sys, "path", list(sys.path))  # the script's shim edits sys.path
    outcomes = []

    for argv in (["--runs", "0"], ["--smoke", "--tickers", "0700.HK"], ["--bogus"]):
        _set_argv(monkeypatch, "run_baseline.py", argv)
        with pytest.raises(SystemExit) as excinfo:
            exec(code, {"__name__": "__main__", "__file__": str(path)})  # noqa: S102
        outcomes.append(excinfo.value.code)
        capsys.readouterr()

    assert outcomes[0] == 2 and outcomes[2] == 2
    assert outcomes[1].startswith("✗ --tickers 含不支持的市场的标的 '0700.HK'")



# =========================================================================== #
# scripts/run_robust.py
# =========================================================================== #

_DAY_ROBUST = "2026-07-01"


@pytest.fixture
def rr(monkeypatch):
    """A fresh ``run_robust.py``: its module-level abort flag and child registry start clean."""
    monkeypatch.delenv("YIALPHA_ROBUST_CHILD_SCRIPT", raising=False)
    return _load("r17_run_robust", "run_robust.py")


def _robust_parse(rr, monkeypatch, *argv):
    _set_argv(monkeypatch, "run_robust.py", ["--tickers", "AAPL", "--date", _DAY_ROBUST, *argv])
    return rr._parse_args()


class _Clock:
    """Replaces ``time.time`` / ``time.sleep`` in the loaded script: no real waiting."""

    def __init__(self, start=1_000.0):
        self.now = start
        self.sleeps: list[float] = []

    def time(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += float(seconds)


class _Seq:
    """Per-call scripted return values for a stubbed helper; the last value repeats."""

    def __init__(self, *values, default=None):
        self.values = list(values)
        self.default = default
        self.calls = 0

    def __call__(self, *args):
        self.calls += 1
        if not self.values:
            return self.default
        return self.values[min(self.calls, len(self.values)) - 1]


class _Child:
    """``Popen`` stand-in: ``codes`` is the ``poll()`` script (last value repeats)."""

    def __init__(self, codes, stage, pid=4242):
        self._codes = list(codes)
        self.stage = stage
        self.pid = pid
        self.returncode = None
        self.waits: list[float | None] = []
        self.kill_calls = 0

    def poll(self):
        code = self._codes.pop(0) if len(self._codes) > 1 else self._codes[0]
        if self.stage.on_poll is not None:
            code = self.stage.on_poll(self, code)
        self.returncode = code
        return code

    def wait(self, timeout=None):
        self.waits.append(timeout)
        if self.stage.wait_hangs and len(self.waits) == 1:
            raise subprocess.TimeoutExpired(cmd="child", timeout=timeout)
        if self.returncode is None:
            self.returncode = self.stage.exit_after_kill
        return self.returncode

    def kill(self):
        self.kill_calls += 1
        self.returncode = self.stage.exit_after_kill


@pytest.fixture
def run(rr, monkeypatch, tmp_path):
    """Drive ``_run_one_ticker`` with scripted children and a fake clock; nothing is spawned.

    ``_states_root`` is pinned to ``tmp_path/logs`` (also the ``--reports-root``
    default); ``_find_new_report`` / ``_core_sentinel_count`` are scripted
    through ``stage.reports`` / ``stage.sentinels``; the registry, ``_reap`` and
    ``_complete_mtime`` stay real.
    """
    stage = SimpleNamespace(
        clock=_Clock(),
        spawns=[],
        scripts=[[0]],          # poll() script per attempt (last repeats): exit 0 at once
        kill_landed=True,
        kill_calls=[],
        reports=_Seq(tmp_path / "reports" / "AAPL_x" / "complete_report.md"),
        sentinels=_Seq(0),
        cpu=_Seq(),
        on_poll=None,
        popen_error=None,
        wait_hangs=False,
        exit_after_kill=1,
        states_root=tmp_path / "logs",
        reports_root=tmp_path / "logs" / "reports",
    )
    monkeypatch.setattr(rr, "_states_root", lambda: stage.states_root)
    monkeypatch.setattr(rr.time, "time", stage.clock.time)
    monkeypatch.setattr(rr.time, "sleep", stage.clock.sleep)

    def popen(cmd, **kwargs):
        record = {"cmd": list(cmd), **kwargs}
        stage.spawns.append(record)
        if stage.popen_error is not None:
            raise stage.popen_error
        script = stage.scripts[min(len(stage.spawns) - 1, len(stage.scripts) - 1)]
        record["child"] = _Child(script, stage)
        return record["child"]

    stage.popen = popen
    monkeypatch.setattr(rr.subprocess, "Popen", popen)
    monkeypatch.setattr(
        rr, "_kill_tree", lambda pid: stage.kill_calls.append(pid) or stage.kill_landed
    )
    monkeypatch.setattr(rr, "_cpu_seconds", lambda pid: stage.cpu(pid))
    monkeypatch.setattr(
        rr, "_find_new_report", lambda root, ticker, pre: stage.reports(root, ticker, pre)
    )
    monkeypatch.setattr(
        rr, "_core_sentinel_count", lambda root, ticker, date: stage.sentinels(root, ticker, date)
    )
    stage.opts = _robust_parse(rr, monkeypatch, "--reports-root", str(stage.reports_root))
    return stage


def _one(run, rr, ticker="AAPL", **overrides):
    for key, value in overrides.items():
        setattr(run.opts, key, value)
    return rr._run_one_ticker(ticker, _DAY_ROBUST, run.opts)


# --------------------------------------------------------------------------- #
# _parse_args()
# --------------------------------------------------------------------------- #


class TestRobustParseArgs:
    def test_r17_robust_parse_defaults(self, rr, monkeypatch, tmp_path):
        monkeypatch.setattr(rr, "_states_root", lambda: tmp_path / "logs")

        opts = _robust_parse(rr, monkeypatch)

        assert vars(opts) == {
            "tickers": ["AAPL"],
            "date": _DAY_ROBUST,
            "asset_type": None,
            "workers": 2,
            "per_ticker_timeout": 1800.0,
            "max_attempts": 3,
            "stall_timeout": 0.0,
            "stall_cpu": 0.5,
            "backoff": 15.0,
            "no_llm_cache": False,
            "reports_root": str(tmp_path / "logs" / "reports"),
            "allow_degraded": False,
            "require_data_quality": True,
        }

    def test_r17_robust_parse_every_flag(self, rr, monkeypatch, tmp_path):
        opts = _robust_parse(
            rr, monkeypatch,
            "--tickers", "SNDK", "INTC", "--asset-type", "crypto_perp", "--workers", "3",
            "--per-ticker-timeout", "900", "--max-attempts", "5", "--stall-timeout", "120",
            "--stall-cpu", "1.5", "--backoff", "0", "--no-llm-cache", "--allow-degraded",
            "--reports-root", str(tmp_path / "elsewhere"),
        )

        assert {k: v for k, v in vars(opts).items() if k != "reports_root"} == {
            "tickers": ["SNDK", "INTC"],
            "date": _DAY_ROBUST,
            "asset_type": "crypto_perp",
            "workers": 3,
            "per_ticker_timeout": 900.0,
            "max_attempts": 5,
            "stall_timeout": 120.0,
            "stall_cpu": 1.5,
            "backoff": 0.0,
            "no_llm_cache": True,
            "allow_degraded": True,
            "require_data_quality": False,
        }
        assert opts.reports_root == str(tmp_path / "elsewhere")

    @pytest.mark.parametrize("asset_type", ["stock", "crypto", "crypto_spot", "crypto_perp"])
    def test_r17_robust_parse_asset_type_choices(self, rr, monkeypatch, asset_type):
        opts = _robust_parse(rr, monkeypatch, "--asset-type", asset_type)

        assert opts.asset_type == asset_type

    @pytest.mark.parametrize(
        ("argv", "message"),
        [
            pytest.param(["--workers", "0"], "--workers must be >= 1 (got 0)", id="workers-zero"),
            pytest.param(["--workers", "-1"], "--workers must be >= 1 (got -1)", id="workers-negative"),
            pytest.param(["--max-attempts", "0"], "--max-attempts must be >= 1 (got 0)", id="attempts-zero"),
            pytest.param(["--per-ticker-timeout", "0"], "--per-ticker-timeout must be > 0 (got 0.0)", id="timeout-zero"),
            pytest.param(["--per-ticker-timeout", "-5"], "--per-ticker-timeout must be > 0 (got -5.0)", id="timeout-negative"),
            pytest.param(["--backoff", "-1"], "--backoff must be >= 0 (got -1.0)", id="backoff-negative"),
            pytest.param(["--stall-timeout", "-1"], "--stall-timeout must be >= 0 (got -1.0)", id="stall-negative"),
            pytest.param(["--asset-type", "perp"], "argument --asset-type: invalid choice: 'perp'", id="asset-type"),
            pytest.param(["--workers", "two"], "argument --workers: invalid int value: 'two'", id="workers-not-int"),
            pytest.param(["--backoff", "soon"], "argument --backoff: invalid float value: 'soon'", id="backoff-not-float"),
            pytest.param(["--bogus"], "unrecognized arguments: --bogus", id="unknown-flag"),
        ],
    )
    def test_r17_robust_parse_rejections(self, rr, monkeypatch, capsys, argv, message):
        _set_argv(monkeypatch, "run_robust.py", ["--tickers", "AAPL", "--date", _DAY_ROBUST, *argv])

        with pytest.raises(SystemExit) as excinfo:
            rr._parse_args()

        assert excinfo.value.code == 2
        assert capsys.readouterr().err.splitlines()[-1].startswith(f"run_robust.py: error: {message}")

    @pytest.mark.parametrize("missing", [[], ["--date", _DAY_ROBUST], ["--tickers", "AAPL"]])
    def test_r17_robust_parse_tickers_and_date_are_required(self, rr, monkeypatch, capsys, missing):
        monkeypatch.setenv("COLUMNS", "300")  # keep the usage line unabbreviated
        _set_argv(monkeypatch, "run_robust.py", missing)

        with pytest.raises(SystemExit) as excinfo:
            rr._parse_args()

        assert excinfo.value.code == 2
        err = capsys.readouterr().err
        assert "--tickers" in err and "--date" in err

    @pytest.mark.parametrize("date", ["2026-7-1", "2026/07/01", "20260701", "2026-07-01 09:30:00", "", "yesterday"])
    def test_r17_robust_parse_date_must_be_canonical(self, rr, monkeypatch, capsys, date):
        _set_argv(monkeypatch, "run_robust.py", ["--tickers", "AAPL", "--date", date])

        with pytest.raises(SystemExit) as excinfo:
            rr._parse_args()

        assert excinfo.value.code == 2
        assert capsys.readouterr().err.splitlines()[-1] == (
            f"run_robust.py: error: --date must be YYYY-MM-DD (got {date!r})"
        )

    @pytest.mark.parametrize(
        ("flag", "value"),
        [
            pytest.param("--per-ticker-timeout", "nan", id="timeout-nan"),
            pytest.param("--per-ticker-timeout", "inf", id="timeout-inf"),
            pytest.param("--per-ticker-timeout", "1e309", id="timeout-overflow-literal"),
            pytest.param("--backoff", "nan", id="backoff-nan"),
            pytest.param("--backoff", "inf", id="backoff-inf"),
            pytest.param("--stall-timeout", "nan", id="stall-nan"),
            pytest.param("--stall-timeout", "inf", id="stall-inf"),
            pytest.param("--stall-cpu", "nan", id="stall-cpu-nan"),
            pytest.param("--stall-cpu", "inf", id="stall-cpu-inf"),
            pytest.param("--stall-cpu", "-1", id="stall-cpu-negative"),
        ],
    )
    def test_r17_robust_parse_rejects_values_that_silently_disable_a_watchdog(
        self, rr, monkeypatch, capsys, flag, value
    ):
        """Round-17: the lower-bound guards compared with ``< 0`` / ``<= 0`` only.

        NaN fails every such comparison and +inf passes it, so
        ``--per-ticker-timeout nan`` was accepted and ``elapsed > nan`` is always
        False -- the wall-clock watchdog could never fire. A NaN / inf backoff
        reached ``time.sleep`` and escaped as ValueError / OverflowError.
        """
        _set_argv(monkeypatch, "run_robust.py", ["--tickers", "AAPL", "--date", _DAY_ROBUST, flag, value])

        with pytest.raises(SystemExit) as excinfo:
            rr._parse_args()

        assert excinfo.value.code == 2
        line = capsys.readouterr().err.splitlines()[-1]
        shown = {"nan": "nan", "inf": "inf", "1e309": "inf", "-1": "-1.0"}[value]
        assert line.startswith(f"run_robust.py: error: {flag} must be")
        assert line.endswith(f"(got {shown})")

    def test_r17_robust_parse_zero_stall_window_is_off_and_allowed(self, rr, monkeypatch):
        opts = _robust_parse(rr, monkeypatch, "--stall-timeout", "0", "--stall-cpu", "0", "--backoff", "0")

        assert (opts.stall_timeout, opts.stall_cpu, opts.backoff) == (0.0, 0.0, 0.0)

    @pytest.mark.parametrize(
        ("flags", "strict", "lenient"),
        [
            pytest.param([], True, False, id="default-strict"),
            pytest.param(["--require-data-quality"], True, False, id="legacy-flag-is-a-noop"),
            pytest.param(["--allow-degraded"], False, True, id="escape-hatch"),
            pytest.param(["--require-data-quality", "--allow-degraded"], False, True, id="both-escape-hatch-wins"),
            pytest.param(["--allow-degraded", "--require-data-quality"], False, True, id="both-either-order"),
        ],
    )
    def test_r17_robust_parse_quality_gate_flags(self, rr, monkeypatch, capsys, flags, strict, lenient):
        opts = _robust_parse(rr, monkeypatch, *flags)

        assert (opts.require_data_quality, opts.allow_degraded) == (strict, lenient)
        both = "--require-data-quality" in flags and "--allow-degraded" in flags
        expected = (
            "⚠ 同时给了 --require-data-quality 与 --allow-degraded：按 --allow-degraded（显式逃生口）执行\n"
            if both else ""
        )
        assert capsys.readouterr().err == expected

    def test_r17_robust_parse_reports_root_default_is_the_states_root_reports_dir(
        self, rr, monkeypatch, tmp_path
    ):
        """Pinned through the real ``_states_root``: an exported var is used verbatim."""
        root = tmp_path / "exported" / "results"
        monkeypatch.setenv("YIALPHA_RESULTS_DIR", str(root))

        opts = _robust_parse(rr, monkeypatch)

        assert opts.reports_root == str(root / "reports")

    def test_r17_robust_parse_tickers_are_not_validated_here(self, rr, monkeypatch, capsys):
        """No charset / market gate in the orchestrator: the child validates (exit 2)."""
        opts = _robust_parse(rr, monkeypatch, "--tickers", "AAPL,MSFT", "0700.HK", "", "brk.b")

        assert opts.tickers == ["AAPL,MSFT", "0700.HK", "", "brk.b"]
        assert capsys.readouterr().err == ""

    @pytest.mark.parametrize(
        ("tickers", "kept", "dropped"),
        [
            pytest.param(["AAPL", "NVDA", "AAPL"], ["AAPL", "NVDA"], ["AAPL"], id="exact"),
            pytest.param(["AAPL", "aapl"], ["AAPL"], ["aapl"], id="case-variant"),
            pytest.param(["SNDK", "SNDK", "SNDK"], ["SNDK"], ["SNDK", "SNDK"], id="triple"),
        ],
    )
    def test_r17_robust_parse_duplicate_tickers_are_dropped_once(
        self, rr, monkeypatch, capsys, tickers, kept, dropped
    ):
        """Round-17: one child is spawned per list entry, so a ticker named twice
        was analyzed twice at once (two children on one ticker's state logs and
        caches, both worker slots on it, two rows in the summary). Same rule as
        ``BatchRunner._dedup``; what was dropped is said on stderr."""
        opts = _robust_parse(rr, monkeypatch, "--tickers", *tickers)

        assert opts.tickers == kept
        assert capsys.readouterr().err == (
            f"⚠ removed {len(dropped)} duplicate ticker(s) from --tickers: {dropped} "
            f"(each ticker is run once)\n"
        )

    def test_r17_robust_help_documents_exactly_the_parsed_options(self, rr, monkeypatch, capsys):
        monkeypatch.setenv("COLUMNS", "200")
        _set_argv(monkeypatch, "run_robust.py", ["--help"])

        with pytest.raises(SystemExit) as excinfo:
            rr._parse_args()

        assert excinfo.value.code == 0
        out = capsys.readouterr().out
        entries = re.findall(r"^  (--[a-z-]+)\b", out, flags=re.M)
        assert entries == [
            "--tickers", "--date", "--asset-type", "--workers", "--per-ticker-timeout",
            "--max-attempts", "--stall-timeout", "--stall-cpu", "--backoff", "--no-llm-cache",
            "--reports-root", "--allow-degraded", "--require-data-quality",
        ]
        # The legacy flag is advertised as the no-op it is.
        assert "（已默认开启，保留兼容；见 --allow-degraded）" in out


# --------------------------------------------------------------------------- #
# _states_root()
# --------------------------------------------------------------------------- #


class TestRobustStatesRootFallbacks:
    """The two branches the watchdog tests leave out: dotenv itself unavailable."""

    @pytest.mark.parametrize("error", [OSError("locked"), PermissionError("denied")])
    def test_r17_robust_states_root_unreadable_env_file_falls_back_to_home(
        self, rr, monkeypatch, error
    ):
        monkeypatch.delenv("YIALPHA_RESULTS_DIR", raising=False)
        def unreadable(path):
            raise error

        monkeypatch.setattr("dotenv.dotenv_values", unreadable)

        assert rr._states_root() == Path.home() / ".yialpha" / "logs"

    def test_r17_robust_states_root_without_python_dotenv_falls_back_to_home(self, rr, monkeypatch):
        monkeypatch.delenv("YIALPHA_RESULTS_DIR", raising=False)
        monkeypatch.setitem(sys.modules, "dotenv", None)  # ``from dotenv import ...`` -> ImportError

        assert rr._states_root() == Path.home() / ".yialpha" / "logs"

    def test_r17_robust_states_root_exported_value_wins_without_reading_dotenv(
        self, rr, monkeypatch, tmp_path
    ):
        monkeypatch.setenv("YIALPHA_RESULTS_DIR", str(tmp_path / "from-env"))
        monkeypatch.setattr("dotenv.dotenv_values", lambda path: pytest.fail(".env must not be read"))

        assert rr._states_root() == tmp_path / "from-env"


# --------------------------------------------------------------------------- #
# _find_new_report() / _core_sentinel_count()
# --------------------------------------------------------------------------- #


def _touch(path, mtime):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("x", encoding="utf-8")
    os.utime(path, (mtime, mtime))
    return path


class TestRobustFindNewReport:
    def test_r17_robust_find_new_report_skips_incomplete_dirs_even_when_newer(self, rr, tmp_path):
        good = _touch(tmp_path / "AAPL_20260701_100000" / "complete_report.md", 100.0)
        (tmp_path / "AAPL_20260701_100500").mkdir()  # newer, still being written
        incomplete = tmp_path / "AAPL_20260701_101000"
        incomplete.mkdir()
        (incomplete / "partial.md").write_text("y", encoding="utf-8")

        assert rr._find_new_report(tmp_path, "AAPL", 50.0) == good
        assert rr._complete_mtime(tmp_path, "AAPL") == 100.0

    def test_r17_robust_find_new_report_only_incomplete_dirs_is_none(self, rr, tmp_path):
        (tmp_path / "AAPL_20260701_100500").mkdir()

        assert rr._find_new_report(tmp_path, "AAPL", 0.0) is None

    def test_r17_robust_find_new_report_newest_wins_and_ties_lose(self, rr, tmp_path):
        older = _touch(tmp_path / "AAPL_20260701_090000" / "complete_report.md", 100.0)
        newer = _touch(tmp_path / "AAPL_20260701_100000" / "complete_report.md", 200.0)
        _touch(tmp_path / "AAPL_20260701_100500" / "complete_report.md", 200.0)

        assert rr._find_new_report(tmp_path, "AAPL", 0.0) == newer
        assert rr._find_new_report(tmp_path, "AAPL", 100.0) == newer
        # A report whose mtime equals the pre-attempt snapshot is not "new".
        assert rr._find_new_report(tmp_path, "AAPL", 200.0) is None
        assert older.exists()


class TestRobustCoreSentinelCount:
    def _log(self, root, body, name="full_states_log_2026-07-01.json", mtime=None):
        path = root / "AAPL" / "YiAlphaStrategy_logs" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body, encoding="utf-8")
        if mtime is not None:
            os.utime(path, (mtime, mtime))
        return path

    @pytest.mark.parametrize("body", ["", "   ", '{"data_quality": {"core_sentinel_count": ', "not json"])
    def test_r17_robust_core_sentinel_count_unreadable_log_is_unknown(self, rr, tmp_path, body):
        self._log(tmp_path, body)

        assert rr._core_sentinel_count(tmp_path, "AAPL", "2026-07-01") is None

    @pytest.mark.parametrize("body", ["[]", '"text"', "42"])
    def test_r17_robust_core_sentinel_count_non_object_log_raises(
        self, rr, tmp_path, body
    ):
        """Pins the current seam: only OSError / ValueError are caught, so a
        hand-written non-object file surfaces as ``orchestrator_error`` for the
        ticker. The graph always writes an object."""
        self._log(tmp_path, body)

        with pytest.raises(AttributeError):
            rr._core_sentinel_count(tmp_path, "AAPL", "2026-07-01")

    @pytest.mark.parametrize(
        ("body", "expected"),
        [
            pytest.param('{"data_quality": {"core_sentinel_count": 0}}', 0, id="clean"),
            pytest.param('{"data_quality": {"core_sentinel_count": 4}}', 4, id="degraded"),
            pytest.param('{"data_quality": []}', None, id="empty-list-block"),
            pytest.param('{"data_quality": {"core_sentinel_count": null}}', None, id="null"),
            pytest.param('{"data_quality": {"core_sentinel_count": 1.5}}', None, id="float"),
            pytest.param('{"data_quality": {"core_sentinel_count": "2"}}', None, id="numeric-string"),
            pytest.param('{"data_quality": {"core_sentinel_count": true}}', True, id="bool-is-an-int"),
            pytest.param('{"final_trade_decision": "x"}', None, id="pre-round-log"),
        ],
    )
    def test_r17_robust_core_sentinel_count_block_shapes(self, rr, tmp_path, body, expected):
        self._log(tmp_path, body)

        assert rr._core_sentinel_count(tmp_path, "AAPL", "2026-07-01") == expected

    def test_r17_robust_core_sentinel_count_newest_file_for_the_date_wins(self, rr, tmp_path):
        self._log(tmp_path, '{"data_quality": {"core_sentinel_count": 7}}', mtime=100.0)
        self._log(
            tmp_path, '{"data_quality": {"core_sentinel_count": 2}}',
            name="full_states_log_2026-07-01_perp.json", mtime=300.0,
        )
        self._log(
            tmp_path, '{"data_quality": {"core_sentinel_count": 9}}',
            name="full_states_log_2026-07-01_spot.json", mtime=200.0,
        )

        assert rr._core_sentinel_count(tmp_path, "AAPL", "2026-07-01") == 2
        assert rr._core_sentinel_count(tmp_path, "AAPL", "2026-07-02") is None


# --------------------------------------------------------------------------- #
# _cpu_seconds() / _kill_all_active()
# --------------------------------------------------------------------------- #


class TestRobustCpuSeconds:
    def test_r17_robust_cpu_seconds_is_windows_only(self, rr, monkeypatch):
        monkeypatch.setattr(rr, "IS_WINDOWS", False)
        monkeypatch.setattr(
            rr.subprocess, "check_output", lambda *a, **k: pytest.fail("no probe off Windows")
        )

        assert rr._cpu_seconds(4242) is None

    @pytest.mark.parametrize("text", ["12.5\n", " 0.0\r\n", "3", "125000.999"])
    def test_r17_robust_cpu_seconds_parses_the_powershell_total(self, rr, monkeypatch, text):
        monkeypatch.setattr(rr, "IS_WINDOWS", True)
        calls = []

        def check_output(cmd, **kwargs):
            calls.append((cmd, kwargs))
            return text

        monkeypatch.setattr(rr.subprocess, "check_output", check_output)

        assert rr._cpu_seconds(4242) == float(text)
        assert calls == [(
            ["powershell", "-NoProfile", "-Command", "(Get-Process -Id 4242).CPU"],
            {"stderr": subprocess.DEVNULL, "text": True, "timeout": 10},
        )]

    @pytest.mark.parametrize("text", ["", "   \n", "N/A"])
    def test_r17_robust_cpu_seconds_empty_or_non_numeric_output_is_none(self, rr, monkeypatch, text):
        monkeypatch.setattr(rr, "IS_WINDOWS", True)
        monkeypatch.setattr(rr.subprocess, "check_output", lambda cmd, **kwargs: text)

        assert rr._cpu_seconds(4242) is None

    @pytest.mark.parametrize(
        "error",
        [
            subprocess.CalledProcessError(1, "powershell"),
            subprocess.TimeoutExpired(cmd="powershell", timeout=10),
            OSError("not found"),
            ValueError("bad pipe"),
        ],
    )
    def test_r17_robust_cpu_seconds_probe_failure_is_none(self, rr, monkeypatch, error):
        """The process is gone, PowerShell is missing or slow: unknown, never a crash."""
        monkeypatch.setattr(rr, "IS_WINDOWS", True)

        def check_output(cmd, **kwargs):
            raise error

        monkeypatch.setattr(rr.subprocess, "check_output", check_output)

        assert rr._cpu_seconds(4242) is None


class TestRobustKillAllActive:
    def test_r17_robust_kill_all_active_kills_only_children_still_running(self, rr, monkeypatch):
        killed = []
        monkeypatch.setattr(rr, "_kill_tree", killed.append)
        alive = SimpleNamespace(pid=111, poll=lambda: None)
        gone = SimpleNamespace(pid=222, poll=lambda: 0)
        other = SimpleNamespace(pid=333, poll=lambda: None)
        for proc in (alive, gone, other):
            rr._register_proc(proc)

        rr._kill_all_active()

        assert killed == [111, 333]
        assert rr._active_procs == [alive, gone, other]  # the workers unregister themselves

    def test_r17_robust_kill_all_active_with_nothing_registered(self, rr, monkeypatch):
        monkeypatch.setattr(rr, "_kill_tree", lambda pid: pytest.fail("nothing to kill"))

        rr._kill_all_active()

    def test_r17_robust_register_unregister_is_idempotent(self, rr):
        proc = SimpleNamespace(pid=1, poll=lambda: None)

        rr._register_proc(proc)
        rr._register_proc(proc)
        assert rr._active_procs == [proc, proc]

        rr._unregister_proc(proc)
        rr._unregister_proc(proc)
        rr._unregister_proc(proc)  # already gone: harmless
        assert rr._active_procs == []


def test_r17_robust_module_reconfigures_console_without_the_shared_bootstrap(rr, monkeypatch):
    """``run_robust.py`` keeps its own inline copy of the UTF-8 console fix."""
    calls = []

    class _Stream:
        def __init__(self, name):
            self.name = name

        def reconfigure(self, **kwargs):
            calls.append((self.name, kwargs))

    monkeypatch.setattr(sys, "stdout", _Stream("stdout"))
    monkeypatch.setattr(sys, "stderr", _Stream("stderr"))

    rr_rel = _load("r17_run_robust_console", "run_robust.py")

    assert calls == [
        ("stdout", {"encoding": "utf-8", "errors": "replace"}),
        ("stderr", {"encoding": "utf-8", "errors": "replace"}),
    ]
    assert (os.name == "nt") == rr_rel.IS_WINDOWS


# --------------------------------------------------------------------------- #
# _run_one_ticker()
# --------------------------------------------------------------------------- #


_WARN = "\u26a0\ufe0f"
_HOURGLASS = "\u23f3"
_LOG_NAME = re.compile(r"robust_AAPL_\d{8}_\d{6}_a(\d)\.log")


def _spawn(run, n=0):
    return run.spawns[n]


def _log_dir(run):
    return run.reports_root.parent / "robust"


def _log_text(run, name):
    return (_log_dir(run) / name).read_text(encoding="utf-8")


class TestRobustOneSpawn:
    def test_r17_robust_one_clean_exit_ok_result_command_and_log(self, run, rr, monkeypatch, capsys):
        monkeypatch.setattr(rr, "IS_WINDOWS", True)
        report = run.reports.values[0]

        result = _one(run, rr)

        assert result == {
            "ticker": "AAPL", "ok": True, "attempts": 1, "reason": "ok (wall 0s)",
            "report_path": report, "log_path": result["log_path"],
        }
        spawn = _spawn(run)
        assert spawn["cmd"] == [
            sys.executable, str(rr._PROJECT_ROOT / "scripts" / "run_batch.py"),
            "--tickers", "AAPL", "--date", _DAY_ROBUST, "--workers", "1", "--no-progress",
        ]
        assert spawn["cwd"] == str(rr._PROJECT_ROOT)
        assert spawn["stderr"] == subprocess.STDOUT
        assert spawn["creationflags"] == rr._CREATE_FLAGS
        assert "start_new_session" not in spawn
        # The per-attempt log is the child's stdout+stderr and is closed again.
        assert Path(spawn["stdout"].name) == result["log_path"]
        assert spawn["stdout"].closed is True
        log_name = result["log_path"].name
        assert _LOG_NAME.fullmatch(log_name) and log_name.endswith("_a1.log")
        assert result["log_path"].parent == _log_dir(run)
        assert _log_text(run, log_name).splitlines()[:2] == [
            "$ " + " ".join(spawn["cmd"]),
            f"# cwd={rr._PROJECT_ROOT} per_ticker_timeout=1800.0s stall_timeout=0.0s",
        ]
        assert rr._active_procs == []
        assert run.kill_calls == []
        out = capsys.readouterr().out.splitlines()
        assert len(out) == 2 and out[1] == f"[AAPL] {_OK} attempt 1 done in 0s → {report}"
        assert out[0] == f"[AAPL] {_PLAY} attempt 1/3 → log {log_name}"

    def test_r17_robust_one_posix_child_starts_its_own_session(self, run, rr, monkeypatch):
        monkeypatch.setattr(rr, "IS_WINDOWS", False)

        _one(run, rr)

        spawn = _spawn(run)
        assert spawn["start_new_session"] is True  # so os.killpg() never reaches this process
        assert "creationflags" not in spawn
        assert spawn["cwd"] == str(rr._PROJECT_ROOT)

    @pytest.mark.parametrize("asset_type", ["stock", "crypto", "crypto_spot", "crypto_perp"])
    def test_r17_robust_one_forwards_asset_type_only_when_set(self, run, rr, asset_type):
        _one(run, rr, asset_type=asset_type)

        assert _spawn(run)["cmd"][-2:] == ["--asset-type", asset_type]
        assert _spawn(run)["cmd"][:-2] == [
            sys.executable, str(rr._PROJECT_ROOT / "scripts" / "run_batch.py"),
            "--tickers", "AAPL", "--date", _DAY_ROBUST, "--workers", "1", "--no-progress",
        ]

    def test_r17_robust_one_child_script_hook_swaps_only_the_program(self, run, rr, monkeypatch, tmp_path):
        child = tmp_path / "crasher.py"
        monkeypatch.setenv("YIALPHA_ROBUST_CHILD_SCRIPT", str(child))

        _one(run, rr, asset_type="crypto")

        assert _spawn(run)["cmd"] == [
            sys.executable, str(child),
            "--tickers", "AAPL", "--date", _DAY_ROBUST, "--workers", "1", "--no-progress",
            "--asset-type", "crypto",
        ]

    def test_r17_robust_one_registers_the_child_and_unregisters_it_before_the_verdict(
        self, run, rr, monkeypatch
    ):
        seen = []
        real_register, real_unregister = rr._register_proc, rr._unregister_proc
        monkeypatch.setattr(rr, "_register_proc", lambda p: (seen.append(("register", p)), real_register(p)))
        monkeypatch.setattr(rr, "_unregister_proc", lambda p: (seen.append(("unregister", p)), real_unregister(p)))
        during = {}

        def find_report(root, ticker, pre):
            during["active"] = list(rr._active_procs)
            return tmp_report

        tmp_report = run.reports_root / "AAPL_x" / "complete_report.md"
        monkeypatch.setattr(rr, "_find_new_report", find_report)

        _one(run, rr)

        assert [event for event, _ in seen] == ["register", "unregister"]
        assert seen[0][1] is seen[1][1]
        assert during["active"] == []  # no longer a Ctrl+C target once the child is gone
        assert rr._active_procs == []

    def test_r17_robust_one_verdict_exception_leaves_no_registered_child(self, run, rr, monkeypatch):
        def explode(*args):
            raise RuntimeError("report lookup failed")

        monkeypatch.setattr(rr, "_find_new_report", explode)

        with pytest.raises(RuntimeError, match="report lookup failed"):
            _one(run, rr)

        assert rr._active_procs == []

    def test_r17_robust_one_spawn_failure_is_reported_once_without_retry(self, run, rr, capsys):
        run.popen_error = FileNotFoundError("python is gone")

        result = _one(run, rr)

        assert result["attempts"] == 1 and result["ok"] is False
        assert result["reason"] == "spawn_failed: python is gone"
        assert result["report_path"] is None and result["log_path"].name.endswith("_a1.log")
        assert len(run.spawns) == 1
        assert run.clock.sleeps == []  # no backoff either
        assert rr._active_procs == []
        assert capsys.readouterr().out.count("\n") == 1  # only the "attempt" line

    def test_r17_robust_one_abort_flag_set_beforehand_launches_nothing(self, run, rr, capsys):
        rr._abort.set()

        result = _one(run, rr)

        assert result == {
            "ticker": "AAPL", "ok": False, "attempts": 0, "reason": "",
            "report_path": None, "log_path": None,
        }
        assert run.spawns == []
        assert capsys.readouterr().out == ""
        assert _log_dir(run).is_dir() and list(_log_dir(run).iterdir()) == []

    def test_r17_robust_one_ticker_is_used_verbatim_in_the_log_name(self, run, rr, capsys):
        result = _one(run, rr, ticker="0700.HK")

        assert result["log_path"].name.startswith("robust_0700.HK_")
        assert _spawn(run)["cmd"][2:4] == ["--tickers", "0700.HK"]

    def test_r17_robust_one_ticker_with_a_path_separator_fails_before_the_spawn(self, run, rr):
        """Pins the current seam: the ticker is interpolated raw into the log file name,
        so ``a/b`` cannot open its log; the reason names a spawn that never happened."""
        result = _one(run, rr, ticker="a/b")

        assert result["ok"] is False and result["attempts"] == 1
        assert result["reason"].startswith("spawn_failed: ")
        assert run.spawns == []


class TestRobustOneChildEnv:
    _KEYS = (
        "YIALPHA_URLOPEN_HARD_TIMEOUT_S", "YIALPHA_FAULT_DUMP_S", "YIALPHA_LLM_TIMEOUT_S",
        "YIALPHA_RESULTS_DIR", "YIALPHA_LLM_CACHE", "YIALPHA_DATA_VACUUM_POLICY", "PYTHONUNBUFFERED",
    )

    def _child_env(self, run, rr, monkeypatch, **opts):
        """Run one attempt and return ``(parent env at spawn, child env)``."""
        captured = {}

        def popen(cmd, **kwargs):
            captured["parent"] = dict(os.environ)
            captured["child"] = dict(kwargs["env"])
            return _Child([0], run)

        monkeypatch.setattr(rr.subprocess, "Popen", popen)
        for key, value in opts.items():
            setattr(run.opts, key, value)
        rr._run_one_ticker("AAPL", _DAY_ROBUST, run.opts)
        return captured["parent"], captured["child"]

    def test_r17_robust_one_child_env_defaults(self, run, rr, monkeypatch):
        for key in self._KEYS:
            monkeypatch.delenv(key, raising=False)

        parent, child = self._child_env(run, rr, monkeypatch)

        assert child == {
            **parent,
            "YIALPHA_URLOPEN_HARD_TIMEOUT_S": "20",
            "YIALPHA_FAULT_DUMP_S": "0",
            "YIALPHA_LLM_TIMEOUT_S": "120",
            "YIALPHA_RESULTS_DIR": str(run.states_root),
            "YIALPHA_LLM_CACHE": "true",
            "PYTHONUNBUFFERED": "1",
        }

    def test_r17_robust_one_child_env_inherits_keys_and_proxy(self, run, rr, monkeypatch):
        for key, value in {
            "DEEPSEEK_API_KEY": "sk-test", "HTTP_PROXY": "http://127.0.0.1:8086",
            "NO_PROXY": "api.deepseek.com", "SOCKS5_PROXY": "socks5h://127.0.0.1:1080",
            "YIALPHA_LLM_PROVIDER": "deepseek",
        }.items():
            monkeypatch.setenv(key, value)

        parent, child = self._child_env(run, rr, monkeypatch)

        for key in ("DEEPSEEK_API_KEY", "HTTP_PROXY", "NO_PROXY", "SOCKS5_PROXY", "YIALPHA_LLM_PROVIDER"):
            assert child[key] == os.environ[key]
        assert {k: v for k, v in child.items() if k not in self._KEYS} == {
            k: v for k, v in parent.items() if k not in self._KEYS
        }

    @pytest.mark.parametrize(
        ("key", "exported"),
        [
            pytest.param("YIALPHA_URLOPEN_HARD_TIMEOUT_S", "45"),
            pytest.param("YIALPHA_FAULT_DUMP_S", "30"),
            pytest.param("YIALPHA_LLM_TIMEOUT_S", "0"),
            pytest.param("YIALPHA_RESULTS_DIR", "C:/elsewhere"),
            pytest.param("YIALPHA_LLM_CACHE", "false"),
            pytest.param("PYTHONUNBUFFERED", "0"),
        ],
    )
    def test_r17_robust_one_child_env_exported_value_wins_over_the_default(
        self, run, rr, monkeypatch, key, exported
    ):
        monkeypatch.setenv(key, exported)

        _, child = self._child_env(run, rr, monkeypatch)

        assert child[key] == exported

    def test_r17_robust_one_child_env_allow_degraded_lowers_the_vacuum_gate(self, run, rr, monkeypatch):
        monkeypatch.delenv("YIALPHA_DATA_VACUUM_POLICY", raising=False)

        _, child = self._child_env(run, rr, monkeypatch, allow_degraded=True)

        assert child["YIALPHA_DATA_VACUUM_POLICY"] == "warn"

    def test_r17_robust_one_child_env_strict_default_does_not_touch_the_vacuum_gate(
        self, run, rr, monkeypatch
    ):
        monkeypatch.delenv("YIALPHA_DATA_VACUUM_POLICY", raising=False)

        _, child = self._child_env(run, rr, monkeypatch)

        assert "YIALPHA_DATA_VACUUM_POLICY" not in child

    def test_r17_robust_one_child_env_exported_vacuum_policy_wins_over_the_flag(
        self, run, rr, monkeypatch
    ):
        monkeypatch.setenv("YIALPHA_DATA_VACUUM_POLICY", "reject")

        _, child = self._child_env(run, rr, monkeypatch, allow_degraded=True)

        assert child["YIALPHA_DATA_VACUUM_POLICY"] == "reject"

    def test_r17_robust_one_child_env_no_llm_cache_overrides_an_exported_on(self, run, rr, monkeypatch):
        monkeypatch.setenv("YIALPHA_LLM_CACHE", "true")

        _, child = self._child_env(run, rr, monkeypatch, no_llm_cache=True)

        assert child["YIALPHA_LLM_CACHE"] == "false"


class TestRobustOneWatchdog:
    def test_r17_robust_one_wall_clock_kill_lands_after_the_timeout(self, run, rr, capsys):
        run.scripts = [[None]]  # alive at every poll
        run.opts.per_ticker_timeout = 100
        run.opts.max_attempts = 1

        result = _one(run, rr)

        # The loop sleeps 5 s between polls: 105 s is the first poll past 100 s.
        assert result["ok"] is False and result["attempts"] == 1
        assert result["reason"] == "killed: wall_clock 105s > 100s (wall 105s)"
        assert run.kill_calls == [4242]
        assert run.spawns[0]["child"].waits == [30.0]  # _reap waits with a bound, once
        assert run.clock.sleeps == [5.0] * 21
        out = capsys.readouterr().out.splitlines()
        assert out[1] == f"[AAPL] {_WARN} killed attempt 1: wall_clock 105s > 100s (wall 105s)"
        assert len(out) == 2  # last attempt: no backoff line

    def test_r17_robust_one_wall_clock_exactly_at_the_limit_is_still_inside(
        self, run, rr
    ):
        """``elapsed > timeout`` is strict: the poll taken at exactly 10 s spares the child."""
        run.scripts = [[None]]  # alive at every poll
        run.opts.per_ticker_timeout = 10
        run.opts.max_attempts = 1

        result = _one(run, rr)

        assert result["reason"] == "killed: wall_clock 15s > 10s (wall 15s)"
        assert run.clock.sleeps == [5.0] * 3  # polls at 0, 5 and 10 s all let it live

    def test_r17_robust_one_child_exiting_at_the_limit_is_a_clean_success(self, run, rr):
        run.opts.per_ticker_timeout = 10
        run.scripts = [[None, None, 0]]  # gone on the poll taken at 10 s

        result = _one(run, rr)

        assert result["ok"] is True and result["reason"] == "ok (wall 10s)"
        assert run.kill_calls == []
        assert run.spawns[0]["child"].waits == []

    def test_r17_robust_one_reap_escalates_when_the_killed_tree_holds_the_pipe(
        self, run, rr, capsys
    ):
        run.scripts = [[None]]  # alive at every poll
        run.opts.max_attempts = 1
        run.opts.per_ticker_timeout = 10
        run.wait_hangs = True
        run.exit_after_kill = 9

        result = _one(run, rr)

        assert result["reason"] == "killed: wall_clock 15s > 10s (wall 15s)"
        assert capsys.readouterr().err.splitlines() == [
            "[AAPL] child did not exit within 30s after kill; force-killing and reaping"
        ]

    @pytest.mark.parametrize(
        ("cpu", "expected_kills_at", "reason"),
        [
            pytest.param([10.0, 10.2], 20, "cpu stall: 0.20s over 20s window", id="idle-in-first-window"),
            pytest.param([10.0, 11.0, 11.1], 40, "cpu stall: 0.10s over 20s window", id="busy-then-idle"),
            # the first window sits exactly on the threshold (no stall), the second is idle
            pytest.param([10.0, 10.5, 10.5], 40, "cpu stall: 0.00s over 20s window", id="at-threshold-then-idle"),
        ],
    )
    def test_r17_robust_one_cpu_stall_detection(self, run, rr, capsys, cpu, expected_kills_at, reason):
        run.opts.stall_timeout = 20
        run.opts.stall_cpu = 0.5
        run.opts.max_attempts = 1
        run.scripts = [[None]]  # alive at every poll
        run.cpu = _Seq(*cpu)

        result = _one(run, rr)

        assert result["reason"] == f"killed: {reason} (wall {expected_kills_at}s)"
        assert run.kill_calls == [4242]
        # CPU is sampled at spawn and once per elapsed window.
        assert run.cpu.calls == len(cpu)
        assert capsys.readouterr().out.splitlines()[1] == (
            f"[AAPL] {_WARN} killed attempt 1: {reason} (wall {expected_kills_at}s)"
        )

    def test_r17_robust_one_stall_window_whose_start_sample_is_unknown_judges_the_next_one(
        self, run, rr
    ):
        run.scripts = [[None]]  # alive at every poll
        run.opts.stall_timeout = 20
        run.opts.max_attempts = 1
        run.cpu = _Seq(None, 4.0, 4.1)

        result = _one(run, rr)

        assert result["reason"] == "killed: cpu stall: 0.10s over 20s window (wall 40s)"
        assert run.cpu.calls == 3

    def test_r17_robust_one_stall_detection_with_no_cpu_numbers_never_kills(self, run, rr):
        run.opts.stall_timeout = 20
        run.scripts = [[None] * 5 + [0]]  # alive through the poll at 20 s, gone at 25 s
        run.cpu = _Seq(None)

        result = _one(run, rr)

        assert result["ok"] is True and result["reason"] == "ok (wall 25s)"
        assert run.kill_calls == []
        assert run.cpu.calls == 2  # at spawn and once when the first window has elapsed

    def test_r17_robust_one_stall_detection_off_by_default_never_probes_cpu(self, run, rr):
        run.scripts = [[None, None, 0]]

        result = _one(run, rr)

        assert result["ok"] is True
        assert run.cpu.calls == 0

    def test_r17_robust_one_wall_clock_wins_when_it_comes_first(self, run, rr):
        run.scripts = [[None]]  # alive at every poll
        run.opts.per_ticker_timeout = 10
        run.opts.stall_timeout = 30
        run.opts.max_attempts = 1
        run.cpu = _Seq(1.0)

        result = _one(run, rr)

        assert result["reason"] == "killed: wall_clock 15s > 10s (wall 15s)"
        assert run.cpu.calls == 1


class TestRobustOneVerdict:
    def test_r17_robust_one_clean_exit_pops_a_degraded_marker_left_by_an_earlier_attempt(
        self, run, rr, capsys
    ):
        run.sentinels = _Seq(3, 0)

        result = _one(run, rr)

        assert result["attempts"] == 2 and result["ok"] is True
        assert result["reason"] == "ok (wall 0s)" and "degraded" not in result
        assert run.clock.sleeps == [15.0]
        out = capsys.readouterr().out.splitlines()
        assert out[1] == f"[AAPL] {_WARN} attempt 1: DEGRADED: 3 core data sentinel(s) (wall 0s) → retry (--require-data-quality)"
        assert out[2] == f"[AAPL] {_HOURGLASS} backoff 15s before retry…"

    @pytest.mark.parametrize("max_attempts", [1, 3])
    def test_r17_robust_one_exit_zero_without_a_new_report_is_a_failure(
        self, run, rr, capsys, max_attempts
    ):
        run.opts.max_attempts = max_attempts
        run.reports = _Seq(None)

        result = _one(run, rr)

        assert result["ok"] is False and result["attempts"] == max_attempts
        assert result["reason"] == "exit 0 but no new complete_report.md"
        assert result["report_path"] is None
        assert len(run.spawns) == max_attempts
        assert run.clock.sleeps == [15.0] * (max_attempts - 1)
        out = capsys.readouterr().out.splitlines()
        assert out[1] == f"[AAPL] {_WARN} exit 0 but no new complete_report.md → retry"
        # One log per attempt.
        names = sorted(p.name for p in _log_dir(run).iterdir())
        assert [n[-7:] for n in names] == [f"_a{i}.log" for i in range(1, max_attempts + 1)]

    @pytest.mark.parametrize("code", [1, 2, -9, 137])
    def test_r17_robust_one_non_zero_exit_is_retried_then_reported_with_its_code(
        self, run, rr, capsys, code
    ):
        run.scripts = [[code]]
        run.opts.max_attempts = 2

        result = _one(run, rr)

        assert result["ok"] is False and result["reason"] == f"exit {code}"
        assert result["report_path"] is None
        assert run.reports.calls == 0 and run.sentinels.calls == 0  # no report lookup on a crash
        out = capsys.readouterr().out.splitlines()
        assert out[1] == f"[AAPL] {_WARN} attempt 1 failed (exit {code}, wall 0s) → retry"
        assert out[-1] == f"[AAPL] {_WARN} attempt 2 failed (exit {code}, wall 0s) → retry"

    def test_r17_robust_one_strict_and_evidence_unavailable_retries(self, run, rr, capsys):
        run.sentinels = _Seq(None)
        run.opts.max_attempts = 2

        result = _one(run, rr)

        assert result["ok"] is False and result["degraded"] is True
        assert result["reason"] == (
            "DEGRADED: data-quality evidence unavailable (no readable data_quality block in "
            "full_states_log) (wall 0s) → retry (--require-data-quality)"
        )
        assert result["report_path"] is None
        assert run.sentinels.calls == 2
        assert capsys.readouterr().out.splitlines()[1] == f"[AAPL] {_WARN} attempt 1: {result['reason']}"

    def test_r17_robust_one_lenient_and_evidence_unavailable_is_ok_with_quality_unknown(
        self, run, rr, capsys
    ):
        run.sentinels = _Seq(None)

        # what --allow-degraded leaves in the namespace: the strict flag is derived
        result = _one(run, rr, allow_degraded=True, require_data_quality=False)

        assert result["ok"] is True and result["attempts"] == 1
        assert result["reason"] == "ok (quality unknown; wall 0s)"
        assert result["report_path"] == run.reports.values[0]
        assert "degraded" not in result
        report = run.reports.values[0]
        assert capsys.readouterr().out.splitlines()[1] == (
            f"[AAPL] {_OK} attempt 1 done in 0s → {report} — data-quality evidence unavailable (older run log?)"
        )

    def test_r17_robust_one_strict_and_core_sentinels_retries(self, run, rr, capsys):
        run.sentinels = _Seq(2)
        run.opts.max_attempts = 2

        result = _one(run, rr)

        assert result["ok"] is False and result["degraded"] is True
        assert result["reason"] == "DEGRADED: 2 core data sentinel(s) (wall 0s) → retry (--require-data-quality)"
        assert capsys.readouterr().out.splitlines()[1] == f"[AAPL] {_WARN} attempt 1: {result['reason']}"

    def test_r17_robust_one_lenient_and_core_sentinels_is_ok_but_degraded(self, run, rr, capsys):
        run.sentinels = _Seq(2)

        result = _one(run, rr, allow_degraded=True, require_data_quality=False)

        assert result["attempts"] == 1
        assert result["ok"] is True and result["degraded"] is True
        assert result["reason"] == "DEGRADED: 2 core data sentinel(s) (wall 0s)"
        report = run.reports.values[0]
        assert result["report_path"] == report
        assert capsys.readouterr().out.splitlines()[1] == (
            f"[AAPL] {_WARN} DEGRADED done in 0s → {report} — 2 core category(ies) had NO data "
            "(see data_quality in full_states_log)"
        )

    def test_r17_robust_one_last_strict_attempt_still_says_retry_in_the_reason(self, run, rr):
        run.sentinels = _Seq(2)
        run.opts.max_attempts = 1

        result = _one(run, rr)

        assert result["reason"].endswith("→ retry (--require-data-quality)")
        assert run.clock.sleeps == []

    @pytest.mark.parametrize("backoff", [0.0, 2.5, 120.0])
    def test_r17_robust_one_backoff_between_attempts_only(self, run, rr, capsys, backoff):
        run.scripts = [[7]]
        run.opts.backoff = backoff
        run.opts.max_attempts = 3

        _one(run, rr)

        assert run.clock.sleeps == [backoff, backoff]
        out = capsys.readouterr().out.splitlines()
        assert out[2] == f"[AAPL] {_HOURGLASS} backoff {backoff:.0f}s before retry…"
        assert out[5] == out[2] and len(out) == 8  # the last failure still says "→ retry"
        assert out[-1] == f"[AAPL] {_WARN} attempt 3 failed (exit 7, wall 0s) → retry"

    def test_r17_robust_one_result_log_path_is_the_last_attempt(self, run, rr):
        run.scripts = [[7], [7], [0]]

        result = _one(run, rr)

        assert result["ok"] is True and result["attempts"] == 3
        assert result["log_path"].name.endswith("_a3.log")
        assert sorted(p.name[-7:] for p in _log_dir(run).iterdir()) == ["_a1.log", "_a2.log", "_a3.log"]

    def test_r17_robust_one_no_backoff_once_aborted(self, run, rr, capsys):
        """Round-17: after Ctrl+C the worker slept the backoff all the same.

        ``main()``'s handler sets ``_abort`` and kills the children, then joins
        the pool; a worker whose child just died only noticed the abort at the
        top of its NEXT iteration -- after ``time.sleep(backoff)`` (15 s by
        default, ``--backoff 300`` made it five minutes) with no way to interrupt
        the sleep on Windows.
        """
        run.opts.max_attempts = 3
        run.opts.backoff = 300.0

        def abort_during_attempt(child, code):
            run.on_poll = None
            rr._abort.set()        # what main()'s Ctrl+C handler does
            child._codes = [1]     # _kill_all_active() killed it: the next poll sees it gone
            return None            # this poll still sees it alive

        run.on_poll = abort_during_attempt

        result = _one(run, rr)

        assert result["ok"] is False and result["attempts"] == 1
        assert len(run.spawns) == 1
        assert [s for s in run.clock.sleeps if s != 5.0] == []
        assert f"{_HOURGLASS} backoff" not in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# main()
# --------------------------------------------------------------------------- #


_SHIELD = "\U0001f6e1️"
_STOP = "\U0001f6d1"
_ABORTED = f"{_STOP} 已中断：已强杀所有活跃子进程，等待工作线程退出…"
_SUMMARY_TITLE = "=== 健壮编排完成：{ok}/{total} 成功 ==="
_SUMMARY_HEADER = "ticker".ljust(10) + " " + "状态".ljust(6) + " " + "尝试".rjust(4) + "  " + "报告/原因"


def _robust_row(ticker, status, attempts, detail):
    """One summary row: 10-wide ticker, glyph, 4-wide attempt count, detail."""
    return ticker.ljust(10) + " " + status.ljust(6) + " " + str(attempts).rjust(4) + "  " + str(detail)


def _robust_result(ticker, *, ok, attempts, reason="", report=None, log=None, **extra):
    return {
        "ticker": ticker, "ok": ok, "attempts": attempts, "reason": reason,
        "report_path": report, "log_path": log, **extra,
    }


@pytest.fixture
def main_run(rr, monkeypatch, tmp_path):
    """``main()`` with ``_run_one_ticker`` replaced by a per-ticker script and a recording pool.

    A ticker without a script succeeds at once; ``stage.scripted[ticker]`` is
    either an exception to raise or a callable returning the result dict.
    """
    stage = SimpleNamespace(calls=[], scripted={}, pool_workers=[], shutdowns=[], kill_all=0)

    def report_of(ticker):
        return tmp_path / "logs" / "reports" / f"{ticker}_20260701_120000" / "complete_report.md"

    def log_of(ticker):
        return tmp_path / "logs" / "robust" / f"robust_{ticker}_20260701_120000_a1.log"

    def fake_one(ticker, date, opts):
        stage.calls.append((ticker, date, opts))
        action = stage.scripted.get(ticker)
        if isinstance(action, BaseException):
            raise action
        if callable(action):
            return action(ticker)
        return _robust_result(
            ticker, ok=True, attempts=1, reason="ok (wall 3s)",
            report=report_of(ticker), log=log_of(ticker),
        )

    monkeypatch.setattr(rr, "_run_one_ticker", fake_one)

    real_pool = rr.ThreadPoolExecutor

    class _Pool(real_pool):
        def __init__(self, max_workers=None, *args, **kwargs):
            super().__init__(max_workers, *args, **kwargs)
            stage.pool_workers.append(max_workers)

        def shutdown(self, wait=True, *args, **kwargs):
            stage.shutdowns.append(wait)
            super().shutdown(wait, *args, **kwargs)

    monkeypatch.setattr(rr, "ThreadPoolExecutor", _Pool)

    real_kill_all = rr._kill_all_active

    def kill_all():
        stage.kill_all += 1
        real_kill_all()

    monkeypatch.setattr(rr, "_kill_all_active", kill_all)
    stage.report_of, stage.log_of = report_of, log_of
    return stage


def _robust_main(rr, monkeypatch, capsys, *argv):
    """``main()`` reads ``sys.argv`` itself; return ``(code, stdout, stderr)``.

    The analysis date is always ``_DAY_ROBUST``; ``argv`` carries the rest.
    """
    _set_argv(monkeypatch, "run_robust.py", ["--date", _DAY_ROBUST, *argv])
    code = rr.main()
    captured = capsys.readouterr()
    return code, captured.out, captured.err


class TestRobustMain:
    def test_r17_robust_main_mixed_outcome_summary_sorted_by_the_ticker_list(
        self, main_run, rr, monkeypatch, capsys, tmp_path
    ):
        nvda_log = tmp_path / "logs" / "robust" / "robust_NVDA_20260701_120000_a3.log"
        main_run.scripted["NVDA"] = lambda t: _robust_result(
            t, ok=False, attempts=3, reason="exit 2", log=nvda_log
        )

        code, out, err = _robust_main(
            rr, monkeypatch, capsys, "--tickers", "NVDA", "AAPL", "--workers", "2"
        )

        assert code == 1
        opts = main_run.calls[0][2]
        assert sorted(t for t, _, _ in main_run.calls) == ["AAPL", "NVDA"]
        assert {d for _, d, _ in main_run.calls} == {_DAY_ROBUST}
        assert all(o is opts for _, _, o in main_run.calls)
        assert opts.tickers == ["NVDA", "AAPL"]
        assert main_run.pool_workers == [2] and main_run.shutdowns == [True]
        assert err == ""
        assert out.splitlines() == [
            f"{_SHIELD} robust orchestrator | tickers=['NVDA', 'AAPL'] date={_DAY_ROBUST} "
            "workers=2 per_ticker_timeout=1800s max_attempts=3 stall_timeout=0s",
            "",
            _SUMMARY_TITLE.format(ok=1, total=2),
            _SUMMARY_HEADER,
            _robust_row("NVDA", _FAIL, 3, f"exit 2 (log={nvda_log})"),
            _robust_row("AAPL", _OK, 1, main_run.report_of("AAPL")),
        ]

    def test_r17_robust_main_all_ok_exit_0(self, main_run, rr, monkeypatch, capsys):
        code, out, err = _robust_main(rr, monkeypatch, capsys, "--tickers", "AAPL", "NVDA", "MSFT")

        assert code == 0
        assert err == ""
        lines = out.splitlines()
        assert lines[2] == _SUMMARY_TITLE.format(ok=3, total=3)
        assert lines[3:] == [
            _SUMMARY_HEADER, *(_robust_row(t, _OK, 1, main_run.report_of(t)) for t in ("AAPL", "NVDA", "MSFT"))
        ]
        assert "log=" not in out  # a success prints its report, never a log

    def test_r17_robust_main_header_mirrors_the_flags(self, main_run, rr, monkeypatch, capsys):
        code, out, _ = _robust_main(
            rr, monkeypatch, capsys, "--tickers", "AAPL", "--workers", "4",
            "--per-ticker-timeout", "900.4", "--max-attempts", "5", "--stall-timeout", "120",
            "--stall-cpu", "2", "--backoff", "60", "--allow-degraded", "--no-llm-cache",
        )

        assert code == 0
        assert out.splitlines()[0] == (
            f"{_SHIELD} robust orchestrator | tickers=['AAPL'] date={_DAY_ROBUST} "
            "workers=4 per_ticker_timeout=900s max_attempts=5 stall_timeout=120s"
        )
        assert main_run.pool_workers == [4]

    @pytest.mark.parametrize(
        "error",
        [
            pytest.param(RuntimeError("boom"), id="runtime-error"),
            pytest.param(ValueError("odd"), id="value-error"),
            pytest.param(SystemExit(3), id="system-exit-in-a-worker"),
        ],
    )
    def test_r17_robust_main_one_ticker_error_is_isolated(
        self, main_run, rr, monkeypatch, capsys, error
    ):
        """Any exception but KeyboardInterrupt becomes that ticker's failure row."""
        main_run.scripted["NVDA"] = error

        code, out, err = _robust_main(
            rr, monkeypatch, capsys, "--tickers", "AAPL", "NVDA", "--workers", "2"
        )

        assert code == 1
        assert err.splitlines() == [f"[NVDA] ❌ orchestrator error: {error!r}"]
        lines = out.splitlines()
        assert lines[2] == _SUMMARY_TITLE.format(ok=1, total=2)
        assert lines[4:] == [
            _robust_row("AAPL", _OK, 1, main_run.report_of("AAPL")),
            _robust_row("NVDA", _FAIL, 0, f"orchestrator_error: {error!r} (log=None)"),
        ]
        assert main_run.shutdowns == [True]

    def test_r17_robust_main_every_ticker_erroring_still_prints_a_summary(self, main_run, rr, monkeypatch, capsys):
        main_run.scripted = {"AAPL": RuntimeError("a"), "NVDA": RuntimeError("b")}

        code, out, err = _robust_main(rr, monkeypatch, capsys, "--tickers", "AAPL", "NVDA")

        assert code == 1
        assert sorted(err.splitlines()) == [
            "[AAPL] ❌ orchestrator error: RuntimeError('a')",
            "[NVDA] ❌ orchestrator error: RuntimeError('b')",
        ]
        assert out.splitlines()[2] == _SUMMARY_TITLE.format(ok=0, total=2)

    def test_r17_robust_main_keyboard_interrupt_in_a_worker_aborts_and_kills_children(
        self, main_run, rr, monkeypatch, capsys
    ):
        """The interrupt is handed to the handler: abort flag set, children killed,
        the ticker that already finished kept, the interrupted one listed as not run."""
        main_run.scripted["AAA"] = KeyboardInterrupt()

        code, out, err = _robust_main(
            rr, monkeypatch, capsys, "--tickers", "BBB", "AAA", "--workers", "1"
        )

        assert code == 1
        assert rr._abort.is_set()
        assert main_run.kill_all == 1
        assert err == f"\n{_ABORTED}\n\n⚠ 1 个 ticker 未完成（中断/编排失败，未运行）: AAA\n"
        lines = out.splitlines()
        assert lines[2] == _SUMMARY_TITLE.format(ok=1, total=2)
        assert lines[4:] == [_robust_row("BBB", _OK, 1, main_run.report_of("BBB"))]
        assert main_run.shutdowns == [True]

    def test_r17_robust_main_interrupt_before_any_result_reports_every_ticker(
        self, main_run, rr, monkeypatch, capsys
    ):
        main_run.scripted["AAA"] = KeyboardInterrupt()

        code, out, err = _robust_main(
            rr, monkeypatch, capsys, "--tickers", "AAA", "BBB", "--workers", "1"
        )

        assert code == 1
        assert err.splitlines() == [
            "", _ABORTED, "", "⚠ 2 个 ticker 未完成（中断/编排失败，未运行）: AAA, BBB",
        ]
        assert out.splitlines()[2:] == [_SUMMARY_TITLE.format(ok=0, total=2), _SUMMARY_HEADER]

    def test_r17_robust_main_ctrl_c_while_waiting_kills_the_registered_children(
        self, main_run, rr, monkeypatch, capsys
    ):
        """Ctrl+C lands in ``as_completed``: every child still running is killed before the join."""
        killed = []
        monkeypatch.setattr(rr, "_kill_tree", killed.append)
        rr._register_proc(SimpleNamespace(pid=111, poll=lambda: None))
        rr._register_proc(SimpleNamespace(pid=222, poll=lambda: 0))

        def interrupted(futs):
            raise KeyboardInterrupt
            yield  # pragma: no cover -- only makes this a generator function

        monkeypatch.setattr(rr, "as_completed", interrupted)

        code, out, err = _robust_main(rr, monkeypatch, capsys, "--tickers", "AAPL")

        assert code == 1
        assert killed == [111]  # the child that already exited is left alone
        assert rr._abort.is_set() and main_run.shutdowns == [True]
        assert out.splitlines()[2] == _SUMMARY_TITLE.format(ok=0, total=1)
        assert err.splitlines()[1] == _ABORTED

    def test_r17_robust_main_duplicate_tickers_are_run_once(self, main_run, rr, monkeypatch, capsys):
        """Round-17, end to end: the list is de-duplicated before the pool sees it."""
        code, out, err = _robust_main(
            rr, monkeypatch, capsys, "--tickers", "AAPL", "NVDA", "aapl", "--workers", "2"
        )

        assert code == 0
        assert sorted(t for t, _, _ in main_run.calls) == ["AAPL", "NVDA"]
        assert err == "⚠ removed 1 duplicate ticker(s) from --tickers: ['aapl'] (each ticker is run once)\n"
        lines = out.splitlines()
        assert lines[0].startswith(f"{_SHIELD} robust orchestrator | tickers=['AAPL', 'NVDA'] date=")
        assert lines[2:] == [
            _SUMMARY_TITLE.format(ok=2, total=2), _SUMMARY_HEADER,
            _robust_row("AAPL", _OK, 1, main_run.report_of("AAPL")),
            _robust_row("NVDA", _OK, 1, main_run.report_of("NVDA")),
        ]

    def test_r17_robust_main_end_to_end_with_the_real_per_ticker_loop(self, run, rr, monkeypatch, capsys):
        run.scripts = [[0], [1], [1]]  # AAPL exits 0 at once; NVDA exits 1 on both attempts

        code, out, err = _robust_main(
            rr, monkeypatch, capsys,
            "--tickers", "AAPL", "NVDA", "--workers", "1", "--max-attempts", "2",
        )

        assert code == 1 and err == ""
        assert [s["cmd"][3] for s in run.spawns] == ["AAPL", "NVDA", "NVDA"]
        assert not rr._abort.is_set() and rr._active_procs == []
        nvda_log = run.spawns[2]["stdout"].name
        lines = out.splitlines()
        assert lines[0].startswith(_SHIELD)
        assert lines[8:] == [
            "",
            _SUMMARY_TITLE.format(ok=1, total=2),
            _SUMMARY_HEADER,
            _robust_row("AAPL", _OK, 1, run.reports.values[0]),
            _robust_row("NVDA", _FAIL, 2, f"exit 1 (log={nvda_log})"),
        ]
        assert "log=" not in lines[11] and nvda_log.endswith("_a2.log")


def test_r17_robust_create_flags_follow_the_platform(rr):
    expected = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
    assert expected == rr._CREATE_FLAGS


def test_r17_robust_main_guard_exits_with_main_code(monkeypatch, capsys):
    """``python scripts/run_robust.py`` exits with ``main()``'s status."""
    path = SCRIPTS_DIR / "run_robust.py"
    code = compile(path.read_text(encoding="utf-8"), str(path), "exec")
    outcomes = []

    for argv in (["--workers", "0"], ["--per-ticker-timeout", "nan"], ["--bogus"]):
        _set_argv(monkeypatch, "run_robust.py", ["--tickers", "AAPL", "--date", _DAY_ROBUST, *argv])
        with pytest.raises(SystemExit) as excinfo:
            exec(code, {"__name__": "__main__", "__file__": str(path)})  # noqa: S102
        outcomes.append(excinfo.value.code)
        capsys.readouterr()

    assert outcomes == [2, 2, 2]


# =========================================================================== #
# The sys.path shim shared by the entry scripts
# =========================================================================== #


@pytest.mark.parametrize(
    "filename",
    [
        "analyze_window.py", "export_ic_dataset.py", "prune_indicators_cli.py", "rank_signals.py",
        "run_analyst_parallel_ab.py", "run_baseline.py", "run_batch.py",
    ],
)
def test_r17_scripts_shim_puts_scripts_dir_on_sys_path_and_bootstrap_the_root_in_front(
    monkeypatch, filename
):
    """Each entry script adds its own directory (for ``_bootstrap``); the shared
    bootstrap then puts the project root in front of it."""
    scripts_dir, root = str(SCRIPTS_DIR), str(PROJECT_ROOT)
    without = [p for p in sys.path if p not in (scripts_dir, root)]
    monkeypatch.setattr(sys, "path", without)
    seen = {}

    _load(f"r17_shim_{filename}", filename, on_executed=lambda path: seen.update(path=list(path)))

    assert seen["path"][0] == root
    assert seen["path"][1] == scripts_dir
    assert seen["path"].count(scripts_dir) == 1
    assert seen["path"][2:] == without
    # The loader hands sys.path back as it found it.
    assert sys.path == without


# =========================================================================== #
# scripts/trade_ticket.py — _fnum
# =========================================================================== #


@pytest.fixture(scope="module")
def tt():
    return _load("r17_trade_ticket", "trade_ticket.py")


class TestTradeTicketFnum:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            pytest.param("1" * 400, float("inf"), id="400-digit-integer-overflows-to-inf"),
            pytest.param("9" * 400, float("inf"), id="large-integer-overflows-to-inf"),
            pytest.param("1e" + "9" * 400, float("inf"), id="huge-exponent"),
            pytest.param("-1e999", float("-inf"), id="negative-overflow"),
            pytest.param("1e-" + "9" * 400, 0.0, id="tiny-exponent-underflows-to-zero"),
            pytest.param("12e0345", float("inf"), id="exponent-345-overflows"),
            pytest.param("1e", 1.0, id="exponent-without-digits-is-dropped"),
            pytest.param("1e+", 1.0, id="exponent-sign-without-digits"),
            pytest.param("1E-", 1.0, id="upper-e-minus"),
            pytest.param("1e+2.5", 100.0, id="fractional-exponent-stops-at-the-dot"),
            pytest.param("1..2", 1.0, id="second-dot-ends-the-number"),
            pytest.param("1,,2", 12.0, id="commas-anywhere-are-removed"),
            pytest.param("1,2,3,4,5,6,7,8,9,0", 1234567890.0, id="every-digit-comma-grouped"),
            pytest.param(",,,1", 1.0, id="leading-commas-skipped"),
            pytest.param("1.", 1.0, id="trailing-dot"),
            pytest.param("+5", 5.0, id="plus-sign-is-not-part-of-the-number"),
            pytest.param("5-", 5.0, id="minus-after-is-not-part-of-the-number"),
            pytest.param("--1", -1.0, id="double-minus"),
            pytest.param("1-2", 1.0, id="range-keeps-the-first"),
            pytest.param("1/2", 1.0, id="fraction-keeps-the-first"),
            pytest.param("0x1F", 0.0, id="hexadecimal-reads-zero"),
            pytest.param("1_000", 1.0, id="underscore-grouping-reads-one"),
            pytest.param("²3", 3.0, id="superscript-is-not-a-digit"),
        ],
    )
    def test_r17_fnum_numeric_edges(self, tt, text, expected):
        out = tt._fnum(text)

        assert out == expected
        assert math.copysign(1.0, out) == math.copysign(1.0, expected)

    @pytest.mark.parametrize(
        "text",
        ["nan", "inf", "-inf", "Infinity", "NaN", "½", "①②", "²", "n/a", "-", ".", ",", "+", " "],
    )
    def test_r17_fnum_no_ascii_digit_is_none(self, tt, text):
        assert tt._fnum(text) is None

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            pytest.param("١٢٣", 123.0, id="arabic-indic"),
            pytest.param("१२३", 123.0, id="devanagari"),
            pytest.param("１２３", 123.0, id="fullwidth"),
            pytest.param("𝟙𝟚𝟛", 123.0, id="mathematical-bold"),
            pytest.param("１,２３", 123.0, id="fullwidth-with-comma"),
            pytest.param("1,٢", 12.0, id="mixed-scripts"),
            pytest.param("١e٢", 100.0, id="arabic-indic-exponent"),
            pytest.param("٣٫٥", 3.0, id="arabic-decimal-separator-stops-the-number"),
        ],
    )
    def test_r17_fnum_unicode_decimal_digits_are_digits(self, tt, text, expected):
        """``\\d`` matches every Unicode decimal digit and ``float()`` accepts them."""
        assert tt._fnum(text) == expected

    def test_r17_fnum_every_unicode_decimal_digit_runs_clean(self, tt):
        import unicodedata

        digits = [chr(c) for c in range(0x110000) if unicodedata.category(chr(c)) == "Nd"]

        assert len(digits) > 600
        for digit in digits:
            assert tt._fnum(digit + digit) == float(digit + digit)

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            pytest.param(".5", 5.0, id="leading-dot"),
            pytest.param("-.5", 5.0, id="minus-leading-dot-loses-sign-and-magnitude"),
            pytest.param("Entry .0050", 50.0, id="sub-dollar-leading-dot"),
        ],
    )
    def test_r17_fnum_leading_dot_decimal_reads_only_the_digits(self, tt, text, expected):
        """Pins the current seam (reported, not changed): the pattern needs a digit
        before the dot, so ``.5`` is read as 5, ``-.5`` loses its sign and
        ``.0050`` is read as 50. The overlay renders with ``%g`` (always
        ``0.5``); an LLM writing ``.5`` in a parsed field (entry / stop /
        price target) would be read ten times too large. No local report or
        state log contains the form."""
        assert tt._fnum(text) == expected

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            pytest.param(5, 5.0, id="int"),
            pytest.param(5.5, 5.5, id="float"),
            pytest.param(-0.0, 0.0, id="negative-zero-keeps-its-sign"),
            pytest.param(True, None, id="true-has-no-digit"),
            pytest.param(float("nan"), None, id="nan"),
            pytest.param(float("inf"), None, id="inf"),
            pytest.param([1], 1.0, id="list-is-stringified"),
            pytest.param({"a": 1}, 1.0, id="dict-is-stringified"),
            pytest.param(b"12", 12.0, id="bytes"),
        ],
    )
    def test_r17_fnum_non_string_input_is_stringified_first(self, tt, value, expected):
        out = tt._fnum(value)

        assert out == expected
        if isinstance(value, float) and value == 0.0:
            assert math.copysign(1.0, out) == math.copysign(1.0, value)

    def test_r17_fnum_value_error_branch_is_unreachable(self, tt, monkeypatch):
        """Why ``_fnum``'s ``except ValueError`` (the two uncovered lines) stays uncovered.

        Whatever the pattern matches is a run of digits/commas with an optional
        dot and exponent; ``float()`` takes every such string (overflow gives
        inf, never an exception). Exhaustive over a 9-character alphabet up to
        length 4 and instrumented: ``float()`` never raises.
        """
        import itertools

        raised = []
        real_float = float

        def spy(text):
            try:
                return real_float(text)
            except ValueError:
                raised.append(text)
                raise

        monkeypatch.setattr(tt, "float", spy, raising=False)  # a module global shadows the builtin
        tried = 0
        for length in range(5):
            for chars in itertools.product("0,.-+eE a", repeat=length):
                tt._fnum("".join(chars))
                tried += 1

        assert tried == 1 + 9 + 81 + 729 + 6561
        assert raised == []


# =========================================================================== #
# scripts/smoke_structured_output.py
# =========================================================================== #


@pytest.fixture
def sso():
    return _load("r17_smoke_structured", "smoke_structured_output.py")


def test_r17_sso_shim_puts_the_project_root_on_sys_path(monkeypatch):
    root = str(PROJECT_ROOT)
    without = [p for p in sys.path if p != root]
    monkeypatch.setattr(sys, "path", without)
    seen = {}

    _load("r17_sso_shim", "smoke_structured_output.py", on_executed=lambda path: seen.update(path=list(path)))

    assert seen["path"] == [root, *without]
    assert sys.path == without


class TestSsoConstants:
    def test_r17_sso_provider_defaults(self, sso):
        assert sso.PROVIDER_DEFAULTS == {
            "openai": ("gpt-5.4-mini", None),
            "google": ("gemini-3.5-flash", None),
            "anthropic": ("claude-sonnet-4-6", None),
            "deepseek": ("deepseek-v4-pro", None),
            "qwen": ("qwen3.7-plus", None),
            "glm": ("glm-5", None),
            "xai": ("grok-4.3", None),
        }

    def test_r17_sso_state_builders(self, sso):
        rm = sso._make_rm_state()
        trader = sso._make_trader_state("PLAN-RM")
        pm = sso._make_pm_state("PLAN-RM", "PLAN-TRADER")

        assert set(rm) == {"company_of_interest", "investment_debate_state"}
        assert rm["company_of_interest"] == "NVDA"
        debate = rm["investment_debate_state"]
        assert set(debate) == {
            "history", "bull_history", "bear_history", "current_response", "judge_decision", "count",
        }
        assert debate["history"] == sso.DEBATE_HISTORY
        assert (debate["current_response"], debate["judge_decision"], debate["count"]) == ("", "", 1)
        assert trader == {"company_of_interest": "NVDA", "investment_plan": "PLAN-RM"}
        assert set(pm) == {
            "company_of_interest", "past_context", "risk_debate_state", "market_report",
            "sentiment_report", "news_report", "fundamentals_report", "investment_plan",
            "trader_investment_plan",
        }
        assert (pm["company_of_interest"], pm["past_context"]) == ("NVDA", "")
        assert (pm["investment_plan"], pm["trader_investment_plan"]) == ("PLAN-RM", "PLAN-TRADER")
        assert (pm["market_report"], pm["fundamentals_report"]) == ("Market report.", "Fundamentals report.")
        risk = pm["risk_debate_state"]
        assert set(risk) == {
            "history", "aggressive_history", "conservative_history", "neutral_history",
            "judge_decision", "current_aggressive_response", "current_conservative_response",
            "current_neutral_response", "count",
        }
        assert (risk["judge_decision"], risk["count"]) == ("", 1)
        # Fresh dicts every call: one agent cannot pollute the next run.
        assert sso._make_rm_state() is not rm and sso._make_rm_state()["investment_debate_state"] is not debate

    def test_r17_sso_print_section(self, sso, capsys):
        bar = "=" * 70

        sso._print_section("[1] Title", "line one\nline two")

        assert capsys.readouterr().out == f"\n{bar}\n[1] Title\n{bar}\nline one\nline two\n"


@pytest.fixture
def chain(sso, monkeypatch):
    """The LLM client factory and the three agent factories become recorders."""
    stage = SimpleNamespace(
        clients=[], llms=[], agents={}, order=[],
        rm_plan="**Recommendation**: Buy\n\nPlan body.",
        trader_plan="**Action**: BUY\n\nFINAL TRANSACTION PROPOSAL: BUY\n",
        pm_decision="**Rating**: Buy\n**Executive Summary**: s\n**Investment Thesis**: t\n",
        rating=None,
    )

    class _Client:
        def __init__(self, provider, model):
            self.provider, self.model = provider, model

        def get_llm(self):
            llm = f"<{self.provider}:{self.model}>"
            stage.llms.append(llm)
            return llm

    def create_llm_client(provider, model):
        client = _Client(provider, model)
        stage.clients.append(client)
        return client

    def agent(name, builds):
        def factory(llm):
            stage.agents[name] = llm

            def run(state):
                stage.order.append((name, state))
                return builds(state)
            return run
        return factory

    monkeypatch.setattr(sso, "create_llm_client", create_llm_client)
    monkeypatch.setattr(sso, "create_research_manager", agent("rm", lambda state: {"investment_plan": stage.rm_plan}))
    monkeypatch.setattr(sso, "create_trader", agent("trader", lambda state: {"trader_investment_plan": stage.trader_plan}))
    monkeypatch.setattr(
        sso, "create_portfolio_manager", agent("pm", lambda state: {"final_trade_decision": stage.pm_decision})
    )
    return stage


def _sso_main(sso, monkeypatch, capsys, *argv):
    _set_argv(monkeypatch, "smoke_structured_output.py", list(argv))
    code = sso.main()
    captured = capsys.readouterr()
    return code, captured.out, captured.err


_SSO_BAR = "=" * 70


def _section(title, content):
    # print() appends its own newline, so a content ending in a newline leaves an extra blank line
    return ["", _SSO_BAR, title, _SSO_BAR, *(content + chr(10)).splitlines()]


class TestSsoArguments:
    @pytest.mark.parametrize("provider", sorted({
        "openai": 0, "google": 0, "anthropic": 0, "deepseek": 0, "qwen": 0, "glm": 0, "xai": 0,
    }))
    def test_r17_sso_default_model_for_every_provider(self, sso, chain, monkeypatch, capsys, provider):
        default = sso.PROVIDER_DEFAULTS[provider][0]

        code, out, _ = _sso_main(sso, monkeypatch, capsys, provider)

        assert code == 0
        assert [(c.provider, c.model) for c in chain.clients] == [(provider, default), (provider, default)]
        assert out.splitlines()[:3] == [f"Provider: {provider}", f"Deep model:  {default}", f"Quick model: {default}"]

    def test_r17_sso_model_overrides(self, sso, chain, monkeypatch, capsys):
        code, out, _ = _sso_main(
            sso, monkeypatch, capsys, "deepseek", "--deep-model", "deepseek-flash", "--quick-model", "custom-id"
        )

        assert code == 0
        assert [(c.provider, c.model) for c in chain.clients] == [
            ("deepseek", "deepseek-flash"), ("deepseek", "custom-id"),
        ]
        assert out.splitlines()[:3] == [
            "Provider: deepseek", "Deep model:  deepseek-flash", "Quick model: custom-id",
        ]

    def test_r17_sso_one_override_leaves_the_other_at_the_default(self, sso, chain, monkeypatch, capsys):
        _sso_main(sso, monkeypatch, capsys, "glm", "--quick-model", "glm-5.3-flash")

        assert [c.model for c in chain.clients] == ["glm-5", "glm-5.3-flash"]

    @pytest.mark.parametrize("argv", [[], ["bogus"], ["OpenAI"], ["deepseek", "extra"]])
    def test_r17_sso_rejects_anything_but_one_known_provider(
        self, sso, chain, monkeypatch, capsys, argv
    ):
        _set_argv(monkeypatch, "smoke_structured_output.py", argv)

        with pytest.raises(SystemExit) as excinfo:
            sso.main()

        assert excinfo.value.code == 2
        assert chain.clients == []
        assert capsys.readouterr().out == ""

    def test_r17_sso_unknown_flag_exits_2(self, sso, chain, monkeypatch, capsys):
        _set_argv(monkeypatch, "smoke_structured_output.py", ["deepseek", "--model", "x"])

        with pytest.raises(SystemExit) as excinfo:
            sso.main()

        assert excinfo.value.code == 2
        assert chain.clients == []


class TestSsoChain:
    def test_r17_sso_passing_chain_prints_each_stage_and_exits_0(self, sso, chain, monkeypatch, capsys):
        code, out, err = _sso_main(sso, monkeypatch, capsys, "deepseek")

        assert (code, err) == (0, "")
        assert chain.agents == {"rm": "<deepseek:deepseek-v4-pro>", "trader": "<deepseek:deepseek-v4-pro>", "pm": "<deepseek:deepseek-v4-pro>"}
        assert out.splitlines() == [
            "Provider: deepseek",
            "Deep model:  deepseek-v4-pro",
            "Quick model: deepseek-v4-pro",
            *_section("[1] Research Manager — investment_plan", chain.rm_plan),
            *_section("[2] Trader — trader_investment_plan", chain.trader_plan),
            *_section("[3] Portfolio Manager — final_trade_decision", chain.pm_decision),
            *_section("[4] SignalProcessor → rating", "Buy"),
            "", _SSO_BAR, "Structure checks", _SSO_BAR,
            "  PASS  Research Manager: contains '**Recommendation**:'",
            "  PASS  Trader: contains '**Action**:'",
            "  PASS  Trader: contains 'FINAL TRANSACTION PROPOSAL:'",
            "  PASS  Portfolio Manager: contains '**Rating**:'",
            "  PASS  Portfolio Manager: contains '**Executive Summary**:'",
            "  PASS  Portfolio Manager: contains '**Investment Thesis**:'",
            "",
            "Smoke PASSED: structured output → rendered markdown chain works for deepseek",
        ]

    def test_r17_sso_each_agent_gets_the_previous_ones_output_and_the_right_llm(
        self, sso, chain, monkeypatch, capsys
    ):
        _sso_main(sso, monkeypatch, capsys, "openai", "--deep-model", "D", "--quick-model", "Q")

        assert chain.agents == {"rm": "<openai:D>", "trader": "<openai:Q>", "pm": "<openai:D>"}
        assert [name for name, _ in chain.order] == ["rm", "trader", "pm"]
        _, rm_state = chain.order[0]
        _, trader_state = chain.order[1]
        _, pm_state = chain.order[2]
        assert rm_state == sso._make_rm_state()
        assert trader_state == sso._make_trader_state(chain.rm_plan)
        assert pm_state == sso._make_pm_state(chain.rm_plan, chain.trader_plan)
        assert trader_state["investment_plan"] == "**Recommendation**: Buy\n\nPlan body."
        assert pm_state["trader_investment_plan"].startswith("**Action**: BUY")

    def test_r17_sso_different_models_route_deep_to_rm_and_pm_quick_to_trader(
        self, sso, chain, monkeypatch, capsys
    ):
        _sso_main(sso, monkeypatch, capsys, "anthropic", "--quick-model", "fast")

        assert chain.agents == {"rm": "<anthropic:claude-sonnet-4-6>", "trader": "<anthropic:fast>", "pm": "<anthropic:claude-sonnet-4-6>"}

    @pytest.mark.parametrize(
        ("field", "text", "missing"),
        [
            pytest.param("trader_plan", "no markers here\n", ["**Action**:", "FINAL TRANSACTION PROPOSAL:"], id="trader"),
            pytest.param(
                "pm_decision", "**Rating**: Sell\n", ["**Executive Summary**:", "**Investment Thesis**:"], id="pm"
            ),
            pytest.param("rm_plan", "Thin.", ["**Recommendation**:"], id="rm"),
        ],
    )
    def test_r17_sso_missing_marker_fails_the_smoke_with_exit_1(
        self, sso, chain, monkeypatch, capsys, field, text, missing
    ):
        setattr(chain, field, text)

        code, out, err = _sso_main(sso, monkeypatch, capsys, "qwen")

        assert (code, err) == (1, "")
        lines = out.splitlines()
        fails = [line for line in lines if line.startswith("  FAIL")]
        passes = [line for line in lines if line.startswith("  PASS")]
        assert fails == [line for marker in missing for line in [f"  FAIL  {{name}}: contains {marker!r}".format(
            name={"trader_plan": "Trader", "pm_decision": "Portfolio Manager", "rm_plan": "Research Manager"}[field]
        )]]
        assert len(fails) == len(missing) and len(passes) == 6 - len(missing)
        assert lines[-2:] == ["", f"Smoke FAILED: {len(missing)} structure check(s) missing."]
        # the stage outputs are still printed in full before the verdict
        assert text.splitlines()[0] in lines

    @pytest.mark.parametrize(
        ("decision", "expected_code", "fails"),
        [
            pytest.param("x **Rating**:Buy **Executive Summary**:y **Investment Thesis**:z", 0, [], id="anywhere-in-the-text"),
            pytest.param(
                "**Rating** : Buy\n**Executive Summary**: y\n**Investment Thesis**: z",
                1, ["**Rating**:"], id="space-before-the-colon-is-a-miss",
            ),
            pytest.param(
                "**Rating**: Buy\n**Executive Summary**: y\n**Investment Thesis** x",
                1, ["**Investment Thesis**:"], id="missing-colon-is-a-miss",
            ),
        ],
    )
    def test_r17_sso_markers_are_plain_substrings(self, sso, chain, monkeypatch, capsys, decision, expected_code, fails):
        chain.pm_decision = decision + "\n"

        code, out, _ = _sso_main(sso, monkeypatch, capsys, "xai")

        assert code == expected_code
        lines = out.splitlines()
        assert [line for line in lines if line.startswith("  FAIL")] == [
            f"  FAIL  Portfolio Manager: contains {marker!r}" for marker in fails
        ]
        assert sum(line.startswith("  PASS") for line in lines) == 6 - len(fails)

    def test_r17_sso_rating_line_is_what_the_real_signal_processor_returns(
        self, sso, chain, monkeypatch, capsys
    ):
        chain.pm_decision = "**Rating**: Overweight\n**Executive Summary**: a\n**Investment Thesis**: b\n"

        code, out, _ = _sso_main(sso, monkeypatch, capsys, "google")

        assert code == 0
        lines = out.splitlines()
        title = lines.index("[4] SignalProcessor → rating")
        assert lines[title - 2 : title + 3] == ["", _SSO_BAR, "[4] SignalProcessor → rating", _SSO_BAR, "Overweight"]

    def test_r17_sso_agent_exception_is_not_caught(self, sso, chain, monkeypatch, capsys):
        def explode(llm):
            raise RuntimeError("provider 500")

        monkeypatch.setattr(sso, "create_trader", explode)

        with pytest.raises(RuntimeError, match="provider 500"):
            _sso_main(sso, monkeypatch, capsys, "deepseek")

        assert len(chain.clients) == 2
        assert "Smoke" not in "\n".join(capsys.readouterr().out.splitlines()[-1:])

    def test_r17_sso_client_factory_exception_is_not_caught(self, sso, chain, monkeypatch, capsys):
        def explode(provider, model):
            raise ValueError(f"no key for {provider}")

        monkeypatch.setattr(sso, "create_llm_client", explode)

        with pytest.raises(ValueError, match="no key for deepseek"):
            _sso_main(sso, monkeypatch, capsys, "deepseek")

        assert chain.order == []
        assert capsys.readouterr().out.splitlines() == [
            "Provider: deepseek", "Deep model:  deepseek-v4-pro", "Quick model: deepseek-v4-pro",
        ]


def test_r17_sso_main_guard_exits_with_main_code(monkeypatch, capsys):
    """``python scripts/smoke_structured_output.py`` exits with ``main()``'s status."""
    path = SCRIPTS_DIR / "smoke_structured_output.py"
    code = compile(path.read_text(encoding="utf-8"), str(path), "exec")

    _set_argv(monkeypatch, "smoke_structured_output.py", ["nope"])
    with pytest.raises(SystemExit) as excinfo:
        exec(code, {"__name__": "__main__", "__file__": str(path)})  # noqa: S102

    assert excinfo.value.code == 2


# =========================================================================== #
# scripts/prune_indicators_cli.py
# =========================================================================== #


@pytest.fixture
def prune():
    return _load("r17_prune_cli", "prune_indicators_cli.py")


_ROWS = 120
_GOOD = [math.sin(i / 3.0) for i in range(_ROWS)]
_WINDOW = ["--window", "20", "--min-consecutive", "10", "--min-observations", "10"]
_PRUNE_ARGS = _WINDOW + ["--min-abs-ic", "0.5"]
_FENCE = "`" * 3


def _noise(seed=7):
    rng = random.Random(seed)
    return [rng.uniform(-1, 1) for _ in range(_ROWS)]


def _write_ic_csv(path, columns):
    """Write an IC CSV from ordered ``(name, values)`` pairs; ``date`` is filled in."""
    data = {}
    if "date" not in dict(columns):
        data["date"] = pd.date_range("2025-01-01", periods=len(columns[0][1]), freq="B")
    data.update(columns)
    pd.DataFrame(data).to_csv(path, index=False)
    return path


def _simple_csv(path):
    return _write_ic_csv(path, [("forward_return", _GOOD), ("good", _GOOD), ("noise", _noise())])


def _assert_close(actual, expected, where="$"):
    """Recursive comparison: floats to 1e-9 relative, everything else exact."""
    if isinstance(expected, dict):
        assert isinstance(actual, dict) and list(actual) == list(expected), where
        for key in expected:
            _assert_close(actual[key], expected[key], f"{where}.{key}")
    elif isinstance(expected, list):
        assert isinstance(actual, list) and len(actual) == len(expected), where
        for i, (a, e) in enumerate(zip(actual, expected, strict=True)):
            _assert_close(a, e, f"{where}[{i}]")
    elif isinstance(expected, float) and not isinstance(expected, bool):
        assert actual == pytest.approx(expected, rel=1e-9, abs=1e-12), where
    else:
        assert actual == expected, where


class TestPruneLoadIcData:
    def test_r17_prune_load_splits_horizon_columns_from_indicators(self, prune, tmp_path):
        horizon = {h: [g * factor for g in _GOOD] for h, factor in (("fwd_ret_5d", 0.9), ("fwd_ret_10d", 0.5))}
        look_alikes = {"FWD_RET_5D": _GOOD, "fwd_ret_xd": _GOOD, "fwd_ret_5dd": _GOOD, "fwd_ret_": _GOOD}
        path = _write_ic_csv(tmp_path / "ic.csv", [
            ("forward_return", _GOOD), ("fwd_ret_5d", horizon["fwd_ret_5d"]), ("alpha", _noise(1)),
            ("fwd_ret_10d", horizon["fwd_ret_10d"]), *look_alikes.items(), ("beta", _noise(2)),
        ])

        forward, indicators, horizon_returns = prune._load_ic_data(str(path))

        assert list(forward) == pytest.approx(_GOOD)
        assert list(indicators) == ["alpha", *look_alikes, "beta"]
        assert list(indicators["alpha"]) == pytest.approx(_noise(1))
        assert list(indicators["beta"]) == pytest.approx(_noise(2))
        for name in look_alikes:
            assert list(indicators[name]) == pytest.approx(_GOOD)
        assert list(horizon_returns) == ["5d", "10d"]
        assert list(horizon_returns["5d"]) == pytest.approx(horizon["fwd_ret_5d"])
        assert list(horizon_returns["10d"]) == pytest.approx(horizon["fwd_ret_10d"])

    def test_r17_prune_load_single_horizon_csv_has_no_horizon_map(self, prune, tmp_path):
        path = _simple_csv(tmp_path / "ic.csv")

        forward, indicators, horizon_returns = prune._load_ic_data(str(path))

        assert list(forward) == pytest.approx(_GOOD)
        assert list(indicators) == ["good", "noise"]
        assert horizon_returns == {}

    @pytest.mark.parametrize(
        ("columns", "message"),
        [
            pytest.param([("date", ["2025-01-01"]), ("x", [1.0])], "CSV must have a 'forward_return' column", id="no-forward"),
            pytest.param([("date", ["2025-01-01"]), ("forward_return", [0.1])], "CSV must have at least one indicator column", id="no-indicator"),
        ],
    )
    def test_r17_prune_load_rejects_incomplete_csvs(self, prune, tmp_path, columns, message):
        path = _write_ic_csv(tmp_path / "ic.csv", columns)

        with pytest.raises(ValueError, match=message):
            prune._load_ic_data(str(path))

    def test_r17_prune_load_without_date_column_is_refused_by_pandas_first(self, prune, tmp_path):
        """Why the script's own "must have a 'date' column" line stays uncovered:
        ``read_csv(parse_dates=["date"])`` raises before the script looks."""
        path = tmp_path / "ic.csv"
        path.write_text("x,forward_return\n1,0.1\n", encoding="utf-8")

        with pytest.raises(ValueError, match="date"):
            prune._load_ic_data(str(path))

    def test_r17_prune_load_unparseable_dates_are_accepted(self, prune, tmp_path):
        """Pins the current seam: the date column is parsed but never used for the math."""
        path = tmp_path / "ic.csv"
        path.write_text("date,forward_return,ind\nnot-a-date,0.1,1\nalso-not,0.2,2\n", encoding="utf-8")

        with pytest.warns(UserWarning, match="Could not infer format"):
            forward, indicators, horizon_returns = prune._load_ic_data(str(path))

        assert list(forward) == pytest.approx([0.1, 0.2])
        assert list(indicators) == ["ind"] and horizon_returns == {}


class TestPruneMainErrors:
    def test_r17_prune_missing_csv_exit_1(self, prune, capsys, tmp_path):
        missing = tmp_path / "nope.csv"

        assert prune.main([str(missing), *_PRUNE_ARGS]) == 1

        captured = capsys.readouterr()
        assert captured.out == ""
        assert captured.err == f"Error: {missing} not found\n"

    @pytest.mark.parametrize(
        ("body", "fragment"),
        [
            pytest.param("date,x\n2025-01-01,1\n", "CSV must have a 'forward_return' column", id="no-forward"),
            pytest.param("date,forward_return\n2025-01-01,0.1\n", "CSV must have at least one indicator column", id="no-indicator"),
            pytest.param("", "", id="empty-file"),
            pytest.param("\x00\x01not a csv at all \xff", "", id="garbage"),
            pytest.param("x,forward_return\n1,0.1\n", "date", id="no-date-column"),
        ],
    )
    def test_r17_prune_unloadable_csv_exit_1_with_the_reason(self, prune, capsys, tmp_path, body, fragment):
        path = tmp_path / "ic.csv"
        path.write_text(body, encoding="utf-8")

        assert prune.main([str(path), *_PRUNE_ARGS]) == 1

        captured = capsys.readouterr()
        assert captured.out == ""
        assert captured.err.startswith("Error loading CSV: ") and fragment in captured.err
        assert captured.err.count("\n") == 1

    @pytest.mark.parametrize(
        ("flag", "value", "message"),
        [
            pytest.param("--window", "0", "Error: --window must be >= 1 (got 0)", id="window-zero"),
            pytest.param("--window", "-5", "Error: --window must be >= 1 (got -5)", id="window-negative"),
            pytest.param("--min-consecutive", "0", "Error: --min-consecutive must be >= 1 (got 0)", id="consecutive-zero"),
            pytest.param("--min-observations", "0", "Error: --min-observations must be >= 1 (got 0)", id="observations-zero"),
            pytest.param("--min-observations", "-1", "Error: --min-observations must be >= 1 (got -1)", id="observations-negative"),
        ],
    )
    def test_r17_prune_degenerate_arguments_exit_1(self, prune, capsys, tmp_path, flag, value, message):
        path = _simple_csv(tmp_path / "ic.csv")

        assert prune.main([str(path), *_PRUNE_ARGS, flag, value]) == 1

        captured = capsys.readouterr()
        assert (captured.out, captured.err) == ("", message + "\n")

    def test_r17_prune_check_order_file_then_window_then_consecutive_then_observations(
        self, prune, capsys, tmp_path
    ):
        path = _simple_csv(tmp_path / "ic.csv")
        all_bad = [str(path), "--window", "0", "--min-consecutive", "0", "--min-observations", "0"]

        assert prune.main([str(tmp_path / "nope.csv"), *all_bad[1:]]) == 1  # file first
        assert prune.main(all_bad) == 1
        assert prune.main([all_bad[0], *all_bad[3:]]) == 1  # no --window
        assert prune.main([all_bad[0], *all_bad[5:]]) == 1  # only --min-observations

        assert capsys.readouterr().err.splitlines() == [
            f"Error: {tmp_path / 'nope.csv'} not found",
            "Error: --window must be >= 1 (got 0)",
            "Error: --min-consecutive must be >= 1 (got 0)",
            "Error: --min-observations must be >= 1 (got 0)",
        ]

    def test_r17_prune_header_only_csv_refuses_a_verdict(self, prune, capsys, tmp_path):
        path = tmp_path / "ic.csv"
        path.write_text("date,forward_return,ind\n", encoding="utf-8")

        assert prune.main([str(path), *_PRUNE_ARGS]) == 1

        captured = capsys.readouterr()
        assert captured.out == ""
        assert captured.err == (
            "Error: CSV carries no data rows (header only) — refusing to "
            "emit a keep/prune verdict built on zero evidence\n"
        )

    @pytest.mark.parametrize("blank", ["", "nan", "NaN", "N/A"])
    def test_r17_prune_all_nan_forward_returns_refuse_a_verdict(self, prune, capsys, tmp_path, blank):
        path = tmp_path / "ic.csv"
        path.write_text(
            "date,forward_return,ind\n" + "".join(f"2025-01-0{d},{blank},{d}\n" for d in (1, 2, 3)),
            encoding="utf-8",
        )

        assert prune.main([str(path), *_PRUNE_ARGS]) == 1

        captured = capsys.readouterr()
        assert captured.out == ""
        assert captured.err == (
            "Error: CSV has 3 rows but forward_return has 0 non-NaN values — forward-return "
            "construction is broken; refusing to emit a keep/prune verdict built on zero evidence\n"
        )

    def test_r17_prune_output_into_a_missing_directory_crashes_after_the_report(
        self, prune, capsys, tmp_path
    ):
        """Pins the current seam: the report is already on stdout, then the write fails
        with a bare FileNotFoundError (interpreter exit status 1)."""
        path = _simple_csv(tmp_path / "ic.csv")

        with pytest.raises(FileNotFoundError):
            prune.main([str(path), *_PRUNE_ARGS, "--output", str(tmp_path / "nope" / "r.md")])

        assert "# IC Indicator Pruning Report" in capsys.readouterr().out

    def test_r17_prune_bad_dates_do_not_stop_the_analysis(self, prune, capsys, tmp_path):
        path = tmp_path / "ic.csv"
        path.write_text("date,forward_return,ind\nnot-a-date,0.1,1\nalso-not,0.2,2\n", encoding="utf-8")

        with pytest.warns(UserWarning, match="Could not infer format"):
            assert prune.main([str(path), *_PRUNE_ARGS]) == 0

        assert "IC Indicator Pruning Report" in capsys.readouterr().out


_SIMPLE_REPORT = "\n".join([
    "# IC Indicator Pruning Report",
    "",
    "- **Kept**: 1 indicator(s) (good)",
    "- **Pruned**: 1 indicator(s) (noise)",
    "",
    "## Kept indicators",
    "",
    "| Indicator | Mean |IC| | Note |",
    "|---|---|---|",
    "| good | 1.000 | survived |",
    "",
    "## Pruned indicators",
    "- **noise** (mean |IC| 0.220)",
    "",
    "## Quantile spread & turnover (primary horizon)",
    "",
    "| Indicator | Q1 | Q2 | Q3 | Q4 | Q5 | Q_hi-Q_lo | Monotonic | Turnover |",
    # 9 header cells, 10 delimiter cells (``---:|`` is repeated once too often): a display
    # defect, pinned as is -- GFM drops the excess, so Monotonic renders right- and Turnover
    # left-aligned.
    "|---|---:|---:|---:|---:|---:|---:|---:|---|---:|",
    "| good | -0.9273 | -0.5302 | 0.0748 | 0.6273 | 0.9463 | 1.8736 | yes | 0.101 |",
    "| noise | 0.0577 | 0.1998 | -0.1080 | -0.0304 | 0.0718 | 0.0141 | no | 0.329 |",
    "",
    "Monotone quantile means make a factor usable as a ranked long-short signal; a spread "
    "concentrated only in the tail quantiles means tail-only usefulness. Turnover near 1.0 = "
    "the ranking reshuffles almost every row (expensive to capture).",
    "",
    "---",
    "",
    "_Window=20, min|IC|=0.5, min_consecutive=10, min_observations=10_",
]) + chr(10)  # the ``report`` string: what --output writes, before print() adds a newline
_FOOTER = "_Window=20, min|IC|=0.5, min_consecutive=10, min_observations=10_"


def _suggestion(kept):
    return "\n".join([
        "",
        "## Config suggestion (REVIEW BEFORE APPLYING)",
        "",
        "The following indicators are candidates for removal from the",
        "market analyst's catalog. After review, set `indicator_battery`",
        "(in default_config.py or via set_config) to the kept names:",
        "",
        _FENCE + "python",
        "# default_config.py",
        f'"indicator_battery": {kept},',
        _FENCE,
        "",
        f"{_WARN} This is a suggestion only. Verify against out-of-sample data and apply manually "
        "— never auto-edit the live config.",
    ]) + "\n"


_GOOD_JSON = {
    "verdict": "keep",
    "mean_abs_ic": 1.0,
    "finite_windows": 101,
    "longest_low_run": 0,
    "quantile_means": [-0.9272912109198178, -0.5302014884648116, 0.0748058942685071, 0.6272941159536775, 0.9463038472603822],
    "quantile_spread": 1.8735950581802001,
    "monotonic": True,
    "turnover": 0.10126403502577501,
}
_NOISE_JSON = {
    "verdict": "prune",
    "mean_abs_ic": 0.21966798183577754,
    "finite_windows": 101,
    "longest_low_run": 73,
    "quantile_means": [0.05769307609404432, 0.19978701100347676, -0.10795732398699605, -0.030385322526082694, 0.07177371751349489],
    "quantile_spread": 0.01408064141945057,
    "monotonic": False,
    "turnover": 0.32893157262905165,
}


class TestPruneMainReport:
    def test_r17_prune_report_file_json_verdict_and_suggestion(self, prune, capsys, tmp_path):
        path = _simple_csv(tmp_path / "ic.csv")
        report_file, json_file = tmp_path / "r.md", tmp_path / "v.json"

        code = prune.main([
            str(path), *_PRUNE_ARGS, "-o", str(report_file), "--json-out", str(json_file), "--suggest-config",
        ])

        assert code == 0
        captured = capsys.readouterr()
        assert captured.out == _SIMPLE_REPORT + chr(10) + _suggestion("['good']")
        assert report_file.read_text(encoding="utf-8") == _SIMPLE_REPORT  # without the suggestion
        assert captured.err == (
            f"\nReport written to {report_file}\n\nJSON verdict written to {json_file}\n"
        )
        raw = json_file.read_text(encoding="utf-8")
        verdict = json.loads(raw)
        assert list(verdict) == ["generated_at", "params", "keep", "prune", "per_indicator"]
        assert raw == json.dumps(verdict, indent=2)
        stamp = datetime.fromisoformat(verdict["generated_at"])
        assert stamp.utcoffset() == timedelta(0)
        del verdict["generated_at"]
        _assert_close(verdict, {
            "params": {"window": 20, "min_abs_ic": 0.5, "min_consecutive": 10, "min_observations": 10},
            "keep": ["good"],
            "prune": ["noise"],
            "per_indicator": {"good": _GOOD_JSON, "noise": _NOISE_JSON},
        })

    def test_r17_prune_decay_and_quantile_tables_with_a_coarse_indicator(self, prune, capsys, tmp_path):
        path = _write_ic_csv(tmp_path / "ic.csv", [
            ("forward_return", _GOOD), ("good", _GOOD), ("noise", _noise()),
            ("coarse", [i % 3 for i in range(_ROWS)]),
            ("fine", [((i * 37) % 101) / 101 for i in range(_ROWS)]),
            ("fwd_ret_5d", [g * 0.9 for g in _GOOD]),
            ("fwd_ret_10d", [g * 0.5 for g in _GOOD]),
        ])
        json_file = tmp_path / "v.json"

        code = prune.main([str(path), *_PRUNE_ARGS, "--json-out", str(json_file)])

        assert code == 0
        captured = capsys.readouterr()
        lines = captured.out.splitlines()
        assert lines[:13] == [
            "# IC Indicator Pruning Report",
            "",
            "- **Kept**: 1 indicator(s) (good)",
            "- **Pruned**: 3 indicator(s) (coarse, fine, noise)",
            "",
            "## Kept indicators",
            "",
            "| Indicator | Mean |IC| | Note |",
            "|---|---|---|",
            "| good | 1.000 | survived |",
            "",
            "## Pruned indicators",
            "- **coarse** (mean |IC| 0.046)",
        ]
        assert lines[13:16] == ["- **fine** (mean |IC| 0.106)", "- **noise** (mean |IC| 0.220)", ""]
        decay_at = lines.index("## IC decay (full-sample IC by forward horizon)")
        assert lines[decay_at:decay_at + 10] == [
            "## IC decay (full-sample IC by forward horizon)",
            "",
            "| Indicator | 5d | 10d |",
            "|---|---:|---:|",
            "| good | 1.000 | 1.000 |",
            "| noise | -0.033 | -0.033 |",
            "| coarse | 0.010 | 0.010 |",
            "| fine | -0.003 | -0.003 |",
            "",
            "IC collapsing toward 0 as the horizon lengthens marks a short-lived signal; "
            "a flat curve supports slower rebalancing.",
        ]
        quant_at = lines.index("| Indicator | Q1 | Q2 | Q3 | Q4 | Q5 | Q_hi-Q_lo | Monotonic | Turnover |")
        assert lines[quant_at - 3:quant_at] == [
            "", "## Quantile spread & turnover (primary horizon)", "",
        ]
        assert lines[quant_at + 2] == "| good | -0.9273 | -0.5302 | 0.0748 | 0.6273 | 0.9463 | 1.8736 | yes | 0.101 |"
        assert lines[quant_at + 3] == "| noise | 0.0577 | 0.1998 | -0.1080 | -0.0304 | 0.0718 | 0.0141 | no | 0.329 |"
        # Ties collapse the coarse indicator to few buckets; the missing quantiles are dashes.
        cells = [cell.strip() for cell in lines[quant_at + 4].strip("|").split("|")]
        assert cells[:3] == ["coarse", "0.0342", "0.0461"]
        assert cells[-3:] == ["0.0119", "yes", "0.446"]
        assert len(cells) == 9 and "—" in cells[3:-3]
        assert lines[quant_at + 5] == "| fine | 0.0419 | 0.0561 | 0.0378 | -0.0016 | 0.0568 | 0.0149 | no | 0.467 |"
        assert lines[-4:] == ["---", "", _FOOTER, ""]
        assert "Config suggestion" not in captured.out
        assert captured.err == f"\nJSON verdict written to {json_file}\n"
        verdict = json.loads(json_file.read_text(encoding="utf-8"))
        assert (verdict["keep"], verdict["prune"]) == (["good"], ["coarse", "fine", "noise"])
        assert list(verdict["per_indicator"]) == ["good", "noise", "coarse", "fine"]
        coarse = verdict["per_indicator"]["coarse"]
        numeric = [cell for cell in cells[1:-3] if cell != "—"]
        assert len(coarse["quantile_means"]) == len(numeric) < 5
        assert [f"{m:.4f}" for m in coarse["quantile_means"]] == numeric
        _assert_close(coarse["ic_by_horizon"], {"5d": 0.009723055853282467, "10d": 0.009723055853282467})
        assert (coarse["longest_low_run"], coarse["finite_windows"], coarse["monotonic"]) == (101, 101, True)
        _assert_close(verdict["per_indicator"]["good"]["ic_by_horizon"], {"5d": 1.0, "10d": 1.0})
        _assert_close(verdict["per_indicator"]["noise"]["ic_by_horizon"], {"5d": -0.03321063962775192, "10d": -0.03321063962775192})

    def test_r17_prune_suggestion_lists_kept_names_in_csv_order(self, prune, capsys, tmp_path):
        path = _write_ic_csv(tmp_path / "ic.csv", [
            ("forward_return", _GOOD), ("zeta", _GOOD), ("noise", _noise()), ("alpha", _GOOD),
        ])

        code = prune.main([str(path), *_PRUNE_ARGS, "--suggest-config"])

        assert code == 0
        out = capsys.readouterr().out
        assert "- **Kept**: 2 indicator(s) (alpha, zeta)" in out  # the report itself is sorted
        assert out.endswith(_FOOTER + chr(10) * 2 + _suggestion("['zeta', 'alpha']"))

    def test_r17_prune_nothing_pruned_prints_no_suggestion(self, prune, capsys, tmp_path):
        path = _simple_csv(tmp_path / "ic.csv")
        json_file = tmp_path / "v.json"

        code = prune.main([
            str(path), "--window", "20", "--min-consecutive", "10", "--min-observations", "500",
            "--suggest-config", "--json-out", str(json_file),
        ])

        assert code == 0
        out = capsys.readouterr().out
        assert "- **Kept**: 2 indicator(s) (good, noise)" in out
        assert "- **Pruned**: 0 indicator(s)" in out
        assert "Config suggestion" not in out
        assert out.endswith(
            "_Window=20, min|IC|=0.03, min_consecutive=10, min_observations=500_" + chr(10) * 2
        )
        verdict = json.loads(json_file.read_text(encoding="utf-8"))
        assert (verdict["keep"], verdict["prune"]) == (["good", "noise"], [])
        assert verdict["params"] == {"window": 20, "min_abs_ic": 0.03, "min_consecutive": 10, "min_observations": 500}
        noise = verdict["per_indicator"]["noise"]
        # 101 finite windows are fewer than the 500 required before any run may prune
        assert noise["verdict"] == "keep" and noise["mean_abs_ic"] == pytest.approx(0.21966798183577754)
        assert noise["finite_windows"] == 101

    def test_r17_prune_default_parameters_in_footer_and_json(self, prune, capsys, tmp_path):
        path = _simple_csv(tmp_path / "ic.csv")

        assert prune.main([str(path), "--json-out", str(tmp_path / "v.json")]) == 0

        out = capsys.readouterr().out
        assert "_Window=60, min|IC|=0.03, min_consecutive=30, min_observations=30_" in out
        assert "## IC decay" not in out  # single-horizon CSV keeps the original shape
        verdict = json.loads((tmp_path / "v.json").read_text(encoding="utf-8"))
        assert verdict["params"] == {"window": 60, "min_abs_ic": 0.03, "min_consecutive": 30, "min_observations": 30}
        good = verdict["per_indicator"]["good"]
        assert (good["mean_abs_ic"], good["finite_windows"], good["longest_low_run"]) == (1.0, 61, 0)

    @pytest.mark.parametrize(
        ("value", "kept", "pruned"),
        [
            pytest.param("0", 2, 0, id="zero-threshold-keeps-all"),
            pytest.param("-1", 2, 0, id="negative-keeps-all"),
            pytest.param("nan", 2, 0, id="nan-keeps-all"),
            pytest.param("inf", 0, 2, id="inf-prunes-all"),
        ],
    )
    def test_r17_prune_degenerate_threshold_still_emits_a_verdict(
        self, prune, capsys, tmp_path, value, kept, pruned
    ):
        """Pins the current seam (reported, not changed): ``--min-abs-ic`` has no guard.
        NaN fails every ``abs(IC) < threshold`` comparison and 0 / negatives never
        trip, so everything is KEPT; +inf prunes everything -- all with exit 0."""
        path = _simple_csv(tmp_path / "ic.csv")

        assert prune.main([str(path), *_WINDOW, "--min-abs-ic", value]) == 0

        out = capsys.readouterr().out
        assert f"- **Kept**: {kept} indicator(s)" in out
        assert f"- **Pruned**: {pruned} indicator(s)" in out


class TestPruneConsole:
    @staticmethod
    def _stream(encoding):
        calls = []
        stream = SimpleNamespace(encoding=encoding, reconfigure=lambda **kwargs: calls.append(kwargs))
        return stream, calls

    @pytest.mark.parametrize("encoding", ["utf-8", "UTF-8", "utf8", "UTF8"])
    def test_r17_prune_console_already_utf8_is_left_alone(self, prune, monkeypatch, encoding):
        out, out_calls = self._stream(encoding)
        err, err_calls = self._stream(encoding)
        monkeypatch.setattr(sys, "stdout", out)
        monkeypatch.setattr(sys, "stderr", err)

        prune._ensure_unicode_stdout()

        assert (out_calls, err_calls) == ([], [])

    @pytest.mark.parametrize("encoding", ["cp1252", "gbk", "ascii", "utf-16", None])
    def test_r17_prune_console_other_encodings_are_switched_to_utf8(self, prune, monkeypatch, encoding):
        out, out_calls = self._stream(encoding)
        err, err_calls = self._stream(encoding)
        monkeypatch.setattr(sys, "stdout", out)
        monkeypatch.setattr(sys, "stderr", err)

        prune._ensure_unicode_stdout()

        assert out_calls == [{"encoding": "utf-8", "errors": "replace"}]
        assert err_calls == [{"encoding": "utf-8", "errors": "replace"}]

    def test_r17_prune_console_stream_without_reconfigure_crashes(self, prune, monkeypatch):
        """Pins the current seam: unlike ``_bootstrap.bootstrap()`` there is no
        ``getattr`` guard here, so a replaced stdout that lacks ``reconfigure``
        (``redirect_stdout(StringIO())``) makes ``main()`` fail before it starts."""
        monkeypatch.setattr(sys, "stdout", io.StringIO())

        with pytest.raises(AttributeError, match="reconfigure"):
            prune._ensure_unicode_stdout()


def test_r17_prune_main_guard_exits_with_main_code(monkeypatch, capsys, tmp_path):
    """``python scripts/prune_indicators_cli.py`` exits with ``main()``'s status."""
    path = SCRIPTS_DIR / "prune_indicators_cli.py"
    code = compile(path.read_text(encoding="utf-8"), str(path), "exec")
    monkeypatch.setattr(sys, "path", list(sys.path))  # the script's shim edits sys.path
    missing = tmp_path / "nope.csv"

    _set_argv(monkeypatch, "prune_indicators_cli.py", [str(missing)])
    with pytest.raises(SystemExit) as excinfo:
        exec(code, {"__name__": "__main__", "__file__": str(path)})  # noqa: S102

    assert excinfo.value.code == 1
    assert capsys.readouterr().err == f"Error: {missing} not found\n"


# =========================================================================== #
# scripts/rank_signals.py
# =========================================================================== #


@pytest.fixture
def rs(monkeypatch, tmp_path):
    """A fresh ``rank_signals.py`` whose reports root is ``tmp_path/reports``."""
    monkeypatch.setenv("YIALPHA_RESULTS_DIR", str(tmp_path))
    return _load("r17_rank_signals", "rank_signals.py")


_GREEN_DOT = "\U0001f7e2"
_RED_DOT = "\U0001f534"
_WHITE_DOT = "\u26aa"
_LONG_BADGE = f"{_GREEN_DOT} 做多"
_SHORT_BADGE = f"{_RED_DOT} 做空"


def _report(root, ticker, stamp="20260918_120000", *, rating=None, pm_target=None, ovl_stop=None,
            ovl_regime=None, ovl_action=None, action=None, entry=None, stop=None, sentiment=None,
            complete=True, mtime=None, sentiment_raw=None):
    """One report tree with the files ``analyze_ticker`` reads; raw strings for every number."""
    d = root / f"{ticker}_{stamp}"
    decision = ["# Portfolio Decision"]
    if rating:
        decision.append(f"**Rating**: {rating}")
    if pm_target is not None:
        decision.append(f"**Price Target**: {pm_target}")
    if ovl_stop is not None:
        decision.append(f"- **Stop Loss**: {ovl_stop}")
    if ovl_regime:
        decision.append(f"- **Drawdown Regime**: {ovl_regime}")
    if ovl_action:
        decision.append(f"- **Action**: {ovl_action}")
    trader = []
    if action:
        trader.append(f"**Action**: {action}")
    if entry is not None:
        trader.append(f"**Entry Price**: {entry}")
    if stop is not None:
        trader.append(f"**Stop Loss**: {stop}")
    texts = {
        "5_portfolio/decision.md": "\n".join(decision) + "\n",
        "3_trading/trader.md": "\n".join(trader) + "\n",
        "1_analysts/market.md": "",
        "1_analysts/sentiment.md": sentiment_raw if sentiment_raw is not None else (
            f"**Overall Sentiment:** **Bullish** (Score: {sentiment}/10)\n" if sentiment is not None else ""
        ),
    }
    for rel, text in texts.items():
        path = d / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    if complete:
        cr = d / "complete_report.md"
        cr.write_text("done\n", encoding="utf-8")
        if mtime is not None:
            os.utime(cr, (mtime, mtime))
    return d


def _rank_main(rs, monkeypatch, capsys, *argv):
    _set_argv(monkeypatch, "rank_signals.py", list(argv))
    rs.main()
    captured = capsys.readouterr()
    return captured.out, captured.err


class TestRankLatestForTicker:
    def test_r17_rank_latest_skips_a_report_that_cannot_be_stat_ed(self, rs, monkeypatch, tmp_path):
        """The window between ``is_file()`` and ``stat()``: unreadable newest is skipped."""
        root = tmp_path / "reports"
        _report(root, "AAPL", "20260918_120000", mtime=100.0)
        _report(root, "AAPL", "20260919_120000", mtime=200.0)
        real_stat, calls = Path.stat, []

        def flaky_stat(self, *args, **kwargs):
            if self == root / "AAPL_20260919_120000" / "complete_report.md":
                calls.append(1)
                if len(calls) == 2:  # is_file() got through; the explicit stat() now fails
                    raise PermissionError("being replaced")
            return real_stat(self, *args, **kwargs)

        monkeypatch.setattr(Path, "stat", flaky_stat)

        assert rs.latest_for_ticker("AAPL") == root / "AAPL_20260918_120000"
        assert len(calls) == 2

    @pytest.mark.parametrize(
        ("shape", "ticker"),
        [
            pytest.param("no-root", "AAPL", id="no-root"),
            pytest.param("incomplete-only", "AAPL", id="incomplete-only"),
            pytest.param("unknown-ticker", "MSFT", id="unknown-ticker"),
            pytest.param("bare-prefix", "AAP", id="AAP-underscore-does-not-reach-AAPL"),
        ],
    )
    def test_r17_rank_latest_without_a_complete_report_is_none(self, rs, tmp_path, shape, ticker):
        root = tmp_path / "reports"
        if shape == "incomplete-only":
            _report(root, "AAPL", complete=False)
        elif shape in ("unknown-ticker", "bare-prefix"):
            _report(root, "AAPL")

        assert rs.latest_for_ticker(ticker) is None

    def test_r17_rank_latest_counts_an_unstamped_sibling(self, rs, tmp_path):
        """Pins the current seam: the lookup is ``glob("<ticker>_*")`` with no stamp-shape
        check, so ``AAA_notes`` (or a backed-up copy) can be ranked as AAA's report; its
        date then renders as a dash and ``--since`` excludes it."""
        root = tmp_path / "reports"
        _report(root, "AAA", "20260918_120000", mtime=100.0, rating="Buy")
        (root / "AAA_notes").mkdir()
        (root / "AAA_notes" / "complete_report.md").write_text("copy\n", encoding="utf-8")
        os.utime(root / "AAA_notes" / "complete_report.md", (300.0, 300.0))

        assert rs.latest_for_ticker("AAA") == root / "AAA_notes"
        row = rs.analyze_ticker("AAA")
        assert row["report_dir"] == str(root / "AAA_notes") and row["report_date"] is None

    def test_r17_rank_report_date_shows_a_malformed_stamp_as_is(self, rs, tmp_path):
        """Pins the current seam: the stamp is sliced, never validated."""
        root = tmp_path / "reports"
        _report(root, "BAD", "20261340_120000", rating="Buy")

        assert rs.analyze_ticker("BAD")["report_date"] == "13-40"


class TestRankParseSentiment:
    @pytest.mark.parametrize(
        "md",
        [
            pytest.param("**Overall Sentiment:** **Bullish** (Score: ./10)", id="english-dot-only"),
            pytest.param("**Overall Sentiment:** **Bullish** (Score: 1.2.3/10)", id="english-two-dots"),
            pytest.param("**Overall Sentiment:** **Bullish** (Score: .../10)", id="english-ellipsis"),
            pytest.param("整体情绪：看涨（评分 ../10）", id="chinese-dot-only"),
            pytest.param("整体情绪 看跌 评分 3.4.5/10", id="chinese-two-dots"),
            pytest.param("整体情绪：中性（评分：…/10）", id="chinese-ellipsis-not-matched"),
            pytest.param("**Overall Sentiment:** **Bullish** (Score: -1/10)", id="negative-has-no-pattern"),
        ],
    )
    def test_r17_rank_sentiment_unparseable_score_is_none(self, rs, md):
        assert rs.parse_sentiment_score(md) is None

    @pytest.mark.parametrize(
        ("md", "expected"),
        [
            pytest.param("**Overall Sentiment:** **Bearish** (Score: 0/10)", 0.0, id="zero"),
            pytest.param("整体情绪：中性（评分：7.25 / 10）", 7.25, id="chinese-full-width-colon"),
            pytest.param("**overall sentiment:** **Neutral**  (score: 6/10)", 6.0, id="case-insensitive"),
            pytest.param("**Overall Sentiment:** **Bullish** (Score: 12.5/10)", 12.5, id="above-range-accepted"),
            pytest.param("**Overall Sentiment:** **Bearish** (Score: 8/10) ... (Score: 2/10)", 8.0, id="first-wins"),
        ],
    )
    def test_r17_rank_sentiment_score_edges(self, rs, md, expected):
        assert rs.parse_sentiment_score(md) == expected

    def test_r17_rank_out_of_range_score_is_clamped_in_the_conviction(self, rs):
        score, comps = rs.conviction("long", "Buy", 12.5, None)

        assert comps == {"rating": 0.90, "sent": 1.0, "rr": None}
        assert score == 93
        assert rs.conviction("short", "Sell", -4.0, None)[1]["sent"] == 1.0


class TestRankM:
    @pytest.mark.parametrize(
        ("value", "text"),
        [
            pytest.param(None, "—", id="none"),
            pytest.param(999.99, "999.99", id="just-below-thousands"),
            pytest.param(1000, "1,000", id="thousands-boundary"),
            pytest.param(1000.4, "1,000", id="thousands-round"),
            pytest.param(65000.0, "65,000", id="thousands"),
            pytest.param(1234567.891, "1,234,568", id="millions"),
            pytest.param(-5000.4, "-5,000", id="negative-thousands"),
            pytest.param(999.999, "1,000", id="rounds-up-into-thousands"),
            pytest.param(100.0, "100", id="trailing-zeros-stripped-after-the-dot-only"),
            pytest.param(92.5, "92.5", id="one-decimal"),
            pytest.param(1.005, "1", id="bankers-rounding-to-integer"),
            pytest.param(20.0, "20", id="tens"),
            pytest.param(0.30000000000000004, "0.3", id="float-noise"),
            pytest.param(0.0009, "0.0009", id="tiny"),
            pytest.param(0.0000114, "1.14e-05", id="micro-price"),
            pytest.param(0.0005, "0.0005", id="half-a-milli"),
            pytest.param(-0.00025, "-0.00025", id="negative-tiny"),
            pytest.param(0.0, "0", id="zero"),
            pytest.param(-0.0, "-0", id="negative-zero"),
            pytest.param(float("nan"), "nan", id="nan"),
            pytest.param(float("inf"), "inf", id="inf"),
        ],
    )
    def test_r17_rank_m_number_formatting(self, rs, value, text):
        assert rs._m(value) == text


_RANK_NOTE = (
    "> 注：「资产」列按 ticker 启发式判定（USDT 后缀 → 永续）；实际执行以 trade_ticket 出单时的 "
    "venue 判别（含 overlay 证据修正）为准。信心分为 — 表示评级缺失、仅 Trader action 兜底方向，"
    "无综合信心分。"
)
_RANK_PREAMBLE = [
    "# 多标的信心排名 · {ok} 个 ticker（展示每类前 {top}）",
    "> 综合信心分 = 评级 55% + 情绪 25% + 盈亏比 20%（缺失分量重归一化）。信心分是**代理分，非校准概率**。"
    f"评级中/英文均识别。{_WARN} = 回撤熔断（框架已拒部署，不进 #1 推荐）。",
    "",
]
_RANK_TABLE_HEAD = [
    "| # | Ticker | 评级 | 信心分 | 情绪/10 | 盈亏比 | 入场 | 止损 | 报告日期 | 资产 | |",
    "|---|---|---|---:|---:|---:|---|---|---|---|---|",
]
_RANK_DISCLAIMER = [
    "---",
    "*免责声明：信心分为代理分（评级+情绪+盈亏比加权），基于 LLM 分析结论与历史公式，非校准概率，"
    "不构成投资建议。杠杆交易有爆仓归零风险。*",
]


def _classic_reports(root):
    _report(root, "AAA", rating="Buy", pm_target="120.00", entry="100.00", stop="92.00", sentiment=8.2)
    _report(root, "DWSH", rating="Sell", entry="30.00", stop="32.00", sentiment=1.0)
    _report(root, "CCC", rating="Hold")


def _ranked_markdown(top=10, with_tickets=False):
    ok = 3  # AAA, DWSH, CCC have reports; BBB does not
    lines = [
        _RANK_PREAMBLE[0].format(ok=ok, top=top),
        _RANK_PREAMBLE[1],
        "",
        f"## {_GREEN_DOT} 做多排名（1）",
        *_RANK_TABLE_HEAD,
        "| 1 | `AAA` | Buy | **80** | 8.2 | 2.5 | 100 | 92 | 09-18 | 美股 | ← **最强多头** |",
        "",
        f"## {_RED_DOT} 做空排名（1）",
        *_RANK_TABLE_HEAD,
        "| 1 | `DWSH` | Sell | **90** | 1.0 | — | 30 | 32 | 09-18 | 美股 | ← **最强空头** |",
        "",
        _RANK_NOTE,
        "",
        f"## {_WHITE_DOT} 观望（1，不建议进场）",
        "`CCC`",
        "",
        f"## {_WARN} 无报告",
        "没找到最新完成报告（可能没分析过，或目录无 complete_report.md）：`BBB`",
        "",
        "## 信心分明细（最强多头 / 最强空头）",
        "| Ticker | 方向 | 评级分 | 情绪分 | 盈亏比分 | 加权后 |",
        "|---|---|---:|---:|---:|---:|",
        f"| `AAA` | {_LONG_BADGE} | 0.90 | 0.82 | 0.50 | **80** |",
        f"| `DWSH` | {_SHORT_BADGE} | 0.90 | 0.90 | — | **90** |",
        "",
        "## 下一步（选一个出执行单）",
        "- 最强多头 `AAA` → `python scripts/trade_ticket.py --ticker AAA --capital <你的资金>`",
        "- 最强空头 `DWSH` → `python scripts/trade_ticket.py --ticker DWSH --capital <你的资金>`",
        "",
        *_RANK_DISCLAIMER,
    ]
    if with_tickets:
        for ticker in ("AAA", "DWSH"):
            lines += ["", "---", f"# 附：`{ticker}` 执行单（capital=10000.0, moderate 档）", f"TICKET({ticker})"]
    return "\n".join(lines) + "\n"


@pytest.fixture
def ticket_stub(rs, monkeypatch):
    calls = []

    def render(report_dir, capital):
        calls.append((report_dir, capital))
        return f"TICKET({report_dir.name.split('_')[0]})"

    monkeypatch.setattr(rs, "_render_ticket_md", render)
    return calls


class TestRankMain:
    def test_r17_rank_main_markdown_with_missing_and_hold(self, rs, monkeypatch, capsys, tmp_path):
        _classic_reports(tmp_path / "reports")

        out, err = _rank_main(rs, monkeypatch, capsys, "--tickers", "aaa", "dwsh", "BBB", "ccc")

        assert err == ""
        assert out == _ranked_markdown()

    def test_r17_rank_main_capital_appends_a_ticket_per_top_pick(self, rs, ticket_stub, monkeypatch, capsys, tmp_path):
        root = tmp_path / "reports"
        _classic_reports(root)

        out, err = _rank_main(rs, monkeypatch, capsys, "--tickers", "AAA", "DWSH", "BBB", "CCC", "--capital", "10000")

        assert (out, err) == (_ranked_markdown(with_tickets=True), "")
        assert ticket_stub == [(root / "AAA_20260918_120000", 10000.0), (root / "DWSH_20260918_120000", 10000.0)]

    def test_r17_rank_main_top_limits_each_table_but_not_the_counts(self, rs, monkeypatch, capsys, tmp_path):
        root = tmp_path / "reports"
        _classic_reports(root)
        _report(root, "ZZZ", "20260917_120000", rating="Overweight", entry="50.00", stop="47.00",
                pm_target="60.00", sentiment=6)

        out, _ = _rank_main(rs, monkeypatch, capsys, "--tickers", "AAA", "ZZZ", "DWSH", "--top", "1")

        lines = out.splitlines()
        assert lines[0] == "# 多标的信心排名 · 3 个 ticker（展示每类前 1）"
        assert lines[3] == f"## {_GREEN_DOT} 做多排名（2）"  # the group is still two strong
        rows = [line for line in lines if line.startswith("| 1 | `") or line.startswith("| 2 | `")]
        assert [row.split("`")[1] for row in rows] == ["AAA", "DWSH"]  # ZZZ (68) cut by --top 1
        assert "`ZZZ` | Overweight | **68**" not in out

    def test_r17_rank_main_json_rows_for_missing_reports(self, rs, monkeypatch, capsys, tmp_path):
        _classic_reports(tmp_path / "reports")

        out, err = _rank_main(rs, monkeypatch, capsys, "--tickers", "AAA", "BBB", "--json")

        assert err == ""
        rows = json.loads(out)
        assert [r["ticker"] for r in rows] == ["AAA", "BBB"]
        assert rows[1] == {"ticker": "BBB", "status": "no_report"}
        assert (rows[0]["status"], rows[0]["direction"], rows[0]["conviction"], rows[0]["rr"]) == ("ok", "long", 80, 2.5)
        assert out == json.dumps(rows, ensure_ascii=False, indent=2, default=str) + "\n"

    def test_r17_rank_main_since_rejects_an_impossible_stamp_date(self, rs, monkeypatch, capsys, tmp_path):
        """8 digits that are not a calendar date: same treatment as no stamp at all."""
        root = tmp_path / "reports"
        _report(root, "AAA", "20260918_120000", rating="Buy", entry="100.00", stop="92.00")
        _report(root, "BADSTAMP", "20261340_120000", rating="Buy", entry="10", stop="9")

        for since in ("2026-01-01", "2026-9-1"):
            out, err = _rank_main(rs, monkeypatch, capsys, "--tickers", "AAA", "BADSTAMP", "--json", "--since", since)
            assert [r["ticker"] for r in json.loads(out)] == ["AAA"]
            assert err == (
                "✗ BADSTAMP：报告目录无（或非法）_YYYYMMDD_HHMMSS 时间戳，"
                f"无法判定 --since={since}，已排除：{root / 'BADSTAMP_20261340_120000'}\n"
            )

    def test_r17_rank_main_since_is_inclusive_and_reads_the_newest_reports_stamp(
        self, rs, monkeypatch, capsys, tmp_path
    ):
        root = tmp_path / "reports"
        _report(root, "AAA", "20260918_120000", rating="Buy", entry="100.00", stop="92.00", mtime=100.0)
        _report(root, "AAA", "20260919_090000", rating="Sell", entry="10.00", stop="11.00", mtime=200.0)
        _report(root, "OLD", "20260918_120000", rating="Buy", entry="10.00", stop="9.00")

        out, err = _rank_main(rs, monkeypatch, capsys, "--tickers", "AAA", "OLD", "--json", "--since", "2026-09-18")
        rows = json.loads(out)

        assert err == ""
        # --since is inclusive; and only the date of the NEWEST report decides, so the
        # 09-18 buy of AAA does not matter once its 09-19 report exists.
        assert [(r["ticker"], r["direction"]) for r in rows] == [("AAA", "short"), ("OLD", "long")]
        out, err = _rank_main(rs, monkeypatch, capsys, "--tickers", "AAA", "OLD", "--json", "--since", "2026-09-19")
        assert [r["ticker"] for r in json.loads(out)] == ["AAA"]
        out, err = _rank_main(rs, monkeypatch, capsys, "--tickers", "AAA", "OLD", "--json", "--since", "2026-09-20")
        assert (json.loads(out), err) == ([], "")

    def test_r17_rank_main_ticker_case_and_dots(self, rs, monkeypatch, capsys, tmp_path):
        root = tmp_path / "reports"
        # A dotted name is left exactly as typed (glob is case-sensitive off Windows);
        # a bare one is upper-cased.
        _report(root, "brk.B", rating="Buy", entry="100.00", stop="92.00")
        _report(root, "AAA", rating="Buy", entry="100.00", stop="92.00")

        out, err = _rank_main(rs, monkeypatch, capsys, "--tickers", "brk.B", "aaa", "--json")

        assert err == ""
        assert [r["ticker"] for r in json.loads(out)] == ["brk.B", "AAA"]


class TestRankTicket:
    def test_r17_rank_render_ticket_md_composes_build_then_render(self, rs, monkeypatch, tmp_path):
        calls = []
        fake = SimpleNamespace(
            build_ticket=lambda report_dir, capital, profile: calls.append(("build", report_dir, capital, profile)) or {"t": 1},
            render_ticket=lambda ticket: calls.append(("render", ticket)) or "# ticket",
        )
        monkeypatch.setitem(sys.modules, "trade_ticket", fake)

        assert rs._render_ticket_md(tmp_path / "R", 5000.0) == "# ticket"
        assert calls == [("build", tmp_path / "R", 5000.0, "moderate"), ("render", {"t": 1})]

    def test_r17_rank_render_ticket_md_on_a_real_report_tree(self, rs, tmp_path):
        report_dir = _report(tmp_path / "reports", "AAA", rating="Buy", entry="100.00", stop="92.00")

        text = rs._render_ticket_md(report_dir, 10000.0)

        assert text.startswith("# 交易执行单 · AAA")
        assert "## ④ 执行建议" in text

    def test_r17_rank_a_failing_ticket_does_not_lose_the_ranking(self, rs, monkeypatch, capsys, tmp_path):
        root = tmp_path / "reports"
        _classic_reports(root)

        def boom(report_dir, capital):
            if report_dir.name.startswith("DWSH"):
                raise RuntimeError("brackets endpoint down")
            return "TICKET(AAA)"

        monkeypatch.setattr(rs, "_render_ticket_md", boom)
        out, err = _rank_main(rs, monkeypatch, capsys, "--tickers", "AAA", "DWSH", "BBB", "CCC", "--capital", "10000")

        assert err == ""
        assert out.endswith(
            "\n---\n# 附：`AAA` 执行单（capital=10000.0, moderate 档）\nTICKET(AAA)"
            "\n\n---\n# 附：`DWSH` 执行单（capital=10000.0, moderate 档）\n_(生成执行单失败：brackets endpoint down)_\n"
        )

    def test_r17_rank_no_pick_means_no_ticket_even_with_capital(self, rs, ticket_stub, monkeypatch, capsys, tmp_path):
        _report(tmp_path / "reports", "CCC", rating="Hold")

        out, err = _rank_main(rs, monkeypatch, capsys, "--tickers", "CCC", "--capital", "10000")

        assert err == "" and ticket_stub == []
        assert out.endswith("\n".join([*_RANK_DISCLAIMER]) + "\n")
        assert "- 本组无可推荐标的（全为观望 / 熔断 / 无报告）。" in out.splitlines()


def test_r17_rank_main_guard_runs_main(monkeypatch, capsys):
    """``python scripts/rank_signals.py`` runs ``main()`` (argparse exits 2 on a bad flag)."""
    path = SCRIPTS_DIR / "rank_signals.py"
    code = compile(path.read_text(encoding="utf-8"), str(path), "exec")
    monkeypatch.setattr(sys, "path", list(sys.path))  # the script's shim edits sys.path

    _set_argv(monkeypatch, "rank_signals.py", ["--top", "-1"])
    with pytest.raises(SystemExit) as excinfo:
        exec(code, {"__name__": "__main__", "__file__": str(path)})  # noqa: S102

    assert excinfo.value.code == 2


# =========================================================================== #
# scripts/run_analyst_parallel_ab.py
# =========================================================================== #


@pytest.fixture
def abg():
    return _load("r17_ab_gate", "run_analyst_parallel_ab.py")


def _failed_run(ticker="AAPL", segment=None):
    """What ``_capture_run`` returns for a propagate that raised."""
    return {
        "ticker": ticker, "rating": "?", "market_report": "", "sentiment_report": "", "news_report": "",
        "fundamentals_report": "", "final_trade_decision": "", "risk_overlay": None, "wall_full": 0.001,
        "analyst_segment": segment, "prediction_analysts": [], "exception": "RuntimeError('endpoint down')",
    }


def _ok_run(ticker="AAPL", segment=10.0, rating="Buy"):
    return {
        "ticker": ticker, "rating": rating, "market_report": "m", "sentiment_report": "s", "news_report": "n",
        "fundamentals_report": "f", "final_trade_decision": f"**Rating**: {rating}", "risk_overlay": None,
        "wall_full": 30.0, "analyst_segment": segment, "prediction_analysts": [], "exception": None,
    }


# --------------------------------------------------------------------------- #
# Optional accelerators
# --------------------------------------------------------------------------- #


class TestAbAccelerators:
    @pytest.mark.parametrize(
        ("blocked", "flag", "names"),
        [
            pytest.param("scipy.stats", "_HAS_SCIPY", ["_scipy_stats"], id="scipy-absent"),
            pytest.param("sklearn.feature_extraction.text", "_HAS_SKLEARN", ["_sk_text", "_sk_metrics"], id="sklearn-text-absent"),
            pytest.param("sklearn.metrics.pairwise", "_HAS_SKLEARN", ["_sk_text", "_sk_metrics"], id="sklearn-metrics-absent"),
        ],
    )
    def test_r17_ab_missing_accelerator_sets_the_fallback_flags(self, monkeypatch, blocked, flag, names):
        monkeypatch.setitem(sys.modules, blocked, None)  # ``import <blocked>`` now raises ImportError

        mod = _load("r17_ab_without_" + blocked.replace(".", "_"), "run_analyst_parallel_ab.py")

        assert getattr(mod, flag) is False
        assert [getattr(mod, name) for name in names] == [None] * len(names)

    def test_r17_ab_without_scipy_every_rating_test_takes_the_manual_path(self, abg, monkeypatch):
        monkeypatch.setitem(sys.modules, "scipy.stats", None)
        mod = _load("r17_ab_without_scipy_stats", "run_analyst_parallel_ab.py")
        dense = ({"Buy": 20, "Hold": 20, "Sell": 20}, {"Buy": 21, "Hold": 19, "Sell": 20})

        p, method = mod.chi_square_test(*dense)

        assert method == "chi-square"
        assert p == mod._chi_square_manual(*dense)
        # and it stays within the documented tolerance of the exact (scipy) one
        assert p == pytest.approx(abg.chi_square_test(*dense)[0], abs=0.02)

    def test_r17_ab_chi_square_fallback_tracks_scipy(self, abg, monkeypatch):
        """The Wilson-Hilferty p-value stays within the documented ~0.01 of the exact one."""
        stats = pytest.importorskip("scipy.stats")
        monkeypatch.setattr(abg, "_scipy_stats", stats)
        monkeypatch.setattr(abg, "_HAS_SCIPY", True)
        tables = [
            ({"Buy": 8, "Hold": 2}, {"Buy": 2, "Hold": 8}),
            ({"Buy": 20, "Hold": 20, "Sell": 20}, {"Buy": 21, "Hold": 19, "Sell": 20}),
            ({"Buy": 30, "Hold": 30, "Sell": 30, "Underweight": 30}, {"Buy": 10, "Hold": 50, "Sell": 40, "Underweight": 20}),
            ({"Buy": 12, "Hold": 12, "Sell": 12, "Underweight": 12, "Overweight": 12},
             {"Buy": 12, "Hold": 12, "Sell": 12, "Underweight": 12, "Overweight": 12}),
        ]
        for hist_a, hist_b in tables:
            p_exact, method = abg.chi_square_test(hist_a, hist_b)
            assert method == "chi-square"
            # the documented ~0.01 is a TAIL accuracy; near p = 1 the approximation is looser
        assert abg._chi_square_manual(hist_a, hist_b) == pytest.approx(
            p_exact, abs=0.01 if p_exact <= 0.5 else 0.02
        )

    def test_r17_ab_tfidf_backends_fit_the_same_space(self, abg, monkeypatch):
        """sklearn's ``TfidfVectorizer()`` defaults and the stdlib TF-IDF agree to 1e-9."""
        sk_text = pytest.importorskip("sklearn.feature_extraction.text")
        sk_metrics = pytest.importorskip("sklearn.metrics.pairwise")
        monkeypatch.setattr(abg, "_HAS_SKLEARN", True)
        monkeypatch.setattr(abg, "_sk_text", sk_text)
        monkeypatch.setattr(abg, "_sk_metrics", sk_metrics)
        texts = [
            "strong upside momentum and expanding margins", "balanced signals mixed price action",
            "breakdown in trend weakening demand", "strong upside momentum, balanced signals",
            "", "margins margins margins",
        ]

        via_sklearn = abg._pairwise_cosines(texts)
        monkeypatch.setattr(abg, "_HAS_SKLEARN", False)
        via_stdlib = abg._pairwise_cosines(texts)

        assert len(via_sklearn) == 6
        assert [v for row in via_stdlib for v in row] == pytest.approx(
            [v for row in via_sklearn for v in row], abs=1e-9
        )
        assert via_sklearn[4] == pytest.approx([0.0] * 6)  # the empty text has no terms

    def test_r17_ab_pairwise_cosines_empty_input(self, abg):
        assert abg._pairwise_cosines([]) == []


# --------------------------------------------------------------------------- #
# Statistic helpers
# --------------------------------------------------------------------------- #


class TestAbChi2Survival:
    @pytest.mark.parametrize("df", [0, -1, -0.0])
    def test_r17_ab_no_degrees_of_freedom_cannot_reject(self, abg, df):
        assert abg._chi2_sf_wilson_hilferty(99.0, df) == 1.0

    @pytest.mark.parametrize("x", [0, -3.5, 0.0])
    def test_r17_ab_non_positive_statistic_cannot_reject(self, abg, x):
        assert abg._chi2_sf_wilson_hilferty(x, 4) == 1.0

    @pytest.mark.parametrize("df", [1, 2, 3, 4])
    def test_r17_ab_approximation_near_the_exact_survival_function(self, abg, df):
        stats = pytest.importorskip("scipy.stats")

        for x in (0.5, 3.84, 9.49, 20.0, 40.0):
            assert abg._chi2_sf_wilson_hilferty(x, df) == pytest.approx(float(stats.chi2.sf(x, df)), abs=0.012)

    def test_r17_ab_hand_computed_value_at_the_five_percent_point(self, abg):
        # df=4, x=9.49 (chi-square's 5 % point): (9.49/4)^(1/3) = 1.3333,
        # z = (1.3333 - (1 - 2/36)) / sqrt(2/36) = 1.650, sf = erfc(1.650 / sqrt 2) / 2.
        assert abg._chi2_sf_wilson_hilferty(9.49, 4) == pytest.approx(0.0495, abs=2e-3)

    def test_r17_ab_survival_is_monotone_and_clamped(self, abg):
        values = [abg._chi2_sf_wilson_hilferty(x, 3) for x in (0.1, 1.0, 5.0, 10.0, 50.0, 500.0)]

        assert values == sorted(values, reverse=True)
        assert abg._chi2_sf_wilson_hilferty(float("inf"), 3) == 0.0
        assert all(0.0 <= v <= 1.0 for v in values)


class TestAbManualChiSquare:
    def test_r17_ab_degenerate_tables_cannot_reject(self, abg):
        assert abg._chi_square_manual({}, {}) == 1.0
        assert abg._chi_square_manual({"Buy": 3}, {"Buy": 5}) == 1.0
        assert abg._chi_square_manual({"x": 0, "y": 0}, {"x": 0, "y": 0}) == 1.0

    def test_r17_ab_category_empty_in_both_legs_still_counts_in_the_degrees_of_freedom(
        self, abg, monkeypatch
    ):
        """Pins the current seam: the zero expected cell is smoothed (no division by zero)
        and adds nothing to the statistic, but the empty category still raises ``df`` --
        so the manual fallback gives another p-value than the scipy path, whose table
        drops categories that are empty in both legs (``_contingency``). Not reachable
        from the pipeline: ``rating_histogram`` only creates keys for ratings that occur.
        """
        seen = {}

        def record(stat, df):
            seen[df] = stat
            return 0.0

        monkeypatch.setattr(abg, "_chi2_sf_wilson_hilferty", record)

        with_hole = abg._chi_square_manual({"x": 3, "y": 1, "z": 0}, {"x": 0, "y": 4, "z": 0})
        without = abg._chi_square_manual({"x": 3, "y": 1}, {"x": 0, "y": 4})

        assert (with_hole, without) == (0.0, 0.0)  # whatever the spy returns
        # rows 4 and 4, columns 3 and 5 of 8: expected 1.5 / 2.5 / 1.5 / 2.5,
        # statistic 1.5 + 0.9 + 1.5 + 0.9; the smoothed hole adds 2e-9
        assert seen[1] == pytest.approx(4.8)
        assert seen[2] == pytest.approx(seen[1], rel=1e-9)  # same statistic, df 2 vs 1

    def test_r17_ab_pearson_statistic_on_a_hand_computed_table(self, abg, monkeypatch):
        seen = {}
        monkeypatch.setattr(abg, "_chi2_sf_wilson_hilferty", lambda stat, df: seen.update(stat=stat, df=df) or 0.5)

        assert abg._chi_square_manual({"Buy": 8, "Hold": 2}, {"Buy": 2, "Hold": 8}) == 0.5
        # every expected cell is 5: four cells of (3^2)/5
        assert seen == {"stat": pytest.approx(7.2), "df": 1}

    def test_r17_ab_scipy_failure_falls_back_to_the_manual_statistic(self, abg, monkeypatch):
        def explode(*args, **kwargs):
            raise ValueError("scipy blew up")

        monkeypatch.setattr(abg, "_HAS_SCIPY", True)
        monkeypatch.setattr(abg, "_scipy_stats", SimpleNamespace(chi2_contingency=explode))
        dense = ({"Buy": 20, "Hold": 20, "Sell": 20}, {"Buy": 21, "Hold": 19, "Sell": 20})

        assert abg.chi_square_test(*dense) == (abg._chi_square_manual(*dense), "chi-square")


class TestAbFisherAndContingency:
    @pytest.mark.parametrize(
        "table",
        [(0, 0, 0, 0), (0, 0, 5, 5), (5, 5, 0, 0), (3, 0, 3, 0), (0, 4, 0, 6)],
    )
    def test_r17_ab_fisher_empty_margin_cannot_reject(self, abg, table):
        assert abg._fisher_exact_2x2(*table) == 1.0

    @pytest.mark.parametrize(
        ("table", "expected"),
        [
            pytest.param((3, 0, 0, 3), 2 / 20, id="hand-computed-2-over-20"),
            pytest.param((5, 0, 0, 5), 2 / 252, id="hand-computed-2-over-252"),
            # tables no more probable than [[8,2],[2,8]]: x = 0, 1, 2 and their mirrors,
            # 2 * (1 + 100 + 2025) out of C(20, 10)
            pytest.param((8, 2, 2, 8), 2 * (1 + 100 + 2025) / 184756, id="symmetric-drift"),
        ],
    )
    def test_r17_ab_fisher_two_sided_values(self, abg, table, expected):
        assert abg._fisher_exact_2x2(*table) == pytest.approx(expected, abs=5e-4)

    def test_r17_ab_fisher_matches_scipy(self, abg):
        stats = pytest.importorskip("scipy.stats")
        rng = random.Random(3)

        compared = 0
        while compared < 40:
            a, b, c, d = (rng.randint(0, 12) for _ in range(4))
            if a + b == 0 or c + d == 0 or a + c == 0 or b + d == 0:
                continue  # scipy refuses empty margins; ours returns 1.0 (pinned above)
            compared += 1
            exact = float(stats.fisher_exact([[a, b], [c, d]])[1])
            assert abg._fisher_exact_2x2(a, b, c, d) == pytest.approx(exact, abs=1e-12)

    @pytest.mark.parametrize(
        ("hist_a", "hist_b"),
        [
            pytest.param({"a": 0, "b": 0}, {"a": 1, "b": 2}, id="empty-leg-a"),
            pytest.param({"a": 1, "b": 2}, {}, id="empty-leg-b"),
            pytest.param({"a": 3}, {"a": 4}, id="single-category"),
            pytest.param({"a": 3, "b": 0}, {"a": 4, "b": 0}, id="single-effective-category"),
            pytest.param({}, {}, id="both-empty"),
        ],
    )
    def test_r17_ab_no_contingency_layout(self, abg, hist_a, hist_b):
        assert abg._contingency(hist_a, hist_b) is None
        assert abg.chi_square_test(hist_a, hist_b) == (1.0, "degenerate (too few categories or samples)")

    def test_r17_ab_contingency_layout_drops_categories_empty_in_both_legs(self, abg):
        assert abg._contingency({"a": 2, "z": 0}, {"b": 3, "z": 0}) == ([[2, 0], [0, 3]], ["a", "b"], 0.8)
        # 10 per leg over 2 categories: every expected cell is 5 -> not sparse
        assert abg._contingency({"a": 5, "b": 5}, {"a": 5, "b": 5}) == ([[5, 5], [5, 5]], ["a", "b"], 5.0)


# --------------------------------------------------------------------------- #
# Gate summary
# --------------------------------------------------------------------------- #


def _passing_metrics():
    return {
        "n_serial": 10, "n_parallel": 10, "dropped_serial": 0, "dropped_parallel": 0,
        "rating_p": 0.9, "overlap_ratio": 0.9,
        "per_field_similarity": {"market_report": {"within_serial": 0.5, "within_parallel": 0.5, "between": 0.6}},
        "risk_determinism_serial": True, "risk_determinism_parallel": True,
        "empty_reports_serial": 0, "empty_reports_parallel": 0,
        "exceptions_serial": 0, "exceptions_parallel": 0,
        "speedup": 3.0, "speedup_na": False, "dry_run": False,
    }


class TestAbSummarizeGateEdges:
    def test_r17_ab_missing_p_value_fails_the_gate(self, abg):
        metrics = _passing_metrics()
        del metrics["rating_p"]

        assert abg.summarize_gate(metrics) == (False, ["rating chi-square: no p-value computed"])
        metrics["rating_p"] = None
        assert abg.summarize_gate(metrics) == (False, ["rating chi-square: no p-value computed"])

    def test_r17_ab_nan_p_value_fails_closed_while_infinity_passes(self, abg):
        metrics = _passing_metrics()
        metrics["rating_p"] = float("nan")

        ok, reasons = abg.summarize_gate(metrics)

        assert ok is False
        assert reasons == [
            "rating chi-square p=nan <= 0.05 (overlap fallback unavailable: n_min=10 >=5, overlap 0.90)"
        ]
        metrics["rating_p"] = float("inf")
        assert abg.summarize_gate(metrics) == (True, [])

    def test_r17_ab_nothing_measured_is_not_a_pass(self, abg):
        metrics = _passing_metrics()
        del metrics["speedup"]

        assert abg.summarize_gate(metrics) == (False, ["parallel speedup not measured"])


# --------------------------------------------------------------------------- #
# A leg that lost every propagate
# --------------------------------------------------------------------------- #


class TestAbCrashedLeg:
    @pytest.mark.parametrize(
        ("serial", "parallel"),
        [
            pytest.param([_failed_run(), _failed_run()], [_failed_run(), _failed_run()], id="both-legs"),
            pytest.param([_failed_run(), _failed_run()], [_ok_run(), _ok_run()], id="serial-leg"),
            pytest.param([_ok_run(), _ok_run()], [_failed_run(), _failed_run()], id="parallel-leg"),
        ],
    )
    def test_r17_ab_all_crashed_leg_is_a_reported_failure_not_a_crash(self, abg, serial, parallel):
        """Round-17: ``all([])`` is True, so an empty leg passed the segment check and
        the mean divided by zero. The whole run then died as an "infra crash" (exit 3,
        no report, no reasons) instead of the criterion-6 / criterion-7 failure those
        criteria exist to report."""
        metrics = abg._compute_metrics(serial, parallel, dry_run=False)

        assert (metrics["speedup"], metrics["speedup_na"]) == (None, True)
        assert (metrics["dropped_serial"], metrics["dropped_parallel"]) == (
            len(serial) if serial[0]["exception"] else 0, len(parallel) if parallel[0]["exception"] else 0,
        )
        ok, reasons = abg.summarize_gate(metrics)
        assert ok is False
        crashed = [leg for leg, runs in (("serial", serial), ("parallel", parallel)) if runs[0]["exception"]]
        assert reasons == [
            *[f"{len(serial)} exception(s) in {leg} leg" for leg in crashed],
            *[f"{leg} leg: all propagate runs failed (no usable samples)" for leg in crashed],
        ]

    def test_r17_ab_partially_crashed_legs_keep_their_speedup(self, abg):
        metrics = abg._compute_metrics([_failed_run(), _ok_run(segment=20.0)], [_failed_run(), _ok_run(segment=5.0)])

        assert (metrics["speedup"], metrics["speedup_na"]) == (4.0, False)
        assert (metrics["dropped_serial"], metrics["exceptions_serial"]) == (1, 1)
        assert abg.summarize_gate(metrics) == (False, [
            "1 exception(s) in serial leg", "1 exception(s) in parallel leg",
        ])


# --------------------------------------------------------------------------- #
# _capture_run()
# --------------------------------------------------------------------------- #


class TestAbCaptureRun:
    def test_r17_ab_capture_run_propagate_exception_becomes_a_failed_sample(self, abg):
        class Graph:
            def propagate(self, ticker, date, asset_type="stock"):
                raise ConnectionError("endpoint down")

        run = abg._capture_run(Graph(), "AAPL", "2026-07-01")

        assert run == {
            "ticker": "AAPL", "rating": "?", "market_report": "", "sentiment_report": "", "news_report": "",
            "fundamentals_report": "", "final_trade_decision": "", "risk_overlay": None,
            "wall_full": run["wall_full"], "analyst_segment": None, "prediction_analysts": [],
            "exception": "ConnectionError('endpoint down')",
        }
        assert run["wall_full"] >= 0.0

    def test_r17_ab_capture_run_tracker_failure_leaves_the_segment_unavailable(self, abg):
        class Tracker:
            def serialize(self):
                raise RuntimeError("telemetry corrupted")

        class Graph:
            perf_tracker = Tracker()

            def propagate(self, ticker, date, asset_type="stock"):
                return {"final_trade_decision": "**Rating**: Buy\n\n## Quantitative Risk Overlay\n- **Target Weight**: 12.5%"}, "Buy"

        run = abg._capture_run(Graph(), "AAPL", "2026-07-01")

        assert run["analyst_segment"] is None and run["exception"] is None
        assert run["rating"] == "Buy" and run["risk_overlay"] == {"target_weight": 12.5, "stop_loss": None, "entry_price": None}

    def test_r17_ab_capture_run_keyboard_interrupt_is_not_swallowed(self, abg):
        class Graph:
            def propagate(self, ticker, date, asset_type="stock"):
                raise KeyboardInterrupt

        with pytest.raises(KeyboardInterrupt):
            abg._capture_run(Graph(), "AAPL", "2026-07-01")

    def _predictions(self, monkeypatch, result):
        import yialpha.ledger.predictions as predictions_module

        calls = []

        def predictions_for_run(run_id):
            calls.append(run_id)
            if isinstance(result, BaseException):
                raise result
            return result

        monkeypatch.setattr(predictions_module, "predictions_for_run", predictions_for_run)
        return calls

    def test_r17_ab_capture_run_collects_the_distinct_prediction_analysts(self, abg, monkeypatch):
        calls = self._predictions(monkeypatch, [
            SimpleNamespace(analyst="news"), SimpleNamespace(analyst="market"), SimpleNamespace(analyst="news"),
        ])
        state = {"run_id": 77, "final_trade_decision": "**Rating**: Hold"}

        class Graph:
            perf_tracker = None

            def propagate(self, ticker, date, asset_type="stock"):
                return state, "Hold"

        run = abg._capture_run(Graph(), "AAPL", "2026-07-01")

        assert run["prediction_analysts"] == ["market", "news"]
        assert calls == ["77"]

    def test_r17_ab_capture_run_ledger_failure_is_fail_soft(self, abg, monkeypatch):
        calls = self._predictions(monkeypatch, RuntimeError("ledger locked"))

        class Graph:
            perf_tracker = None

            def propagate(self, ticker, date, asset_type="stock"):
                return {"run_id": "run-1"}, "Hold"

        run = abg._capture_run(Graph(), "AAPL", "2026-07-01")

        assert run["prediction_analysts"] == [] and run["exception"] is None
        assert calls == ["run-1"]

    def test_r17_ab_capture_run_without_run_id_never_opens_the_ledger(self, abg, monkeypatch):
        import yialpha.ledger.predictions as predictions_module

        def never(run_id):
            pytest.fail("ledger must not be read")

        monkeypatch.setattr(predictions_module, "predictions_for_run", never)
        calls = []

        class Graph:
            perf_tracker = None

            def propagate(self, ticker, date, asset_type="stock"):
                return {"final_trade_decision": "x"}, "Hold"

        run = abg._capture_run(Graph(), "AAPL", "2026-07-01")

        assert run["prediction_analysts"] == [] and calls == []


# --------------------------------------------------------------------------- #
# Report writer
# --------------------------------------------------------------------------- #


def _failing_metrics():
    return {
        "n_serial": 3, "n_parallel": 3, "exceptions_serial": 3, "exceptions_parallel": 0,
        "dropped_serial": 3, "dropped_parallel": 0,
        "prediction_attribution_measured": True, "prediction_attribution_ok": False,
        "speedup": None, "speedup_na": False, "dry_run": False,
        "serial_hist": {"Buy": 2, "Weird": 1}, "parallel_hist": {"Buy": 3},
        "rating_p": None, "overlap_ratio": 0.5,
        "per_field_similarity": {"market_report": {"within_serial": 0.5, "within_parallel": 0.25, "between": 0.3}},
        "risk_determinism_serial": None, "risk_determinism_parallel": True,
        "empty_reports_serial": 0, "empty_reports_parallel": 1,
    }


_AB_REASONS = [
    "rating chi-square: no p-value computed",
    "1 empty report(s) in parallel leg",
    "parallel speedup not measured",
    "3 exception(s) in serial leg",
    "serial leg: all propagate runs failed (no usable samples)",
    "prediction attribution: every parallel run's predictions land on a single analyst (__active__ capture race)",
]


class TestAbWriteReport:
    def test_r17_ab_report_renders_every_failure_and_every_na(self, abg, tmp_path):
        path = tmp_path / "nested" / "parallel_ab_AAPL_2026-07-01.md"

        abg.write_report(path, "AAPL", "2026-07-01", 3, _failing_metrics())

        assert path.read_text(encoding="utf-8") == "\n".join([
            "# Analyst Parallel A/B Gate — AAPL @ 2026-07-01",
            "",
            "- Runs per leg: **3**",
            "- Serial leg exceptions: 3",
            "- Parallel leg exceptions: 0",
            "- Dropped from stats (propagate failed): serial 3, parallel 0",
            "- Prediction attribution (S3-a race probe): FAILED (single-analyst collapse)",
            "- Parallel speedup: not measured",
            "",
            "## Rating distribution",
            "",
            "| Rating | Serial | Parallel |",
            "|---|---:|---:|",
            "| Buy | 2 | 3 |",
            "| Overweight | 0 | 0 |",
            "| Hold | 0 | 0 |",
            "| Underweight | 0 | 0 |",
            "| Sell | 0 | 0 |",
            "| Weird | 1 | 0 |",
            "",
            "- Chi-square p = N/A",
            "- Distribution overlap = 0.50",
            "",
            "## Report similarity (TF-IDF cosine)",
            "",
            "| Field | within-serial | within-parallel | between | between >= min(within)? |",
            "|---|---:|---:|---:|:---:|",
            "| market_report | 0.500 | 0.250 | 0.300 | YES |",
            "",
            "## Risk-overlay determinism",
            "",
            "- Serial leg: N/A (overlay present but no numbers parsed; not verified)",
            "- Parallel leg: PASS",
            "",
            "## Verdict: FAIL",
            "",
            "Reasons:",
            *[f"- {reason}" for reason in _AB_REASONS],
            "",
        ])

    @pytest.mark.parametrize(
        ("measured", "ok", "line"),
        [
            pytest.param(False, False, None, id="not-measured-is-not-rendered"),
            pytest.param(True, True, "- Prediction attribution (S3-a race probe): OK (2+ distinct analysts in a parallel run)", id="ok"),
            pytest.param(True, False, "- Prediction attribution (S3-a race probe): FAILED (single-analyst collapse)", id="collapsed"),
        ],
    )
    def test_r17_ab_report_prediction_attribution_lines(self, abg, tmp_path, measured, ok, line):
        metrics = _passing_metrics()
        metrics.update({"prediction_attribution_measured": measured, "prediction_attribution_ok": ok})
        path = tmp_path / "r.md"

        abg.write_report(path, "AAPL", "2026-07-01", 10, metrics)

        lines = path.read_text(encoding="utf-8").splitlines()
        rendered = [item for item in lines if "Prediction attribution" in item]
        assert rendered == ([line] if line else [])
        assert lines[3:6] == ["- Serial leg exceptions: 0", "- Parallel leg exceptions: 0",
                              "- Dropped from stats (propagate failed): serial 0, parallel 0"]

    @pytest.mark.parametrize(
        ("speedup", "expected"),
        [
            pytest.param({"speedup": 3.0}, "- Parallel speedup: **3.00x** (threshold >= 2.5x)"),
            pytest.param({"speedup": 1.2}, "- Parallel speedup: **1.20x** (threshold >= 2.5x)"),
            pytest.param({"speedup": None}, "- Parallel speedup: not measured"),
        ],
    )
    def test_r17_ab_report_speedup_lines(self, abg, tmp_path, speedup, expected):
        metrics = {**_passing_metrics(), **speedup}

        lines = _report_lines(abg, tmp_path, metrics)

        assert [item for item in lines if "Parallel speedup" in item] == [expected]


def _report_lines(abg, tmp_path, metrics, *, dry_run=False, n=10):
    path = tmp_path / f"r{len(list(tmp_path.iterdir()))}.md"
    abg.write_report(path, "AAPL", "2026-07-01", n, metrics, dry_run=dry_run)
    return path.read_text(encoding="utf-8").splitlines()


class TestAbRealRun:
    def _args(self, tmp_path, n=2):
        return SimpleNamespace(tickers=["AAPL"], date="2026-07-01", n=n, out_dir=str(tmp_path),
                               max_threads=None, seed=42)

    def test_r17_ab_every_propagate_failing_ends_in_a_written_fail_report(self, abg, monkeypatch, capsys, tmp_path):
        """The end-to-end shape of the round-17 fix: the per-run failure count and the
        all-runs-failed reasons reach the console and the report."""
        class _Crashing:
            perf_tracker = None

            def __init__(self):
                self.calls = 0

            def propagate(self, ticker, date, asset_type="stock"):
                self.calls += 1
                raise ConnectionError("endpoint down")

        graphs = {"serial": _Crashing(), "parallel": _Crashing()}
        monkeypatch.setattr(abg, "_build_graph", lambda leg, args: graphs[leg])
        monkeypatch.setitem(DEFAULT_CONFIG, "llm_cache", False)  # keep the preflight quiet
        monkeypatch.setitem(DEFAULT_CONFIG, "memory_enabled", False)
        monkeypatch.setitem(DEFAULT_CONFIG, "llm_rate_limiter", False)

        code = abg._real_run(self._args(tmp_path))

        out = capsys.readouterr().out
        report = tmp_path / "parallel_ab_AAPL_2026-07-01.md"
        assert code == 1
        assert out.splitlines() == [  # the ticker banner is preceded by a blank line
            "",
            "=== AAPL @ 2026-07-01 | n=2 per leg ===",
            "  [AAPL] serial warm-up (discarded) ...",
            "  [AAPL] serial run 1/2 ...",
            "  [AAPL] serial run 2/2 ...",
            "  [AAPL] parallel warm-up (discarded) ...",
            "  [AAPL] parallel run 1/2 ...",
            "  [AAPL] parallel run 2/2 ...",
            f"[AAPL] report -> {report}",
            "[AAPL] verdict: FAIL",
            "    - 2 exception(s) in serial leg",
            "    - 2 exception(s) in parallel leg",
            "    - serial leg: all propagate runs failed (no usable samples)",
            "    - parallel leg: all propagate runs failed (no usable samples)",
        ]
        assert out.endswith("no usable samples)\n")
        assert graphs["serial"].calls == graphs["parallel"].calls == 3  # warm-up + n
        text = report.read_text(encoding="utf-8")
        assert "## Verdict: FAIL" in text
        for reason in (
            "2 exception(s) in serial leg", "2 exception(s) in parallel leg",
            "serial leg: all propagate runs failed (no usable samples)",
            "parallel leg: all propagate runs failed (no usable samples)",
        ):
            assert f"- {reason}" in text
        metrics = json.loads((tmp_path / "parallel_ab_AAPL_2026-07-01.json").read_text(encoding="utf-8"))
        assert (metrics["speedup"], metrics["speedup_na"], metrics["dropped_serial"]) == (None, True, 2)

    def test_r17_ab_harness_crash_is_exit_3_and_writes_nothing(self, abg, monkeypatch, capsys, tmp_path):
        def explode(leg, args):
            raise KeyError("quick_think_llm")

        monkeypatch.setattr(abg, "_build_graph", explode)
        monkeypatch.setitem(DEFAULT_CONFIG, "llm_cache", False)  # keep the preflight quiet
        monkeypatch.setitem(DEFAULT_CONFIG, "memory_enabled", False)
        monkeypatch.setitem(DEFAULT_CONFIG, "llm_rate_limiter", False)

        code = abg._real_run(self._args(tmp_path))

        captured = capsys.readouterr()
        assert code == 3
        assert captured.err.strip() == (
            "[infra] A/B harness crashed before a verdict could be reached "
            "(this is NOT a statistical FAIL): KeyError('quick_think_llm')"
        )
        assert "verdict" not in captured.out
        assert list(tmp_path.iterdir()) == []


# --------------------------------------------------------------------------- #
# Preflight / graph builder / dry-run driver
# --------------------------------------------------------------------------- #


class TestAbPreflight:
    def _warn(self, abg, monkeypatch, **overrides):
        for key, value in overrides.items():
            monkeypatch.setitem(DEFAULT_CONFIG, key, value)

    def test_r17_ab_clean_config_prints_nothing(self, abg, monkeypatch, capsys):
        self._warn(abg, monkeypatch, llm_cache=False, memory_enabled=False, llm_rate_limiter=True, llm_rpm=120)

        abg._preflight_warn(None)

        assert capsys.readouterr().err == ""

    def test_r17_ab_three_warnings_in_a_fixed_order(self, abg, monkeypatch, capsys):
        self._warn(abg, monkeypatch, llm_cache=True, memory_enabled=True, llm_rate_limiter=True, llm_rpm=60)

        abg._preflight_warn(None)

        err = capsys.readouterr().err.splitlines()
        assert len(err) == 3
        assert err[0].startswith("[preflight] WARNING: YIALPHA_LLM_CACHE=true is set.")
        assert err[1] == (
            "[preflight] WARNING: memory_enabled=true. A shared memory log makes later runs "
            "depend on earlier ones (non-i.i.d. legs). This script FORCES memory_enabled=False on both legs."
        )
        assert err[2] == (
            "[preflight] WARNING: llm_rate_limiter is ON with llm_rpm=60 (< 120). The parallel-leg "
            "speedup is gated by the shared limiter bucket; consider raising llm_rpm or "
            "disabling the limiter for this gate run."
        )

    @pytest.mark.parametrize(
        ("overrides", "warns"),
        [
            pytest.param({"llm_rate_limiter": False, "llm_rpm": 6}, False, id="limiter-off"),
            pytest.param({"llm_rate_limiter": True, "llm_rpm": 119}, True, id="just-under"),
            pytest.param({"llm_rate_limiter": True, "llm_rpm": 120}, False, id="at-120"),
            pytest.param({"llm_rate_limiter": True, "llm_rpm": "30"}, True, id="numeric-string"),
        ],
    )
    def test_r17_ab_limiter_warning_threshold(self, abg, monkeypatch, capsys, overrides, warns):
        self._warn(abg, monkeypatch, llm_cache=False, memory_enabled=False, **overrides)

        abg._preflight_warn(None)

        assert ("llm_rate_limiter is ON" in capsys.readouterr().err) is warns

    def test_r17_ab_unimportable_config_stops_the_preflight_with_a_note(self, abg, monkeypatch, capsys):
        self._warn(abg, monkeypatch, llm_cache=True)
        monkeypatch.setitem(sys.modules, "yialpha.default_config", None)

        abg._preflight_warn(None)

        err = capsys.readouterr().err
        assert err.startswith("[preflight] could not import DEFAULT_CONFIG: ")
        assert "yialpha.default_config" in err and "YIALPHA_LLM_CACHE" not in err


class TestAbBuildGraph:
    def _built(self, abg, monkeypatch, leg, max_threads, **overrides):
        import yialpha.graph.trading_graph as trading_graph

        seen = []

        def fake_graph(selected_analysts=None, debug=None, config=None, **kwargs):
            seen.append(dict(config or {}))
            return "graph"

        monkeypatch.setattr(trading_graph, "YiAlphaGraph", fake_graph)
        for key, value in overrides.items():
            monkeypatch.setitem(DEFAULT_CONFIG, key, value)
        result = abg._build_graph(leg, SimpleNamespace(max_threads=max_threads))
        assert result == "graph" and len(seen) == 1
        return seen[0]

    def test_r17_ab_serial_leg_config(self, abg, monkeypatch):
        config = self._built(abg, monkeypatch, "serial", None, llm_cache=True, memory_enabled=True, batch_workers=8)

        assert (config["llm_cache"], config["memory_enabled"], config["batch_workers"]) == (False, False, 1)
        assert (config["node_perf_telemetry"], config["analyst_parallel"]) == (True, False)

    def test_r17_ab_parallel_leg_config(self, abg, monkeypatch):
        config = self._built(abg, monkeypatch, "parallel", None, llm_cache=True, memory_enabled=True)

        assert (config["llm_cache"], config["memory_enabled"], config["analyst_parallel"]) == (False, False, True)
        assert config["node_perf_telemetry"] is True and config["batch_workers"] == 1

    def test_r17_ab_max_threads_reaches_only_the_parallel_leg(self, abg, monkeypatch):
        parallel = self._built(abg, monkeypatch, "parallel", 4, analyst_parallel_max_threads=99)
        serial = self._built(abg, monkeypatch, "serial", 4, analyst_parallel_max_threads=99)

        assert parallel["analyst_parallel_max_threads"] == 4
        assert serial["analyst_parallel_max_threads"] == 99  # untouched on the serial leg

    def test_r17_ab_without_max_threads_the_config_value_stands(self, abg, monkeypatch):
        parallel = self._built(abg, monkeypatch, "parallel", None, analyst_parallel_max_threads=7)

        assert parallel["analyst_parallel_max_threads"] == 7

    def test_r17_ab_the_default_config_itself_is_not_mutated(self, abg, monkeypatch):
        import yialpha.graph.trading_graph as trading_graph

        monkeypatch.setattr(trading_graph, "YiAlphaGraph", lambda **kwargs: "graph")
        monkeypatch.setitem(DEFAULT_CONFIG, "llm_cache", True)
        before = dict(DEFAULT_CONFIG)

        abg._build_graph("parallel", SimpleNamespace(max_threads=3))

        assert dict(DEFAULT_CONFIG) == before


class TestAbDryRunDriver:
    def test_r17_ab_dry_run_prints_failures_but_still_exits_zero(self, abg, monkeypatch, capsys, tmp_path):
        real = abg._compute_metrics

        def with_a_crash(serial, parallel, *, dry_run=False):
            return {**real(serial, parallel, dry_run=dry_run), "exceptions_parallel": 1}

        monkeypatch.setattr(abg, "_compute_metrics", with_a_crash)
        args = SimpleNamespace(tickers=["AAPL", "MSFT"], date="2026-07-01", n=4, out_dir=str(tmp_path),
                               max_threads=None, seed=42)

        code = abg._dry_run(args)

        assert code == 0
        out = capsys.readouterr().out
        assert out.splitlines() == [
            line
            for ticker in ("AAPL", "MSFT")
            for line in (
                f"[{ticker}] dry-run report -> {tmp_path / f'parallel_ab_{ticker}_2026-07-01.md'}",
                f"[{ticker}] verdict: FAIL (dry-run, exit 0 regardless)",
                "    - 1 exception(s) in parallel leg",
            )
        ]
        assert sorted(p.name for p in tmp_path.iterdir()) == sorted(
            f"parallel_ab_{ticker}_2026-07-01.{ext}" for ticker in ("AAPL", "MSFT") for ext in ("md", "json")
        )
        metrics = json.loads((tmp_path / "parallel_ab_AAPL_2026-07-01.json").read_text(encoding="utf-8"))
        assert metrics["dry_run"] is True and metrics["speedup_na"] is True and metrics["n_serial"] == 4


def test_r17_ab_main_guard_exits_with_main_code(monkeypatch, capsys):
    """``python scripts/run_analyst_parallel_ab.py`` exits with ``main()``'s status."""
    path = SCRIPTS_DIR / "run_analyst_parallel_ab.py"
    code = compile(path.read_text(encoding="utf-8"), str(path), "exec")
    monkeypatch.setattr(sys, "path", list(sys.path))  # the script's shim edits sys.path

    _set_argv(monkeypatch, "run_analyst_parallel_ab.py", ["--tickers", "AAPL", "--date", "2026-07-01", "--n", "0"])
    with pytest.raises(SystemExit) as excinfo:
        exec(code, {"__name__": "__main__", "__file__": str(path)})  # noqa: S102

    assert excinfo.value.code == 2
    assert capsys.readouterr().err == "--n must be >= 1\n"


# --------------------------------------------------------------------------- #
# Order independence of the card itself
# --------------------------------------------------------------------------- #


def test_r17_default_config_is_resolved_on_every_use(monkeypatch):
    """Why ``DEFAULT_CONFIG`` in this file is a proxy.

    ``tests/test_env_overrides.py`` and ``tests/test_web_store_results_dir.py`` reload
    ``yialpha.default_config`` and leave a new dict behind. A dict bound at collection
    time then disagrees with every module loaded afterwards (the 20 order-dependent
    failures of the first full-suite run: ``llm_max_retries`` 6 on the dev box's ``.env``
    vs 2 after the reload). Here the same rebinding is done on purpose, before the
    script is loaded -- the order the full suite produces.
    """
    stale = dict(_default_config_module.DEFAULT_CONFIG)
    rebound = {**stale, "results_dir": "rebound-by-a-reload"}
    monkeypatch.setattr(_default_config_module, "DEFAULT_CONFIG", rebound)

    assert DEFAULT_CONFIG["results_dir"] == "rebound-by-a-reload"
    assert stale != DEFAULT_CONFIG and rebound == dict(DEFAULT_CONFIG)
    assert len(rebound) == len(DEFAULT_CONFIG) and "results_dir" in DEFAULT_CONFIG
    assert DEFAULT_CONFIG.get("results_dir") == "rebound-by-a-reload"
    assert {**rebound, "extra": 1} == {**DEFAULT_CONFIG, "extra": 1}

    # what a script loaded NOW sees is what the proxy answers
    monkeypatch.setitem(DEFAULT_CONFIG, "memory_enabled", True)
    script = _load("r17_run_baseline_after_reload", "run_baseline.py")
    built = []

    class _Graph:
        def __init__(self, selected_analysts, debug=False, config=None):
            built.append({
                "selected_analysts": tuple(selected_analysts), "debug": debug, "config": dict(config or {}),
            })

    monkeypatch.setattr(script, "YiAlphaGraph", _Graph)

    script._build_graph(risk_enabled=False)

    assert built == [{
        "selected_analysts": ("market", "social", "news", "fundamentals"),
        "debug": False,
        "config": {**rebound, "memory_enabled": False, "risk_enabled": False},
    }]
    assert rebound["memory_enabled"] is True  # the setitem reached the live dict itself
