#!/usr/bin/env python
"""健壮的按-ticker 子进程分析编排器（自动杀卡死 + 重跑）。

每个 ticker 跑在独立 OS 子进程里（`python scripts/run_batch.py --tickers <T>
--workers 1`），套一层硬墙钟看门狗；一旦超时或 CPU 停滞，用 `taskkill /F /T`
杀掉整棵进程树并重试（最多 N 次）。OS 级强杀不依赖 Python 能否打断 ssl.read，
所以无论 DeepSeek(httpx) 还是 urllib 那条路卡死都能恢复。

不 import 任何 agent / dataflow 源码（遵守「铁律不改 agent」），只 subprocess
调用 scripts/run_batch.py。

背景：in-process 的 BatchRunner 把所有 ticker 跑在同一进程里，一个调用卡死
→ 整个图冻结、无法自救。本脚本把「单 ticker」下沉到独立 OS 进程，卡死即可
强杀重跑，互不影响。

用法（项目根目录）：
    python scripts/run_robust.py --tickers SNDK INTC --date 2026-07-01 --workers 2

退出码：全部成功 0；任一失败（达到 max-attempts 仍无新报告）1。
"""

from __future__ import annotations

import sys

# Windows 控制台默认 GBK，打印 emoji/中文会触发 UnicodeEncodeError；强制 utf-8。
for _stream in (sys.stdout, sys.stderr):
    with __import__("contextlib").suppress(AttributeError, ValueError):
        _stream.reconfigure(encoding="utf-8", errors="replace")

import argparse
import contextlib
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

# Allow running as `python scripts/run_robust.py` without an editable install.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent

IS_WINDOWS = os.name == "nt"
# Kill the whole process tree on timeout; CREATE_NEW_PROCESS_GROUP gives a clean
# tree root so /T reliably reaches every descendant (grandchild datafetch etc.).
_CREATE_FLAGS = subprocess.CREATE_NEW_PROCESS_GROUP if IS_WINDOWS else 0

# Default shim location auto-loads sitecustomize at interpreter startup (urllib
# hard timeout + socket backstop). Override with YIAGENTS_TIMEOUT_SHIM_DIR.
_DEFAULT_SHIM_DIR = os.environ.get(
    "YIAGENTS_TIMEOUT_SHIM_DIR",
    str(Path(os.environ.get("TEMP", str(Path.home()))) / "yiagents_timeout_shim"),
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="按-ticker 子进程编排器：硬墙钟看门狗 + 卡死强杀 + 重跑。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--tickers", nargs="+", required=True, help="代码列表，如 SNDK INTC")
    p.add_argument("--date", required=True, help="分析日期 YYYY-MM-DD")
    p.add_argument(
        "--asset-type",
        default=None,
        choices=["stock", "crypto", "crypto_spot", "crypto_perp"],
        help="透传给 run_batch --asset-type（不传则不附加，与历史字节一致）；"
        "crypto_spot=Binance 现货，crypto_perp=Binance USDT-M 永续",
    )
    p.add_argument("--workers", type=int, default=2, help="并发子进程数 K（默认 2）")
    p.add_argument(
        "--per-ticker-timeout",
        type=float,
        default=1800.0,
        help="单 ticker 硬墙钟上限秒（默认 1800=30min，覆盖健康 ~20min + 余量）",
    )
    p.add_argument(
        "--max-attempts",
        type=int,
        default=3,
        help="单 ticker 最多尝试次数（含首次，默认 3）",
    )
    p.add_argument(
        "--stall-timeout",
        type=float,
        default=0.0,
        help="CPU 停滞检测窗口秒（0=关闭，默认关；开启则窗口内 CPU 增量<阈值判卡死）",
    )
    p.add_argument(
        "--stall-cpu",
        type=float,
        default=0.5,
        help="停滞窗口内 CPU 增量阈值秒（默认 0.5；--stall-timeout>0 时生效）",
    )
    p.add_argument("--backoff", type=float, default=15.0, help="重试间退避秒（默认 15）")
    p.add_argument(
        "--reports-root",
        default=str(Path.home() / ".yiagents" / "logs" / "reports"),
        help="报告根目录（默认 ~/.yiagents/logs/reports）",
    )
    return p.parse_args()


def _reports_root(reports_root: str) -> Path:
    root = Path(reports_root).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    return root


def _complete_mtime(reports_root: Path, ticker: str) -> float:
    """该 ticker 现存报告中 complete_report.md 的最新 mtime（无则 0）。

    用 complete_report.md 的 mtime（而非目录 mtime）做快照：它在 propagate 末尾
    最后写出，mtime>快照 才代表「本次新产出了一份完整报告」。
    """
    newest = 0.0
    for d in reports_root.glob(f"{ticker}_*"):
        cr = d / "complete_report.md"
        if cr.is_file():
            with contextlib.suppress(OSError):
                newest = max(newest, cr.stat().st_mtime)
    return newest


def _find_new_report(reports_root: Path, ticker: str, pre_mtime: float) -> Path | None:
    """找一份 mtime 比 pre_mtime 新且含 complete_report.md 的 <TICKER>_<stamp>/。"""
    best: Path | None = None
    best_mtime = pre_mtime
    for d in reports_root.glob(f"{ticker}_*"):
        cr = d / "complete_report.md"
        if not cr.is_file():
            continue
        with contextlib.suppress(OSError):
            m = cr.stat().st_mtime
            if m > best_mtime:
                best_mtime, best = m, cr
    return best


def _kill_tree(pid: int) -> None:
    """强杀整棵进程树（Windows: taskkill /F /T /PID；POSIX: kill -9 进程组）。"""
    if IS_WINDOWS:
        # /F 强制 /T 含所有子进程；忽略「进程已退出」的退出码 128。
        subprocess.run(
            ["taskkill", "/F", "/T", "/PID", str(pid)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    else:
        with contextlib.suppress(ProcessLookupError, OSError):
            os.killpg(os.getpgid(pid), 9)


def _reap(proc: subprocess.Popen, label: str, timeout: float = 30.0) -> None:
    """等被强杀的子进程退出；30s 仍不退则升级 proc.kill() 强 reap。

    ``proc.wait(timeout=...)`` 在子进程无视 kill（如抓数据的孙进程占着管道
    不放）时会抛 ``TimeoutExpired``。若不捕获，该异常会窜出
    ``_run_one_ticker`` → ``ThreadPoolExecutor`` → ``fut.result()`` → 整个
    编排器崩溃，丢失所有其它 ticker 已完成的结果。这里捕获后升级强杀并
    无条件 reap，让重试循环继续。
    """
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        print(
            f"[{label}] child did not exit within {timeout:.0f}s after kill; "
            f"force-killing and reaping",
            file=sys.stderr,
            flush=True,
        )
        with contextlib.suppress(ProcessLookupError, OSError):
            proc.kill()
        # 无 timeout 充分 reap，避免僵尸；此分支下进程已不可恢复。
        with contextlib.suppress(Exception):  # noqa: BLE001 -- best-effort reap
            proc.wait()


def _cpu_seconds(pid: int) -> float | None:
    """子进程累计 CPU 秒（PowerShell Get-Process；不可用/已退出返回 None）。"""
    if not IS_WINDOWS:
        return None
    try:
        out = subprocess.check_output(
            ["powershell", "-NoProfile", "-Command", f"(Get-Process -Id {pid}).CPU"],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=10,
        ).strip()
        return float(out) if out else None
    except (subprocess.SubprocessError, ValueError, OSError):
        return None


def _run_one_ticker(ticker: str, date: str, opts: argparse.Namespace) -> dict:
    """单 ticker 的「启动子进程 → 看门狗 → 杀/重试」循环。返回结果 dict。"""
    reports_root = _reports_root(opts.reports_root)
    log_dir = reports_root.parent / "robust"
    log_dir.mkdir(parents=True, exist_ok=True)

    result = {
        "ticker": ticker,
        "ok": False,
        "attempts": 0,
        "reason": "",
        "report_path": None,
        "log_path": None,
    }

    cmd_base = [
        sys.executable,
        str(_PROJECT_ROOT / "scripts" / "run_batch.py"),
        "--tickers",
        ticker,
        "--date",
        date,
        "--workers",
        "1",
        "--no-progress",
    ]
    # Only forward --asset-type when explicitly set, so a normal run's cmd_base
    # stays byte-identical to the historical watchdog contract (no extra argv).
    if getattr(opts, "asset_type", None):
        cmd_base += ["--asset-type", opts.asset_type]

    for attempt in range(1, opts.max_attempts + 1):
        result["attempts"] = attempt
        pre_mtime = _complete_mtime(reports_root, ticker)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = log_dir / f"robust_{ticker}_{stamp}_a{attempt}.log"
        result["log_path"] = log_path

        # 子进程 env：cwd=项目根让 .env 自动加载；注入 timeout shim 到 PYTHONPATH。
        child_env = os.environ.copy()
        child_env["PYTHONPATH"] = os.pathsep.join(
            [p for p in [_DEFAULT_SHIM_DIR, child_env.get("PYTHONPATH", "")] if p]
        )
        child_env.setdefault("YIAGENTS_URLOPEN_HARD_TIMEOUT_S", "20")
        child_env.setdefault("YIAGENTS_FAULT_DUMP_S", "0")

        print(
            f"[{ticker}] ▶️ attempt {attempt}/{opts.max_attempts} → "
            f"log {log_path.name}",
            flush=True,
        )
        t0 = time.time()
        kill_reason = ""
        try:
            with open(log_path, "w", encoding="utf-8", errors="replace") as logf:
                logf.write(
                    f"$ {' '.join(cmd_base)}\n# cwd={_PROJECT_ROOT} "
                    f"per_ticker_timeout={opts.per_ticker_timeout}s "
                    f"stall_timeout={opts.stall_timeout}s\n"
                )
                logf.flush()
                # Windows: CREATE_NEW_PROCESS_GROUP gives a clean tree root so
                # taskkill /T reaches every descendant. POSIX: start_new_session
                # puts the child in its own process group so os.killpg() below
                # kills the child tree and NOT this orchestrator (without it the
                # child shares our pgid and the watchdog would suicide). Each
                # kwarg is platform-only — Popen rejects start_new_session on
                # Windows and creationflags is a no-op 0 on POSIX — so the
                # Windows argv/flags stay byte-identical to the old behavior.
                popen_kwargs = dict(
                    cwd=str(_PROJECT_ROOT),
                    env=child_env,
                    stdout=logf,
                    stderr=subprocess.STDOUT,
                )
                if IS_WINDOWS:
                    popen_kwargs["creationflags"] = _CREATE_FLAGS
                else:
                    popen_kwargs["start_new_session"] = True
                proc = subprocess.Popen(cmd_base, **popen_kwargs)
        except OSError as exc:
            result["reason"] = f"spawn_failed: {exc}"
            break

        # —— 看门狗循环 ——
        win_start = (time.time(), _cpu_seconds(proc.pid))
        while True:
            rc = proc.poll()
            if rc is not None:
                break
            elapsed = time.time() - t0
            # 硬墙钟
            if elapsed > opts.per_ticker_timeout:
                kill_reason = (
                    f"wall_clock {elapsed:.0f}s > {opts.per_ticker_timeout:.0f}s"
                )
                _kill_tree(proc.pid)
                _reap(proc, ticker)
                break
            # CPU 停滞（可选）
            if opts.stall_timeout > 0:
                now = time.time()
                if now - win_start[0] >= opts.stall_timeout:
                    cpu_now = _cpu_seconds(proc.pid)
                    if cpu_now is not None and win_start[1] is not None:
                        delta = cpu_now - win_start[1]
                        if delta < opts.stall_cpu:
                            kill_reason = (
                                f"cpu stall: {delta:.2f}s over "
                                f"{opts.stall_timeout:.0f}s window"
                            )
                            _kill_tree(proc.pid)
                            proc.wait(timeout=30)
                            break
                    win_start = (now, cpu_now)
            time.sleep(5)

        rc = proc.returncode
        wall = time.time() - t0

        if kill_reason:
            print(
                f"[{ticker}] ⚠️ killed attempt {attempt}: {kill_reason} (wall {wall:.0f}s)",
                flush=True,
            )
        elif rc == 0:
            new_report = _find_new_report(reports_root, ticker, pre_mtime)
            if new_report:
                result["ok"] = True
                result["report_path"] = new_report
                result["reason"] = f"ok (wall {wall:.0f}s)"
                print(
                    f"[{ticker}] ✅ attempt {attempt} done in {wall:.0f}s → {new_report}",
                    flush=True,
                )
                break
            # 退出码 0 但没产出新报告：视为失败重跑。
            result["reason"] = "exit 0 but no new complete_report.md"
            print(f"[{ticker}] ⚠️ {result['reason']} → retry", flush=True)
        else:
            result["reason"] = f"exit {rc}"
            print(
                f"[{ticker}] ⚠️ attempt {attempt} failed (exit {rc}, wall {wall:.0f}s) → retry",
                flush=True,
            )

        if attempt < opts.max_attempts:
            print(f"[{ticker}] ⏳ backoff {opts.backoff:.0f}s before retry…", flush=True)
            time.sleep(opts.backoff)

    return result


def main() -> int:
    opts = _parse_args()

    print(
        f"🛡️ robust orchestrator | tickers={opts.tickers} date={opts.date} "
        f"workers={opts.workers} per_ticker_timeout={opts.per_ticker_timeout:.0f}s "
        f"max_attempts={opts.max_attempts} stall_timeout={opts.stall_timeout:.0f}s"
    )

    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=opts.workers) as pool:
        futs = {
            pool.submit(_run_one_ticker, t, opts.date, opts): t for t in opts.tickers
        }
        for fut in as_completed(futs):
            # 一个 ticker 的看门狗循环抛异常（如 _reap 之外的未预期错误）
            # 绝不能炸掉整批：构造一个失败 result，对齐 in-process BatchRunner
            # 已有的容错风格，让其它 ticker 的成果照样落盘/汇报。
            try:
                results.append(fut.result())
            except Exception as exc:  # noqa: BLE001 -- isolate per-ticker failure
                ticker = futs[fut]
                print(
                    f"[{ticker}] ❌ orchestrator error: {exc!r}",
                    file=sys.stderr,
                    flush=True,
                )
                results.append({
                    "ticker": ticker,
                    "ok": False,
                    "attempts": 0,
                    "reason": f"orchestrator_error: {exc!r}",
                    "report_path": None,
                    "log_path": None,
                })

    results.sort(key=lambda r: opts.tickers.index(r["ticker"]))
    ok = sum(1 for r in results if r["ok"])

    print(f"\n=== 健壮编排完成：{ok}/{len(results)} 成功 ===")
    print(f"{'ticker':<10} {'状态':<6} {'尝试':>4}  {'报告/原因'}")
    for r in results:
        status = "✅" if r["ok"] else "❌"
        detail = str(r["report_path"]) if r["ok"] else f"{r['reason']} (log={r['log_path']})"
        print(f"{r['ticker']:<10} {status:<6} {r['attempts']:>4}  {detail}")

    return 0 if ok == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
