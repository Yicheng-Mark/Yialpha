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
    _reconfigure = getattr(_stream, "reconfigure", None)
    if callable(_reconfigure):
        with __import__("contextlib").suppress(AttributeError, ValueError):
            _reconfigure(encoding="utf-8", errors="replace")

# noqa: E402 — imports follow the UTF-8 reconfigure guard above; reordering would
# re-introduce UnicodeEncodeError when printing ❌/✅/中文 on a GBK Windows console.
from yialpha.logging_config import setup_logging  # noqa: E402

setup_logging()  # noqa: E402 — centralised logging before any other import fires

# noqa: E402 — imports follow the UTF-8 reconfigure guard above; reordering would
# re-introduce UnicodeEncodeError when printing ❌/✅/中文 on a GBK Windows console.
import argparse  # noqa: E402
import contextlib  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import subprocess  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ThreadPoolExecutor, as_completed  # noqa: E402
from datetime import datetime  # noqa: E402
from pathlib import Path  # noqa: E402

# Allow running as `python scripts/run_robust.py` without an editable install.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent

IS_WINDOWS = os.name == "nt"
# Kill the whole process tree on timeout; CREATE_NEW_PROCESS_GROUP gives a clean
# tree root so /T reliably reaches every descendant (grandchild datafetch etc.).
_CREATE_FLAGS = subprocess.CREATE_NEW_PROCESS_GROUP if IS_WINDOWS else 0

# Default shim location auto-loads sitecustomize at interpreter startup (urllib
# hard timeout + socket backstop). Override with YIALPHA_TIMEOUT_SHIM_DIR.
_DEFAULT_SHIM_DIR = os.environ.get(
    "YIALPHA_TIMEOUT_SHIM_DIR",
    str(Path(os.environ.get("TEMP", str(Path.home()))) / "yialpha_timeout_shim"),
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
        "--no-llm-cache",
        action="store_true",
        help="禁用 per-call LLM 响应缓存（默认开）。robust 重跑默认复用缓存："
        "首跑缓存冷=字节等价；卡死重试时已完成节点按 (model+prompt+temp+tools) "
        "回放、未产出 generation 的卡死节点重新执行——正是 hang 恢复想要的行为，"
        "避免每次重试把整张图重新计费。setdefault 尊重用户在 env 里显式设的值；"
        "本 flag 强制关闭。注意：勿用于 A/B gate / DSR 多 run 分布检验（那些仍需关）",
    )
    p.add_argument(
        "--reports-root",
        default=str(
            Path(os.getenv("YIALPHA_RESULTS_DIR", Path.home() / ".yialpha" / "logs"))
            / "reports"
        ),
        help="报告根目录（默认 $YIALPHA_RESULTS_DIR/reports，回退 ~/.yialpha/logs/reports）",
    )
    p.add_argument(
        "--allow-degraded",
        action="store_true",
        help="接受降级 run：把子进程的 data_vacuum_policy 降为 warn（数据真空仍产出 "
        "DEGRADED 报告而非类型化失败），且不再因核心数据降级重跑。这是旧默认行为"
        "的逃生口；默认（不开）= 质量闸门全开：真空 run 在 trader 节点抛 "
        "DataVacuumError、退出码非 0，部分降级报告也按失败重跑。",
    )
    p.add_argument(
        "--require-data-quality",
        action="store_true",
        help="（已默认开启，保留兼容；见 --allow-degraded）",
    )
    opts = p.parse_args()
    # Quality gate is ON by default now: the only way back to the old
    # "accept a degraded report" semantics is the explicit --allow-degraded
    # escape hatch. (--require-data-quality stays accepted as a no-op for
    # compat with existing invocations/docs.)
    opts.require_data_quality = not opts.allow_degraded
    return opts


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


def _core_sentinel_count(reports_root: Path, ticker: str, date: str) -> int | None:
    """Read the finished run's data_quality block; None when unavailable.

    ``full_states_log_<date>.json`` (written atomically by the graph before
    complete_report.md) carries ``data_quality.core_sentinel_count`` — how
    many CORE categories degraded to NO_DATA_AVAILABLE during the run. This
    distinguishes a fully-fed report from a data-vacuum HOLD: a brand-new
    complete_report.md alone no longer proves the run actually had data.
    Returns None when the log is missing/unreadable or predates the
    data_quality field (older runs) — unknown, not zero.
    """
    log = (
        reports_root.parent / ticker / "YiAlphaStrategy_logs"
        / f"full_states_log_{date}.json"
    )
    try:
        data = json.loads(log.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    block = data.get("data_quality") or {}
    count = block.get("core_sentinel_count")
    return count if isinstance(count, int) else None


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
            os.killpg(os.getpgid(pid), 9)  # type: ignore[attr-defined]


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


# ---- Ctrl+C / abort tracking ------------------------------------------------
# Default Ctrl+C behaviour hangs for up to per_ticker_timeout: the `with
# ThreadPoolExecutor` form runs shutdown(wait=True) in __exit__, which joins
# worker threads stuck in time.sleep(5)/proc.poll() (non-interruptible on
# Windows), while the CREATE_NEW_PROCESS_GROUP children ignore Ctrl+C and keep
# burning API quota. main() breaks this by (a) killing every active child and
# (b) setting _abort before shutdown joins — each worker's proc.poll() then
# turns non-None, its watchdog loop exits, and shutdown rejoins in seconds.
_abort = threading.Event()
_active_procs: list[subprocess.Popen] = []
_active_lock = threading.Lock()


def _register_proc(proc: subprocess.Popen) -> None:
    with _active_lock:
        _active_procs.append(proc)


def _unregister_proc(proc: subprocess.Popen) -> None:
    # already removed (Ctrl+C handler killed and cleared it) is harmless.
    with _active_lock, contextlib.suppress(ValueError):
        _active_procs.remove(proc)


def _kill_all_active() -> None:
    """Kill every registered child process tree (Ctrl+C / abort path)."""
    with _active_lock:
        procs = list(_active_procs)
    for proc in procs:
        if proc.poll() is None:
            _kill_tree(proc.pid)


def _apply_robust_llm_cache(child_env: dict, no_llm_cache: bool) -> None:
    """Set ``YIALPHA_LLM_CACHE`` on a robust child subprocess env, in place.

    Default-on for hang-recovery: a retry replays the completed nodes' cached
    LLM generations and only re-bills the call that actually hung (which never
    produced a generation, so it was never cached). First attempt is a cache
    miss → byte-equivalent to running with the cache off. ``--no-llm-cache``
    forces it off. A user who already exported ``YIALPHA_LLM_CACHE`` is
    respected (``setdefault``) unless ``--no-llm-cache`` explicitly overrides —
    run_robust is live single-config analysis, not an A/B-gate / DSR
    distribution measurement, so the response_cache distribution caveat does
    not apply.
    """
    if no_llm_cache:
        child_env["YIALPHA_LLM_CACHE"] = "false"
    else:
        child_env.setdefault("YIALPHA_LLM_CACHE", "true")


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
    # 可测试性 hook：用自定义脚本替换 run_batch 子进程（默认关 = 字节等价）。
    # 用于韧性测试：注入确定崩溃的 crasher 验证看门狗 + 重试契约。
    _child_script = os.environ.get("YIALPHA_ROBUST_CHILD_SCRIPT")
    if _child_script:
        cmd_base = [sys.executable, _child_script] + cmd_base[2:]

    for attempt in range(1, opts.max_attempts + 1):
        if _abort.is_set():
            break  # Ctrl+C during a prior attempt: stop launching fresh children
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
        child_env.setdefault("YIALPHA_URLOPEN_HARD_TIMEOUT_S", "20")
        child_env.setdefault("YIALPHA_FAULT_DUMP_S", "0")
        # Hang-recovery，与 _apply_robust_llm_cache 同一动机：半开 LLM 连接默认
        # 无限空转，只能等 OS 级看门狗把整个 attempt 杀掉重来；120s 读超时把它
        # 变成 APITimeoutError → SDK 内置重试秒级恢复。健康网络下永不触发 =
        # 字节等价，只改失败路径的恢复速度。setdefault — 用户显式导出的
        # YIALPHA_LLM_TIMEOUT_S（含 0=显式关闭）优先。
        child_env.setdefault("YIALPHA_LLM_TIMEOUT_S", "120")
        # 同理：BaoStock TCP 会话的瞬时故障（解码错/超时/连接重置）vendor 层
        # 默认 0 次（字节等价）；robust 子进程给 2 次退避重试，避免单次抖动
        # 把 a_share_native 类目打成哨兵、再由整个 attempt 重跑兜底。
        child_env.setdefault("YIALPHA_BAOSTOCK_RETRIES", "2")
        # --allow-degraded opts the child's data-vacuum gate down to warn as
        # well: without this the child would still raise DataVacuumError at the
        # trader node and never produce the degraded report the operator asked
        # to keep. setdefault — an explicit env var wins over the flag.
        if opts.allow_degraded:
            child_env.setdefault("YIALPHA_DATA_VACUUM_POLICY", "warn")
        # 让 run_batch 子进程 stdout/stderr 实时 flush：崩溃 traceback 不会闷在块缓冲里
        # 丢失（AAPL#1 偶发崩溃时 a1 日志只剩 7 行就是这个盲区）。字节等价——只改
        # flush 时机，不改输出内容；run_batch 的 LLM 决策不读自己的 stdout。
        child_env.setdefault("PYTHONUNBUFFERED", "1")
        _apply_robust_llm_cache(child_env, opts.no_llm_cache)

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
                if IS_WINDOWS:
                    proc = subprocess.Popen(
                        cmd_base,
                        cwd=str(_PROJECT_ROOT),
                        env=child_env,
                        stdout=logf,
                        stderr=subprocess.STDOUT,
                        creationflags=_CREATE_FLAGS,
                    )
                else:
                    proc = subprocess.Popen(
                        cmd_base,
                        cwd=str(_PROJECT_ROOT),
                        env=child_env,
                        stdout=logf,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                _register_proc(proc)  # so the Ctrl+C handler can reach this child
        except OSError as exc:
            result["reason"] = f"spawn_failed: {exc}"
            break

        # —— 看门狗循环 ——
        # Only sample CPU when stall detection is on (the sole reader of
        # win_start[1] is the `if opts.stall_timeout > 0` block below). With it
        # off (the default), each _cpu_seconds call spawns a PowerShell process
        # (0.5–2s cold start) per attempt whose result is never read.
        win_start = (
            time.time(),
            _cpu_seconds(proc.pid) if opts.stall_timeout > 0 else None,
        )
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
                            # Mirror the wall-clock kill path: _reap catches the
                            # TimeoutExpired that proc.wait(timeout=30) raises when
                            # a killed grandchild (the ssl.read hang case) still
                            # holds the stdout pipe, escalates to proc.kill(), and
                            # unconditionally reaps. A bare proc.wait(timeout=30)
                            # here would let that exception escape _run_one_ticker
                            # → the pool → fut.result(), dropping this ticker's
                            # result and skipping its remaining retries.
                            _reap(proc, ticker)
                            break
                    win_start = (now, cpu_now)
            time.sleep(5)

        # Child has exited (or was killed+reaped above): drop it from the active
        # set so the Ctrl+C handler no longer targets it. Done before the result
        # verdict so an exception in the verdict block cannot leak a stale entry.
        _unregister_proc(proc)
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
                # A new report exists — but did the run actually HAVE data?
                # core_sentinel_count > 0 means core categories degraded to
                # NO_DATA_AVAILABLE (a data-vacuum HOLD). Default semantics
                # stay "ok" (exit-code contract unchanged); the run is marked
                # DEGRADED so the summary shows it. --require-data-quality
                # escalates it to a failure + retry for operators who would
                # rather re-pull than keep an empty report.
                # sentinels is None when the evidence itself is unavailable
                # (missing/unreadable log, or a run predating the
                # data_quality field) — unknown, not zero. Strict mode cannot
                # verify quality then, so it retries; default mode stays ok
                # but says "unknown" instead of silently looking verified.
                sentinels = _core_sentinel_count(reports_root, ticker, opts.date)
                if sentinels is None:
                    if opts.require_data_quality:
                        result["degraded"] = True
                        result["reason"] = (
                            "DEGRADED: data-quality evidence unavailable (no "
                            "readable data_quality block in full_states_log) "
                            f"(wall {wall:.0f}s) → retry (--require-data-quality)"
                        )
                        print(
                            f"[{ticker}] ⚠️ attempt {attempt}: {result['reason']}",
                            flush=True,
                        )
                    else:
                        result["ok"] = True
                        result["report_path"] = new_report
                        result["reason"] = f"ok (quality unknown; wall {wall:.0f}s)"
                        print(
                            f"[{ticker}] ✅ attempt {attempt} done in {wall:.0f}s → "
                            f"{new_report} — data-quality evidence unavailable "
                            "(older run log?)",
                            flush=True,
                        )
                        break
                elif sentinels:
                    result["degraded"] = True
                    result["reason"] = (
                        f"DEGRADED: {sentinels} core data sentinel(s) "
                        f"(wall {wall:.0f}s)"
                    )
                    if opts.require_data_quality:
                        result["reason"] += " → retry (--require-data-quality)"
                        print(
                            f"[{ticker}] ⚠️ attempt {attempt}: {result['reason']}",
                            flush=True,
                        )
                    else:
                        result["ok"] = True
                        result["report_path"] = new_report
                        print(
                            f"[{ticker}] ⚠️ DEGRADED done in {wall:.0f}s → "
                            f"{new_report} — {sentinels} core category(ies) had "
                            f"NO data (see data_quality in full_states_log)",
                            flush=True,
                        )
                        break
                else:
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
    # Manage the pool manually (not `with`) so a KeyboardInterrupt can kill every
    # in-flight child BEFORE shutdown joins the worker threads. With the default
    # `with` form, __exit__'s shutdown(wait=True) would run first, joining threads
    # stuck in time.sleep(5)/proc.poll() (non-interruptible on Windows) while the
    # CREATE_NEW_PROCESS_GROUP children ignore Ctrl+C and keep running — a hang
    # of up to per_ticker_timeout (default 1800s). Here the except kills children
    # first; worker threads then see proc.poll() turn non-None, exit their
    # watchdog loop, and shutdown(wait=True) in finally rejoins within seconds.
    pool = ThreadPoolExecutor(max_workers=opts.workers)
    try:
        futs = {
            pool.submit(_run_one_ticker, t, opts.date, opts): t for t in opts.tickers
        }
        for fut in as_completed(futs):
            ticker = futs[fut]
            # 一个 ticker 的看门狗循环抛异常（如 _reap 之外的未预期错误，或线程内
            # SystemExit）绝不能炸掉整批：构造一个失败 result，对齐 in-process
            # BatchRunner 已有的容错风格，让其它 ticker 的成果照样落盘/汇报。
            # 关键：用 BaseException 而非 Exception —— SystemExit 继承 BaseException
            # 而非 Exception，若不捕获会传播出 try/finally，跳过末尾的汇总表打印
            # （表现为「没重试、没汇总就退出」）。KeyboardInterrupt 单独 re-raise，
            # 交给外层 except KeyboardInterrupt 做 _abort + 杀子清理。
            try:
                results.append(fut.result())
            except KeyboardInterrupt:
                raise
            except BaseException as exc:  # noqa: BLE001 -- isolate per-ticker failure
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
    except KeyboardInterrupt:
        _abort.set()  # stop workers mid-retry from launching fresh children
        _kill_all_active()
        print(
            "\n🛑 已中断：已强杀所有活跃子进程，等待工作线程退出…",
            file=sys.stderr, flush=True,
        )
    finally:
        pool.shutdown(wait=True)

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
