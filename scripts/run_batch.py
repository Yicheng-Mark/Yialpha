#!/usr/bin/env python
"""批量并发分析多个 ticker（一个 API key 驱动多个 agent）。

把一批 ticker 经 BatchRunner 并发跑 propagate()——每个 ticker 走和单 ticker
完全相同的图，并发层只叠在 propagate() 之上，不碰任何 agent 的输入/深度。
一个批次只能含同一资产类别（全股票或全加密货币），因为所有 worker 共享同一
全局 config（dataflows/config.py 的 set_config）。

用法（项目根目录）：
    # 并发分析 3 只股票（K=3），默认 batch_concurrency 需为 true 或显式 --workers
    python scripts/run_batch.py --tickers AAPL NVDA MSFT --date 2026-06-27 --workers 3

    # 加密货币一批
    python scripts/run_batch.py --tickers BTC-USD ETH-USD SOL-USD --date 2026-06-27 --workers 3

    # 强制串行（K=1，与今天行为一致，便于做 G1 对照基线）
    python scripts/run_batch.py --tickers AAPL NVDA --date 2026-06-27 --workers 1

参数说明：
    --tickers      代码列表（nargs +），必填
    --date         分析日期 YYYY-MM-DD，必填
    --asset-type   stock | crypto | auto（auto=按首个 ticker 自动判定），默认 auto
    --workers      并发数 K（池大小）；不传则用 YIAGENTS_BATCH_WORKERS（默认 3）
    --no-progress  关闭 tqdm 进度条
    --out          （可选）覆盖 results_dir

注意：K 受 DeepSeek RPM / SOCKS5 代理并发上限 / 厂商限流约束。默认 K=3，先用
小批测，确认无 429 风暴再放大。一键回滚串行：YIAGENTS_BATCH_CONCURRENCY=false。
"""

from __future__ import annotations

import argparse
import contextlib
import sys
from datetime import datetime
from pathlib import Path

# Allow running as `python scripts/run_batch.py` without an editable install.
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# Windows 控制台默认 GBK，打印 ✅/❌/中文会触发 UnicodeEncodeError；强制 utf-8。
for _stream in (sys.stdout, sys.stderr):
    _reconfigure = getattr(_stream, "reconfigure", None)
    if callable(_reconfigure):
        with contextlib.suppress(AttributeError, ValueError):
            _reconfigure(encoding="utf-8", errors="replace")

from yiagents.logging_config import setup_logging  # noqa: E402

setup_logging()

from yiagents.batch.runner import BatchRunner  # noqa: E402


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="批量并发分析多个 ticker（同一资产类别）。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--tickers", nargs="+", required=True, help="代码列表，如 AAPL NVDA MSFT")
    p.add_argument("--date", required=True, help="分析日期 YYYY-MM-DD")
    p.add_argument(
        "--asset-type",
        choices=["auto", "stock", "crypto", "crypto_spot", "crypto_perp"],
        default="auto",
        help="auto=按首个 ticker 判定；一个批次需同一类；crypto_spot=Binance 现货，"
        "crypto_perp=Binance USDT-M 永续（均显式指定）",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=None,
        help="并发数 K（不传则用 YIAGENTS_BATCH_WORKERS，默认 3）",
    )
    p.add_argument("--no-progress", action="store_true", help="关闭进度条")
    p.add_argument("--out", default=None, help="覆盖 results_dir")
    return p.parse_args()


def main() -> int:
    args = _parse_args()

    # Shared validation/config assembly (the single source both this script
    # and `yiagents batch` agree on — the validation, worker-override and
    # config code that used to be duplicated here now lives in
    # yiagents.batch.runner.prepare_batch_run).
    from yiagents.batch.runner import (
        BatchInputError,
        batch_selected_analysts,
        prepare_batch_run,
    )

    try:
        config, asset_type = prepare_batch_run(
            args.tickers, args.date, args.asset_type,
            workers=args.workers, results_dir=args.out,
        )
    except BatchInputError as exc:
        print(f"❌ {exc}")
        return 2
    if args.asset_type == "auto":
        print(f"ℹ️  资产类别自动判定：{asset_type}（按 {args.tickers[0]}）")

    workers_hint = "" if args.workers is None else f" (K={args.workers})"
    print(
        f"▶️  批量分析 {len(args.tickers)} 个 ticker | date={args.date} | "
        f"type={asset_type}{workers_hint}"
    )

    runner_start = datetime.now()
    with BatchRunner(
        config,
        workers=args.workers,
        selected_analysts=batch_selected_analysts(asset_type, args.tickers),
        progress=not args.no_progress,
    ) as runner:
        results = runner.run(args.tickers, args.date, asset_type=asset_type)
    total_elapsed = (datetime.now() - runner_start).total_seconds()

    # 汇总表
    ok = sum(1 for r in results if r["error"] is None)
    print(f"\n=== 批量完成：{ok}/{len(results)} 成功，墙钟 {total_elapsed:.1f}s ===")
    print(f"{'ticker':<12} {'状态':<6} {'墙钟':>8}  详情")
    for r in results:
        ticker = r["ticker"]
        if r["error"] is None:
            elapsed = f"{r['elapsed']:.1f}s"
            report = str(r["report_path"]) if r["report_path"] else ""
            print(f"{ticker:<12} {'✅':<6} {elapsed:>8}  {report}")
        else:
            print(f"{ticker:<12} {'❌':<6} {'-':>8}  {type(r['error']).__name__}: {r['error']}")

    return 0 if ok == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
