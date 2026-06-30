#!/usr/bin/env python
"""一键执行脚本：把 Phase 0/1/3/4 串起来跑真实回测。

用法（在项目根目录）：
    # -1) 起飞检查 —— 零 LLM 成本自检（PySocks/代理端口/yfinance/DeepSeek key），最先跑这个
    python scripts/run_baseline.py --preflight --ticker AAPL

    # 0) 冒烟测试 —— 1 只票 1 个日期，确认 LLM/网络/key 都通（最便宜，先跑这个）
    python scripts/run_baseline.py --smoke --ticker AAPL --date 2026-03-15

    # 1) 基线回测 —— 现状系统（LLM 决策，简单仓位），出基线报告
    python scripts/run_baseline.py --baseline --tickers AAPL NVDA

    # 2) 完整 A/B —— 基线 vs 风控增强，跑闸门判定，出报告+仪表盘
    python scripts/run_baseline.py --full --tickers AAPL NVDA --runs 2

参数说明：
    --tickers      代码列表，默认 AAPL NVDA（A股用 600519.SS）
    --date         冒烟测试的单个日期 YYYY-MM-DD
    --start/--end  回测区间，默认 end=今天-60天，start=end-180天
    --step         再平衡间隔交易日，默认 10（约两周）
    --holding-days 持仓天数，默认 5
    --runs         每只票跑几次取分布（LLM 非确定性），默认 2
    --rebalance    跑风控的再平衡日期数，默认 6（控制成本）
    --cost-bps     单边交易成本 bp，默认 5
    --out          输出目录，默认 ./backtest_output

注意：每个 propagate = 一次完整 LLM 图（4分析师+辩论+交易员+风控辩论+PM），
成本随 (tickers × 日期数 × runs) 线性增长。先用 --preflight 自检（零 LLM 成本），
全绿再 --smoke 验证，最后逐步放大。
"""

from __future__ import annotations

import argparse
import contextlib
import queue
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path

# Allow running as `python scripts/run_baseline.py` without an editable install:
# ensure the project root (parent of this scripts/ dir) is on sys.path.
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# Windows 控制台默认 GBK(cp936)，打印 ✅/❌ 等 Unicode 会触发 UnicodeEncodeError；
# 强制标准输出/错误流用 utf-8（Python 3.7+），让所有模式的中文与符号都能正常显示。
for _stream in (sys.stdout, sys.stderr):
    with contextlib.suppress(AttributeError, ValueError):
        _stream.reconfigure(encoding="utf-8", errors="replace")

import pandas as pd  # noqa: E402

from yiagents.backtest.engine import run_backtest  # noqa: E402
from yiagents.backtest.report import write_report  # noqa: E402
from yiagents.backtest.validation_gate import evaluate_gate  # noqa: E402
from yiagents.default_config import DEFAULT_CONFIG  # noqa: E402
from yiagents.graph.trading_graph import YiAgentsGraph  # noqa: E402
from yiagents.monitoring.dashboard import write_dashboard  # noqa: E402
from yiagents.risk.manager import RiskManager, build_backtest_weight_fn  # noqa: E402


def _rebalance_dates(start: str, end: str, step: int, n: int) -> list[str]:
    """从 [start,end] 里均匀抽 n 个交易日，间隔约 step 个交易日。"""
    idx = pd.bdate_range(start, end)
    if len(idx) < n:
        return [d.strftime("%Y-%m-%d") for d in idx]
    # 从末尾往前抽，保证有足够的"未来"价格做持仓 mark
    picked = [idx[-1 - step * i] for i in range(n) if i * step < len(idx)]
    picked.reverse()
    return [d.strftime("%Y-%m-%d") for d in picked]


def _build_graph(debug: bool = False) -> YiAgentsGraph:
    config = DEFAULT_CONFIG.copy()
    return YiAgentsGraph(debug=debug, config=config)


def _map_tickers(tickers, task, workers: int = 1):
    """Run ``task(ticker, graph) -> result`` for each ticker; concurrent across tickers.

    Each ticker borrows ONE graph instance from a pool of K (one graph per worker).
    ``propagate`` mutates instance state (self.ticker / self.curr_state / self.graph
    recompile), so a single shared graph cannot serve concurrent tickers — hence the
    pool, never shared across threads. Within a ticker, dates/runs stay serial (the
    task body loops them). ``workers`` <= 1 takes a strict serial path with one graph
    and is byte-equivalent to the previous single-``ta`` loop. Results come back in
    input ticker order.
    """
    if workers <= 1:
        graph = _build_graph(debug=False)
        return [task(t, graph) for t in tickers]

    pool: queue.Queue = queue.Queue()
    for _ in range(workers):
        pool.put(_build_graph(debug=False))

    results: dict = {}

    def run(t):
        graph = pool.get()
        try:
            return task(t, graph)
        finally:
            pool.put(graph)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        future_to_ticker = {ex.submit(run, t): t for t in tickers}
        for fut in as_completed(future_to_ticker):
            results[future_to_ticker[fut]] = fut.result()
    return [results[t] for t in tickers]


def preflight(ticker: str) -> int:
    """档 -1：零/近零成本自检——确认 smoke 能跑通（不构造 graph、不调 LLM 图）。

    5 项检查：依赖(含 PySocks) / env+key / 代理端口 / yfinance 实拉 / DeepSeek 探活。
    DeepSeek 用免费 GET /v1/models（非 chat completion），走 NO_PROXY 直连，零 LLM 成本。
    """
    import importlib
    import os
    import socket

    print(f"\n=== 起飞检查（preflight）：{ticker} ===")
    ok = True

    def check(name, cond, hint=""):
        nonlocal ok
        if not cond:
            ok = False
        line = f"  {'✅' if cond else '❌'} {name}"
        if not cond and hint:
            line += f"  → {hint}"
        print(line)

    # 1) 关键 Python 依赖（PySocks 是 requests/yfinance 走 socks5h 代理的前提）
    for mod in ["socks", "yfinance", "pandas", "httpx", "dotenv"]:
        try:
            importlib.import_module(mod)
            check(f"依赖 {mod}", True)
        except ImportError:
            hint = 'pip install "requests[socks]"  # 装 PySocks' if mod == "socks" else f"pip install {mod}"
            check(f"依赖 {mod}", False, hint)

    # 2) env / key（.env 由 yiagents 导入时通过 find_dotenv(usecwd=True) 加载，
    #    故必须从项目根目录运行，否则 os.environ 里拿不到 key）
    key = os.environ.get("DEEPSEEK_API_KEY", "")
    check("DEEPSEEK_API_KEY 已设置", bool(key),
          ".env 未加载——确认从项目根目录（含 .env 的目录）运行")
    check("LLM provider 已设置", bool(os.environ.get("YIAGENTS_LLM_PROVIDER")),
          ".env 里 YIAGENTS_LLM_PROVIDER 缺失")

    # 3) 代理端口可达（解析 HTTPS_PROXY 里的 host:port，TCP 探测）
    proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("HTTP_PROXY", "")
    host, port = "127.0.0.1", 1080
    if "://" in proxy:
        hp = proxy.split("://", 1)[1].split("/", 1)[0]
        if ":" in hp:
            host = hp.split(":", 1)[0]
            with contextlib.suppress(ValueError):
                port = int(hp.rsplit(":", 1)[1])
    try:
        socket.create_connection((host, port), timeout=3).close()
        check(f"代理端口 {host}:{port} 可达", True)
    except OSError:
        check(f"代理端口 {host}:{port} 可达", False, "确认 V2Ray/Xray 在监听该端口")

    # 4) yfinance 实拉（代理+数据 的真证明；失败时打印原始异常定位）
    try:
        import yfinance as yf
        df = yf.Ticker(ticker).history(period="5d", auto_adjust=True)
        check(f"yfinance 拉到 {ticker} 数据", len(df) > 0,
              "代理/PySocks/Yahoo 限流；若 PySocks 已装仍失败，检查 yfinance 底层是否走 curl_cffi")
    except Exception as e:  # noqa: BLE001
        check(f"yfinance 拉到 {ticker} 数据", False, repr(e)[:140])

    # 5) DeepSeek 连通（free GET /v1/models，走直连 NO_PROXY，零 LLM 成本）
    if key:
        try:
            import httpx
            r = httpx.get("https://api.deepseek.com/v1/models",
                          headers={"Authorization": f"Bearer {key}"}, timeout=12)
            hint = f"HTTP {r.status_code}"
            if r.status_code in (401, 403):
                hint += "（key 无效或未充值）"
            check("DeepSeek API 可达且 key 有效", r.status_code == 200, hint)
        except Exception as e:  # noqa: BLE001
            check("DeepSeek API 可达且 key 有效", False, repr(e)[:140])
    else:
        check("DeepSeek API 可达且 key 有效", False, "无 key，跳过（见上）")

    verdict = "✅ 全部通过，可以跑 --smoke" if ok else "❌ 有未通过项，先修上面 ❌ 再跑 --smoke"
    print(f"\n  → {verdict}")
    return 0 if ok else 2


def smoke(ticker: str, date: str) -> int:
    """档 0：跑一次 propagate，确认链路通。"""
    print(f"\n=== 冒烟测试：{ticker} @ {date} ===")
    try:
        ta = _build_graph(debug=False)
        final_state, rating = ta.propagate(ticker, date)
        print(f"评级: {rating}")
        decision = (final_state or {}).get("final_trade_decision", "")
        print(f"决策摘要（前 400 字）:\n{(decision or '')[:400]}")
        print("\n✅ 链路打通，LLM/网络/key 都正常。可以进 --baseline。")
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"\n❌ 冒烟失败：{exc}", file=sys.stderr)
        print("排查：1) 代理是否通  2) DEEPSEEK_API_KEY 是否有效  "
              "3) yfinance 能否拉到该票该日数据", file=sys.stderr)
        return 2


def baseline_backtest(tickers, start, end, step, n_dates, holding_days, cost_bps, runs, out, workers=1):
    """档 1：现状基线（简单评级→仓位）。

    ``workers`` > 1 并发跑多个 ticker（每票借一个独立 graph 实例，内部
    dates/runs 仍串行）。默认 ``workers``=1 与今天逐票串行完全等价。
    """
    print(f"\n=== 基线回测：{tickers} | {start}→{end} | {n_dates}×再平衡 ×{runs} 次 ===")
    dates = _rebalance_dates(start, end, step, n_dates)
    print(f"  再平衡日期 {dates}")

    def per_ticker(t, ta):
        out_res = []
        for r in range(runs):
            print(f"    [{t}] run {r} ...", flush=True)
            res = run_backtest(ta, t, dates, holding_days=holding_days,
                               cost_bps=cost_bps, run_tag=f"base_r{r}")
            m = res.metrics
            print(f"      [{t}] 总收益 {m.total_return:.2%} | Sharpe {m.sharpe:.2f} | "
                  f"MDD {m.max_drawdown:.2%} | vs B&H alpha {m.alpha_vs_buyhold:.2%}")
            out_res.append(res)
        return out_res

    nested = _map_tickers(tickers, per_ticker, workers)
    all_results = [r for sub in nested for r in sub]
    path = write_report(all_results, results_dir=out)
    write_dashboard(all_results, results_dir=out)
    print(f"\n📊 基线报告: {path}")
    print(f"📊 仪表盘(用浏览器打开): {Path(out) / 'monitoring' / 'dashboard.html'}")
    return all_results


def full_ab(tickers, start, end, step, n_dates, holding_days, cost_bps, runs, out, workers=1):
    """档 2：基线 vs 风控增强 + 闸门判定。

    ``workers`` > 1 并发跑多个 ticker（每票借独立 graph 实例）。默认 ``workers``=1
    与今天逐票串行等价；闸门判定与报告聚合始终串行（确定性顺序）。
    """
    print(f"\n=== 完整 A/B：基线 vs Phase-1 风控 | {tickers} ===")

    # 启用风控配置
    risk_cfg = DEFAULT_CONFIG.copy()
    risk_cfg["risk_enabled"] = True
    rm = RiskManager.from_config(risk_cfg)
    dates = _rebalance_dates(start, end, step, n_dates)
    print(f"  再平衡日期 {dates}")

    def per_ticker(t, ta):
        print(f"\n  [{t}] 开始", flush=True)
        base = run_backtest(ta, t, dates, holding_days=holding_days,
                            cost_bps=cost_bps, run_tag=f"ab_base_{t}")
        mb = base.metrics
        print(f"    [{t}] 基线:    总收益 {mb.total_return:.2%} | Sharpe {mb.sharpe:.2f} | "
              f"MDD {mb.max_drawdown:.2%}")

        improved = []
        wfn = build_backtest_weight_fn(rm, t)
        for r in range(runs):
            res = run_backtest(ta, t, dates, holding_days=holding_days,
                               cost_bps=cost_bps, weight_fn=wfn,
                               run_tag=f"ab_risk_{t}_r{r}")
            mi = res.metrics
            print(f"    [{t}] 风控 run{r}: 总收益 {mi.total_return:.2%} | Sharpe {mi.sharpe:.2f} | "
                  f"MDD {mi.max_drawdown:.2%} | DSR {mi.deflated_sharpe:.2f}")
            improved.append(res)
        return (t, base, improved)

    gate_inputs = _map_tickers(tickers, per_ticker, workers)

    # 每只票独立判定闸门（串行，确定性顺序）
    all_render = []
    for t, base, improved in gate_inputs:
        verdict = evaluate_gate(base, improved)
        md = verdict.render()
        print(f"\n  [{t}] 闸门判定: {'✅ PASS' if verdict.passes else '❌ FAIL'} "
              f"| DSR {verdict.mean_dsr:.2f} | 跑赢B&H {verdict.beats_buyhold}")
        print(f"    建议: {verdict.recommendation[:160]}...")
        (Path(out) / f"gate_{t}.md").write_text(md, encoding="utf-8")
        all_render += [base] + improved

    write_report(all_render, results_dir=out)
    write_dashboard([gi[1] for gi in gate_inputs], results_dir=out, kill_switch=False)
    print(f"\n📊 报告/仪表盘/闸门判定 都写到: {out}")


def main():
    p = argparse.ArgumentParser(description="YiAgents 一键回测")
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true", help="起飞检查：零成本自检（最先跑这个）")
    mode.add_argument("--smoke", action="store_true", help="冒烟：1票1日")
    mode.add_argument("--baseline", action="store_true", help="基线回测")
    mode.add_argument("--full", action="store_true", help="基线 vs 风控 A/B + 闸门")
    p.add_argument("--tickers", nargs="+", default=["AAPL", "NVDA"])
    p.add_argument("--ticker", default="AAPL", help="冒烟用单只")
    p.add_argument("--date", default="2026-03-15", help="冒烟日期")
    p.add_argument("--start", default=None)
    p.add_argument("--end", default=None)
    p.add_argument("--step", type=int, default=10)
    p.add_argument("--rebalance", type=int, default=6, help="再平衡次数")
    p.add_argument("--holding-days", type=int, default=5)
    p.add_argument("--cost-bps", type=float, default=5.0)
    p.add_argument("--runs", type=int, default=2)
    p.add_argument("--workers", type=int, default=1,
                   help="跨 ticker 并发数 K（1=串行，与今天等价；>1 并发，受 DeepSeek RPM/代理约束）")
    p.add_argument("--out", default="backtest_output")
    args = p.parse_args()

    end = args.end or (datetime.now() - timedelta(days=60)).strftime("%Y-%m-%d")
    start = args.start or (datetime.strptime(end, "%Y-%m-%d")
                           - timedelta(days=180)).strftime("%Y-%m-%d")

    if args.preflight:
        sys.exit(preflight(args.ticker))
    elif args.smoke:
        sys.exit(smoke(args.ticker, args.date))
    elif args.baseline:
        baseline_backtest(args.tickers, start, end, args.step, args.rebalance,
                          args.holding_days, args.cost_bps, args.runs, args.out,
                          workers=args.workers)
    else:
        full_ab(args.tickers, start, end, args.step, args.rebalance,
                args.holding_days, args.cost_bps, args.runs, args.out,
                workers=args.workers)


if __name__ == "__main__":
    main()
