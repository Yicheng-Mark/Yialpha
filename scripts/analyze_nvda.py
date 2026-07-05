"""一次性深度分析脚本：对单只票跑完整 propagate，落盘报告并 dump 各节观点。

用法： python scripts/analyze_nvda.py --ticker NVDA --date 2026-06-26
非 smoke：提取全部分析师/研究/交易/风险/PM 节，写到 dump 文件供汇总。
"""
import argparse
import json
import sys

# Windows 控制台默认 GBK，打印 ❌/✅ 会触发 UnicodeEncodeError；强制 utf-8。
for _stream in (sys.stdout, sys.stderr):
    with __import__("contextlib").suppress(AttributeError, ValueError):
        _stream.reconfigure(encoding="utf-8", errors="replace")

from pathlib import Path

from yiagents.default_config import DEFAULT_CONFIG
from yiagents.graph.trading_graph import YiAgentsGraph


def main() -> int:
    p = argparse.ArgumentParser(description="YiAgents 单票深度分析")
    p.add_argument("--ticker", default="NVDA")
    p.add_argument("--date", default="2026-06-26")
    p.add_argument("--out", default="analysis_output")
    args = p.parse_args()

    print(f"\n=== 深度分析：{args.ticker} @ {args.date} ===", flush=True)
    cfg = DEFAULT_CONFIG.copy()
    ta = YiAgentsGraph(debug=False, config=cfg)

    final_state, rating = ta.propagate(args.ticker, args.date)
    if final_state is None:
        print("❌ propagate 返回空状态", file=sys.stderr)
        return 2

    # 落盘完整报告树（与 CLI 一致）
    report_path = ta.save_reports(final_state, args.ticker)
    print(f"\n报告已写入：{report_path}", flush=True)

    # dump 全部关键节到 JSON，供汇总
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    dump = {
        "ticker": args.ticker,
        "date": args.date,
        "rating": rating,
        "market_report": final_state.get("market_report", ""),
        "sentiment_report": final_state.get("sentiment_report", ""),
        "news_report": final_state.get("news_report", ""),
        "fundamentals_report": final_state.get("fundamentals_report", ""),
        "investment_plan": final_state.get("investment_plan", ""),
        "trader_investment_plan": final_state.get("trader_investment_plan", ""),
        "final_trade_decision": final_state.get("final_trade_decision", ""),
        "investment_debate_state": final_state.get("investment_debate_state", {}),
        "risk_debate_state": final_state.get("risk_debate_state", {}),
        "final_portfolio_decision": final_state.get("final_portfolio_decision", ""),
    }
    dump_path = out_dir / f"{args.ticker}_{args.date}.json"
    dump_path.write_text(json.dumps(dump, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"结构化 dump：{dump_path}", flush=True)
    print(f"\n评级: {rating}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
