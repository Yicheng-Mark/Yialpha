"""多日窗口决策序列 + 量化风控叠加层。

对一只票在多个交易日各跑一次完整 propagate（每日独立 graph，无跨日记忆），
全程开 risk_enabled，传一个固定 $100k 全现金假设组合，让 Kelly/ATR/CVaR 层
在每日评级+ATR 下给出目标仓位/止损/regime，观察随行情下跌的演化。

用法：
  python scripts/analyze_window.py --ticker NVDA \
      --dates 2026-05-14 2026-05-29 2026-06-10 2026-06-18 2026-06-26 \
      --equity 100000 --risk --out analysis_output
"""
import argparse
import json
import re
import sys
from pathlib import Path

from yiagents.default_config import DEFAULT_CONFIG
from yiagents.graph.trading_graph import YiAgentsGraph

# overlay 段落里要抽取的字段
_FIELDS = {
    "action": r"\*\*Action\*\*:\s*(.+)",
    "target_weight": r"\*\*Target Weight\*\*:\s*([0-9.]+%)",
    "position_value": r"\*\*Target Weight\*\*:.*?\(([-0-9,]+)\)",
    "stop_loss": r"\*\*Stop Loss\*\*:\s*([-0-9.]+)",
    "entry": r"\*\*Entry Reference\*\*:\s*([-0-9.]+)",
    "regime": r"\*\*Drawdown Regime\*\*:\s*(\S+)",
}


def parse_overlay(decision_md: str) -> dict:
    """从 PM 最终决策（含追加的 overlay 段）抽取风控层数值。"""
    out = {}
    for key, pat in _FIELDS.items():
        m = re.search(pat, decision_md)
        if m:
            out[key] = m.group(1).strip()
    return out


def run_one(ticker: str, date: str, equity: float, risk: bool) -> dict:
    cfg = DEFAULT_CONFIG.copy()
    cfg["risk_enabled"] = risk
    ta = YiAgentsGraph(debug=False, config=cfg)
    portfolio_state = {"cash": equity, "equity": equity, "positions": {},
                       "sectors": {}, "returns_history": [], "trade_history": []}
    final_state, rating = ta.propagate(ticker, date, portfolio_state=portfolio_state)
    decision = (final_state or {}).get("final_trade_decision", "")
    overlay = parse_overlay(decision)
    return {
        "date": date, "rating": rating,
        "price": overlay.get("entry"),
        "overlay": overlay,
        "decision_excerpt": decision[:1200],
        "full_decision": decision,
    }


def main() -> int:
    p = argparse.ArgumentParser(description="YiAgents 多日窗口 + 风控序列")
    p.add_argument("--ticker", default="NVDA")
    p.add_argument("--dates", nargs="+", default=[
        "2026-05-14", "2026-05-29", "2026-06-10", "2026-06-18", "2026-06-26"])
    p.add_argument("--equity", type=float, default=100000.0)
    p.add_argument("--risk", action="store_true", default=True)
    p.add_argument("--no-risk", dest="risk", action="store_false")
    p.add_argument("--out", default="analysis_output")
    args = p.parse_args()

    print(f"\n=== 窗口序列：{args.ticker}  risk_enabled={args.risk}  equity=${args.equity:,.0f} ===",
          flush=True)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"window_{args.ticker}.json"
    # 增量落盘：每跑完一天就写一次，最后一天卡住也不丢前面的成果
    rows = []
    for i, date in enumerate(args.dates, 1):
        print(f"\n--- [{i}/{len(args.dates)}] {args.ticker} @ {date} 开始 ---", flush=True)
        try:
            rec = run_one(args.ticker, date, args.equity, args.risk)
        except Exception as exc:  # noqa: BLE001
            print(f"❌ {date} 失败：{exc}", file=sys.stderr, flush=True)
            rec = {"date": date, "rating": "ERROR", "overlay": {}, "decision_excerpt": str(exc)}
        ov = rec.get("overlay", {})
        print(f"   评级={rec['rating']}  价={ov.get('entry','-')}  "
              f"动作={ov.get('action','-')}  目标仓位={ov.get('target_weight','-')}  "
              f"止损={ov.get('stop_loss','-')}  regime={ov.get('regime','-')}", flush=True)
        rows.append(rec)
        path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"   [增量落盘] 已写入第 {i} 天 → {path}", flush=True)

    print(f"\n全部完成，dump：{path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
