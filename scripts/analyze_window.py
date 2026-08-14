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
import sys

# Windows 控制台默认 GBK，打印 ❌/✅ 会触发 UnicodeEncodeError；强制 utf-8。
# 这对 except 分支里打印 ❌ 尤其关键——否则 UnicodeEncodeError 会覆盖真实异常。
for _stream in (sys.stdout, sys.stderr):
    _reconfigure = getattr(_stream, "reconfigure", None)
    if callable(_reconfigure):
        with __import__("contextlib").suppress(AttributeError, ValueError):
            _reconfigure(encoding="utf-8", errors="replace")

# noqa: E402 — imports sit after the UTF-8 reconfigure guard above (Windows GBK shim).
from pathlib import Path  # noqa: E402

from yiagents.default_config import DEFAULT_CONFIG  # noqa: E402
from yiagents.graph.overlay_fields import parse_overlay  # noqa: E402
from yiagents.graph.trading_graph import YiAgentsGraph  # noqa: E402


def run_one(ticker: str, date: str, equity: float, risk: bool) -> dict:
    cfg = DEFAULT_CONFIG.copy()
    cfg["risk_enabled"] = risk
    ta = YiAgentsGraph(debug=False, config=cfg)
    portfolio_state = {"cash": equity, "equity": equity, "positions": {},
                       "sectors": {}, "returns_history": [], "trade_history": []}
    final_state, rating = ta.propagate(ticker, date, portfolio_state=portfolio_state)
    decision = (final_state or {}).get("final_trade_decision", "")
    overlay = parse_overlay(decision) or {}
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
