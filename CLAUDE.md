# YiAgents

AI 多智能体量化投资框架（前身 TradingAgents v0.3.0 fork，已于 2026-06-27 彻底重命名为 YiAgents）。

- **GitHub**：https://github.com/zhang12120113-creator/Yiagents （SSH push）
- **环境**：Windows 11 + git-bash，DeepSeek API

## 命名约定（已统一）

| 项 | 值 |
|---|---|
| 展示名 | YiAgents |
| 包名 / import / CLI | `yiagents` |
| 环境变量前缀 | `YIAGENTS_*` |
| 用户数据目录 | `~/.yiagents/` |
| 主类 | `YiAgentsGraph` |

前身 TradingAgents 的学术引用（arXiv:2412.20138 / bibtex / CHANGELOG 历史条目）保留原名。

## 叙事红线（硬性要求）

在 README、文档、对外描述中，YiAgents 一律按**基于 99 篇 LLM 金融 / 多智能体 / 风控文献独立设计**的框架呈现：

- **禁止**提到 TradingAgents。
- **禁止**用「fork / 深度定制 / 上游」等暗示衍生的措辞。
- 文献支撑统一指向 `REFERENCES.md`（已剔除 TradingAgents 条目）。

## 模型偏好

DeepSeek 的两个思考引擎（quick/deep）**一律用 `deepseek-v4-pro`**，不要 `deepseek-v4-flash`。`capabilities.py` 里仍登记 flash 以兼容手输，不要误删。

## 铁律

**并发 / 性能层不得改动任何 agent 的输入 / 能力 / 深度**——并发层不引入新随机性，串行 vs 并发的分布必须一致。

## 运行规范

- **必须先 `cd` 到项目根再跑**：`yiagents/__init__.py` 用 `load_dotenv(usecwd=True)` 注入 `HTTP_PROXY`/`NO_PROXY`/`DEEPSEEK_API_KEY`；在别处裸跑会 DNS 解析失败 / 缺 key。
- **跑 ≥1 个 ticker 一律优先 `scripts/run_robust.py`**（per-ticker 独立子进程 + 看门狗 + OS 级强杀重跑），别裸跑 in-process batch——VPN / DeepSeek mid-response 卡死在进程内不可恢复。
  ```bash
  python scripts/run_robust.py --tickers SNDK INTC --date 2026-07-01 \
    --workers 2 --per-ticker-timeout 1800 --max-attempts 3
  ```
- **单 ticker 墙钟 ~20 分钟**（deepseek-v4-pro 推理模型），规划按 ≥20 min/ticker 估。
- **DeepSeek 直连偶发 APIConnectionError / DNS 瞬时失败**属正常抖动：对失败 ticker 单独 `python scripts/run_batch.py --tickers <T> --date <D>` 重跑即可恢复。
- 报告产物：`~/.yiagents/logs/reports/<TICKER>_<ts>/`；逐 ticker 完成态看 `~/.yiagents/logs/<TICKER>/YiAgentsStrategy_logs/full_states_log_<date>.json`（完成才落盘，可作进度信号）。

## 环境（已验证可用，别再重复诊断）

- DeepSeek key 有效，**走直连**：`.env` 里 `NO_PROXY=api.deepseek.com`。可用模型 `deepseek-v4-pro` / `deepseek-v4-flash`。
- yfinance / 行情 / 财报必须经 `socks5h://127.0.0.1:1080`（SOCKS5 代理，FLASH-CAT VPN）。代理一断 → 数据抓取永久挂起。
- **Windows 控制台是 GBK(cp936)**，打印 ✅/❌ 会 `UnicodeEncodeError`；入口脚本顶部须 `sys.stdout.reconfigure(utf-8)`，兜底用 `PYTHONUTF8=1`。
- Reddit RSS 429、`FRED_API_KEY not set` 是**非致命降级**，不影响评级输出，不用装 FRED key。

## 验证路线（`scripts/run_baseline.py`）

1. `--preflight`（零 LLM 成本自检）
2. `--smoke`（1 票 1 日，跑通整条 LLM 图）
3. `--baseline`（基线回测）
4. `--full`（基线 vs 风控 A/B + 闸门判定）——闸门 PASS 才做券商适配
