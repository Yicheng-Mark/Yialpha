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

DeepSeek 两档分工（2026-07-05 起，原「一律 v4-pro」已废弃——单 ticker 墙钟 ~20min 太慢）：

- **deep 通道（Research Manager / Portfolio Manager）用 `deepseek-v4-pro`** —— 重裁决，推理深度决定质量。
- **quick 通道（4 分析师 / Bull-Bear 辩论 / 风控三方辩论 / Trader / 反思 / 信号提取）用 `deepseek-v4-flash`** —— 轻量多轮，速度优先；墙钟降到 ~8-10min。

切换只改 `.env` 的 `YIAGENTS_QUICK_THINK_LLM`（第 8 行）。`capabilities.py` 仍登记 pro/flash 两模型，不要误删。

## 铁律

**并发 / 性能层不得改动任何 agent 的输入 / 能力 / 深度**——并发层不引入新随机性，串行 vs 并发的分布必须一致。

## 性能 / 遥测层（零影响，全部默认关 = 字节等价）

并发 / 传输 / 观测层一律不触 agent 输入；以下开关全部**默认关（或等价值）= 与今天字节等价**，按需开：

| 开关（env） | 默认 | 作用 | 备注 / 产物 |
|---|---|---|---|
| `YIAGENTS_LLM_TIMEOUT_S` | 120（已设） | 单次 LLM 读超时；半开连接 → `APITimeoutError` → SDK 内置重试恢复 | `openai_client.py` 在线读；消除偶发 30min 卡死 |
| `YIAGENTS_HTTP_KEEPALIVE` | true（已设） | 进程级共享 `httpx.Client`，复用 TLS/SOCKS5 连接 | 仅 OpenAI 兼容 provider；DeepSeek 直连适用 |
| `YIAGENTS_LLM_MAX_RETRIES` | 2（= langchain 默认，等价） | 单调用重试次数；抖动期可调低 | 默认值与历史字节一致；外层靠 `run_robust` 看门狗兜底 |
| `YIAGENTS_NODE_PERF_TELEMETRY` | false | 节点级墙钟 + token 遥测（包装每个节点 handler） | 产物 `node_perf_<date>.json`（紧邻 `full_states_log`）；`run_baseline --profile` 一键开。注：`ToolNode` 不包装（Runnable 非 plain callable） |
| `YIAGENTS_ANALYST_PARALLEL` | false | 4 分析师并行（单一 `Analyst Fanout` 节点内跑 4 个独立子图） | **必须先过 A/B gate 才翻默认**；与 `llm_rate_limiter` 强交互（rpm<120 实际仅 ~1.5×） |
| `YIAGENTS_ANALYST_PARALLEL_MAX_THREADS` | 16 | 嵌套并发上限；`batch_workers*4` 超限则静默回退串行 | |
| `YIAGENTS_BINANCE_PROACTIVE_BACKOFF` | false | Binance perp vendor 读 `X-MBX-USED-WEIGHT-1M` 头主动退避 | 仅改"何时发请求"不改数据；线程安全（BatchRunner 并发）；反应式 429/418 仍是兜底。配 `YIAGENTS_BINANCE_WEIGHT_THRESHOLD`（fapi 默认 2400/min） |

- **`clear_node` 是串行耦合点**：它枚举共享 `state["messages"]` 全部 ID 删消息，仅串行安全 → 分析师并行用「每分析师独立子图」结构规避（父图共享 messages 永不并发写）。已逐一读码确认：下游节点（Bull / Bear / Research Manager / Trader / 3 风控辩手 / Portfolio Manager）**均不读 `messages`**，只读各自 `*_report` + debate state，故父图 messages 残留不影响任何下游决策。
- **遥测一键**：`python scripts/run_baseline.py --smoke --profile --ticker <T> --date <D>` 跑完打印「节点→墙钟占比 + token」表，定位真实瓶颈。
- **A/B gate（用户自跑，不在默认流程）**：`python scripts/run_analyst_parallel_ab.py --tickers <T> --date <D> --n 10`；指标=①评级卡方 p>0.05 ②4 份 `*_report` TF-IDF 余弦 between≥within ③同评级风控 overlay 数值字节一致 ④分析师段墙钟 ≥2.5×。**全过才翻 `YIAGENTS_ANALYST_PARALLEL=true`**。先 `--dry-run`（零 LLM 成本）验证脚本本身。

## 运行规范

- **必须先 `cd` 到项目根再跑**：`yiagents/__init__.py` 用 `load_dotenv(usecwd=True)` 注入 `HTTP_PROXY`/`NO_PROXY`/`DEEPSEEK_API_KEY`；在别处裸跑会 DNS 解析失败 / 缺 key。
- **跑 ≥1 个 ticker 一律优先 `scripts/run_robust.py`**（per-ticker 独立子进程 + 看门狗 + OS 级强杀重跑），别裸跑 in-process batch——VPN / DeepSeek mid-response 卡死在进程内不可恢复。
  ```bash
  python scripts/run_robust.py --tickers SNDK INTC --date 2026-07-01 \
    --workers 2 --per-ticker-timeout 1800 --max-attempts 3
  ```
- **单 ticker 墙钟 ~8-10 分钟**（quick=`deepseek-v4-flash` / deep=`deepseek-v4-pro` 分工后；原「一律 pro」时为 ~20min）。规划按 ≥10 min/ticker 估。
- **DeepSeek 直连偶发 APIConnectionError / DNS 瞬时失败**属正常抖动：对失败 ticker 单独 `python scripts/run_batch.py --tickers <T> --date <D>` 重跑即可恢复。
- 报告产物：`~/.yiagents/logs/reports/<TICKER>_<ts>/`；逐 ticker 完成态看 `~/.yiagents/logs/<TICKER>/YiAgentsStrategy_logs/full_states_log_<date>.json`（完成才落盘，可作进度信号）。

## 环境（已验证可用，别再重复诊断）

- DeepSeek key 有效，**走直连**：`.env` 里 `NO_PROXY=api.deepseek.com`。可用模型 `deepseek-v4-pro` / `deepseek-v4-flash`。
- yfinance / 行情 / 财报必须经 `socks5h://127.0.0.1:1080`（SOCKS5 代理，FLASH-CAT VPN）。代理一断 → 数据抓取永久挂起。
- **Windows 控制台是 GBK(cp936)**，打印 ✅/❌ 会 `UnicodeEncodeError`；入口脚本顶部须 `sys.stdout.reconfigure(utf-8)`，兜底用 `PYTHONUTF8=1`。
- Reddit RSS 429、`FRED_API_KEY not set` 是**非致命降级**，不影响评级输出，不用装 FRED key。

## 验证路线（`scripts/run_baseline.py`）

1. `--preflight`（零 LLM 成本自检）
2. `--smoke`（1 票 1 日，跑通整条 LLM 图）；加 `--profile` 同跑且打印「节点→墙钟占比」遥测表（零影响，见上「性能 / 遥测层」）
3. `--baseline`（基线回测）
4. `--full`（基线 vs 风控 A/B + 闸门判定）——闸门 PASS 才做券商适配

分析师并行的分布等价性 gate 走 `scripts/run_analyst_parallel_ab.py`（独立于上述回测路线，见上）。
