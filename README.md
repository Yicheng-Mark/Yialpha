<p align="center">
  <img src="assets/yiagents-logo.svg" alt="YiAgents" width="480">
</p>

# YiAgents

> 基于 [TradingAgents](https://arxiv.org/abs/2412.20138)（论文 *TradingAgents: Multi-Agents LLM Financial Trading Framework*）深度定制的多智能体量化交易框架。
>
> 本项目由 TradingAgents 深度定制并**彻底重命名**为 YiAgents：包名 / import / CLI 命令统一为 `yiagents`，环境变量前缀为 `YIAGENTS_*`，数据目录为 `~/.yiagents/`。学术引用仍指向原工作 TradingAgents（见文末）。

---

## 这是什么

YiAgents 用一组分工明确的 **LLM 智能体**模拟真实交易公司的运作：基本面 / 情绪 / 新闻 / 技术分析师产出观点，多空研究员结构化辩论，交易员给出提案，风控团队与组合经理做最终裁决。在此之上，本 fork 叠加了一层**确定性量化风控**（Kelly 仓位 / ATR 止损 / 熔断 / CVaR）和一套**四档验证 + 回测闸门**流程，把"研究玩具"往"可上线工程"推。

> ⚠️ **仅用于研究。** 交易表现受模型、温度、数据质量、调仓周期等诸多非确定性因素影响，不构成任何金融、投资或交易建议。

---

## 相对上游的定制

| 模块 | 位置 | 说明 |
| ------ | ------ | ------ |
| **四档执行脚本** | [scripts/run_baseline.py](scripts/run_baseline.py) | preflight（零成本自检）→ smoke（冒烟）→ baseline（基线回测）→ full（A/B + 闸门） |
| **量化风控叠加层** | [yiagents/risk/](yiagents/risk/) | `kelly.py` / `atr_stop.py` / `breaker.py` / `cvar.py` / `manager.py`，LLM 定方向、数学定仓位与止损 |
| **回测引擎与闸门** | [yiagents/backtest/](yiagents/backtest/) | `engine.py` / `metrics.py`（含 Deflated Sharpe）/ `validation_gate.py`（PASS/FAIL 判定）/ `ic.py`（信息系数） |
| **监控仪表盘** | [yiagents/monitoring/dashboard.py](yiagents/monitoring/dashboard.py) | HTML 仪表盘，浏览器直开 |
| **浏览器券商执行层** | [yiagents/execution/browser_broker.py](yiagents/execution/browser_broker.py) | 桩实现 + 全局 kill switch（`YIAGENTS_KILL_SWITCH`） |
| **多供应商数据路由** | [yiagents/dataflows/](yiagents/dataflows/) | yfinance / Alpha Vantage / FRED / Polymarket / Reddit / StockTwits / 浏览器另类数据 |
| **FinCoT 结构化提示词** | [yiagents/default_config.py](yiagents/default_config.py) | `fin_cot_prompts`：去人格化的"任务→推理步骤→输出约束"提示，默认关 |

> 上游已有的多 LLM 提供商、LangGraph 编排、多语言、记忆闭环、检查点恢复等能力全部保留。

---

## 架构

```text
数据层 (yfinance/AV/FRED/Polymarket)
        │
   ┌────▼───── Analyst Team（串行）────────────────┐
   │  Fundamentals · Sentiment · News · Technical  │
   └───────────────────────────────────────────────┘
        │
   ┌────▼───── Researcher Team ────────────────────┐
   │  Bull ⇄ Bear 多轮结构化辩论（max_debate_rounds）│
   └───────────────────────────────────────────────┘
        │
   Research Manager ──► Trader（出提案）
        │
   ┌────▼───── Risk Management 三方辩论 ───────────┐
   │  Risk ← Trader / Researcher（max_risk_rounds）│
   └───────────────────────────────────────────────┘
        │
   Portfolio Manager（批准/否决）
        │
   [可选] 量化风控叠加（risk_enabled）→ 仓位/止损/敞口再校准
        │
   决策 + 记忆闭环（写入 ~/.yiagents/memory/）
```

15 个技术指标 · 五档评级（强买到强卖）· 结构化输出。

---

## 快速开始（四档验证流程）

**强烈建议按顺序跑这四档**，从零成本自检逐步放大到完整 A/B，避免一上来就烧 LLM 配额。

```bash
# 档 -1｜起飞检查：零 LLM 成本，自检 依赖(含PySocks)/env/代理端口/yfinance实拉/DeepSeek探活
python scripts/run_baseline.py --preflight --ticker AAPL

# 档 0｜冒烟测试：1 只票 1 个日期，确认 LLM/网络/key 全通（最便宜的完整链路）
python scripts/run_baseline.py --smoke --ticker AAPL --date 2026-03-15

# 档 1｜基线回测：现状系统（LLM 决策 + 简单仓位），出基线报告 + 仪表盘
python scripts/run_baseline.py --baseline --tickers AAPL NVDA

# 档 2｜完整 A/B：基线 vs Phase-1 风控增强 + 闸门 PASS/FAIL 判定
python scripts/run_baseline.py --full --tickers AAPL NVDA --runs 2
```

常用参数：`--tickers`（A股 `600519.SS`）/ `--start --end` / `--step`（调仓间隔，默认 10）/ `--rebalance`（调仓次数，默认 6）/ `--holding-days` / `--cost-bps`（单边成本，默认 5bp）/ `--runs`（LLM 非确定性，每票跑几次取分布）/ `--out`（默认 `./backtest_output`）。

> 每次 `propagate()` = 一次完整 LLM 图（4 分析师 + 辩论 + 交易员 + 风控辩论 + PM）。成本随 `tickers × 日期数 × runs` 线性增长。**先 preflight 全绿，再 smoke，最后放大。**

---

## 安装

```bash
conda create -n yiagents python=3.12
conda activate yiagents
pip install .
```

Docker：

```bash
cp .env.example .env        # 填入 API key
docker compose run --rm yiagents
```

## 配置环境

YiAgents 支持多 LLM 提供商，在 `.env` 里设其一即可（本仓库脚本默认走 **DeepSeek**）：

```bash
DEEPSEEK_API_KEY=...            # DeepSeek（脚本默认）
OPENAI_API_KEY=...              # OpenAI（GPT）
ANTHROPIC_API_KEY=...           # Anthropic（Claude）
GOOGLE_API_KEY=...              # Google（Gemini）
XAI_API_KEY=...                 # xAI（Grok）
DASHSCOPE_API_KEY=...           # 通义千问（国际）
DASHSCOPE_CN_API_KEY=...        # 通义千问（国内）
ZHIPU_API_KEY=...               # GLM（Z.AI 国际）
ZHIPU_CN_API_KEY=...            # GLM（BigModel 国内）
MINIMAX_API_KEY=...             # MiniMax（全球）
ALPHA_VANTAGE_API_KEY=...       # Alpha Vantage
FRED_API_KEY=...                # 美联储宏观数据
```

**本机代理（重要）：** 若经 SOCKS5 代理访问 Yahoo/LLM，需安装 PySocks（`pip install "requests[socks]"`）并设 `HTTPS_PROXY`。`preflight` 会自动探测代理端口与 PySocks 依赖。

切换提供商 / 模型无需改代码，全部走环境变量：

```bash
export YIAGENTS_LLM_PROVIDER=deepseek       # openai/google/anthropic/deepseek/groq/ollama/openai_compatible
export YIAGENTS_DEEP_THINK_LLM=deepseek-chat
export YIAGENTS_QUICK_THINK_LLM=deepseek-chat
export YIAGENTS_OUTPUT_LANGUAGE=Chinese     # 分析师报告与最终决策输出语言
```

> 配置项与类型强制的 env 映射见 [yiagents/default_config.py](yiagents/default_config.py)（`_ENV_OVERRIDES`）。拼错的布尔/数值会在启动时报错，而非静默用默认值。

---

## CLI 用法

```bash
yiagents          # 已安装命令
python -m cli.main     # 或直接从源码运行
```

交互界面可选 ticker、分析日期、LLM 提供商、研究深度等，结果边算边显示。

**支持市场**（Yahoo Finance 覆盖范围，用交易所后缀的 ticker；公司身份与 alpha 基准自动按市场解析）：

- 美股：`AAPL`、`SPY`
- 港股：`0700.HK` · 东京：`7203.T` · 伦敦：`AZN.L`
- 印度：`RELIANCE.NS`/`.BO` · 加拿大：`.TO` · 澳洲：`.AX`
- A股：上交所 `.SS`、深交所 `.SZ`（如 `600519.SS` 贵州茅台）
- 加密：`BTC-USD`、`ETH-USD`

---

## Python 调用

```python
from yiagents.graph.trading_graph import YiAgentsGraph
from yiagents.default_config import DEFAULT_CONFIG

# DEFAULT_CONFIG 已自动套用 YIAGENTS_* 环境变量覆盖
ta = YiAgentsGraph(debug=True, config=DEFAULT_CONFIG.copy())

_, decision = ta.propagate("NVDA", "2026-01-15")
print(decision)
```

启用量化风控叠加层：

```python
config = DEFAULT_CONFIG.copy()
config["risk_enabled"] = True          # 开启确定性风控（默认关，保持基线可复现）
config["kelly_fraction"] = 0.25        # 四分之一 Kelly
config["max_single_position"] = 0.20   # 单票 ≤ 20% 净值
config["max_single_sector"] = 0.30     # 单行业 ≤ 30%
config["max_drawdown_hard_stop"] = 0.15# 回撤熔断
config["atr_stop_mult"] = 2.0          # 止损 = 最新收盘 - 2×ATR（多头）

ta = YiAgentsGraph(config=config)
_, decision = ta.propagate("NVDA", "2026-01-15")
```

全部配置项见 [yiagents/default_config.py](yiagents/default_config.py)。

---

## 回测与闸门

档 2（`--full`）会跑基线 vs 风控增强的 A/B，并对每只票独立判定**验证闸门**：

- **Deflated Sharpe Ratio（DSR）** —— 对多次抽样做多重检验校正，惩罚过拟合
- **是否跑赢买入持有**、PASS/FAIL 结论、改进建议
- 报告、仪表盘、闸门判定写入 `--out`（默认 `backtest_output/`）

```text
[AAPL] 闸门判定: ✅ PASS | DSR 1.42 | 跑赢B&H True
[MSFT] 闸门判定: ❌ FAIL | DSR -0.31 | 跑赢B&H False
```

---

## 持久化与恢复

**决策日志（默认开启）：** 每次完成的运行把决策追加到 `~/.yiagents/memory/trading_memory.md`。下次同 ticker 运行时，自动拉取实现收益（含相对基准的 alpha）、生成反思，并把最近同 ticker 决策与跨 ticker 教训注入组合经理提示，形成"吃一堑长一智"闭环。路径用 `YIAGENTS_MEMORY_LOG_PATH` 覆盖。

**检查点恢复（opt-in）：** 用 `--checkpoint` 开启，LangGraph 在每个节点后存档，崩溃/中断可从最后一个成功步骤续跑，成功完成后自动清理。按 ticker 的 SQLite 库位于 `~/.yiagents/cache/checkpoints/<TICKER>.db`（`YIAGENTS_CACHE_DIR` 覆盖）。

```bash
yiagents analyze --checkpoint           # 本轮启用
yiagents analyze --clear-checkpoints    # 运行前重置
```

---

## 项目结构

```text
.
├── yiagents/            # 核心包（内部包名，对外 import 名）
│   ├── agents/               # analysts / researchers / managers / trader / risk_mgmt
│   ├── dataflows/            # yfinance/Alpha Vantage/FRED/Polymarket/Reddit/StockTwists/浏览器数据
│   ├── graph/                # LangGraph 编排：trading_graph/propagation/reflection/signal_processing
│   ├── risk/                 # kelly/atr_stop/breaker/cvar/manager（量化风控叠加层）
│   ├── backtest/             # engine/metrics/validation_gate/ic/report/cache
│   ├── execution/            # browser_broker（浏览器券商 + kill switch）
│   ├── monitoring/           # dashboard（HTML 仪表盘）
│   ├── llm_clients/          # 多 LLM 提供商适配
│   ├── default_config.py     # 配置 + env 映射
│   └── reporting.py
├── cli/                      # 交互式 CLI
├── scripts/
│   ├── run_baseline.py       # 四档：preflight/smoke/baseline/full
│   └── smoke_structured_output.py
├── tests/                    # 715+ 测试，覆盖数据/风控/回测/闸门/多提供商/i18n
└── pyproject.toml
```

---

## 可复现性

YiAgents 是 LLM 驱动的，**同一 ticker + 日期的两次运行可能不同** —— 这是语言模型研究的固有特性，不是缺陷。来源：

- **模型采样非确定性**：即使固定温度，提供商也不保证逐字节一致；推理模型（默认 GPT-5.x 系）内部推理本身就在采样，波动更大。
- **实时数据在变**：新闻 / StockTwits / Reddit 随时间返回不同内容，即便固定历史交易日，社情舆情仍反映"当下"。

降低波动的手段：调低 `temperature`（`YIAGENTS_TEMPERATURE`），或显式选非推理模型（Custom model ID）。已确定性化的部分：分析公司身份在 agent 运行前由 ticker 解析锁定；市场分析师的精确价格/指标取自已校验的数据快照。

回测结果不保证对齐任何已发表数字，请把它当作**研究多智能体分析的脚手架**，而非一条有固定可复制收益的策略。

---

## 致谢与引用

本框架在上游 TradingAgents 基础上定制。若 YiAgents 对你有帮助，请引用原工作：

```bibtex
@misc{xiao2025tradingagentsmultiagentsllmfinancial,
      title={TradingAgents: Multi-Agents LLM Financial Trading Framework},
      author={Yijia Xiao and Edward Sun and Di Luo and Wei Wang},
      year={2025},
      eprint={2412.20138},
      archivePrefix={arXiv},
      primaryClass={q-fin.TR},
      url={https://arxiv.org/abs/2412.20138},
}
```
