<p align="center">
  <img src="assets/yiagents-logo.svg" alt="YiAgents" width="480">
</p>

<h1 align="center">YiAgents</h1>

<p align="center">
  面向研究的 <b>多智能体 LLM 量化交易框架</b>
</p>

<p align="center">
  <a href="README.md">English</a> &nbsp;|&nbsp; <a href="README.zh-CN.md">中文</a>
</p>

<p align="center">
  <img alt="version" src="https://img.shields.io/badge/version-0.3.0-blue">
  <img alt="python" src="https://img.shields.io/badge/python-3.10%2B-blue">
  <img alt="status" src="https://img.shields.io/badge/status-Research%20Only-orange">
  <img alt="license" src="https://img.shields.io/badge/license-MIT-green">
</p>

> 设计方法学源自 **99 篇** LLM 金融推理 / 多智能体 / 量化风控 / 回测严谨性 / 对抗安全文献（见 [研究基础与文献支撑](#研究基础与文献支撑)，完整文献见 [REFERENCES.md](REFERENCES.md)）。
>
> 包名 / import / CLI 命令统一为 `yiagents`，环境变量前缀 `YIAGENTS_*`，数据目录 `~/.yiagents/`。

---

## 这是什么

YiAgents 用一组分工明确的 **LLM 智能体**模拟真实交易公司的运作：基本面 / 情绪 / 新闻 / 技术分析师产出观点，多空研究员结构化辩论，交易员给出提案，风控团队与组合经理做最终裁决。在此之上，框架叠加一层**确定性量化风控**（Kelly 仓位 / ATR 止损 / 熔断 / CVaR）和一套**四档验证 + 回测闸门**流程，把"研究玩具"往"可上线工程"推。

> ⚠️ **仅用于研究。** 交易表现受模型、温度、数据质量、调仓周期等诸多非确定性因素影响，**不构成任何金融、投资或交易建议**。

---

## 能分析什么

给 YiAgents 一个 **ticker + 日期**，它会从四个维度分析，经过多轮辩论与风控裁决，输出带评级、仓位、止损的结构化交易决策。

**支持的资产**（Yahoo Finance 覆盖范围，用交易所后缀的 ticker；公司身份与 alpha 基准自动按市场解析）：

| 市场 | 示例 ticker |
| ------ | ------ |
| 美股 | `AAPL`、`SPY` |
| A 股 | `600519.SS`（贵州茅台）、`000001.SZ` |
| 港股 / 东京 / 伦敦 | `0700.HK` / `7203.T` / `AZN.L` |
| 印度 / 加拿大 / 澳洲 | `RELIANCE.NS` / `SHOP.TO` / `BHP.AX` |
| 加密货币 | `BTC-USD`、`ETH-USD` |

**四个分析维度**（分析师团队，可勾选）：

| 分析师 | 维度 | 数据来源 |
| ------ | ------ | ------ |
| Market Analyst | 技术面：从指标库（MACD / RSI / 布林带 / ATR / VWMA / SMA / EMA 等）按市况选最多 8 个互补指标 | yfinance / Alpha Vantage |
| Sentiment Analyst | 社交情绪 | Reddit、StockTwits |
| News Analyst | 个股新闻 + 宏观/全球新闻（美联储、地缘、央行政策等） | yfinance / Alpha Vantage News |
| Fundamentals Analyst | 财务基本面 | yfinance / Alpha Vantage |

宏观数据走 FRED（美联储），事件概率走 Polymarket（预测市场），另类数据可走浏览器采集。

### 加密货币分析模式

加密货币有三种可选模式。`crypto`（Yahoo 现货）为默认；`crypto_spot` 与 `crypto_perp` 是**只读公共行情、无鉴权、无下单**的 Binance 分析轨道（Track A，分析专用）。

| 模式 | `--asset-type` | 数据源 | 绑定工具 | 备注 |
| ------ | ------ | ------ | ------ | ------ |
| Yahoo 现货（默认） | `crypto` | Yahoo Finance（`BTC-USD`） | indicators + verified_snapshot | 默认加密路径；无 funding / OI / 杠杆 |
| Binance 现货 | `crypto_spot` | Binance 现货（`api.binance.com`，可切镜像） | spot_klines / ticker24 / **spot_perp_basis** + indicators + verified_snapshot | 全新跨 venue 基差 = 永续收盘 − 现货收盘（新 alpha 维度） |
| Binance 永续 | `crypto_perp` | Binance USDT-M 永续（`fapi.binance.com`） | 6 个原生工具：klines / funding / open_interest / long_short_ratio / taker_buy_sell / basis | 无 RSI/MACD（直读 klines）；隐藏 Yahoo 工具避免符号解析到错误现货对 |

加密模式下 Fundamentals Analyst 会自动剔除（永续 / 现货对无基本面）。所有 Binance 请求为手写 `requests`（**非官方 SDK**），复用已验证的 SOCKS5 代理，按产品线独立限流，并保留反应式 429/418 兜底。

---

## 架构

```text
数据层 (yfinance / Alpha Vantage / FRED / Polymarket / Reddit / StockTwits / Binance)
        │
   ┌────▼───── Analyst Team（串行，或开并行的标志位）──────────┐
   │  Fundamentals · Sentiment · News · Technical              │
   └───────────────────────────────────────────────────────────┘
        │
   ┌────▼───── Researcher Team ────────────────────────────────┐
   │  Bull ⇄ Bear 多轮结构化辩论（max_debate_rounds）           │
   └───────────────────────────────────────────────────────────┘
        │
   Research Manager ──► Trader（出提案）
        │
   ┌────▼───── Risk Management 三方辩论 ───────────────────────┐
   │  Aggressive · Neutral · Conservative                      │
   │  （max_risk_discuss_rounds / env YIAGENTS_MAX_RISK_ROUNDS）│
   └───────────────────────────────────────────────────────────┘
        │
   Portfolio Manager（批准 / 否决）
        │
   量化风控叠加（risk_enabled）→ 仓位 / 止损 / 敞口再校准
        │
   决策 + 记忆闭环（写入 ~/.yiagents/memory/）
```

- **评级**：Research Manager / Portfolio Manager 用**五档**（Buy / Overweight / Hold / Underweight / Sell）；Trader 用三档（Buy / Hold / Sell）。
- **数值安全**：精确价格 / 指标取自已校验的市场数据快照，LLM 只定方向，数学定仓位与止损。

---

## 安装

```bash
conda create -n yiagents python=3.12
conda activate yiagents
pip install -e .

# 可选 extras
pip install -e ".[web]"        # 本地 Web UI 的 FastAPI + Uvicorn
pip install -e ".[bedrock]"    # Amazon Bedrock 提供商（AWS SigV4）
pip install -e ".[dev]"        # ruff / pytest
```

Docker：

```bash
cp .env.example .env        # 填入 API key
docker compose run --rm yiagents
```

> `PySocks` 是关键依赖：yfinance 与 Binance vendor 经 SOCKS5 代理访问网络，正是 PySocks 让 `requests` 能透过代理解析主机名，而不是卡在 DNS。

---

## 配置环境

YiAgents 支持多 LLM 提供商，在 `.env` 里设其一即可（**本仓库脚本默认走 DeepSeek**）：

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

**推荐的 DeepSeek 分工** —— 重裁决走 deep 通道，轻量多轮走 quick 通道（单 ticker 墙钟 ~8–10 分钟）：

```bash
YIAGENTS_LLM_PROVIDER=deepseek
YIAGENTS_DEEP_THINK_LLM=deepseek-v4-pro     # Research Manager / Portfolio Manager
YIAGENTS_QUICK_THINK_LLM=deepseek-v4-flash  # 4 分析师 / 辩论 / Trader / 反思
YIAGENTS_OUTPUT_LANGUAGE=Chinese             # 分析师报告与最终决策输出语言
```

**本机代理（重要）：** 若经 SOCKS5 代理访问 Yahoo / Binance / LLM，需安装 PySocks（`pip install "requests[socks]"`）并设 `HTTPS_PROXY`/`HTTP_PROXY`。`preflight` 会自动探测代理端口与 PySocks 依赖。

> 配置项与类型强制的 env 映射见 [yiagents/default_config.py](yiagents/default_config.py)（`_ENV_OVERRIDES`）。拼错的布尔 / 数值会在启动时报错，而非静默用默认值。

---

## CLI 用法

安装后得到 `yiagents` 命令；也可 `python -m cli.main` 从源码运行。

### 单只分析：`yiagents analyze`

```bash
yiagents analyze
```

交互式选择 ticker、分析日期、输出语言、分析师、研究深度、LLM 提供商与模型，结果边算边显示，结束时输出五段式完整报告并询问是否保存。

```bash
yiagents analyze --checkpoint          # 本轮启用检查点（崩溃可续跑）
yiagents analyze --clear-checkpoints   # 运行前清空所有检查点
```

### 批量并发：`yiagents batch`

一次并发分析**同一资产类别**的多只标的，每个标的跑与单票完全相同的完整链路，共享一个 API key。

```bash
# 多只美股，同一日期
yiagents batch -t AAPL -t NVDA -t MSFT -d 2026-06-30

# 多只 A 股
yiagents batch -t 600519.SS -t 000858.SZ -t 601318.SS -d 2026-06-30

# 加密一批（默认 Yahoo 现货）
yiagents batch -t BTC-USD -t ETH-USD -t SOL-USD -d 2026-06-30 -w 3

# Binance 永续分析（Track A，只读公共行情）
yiagents batch -t BTCUSDT -t ETHUSDT -d 2026-06-30 --asset-type crypto_perp

# Binance 现货分析（含跨 venue 现货-永续基差）
yiagents batch -t BTCUSDT -t ETHUSDT -d 2026-06-30 --asset-type crypto_spot
```

| 参数 | 说明 |
| ------ | ------ |
| `-t / --ticker` | 标的代码，重复 `-t` 指定多个；**一个批次只能同一资产类别**（全股票或全加密） |
| `-d / --date` | 分析日期 `YYYY-MM-DD` |
| `--asset-type` | `stock` / `crypto` / `crypto_spot` / `crypto_perp` / `auto`（auto = 按首个 ticker 推断，混类会报错） |
| `-w / --workers` | 并发池大小 K，默认 `YIAGENTS_BATCH_WORKERS=3` |

**并发是安全的**：每个 worker 线程独占一个图实例（无竞态），记忆日志与 OHLCV 缓存用 filelock 序列化，单票失败不连累整批，且每只标的的分析与串行跑**字节等价**——并发层叠在 `propagate()` 之上，不改任何 agent 输入 / 深度 / 推理参数。详见 [yiagents/batch/runner.py](yiagents/batch/runner.py)。

---

## Web UI

一个轻量 FastAPI 应用 + 原生 HTML/JS 前端（[web/](web/)），可在浏览器里浏览历史分析、提交新分析——**不触碰任何 agent / graph / dataflow 代码**。深色"量化终端"质感，带 🌐 中英文 i18n（同时切换界面语言与报告语言）。

```bash
pip install -e ".[web]"          # fastapi + uvicorn（一次性）
python web/app.py                # 启动 http://127.0.0.1:8000
```

> 必须从**项目根**启动（包含 `.env`、`yiagents/`、`scripts/` 的目录）：`yiagents/__init__.py` 用 `load_dotenv(usecwd=True)` 加载 `.env`，在别处启动会令 DeepSeek key 与 SOCKS5 代理缺失，spawn 出来的 `run_robust` 子进程会继承这个坏环境。

- **浏览**：ticker 网格 → 每只的日期 → 完整报告视图（评级徽章 + 量化风控 overlay 卡片 + 5 个可折叠章节 + 可选 node-perf 柱状图）。
- **提交**：表单 spawn `scripts/run_robust.py`（与 CLI 同一条看门狗路径）；前端每 4 秒轮询并链接完成的报告。同一时刻只允许一个分析（运行中返回 409）。
- **API**：`GET /api/tickers`、`GET /api/tickers/{t}/runs[/{date}]`、`POST /api/analyze`、`GET /api/tasks/{id}`、`GET /api/health`。

架构与完整端点说明见 [web/README.md](web/README.md)。

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

调节量化风控叠加层：

```python
config = DEFAULT_CONFIG.copy()
config["risk_enabled"] = True          # 默认开启（生产形态）
config["kelly_fraction"] = 0.25        # 四分之一 Kelly
config["max_single_position"] = 0.20   # 单票 ≤ 20% 净值
config["max_single_sector"] = 0.30     # 单行业 ≤ 30%
config["max_drawdown_hard_stop"] = 0.15# 回撤熔断
config["atr_stop_mult"] = 2.0          # 止损 = 最新收盘 - 2×ATR（多头）

ta = YiAgentsGraph(config=config)
_, decision = ta.propagate("NVDA", "2026-01-15")
```

`ta.save_reports(final_state, ticker)` 可在无头 / API 场景写出与 CLI 相同的报告树。全部配置项见 [yiagents/default_config.py](yiagents/default_config.py)。

---

## 量化风控叠加层

LLM 定方向，数学定仓位与风险（[yiagents/risk/](yiagents/risk/)）：

| 机制 | 文件 | 作用 |
| ------ | ------ | ------ |
| Kelly 仓位 | [kelly.py](yiagents/risk/kelly.py) | 按胜率 / 赔率定最优仓位，可调分数 |
| ATR 止损 | [atr_stop.py](yiagents/risk/atr_stop.py) | 止损 = 收盘 − N×ATR（多头） |
| 回撤熔断 | [breaker.py](yiagents/risk/breaker.py) | 超过最大回撤即平仓冷却 |
| CVaR | [cvar.py](yiagents/risk/cvar.py) | 条件风险价值，尾部风险约束 |
| 总线 | [manager.py](yiagents/risk/manager.py) | 汇聚以上，覆盖单票 / 行业 / 敞口上限 |

`risk_enabled` 默认**开启**（推荐的生产形态）：风控经理确定性地改写仓位 / 止损 / 敞口，LLM 只保留方向。`scripts/run_baseline.py` 按模式显式设置（`--baseline` 关，建立 Phase-0 基线；`--full` 开，做 A/B）。

---

## 回测与验证闸门

**四档验证脚本** [scripts/run_baseline.py](scripts/run_baseline.py)，强烈建议按顺序跑，从零成本自检逐步放大到完整 A/B，避免一上来烧 LLM 配额：

```bash
# 档 -1｜起飞检查：零 LLM 成本，自检 依赖(含PySocks)/env/代理端口/yfinance 实拉/DeepSeek 探活
python scripts/run_baseline.py --preflight --ticker AAPL

# 档  0｜冒烟测试：1 只票 1 个日期，确认 LLM/网络/key 全通（最便宜的完整链路）
python scripts/run_baseline.py --smoke --ticker AAPL --date 2026-03-15

# 档  1｜基线回测：现状系统（LLM 决策 + 简单仓位），出基线报告 + 仪表盘
python scripts/run_baseline.py --baseline --tickers AAPL NVDA

# 档  2｜完整 A/B：基线 vs Phase-1 风控增强 + 闸门 PASS/FAIL 判定
python scripts/run_baseline.py --full --tickers AAPL NVDA --runs 2
```

常用参数：`--tickers`（A股 `600519.SS`）/ `--start --end` / `--step`（调仓间隔，默认 10）/ `--rebalance`（调仓次数，默认 6）/ `--holding-days` / `--cost-bps`（单边成本，默认 5bp）/ `--runs`（LLM 非确定性，每票跑几次取分布）/ `--workers`（跨 ticker 并发）/ `--out`（默认 `./backtest_output`）。

> 每次 `propagate()` = 一次完整 LLM 图（4 分析师 + 辩论 + 交易员 + 风控辩论 + PM）。成本随 `tickers × 日期数 × runs` 线性增长。**先 preflight 全绿，再 smoke，最后放大。**

档 2（`--full`）跑基线 vs 风控增强的 A/B，并对每只票独立判定**验证闸门**（[yiagents/backtest/validation_gate.py](yiagents/backtest/validation_gate.py)）：

- **Deflated Sharpe Ratio（DSR）** —— 对多次抽样做多重检验校正，惩罚过拟合（[metrics.py](yiagents/backtest/metrics.py)）
- **是否跑赢买入持有**、PASS / FAIL 结论、改进建议
- 报告、仪表盘、闸门判定写入 `--out`（默认 `backtest_output/`）

```text
[AAPL] 闸门判定: ✅ PASS | DSR 1.42 | 跑赢B&H True
[MSFT] 闸门判定: ❌ FAIL | DSR -0.31 | 跑赢B&H False
```

---

## 性能与可观测层

并发 / 传输 / 观测层一律不触任何 agent 的输入。以下开关全部**默认关（或等价值）= 与串行基线字节等价**，按需开：

| 开关（env） | 默认 | 作用 |
| ------ | ------ | ------ |
| `YIAGENTS_LLM_TIMEOUT_S` | 120 | 单次 LLM 读超时；半开连接 → `APITimeoutError` → SDK 内置重试恢复 |
| `YIAGENTS_HTTP_KEEPALIVE` | false | 进程级共享 `httpx.Client`，复用 TLS / SOCKS5 连接 |
| `YIAGENTS_LLM_MAX_RETRIES` | 2 | 单调用重试次数（= langchain 默认，等价） |
| `YIAGENTS_NODE_PERF_TELEMETRY` | false | 节点级墙钟 + token 遥测；`--profile` 一键开，产物 `node_perf_<date>.json` |
| `YIAGENTS_STREAM_TELEMETRY` | false | 流式图 + 记录每位分析师墙钟（终态与 invoke 一致） |
| `YIAGENTS_ANALYST_PARALLEL` | false | 4 分析师在单一包装节点内并行（各自独立子图）。**必须先过 A/B gate 才翻默认** |
| `YIAGENTS_LLM_RATE_LIMITER` | false | 可选的共享 RPM 限流器 |
| `YIAGENTS_BINANCE_PROACTIVE_BACKOFF` | false | Binance 永续 vendor 读 `X-MBX-USED-WEIGHT-1M` 头主动退避 |
| `YIAGENTS_BINANCE_SPOT_MIRROR` | false | 现货走免 key 镜像 `data-api.binance.vision` |

遥测一键：`python scripts/run_baseline.py --smoke --profile --ticker <T> --date <D>` 跑完打印「节点→墙钟占比 + token」表，定位真实瓶颈。

---

## 持久化与恢复

**决策日志（默认开启）：** 每次完成的运行把决策追加到 `~/.yiagents/memory/trading_memory.md`。下次同 ticker 运行时，自动拉取实现收益（含相对基准的 alpha）、生成反思，并把最近同 ticker 决策与跨 ticker 教训注入组合经理提示，形成"吃一堑长一智"闭环。路径用 `YIAGENTS_MEMORY_LOG_PATH` 覆盖。

**检查点恢复（opt-in）：** 用 `--checkpoint` 开启，LangGraph 在每个节点后存档，崩溃 / 中断可从最后一个成功步骤续跑，成功完成后自动清理。按 ticker 的 SQLite 库位于 `~/.yiagents/cache/checkpoints/<TICKER>.db`（`YIAGENTS_CACHE_DIR` 覆盖）。

**全局熔断：** `YIAGENTS_KILL_SWITCH=true` 时，浏览器券商执行层拒绝提交任何新订单（[browser_broker.py](yiagents/execution/browser_broker.py)）。

---

## 脚本一览

| 脚本 | 用途 |
| ------ | ------ |
| [scripts/run_baseline.py](scripts/run_baseline.py) | 四档：preflight / smoke / baseline / full |
| [scripts/run_robust.py](scripts/run_robust.py) | 按-ticker 子进程编排 + 看门狗 + OS 级强杀重跑（≥1 ticker 首选） |
| [scripts/run_batch.py](scripts/run_batch.py) | 批量并发分析多只 ticker（等价于 `yiagents batch`） |
| [scripts/run_analyst_parallel_ab.py](scripts/run_analyst_parallel_ab.py) | 验证分析师并行与串行分布等价的 A/B 闸门 |
| [scripts/smoke_structured_output.py](scripts/smoke_structured_output.py) | 针对任意提供商验证三个结构化输出 agent |

---

## 项目结构

```text
.
├── yiagents/            # 核心包（内部包名，对外 import 名）
│   ├── agents/               # analysts / researchers / managers / trader / risk_mgmt
│   ├── dataflows/            # yfinance / Alpha Vantage / FRED / Polymarket / Reddit / StockTwits / Binance / 浏览器数据
│   ├── graph/                # LangGraph 编排：trading_graph / propagation / reflection / signal_processing / perf_telemetry
│   ├── risk/                 # kelly / atr_stop / breaker / cvar / manager（量化风控叠加层）
│   ├── backtest/             # engine / metrics / validation_gate / ic / report / cache
│   ├── batch/                # 多 ticker 并发 runner + filelock
│   ├── execution/            # browser_broker（浏览器券商 + kill switch）
│   ├── monitoring/           # dashboard（HTML 仪表盘）
│   ├── llm_clients/          # 多 LLM 提供商适配 + 限流器 + 共享 httpx
│   ├── default_config.py     # 配置 + env 映射
│   └── reporting.py
├── cli/                      # 交互式 CLI（analyze / batch）
├── web/                      # FastAPI Web UI（原地运行，不打包）
├── scripts/                  # run_baseline / run_robust / run_batch / run_analyst_parallel_ab / …
├── tests/                    # 测试套件——数据/风控/回测/闸门/多提供商/i18n/并发
└── pyproject.toml
```

---

## 可复现性

YiAgents 是 LLM 驱动的，**同一 ticker + 日期的两次运行可能不同** —— 这是语言模型研究的固有特性，不是缺陷。来源：

- **模型采样非确定性**：即使固定温度，提供商也不保证逐字节一致；推理模型内部推理本身就在采样，波动更大。
- **实时数据在变**：新闻 / StockTwits / Reddit 随时间返回不同内容，即便固定历史交易日，社情舆情仍反映"当下"。

降低波动的手段：调低 `temperature`（`YIAGENTS_TEMPERATURE`），或显式选非推理模型。已确定性化的部分：分析公司身份在 agent 运行前由 ticker 解析锁定；市场分析师的精确价格 / 指标取自已校验的数据快照。

回测结果不保证对齐任何已发表数字，请把它当作**研究多智能体分析的脚手架**，而非一条有固定可复制收益的策略。

---

## 研究基础与文献支撑

YiAgents 的每一层关键机制都对应已发表的研究成果，而非凭空设计。下表把 14 个研究方向映射到框架中的落地位置（代表文献仅作示列，完整 99 篇见 [REFERENCES.md](REFERENCES.md)）：

| 研究支柱 | 代表文献 | 在 YiAgents 中的落地 |
| ------ | ------ | ------ |
| 基准与"Alpha 幻觉" | FINSABER (Li 2025) · The Alpha Illusion (Jang 2026) · AlphaQuanter (2025) | [validation_gate.py](yiagents/backtest/validation_gate.py)：DSR + 跑赢买入持有判定 |
| 多智能体决策 | Debate or Vote (NeurIPS 2025) · MA-PoP (2026) · S2-MAD (2025) | Bull/Bear 与 Risk 多轮辩论（`max_debate_rounds` / `max_risk_discuss_rounds`） |
| 推理优化 | FinCoT (2025) · Program-of-Thoughts · Overthinking 早退 (2025) | [fin_cot_prompts](yiagents/default_config.py)：去人格化结构化提示 |
| 记忆与抗遗忘 | FinMem (TBDATA 2025) · Reflexion (NeurIPS 2023) · AlphaAgent (2025) | [memory 闭环](yiagents/graph/)：决策日志 + 反思 + 跨票教训注入 |
| 幻觉与数值验证 | Chain-of-Verification (ICML 2024) · DeBERTa-NLI · HHEM | 数值走解释器 / 校验路径，LLM 只定方向 |
| LLM + 传统量化 | LLM-MAS-DRL (2024) · AlphaCrafter (2024) · FinCon (2024) | LLM 出观点、量化层定仓位 / 止损的混合架构 |
| 市场状态识别 | HMM Regime · 级联控制器 (2024) | 趋势 / 震荡 / 高波动 / 危机状态适配（增强模块，路线图） |
| 风险与仓位 | HRP (Lopez de Prado) · Sentinel/ATR · CVaR 双层 (FinCon) | [risk/](yiagents/risk/)：Kelly + ATR 止损 + 熔断 + CVaR |
| 回测严谨性 | FinCAD (2025) · CPCV · Deflated Sharpe (Lopez de Prado) | [backtest/](yiagents/backtest/)：参数化前视偏差校正 + DSR |
| 对抗鲁棒性 | MemMorph (2025) · SMSR (2025) · Spotlighting (2025) | 工具调用 / 记忆投毒 / 提示注入防护（路线图） |
| 成本工程 | GPTCache · 模型级联 · DAG 编排 (2025) | 多提供商路由 + 检查点续跑 + 四档成本递增验证 |
| 情绪与另类数据 | FinAgent (KDD 2024) · 少样本股票预测 (Deng 2024) | [dataflows/](yiagents/dataflows/)：Reddit / StockTwits / Polymarket / 浏览器 |
| 可解释性 | CFA XAI 报告 (2025) · CoT 可视化 | 结构化报告 + [dashboard](yiagents/monitoring/dashboard.py) + 决策日志 |
| 合规与安全 | EU AI Act · AIBOM (2025) · 零信任架构 | `YIAGENTS_KILL_SWITCH` + 仅研究用途声明 |

> 标注"路线图"的机制已纳入设计、尚未全部落地。

---

## 引用

若本框架对你的工作有帮助，请引用：

```bibtex
@misc{yiagents2026,
      title={YiAgents: A Research-Oriented Multi-Agent LLM Quantitative Trading Framework},
      author={Mark},
      year={2026},
      url={https://github.com/zhang12120113-creator/Yiagents},
}
```

完整文献支撑（99 篇，按 14 个研究方向分类）见 [REFERENCES.md](REFERENCES.md)。

## 免责声明

本项目仅供学术与工程研究使用，**不构成任何金融、投资或交易建议**。金融市场交易存在重大风险，过往表现不代表未来收益。基于本项目的任何决策与损失，作者不承担任何责任。
