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
| `YIAGENTS_LLM_CACHE` | false | per-call LLM 响应磁盘缓存（langchain 全局 `set_llm_cache` + `llm_clients/response_cache.py` `DiskLLMCache`）：相同 (model+prompt+temperature+绑定 tools/结构化 schema) 回放缓存的 `ChatGeneration` 而非重调模型 | 默认关=无缓存无 I/O（字节等价）；迭代重跑同一 smoke/单次分析省中间 ~11 个 agent 调用计费；产物 `~/.yiagents/cache/llm_responses/`。**勿与 `run_analyst_parallel_ab.py` / `run_baseline --full` DSR 同用**——会压扁温度>0 多 run 分布；回测整图重跑已由 `backtest/cache.py` DecisionCache 覆盖 |
| `YIAGENTS_NODE_PERF_TELEMETRY` | false | 节点级墙钟 + token 遥测（包装每个节点 handler） | 产物 `node_perf_<date>.json`（紧邻 `full_states_log`）；`run_baseline --profile` 一键开。注：`ToolNode` 不包装（Runnable 非 plain callable） |
| `YIAGENTS_ANALYST_PARALLEL` | false | 4 分析师并行（单一 `Analyst Fanout` 节点内跑 4 个独立子图） | **必须先过 A/B gate 才翻默认**；与 `llm_rate_limiter` 强交互（rpm<120 实际仅 ~1.5×） |
| `YIAGENTS_ANALYST_PARALLEL_MAX_THREADS` | 16 | 嵌套并发上限；`batch_workers*4` 超限则静默回退串行 | |
| `YIAGENTS_BINANCE_PROACTIVE_BACKOFF` | false | Binance perp vendor 读 `X-MBX-USED-WEIGHT-1M` 头主动退避 | 仅改"何时发请求"不改数据；线程安全（BatchRunner 并发）；反应式 429/418 仍是兜底。配 `YIAGENTS_BINANCE_WEIGHT_THRESHOLD`（fapi 默认 2400/min） |
| `YIAGENTS_BINANCE_SPOT_MIRROR` | false | crypto_spot 现货行情走免 key 镜像 `data-api.binance.vision`（默认 `api.binance.com`） | 仅改现货 host，不改数据；现货为新代码、无既有输出可扰；现货限流独立预算 `get_binance_weight_limiter("spot")` |
| `YIAGENTS_BINANCE_HTTP_KEEPALIVE` | false | 进程级共享 `requests.Session`，跨调用复用 TLS/SOCKS5 连接（仿 LLM 客户端 `YIAGENTS_HTTP_KEEPALIVE`） | 仅传输层（同 URL/params/头 → 同响应字节）；`dataflows/binance_http.py` 单例；urllib3 PoolManager 并发只读线程安全 |
| `YIAGENTS_BINANCE_HTTP_RETRIES` | 0 | 瞬时传输错误（DNS/超时/TLS）+ 5xx 指数退避重试（仿 `yf_retry`，base 2s） | `0`=原样抛（字节等价）；耗尽→`NoMarketDataError`（router 降级）；**不**重试 429/418（仍走反应式 `VendorRateLimitError`） |
| `YIAGENTS_BINANCE_HONOR_RETRY_AFTER` | false | 429/418 读 `Retry-After` 头，≤60s 则 sleep 后再抛（让 IP 禁令窗口过期） | 默认关=立即抛（字节等价）；>60s 的长禁令不内联 sleep，交给 `run_robust` 重跑 |
| `YIAGENTS_SEC_OWNERSHIP` | false | 新 optional category `sec_ownership`：Form4 内幕交易 + FTD（fail-to-deliver）两工具暴露给 fundamentals 分析师 | 默认关=工具列表/prompt/能力字节等价（同 `valuation_tools` 契约）；复用 `sec_edgar.py` 基建（CIK/`_sec_get`/缓存/限流，**不改 sec_edgar 一行**）；美股专属（Form4 需 CIK，非美股→`NO_DATA_AVAILABLE`；FTD 按 ticker，非美股自然无行）；PIT（Form4 `filingDate` / FTD `cutoff+发布滞后`）；新 category 默认配置不引用=未 opt-in 零触达。**13F 拆 B2.1**（EFTS 按 CUSIP 检索 + 逐 holder 大 XML 聚合，重 I/O 偏噪，干净聚合需商业源） |
| `YIAGENTS_FTD_PUB_LAG_DAYS` | 10 | FTD 半月度文件 PIT 发布滞后：`cutoff + lag ≤ curr_date` 才可见 | 默认 10 天保守值（SEC 在 cutoff 后数天才发布）；防回测偷看尚未发布的文件 |

- **`clear_node` 是串行耦合点**：它枚举共享 `state["messages"]` 全部 ID 删消息，仅串行安全 → 分析师并行用「每分析师独立子图」结构规避（父图共享 messages 永不并发写）。已逐一读码确认：下游节点（Bull / Bear / Research Manager / Trader / 3 风控辩手 / Portfolio Manager）**均不读 `messages`**，只读各自 `*_report` + debate state，故父图 messages 残留不影响任何下游决策。
- **遥测一键**：`python scripts/run_baseline.py --smoke --profile --ticker <T> --date <D>` 跑完打印「节点→墙钟占比 + token」表，定位真实瓶颈。
- **A/B gate（用户自跑，不在默认流程）**：`python scripts/run_analyst_parallel_ab.py --tickers <T> --date <D> --n 10`；指标=①评级卡方 p>0.05 ②4 份 `*_report` TF-IDF 余弦 between≥within ③同评级风控 overlay 数值字节一致 ④分析师段墙钟 ≥2.5×。**全过才翻 `YIAGENTS_ANALYST_PARALLEL=true`**。先 `--dry-run`（零 LLM 成本）验证脚本本身。

## 数据正确性层（默认开 = 修正既有 lookahead，非字节等价）

与上面「性能 / 遥测层（默认关 = 字节等价）」不同，以下是**回测正确性 bugfix**：默认开、且故意改变回测输出——修正前的输出含未来信息。均落 dataflows / backtest 层，不触任何 agent 输入，不引入新随机性，串/并发分布一致。

- **`YIAGENTS_FUNDAMENTALS_FILING_LAG_DAYS`（默认 45）**：基本面三表（BS/CF/IS）按 `fiscalDateEnding + 公告滞后 ≤ curr_date` 过滤，而非按报告期末。修正「Q3 报表 9/30 结束、10/1 即被回测可见」的 lookahead（10-Q 实际要 ~40 天后才公告）；45 天对齐 SEC large accelerated filer 节奏（10-K≈60d / 10-Q≈40d）。设 0 回退旧行为。实现 `dataflows/utils.py:is_filing_public`。价格层早有 PIT（`stockstats_utils.py` cutoff + stale guard），此为基本面层补齐。
- **overview 快照（无开关，降级）**：yfinance `.info` / AV `OVERVIEW` 是今天单点值、无日期维度，`curr_date < 今天` 时直接抛 `NoMarketDataError` → router 转 `NO_DATA_AVAILABLE` sentinel（fundamentals 分析师 grounding rule 接住，如实写「data not available」）。降级而非过滤——`.info` 无历史维度可供按日选取；若需恢复 PE/marketCap/EPS/beta 须单独立项从三表 + 历史价重构（L 工作量）。回测中 fundamentals 分析师因此失去 overview、改用三表（已正确 PIT 过滤）。
- **DSR 闸门（无开关，bugfix）**：`run_backtest` 现把 `n_trials` 透传给 `compute_metrics`（默认 1 = 字节等价）；`run_baseline --full` 设 `n_trials=2`（baseline + risk-improved 两个独立配置；`runs` 是同配置重复测量，不计入多重检验），让 `validation_gate` 的 DSR hurdle 真正抬起（此前 `engine.py` 硬编码 1 → `_expected_max_z` 返回 0 → hurdle 坍缩 → gate 近乎恒真）。
- **文献支撑**：PIT lookahead / 过拟合校验见 `REFERENCES.md` #59（look-ahead bias）/ #60 CPCV / #61 DSR（均 López de Prado 系），无需新增条目。
- **回测统计归因（opt-in，默认关 = 字节等价）**：`run_backtest` 两个 advisory 后处理，均纯 post-processing（不触 agent、不改 equity 衍生指标、fail-open）：
  - `factor_model`（参数默认 `None` = 跳过）：Fama-French 归因，`"3"`(Mkt-RF/SMB/HML) / `"5"`(+RMW/CMA)，从 French 数据源拉因子矩阵回归策略日收益，填 `metrics.factor_alpha/factor_betas/factor_r_squared`。`run_baseline --baseline/--full` 硬编码 `factor_model="3"`（已是生产路径）；编程式调用默认 None = 字节等价。实现 `backtest/engine.py:_attribute_factors` + `backtest/factor_model.py`。
  - `event_study`（参数默认 `False` = 跳过）：市场模型事件研究，对每个决策日用决策前 250 天估计窗拟合 `R_asset = a + b*R_benchmark`，检验持仓窗 CAR 是否显著非零（mean CAR + Brown&Warner cross-sectional t + bootstrap 95% CI），填 `metrics.event_study_*`。是对朴素 `alpha_vs_index`（无 β 控制、无显著性）的统计强化。需决策日前 ~250 天数据，故 opt-in 时额外 wide-window 重拉 asset+benchmark（`first_event - 400d`）。实现 `backtest/engine.py:_run_event_study` + `backtest/event_study.py`；report opt-in 渲染段。
  - 二者 fail-open：因子文件缺失 / 估计窗不足 / benchmark 拉不到 → 字段留默认 None，不中断回测。

## Binance 资产类型（crypto_perp / crypto_spot）

两条 Binance 分析轨道，均为**只读公共行情、无鉴权、无下单**（Track A，分析专用）。手写 `requests`（**非官方 SDK**），复用已验证的 SOCKS5 代理（`_proxies()`）+ 产品线独立限流（`get_binance_weight_limiter("fapi"|"spot")`）+ 反应式 429/418 兜底 + 可选类目降级。

| 资产类型 | `--asset-type` | 数据源 | 分析师绑定工具 | 备注 |
|---|---|---|---|---|
| `crypto_perp` | `crypto_perp` | Binance USDT-M 永续（`fapi.binance.com`） | 6 个 perp 原生工具：klines/funding/OI/long_short_ratio/taker_buy_sell/basis | 隐藏 Yahoo 工具（符号会解析到错误 spot 对）；无 RSI/MACD 指标（直读 klines） |
| `crypto_spot` | `crypto_spot` | Binance 现货（默认 `api.binance.com`，镜像可切） | 5 工具：spot_klines/ticker24/**spot_perp_basis** + indicators + verified_snapshot | 现货无 funding/OI/杠杆；保留 indicators（符号解析正确）；**spot_perp_basis 是全新 alpha 维度**（跨 venue 基差 = 永续收盘 − 现货收盘） |

**字节等价**：两模式均为 asset_type 新分支；泛化的 `_http_get`/`_paginate_history` 用默认参数（`base=_FAPI_BASE, weight_key="fapi"`），perp/stock/crypto 路径字节不变（`tests/test_crypto_perp_mode.py` 零修改全绿为证）。

**Track B（执行 / 实时轨道，未实施）**：官方模块化 SDK `binance-sdk-derivatives-trading-usds-futures` + `binance-sdk-spot` 是实时 WS 行流 + 下单 + user-data stream 的预设基础（`dataflows/binance.py` docstring 已留 "Track B (execution) will bring the official SDK"）。**当前不迁 SDK**——对分析层 REST 公共行情无净收益，且 SDK 未文档化支持 `socks5h://`（需额外 `requests[socks]`/`aiohttp-socks`）。执行轨道单独启动时再引入。离线 SDK 源码副本 + Spot API 文档存于仓库外 `D:\edge download\binance-connector-python-master\` 与 `D:\edge download\binance-spot-api-docs-master\`（2026-07-08 二次确认：SOCKS5h 零支持、不接收外部 `requests.Session`、5xx 重试有隐性 bug、依赖重——迁分析层净负，结论不变）。

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
