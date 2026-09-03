# YiAlpha

AI 多智能体量化投资框架（前身 TradingAgents v0.3.0 fork，已于 2026-06-27 彻底重命名为 YiAlpha）。

- **GitHub**：https://github.com/Yicheng-Mark/Yialpha
- **环境**：Windows 11 + git-bash，GLM Coding Plan API（BigModel 国内端点）

## 命名约定（已统一）

| 项 | 值 |
|---|---|
| 展示名 | YiAlpha |
| 包名 / import / CLI | `yialpha` |
| 环境变量前缀 | `YIALPHA_*` |
| 用户数据目录 | `~/.yialpha/` |
| 主类 | `YiAlphaGraph` |

前身 TradingAgents 的学术引用（arXiv:2412.20138 / bibtex / CHANGELOG 历史条目）保留原名。

## 叙事红线（硬性要求）

在 README、文档、对外描述中，YiAlpha 一律按**基于 99 篇 LLM 金融 / 多智能体 / 风控文献独立设计**的框架呈现：

- **禁止**提到 TradingAgents。
- **禁止**用「fork / 深度定制 / 上游」等暗示衍生的措辞。
- 文献支撑统一指向 `REFERENCES.md`（已剔除 TradingAgents 条目）。

## 模型偏好

GLM 两档分工（2026-09-03 从 DeepSeek 切到 GLM Coding Plan，BigModel 国内 coding 端点 + 三 key 池轮换；两档哲学不变）：

- **deep 通道（Research Manager / Portfolio Manager）用 `glm-5.3`** —— 重裁决，推理深度决定质量。
- **quick 通道（4 分析师 / Trader / 反思 / 信号提取）+ 辩论层（Bull-Bear / 风控三方辩论，`YIALPHA_DEBATE_LLM`）用 `glm-5.3-flash`** —— 轻量多轮，速度优先；coding plan 按套餐限额（并发/每 5h 提示数），flash 层不占旗舰并发额度。

**Key 池**（`yialpha/llm_clients/key_pool.py`）：`ZHIPU_CN_API_KEYS` 三把 key，**每个请求轮流取下一把活 key**（401/403 该 key 本进程内踢出；429 冷却 `YIALPHA_LLM_KEY_COOLDOWN_S` 秒默认 120 后回归）。coding plan 端点 `ZHIPU_CN_BASE_URL=https://open.bigmodel.cn/api/coding/paas/v4/` 与按量端点不通用。单 key（仅 `ZHIPU_CN_API_KEY`）时池不启用，行为与旧版字节等价。

切回 DeepSeek 需在 `.env` 重配 provider/模型并重新填 key（旧 key 已于 2026-09-03 删除）。`capabilities.py` 仍登记 deepseek pro/flash 两模型，不要误删。

## 铁律

**并发 / 性能层不得改动任何 agent 的输入 / 能力 / 深度**——并发层不引入新随机性，串行 vs 并发的分布必须一致。

## 性能 / 遥测层（零影响，全部默认关 = 字节等价）

并发 / 传输 / 观测层一律不触 agent 输入；以下开关全部**默认关（或等价值）= 与今天字节等价**，按需开：

| 开关（env） | 默认 | 作用 | 备注 / 产物 |
|---|---|---|---|
| `YIALPHA_LLM_TIMEOUT_S` | off（未设；生产建议 120） | 单次 LLM 读超时；半开连接 → `APITimeoutError` → SDK 内置重试恢复 | `openai_client.py` 在线读；消除偶发 30min 卡死。默认 OFF 会在首次调用时发出一次性警告 |
| `YIALPHA_HTTP_KEEPALIVE` | true（已设） | 进程级共享 `httpx.Client`，复用 TLS/SOCKS5 连接 | 仅 OpenAI 兼容 provider；key 池生效时由池的轮换 client 接管（同样是共享 keepalive，见「模型偏好」） |
| `YIALPHA_LLM_MAX_RETRIES` | 2（= langchain 默认，等价） | 单调用重试次数；抖动期可调低 | 默认值与历史字节一致；外层靠 `run_robust` 看门狗兜底 |
| `YIALPHA_LLM_CACHE` | false | per-call LLM 响应磁盘缓存（langchain 全局 `set_llm_cache` + `llm_clients/response_cache.py` `DiskLLMCache`）：相同 (model+prompt+temperature+绑定 tools/结构化 schema) 回放缓存的 `ChatGeneration` 而非重调模型 | 默认关=无缓存无 I/O（字节等价）；迭代重跑同一 smoke/单次分析省中间 ~11 个 agent 调用计费；产物 `~/.yialpha/cache/llm_responses/`。**勿与 `run_analyst_parallel_ab.py` / `run_baseline --full` DSR 同用**——会压扁温度>0 多 run 分布；回测整图重跑已由 `backtest/cache.py` DecisionCache 覆盖。**`run_robust.py` 默认为自己的运行开**（`setdefault` 尊重 env 显式值）：重跑回放已完成节点的 LLM 生成、卡死节点重跑 = hang 恢复语义，省整图重计费；run_robust 是 live 单配置分析、不触达上述分布 caveat；`--no-llm-cache` 关 |
| `YIALPHA_NODE_PERF_TELEMETRY` | false | 节点级墙钟 + token 遥测（包装每个节点 handler） | 产物 `node_perf_<date>.json`（紧邻 `full_states_log`）；`run_baseline --profile` 一键开。注：`ToolNode` 不包装（Runnable 非 plain callable） |
| `YIALPHA_ANALYST_PARALLEL` | false | 4 分析师并行（单一 `Analyst Fanout` 节点内跑 4 个独立子图） | **必须先过 A/B gate 才翻默认**；与 `llm_rate_limiter` 强交互（rpm<120 实际仅 ~1.5×） |
| `YIALPHA_ANALYST_PARALLEL_MAX_THREADS` | 16 | 嵌套并发上限；`batch_workers*4` 超限则静默回退串行 | |
| `YIALPHA_BINANCE_PROACTIVE_BACKOFF` | false | Binance perp vendor 读 `X-MBX-USED-WEIGHT-1M` 头主动退避 | 仅改"何时发请求"不改数据；线程安全（BatchRunner 并发）；反应式 429/418 仍是兜底。配 `YIALPHA_BINANCE_WEIGHT_THRESHOLD`（fapi 默认 2400/min） |
| `YIALPHA_BINANCE_SPOT_MIRROR` | false | crypto_spot 现货行情走免 key 镜像 `data-api.binance.vision`（默认 `api.binance.com`） | 仅改现货 host，不改数据；现货为新代码、无既有输出可扰；现货限流独立预算 `get_binance_weight_limiter("spot")` |
| `YIALPHA_BINANCE_HTTP_KEEPALIVE` | false | 进程级共享 `requests.Session`，跨调用复用 TLS/SOCKS5 连接（仿 LLM 客户端 `YIALPHA_HTTP_KEEPALIVE`） | 仅传输层（同 URL/params/头 → 同响应字节）；`dataflows/binance_http.py` 单例；urllib3 PoolManager 并发只读线程安全 |
| `YIALPHA_BINANCE_HTTP_RETRIES` | 0 | 瞬时传输错误（DNS/超时/TLS）+ 5xx 指数退避重试（仿 `yf_retry`，base 2s） | `0`=原样抛（字节等价）；耗尽→`NoMarketDataError`（router 降级）；**不**重试 429/418（仍走反应式 `VendorRateLimitError`） |
| `YIALPHA_BINANCE_HONOR_RETRY_AFTER` | false | 429/418 读 `Retry-After` 头，≤60s 则 sleep 后再抛（让 IP 禁令窗口过期） | 默认关=立即抛（字节等价）；>60s 的长禁令不内联 sleep，交给 `run_robust` 重跑 |
| `YIALPHA_SEC_OWNERSHIP` | false | optional category `sec_ownership`：Form4 内幕交易 + FTD + 13F 机构持仓三工具暴露给 fundamentals 分析师 | 默认关=工具列表/prompt/能力字节等价（同 `valuation_tools` 契约）；复用 `sec_edgar.py` 基建（CIK/`_sec_get`/`_cached_or_fetch`/`_fetch_company_facts`/缓存/限流，**不改 sec_edgar 一行**）；美股专属（Form4/13F 需 CIK，非美股→`NO_DATA_AVAILABLE`；FTD 按 ticker，非美股自然无行）；PIT（Form4 `filingDate` / FTD `cutoff+发布滞后` / 13F 数据集 `period-end+发布滞后`）；新 category 默认配置不引用=未 opt-in 零触达。13F（B2.1）走 SEC 批量 Form 13F Data Sets（每季一 ZIP，COVER+HOLDING TSV，本地按 CUSIP 反向聚合，1 请求/季）；CUSIP 单源 companyfacts `dei:EntityCusip`（缺失降级）；45 天发布窗口内诚实"未发布"，不做 EFTS |
| `YIALPHA_FTD_PUB_LAG_DAYS` | 10 | FTD 半月度文件 PIT 发布滞后：`cutoff + lag ≤ curr_date` 才可见 | 默认 10 天保守值（SEC 在 cutoff 后数天才发布）；防回测偷看尚未发布的文件 |
| `YIALPHA_SEC_13F_PUB_LAG_DAYS` | 5 | 13F 批量数据集 PIT 发布滞后：`数据集 period-end + lag ≤ curr_date` 才可见 | 默认 5 天保守值（SEC 在窗口结束后数天才发 ZIP）；防回测偷看尚未发布的季度 ZIP；45 天窗口内诚实降级，不做 EFTS |
| `YIALPHA_MARKET_REGIME` | false | 市场动荡指数（FinRL `calculate_turbulence` 单资产降维 = 基准 252 日滚动收益的平方 z）：开时给**保守风控 debater** 的数据源段追加一行市场压力读数 | 默认关=保守 debater prompt 字节等价（同 `sec_ownership`/`valuation_tools` 契约）；市场层**事前**信号，与组合层事后 `DrawdownBreaker` 互补；advisory + fail-soft（取数失败→不加行，不报错）；`yialpha/dataflows/market_regime.py`（MIT 头注明 FinRL 来源），benchmark 经 `_resolve_benchmark` 同款后缀解析（crypto→SPY 代理，可 `benchmark_ticker` 覆盖） |
| `YIALPHA_A_SHARE_NATIVE` | false | 原生 A 股数据源 optional category `a_share_native`：fundamentals 分析师获 `get_a_share_fundamentals_native`（BaoStock 日级 PE/PB/PS/PCF-TTM）+ `get_a_share_ohlc_native`（BaoStock 前复权 OHLCV），news 分析师获 `get_a_share_news_native`（AKShare 东财个股新闻）；补 yfinance 对 A 股覆盖薄弱的真缺口 | 默认关=工具列表/prompt/能力字节等价（同 `a_stock`/`sec_ownership` 契约）；双重门控（flag on **且** `is_a_stock`）；**免费/keyless**（BaoStock TCP socket 直连 `baostock.com:9001` 不读 HTTP_PROXY 无 hang；AKShare/Tushare 走国内 HTTP，env pop/restore 直连绕 SOCKS5，仿 eastmoney `trust_env=False`）；**惰性可选依赖** `pip install "yialpha[a-share]"`（baostock+akshare）/`[a-share-tushare]`，默认关/未装→`NoMarketDataError`→sentinel，零开销；PIT（行 `date <= curr_date`，Tushare 报表 `ann_date`、新闻 `datetime`）；fail-soft；Tushare 是 opt-in 质量档（需 `TUSHARE_TOKEN`，缺→`VendorNotConfiguredError`，router 跳到 keyless vendor）。`yialpha/dataflows/{baostock_vendor,akshare_vendor,tushare_vendor}.py`。CN-fork endpoint 参考，非代码复制 |
| `TUSHARE_TOKEN` | 未设 | Tushare Pro token（opt-in 质量档，裸命名同 `DEEPSEEK_API_KEY`，**无 `YIALPHA_` 前缀**）；只进 `.env`（gitignore） | 缺→`VendorNotConfiguredError`（router 跳到 BaoStock/AKShare keyless vendor，不中断）；走 `api.tushare.pro` 国内 HTTP（env pop/restore 直连）；tier→calls/min + safety margin |
| `YIALPHA_CHECKPOINT_ENABLED` | false | 崩溃可续跑：per-ticker `SqliteSaver`，从上一个成功节点恢复，成功后自动清 checkpoint | 默认关=字节等价；开时 `thread_id` 折入图形状签名（`analysts`/`debate`/`risk`/`asset`/`parallel` 五项），换配置续跑不命中旧图形状的陈旧 checkpoint（#1089）；`run_robust` 走子进程重启不依赖它 |

- **`clear_node` 是串行耦合点**：它枚举共享 `state["messages"]` 全部 ID 删消息，仅串行安全 → 分析师并行用「每分析师独立子图」结构规避（父图共享 messages 永不并发写）。已逐一读码确认：下游节点（Bull / Bear / Research Manager / Trader / 3 风控辩手 / Portfolio Manager）**均不读 `messages`**，只读各自 `*_report` + debate state，故父图 messages 残留不影响任何下游决策。
- **遥测一键**：`python scripts/run_baseline.py --smoke --profile --ticker <T> --date <D>` 跑完打印「节点→墙钟占比 + token」表，定位真实瓶颈。
- **A/B gate（用户自跑，不在默认流程）**：`python scripts/run_analyst_parallel_ab.py --tickers <T> --date <D> --n 10`；指标=①评级卡方 p>0.05 ②4 份 `*_report` TF-IDF 余弦 between≥within ③同评级风控 overlay 数值字节一致 ④分析师段墙钟 ≥2.5×。**全过才翻 `YIALPHA_ANALYST_PARALLEL=true`**。先 `--dry-run`（零 LLM 成本）验证脚本本身。

## 数据正确性层（默认开 = 修正既有 lookahead，非字节等价）

与上面「性能 / 遥测层（默认关 = 字节等价）」不同，以下是**回测正确性 bugfix**：默认开、且故意改变回测输出——修正前的输出含未来信息。均落 dataflows / backtest 层，不触任何 agent 输入，不引入新随机性，串/并发分布一致。

- **`YIALPHA_FUNDAMENTALS_FILING_LAG_DAYS`（默认 45）**：基本面三表（BS/CF/IS）按 `fiscalDateEnding + 公告滞后 ≤ curr_date` 过滤，而非按报告期末。修正「Q3 报表 9/30 结束、10/1 即被回测可见」的 lookahead（10-Q 实际要 ~40 天后才公告）；45 天对齐 SEC large accelerated filer 节奏（10-K≈60d / 10-Q≈40d）。设 0 回退旧行为。实现 `dataflows/utils.py:is_filing_public`。价格层早有 PIT（`stockstats_utils.py` cutoff + stale guard），此为基本面层补齐。
- **overview 快照（无开关，降级）**：yfinance `.info` / AV `OVERVIEW` 是今天单点值、无日期维度，`curr_date < 今天` 时直接抛 `NoMarketDataError` → router 转 `NO_DATA_AVAILABLE` sentinel（fundamentals 分析师 grounding rule 接住，如实写「data not available」）。降级而非过滤——`.info` 无历史维度可供按日选取；若需恢复 PE/marketCap/EPS/beta 须单独立项从三表 + 历史价重构（L 工作量）。回测中 fundamentals 分析师因此失去 overview、改用三表（已正确 PIT 过滤）。
- **DSR 闸门（无开关，bugfix）**：`run_backtest` 现把 `n_trials` 透传给 `compute_metrics`（默认 1 = 字节等价）；`run_baseline --full` 设 `n_trials=2`（baseline + risk-improved 两个独立配置；`runs` 是同配置重复测量，不计入多重检验），让 `validation_gate` 的 DSR hurdle 真正抬起（此前 `engine.py` 硬编码 1 → `_expected_max_z` 返回 0 → hurdle 坍缩 → gate 近乎恒真）。
- **文献支撑**：PIT lookahead / 过拟合校验见 `REFERENCES.md` #59（look-ahead bias）/ #60 CPCV / #61 DSR（均 López de Prado 系），无需新增条目。
- **回测统计归因（opt-in，默认关 = 字节等价）**：`run_backtest` 两个 advisory 后处理，均纯 post-processing（不触 agent、不改 equity 衍生指标、fail-open）：
  - `factor_model`（参数默认 `None` = 跳过）：Fama-French 归因，`"3"`(Mkt-RF/SMB/HML) / `"5"`(+RMW/CMA)，从 French 数据源拉因子矩阵回归策略日收益，填 `metrics.factor_alpha/factor_betas/factor_r_squared`。`run_baseline --baseline/--full` 硬编码 `factor_model="3"`（已是生产路径）；编程式调用默认 None = 字节等价。实现 `backtest/engine.py:_attribute_factors` + `backtest/factor_model.py`。
  - `event_study`（参数默认 `False` = 跳过）：市场模型事件研究，对每个决策日用决策前 250 天估计窗拟合 `R_asset = a + b*R_benchmark`，检验持仓窗 CAR 是否显著非零（mean CAR + Brown&Warner cross-sectional t + bootstrap 95% CI），填 `metrics.event_study_*`。是对朴素 `alpha_vs_index`（无 β 控制、无显著性）的统计强化。需决策日前 ~250 天数据，故 opt-in 时额外 wide-window 重拉 asset+benchmark（`first_event - 400d`）。实现 `backtest/engine.py:_run_event_study` + `backtest/event_study.py`；report opt-in 渲染段。
  - 二者 fail-open：因子文件缺失 / 估计窗不足 / benchmark 拉不到 → 字段留默认 None，不中断回测。

## 自我改进闭环（light wiring，2026-08-14）

IC 剪枝结论的**真实落地路径**（此前 `prune_indicators_cli.py --suggest-config` 指向不存在的 `indicator_battery` 幻影键，指标清单只能手改 prompt 常量）：

- **`indicator_battery` 配置键**（`default_config.py`，默认 `None` = 全目录字节等价）：market analyst 的指标目录现由结构化 `_INDICATOR_SECTIONS` 渲染（`agents/analysts/market_analyst.py`），list 值只保留所指指标、空节整节消失；未知名 warning + 忽略，全未知/空列表 warning + 回退全目录（分析师永远有工具词汇）；`yialpha config-check` 校验名单（❌ 标记但不阻断 exit code）。默认渲染与重构前 hand-written 字面量**字节等价**（`tests/test_indicator_battery.py` 固化 git HEAD 快照）。
- **`yialpha snapshot record/diff/list`**：`config_snapshot.py`（机制完整但此前运行时零调用）的 CLI 入口。人工流程（fail-closed，永不自动改 live config）：`scripts/prune_indicators_cli.py --suggest-config` 产出证据 → 人工审查 → 人工改 `indicator_battery` → `yialpha snapshot record --reason ... --evidence ...` 落 append-only 快照（`<data_cache_dir>/config_history/`，带 git commit + 指纹）→ `run_analyst_parallel_ab.py` A/B 对比 → `snapshot diff` 检测漂移。`yialpha analyze`（交互入口）启动时自动比对最新快照指纹，漂移打一行 WARNING 指向 `snapshot diff`（advisory，不阻断）。
- 13 处残留静默退化已收口（2026-08-14 审计）：`yfinance_news.py` 两处错误字符串改 re-raise（news 是 core 类别，fail-closed 契约自此完整）；13F 畸形 filing_date 由 fail-open `pass` 改 fail-closed `continue`（对齐 CUSIP 门）；baostock 季报失败 / 回测基准 SPY 回退 / checkpoint 清理失败 / fin_cot prompt 降级 / memory 畸形日期 / Binance 恢复查询失败全部补日志。

### 证据链全机械化（2026-08-14 二期）

此前链条里每个箭头都要人手工搬运数据；现在**证据的采集与流转全部机械化了，只有「应用」一步留给人**（fail-closed 哲学不变）：

- **IC 数据集导出器 `scripts/export_ic_dataset.py`**（此前最上游断点：prune CLI 需要的 `date, forward_return, <指标>` CSV 只能人手拼）：从 OHLCV 缓存 + stockstats 直接算指标列 + `shift(-N)` 前瞻收益（末尾 N 行 drop——不伪造未实现的 forward_return），产出的 CSV 直接喂 prune CLI。指标名对 `INDICATOR_NAMES` 白名单校验；算不出的指标 skip + WARNING（绝不零填充）。
- **prune CLI `--json-out`**（此前建议只 print 到 stdout）：写结构化 verdict（params + keep/prune + 每指标 mean|IC|/finite_windows/longest_low_run），下游工具无需解析 markdown。
- **数据质量结构化落盘**（此前"全数据源失败仍产出 HOLD 报告"只有 LLM 散文说明）：router 每发一个 `NO_DATA_AVAILABLE`/`DATA_UNAVAILABLE` 哨兵就记一条 `{method, kind, detail}`（`dataflows/quality.py`，ContextVar per-run）；`_log_state` 把 `pm_rating`（PM 结构化评级，此前被拍平进 markdown）+ `data_quality` 块写进 `full_states_log_<date>.json`。
- **run_robust DEGRADED 判定**（此前"有新 complete_report.md 即成功"）：成功后读 full_states_log 的 `data_quality.core_sentinel_count`，>0 打 DEGRADED 标记；质量闸门**默认开启**（2026-08-16 反转：降级报告按失败重跑），`--allow-degraded` 是回到旧行为的逃生口（同时把子进程 `YIALPHA_DATA_VACUUM_POLICY` 降为 warn）；旧 `--require-data-quality` 保留为 no-op 兼容。旧版日志（无该字段）读作 None=未知，绝不误判为 0。
- 完整人工流程：`export_ic_dataset.py NVDA --horizon 5` → `prune_indicators_cli.py ic_data/NVDA_5d.csv --json-out ...` → 人工审查 → 人工改 `indicator_battery` → `snapshot record` → A/B → `snapshot diff`。

### 数据真空闸门 + 证据账本补盲（2026-08-16 T0 批）

此前"核心数据全超时仍产出正常样子的 HOLD 报告"是头号信任问题；本批把它从"可检测"升级为"默认不可通过"：

- **`data_vacuum_policy` 配置**（env `YIALPHA_DATA_VACUUM_POLICY`，默认 `reject`）：`quality.check_data_vacuum()` 在 **Trader 节点入口**执行（`graph/setup.py` 用 `gate_on_data_vacuum` 包装，证据在分析师全部跑完后即完整，且在任何决策阶段 LLM 计费之前）；**真空判定 = 有核心哨兵事件且 `record_success` 集合为空**（"尝试过且零成功"，比"全 sentineled"更精确）。`reject` 抛类型化 `DataVacuumError`（BatchRunner 逐 ticker 捕获 → 表格 + 退出码 1；run_robust 子进程非零退出 → 重跑/失败）；`warn` 维持旧行为。非法取值**fail-closed 到 reject**。交互式 `yialpha analyze` 自动降为 warn（有人在看实时面板），显式 env 优先。
- **证据账本补盲（7 处）**：router 核心类 `raise first_error` 前记 `KIND_CORE_ERROR`（新 kind，**计入 `core_sentinel_count`**）；4 个直连工具（market_data_validator / price_structure / weekly_indicators / binance_indicator_tools）的 `DATA_UNAVAILABLE` 全部记 `KIND_OPTIONAL_UNAVAILABLE`（它们绕过 router，此前是账本盲区）；reddit 真降级记证据（OAuth 拿不到 token / OAuth 失败退 RSS / RSS 自身失败 / 全程零帖），**无凭据走 RSS 是设计内默认、不记**（否则每个无 key run 都是噪声）。
- **`record_success(method)`**（同 record_sentinel 契约，永不抛异常）：router 成功返回且类目非 OPTIONAL 时记录；`summarize_quality` 新增 `core_ok_count` / `core_error_count` / `degraded_count`（= core 哨兵 + stale-cache，仅报告用，真空判定仍只看核心证据）/ `data_vacuum` 布尔。
- **默认多源链**（T0-3）：`data_vendors` 四个核心类目默认 `yfinance,alpha_vantage`（fundamentals 另加 `sec_edgar`）；**yfinance 永远在链首**（无 key 无限流），AV 免费档限流只作链尾；`config-check` 对"链含 AV 但 key 缺失"打 ⚠。
- **超时默认开**：`YIALPHA_HTTP_TIMEOUT_S` 默认 30s（原 opt-in；`0` 显式关闭）；BaoStock 裸 TCP 新增 `YIALPHA_BAOSTOCK_TIMEOUT_S`（默认 30s，`_BaostockSession` 生命周期内 save/restore `socket.setdefaulttimeout`，requests 层 shim 够不到裸 socket）。
- **决策时价格入档**（T0-5）：`full_states_log` 新增 `price_at_decision` / `price_at_decision_basis` / `asset_type`（`_apply_risk_overlay` 结尾挂 `decision.entry_price`；overlay 未跑时 `_log_state` 兜底直调记忆化的 `_latest_close_and_atr`）；旧日志缺字段向后兼容，Web 的 overlay 正则回捞保留。这是评级↔结果验证闭环（verify-history / /api/accuracy）的锚点。

### 评级↔结果验证闭环 + memory-resolve（2026-08-16 T2 批）

- **`yialpha verify-history`**（`yialpha/accuracy.py`）：扫全部 full_states_log → 每 (ticker, 日期, 评级) 取 PIT 前向收益（现货 `get_YFin_history_cached`，perp/spot 走 `binance_klines_frame`，按 `asset_type` 分流；旧日志缺 asset_type 时 USDT 后缀推断为 perp 且在报告里标注）→ 方向命中率（Buy/Overweight/Sell/Underweight）+ Hold 机会成本均值 + 分评级/分标的表（**全部带 n**）；horizon 未走完 = pending 绝不计分。产物 `accuracy/accuracy_report.{json,md}`；`GET /api/accuracy` + Web「评级准确率」视图只读服务该产物（未生成时诚实地 `available:false` + 命令提示，绝不伪造空报告）。
- **`yialpha memory-resolve`**：按需清扫全部 ticker 的 pending 记忆条目（此前只有同 ticker 重跑才解析）。解析核心下沉到 `yialpha/graph/memory_resolution.py`（graph 的 `_resolve_pending_entries` 委托它），收益计算下沉到 `yialpha/accuracy.fetch_returns_yf`（graph 的 `_fetch_returns` 同步改为薄委托——两处不再可能漂移）。每条解析一次反思 LLM 调用。

### IC 闭环半自动化 + indicator_ic_context（2026-08-16 T1 批）

- **`yialpha ic-cycle`**：一条命令跑完 export → prune 判定 → 打印建议。数据集构建下沉到包内 `yialpha/backtest/ic_dataset.py`（`build_ic_frame`/`export_ic_datasets`/`prune_verdict_for_csv`/`run_ic_cycle`；export 脚本变薄委托，argv 契约不变——脚本不在 wheel 里，CLI 只能调包内实现）。判定 JSON 与 prune CLI 的 `--json-out` 同构；输出各 ticker 判定表 + 跨 ticker `indicator_battery` 交集建议 + `snapshot record --evidence <prune.json>` 提示。**永不自动改 live config**（人审是契约）。
- **`.github/workflows/ic-cycle.yml`**：每周六 04:30 UTC（+手动 dispatch），独立 concurrency group（CI 组会被 push 取消），ic_data/ 作 90 天 artifact；空数据集 = 任务失败（空 artifact 冒充证据更糟）。
- **`snapshot record --evidence` 卫生**：path 形态（单 token + 熟知后缀）但文件不存在 → 大声警告不阻断；自由文本描述合法不响。
- **`indicator_ic_context`**（env `YIALPHA_INDICATOR_IC_CONTEXT`，**默认关** = prompt 与 A/B 基线字节等价）：开启后 market analyst 系统消息追加一条 advisory——各指标 trailing mean |IC|（`ic_data/*.prune.json` 的 `per_indicator.mean_abs_ic` 跨文件平均，只渲染目录内已知指标；无 verdict → 不追加）。**`.prune.json` 第一次有了运行时消费者**。

## Binance 资产类型（crypto_perp / crypto_spot）

两条 Binance 分析轨道，均为**只读公共行情、无鉴权、无下单**（Track A，分析专用）。手写 `requests`（**非官方 SDK**），复用已验证的 SOCKS5 代理（`_proxies()`）+ 产品线独立限流（`get_binance_weight_limiter("fapi"|"spot")`）+ 反应式 429/418 兜底 + 可选类目降级。

| 资产类型 | `--asset-type` | 数据源 | 分析师绑定工具 | 备注 |
|---|---|---|---|---|
| `crypto_perp` | `crypto_perp` | Binance USDT-M 永续（`fapi.binance.com`）+ data.binance.vision 深历史归档 | 基础 5（live+历史）：klines / funding / indicators / **vision_metrics / vision_book_depth**；live 另含 6：OI / long_short_ratio / taker_buy_sell / basis / premium_index / depth_snapshot，另加 live web_search（合计 live 12 / 历史 5，名单由 `test_crypto_perp_mode.py` 钉住） | 隐藏 Yahoo 工具（符号会解析到错误 spot 对）；RSI/MACD 由 `get_binance_indicators` 在 perp K 线上计算（stockstats 全电池，2026-08-15）；vision 归档 = PIT 正确的多年深历史持仓/盘口（REST 持仓端点仅保 30 天） |
| `crypto_spot` | `crypto_spot` | Binance 现货（默认 `api.binance.com`，镜像可切） | 基础 10（live+历史）：get_stock_data / spot_klines / spot_indicators / indicators / verified_snapshot / 周线 / S-R / 量价 / K线形态 / 相对强度；live 另含 2：ticker24 / **spot_perp_basis**，另加 live web_search（合计 live 13 / 历史 10） | 现货无 funding/OI/杠杆；保留 indicators（符号解析正确）；**spot_perp_basis 是全新 alpha 维度**（跨 venue 基差 = 永续收盘 − 现货收盘） |

**字节等价**：两模式均为 asset_type 新分支；泛化的 `_http_get`/`_paginate_history` 用默认参数（`base=_FAPI_BASE, weight_key="fapi"`），perp/stock/crypto 路径字节不变（引入当时 `tests/test_crypto_perp_mode.py` 零修改全绿为证；后续加工具时该测试同步更新钉名单）。

**Track B（执行 / 实时轨道，骨架已就位、网关未接）**：官方模块化 SDK `binance-sdk-derivatives-trading-usds-futures` + `binance-sdk-spot` 是实时 WS 行流 + 下单 + user-data stream 的预设基础（`dataflows/binance.py` docstring 已留 "Track B (execution) will bring the official SDK"）。**当前不迁 SDK**——对分析层 REST 公共行情无净收益，且 SDK 未文档化支持 `socks5h://`（需额外 `requests[socks]`/`aiohttp-socks`）。执行轨道单独启动时再引入。离线 SDK 源码副本 + Spot API 文档存于仓库外 `D:\edge download\binance-connector-python-master\` 与 `D:\edge download\binance-spot-api-docs-master\`（2026-07-08 二次确认：SOCKS5h 零支持、不接收外部 `requests.Session`、5xx 重试有隐性 bug、依赖重——迁分析层净负，结论不变）。

**执行层（已落地，默认关 = 字节等价）**：`yialpha/execution/` 现有四件套——`domain.py`（vnpy MIT 移植的交易域 dataclass + enum，精简到单标的批量下单子集，去 i18n）、`gateway.py`（薄同步 `BaseGateway` ABC：`send_order` 同步返回 `OrderData`、`on_*` 回调降级 no-op，**不**移植 EventEngine 后台线程）、`bridge.py`（`decision_to_order_requests` 纯函数，方向判定逐字对齐 `scripts/trade_ticket.py:decide_direction`，rating 优先、action 兜底、Hold→空单）、`binance_gateway.py`（**Track B 首个具体网关 `BinanceGateway`**，官方模块化 SDK `binance-sdk-spot`/`binance-sdk-derivatives-trading-usds-futures`，MIT，`binance-common` v4.1.0+）。`BinanceGateway`：spot+perp 两产品线，LIMIT/MARKET 入场单（`new_order` + `new_client_order_id` 幂等）+ 查单/撤单/账户/持仓；SDK 懒导入（默认关时不依赖 SDK 安装）；testnet 默认（`base_path` 覆盖 `*_TESTNET_URL`），主网需 `YIALPHA_EXECUTION_MAINNET=true`；proxy 从 `HTTPS_PROXY` 拆 `socks5h` 注入 `ConfigurationRestAPI(proxy=…)`（v4.1.0 一等支持，推翻 2026-07-08「SOCKS5h 零支持」的旧推迟结论）；**fail-closed**：开关关/缺 key/未连接/不支持类型 → REJECTED 或 `VendorNotConfiguredError`；**绝不盲补单**：SDK 对 POST 零重试，模糊失败先 `query_order` 查态、查不到→REJECTED（不重复成交），418 IP 禁令不重试。配套 `ExecutionEnableSwitch`（镜像 `KillSwitch`、`YIALPHA_EXECUTION_ENABLED`，malformed→关）。执行层整体**仍未挂进 LangGraph 图**、零调用方、默认关 = 字节等价；现有 `browser_broker.py`（fail-closed Playwright 兜底）一字未动。待续：挂图（默认关节点，独立 guarded 步骤，触发 isolation 门）、futures 条件单 `new_algo_order`（`trade_ticket.py` ATR 止损自动落）、user-data-stream 实时 `on_*` 回调、order-count 预算限流、`BrowserBrokerGateway` adapter / A 股 `vnpy_ctp` 腿。详见 `yialpha/execution/{domain,gateway,bridge,binance_gateway}.py` 顶部 docstring 与 `tests/test_execution_*.py`。

| 执行 env | 默认 | 作用 |
|---|---|---|
| `YIALPHA_EXECUTION_ENABLED` | false | `BinanceGateway` 使能开关（镜像 `KillSwitch`，order-time 直读 env，malformed→关）。关=零 SDK 导入、零图拓扑变化、字节等价 |
| `BINANCE_API_KEY` / `BINANCE_API_SECRET` | 未设 | 交易密钥（裸命名，同 `DEEPSEEK_API_KEY`，**无 `YIALPHA_` 前缀**）；只进 `.env`（gitignore），缺 → `VendorNotConfiguredError` |
| `YIALPHA_EXECUTION_MAINNET` | false | true 才连主网；默认 testnet |
| `YIALPHA_EXECUTION_TIMEOUT_MS` | 10000 | 单次 SDK REST 读超时（ms） |

## 运行规范

- **必须先 `cd` 到项目根再跑**：`yialpha/__init__.py` 用 `load_dotenv(usecwd=True)` 注入 `HTTP_PROXY`/`NO_PROXY`/`ZHIPU_CN_API_KEYS`；在别处裸跑会 DNS 解析失败 / 缺 key。
- **跑 ≥1 个 ticker 一律优先 `scripts/run_robust.py`**（per-ticker 独立子进程 + 看门狗 + OS 级强杀重跑），别裸跑 in-process batch——VPN / LLM mid-response 卡死在进程内不可恢复。
  ```bash
  python scripts/run_robust.py --tickers SNDK INTC --date 2026-07-01 \
    --workers 2 --per-ticker-timeout 1800 --max-attempts 3
  ```
- **单 ticker 墙钟 ~8-10 分钟**（flash/旗舰两档分工；DeepSeek 时代实测值，GLM 量级相近）。规划按 ≥10 min/ticker 估。
- **LLM 直连偶发 APIConnectionError / DNS 瞬时失败**属正常抖动：对失败 ticker 单独 `python scripts/run_batch.py --tickers <T> --date <D>` 重跑即可恢复。
- 报告产物：`~/.yialpha/logs/reports/<TICKER>_<ts>/`；逐 ticker 完成态看 `~/.yialpha/logs/<TICKER>/YiAlphaStrategy_logs/full_states_log_<date>.json`（完成才落盘，可作进度信号）。

## 环境（已验证可用，别再重复诊断）

- GLM Coding Plan 三把 key 有效（2026-09-03 实测），**走直连**：`.env` 里 `NO_PROXY=open.bigmodel.cn`。端点 `ZHIPU_CN_BASE_URL=https://open.bigmodel.cn/api/coding/paas/v4/`（coding plan 专用）。可用模型 `glm-5.3` / `glm-5.3-flash`。
- yfinance / 行情 / 财报必须经 `socks5h://127.0.0.1:1080`（SOCKS5 代理，FLASH-CAT VPN）。代理一断 → 数据抓取永久挂起。
- **Windows 控制台是 GBK(cp936)**，打印 ✅/❌ 会 `UnicodeEncodeError`；入口脚本顶部须 `sys.stdout.reconfigure(utf-8)`，兜底用 `PYTHONUTF8=1`。
- Reddit RSS 429、`FRED_API_KEY not set` 是**非致命降级**，不影响评级输出，不用装 FRED key。

## 验证路线（`scripts/run_baseline.py`）

1. `--preflight`（零 LLM 成本自检）
2. `--smoke`（1 票 1 日，跑通整条 LLM 图）；加 `--profile` 同跑且打印「节点→墙钟占比」遥测表（零影响，见上「性能 / 遥测层」）
3. `--baseline`（基线回测）
4. `--full`（基线 vs 风控 A/B + 闸门判定）——闸门 PASS 才做券商适配

分析师并行的分布等价性 gate 走 `scripts/run_analyst_parallel_ab.py`（独立于上述回测路线，见上）。
