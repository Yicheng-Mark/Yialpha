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

**DeepSeek 全通道统一（2026-09-30 晚生效，当前状态）**：GLM coding plan 端点当日病态 ~5h（拒连/建连后挂死/滴字节吊流，详见 memory `glm-endpoint-sick-day-2026-09-30`），整链切回 DeepSeek。

- **deep / quick / debate 三通道全部 `deepseek-flash`**（display name **DeepSeek-V4.1-Flash**，`/models` 实测 id，2026-09-30 验证；1M ctx，thinking 档——`capabilities.py` 注册不发 `tool_choice`、需 `reasoning_content` 回传）。
- 端点 `https://api.deepseek.com` 走 `NO_PROXY` 直连；`YIALPHA_LLM_TIMEOUT_S=120` + `YIALPHA_LLM_MAX_RETRIES=6` 抗半开挂死与 DNS 瞬时抖动（`.env` 内有实测注释）。DeepSeek 单 key（`DEEPSEEK_API_KEY`）不经 key 池。

**GLM 备用（切回只改 `.env` 的 provider + 三个模型行，四把 key 已保留在 `ZHIPU_CN_API_KEYS`）**：两档分工 deep=`glm-5.3` / quick+debate=`glm-5.3-flash`；**Key 池**（`yialpha/llm_clients/key_pool.py`）：`ZHIPU_CN_API_KEYS` 四把 key（2026-09-19 扩容；多项目共用，单 key 的 5h 窗口可能被其他项目耗尽——429 冷却轮换正是为此设计），**每个请求轮流取下一把活 key**（401/403 该 key 本进程内踢出；429 冷却 `YIALPHA_LLM_KEY_COOLDOWN_S` 秒默认 120 后回归）。coding plan 端点 `ZHIPU_CN_BASE_URL=https://open.bigmodel.cn/api/coding/paas/v4/` 与按量端点不通用。单 key（仅 `ZHIPU_CN_API_KEY`）时池不启用。**Tavily 同款池**（`TAVILY_API_KEYS` + `TAVILY_API_KEY` 合并去重轮换，现 5 把）。

`capabilities.py` 登记的 deepseek 模型档案（v1-3 chat 档 / v4+ thinking 档 / `deepseek-flash` / `deepseek-v4-pro`）继续维护，不要误删。

## 铁律

**并发 / 性能层不得改动任何 agent 的输入 / 能力 / 深度**——并发层不引入新随机性，串行 vs 并发的分布必须一致。

**输入闸门（硬性）**：仅接受美股上市股票/ETF 与加密货币标的。非美股、非加密的 ticker（A 股 / 港股 / 东京 / 伦敦 / 印度 / 加拿大 / 澳洲交易所后缀及外汇、商品、指数 CDN 别名）在 CLI / batch / web / `propagate()` 各入口直接抛 `UnsupportedMarketError`（`yialpha/dataflows/symbol_utils.py`），不做静默解析或降级。

## 性能 / 遥测层（零影响，全部默认关 = 字节等价）

并发 / 传输 / 观测层一律不触 agent 输入；以下开关全部**默认关（或等价值）= 与今天字节等价**，按需开：

| 开关（env） | 默认 | 作用 | 备注 / 产物 |
|---|---|---|---|
| `YIALPHA_LLM_TIMEOUT_S` | off（未设；生产建议 120） | 单次 LLM 读超时；半开连接 → `APITimeoutError` → SDK 内置重试恢复 | `openai_client.py` 在线读；消除偶发 30min 卡死。默认 OFF 会在首次调用时发出一次性警告。**`run_robust.py` 为子进程 setdefault 120**（2026-09-19：hang-recovery 语义，与 LLM 缓存 setdefault 同一先例——半开挂死从「等看门狗杀整个 attempt」变「120s 超时 → SDK 重试秒级恢复」；用户显式导出的值含 0 优先）。**`0` = 显式关闭**（2026-09-22 round-6：此前 0 被当非法值弹回 120s 云默认，与文档相反；现解析为 `timeout=None`，挂死恢复只依赖看门狗） |
| `YIALPHA_LLM_POOL_STREAM_USAGE` | true | key 池 streaming 模式下显式请求 usage 块（`stream_options.include_usage`），否则 token 遥测归零 | 2026-09-22 round-6：自定义 http_client（池的）会关掉 langchain 的自动 `stream_usage`；false 是端点拒绝 stream_options 时的逃生口 |
| `YIALPHA_HTTP_KEEPALIVE` | true（已设） | 进程级共享 `httpx.Client`，复用 TLS/SOCKS5 连接 | 仅 OpenAI 兼容 provider；key 池生效时由池的轮换 client 接管（同样是共享 keepalive，见「模型偏好」） |
| `YIALPHA_LLM_MAX_RETRIES` | 2（= langchain 默认，等价） | 单调用重试次数；抖动期可调低 | 默认值与历史字节一致；外层靠 `run_robust` 看门狗兜底 |
| `YIALPHA_LLM_CACHE` | false | per-call LLM 响应磁盘缓存（langchain 全局 `set_llm_cache` + `llm_clients/response_cache.py` `DiskLLMCache`）：相同 (model+prompt+temperature+绑定 tools/结构化 schema) 回放缓存的 `ChatGeneration` 而非重调模型 | 默认关=无缓存无 I/O（字节等价）；迭代重跑同一 smoke/单次分析省中间 ~11 个 agent 调用计费；产物 `~/.yialpha/cache/llm_responses/`。**勿与 `run_analyst_parallel_ab.py` / `run_baseline --full` DSR 同用**——会压扁温度>0 多 run 分布；回测整图重跑已由 `backtest/cache.py` DecisionCache 覆盖。**`run_robust.py` 默认为自己的运行开**（`setdefault` 尊重 env 显式值）：重跑回放已完成节点的 LLM 生成、卡死节点重跑 = hang 恢复语义，省整图重计费；run_robust 是 live 单配置分析、不触达上述分布 caveat；`--no-llm-cache` 关 |
| `YIALPHA_NODE_PERF_TELEMETRY` | false | 节点级墙钟 + token 遥测（包装每个节点 handler） | 产物 `node_perf_<date>.json`（紧邻 `full_states_log`）；`run_baseline --profile` 一键开。`ToolNode` 走 Runnable 感知包装（round-11 前计零，曾系统性压低 A/B gate 串行腿）；并行模式子图内 tool 节点不单独包装、墙钟折入父级 `Analyst Fanout` 节点 |
| `YIALPHA_ANALYST_PARALLEL` | false | 4 分析师并行（单一 `Analyst Fanout` 节点内跑 4 个独立子图） | **必须先过 A/B gate 才翻默认**；与 `llm_rate_limiter` 强交互（rpm<120 实际仅 ~1.5×） |
| `YIALPHA_ANALYST_PARALLEL_MAX_THREADS` | 16 | 嵌套并发上限：fanout 节点内 ThreadPoolExecutor 的 `max_workers=max(1, N)`（≤0 钳到 1；钳制只减少并发数，**绝不回退串行**） | |
| `YIALPHA_BINANCE_PROACTIVE_BACKOFF` | false | Binance perp vendor 读 `X-MBX-USED-WEIGHT-1M` 头主动退避 | 仅改"何时发请求"不改数据；线程安全（BatchRunner 并发）；反应式 429/418 仍是兜底。配 `YIALPHA_BINANCE_WEIGHT_THRESHOLD`（fapi 默认 2400/min） |
| `YIALPHA_BINANCE_SPOT_MIRROR` | false | crypto_spot 现货行情走免 key 镜像 `data-api.binance.vision`（默认 `api.binance.com`） | 仅改现货 host，不改数据；现货为新代码、无既有输出可扰；现货限流独立预算 `get_binance_weight_limiter("spot")` |
| `YIALPHA_BINANCE_HTTP_KEEPALIVE` | false | 进程级共享 `requests.Session`，跨调用复用 TLS/SOCKS5 连接（仿 LLM 客户端 `YIALPHA_HTTP_KEEPALIVE`） | 仅传输层（同 URL/params/头 → 同响应字节）；`dataflows/binance_http.py` 单例；urllib3 PoolManager 并发只读线程安全 |
| `YIALPHA_BINANCE_HTTP_RETRIES` | 0 | 瞬时传输错误（DNS/超时/TLS）+ 5xx 指数退避重试（仿 `yf_retry`，base 2s） | `0`=原样抛（字节等价）；耗尽→`NoMarketDataError`（router 降级）；**不**重试 429/418（仍走反应式 `VendorRateLimitError`）；负值钳 0 + 警告（round-12：`-1` 曾使 `range(max_retries+1)` 零次尝试 → 传输层裸崩，报错不指名 env 也不指名 vendor） |
| `YIALPHA_BINANCE_HONOR_RETRY_AFTER` | false | 429/418 读 `Retry-After` 头，≤60s 则 sleep 后再抛（让 IP 禁令窗口过期） | 默认关=立即抛（字节等价）；>60s 的长禁令不内联 sleep，交给 `run_robust` 重跑 |
| `YIALPHA_SEC_OWNERSHIP` | false | optional category `sec_ownership`：Form4 内幕交易 + FTD + 13F 机构持仓三工具暴露给 fundamentals 分析师 | 默认关=工具列表/prompt/能力字节等价（同 `valuation_tools` 契约）；复用 `sec_edgar.py` 基建（CIK/`_sec_get`/`_cached_or_fetch`/`_fetch_company_facts`/缓存/限流，**不改 sec_edgar 一行**）；美股专属（Form4/13F 需 CIK，非美股→`NO_DATA_AVAILABLE`；FTD 按 ticker，非美股自然无行）；PIT（Form4 `filingDate` / FTD `cutoff+发布滞后` / 13F 数据集 `period-end+发布滞后`）；新 category 默认配置不引用=未 opt-in 零触达。13F（B2.1）走 SEC 批量 Form 13F Data Sets（每季一 ZIP，COVER+HOLDING TSV，本地按 CUSIP 反向聚合，1 请求/季）；CUSIP 单源 companyfacts `dei:EntityCusip`（缺失降级）；45 天发布窗口内诚实"未发布"，不做 EFTS |
| `YIALPHA_FTD_PUB_LAG_DAYS` | 10 | FTD 半月度文件 PIT 发布滞后：`cutoff + lag ≤ curr_date` 才可见 | 默认 10 天保守值（SEC 在 cutoff 后数天才发布）；防回测偷看尚未发布的文件 |
| `YIALPHA_SEC_13F_PUB_LAG_DAYS` | 45 | 13F 批量数据集 PIT 发布滞后：`数据集 period-end + lag ≤ curr_date` 才可见 | 默认 45 天（对齐 13F 法定 45 天申报窗口；SEC 在申报窗口结束后才发布季度 ZIP——round-11 修正文档漂移：此前文档写 5，代码两处一直为 45）；防回测偷看尚未发布的季度 ZIP；窗口内诚实降级，不做 EFTS |
| `YIALPHA_MARKET_REGIME` | false | 市场动荡指数（FinRL `calculate_turbulence` 单资产降维 = 基准 252 日滚动收益的平方 z）：开时给**保守风控 debater** 的数据源段追加一行市场压力读数 | 默认关=保守 debater prompt 字节等价（同 `sec_ownership`/`valuation_tools` 契约）；市场层**事前**信号，与组合层事后 `DrawdownBreaker` 互补；advisory + fail-soft（取数失败→不加行，不报错）；`yialpha/dataflows/market_regime.py`（MIT 头注明 FinRL 来源），benchmark 经 `_resolve_benchmark` 同款后缀解析（纯币家族→BTCUSDT 原生基准（渲染行显式标注）；tokenized-stock perp 与股票→SPY，可 `benchmark_ticker` 覆盖） |
| `YIALPHA_CRYPTO_FEAR_GREED` | false | 加密恐惧贪婪指数（alternative.me keyless）：开时给**保守风控 debater** 的数据源段追加一行市场情绪带读数（0-100 + 标签 + 前一日值）——交易所内多空比之外的 survey+composite 外部视角 crowding 线索 | 默认关=debater prompt 字节等价（同 `market_regime` 契约，2026-09-29 落地）；仅 crypto 家族 run（pure perp / spot / legacy）适用，tokenized-stock perp 与股票 run 零触达零抓取（适用但取数失败→显式 unavailable 行，fail-visible）；PIT=client-side 时间戳过滤（行内披露 as-of 日期；`start/end` 端点参数实测不可靠故弃用），`limit=1500` 覆盖约 2022-08 起，更早日期诚实 unavailable；磁盘缓存 1 天 + 1s 节流；`yialpha/dataflows/fear_greed.py`，`tests/test_fear_greed.py` 14 例钉死 |
| `YIALPHA_CRYPTO_NATIVE` | false | 纯币基本面 optional category `crypto_native`：纯币（`pure_crypto_perp` / `crypto_spot` / legacy `crypto`）的 fundamentals 分析师**复活**——`get_crypto_token_fundamentals`（CoinGecko 免费档：市值/排名/供给/FDV/ATH-ATL，live 快照；历史=日级 PIT 序列，免费档限 trailing ~365 天，更早诚实降级）+ `get_crypto_network_fundamentals`（DefiLlama：链 TVL 全历史 + 协议 TVL（治理代币符号匹配取最高 TVL 版并披露，$1M 可信度地板挡 SHIB→shibex 类撞名）+ live 链排名；链解析=静态表→/v2/chains 目录 `tokenSymbol`→协议主链，新链零代码自动扩）；**替换**（非追加）股票工具集 + crypto 化 persona prompt（market 分析师 perp 先例），股票 bundle 跳过（防核心哨兵误伤真空闸门） | 默认关=纯币仍跳过、其他运行字节等价（双重门控：开关 AND 纯加密 instrument class，默认关 = 零工具绑定、零 prompt 变化）；门=routing 谓词 `crypto_native_fundamentals_run`（flag on **且** `instrument_class ∈ {pure_crypto_perp, crypto_spot}`，unknown_perp 诚实跳过）——节点运行时门与 CLI/batch 选择过滤**单源共享**；符号归一=乘数前缀（1000/1000000）解包 + legacy 破折号（BTC-USD→BTC）；**免费/keyless 外网源走 SOCKS5**（`proxy_map()`，CoinGecko 必须浏览器 UA——默认 python UA 边缘 403；节流 `MinIntervalThrottle` 4s/1s）；无可选依赖（requests 核心）；PIT（行 `date <= curr_date`；current-only 字段绝不进历史视图，含协议标题 current-TVL 与链排名）；逐腿 fail-soft（一条腿挂不丢另一腿已渲染证据）+ mtime 解析 memo；`_SIGNATURE_CONFIG_KEYS` + crypto_native（checkpoint 失效语义）。`yialpha/dataflows/{coingecko_vendor,defillama_vendor}.py`；`tests/test_crypto_native.py` 37 例钉死 |
| `YIALPHA_CHECKPOINT_ENABLED` | false | 崩溃可续跑：per-ticker `SqliteSaver`，从上一个成功节点恢复，成功后自动清 checkpoint | 默认关=字节等价；开时 `thread_id` 折入图形状签名（`analysts`/`debate`/`risk`/`asset`/`parallel` 五项 + `config=` 段：模型档位与 sec_ownership / crypto_native / valuation_tools / indicator_battery / indicator_ic_context / market_regime / crypto_fear_greed / data_vendors 的 sorted-JSON，2026-09-19——输入语义配置翻转不回放旧工具产出的报告），换配置续跑不命中旧图形状的陈旧 checkpoint（#1089）；检测到 checkpoint 时 `_run_graph` 以 `None` 输入 invoke/langgraph（非 None 输入一律从 `__start__` 重跑全图——旧"resume"实为全图重计费，2026-09-19 修）；`run_robust` 走子进程重启不依赖它 |
| `YIALPHA_POSITIONING_SPLIT` | false | crypto_perp 专属 Positioning Analyst（V2.3 拆分的 perp 持仓结构半，结构化输出；位置钉在 market 之后） | 默认关=字节等价。round-11 修通端到端死开关：此前 filter 追加了该分析师但交互选择漏斗 / batch union / 实时面板 / web 卡片四处各自静默丢弃——flag 开也等于没开；单源 `filter_analysts_for_asset_type`，图构建前经 `resolve_run_analysts` 重应用 |

- **`clear_node` 是串行耦合点**：它枚举共享 `state["messages"]` 全部 ID 删消息，仅串行安全 → 分析师并行用「每分析师独立子图」结构规避（父图共享 messages 永不并发写）。已逐一读码确认：下游节点（Bull / Bear / Research Manager / Trader / 3 风控辩手 / Portfolio Manager）**均不读 `messages`**，只读各自 `*_report` + debate state，故父图 messages 残留不影响任何下游决策。
- **遥测一键**：`python scripts/run_baseline.py --smoke --profile --ticker <T> --date <D>` 跑完打印「节点→墙钟占比 + token」表，定位真实瓶颈。
- **A/B gate（用户自跑，不在默认流程）**：`python scripts/run_analyst_parallel_ab.py --tickers <T> --date <D> --n 10`；指标=①评级卡方 p>0.05 ②4 份 `*_report` TF-IDF 余弦 between≥within ③同评级风控 overlay 数值字节一致 ④分析师段墙钟 ≥2.5×。**全过才翻 `YIALPHA_ANALYST_PARALLEL=true`**。先 `--dry-run`（零 LLM 成本）验证脚本本身。

## 数据正确性层（默认开 = 修正既有 lookahead，非字节等价）

与上面「性能 / 遥测层（默认关 = 字节等价）」不同，以下是**回测正确性 bugfix**：默认开、且故意改变回测输出——修正前的输出含未来信息。均落 dataflows / backtest 层，不触任何 agent 输入，不引入新随机性，串/并发分布一致。

- **`YIALPHA_FUNDAMENTALS_FILING_LAG_DAYS`（默认 45）**：基本面三表（BS/CF/IS）按 `fiscalDateEnding + 公告滞后 ≤ curr_date` 过滤，而非按报告期末。修正「Q3 报表 9/30 结束、10/1 即被回测可见」的 lookahead（10-Q 实际要 ~40 天后才公告）；45 天对齐 SEC large accelerated filer 节奏（10-K≈60d / 10-Q≈40d）。设 0 回退旧行为。实现 `dataflows/utils.py:is_filing_public`。价格层早有 PIT（`stockstats_utils.py` cutoff + stale guard），此为基本面层补齐。
- **overview 快照（无开关，降级）**：yfinance `.info` / AV `OVERVIEW` 是今天单点值、无日期维度，`curr_date < 今天` 时直接抛 `NoMarketDataError` → router 转 `NO_DATA_AVAILABLE` sentinel（fundamentals 分析师 grounding rule 接住，如实写「data not available」）。降级而非过滤——`.info` 无历史维度可供按日选取；若需恢复 PE/marketCap/EPS/beta 须单独立项从三表 + 历史价重构（L 工作量）。回测中 fundamentals 分析师因此失去 overview、改用三表（已正确 PIT 过滤）。
- **DSR 闸门（无开关，bugfix）**：`run_backtest` 现把 `n_trials` 透传给 `compute_metrics`（默认 1 = 字节等价）；`run_baseline --full` 设 `n_trials=2`（baseline + risk-improved 两个独立配置；`runs` 是同配置重复测量，不计入多重检验），让 `validation_gate` 的 DSR hurdle 真正抬起（此前 `engine.py` 硬编码 1 → `_expected_max_z` 返回 0 → hurdle 坍缩 → gate 近乎恒真）。
- **perp bundle 单腿超时改默认（round-11，`YIALPHA_PERP_BUNDLE_LEG_TIMEOUT_S` 默认 300）**：旧硬编码 60s 把 `YIALPHA_BINANCE_PROACTIVE_BACKOFF`（单次 acquire 可睡 ~60s）/ `BINANCE_HTTP_RETRIES` 退避叠加的**合法慢腿**误判 TimeoutError → 核心哨兵 → 整单 NO_TRADE——**默认即生效的行为修正**（故记本段而非开关表）。纯标注阈值：`ThreadPoolExecutor.__exit__` 本就 join 到底，抬高不延长最坏墙钟（真挂死归 run_robust OS 看门狗）。`0` = 显式关闭（无限等，与 pool join 语义一致）；负数/垃圾值警告回落 300。
- **文献支撑**：PIT lookahead / 过拟合校验见 `REFERENCES.md` #59（look-ahead bias）/ #60 CPCV / #61 DSR（均 López de Prado 系），无需新增条目。
- **回测统计归因（opt-in，默认关 = 字节等价）**：`run_backtest` 两个 advisory 后处理，均纯 post-processing（不触 agent、不改 equity 衍生指标、fail-open）：
  - `factor_model`（参数默认 `None` = 跳过）：Fama-French 归因，`"3"`(Mkt-RF/SMB/HML) / `"5"`(+RMW/CMA)，从 French 数据源拉因子矩阵回归策略日收益，填 `metrics.factor_alpha/factor_betas/factor_r_squared`。`run_baseline --baseline/--full` 硬编码 `factor_model="3"`（已是生产路径）；编程式调用默认 None = 字节等价。实现 `backtest/engine.py:_attribute_factors` + `backtest/factor_model.py`。
  - `event_study`（参数默认 `False` = 跳过）：市场模型事件研究，对每个决策日用决策前 250 天估计窗拟合 `R_asset = a + b*R_benchmark`，检验持仓窗 CAR 是否显著非零（mean CAR + Brown&Warner cross-sectional t + bootstrap 95% CI），填 `metrics.event_study_*`。是对朴素 `alpha_vs_index`（无 β 控制、无显著性）的统计强化。需决策日前 ~250 天数据，故 opt-in 时额外 wide-window 重拉 asset+benchmark（`first_event - 400d`）。**CAR 窗口跟随 `holding_days`（round-12：此前固定模块默认 `(0, 20)`——21 个 bar 的漂移，与"holding window"承诺相悖）**；窗口实际值写入 `metrics.event_study_window` 并在报告披露；Binance 形基准（BTCUSDT 等）的 index 腿走 run 自己的 Binance provider 而非 yfinance（round-12：`normalize_symbol` 会把 BTCUSDT 静默换成 Yahoo 现货 BTC-USD，工具错配折进"alpha"）。实现 `backtest/engine.py:_run_event_study` + `backtest/event_study.py`；report opt-in 渲染段。
  - 二者 fail-open：因子文件缺失 / 估计窗不足 / benchmark 拉不到 → 字段留默认 None，不中断回测。

## 自我改进闭环（light wiring，2026-08-14）

IC 剪枝结论的**真实落地路径**（此前 `prune_indicators_cli.py --suggest-config` 指向不存在的 `indicator_battery` 幻影键，指标清单只能手改 prompt 常量）：

- **`indicator_battery` 配置键**（`default_config.py`，默认 `None` = 全目录字节等价）：market analyst 的指标目录现由结构化 `_INDICATOR_SECTIONS` 渲染（`agents/analysts/market_analyst.py`），list 值只保留所指指标、空节整节消失；未知名 warning + 忽略，全未知/空列表 warning + 回退全目录（分析师永远有工具词汇）；`yialpha config-check` 校验名单（❌ 标记但不阻断 exit code）。默认渲染与重构前 hand-written 字面量**字节等价**（`tests/test_indicator_battery.py` 固化 git HEAD 快照）。
- **`yialpha snapshot record/diff/list`**：`config_snapshot.py`（机制完整但此前运行时零调用）的 CLI 入口。人工流程（fail-closed，永不自动改 live config）：`scripts/prune_indicators_cli.py --suggest-config` 产出证据 → 人工审查 → 人工改 `indicator_battery` → `yialpha snapshot record --reason ... --evidence ...` 落 append-only 快照（`<data_cache_dir>/config_history/`，带 git commit + 指纹）→ `run_analyst_parallel_ab.py` A/B 对比 → `snapshot diff` 检测漂移。`yialpha analyze`（交互入口）启动时自动比对最新快照指纹，漂移打一行 WARNING 指向 `snapshot diff`（advisory，不阻断）。
- 13 处残留静默退化已收口（2026-08-14 审计）：`yfinance_news.py` 两处错误字符串改 re-raise（news 是 core 类别，fail-closed 契约自此完整）；13F 畸形 filing_date 由 fail-open `pass` 改 fail-closed `continue`（对齐 CUSIP 门）；回测基准 SPY 回退 / checkpoint 清理失败 / fin_cot prompt 降级 / memory 畸形日期 / Binance 恢复查询失败全部补日志。

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
- **默认多源链**（T0-3）：`data_vendors` 四个核心类目默认 `yfinance,alpha_vantage`；**价格 / 指标 / 新闻三条链 yfinance 在链首**（无 key 无限流），AV 免费档限流只作链尾；**fundamentals 是有意的例外**（PR5，2026-09，见 `default_config.py` 注释）：`sec_edgar,yfinance,alpha_vantage`——SEC 申报带真实 `filed` 日期，是 PIT 的 ground truth，优于 yfinance / AV 的 `period_end + 45 天`启发式；`config-check` 对"链含 AV 但 key 缺失"打 ⚠。
- **超时默认开**：`YIALPHA_HTTP_TIMEOUT_S` 默认 30s（原 opt-in；`0` 显式关闭）。
- **决策时价格入档**（T0-5）：`full_states_log` 新增 `price_at_decision` / `price_at_decision_basis` / `asset_type`（`_apply_risk_overlay` 结尾挂 `decision.entry_price`；overlay 未跑时 `_log_state` 兜底直调记忆化的 `_latest_close_and_atr`）；旧日志缺字段向后兼容，Web 的 overlay 正则回捞保留。这是评级↔结果验证闭环（verify-history / /api/accuracy）的锚点。
- **perp 价格簿记账升级（2026-09-30 round-12）**：router 哨兵侧与成功侧/`classify_quality` 共用单一严重度源（`quality.sentinel_is_core`）——perp/spot 类目里的价格簿引擎（klines/indicators，非 index qualifier）失败改记 **CORE** kind。此前 `core_sentinel_count=0` 而 tradeability gate 判同一证据 DEGRADED_CRITICAL，run_robust 的 DEGRADED 判定对 perp 核心故障失明；现在纯币 run 的 klines 全挂会像股票 run 一样触发 Trader 入口真空闸门（reject 策略下重跑，而非产出无价格簿的报告）。同批：`_states_log_stem` 对带时刻的 trade_date（`"YYYY-MM-DD HH:MM:SS"` live 入口形态）取日期部——冒号文件名在 Windows 非法，整份 on-disk 证据曾静默丢失；fear_greed / market_regime 的同款双形式日期解析同步修复。

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

**执行层（已移除，2026-11-22）**：原 `yialpha/execution/`（domain/gateway/bridge/binance_gateway/browser_broker 五文件）连同 `tests/test_execution_*.py`、`browser_broker`/`base_gateway`/`pre_trade_risk_check` 测试整体删除——它从未挂进 LangGraph 图、零生产调用方，项目定位收敛为美股+加密的**分析-only**（报告 + 回测 + 票据）。连带清理：`YIALPHA_EXECUTION_{ENABLED,MAINNET,TIMEOUT_MS,LEVERAGE}` 四个 direct-read env 键、`.env.example` 执行段、CI mypy 的 `yialpha/execution` 范围、pyproject 的 execution mypy override、`p0_shadow/guard.py` 前缀黑名单里的 `yialpha.execution`。**保留**：`YIALPHA_ANALYSIS_ONLY` / `YIALPHA_LIVE_EXECUTION_ENABLED`（p0 shadow manifest 快照断言读取这两个 config 键，恒为默认 false/true）、`YIALPHA_KILL_SWITCH`（monitoring dashboard 的 halt banner）、`BINANCE_API_KEY/SECRET`（perp 票据分析路径 `binance_brackets`/`perp_bundle`/`risk/perp_ticket` 的只读数据访问，**不是**下单密钥）。若未来重启实盘轨道，从 git 历史找回（本段之前的版本有完整设计记录）。

| 边界 env | 默认 | 作用 |
|---|---|---|
| `YIALPHA_ANALYSIS_ONLY` | true | 分析-only 边界；true/未设/malformed = 分析模式 |
| `YIALPHA_LIVE_EXECUTION_ENABLED` | false | 历史遗留键：执行层已删，仅 p0 shadow manifest 快照断言其为 false |
| `BINANCE_API_KEY` / `BINANCE_API_SECRET` | 未设 | Binance 只读数据密钥（perp 票据分析用；裸命名，同 `DEEPSEEK_API_KEY`，**无 `YIALPHA_` 前缀**）；只进 `.env`（gitignore），缺 → 部分数据降级 |
| `YIALPHA_KILL_SWITCH` | off | monitoring dashboard 的 halt banner 开关 |

## 运行规范

- **必须先 `cd` 到项目根再跑**：`yialpha/__init__.py` 用 `load_dotenv(usecwd=True)` 注入 `HTTP_PROXY`/`NO_PROXY`/`ZHIPU_CN_API_KEYS`；在别处裸跑会 DNS 解析失败 / 缺 key。
- **跑 ≥1 个 ticker 一律优先 `scripts/run_robust.py`**（per-ticker 独立子进程 + 看门狗 + OS 级强杀重跑），别裸跑 in-process batch——VPN / LLM mid-response 卡死在进程内不可恢复。
  ```bash
  python scripts/run_robust.py --tickers SNDK INTC --date 2026-07-01 \
    --workers 2 --per-ticker-timeout 1800 --max-attempts 3
  ```
- **单 ticker 墙钟 ~8-10 分钟**（flash/旗舰两档分工；DeepSeek 时代实测值，GLM 量级相近）。规划按 ≥10 min/ticker 估。
- **LLM 直连偶发 APIConnectionError / DNS 瞬时失败**属正常抖动：对失败 ticker 单独 `python scripts/run_batch.py --tickers <T> --date <D>` 重跑即可恢复。
- 报告产物：`~/.yialpha/logs/reports/<TICKER>_<ts>/`；逐 ticker 完成态看 `~/.yialpha/logs/<TICKER>/YiAlphaStrategy_logs/full_states_log_<date>[_perp|_spot].json`（完成才落盘，可作进度信号；2026-09-19 起 crypto 两 venue 各一份文件、同日 perp+spot 不再互相覆盖，股票/legacy 无后缀）。

## 测试密闭性与 CI 对齐（round 16，硬性）

- **全套件默认离线**：`tests/conftest.py` 的 autouse fixture `_hermetic_no_network`（类 `NetworkGuard`）在三层拦截真实网络 I/O——① Python socket（requests / urllib3 / 同步 httpx）；② Windows asyncio Proactor 的 `sock_connect`（异步 httpx 走 `ConnectEx`，不经过 `socket.connect`）；③ `curl_cffi` 的 `Session.request`（yfinance 只要装了 curl_cffi 就默认建 libcurl 会话，完全绕开 Python socket）。当场抛 `NetworkAccessError`，并在 teardown 复查记录——**异常被产品的 `except Exception` 吞掉也会失败**。`-m integration` 是唯一的豁免。守卫自身的契约由 `tests/test_round16_network_guard.py` 钉住（含「连到回环地址上的代理那一跳也算网络访问」）。
- **读失败信息**：`test attempted real network I/O (connect to ('127.0.0.1', 8086)) via 文件:行号 函数 <- 调用者 ...`——最内层项目帧就是该 mock 的接缝。先看端口：`8086` 之类是本机代理（请求真的出去了）；Windows 上 `socket.socketpair()` 的回环连接是进程内管道，已放行。
- **写测试的规矩**：(a) 从别的模块 import 进来的名字，在**使用它的那个模块**的命名空间里 patch（`_cik_for_ticker` 是 `sec_ownership` 从 `sec_edgar` 导入的，要 patch `sec_ownership._cik_for_ticker`）；(b) 函数体内惰性 import 的名字 patch **源头模块**（`get_user_selections` 里的 `warm_equity_perp_bases` → `yialpha.dataflows.binance.warm_equity_perp_bases`）；(c) 后台预取线程的取数函数也要 stub；(d) 重试退避的 `time.sleep` 用 `monkeypatch.setattr(netretry.time, "sleep", lambda _s: None)` 打掉，不让测试真等；(e) HTTP 客户端测试自带 `trust_env=False` 或显式代理配置，不依赖机器的代理设置；(f) 夹具里的「绝对路径」必须在每个平台都是绝对路径（`C:/x` 只在 Windows 上是）；(g) 「今天」取产品自己的时钟——`stockstats_utils._utc_today()` 或 `datetime.now(UTC).date()`，不用 `date.today()` / `pd.Timestamp.today()`（yfinance 与 crypto 数据层按 UTC 日历判「今天」，交互式 CLI 的默认日期是本地日历，两者在本地 00:00–08:00 不是同一天）；要「明天」取本地 +2 天（+1 天可能正好是 UTC 今天，仍被当成 live）；能冻结时钟就冻结。
- **测试文件要过 `tests/test_round17_test_hygiene.py` 的 12 个 pin**：它们对 `tests/` 下每个 `.py` 做 AST 扫描——文件必须能解析；没有空测试体或恒真断言；`skip` / `xfail` 必须带 `reason=`；超过 0.05 s 的真实 `time.sleep`、盘符字面量（`C:/…`）、`sys.path` 改动都要在允许清单里带理由（清单是精确集合，双向核对）；pytest 风格的测试、夹具、辅助函数不直接写 `os.environ`（用 `monkeypatch`）；警告过滤不超出作用域；`pyproject` 只忽略 `DeprecationWarning`；测试里触发的 `importlib.reload` 必须撤销——测试前 `saved = dict(mod.__dict__)`，测试后 `mod.__dict__.clear(); mod.__dict__.update(saved)`（`reload` 是在原模块自己的 `__dict__` 里重新执行，「再 reload 一次恢复」不可靠；泄漏会让同一进程里后续做身份比较的测试变红，而且只有全量才看得出来）；先跑的文件不许把会话级对象换成别的对象。新增测试文件写完先跑它（3 秒多）。
- **本机全绿 ≠ CI 全绿**。CI 的测试 job 用 `pip install -e ".[dev,web]"` **无约束解析**，拿到的是当天最新的依赖（实测比 `requirements.lock` 新：pandas 3.1 / langchain-core 1.8 / 三个 LLM SDK 均跨大版本），且没有 `.env`、没有代理、时钟是 UTC，三条泳道里两条是 Linux。「只在开发机验证」让 CI 从 2026-09-18 一直红到 2026-11-23（最后一次成功 `db79063` 之后连续 25 次失败，含 v1.0.0 发布提交）：一类是 22 个测试在打真网络（开发机代理开着就「通过」），一类是 Windows 盘符路径夹具（在 Linux 上是相对路径）。提交前按 CI 条件自测：全新 venv + 无约束安装 + 清空 `HTTP(S)_PROXY` / `NO_PROXY` / 各 `*_API_KEY` / `YIALPHA_*` + `TZ=UTC`，并在 Linux（WSL）上也跑一遍。
- **本机全绿还有第二层：调度器和文件系统的顺序**（2026-11-28）。16 条泳道 0 failed 之后，CI 首跑仍在两条 ubuntu 泳道上红了一条测试，推送前另查出五条只在特定条件下才红。四类写法：① 测试断言任务被工作线程**执行**的先后——产品只承诺结果顺序，Linux 有负载时约 1/5 失败，Windows 从不；② 断言 Rich 输出里某个子串完整——无终端时按 80 列折行，折在哪取决于 `tmp_path` 的长度，GitHub ubuntu runner 的临时目录正落在会折断文件名的区间；③ 断言目录遍历的顺序——`Path.glob` 的顺序属于文件系统（NTFS 排序、WSL 新建的目录是创建顺序、runner 的 ext4 两者都不是）；④ 往工作目录写文件（同一份检出里并行跑两次全量会互相删）。大改动推送前，除了回归矩阵再做三项筛查：(a) 打乱目录顺序的全量——`SHUFFLE_SEED=<种子> PYTHONPATH=tests/_plugins python -m pytest -q -p shuffle_plugin`，换两三个种子（每周的 CI job `tests, scrambled directory order` 用 run id 当种子，失败时照摘要里的命令复现）；(b) 带负载重复全量——多个 pytest 进程并行加占位进程，Linux 上 `taskset -c 0,1` 绑核，**每个进程各用一棵副本树**；(c) 临时目录长度扫描——`--basetemp=<另一种长度的路径>` 跑驱动 CLI 的测试文件。写测试时：不断言执行顺序或平局的胜者（用 `sorted(...)` / `in {a, b}`，或用 `Event` 把「已在跑」变成事实）；要断言输出里的完整路径，先 `monkeypatch.setenv("COLUMNS", "400")`；临时文件只写 `tmp_path`。
- **scipy 是「可选加速器」但测试需要它**：`backtest/{metrics,ic,event_study}.py` 与 A/B gate 带 numpy/math 回退；钉「回退 == scipy 到 1e-12」的测试 `importorskip("scipy")`，所以 scipy 在 `[dev]` 里（运行时安装不变，它此前从未被声明，那 8 个等价性测试在 CI 上一直被静默跳过）。涉及 scipy 的测试要同时断言回退分支（monkeypatch `_HAVE_SCIPY=False`），否则干净安装上那部分永远不被执行。

## 环境（已验证可用，别再重复诊断）

- 当前 LLM = **DeepSeek 直连**（2026-09-30 实测：key 有效、`deepseek-flash` / `deepseek-v4-pro` 在 served 列表、chat 端点回话正常）；`NO_PROXY=api.deepseek.com,...`。GLM 四把 key + coding plan 端点保留在 `.env` 备切回（见「模型偏好」）。
- yfinance / 行情 / 财报必须经 VPN 代理（FLASH-CAT）。**双端口实况（2026-09-30）**：机器级用户环境变量 `HTTP_PROXY/HTTPS_PROXY=http://127.0.0.1:8086`（load_dotenv 不覆盖已设值，故实际出口常是 8086 而非 `.env` 写的 `socks5h://127.0.0.1:1080`）；两端口均为活代理，1080 实测 yfinance 可拉数。代理一断 → 数据抓取永久挂起（preflight 的 yfinance ❌ 先怀疑代理没开）。
- **Windows 控制台是 GBK(cp936)**，打印 ✅/❌ 会 `UnicodeEncodeError`；入口脚本顶部须 `sys.stdout.reconfigure(utf-8)`，兜底用 `PYTHONUTF8=1`。
- Reddit RSS 429、`FRED_API_KEY not set` 是**非致命降级**，不影响评级输出，不用装 FRED key。

## 验证路线（`scripts/run_baseline.py`）

1. `--preflight`（零 LLM 成本自检）
2. `--smoke`（1 票 1 日，跑通整条 LLM 图）；加 `--profile` 同跑且打印「节点→墙钟占比」遥测表（零影响，见上「性能 / 遥测层」）
3. `--baseline`（基线回测）
4. `--full`（基线 vs 风控 A/B + 闸门判定）——闸门 PASS 才做券商适配

分析师并行的分布等价性 gate 走 `scripts/run_analyst_parallel_ab.py`（独立于上述回测路线，见上）。
