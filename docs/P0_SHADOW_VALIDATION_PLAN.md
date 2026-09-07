# P0 独立 shadow 真实验证方案

准备日期：2026-09-06。状态：**A 阶段方案交付；B0 离线准备已于 2026-09-07 实现并通过离线验证；B1/B2/B3 尚未执行**。
本文件依据本地代码静态审阅编写；没有访问供应商、创建真实预测、评分真实样本或运行 LLM。

**B0 完成记录（2026-09-07）**：第 2、3 节"待实现"的 runner、路径 guard、原始响应 recorder、manifest 和逐行审计
已实现于 `scripts/p0_shadow/`（guard.py / artifacts.py / audit.py / runner.py）及 CLI 入口 `scripts/p0_shadow_validation.py`
（子命令 `offline-demo` / `inspect`，仅离线 fixture 与只读检查，无真实采集入口）。
端到端离线验证：baseline 场景 12 条预测 → 11 complete + 1 incomplete（U-MU 周末 1d 同根终态）；
missing-funding 场景 → 8 complete + 4 incomplete（BTC 三期限 7d 截止终态 + 周末同根）。
本轮修复了 B0 实现中的 6 处缺陷（见 P0_TIME_CONTRACT.md 2026-09-07 补充）。
20 个测试文件离线回归 **381 passed**，网络拦截计数 0；Ruff 与 4 个 B0 模块限定 mypy 通过。
B0 完成不自动启动 B1；B1 开始前仍须用户明确授权。
适用契约见 [P0_TIME_CONTRACT.md](P0_TIME_CONTRACT.md)，阶段授权边界见
[P0_NEXT_SESSION_PLAN.md](P0_NEXT_SESSION_PLAN.md)。此前 316 项、本轮 338 项离线通过记录均不能替代本方案的真实证据。

## 1. 验证目标与不变边界

验证形成时冻结参考快照、1/5/21 自然日固定期限、到期数据与费用归因、不可评分终态、幂等和汇总来源是否可追溯。
全程 `analysis_only=true` / shadow，业务要求 `live_execution=false` 对应现有配置
`live_execution_enabled=false`；禁止执行交易、调用 LLM、运行完整分析图、重跑原验收。
不读写 `.env` 或 `.env.enterprise`；不得通过包初始化间接读取。
不提交、推送、部署，不进入 P1/P2/P3，不建立自动化或后台轮询。

该口径是**日线参考收益基准**。窗口可能包含预测形成前的部分时间，费用为 shadow 成本模型；
不得描述为提交后可执行策略收益、真实成交损益、预测有效性或盈利性验收。
通过标准允许有理由且证据充分的 `incomplete`，不追求全部 `complete`。

原目录 `analysis_output/shadow-glm-round6-20260905`、
`analysis_output/shadow-glm-round6b-20260905` 保持原样。
旧 round6b 的 24 条盘中 CONTRACT 无冻结参考快照，等待不能修复历史可知性；本方案不复制它们充当新样本，
不重建预测时快照，不覆盖任何已有 outcome，不提供复评入口。

## 2. 分阶段推进与当前缺少的入口

| 阶段 | 具体工作 | 阶段交付与下一步条件 |
|---|---|---|
| B0：离线准备实现 | 实现下述独立 runner 的路径 guard、证据捕获、清单、只读报告；使用模拟数据、临时账本、网络阻断检验 | 入口代码和离线结果可审阅；当前 A 阶段只列需求，未实现这些脚本 |
| B1：形成时真实快照 | 在实际形成时间创建流程样本，保存同次取价的原始证据、冻结 timing、成本假设和 ID 映射 | 单样本及矩阵快照审计通过；只做 B1 不顺带启动到期评分或 LLM |
| B2：自然到期归因 | 在各真实到期时点之后手动运行评分，记录供应商证据、实际窗口、归因与状态 | 分别交付 1d、5d、21d 真实观察结果；存在补数需求时继续到固定截止 |
| B3：生命周期与汇总收尾 | 手动补数观察、终态/幂等核验、独立 scoreboard 和拒绝清单 | 原件保护成立，结果可复核；未发生的故障分支明确标为仅离线验证 |

后续只有明确安排的阶段才执行，已经明确授权的同一阶段无需重复确认。
B0 完成并不自动启动 B1；B1 完成也不自动创建长期监控。
新样本 21d 必须自然等待；必要数据仍缺失时，观察可到形成后 28 天。

当前已有可调用的 Python 接口如下；它们是 runner 的构件，**不是已存在的安全验证命令**：

| 已有接口 | 用途及需要补的外围能力 |
|---|---|
| `yialpha.dataflows.config.set_config/get_config` | 进程当前上下文配置；必须显式设置并核验绝对路径，不能依赖环境默认值 |
| `ledger.evidence.register_run`、`ledger.run_context.set_ledger_run_context` | 创建并绑定独立 run；绑定上下文本身不创建 run 行 |
| `agents.utils.prediction_tools.begin_prediction_capture`、`make_submit_prediction_tool`、`flush_predictions` | 用固定输入调用真实 submit 工具路径，由确定性行情代码取价；不能自行拼 timing 冒充该路径 |
| `ledger.time_contract.capture_prediction_timing` | 真实参考捕获；只存数值、时间、来源标记及错误，没有供应商原始响应存储 |
| `ledger.evidence.record_evidence` | 存证据 metadata 和 payload SHA-256，不保存 payload 正文；必须另存原始文件 |
| `ledger.tickets_mirror.attach_ticket` | 写入 shadow 费用分解镜像；失败会被日志捕获，需要实际读回核验 |
| `ledger.outcomes.pending_predictions` | 只读待评分队列；任意已有 outcome 均退出自动队列 |
| `ledger.outcome_compute.compute_outcomes(now_as_of, limit=...)` | 会访问供应商并追加 outcome；必须经 B2 专用入口且采用当时实际 UTC 时钟 |
| `ledger.scoreboard.build_scoreboard(scoring_version=...)`、`render_scoreboard_markdown` | 只读汇总；需另加源账本指纹、行级纳入/排除审计及结果文件存储 |

现有 CLI `scoreboard` 在渲染前会调用 `compute_outcomes`，因此不能拿它做 B1 只读检查。
仅指定 `--results-dir` 不会隔离 ledger 或 cache。当前没有同时满足本方案保护、原始证据和清单要求的已确认 CLI；
不提供未经实现的脚本名、参数或可复制执行命令。

## 3. 独立目录、账本与入口 guard

后续真实 cohort 使用一个全新目录，形如
`C:\Users\warri\Desktop\Yialpha\analysis_output\p0-shadow-validation-<实际UTC开始时间>-<唯一ID>`。
尖括号表示待 B1 开始时生成的值，本阶段不创建目录。
每个 cohort 的计划结构：

| 路径（相对 cohort 根目录） | 内容 |
|---|---|
| `manifest.json` | cohort ID、流程样本标记、允许标的/范围、实际启动 UTC、代码与依赖版本、配置摘要、路径白名单 |
| `ledger/portfolio.db` | 唯一真实流程验证账本；SQLite WAL/SHM 文件也必须留在此目录 |
| `cache/<attempt_id>/data/` | 每次采集或评分进程独立的项目供应商 cache，避免旧结果掩盖重试数据状态 |
| `cache/<attempt_id>/yfinance-internal/` | yfinance 自身 timezone/cookie/ISIN cache；内部 cookie 不作为报告证据发布 |
| `raw/<request_id>/` | 同次请求响应、规范化帧、请求参数、响应摘要、UTC 观测时间及 SHA-256；不可覆盖 |
| `snapshots/<prediction_id>.json` | 账本 timing 原值、entry 原始证据关联、提交输入、接受/flush/落账观察记录 |
| `costs/<run_id>.json` | 预先冻结成本假设、费用四项、来源/版本/形成时间、ticket ID 与 payload hash |
| `attempts/<实际UTC>-<唯一ID>/` | 单次队列前后快照、评分明细、数据缺口、手算归因、运行日志和导出表 |
| `reports/<attempt_id>/` | scoreboard JSON/Markdown、源账本清单、行级纳入/排除原因、人工核对结论 |
| `checkpoints/<attempt_id>/` | 已关闭连接后的数据库校验或 SQLite 一致备份及其 hash；不直接复制活动 WAL 数据库 |

离线故障注入使用系统临时目录下的另一套账本和模拟证据，不写入上述真实 cohort。
未来真实研究/LLM 样本使用第三套独立目录和账本；默认 scoreboard 聚合当前账本所有合格行，
仅靠备注不能防止流程样本与研究样本混入。

B0 必须实现并验证以下启动顺序：

1. 使用项目 `.venv-dev\Scripts\python.exe` 启动全新进程，保留实际 UTC 时钟。
   在导入任何 `yialpha` 模块之前，将 `dotenv.find_dotenv` 和 `dotenv.load_dotenv` 替换为不读取文件的函数；
   启动审计同时拒绝 `.env` 家族文件的读取和写入，不输出环境变量全集或密钥。
2. 先解析并验证本次 cohort 根目录及所有目标的绝对规范路径。拒绝空值、默认目录、受保护目录、目录穿越，
   拒绝经符号链接/junction 解析后跑出根目录；不能用字符串前缀代替路径包含判断。
   cohort 创建不得覆盖已有目录；重试必须提供 manifest 中已有 cohort ID 并核验路径一致。
3. 设置当前配置：`ledger_db_path`、`data_cache_dir`、`results_dir`、`memory_log_path` 均指向独立根下；
   显式强制 `analysis_only=true`、`live_execution_enabled=false`、`llm_cache=false`。
   runner 只调用上述窄接口，不运行图、执行适配器、模型工厂或 memory 写入流程。
4. 在首次 yfinance 访问前调用本机已存在的 `yfinance.set_tz_cache_location` 重定向内部 cache。
   本机实现同时设置 timezone、cookie、ISIN 三类存储；只有 `data_cache_dir` 隔离不充分。
   每次运行使用新进程，避免 Binance 历史进程内 memo 跨 attempt 混用。
5. 第一条 SQL 或供应商请求之前，读回 `ledger_db_path()`、`cache_base_dir()` 和配置，核验所有写入目标。
   后续写入也要受根目录 allowlist 约束：SQLite、证据、CSV cache、临时文件和报告均不得落到
   `C:\Users\warri\.yialpha`、项目默认输出或 round6/round6b。
   初版 runner 使用单线程；以后若加入线程，必须传播 ContextVar 配置并逐 worker 核验，不靠父线程配置推断。
6. B0 使用临时路径和模拟供应商验证 guard 失败时在网络和账本写入之前终止；网络阻断需覆盖 HTTP 客户端及底层连接，
   特别核验 yfinance 使用的传输层，不把仅替换 `socket.socket.connect` 当成所有第三方传输均被拦截的证明。
   B1/B2 只开放本阶段所需公开行情来源，网络权限不包含交易或模型调用。
7. 每次保存代码快照标识：HEAD、dirty diff hash、新增文件 hash、Python/相关依赖版本、允许配置白名单及摘要。
   不用保存完整环境配置来换取可追溯性。开始/结束核对保护目录清单；不将新基线冒充历史 611 文件基线。

以上 guard、请求 recorder、manifest writer 和行级审计目前仍是**待实现/待离线验证**的 B0 工作。
已有生产时间契约不因这些外围准备需求而重做。

## 4. B1 样本矩阵与形成时证据

最小主矩阵为 12 条 horizon 记录。每个 instrument/scope cohort 使用自己的 run，
同一 run 的三个期限可在一次真实 submit 中同时接受，因此共享那次冻结快照；不同 run 的形成时间不能强行一致。

| run 分组 | 注册/路由身份 | capture instrument / scope | 期限 | 必须观察的结果 |
|---|---|---|---|---|
| C-BTC | BTCUSDT，`crypto_perp / pure_crypto_perp` | BTCUSDT / CONTRACT | 1、5、21d | 合约精确日线、真实资金费和单次成本归因 |
| C-ETH | ETHUSDT，`crypto_perp / pure_crypto_perp` | ETHUSDT / CONTRACT | 1、5、21d | 同上，独立来源证据 |
| C-MU | MUUSDT，`crypto_perp / stock_perp` | MUUSDT / CONTRACT | 1、5、21d | 24/7 合约日线及实际资金费频率，不套股票周末规则 |
| U-MU-WE | MUUSDT，`crypto_perp / stock_perp` | MU / UNDERLYING | 1、5、21d | 实际 UTC 周六或周日形成；1d 同根终态，5d/21d 验证精确股票终点和复权一致性 |

标的分类及上市状态在 B1 由当时公开供应商资料留证；表中是计划覆盖身份，不是本阶段已验证的当前供应商状态。
MU CONTRACT 和 MU UNDERLYING 分 run，避免 scorer 按 run 选择最新 ticket 时串用两种成本。
若不能在实际 UTC 周末运行，则 U-MU-WE 等下一个实际周末；不能把当下调用回填到 9/5。
每条 `horizon_end = timing.prediction_formed_at + horizon_days`，不以运行登记、研究截止或文件名时间代替形成时钟。

固定输入在 B1 前写进 cohort manifest，并明确 `sample_kind=process_validation_fixed_input`、`llm_used=false`。
可使用固定 `prob_up=0.5`、固定方向和可通过现有校验的固定 analyst 名称；这些取值没有研究含义。
不增添预测 schema 字段，不伪装为分析师结论；样本标记存外部 manifest 和报告，映射全部 run/prediction ID。
不开展 Brier、方向胜率等指标的能力解释，即使 scoreboard 能计算数值。

B1 每个 run 的实际步骤：

1. 登记实际 `analysis_as_of` 和 `runs.created_at`，绑定 run context，冻结该 scope 的 shadow 费用方案并镜像 ticket。
2. 开启同次请求 recorder，调用 `begin_prediction_capture`，通过 `make_submit_prediction_tool` 的实际工具调用提交固定 horizon 输入。
   recorder 旁路保存真实调用收到的结果，不替换供应商数据、不修改生产时钟、不自行构造参考价。
3. 保留 submit 返回文本，并核验接受的 horizon。接受文本中的 committed 不能单独证明 SQLite 写入；
   调用 `flush_predictions` 后读回全部 3 条预测、timing、run/ticket 关联和行数。
   明确 `prediction_formed_at` 早于或等于实际落账，延迟 flush 不能重新取价或重置形成时间。
4. 在同一 capture 中重复相同 submit，核对未增加取价请求、未变 timing；重复 flush 后行数和内容保持不变。
   冲突提交、强制异常或时钟调整留给离线临时账本，不在真实主矩阵注入。
   同 horizon 并行相同/冲突重交及接受期间 flush 已有确定性离线回归；B0 必须保留这一覆盖，
   B1 单线程 runner 不把未发生的并发分支描述为真实观察。
5. 保存读取的账本原值及请求证据链接。错误快照同样留证；不能为得到完整矩阵而删除它、补写旧参考或降回 legacy。
   若另建替代样本，以新 run、新实际形成时间追加，并披露原失败样本，不选择性隐藏失败率。

每个价格快照的最小证据包：

| 证据 | 保存和核验内容 |
|---|---|
| 请求身份 | source/venue、canonical symbol、endpoint、interval、price_type、请求起止日期、`closed_as_of`、请求 ID、参数摘要 |
| 原始与规范化内容 | 同次请求的响应正文/行、供应商时间字段、价格字段、规范化 DataFrame/CSV、二者 SHA-256 和解析版本；不把事后另一次查询冒充形成时原始响应 |
| 本地 UTC | 请求开始、响应收到、规范化完成/本地观测、预测接受、flush、账本 created_at；每项带时区，另记主机时钟异常或校准证据 |
| 冻结 timing | `version`、`prediction_formed_at`、`reference_price`、`reference_price_at`、`reference_available_at`、`reference_observed_at`、`reference_source`、`reference_error` 全字段 |
| 时序判定 | 有效价格有限且大于零；`price_at <= available_at <= observed_at <= formed_at`；精确应有日线仅一条且已闭合，不用更老日线补位 |
| 关联与来源 | cohort/run/prediction ID、原始证据文件 hash、可用 evidence ID、代码/依赖版本和缓存命中状态 |

CONTRACT 当前路径为 `binance_klines_frame(... interval="1d", venue="binance_perp", price_type="last", closed_as_of=...)`，
冻结来源 `binance_perp:1d:last`；需保存 kline open time、close time 和 Close，核验 D 日线的保守可知界限为 D+1 00:00 UTC。
Binance 的实际 closeTime 可能表达毫秒级闭合边界，报告保留原值和转换规则，不能直接将两种时间文字当成矛盾或相等。

UNDERLYING 当前路径为 `get_YFin_history_cached`，冻结来源 `yfinance:1d:close`；
保存原时区索引、转换后日期、Close、公司行动字段及本机 yfinance 版本和 `history` 的实际默认参数。
代码没有显式传入 `auto_adjust`，因此不能把当前 Close 写成已证实的未复权成交收盘价。
框架返回的 CSV 是规范化证据，不能标为 Yahoo 原始 HTTP 响应。
若 recorder 无法保存供应商原始响应，要明确证据层级和缺口，不能宣称原始响应审计通过。

`reference_available_at` 当前等于保守日线闭合时刻，是契约规则，**不是供应商历史发布时间的实测值**。
真实响应收到时间能够证明此次观测时已有此数据；如果供应商未给历史发布时刻，保留“未提供”并披露规则来源，
不将本地下载时间或推测发布时间回填为供应商证据。
`reference_source` 字符串或 evidence payload hash 本身也不能证明取价来源，必须可追到同次请求保留内容。
缓存命中需要关联原始生成证据、mtime/年龄和 stale warning；B1 新 attempt cache 默认空，不能继承现存真实 cache。
证据 PIT 校验必须使用字段真实语义，不能把晚取得的新观测记录伪装成研究截止前形成的证据来绕过 gate。
无法合法挂到 prediction.evidence_ids 的观测日志保留外部审计关联，并明确它与预测输入证据的区别。

## 5. 费用 provenance 与到期独立归因

无交易的 shadow 流程不能声称有真实 fill、佣金账单或实测成交滑点。
费用方案在预测形成前冻结，至少含 `entry_fee_bps`、`exit_fee_bps`、
`entry_slippage_bps`、`exit_slippage_bps` 四项，均有限且非负；注明单位为 bps，
来源属于公开费率表、已有版本化成本模型参数或明确的流程测试假设。
公开资料写来源 URL、检索 UTC、适用市场/费率档/假设；模型或假设写版本、决定时间、参数和依据。
没有可验证账户费率时不用账户费率措辞；滑点只作假设，不补称真实市场成交证据。

保存 `COST_MODEL_VERSION`、`TICKET_VERSION`、ticket ID、原 payload hash 和全部分解。
每 run 固定一个 ticket，形成后不追加更晚 ticket 改变 scorer 的“latest ticket”选择；
三个 horizon 使用同一明确成本口径，各自对一个假设持有窗口只计一轮 entry/exit，不能按持有天数放大。
只有 `estimated_cost` 总额不满足新口径要求；不能以零成本默默补齐，不能将 expected funding 再计入成本。
费用缺口自然出现时按生产 pending/7d 规则留证；人为制造费用缺失只属于离线测试。

B2 对每条预测独立核算并与账本比较：

| 项目 | 核验规则 |
|---|---|
| 固定期限 | 使用原 timing 的 formed UTC 加 1/5/21 个自然日；调用时 `now_as_of` 取真实当时 UTC，不传未来时钟 |
| entry | 使用冻结 reference_price，不重新选价；entry 时间等于形成前应有的精确日线保守闭合时间 |
| endpoint | CONTRACT 为 horizon_end UTC 日期前一天日线；UNDERLYING 再退到最近周一至周五；日线必须精确存在，不以更老日线替补 |
| 实际窗口 | `window_start=reference_price_at`；`window_end=endpoint_day+1 00:00 UTC`；记录名义期限与实际窗口之差 |
| 价格收益 | `exit_price / frozen_reference_price - 1`，保存两个数值与对应证据 |
| 资金费 | CONTRACT 仅累计 `(window_start, window_end]` 内真实结算 rate；`funding_pnl=-sum(rate)`，方向无关的 long-pay 基准 |
| 费用 | `fees=(entry_fee_bps+exit_fee_bps)/10000`；`slippage=(entry_slippage_bps+exit_slippage_bps)/10000`，分别只计一次 |
| 校准标签 | `net_return=price_return+funding_pnl-fees-slippage`；UNDERLYING 无合约 funding leg；策略方向 view 不替代该标签 |
| MU 诊断 | underlying/basis 只用同一精确端点；CONTRACT 的 underlying 诊断缺失可披露，不拼其他窗口且不自动否定完整核心归因 |
| 股票基准 | 当次取回原参考日 Close 与冻结价在 `rel_tol=1e-9, abs_tol=1e-12` 下匹配；不替换冻结价；变化按 `equity_reference_basis_changed` 终态 |
| 结果可用时间 | 新口径严格要求 `outcome_available_at == window_end`；另存本次完整价格/资金费/费用证据观测和落账时间，不能当作整份结果实际发布时刻 |
| 重复日线 | 所需终点或股票原参考日 Close 矛盾时 `price_bar_conflict:日期`，进入 pending/7d incomplete；同值重复可折叠，无关回看日期冲突不阻挡窗口；保留原始顺序便于审计 |

资金费证据包必须保留 `/fapi/v1/fundingRate` 各分页请求和原始行、CSV、全部原始 fundingTime 毫秒值、
窗口内去重后的结算表、窗口前的网格锚点、排除的窗口外行以及求和结果。
相同时间相同 rate 只算一次；相同时间冲突 rate 必须拒绝；检查首段、中段、尾段预期 slot，不只看总条数。
同时保留供应商返回的 `/fapi/v1/fundingInfo` 信息及观测 UTC；没有返回该标的、请求失败、推断回退必须区分记录。

静态代码边界：供应商 CSV header 会披露 fundingInfo 或观测间隔；生产 scorer 自己按收到的结算时刻众数推断 cadence，
并不消费 header 的权威频率。**众数不能单独证明历史频率未变或每条应有结算都齐全**。
手工独立核验需查明该实际窗口的结算安排及变更，尤其 MUUSDT；当前 fundingInfo 不能自动证明整个过去 21d 的频率。
遇到频率切换或有序缺失但 scorer 仍给 complete，应标为验证差异、停止该结果的验收结论并记录最小复现线索，
不能修改真实原始数据或将其直接宣称为归因通过。修复/复评是具体缺陷的后续安排，不覆盖已写 outcome。

## 6. 生命周期、人工检查时点与真实/模拟证据分栏

B1 成功落账后生成每条记录的真实 `formed_at`、`horizon_end`、`retry_deadline=horizon_end+7d` 检查表。
只在人工启动时运行；检查表不是已创建的日程、提醒或自动任务。
自然到期前可只读查看队列，不人为推进时钟。到期后首次人工评分必须记实际运行 UTC，
未能恰好在到期瞬间运行时诚实记录延迟，固定窗口仍保持原值。

| 情况 | 真实主矩阵的期望 | 证据与范围 |
|---|---|---|
| horizon_end 之前 | 不进入到期队列，不写 outcome | 实际时钟、预测原 timing、队列快照 |
| MU 周末 1d 同根 | 到期后 `incomplete / no_new_trading_session`，不等周一改端点 | 形成与终点应有日线均为同一周五；规则证据，不要求再拉股票行情 |
| 形成时 reference_error | 到期评分为 `incomplete / reference_unavailable_at_prediction...`，不能靠事后行情补回 | 原错误快照和真实供应商错误记录 |
| 终点/资金费/费用未齐或可补数供应商异常 | 到期后、retry_deadline 前为内存 pending，outcomes 表零新增 | 保存 attempt 明细的 reason/legs_missing 和队列；不能只看 outcomes 表误认为未处理 |
| 所需日线存在冲突 Close | `price_bar_conflict:日期`，适用 7d 补数 | 不按输入顺序挑最后一条；真实出现才标为 real_observed，人为制造只能离线注入 |
| 至 retry_deadline 仍缺必要数据 | 在截止之后下一次实际人工调用写 incomplete，并退出队列 | 记录实际调用延迟及持久化原因；没有后台调用就没有自动写入保证 |
| 股票原参考日证据缺失 | `equity_reference_basis_unverifiable`，适用 7d 补数 | 原参考与本次取价范围/缺口证据 |
| 股票参考价复权变化 | `incomplete / equity_reference_basis_changed`，不自动换算公司行动 | 两次原始价、容差计算及可用公司行动资料 |
| 已有 complete/incomplete（或历史 pending） | 下次队列不再纳入，已有行不变 | 前后 logical rows 和 ID/hash 比较；本方案不在真实主矩阵人工插历史 pending |
| 无 complete 的头页、单样本 failed | 后续样本仍可处理，failed 行隔离 | 真实发生则记录队列顺序/limit/明细；未发生则引用离线证据，不制造线上故障 |

当前计算报告不持久化 pending 的完整 scoring_context；runner 需从原 timing、请求证据和本次 detail 补全 attempt 审计，
明确哪些字段来自账本、哪些是验证器派生值，不把派生值冒充已落账 metadata。
`compute_outcomes` 在遇到有 complete 的一页后结束本次调用；后页仍可能有已到期样本。
每次报告列出队列剩余 ID，必要时在同一已授权人工检查中继续调用；若只剩 pending/failed 且没有可推进记录就停止，
不创建忙循环，不以“本次未报错”代替全矩阵处理结论。

真实 B1/B2 的取价、自然期限、同根终态、实际幂等与供应商缺口属于 `real_observed`。
丢弃数据行、伪造重复/冲突率、冻结或跳跃时钟、强制供应商错误、造旧 pending、造复权变化和队列异常，
只在另一个临时账本以模拟数据运行，属于 `offline_fault_injection`。
如果真实期间没有 pending 或持续 7d 缺数，报告应写“未自然观察，离线逻辑已覆盖”，
不能将离线跳时钟写成真实 7d 生命周期通过，也不为覆盖率污染主矩阵。

重复检查采用真实时间再调用：已有 terminal outcome 应退出队列，原 prediction/ticket/outcome 内容 hash 与数量不变；
pending 的下一次返回可因真实新数据而变化，但仍不覆盖旧行，因为它此前没有 outcome。
outcome writer 的冲突 metadata 改写拒绝在离线临时账本验证；真实账本只做无冲突重放。

## 7. Scoreboard、交付清单与退出条件

每次 scoreboard 调用必须在已经核验路径的独立进程/上下文中进行，使用
`build_scoreboard(scoring_version="close_reference_v1")` 生成当前 cohort 新口径视图。
同时保存未筛选视图作版本核对：本真实新样本账本不应有 legacy 行；如出现则作为来源污染调查。
`legacy_daily_v1` 与 `close_reference_v1` 分桶、混合版本警告通过已有离线临时账本测试验证，
不为展示混合版本向真实 cohort 填入旧 round6b 数据。

默认 scoreboard 仅显示合格 complete 的聚合，没有源账本路径、snapshot hash 或逐行拒绝清单。
待实现 runner 必须另附：

- 源账本绝对路径、cohort ID、schema_meta 版本、代码 HEAD/dirty fingerprint、取数 UTC、
  一致备份或关闭连接后的 DB hash；SQLite 活动 WAL 状态下不能只 hash 主 DB 声称代表完整源数据。
- 总 predictions/到期数、内存 pending、持久化 complete/incomplete、failed、尚未到期、队列剩余 ID；
  区分“状态 complete”和“scoreboard 合格 complete”。
- 每条预测的纳入/排除结论及证据：timing 原值、scoring_context、缺失腿、finite 校验、net 重算、
  版本、对应原 snapshot 匹配、原始请求和 ticket hash。scoreboard 目前静默过滤不合格行，
  不得把不存在的内置拒绝 reason API 当已提供；验证器需输出自己的规则标识并与生产资格判断核对。
  新口径还必须核对可用时间等于固定 window_end、价格类 fees/slippage 非负；不能通过重算 net 掩盖负成本。
- `SCHEMA_VERSION=v1`、SQLite 内部 schema 4、`FEATURE_VERSION=v2`、`COST_MODEL_VERSION=v1`、
  `TICKET_VERSION=v1`、prediction/outcome `close_reference_v1` 分别记录，不能混为一个 schema 版本。
- 分析标题和每张导出表标记“固定输入流程验证，非预测能力评估”；不得与真实研究样本合并解释校准或胜率。

阶段结果采用“通过 / 有明确原因不可评分 / 验证差异 / 未自然观察 / 尚未到期”，
同时保留真实 status；验证报告状态不修改生产 outcome。

退出条件：

1. 12 条主矩阵记录均在真实当时形成并保留可追溯快照或明确失败证据；未达到的覆盖项列为未完成，不能用回填补齐。
2. 1/5/21d 已自然到期并被实际检查；仍缺必要数据者已自然到固定 7d 截止后的实际检查，或诚实标明尚待观察。
   每项具有真实终点/资金费/费用来源核验或不可评分原因；出现生产计算与独立证据差异则不得宣告该项通过。
3. 同根终态、实际幂等、队列和 scoreboard 来源可复核；未自然发生的故障分支明确只具离线证据。
4. 保护目录和现有历史账本未变化；未执行交易、LLM、原验收、P1/P2/P3、自动化、提交或部署。
5. 报告不以全 complete 为目标，不将日线基准、模型费用或固定输入概率描述成实盘表现。

之后若开展真实研究预测积累和校准评估，需要独立安排研究样本来源、预先确定指标、样本外划分、
模型/数据/配置版本和失败样本统计。本方案结束不等于这项工作已开始或已完成。
