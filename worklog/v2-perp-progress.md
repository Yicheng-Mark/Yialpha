# V2 永续专项进度锚点（V2.1→V2.4）

> 用途：主智能体上下文压缩后从此文件恢复状态。每批完成即更新。
> 基线：`adec679`（PR1–PR7+修复批，用户已提交）。每版三绿后 commit + tag。

## 总体路线（已获用户批准 2026-09-03）

- **只动永续**：一切新能力以 `asset_type == "crypto_perp"` 为门；stock/crypto/crypto_spot 路径零行为改动，byte-compat 测试钉死。
- 版本顺序：V2.1 Measurability（record/shadow）→ V2.2 Context → V2.3 Specialization → V2.4 Portfolio Control。
- 执行协议：子智能体按批次实现（不重叠文件可并行），主智能体管契约/审阅/高危接线/门禁/commit+tag。
- 门禁命令：`python -m pytest -q` / `python -m ruff check .` / `python -m mypy --no-site-packages --follow-imports=skip <scoped modules>`（对齐 CI）。

## 批次状态

| 批次 | 内容 | 状态 |
|---|---|---|
| 前置 | versions/config/sqlite.py/包骨架/conftest 隔离/worklog | ✅ 完成（ruff 绿、33 tests 切片绿、账本冒烟通过） |
| V2.1-A | instruments 包 + routing/binance 集成 | ✅ 完成（58 新测试；主智能体复核 98 tests + ruff 绿） |
| V2.1-B | ledger 包 models/evidence/predictions/outcomes | ✅ 完成（38 新测试；主智能体复核 + mypy 14 文件绿） |
| V2.1-D | perp 包 quote_fx + fair_value + tickets/schemas 字段 | ✅ 完成（25 新测试；主智能体复核 44 tests + ruff 绿） |
| V2.1-C | submit_prediction 工具 + 分析师 evidence 接线（依赖 A/B） | ✅ 完成（28 新测试 + 63 byte-compat；主智能体复核绿） |
| V2.1-E | compute_outcomes + scoreboard CLI（依赖 A/B） | ✅ 完成（23 新测试；主智能体复核绿） |
| V2.1-F | run_id 穿链/ticket linkage+mirror/bridge 接线/Web API（主智能体） | ✅ 完成（9 新测试；fair_value basis 改诊断项 16 测试） |
| V2.1 门禁 | 全量三绿 + byte-compat → commit + tag v2.1.0 | ✅ **2819 passed / ruff clean / mypy 159 文件 clean** |

## V2.2 Context（shadow 默认）— ✅ 完成（2026-09-03，2844 passed / ruff clean / mypy 160 文件 clean）

批次：单一大子智能体（无并行冲突，避免接口错配）。冻结决定：
- 新包 `yialpha/regime/`（state.py：RegimeState + regime_id=sha256(REGIME_VERSION+canonical inputs) 确定性哈希 + render 块；compute.py：perp 共用/纯币/股票永续三组字段，只用既有 PIT seam）
- **迁移 v2**：新表 regimes（regime_id PK + payload + regime_version + analysis_as_of + computed_at）；predictions/outcomes 各加 regime_id 列（additive ALTER）
- 注入：market analyst（仅 perp run）evidence 消息（flag `regime_state` 生产默认开/conftest 关）；ticket.regime_id 填充；predictions 提交时带 regime_id（run context 扩展或 prediction_tools 传递）
- REGIME_VERSION→v2；scoreboard 增 by_regime 切片；vol_estimators 年化按 instrument_class 走 sessions.py（遗留项收口）
- 历史回放仅 PIT 源；不可得→regime_id=None+披露（不泄漏）

### 18. V2.2 落点（2026-09-03）

- **RegimeState 契约**：regime_id = "G"+sha256(REGIME_VERSION + canonical sorted JSON(全部输入字段，剔除 regime_id/regime_version/computed_at))[:12]；computed_at 剔除是幂等关键（同输入重算同 id → INSERT OR IGNORE 去重）；missing_inputs 参与 preimage（缺腿的 regime 是**另一个** regime，不与完整版合并）。全数值/分类输入缺失 → 返回 None（披露"regime unavailable"，绝不伪造 id）。
- **分类器冻结**（REGIME_VERSION v2，详见 versions.py/state.py docstring）：trend=close vs SMA50/200（both-above=up/both-below=down/else range，需 200 行）；realized_vol_pct=20 日收益 stdev（ddof=1，百分比，**不年化**）；funding_pct=7 日结算净和；oi_pct=30 日窗口分位；stress 触发=|funding|≥1%/7d ∨ |basis|≥50bps ∨ |gap|≥100bps ∨ OI≥90pct+1d+10%；depth=spread≤2/≤10bps + ±50bps notional ≥$2M/≥$500k；session=NYSE-equivalent ET 桶（date-only=工作日 regular/周末 closed，DST 用手写 2nd-Sun-3月..1st-Sun-11月 规则，免 tzdata）。earnings_calendar/sector_index 为诚实缺口（None+missing，不造数）。
- **历史 PIT 过滤**：`is_historical_date(end_date)` 为真 → 深度等 LIVE_ONLY 腿**不 fetch 不泄漏**（missing_inputs 记 "depth_bands"），只用 klines/funding 历史/OI/LSR/taker 窗口（30 日保留期外自然降级为 missing）。market analyst 注入 replayability：历史=PIT_REPLAYABLE / live=LIVE_ONLY（镜像 bundle 契约）。
- **迁移 v2**：ALTER 无 IF NOT EXISTS → 每个 ALTER 独立小事务 + "duplicate column name" OperationalError 吞掉（并发竞态败者视为成功；独立事务保证 duplicate 错误不会回滚 CREATE TABLE）。schema_meta='2' 最后写。
- **穿链**：LedgerRunContext.regime_id 加性字段（set_ledger_run_context 同名 keyword，嵌在 prediction_ledger 门内——flag off 无 context 无 regime）；`_run_graph` 在 run 绑定后计算+upsert+**重新 force-bind** context（analyst 从 regime_by_id 读存储块，**不重算**）；final_state/log entry `regime_id` 键存在性=信号（off 字节不变）；ticket.regime_id 从 context 填（off 保持 None，pinned 测试钉住）；predictions regime_id 计入**内容比较**（同 id 换 regime=不可变冲突）；outcome 行经 pending dict 透传 regime_id。
- **vol 年化收口**：`vol_estimators.sessions_per_year(instrument_class)` 经 `trading_sessions_between` 固定窗口（2025-01-01→2026-01-01）派生：stock_perp=261（2025 工作日数——可推导的 ~252 类因子；Binance 假日历未机器验证，252 纯惯例无法重构，已 docstring 披露）、pure_crypto_perp=365、其他=252。`periods_per_year_for(asset_type, instrument_class=None)` 加性参数，已知类优先；无类判定保持历史行为（纯币/股票字节不变，仅 stock_perp 路径变化）。接线：binance_indicator_tools（venue=perp 时用 `stock_perp_underlying` 判类）+ market_regime.classify_vol_state（新 instrument_class keyword，format_regime_context 判类传入）。
- **conftest**：`_runtime_ledger_isolated` 增 `"regime_state": False`；.env.example 增 YIALPHA_REGIME_STATE 注释行（双向覆盖测试钉）。
- **测试**：tests/test_regime_state.py（25 个）：id 确定性/输入敏感/版本隔离/computed_at 剔除、纯币 live 装配、历史模式 LIVE 腿零调用零泄漏、stock_perp 装配+诚实缺口、全缺→None、session 桶、迁移 v2 幂等+重复列恢复、store 往返、predictions 带载+regime 冲突、outcome 透传、`_run_graph` 穿链 flag on/off/uncomputable、ticket 填充 on/None off、analyst 注入 on/off 字节不变+PIT tag、scoreboard by_regime+no_regime 桶、sessions_per_year 双类钉值、indicator 工具年化因数捕获。

## V2.3 Specialization — ✅ 完成（2026-09-03，2862 passed / ruff clean / mypy 165 文件 clean）

**执行注记**：子智能体实施途中撞 5h 用量上限终止（留下完整接线层）；主智能体接管补齐 config/env/conftest/CLI 注册/POSITIONING outcome 定价/PM 双观点透传/desired_side/全部测试（+18 测试）。落点：
- **Positioning 分析师**：`positioning_analyst.py`（sentiment 型结构化输出无工具循环；复用 market 的 run_cached bundle 键=每 run 一次 fetch 服务两分析师）；`PositioningReport` **extra="forbid"**（direction 键直接 ValidationError——冻结是结构性的）；盲预测 scope=POSITIONING（direction=funding 累计和符号，`direction_note` 注入工具描述，工具描述插槽在 Args 前）；`render_positioning_block`（funding/OI/LSR/taker/depth/ADL/basis 段）。
- **拆分裁决（偏差）**：flag-on 时 market analyst 的 bundle 块**保持不变**（positioning 数据在两块中重复而非从 market 移除）——重复优于数据丢失，且保证 flag-off/on 对现有报告字节零变化；若 A/B 后转默认再考虑去重。
- **接线**：`analyst_execution` ANALYST_NODE_SPECS + 计划构建器把 positioning 钉在 market 后（任何入口顺序）；`conditional_logic.should_continue_positioning`；trading_graph 注册休眠 ToolNode([submit_prediction])（接线契约满足）；AgentState.positioning_report；CLI AnalystType.POSITIONING **不可用户选择**，`filter_analysts_for_asset_type` flag-on+crypto_perp 时追加（CLI 与 batch 同一门）。
- **POSITIONING outcome**：`_funding_window_sum` 重构（原始累计和+最后结算日，`_funding_pnl_leg` 变薄壳）；分支在 MACRO 后、时间自算（**坑：分支在 analysis_dt 定义前，引用即 UnboundLocalError**）；net_return=realized funding sum（方向无关，钉死），无价格/成本腿，缺口→incomplete(funding_window)。
- **PM 双观点**：PortfolioDecision +underlying_direction/contract_direction/basis_view（schema Field 描述驱动 LLM，**不加 system prompt 行**——prompt 字节不变）；render_pm_decision 仅填充时加行（None 字节不变钉死）；`_decision_fields_dict` 透传非 None。
- **desired_side_from_decision**（tickets.py，record-only 供 V2.4）：Buy/Overweight→LONG、Hold→FLAT、Underweight→REDUCE、Sell→CLOSE、未知→FLAT；**永不返回 SHORT**（仅 V2.4 显式结构化字段可开空）；双观点字段不参与（保守读法）。
- **链上**：`onchain_flows.py` blockchain.info charts（keyless，estimated-transaction-volume-usd + n-unique-addresses，30d 窗，PIT 过滤未来点，PIT_REPLAYABLE，available_at=最后点时刻）；capability-absent 披露渲染；evidence category "onchain"（新类别，social 会误标）。
- **config**：positioning_split=False（默认关待 A/B）/onchain_evidence=False + env 行 + conftest 扩展 + .env.example。

## V2.4 Portfolio Control — ✅ 完成（2026-09-03，2941 passed / ruff clean / mypy 169 文件 clean）

批次：K（子智能体，signed_math/constraints/resolver 纯数学，51 测试）+ M（子智能体，backtest 分轨，19 测试）+ L（主智能体，流水线）+ N（主智能体，Web API）。落点：
- **单旋钮裁决**：`portfolio_control_mode`=legacy(默认)/shadow/enforced 取代 RFC 三 flag（staged rollout 一个旋钮表达完整）。
- **K**：RiskDecision{rule,PASS|RESIZE|VETO,multiplier∈[0,1],reasons,metrics}（post_init 校验 VETO⇒0）；五硬约束冻结顺序 global_gross→asset_class→single_concentration→directional_concentration→correlation_cluster（函数名=rule 名）；multiplier=limit/current clip；超限=RESIZE 非 VETO（VETO 留给非有限输入/退化）；directional=候选方向净额（对冲净掉）；cluster_of=None→单符号簇（max_cluster=第二道符号帽，诚实：v1 无相关性分组数据）；约束崩溃→fail-closed VETO("constraint_error")；advisory（VaR/CVaR/beta/corr/USDT depeg/Binance outage/BTC×财报 gap 复合压力）只算不缩。signed_math：funding=-sign×notional×rate、镜像 ATR 止损/强平、同 bar 双触发保守先强平。
- **M**：engine `_perp_instrument_class`（memo per (asset_type,symbol)，委托 routing）；年化走 `periods_per_year_for`（stock_perp=261/pure=365/equity=252，**crypto_spot 365 钉值靠"非 perp 家族不传 class"保住**）；corporate_actions 可选参数（拆股乘性/分红加性，按 event_date 前调整，None=字节不变）；fill 审计=已在下一**存在**bar 成交（24/7 与跳日两形状新钉）；MMR/同 bar 短侧新钉。**主智能体收口：执行侧 8 处 `asset_type=="crypto_perp"` 全部 re-key 为 is_perp_family**（M 只改了验证门；funding/model_liquidation/stop-sim/mark 触发/config/metrics 若不改，stock perp 可过验证却无强平建模——危险缺口）。
- **L**：迁移 v3（portfolio_snapshots+positions）；`ledger/portfolio.py`（commit_final_ticket 单事务原子写 Final Ticket+Snapshot+Position，全 INSERT OR IGNORE 幂等；**同符号替换语义**：开新仓先关同符号旧仓 closed_at——生命周期转移非改写，重试不变量在符号级成立）；`_apply_portfolio_control`（legacy no-op 字节不变；shadow=全管道计算+[SHADOW] 段渲染+final_state 记录零写库；enforced=resolver 权威：final_size/status APPROVED|RESIZED|VETOED/margin_mode=ISOLATED/risk_decision_ids=5 rule 名/portfolio_snapshot_id 落票+原子入账；**REDUCE/CLOSE→FLAT 入场候选**（V2.4 管 entry 侧，缩/平存量走持仓生命周期 close 路径）；side=显式 desired_side 或保守 rating 映射，resolver 永不改 side）；enforced 下 `_link_ticket_to_ledger` 跳过 attach_ticket（原子提交独占票据写，防 CANDIDATE payload 抢占 OR IGNORE）；**显式 SHORT 过渡规则**：评分 Sell 时 legacy weight=0 → 显式空头用 kelly_fraction×max_single_position=5% 保守默认入场（披露，待 signed-Kelly）；rating 回退链 pm_fields.rating→final_state.pm_rating；equity 从 PortfolioState 传入。
- **N**：web /api/tickets/{id}、/api/portfolio/snapshots/{id}、/api/positions（store 只读加载器+404 语义）；RFC17 矩阵审计=性质 8 条+集成 13 场景全部由各版本测试覆盖（K51+L9+M19+既有钉值）。
- **坑**：pm_decision_fields 不总带 rating（测试壳）→ overlay 权威源 pm_rating 回退必加；mypy Side literal 需 cast 收窄；test_frozen_field_set pin 每版加字段都要补（V2.1 先例）。

## 验收收敛批（2026-09-03，用户指令：停止扩功能，进入独立验收+shadow 样本积累）— ✅ 完成（2948 passed / ruff / mypy 169 文件；commit 不加版本 tag）

21. **验收批落点**：状态定义=「代码交付完成，启用验收待完成」。五项收敛：
    1. **Ticket-first 顺序核实**：enforced 实际执行序 PM→策略初始定尺（RiskManager 单票据 Kelly/breaker/CVaR/ATR，成为 candidate 的 proposed_size，非组合风控）→Candidate→Snapshot→RiskDecision[]→Resolver→Final——`test_v24_acceptance.py` 用 6 点执行 trace 钉死；docstring 明确「策略初始定尺 proposes / 组合风险约束（严格在 candidate 之后）只能缩或拒」两层区分。
    2. **enforced 硬拒绝**：新增 eligibility 资格门（**非第六组合约束**，五约束冻结不动）：unknown_perp / tradeability NO_TRADE / stock_perp FX 桥不完整（underlying 有 USD 目标但无换算）→ 作为 rule="eligibility" 的 VETO RiskDecision 进 resolver → Final Ticket 真正 VETOED+零仓位+veto_reasons；shadow 档显示 WOULD VETO 不阻断；五约束照跑留审计。
    3. **空头定尺如实标注**：kelly×max_single=5% 明确标 TRANSITIONAL HEURISTIC not signed-Kelly（overlay 行+final_state.short_sizing="heuristic"）；空头缩仓测试证明 resize 只缩不翻符号（gross 超限→5%缩至更小负值）；不加更复杂定尺模型。
    4. **261=日历假设**：`SESSION_CALENDAR_CAVEAT` 常量；engine stock_perp backtest 的 config_summary 强制携带 session_calendar_assumption（weekday-count / NOT verified / 假日/DST/闭市未核实）；report.md 渲染 ⚠️ 行；纯币无该键（真 24/7）。
    5. **Mimosa 完整审计完成**：scanId `scan-2026-09-03T02-29-00.403Z-630d1fa5af3c`，seal `sha256:f2ea8697…05dc`，**30 findings（21 high/9 medium）全部位于 V2 之前的旧代码，V2 新面（账本事务/幂等/约束/resolver/新 web 端点）零 finding**。分布：pot_executor 代码注入×3（默认关 flag 已知风险）、app.js 前端污点启发×13、内部路径拼接污点×4（用户输入边界已有 safe_ticker_component/_validate_path_ticker/sanitize_cache_filename）、固定 vendor 主机 SSRF×7、reddit/sec_ownership XML 实体扩展×2（可后续 defusedxml 加固，分析只读路径非验收阻塞）。**审计结论按 Mimosa 纪律：不宣称项目完全安全**。
    - 用户路线指令：positioning_split 保持关闭（A/B 不作为加功能理由）；legacy 默认，新流程先 shadow（收集旧新差异/否决原因/空头票据/净收益归因），验收通过后才 enforced；enforced 仍是分析与账本约束非实盘授权。下一阶段目标：每张永续票据可解释、可重放、被后续结果检验。

## Shadow 验收批（2026-09-03，基线锁定 `6cbfe1d`；2960 passed / ruff / mypy 三绿；不加版本 tag）

22. **Shadow 验收五项落点**（用户指令：开发收住、只验证运行证据；先验收运行正确性再评估预测表现，两者不混为一个"通过"；positioning_split 保持 false）：
    1. **非空组合快照**：`portfolio.load_positions_input()`=ledger open_positions ∪ 操作员只读 JSON 文件（config `portfolio_positions_file`，env YIALPHA_PORTFOLIO_POSITIONS_FILE；同符号文件行替换 ledger 行；坏文件降级 WARNING）；每张 snapshot/记录携带 `snapshot_source` 标签（empty/ledger/file/ledger+file:name）——shadow 预演永不静默对空组合。测试覆盖已有多头+已有空头+同符号近限（shadow 3 持仓预演）与超限（asset_class 0.9>0.6 + directional 0.9>0.8 双 RESIZE 钉值）。**注**：文件行不带 instrument_class 时默认 unknown_perp 会拆散类聚合——操作员文件应显式给类（config 注释已说明）。
    2. **拒绝开仓≠批准平仓**：`commit_final_ticket` 增 `close_existing`（与 open_position 互斥）：开仓=同符号先关后开（重试幂等）；**批准平仓**=仅 APPROVED 且 side=FLAT 且 legacy_intent=CLOSE 才关仓不开新；**VETO/FLAT/REDUCE=双标志皆 False，旧仓一行不动**（钉死测试：种子仓 0.12 经 VETO 后原封不动）。REDUCE 的部分减量留给持仓生命周期（未实现，显式不假装）。
    3. **完整对账依据**：shadow 记录扩为 16 键自包含 dict（mode/ticket_id/tradeability/reference_price/side/proposed/resolver/decisions/eligibility/candidate/snapshot_positions/snapshot_source/limits/config/short_sizing/snapshot_id），**经 `_log_state` presence-gated 持久化进 full_states_log**（此前只 riding state 不落日志——已修）；零无法解释翻方向/放大/关键缺失仍批准由性质断言钉住（side 回显、final≤proposed、multiplier∈[0,1]、5 rule 名齐）。
    4. **scoreboard 样本入口**：写库矩阵精确化——shadow 写 **0** 行 portfolio_snapshots/positions/Final Ticket；prediction_ledger 记录阶段（runs/evidence/predictions/outcomes + tickets 镜像 CANDIDATE payload）与 portfolio_control_mode **完全独立**（钉死测试：shadow 下 prediction 提交正常、镜像有 CANDIDATE 行、组合三表零行）。评分入口=`yialpha scoreboard`→compute_outcomes→pending_predictions（仅 horizon 完整到期；funding 缺口=腿缺失 fail-closed；价格/funding/fees/slippage 分腿核对在 V2.1 测试已钉）。
    5. **30 findings 逐项处置**（scan-2026-09-03T02-29-00.403Z）：
       - **真实可达（2）→已修复**：reddit.py RSS / sec_ownership.py Form4 的 XML 实体扩展——新增 `dataflows/utils.safe_xml_root`：解析前拒 DOCTYPE/ENTITY（大小写不敏感全文扫描；合法 reddit/SEC feed 两者皆无）；sec 侧包装为 NoMarketDataError typed miss，reddit 侧 fail-soft。测试钉死（恶意 DTD 拒绝+干净 RSS 可解析）。
       - **受条件限制（11）**：pot_executor 代码注入×3（pot_enabled 默认 False+独立显式 opt-in；开启即真实可达——保持关闭，不宣称已缓解）；路径穿越×4 中 perf_telemetry（路径来自 operator config 的 results_dir+ticker，操作员信任边界内）+ cli/main.py:1192（section_name 内部常量表）；SSRF×7 中 alpha_vantage/reddit/factor_model（固定 https 主机+urlencode 查询/符号经上游 normalize；无用户可控绝对 URL——内网不可达但保留条件标记）。
       - **误报（17）**：app.js renderError/drawRatingCompare XSS×7（sink 链 errorBox→`esc()`=`window.YiUtil.escapeHTML` 已验证转义）；fetchJSON SSRF×4（客户端同源相对 /api/* 路径，无用户可控绝对 URL）；跨文件污点×2（common.js escapeHTML sink）；stocktwits.py SSRF（`safe_ticker_component` 已 sanitize，代码注释自证）；trading_graph/run_robust/cli 其余路径穿越×3（内部常量拼接或 safe_ticker 边界）。
       - 处置原则（用户指令）：不因"旧代码/只读分析"认定非阻塞——XML 两点虽在只读分析路径仍按真实可达修复；PoT 按条件可达标记并保持默认关闭。
    6. 261 日历披露维持假设标签（假日/DST/闭市/下一可交易 bar 未核验；相关年化结果保留 ⚠️），不算"日历验证通过"。

## 2026-09-04 晚间批（子智能体并行：SEC UA 销账 / 日历机器核实 261→365 / housekeeping / enforced 门槛写死）— ✅ 完成（3059 passed / ruff clean / mypy 179 文件 clean；`e0b961c` + `ff2e9db` + 本批 2 commit）

23. **A1 SEC fair-access UA 销账**（round-3e 起 sec_edgar 403 欠账）：`YIALPHA_SEC_USER_AGENT="YiAlpha research zhang12120113@gmail.com"` 落 .env；.env.example 行按 YIALPHA_LEDGER_DB 先例保留注释文档形式（env-example 双向钉测试 3 项绿——getenv 直读键不激活进 example）。验证：MU live fundamentals 实抓 SEC 腿 HTTP 200，companyfacts（CIK 723125）落 `~/.yialpha/cache/sec/`。
24. **A2 exchangeInfo 时段字段核实（机器证据）**：MUUSDT 共 24 keys，无任何 session/tradingHours/calendar 字段（时间类仅 onboardDate/deliveryDate/timeInForce）→ **已核实缺失**，exchangeInfo 路径关闭；onboardDate=2026-04-07T13:30Z（美东 09:30，TradFi 锚定印证）。
25. **A3 假日/周末停市核实（机器证据，口径反转）**：MUUSDT 1d klines（窗口 2026-05-18→07-10，54 bar；上架晚于 2025 假日故取 2026 已过假日）——三个美股全日假日（05-25 Memorial / 06-19 Juneteenth / 07-03 Independence observed）**全部带量 bar**（5.1万–36万张，与相邻交易日同量级），14 个周末 bar 全带量，零量日=0，假日/周末量能≈平日 1/4–1/3（更薄但不改日历）。**结论：Binance 股票永续 24/7 全年无休，261（2025 工作日数）与 252（NYSE）皆证伪 → 年化因子 365**（与 pure_crypto_perp 同口径；261 把年化波动低估 sqrt(365/261)≈1.18 倍）。
26. **261→365 修正（本批执行，7 文件）**：vol_estimators stock_perp 分支改返回 CRYPTO_TRADING_DAYS_PER_YEAR（`_SESSIONS_WINDOW` 保留，pure 分支仍引用非死代码）；sessions.py 只改披露文案（SESSION_CALENDAR_CAVEAT="klines-verified 24/7 … factor 365"，常量与 weekday 计数语义零改动）；engine.py `config_summary["session_calendar_assumption"]` 值+注释同步（键名与装配逻辑不变）；binance_indicator_tools 注释同步；钉值测试 261→365（test_backtest_perp_classes / test_regime_state / test_v24_acceptance；report.py 渲染动态取 config 无需改）。**未动** V2.2 冻结的 ET session 桶（date-only=工作日 regular/周末 closed）——牵连 regime_id 语义需 bump REGIME_VERSION，与"regime 标签 vs 实际 24/7 交易"的偏差一起留用户裁决。**范围外残留（下一批）**：market_regime.py 3 处 docstring、routing.py:48 注释、fundamentals_analyst.py:151 提示词仍写 "published TradFi sessions"（被 test_news_contract_angle.py:264-271 钉死，需连测试一起改）。
27. **E housekeeping**：ci.yml checkout v4→v5 / setup-python v5→v6 实际 **16 处**（8 job×2）；test_deepseek_reasoning.py 改造为 **GLM 等价 live-call**（GLM key 在位才真跑、缺 key 干净 skip、无永久 skip；DeepSeek 单测保留——客户端仍是活代码）；get_binance_basis 的 in-200 -4104 重抛 NoMarketDataError 以结构性缺失开头（"no basis data for this symbol"，vendor_code 附带 debuggable，errors.py 加可选属性 + 3 新单测）。
28. **F enforced 晋升门槛（写死，数字用户保留调整权）**：≥5 轮 shadow × ≥4 符号 × ≥30 条成熟预测 × 0 条失败 invariant × time_and_link_audit 全绿 × 校准方向合理 → 允许**单符号**试运行 enforced；enforced→实盘永远单独经用户显式授权。每次 Track B scoreboard 批读对照汇报进度缺口。

## 已冻结的契约决定

1. **中央账本**：单文件 SQLite，config `ledger_db_path`（默认 `~/.yialpha/ledger/portfolio.db`，env `YIALPHA_LEDGER_DB`）。WAL + busy_timeout=5000 + BEGIN IMMEDIATE 原子提交；版本管理用 **schema_meta 表**（`schema_version` 行，非 PRAGMA user_version——Mimosa 钩子拦 f-string PRAGMA）；append-only（无 UPDATE 路径）。迁移 v1 = runs / instrument_snapshots / evidence / predictions / outcomes / tickets 六表（DDL 见 `yialpha/ledger/sqlite.py::_migrate`）。**所有 SQL 必须字面量+参数绑定**（Mimosa 钩子拦截变量 SQL，已两次拦截验证）。
2. **确定性 ID（幂等）**：evidence_id/prediction_id 由内容哈希派生（"E"/"P"+12hex），重复提交 INSERT OR IGNORE——工具循环重入不会重复入账；run 重试不重复。
3. **horizon 阶梯**：`HORIZON_LADDER_DAYS = (1, 5, 21)` 天（冻结进 FEATURE_VERSION v2 docstring）；批次 B 须核对 accuracy.py 现有 horizon，若已有既定阶梯则采纳同值并回改 versions.py docstring。
4. **instrument_class 语义（record 阶段）**：routing 判定链 = registry PIT 快照（available_at<=as_of）→ 现有 warm/seed 逻辑 → 仍不可证时（且 flag 开）`unknown_perp`（只披露，不 veto；enforced 在 V2.4）。**不做**"历史无 registry 快照→一律 unknown"——那会回退历史 stock_perp 的 fundamentals 路由，违反 record 阶段零行为变化原则。非美 EQUITY（HK/KR/CN）、COMMODITY、INDEX → instrument_class=unknown_perp + `unsupported_reason` 字段点名。
5. **FX**：USDTUSD = 1/Binance 现货 USDCUSDT 收盘（用户选定）；live-only（历史回放不可用→披露）；缺价 record 阶段只记 shadow DEGRADED_CRITICAL 判定，不真 veto。
6. **flag 布局**（生产默认开=record，conftest autouse 测试关）：`instrument_registry` / `prediction_ledger` / `stock_perp_fair_value`；DB 路径 `ledger_db_path`。conftest fixture 名 `_runtime_ledger_isolated`（字母序在 `_isolate_config` 后）。
7. **时间字段**：全部 ISO-8601 UTC 字符串（`utc_now_iso()`）；禁裸 time/date/timestamp。
8. **预测 scope**：market/sentiment=CONTRACT；fundamentals/news：stock_perp→UNDERLYING、纯币→CONTRACT。direction∈{up,down,flat}，prob_up=P(return>0)。
9. **FEATURE_VERSION → v2**（本版唯一 bump；SCHEMA_VERSION 不 bump——AnalystPrediction 是新 schema 的 v1）。
10. **批次文件边界**（防并行冲突）：A=routing.py+binance.py+instruments/；B=ledger/(sqlite.py 除外)+versions.py docstring 核对；D=perp/+tickets.py+schemas.py（全部 additive 可选字段，不 bump 版本）；C=四 analysts+agent 工具注册；E=outcome 计算+CLI；F=trading_graph+web+conftest。
11. **corporate_actions 范围调整**：V2.1 只做 dataclass+校验+渲染骨架，**无 DB 表无 vendor**（持久化随 V2.4 回测接入再加迁移 v2）——避免 schema 反复。
12. **D 批次落点（2026-09-03）**：quote_fx 复用 `binance_klines_frame(venue="binance_spot")` seam（测试桩 `yialpha.perp.quote_fx.binance_klines_frame`）；缓存 `vendor_cache_dir("quote_fx")/usdcusdt_<date>.json` ttl=1h fail_open；depeg 带 [0.95,1.05]（close）/[0.9,1.1]（fx 入参）；`usdt_usd_as_of` 仅 today 委托 fetch、过去/未来/不可解析→None。fair_value 纯函数：`target_usd ÷ usdt_usd × (1+expected_basis)`，missing_inputs={underlying_target,fx,fx_out_of_band,current_basis}，chain 全字面量渲染（存储值永不取整）。tickets 5 个加性字段插在 analysis_as_of 前；schemas 4 个加性字段（USD/USDT/last/mark 校验）渲染字节不变（三方 byte-identity 测试钉死）。**已知偏差**：tests/test_v2_tickets.py 的 `test_frozen_field_set_is_complete` 是精确集合 pin，必须加 5 个新字段名才能与加性扩展共存（合理）。
13. **A 批次落点**：sessions 常量 canonical 家迁至 `instruments/sessions.py`（routing 再导出）；registry 键=exchangeInfo symbol 原样（大写紧凑），USDC 结尾输入额外探测 USDT 孪生；PIT 归一化：date-only as_of=当日 UTC 零点（保守防泄漏，测试钉死）；`unsupported_reason` 用连字符 `"unsupported-contract-type:<TYPE>"`（非下划线，测试已钉）；routing 矩阵：registry evidence>warm/seed；registry 空且 warm/seed 正向分类→保持原答案；registry 空且 pure-by-default→unknown_perp；flag off 字节不变。warm 集成 `_persist_instrument_registry` 零额外 HTTP、fail-soft WARNING。
14. **B 批次落点**：**HORIZON_LADDER_DAYS=(1,5,21) 维持**（accuracy.py 只有 holding_days=5 单参数无固定阶梯）；ID preimage `evidence|{run_id}|{payload_hash}` / `prediction|{run_id}|{analyst}|{scope}|{horizon}|r{revision}`（revision 判别器防 revise 撞 id）/ `outcome|{prediction_id}|{horizon}`，sha256 前 12 hex + E/P/O 前缀；时间归一化双规则：流逝判定 date=当日零点，PIT 准入 date=当日 23:59:59（否则生产 date-only trade date 会全量误报 PIT 违规）；legs_missing/evidence_ids 存排序 JSON（内容比较顺序无关）；不可变=同事务内 check-then-insert，UNIQUE 永不触发；FK 违规不被 INSERT OR IGNORE 吞（已验证）。
15. **E 批次落点**：outcome_compute seams=`binance_klines_frame`/`get_binance_funding_rate`/`get_YFin_history_cached`（模块级绑定可桩）；票据成本从 tickets 镜像表按 run_id 取（**F 必须写镜像，否则 outcome 的 fees/slippage 腿永远缺失**）；funding=`−sign×Σrate`（up=+1/down=−1/flat=0，缺口=腿缺失 fail-closed）；exit bar 不严格晚于 entry → 不写留 pending；entry bar 缺 → incomplete(contract_price)；UNDERLYING scope 净收益=cpr−fees−slippage（funding 结构性 n/a）；MACRO scope 跳过；ticket 只有标量 estimated_cost 时全部记 fees、slippage=0（净值不变）。scoreboard：complete outcomes⋈predictions⋈runs；指标 accuracy/Brier/log-loss/ECE(10 bins)；切片 analyst/class/horizon/direction/evidence 覆盖桶；V3_MIN_SAMPLES_PER_CELL=30 只展示不调权。CLI `yialpha scoreboard`（typer，写 `<results_dir>/scoreboard/scoreboard.{json,md}`）。
16. **candidate-first 前移裁决（F 批，主智能体 2026-09-03）**：`build_candidate_ticket` 消费 risk sizing 输出（target_weight 进 perp 杠杆计算、entry/stop 直接入票），V2.1 强行前移会改变 pinned 钉值、违反 record 阶段零行为变化。**完整 candidate-first（PM 后建 candidate、risk 只缩不放、proposed/final 拆分）归 V2.4 resolver**；V2.1 交付 linkage（run_id/prediction_ids/evidence_ids 填充）+ tickets 镜像 + bridge 字段。ExecutionTicket 非 frozen（可 build 后赋值）。
17. **F 批落点**：`_run_graph` flag 门控绑定 run context（new_run_id="R"+12hex）+ register_run + `ensure_prediction_capture_scope()` + finally reset；state `run_id` 仅 flag-on 存在（键存在性即信号，off 字节不变）；`_link_ticket_to_ledger`（linkage + mirror + bridge，全 fail-soft）；bridge 输入链：pm_fields.underlying_price_target → price_target(若 price_target_currency=USD)；current_basis=(last−mark)/mark；**fair_value 修正：current_basis 改为诊断项不阻断换算**（D 实现与 docstring 矛盾，按 RFC 公式语义裁决）；overlay 披露 fx line + bridge block + fx 缺失 shadow verdict 行；PM `_decision_fields_dict` 增 4 个可选字段（仅非 None 入 dict，日志字节不变）；`ledger/tickets_mirror.py`（attach_ticket INSERT OR IGNORE + ticket_for_run 读）；web：/api/runs/{run_id}/predictions、/api/outcomes、/api/calibration + accuracy 页 calibration 卡片（i18n zh/en）；.env.example 补 3 个 flag 行（test_env_example_coverage 双向钉：override 键必须文档化、example 键必须真实——YIALPHA_LEDGER_DB 走 getenv 直读不能进 example）。**坑**：sed 直写被 Mimosa 拦，必须 Edit 工具；monkeypatch 桩函数记得带被替函数的形参。

## 用户决定记录

- 2026-09-03：V2.1→V2.4 连续推进；基线用户自提交（已验证 `adec679` 干净）；盲预测用 submit_prediction 工具调用；FX 用 Binance 现货 USDC 反推；**重点只做永续，其他资产不动**；**最大化子智能体使用**（上下文预算）。
- 2026-09-04（晚间行动计划）：A3 klines 证据驱动 261→365 当晚直接执行；F 门槛数字按用户提案写死；Track B 09-05 晚首试 scoreboard（严格可读 09-06 08:00 后，未成熟就顺延）；Track C 每晚 BTC+ETH+MU 固定班底+每周轮换 1 槽，发射窗本地 08:00–23:00，每轮复制 round-5 脚本目录+provenance code_state=当晚 HEAD；Mimosa deep 扫描由用户 UI 触发（outputDir `C:\Users\warri\.mimosa-scans\yialpha`，agent 无触发路径——扩展市场亦无连接器）。

## 下一步

- 09-05 08:00 后：round-6 发射（BTC+ETH+MU；复制 round-5 脚本目录，code_state=当晚 HEAD；验收=preflight/reconcile/scoreboard 三 PASS）。
- 09-05 晚：Track B scoreboard 首批（1d outcome 未成熟就顺延到 09-06 08:00 后硬读）：`YIALPHA_LEDGER_DB=analysis_output/shadow-glm-round5-20260904/ledger.db .venv-dev/Scripts/yialpha.exe scoreboard --results-dir <round5 目录>`。看四样：① 三分析师盲预测 accuracy/Brier；② live SHORT（MU −0.05）1d 兑现；③ ECE 只看方向；④ by_regime 切片。
- 用户 UI 触发 Mimosa deep 扫描 → 密封报告按真实可达/条件可达/误报三分类处置 → roadmap 销账。
- 下一批：365 范围外残留（第 26 条）+ ET session 桶/regime_id 语义裁决。
