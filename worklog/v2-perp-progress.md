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

## 下一步

前置完成后并行派发 A/B/D 三个后台子智能体 → 完成后派发 C/E → 主智能体做 F → 门禁 → tag v2.1.0。
