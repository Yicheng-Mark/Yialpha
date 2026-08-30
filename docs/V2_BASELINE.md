# V2 Implementation Baseline（冻结版 · Final）

> 状态：**冻结**。本文档是 V2 的 PRD、开发任务拆解依据、Code Review checklist
> 与测试验收标准。架构讨论已终结（2026-08-18 四轮评审收敛）；后续所有优化必须
> 服从本文的不变量。版本演进 V2.0→V2.4，每版本三绿（pytest / mypy 0 / ruff
> clean）后 tag 再进下一版本。

## V2 阶段目标

不是提升单次分析复杂度，而是建立**可验证（Verifiable）、可归因（Attributable）、
可回放（Replayable）、可演进（Evolvable）**的量化决策基础设施。

不追求增加 Agent 数量、不追求增加工具数量、不追求更复杂 Prompt。

核心资产链（任何未来能力必须挂载在该链路上）：

```
Evidence Ledger → Prediction Ledger → Decision Layer → Execution Ledger
→ Portfolio Ledger → Outcome Ledger → (V3 Weight Learning)
```

## Architecture Enforcement Rules

1. **Ticket-first**：所有交易动作必经
   `Decision → Candidate Ticket → RiskDecision → Ticket Resolver → Final Ticket → Position`。
   禁止 Agent output 直接下单；禁止 Risk 模块静默改写 PM decision。
   PM=观点层 / Risk=约束层 / Portfolio=状态层，三个 namespace 永久隔离。
2. **Prediction immutable**：预测入 Prediction Ledger 后不可修改；后续观点变化
   记 `original_prediction + debate_revision`，禁止覆盖——系统必须能回答"当时
   模型真正相信什么"。
3. **PIT 时间优先**：所有历史分析满足 `available_at <= analysis_as_of`；任何
   新数据源必须提供 `event_time / available_at / created_at` 三个时间字段。
4. **Risk 是约束，不是决策者**：Risk 只输出
   `RiskDecision{action: PASS|RESIZE|VETO, multiplier, reasons}`，禁止输出
   BUY/SELL。方向属 Research/PM，风险属 Risk。
5. **Analyst 输出必须可评分**：新增 Analyst 必须提供 prediction + horizon +
   confidence + evidence_refs + outcome linkage，否则不得进入 V3 数据池。

**Code Review 第一规则：任何绕过 Ticket、Prediction Snapshot、RiskDecision、
PIT 时间语义的快捷实现，均视为架构违规。**

## 时间字段命名（强制）

| 字段 | 语义 |
|---|---|
| `analysis_as_of` | 模型看到世界的时间 |
| `available_at` | 数据真正可获得时间 |
| `event_time` | 事件发生时间 |
| `created_at` | 系统写入时间 |

禁用裸 `time` / `date` / `timestamp`。

## 代码内版本戳

所有 schema 与账本对象携带（定义见 `yiagents/versions.py`；不靠 git 反推）：
`schema_version` / `feature_version` / `cost_model_version` / `regime_version` /
`ticket_version`。字段重命名/删除/重定义必须 bump；带默认值的可选新增不算。

## 账本关联主键链（I2）

`run_id → prediction_id → decision_id → ticket_id → position_id`，层级关系
`Run → {Evidence[], Prediction[], Decision, Ticket → Position}`。Evidence 为
**多对多**（一条 Fed 降息证据可同时影响 BTC/SPX/Gold）：prediction/decision
通过 `evidence_ids[]` 反向引用，`ticket → evidence` 不做唯一方向。

## 五版本路线

| 版本 | 主题 | 核心交付 |
|---|---|---|
| V2.0 | Tradeability | CostModel、方向化 edge、critical-data gate、DerivativesStress、PortfolioDecision 扩展、ExecutionTicket schema |
| V2.1 | Measurability | 盲 AnalystPrediction（多 horizon forecasts）、不可变预测快照、OutcomeRecord、归因、计分板 |
| V2.2 | Context | PIT RegimeState（regime_id/feature_version/confidence_components）、全链注入、regime 归因 |
| V2.3 | Specialization | positioning_split 双态开关；Market=Price / Positioning=Market Structure（只出 bias 不出方向）/ Sentiment=Social / News=Events / Fundamentals=Business |
| V2.4 | Portfolio Control | Candidate → Snapshot → 预演 → RiskDecision[]（multiplier 取 min）→ Resolver → Final Ticket → SQLite 原子入账 |

V2.4 组合硬约束仅 5 项且固定顺序：global gross → asset class → single
concentration → directional concentration → correlation cluster。VaR/CVaR/
beta/相关矩阵第一版只计算+展示+入档，不参与硬 resize。

## V2.4 后禁止事项与 V3 启动条件

V2.4 完成后**不加 Agent、不加 Debate Agent、不加预测字段、不加风险硬约束、
不人工调 Analyst 权重**，进入 Sample Accumulation Phase（只收集
Analyst × Asset × Horizon × Regime × Outcome）。

**V3（Dynamic Weighting）启动条件是样本规模**——每 Analyst × Asset Type ×
Horizon × 主要 Regime 具有足够 OOS outcome 样本——不是日期，不是代码完成。

## 冻结原则

> V2 不负责让系统"更会预测"，V2 负责让系统知道"自己为什么预测、预测是否
> 有效、风险为什么允许执行"。
