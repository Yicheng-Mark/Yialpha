# P0 A 阶段最终审阅记录

日期：2026-09-06。结论：**A 阶段已完成；B 阶段真实验证未开始**。
主智能体与三个子智能体并行审阅时间/持久化、到期评分/汇总、真实验证方案，另做方案独立复核。
仅修复可复现问题，未从头实现 P0。未发现本次审阅范围内仍未解释的阻断问题。

## 1. 基线、范围与原件保护

- 开工和交付 HEAD 均为 `62b1fa02dce44a693a75603e0085dd4a351a5f8a`；未提交、推送或部署。
- 接手时已有 15 个已跟踪文件修改、6 个未跟踪文件；20 个前一轮验证文件与交接哈希逐一一致，另有交接文档。
  现有未提交修改全部保留，没有 reset、checkout 覆盖或重建旧 P0。
- 本轮在原文件中增量修改 3 个生产模块、4 个测试文件、2 个文档；新增本文和真实验证方案。
  其余原有 12 个文件与本轮开工哈希相同。
- `analysis_output/shadow-glm-round6-20260905` 与 `analysis_output/shadow-glm-round6b-20260905`：
  开工及结束都与原历史基线对比，均为 **611 个文件、SHA-256 零差异**。没有重跑原验收或在原账本评分。
- 未找到适用于本工作区的额外 AGENTS.md；以用户约束和两份交接/契约文档执行。

原历史保护基线：`C:\Users\warri\AppData\Local\Temp\yialpha-p0-round6-manifest-before.json`。
本轮开工/结束清单、离线入口及最终日志保存于：
`C:\Users\warri\AppData\Local\Temp\yialpha-p0-a-review-245b90c1d07f4f8ca1b0d5566a444e74`。
临时文件可能被系统清理；不能在其消失后重建并冒称同一历史证据。

## 2. 具体发现与必要修改

| 发现 | 修复与验证证据 |
|---|---|
| ToolNode 同 horizon 并行调用在取价期间越过同一检查，重复接受并覆盖 timing；flush 因重复期限失败 | `prediction_tools.py` 每 capture 共享锁，检查/冻结/接受及 flush 保持一致；同内容不重取价，冲突拒绝，接受中的 flush 等待，写入失败可重试。4 项定向检查修复前 3 failed / 1 passed，修复后 4 passed |
| 终点日线存在不同 Close 时按输入顺序取最后一条；105/120 对调会使 net 从 0.198 变为 0.048 | `outcome_compute.py` 仅检查新窗口所需终点及股票原参考日，冲突返回 `price_bar_conflict:日期`，pending → 固定 7d 后 incomplete；同值重复及无关日线冲突不阻挡；legacy 归一化保持兼容 |
| 新 scoreboard 接纳晚于固定终点甚至期限的 outcome_available_at，也接纳可重新合成 net 的负费用/滑点 | `scoreboard.py` 新口径严格要求 available 等于 window_end，价格类费用与滑点非负；保留 POSITIONING 标签及 legacy 历史语义 |
| 旧分析师/票据测试会顺带运行 advisory regime_context 和 overlay stress，触发网络保护层 | `tests/conftest.py` 默认关闭无关 regime_context、模拟 stress 披露；专门测试可显式启用并模拟该功能。最终统一回归零网络尝试 |

评分/汇总组新增 18 项定向检查，修复前 **11 failed / 7 passed**，修复后 **18 passed**；
包含冲突正反顺序、补数恢复/截止、同值去重、无关日期、三类 scope 时间及负费用/滑点的正反检查。
合计新增 22 项，测试文件为 `test_prediction_tool.py`、`test_p0_review.py`、`test_scoreboard.py`。

## 3. 最终离线验证

| 检查 | 实际结果 |
|---|---|
| 契约列出的同一 18 文件回归 | **338 passed in 7.34s**，入口退出码 0 |
| 网络隔离 | Python socket connect/connect_ex/sendto/DNS 与原生 curl 传输均阻断；最终拦截计数 **0** |
| dotenv 与数据路径 | 项目导入前禁用 dotenv 并审计拒绝 `.env` open；模拟行情、临时账本、临时项目 cache、临时 yfinance 内部 cache |
| Ruff | 19 个改动 Python 文件通过；最后 fixture 增量也通过 |
| mypy | 9 个生产模块，`--no-site-packages --follow-imports=skip`，通过 |
| diff/HEAD/原件 | `git diff --check` 通过；HEAD 未变；保护目录 611 文件零差异 |

复现命令与 18 文件清单见 [P0_TIME_CONTRACT.md](P0_TIME_CONTRACT.md)。
这些是指定范围离线回归，不是全仓测试，也不是真实到期验收。

隔离诊断偏差：首次加严运行的 338 项断言已过，但 105 次 advisory 行情尝试被传输保护层截断，入口因此报失败。
那次尚未先重定向 yfinance 内部 cache，可能读取现存时区/cookie cache；仅查看文件元数据发现原两份 DB 修改时间未变，
不能据此声称从未访问。随后先重定向临时内部 cache，定位并修正 fixture 遗漏，最终达到零网络尝试。
未查看/输出 cookie 值、未读写 `.env`；此偏差没有涉及 round6/round6b 或真实评分。

## 4. 已确认的语义与尚未完成的工作

- 新口径是日线参考收益基准，可能含形成前时段，不能解释为提交后实际持仓收益。
- 股票只采用保守工作日规则；未实现完整节假日日历。复权基准变化拒绝评分，未实现公司行动自动换算。
- 旧行不回填；没有历史参考价重建、覆盖 outcome 或追加复评入口。旧 round6b 的 24 条盘中 CONTRACT 不能以全部 complete 为目标。
- 真实快照的原始供应商响应、可知时间证据、1/5/21d 真实终点、历史资金费频率/覆盖、成本 provenance 尚未验证。
  生产资金费覆盖判定仍依赖观测间隔推断，下一阶段须独立核验实际窗口，不能把推断当作供应商完整性证明。
- 固定输入只能证明流程；真实研究预测样本积累、校准评估及 P1/P2/P3 均未执行。

## 5. 下一阶段具体执行方案

完整可审阅方案：[P0_SHADOW_VALIDATION_PLAN.md](P0_SHADOW_VALIDATION_PLAN.md)。

1. **B0 离线入口准备**：实现尚不存在的独立 runner、路径 guard、原始响应 recorder、manifest 和逐行审计。
   使用临时目录/模拟数据检验；避免 CLI scoreboard 自带评分副作用，并隔离 yfinance 内部 cache。
2. **B1 真实形成时快照**：另行安排后，在独立 cohort/账本实际提交 BTCUSDT、ETHUSDT、MUUSDT CONTRACT 各 1/5/21d，
   以及实际周末 MU UNDERLYING 1/5/21d，共至少 12 条流程记录；保留错误样本，不能回填过去的形成时间。
3. **B2/B3 自然到期与汇总**：按每条真实 formed_at 推导到期/补数截止表，人工启动已安排的检查，核对窗口、费用、缺失、幂等、
   队列与 scoreboard 来源；21d 自然等待不可省略，缺数观察可能到形成后 28 天。

本阶段仅交付方案，未创建真实 cohort、真实账本样本、采集任务、评分任务、自动化或 LLM 任务。
后续已明确安排的阶段无需重复确认；当前 A 阶段完成不扩大为 B 或 P1/P2/P3 的执行授权。
