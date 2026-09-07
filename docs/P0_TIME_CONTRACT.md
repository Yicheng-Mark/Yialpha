# P0：预测时间与到期评分闭环

适用范围：USDT 永续分析的离线修复；analysis_only / shadow，live_execution=false。
本次未更改执行开关、票据定义、资金费符号、策略收益定义；未运行真实 LLM、真实到期评分、提交、推送或部署。
前一轮 P0 的交接与开工 HEAD 均为 `62b1fa02dce44a693a75603e0085dd4a351a5f8a`，当时开工工作区干净。
2026-09-06 A 阶段接续审阅时 HEAD 未变，工作区已有未提交 P0 diff，已按原清单核验并保留。

## 根因与复现

原评分器按 analysis_as_of 的日期选择当日日线作为 entry；盘中预测做出时，该日线到次日 UTC 午夜才闭合。
正确的 intraday knowability gate 因而将完整行情下的 BTCUSDT、ETHUSDT、MUUSDT 合约 1/5/21d 样本写为 incomplete。
等待更久无法改变历史 entry 的可知性。

MU UNDERLYING 周六形成、周日到期的 1d 预测，两端都解析为周五日线。
原代码把相同端点当作尚待收盘，反复 pending；固定周日期限不会因为等到周一而增加交易时段。
同时，outcomes 只允许追加且同一 prediction/horizon 不允许冲突改写，但 pending/incomplete 又反复入队；行情或成本数据变化后会反复失败。

修复前先以全模拟行情、临时账本运行 11 项特征测试：3 个合约 × 3 个期限、MU 周末固定窗口、incomplete 冲突重试全部复现。
另有 2 项预期结果退休的测试在旧实现上失败。旧 gate 测试保留，修复没有放宽 gate 或挪动预测时间。

## 新预测：close_reference_v1

SQLite schema 4 只新增 nullable 的 predictions.timing 和 outcomes.scoring_context；旧行仍为 NULL。
价格/时间口径有独立版本戳，不修改原有 FEATURE_VERSION 或 TICKET_VERSION。

| 时间 | 语义 |
|---|---|
| runs.created_at | 运行登记时间；不是预测形成时间 |
| analysis_as_of | 原运行的研究/PIT 截止时间，原值保留 |
| timing.prediction_formed_at | submit_prediction 接受有效预测的时间；延迟 flush 不重置 |
| timing.reference_price_at | 参考日线的保守闭合时刻：标记为 D 的日线取 D+1 00:00 UTC |
| timing.reference_available_at | 此价格可知的保守界限，不能晚于观测或预测形成 |
| timing.reference_observed_at | 本地代码取得参考快照的时间 |
| predictions.created_at | 真正写入账本的时间，区别于预测形成 |
| scoring_context.horizon_end | prediction_formed_at + 1/5/21 个自然日 |
| scoring_context.window_start / window_end | 价格收益与资金费共同使用的实际参考价窗口 |
| outcome_available_at | 完整终点日线的保守闭合时间；不是整份结果落账时间 |

**冻结参考价。** submit_prediction 由确定性行情代码获取参考，不接收 LLM 自报参考价。
价格、价格时间、可知时间、观测时间与来源一起冻结；价格必须有限且为正，时间必须带时区，满足
`price_at <= available_at <= observed_at <= formed_at`。同一已接受 horizon 的冲突重交被拒绝；相同内容重交不重新取价。
分次补交的不同 horizon 分别保留自身的形成时间。延迟 flush、之后的行情或历史数据修订均不能改动已冻结参考。
ToolNode 并行重交时，每个 capture 的检查、快照获取、接受与 flush 共享互斥锁；同一 horizon 只接受一次，
相同内容不重复取价，冲突内容拒绝。flush 等待正在进行的接受操作后重新检查状态；落账失败可重试且不重取参考价。
新口径预测的 revision 必须提供自己的新快照，不能继承旧时间，也不能因漏传 timing 静默降回 legacy。

**每日精度。** 永续使用形成时点前最近已收盘日线，来源 `binance_perp:1d:last`。
例如 9/5 12:00 UTC 形成的预测使用 9/4 日线、价格时间为 9/5 00:00 UTC；参考通常早于形成时间不到 24 小时。
UNDERLYING 使用 `yfinance:1d:close`，按相同 UTC 闭合界限选择最近工作日。
若精确应有日线缺失，记录 reference_error，不以更老价格补位，也不退回旧评分口径。
这是明确的**日线参考收益基准**，不是预测形成瞬间的价格或可执行成交价；不能用它声称消除了盘中价格延迟。

**固定终点。** 到期时，永续终点是 horizon_end 前最后已收盘日线，即 horizon_end 的 UTC 日期减一天。
股票再退到最近工作日，必须取得该精确日线；缺口不得由更早的可用数据替代。
不采用完整交易所节假日日历：工作日假期在此版本中会作为数据/日历缺口明确不可评分，不能伪装为完整。
期限到达前不评分，未闭合端点不评分，不把未来当日最终收盘价用于盘中参考或终点。
新口径到期取价对所需终点、股票原参考日检查重复 Close：完全相同的重复可以折叠，
矛盾值返回 `price_bar_conflict:日期`，按 7 天补数规则 pending → incomplete，不随输入顺序选择最后一条。
无关回看日期的冲突不阻挡本窗口；MU CONTRACT 的股票诊断冲突只使该诊断缺失。旧口径日线归一化保持兼容。
跨周末两端相同直接记 `incomplete / no_new_trading_session`，不将周日期限顺延到周一。

**统一持有窗口。** 新预测的价格收益为 `exit_price / frozen_reference_price - 1`；
资金费严格累计 `(reference_price_at, exit_close]`，实际窗口在 metadata 持久化。
例如 9/5 12:00 的永续 1d，期限为 9/6 12:00，价格窗口为 9/5 00:00 → 9/6 00:00。
5d 与 21d 使用同样规则。由此也能看出，窗口可能包含形成前的部分时段；它是日线基准，不能解释为提交后实际策略持仓收益。
股票永续的 underlying/basis 只作诊断；新口径无法匹配同一组端点时保留诊断缺失，不拼接其他窗口。

**股票复权基准。** 当前 yfinance history 的 Close 会随拆股、除息等调整历史价格。
到期新拉取的 Close 与此前冻结的 Close 未必使用同一基准；例如冻结 100，拆股后终点 55，不能直接声称亏损 45%。
新股票评分先比较本次行情中的原参考日收盘价与冻结值（相对容差 `1e-9`、绝对容差 `1e-12`）。
该重读仅校验基准，不替换冻结 entry。发生变化时写 `incomplete / equity_reference_basis_changed`；
无法取得原参考日数据时按 `equity_reference_basis_unverifiable` 补数，最多 7 天。
本版本不自动换算公司行动，也不把变化后的历史价格冒充预测时参考价；基准匹配的 5d/21d 窗口仍能正常评分。

**费用与标签。** 新评分必须有 entry/exit fee 与 slippage 分解，分别只收一次，不收 expected funding。
只有 estimated_cost 总额时无法排除其携带的预估资金费，按缺失 ticket_cost 处理。
资金费仍采用方向无关的 long-pay 基准 `funding_pnl = -sum(rate)`；
校准用 `net_return = price_return + funding_pnl - fees - slippage`，不是多空策略收益。
现有多空策略 view 仍单独披露：方向符号只作用于价格与 long funding，费用永远是成本，flat 为零。
POSITIONING 保留累计 funding rate 本身作为标签，新样本只把窗口起点换为明确记录的形成时间；不加价格或交易费用。

## 生命周期与历史兼容

| 情况 | 行为 |
|---|---|
| 未到 horizon_end | 不进入到期队列 |
| 新样本缺终点、资金费或费用分解 | 只返回内存 pending，不写占据唯一键的 outcome |
| 到 horizon_end + 7 自然日仍缺必要数据 | 写 incomplete、缺失腿和持久化原因，退出自动队列 |
| 新样本缺预测时参考价/时间非法/同根固定窗口 | 直接写不可评分 incomplete，不等待未来行情重建参考 |
| 已有任意 outcome（含历史 pending/incomplete） | 不覆盖、自动队列不再重复处理 |
| 新口径价格/资金费供应商异常 | 与数据缺口相同：pending → 固定 7 天期限后 incomplete |
| 其他单样本账本异常、旧口径供应商异常 | 报 failed，隔离错误并继续后续样本；未写 outcome 的异常可重试 |

7 天是这个版本的显式补数宽限，不改预测期限或实际持有窗口。它不是自动保证数据最终齐全。
旧口径缺 entry/funding/cost 的即时 incomplete 行为保留；旧终点缺口也使用上述 7 天停止重试规则。
工作队列分页时，零 complete 的一页会继续走向后页，永久缺口、终态与单行异常均不阻挡后续可评分样本。
相同内容 outcome 写入幂等，包含 metadata 的冲突重写仍拒绝。

旧 predictions 无 timing 的评分明确标为 `legacy_daily_v1`；已有 outcomes 无 metadata 也归入 legacy。
保留旧日期标签的日线近似和旧 scalar-cost 兼容行为，实际窗口、来源与 entry_precision 在新计算的 legacy outcome 中披露。
该旧口径的 exit 可能晚于名义 horizon_end，因此不能与新口径当作同一定义。
旧盘中 gate 仍生效，无法证明预测时已知的 entry 不被改造成 complete。
24/7 合约旧样本还必须取得分析当天的精确 entry 日线；当天缺失时写 `legacy_entry_bar_missing`，
不能退用前日价格而绕过 gate。股票周末仍使用明确的工作日规则。
本 P0 不实施自动历史参考补写或推断；只有原有路径中历史时点已经可验证的样本能够继续评分。
若要恢复已有不可变 outcome，需另行设计带独立版本/关联的追加复评记录，本次没有覆盖或复评入口。

scoreboard 只汇总合格、有限的 complete 标签，新口径还校验时间窗口、必要归因腿与 net 组合，并逐项核对原 prediction.timing。
新口径要求 `outcome_available_at == window_end`（POSITIONING 也适用）；价格类标签的 fees/slippage 必须非负，
不能用晚取得的结果时间替代固定终点，或把负成本重新合成 net 后混入统计。legacy 历史兼容口径不变。
仅在 outcome 自报新版本、原预测却无快照，或 metadata 与原快照不符的样本均被拒绝。
新增按 scoring_version 的分桶/筛选及混合版本披露；不可评分样本不会通过 partial net 或策略 view 混入校准。
旧日期近似样本仍属于明确的 legacy 统计，不能作为新精确时间契约的验收证据。

## round6b 的边界与尚待完成的验证

交接信息：preflight/analyze/reconcile/scoreboard 退出码均为 0，reconcile PASS，
BTCUSDT/ETHUSDT/MUUSDT acceptance PASS；predictions=30、evidence=21、complete_outcomes=0。
这是流程与一致性验收，不是预测表现验收。本次未重跑该阶段或修改 round6/round6b 文件。

- 24 条盘中 CONTRACT：无冻结参考价，继续按旧 gate 保留不可评分；单纯等待无法修复。
- MU 的两条周末 UNDERLYING 1d：固定同根窗口转为明确不可评分终态。
- MU 的其余四条 UNDERLYING 5d/21d：保留旧日线评分路径，仍取决于闭合行情与必要成本等数据完整性。
- 本次没有在真实账本上运行上述到期过程，以上是代码路径及离线模拟确认，不是原样本实际结果。

待另行安排：新预测真实提交参考快照审计；1/5/21d 自然到期后的真实数据覆盖、资金费结算与成本归因核验；
真实账户外的 shadow scoreboard 样本积累与预测校准评估。P1 多空仓位意图、P2 临近使用复核、P3 多角色增量实验均未执行。
离线测试不能证明预测有效、盈利或实盘可用。

## 离线验证记录

测试均使用模拟行情、临时目录、临时账本，不调用真实 LLM。
交接给定的 Python 路径不存在，实际使用 `C:\Users\warri\Desktop\Yialpha\.venv-dev\Scripts\python.exe`。
首次 11 项特征复现使用普通 pytest 入口；随后发现包初始化默认调用 dotenv，存在隐式读取 .env 的路径。
后续所有回归入口在导入 pytest/项目包前将 `dotenv.find_dotenv` 和 `dotenv.load_dotenv` 置为无读取函数；未输出或更改密钥文件。
前一轮验证（2026-09-06；A 阶段后续结果见下节）：

| 检查 | 结果 |
|---|---|
| 下列 18 个测试文件，禁用 dotenv 和 socket 网络连接 | **316 passed in 9.83s**，退出码 0 |
| 全部改动 Python 文件 Ruff | All checks passed |
| 9 个改动生产模块 mypy，`--no-site-packages --follow-imports=skip` | Success: no issues found |
| `git diff --check` | 通过 |
| round6/round6b 开工前后逐文件 SHA-256 | 原 611 个、现 611 个，0 个差异 |
| HEAD | 仍为交接 commit；未提交、推送或部署 |

回归覆盖原有 gate、三标的全部期限、新参考冻结、形成/落账时间区别、未闭合/缺口终点、
周末同根终态、供应商异常补数截止、资金费首部缺口/重复证据、费用单次归因、
写入幂等/冲突拒绝、队列无饥饿、新旧 schema 只读兼容/增量迁移、原快照与 scoreboard 关联验证，
以及 evidence、regime、POSITIONING、票据镜像、portfolio 与 execution isolation 的现有回归。
首次特征测试的错误行为断言已转换成修复后终态断言；原 intraday gate 的不可评分保护仍被测试。

在 309 项通过后的补充审查中，再以离线测试复现了两个遗漏：旧合约 entry 当天缺失可绕过 gate，
以及股票复权基准变化后仍把混基准收益记为 complete。修复后补充 7 项正反向检查，
形成上述 316 项最终结果；新增检查包含股票基准证据缺失的补数/终态与 scoreboard 排除。

## A 阶段最终审阅补充（2026-09-06）

在保留已有修改的基础上，三路并行审阅复现并修复了：并行同期限重交与 flush 竞态、
所需日线矛盾 Close 被顺序覆盖、新 scoreboard 接纳错误可用时间和负费用/滑点。
新增 22 项正反向回归：并发组修复前 3 failed / 1 passed，评分组修复前 11 failed / 7 passed，修复后均通过。
还修正了离线 fixture 的既有隔离遗漏：关闭无关测试的 advisory `regime_context`，并模拟 overlay stress 披露，
保留专门测试对供应商/计算函数的独立模拟，不改变生产功能开关。

最终同一 18 文件范围：**338 passed in 7.34s**，入口退出码 0，网络拦截计数 **0**。
Ruff、限定 9 模块 mypy 和保护目录/最终 diff 核验见 [P0_FINAL_REVIEW.md](P0_FINAL_REVIEW.md)。
下一阶段具体矩阵、独立账本/cache/原始证据、待实现入口与自然到期退出条件见
[P0_SHADOW_VALIDATION_PLAN.md](P0_SHADOW_VALIDATION_PLAN.md)；本阶段没有执行真实验证。

隔离诊断过程如实保留：首轮 338 项断言虽通过，但加严入口拦截了 105 次既有 advisory 行情请求，因此入口以 1 退出；
调用均在连接/DNS 或原生 curl 传输前阻断，没有发出真实请求。该轮未预先重定向 yfinance 自身 cache，
可能读取了现存时区/cookie cache；检查文件元数据时原两份 DB 的修改时间未变，但这不能证明从未访问。
随后先将 yfinance 内部 cache 重定向临时目录，定位并修正两条遗漏路径，才得到上述零网络尝试的最终结果。
没有查看或输出 cookie 值，没有读写 `.env`，也没有通过重跑原验收验证这些修复。

在项目根目录可用以下 PowerShell 命令复现当前同一组离线测试（原生 curl、dotenv 与内部 cache 均隔离）：

```powershell
$p0Tests = @'
import os
import sys
import tempfile
from pathlib import Path
os.environ["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
def deny_env_open(event, args):
    if event == "open" and isinstance(args[0], (str, bytes, os.PathLike)):
        name = os.fsdecode(args[0]).replace("\\", "/").rsplit("/", 1)[-1]
        if name == ".env" or name.startswith(".env."):
            raise RuntimeError(".env access forbidden during P0 regression")
sys.addaudithook(deny_env_open)
import dotenv
import dotenv.main
dotenv.find_dotenv = dotenv.main.find_dotenv = lambda *a, **k: ""
dotenv.load_dotenv = dotenv.main.load_dotenv = lambda *a, **k: False
import socket
blocked_network = []
def no_network(*args, **kwargs):
    blocked_network.append(True)
    raise RuntimeError("network disabled for P0 offline regression")
socket.socket.connect = no_network
socket.socket.connect_ex = no_network
socket.socket.sendto = no_network
socket.getaddrinfo = no_network
import curl_cffi
curl_cffi.Curl.perform = no_network
import yfinance
p0_temp = Path(tempfile.mkdtemp(prefix="yialpha-p0-offline-"))
yfinance.set_tz_cache_location(str(p0_temp / "yfinance-internal"))
import pytest
tests = [
    "tests/test_outcome_compute.py", "tests/test_ledger_outcomes.py",
    "tests/test_ledger_predictions.py", "tests/test_prediction_tool.py",
    "tests/test_scoreboard.py", "tests/test_prediction_time_contract.py",
    "tests/test_p0_time_regressions.py", "tests/test_p0_review.py",
    "tests/test_regime_state.py", "tests/test_ledger_transaction.py",
    "tests/test_ledger_replay_idempotency.py", "tests/test_tickets_mirror.py",
    "tests/test_positioning_analyst.py", "tests/test_ledger_evidence.py",
    "tests/test_evidence_time_contract.py", "tests/test_v21_ledger_wiring.py",
    "tests/test_v24_portfolio_control.py", "tests/test_execution_isolation.py",
]
result = pytest.main(tests + ["-q", "--basetemp", str(p0_temp / "pytest")])
print(f"P0 blocked network attempts: {len(blocked_network)}")
raise SystemExit(result or (1 if blocked_network else 0))
'@
& .\.venv-dev\Scripts\python.exe -c $p0Tests
```

## B0 实现与离线验证补充（2026-09-07）

B0 离线入口（runner / guard / recorder / manifest / 逐行审计 / CLI）实现完成后，端到端离线
demo 首次暴露 6 处缺陷，均已修复并补回归：

| 缺陷 | 修复 |
|---|---|
| `sqlite3.connect/handle` 审计事件对被路径检查中止的半初始化连接触发，`execute` 抛 `ProgrammingError` 且 authorizer 未装配 | PRAGMA/authorizer 装配移入 `isolated_runtime` 内对 `sqlite3.connect` 的包裹补丁；路径检查仍在建文件前于审计钩子失败关闭 |
| 审计钩子拦截 `socket.gethostname`（pandas 导入时 `platform.uname()` 的本地调用，无网络副作用） | 白名单 `socket.gethostname`，其余 socket 事件仍拒绝并计数 |
| `runner.PROJECT_ROOT` 多取一层 `.parent` 指向 Desktop，git 快照调用失败 | 修正为 `parents[2]`（与 guard 一致） |
| filelock（yfinance 依赖）导入时在系统临时目录探测 symlink，被写入白名单拦截 | 进入隔离 runtime 前预导入 `filelock` / `yfinance`（cache manager 保持惰性，guard 的已初始化检查仍生效） |
| `Path.as_uri()` 三斜杠 `file:///C:/...` 被当作 UNC `//` 拒绝 | `_sqlite_path` Windows 盘符 URI 归一化接受任意前导斜杠数，UNC 仍拒绝 |
| `build_audit`/`write_audit` 用备份路径反推 cohort 根、`resolve(".")` 被路径规则拒绝 | `_report` 直接传活动账本路径；`write_audit` 改用 `require_runtime()` 根比对 |

新增端到端回归：`tests/test_p0_shadow_artifacts.py` 两个 runner 场景（baseline 12→11+1、
missing-funding 12→8+4）；`.env` 家族拒绝断言放宽为接受先注册的回归入口钩子抛出的 `RuntimeError`
（两种入口均失败关闭）。最终 20 文件回归 **381 passed in 9.73s**，网络拦截 0；
Ruff 与 `scripts/p0_shadow` 4 模块限定 mypy 通过；round6/round6b 保护目录 611 文件未变。

推送前全仓回归（2026-09-07，普通 pytest 入口）：**3348 passed / 4 skipped（环境性 skip）**。
暴露并修复两处顺序依赖：①全仓早先 dataflow 测试初始化 yfinance 进程级 cache manager，
guard 的"新进程"前置检查失败——两个 B0 测试文件加 autouse fixture 以 `set_location`
原位复位 `_db`；②conftest 的 stress-line 桩改为止拦截 3 天窗口内的近期日期分支，
历史日期走真实渲染（本就不抓取），`test_v2_overlay_ticket` 历史披露断言恢复通过。
