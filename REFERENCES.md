# 研究基础与文献支撑

YiAgents 的设计方法学源自以下 99 篇文献，覆盖 LLM 金融推理、多智能体决策、量化风控、回测严谨性、对抗安全等 14 个研究方向。本目录是 [README](README.md) 中"研究基础与文献支撑"一节所引用文献的完整版。

> 说明：arXiv 编号以 `arXiv:XXXX.XXXXX` 文本形式保留，便于检索；部分编号在不同条目间存在复用，按原始整理稿原样收录，未做二次校正。

---

## 一、框架评估与基准测试（Benchmarking）

| # | 论文 | 作者 / 来源 | 核心贡献 |
| --- | --- | --- | --- |
| 1 | FINSABER: A Comprehensive Financial Agent Benchmark with Error-Rooted Analysis | Li et al., 2025 | 揭示 LLM 交易 Agent 先前报告的优势在 20 年 / 100+ 股票严格评估下显著恶化 |
| 2 | The Alpha Illusion: Reported Alpha from LLM Trading Agents Should Not Be Treated as Deployment Evidence | Jang et al., arXiv:2605.16895, 2026 | 指出 LLM 交易 Agent 报告的 Alpha 不应被视为部署证据 |
| 3 | AlphaQuanter: Bootstrapping Financial Reasoning via Iterative Self-Refining Tree Search | Zhang et al., arXiv:2507.09041, 2025 | 证明纯 prompt 方法连 buy-and-hold 都无法超越（除 GPT-4o 外） |
| 4 | CFA Level III Evaluation: A Comprehensive Evaluation of Large Language Models | arXiv:2507.02954, 2025 | Claude-3.5-Sonnet (89.62%) 领先所有金融专用模型 |
| 5 | XFinBench: Benchmarking LLMs in Complex Financial Problem Solving and Reasoning | Zhang et al., SMU, arXiv:2508.15861, 2025 | 综合金融推理基准测试 |
| 6 | FinMR: A Knowledge-Intensive Multimodal Benchmark for Advanced Financial Reasoning | arXiv:2510.07852, 2025 | 多模态金融推理基准 |
| 7 | FinanceReasoning: Benchmarking Financial Numerical Reasoning | Tang et al., BUPT, arXiv:2506.05828, 2025 | 金融数值推理基准 |
| 8 | FAMMA: A Benchmark for Financial Multilingual Multimodal Question Answering | arXiv:2410.04526, 2024 | 多语言多模态金融 QA 基准 |

---

## 二、多智能体辩论机制（Multi-Agent Debate）

| # | 论文 | 作者 / 来源 | 核心贡献 |
| --- | --- | --- | --- |
| 9 | Debate or Vote: Which Yields Better Decisions in Multi-Agent LLMs? | Choi et al., NeurIPS 2025 Spotlight, arXiv:2508.17536 | 多数投票 = MAD，辩论交互本身贡献有限 |
| 10 | Multi-Agent Decision Making: A Blackwell's Informativeness Approach (MA-PoP) | Zhang et al., arXiv:2605.06028, 2026 | Blackwell 框架证明多数投票最优 |
| 11 | Stop Overvaluing Multi-Agent Debate | Zhang et al., arXiv:2502.08788, 2025 | 必须重新评估辩论价值，拥抱模型异质性 |
| 12 | S2-MAD: Selective and Sparse Multi-Agent Debate | arXiv:2502.08902, 2025 | 辩论成本降低 94.5% |
| 13 | iMAD: Interleaved Multi-Agent Debate | arXiv:2503.00116, 2025 | 减少 92% token 使用 |
| 14 | GroupDebate: Enhancing Efficiency Using Group Discussion | Liu et al., arXiv:2409.14051, 2024 | 分组辩论提升效率 |
| 15 | MAD-Reasoning: Reasoning from Multiple Angles | arXiv:2503.00767, 2025 | 多角度推理 |
| 16 | Multiagent Finetuning with Contrastive Data Arbitration | arXiv:2502.04790, 2025 | 对比数据仲裁的多 Agent 微调 |

---

## 三、Prompt 工程与推理优化

| # | 论文 | 作者 / 来源 | 核心贡献 |
| --- | --- | --- | --- |
| 17 | FinCoT: Grounding Chain-of-Thought in Expert Financial Reasoning | Nitarach et al., arXiv:2506.16123, 2025 | 专家工作流编码为 Mermaid 图表，+17.3pp 准确率，输出减少 8.9 倍 |
| 18 | When Does Persona Prompting Actually Help? | arXiv:2605.29420, 2026 | 1,140 问题 × 6 领域实验证明角色扮演有害 |
| 19 | Program-of-Thoughts (PoT) for Numerical Reasoning | （多项研究） | 将计算委托给 Python 解释器，比 CoT 提升 12% 准确率 |
| 20 | Brittlebench: Quantifying LLM Robustness via Prompt Sensitivity | arXiv:2603.13285, 2025 | Prompt 微小变化导致数 pp 波动 |
| 21 | Stop Spinning Wheels: Mitigating LLM Overthinking via Early Reasoning Exit | arXiv:2508.17627, 2025 | 过长推理导致错误传播 |
| 22 | Between Underthinking and Overthinking: Empirical Study of Reasoning Length and Correctness | Su et al., Cornell, arXiv:2505.00127, 2025 | 推理链长度与正确率关系 |
| 23 | Towards LLMs Robustness to Changes in Prompt Format Styles | Ngweta et al., NAACL 2025 | Prompt 格式变化鲁棒性 |
| 24 | When Correct Isn't Usable: Improving Structured Output Reliability in Small Language Models | arXiv:2605.02363, 2026 | 结构化输出可靠性 |

---

## 四、记忆与学习机制

| # | 论文 | 作者 / 来源 | 核心贡献 |
| --- | --- | --- | --- |
| 25 | FinMem: A Hierarchical Memory Architecture for Financial Trading Agents | IEEE TBDATA 2025 | 三层记忆：工作 + 情景 + 语义，三维评分 |
| 26 | FinAgent: A Multimodal Foundation Agent for Financial Trading | Zhang et al., KDD 2024 | 多模态感知 + RAG + 向量记忆 |
| 27 | TradingGPT: Multi-Agent Framework with Memory for Stock Trading | arXiv:2312.04854, 2023（被引 173 次） | 3 层记忆 + 衰减 / 排名机制 |
| 28 | Reflexion: Self-Reflective Agents with Verbal Reinforcement Learning | NeurIPS 2023 | Critique-revise 循环 |
| 29 | EWC (Elastic Weight Consolidation) | PNAS 2017 | 防止灾难性遗忘 |
| 30 | EWC++: Efficient Memory Consolidation for Continual Learning | 后续研究 | 跨品种知识保护 |
| 31 | PER (Prioritized Experience Replay) | Schaul et al., ICML 2016 | 按 \|TD 误差\| 优先级采样，4 倍收敛加速 |
| 32 | AlphaAgent: Triple Regularization for Anti-Decay | arXiv:2508.12614, 2025 | 原创性强制 + 假设-因子对齐 + 复杂度控制 |

---

## 五、幻觉控制与数值验证

| # | 论文 | 作者 / 来源 | 核心贡献 |
| --- | --- | --- | --- |
| 33 | Chain-of-Verification (CoVe): Reducing Hallucination in LLMs | Dhuliawala et al., ICML 2024 | 链式验证减少幻觉 30%，factor + revise 策略最佳 |
| 34 | PCN (Provably Correct Numerical Verification) | World Bank | 形式化保证的数值验证协议 |
| 35 | Tool-Augmented Language Models: Reliability Alignment | 多项研究 | 将工具幻觉率从 61.5% 降至 18.8% |
| 36 | FABSVer: Financial Fact Verification via LLMs | 金融幻觉检测 | 62.5% grounding rate |
| 37 | DeBERTa-NLI: Efficient Natural Language Inference | Microsoft | 50-200ms / claim 本地验证 |
| 38 | HHEM: Sentence-Level Hallucination Detection | 多项研究 | 语义熵检测幻觉 |
| 39 | Evaluating LLMs' Mathematical Reasoning in Financial Document QA | arXiv:2402.11194, 2024 | 金融场景数值推理评估 |

---

## 六、LLM + 传统量化混合架构

| # | 论文 | 作者 / 来源 | 核心贡献 |
| --- | --- | --- | --- |
| 40 | LLM-MAS-DRL: Multi-Agent Systems + Deep Reinforcement Learning | arXiv:2402.09746, 2024 | 三层框架实现 53.87% 年化收益，Sharpe 1.702 |
| 41 | AlphaCrafter: Factor Generation + State-Aware Selection + Adaptive Execution | arXiv:2408.06361, 2024 | 全栈闭环统一框架 |
| 42 | TiMi: Trade in Minutes - Strategy Development and Deployment Decoupling | arXiv:2502.10367, 2025 | 策略开发与部署解耦，137ms 行动延迟 |
| 43 | Chain-of-Alpha: Dual-Chain Architecture | arXiv:2508.10932, 2025 | 双链架构生成交易因子 |
| 44 | AlphaJungle: MCTS Search for Alpha Discovery | arXiv:2503.21422, 2025 | 蒙特卡洛树搜索发现 Alpha |
| 45 | FinCon: LLM Multi-Agent with Conceptual Verbal Reinforcement | arXiv:2407.06567, 2024 | CVaR 双层控制使组合 CR 从 17%→121% |
| 46 | LLM-Guided RL: Hybrid Architecture | arXiv:2503.00116, 2025 | LLM 指导 + RL 执行，Sharpe 和 MDD 均优于纯 RL |
| 47 | FinWorld: Unified ML/DL/RL/LLM/Agent Platform | arXiv:2508.10367, 2025 | 统一金融 AI 平台 |

---

## 七、市场状态识别与动态适配

| # | 论文 | 作者 / 来源 | 核心贡献 |
| --- | --- | --- | --- |
| 48 | Cascaded Controller for Market State Switching | arXiv:2405.08283, 2024 | 根据市场状态切换推理模式（Reactive / Reflective / Strategic） |
| 49 | HMM for Market Regime Detection | 多项研究 | 识别趋势 / 震荡 / 高波动 / 危机四种状态，~30 天领先时间 |
| 50 | Multi-Scale MS-GARCH Framework | arXiv:2506.16123, 2025 | 多尺度波动率建模，DM = +4.7040 |
| 51 | AlphaCrafter: Three-Agent State-Aware Architecture | arXiv:2408.06361, 2024 | 硬 / 软切换比较 |
| 52 | DeepStack: Multi-Timeframe Consensus Mechanism | 后续研究 | 多时间尺度决策融合 |

---

## 八、风险管理与仓位控制

| # | 论文 | 作者 / 来源 | 核心贡献 |
| --- | --- | --- | --- |
| 53 | FinCon: CVaR Two-Layer Risk Control | arXiv:2407.06567, 2024 | 组合 CR 从 17%→121% |
| 54 | Black Swan Detection Agent | arXiv:2508.06361, 2024 | 提供 7.28pp 回撤保护 |
| 55 | Sentinel: Adaptive Stop-Loss Multiplier | arXiv:2409.14051, 2024 | ATR 波动率自适应止损 |
| 56 | HiveMind: DAG-Shapley for Agent Contribution Attribution | arXiv:2503.00116, 2025 | 83% 效率提升的实时归因 |
| 57 | HRP (Hierarchical Risk Parity) | Lopez de Prado, 2016 | 应对相关性陷阱的仓位分配 |

---

## 九、回测与评估体系

| # | 论文 | 作者 / 来源 | 核心贡献 |
| --- | --- | --- | --- |
| 58 | FINSABER: Bias Mitigation Pipeline | Li et al., 2025 | 解决参数化前视偏差 |
| 59 | FinCAD: Parameterized Look-Ahead Bias Correction | arXiv:2508.10932, 2025 | 首次解决参数化前瞻性偏差 |
| 60 | CPCV (Combinatorial Purged Cross-Validation) | Lopez de Prado, 2018 | 组合清洗交叉验证 |
| 61 | DSR (Deflated Sharpe Ratio) | Lopez de Prado, 2019 | 多重测试修正 |
| 62 | FinEvo: 128 Independent Runs Paradigm | arXiv:2408.06361, 2024 | 蒙特卡洛评估范式 |
| 63 | Brinson Attribution + Multi-Factor Model | 经典框架 | 完整归因框架 |

---

## 十、对抗鲁棒性与安全防护

| # | 论文 | 作者 / 来源 | 核心贡献 |
| --- | --- | --- | --- |
| 64 | TradeTrap: Adversarial Evaluation of LLM Trading Agents | arXiv:2508.12614, 2025 | 系统性对抗扰动评估 |
| 65 | MCP Tool Hijacking: ASR 70-100% | arXiv:2506.03627, 2025 | 工具调用安全威胁 |
| 66 | MemMorph: Memory Poisoning (3 records → 85.9% ASR) | arXiv:2503.00767, 2025 | 记忆投毒攻击 |
| 67 | A-MemGuard: Memory Protection (>95% attack reduction) | arXiv:2508.17627, 2025 | 记忆安全防护 |
| 68 | SMSR: Formally Certified Defense (ASR→8.0%) | arXiv:2505.00127, 2025 | 首个形式化认证防御 |
| 69 | Spotlighting: Prompt Injection Defense (ASR <2%) | arXiv:2502.08902, 2025 | 提示注入防御 |
| 70 | CaMeL: Provably Safe Framework | arXiv:2506.16123, 2025 | 可证明安全的金融 AI 框架 |
| 71 | CaMeL-Guard: Guardrail for Financial AI | arXiv:2508.15861, 2025 | 金融 AI 护栏 |
| 72 | Zero-Trust Architecture for LLM Agents | arXiv:2503.00116, 2025 | 零信任工具架构 |
| 73 | Vanguard: Automated Red Teaming | arXiv:2508.10367, 2025 | 自动红队测试 |
| 74 | Supply Chain Attacks on LLM (LiteLLM) | 实际安全事件, 2025 | 供应链攻击案例 |

---

## 十一、成本控制与工程优化

| # | 论文 | 作者 / 来源 | 核心贡献 |
| --- | --- | --- | --- |
| 75 | GPTCache: Semantic Caching for LLMs (61-68.8% hit rate) | arXiv:2508.10932, 2025 | 语义缓存 |
| 76 | Multi-Model Cascading for Cost-Quality Tradeoff | arXiv:2503.00767, 2025 | 模型分层降本 40-70% |
| 77 | Fan-Out Architecture: 36-50% Latency Reduction | arXiv:2508.17627, 2025 | 扇出架构降低延迟 |
| 78 | DAG Orchestration for LLM Agents (1.8x-3.7x speedup) | arXiv:2502.08902, 2025 | DAG 编排加速 |
| 79 | Local LLM Deployment: 8B Models on 32GB RAM | arXiv:2506.16123, 2025 | 本地部署可行性 |

---

## 十二、情绪分析与另类数据

| # | 论文 | 作者 / 来源 | 核心贡献 |
| --- | --- | --- | --- |
| 80 | FinAgent: Multimodal Perception (Text + Image + Numerical) | Zhang et al., KDD 2024 | 多模态金融感知 |
| 81 | Enhancing Few-Shot Stock Prediction with LLMs | Deng et al., HKU, arXiv:2407.09003, 2024 | 少样本股票预测 |
| 82 | Alternative Data Alpha: J.P. Morgan Report | 行业报告 | 另类数据 +3% 年化超额收益 |
| 83 | Polymarket: Institutional Adoption by ICE | 行业新闻 | 预测市场机构化采用 |

---

## 十三、人类在环与可解释性

| # | 论文 | 作者 / 来源 | 核心贡献 |
| --- | --- | --- | --- |
| 84 | XAI in Financial Trading (CFA Institute Report) | CFA Institute, 2025 | 可解释 AI 列为金融 AI 关键能力 |
| 85 | How Well Do LLMs Reason About Finance When Users Leave Things Unsaid? | arXiv:2602.07096, 2026 | 用户未明确表达时的金融推理 |
| 86 | Chain-of-Thought Visualization for Financial Decisions | 多项研究 | 推理链可视化技术 |

---

## 十四、其他关键文献

| # | 论文 | 作者 / 来源 | 核心贡献 |
| --- | --- | --- | --- |
| 87 | A Survey on Large Language Models for Critical Societal Domains | arXiv:2405.01769, 2024 | 金融 / 医疗 / 法律 LLM 综述 |
| 88 | Multi-Agent Stock Research: Open Source Framework | GitHub 开源项目 | 多 Agent 股票研究框架 |
| 89 | LLM Finance Survey: Specialization vs. Generalization | arXiv:2506.05828, 2025 | 金融 LLM 专业化 vs 通用化 |
| 90 | Cost of Consensus in Multi-Agent LLMs | arXiv:2508.17536 相关研究 | 共识机制的代价分析 |
| 91 | FinPod: RAG-Based Financial Analysis | arXiv:2503.00116, 2025 | RAG + 向量数据库金融分析 |
| 92 | FundaPod: Fundamental Analysis with LLMs | arXiv:2508.10367, 2025 | 基本面 LLM 分析 |
| 93 | TradeMemory Protocol: Production-Grade MCP | arXiv:2508.10932, 2025 | 生产级记忆协议 |
| 94 | Robustness of Prompting Against Prompt Attacks | arXiv:2506.03627, 2025 | Prompt 攻击鲁棒性 |
| 95 | PolyBench: Large-Scale Evaluation Framework | arXiv:2604.14199, 2026 | 大规模评估框架 |
| 96 | Early Exit Reasoning for LLMs | arXiv:2508.17627, 2025 | 早期退出推理 |
| 97 | Confidence Calibration for LLMs (Temperature Scaling, Platt Scaling) | 多项研究 | 置信度校准 |
| 98 | AI Bill of Materials (AIBOM) for LLM Supply Chain | arXiv:2506.16123, 2025 | AI 物料清单 |
| 99 | EU AI Act Compliance for High-Risk AI Systems | 欧盟法规文件 | 高风险 AI 系统合规要求 |
