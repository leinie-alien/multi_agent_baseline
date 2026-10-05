# 多智能体 LLM 推理基线实验——最终报告

> 本报告面向答辩/交付,按 `whole_request.md` 的考核要求组织。过程记录、踩坑细节见 `LEARNING_NOTES.md`;设计动机见 `PLAN.md`。

**状态:草稿骨架,带 [TODO] 标记的部分等 PEV 重跑完成后填充,其余部分数据已齐全。**

---

## 1. 实验配置声明

答辩要求明确说明的四项(`whole_request.md` 第47节):

| 项目 | 配置 |
|---|---|
| 模型型号 | Qwen3.5-9B |
| 模型版本 | [TODO: 填具体 revision/commit hash,或说明用的是 HuggingFace 上哪个快照,路径见 `src/config.py` 的 `model_path`] |
| 是否量化 | 未量化(bf16) |
| 推理框架 | vLLM |

压缩/内存优化配置(`PLAN.md` 要求这类配置也要写进报告,不算模型量化但影响吞吐和显存占用):

| 参数 | 值 | 作用 |
|---|---|---|
| `tensor_parallel_size` | 2 | 模型切分到 2 张 GPU |
| `gpu_memory_utilization` | 0.85 | vLLM 显存预算上限 |
| `max_model_len` | 36864 | 上下文窗口上限(模型本身支持到262144,这里按显存预算保守设) |
| `max_num_seqs` | 32 | 最大并发序列数,从默认256调低以避免启动阶段OOM |
| `max_tokens` | 32000 | 三个基线统一的最大输出长度上限 |

硬件环境:4 × NVIDIA GeForce RTX 4090,实验使用 `CUDA_VISIBLE_DEVICES=0,1`。

---

## 2. 数据集与评测方法

- 数据集:MATH-500,按 5 个难度(Level 1-5)各抽 40 题,共 200 题,固定题目列表见 `data/selected_ids.txt`。
- 答案提取:`src/extract_answer.py` 的 `extract_answer()`,优先解析 `\boxed{}`,兜底处理分数/小数/LaTeX表达式。
- 正确性判断:`is_equiv()`,数学等价判断而非字符串完全匹配。
- 被截断(`finish_reason == "length"`)的输出禁用兜底提取规则(截断点附近的数字不代表模型真实结论),诚实返回 `None` 而非伪造答案。

---

## 3. 方法总览

| 方法 | 类型 | 说明 |
|---|---|---|
| Direct | 单智能体基线 | 直接给出答案,不要求显式推理过程 |
| CoT | 单智能体基线 | 要求显式逐步推理 |
| Self-Consistency | 伪协作基线 | N=5独立采样+多数投票,考核要求明确排除此类"同模型多次采样"算作真多智能体协作 |
| Debate | 真多智能体协作 | Proposer(3 Solver)→Critic→Refine→Judge,角色间有真实信息交互 |
| PEV | 真多智能体协作 | Planner→Executor→Verifier,功能分工,Verifier判FAIL可打回Planner重新规划(最多2次反馈) |

Self-Refine(单模型自我修正循环)被明确排除,排除依据见 `PLAN.md` 的"明确排除项"章节:核心是信息流结构(有无多份独立候选互相验证),不是"是否单模型"。

每种方法均用 3 个 seed(41/42/43)各跑一次,全部关闭原生 thinking(`--disable-thinking`,依据见第5节)。

---

## 4. 整体结果对比表

全部数据来自 `results_no_thinking/<method>/run_seed{41,42,43}/summary.json`,3 seed 均值。

| Method | Accuracy | Avg Input Tokens | Avg Output Tokens | Avg Total Tokens | 截断率 |
|---|---:|---:|---:|---:|---:|
| Direct | 94.17% | ~108 | ~2,334 | 2,443 | 3.83% |
| CoT | 94.00% | ~118 | ~2,209 | 2,328 | 2.67% |
| Self-Consistency | 94.17% | ~591 | ~10,772 | 11,363 | 2.33% |
| Debate | 94.50% | ~821 | ~6,602 | 7,423 | 1.63% |
| PEV | [TODO: 重跑后填] | [TODO] | [TODO] | [TODO] | [TODO] |

> PEV 当前数据(93.17%,方差91.0%/93.5%/95.0%)基于已确认的 Verifier 判决解析 bug 得出,**不可用于最终结论**,见第7节。

---

## 5. Thinking 开关的影响(补充实验)

对三个基线额外跑了一组强制关闭 thinking 的对照(`results_no_thinking/`),用于判断 Qwen3.5-9B 的原生 thinking 模式是否值得保留:

| 方法 | 默认(开thinking) | 关闭thinking | 准确率变化 | token节省 |
|---|---|---|---|---|
| CoT | 94.00%,7,092 tokens | 94.00%,2,209 tokens | 0 | -68.85% |
| Self-Consistency | 95.00%,35,562 tokens | 94.17%,10,771 tokens | -0.83pp | -69.71% |

结论:原生 thinking 在本实验设置下更接近"token 浪费"而非"必要的推理增强"——关闭后 CoT 不掉分省7成token,SC仅掉0.83个百分点但同样省7成token,且收益不集中在难题(Level 5反而是负收益)。基于此,Debate/PEV 的正式实验直接采用关闭 thinking 的配置。详细数据见 `LEARNING_NOTES.md` 2026-09-23 记录。

---

## 6. 分难度(Level 1-5)对比

3 seed 均值,数据来自 `scripts/analyze_collaboration.py` 的输出。

### 准确率

| Method | L1 | L2 | L3 | L4 | L5 |
|---|---:|---:|---:|---:|---:|
| Direct | 97.50% | 95.00% | 94.17% | 95.00% | 89.17% |
| CoT | 95.83% | 94.17% | 94.17% | 94.17% | 91.67% |
| Self-Consistency | 95.00% | 95.00% | 95.00% | 95.00% | 90.83% |
| Debate | 96.67% | 95.83% | 95.00% | 95.00% | 90.00% |
| PEV | [TODO: 重跑后填] | [TODO] | [TODO] | [TODO] | [TODO] |

### 平均 Total Tokens

| Method | L1 | L2 | L3 | L4 | L5 |
|---|---:|---:|---:|---:|---:|
| Direct | 782.9 | 1,039.5 | 1,686.7 | 1,834.0 | 6,870.9 |
| CoT | 1,063.4 | 945.1 | 1,464.0 | 2,192.5 | 5,973.4 |
| Self-Consistency | 4,633.2 | 5,118.6 | 8,514.9 | 10,662.7 | 27,884.7 |
| Debate | 3,054.2 | 3,697.1 | 4,967.6 | 5,971.1 | 19,425.0 |
| PEV | [TODO: 重跑后填] | [TODO] | [TODO] | [TODO] | [TODO] |

**观察**:协作收益不集中在难题(Level 4-5)上,Debate 的优势主要体现在 Level 1-3,这跟 thinking 分析得出的"收益不集中在难题"结论一致。

---

## 7. 协作机制深度分析(考核重点:不只看总分)

### 7.1 Debate:早停占 95.83%,真正协作发生时是净正收益

- 600 道题(3seed×200)中,**575 道(95.83%)三个 Solver 第一轮就一致,直接早停**,只有 25 道(4.17%)真正进入 Critic→Refine 阶段。
- 这 25 道题 75 个 solver 位的修正结果:**wrong→right 13 次(17.33%),right→wrong 1 次(1.33%)**。`PLAN.md` 预判的"小模型当Critic从众改错"风险确实存在,但发生频率远低于"协作改对",净比例 13:1,是正收益。
- Judge 阶段只触发 8 次,样本量太小,未能得出其独立价值的结论。
- 与 Self-Consistency 逐题配对比较:562题同对,30题同错,5题SC错Debate对,3题SC对Debate错,**净赢2题**,且只用约65%的token。

**结论**:Debate 的优势不是靠复杂协作全程发力,而是"3人投票+早停"几乎零成本复刻了 Self-Consistency 的稳定性,外加一个真正发生时net-positive的纠错环节。

### 7.2 PEV:发现并修复了一个核心 bug,结果待重新验证

**[TODO: 本节需要在 PEV 修复后重跑验证完成后重写]**

初次分析发现一个反直觉结果:三个seed的 Verifier `fail` 全部是 0(600次运行从未判过FAIL),一开始怀疑是模型当Verifier过于宽松,但读代码后确认是**实现bug**:verify阶段只检查输出里有没有 `\boxed{}`,从未真正解析模型给出的 PASS/FAIL 判决文字。模型有"总要给个答案"的倾向,即使正文判断所有候选FAIL,仍可能习惯性给出一个boxed答案,导致该机制从未被正确触发。

修复方案:要求Verifier明确输出 `Overall: PASS/FAIL` 判决行,代码解析该行,只有"判决PASS且有boxed答案"才算真正通过,否则进入反馈重试。修复代码已完成并通过21个单测,**但尚未在真实GPU上验证、PEV的200题数据尚未用修复后的代码重跑**(详见 `LEARNING_NOTES.md` 2026-10-05 记录)。

修复前的数据(仅供参考,不代表修复后的真实协作效果):
- PEV 整体准确率 93.17%(方差大:91.0%/93.5%/95.0%),低于三个基线和Debate
- 实质退化为"2个executor候选+从不拒绝的仲裁步骤",仲裁本身的对错波动直接决定了方差
- 与Self-Consistency逐题配对:净输6题

[TODO: 重跑后补充真实的 Verifier拦截率/误判率/反馈重试触发率数据]

---

## 8. 核心问题回答(`whole_request.md` 第十五节)

### Q1: CoT 相比 Direct,是否提高了准确率?

数据:Direct 94.17% vs CoT 94.00%(关闭thinking)。**没有观测到提升**,两者基本持平,CoT反而略低(差异在3个seed的波动范围内)。这跟直觉("逐步推理应该更准")不符,可能原因:Qwen3.5-9B在这个难度的MATH题上,直接给答案已经接近模型能力上限,显式CoT结构本身没有带来额外推理增益,反而多出的输出长度增加了被截断的风险窗口。

### Q2: Self-Consistency 相比单次 CoT,是否进一步提高准确率?

数据:CoT 94.00% vs Self-Consistency 94.17%(关闭thinking)。**小幅提升(+0.17pp)**,代价是token消耗从~2,328增至~11,363(约4.9倍)。

### Q3: 准确率提升的代价是多少?Accuracy Gain vs Token Cost

| 方法 | Accuracy | Total Tokens | 相对Direct的token倍数 |
|---|---:|---:|---:|
| Direct | 94.17% | 2,443 | 1.0× |
| CoT | 94.00% | 2,328 | 0.95× |
| Self-Consistency | 94.17% | 11,363 | 4.65× |
| Debate | 94.50% | 7,423 | 3.04× |
| PEV | [TODO] | [TODO] | [TODO] |

**关键发现**:真多智能体协作(Debate)用比Self-Consistency更少的token(3.04×对4.65×),拿到了比三个基线都高的准确率(94.50%),是目前数据里综合性价比最好的方法。Self-Consistency作为"伪协作"基线,token成本最高但准确率提升有限,印证了考核设计的初始假设——多次独立采样投票的边际收益递减明显。

---

## 9. 待完成事项

- [ ] 验证 PEV Verifier 修复在真实 GPU 上生效(`--limit 5` sanity check)
- [ ] 重跑 PEV 200题正式实验(3 seed,覆盖 `results_no_thinking/pev/`)
- [ ] 重新跑 `scripts/analyze_collaboration.py`,更新第4/6/7.2/8节的 PEV 数据
- [ ] 补充模型版本/revision信息(第1节)
- [ ] (可选)如果时间允许,考虑 `PLAN.md` 里的拓展方案 C(Method-Diversity Ensemble)作为第三个协作架构补充对比
