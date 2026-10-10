# 多智能体推理基线实验 — 代码架构设计

## Context

目标是对比 Direct / CoT / Self-Consistency 三种推理方法在 MATH-500 上的准确率与 Token 成本。已给定固定的 200 道题目 ID 列表(来自 `HuggingFaceH4/math-500`),先把代码架构和逻辑搭好,环境和大规模运行留到之后。已确认的关键设计决策:

- Direct 和 CoT 也使用非零 temperature(如 0.6~0.7),这样"每种方法重复 3 次取均值"的实验设计对三种方法都有意义,也符合文档里"小模型结果有波动"的说法。
- Self-Consistency 用 N=5、temperature=0.7,同样重复 3 次(每次用不同 seed)。

## 整体目录结构

```
multi_agent_baseline/
├── data/
│   ├── selected_ids.txt          # 用户提供的200个题目unique_id,一行一个
│   └── math500_200.json          # 解析后的统一数据格式(需联网环境生成)
├── src/
│   ├── config.py                 # 全局配置:模型路径、采样参数、重复次数、seed等
│   ├── model_runner.py           # 统一模型调用接口 ModelRunner(封装 vLLM)
│   ├── prompts.py                # Direct / CoT 的 prompt 模板(chat 格式)
│   ├── extract_answer.py         # 答案提取 + MATH 官方等价判断逻辑(is_equiv 等)
│   ├── data_loader.py            # 加载 math500_200.json,按 level 分组等工具函数
│   └── methods/
│       ├── __init__.py
│       ├── base.py               # 公共的"跑一批题目+计分+存结果"逻辑
│       ├── direct.py
│       ├── cot.py
│       └── self_consistency.py
├── scripts/
│   ├── prepare_data.py           # 用 selected_ids.txt 从 HF 数据集筛出 math500_200.json
│   ├── run_experiment.py         # 统一入口:--method direct|cot|sc --repeat 3 --limit N
│   └── aggregate_results.py      # 汇总所有 run,生成整体表 + 分难度表(含均值)
├── results/
│   └── <method>/run_seed<seed>/
│       ├── per_question.jsonl
│       └── summary.json
├── configs/
│   └── config.yaml               # 可覆盖 src/config.py 默认值
├── tests/
│   └── test_extract_answer.py
└── README.md
```

相比文档建议结构的调整:三个 baseline 脚本合并为一个 `scripts/run_experiment.py` + `src/methods/*.py`,因为三者除 prompt/采样策略外逻辑高度重合(读数据→调用 ModelRunner→提取答案→判分→记录 token→存结果),拆成三个独立脚本会导致这段逻辑复制三份。`src/methods/base.py` 放公共流程,三个方法文件只提供 prompt 构造 + 采样参数 + (SC 特有的)投票逻辑。

## 核心组件设计

### 1. `data/selected_ids.txt` + `scripts/prepare_data.py`

把用户给的 200 个 `test/xxx/NNNN.json` 形式的 ID 写入 `data/selected_ids.txt`(一行一个)。`prepare_data.py`:
1. `datasets.load_dataset("HuggingFaceH4/math-500")` 加载全量 500 题(字段包括 `problem`, `solution`, `answer`, `subject`, `level`, `unique_id`)。
2. 按 `unique_id` 过滤出 selected_ids.txt 里的 200 道题。
3. 输出统一格式 `data/math500_200.json`,每条记录:
   ```json
   {
     "question_id": "q001",
     "unique_id": "test/algebra/2470.json",
     "subject": "algebra",
     "level": 3,
     "problem": "...",
     "answer": "..."
   }
   ```
4. 校验:确认最终恰好 200 条,且打印每个 level 的实际数量(理论上每级40,若不满40打印警告而非静默通过)。

这一步需要联网访问 HuggingFace,属于环境相关操作,先写好脚本,不在本阶段执行。

### 2. `src/model_runner.py` — 统一模型接口

```python
class ModelRunner:
    def __init__(self, model_path, tensor_parallel_size, gpu_memory_utilization, max_model_len):
        self.llm = LLM(model=model_path, tensor_parallel_size=tensor_parallel_size,
                        gpu_memory_utilization=gpu_memory_utilization, max_model_len=max_model_len)
        self.tokenizer = self.llm.get_tokenizer()

    def build_prompt(self, messages: list[dict]) -> str:
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def generate(self, prompts: list[str], temperature: float, max_tokens: int,
                 n: int = 1, seed: int | None = None) -> list[PromptResult]:
        sampling_params = SamplingParams(temperature=temperature, max_tokens=max_tokens, n=n, seed=seed)
        outputs = self.llm.generate(prompts, sampling_params)
        # 每个 PromptResult 包含: input_tokens, samples: [{text, output_tokens}]
```

关键点:
- 三个方法全部通过这一个类调用模型,保证同一 tokenizer、同一推理框架、同一 `max_tokens` 上限,避免"不同最大输出长度"的不公平问题。
- Token 计数直接用 vLLM 返回的 `prompt_token_ids` / `completion.token_ids` 的长度,不用字符串估算。
- `n` 参数原生支持 Self-Consistency 的多次采样。
- Token 统计口径:Self-Consistency 的 `total_input_tokens` 按 `input_tokens × n` 计,`total_output_tokens` 为 N 个 completion 的 output_tokens 之和,summary 里注明统计口径。

### 3. `src/prompts.py`

Direct / CoT 分别用不同 prompt 模板,末尾统一加一句格式要求 `"Please put your final answer within \\boxed{}."`,这样三种方法的答案提取逻辑可以完全统一。Self-Consistency 复用 CoT 的 prompt,只是采样参数不同(n=5, temperature=0.7)。

### 4. `src/extract_answer.py` — 答案提取与评测

基于 Hendrycks MATH 官方评测逻辑重新实现(不引入外部依赖,避免联网/版本问题):
- `last_boxed_only_string` / `remove_boxed`:提取最后一个 `\boxed{...}`
- `extract_answer`:找不到 boxed 则用 "final answer is..." 等模式兜底,最后兜底文本里最后一个数字/分数
- `normalize_answer`:去空格、统一分数写法、百分号、逗号千分位等
- `is_equiv`:先 normalize 再字符串比较,能转数值的再比较数值

三种方法都调用同一套逻辑,保证评测标准一致。

### 5. `src/methods/base.py` — 公共运行流程

统一处理:构造 prompts → 调用 ModelRunner.generate → 提取答案(单个或投票)→ 判分 → 记录 token → 组装结果。三个方法文件各自只提供 `message_builder`、采样参数,以及 SC 特有的 `majority_vote()`(处理平票,记录是否发生平票)。

### 6. `scripts/run_experiment.py`

统一入口:
```bash
python scripts/run_experiment.py --method direct --repeat 3 --limit 5   # sanity check
python scripts/run_experiment.py --method self_consistency --repeat 3 --n 5
```
结果写入 `results/<method>/run_seed<seed>/per_question.jsonl` + `summary.json`。

### 7. `scripts/aggregate_results.py`

汇总所有 run,产出方法整体表(Accuracy/Token mean±std)和分难度表,输出 markdown + csv。

## 当前阶段范围

本阶段(无 GPU/vLLM 环境)完成:完整代码架构 + `extract_answer` 的可本机验证单元测试。`prepare_data.py`(需联网)、sanity check、完整实验留给服务器环境阶段。

## 参考同需求的另一版实现(`multi_agent_baseline_0`)后的改动

对比了同目录下另一个从零实现的版本(用 `vllm serve` + OpenAI HTTP API,而不是 `LLM()` 直接调用),借鉴了两点、修了两个由此发现的 bug:

- **新增 `scripts/run_sanity_check.py`**:固定每个难度抽 1 题(共 5 题),依次跑三种方法,逐题打印 ✓/✗、预测答案、ground truth、token 数,最后打印三方法对比表。此前 `run_experiment.py --limit 5` 只是取数据集前 5 题(非分层),也没有这种交互式打印,不满足文档第十六节"用 2~5 题验证 pipeline"的要求。
- **`QuestionRecord` 新增 `raw_outputs` 字段**:存模型原始输出全文,之前只存了提取后的答案,排查"为什么某题答案提取失败"时无从下手。
- **修复 bug:`majority_vote` 投票前未做归一化**:原实现直接对 `extract_answer` 的原始字符串计票,会把 `1/2`、`\frac{1}{2}`、`0.5` 当成三个不同选项分票,稀释真实票数,可能改变投票结果。现在投票前先用 `normalize_answer` 归一化再分组计票,但 winner 仍返回原始文本(不是归一化后的中间形式),保证存入结果的 `prediction` 字段可读。
- **修复 bug(改上一条时引入又发现的):用原始答案字符串当 dict key 会丢失重复计数**。第一次修复时写成 `{a: normalize_answer(a) for a in valid}`,如果同一个原始答案重复出现(如 5 次采样里两次都输出 `"3"`),dict 会把重复 key 合并掉,计票直接漏掉重复次数,平票判断也会跟着错。改成用列表保存归一化结果、按下标取胜者,不再依赖 dict 去重。
- **`ModelRunner` 新增 `shutdown()` + `__enter__`/`__exit__`**:呼应 `LEARNING_NOTES.md` 记录的孤儿进程教训,尽力释放 vLLM 子进程与显存,`run_experiment.py` 已包进 `try/finally` 调用。
- 评估但未采用:改成 `vllm serve` + HTTP 的架构。原因:`LLM()` 原生支持 `n` 参数一次批量完成多次采样,比 HTTP + 循环调用更高效,且能达到同样的资源清理效果,已在讨论后确认保留现有架构。

---

# 多智能体协作扩展 — 下一阶段设计(主线锁定,避免中途跑偏)

## 任务要求回顾

考核要求(3)(4)是对比基线:单智能体 Direct、单智能体 CoT、多次采样投票(Self-Consistency)——上面 Direct/CoT/Self-Consistency 三个方法**继续保留,不作废**,200 题正式跑批完成后直接构成对比表的基线三行。

核心任务是(基于此)设计一个**真正的多智能体协作系统**:要求"角色之间存在分工与信息交互",明确排除"对同一模型多次独立采样后的简单投票集成"——即 Self-Consistency 本身不算数,不能靠换个名字重新交一遍。理想目标是准确率显著且稳定优于三个基线;考核重点是对协作机制在何种情形下有效/无效/负面的分析深度,不只是分数。

需要记录:模型、prompt、采样次数与聚合方式、token 消耗(多次运行均值)、准确率(多次运行均值)、量化/内存优化等压缩技术配置(现有 `tensor_parallel_size`/`gpu_memory_utilization`/`max_num_seqs` 已经算这一类,需要写进最终报告)。

## 主线:两个协作架构都做

### 主线 A:Debate(Proposer-Critic-Refine,对称角色)

1. 第一轮:2-3 个 Solver 角色(同一基座模型,不同角色 prompt,类似 Self-Consistency 的初始多样性但角色身份不同)各自独立解题。
2. 第二轮:Critic 角色看到所有 Solver 的**完整推理过程**(不只是最终答案),指出分歧点、检查每一步计算/逻辑错误。
3. 第三轮:Solver 们看到 Critic 的批评和彼此的解法后修正自己的答案。
4. 终局:Judge 角色读所有修正后的解法 + 批评摘要给出最终答案(或最后一轮再投票)。
5. **早停条件**:第一轮所有 Solver 已经一致时跳过 Critic/Refine 直接收敛,节省 token,同时这个早停比例本身是一个分析指标。

已识别风险:小模型当 Critic 时批评本身可能是错的(sycophancy/从众风险),Solver 顺着错误批评改错答案,这是"协作产生负面影响"的典型场景,分析时要专门统计。

### 主线 B:PEV(Planner-Executor-Verifier,非对称角色,功能分工而非内容分工)

1. Planner 生成 1-2 个候选解题策略(只出思路大纲,不做具体运算,出错空间小)。
2. Executor 严格按每个策略走完整推理,产出候选答案(不自己重新决定思路)。
3. Verifier 做机械性复核(代入验证/独立重新推导对比),不做自然语言点评,给出通过/不通过或修正后答案。
4. 若 Verifier 判定不通过或分歧未解决,反馈回 Planner 重新出策略(有限反馈轮次,而不是无条件多轮打嘴仗)。

相比 Debate,PEV 把"自然语言批评"这个对小模型较难的动作换成"机械代入检验",预期更稳定;分析时统计 Verifier 拦截住的错误比例(协作生效证据)与误判把对的推翻的比例(协作负面证据)。

## 拓展(可选,主线做扎实后再考虑,不摊薄主线精力)

### 拓展 C:Method-Diversity Ensemble + Reconciler(已实现,2026-10-10)

不同角色被要求用指定的不同解题方法(代数换元/几何作图/数值代入等),多样性来自方法本身而非 temperature 随机采样。Reconciler 只检查多方法是否收敛,不一致时挑一两种重新独立推导。交互只发生在最后仲裁环节,避免 Debate 的从众风险,工程成本低,可复用主线的 orchestrator。

**实现结果**:代码在 `src/multi_agent/mde.py`+`roles.py` 的 `make_mde_functions`,3个Solver强制分配 Algebraic/Geometric/Numerical 三种方法,`solver_temperature=0.3`(刻意压低以排除采样噪声的混淆,跟Debate的0.7形成对照组)。200题×3seed正式跑批:准确率94.33%(六个方法里第二高,仅次于Debate),avg_total_tokens 6,613,标准差0.29pp(与Router并列最低)。核心发现:solve阶段分歧率8.50%,约为Debate(4.17%)的2倍——证明"强制方法多样性"确实比"temperature采样多样性"制造更多真实分歧;但Reconciler介入后的改对/改错比例是4:4(net neutral),不像Debate的critic/refine那样展现13:1的净正收益,说明"制造更多分歧"和"更好地解决分歧"是两件独立的事。设计审查阶段还顺带发现了Debate现有代码一处归因脱节的技术债(Judge fallback时`raw_outputs`存证错位),已记录待修复,不影响已报告结果。详细结果见 `FINAL_REPORT.md` 第7.4节,过程记录见 `LEARNING_NOTES.md` 2026-10-09/10条目。

### 拓展 D:Router → Domain Specialist → 轻量 Verifier(已实现,2026-10-09)

Router 先判断题目类型(代数/几何/组合/数论/微积分),据此选择对应专家 prompt,专家作答后接轻量 Verifier。信息交互体现在"上游分类结果决定下游角色行为"。额外好处:可以按题目 subject(而非只按 Level)分层分析协作效果,前提是 `math500_200.json` 里保留了 `subject` 字段。

**实现结果**:代码在 `src/multi_agent/router.py`+`roles.py` 的 `make_router_functions`,7个类别用数据集原生英文 subject 字符串(非中文归类名,避免分析时需要模糊映射)。200题×3seed正式跑批:准确率94.17%,avg_total_tokens 4,970(是三个真协作架构里最低,仅为Debate的67%、PEV的49%),标准差0.29pp(六个方法里最低)。路由分类准确率46.67%,但分类正确(95.36%)vs分类错误(93.12%)时最终准确率只差2.24pp——证明"上游决策影响下游行为"是真实机制但影响幅度有限。详细结果见 `FINAL_REPORT.md` 第7.3节,过程记录见 `LEARNING_NOTES.md` 2026-10-09条目。

### 明确排除项:单模型自我修正循环(Self-Refine)

同一角色 prompt 自己反复看自己上一轮输出修正,没有真正的角色分工,精神上等同于"采样变成了迭代"的 Self-Consistency,不满足"角色分工+信息交互"要求,不作为候选方案。

**排除标准的澄清(避免与"Direct/CoT 也是单智能体"混淆)**:排除 Self-Refine 的依据不是"是否只用了一个模型实例",而是"有没有真正的角色分工+信息交叉"。Direct/CoT 本身就是单智能体基线,定位一直是对比基线,不需要也不宣称是协作,不构成排除的理由。Self-Refine 的问题在信息流结构:从头到尾只有一条串行的自我对话链(自己生成→自己看着上一轮输出批评自己→自己改),没有第二份独立产出进来对照,不满足"多份独立候选互相验证"这个协作的核心特征。对比 Debate:3 个 Solver 各自独立解题产出彼此不同的候选,Critic 同时看到多份独立结果找分歧点,Solver 再参考"别人的解法+批评"修正——这里有真正的信息交叉(一个角色的输出影响另一个角色的行为,且输入是多份独立候选)。Self-Refine 哪怕用 prompt 把同一次生成"扮演"成 solver/critic 两个角色,骨子里仍是单线自我修正,故排除。

## 代码架构改动(不碰现有三个基线代码)

新增平行子包,复用现有 `config.py` / `prompts.py` / `model_runner.py` / `results_io.py` 不变:

- `src/multi_agent/roles.py`:各角色 prompt 构造函数(`build_solver_messages`、`build_critic_messages(solver_outputs)`、`build_planner_messages`、`build_executor_messages(plan)`、`build_verifier_messages(question, answer)` 等),部分函数接收上一轮输出作为输入,这就是"信息交互"的代码落地。
- `src/multi_agent/orchestrator.py`:通用多轮调度器。每轮把当前待处理题目一次性 batch 调 `runner.generate`(保持现有代码"一次性处理全部题目"的效率做法,不逐题调用),按每题结果决定是否进入下一轮(早停判断/反馈判断),未收敛的题目带上下文进入下一轮。Debate 和 PEV 共用这份调度器,区别只在传入的角色 prompt 函数和"是否继续下一轮"的判断逻辑。
- `src/multi_agent/debate.py`、`src/multi_agent/pev.py`:分别用 orchestrator 拼出两套具体流程。
- `src/methods/base.py` 里 `QuestionRecord` 新增字段(纯新增,不改现有字段,基线兼容不受影响):`rounds_used: int`、`agent_trace: list[dict]`(角色名、轮次、生成文本、token 数明细)。汇总口径仍能填进现有 `summary.json` 格式(`input_tokens`/`output_tokens`/`total_tokens`)与基线直接对比,同时保留角色级明细供后续分析。
- `config.py` 新增 `DebateConfig`(如 `num_solvers`、`max_rounds`、`critic_temperature`)、`PEVConfig`(如 `num_strategies`、`max_feedback_rounds`),作为"用了几个 agent、最多几轮、聚合/终止方式"这类考核要求记录项的唯一代码来源。

## 分析计划(呼应考核对"深度分析"的要求)

- 按 Level(1-5,数据已有此字段)分层对比协作 vs 基线的准确率/token 增量:预期 Level 1-2 协作收益小甚至负(纯烧 token 或 Critic 误判改错),Level 4-5 协作可能有实质纠错。
- Debate:统计因初始一致而早停的题目占比、Critic 批评被 Solver 采纳后"改对"与"改错"的比例。
- PEV:统计 Verifier 拦截住的错误比例、误判把对的答案推翻的比例、平均反馈轮次。
- 如果做拓展 D:按 subject 分层看协作效果,作为除 Level 之外的第二个分析维度。
- 所有对比表(基线三个 + 协作两个,拓展视时间加入)统一走同一 `summary.json` 格式,汇总脚本 `scripts/aggregate_results.py` 需要能兼容新方法名。

## 当前状态

Direct/CoT/Self-Consistency 三个基线的 sanity check(每难度抽 1 题,共 5 题)已跑通,`test_methods_base.py` 已覆盖截断兜底逻辑。下一步:200 题正式跑批(3 个基线 × `DEFAULT_SEEDS` 三个 seed),完成后再启动主线 A/B 的代码实现。
