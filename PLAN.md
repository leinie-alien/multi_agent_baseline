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
