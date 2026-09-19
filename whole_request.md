# 多智能体 LLM 推理基线实验——任务说明

## 一、实验背景

需要完成一个关于 **LLM 推理方法 / 多智能体推理基线** 的实验。

核心目标是：

> 在相同的基座模型、相同的数据集上，对比不同推理方法的效果与 Token 成本。

最低要求比较以下三种方法：

1. 单智能体直接作答（Direct Answer）

2. 单智能体思维链（Chain-of-Thought, CoT）

3. 多次采样投票（Self-Consistency / Best-of-N）

最终需要比较它们的：

- Accuracy（准确率）

- Token 消耗（Input Tokens + Output Tokens）

---

# 二、实验要求

## 1. 基座模型

必须优先使用：

**Qwen3.5-9B**

任务文档允许在以下两个模型中任选：

- Qwen3.5-9B

- Qwen3.5-4B

当前选择：

**Qwen3.5-9B**

不建议更换其他模型。

最终答辩需要能够说明：

- 模型型号

- 模型版本

- 是否量化

- 推理框架

当前计划：

```text
Model: Qwen3.5-9B
Quantization: 暂不量化
Inference Framework: vLLM
```

---

# 三、数据集

必须使用：

**MATH-500**

实验不能只随机跑一些题。

要求按照数据集的五个 difficulty level 进行抽样：

```text
Level 1：40 道
Level 2：40 道
Level 3：40 道
Level 4：40 道
Level 5：40 道
----------------
总计：200 道
```

因此最终实验规模：

**200 questions**

如果后续有余力，可以增加其他数据集，但不是必须的。

当前优先把 MATH-500 的 200 道题完整跑通。

---

# 四、需要实现的三种方法

## 方法 1：Single Agent Direct Answer

最简单的 baseline。

流程：

```text
MATH Problem
     ↓
Qwen3.5-9B
     ↓
Final Answer
```

模型直接回答问题，不要求显式思维链。

例如 Prompt 可以要求模型：

```text
Solve the following math problem and give the final answer.

Problem:
{question}
```

需要记录：

- 输入 Token 数

- 输出 Token 数

- 最终答案

- 是否正确

- 推理时间（建议记录，但不是硬性指标）

---

# 五、方法 2：Single Agent CoT

单个 Qwen3.5-9B 进行显式推理。

流程：

```text
MATH Problem
     ↓
Qwen3.5-9B
     ↓
Reasoning / Step-by-step solution
     ↓
Final Answer
```

Prompt 可以要求：

```text
Solve the following math problem step by step.
Explain your reasoning clearly and provide the final answer.

Problem:
{question}
```

需要记录：

- Input Tokens

- Output Tokens

- 最终答案

- 是否正确

- 推理时间

注意：

**Direct 和 CoT 必须使用完全相同的模型和数据集。**

---

# 六、方法 3：Self-Consistency / Best-of-N

这是多次采样方法。

对于同一道题，让同一个模型独立生成多个答案，然后进行投票。

例如设置：

```text
N = 5
```

流程：

```text
                 ┌── Sample 1 ──→ Answer 1
                 │
                 ├── Sample 2 ──→ Answer 2
                 │
Question ────────┼── Sample 3 ──→ Answer 3
                 │
                 ├── Sample 4 ──→ Answer 4
                 │
                 └── Sample 5 ──→ Answer 5
                              ↓
                           Majority Vote
                              ↓
                         Final Answer
```

每次 Sample 都应该是独立采样。

可以使用：

```text
temperature > 0
```

从而产生不同的推理结果。

例如：

```text
N = 5
temperature = 0.7
```

然后提取每次生成的最终答案：

```text
Answer 1
Answer 2
Answer 3
Answer 4
Answer 5
```

进行 majority voting。

最终：

```text
Final Answer = 出现次数最多的答案
```

需要记录：

- 总 Input Tokens

- 总 Output Tokens

- 每个 Sample 的答案

- 投票结果

- 最终答案

- 是否正确

- 推理时间

特别注意：

**Self-Consistency 的 Token 消耗应该统计 5 次采样的总消耗，而不是只统计最后投票结果。**

---

# 七、答案评测

MATH-500 是数学题，因此不能简单地用字符串完全相等判断。

需要从模型输出中提取最终答案，然后与标准答案进行比较。

建议实现一个统一的：

```python
extract_answer()
```

例如模型可能输出：

```text
Therefore, the final answer is \boxed{42}.
```

需要提取：

```text
42
```

然后和 ground truth 比较。

需要尽量处理：

- `\boxed{...}`

- 最终答案

- 数字

- 分数

- 小数

- LaTeX 表达式

最好优先参考 MATH 数据集已有的答案评测逻辑，而不是自己简单写字符串比较。

---

# 八、Token 统计

实验要求明确统计：

> 输入与输出 Token 总量。

所以每次请求都要记录：

```text
input_tokens
output_tokens
total_tokens
```

其中：

```text
total_tokens = input_tokens + output_tokens
```

最终每种方法都需要统计：

```text
平均 Input Tokens
平均 Output Tokens
平均 Total Tokens
```

Self-Consistency 要统计所有 N 次采样的总 Token。

---

# 九、准确率统计

最基本：

```text
Accuracy = 正确题数 / 总题数
```

例如：

```text
Direct:
正确 120 / 200
Accuracy = 60%
```

同时建议按照 difficulty level 分别统计：

```text
Level 1 Accuracy
Level 2 Accuracy
Level 3 Accuracy
Level 4 Accuracy
Level 5 Accuracy
Overall Accuracy
```

最终希望得到类似：

| Method           | L1  | L2  | L3  | L4  | L5  | Overall |
| ---------------- | --- | --- | --- | --- | --- | ------- |
| Direct           |     |     |     |     |     |         |
| CoT              |     |     |     |     |     |         |
| Self-Consistency |     |     |     |     |     |         |

---

# 十、随机性与多次实验

任务文档指出：

> 小模型结果存在波动，建议多次运行并报告均值，并注明随机性设置。

因此建议：

**每种方法运行 3 次。**

例如：

```text
Direct × 3
CoT × 3
Self-Consistency × 3
```

记录每次实验的：

```text
seed
temperature
N
accuracy
token usage
```

最终报告：

```text
Mean Accuracy
Mean Token Usage
```

如果实验成本太高，可以先完成一次完整实验，再决定是否增加重复次数。

---

# 十一、当前服务器环境

当前实验服务器：

```text
4 × NVIDIA GeForce RTX 4090
```

目前使用：

```text
GPU 0
GPU 1
```

运行方式：

```bash
CUDA_VISIBLE_DEVICES=0,1
```

vLLM：

```text
vLLM
```

模型：

```text
/data/lsl/cache/huggingface/models/Qwen3.5-9B
```

当前项目目录：

```text
/data/lsl/projects/multi_agent_baseline
```

Python 环境：

```text
/data/lsl/projects/multi_agent_baseline/.venv
```

激活：

```bash
cd /data/lsl/projects/multi_agent_baseline
source .venv/bin/activate
```

当前 PyTorch：

```text
torch 2.5.1+cu121
```

当前环境已经确认：

```text
torch.cuda.is_available() == True
```

GPU 可以正常识别：

```text
NVIDIA GeForce RTX 4090
```

模型已经下载完成：

```text
/data/lsl/cache/huggingface/models/Qwen3.5-9B
```

---

# 十二、当前最重要的事情

不要重新折腾服务器环境。

当前环境已经基本配置完成。

接下来应该进入**实验代码开发阶段**。

推荐项目结构：

```text
multi_agent_baseline/
├── .venv/
├── data/
│   └── math500_200.json
│
├── src/
│   ├── model.py
│   ├── direct.py
│   ├── cot.py
│   ├── self_consistency.py
│   ├── evaluator.py
│   └── utils.py
│
├── scripts/
│   ├── run_direct.py
│   ├── run_cot.py
│   └── run_self_consistency.py
│
├── results/
│   ├── direct/
│   ├── cot/
│   └── self_consistency/
│
└── README.md
```

不要求严格按照这个结构，可以根据实际情况调整。

---

# 十三、代码设计要求

最好设计一个统一的模型调用接口。

例如：

```python
class ModelRunner:
    def generate(...):
        ...
```

然后三个 baseline 都调用同一个模型接口。

这样可以保证：

**三个方法除了 Prompt / Sampling Strategy 不同之外，其他条件尽可能一致。**

需要避免：

- 不同模型

- 不同 tokenizer

- 不同推理框架

- 不同数据

- 不同最大输出长度

否则实验不公平。

---

# 十四、实验结果最终需要输出

至少生成以下结果。

## 1. 每道题的详细结果

例如：

```text
question_id
level
method
prediction
ground_truth
correct
input_tokens
output_tokens
total_tokens
```

Self-Consistency 还应该记录：

```text
sample_1
sample_2
sample_3
sample_4
sample_5
vote_result
```

---

## 2. 方法整体结果

例如：

| Method           | Accuracy | Avg Input Tokens | Avg Output Tokens | Avg Total Tokens |
| ---------------- | -------- | ---------------- | ----------------- | ---------------- |
| Direct           |          |                  |                   |                  |
| CoT              |          |                  |                   |                  |
| Self-Consistency |          |                  |                   |                  |

---

## 3. 不同难度结果

| Method           | Level 1 | Level 2 | Level 3 | Level 4 | Level 5 |
| ---------------- | ------- | ------- | ------- | ------- | ------- |
| Direct           |         |         |         |         |         |
| CoT              |         |         |         |         |         |
| Self-Consistency |         |         |         |         |         |

---

# 十五、最终实验想回答的问题

这个实验最终不是单纯为了跑三个程序，而是想回答：

### 问题 1

CoT 相比直接回答：

> **是否能提高数学推理准确率？**

### 问题 2

Self-Consistency 相比单次 CoT：

> **是否进一步提高准确率？**

### 问题 3

准确率提高的代价是多少？

即：

> **Accuracy Gain vs Token Cost**

例如最终可能得到：

```text
Direct
Accuracy: 55%
Token: 500

CoT
Accuracy: 63%
Token: 1200

Self-Consistency (N=5)
Accuracy: 68%
Token: 6000
```

那么就可以分析：

> CoT 用约 2.4× Token 换来了准确率提升；Self-Consistency 进一步提升准确率，但 Token 成本大幅增加。

这就是整个实验最核心的分析。

---

# 十六、请 AI 协助完成的工作

请基于以上要求协助完成这个实验。

**第一阶段不要直接开始大规模运行。**

先完成：

1. 检查当前项目环境

2. 确认 Qwen3.5-9B 可以正常进行一次推理

3. 确认 MATH-500 数据格式

4. 确认 200 道题的抽样方式

5. 设计统一的数据格式

6. 设计三个 baseline 的代码结构

7. 设计答案提取与评测方法

8. 设计 Token 统计方式

9. 先用少量题目进行 sanity check

10. 确认结果正确后，再跑完整 200 题

**不要一开始就跑 200 × 3 × 5 次的大实验。**

先用 2～5 道题验证整个 pipeline：

```text
数据读取
→ 模型推理
→ 答案提取
→ 正确性判断
→ Token 统计
→ 保存结果
```

确认 pipeline 没问题之后，再正式运行完整实验。