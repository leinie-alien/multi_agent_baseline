# Multi-Agent Baseline: Direct vs CoT vs Self-Consistency on MATH-500

对比 Direct Answer / Chain-of-Thought / Self-Consistency 三种推理方法在 MATH-500(200题,5个难度各40题)上的准确率与 Token 成本。详细设计见 `PLAN.md`。

## 三种方法是什么

- **Direct(直接作答)**:把题目丢给模型,让它直接说答案,不要求解题过程。类似问别人"3的平方是多少",对方脱口而出"9",没有过程。最省 token、跑得最快,但复杂题容易因为没有"逐步思考"的机会而直接算错,通常准确率最低。

- **CoT(思维链,Chain-of-Thought)**:要求模型把解题步骤写出来,一步一步推,最后再给答案。相当于给模型的推理搭了脚手架,减少跳步出错。通常比 Direct 准确率高,但输出变长,token 消耗明显增加。

- **Self-Consistency(自我一致性 / 多数投票)**:让模型对同一道题独立"想"很多遍(本项目 N=5,每次带一点随机性,所以 5 次的解法/答案可能不完全一样),再看这 5 次里哪个答案出现次数最多,采信那个。类似 5 个人分别独立做同一道题,少数服从多数——如果模型走对了推理路径,大概率稳定得到同一个答案,走偏了的错误路径反而五花八门、彼此不一致,所以"最常出现的答案"更可能是对的。通常三者中准确率最高,但要跑 5 次完整 CoT,token 成本约是单次 CoT 的 5 倍。

三者是层层递进的实验设计:Direct 是最省钱的基线,CoT 用更多 token 换更高准确率,Self-Consistency 再用更多 token 换更进一步的准确率提升。实验最终要回答的问题就是:每一步"加钱"到底值不值——多花的 token,换来的准确率提升有多少。

## 目录结构

```
data/            题目 ID 列表 + 准备好的数据集
src/             核心代码(模型接口、prompt、答案提取、三种方法)
scripts/         数据准备 / 运行实验 / 汇总结果 的入口脚本
results/         每次运行的逐题结果与汇总
tests/           不依赖 GPU 的单元测试(答案提取与等价判断)
```

## 使用步骤

1. 准备数据(需要联网访问 HuggingFace):
   ```bash
   python scripts/prepare_data.py
   ```
2. Sanity check(先用少量题目验证 pipeline,需要 vLLM + GPU 环境):
   ```bash
   python scripts/run_experiment.py --method direct --limit 5 --repeat 1
   python scripts/run_experiment.py --method cot --limit 5 --repeat 1
   python scripts/run_experiment.py --method self_consistency --limit 5 --repeat 1
   ```
3. 确认无误后跑完整实验(每种方法默认重复 3 次,seed 见 `src/config.py`):
   ```bash
   python scripts/run_experiment.py --method direct
   python scripts/run_experiment.py --method cot
   python scripts/run_experiment.py --method self_consistency
   ```
4. 汇总结果:
   ```bash
   python scripts/aggregate_results.py
   ```
   生成 `results/summary_overall.md`(方法整体对比)和 `results/summary_by_level.md`(按难度分层)。

## 关键设计

- 三种方法共用同一个 `ModelRunner`(`src/model_runner.py`),同一 tokenizer、同一 `max_tokens`,只有 prompt 和采样参数不同,保证公平对比。
- Direct / CoT 使用 temperature=0.6,Self-Consistency 使用 temperature=0.7、N=5,这样"重复3次取均值"对三种方法都有统计意义。
- 答案提取与等价判断(`src/extract_answer.py`)参考 Hendrycks MATH 官方评测逻辑,处理 `\boxed{}`、分数、小数、百分号、单位等情况。
- Self-Consistency 的 token 统计按 N 次独立调用的口径计入输入 token(`input_tokens * n`),不会低估成本。

## 测试

```bash
python -m pytest tests/
```

`src/model_runner.py` 依赖 vLLM,只能在服务器 GPU 环境下运行,本机测试只覆盖答案提取逻辑。
