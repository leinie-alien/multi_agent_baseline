"""三种方法共用的运行流程:构造 prompt -> 调用模型 -> 提取答案(单个或投票) -> 判分 -> 记录 token。"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from ..extract_answer import extract_answer, is_equiv
from ..model_runner import ModelRunner


@dataclass
class QuestionRecord:
    question_id: str
    level: int
    method: str
    prediction: str | None
    ground_truth: str
    correct: bool
    input_tokens: int
    output_tokens: int
    total_tokens: int
    raw_outputs: list[str] = field(default_factory=list)  # 模型原始输出全文,便于调试提取失败的情况
    finish_reasons: list[str] = field(default_factory=list)  # 每个采样对应的 vLLM finish_reason,"length" 表示被 max_tokens 截断
    samples: list[str] | None = None  # 仅 Self-Consistency 填充:每个采样的提取答案
    vote_result: str | None = None  # 仅 Self-Consistency 填充
    rounds_used: int = 1  # 仅多智能体方法填充:该题实际跑了几轮(受益于早停/未触发满量反馈轮次)
    agent_trace: list[dict] = field(default_factory=list)  # 仅多智能体方法填充:每轮每个角色的输出记录,元素形如
    # {"round": int, "role": str, "text": str, "output_tokens": int, "finish_reason": str}


@dataclass
class RunResult:
    method: str
    seed: int
    records: list[QuestionRecord] = field(default_factory=list)

    @property
    def accuracy(self) -> float:
        if not self.records:
            return 0.0
        return sum(r.correct for r in self.records) / len(self.records)

    @property
    def avg_input_tokens(self) -> float:
        return sum(r.input_tokens for r in self.records) / len(self.records)

    @property
    def avg_output_tokens(self) -> float:
        return sum(r.output_tokens for r in self.records) / len(self.records)

    @property
    def avg_total_tokens(self) -> float:
        return sum(r.total_tokens for r in self.records) / len(self.records)

    @property
    def truncation_rate(self) -> float:
        """被 max_tokens 截断的采样占全部采样的比例(finish_reason == "length")。"""
        total_samples = sum(len(r.finish_reasons) for r in self.records)
        if total_samples == 0:
            return 0.0
        truncated = sum(fr == "length" for r in self.records for fr in r.finish_reasons)
        return truncated / total_samples


MessageBuilder = Callable[[str], list[dict]]
VoteFn = Callable[[list[str | None]], tuple[str | None, str]]


def run_single_method(
    runner: ModelRunner,
    questions: list[dict],
    *,
    message_builder: MessageBuilder,
    temperature: float,
    max_tokens: int,
    n: int,
    seed: int,
    method_name: str,
    vote_fn: VoteFn | None = None,
    enable_thinking: bool | None = None,
    repetition_penalty: float = 1.0,
) -> RunResult:
    """跑一批题目并返回结果。n=1 时直接取唯一采样的答案;n>1 时必须提供 vote_fn 做投票。

    enable_thinking 透传给 ModelRunner.build_prompt:Direct 传 False 关闭思考,
    CoT/Self-Consistency 保持默认 None(保留思考,由模型自己决定/走 <think> 分支)。
    """
    if n > 1 and vote_fn is None:
        raise ValueError("n > 1 (Self-Consistency) 时必须提供 vote_fn")

    prompts = [
        runner.build_prompt(message_builder(q["problem"]), enable_thinking=enable_thinking)
        for q in questions
    ]
    outputs = runner.generate(
        prompts,
        temperature=temperature,
        max_tokens=max_tokens,
        n=n,
        seed=seed,
        repetition_penalty=repetition_penalty,
    )

    records: list[QuestionRecord] = []
    for q, out in zip(questions, outputs):
        # 被截断(finish_reason == "length")的采样禁用兜底提取规则:截断点是硬断开的,
        # 兜底规则抓到的"最后一个数字"只是断点附近偶然出现的数字,不是模型的真实结论,
        # 当成一次有效预测会把截断噪音伪装成答案(见 extract_answer 的 allow_fallback 说明)。
        sample_answers = [
            extract_answer(s.text, allow_fallback=s.finish_reason != "length")
            for s in out.samples
        ]
        output_tokens = sum(s.output_tokens for s in out.samples)
        # Token 统计口径:多次采样共用同一份输入,但按“公平对比等价于 N 次独立调用”的口径,
        # 输入 token 按 input_tokens * n 计入。
        input_tokens = out.input_tokens * n

        vote_result: str | None = None
        if n == 1:
            prediction = sample_answers[0]
        else:
            prediction, vote_result = vote_fn(sample_answers)

        correct = is_equiv(prediction, q["answer"]) if prediction is not None else False

        records.append(
            QuestionRecord(
                question_id=q["question_id"],
                level=q["level"],
                method=method_name,
                prediction=prediction,
                ground_truth=q["answer"],
                correct=correct,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=input_tokens + output_tokens,
                raw_outputs=[s.text for s in out.samples],
                finish_reasons=[s.finish_reason for s in out.samples],
                samples=sample_answers if n > 1 else None,
                vote_result=vote_result,
            )
        )

    return RunResult(method=method_name, seed=seed, records=records)
