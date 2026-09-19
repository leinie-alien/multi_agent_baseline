import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.methods.base import run_single_method  # noqa: E402
from src.model_runner import PromptResult, SampleResult  # noqa: E402


class FakeRunner:
    """伪造 ModelRunner:只实现 run_single_method 依赖的两个方法,不需要真实 vLLM。"""

    def __init__(self, samples: list[SampleResult]) -> None:
        self._samples = samples

    def build_prompt(self, messages, enable_thinking=None):
        return "fake prompt"

    def generate(self, prompts, temperature, max_tokens, n=1, seed=None, repetition_penalty=1.0):
        return [
            PromptResult(prompt=p, input_tokens=10, samples=self._samples)
            for p in prompts
        ]


def _question():
    return {"question_id": "q1", "level": 5, "problem": "1+1=?", "answer": "2"}


def test_truncated_sample_without_boxed_yields_no_prediction():
    # 复现 q161 场景:被 max_tokens 截断且没写到 \boxed,兜底规则本可以从文本里
    # 抓到"1",但那只是截断点附近偶然出现的数字,不是模型的真实结论。
    # run_single_method 必须按 finish_reason == "length" 关掉 allow_fallback,
    # 让 prediction 落到 None,而不是伪造出这个"1"。
    samples = [
        SampleResult(
            text="Case 1: product is 33, so a_{n-1} could be 1",
            output_tokens=100,
            finish_reason="length",
        )
    ]
    result = run_single_method(
        FakeRunner(samples),
        [_question()],
        message_builder=lambda p: [{"role": "user", "content": p}],
        temperature=0.6,
        max_tokens=100,
        n=1,
        seed=42,
        method_name="cot",
    )
    assert result.records[0].prediction is None


def test_normal_finish_without_boxed_still_uses_fallback():
    # 正常结束(finish_reason == "stop")没写 \boxed 时,兜底规则应该照常生效,
    # 不应该被截断相关的改动误伤。
    samples = [
        SampleResult(
            text="Case 1: product is 33, so a_{n-1} could be 1",
            output_tokens=50,
            finish_reason="stop",
        )
    ]
    result = run_single_method(
        FakeRunner(samples),
        [_question()],
        message_builder=lambda p: [{"role": "user", "content": p}],
        temperature=0.6,
        max_tokens=100,
        n=1,
        seed=42,
        method_name="cot",
    )
    assert result.records[0].prediction == "1"
