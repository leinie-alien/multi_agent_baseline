from ..config import DIRECT_SAMPLING
from ..model_runner import ModelRunner
from ..prompts import build_direct_messages
from .base import RunResult, run_single_method


def run_direct(
    runner: ModelRunner,
    questions: list[dict],
    *,
    max_tokens: int,
    seed: int,
    enable_thinking: bool | None = None,
) -> RunResult:
    """enable_thinking 参数仅为和 run_cot/run_self_consistency 保持统一签名(方便 CLI
    统一调用),Direct 方法始终强制关闭思考,忽略传入值。"""
    del enable_thinking
    return run_single_method(
        runner,
        questions,
        message_builder=build_direct_messages,
        temperature=DIRECT_SAMPLING.temperature,
        max_tokens=max_tokens,
        n=DIRECT_SAMPLING.n,
        seed=seed,
        method_name="direct",
        enable_thinking=False,
        repetition_penalty=DIRECT_SAMPLING.repetition_penalty,
    )
