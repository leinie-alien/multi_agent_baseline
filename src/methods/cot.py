from ..config import COT_SAMPLING
from ..model_runner import ModelRunner
from ..prompts import build_cot_messages
from .base import RunResult, run_single_method


def run_cot(runner: ModelRunner, questions: list[dict], *, max_tokens: int, seed: int) -> RunResult:
    return run_single_method(
        runner,
        questions,
        message_builder=build_cot_messages,
        temperature=COT_SAMPLING.temperature,
        max_tokens=max_tokens,
        n=COT_SAMPLING.n,
        seed=seed,
        method_name="cot",
    )
