from ..config import DIRECT_SAMPLING
from ..model_runner import ModelRunner
from ..prompts import build_direct_messages
from .base import RunResult, run_single_method


def run_direct(runner: ModelRunner, questions: list[dict], *, max_tokens: int, seed: int) -> RunResult:
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
    )
