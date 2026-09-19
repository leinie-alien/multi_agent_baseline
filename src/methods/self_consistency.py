"""Self-Consistency: 复用 CoT 的 prompt,多次采样后多数投票。"""
from collections import Counter

from ..config import SELF_CONSISTENCY_SAMPLING
from ..extract_answer import normalize_answer
from ..model_runner import ModelRunner
from ..prompts import build_cot_messages
from .base import RunResult, run_single_method


def majority_vote(answers: list[str | None]) -> tuple[str | None, str]:
    """多数投票。返回 (最终答案, 投票结果描述)。

    投票前先用 normalize_answer 归一化,避免同一个答案的不同写法
    (如 "1/2" 与 "\\frac{1}{2}")被当成不同选项而分票、稀释票数。
    忽略 None(提取失败的采样);若全部为 None 则返回 (None, "no_valid_answer")。
    平票时取先出现的最高票答案,并在描述里标注 tie。
    """
    valid = [a for a in answers if a is not None]
    if not valid:
        return None, "no_valid_answer"

    # 用列表(而非以原始答案为 key 的 dict)保存归一化结果,避免重复的原始答案
    # (如两次都输出 "3")被当成 dict 的同一个 key 合并掉,导致计票丢失重复次数。
    normalized = [normalize_answer(a) for a in valid]
    counts = Counter(normalized)
    max_count = max(counts.values())
    # winner 取原始(未归一化)文本,只用归一化结果分组计票,这样存入结果的
    # prediction 是可读的原始答案(如 "1/2"),而不是 normalize_answer 内部的中间形式。
    winner_idx = next(i for i, n in enumerate(normalized) if counts[n] == max_count)
    winner = valid[winner_idx]

    tied = sum(1 for c in counts.values() if c == max_count) > 1
    vote_desc = f"{winner} ({max_count}/{len(valid)})" + (" [tie]" if tied else "")
    return winner, vote_desc


def run_self_consistency(runner: ModelRunner, questions: list[dict], *, max_tokens: int, seed: int) -> RunResult:
    return run_single_method(
        runner,
        questions,
        message_builder=build_cot_messages,
        temperature=SELF_CONSISTENCY_SAMPLING.temperature,
        max_tokens=max_tokens,
        n=SELF_CONSISTENCY_SAMPLING.n,
        seed=seed,
        method_name="self_consistency",
        vote_fn=majority_vote,
    )
