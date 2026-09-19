"""Sanity Check 脚本:每个难度抽 1 题(共 5 题),依次跑三种方法,验证完整 pipeline:

    数据读取 -> 模型推理 -> 答案提取 -> 正确性判断 -> Token 统计 -> 保存结果

用法:
    python scripts/run_sanity_check.py

需要 vLLM + GPU 环境,留给服务器环境阶段执行。确认无误后再用
scripts/run_experiment.py 跑完整 200 题实验。
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DATA_PATH, ModelConfig  # noqa: E402
from src.data_loader import group_by_level, load_questions  # noqa: E402
from src.methods.cot import run_cot  # noqa: E402
from src.methods.direct import run_direct  # noqa: E402
from src.methods.self_consistency import run_self_consistency  # noqa: E402
from src.model_runner import ModelRunner  # noqa: E402


def pick_one_per_level(questions: list[dict]) -> list[dict]:
    """每个难度取第一题,凑成 5 题的分层小样本。"""
    groups = group_by_level(questions)
    picked = []
    for level in sorted(groups.keys()):
        picked.append(groups[level][0])
    return picked


def print_record(record) -> None:
    status = "✓" if record.correct else "✗"
    print(f"  [{status}] {record.question_id} (Level {record.level})")
    print(f"      Prediction:    {record.prediction!r}")
    print(f"      Ground Truth:  {record.ground_truth!r}")
    print(f"      Tokens: in={record.input_tokens} out={record.output_tokens} total={record.total_tokens}")
    if record.samples is not None:
        print(f"      Samples: {record.samples}")
        print(f"      Vote:    {record.vote_result}")


def print_summary(method: str, run_result) -> None:
    print(f"\n  ---- {method} summary ----")
    print(f"  Accuracy: {run_result.accuracy:.2%} "
          f"({sum(r.correct for r in run_result.records)}/{len(run_result.records)})")
    print(f"  Avg Input Tokens:  {run_result.avg_input_tokens:.1f}")
    print(f"  Avg Output Tokens: {run_result.avg_output_tokens:.1f}")
    print(f"  Avg Total Tokens:  {run_result.avg_total_tokens:.1f}")


def main() -> None:
    print("=" * 60)
    print("  MULTI-AGENT BASELINE - SANITY CHECK (5 questions, 1 per level)")
    print("=" * 60)

    questions = load_questions(str(PROJECT_ROOT / DATA_PATH))
    sample = pick_one_per_level(questions)
    print(f"Sampled {len(sample)} questions:")
    for q in sample:
        print(f"  {q['question_id']} Level {q['level']} ({q['subject']}): {q['problem'][:60]}...")

    model_config = ModelConfig()
    runner = ModelRunner(
        model_path=model_config.model_path,
        tensor_parallel_size=model_config.tensor_parallel_size,
        gpu_memory_utilization=model_config.gpu_memory_utilization,
        max_model_len=model_config.max_model_len,
        max_num_seqs=model_config.max_num_seqs,
    )

    seed = 42
    try:
        print("\n[1/3] Running Direct...")
        direct_result = run_direct(runner, sample, max_tokens=model_config.max_tokens, seed=seed)
        for r in direct_result.records:
            print_record(r)
        print_summary("direct", direct_result)

        print("\n[2/3] Running CoT...")
        cot_result = run_cot(runner, sample, max_tokens=model_config.max_tokens, seed=seed)
        for r in cot_result.records:
            print_record(r)
        print_summary("cot", cot_result)

        print("\n[3/3] Running Self-Consistency...")
        sc_result = run_self_consistency(runner, sample, max_tokens=model_config.max_tokens, seed=seed)
        for r in sc_result.records:
            print_record(r)
        print_summary("self_consistency", sc_result)

        print("\n" + "=" * 60)
        print(f"  {'Method':<20} {'Accuracy':>10} {'Avg Input':>10} {'Avg Output':>11} {'Avg Total':>10}")
        print(f"  {'-' * 61}")
        for method, result in [
            ("Direct", direct_result),
            ("CoT", cot_result),
            ("Self-Consistency", sc_result),
        ]:
            print(
                f"  {method:<20} {result.accuracy:>9.2%} "
                f"{result.avg_input_tokens:>10.1f} {result.avg_output_tokens:>11.1f} "
                f"{result.avg_total_tokens:>10.1f}"
            )
        print("=" * 60)
    finally:
        runner.shutdown()

    print("\nSanity check complete. Pipeline 无误后可运行 scripts/run_experiment.py 跑完整实验。")


if __name__ == "__main__":
    main()
