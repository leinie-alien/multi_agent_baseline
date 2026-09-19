"""实验统一入口。

用法示例:
    python scripts/run_experiment.py --method direct --limit 5          # sanity check
    python scripts/run_experiment.py --method cot --repeat 3
    python scripts/run_experiment.py --method self_consistency --repeat 3

需要 vLLM + GPU 环境,留给服务器环境阶段执行。
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (  # noqa: E402
    DATA_PATH,
    DEFAULT_SEEDS,
    ModelConfig,
    RESULTS_DIR,
)
from src.data_loader import load_questions  # noqa: E402
from src.methods.cot import run_cot  # noqa: E402
from src.methods.direct import run_direct  # noqa: E402
from src.methods.self_consistency import run_self_consistency  # noqa: E402
from src.model_runner import ModelRunner  # noqa: E402

METHOD_RUNNERS = {
    "direct": run_direct,
    "cot": run_cot,
    "self_consistency": run_self_consistency,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行单个方法的实验")
    parser.add_argument("--method", required=True, choices=list(METHOD_RUNNERS.keys()))
    parser.add_argument("--repeat", type=int, default=3, help="重复运行次数,默认 3")
    parser.add_argument("--seeds", type=int, nargs="*", default=None, help="覆盖默认 seed 列表")
    parser.add_argument("--limit", type=int, default=None, help="只跑前 N 题,用于 sanity check")
    parser.add_argument("--data-path", default=str(PROJECT_ROOT / DATA_PATH))
    parser.add_argument("--results-dir", default=str(PROJECT_ROOT / RESULTS_DIR))
    return parser.parse_args()


def save_raw_outputs(run_result, out_dir: Path) -> None:
    """把每题的模型原始输出另存成纯文本文件,方便直接打开肉眼检查
    (比如判断答案是否被 max_tokens 截断),不用去 per_question.jsonl 里翻转义字符串。
    """
    raw_dir = out_dir / "raw_outputs"
    raw_dir.mkdir(parents=True, exist_ok=True)
    for record in run_result.records:
        lines = []
        for i, text in enumerate(record.raw_outputs):
            if len(record.raw_outputs) > 1:
                lines.append(f"===== sample {i} =====")
            lines.append(text)
        (raw_dir / f"{record.question_id}.txt").write_text(
            "\n".join(lines), encoding="utf-8"
        )


def save_run_result(run_result, results_dir: Path) -> None:
    out_dir = results_dir / run_result.method / f"run_seed{run_result.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "per_question.jsonl", "w", encoding="utf-8") as f:
        for record in run_result.records:
            f.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")

    save_raw_outputs(run_result, out_dir)

    summary = {
        "method": run_result.method,
        "seed": run_result.seed,
        "num_questions": len(run_result.records),
        "accuracy": run_result.accuracy,
        "avg_input_tokens": run_result.avg_input_tokens,
        "avg_output_tokens": run_result.avg_output_tokens,
        "avg_total_tokens": run_result.avg_total_tokens,
        "truncation_rate": run_result.truncation_rate,
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[{run_result.method} seed={run_result.seed}] "
          f"accuracy={run_result.accuracy:.3f} "
          f"avg_total_tokens={run_result.avg_total_tokens:.1f} "
          f"-> {out_dir}")


def main() -> None:
    args = parse_args()

    questions = load_questions(args.data_path)
    if args.limit is not None:
        questions = questions[: args.limit]

    seeds = args.seeds if args.seeds else DEFAULT_SEEDS[: args.repeat]

    model_config = ModelConfig()
    runner = ModelRunner(
        model_path=model_config.model_path,
        tensor_parallel_size=model_config.tensor_parallel_size,
        gpu_memory_utilization=model_config.gpu_memory_utilization,
        max_model_len=model_config.max_model_len,
        max_num_seqs=model_config.max_num_seqs,
    )

    run_fn = METHOD_RUNNERS[args.method]
    results_dir = Path(args.results_dir)

    try:
        for seed in seeds:
            run_result = run_fn(runner, questions, max_tokens=model_config.max_tokens, seed=seed)
            save_run_result(run_result, results_dir)
    finally:
        # 确保 vLLM 子进程被清理,避免孤儿进程占用显存(参见 LEARNING_NOTES.md)
        runner.shutdown()


if __name__ == "__main__":
    main()
