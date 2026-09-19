"""结果落盘:per_question.jsonl / raw_outputs/ / summary.json,run_experiment.py 和 run_sanity_check.py 共用。"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path


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


def save_run_result(run_result, results_dir: Path) -> Path:
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
          f"truncation_rate={run_result.truncation_rate:.1%} "
          f"-> {out_dir}")

    return out_dir
