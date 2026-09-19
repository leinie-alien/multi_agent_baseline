"""汇总 results/ 下所有方法的所有 run,产出:
1. 方法整体表: Accuracy / Avg Input/Output/Total Tokens (多次 run 的 mean ± std)
2. 分难度表: 每方法每 level 的 accuracy (mean ± std)

输出到 results/summary_overall.md 和 results/summary_by_level.md。
"""
from __future__ import annotations

import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import RESULTS_DIR  # noqa: E402


def mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    mean = statistics.mean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return mean, std


def fmt(mean: float, std: float, pct: bool = False) -> str:
    if pct:
        return f"{mean * 100:.1f}% ± {std * 100:.1f}"
    return f"{mean:.1f} ± {std:.1f}"


def load_runs(results_dir: Path) -> dict[str, list[dict]]:
    """返回 {method: [summary_dict, ...]}"""
    runs_by_method: dict[str, list[dict]] = defaultdict(list)
    for method_dir in sorted(results_dir.iterdir()):
        if not method_dir.is_dir():
            continue
        method = method_dir.name
        for run_dir in sorted(method_dir.glob("run_seed*")):
            summary_path = run_dir / "summary.json"
            per_question_path = run_dir / "per_question.jsonl"
            if not summary_path.exists() or not per_question_path.exists():
                continue
            with open(summary_path, encoding="utf-8") as f:
                summary = json.load(f)
            records = []
            with open(per_question_path, encoding="utf-8") as f:
                for line in f:
                    records.append(json.loads(line))
            summary["records"] = records
            runs_by_method[method].append(summary)
    return runs_by_method


def build_overall_table(runs_by_method: dict[str, list[dict]]) -> str:
    lines = [
        "| Method | Accuracy | Avg Input Tokens | Avg Output Tokens | Avg Total Tokens | Runs |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for method, runs in runs_by_method.items():
        acc_mean, acc_std = mean_std([r["accuracy"] for r in runs])
        in_mean, in_std = mean_std([r["avg_input_tokens"] for r in runs])
        out_mean, out_std = mean_std([r["avg_output_tokens"] for r in runs])
        tot_mean, tot_std = mean_std([r["avg_total_tokens"] for r in runs])
        lines.append(
            f"| {method} | {fmt(acc_mean, acc_std, pct=True)} | "
            f"{fmt(in_mean, in_std)} | {fmt(out_mean, out_std)} | "
            f"{fmt(tot_mean, tot_std)} | {len(runs)} |"
        )
    return "\n".join(lines)


def build_by_level_table(runs_by_method: dict[str, list[dict]]) -> str:
    levels = [1, 2, 3, 4, 5]
    header = "| Method | " + " | ".join(f"Level {lv}" for lv in levels) + " |"
    sep = "| --- | " + " | ".join("---" for _ in levels) + " |"
    lines = [header, sep]

    for method, runs in runs_by_method.items():
        cells = []
        for lv in levels:
            per_run_acc = []
            for run in runs:
                lv_records = [r for r in run["records"] if r["level"] == lv]
                if lv_records:
                    acc = sum(r["correct"] for r in lv_records) / len(lv_records)
                    per_run_acc.append(acc)
            mean, std = mean_std(per_run_acc)
            cells.append(fmt(mean, std, pct=True) if per_run_acc else "N/A")
        lines.append(f"| {method} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    results_dir = PROJECT_ROOT / RESULTS_DIR
    runs_by_method = load_runs(results_dir)

    if not runs_by_method:
        print(f"未在 {results_dir} 下找到任何结果,请先运行 scripts/run_experiment.py")
        return

    overall_table = build_overall_table(runs_by_method)
    by_level_table = build_by_level_table(runs_by_method)

    (results_dir / "summary_overall.md").write_text(overall_table, encoding="utf-8")
    (results_dir / "summary_by_level.md").write_text(by_level_table, encoding="utf-8")

    print("=== 方法整体结果 ===")
    print(overall_table)
    print("\n=== 分难度结果 ===")
    print(by_level_table)


if __name__ == "__main__":
    main()
