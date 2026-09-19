"""从 HuggingFaceH4/math-500 数据集中筛出 data/selected_ids.txt 指定的 200 道题,
输出统一格式 data/math500_200.json。

需要联网访问 HuggingFace,留给服务器/联网环境阶段执行。
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SELECTED_IDS_PATH = PROJECT_ROOT / "data" / "selected_ids.txt"
OUTPUT_PATH = PROJECT_ROOT / "data" / "math500_200.json"


def load_selected_ids(path: Path) -> list[str]:
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def main() -> None:
    from datasets import load_dataset  # 延迟导入,避免本机无网络/无依赖时 import 失败

    selected_ids = load_selected_ids(SELECTED_IDS_PATH)
    selected_set = set(selected_ids)
    if len(selected_ids) != len(selected_set):
        dup = [k for k, v in Counter(selected_ids).items() if v > 1]
        raise ValueError(f"selected_ids.txt 中存在重复 ID: {dup}")

    dataset = load_dataset("HuggingFaceH4/math-500", split="test")

    by_unique_id = {row["unique_id"]: row for row in dataset}

    missing = [uid for uid in selected_ids if uid not in by_unique_id]
    if missing:
        raise ValueError(f"以下 unique_id 在数据集中未找到: {missing}")

    questions = []
    for i, uid in enumerate(selected_ids, start=1):
        row = by_unique_id[uid]
        questions.append(
            {
                "question_id": f"q{i:03d}",
                "unique_id": uid,
                "subject": row["subject"],
                "level": int(row["level"]),
                "problem": row["problem"],
                "answer": row["answer"],
            }
        )

    if len(questions) != 200:
        raise ValueError(f"最终题目数量不是 200,而是 {len(questions)}")

    level_counts = Counter(q["level"] for q in questions)
    for level in range(1, 6):
        count = level_counts.get(level, 0)
        if count != 40:
            print(f"[WARNING] Level {level} 实际数量为 {count},预期 40")
        else:
            print(f"Level {level}: {count} 题")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(questions, f, ensure_ascii=False, indent=2)

    print(f"已写入 {OUTPUT_PATH},共 {len(questions)} 题")


if __name__ == "__main__":
    main()
