"""加载统一格式的题目数据 (data/math500_200.json)。"""
from __future__ import annotations

import json
from pathlib import Path


def load_questions(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        questions: list[dict] = json.load(f)
    return questions


def group_by_level(questions: list[dict]) -> dict[int, list[dict]]:
    groups: dict[int, list[dict]] = {}
    for q in questions:
        groups.setdefault(q["level"], []).append(q)
    return groups
