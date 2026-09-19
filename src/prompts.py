"""Direct / CoT 的 prompt 构造。Self-Consistency 复用 CoT 的 prompt。"""
from .config import FINAL_ANSWER_INSTRUCTION


def build_direct_messages(question: str) -> list[dict]:
    content = (
        "Solve the following math problem and give the final answer.\n\n"
        f"Problem:\n{question}"
        f"{FINAL_ANSWER_INSTRUCTION}"
    )
    return [{"role": "user", "content": content}]


def build_cot_messages(question: str) -> list[dict]:
    content = (
        "Solve the following math problem step by step.\n"
        "Explain your reasoning clearly and provide the final answer.\n\n"
        f"Problem:\n{question}"
        f"{FINAL_ANSWER_INSTRUCTION}"
    )
    return [{"role": "user", "content": content}]
