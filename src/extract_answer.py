"""答案提取与等价判断。

核心等价判断逻辑参考 Hendrycks et al. MATH 数据集官方评测脚本
(https://github.com/hendrycks/math, math_equivalence.py) 的做法:
先做 LaTeX/格式归一化,再做字符串比较,而不是简单的原始字符串相等。
"""
from __future__ import annotations

import re


def last_boxed_only_string(text: str) -> str | None:
    """提取文本中最后一个花括号配对完整的 \\boxed{...} 或 \\fbox{...} 整体子串。

    从后往前找 \\boxed/\\fbox 出现位置;若最靠后的一个因文本被截断而花括号未配对完整
    (例如生成在 max_tokens 处截断,恰好断在某次重复的 \\boxed{...} 中间),则跳过它,
    继续往前找上一个配对完整的,而不是直接判定整段提取失败。
    """
    search_end = len(text)
    while True:
        idx = text.rfind("\\boxed", 0, search_end)
        fbox_idx = text.rfind("\\fbox", 0, search_end)
        idx = max(idx, fbox_idx)
        if idx < 0:
            return None

        i = idx
        right_brace_idx = None
        num_left_braces_open = 0
        while i < len(text):
            if text[i] == "{":
                num_left_braces_open += 1
            if text[i] == "}":
                num_left_braces_open -= 1
                if num_left_braces_open == 0:
                    right_brace_idx = i
                    break
            i += 1

        if right_brace_idx is not None:
            return text[idx:right_brace_idx + 1]

        # 括号未配对完整(通常是截断导致),从这个出现位置之前继续找上一个
        search_end = idx


def remove_boxed(boxed_str: str) -> str:
    """从 \\boxed{xxx} 或 \\fbox{xxx} 中取出 xxx。"""
    for prefix in ("\\boxed{", "\\fbox{"):
        if boxed_str.startswith(prefix) and boxed_str.endswith("}"):
            return boxed_str[len(prefix):-1]
    # 兜底:\boxed xxx (无花括号,极少见)
    left = "\\boxed "
    if boxed_str.startswith(left):
        return boxed_str[len(left):]
    return boxed_str


_FINAL_ANSWER_PATTERNS = [
    re.compile(r"final answer is[:\s]*\$?(.+?)\$?\s*\.?\s*$", re.IGNORECASE),
    re.compile(r"answer\s*[:=]\s*\$?(.+?)\$?\s*\.?\s*$", re.IGNORECASE),
    re.compile(r"答案[是为:：]\s*(.+?)\s*[。\.]?\s*$"),
]

_LAST_NUMBER_PATTERN = re.compile(r"-?\d+(?:\.\d+)?(?:/\d+)?")


def extract_answer(model_output: str, allow_fallback: bool = True) -> str | None:
    """从模型输出中提取最终答案字符串。

    优先级: \\boxed{} > "final answer is / answer:" 等模式(取最后一行匹配) > 文本中最后一个数字/分数。

    allow_fallback=False 时只认 \\boxed{}/\\fbox{},不使用后两级兜底规则。用于输出被
    max_tokens 截断且找不到 \\boxed 的情况:此时文本是在任意位置硬断开的,兜底规则抓到的
    "最后一个数字"只是断点附近偶然出现的数字,不是模型的真实结论,当成答案会把截断噪音
    伪装成一次预测(误判对/误判错都可能发生),详见 LEARNING_NOTES.md q161 相关记录。
    """
    if not model_output:
        return None

    boxed = last_boxed_only_string(model_output)
    if boxed is not None:
        return remove_boxed(boxed).strip()

    if not allow_fallback:
        return None

    for line in reversed(model_output.strip().splitlines()):
        line = line.strip()
        if not line:
            continue
        for pattern in _FINAL_ANSWER_PATTERNS:
            m = pattern.search(line)
            if m:
                return m.group(1).strip().rstrip("$").strip()

    numbers = _LAST_NUMBER_PATTERN.findall(model_output)
    if numbers:
        return numbers[-1]

    return None


def _fix_fracs(string: str) -> str:
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if len(substr) > 0 and substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except AssertionError:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    return new_str


def _fix_a_slash_b(string: str) -> str:
    if len(string.split("/")) != 2:
        return string
    a_str, b_str = string.split("/")
    try:
        a = int(a_str)
        b = int(b_str)
        if string == f"{a}/{b}":
            return f"\\frac{{{a}}}{{{b}}}"
        return string
    except ValueError:
        return string


def _remove_right_units(string: str) -> str:
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        return splits[0]
    return string


def _fix_sqrt(string: str) -> str:
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if len(split) > 0 and split[0] != "{":
            a = split[0]
            new_substr = "\\sqrt{" + a + "}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr
    return new_string


def normalize_answer(string: str | None) -> str:
    """尽量还原 MATH 官方 is_equiv 的归一化步骤。"""
    if string is None:
        return ""

    string = string.strip()
    string = string.replace("\n", "")
    string = string.rstrip(".")
    string = string.replace("\\!", "")
    string = string.replace("\\\\", "\\")
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")
    string = string.replace("\\$", "")
    string = string.replace("$", "")
    string = _remove_right_units(string)
    string = string.replace("\\%", "")
    string = string.replace("%", "")
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    if len(string) > 0 and string[0] == ".":
        string = "0" + string

    if len(string.split("=")) == 2:
        parts = string.split("=")
        if len(parts[0]) <= 2:
            string = parts[1]

    string = _fix_sqrt(string)
    string = string.replace(" ", "")
    string = _fix_fracs(string)
    if string == "0.5":
        string = "\\frac{1}{2}"
    string = _fix_a_slash_b(string)

    string = string.strip("{}")
    string = string.strip("\\")
    string = string.strip()

    return string


def _try_parse_number(string: str) -> float | None:
    try:
        return float(string)
    except ValueError:
        m = re.fullmatch(r"\\frac\{(-?\d+(?:\.\d+)?)\}\{(-?\d+(?:\.\d+)?)\}", string)
        if m:
            num, denom = float(m.group(1)), float(m.group(2))
            if denom != 0:
                return num / denom
        m = re.fullmatch(r"(-?\d+(?:\.\d+)?)/(-?\d+(?:\.\d+)?)", string)
        if m:
            num, denom = float(m.group(1)), float(m.group(2))
            if denom != 0:
                return num / denom
    return None


def is_equiv(pred: str | None, ground_truth: str | None) -> bool:
    """判断预测答案与标准答案是否等价。先做字符串归一化比较,再尝试数值比较兜底。"""
    if pred is None or ground_truth is None:
        return False

    try:
        norm_pred = normalize_answer(pred)
        norm_gt = normalize_answer(ground_truth)
        if norm_pred == norm_gt:
            return True
    except Exception:
        pass

    val_pred = _try_parse_number(pred.strip())
    val_gt = _try_parse_number(ground_truth.strip())
    if val_pred is not None and val_gt is not None:
        return abs(val_pred - val_gt) < 1e-4

    return False
