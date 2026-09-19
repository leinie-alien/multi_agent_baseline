import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.extract_answer import extract_answer, is_equiv, normalize_answer  # noqa: E402


def test_extract_boxed_simple():
    text = "Some reasoning...\nTherefore, the final answer is \\boxed{42}."
    assert extract_answer(text) == "42"


def test_extract_boxed_last_one_wins():
    text = "First I thought \\boxed{1}, but actually \\boxed{2}."
    assert extract_answer(text) == "2"


def test_extract_boxed_with_nested_braces():
    text = "The answer is \\boxed{\\frac{1}{2}}."
    assert extract_answer(text) == "\\frac{1}{2}"


def test_extract_fallback_final_answer_pattern():
    text = "After solving, the final answer is 7"
    assert extract_answer(text) == "7"


def test_extract_fallback_last_number():
    text = "The result comes out to 3.14 after simplification"
    assert extract_answer(text) == "3.14"


def test_extract_no_answer():
    assert extract_answer("") is None
    assert extract_answer("blah blah no numbers or boxed here") is None


def test_extract_allow_fallback_false_without_boxed_returns_none():
    # 复现截断且未写到 \boxed 的场景:兜底规则本可以从文本里抓到一个数字,
    # 但那只是截断点附近偶然出现的数字,不是模型的真实结论,应返回 None 而非伪造答案。
    text = "Case 1: product is 33, so a_{n-1} could be 1"
    assert extract_answer(text, allow_fallback=True) == "1"
    assert extract_answer(text, allow_fallback=False) is None


def test_extract_allow_fallback_false_still_returns_boxed():
    text = "Thus the final answer is \\boxed{14}."
    assert extract_answer(text, allow_fallback=False) == "14"


def test_extract_boxed_truncated_repetition_falls_back_to_earlier_complete_one():
    # 复现 q161 场景:模型正确给出 \boxed{14} 后陷入重复循环,
    # 最终在 max_tokens 处截断,恰好断在某次重复的 \boxed{ 中间(缺右花括号)。
    text = "Thus the answer is \\boxed{14}.\n\n\\boxed{14}\n\n\\boxed{1"
    assert extract_answer(text) == "14"


def test_is_equiv_exact_match():
    assert is_equiv("42", "42")


def test_is_equiv_fraction_forms():
    assert is_equiv("1/2", "\\frac{1}{2}")
    assert is_equiv("0.5", "\\frac{1}{2}")


def test_is_equiv_whitespace_and_dollar():
    assert is_equiv("$42$", "42")
    assert is_equiv(" 42 ", "42")


def test_is_equiv_percent():
    assert is_equiv("50\\%", "50")


def test_is_equiv_units_stripped():
    assert is_equiv("5\\text{ cm}", "5")


def test_is_equiv_mismatch():
    assert not is_equiv("42", "43")


def test_is_equiv_none_prediction():
    assert not is_equiv(None, "42")


def test_normalize_answer_sqrt():
    assert normalize_answer("\\sqrt3") == normalize_answer("\\sqrt{3}")


def test_is_equiv_numeric_fallback():
    assert is_equiv("2/4", "0.5")
