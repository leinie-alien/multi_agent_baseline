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
