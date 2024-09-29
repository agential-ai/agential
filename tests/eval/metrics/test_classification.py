"""Unit tests for classification evaluation metrics."""

from agential.core.llm import MockLLM
from agential.eval.metrics.classification import (
    EM,
    f1,
    fuzzy_EM,
    llm_as_judge_eval,
    normalize_answer,
    parse_first_number,
    precision,
    recall,
    remove_articles,
    remove_punc,
    white_space_fix,
)


def test_remove_articles() -> None:
    """Test remove_articles function."""
    sample_text = (
        "A fox jumped over the fence. An apple was on the table. The quick brown fox."
    )

    result = remove_articles(sample_text)
    expected = (
        "A fox jumped over   fence. An apple was on   table. The quick brown fox."
    )
    assert result == expected, f"Test failed: Expected '{expected}', got '{result}'"


def test_white_space_fix() -> None:
    """Test white_space_fix function."""
    sample_text = "over   fence"
    result = white_space_fix(sample_text)
    assert result == "over fence"


def test_remove_punc() -> None:
    """Test remove_punc function."""
    sample_text = "abcd.,"
    result = remove_punc(sample_text)
    assert result == "abcd"


def test_normalize_answer() -> None:
    """Test normalize_answer function."""
    sample_text = (
        "A fox jumped over the fence. An apple was on the table. The quick brown fox."
    )

    result = normalize_answer(sample_text)
    expected = "fox jumped over fence apple was on table quick brown fox"
    assert result == expected


def test_parse_first_number() -> None:
    """Test parse_first_number."""
    number = parse_first_number("145 monkeys in a group 1230")
    assert number == "145.0"

    number = parse_first_number("some random text")
    assert number == ""


def test_llm_as_judge_eval() -> None:
    """Test llm_as_judge_eval function."""
    llm = MockLLM("gpt-3.5-turbo", responses=["1"])
    assert llm_as_judge_eval(llm, "What's the capital of France?", "Paris", "Paris")

    llm = MockLLM("gpt-3.5-turbo", responses=["abc"])
    assert not llm_as_judge_eval(
        llm, "What's the capital of France?", "Parismm", "Paris"
    )

    llm = MockLLM("gpt-3.5-turbo", responses=["1"])
    assert llm_as_judge_eval(
        llm,
        "What's the capital of France?",
        ["Paris", "France"],
        ["Paris", "France"],
        with_em=False,
    )


def test_em() -> None:
    """Test EM function."""
    # Test cases for exact match without normalization and numeric comparison.
    assert EM("Paris", "Paris", normalize=False, is_numeric=False) == True
    assert EM("Paris", "Berlin", normalize=False, is_numeric=False) == False

    # Test cases for exact match with normalization.
    assert EM(" Paris ", "paris", normalize=True, is_numeric=False) == True
    assert EM("Paris", "Berlin", normalize=True, is_numeric=False) == False

    # Test cases for exact match with numeric comparison.
    assert EM("3.14", "3.14", normalize=False, is_numeric=True) == True
    assert EM("3.14", "3.15", normalize=False, is_numeric=True) == False

    # Test cases for exact match with numeric comparison and normalization.
    assert EM(" 3.14 ", "3.14", normalize=True, is_numeric=True) == True
    assert EM("3.14", "3.15", normalize=True, is_numeric=True) == False

    # Test cases for exact match with numeric comparison (fuzzy should not apply).
    assert EM("3.14", "3.14", normalize=False, is_numeric=True) == True
    assert EM("3.14", "3.15", normalize=False, is_numeric=True) == False

    # Test cases for exact match with all options enabled.
    assert EM(" 3.14 ", "3.14", normalize=True, is_numeric=True) == True
    assert EM("3.14", "3.15", normalize=True, is_numeric=True) == False


def test_fuzzy_em() -> None:
    """Test fuzzy_EM function."""
    # Test cases for fuzzy match with default normalization.
    assert fuzzy_EM("Paris", "Pariss", normalize=False) == False
    assert fuzzy_EM("Paris", "Berlin", normalize=False) == False

    # Test cases for fuzzy match with normalization.
    assert fuzzy_EM(" Paris ", "pariss", normalize=True) == False
    assert fuzzy_EM("Paris", "Berlin", normalize=True) == False


def test_precision() -> None:
    """Test the precision calculation function."""
    # Test case 1: Perfect precision
    answer = "Paris is the capital of France"
    key = "Paris is the capital of France"
    assert precision(answer, key) == 1.0, "Failed: Perfect precision test"

    # Test case 2: Partial precision
    answer = "Paris is the capital of France"
    key = "Paris is a city in France"
    assert precision(answer, key) == 0.6, "Failed: Partial precision test"

    # Test case 3: No overlap
    answer = "Berlin is the capital of Germany"
    key = "Paris is the capital of France"
    assert precision(answer, key) == 0.6, "Failed: No overlap precision test"

    # Test case 4: Empty answer
    answer = ""
    key = "Paris is the capital of France"
    assert precision(answer, key) == 0.0, "Failed: Empty answer precision test"


def test_recall() -> None:
    """Test the recall calculation function."""
    # Test case 1: Perfect recall
    answer = "Paris is the capital of France"
    key = "Paris is the capital of France"
    assert recall(answer, key) == 1.0, "Failed: Perfect recall test"

    # Test case 2: Partial recall
    answer = "Paris is the capital of France"
    key = "Paris is a city in France"
    assert recall(answer, key) == 0.6, "Failed: Partial recall test"

    # Test case 3: No overlap
    answer = "Berlin is the capital of Germany"
    key = "Paris is the capital of France"
    assert recall(answer, key) == 0.6, "Failed: No overlap recall test"

    # Test case 4: Empty key
    answer = "Paris is the capital of France"
    key = ""
    assert recall(answer, key) == 0.0, "Failed: Empty key recall test"


def test_f1() -> None:
    """Test the F1 score calculation function."""
    # Test case 1: Perfect F1
    answer = "Paris is the capital of France"
    key = "Paris is the capital of France"
    assert f1(answer, key) == 1.0, "Failed: Perfect F1 test"

    # Test case 2: Partial F1
    answer = "Paris is the capital of France"
    key = "Paris is a city in France"
    precision_val = 0.8  # 4 common words / 5 predicted words
    recall_val = 1.0  # 4 common words / 4 ground truth words
    expected_f1 = 0.6
    assert f1(answer, key) == expected_f1, "Failed: Partial F1 test"

    # Test case 3: No overlap
    answer = "Berlin is the capital of Germany"
    key = "Paris is the capital of France"
    assert f1(answer, key) == 0.6, "Failed: No overlap F1 test"

    # Test case 4: Empty answer and key
    answer = ""
    key = ""
    assert f1(answer, key) == 0.0, "Failed: Empty answer and key F1 test"
