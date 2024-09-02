"""Unit tests for classification evaluation metrics."""

from agential.eval.metrics.classification import (
    EM,
    f1,
    normalize_answer,
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


def test_em() -> None:
    """Test EM function."""
    sample_answer = None
    sample_key = None
    result = EM(sample_answer, sample_key)
    assert not result

    sample_answer = (
        "A fox jumped over the fence. An apple was on the table. The quick brown fox."
    )
    sample_key = (
        "A fox jumped over the fence. An apple was on the table. The quick brown fox."
    )
    result = EM(sample_answer, sample_key)
    assert result

    sample_answer = (
        "1A fox jumped over the fence. An apple was on the table. The quick brown fox."
    )
    sample_key = (
        "A fox jumped over the fence. An apple was on the table. The quick brown fox."
    )
    result = EM(sample_answer, sample_key)
    assert not result

    sample_answer = " A fox jumped over the fence. "
    sample_key = "A fox jumped over the fence."
    result = EM(sample_answer, sample_key, normalize=False)
    assert not result

    sample_answer = "A fox jumped over the fence."
    sample_key = "A fox jumped over the fence."
    result = EM(sample_answer, sample_key, normalize=False)
    assert result


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
