"""Unit tests for EM evaluation metric."""

from agential.eval.em import (
    EM,
    normalize_answer,
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
