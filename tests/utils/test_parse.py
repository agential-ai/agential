"""Unit tests for parsing-related functions."""
from discussion_agents.utils.parse import (
    parse_list, 
    parse_numbered_list, 
    remove_name,
    clean_str, 
    get_page_obs,
    construct_lookup_list,
    remove_articles,
    white_space_fix,
    remove_punc,
    normalize_answer
)


def test_parse_list() -> None:
    """Test parse list function."""
    gt = ["Item 1", "Item 2", "Item 3", "Item 4"]

    x = "1. Item 1\n2. Item 2\n3. Item 3\n\n4. Item 4"
    out = parse_list(x)

    assert len(gt) == len(out)
    for i, j in zip(gt, out):
        assert i == j


def test_parse_numbered_list() -> None:
    """Test parse_numbered_list function."""
    gt = ["Item One", "Item Two", "Item Three"]

    input_text = "1) Item One.\n2) Item Two.\n3) Item Three,\n"
    out = parse_numbered_list(input_text)

    assert len(gt) == len(out)
    for i, j in zip(gt, out):
        assert i == j


def test_remove_name() -> None:
    """Test remove_name function."""
    gt = "Smith"

    x = "John Smith"
    out = remove_name(x, "John")

    assert out == gt


def test_clean_str() -> None:
    """Test clean_str function."""
    s = "a string"
    out = clean_str(s)
    assert out == "a string"

    gt = "Café â\x80\x93 Büro"
    out = clean_str("Caf\u00e9 \xe2\x80\x93 B\u00fcro")
    assert out == gt


def test_get_page_obs() -> None:
    """Test get_page_obs function."""
    sample_page = """
    The quick brown fox jumps over the lazy dog. This sentence contains all the letters of the alphabet. 
    It's often used for typing practice. 

    Science is fascinating. It covers many areas, like physics, biology, and chemistry. Physics deals with the fundamental particles of the universe. Biology is the study of living beings. Chemistry focuses on the composition of substances. 

    Mathematics is the language of science. It is used to describe the laws of nature.
    """
    
    gt = "The quick brown fox jumps over the lazy dog. This sentence contains all the letters of the alphabet.. It's often used for typing practice.. Science is fascinating. It covers many areas, like physics, biology, and chemistry."

    result = get_page_obs(sample_page)
    assert result == gt


def test_construct_lookup_list() -> None:
    """Test construct_lookup_list function."""
    sample_page = """
    The quick brown fox jumps over the lazy dog. Foxes are wild animals. 
    In many cultures, foxes are depicted as cunning creatures. 

    This paragraph talks about other animals. For instance, lions are considered the king of the jungle.
    """

    keyword1 = "fox"
    keyword2 = "Lions"

    # Test with keyword "fox".
    result1 = construct_lookup_list(keyword1, sample_page)
    expected1 = [
        'The quick brown fox jumps over the lazy dog.',
        'Foxes are wild animals..',
        'In many cultures, foxes are depicted as cunning creatures..'
    ]
    assert result1 == expected1

    # Test with keyword "Lions".
    result2 = construct_lookup_list(keyword2, sample_page)
    expected2 = ['For instance, lions are considered the king of the jungle..']
    assert result2 == expected2, "Test with keyword 'Lions' failed."

    # Test with no page provided.
    result3 = construct_lookup_list(keyword1)
    expected3 = []
    assert result3 == expected3, "Test with no page provided failed."


def test_remove_articles() -> None:
    """Test remove_articles function."""
    sample_text = "A fox jumped over the fence. An apple was on the table. The quick brown fox."

    result = remove_articles(sample_text)
    expected = 'A fox jumped over   fence. An apple was on   table. The quick brown fox.'
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
    sample_text = "A fox jumped over the fence. An apple was on the table. The quick brown fox."

    result = normalize_answer(sample_text)
    expected = 'fox jumped over fence apple was on table quick brown fox'
    assert result == expected
