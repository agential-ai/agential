"""Unit tests for parsing-related functions."""
from discussion_agents.utils.parse import parse_list, parse_numbered_list, remove_name


def test_parse_list():
    """Test parse list function."""
    gt = ["Item 1", "Item 2", "Item 3", "Item 4"]

    x = "1. Item 1\n2. Item 2\n3. Item 3\n\n4. Item 4"
    out = parse_list(x)

    assert len(gt) == len(out)
    for i, j in zip(gt, out):
        assert i == j


def test_parse_numbered_list():
    """Test parse_numbered_list function."""
    gt = ["Item One", "Item Two", "Item Three"]

    input_text = "1) Item One.\n2) Item Two.\n3) Item Three,\n"
    out = parse_numbered_list(input_text)

    assert len(gt) == len(out)
    for i, j in zip(gt, out):
        assert i == j


def test_remove_name():
    """Test remove_name function."""
    gt = "Smith"

    x = "John Smith"
    out = remove_name(x, "John")

    assert out == gt
