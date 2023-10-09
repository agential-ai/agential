"""Unit tests for parsing-related functions."""
from discussion_agents.utils.parse import parse_list


def test_parse_list():
    """Test parse list function."""
    gt = ["Item 1", "Item 2", "Item 3", "Item 4"]

    x = "1. Item 1\n2. Item 2\n3. Item 3\n\n4. Item 4"
    out = parse_list(x)

    assert len(gt) == len(out)
    for i, j in zip(gt, out):
        assert i == j
