"""Unit tests for general util functions."""

from discussion_agents.utils.general import shuffle_chunk_list, safe_execute, remove_comment


def test_shuffle_chunk_list() -> None:
    """Test shuffle_chunk_list."""
    lst = list(range(10)) + list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    chunked_lst = shuffle_chunk_list(lst, k=3)
    gt_chunked_lst = [
        [7, 1, "H"],
        ["F", "E", 4],
        ["N", 3, "L"],
        ["T", "X", 2],
        ["I", "D", "Y"],
        [0, "O", 6],
        ["Z", "G", "K"],
        [8, "S", "B"],
        ["A", "V", "Q"],
        ["M", "P", "U"],
        ["J", "W", "R"],
        [5, "C", 9],
    ]
    assert chunked_lst == gt_chunked_lst

def test_safe_excute():
    code_string_1 = """import numpy as np
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
an = np.array([5, 7, 9])
answer = a + b"""

    code_string_2 = """budget = 1000
food = 0.3
accommodation = 0.15
entertainment = 0.25
coursework_materials = 1 - food - accommodation - entertainment
answer = budget * coursework_materials
"""
    an, report = safe_execute(code_string_1)
    assert report == 'Done'

    an, report = safe_execute(code_string_2)
    print(an, report)
    assert report == 'Done'

def test_remove_comments():
    code = """# This is a comment\n# Another comment\nint x = 1"""
    expected = "int x = 1"
    result = remove_comment(code)
    assert result == expected