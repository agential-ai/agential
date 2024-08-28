"""Unit tests for general util functions."""

from agential.utils.general import safe_execute, shuffle_chunk_list


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


# Ref: https://github.com/microsoft/ProphetNet/blob/master/CRITIC/src/tools/interpreter_api.py.
def test_safe_execute() -> None:
    """Test safe_execute function."""
    code_string = """budget = 1000\nfood = 0.3\naccommodation = 0.15\nentertainment = 0.25\ncoursework_materials = 1 - food - accommodation - entertainment\nanswer = budget * coursework_materials"""
    answer, report = safe_execute(code_string)
    assert int(answer[0]) == 299
    assert report == "Done"
