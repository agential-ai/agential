"""Unit tests for general util functions."""

from agential.utils.general import safe_execute, shuffle_chunk_list, get_token_cost


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


def test_get_token_cost() -> None:
    """Test get_token_cost function."""
    # Test with sample token counts and model.
    prompt_tokens = 100
    completion_tokens = 50
    model = "gpt-3.5-turbo"

    cost_breakdown = get_token_cost(prompt_tokens, completion_tokens, model)

    assert isinstance(cost_breakdown, dict)
    assert "prompt_tokens_cost" in cost_breakdown
    assert "completion_tokens_cost" in cost_breakdown
    assert "total_tokens_cost" in cost_breakdown

    assert isinstance(cost_breakdown["prompt_tokens_cost"], float)
    assert isinstance(cost_breakdown["completion_tokens_cost"], float)
    assert isinstance(cost_breakdown["total_tokens_cost"], float)

    assert cost_breakdown["prompt_tokens_cost"] > 0
    assert cost_breakdown["completion_tokens_cost"] > 0
    assert cost_breakdown["total_tokens_cost"] == (
        cost_breakdown["prompt_tokens_cost"] + cost_breakdown["completion_tokens_cost"]
    )

    # Test with different token counts and model.
    prompt_tokens = 200
    completion_tokens = 100
    model = "gpt-4"

    cost_breakdown = get_token_cost(prompt_tokens, completion_tokens, model)

    assert isinstance(cost_breakdown, dict)
    assert "prompt_tokens_cost" in cost_breakdown
    assert "completion_tokens_cost" in cost_breakdown
    assert "total_tokens_cost" in cost_breakdown

    assert isinstance(cost_breakdown["prompt_tokens_cost"], float)
    assert isinstance(cost_breakdown["completion_tokens_cost"], float)
    assert isinstance(cost_breakdown["total_tokens_cost"], float)

    assert cost_breakdown["prompt_tokens_cost"] > 0
    assert cost_breakdown["completion_tokens_cost"] > 0
    assert cost_breakdown["total_tokens_cost"] == (
        cost_breakdown["prompt_tokens_cost"] + cost_breakdown["completion_tokens_cost"]
    )

