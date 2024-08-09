"""Unit tests for general util functions."""

from agential.llm.llm import ModelResponse, Usage
from agential.utils.general import get_token_and_cost, safe_execute, shuffle_chunk_list


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


def test_get_token_and_cost() -> None:
    """Test get_token_and_cost function."""
    # Create a mock ModelResponse object.

    # Test with sample token counts and model.
    prompt_tokens = 100
    completion_tokens = 50
    model = "gpt-3.5-turbo"

    usage = Usage()
    usage.prompt_tokens = prompt_tokens
    usage.completion_tokens = completion_tokens
    usage.total_tokens = prompt_tokens + completion_tokens

    response = ModelResponse()
    response.choices = []
    response.usage = usage
    response.model = model
    response.time_taken = 0.5
    
    token_and_cost = get_token_and_cost(response)

    assert isinstance(token_and_cost, dict)
    assert "prompt_tokens" in token_and_cost
    assert "completion_tokens" in token_and_cost
    assert "total_tokens" in token_and_cost
    assert "prompt_tokens_cost" in token_and_cost
    assert "completion_tokens_cost" in token_and_cost
    assert "total_tokens_cost" in token_and_cost

    assert isinstance(token_and_cost["prompt_tokens"], int)
    assert isinstance(token_and_cost["completion_tokens"], int)
    assert isinstance(token_and_cost["total_tokens"], int)
    assert isinstance(token_and_cost["prompt_tokens_cost"], float)
    assert isinstance(token_and_cost["completion_tokens_cost"], float)
    assert isinstance(token_and_cost["total_tokens_cost"], float)

    assert token_and_cost["prompt_tokens"] == prompt_tokens
    assert token_and_cost["completion_tokens"] == completion_tokens
    assert token_and_cost["total_tokens"] == prompt_tokens + completion_tokens
    assert token_and_cost["prompt_tokens_cost"] > 0
    assert token_and_cost["completion_tokens_cost"] > 0
    assert token_and_cost["total_tokens_cost"] == (
        token_and_cost["prompt_tokens_cost"] + token_and_cost["completion_tokens_cost"]
    )
    assert token_and_cost["time_sec"] == 0.5

    # Test with different token counts and model.
    prompt_tokens = 200
    completion_tokens = 100
    model = "gpt-4"

    usage = Usage()
    usage.prompt_tokens = prompt_tokens
    usage.completion_tokens = completion_tokens
    usage.total_tokens = prompt_tokens + completion_tokens

    response = ModelResponse()
    response.choices = []
    response.usage = usage
    response.model = model
    response.time_taken = 0.5
    
    token_and_cost = get_token_and_cost(response)

    assert isinstance(token_and_cost, dict)
    assert "prompt_tokens" in token_and_cost
    assert "completion_tokens" in token_and_cost
    assert "total_tokens" in token_and_cost
    assert "prompt_tokens_cost" in token_and_cost
    assert "completion_tokens_cost" in token_and_cost
    assert "total_tokens_cost" in token_and_cost

    assert isinstance(token_and_cost["prompt_tokens"], int)
    assert isinstance(token_and_cost["completion_tokens"], int)
    assert isinstance(token_and_cost["total_tokens"], int)
    assert isinstance(token_and_cost["prompt_tokens_cost"], float)
    assert isinstance(token_and_cost["completion_tokens_cost"], float)
    assert isinstance(token_and_cost["total_tokens_cost"], float)

    assert token_and_cost["prompt_tokens"] == prompt_tokens
    assert token_and_cost["completion_tokens"] == completion_tokens
    assert token_and_cost["total_tokens"] == prompt_tokens + completion_tokens
    assert token_and_cost["prompt_tokens_cost"] > 0
    assert token_and_cost["completion_tokens_cost"] > 0
    assert token_and_cost["total_tokens_cost"] == (
        token_and_cost["prompt_tokens_cost"] + token_and_cost["completion_tokens_cost"]
    )
    assert token_and_cost["time_sec"] == 0.5