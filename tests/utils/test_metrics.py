"""Unit tests for metric utilities."""

from agential.llm.llm import ModelResponse, Usage
from agential.utils.metrics import PromptMetrics, get_token_cost_time


def test_get_token_cost_time() -> None:
    """Test get_token_cost_time function."""
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

    token_cost_time = get_token_cost_time(response)

    assert token_cost_time == PromptMetrics(
        prompt_tokens=100,
        completion_tokens=50,
        total_tokens=150,
        prompt_cost=0.00015000000000000001,
        completion_cost=9.999999999999999e-05,
        total_cost=0.00025,
        prompt_time=0.5,
    )

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

    token_cost_time = get_token_cost_time(response)

    assert token_cost_time == PromptMetrics(
        prompt_tokens=200,
        completion_tokens=100,
        total_tokens=300,
        prompt_cost=0.006,
        completion_cost=0.006,
        total_cost=0.012,
        prompt_time=0.5,
    )
