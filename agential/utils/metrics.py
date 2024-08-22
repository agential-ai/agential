"""Utility functions for prompt metrics calculation."""

from litellm import cost_per_token
from pydantic import BaseModel, Field

from agential.llm.llm import ModelResponse


class PromptInfo(BaseModel):
    """Prompt info Pydantic output class.

    Attributes:
        input_text (str): The input text.
        output_text (str): The output text.
        prompt_tokens (int): The number of tokens in the prompt.
        completion_tokens (int): The number of tokens in the completion.
        total_tokens (int): The total number of tokens in the prompt and completion.
        prompt_cost (float): The cost of the prompt tokens in dollars.
        completion_cost (float): The cost of the completion tokens in dollars.
        total_cost (float): The total cost of the prompt and completion tokens in dollars.
        prompt_time (float): The time it took to generate the prompt in seconds.
    """

    input_text: str = Field(..., description="The input text.")
    output_text: str = Field(..., description="The output text.")
    prompt_tokens: int = Field(..., description="The number of tokens in the prompt.")
    completion_tokens: int = Field(
        ..., description="The number of tokens in the completion."
    )
    total_tokens: int = Field(
        ..., description="The total number of tokens in the prompt and completion."
    )
    prompt_cost: float = Field(
        ..., description="The cost of the prompt tokens in dollars."
    )
    completion_cost: float = Field(
        ..., description="The cost of the completion tokens in dollars."
    )
    total_cost: float = Field(
        ...,
        description="The total cost of the prompt and completion tokens in dollars.",
    )
    prompt_time: float = Field(
        ..., description="The time taken to generate the response in seconds."
    )


def get_prompt_info(response: ModelResponse) -> PromptInfo:
    """Calculates the token usage and cost of a prompt and completion in dollars.

    Args:
        response (ModelResponse): The response object containing the usage information.

    Returns:
        PromptInfo: A Pydantic object containing the token usage and cost breakdown:
            - "prompt_tokens": The number of tokens in the prompt.
            - "completion_tokens": The number of tokens in the completion.
            - "total_tokens": The total number of tokens in the prompt and completion.
            - "prompt_cost": The cost of the prompt tokens in dollars.
            - "completion_cost": The cost of the completion tokens in dollars.
            - "total_cost": The total cost of the prompt and completion tokens in dollars.
            - "prompt_time": The time taken to generate the response in seconds.
    """
    prompt_tokens_cost_usd_dollar, completion_tokens_cost_usd_dollar = cost_per_token(
        model=response.model,
        prompt_tokens=response.usage.prompt_tokens,
        completion_tokens=response.usage.completion_tokens,
    )

    return PromptInfo(
        prompt_tokens=response.usage.prompt_tokens,
        completion_tokens=response.usage.completion_tokens,
        total_tokens=response.usage.total_tokens,
        prompt_cost=prompt_tokens_cost_usd_dollar,
        completion_cost=completion_tokens_cost_usd_dollar,
        total_cost=prompt_tokens_cost_usd_dollar + completion_tokens_cost_usd_dollar,
        prompt_time=response.time_taken,
    )
