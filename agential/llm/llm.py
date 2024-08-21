"""Simple LLM wrapper for LiteLLM's completion function."""

import time

from abc import ABC, abstractmethod
from typing import Any, List

from litellm import completion


class Message:
    """Represents a message with content."""

    content: str


class Choices:
    """Represents a choice with a message."""

    message: Message


class Usage:
    """Represents usage information."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ModelResponse:
    """Represents a model response with choices."""

    choices: List[Choices]
    usage: Usage
    model: str
    time_taken: float


class BaseLLM(ABC):
    """Base class for LLM."""

    def __init__(self, model: str) -> None:
        """Initialize."""
        self.model = model

    @abstractmethod
    def __call__(self, *args: Any, **kwargs: Any) -> ModelResponse:
        """Generate a mock response.

        Args:
            args (Any): Additional arguments.
            **kwargs (Any): Additional keyword arguments.

        Returns:
            ModelResponse: A mock response from the predefined list of responses.
        """
        pass


from litellm import cost_per_token
from pydantic import BaseModel, Field

from agential.llm.llm import ModelResponse


class PromptMetrics(BaseModel):
    """Prompt metrics Pydantic output class.

    Attributes:
        prompt_tokens (int): The number of tokens in the prompt.
        completion_tokens (int): The number of tokens in the completion.
        total_tokens (int): The total number of tokens in the prompt and completion.
        prompt_cost (float): The cost of the prompt tokens in dollars.
        completion_cost (float): The cost of the completion tokens in dollars.
        total_cost (float): The total cost of the prompt and completion tokens in dollars.
        prompt_time (float): The time it took to generate the prompt in seconds.
    """

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


def get_token_cost_time(response: ModelResponse) -> PromptMetrics:
    """Calculates the token usage and cost of a prompt and completion in dollars.

    Args:
        response (ModelResponse): The response object containing the usage information.

    Returns:
        PromptMetrics: A Pydantic object containing the token usage and cost breakdown:
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

    return PromptMetrics(
        prompt_tokens=response.usage.prompt_tokens,
        completion_tokens=response.usage.completion_tokens,
        total_tokens=response.usage.total_tokens,
        prompt_cost=prompt_tokens_cost_usd_dollar,
        completion_cost=completion_tokens_cost_usd_dollar,
        total_cost=prompt_tokens_cost_usd_dollar + completion_tokens_cost_usd_dollar,
        prompt_time=response.time_taken,
    )


class LLM(BaseLLM):
    """Simple LLM wrapper for LiteLLM's completion function.

    Parameters:
        model (str): The name or identifier of the language model to use.
        kwargs (Any): Additional keyword arguments to pass to the completion function.
    """

    def __init__(self, model: str, **kwargs) -> None:
        """Initialize."""
        super().__init__(model=model)
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_tokens = 0
        self.total_prompt_cost = 0
        self.total_completion_cost = 0
        self.total_cost = 0
        self.total_prompt_time = 0
        self.kwargs = kwargs

    def __call__(self, prompt: str, **kwargs: Any) -> ModelResponse:
        """Generate a response using the language model.

        Args:
            prompt (str): The input prompt for the language model.
            **kwargs (Any): Additional keyword arguments to pass to the completion function.

        Returns:
            ModelResponse: The response from the language model, typically containing generated text and metadata.
        """
        start_time = time.time()
        merged_kwargs = {**self.kwargs, **kwargs}
        response = completion(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            **merged_kwargs,
        )
        end_time = time.time()

        response.time_taken = end_time - start_time

        o = get_token_cost_time(response)
        self.total_prompt_tokens += o.prompt_tokens
        self.total_completion_tokens += o.completion_tokens
        self.total_tokens += o.total_tokens
        self.total_prompt_cost += o.prompt_cost
        self.total_completion_cost += o.completion_cost
        self.total_cost += o.total_cost
        self.total_prompt_time += o.prompt_time

        return response


class MockLLM(BaseLLM):
    """Mock LLM class for testing purposes.

    Parameters:
        model (str): The name or identifier of the language model to use.
        responses (List[str]): The list of predefined responses to return.
    """

    def __init__(self, model: str, responses: List[str]):
        """Initialize."""
        super().__init__(model=model)
        self.responses = responses
        self.current_index = 0

    def __call__(self, prompt: str, **kwargs: Any) -> ModelResponse:
        """Generate a mock response.

        Args:
            prompt (str): The input prompt (ignored in this mock implementation).
            **kwargs (Any): Additional keyword arguments (ignored in this mock implementation).

        Returns:
            ModelResponse: A mock response containing the next predefined text in the list.
        """
        response = self.responses[self.current_index]
        self.current_index = (self.current_index + 1) % len(self.responses)

        response = completion(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            mock_response=response,
            **kwargs,
        )

        response.time_taken = 0.5
        return response
