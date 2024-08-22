"""Simple LLM wrapper for LiteLLM's completion function."""

import time

from abc import ABC, abstractmethod
from typing import Any, List
from pydantic import BaseModel, Field

from litellm import completion, cost_per_token


class Response(BaseModel):
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


class BaseLLM(ABC):
    """Base class for LLM."""

    def __init__(self, model: str) -> None:
        """Initialize."""
        self.model = model

    @abstractmethod
    def __call__(self, *args: Any, **kwargs: Any) -> Response:
        """Generate a mock response.

        Args:
            args (Any): Additional arguments.
            **kwargs (Any): Additional keyword arguments.

        Returns:
            Response: A mock response from the predefined list of responses.
        """
        pass


class LLM(BaseLLM):
    """Simple LLM wrapper for LiteLLM's completion function.

    Parameters:
        model (str): The name or identifier of the language model to use.
        kwargs (Any): Additional keyword arguments to pass to the completion function.
    """

    def __init__(self, model: str, **kwargs) -> None:
        """Initialize."""
        super().__init__(model=model)
        self.kwargs = kwargs

    def __call__(self, prompt: str, **kwargs: Any) -> Response:
        """Generate a response using the language model.

        Args:
            prompt (str): The input prompt for the language model.
            **kwargs (Any): Additional keyword arguments to pass to the completion function.

        Returns:
            Response: The response from the language model, typically containing generated text and metadata.
        """
        merged_kwargs = {**self.kwargs, **kwargs}
        start_time = time.time()
        response = completion(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            **merged_kwargs,
        )
        end_time = time.time()

        time_taken = end_time - start_time

        prompt_tokens_cost_usd_dollar, completion_tokens_cost_usd_dollar = cost_per_token(
            model=response.model,
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
        )

        return Response(
            input_text=prompt,
            output_text=response.choices[0].message.content,
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
            total_tokens=response.usage.total_tokens,
            prompt_cost=prompt_tokens_cost_usd_dollar,
            completion_cost=completion_tokens_cost_usd_dollar,
            total_cost=prompt_tokens_cost_usd_dollar + completion_tokens_cost_usd_dollar,
            prompt_time=time_taken,
        )
    

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

    def __call__(self, prompt: str, **kwargs: Any) -> Response:
        """Generate a mock response.

        Args:
            prompt (str): The input prompt (ignored in this mock implementation).
            **kwargs (Any): Additional keyword arguments (ignored in this mock implementation).

        Returns:
            Response: A mock response containing the next predefined text in the list.
        """
        response = self.responses[self.current_index]
        self.current_index = (self.current_index + 1) % len(self.responses)

        response = completion(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            mock_response=response,
            **kwargs,
        )
    
        prompt_tokens_cost_usd_dollar, completion_tokens_cost_usd_dollar = cost_per_token(
            model=response.model,
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
        )

        return Response(
            input_text=prompt,
            output_text=response.choices[0].message.content,
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
            total_tokens=response.usage.total_tokens,
            prompt_cost=prompt_tokens_cost_usd_dollar,
            completion_cost=completion_tokens_cost_usd_dollar,
            total_cost=prompt_tokens_cost_usd_dollar + completion_tokens_cost_usd_dollar,
            prompt_time=0.5,
        )
