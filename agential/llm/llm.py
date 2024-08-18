"""Simple LLM wrapper for LiteLLM's completion function."""

import time

from abc import ABC, abstractmethod
from typing import Any, List
from pydantic import BaseModel, Field

from litellm import completion, cost_per_token

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

class LLM(BaseLLM):
    """Simple LLM wrapper for LiteLLM's completion function.

    Parameters:
        model (str): The name or identifier of the language model to use.
    """

    def __init__(self, model: str) -> None:
        """Initialize."""
        super().__init__(model=model)

    def __call__(self, prompt: str, **kwargs: Any) -> ModelResponse:
        """Generate a response using the language model.

        Args:
            prompt (str): The input prompt for the language model.
            **kwargs (Any): Additional keyword arguments to pass to the completion function.

        Returns:
            ModelResponse: The response from the language model, typically containing generated text and metadata.
        """
        start_time = time.time()
        response = completion(
            model=self.model, messages=[{"role": "user", "content": prompt}], **kwargs
        )
        end_time = time.time()

        response.time_taken = end_time - start_time
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
