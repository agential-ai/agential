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
    """Extended ModelResponse with additional attributes."""

    def __init__(self, choices: List[Choices], usage: Usage, model: str, time_taken: float, input_text: str, output_text: str) -> None:
        """"Initialize."""
        self.choices = choices
        self.usage = usage
        self.model = model
        self.time_taken = time_taken
        self.input_text = input_text
        self.output_text = output_text


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
        kwargs (Any): Additional keyword arguments to pass to the completion function.
    """

    def __init__(self, model: str, **kwargs) -> None:
        """Initialize."""
        super().__init__(model=model)
        self.kwargs = kwargs

    def __call__(self, prompt: str, **kwargs: Any) -> ModelResponse:
        """Generate a response using the language model.

        Args:
            prompt (str): The input prompt for the language model.
            **kwargs (Any): Additional keyword arguments to pass to the completion function.

        Returns:
            ModelResponse: The response from the language model, typically containing generated text and metadata.
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

        return ModelResponse(
            choices=response.choices,
            usage=response.usage,
            model=response.model,
            time_taken=time_taken,
            input_text=prompt,
            output_text=response.choices[0].message.content,
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
