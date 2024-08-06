"""Simple LLM wrapper for LiteLLM's completion function."""

from typing import List, Any
from abc import ABC, abstractmethod
from litellm import completion
from litellm.types.utils import ModelResponse

class BaseLLM(ABC):
    """Base class for LLM."""

    @abstractmethod
    def __call__(self, *args: Any, **kwargs: Any) -> ModelResponse:
        """Generate a mock response.

        Args:
            prompt (str): The input prompt.
            **kwargs: Additional keyword arguments.

        Returns:
            ModelResponse: A mock response from the predefined list of responses.
        """    
        pass


class LLM(BaseLLM):
    """Simple LLM wrapper for LiteLLM's completion function."""

    def __init__(self, model: str) -> None:
        """Initialize the LLM with a specified model.

        Args:
            model (str): The name or identifier of the language model to use.
        """
        self.model = model

    def __call__(self, prompt: str, **kwargs) -> ModelResponse:
        """Generate a response using the language model.

        Args:
            prompt (str): The input prompt for the language model.
            **kwargs: Additional keyword arguments to pass to the completion function.

        Returns:
            ModelResponse: The response from the language model, typically containing generated text and metadata.
        """
        response = completion(model=self.model, messages=[{"role": "user", "content": prompt}], **kwargs)
        return response

class MockLLM(BaseLLM):
    """Mock LLM class for testing purposes."""

    def __init__(self, model: str, responses: List[str]):
        """Initialize the MockLLM with a list of predefined responses.

        Args:
            model (str): The name or identifier of the language model to use.
            responses (List[str]): The list of predefined responses to return.
        """
        self.model = model
        self.responses = responses
        self.current_index = 0

    def __call__(self, prompt: str, **kwargs) -> ModelResponse:
        """Generate a mock response.

        Args:
            prompt (str): The input prompt (ignored in this mock implementation).
            **kwargs: Additional keyword arguments (ignored in this mock implementation).

        Returns:
            ModelResponse: A mock response containing the next predefined text in the list.
        """
        response = self.responses[self.current_index]
        self.current_index = (self.current_index + 1) % len(self.responses)

        response = completion(model=self.model, messages=[{"role": "user", "content": prompt}], mock_response=response, **kwargs)
        return response