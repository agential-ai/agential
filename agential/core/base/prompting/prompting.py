"""Base prompting interface class."""

from abc import ABC, abstractmethod
from typing import Any, Dict

from agential.core.base.prompting.output import BasePromptingOutput
from agential.core.base.prompting.strategies import BasePromptingStrategy
from agential.llm.llm import BaseLLM


class BasePrompting(ABC):
    """Base prompting method class providing a general interface for prompt method operations.

    Parameters:
        llm (BaseLLM): An instance of a language model used for generating initial answers
            and critiques.
        benchmark (str): The benchmark.
        testing (bool, optional): Whether to run in testing mode. Defaults to False.
    """

    def __init__(
        self,
        llm: BaseLLM,
        benchmark: str,
        testing: bool = False,
    ) -> None:
        """Initialization."""
        super().__init__()
        self.llm = llm
        self.benchmark = benchmark
        self.testing = testing

    @abstractmethod
    def get_fewshots(
        self, benchmark: str, fewshot_type: str, **kwargs: Any
    ) -> Dict[str, str]:
        """Retrieve few-shot examples based on the benchmark.

        Args:
            benchmark (str): The benchmark name.
            fewshot_type (str): The benchmark few-shot type.
            **kwargs (Any): Additional arguments.

        Returns:
            Dict[str, str]: A dictionary of few-shot examples.
        """
        raise NotImplementedError

    @abstractmethod
    def get_prompts(self, benchmark: str, **kwargs: Any) -> Dict[str, str]:
        """Retrieve the prompt instructions based on the benchmark.

        Args:
            benchmark (str): The benchmark name.
            **kwargs (Any): Additional arguments.

        Returns:
            Dict[str, str]: A dictionary of prompt instructions.
        """
        raise NotImplementedError

    @abstractmethod
    def get_strategy(self, benchmark: str, **kwargs: Any) -> BasePromptingStrategy:
        """Returns an instance of the appropriate strategy based on the provided benchmark.

        Args:
            benchmark (str): The benchmark name.
            **kwargs (Dict[str, Any]): Additional keyword arguments to pass to
                the strategy's constructor.

        Returns:
            BasePromptingStrategy: An instance of the appropriate strategy.
        """
        raise NotImplementedError

    @abstractmethod
    def generate(self, *args: Any, **kwargs: Any) -> BasePromptingOutput:
        """Generate a response.

        Args:
            *args (Any): Additional arguments.
            **kwargs (Any): Additional keyword arguments.

        Returns:
                BasePromptingOutput: The generated response.
        """
        raise NotImplementedError
