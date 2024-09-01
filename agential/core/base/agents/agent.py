"""Base agent interface class."""

from abc import ABC, abstractmethod
from typing import Any, Dict

from agential.core.base.agents.output import BaseAgentOutput
from agential.core.base.agents.strategies import BaseAgentStrategy
from agential.llm.llm import BaseLLM


class BaseAgent(ABC):
    """Base agent class providing a general interface for agent operations.

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
    def get_strategy(self, benchmark: str, **kwargs: Any) -> BaseAgentStrategy:
        """Returns an instance of the appropriate strategy based on the provided benchmark.

        Args:
            benchmark (str): The benchmark name.
            **kwargs (Dict[str, Any]): Additional keyword arguments to pass to
                the strategy's constructor.

        Returns:
            BaseAgentStrategy: An instance of the appropriate strategy.
        """
        raise NotImplementedError

    @abstractmethod
    def generate(self, *args: Any, **kwargs: Any) -> BaseAgentOutput:
        """Generate a response.

        Args:
            *args (Any): Additional arguments.
            **kwargs (Any): Additional keyword arguments.

        Returns:
                BaseAgentOutput: The generated response.
        """
        raise NotImplementedError
