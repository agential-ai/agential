"""Base agent interface class."""

from abc import abstractmethod
from typing import Any

from agential.agents.base.output import BaseAgentOutput
from agential.agents.base.strategies import BaseAgentStrategy
from agential.core.base.method import BaseMethod
from agential.core.llm import BaseLLM


class BaseAgent(BaseMethod):
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
        super().__init__(llm=llm, benchmark=benchmark, testing=testing)

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
