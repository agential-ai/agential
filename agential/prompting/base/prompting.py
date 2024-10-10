"""Base prompting interface class."""

from abc import abstractmethod
from typing import Any

from agential.core.base.method import BaseMethod
from agential.core.llm import BaseLLM
from agential.prompting.base.output import BasePromptingOutput
from agential.prompting.base.strategies import BasePromptingStrategy


class BasePrompting(BaseMethod):
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
        super().__init__(llm=llm, benchmark=benchmark, testing=testing)

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
