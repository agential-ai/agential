"""Base agent interface class."""

from abc import ABC, abstractmethod

class BaseAgent(ABC):
    """Base agent class providing a general interface for agent operations.

    Parameters:
        llm (BaseLLM): An instance of a language model used for generating initial answers
            and critiques.
        benchmark (str): The benchmark.
        testing (bool, optional): Whether to run in testing mode. Defaults to False.
    """

    def __init__(self, llm, benchmark, testing=False):
        """Initialization."""
        self.llm = llm
        self.benchmark = benchmark
        self.testing = testing

    @abstractmethod
    def generate(self, *args, **kwargs):
        """Generate a response.

        Args:
            *args (Any): Additional arguments.
            **kwargs (Any): Additional keyword arguments.

        Returns:
            BaseAgentOutput: The generated response.
        """
        raise NotImplementedError
