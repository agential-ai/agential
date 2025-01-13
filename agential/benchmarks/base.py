"""Base benchmark class."""

from abc import ABC, abstractmethod
from typing import Any


class BaseBenchmark(ABC):
    """Abstract base class for defining a benchmark.

    The `BaseBenchmark` class defines the common interface for evaluating performance, providing
    inputs, and retrieving outputs during benchmark execution. Subclasses should implement the
    necessary methods to perform specific benchmarking tasks.

    Attributes:
        kwargs (dict): A dictionary of keyword arguments used for configuring or initializing
            the benchmark environment.
    """

    def __init__(self, **kwargs: Any) -> None:
        """Initializes the benchmark with the provided keyword arguments.

        Args:
            **kwargs: Configuration parameters for the benchmark. These could include settings
                for the environment, task-specific parameters, or any other necessary
                information for initializing the benchmark.
        """
        self.kwargs = kwargs

    @abstractmethod
    def evaluate(self) -> float:
        """Evaluates the performance of the agent, model, or system in the benchmark.

        This method must be implemented by subclasses to define how the benchmark is scored or
        evaluated. It should return an evaluation metric or score that indicates how well the
        agent or system performed on the task.

        Returns:
            float: The evaluation result, which could be a score, a metric, or other form of outcome
                depending on the task.

        Raises:
            NotImplementedError: If the method is not implemented in a subclass.
        """
        raise NotImplementedError
