"""Base benchmark class."""

from abc import ABC, abstractmethod
from typing import Any


class BaseBenchmark(ABC):
    """
    Abstract base class for defining a benchmark. This class serves as a foundation for creating
    various benchmark classes that test agents, models, or systems in different domains.

    The `BaseBenchmark` class defines the common interface for evaluating performance, providing
    inputs, and retrieving outputs during benchmark execution. Subclasses should implement the
    necessary methods to perform specific benchmarking tasks.

    Attributes:
        kwargs (dict): A dictionary of keyword arguments used for configuring or initializing
                       the benchmark environment.

    Methods:
        __init__(**kwargs):
            Initializes the benchmark with the given keyword arguments.

        evaluate():
            Abstract method that evaluates the performance of the agent or system on the benchmark.
            Implementations should define the evaluation logic based on the task at hand.

        get_input():
            Abstract method that retrieves the input(s) for the benchmark. This could involve
            retrieving a dataset, generating a test case, or preparing the environment for execution.

        get_output():
            Abstract method that retrieves the output(s) from the benchmark execution. This could
            involve collecting results, processing responses, or gathering system output.

    """

    def __init__(self, **kwargs: Any) -> None:
        """
        Initializes the benchmark with the provided keyword arguments.

        Args:
            **kwargs: Configuration parameters for the benchmark. These could include settings
                      for the environment, task-specific parameters, or any other necessary
                      information for initializing the benchmark.

        """
        self.kwargs = kwargs

    @abstractmethod
    def evaluate(self) -> float:
        """
        Evaluates the performance of the agent, model, or system in the benchmark.

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

    @abstractmethod
    def get_input(self) -> Any:
        """
        Retrieves the input(s) for the benchmark.

        This method must be implemented by subclasses to define how the inputs are prepared or
        accessed for the benchmark. The input could be a dataset, a task description, or any
        other form of information required for the agent or system to perform its task.

        Returns:
            Any: The input required for the benchmark task.

        Raises:
            NotImplementedError: If the method is not implemented in a subclass.
        """
        raise NotImplementedError

    @abstractmethod
    def get_output(self) -> Any:
        """
        Retrieves the output(s) from the benchmark execution.

        This method must be implemented by subclasses to define how the output of the benchmark
        is collected or processed. The output could be the result of the agent's response, system
        output, or any other relevant result produced during the benchmark execution.

        Returns:
            Any: The output of the benchmark execution.

        Raises:
            NotImplementedError: If the method is not implemented in a subclass.
        """
        raise NotImplementedError
