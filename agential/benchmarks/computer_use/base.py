"""Base benchmark class for computer-use benchmarks."""

from abc import abstractmethod
from typing import Any

from agential.benchmarks.base import BaseBenchmark


class BaseComputerUseBenchmark(BaseBenchmark):
    """Abstract base class for computer-use benchmarks.

    Subclasses should implement the methods to manage the lifecycle of the benchmark task,
    including setup, execution steps, and teardown.
    """

    def __init__(self, **kwargs: Any) -> None:
        """Initializes the computer-use benchmark with the provided configuration parameters.

        Args:
            **kwargs: Configuration parameters, passed on to the parent class for initializing
                the benchmark environment.
        """
        super().__init__(**kwargs)

    @abstractmethod
    def close(self) -> None:
        """Closes the benchmark environment or any associated resources.

        This method should be implemented by subclasses to define how the benchmark environment
        is closed or cleaned up after the benchmark task is finished. This may involve closing
        applications, releasing resources, or any other necessary teardown operations.

        Raises:
            NotImplementedError: If the method is not implemented in a subclass.
        """
        raise NotImplementedError

    @abstractmethod
    def reset(self, **kargs: Any) -> Any:
        """Resets the benchmark task to its initial state.

        This method should be implemented by subclasses to define how the benchmark environment
        or task is reset. It could involve resetting the environment, clearing data, or preparing
        the system for a new evaluation.

        Returns:
            The result of the reset operation, which could be an updated state or configuration
            of the environment.

        Raises:
            NotImplementedError: If the method is not implemented in a subclass.
        """
        raise NotImplementedError

    @abstractmethod
    def step(self, **kargs: Any) -> Any:
        """Executes a single step of the benchmark task.

        This method should be implemented by subclasses to define the actions that occur at each
        step of the benchmark. This could involve interacting with the environment, making decisions,
        or performing specific tasks as part of the benchmark evaluation.

        Returns:
            The result of the step, which could include updated states, scores, or other relevant
                outcomes from the step execution.

        Raises:
            NotImplementedError: If the method is not implemented in a subclass.
        """
        raise NotImplementedError
