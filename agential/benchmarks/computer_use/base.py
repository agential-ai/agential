"""Base benchmark class for computer-use benchmarks."""

from typing import Any

import gymnasium as gym

from agential.benchmarks.base import BaseBenchmark


class BaseComputerUseBenchmark(gym.Env, BaseBenchmark):
    """Abstract base class for computer-use benchmarks."""

    def __init__(self, **kwargs: Any) -> None:
        """Initializes the computer-use benchmark."""
        super().__init__(**kwargs)
