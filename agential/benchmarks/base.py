"""Base benchmark class."""

from abc import ABC
from typing import Any


class BaseBenchmark(ABC):
    """Abstract base class for defining a benchmark."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the base benchmark class."""
        super().__init__()
