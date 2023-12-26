"""Base agent interface class."""

from abc import ABC
from typing import Any

from pydantic.v1 import BaseModel


class BaseAgent(BaseModel, ABC):
    """Base agent class providing a general interface for agent operations."""

    def plan(self) -> Any:
        """Optionally implementable method to plan an action or set of actions."""
        raise NotImplementedError("Plan method not implemented.")

    def reflect(self) -> Any:
        """Optionally implementable method to reflect on the current state or past actions."""
        raise NotImplementedError("Reflect method not implemented.")

    def score(self) -> Any:
        """Optionally implementable method to score or evaluate something."""
        raise NotImplementedError("Score method not implemented.")

    def retrieve(self) -> Any:
        """Optionally implementable method to retrieve memory."""
        raise NotImplementedError("Retrieve method not implemented.")

    def generate(self) -> Any:
        """Optionally implementable method to generate a response."""
        raise NotImplementedError("Generate method not implemented.")
