"""Base Persona interface."""

from abc import ABC
from typing import Optional


class BasePersona(ABC):
    """Base persona class.

    Attributes:
        name (Optional[str]): Name of the persona.
        age (Optional[int]): Age of the persona.
        traits (Optional[str]): Traits describing the persona.
        status (Optional[str]): Status of the persona.
        lifestyle (Optional[str]): Lifestyle of the persona.
    """

    def __init__(
        self,
        name: Optional[str] = None,
        age: Optional[int] = None,
        traits: Optional[str] = None,
        status: Optional[str] = None,
        lifestyle: Optional[str] = None,
    ) -> None:
        """Initialization."""
        super().__init__()
        self.name = name
        self.age = age
        self.traits = traits
        self.status = status
        self.lifestyle = lifestyle
