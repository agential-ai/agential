"""Base Persona interface."""

from abc import ABC

from pydantic.v1 import BaseModel


class BasePersona(BaseModel, ABC):
    """Base persona class."""

    name: str = None
    age: int = None
    traits: str = None
    status: str = None
    lifestyle: str = None
