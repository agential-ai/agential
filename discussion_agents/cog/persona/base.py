"""Base Persona interface."""

from abc import ABC
from typing import Optional

from pydantic.v1 import BaseModel


class BasePersona(BaseModel, ABC):
    """Base persona class."""

    name: Optional[str] = None
    age: Optional[int] = None
    traits: Optional[str] = None
    status: Optional[str] = None
    lifestyle: Optional[str] = None
