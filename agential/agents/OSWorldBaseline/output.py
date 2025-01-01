"""OSWorldBaseline structured output module."""

from typing import Any, Dict

from pydantic import BaseModel, Field

from agential.agents.base.output import BaseAgentOutput


class OSWorldBaseOutput(BaseAgentOutput):
    """OSWorldBaseOutput structured output class.

    Attributes:
        additional_info Dict[str, Any]: A dictionary of observations, thoughts, and actions of OSWorldBaselineAgent Output.
    """

    additional_info: Dict[str, Any] = Field(
        ...,
        description="A dictionary of observations, thoughts, and actions of OSWorldBaselineAgent Output.",
    )
