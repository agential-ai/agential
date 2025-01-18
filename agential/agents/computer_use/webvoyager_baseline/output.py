"""WebVoyagerBaseline structured output module."""

from typing import Any, Dict

from pydantic import Field

from agential.agents.base.output import BaseAgentOutput


class WebVoyagerBaseOutput(BaseAgentOutput):
    """WebVoyagerBaseOutput structured output class.

    Attributes:
        additional_info Dict[str, Any]: A dictionary of observations, thoughts, and actions of WebVoyagerBaselineAgent Output.
    """

    additional_info: Dict[str, Any] = Field(
        ...,
        description="A dictionary of observations, thoughts, and actions of WebVoyagerBaselineAgent Output.",
    )
