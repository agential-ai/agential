"""ExpeL structured output module."""

from typing import Any, Dict

from pydantic import BaseModel, Field


class ExpeLOutput(BaseModel):
    """ExpeL structured output for experiences.

    Attributes:
        examples (str): The examples to be included in the output.
        insights (str): Additional insights to be included in the output.
        experience (Dict[str, Any]): The current experience.
        experience_memory (Dict[str, Any]): The experience memory.
        insight_memory (Dict[str, Any]): The insight memory.
        prompt_metrics (Dict[str, Any]): The prompt metrics.
    """

    examples: str = Field(..., description="The examples to be included in the output.")
    insights: str = Field(
        "", description="Additional insights to be included in the output."
    )
    experience: Dict[str, Any] = Field(..., description="The current experience.")
    experience_memory: Dict[str, Any] = Field(..., description="The experience memory.")
    insight_memory: Dict[str, Any] = Field(..., description="The insight memory.")
    prompt_metrics: Dict[str, Any] = Field(
        ..., description="The prompt metrics."
    )