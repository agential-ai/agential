"""ExpeL structured output module."""

from typing import Any, Dict, List

from pydantic import BaseModel, Field

from agential.llm.llm import Response


class ExpeLOutput(BaseModel):
    """ExpeL structured output for experiences.

    Attributes:
        examples (str): The examples to be included in the output.
        insights (str): Additional insights to be included in the output.
        experience (Dict[str, Any]): The current experience.
        experience_memory (Dict[str, Any]): The experience memory.
        insight_memory (Dict[str, Any]): The insight memory.
        experience_memory_response (List[Response]): The experience memory responses.
        insight_memory_response (List[Response]): The insight memory responses.
    """

    examples: str = Field(..., description="The examples to be included in the output.")
    insights: str = Field(
        ..., description="Additional insights to be included in the output."
    )
    experience: Dict[str, Any] = Field(..., description="The current experience.")
    experience_memory: Dict[str, Any] = Field(..., description="The experience memory.")
    insight_memory: Dict[str, Any] = Field(..., description="The insight memory.")
    experience_memory_response: List[Response] = Field(
        ..., description="The experience memory responses."
    )
    insight_memory_response: List[Response] = Field(
        ..., description="The insight memory responses."
    )
