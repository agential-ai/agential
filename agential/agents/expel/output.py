"""ExpeL structured output module."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from agential.core.base.output import BaseOutput
from agential.llm.llm import Response


class ExpeLGenerateOutput(BaseModel):
    """ExpeL structured output for experiences.

    Attributes:
        examples (str): The examples to be included in the output.
        insights (str): Additional insights to be included in the output.
        experience (Dict[str, Any]): The current experience.
        experience_memory (Dict[str, Any]): The experience memory.
        insight_memory (Dict[str, Any]): The insight memory.
        compares_response (Optional[List[List[Response]]]): The insight memory comparison responses.
        successes_response (Optional[List[List[Response]]]): The insight memory successful responses.
    """

    examples: str = Field(..., description="The examples to be included in the output.")
    insights: str = Field(
        ..., description="Additional insights to be included in the output."
    )
    experience: Dict[str, Any] = Field(..., description="The current experience.")
    experience_memory: Dict[str, Any] = Field(..., description="The experience memory.")
    insight_memory: Dict[str, Any] = Field(..., description="The insight memory.")
    compares_response: Optional[List[List[Response]]] = Field(
        ..., description="The insight memory comparison responses."
    )
    successes_response: Optional[List[List[Response]]] = Field(
        ..., description="The insight memory successful responses."
    )


class ExpeLOutput(BaseOutput):
    """ExpeL Pydantic output class.

    Attributes:
        additional_info (ExpeLGenerateOutput): The ExpeL generation outputs.
    """

    additional_info: ExpeLGenerateOutput = Field(
        ..., description="The ExpeLGenerateOutput."
    )
