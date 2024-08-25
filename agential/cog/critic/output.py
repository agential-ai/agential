"""CRITIC structured output module."""

from typing import Any, Dict, List

from pydantic import BaseModel, Field

from agential.cog.base.output import BaseOutput


class CriticStepOutput(BaseModel):
    """Critic step Pydantic output class.

    Attributes:
        answer (str): The answer generated by the agent.
        critique (str): The critique of the answer generated by the agent.
        external_tool_info (Dict[str, Any]): The query requested by the agent.
    """

    answer: str = Field(..., description="The answer generated by the agent.")
    critique: str = Field(..., description="The answer's critique.")
    external_tool_info: Dict[str, Any] = Field(
        ..., description="The external tool outputs."
    )


class CriticOutput(BaseOutput):
    """Critic Pydantic output class.

    Attributes:
        additional_info (List[CriticStepOutput]): The additional info.
    """

    additional_info: List[CriticStepOutput] = Field(
        ..., description="The additional info."
    )