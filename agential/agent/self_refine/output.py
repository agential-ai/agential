"""Self-Refine structured output module."""

from typing import List

from pydantic import BaseModel, Field

from agential.core.base.output import BaseOutput
from agential.llm.llm import Response


class SelfRefineStepOutput(BaseModel):
    """Self-Refine Pydantic output class.

    Attributes:
        answer (str): The answer generated by the agent.
        critique (str): The critique of the answer generated by the agent.
        answer_response (Response): The response of the answer generated by the agent.
        critique_response (Response): The response of the critique generated by the agent.
    """

    answer: str = Field(..., description="The answer generated by the agent.")
    critique: str = Field(..., description="The answer's critique.")
    answer_response: Response = Field(..., description="The answer's response.")
    critique_response: Response = Field(..., description="The critique's response.")


class SelfRefineOutput(BaseOutput):
    """Self-Refine Pydantic output class.

    Attributes:
        additional_info (List[SelfRefineStepOutput]): Additional information about the steps.
    """

    additional_info: List[SelfRefineStepOutput] = Field(
        ..., description="Additional information about the steps."
    )