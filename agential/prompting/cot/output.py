"""CoT structured output module."""

from typing import List

from pydantic import BaseModel, Field

from agential.core.base.prompting.output import BasePromptingOutput
from agential.llm.llm import Response


class CoTStepOutput(BaseModel):
    """CoT step Pydantic output class.

    Attributes:
        thought (str): The thought of the step.
        answer (str): The answer of the step.
        thought_response (Response): The llm response of the thought.
        answer_response (Response): The llm response of the answer.
    """

    thought: str = Field(..., description="The thought of the step.")
    answer: str = Field(..., description="The answer of the step.")
    thought_response: Response = Field(
        ..., description="The llm response of the thought."
    )
    answer_response: Response = Field(
        ..., description="The llm response of the answer."
    )


class CoTOutput(BasePromptingOutput):
    """CoT Pydantic output class.

    Attributes:
        answer (List[List[str]]): The list of list of answers.
        additional_info (List[List[CoTStepOutput]]): The list of list of llm responses information.
    """

    answer: List[List[str]] = Field(..., description="The list of list of answers.")
    additional_info: List[List[CoTStepOutput]] = Field(
        ..., description="The list of list of llm responses information."
    )
