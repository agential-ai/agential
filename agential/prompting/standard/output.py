"""Standard prompting output module."""

from typing import List

from pydantic import BaseModel, Field

from agential.core.base.prompting.output import BasePromptingOutput
from agential.llm.llm import Response


class StandardStepOutput(BaseModel):
    """Standard step Pydantic output class.

    Attributes:
        answer (str): The answer of the step.
        answer_response (Response): The llm response of the answer.
    """

    answer: str = Field(..., description="The answer of the step.")
    answer_response: Response = Field(
        ..., description="The llm response of the answer."
    )


class StandardOutput(BasePromptingOutput):
    """Standard Pydantic output class.

    Attributes:
        answer (List[List[str]]): The list of list of answers.
        additional_info (List[List[StandardStepOutput]]): The list of list of llm responses information.
    """

    answer: List[List[str]] = Field(..., description="The list of list of answers.")
    additional_info: List[List[StandardStepOutput]] = Field(
        ..., description="The list of list of llm responses information."
    )
