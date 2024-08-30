"""CoT structured output module."""

from pydantic import BaseModel, Field

from agential.core.base.output import BaseOutput
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


class CoTOutput(BaseOutput):
    """CoT Pydantic output class.

    Attributes:
        additional_info (CoTStepOutput): The llm response information.
    """

    additional_info: CoTStepOutput = Field(
        ..., description="The llm response information."
    )
