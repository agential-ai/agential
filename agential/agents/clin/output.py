"""Structured output for CLIN."""

from pydantic import BaseModel, Field
from agential.agents.base.output import BaseAgentOutput


class CLINReActStepOutput(BaseModel):
    pass

class CLINStepOutput(BaseModel):
    pass

class CLINOutput(BaseAgentOutput):
    """Structured output for CLIN."""
    additional_info: str = Field(description="The answer to the question.")