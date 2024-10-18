"""Structured output for CLIN."""

from pydantic import BaseModel, Field


class CLINOutput(BaseModel):
    """Structured output for CLIN."""
    answer: str = Field(description="The answer to the question.")