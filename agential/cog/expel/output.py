"""ExpeL structured output module."""

from typing import List

from pydantic import BaseModel, Field

from agential.cog.reflexion.output import ReflexionReActOutput


class ExpeLExperienceOutput(BaseModel):
    """ExpeL structured output for experiences.

    Attributes:
        question (str): The question.
        key (str): The key.
        trajectory (List[ReflexionReActOutput]): The ReflexionReAct trajectory.
        reflections (List[str]): The reflections generated by the agent.
    """

    question: str = Field(..., description="The question.")
    key: str = Field(..., description="The key.")
    trajectory: List[ReflexionReActOutput] = Field(
        ..., description="The ReflexionReAct trajectory."
    )
    reflections: List[str] = Field(
        ..., description="The reflections generated by the agent."
    )
