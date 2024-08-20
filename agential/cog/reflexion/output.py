"""Reflexion structured output module."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from agential.cog.base.output import BaseOutput
from agential.utils.metrics import PromptMetrics


class ReflexionCoTStepOutput(BaseModel):
    """ReflexionCoT step Pydantic output class.

    Attributes:
        thought (str): The thought process of the agent.
        action_type (str): The type of action performed by the agent.
        observation (str): The observation made by the agent.
        answer (str): The answer generated by the agent.
        is_correct (bool): Indicates if the action was correct.
        reflections (List[str]): Additional reflections on the action.
        thought_metrics (PromptMetrics): Thought metrics.
        action_metrics (PromptMetrics): Action metrics.
        reflection_metrics (Optional[PromptMetrics]): Reflection metrics.
    """

    thought: str = Field(..., description="The thought process of the agent.")
    action_type: str = Field(
        ..., description="The type of action performed by the agent."
    )
    observation: str = Field(..., description="The observation made by the agent.")
    answer: str = Field(..., description="The answer generated by the agent.")
    is_correct: bool = Field(..., description="Indicates if the action was correct.")
    reflections: List[str] = Field(
        ..., description="Additional reflections on the action."
    )
    thought_metrics: PromptMetrics = Field(..., description="Thought metrics.")
    action_metrics: PromptMetrics = Field(..., description="Action metrics.")
    reflection_metrics: Optional[PromptMetrics] = Field(
        ..., description="Reflection metrics."
    )


class ReflexionCoTOutput(BaseOutput):
    """ReflexionCoT Pydantic output class.

    Attributes:
        additional_info (List[ReflexionCoTStepOutput]): The list of ReflexionCoT step outputs.
    """

    additional_info: List[ReflexionCoTStepOutput] = Field(
        ..., description="The list of ReflexionCoTStepOutput."
    )


class ReflexionReActReActStepOutput(BaseModel):
    """ReflexionReAct ReAct Step Pydantic output class.

    Attributes:
        thought (str): The thought process of the agent.
        action_type (str): The type of action performed by the agent.
        query (str): The query requested by the agent.
        observation (str): The observation made by the agent.
        answer (str): The answer generated by the agent.
        external_tool_info (Dict[str, Any]): The external tool outputs.
        is_correct (bool): Indicates if the action was correct.
        thought_metrics (PromptMetrics): Thought metrics.
        action_metrics (PromptMetrics): Action metrics.
    """

    thought: str = Field(..., description="The thought process of the agent.")
    action_type: str = Field(
        ..., description="The type of action performed by the agent."
    )
    query: str = Field(..., description="The query requested by the agent.")
    observation: str = Field(..., description="The observation made by the agent.")
    answer: str = Field(..., description="The answer generated by the agent.")
    external_tool_info: Dict[str, Any] = Field(
        ..., description="The external tool outputs."
    )
    is_correct: bool = Field(..., description="Indicates if the action was correct.")
    thought_metrics: PromptMetrics = Field(
        ..., description="Prompt metrics for the thought."
    )
    action_metrics: PromptMetrics = Field(
        ..., description="Prompt metrics for the thought."
    )


class ReflexionReActStepOutput(BaseModel):
    """ReflexionReAct Pydantic output class.

    Attributes:
        steps (List[ReflexionReActStepOutput]): The output of each step of the ReflexionReAct agent.
        reflections (List[str]): The reflections generated by the ReflexionReAct agent.
        reflection_metrics (Optional[PromptMetrics]): Prompt metrics for reflection.
    """

    steps: List[ReflexionReActReActStepOutput] = Field(
        ..., description="The output of each step of the ReflexionReAct agent."
    )
    reflections: List[str] = Field(
        ..., description="The reflections generated by the ReflexionReAct agent."
    )
    reflection_metrics: Optional[PromptMetrics] = Field(
        ..., description="Prompt metrics for reflection."
    )


class ReflexionReActOutput(BaseOutput):
    """ReflexionReAct Pydantic output class.

    Attributes:
        additional_info (List[ReflexionReActStepOutput]): The list of ReflexionReAct step outputs.
    """

    additional_info: List[ReflexionReActStepOutput] = Field(
        ..., description="The list of ReflexionReActStepOutput."
    )