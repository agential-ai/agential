"""LATS structured output module."""

from typing import Any, Dict, List

from pydantic import BaseModel, Field

from agential.cog.base.output import BaseOutput
from agential.utils.general import PromptMetrics


class LATSReActStepOutput(BaseModel):
    """LATS ReAct Pydantic output class.

    Attributes:
        thought (str): The thought process of the agent.
        action_type (str): The type of action performed by the agent.
        query (str): The query requested by the agent.
        observation (str): The observation made by the agent.
        answer (str): The answer generated by the agent.
        external_tool_info (Dict[str, Any]): The external tool outputs.
        prompt_metrics (Dict[str, Any]): The prompt metrics including token usage, cost, and latency.
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
    thought_metrics: PromptMetrics = Field(
        ..., description="The thought metrics including token usage, cost, and latency."
    )
    action_metrics: PromptMetrics = Field(
        ..., description="The thought metrics including token usage, cost, and latency."
    )


class LATSSimulationOutput(BaseModel):
    """LATS simulation Pydantic output class.

    Attributes:
        current_node (Dict[str, Any]): The current node.
        children_nodes (List[Dict[str, Any]]): The children nodes of the current node.
        values (List[Dict[str, Any]]): The values of the children nodes.
    """

    current_node: Dict[str, Any] = Field(..., description="The current node.")
    children_nodes: List[Dict[str, Any]] = Field(
        ...,
        description="The children nodes of the current node.",
    )
    values: List[Dict[str, Any]] = Field(
        ...,
        description="The values of the children nodes.",
    )


class LATSStepOutput(BaseModel):
    """LATS Pydantic output class.

    Attributes:
        iteration (int): The iteration number.
        current_node (Dict[str, Any]): The current node.
        children_nodes (List[Dict[str, Any]]): The children nodes of the current node.
        values (List[Dict[str, Any]]): The values of the children nodes.
        simulation_reward (float): The reward of the simulation from the current node's most valuable child node.
        simulation_terminal_node (Dict[str, Any]): The terminal node of the simulation.
        simulation_results (List[LATSSimulationOutput]): The results of the simulation.
        prompt_metrics (Dict[str, Any]): The metrics of the prompt including token usage, cost, and latency.
    """

    iteration: int = Field(..., description="The iteration number.")
    current_node: Dict[str, Any] = Field(..., description="The current node.")
    children_nodes: List[Dict[str, Any]] = Field(
        ...,
        description="The children nodes of the current node.",
    )
    values: List[Dict[str, Any]] = Field(
        ...,
        description="The values of the children nodes.",
    )
    simulation_reward: float = Field(
        ...,
        description="The reward of the simulation from the current node's most valuable child node.",
    )
    simulation_terminal_node: Dict[str, Any] = Field(
        ...,
        description="The terminal node of the simulation.",
    )
    simulation_results: List[LATSSimulationOutput] = Field(
        ...,
        description="The results of the simulation.",
    )
    prompt_metrics: Dict[str, Any] = Field(
        ...,
        description="The metrics of the prompt.",
    )
