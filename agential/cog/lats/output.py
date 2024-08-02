"""LATS structured output module."""

from typing import List, Dict, Any

from pydantic import BaseModel, Field


class LATSSimulationOutput(BaseModel):
    """LATS simulation Pydantic output class.

    Attributes:
        current_node (Dict[str, Any]): The current node.
        children_nodes (List[Dict[str, Any]]): The children nodes of the current node.
        values (List[float]): The values of the children nodes.
    """

    current_node: Dict[str, Any] = Field(..., description="The current node.")
    children_nodes: List[Dict[str, Any]] = Field(
        ...,
        description="The children nodes of the current node.",
    )
    values: List[float] = Field(
        ...,
        description="The values of the children nodes.",
    )


class LATSOutput(BaseModel):
    """LATS Pydantic output class.

    Attributes:
        iteration (int): The iteration number.
        current_node (Dict[str, Any]): The current node.
        children_nodes (List[Dict[str, Any]]): The children nodes of the current node.
        values (List[float]): The values of the children nodes.
        simulation_reward (float): The reward of the simulation from the current node's most valuable child node.
        simulation_terminal_node (Dict[str, Any]): The terminal node of the simulation.
        simulation_results (List[LATSSimulationOutput]): The results of the simulation.
    """

    iteration: int = Field(..., description="The iteration number.")
    current_node: Dict[str, Any] = Field(..., description="The current node.")
    children_nodes: List[Dict[str, Any]] = Field(
        ...,
        description="The children nodes of the current node.",
    )
    values: List[float] = Field(
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
