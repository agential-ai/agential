"""LATS structured output module."""

from typing import List

from pydantic import BaseModel, Field
from agential.cog.lats.node import Node


class LATSSimulationOutput(BaseModel):
    """LATS simulation Pydantic output class.
    
    Attributes:

    """
    

class LATSStepOutput(BaseModel):
    """LATS step Pydantic output class.

    Attributes:

    """

    current_node: Node = Field(..., description="The current node.")
    children_nodes: List[Node] = Field(
        ...,
        description="The children nodes of the current node.",
    )
    values: List[float] = Field(
        ...,
        description="The values of the children nodes.",
    )
