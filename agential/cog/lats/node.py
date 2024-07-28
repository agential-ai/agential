"""Node class for LATS."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List

import numpy as np

from agential.cog.react.output import ReActOutput


class BaseNode(ABC):
    """Abstract base class for nodes in a tree structure."""

    @abstractmethod
    def uct(self) -> float:
        """Calculate the Upper Confidence Bound for Trees (UCT) value."""
        pass

    @abstractmethod
    def add_children(self, children: List["BaseNode"]) -> None:
        """Add child nodes to the current node."""
        pass

    @abstractmethod
    def to_dict(self) -> Dict:
        """Convert the node to a dictionary representation."""
        pass


class Node(BaseNode):
    """Concrete implementation of a node in a tree structure.

    Attributes:
        state: The state of the node. Defaults to None.
        parent: The parent node. Defaults to None.
        children: List of child nodes. Defaults to None (empty list).
        visits: Number of times the node has been visited. Defaults to 0.
        value: The value of the node. Defaults to 0.
        depth: The depth of the node in the tree. Defaults to 0 or parent node's depth + 1.
        is_terminal: Whether the node is a terminal node. Defaults to False.
        reward: The reward associated with the node. Defaults to 0.
    """

    def __init__(
        self,
        state=None,
        parent=None,
        children=None,
        visits=0,
        value=0,
        depth=0,
        is_terminal=False,
        reward=0,
    ) -> None:
        """Initialization."""
        self.state = (
            ReActOutput(
                **{
                    "thought": "",
                    "action_type": "",
                    "query": "",
                    "observation": "",
                    "answer": "",
                    "external_tool_info": {},
                }
            )
            if state is None
            else state
        )
        self.parent = parent
        self.children = [] if children is None else children
        self.visits = visits
        self.value = value
        self.depth = (
            depth if parent is None else parent.depth + 1
        )
        self.is_terminal = is_terminal
        self.reward = reward

    def uct(self) -> float:
        """Calculate the Upper Confidence Bound for Trees (UCT) value.

        Returns:
            The UCT value of the node.
        """
        if self.visits == 0:
            return self.value
        return self.value / self.visits + np.sqrt(
            2 * np.log(self.parent.visits) / self.visits
        )

    def add_children(self, children) -> None:
        """Add child nodes to the current node.

        Args:
            children: List of child nodes to be added.
        """
        self.children.extend(children)

    def to_dict(self) -> Dict[str, Any]:
        """Convert the node to a dictionary representation.

        Returns:
            A dictionary representation of the node.
        """
        return {
            "state": self.state,
            "visits": self.visits,
            "value": self.value,
            "depth": self.depth,
            "is_terminal": self.is_terminal,
            "reward": self.reward,
        }
