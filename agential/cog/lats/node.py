import numpy as np
from abc import ABC, abstractmethod

class BaseNode(ABC):
    @abstractmethod
    def uct(self):
        pass

    @abstractmethod
    def add_children(self, children):
        pass

    @abstractmethod
    def to_dict(self):
        pass
    

class Node(BaseNode):
    def __init__(
        self,
        state=None,
        parent=None,
        children=None,
        visits=0,
        value=0,
        depth=None,
        is_terminal=False,
        reward=0,
    ):
        self.state = (
            {"thought": "", "action": "", "observation": ""} if state is None else state
        )
        self.parent = parent
        self.children = [] if children is None else children
        self.visits = visits
        self.value = value
        self.depth = (
            0 if parent is None else parent.depth + 1 if depth is None else depth
        )
        self.is_terminal = is_terminal
        self.reward = reward

    def uct(self):
        if self.visits == 0:
            return self.value
        return self.value / self.visits + np.sqrt(
            2 * np.log(self.parent.visits) / self.visits
        )

    def add_children(self, children):
        self.children.extend(children)

    def to_dict(self):
        return {
            "state": self.state,
            "parent": self.parent.to_dict() if self.parent else None,
            "children": [child.to_dict() for child in self.children],
            "visits": self.visits,
            "value": self.value,
            "depth": self.depth,
            "is_terminal": self.is_terminal,
            "reward": self.reward,
        }