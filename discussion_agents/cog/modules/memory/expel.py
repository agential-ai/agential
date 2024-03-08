"""ExpeL's memory implementations.

Original Paper: https://arxiv.org/abs/2308.10144
Paper Repository: https://github.com/LeapLabTHU/ExpeL
"""

from typing import Dict, Any
from discussion_agents.cog.modules.memory.base import BaseMemory
from langchain.vectorstores import FAISS


class ExpeLExperienceMemory(BaseMemory):
    def __init__(self) -> None:
        """Initialization."""
        super().__init__()

    def clear(self) -> None:
        pass

    def add_memories(self) -> None:
        pass

    def load_memories(self) -> Dict[str, Any]:
        pass

    def show_memories(self) -> Dict[str, Any]:
        pass