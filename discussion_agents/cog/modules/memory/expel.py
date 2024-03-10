"""ExpeL's memory implementations.

Original Paper: https://arxiv.org/abs/2308.10144
Paper Repository: https://github.com/LeapLabTHU/ExpeL
"""

from typing import Any, Dict, List, Tuple

from langchain.vectorstores import FAISS

from discussion_agents.cog.modules.memory.base import BaseMemory


class ExpeLExperienceMemory(BaseMemory):
    def __init__(
        self,
        question: str,
        key: str,
        trajectories: List[Tuple[bool, bool, List[Tuple[str, str, str]]]],
    ) -> None:
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
