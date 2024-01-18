"""ReAct's memory implementation.

Original Paper: https://arxiv.org/abs/2210.03629
Paper Repository: https://github.com/ysymyth/ReAct
LangChain: https://github.com/langchain-ai/langchain
LangChain ReAct: https://python.langchain.com/docs/modules/agents/agent_types/react
"""
from typing import Any, Dict, Optional

from discussion_agents.cog.modules.memory.base import BaseMemory


class ReActMemory(BaseMemory):
    """A memory storage class for ReAct.

    It stores, retrieves, and manages text-based memories (observations) in a scratchpad (str).

    Attributes:
        scratchpad (str): A string attribute that stores all the memories.
    """

    def __init__(self, scratchpad: Optional[str] = None) -> None:
        super().__init__()
        self.scratchpad = scratchpad if scratchpad else ""

    def clear(
        self,
    ) -> None:
        """Clears the contents of the scratchpad.

        This method resets the scratchpad to an empty string, erasing all stored memories.
        """
        self.scratchpad = ""

    def add_memories(self, observation: str) -> None:
        """Adds a new observation to the scratchpad.

        This method appends the given observation text to the existing contents of the scratchpad.

        Args:
            observation (str): The observation text to be added to the memory.

        """
        self.scratchpad += observation

    def load_memories(self, input_key: str = "scratchpad") -> Dict[str, Any]:
        """Retrieves all stored memories.

        `show_memories` and `load_memories` are identical in ReAct's case.

        Args:
            input_key (str, optional): The key used to access memories. Defaults to "scratchpad".

        Returns:
            Dict[str, Any]: A dictionary containing the stored memories, accessible via the provided input key.
        """
        return {input_key: self.scratchpad}

    def show_memories(self, input_key: str = "scratchpad") -> Dict[str, Any]:
        """Displays all stored memories.

        Args:
            input_key (str, optional): The key used to access memories. Defaults to "scratchpad".

        Returns:
            Dict[str, Any]: A dictionary containing the stored memories, accessible via the provided input key.
        """
        return {input_key: self.scratchpad}
