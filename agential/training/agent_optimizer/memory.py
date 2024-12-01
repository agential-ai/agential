"""CLIN memory class."""

from copy import deepcopy
from typing import Any, Dict, List

from agential.agents.base.modules.memory import BaseMemory


class AgentOptimizerMemory(BaseMemory):
    """AgentOptimizer Memory implementation.

    Attributes:
        memories (Dict[str, List[Dict[str, Any]]]): A dictionary of memories.
        meta_summaries (Dict[str, List[str]]): A dictionary of meta summaries.
        history (List[str]): A list of history.
        k (int): The number of memories to store.
    """

    def __init__(
        self,
        functions_list: List[str] = {},
        history: List[str] = [],
        k: int = 10,
    ) -> None:
        """Initialize."""
        super().__init__()
        self.functions_list = deepcopy(functions_list)
        self.history = deepcopy(history)
        self.k = k

    def clear(self) -> None:
        """Clear all memories."""
        self.functions_list = []
        self.history = []

    def add_function(
        self,
        name: str,
        description: str,
        arguments: Dict[str, Any],
        packages: str,
        code: str,
    ) -> None:
        """Add a function to the Agent Optimizer function list.

        Args:

        name (str): The name of the function.
        description (str): The description of the function.
        arguments (Dict[str, Any]): The arguments of the function.
        packages (str): The packages used in the function.
        code (str): The code of the function.
        """

        func_info = {
            "name": name,
            "description": description,
            "arguments": arguments,
            "packages": packages,
            "code": code,
        }

        self.functions_list.append(func_info)

    def revise_function(
        self,
        name: str,
        description: str,
        arguments: Dict[str, Any],
        packages: str,
        code: str,
    ) -> None:
        """Revises a function in the Agent Optimizer function list.

        Args:

        name (str): The name of the function.
        description (str): The description of the function.
        arguments (Dict[str, Any]): The arguments of the function.
        packages (str): The packages used in the function.
        code (str): The code of the function.
        """

        for func_info in self.functions_list:
            if func_info["name"] == name:
                func_info["description"] = description
                func_info["arguments"] = arguments
                func_info["packages"] = packages
                func_info["code"] = code

                break

    def remove_function(
        self,
        name: str,
    ) -> None:
        """Removes a function from the Agent Optimizer function list.

        Args:
        name (str): The name of the function.
        """

        for func_info in self.functions_list:
            if func_info["name"] == name:
                self.functions_list.remove(func_info)

    def load_memories(self, question: str) -> Dict[str, Any]:
        """Load all memories and return as a dictionary.

        Args:
            question (str): The question asked.

        Returns:
            Dict[str, Any]: A dictionary containing all stored memories.
        """
        if question not in self.memories:
            return {"previous_trials": "", "latest_summaries": ""}

        previous_trials = "\n\n---\n\n".join(
            [trial["trial"] for trial in self.memories[question]]
        )

        latest_summaries = self.memories[question][-1]["summaries"]

        return {
            "previous_trials": previous_trials,
            "latest_summaries": latest_summaries,
        }

    # load funcs

    def show_memories(self) -> Dict[str, Any]:
        """Show all memories.

        Returns:
            Dict[str, Any]: A dictionary containing all stored memories.
        """
        return {
            "summaries": self.memories,
            "meta_summaries": self.meta_summaries,
            "history": self.history,
        }
