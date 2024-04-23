"""Self-Refine Agent.

Original Webpage: https://selfrefine.info/
Paper Repository: https://github.com/madaan/self-refine
"""

from typing import Any, Dict, List, Optional

from agential.cog.modules.memory.base import BaseMemory


class SelfRefineMemory(BaseMemory):
    """A class to store, retrieve, and manage solution and feedback memories in a self-refinement context.

    Attributes:
        solution (Optional[List[str]]): A list to store solution memories.
        feedback (Optional[List[str]]): A list to store feedback memories.
    """

    def __init__(
        self, solution: Optional[List] = None, feedback: Optional[List] = None
    ) -> None:
        """Initialization."""
        super().__init__()

        self.solution = solution if solution else []
        self.feedback = feedback if feedback else []

    def clear(self) -> None:
        """Clears both the solution and feedback memories, resetting them to empty lists."""
        self.solution = []
        self.feedback = []

    def add_memories(self, solution: str, feedback: str) -> None:
        """Adds a new pair of solution and feedback to the respective memories.

        Args:
            solution (str): The solution to add to the solution memories.
            feedback (str): The feedback to add to the feedback memories.
        """
        self.solution.append(solution)
        self.feedback.append(feedback)

    def load_memories(
        self, solution_key: str = "solution", feedback_key: str = "feedback"
    ) -> Dict[str, Any]:
        """Loads and returns the stored solution and feedback memories as a dictionary.

        Args:
            solution_key (str, optional): The key name for the solution memories in the returned dictionary. Defaults to "solution".
            feedback_key (str, optional): The key name for the feedback memories in the returned dictionary. Defaults to "feedback".

        Returns:
            Dict[str, Any]: A dictionary containing the solution and feedback memories.
        """
        return {solution_key: self.solution, feedback_key: self.feedback}

    def show_memories(
        self, solution_key: str = "solution", feedback_key: str = "feedback"
    ) -> Dict[str, Any]:
        """A convenience method that behaves similarly to `load_memories`, designed for displaying the stored memories.

        Args:
            solution_key (str, optional): The key name for the solution memories in the returned dictionary. Defaults to "solution".
            feedback_key (str, optional): The key name for the feedback memories in the returned dictionary. Defaults to "feedback".

        Returns:
            Dict[str, Any]: A dictionary containing the solution and feedback memories.
        """
        return {solution_key: self.solution, feedback_key: self.feedback}
