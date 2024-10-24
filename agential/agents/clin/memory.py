"""CLIN memory class."""

from typing import Any, Dict

from agential.agents.base.modules.memory import BaseMemory


class CLINMemory(BaseMemory):
    """CLIN Memory implementation."""

    def __init__(self, k: int = 10) -> None:
        """Initialize."""
        super().__init__()
        self.k = k
        self.memories = []

    def clear(self) -> None:
        """Clear all memories."""
        self.memories = []

    def add_memories(
        self,
        question: str,
        summary: str,
        eval_report: str,
        is_correct: bool,
    ) -> None:
        """Add summaries to the CLIN Memory.

        Args:
            question (str): The question asked.
            summary (str): The summary of the question.
            eval_report (str): The evaluation report of the question.
            is_correct (bool): Whether the question was answered correctly.

        Returns:
            None
        """
        self.memories.append(
            {
                "question": question,
                "summary": summary,
                "trial": f"Question: {question}\n{summary}\nEVALUATION REPORT: {eval_report}",
                "is_correct": is_correct,
            }
        )

    def load_memories(self) -> Dict[str, Any]:
        """Load all memories and return as a dictionary.

        Returns:
            Dict[str, Any]: A dictionary containing all stored memories.
        """
        previous_successful_trials = [trial for trial in self.memories if trial["is_correct"]]
        previous_successful_k_trials = previous_successful_trials[-self.k:]
        return {"previous_successful_k_trials": previous_successful_k_trials}

    def show_memories(self) -> Dict[str, Any]:
        """Show all memories.

        Returns:
            Dict[str, Any]: A dictionary containing all stored memories.
        """
        return {"previous_trials": self.memories}

    