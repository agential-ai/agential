"""CLIN memory class."""

from typing import Any, Dict, List

from agential.agents.base.modules.memory import BaseMemory


class CLINMemory(BaseMemory):
    """CLIN Memory implementation."""

    def __init__(self, k: int = 10) -> None:
        """Initialize."""
        super().__init__()
        self.k = k
        self.memories: Dict[str, List[Dict[str, str]]] = {}
        self.meta_summaries: Dict[str, List[str]] = {}
        self.history: List[str] = []

    def clear(self) -> None:
        """Clear all memories."""
        self.memories = {}
        self.meta_summaries = {}
        self.history = []

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
        """
        if question not in self.memories:
            self.memories[question] = []

        self.memories[question].append(
            {
                "summary": summary,
                "trial": f"Question: {question}\n{summary}\nEVALUATION REPORT: {eval_report}",
                "is_correct": is_correct,
            }
        )

    def add_meta_summaries(self, question: str, meta_summary: str) -> None:
        """Add meta-summaries to the CLINMemory.

        Args:
            question (str): The question asked.
            meta_summary (str): The meta-summaries.

        Returns:
            None
        """
        if question not in self.meta_summaries:
            self.meta_summaries[question] = []

        self.meta_summaries[question].append(meta_summary)
        self.history.append(question)

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

        latest_summaries = self.memories[question][-1]["summary"]

        return {
            "previous_trials": previous_trials,
            "latest_summaries": latest_summaries,
        }

    def load_meta_summaries(self) -> Dict[str, Any]:
        """Load all meta-summaries and return as a dictionary.

        Returns:
            Dict[str, Any]: A dictionary containing all stored meta-summaries.
        """
        latest_meta_summaries = "\n\n---\n\n".join(
            [
                f"Question: {question}\n{self.meta_summaries[question][-1]}"
                for question in self.history[-self.k :]
            ]
        )

        return {"meta_summaries": latest_meta_summaries}

    def show_memories(self) -> Dict[str, Any]:
        """Show all memories.

        Returns:
            Dict[str, Any]: A dictionary containing all stored memories.
        """
        return {"previous_trials": self.memories}
