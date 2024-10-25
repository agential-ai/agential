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

    def clear(self) -> None:
        """Clear all memories."""
        self.memories = {}
        self.meta_summaries = {}

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
        if question not in self.memories: 
            self.memories[question] = []

        self.memories[question].append(
            {
                "summary": summary,
                "trial": f"Question: {question}\n{summary}\nEVALUATION REPORT: {eval_report}",
                "is_correct": is_correct,
            }
        )

    def add_meta_summary(self, question: str, meta_summary: str) -> None:
        """Add a meta-summary to the CLIN Memory.

        Args:
            question (str): The question asked.
            meta_summary (str): The meta-summary of the question.

        Returns:
            None
        """
        if question not in self.meta_summaries:
            self.meta_summaries[question] = []

        self.meta_summaries[question].append(meta_summary)
            
    def load_memories(self, question: str, load_meta_summary: bool) -> Dict[str, Any]:
        """Load all memories and return as a dictionary.

        Args:
            question (str): The question asked.
            load_meta_summary (bool): Whether to load the meta-summary.

        Returns:
            Dict[str, Any]: A dictionary containing all stored memories.
        """
        if question not in self.memories:
            return {"previous_trials": "", "meta_summaries": ""}
        
        previous_trials = "\n\n---\n\n".join([trial['trial'] for trial in self.memories[question]])
        
        latest_meta_summaries = ""
        if load_meta_summary and question in self.meta_summaries:
            latest_meta_summaries = self.meta_summaries[question][-1]

        latest_summaries = ""
        if question in self.memories:
            latest_summaries = self.memories[question][-1]['summary']

        return {
            "previous_trials": previous_trials, 
            "latest_meta_summaries": latest_meta_summaries, 
            "latest_summaries": latest_summaries
        }

    def show_memories(self) -> Dict[str, Any]:
        """Show all memories.

        Returns:
            Dict[str, Any]: A dictionary containing all stored memories.
        """
        return {"previous_trials": self.memories}

    