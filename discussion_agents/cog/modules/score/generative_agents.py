"""Scoring module for Generative Agents."""

from typing import Any, List, Optional, Union

from discussion_agents.cog.functional.generative_agents import (
    score_memories_importance,
)
from discussion_agents.cog.modules.score.base import BaseScorer


class GenerativeAgentScorer(BaseScorer):
    """A scoring class for Generative Agents, specialized in assessing memory content relevance.

    This class extends the BaseScorer and is tailored for scoring the importance or relevance of memory
    contents in the context of a generative agent. It uses a language model (LLM) to evaluate the memory contents
    against a set of relevant memories. The scoring is based on the degree of relevance or importance
    of the memory contents, quantified as numerical scores.

    Attributes:
        llm (LLM): An instance of a language model used for scoring the memories.
            This model plays a key role in determining the relevance of memory contents.

    The class primarily provides a `score` method that takes memory contents and relevant memories as
    inputs and returns a list of importance scores for each memory content.
    """

    def __init__(self, llm: Any) -> None:
        """Initialization."""
        super().__init__(llm)

    def score(
        self,
        memory_contents: Union[str, List[str]],
        relevant_memories: Union[str, List[str]],
        importance_weight: Optional[float] = 0.15,
    ) -> List[float]:
        """Scores the importance of memory contents based on their relevance to a set of given memories.

        This method utilizes a language model (LLM) to evaluate the importance of each memory content in relation to the relevant memories.
        The importance is calculated as a float score for each memory content, indicating its relevance.

        Args:
            memory_contents (Union[str, List[str]]): A single memory or a list of memory contents to be scored.
                Each memory content is a string representing an individual memory.
            relevant_memories (Union[str, List[str]]): A single relevant memory or a list of relevant memories that the memory contents
                are being compared to. Each relevant memory is a string.
            importance_weight (float): A weight factor (default: 0.15) used in the scoring calculation
                to adjust the influence of certain criteria in the final score.

        Returns:
            List[float]: A list of float scores corresponding to the importance of each memory content in relation to the relevant memories.
                The scores are calculated using the language model and the importance weight attribute of the class.

        The method delegates the scoring to the `score_memories_importance` function,
        passing the language model, importance weight, and the provided memory contents and relevant memories as arguments.
        """
        return score_memories_importance(
            memory_contents=memory_contents,
            relevant_memories=relevant_memories,
            llm=self.llm,
            importance_weight=importance_weight,
        )

    def clear(self) -> None:
        """Clears any internal state."""
        pass