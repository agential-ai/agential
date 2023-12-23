"""Scoring module for Generative Agents."""

from typing import List, Union

from langchain_core.language_models import LLM

from discussion_agents.cog.functional.generative_agents import (
    score_memories_importance,
)
from discussion_agents.cog.modules.score.base import BaseScorer


class GenerativeAgentScorer(BaseScorer):
    llm: LLM
    importance_weight: float = 0.15

    def score(
        self,
        memory_contents: Union[str, List[str]],
        relevant_memories: Union[str, List[str]],
    ) -> List[float]:
        return score_memories_importance(
            memory_contents=memory_contents,
            relevant_memories=relevant_memories,
            llm=self.llm,
            importance_weight=self.importance_weight,
        )
