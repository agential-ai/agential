"""Scoring module for Generative Agents."""

from typing import Union, List

from langchain_core.language_models import LLM

from discussion_agents.cog.modules.scoring.base import BaseScoring

class GenerativeAgentScorer(BaseScoring):
    llm: LLM

    def score(
        self,
        memory_contents: Union[str, List[str]],
        relevant_memories: Union[str, List[str]],
    ) -> List[float]:
