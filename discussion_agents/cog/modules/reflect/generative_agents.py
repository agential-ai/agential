"""Reflecting module for Generative Agents."""
from datetime import datetime
from typing import List, Optional, Union

from langchain_core.language_models import LLM
from langchain_core.retrievers import BaseRetriever

from discussion_agents.cog.functional.generative_agents import (
    reflect,
)
from discussion_agents.cog.modules.reflect.base import BaseReflector


class GenerativeAgentReflector(BaseReflector):
    llm: LLM
    retriever: BaseRetriever

    def reflect(
        self,
        observations: Union[str, List[str]],
        now: Optional[datetime] = None,
    ) -> List[str]:
        _, insights = reflect(
            observations=observations, llm=self.llm, retriever=self.retriever, now=now
        )
        return insights
