"""Base ExpeL Agent strategy class."""

from abc import abstractmethod
from typing import Any, Dict, Tuple, Optional

from langchain_core.language_models.chat_models import BaseChatModel

from agential.base.strategies import BaseStrategy
from agential.cog.expel.memory import (
    ExpeLExperienceMemory,
    ExpeLInsightMemory,
)
from agential.cog.reflexion.agent import ReflexionReActAgent


class ExpeLBaseStrategy(BaseStrategy):
    """An abstract base class for defining strategies for the ExpeL Agent.

    Attributes:
        llm (BaseChatModel): The language model used for generating answers and critiques.
        reflexion_react_agent (ReflexionReActAgent): The ReflexionReAct agent.
        experience_memory (ExpeLExperienceMemory): Memory module for storing experiences.
        insight_memory (ExpeLInsightMemory): Memory module for storing insights derived from experiences.
        success_batch_size (int): Batch size for processing success experiences in generating insights.
    """

    def __init__(
        self, 
        llm: BaseChatModel,
        reflexion_react_agent: ReflexionReActAgent,
        experience_memory: ExpeLExperienceMemory,
        insight_memory: ExpeLInsightMemory,
        success_batch_size: int,
    ) -> None:
        """Initialization."""
        super().__init__(llm)
        self.reflexion_react_agent = reflexion_react_agent
        self.success_batch_size = success_batch_size
        self.insight_memory = insight_memory
        self.experience_memory = experience_memory

    @abstractmethod
    def get_dynamic_examples(
        self,
        question: str,
        examples: str,
        k_docs: int,
        num_fewshots: int,
        max_fewshot_tokens: int,
        reranker_strategy: str,
        additional_keys: Dict[str, Any]
    ) -> Tuple[str, Dict[str, str]]:
        pass

    @abstractmethod
    def gather_experience(self):
        pass

    @abstractmethod
    def extract_insights(self):
        pass

    @abstractmethod
    def update_insights(self):
        pass
