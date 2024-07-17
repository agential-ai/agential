"""Base ExpeL Agent strategy class."""

from abc import abstractmethod
from typing import Any, Dict, List, Tuple, Union

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
        additional_keys: Dict[str, Any],
    ) -> Tuple[str, Dict[str, str]]:
        """Generates dynamic examples for a given question.
        
        Args:
            question (str): The question to generate examples for.
            examples (str): The examples to use for generating dynamic examples.
            k_docs (int): The number of documents to retrieve for generating the examples.
            num_fewshots (int): The number of few-shot examples to generate.
            max_fewshot_tokens (int): The maximum number of tokens for the few-shot examples.
            reranker_strategy (str): The strategy to use for reranking the generated examples.
            additional_keys (Dict[str, Any]): Additional keys to associate with the generated examples.
        
        Returns:
            Tuple[str, Dict[str, str]]: The generated examples and a dictionary of additional keys.
        """
        pass

    @abstractmethod
    def gather_experience(
        self,
        questions: List[str],
        keys: List[str],
        examples: str,
        prompt: str,
        reflect_examples: str,
        reflect_prompt: str,
        reflect_strategy: str,
        additional_keys: Union[List[Dict[str, str]], Dict[str, str]],
        reflect_additional_keys: Union[List[Dict[str, str]], Dict[str, str]],
        patience: int,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Gathers experience by executing a series of steps.
        
        Args:
            questions (List[str]): A list of questions to gather experiences for.
            keys (List[str]): A list of keys to associate with the gathered experiences.
            examples (str): The examples to use for generating dynamic examples.
            prompt (str): The prompt to use for generating dynamic examples.
            reflect_examples (str): The examples to use for the reflection strategy.
            reflect_prompt (str): The prompt to use for the reflection strategy.
            reflect_strategy (str): The strategy to use for the reflection process.
            additional_keys (Union[List[Dict[str, str]], Dict[str, str]]): Additional keys to associate with the gathered experiences.
            reflect_additional_keys (Union[List[Dict[str, str]], Dict[str, str]]): Additional keys to associate with the insights generated from the reflection process.
            patience (int): The number of attempts to make before giving up on gathering an experience.
            **kwargs (Any): Additional keyword arguments to pass to the underlying methods.
        
        Returns:
            Dict[str, Any]: A dictionary containing the gathered experiences and insights.
        """
        pass

    @abstractmethod
    def extract_insights(self, experiences: Dict[str, Any]) -> None:
        """Extracts insights from the provided experiences.
        
        Args:
            experiences (Dict[str, Any]): A dictionary of experiences to extract insights from.
        """
        pass

    @abstractmethod
    def update_insights(self, operations: List[Tuple[str, str]]) -> None:
        """Updates the insights in the insight memory based on the provided operations.
        
        Args:
            operations (List[Tuple[str, str]]): A list of tuples, where each tuple contains a key and a value to update in the insight memory.
        """
        pass
