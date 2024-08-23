"""Base ExpeL Agent strategy class."""

from abc import abstractmethod
from typing import Any, Dict, List, Optional, Tuple

from agential.cog.base.strategies import BaseStrategy
from agential.cog.expel.memory import (
    ExpeLExperienceMemory,
    ExpeLInsightMemory,
)
from agential.cog.reflexion.agent import ReflexionReActAgent
from agential.llm.llm import BaseLLM, Response


class ExpeLBaseStrategy(BaseStrategy):
    """An abstract base class for defining strategies for the ExpeL Agent.

    Attributes:
        llm (BaseLLM): The language model used for generating answers and critiques.
        reflexion_react_agent (ReflexionReActAgent): The ReflexionReAct agent.
        experience_memory (ExpeLExperienceMemory): Memory module for storing experiences.
        insight_memory (ExpeLInsightMemory): Memory module for storing insights derived from experiences.
        success_batch_size (int): Batch size for processing success experiences in generating insights.
    """

    def __init__(
        self,
        llm: BaseLLM,
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
        reranker_strategy: Optional[str],
        additional_keys: Dict[str, Any],
    ) -> Tuple[str, Dict[str, str]]:
        """Generates dynamic examples for a given question.

        Args:
            question (str): The question to generate examples for.
            examples (str): The examples to use for generating dynamic examples.
            k_docs (int): The number of documents to retrieve for generating the examples.
            num_fewshots (int): The number of few-shot examples to generate.
            max_fewshot_tokens (int): The maximum number of tokens for the few-shot examples.
            reranker_strategy (Optional[str]): The strategy to use for reranking the generated examples.
            additional_keys (Dict[str, Any]): Additional keys to associate with the generated examples.

        Returns:
            Tuple[str, Dict[str, str]]: The generated examples and a dictionary of additional keys.
        """
        raise NotImplementedError

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
        additional_keys: List[Dict[str, str]],
        reflect_additional_keys: List[Dict[str, str]],
        patience: int,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """Gathers experience by executing a series of steps.

        Args:
            questions (List[str]): A list of questions to gather experiences for.
            keys (List[str]): A list of keys to associate with the gathered experiences.
            examples (str): The examples to use for generating dynamic examples.
            prompt (str): The prompt to use for generating dynamic examples.
            reflect_examples (str): The examples to use for the reflection strategy.
            reflect_prompt (str): The prompt to use for the reflection strategy.
            reflect_strategy (str): The strategy to use for the reflection process.
            additional_keys (List[Dict[str, str]]): Additional keys to associate with the gathered experiences.
            reflect_additional_keys (List[Dict[str, str]]): Additional keys to associate with the insights generated from the reflection process.
            patience (int): The number of attempts to make before giving up on gathering an experience.
            **kwargs (Any): Additional keyword arguments to pass to the underlying methods.

        Returns:
            List[Dict[str, Any]]: A list of experiences gathered.
        """
        raise NotImplementedError

    @abstractmethod
    def extract_insights(self, experiences: List[Dict[str, Any]]) -> Tuple[List[Response], List[Response]]:
        """Extracts insights from the provided experiences and updates the `InsightMemory` accordingly.

        This method is responsible for analyzing the successful and failed trials in the provided experiences, comparing them, and generating insights that are then stored in the `InsightMemory`. The insights are generated using the `get_operations_compare` and `get_operations_success` functions, and the `update_insights` method is used to apply the generated operations to the `InsightMemory`.
        The method first categorizes the experiences into "compare" and "success" categories, and then processes the experiences in batches. For the "compare" category, it compares the successful trial with all previous failed trials and generates insights using the `get_operations_compare` function. For the "success" category, it concatenates the successful trials and generates insights using the `get_operations_success` function.

        Args:
            experiences (List[Dict[str, Any]]): A dictionary containing the experiences to be processed, including questions, trajectories, and other relevant data.

        Return:
            List[Response]: A list of compare responses.
            List[Response]: A list of success responses.
        """
        raise NotImplementedError

    @abstractmethod
    def update_insights(self, operations: List[Tuple[str, str]]) -> None:
        """Updates the insights in the insight memory based on the provided operations.

        Args:
            operations (List[Tuple[str, str]]): A list of tuples, where each tuple contains a key and a value to update in the insight memory.
        """
        raise NotImplementedError

    @abstractmethod
    def create_output_dict(
        self,
        examples: str,
        additional_keys: Dict[str, str],
        experience: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Creates and returns an output dictionary containing the current state of the agent.

        Args:
            examples (str): The examples to be included in the output.
            additional_keys (Dict[str, str]): Additional key-value pairs to be included in the output.
            experience (List[Dict[str, Any]]): The current experience to be included in the output.

        Returns:
            Dict[str, Any]: A dictionary containing the current state of the agent, including examples, additional keys, and experience.
        """
        raise NotImplementedError
