"""Base ExpeL Agent strategy class."""

from abc import abstractmethod
from typing import Any, Dict, List, Optional, Tuple

from agential.agents.expel.memory import (
    ExpeLExperienceMemory,
    ExpeLInsightMemory,
)
from agential.agents.expel.output import ExpeLOutput
from agential.agents.reflexion.agent import ReflexionReAct
from agential.core.base.agents.strategies import BaseAgentStrategy
from agential.llm.llm import BaseLLM, Response


class ExpeLBaseStrategy(BaseAgentStrategy):
    """An abstract base class for defining strategies for the ExpeL Agent.

    Attributes:
        llm (BaseLLM): The language model used for generating answers and critiques.
        reflexion_react_agent (ReflexionReAct): The ReflexionReAct agent.
        experience_memory (ExpeLExperienceMemory): Memory module for storing experiences.
        insight_memory (ExpeLInsightMemory): Memory module for storing insights derived from experiences.
        success_batch_size (int): Batch size for processing success experiences in generating insights.
        testing (bool): Whether to run in testing mode. Defaults to False.
    """

    def __init__(
        self,
        llm: BaseLLM,
        reflexion_react_agent: ReflexionReAct,
        experience_memory: ExpeLExperienceMemory,
        insight_memory: ExpeLInsightMemory,
        success_batch_size: int,
        testing: bool = False,
    ) -> None:
        """Initialization."""
        super().__init__(llm=llm, testing=testing)
        self.reflexion_react_agent = reflexion_react_agent
        self.success_batch_size = success_batch_size
        self.insight_memory = insight_memory
        self.experience_memory = experience_memory

    @abstractmethod
    def generate(
        self,
        question: str,
        key: str,
        examples: str,
        prompt: str,
        reflect_examples: str,
        reflect_prompt: str,
        reflect_strategy: str,
        additional_keys: Dict[str, str],
        reflect_additional_keys: Dict[str, str],
        use_dynamic_examples: bool,
        extract_insights: bool,
        patience: int,
        k_docs: int,
        num_fewshots: int,
        max_fewshot_tokens: int,
        reranker_strategy: Optional[str],
        reset: bool,
    ) -> ExpeLOutput:
        """Collects and stores experiences from interactions based on specified questions and strategies.

        This method invokes the ReflexionReAct agent to process a set of questions with corresponding keys,
        using the provided strategy, prompts, and examples. It captures the trajectories of the agent's reasoning
        and reflection process, storing them for future analysis and insight extraction.

        Parameters:
            questions (List[str]): A list of questions for the agent to process.
            keys (List[str]): Corresponding keys to the questions, used for internal tracking and analysis.
            examples (str): Examples to provide context or guidance for the ReflexionReAct agent.
            prompt (str): The initial prompt or instruction to guide the ReflexionReAct agent's process.
            reflect_examples (str): Examples specifically for the reflection phase of processing.
            reflect_prompt (str): The prompt or instruction guiding the reflection process.
            reflect_strategy (Optional[str]): The strategy to use for processing questions.
            additional_keys (Dict[str, str]): The additional keys.
            reflect_additional_keys (Dict[str, str]): Additional keys for the reflection phase.
            use_dynamic_examples (bool): A boolean specifying whether or not to use dynamic examples from ExpeL's memory.
            extract_insights (bool): Whether to extract insights from the experiences.
            patience (int): The number of times to retry the agent's process if it fails.
            k_docs (int): The number of documents to retrieve for the fewshot.
            num_fewshots (int): The number of examples to use for the fewshot.
            max_fewshot_tokens (int): The maximum number of tokens to use for the fewshot.
            reranker_strategy (Optional[str]): The strategy to use for re-ranking the retrieved.
            reset (bool): Whether to reset the agent's state for a new problem-solving session.

        Returns:
            ExpeLOutput: The output of the ExpeL agent.
        """
        raise NotImplementedError

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

        Returns:
            List[Dict[str, Any]]: A list of experiences gathered.
        """
        raise NotImplementedError

    @abstractmethod
    def extract_insights(
        self, experiences: List[Dict[str, Any]]
    ) -> Tuple[List[Response], List[Response]]:
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
        """Updates the insights in the `InsightMemory` based on the provided operations.

        The `operations` parameter is a list of tuples, where each tuple contains an operation type and an insight. The supported operation types are:
        - "REMOVE": Removes the insight from the `InsightMemory`.
        - "AGREE": Increases the score of the insight in the `InsightMemory`.
        - "EDIT": Updates the insight in the `InsightMemory` with the provided insight.
        - "ADD": Adds a new insight to the `InsightMemory` with a score of 2.

        This method is responsible for applying the various operations to the insights stored in the `InsightMemory`.

        Args:
            operations (List[Tuple[str, str]]): A list of tuples, where each tuple contains an operation type and an insight.
        """
        raise NotImplementedError

    def reset(self) -> None:
        """Resets the ExperienceMemory and InsightMemory."""
        raise NotImplementedError
