"""Base Self-Refine Agent strategy class."""

from abc import abstractmethod
from typing import Any, Dict, Tuple

from agential.cog.base.strategies import BaseStrategy
from agential.cog.self_refine.output import SelfRefineOutput
from agential.llm.llm import BaseLLM, Response


class SelfRefineBaseStrategy(BaseStrategy):
    """An abstract base class for defining strategies for the Self-Refine Agent.

    Attributes:
        llm (BaseLLM): The language model used for generating answers and critiques.
        patience (int): The number of interactions to tolerate the same incorrect answer
            before halting further attempts.
        testing (bool): Whether the strategy is being used for testing. Defaults to False.
    """

    def __init__(self, llm: BaseLLM, patience: int, testing: bool = True) -> None:
        """Initialization."""
        super().__init__(llm=llm, testing=testing)
        self.patience = patience

    @abstractmethod
    def generate(
        self,
        question: str,
        examples: str,
        prompt: str,
        critique_examples: str,
        critique_prompt: str,
        refine_examples: str,
        refine_prompt: str,
        additional_keys: Dict[str, str],
        critique_additional_keys: Dict[str, str],
        refine_additional_keys: Dict[str, str],
        max_interactions: int,
        reset: bool,    
    ) -> SelfRefineOutput:
        """Generates a refined solution for a given question through an iterative self-refinement process.
        Args:
            question (str): The question or problem to solve.
            examples (str): Precedent examples to guide initial solution generation.
            prompt (str): Instructional prompt for initial solution generation.
            critique_examples (str): Precedent examples to guide critique generation.
            critique_prompt (str): Instructional prompt for critique generation.
            refine_examples (str): Precedent examples to guide solution refinement.
            refine_prompt (str): Instructional prompt for refining the solution.
            additional_keys (Dict[str, str]): Additional keys to format the prompt.
            critique_additional_keys (Dict[str, str]): Additional keys to format the critique_prompt.
            refine_additional_keys (Dict[str, str]): Additional keys to format the refine_prompt.
            fewshot_type (str): The type of few-shot examples to use.
            max_interactions (int): Maximum number of refinement iterations.
            reset (bool): Resets the agent's state.
        Returns:
            SelfRefineOutput: The agent's output.
        """
        raise NotImplementedError

    @abstractmethod
    def generate_answer(
        self,
        question: str,
        examples: str,
        prompt: str,
        additional_keys: Dict[str, str],
    ) -> Tuple[str, Response]:
        """Generates an answer for the given question using the provided prompt and examples.

        Args:
            question (str): The question to generate an answer for.
            examples (str): Few-shot examples to guide the language model.
            prompt (str): The prompt to generate an answer.
            additional_keys (Dict[str, str]): Additional keys for the prompt.

        Returns:
            Tuple[str, Response]: The generated answer and the response from the language model.
        """
        raise NotImplementedError

    @abstractmethod
    def generate_critique(
        self,
        question: str,
        examples: str,
        answer: str,
        prompt: str,
        additional_keys: Dict[str, str],
    ) -> Tuple[str, bool, Response]:
        """Generates a critique for the provided answer using the given prompt and examples.

        Stops early if patience is reached and answer remains the same.

        Args:
            question (str): The qa question that was answered.
            examples (str): Few-shot examples to guide the language model in generating the critique.
            answer (str): The answer to be critiqued.
            prompt (str): The prompt to generate a critique.
            additional_keys (Dict[str, str]): Additional keys for the prompt.

        Returns:
            Tuple[str, bool, Response]: The critique, a boolean indicating it's finished, and the model response.
        """
        raise NotImplementedError
    
    @abstractmethod
    def update_answer_based_on_critique(
        self,
        question: str,
        examples: str,
        answer: str,
        critique: str,
        prompt: str,
        additional_keys: Dict[str, str],
    ) -> Tuple[str, Response]:
        """Updates the answer based on the given critique.

        Args:
            question: The question that was answered by the language model.
            examples: Few-shot examples to guide the language model.
            answer: The answer provided by the language model.
            critique: The critique of the answer.
            prompt: The prompt to be used for generating the updated answer.
            additional_keys: Additional context or parameters to include in the critique prompt.

        Returns:
            Tuple[str, Response]: The updated answer and the model response.
        """
        raise NotImplementedError

    @abstractmethod
    def halting_condition(self, finished: bool) -> bool:
        """Checks if the halting condition is met.

        Args:
            finished (bool): Whether the interaction has finished.

        Returns:
            bool: True if the halting condition is met, False otherwise.
        """
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> None:
        """Resets the strategy to its initial state."""
        raise NotImplementedError