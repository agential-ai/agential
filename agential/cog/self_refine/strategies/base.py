"""Base Self-Refine Agent strategy class."""

from abc import abstractmethod
from typing import Any, Dict

from agential.cog.base.strategies import BaseStrategy
from agential.cog.self_refine.output import SelfRefineOutput
from agential.llm.llm import BaseLLM


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
    def generate_critique(
        self,
        question: str,
        examples: str,
        answer: str,
        prompt: str,
        additional_keys: Dict[str, str],
    ) -> str:
        """Generates a critique of the provided answer using the given language model, question, examples, and prompt.

        Args:
            question (str): The question that was answered by the language model.
            examples (str): Few-shot examples to guide the language model in generating the critique.
            answer (str): The answer to be critiqued.
            prompt (str): The instruction template used to prompt the language model for the critique.
            additional_keys (Dict[str, str]): Additional keys to format the critique prompt.

        Returns:
            str: The generated critique.
        """
        raise NotImplementedError

    @abstractmethod
    def create_output_dict(self, answer: str, critique: str) -> Dict[str, Any]:
        """Creates a dictionary containing the answer and critique.

        Args:
            answer (str): The original answer.
            critique (str): The generated critique.

        Returns:
            Dict[str, Any]: A dictionary containing the answer and critique.
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
    ) -> str:
        """Updates the answer based on the provided critique using the given language model and question.

        Args:
            question (str): The question that was answered by the language model.
            examples (str): Few-shot examples to guide the language model in generating the updated answer.
            answer (str): The original answer to be updated.
            critique (str): The critique of the original answer.
            prompt (str): The instruction template used to prompt the language model for the update.
            additional_keys (Dict[str, str]): Additional keys to format the update prompt.

        Returns:
            str: The updated answer.
        """
        raise NotImplementedError

    @abstractmethod
    def halting_condition(self) -> bool:
        """Determines whether the critique meets the halting condition for stopping further updates.

        Returns:
            bool: True if the halting condition is met, False otherwise.
        """
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> None:
        """Resets the strategy to its initial state."""
        raise NotImplementedError