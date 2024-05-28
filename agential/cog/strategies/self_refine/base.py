"""Base Self-Refine Agent strategy class."""

from abc import abstractmethod
from typing import Dict

from langchain_core.language_models.chat_models import BaseChatModel

from agential.cog.strategies.base import BaseStrategy


class SelfRefineBaseStrategy(BaseStrategy):
    """An abstract base class for defining strategies for the Self-Refine Agent."""

    def __init__(self, llm: BaseChatModel) -> None:
        """Initialization."""
        super().__init__(llm)

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
        pass

    @abstractmethod
    def create_output_dict(self, answer: str, critique: str) -> Dict[str, str]:
        """Creates a dictionary containing the answer and critique.

        Args:
            answer (str): The original answer.
            critique (str): The generated critique.

        Returns:
            Dict[str, str]: A dictionary containing the answer and critique.
        """
        pass

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
        pass

    @abstractmethod
    def halting_condition(self) -> bool:
        """Determines whether the critique meets the halting condition for stopping further updates.

        Returns:
            bool: True if the halting condition is met, False otherwise.
        """
        pass
