"""Base CRITIC Agent strategy class."""

from abc import abstractmethod
from typing import Any, Dict, Tuple

from agential.cog.base.strategies import BaseStrategy
from agential.cog.critic.output import CriticOutput
from agential.llm.llm import BaseLLM, Response


class CriticBaseStrategy(BaseStrategy):
    """An abstract base class for defining strategies for the CRITIC Agent.
    
    Attributes:
        llm (BaseLLM): An instance of a language model used for generating responses.
        testing (bool): Whether the generation is for testing purposes. Defaults to False.
    """

    def __init__(self, llm: BaseLLM, testing: bool = False) -> None:
        """Initialization."""
        super().__init__(llm=llm, testing=testing)

    @abstractmethod
    def generate(
        self,
        question: str,
        examples: str,
        prompt: str,
        critique_examples: str,
        critique_prompt: str,
        additional_keys: Dict[str, str],
        critique_additional_keys: Dict[str, str],
        max_interactions: int,
        use_tool: bool,
        reset: bool,
    ) -> CriticOutput:
        """Generates an answer and critique for the given question using the provided examples and prompts.

        Args:
            question (str): The question to be answered.
            examples (str): Few-shot examples to guide the language model in generating the answer.
            prompt (str): The instruction template used to prompt the language model for the answer.
            critique_examples (str): Few-shot examples to guide the language model in generating the critique.
            critique_prompt (str): The instruction template used to prompt the language model for the critique.
            additional_keys (Dict[str, str]): Additional keys to format the answer and critique prompts.
            critique_additional_keys (Dict[str, str]): Additional keys to format the critique prompt.
            max_interactions (int): The maximum number of interactions to perform.
            use_tool (bool): Whether to use a tool for generating the critique.
            reset (bool): Whether to reset the strategy.

        Returns:
            CriticOutput: The generated answer and critique.
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
        """Generates an answer to the given question using the provided examples and prompt.

        Args:
            question (str): The question to be answered.
            examples (str): Few-shot examples to guide the language model in generating the answer.
            prompt (str): The instruction template used to prompt the language model for the answer.
            additional_keys (Dict[str, str]): Additional keys to format the answer prompt.

        Returns:
            Tuple[str, Response]: The generated answer and model response.
        """
        raise NotImplementedError

    @abstractmethod
    def generate_critique(
        self,
        idx: int,
        question: str,
        examples: str,
        answer: str,
        critique: str,
        prompt: str,
        additional_keys: Dict[str, str],
        use_tool: bool,
        max_interactions: int,
    ) -> Tuple[str, Dict[str, Any]]:
        """Generates a critique of the provided answer using the given language model, question, examples, and prompt.

        Args:
            idx (int): The index of the current interaction.
            question (str): The question that was answered by the language model.
            examples (str): Few-shot examples to guide the language model in generating the critique.
            answer (str): The answer to be critiqued.
            critique (str): The previous critique, if any.
            prompt (str): The instruction template used to prompt the language model for the critique.
            additional_keys (Dict[str, str]): Additional keys to format the critique prompt.
            use_tool (bool): Whether to use an external tool for generating the critique.
            max_interactions (int): The maximum number of interactions to perform.
        
        Returns:
            Tuple[str, Dict[str, Any]]: The generated critique and any external tool information.
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
        external_tool_info: Dict[str, str],
    ) -> str:
        """Updates the answer based on the provided critique.

        Args:
            question (str): The question that was answered by the language model.
            examples (str): Few-shot examples to guide the language model in generating the updated answer.
            answer (str): The answer to be updated.
            critique (str): The critique of the answer.
            prompt (str): The instruction template used to prompt the language model for the updated answer.
            additional_keys (Dict[str, str]): Additional keys to format the updated answer prompt.
            external_tool_info (Dict[str, str]): Information about any external tool used for generating the critique.

        Returns:
            str: The updated answer.
        """
        raise NotImplementedError

    @abstractmethod
    def halting_condition(self) -> bool:
        """Checks if the halting condition is met.

        Returns:
            bool: True if the halting condition is met, False otherwise.
        """
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> None:
        """Resets the state of the critic."""
        raise NotImplementedError