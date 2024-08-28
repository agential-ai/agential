"""Base CRITIC Agent strategy class."""

from abc import abstractmethod
from typing import Any, Dict, List, Tuple

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
        critique_examples: str,
        prompt: str,
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
            critique_examples (str): Few-shot examples to guide the language model in generating the critique.
            prompt (str): The instruction template used to prompt the language model for the answer.
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
    ) -> Tuple[str, List[Response]]:
        """Generates an answer to the given question using the provided examples and prompt.

        Args:
            question (str): The question to be answered.
            examples (str): Few-shot examples to guide the language model in generating the answer.
            prompt (str): The instruction template used to prompt the language model for the answer.
            additional_keys (Dict[str, str]): Additional keys to format the answer prompt.

        Returns:
            Tuple[str, List[Response]]: The generated answer and model responses.
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
    ) -> Tuple[str, Dict[str, Any], bool, List[Response]]:
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
            Tuple[str, Dict[str, Any], bool, List[Response]]: The generated critique, any external tool information, a boolean for if it finished, and the responses.
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
    ) -> Tuple[str, List[Response]]:
        """Updates the answer based on the provided critique using the given language model and question.

        Args:
            question (str): The question that was answered by the language model.
            examples (str): Few-shot examples to guide the language model in generating the updated answer.
            answer (str): The original answer to be updated.
            critique (str): The critique of the original answer.
            prompt (str): The instruction template used to prompt the language model for the update.
            additional_keys (Dict[str, str]): Additional keys to format the update prompt.
            external_tool_info (Dict[str, str]): Information from any external tools used during the critique.

        Returns:
            str: The updated answer.
            List[Response]: The responses from the critique.
        """
        raise NotImplementedError

    @abstractmethod
    def create_output_dict(
        self,
        finished: bool,
        answer: str,
        critique: str,
        external_tool_info: Dict[str, Any],
        answer_response: List[Response],
        critique_response: List[Response],
    ) -> Dict[str, Any]:
        """Creates a dictionary containing the answer and critique, along with any additional key updates.

        Args:
            finished (bool): Whether the critique process has finished.
            answer (str): The original answer.
            critique (str): The generated critique.
            external_tool_info (Dict[str, Any]): Information from any external tools used during the critique.
            answer_response (List[Response]): The responses from the answer.
            critique_response (List[Response]): The responses from the critique.

        Returns:
            Dict[str, Any]: A dictionary containing the answer, critique, and additional key updates.
        """
        raise NotImplementedError

    @abstractmethod
    def halting_condition(self, finished: bool) -> bool:
        """Checks if the halting condition is met.

        Args:
            finished (bool): Whether the interaction

        Returns:
            bool: True if the halting condition is met, False otherwise.
        """
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> None:
        """Resets the strategy's internal state."""
        raise NotImplementedError
