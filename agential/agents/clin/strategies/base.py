"""CLIN base strategy."""

from abc import abstractmethod
from typing import Dict, List, Tuple

from tiktoken import Encoding

from agential.agents.base.strategies import BaseAgentStrategy
from agential.agents.clin.output import CLINOutput
from agential.core.llm import BaseLLM, Response


class CLINBaseStrategy(BaseAgentStrategy):
    """An abstract base class for defining strategies for the CLIN Agent.

    Attributes:
        llm (BaseLLM): An instance of a language model used for generating responses.
        max_trials (int): The maximum number of trials allowed.
        max_steps (int): The maximum number of steps allowed.
        max_tokens (int): The maximum number of tokens allowed.
        enc (Encoding): The encoding for tokenization.
        testing (bool): Whether the generation is for testing purposes. Defaults to False.
    """

    def __init__(
        self,
        llm: BaseLLM,
        max_trials: int,
        max_steps: int,
        max_tokens: int,
        enc: Encoding,
        testing: bool = False,
    ) -> None:
        """Initialization."""
        super().__init__(llm=llm, testing=testing)
        self.max_trials = max_trials
        self.max_steps = max_steps
        self.max_tokens = max_tokens
        self.enc = enc

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
    ) -> CLINOutput:
        """Generates an answer.

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
            CLINOutput: The generated answer and critique.
        """
        raise NotImplementedError

    @abstractmethod
    def generate_action(
        self,
        question: str,
        examples: str,
        prompt: str,
        additional_keys: Dict[str, str],
    ) -> Tuple[str, List[Response]]:
        """Generates an action.

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
    def generate_summaries(self) -> Tuple[str, Response]:
        pass

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
    def step_halting_condition(self, finished: bool) -> bool:
        """Checks if the halting condition is met."""
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> None:
        """Resets the strategy's internal state."""
        raise NotImplementedError
