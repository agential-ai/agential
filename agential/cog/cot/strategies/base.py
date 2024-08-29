"""CoT base strategy."""

from abc import abstractmethod
from typing import Dict

from agential.cog.base.strategies import BaseStrategy
from agential.cog.cot.output import CoTOutput
from agential.llm.llm import BaseLLM


class CoTBaseStrategy(BaseStrategy):
    """An abstract base class for defining strategies for the CoT Agent.

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
        additional_keys: Dict[str, str],
    ) -> CoTOutput:
        """Generates an answer and critique for the given question using the provided examples and prompts.

        Args:
            question (str): The question to be answered.
            examples (str): Few-shot examples to guide the language model in generating the answer.
            prompt (str): The instruction template used to prompt the language model for the answer.
            additional_keys (Dict[str, str]): Additional keys to format the answer prompt.

        Returns:
            CoTOutput: The output of the CoT strategy.
        """
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> None:
        """Resets the strategy's internal state."""
        raise NotImplementedError
