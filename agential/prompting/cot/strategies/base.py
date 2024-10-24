"""CoT base strategy."""

from abc import abstractmethod
from typing import Dict, List, Optional

from agential.core.llm import BaseLLM
from agential.prompting.base.strategies import BasePromptingStrategy
from agential.prompting.cot.output import CoTOutput


class CoTBaseStrategy(BasePromptingStrategy):
    """An abstract base class for defining strategies for the CoT prompting method.

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
        key: str,
        examples: str,
        prompt: str,
        additional_keys: Dict[str, str],
        num_retries: int,
        warming: List[Optional[float]],
    ) -> CoTOutput:
        """Generates an answer and critique for the given question using the provided examples and prompts.

        Args:
            question (str): The question to be answered.
            key (str): The answer.
            examples (str): Few-shot examples to guide the language model in generating the answer.
            prompt (str): The instruction template used to prompt the language model for the answer.
            additional_keys (Dict[str, str]): Additional keys to format the answer prompt.
            num_retries (int): Number of retries.
            warming (List[Optional[float]]): List of warmup temperatures.

        Returns:
            CoTOutput: The output of the CoT strategy.
        """
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> None:
        """Resets the strategy's internal state."""
        raise NotImplementedError
