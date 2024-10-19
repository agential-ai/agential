"""Generic base strategy class."""

from agential.core.base.strategies import BaseStrategy


class BasePromptingStrategy(BaseStrategy):
    """An abstract base class for defining prompting method strategies for generating responses.

    Parameters:
        llm (BaseLLM): An instance of a language model used for generating responses.
        testing (bool): Whether the generation is for testing purposes. Defaults to False.
    """

    pass
