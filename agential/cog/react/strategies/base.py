"""Base ReAct Agent strategy class."""

from abc import abstractmethod
from typing import Any, Dict, Tuple

from tiktoken import Encoding

from agential.cog.base.strategies import BaseStrategy
from agential.cog.react.output import ReActOutput
from agential.llm.llm import BaseLLM, ModelResponse


class ReActBaseStrategy(BaseStrategy):
    """An abstract base class for defining strategies for the ReAct Agent.

    Attributes:
        llm (BaseLLM): The language model used for generating answers and critiques.
        max_steps (int): The maximum number of steps the agent can take.
        max_tokens (int): The maximum number of tokens allowed for a response.
        enc (Encoding): The encoding used for the language model.
        testing (bool): Whether the generation is for testing purposes. Defaults to False.
    """

    def __init__(
        self,
        llm: BaseLLM,
        max_steps: int,
        max_tokens: int,
        enc: Encoding,
        testing: bool = False,
    ) -> None:
        """Initialization."""
        super().__init__(llm, testing)
        self.max_steps = max_steps
        self.max_tokens = max_tokens
        self.enc = enc

    @abstractmethod
    def generate(
        self,
        question: str,
        examples: str,
        prompt: str,
        additional_keys: Dict[str, str],
        reset: bool,
    ) -> ReActOutput:
        """Generates a thought based on the question, examples, and prompt.

        Args:
            question (str): The question to be answered.
            examples (str): Examples to guide the generation process.
            prompt (str): The prompt used for generating the thought.
            additional_keys (Dict[str, str]): Additional keys for the generation process.
            reset (bool): Whether to reset the strategy.

        Returns:
            ReactOutput: The output of the generation process.
        """
        raise NotImplementedError

    @abstractmethod
    def generate_thought(
        self,
        idx: int,
        scratchpad: str,
        question: str,
        examples: str,
        prompt: str,
        additional_keys: Dict[str, str],
    ) -> Tuple[str, str, ModelResponse]:
        """Generates a thought based on the question, examples, and prompt.

        Args:
            idx (int): The index of the thought.
            scratchpad (str): The scratchpad used for generating the thought.
            question (str): The question to be answered.
            examples (str): Examples to guide the generation process.
            prompt (str): The prompt used for generating the thought.
            additional_keys (Dict[str, str]): Additional keys for the generation process.

        Returns:
            Tuple[str, str, ModelResponse]: The scratchpad, generated thought, and model response.
        """
        raise NotImplementedError

    @abstractmethod
    def generate_action(
        self,
        idx: int,
        scratchpad: str,
        question: str,
        examples: str,
        prompt: str,
        additional_keys: Dict[str, str],
    ) -> Tuple[str, str, str, ModelResponse]:
        """Generates an action based on the question, examples, and prompt.

        Args:
            idx (int): The index of the action.
            scratchpad (str): The scratchpad containing the previous steps.
            question (str): The question to be answered.
            examples (str): Examples to guide the generation process.
            prompt (str): The prompt used for generating the action.
            additional_keys (Dict[str, str]): Additional keys for the generation process.

        Returns:
            Tuple[str, str, str, ModelResponse]: The scratchpad, generated action type and query, and model response.
        """
        raise NotImplementedError

    @abstractmethod
    def generate_observation(
        self, idx: int, scratchpad: str, action_type: str, query: str
    ) -> Tuple[str, str, str, bool, Dict[str, Any]]:
        """Generates an observation based on the action type and query.

        Args:
            idx (int): The index of the observation.
            scratchpad (str): The scratchpad containing the previous steps.
            action_type (str): The type of action to be performed.
            query (str): The query for the action.

        Returns:
            Tuple[str, str, str, bool, Dict[str, Any]]: The scratchpad, the answer, observation, whether the query is correct, and the observation metrics.
        """
        raise NotImplementedError

    @abstractmethod
    def halting_condition(
        self,
        finished: bool,
        idx: int,
        question: str,
        scratchpad: str,
        examples: str,
        prompt: str,
        additional_keys: Dict[str, str],
    ) -> bool:
        """Determines whether the halting condition has been met.

        Args:
            finished (bool): Whether the agent has finished its task.
            idx (int): The current step index.
            question (str): The question being answered.
            scratchpad (str): The scratchpad containing the agent's thoughts and actions.
            examples (str): Examples to guide the generation process.
            prompt (str): The prompt used for generating the thought and action.
            additional_keys (Dict[str, str]): Additional keys for the generation process.

        Returns:
            bool: True if the halting condition is met, False otherwise.
        """
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> None:
        """Resets the agent's state."""
        raise NotImplementedError
