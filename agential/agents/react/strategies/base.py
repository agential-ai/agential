"""Base ReAct Agent strategy class."""

from abc import abstractmethod
from typing import Any, Dict, Tuple

from agential.agents.base.strategies import BaseAgentStrategy
from agential.agents.react.output import ReActOutput
from agential.core.llm import BaseLLM, Response


class ReActBaseStrategy(BaseAgentStrategy):
    """An abstract base class for defining strategies for the ReAct Agent.

    Attributes:
        llm (BaseLLM): The language model used for generating answers and critiques.
        max_steps (int): The maximum number of steps the agent can take.
        testing (bool): Whether the generation is for testing purposes. Defaults to False.
    """

    def __init__(
        self,
        llm: BaseLLM,
        max_steps: int,
        testing: bool = False,
    ) -> None:
        """Initialization."""
        super().__init__(llm=llm, testing=testing)
        self.max_steps = max_steps

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
    ) -> Tuple[str, str, Response]:
        """Generates a thought based on the question, examples, and prompt.

        Args:
            idx (int): The index of the thought.
            scratchpad (str): The scratchpad used for generating the thought.
            question (str): The question to be answered.
            examples (str): Examples to guide the generation process.
            prompt (str): The prompt used for generating the thought.
            additional_keys (Dict[str, str]): Additional keys for the generation process.

        Returns:
            Tuple[str, str, Response]: The updated scratchpad, the generated thought, and the metrics for the thought.
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
    ) -> Tuple[str, str, str, Response]:
        """Generates an action based on the question, examples, and prompt.

        Args:
            idx (int): The index of the action.
            scratchpad (str): The scratchpad containing the previous steps.
            question (str): The question to be answered.
            examples (str): Examples to guide the generation process.
            prompt (str): The prompt used for generating the action.
            additional_keys (Dict[str, str]): Additional keys for the generation process.

        Returns:
            Tuple[str, str, str, Response]: The updated scratchpad, the generated action, the action type, and the metrics for the action.
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
    ) -> bool:
        """Determines whether the current iteration of the task should be halted based on various conditions.

        Args:
            finished (bool): Whether the task has been completed.
            idx (int): The current index of the iteration.
            question (str): The question being answered.
            scratchpad (str): The current state of the scratchpad.
            examples (str): Examples provided for the task.
            prompt (str): The prompt used to generate the action.
            additional_keys (Dict[str, str]): Additional key-value pairs to pass to the language model.

        Returns:
            bool: True if the task should be halted, False otherwise.
        """
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> None:
        """Resets the agent's state."""
        raise NotImplementedError
