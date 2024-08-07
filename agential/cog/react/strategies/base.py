"""Base ReAct Agent strategy class."""

from abc import abstractmethod
from typing import Any, Dict, Tuple

from tiktoken import Encoding

from agential.cog.base.strategies import BaseStrategy


class ReActBaseStrategy(BaseStrategy):
    """An abstract base class for defining strategies for the ReAct Agent.

    Attributes:
        llm (str): The language model used for generating answers and critiques.
        max_steps (int): The maximum number of steps the agent can take.
        max_tokens (int): The maximum number of tokens allowed for a response.
        enc (Encoding): The encoding used for the language model.
    """

    def __init__(
        self,
        llm: str,
        max_steps: int,
        max_tokens: int,
        enc: Encoding,
    ) -> None:
        """Initialization."""
        super().__init__(llm)
        self.max_steps = max_steps
        self.max_tokens = max_tokens
        self.enc = enc

    @abstractmethod
    def generate_action(
        self,
        question: str,
        examples: str,
        prompt: str,
        additional_keys: Dict[str, str],
    ) -> Tuple[str, str]:
        """Generates an action based on the question, examples, and prompt.

        Args:
            question (str): The question to be answered.
            examples (str): Examples to guide the generation process.
            prompt (str): The prompt used for generating the action.
            additional_keys (Dict[str, str]): Additional keys for the generation process.

        Returns:
            Tuple[str, str]: The generated action type and query.
        """
        pass

    @abstractmethod
    def generate_observation(
        self, idx: int, action_type: str, query: str
    ) -> Tuple[str, Dict[str, Any]]:
        """Generates an observation based on the action type and query.

        Args:
            idx (int): The index of the observation.
            action_type (str): The type of action to be performed.
            query (str): The query for the action.

        Returns:
            Tuple[str, Dict[str, Any]]: The generated observation and external tool outputs.
        """
        pass

    @abstractmethod
    def create_output_dict(
        self,
        thought: str,
        action_type: str,
        query: str,
        obs: str,
        external_tool_info: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Creates a dictionary of the output components.

        Args:
            thought (str): The generated thought.
            action_type (str): The type of action performed.
            query (str): The query for the action.
            obs (str): The generated observation.
            external_tool_info (Dict[str, Any]): The external tool outputs.

        Returns:
            Dict[str, Any]: A dictionary containing the thought, action type, query, observation, answer, and external tool output.
        """
        pass

    @abstractmethod
    def halting_condition(
        self,
        idx: int,
        question: str,
        examples: str,
        prompt: str,
        additional_keys: Dict[str, str],
    ) -> bool:
        """Determines whether the halting condition has been met.

        Args:
            idx (int): The current step index.
            question (str): The question being answered.
            examples (str): Examples to guide the generation process.
            prompt (str): The prompt used for generating the thought and action.
            additional_keys (Dict[str, str]): Additional keys for the generation process.

        Returns:
            bool: True if the halting condition is met, False otherwise.
        """
        pass
