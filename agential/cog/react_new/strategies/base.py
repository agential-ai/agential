"""Base ReAct Agent strategy class."""

from abc import abstractmethod
from typing import Any, Dict, Tuple

from tiktoken import Encoding

from agential.cog.base.strategies import BaseStrategy
from agential.llm.llm import BaseLLM, ModelResponse
from agential.cog.react_new.output import ReActOutput



class ReActBaseStrategy(BaseStrategy):
    """An abstract base class for defining strategies for the ReAct Agent.

    Attributes:
        llm (BaseLLM): The language model used for generating answers and critiques.
        max_steps (int): The maximum number of steps the agent can take.
        max_tokens (int): The maximum number of tokens allowed for a response.
        enc (Encoding): The encoding used for the language model.
    """

    def __init__(
        self,
        llm: BaseLLM,
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
    def generate(
        self,
        question: str,
        examples: str,
        prompt: str,
        additional_keys: Dict[str, str],
    ) -> ReActOutput:
        """Generates a thought based on the question, examples, and prompt.
        
        Args:
            question (str): The question to be answered.
            examples (str): Examples to guide the generation process.
            prompt (str): The prompt used for generating the thought.
            additional_keys (Dict[str, str]): Additional keys for the generation process.
            **kwargs (Any): Additional arguments.
         
        Returns:
            ReactOutput: The output of the generation process.
        """

        pass



    @abstractmethod
    def generate_thought(
        self,
        scratchpad: str,
        question: str,
        examples: str,
        prompt: str,
        additional_keys: Dict[str, str],
    ) -> Tuple[str, ModelResponse]:
        """Generates a thought based on the question, examples, and prompt.

        Args:
            scratchpad (str): The scratchpad used for generating the thought.
            question (str): The question to be answered.
            examples (str): Examples to guide the generation process.
            prompt (str): The prompt used for generating the thought.
            additional_keys (Dict[str, str]): Additional keys for the generation process.

        Returns:
            Tuple[str, ModelResponse]: The generated thought.
        """
        pass

    @abstractmethod
    def generate_action(
        self,
        scratchpad: str,
        question: str,
        examples: str,
        prompt: str,
        additional_keys: Dict[str, str],
    ) -> Tuple[str, str, ModelResponse]:
        """Generates an action based on the question, examples, and prompt.

        Args:
            scratchpad (str): The scratchpad containing the previous steps.
            question (str): The question to be answered.
            examples (str): Examples to guide the generation process.
            prompt (str): The prompt used for generating the action.
            additional_keys (Dict[str, str]): Additional keys for the generation process.

        Returns:
            Tuple[str, str, ModelResponse]: The generated action type and query.
        """
        pass

    @abstractmethod
    def generate_observation(
        self, action_type: str, query: str
    ) -> Tuple[str, str, bool, Dict[str, Any]]:
        """Generates an observation based on the action type and query.

        Args:
            action_type (str): The type of action to be performed.
            query (str): The query for the action.

        Returns:
            Tuple[str, str, bool, Dict[str, Any]]: The generated observation, the observation type, whether the observation is correct, and the observation metrics.
        """
        pass

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
        pass

    @abstractmethod
    def reset(self) -> None:
        """Resets the agent's state."""
        
        pass