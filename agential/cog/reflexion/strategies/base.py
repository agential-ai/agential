"""Base Reflexion Agent strategy class."""

from abc import abstractmethod
from typing import Any, Dict, List, Optional, Tuple

from tiktoken import Encoding

from agential.cog.base.strategies import BaseStrategy
from agential.cog.reflexion.output import ReflexionReActStepOutput
from agential.cog.reflexion.reflect import (
    ReflexionCoTReflector,
    ReflexionReActReflector,
)
from agential.llm.llm import BaseLLM
from agential.utils.metrics import PromptMetrics


class ReflexionCoTBaseStrategy(BaseStrategy):
    """An abstract base class for defining strategies for the ReflexionCoT Agent.

    Attributes:
        llm (BaseLLM): The language model used for generating answers and critiques.
        reflector (Optional[ReflexionCoTReflector]): The reflector used for generating reflections.
        max_reflections (int): The maximum number of reflections allowed.
        max_trials (int): The maximum number of trials allowed.
        testing (bool): Whether the strategy is being used for testing. Defaults to False.
    """

    def __init__(
        self,
        llm: BaseLLM,
        reflector: ReflexionCoTReflector,
        max_reflections: int,
        max_trials: int,
        testing: bool = False,
    ) -> None:
        """Initialization."""
        super().__init__(llm=llm, testing=testing)
        self.reflector = reflector
        self.max_reflections = max_reflections
        self.max_trials = max_trials

    @abstractmethod
    def generate_thought(
        self,
        idx: int,
        scratchpad: str,
        question: str,
        examples: str,
        reflections: str,
        prompt: str,
        additional_keys: Dict[str, str],
    ) -> Tuple[str, str, PromptMetrics]:
        """Generates a thought based on the question, examples, and prompt.

        Args:
            idx (int): The index of the thought.
            scratchpad (str): The scratchpad containing previous thoughts.
            question (str): The question to be answered.
            examples (str): Examples to guide the generation process.
            reflections (str): Reflections to consider during generation.
            prompt (str): The prompt used for generating the thought.
            additional_keys (Dict[str, str]): Additional keys for the generation process.

        Returns:
            Tuple[str, str, PromptMetrics]: The updated scratchpad, the generated thought, and the metrics for the thought.
        """
        raise NotImplementedError
    
    @abstractmethod
    def generate_action(
        self,
        idx: int,
        scratchpad: str,
        question: str,
        examples: str,
        reflections: str,
        prompt: str,
        additional_keys: Dict[str, str],
    ) -> Tuple[str, str, str, PromptMetrics]:
        """Generates an action based on the question, examples, and prompt.

        Args:
            idx (int): The current index of the action.
            scratchpad (str): The current state of the scratchpad.
            question (str): The question to be answered.
            examples (str): Examples to guide the generation process.
            reflections (str): Reflections to consider during generation.
            prompt (str): The prompt used for generating the action.
            additional_keys (Dict[str, str]): Additional keys for the generation process.
            **kwargs (Any): Additional arguments.

        Returns:
            Tuple[str, str, str, PromptMetrics]: The updated scratchpad, the generated action, the action type, and the metrics for the action.
        """
        raise NotImplementedError

    @abstractmethod
    def generate_observation(
        self, idx: int, scratchpad: str, action_type: str, query: str, key: str
    ) -> Tuple[str, str, bool, str, bool]:
        """Generates an observation based on the action type and query.

        Args:
            idx (int): The current index of the observation.
            scratchpad (str): The current state of the scratchpad.
            action_type (str): The type of action to be performed.
            query (str): The query for the action.
            key (str): The key for the observation.

        Returns:
            Tuple[str, str, bool, str, bool]: The updated scratchpad, the answer, a boolean indicating if the observation is correct, the observation itself, and a boolean indicating if the observation is finished.
        """
        raise NotImplementedError

    @abstractmethod
    def halting_condition(
        self,
        idx: int,
        key: str,
        answer: str,
    ) -> bool:
        """Determines whether the halting condition has been met.

        Args:
            idx (int): The current step index.
            key (str): The key for the observation.
            answer (str): The answer generated.

        Returns:
            bool: True if the halting condition is met, False otherwise.
        """
        raise NotImplementedError

    @abstractmethod
    def reflect_condition(
        self,
        idx: int,
        reflect_strategy: Optional[str],
        key: str,
        answer: str,
    ) -> bool:
        """Determines whether the reflection condition has been met.

        Args:
            idx (int): The current step.
            reflect_strategy (Optional[str]): The strategy to use for reflection.
            key (str): The key for the observation.
            answer (str): The answer generated.
            
        Returns:
            bool: True if the reflection condition is met, False otherwise.
        """
        raise NotImplementedError
    
    @abstractmethod
    def reflect(
        self,
        scratchpad: str,
        reflect_strategy: str,
        question: str,
        examples: str,
        prompt: str,
        additional_keys: Dict[str, str],
    ) -> Tuple[List[str], str, PromptMetrics]:
        """Reflects on a given question, context, examples, prompt, and additional keys using the specified reflection strategy.

        Args:
            scratchpad (str): The scratchpad containing previous reflections.
            reflect_strategy (str): The strategy to use for reflection.
            question (str): The question to be reflected upon.
            examples (str): Examples to guide the reflection process.
            prompt (str): The prompt or instruction to guide the reflection.
            additional_keys (Dict[str, str]): Additional keys for the reflection process.

        Returns:
            Tuple[List[str], str, PromptMetrics]: The reflections, the reflection string, and the metrics.
        """
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> None:
        """Resets the internal state of the strategy."""
        raise NotImplementedError
    

class ReflexionReActBaseStrategy(BaseStrategy):
    """An abstract base class for defining strategies for the ReflexionReAct Agent.

    Attributes:
        llm (BaseLLM): The language model used for generating answers and critiques.
        reflector (Optional[ReflexionReActReflector]): The reflector used for generating reflections.
        max_reflections (int): The maximum number of reflections allowed.
        max_trials (int): The maximum number of trials allowed.
        max_steps (int): The maximum number of steps allowed.
        max_tokens (int): The maximum number of tokens allowed.
        enc (Encoding): The encoding for tokenization.
    """

    def __init__(
        self,
        llm: BaseLLM,
        reflector: ReflexionReActReflector,
        max_reflections: int,
        max_trials: int,
        max_steps: int,
        max_tokens: int,
        enc: Encoding,
    ) -> None:
        """Initialization."""
        super().__init__(llm)
        self.reflector = reflector
        self.max_reflections = max_reflections
        self.max_trials = max_trials
        self.max_steps = max_steps
        self.max_tokens = max_tokens
        self.enc = enc

    @abstractmethod
    def generate_action(
        self,
        question: str,
        examples: str,
        reflections: str,
        prompt: str,
        additional_keys: Dict[str, str],
        **kwargs: Any,
    ) -> Tuple[str, str]:
        """Generates an action based on the question, examples, and prompt.

        Args:
            question (str): The question to be answered.
            examples (str): Examples to guide the generation process.
            reflections (str): Reflections to guide the generation process.
            prompt (str): The prompt used for generating the action.
            additional_keys (Dict[str, str]): Additional keys for the generation process.
            **kwargs (Any): Additional arguments.

        Returns:
            Tuple[str, str]: The generated action type and query.
        """
        raise NotImplementedError

    @abstractmethod
    def generate_observation(
        self, step_idx: int, action_type: str, query: str, key: str
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """Generates an observation based on the action type and query.

        Args:
            step_idx (int): The index of the step.
            action_type (str): The type of action to be performed.
            query (str): The query for the action.
            key (str): The key for the observation.

        Returns:
            Tuple[bool, str, Dict[str, Any]]: A tuple containing a boolean indicating whether the answer is correct, a string representing the observation,
                and a dictionary of the external tool outputs.
        """
        raise NotImplementedError

    @abstractmethod
    def create_output_dict(
        self, react_out: List[ReflexionReActStepOutput], reflections: List[str]
    ) -> Dict[str, Any]:
        """Creates a dictionary of the output components.

        Args:
            react_out (List[ReflexionReActStepOutput]): The output from the ReAct agent.
            reflections (List[str]): The output from the ReAct reflections.

        Returns:
            Dict[str, Any]: A dictionary containing the ReAct output and the reflections.
        """
        raise NotImplementedError

    @abstractmethod
    def react_create_output_dict(
        self,
        thought: str,
        action_type: str,
        query: str,
        obs: str,
        external_tool_info: Dict[str, Any],
        is_correct: bool,
    ) -> Dict[str, Any]:
        """Creates a dictionary of the output components.

        Args:
            thought (str): The generated thought.
            action_type (str): The type of action performed.
            query (str): The query for the action.
            obs (str): The generated observation.
            external_tool_info (Dict[str, Any]): The external tool outputs.
            is_correct (bool): Whether the observation is correct.

        Returns:
            Dict[str, Any]: A dictionary containing the thought, action type, observation, answer, external_tool_info, and is_correct.
        """
        raise NotImplementedError

    @abstractmethod
    def halting_condition(self, idx: int, key: str, **kwargs: Any) -> bool:
        """Determines whether the halting condition has been met.

        Args:
            idx (int): The current step index.
            key (str): The key for the observation.
            **kwargs (Any): Additional arguments.

        Returns:
            bool: True if the halting condition is met, False otherwise.
        """
        raise NotImplementedError

    @abstractmethod
    def react_halting_condition(
        self,
        step_idx: int,
        question: str,
        examples: str,
        reflections: str,
        prompt: str,
        additional_keys: Dict[str, str],
        **kwargs: Any,
    ) -> bool:
        """Determines whether the halting condition for the ReAct agent has been met.

        Args:
            step_idx (int): The index of the current step.
            question (str): The question to be answered.
            examples (str): Examples to guide the generation process.
            reflections (str): Reflections to guide the generation process.
            prompt (str): The prompt used for generating the action.
            additional_keys (Dict[str, str]): Additional keys for the generation process.
            kwargs (Dict[str, Any]): Additional keyword arguments.

        Returns:
            bool: True if the halting condition is met, False otherwise.
        """
        raise NotImplementedError

    @abstractmethod
    def reflect(
        self,
        reflect_strategy: str,
        question: str,
        examples: str,
        prompt: str,
        additional_keys: Dict[str, str],
    ) -> Tuple[List[str], str]:
        """An abstract method that defines the behavior for reflecting on a given question, context, examples, prompt, and additional keys.

        Args:
            reflect_strategy (str): The strategy to use for reflection.
            question (str): The question to be reflected upon.
            examples (str): Examples to guide the reflection process.
            prompt (str): The prompt or instruction to guide the reflection.
            additional_keys (Dict[str, str]): Additional keys for the reflection process.

        Returns:
            Tuple[List[str], str]: The reflections and reflection string.
        """
        raise NotImplementedError

    @abstractmethod
    def reflect_condition(
        self,
        step_idx: int,
        reflect_strategy: Optional[str],
        question: str,
        examples: str,
        key: str,
        prompt: str,
        additional_keys: Dict[str, str],
        **kwargs: Dict[str, str],
    ) -> bool:
        """Determines whether the reflection condition has been met.

        Args:
            step_idx (int): The current step index.
            reflect_strategy (Optional[str]): The strategy to use for reflection.
            question (str): The question to be reflected upon.
            examples (str): Examples to guide the reflection process.
            key (str): The key for the observation.
            prompt (str): The prompt or instruction to guide the reflection.
            additional_keys (Dict[str, str]): Additional keys for the reflection process.
            kwargs (Dict[str, str]): Additional keyword arguments.

        Returns:
            bool: True if the reflection condition is met, False otherwise.
        """
        raise NotImplementedError
