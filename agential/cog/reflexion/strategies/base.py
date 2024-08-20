"""Base Reflexion Agent strategy class."""

from abc import abstractmethod
from typing import Any, Dict, List, Optional, Tuple

from tiktoken import Encoding

from agential.cog.base.strategies import BaseStrategy
from agential.cog.reflexion.output import (
    ReflexionCoTOutput,
    ReflexionReActOutput,
    ReflexionReActReActStepOutput,
    ReflexionReActStepOutput,
)
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
    def generate(
        self,
        question: str,
        key: str,
        examples: str,
        reflect_examples: str,
        prompt: str,
        reflect_prompt: str,
        reflect_strategy: str,
        additional_keys: Dict[str, str],
        reflect_additional_keys: Dict[str, str],
        patience: int,
        reset: bool,
    ) -> ReflexionCoTOutput:
        """Generates a thought based on the question, examples, and prompt.

        Args:
            question (str): The question to be answered.
            key (str): The key for the output.
            examples (str): Examples to guide the generation process.
            reflect_examples (str): Examples to guide the reflection process.
            prompt (str): The prompt to guide the generation process.
            reflect_prompt (str): The prompt to guide the reflection process.
            reflect_strategy (str): The strategy to use for reflection.
            additional_keys (Dict[str, str]): Additional keys to include in the output.
            reflect_additional_keys (Dict[str, str]): Additional keys to include in the reflection output.
            patience (int): The patience level for the agent.
            reset (bool): Whether to reset the agent.

        Returns:
            ReflexionCoTOutput: The output of the agent.
        """
        raise NotImplementedError

    @abstractmethod
    def generate_thought(
        self,
        scratchpad: str,
        question: str,
        examples: str,
        reflections: str,
        prompt: str,
        additional_keys: Dict[str, str],
    ) -> Tuple[str, str, PromptMetrics]:
        """Generates a thought based on the question, examples, and prompt.

        Args:
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
        scratchpad: str,
        question: str,
        examples: str,
        reflections: str,
        prompt: str,
        additional_keys: Dict[str, str],
    ) -> Tuple[str, str, str, PromptMetrics]:
        """Generates an action based on the question, examples, and prompt.

        Args:
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
        self, scratchpad: str, action_type: str, query: str, key: str
    ) -> Tuple[str, str, bool, str]:
        """Generates an observation based on the action type and query.

        Args:
            scratchpad (str): The current state of the scratchpad.
            action_type (str): The type of action to be performed.
            query (str): The query for the action.
            key (str): The key for the observation.

        Returns:
            Tuple[str, str, bool, str, bool]: The updated scratchpad, the answer, a boolean indicating if the observation is correct, and the observation itself.
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
        testing (bool): Whether to run in testing mode. Defaults to False.
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
        testing: bool = False,
    ) -> None:
        """Initialization."""
        super().__init__(llm=llm, testing=testing)
        self.reflector = reflector
        self.max_reflections = max_reflections
        self.max_trials = max_trials
        self.max_steps = max_steps
        self.max_tokens = max_tokens
        self.enc = enc

    @abstractmethod
    def generate(
        self,
        question: str,
        key: str,
        examples: str,
        reflect_examples: str,
        prompt: str,
        reflect_prompt: str,
        reflect_strategy: str,
        additional_keys: Dict[str, str],
        reflect_additional_keys: Dict[str, str],
        patience: int,
        reset: bool,
    ) -> ReflexionReActOutput:
        """Generates a thought based on the question, examples, and prompt.

        Args:
            question (str): The question to be answered.
            key (str): The key for the output.
            examples (str): Examples to guide the generation process.
            reflect_examples (str): Examples to guide the reflection process.
            prompt (str): The prompt to guide the generation process.
            reflect_prompt (str): The prompt to guide the reflection process.
            reflect_strategy (str): The strategy to use for reflection.
            additional_keys (Dict[str, str]): Additional keys to include in the output.
            reflect_additional_keys (Dict[str, str]): Additional keys to include in the reflection output.
            patience (int): The patience level for the agent.
            reset (bool): Whether to reset the agent.

        Returns:
            ReflexionReActOutput: The output of the agent.
        """
        raise NotImplementedError

    @abstractmethod
    def generate_react(
        self,
        question: str,
        key: str,
        examples: str,
        reflections: str,
        prompt: str,
        additional_keys: Dict[str, str] = {},
    ) -> Tuple[int, bool, str, bool, str, List[ReflexionReActReActStepOutput]]:
        """Generates a reaction based on the given question, key, examples, reflections, prompt, and additional keys.

        Args:
            question (str): The question to be answered.
            key (str): The key for the observation.
            examples (str): Examples to guide the reaction process.
            reflections (str): The reflections to guide the reaction process.
            prompt (str): The prompt or instruction to guide the reaction.
            additional_keys (Dict[str, str]): Additional keys for the reaction process.

        Returns:
            Tuple[int, bool, str, bool, str, List[ReflexionReActReActStepOutput]]: The reaction, whether the reaction is finished, the answer, whether the reaction is valid, the scratchpad, and the steps.
        """
        raise NotImplementedError

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
        """Generates a thought based on the given question, examples, reflections, prompt, and additional keys.

        Args:
            idx (int): The current step.
            scratchpad (str): The scratchpad containing previous thoughts and reflections.
            question (str): The question to generate a thought for.
            examples (str): Examples to guide the thought generation process.
            reflections (str): Reflections to consider during the thought generation process.
            prompt (str): The prompt or instruction to guide the thought generation.
            additional_keys (Dict[str, str]): Additional keys for the thought generation process.

        Returns:
            Tuple[str, str, PromptMetrics]: The updated scratchpad, the generated thought, and the thought metrics.
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
        """Generate an action for the current step in the reasoning process.

        Args:
            idx (int): The current step index.
            scratchpad (str): The scratchpad containing previous thoughts and actions.
            question (str): The main question or task to be addressed.
            examples (str): Relevant examples to provide context for action generation.
            trajectory (str): The current trajectory or history of thoughts and actions.
            reflections (str): Previous reflections to guide the action generation.
            depth (int): The current depth in the search tree.
            prompt (str): The prompt template for action generation.
            additional_keys (Dict[str, str]): Additional keys for prompt formatting.

        Returns:
            Tuple[str, str, str, PromptMetrics]: A tuple containing the updated trajectory, action type, query, and the metrics.
        """
        raise NotImplementedError

    @abstractmethod
    def generate_observation(
        self, idx: int, scratchpad: str, action_type: str, query: str, key: str
    ) -> Tuple[str, str, bool, bool, str, Dict[str, Any]]:
        """Generate an observation based on the given inputs.

        Args:
            idx (int): The current index of the observation.
            scratchpad (str): The current state of the scratchpad.
            action_type (str): The type of action performed.
            query (str): The query or action to observe.
            key (str): The key for the observation.

        Returns:
            Tuple[str, str, str, bool, Dict[str, Any]]: A tuple containing:
                - The updated scratchpad.
                - The answer.
                - A boolean indicating if finished.
                - The generated observation.
                - A boolean indicating if the task is finished.
                - The observation.
                - A dictionary with additional information.
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
    def react_halting_condition(
        self,
        finished: bool,
        idx: int,
        scratchpad: str,
        question: str,
        examples: str,
        reflections: str,
        prompt: str,
        additional_keys: Dict[str, str],
    ) -> bool:
        """Determine whether the halting condition has been met in the ReflexionReAct agent.

        Args:
            finished (bool): A boolean indicating whether the task is finished.
            idx (int): The index of the current step.
            scratchpad (str): The scratchpad containing previous thoughts and actions.
            question (str): The question to generate an action for.
            examples (str): Examples to guide the action generation process.
            reflections (str): Reflections to consider during the action generation process.
            prompt (str): The prompt or instruction to guide the action generation.
            additional_keys (Dict[str, str]): Additional keys for the action generation process.

        Returns:
            bool: True if the halting condition is met, False otherwise. The halting condition is met when the answer is not correct and the current step index is less than the maximum number of steps plus one.
        """
        raise NotImplementedError

    @abstractmethod
    def reflect_condition(
        self,
        answer: str,
        finished: bool,
        idx: int,
        scratchpad: str,
        reflect_strategy: Optional[str],
        question: str,
        examples: str,
        key: str,
        prompt: str,
        additional_keys: Dict[str, str],
    ) -> bool:
        """Determine whether the reflection condition has been met in the ReflexionReAct agent.

        Args:
            answer (str): The answer generated.
            finished (bool): A boolean indicating whether the task is finished.
            idx (int): The index of the current step.
            scratchpad (str): The scratchpad containing previous thoughts and actions.
            reflect_strategy (Optional[str]): The strategy to use for reflection.
            question (str): The question to be reflected upon.
            examples (str): Examples to guide the reflection process.
            key (str): The key for the observation.
            prompt (str): The prompt or instruction to guide the reflection.
            additional_keys (Dict[str, str]): Additional keys for the reflection process.

        Returns:
            bool: True if the reflection condition is met, False otherwise. The reflection condition is met when the agent is halted, the answer is not correct, and the reflection strategy is provided.
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
            scratchpad (str): The scratchpad containing previous thoughts and actions.
            reflect_strategy (str): The strategy to use for reflection.
            question (str): The question to be reflected upon.
            examples (str): Examples to guide the reflection process.
            prompt (str): The prompt or instruction to guide the reflection.
            additional_keys (Dict[str, str]): Additional keys for the reflection process.

        Returns:
            Tuple[List[str], str, PromptMetrics]: The reflections, reflection string, and the metrics for the reflection process.
        """
        raise NotImplementedError

    def reset(self) -> None:
        """Resets the internal state of the strategy."""
        raise NotImplementedError
