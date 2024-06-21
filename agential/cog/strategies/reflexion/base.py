"""Base Reflexion Agent strategy class."""

from abc import abstractmethod
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.language_models.chat_models import BaseChatModel

from agential.cog.strategies.base import BaseStrategy


class ReflexionCoTBaseStrategy(BaseStrategy):
    """An abstract base class for defining strategies for the ReflexionCoT Agent.

    Attributes:
        llm (BaseChatModel): The language model used for generating answers and critiques.
    """

    def __init__(self, llm: BaseChatModel) -> None:
        """Initialization."""
        super().__init__(llm)

    @abstractmethod
    def generate_action(
        self,
        question: str,
        examples: str,
        reflections: str,
        prompt: str,
        additional_keys: Dict[str, str],
    ) -> Tuple[str, str]:
        """Generates an action based on the question, examples, and prompt.

        Args:
            question (str): The question to be answered.
            examples (str): Examples to guide the generation process.
            reflections (str): Reflections to guide the generation process.
            prompt (str): The prompt used for generating the action.
            additional_keys (Dict[str, str]): Additional keys for the generation process.

        Returns:
            Tuple[str, str]: The generated action type and query.
        """
        pass

    @abstractmethod
    def generate_observation(
        self, action_type: str, query: str, key: str
    ) -> Tuple[bool, str]:
        """Generates an observation based on the action type and query.

        Args:
            action_type (str): The type of action to be performed.
            query (str): The query for the action.
            key (str): The key for the observation.

        Returns:
            Tuple[bool, str]: The generated observation.
        """
        pass

    @abstractmethod
    def create_output_dict(
        self,
        thought: str,
        action_type: str,
        obs: str,
        is_correct: bool,
        reflections: List[str],
    ) -> Dict[str, Any]:
        """Creates a dictionary of the output components.

        Args:
            thought (str): The generated thought.
            action_type (str): The type of action performed.
            obs (str): The generated observation.
            is_correct (bool): Whether the observation is correct.
            reflections (List[str]): A list of reflections.

        Returns:
            Dict[str, Any]: A dictionary containing the thought, action type, observation, answer, is_correct, and a list of reflections.
        """
        pass

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
        pass

    @abstractmethod
    def reflect(
        self,
        reflection_strategy: str,
        question: str,
        examples: str,
        prompt: str,
        additional_keys: Dict[str, str],
    ) -> Tuple[List[str], str]:
        """An abstract method that defines the behavior for reflecting on a given question, context, examples, prompt, and additional keys.

        Args:
            reflection_strategy (str): The strategy to use for reflection.
            question (str): The question to be reflected upon.
            examples (str): Examples to guide the reflection process.
            prompt (str): The prompt or instruction to guide the reflection.
            additional_keys (Dict[str, str]): Additional keys for the reflection process.

        Returns:
            Tuple[List[str], str]: The reflection string.
        """
        pass

    @abstractmethod
    def reflect_condition(
        self, idx: int, reflection_strategy: Optional[str], key: str
    ) -> bool:
        """Determines whether the reflection condition has been met.

        Args:
            idx (int): The current step.
            reflection_strategy (Optional[str]): The strategy to use for reflection.
            key (str): The key for the observation.

        Returns:
            bool: True if the reflection condition is met, False otherwise.
        """
        pass


class ReflexionReActBaseStrategy(BaseStrategy):
    """An abstract base class for defining strategies for the ReflexionReAct Agent.

    Attributes:
        llm (BaseChatModel): The language model used for generating answers and critiques.
    """

    def __init__(self, llm: BaseChatModel) -> None:
        """Initialization."""
        super().__init__(llm)

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
        pass

    @abstractmethod
    def generate_observation(
        self, step_idx: int, action_type: str, query: str, key: str
    ) -> Tuple[bool, str]:
        """Generates an observation based on the action type and query.

        Args:
            step_idx (int): The index of the step.
            action_type (str): The type of action to be performed.
            query (str): The query for the action.
            key (str): The key for the observation.

        Returns:
            str: The generated observation.
        """
        pass

    @abstractmethod
    def create_output_dict(
        self, react_out: List[Dict[str, Any]], reflections: List[str]
    ) -> Dict[str, Any]:
        """Creates a dictionary of the output components.

        Args:
            react_out (List[Dict[str, Any]]): The output from the ReAct agent.
            reflections (List[str]): The output from the ReAct reflections.

        Returns:
            Dict[str, Any]: A dictionary containing the ReAct output and the reflections.
        """
        pass

    @abstractmethod
    def react_create_output_dict(
        self, thought: str, action_type: str, query: str, obs: str, is_correct: bool
    ) -> Dict[str, str]:
        """Creates a dictionary of the output components.

        Args:
            thought (str): The generated thought.
            action_type (str): The type of action performed.
            query (str): The query for the action.
            obs (str): The generated observation.
            is_correct (bool): Whether the observation is correct.

        Returns:
            Dict[str, str]: A dictionary containing the thought, action type, observation, answer, and is_correct.
        """
        pass

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
        pass

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
        pass

    @abstractmethod
    def reflect(
        self,
        reflection_strategy: str,
        question: str,
        examples: str,
        prompt: str,
        additional_keys: Dict[str, str],
    ) -> Tuple[List[str], str]:
        """An abstract method that defines the behavior for reflecting on a given question, context, examples, prompt, and additional keys.

        Args:
            reflection_strategy (str): The strategy to use for reflection.
            question (str): The question to be reflected upon.
            examples (str): Examples to guide the reflection process.
            prompt (str): The prompt or instruction to guide the reflection.
            additional_keys (Dict[str, str]): Additional keys for the reflection process.

        Returns:
            Tuple[List[str], str]: The reflections and reflection string.
        """
        pass

    @abstractmethod
    def reflect_condition(
        self,
        step_idx: int,
        reflection_strategy: Optional[str],
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
            reflection_strategy (Optional[str]): The strategy to use for reflection.
            question (str): The question to be reflected upon.
            examples (str): Examples to guide the reflection process.
            key (str): The key for the observation.
            prompt (str): The prompt or instruction to guide the reflection.
            additional_keys (Dict[str, str]): Additional keys for the reflection process.
            kwargs (Dict[str, str]): Additional keyword arguments.

        Returns:
            bool: True if the reflection condition is met, False otherwise.
        """
        pass
