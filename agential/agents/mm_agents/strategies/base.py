"""Base MM (OSWorld) Agent strategy class."""

import logging
from abc import abstractmethod
from typing import Any, Dict, List, Tuple

from agential.agents.base.strategies import BaseAgentStrategy
from agential.core.llm import BaseLLM

class MM_AgentBaseStrategy(BaseAgentStrategy):
    """An abstract base class for defining strategies for the MM (OSWorld Baseline) Agent.

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
        testing: bool = False,
    ) -> None:
        """Initialization."""
        super().__init__(llm, testing)

    @abstractmethod
    def generate(
        self, 
        platform: str,
        model: str,
        max_tokens: int,
        top_p: float,
        temperature: float,
        action_space: str,
        observation_type: str, 
        max_trajectory_length: int,
        a11y_tree_max_tokens: int,
        observations: List, 
        actions: List, 
        thoughts: List,
        _system_message: str,
        instruction: str,
        obs: Dict,
    ) -> Tuple[str, str, List, List, List, List]:
        raise NotImplementedError

    @abstractmethod
    def generate_observations(
        self, 
        _platform: str,
        observation_type: str, 
        max_trajectory_length: int,
        a11y_tree_max_tokens: int,
        observations: List, 
        actions: List, 
        thoughts: List,
        _system_message: str,
        instruction: str,
        obs: Dict,
        logger: logging.Logger
    ) -> Tuple[List, List, List, List]:
        raise NotImplementedError

    @abstractmethod
    def generate_thoughts(
        model: str,
        max_tokens: int,
        top_p: float,
        temperature: float,
        observation_type: str, 
        response: str,
        logger: logging.Logger,
        messages: List,
    ) -> str:
        raise NotImplementedError

    @abstractmethod
    def generate_actions(
        self,
        action_space: str,
        observation_type: str,
        response: str,
        masks: List,
    ) -> Tuple[str, List]:
        raise NotImplementedError

    @abstractmethod
    def reset(
        self,
        actions: List,
        thought: List,
        observations: List
    ) -> Tuple[List, List, List]:
        raise NotImplementedError