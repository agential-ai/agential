"""Base (OSWorld) Agent strategy class."""

from abc import abstractmethod
from typing import Any, Dict, List, Tuple

from agential.agents.base.strategies import BaseAgentStrategy
from agential.agents.osworld_baseline.output import OSWorldBaseOutput
from agential.core.llm import BaseLLM, Response


class OSWorldBaseStrategy(BaseAgentStrategy):
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
        """Initializes the OSWorldBaseStrategy class.

        Args:
            llm (BaseLLM): The language model instance.
            testing (bool): Indicates whether the strategy is being initialized for testing. Defaults to False.
        """
        super().__init__(llm=llm, testing=testing)

    @abstractmethod
    def generate(
        self,
        platform: str,
        max_tokens: int,
        top_p: float,
        temperature: float,
        action_space: str,
        observation_type: str,
        max_trajectory_length: int,
        a11y_tree_max_tokens: int,
        observations: List[Any],
        actions: List[Dict[str, Any]],
        thoughts: List[str],
        _system_message: str,
        instruction: str,
        obs: Dict[str, Any],
    ) -> OSWorldBaseOutput:
        """Generates a new step for the agent, including actions and thoughts.

        Args:
            platform (str): The platform type (e.g., "ubuntu" or "windows").
            model (BaseLLM): The language model instance used for response generation.
            max_tokens (int): The maximum number of tokens allowed for a response.
            top_p (float): Top-p sampling parameter for response generation.
            temperature (float): Temperature parameter for response generation.
            action_space (str): The action space available for the agent.
            observation_type (str): The type of observations to process.
            max_trajectory_length (int): The maximum length of the agent's trajectory.
            a11y_tree_max_tokens (int): Maximum tokens for the accessibility tree.
            observations (List[Any]): A list of past observations.
            actions (List[Dict[str, Any]]): A list of past actions.
            thoughts (List[str]): A list of past thoughts.
            _system_message (str): The system message for context.
            instruction (str): The instruction provided to the agent.
            obs (Dict): The current observation.

        Returns:
            OSWorldBaseOutput: The output of the generation process.
        """
        raise NotImplementedError

    @abstractmethod
    def generate_observation(
        self,
        _platform: str,
        observation_type: str,
        max_trajectory_length: int,
        a11y_tree_max_tokens: int,
        observations: List[Any],
        actions: List[Dict[str, Any]],
        thoughts: List[str],
        _system_message: str,
        instruction: str,
        obs: Dict[str, Any],
    ) -> Tuple[List[int], List[Any], List[Dict[str, Any]], List[str]]:
        """Generates a new observation based on the agent's environment and context.

        Args:
            _platform (str): The platform type (e.g., "ubuntu" or "windows").
            observation_type (str): The type of observation to generate.
            max_trajectory_length (int): The maximum length of the agent's trajectory.
            a11y_tree_max_tokens (int): Maximum tokens for the accessibility tree.
            observations (List[Any]): A list of past observations.
            actions (List[Dict[str, Any]]): A list of past actions.
            thoughts (List[str]): A list of past thoughts.
            _system_message (str): The system message for context.
            instruction (str): The instruction provided to the agent.
            obs (Dict): The current observation.

        Returns:
            Tuple[List[int], List[Any], List[Dict[str, Any]], List[str]]: A tuple containing updated observations, actions, thoughts, and additional information.
        """
        raise NotImplementedError

    @abstractmethod
    def generate_thought(
        self,
        payload: Dict[str, Any],
    ) -> Response:
        """Generates a thought for the agent using the provided model and payload.

        Args:
            payload (Dict[str, Any]): A dictionary containing input parameters for the thought generation.
            model (BaseLLM): The language model instance used for thought generation.

        Returns:
            str: The generated thought as a string.
        """
        raise NotImplementedError

    @abstractmethod
    def generate_action(
        self,
        action_space: str,
        observation_type: str,
        actions_list: List[Any],
        response: str,
        masks: List[int],
    ) -> Tuple[List[str], List[Any]]:
        """Generates an action for the agent based on the response and context.

        Args:
            action_space (str): The action space available for the agent.
            observation_type (str): The type of observation to consider.
            actions_list (List[Any]): The list of past actions.
            response (str): The response generated by the model.
            masks (List[int]): A list of masks for parsing actions.

        Returns:
            Tuple[List[str], List[str]]: A tuple containing the generated actions and the updated actions list.
        """
        raise NotImplementedError

    @abstractmethod
    def reset(
        self, actions: List[Dict[str, Any]], thought: List[str], observations: List[Any]
    ) -> Tuple[List[str], List[Dict[str, Any]], List[Any]]:
        """Resets the agent's internal state, including actions, thoughts, and observations.

        Args:
            actions (List[Dict[str, Any]]): The list of past actions to reset.
            thought (List[str]): The list of past thoughts to reset.
            observations (List[Any]): The list of past observations to reset.

        Returns:
            Tuple[List[str], List[Dict[str, Any]], List[Any]]: A tuple containing the reset actions, thoughts, and observations.
        """
        raise NotImplementedError
