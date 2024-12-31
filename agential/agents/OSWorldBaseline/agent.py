"""OSWorldBaseline Agent.

Original Paper: https://arxiv.org/abs/2404.07972
Paper Repository: https://github.com/xlang-ai/OSWorld/tree/main
"""

import logging

from typing import Any, Dict, List, Tuple

import backoff
import openai

from google.api_core.exceptions import (
    BadRequest,
    InternalServerError,
    InvalidArgument,
    ResourceExhausted,
)
from requests.exceptions import SSLError

from agential.agents.OSWorldBaseline.prompts import (
    SYS_PROMPT_IN_A11Y_OUT_ACTION,
    SYS_PROMPT_IN_A11Y_OUT_CODE,
    SYS_PROMPT_IN_BOTH_OUT_ACTION,
    SYS_PROMPT_IN_BOTH_OUT_CODE,
    SYS_PROMPT_IN_SCREENSHOT_OUT_ACTION,
    SYS_PROMPT_IN_SCREENSHOT_OUT_CODE,
    SYS_PROMPT_IN_SOM_OUT_TAG,
)
from agential.agents.OSWorldBaseline.strategies.base import (
    OSWorldBaselineAgentBaseStrategy,
)
from agential.agents.OSWorldBaseline.strategies.general import (
    OSWorldBaselineAgentGeneralStrategy,
)
from agential.core.llm import LLM, BaseLLM

OSWORLDBASELINEAGENT_STRATEGRIES = {"osworld": OSWorldBaselineAgentGeneralStrategy}

logger = logging.getLogger("desktopenv.agent")

pure_text_settings = ["a11y_tree"]


class OSWorldBaselineAgent:
    """An agent designed for OSWorld environments, capable of processing observations and generating actions.

    Attributes:
        platform (str): The platform on which the agent operates (e.g., 'ubuntu').
        model (BaseLLM): The language model used for generating responses and processing instructions.
        max_tokens (int): Maximum tokens for the response.
        top_p (float): Probability mass for nucleus sampling.
        temperature (float): Temperature parameter for controlling randomness.
        action_space (str): The available action space for the agent.
        observation_type (str): The type of observation provided (e.g., 'screenshot', 'a11y_tree').
        max_trajectory_length (int): Maximum steps allowed in a trajectory.
        a11y_tree_max_tokens (int): Maximum tokens for accessibility tree observations.
        testing (bool): If the agent is in testing mode.
        benchmark (str): The benchmark name the agent is designed for.
        strategy (OSWorldBaselineAgentBaseStrategy): The strategy used by the agent.
        thoughts (List): Accumulated thoughts during the agent's operation.
        actions (List): Actions taken by the agent.
        observations (List): Observations received by the agent.
    """

    def __init__(
        self,
        platform: str = "ubuntu",
        model: BaseLLM = LLM(model="gpt-4o"),
        max_tokens: int = 1500,
        top_p: float = 0.9,
        temperature: float = 0.5,
        action_space: str = "computer_13",
        observation_type: str = "screenshot_a11y_tree",
        max_trajectory_length: int = 3,
        a11y_tree_max_tokens: int = 10000,
        testing: bool = False,
        benchmark: str = "osworld",
        **strategy_kwargs: Any,
    ):
        """Initializes the OSWorldBaselineAgent.

        Args:
            platform (str): The platform on which the agent operates.
            model (BaseLLM): The language model instance.
            max_tokens (int): Maximum number of tokens for responses.
            top_p (float): Nucleus sampling probability.
            temperature (float): Sampling temperature.
            action_space (str): The action space type.
            observation_type (str): The type of observations.
            max_trajectory_length (int): Maximum number of steps in a trajectory.
            a11y_tree_max_tokens (int): Maximum tokens for accessibility tree observations.
            testing (bool): Whether the agent is in testing mode.
            benchmark (str): The benchmark for this agent.
            **strategy_kwargs (Any): Additional arguments for the strategy.
        """
        self.platform = platform
        self.model = model
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.temperature = temperature
        self.action_space = action_space
        self.observation_type = observation_type
        self.max_trajectory_length = max_trajectory_length
        self.a11y_tree_max_tokens = a11y_tree_max_tokens
        self.testing = testing
        self.benchmark = benchmark

        self.thoughts: List = []
        self.actions: List = []
        self.observations: List = []

        self.strategy = OSWorldBaselineAgent.get_strategy(
            benchmark=self.benchmark,
            testing=self.testing,
            **strategy_kwargs,
        )

    @backoff.on_exception(
        backoff.constant,
        # here you should add more model exceptions as you want,
        # but you are forbidden to add "Exception", that is, a common type of exception
        # because we want to catch this kind of Exception in the outside to ensure each example won't exceed the time limit
        (
            # General exceptions
            SSLError,
            # OpenAI exceptions
            openai.RateLimitError,
            openai.BadRequestError,
            openai.InternalServerError,
            # Google exceptions
            InvalidArgument,
            ResourceExhausted,
            InternalServerError,
            BadRequest,
            # Groq exceptions
            # todo: check
        ),
        interval=30,
        max_tries=10,
    )
    def get_prompts(self) -> str:
        """Retrieve the appropriate system prompt based on the observation type and action space.

        Returns:
            str: The system prompt for the agent.

        Raises:
            ValueError: If the action space or observation type is invalid.
        """
        if self.observation_type == "screenshot":
            if self.action_space == "computer_13":
                return SYS_PROMPT_IN_SCREENSHOT_OUT_ACTION
            elif self.action_space == "pyautogui":
                return SYS_PROMPT_IN_SCREENSHOT_OUT_CODE
            else:
                raise ValueError("Invalid action space: " + self.action_space)
        elif self.observation_type == "a11y_tree":
            if self.action_space == "computer_13":
                return SYS_PROMPT_IN_A11Y_OUT_ACTION
            elif self.action_space == "pyautogui":
                return SYS_PROMPT_IN_A11Y_OUT_CODE
            else:
                raise ValueError("Invalid action space: " + self.action_space)
        elif self.observation_type == "screenshot_a11y_tree":
            if self.action_space == "computer_13":
                return SYS_PROMPT_IN_BOTH_OUT_ACTION
            elif self.action_space == "pyautogui":
                return SYS_PROMPT_IN_BOTH_OUT_CODE
            else:
                raise ValueError("Invalid action space: " + self.action_space)
        elif self.observation_type == "som":
            if self.action_space == "computer_13":
                raise ValueError("Invalid action space: " + self.action_space)
            elif self.action_space == "pyautogui":
                return SYS_PROMPT_IN_SOM_OUT_TAG
            else:
                raise ValueError("Invalid action space: " + self.action_space)
        else:
            raise ValueError("Invalid experiment type: " + self.observation_type)

    @staticmethod
    def get_strategy(benchmark: str, **kwargs: Any) -> OSWorldBaselineAgentBaseStrategy:
        """Returns the strategy corresponding to the benchmark.

        Args:
            benchmark (str): The benchmark name.
            **kwargs (Any): Additional arguments for the strategy.

        Returns:
            OSWorldBaselineAgentBaseStrategy: The strategy instance.

        Raises:
            ValueError: If the benchmark is unsupported.
        """
        if benchmark not in OSWORLDBASELINEAGENT_STRATEGRIES:
            raise ValueError(f"Unsupported benchmark: {benchmark} for agent ReAct")

        strategy = OSWORLDBASELINEAGENT_STRATEGRIES[benchmark]
        return strategy(**kwargs)

    def generate(
        self, instruction: str, obs: Dict, prompt: str = ""
    ) -> Tuple[str, List, List]:
        """Processes a given instruction and observations to generate a response.

        Args:
            instruction (str): Instruction for the agent.
            obs (Dict): Observations from the environment.
            prompt (str, optional): Predefined prompt for the agent. Defaults to "".

        Returns:
            Tuple[str, List, List]: A response from the agent, the list of actions,
                and additional messages.
        """
        if not prompt:
            prompt = self.get_prompts()

        response, actions, self.actions, self.thoughts, self.observations, messages = (
            self.strategy.generate(
                platform=self.platform,
                model=self.model,
                max_tokens=self.max_tokens,
                top_p=self.top_p,
                temperature=self.temperature,
                action_space=self.action_space,
                observation_type=self.observation_type,
                max_trajectory_length=self.max_trajectory_length,
                a11y_tree_max_tokens=self.a11y_tree_max_tokens,
                observations=self.observations,
                actions=self.actions,
                thoughts=self.thoughts,
                _system_message=prompt,
                instruction=instruction,
                obs=obs,
            )
        )

        return response, actions, messages
