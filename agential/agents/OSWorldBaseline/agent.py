"""OSWorldBaseline Agent.

Original Paper: https://arxiv.org/abs/2404.07972
Paper Repository: https://github.com/xlang-ai/OSWorld/tree/main
"""

import logging
from typing import Dict, Any, Tuple, List

import backoff
import openai
from google.api_core.exceptions import InvalidArgument, ResourceExhausted, InternalServerError, BadRequest
from requests.exceptions import SSLError

from agential.agents.OSWorldBaseline.prompts import (
    SYS_PROMPT_IN_SCREENSHOT_OUT_CODE, 
    SYS_PROMPT_IN_SCREENSHOT_OUT_ACTION,
    SYS_PROMPT_IN_A11Y_OUT_CODE, 
    SYS_PROMPT_IN_A11Y_OUT_ACTION,
    SYS_PROMPT_IN_BOTH_OUT_CODE, 
    SYS_PROMPT_IN_BOTH_OUT_ACTION,
    SYS_PROMPT_IN_SOM_OUT_TAG
)
from agential.agents.OSWorldBaseline.strategies.base import OSWorldBaselineAgentBaseStrategy
from agential.agents.OSWorldBaseline.strategies.general import OSWorldBaselineAgentGeneralStrategy

OSWORLDBASELINEAGENT_STRATEGRIES = {
    "osworld": OSWorldBaselineAgentGeneralStrategy
}

logger = logging.getLogger("desktopenv.agent")

pure_text_settings = ['a11y_tree']

class OSWorldBaselineAgent:
    def __init__(
            self,
            platform: str = "ubuntu",
            model: str = "gpt-4-vision-preview",
            max_tokens: int = 1500,
            top_p: float = 0.9,
            temperature: float = 0.5,
            action_space: str = "computer_13",
            observation_type: str = "screenshot_a11y_tree",
            # observation_type can be in ["screenshot", "a11y_tree", "screenshot_a11y_tree", "som"]
            max_trajectory_length: int = 3,
            a11y_tree_max_tokens:int = 10000,
            testing: bool = False,
            **strategy_kwargs: Any,
            # strategy_kwargs is a dictinary
    ):
        self.platform = platform
        self.model = model
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.temperature = temperature
        self.action_space = action_space
        self.observation_type = observation_type
        self.max_trajectory_length = max_trajectory_length
        self.a11y_tree_max_tokens = a11y_tree_max_tokens
        self.testing=testing

        self.thoughts: List= []
        self.actions: List= []
        self.observations: List = []

        self.strategy = OSWorldBaselineAgent.get_strategy(
            benchmark="osworld",
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
        max_tries=10
    )

    def get_prompts(self) -> str:
        """Retrieve the prompt instruction based on the benchmark.

        Args:
            benchmark (str): The benchmark name.
            **kwargs (Any): Additional arguments.

        Returns:
            str: A prompt instruction.
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
        """Returns an instance of the appropriate ReAct strategy based on the provided benchmark.

        Args:
            benchmark (str): The benchmark name.
            **kwargs (Any): Additional keyword arguments to pass to
                the strategy's constructor.

        Returns:
            OSWorldBaselineAgentBaseStrategy: An instance of the appropriate ReAct strategy.
        """
        if benchmark not in OSWORLDBASELINEAGENT_STRATEGRIES:
            raise ValueError(f"Unsupported benchmark: {benchmark} for agent ReAct")

        strategy = OSWORLDBASELINEAGENT_STRATEGRIES[benchmark]
        return strategy(**kwargs)

    def generate(
        self,
        instruction: str, 
        obs: Dict,
        prompt: str = ""
    ) -> Tuple[str, str, List]:
        """Processes a given question through ReAct.

        Iteratively applies the think-act-observe cycle to generate an answer for the question.
        The process continues until the operation is halted based on certain conditions.

        Args:
            instruction (str): Instruct agent what to do
            obs (Dict[str, str]): Observation of the environments.
            prompt (str, optional): Prompt template string. Defaults to "".

        Returns:
            Tuple[str, str]: The response from agent and actions that will be taken next.
        """
        if not prompt:
            prompt = self.get_prompts()

        response, actions, self.actions, self.thoughts, self.observations, messages= self.strategy.generate(
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
            obs=obs
        )

        return response, actions, messages
