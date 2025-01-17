"""OSWorldBaseline Agent.

Original Paper: https://arxiv.org/abs/2404.07972
Paper Repository: https://github.com/xlang-ai/OSWorld/tree/main
"""

from typing import Any, Dict, List

from agential.agents.base.agent import BaseAgent
from agential.agents.computer_use.osworld_baseline.output import OSWorldBaseOutput
from agential.agents.computer_use.osworld_baseline.prompts import (
    SYS_PROMPT_IN_A11Y_OUT_ACTION,
    SYS_PROMPT_IN_A11Y_OUT_CODE,
    SYS_PROMPT_IN_BOTH_OUT_ACTION,
    SYS_PROMPT_IN_BOTH_OUT_CODE,
    SYS_PROMPT_IN_SCREENSHOT_OUT_ACTION,
    SYS_PROMPT_IN_SCREENSHOT_OUT_CODE,
    SYS_PROMPT_IN_SOM_OUT_TAG,
)
from agential.agents.computer_use.osworld_baseline.strategies.base import (
    OSWorldBaseStrategy,
)
from agential.agents.computer_use.osworld_baseline.strategies.general import (
    OSWorldBaseGeneralStrategy,
)
from agential.core.llm import LLM, BaseLLM

OSWORLD_BASELINE_AGENT_STRATEGIES = {"osworld": OSWorldBaseGeneralStrategy}

pure_text_settings = ["a11y_tree"]


class OSWorldBaseline(BaseAgent):
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
        strategy (OSWorldBaseStrategy): The strategy used by the agent.
        thoughts (List): Accumulated thoughts during the agent's operation.
        actions (List): Actions taken by the agent.
        observations (List): Observations received by the agent.
    """

    def __init__(
        self,
        platform: str = "ubuntu",
        llm: BaseLLM = LLM(model="gpt-4o"),
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
        """Initializes the OSWorldBaseline.

        Args:
            platform (str): The platform on which the agent operates.
            llm (BaseLLM): The language model instance.
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
        super().__init__(llm=llm, benchmark=benchmark, testing=testing)
        self.platform = platform
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.temperature = temperature
        self.action_space = action_space
        self.observation_type = observation_type
        self.max_trajectory_length = max_trajectory_length
        self.a11y_tree_max_tokens = a11y_tree_max_tokens

        self.thoughts: List = []
        self.actions: List = []
        self.observations: List = []

        self.strategy = OSWorldBaseline.get_strategy(
            benchmark=self.benchmark,
            llm=self.llm,
            testing=self.testing,
            **strategy_kwargs,
        )

    def get_prompts(self, benchmark: str = "", **kwargs: Any) -> Dict[str, str]:
        """Retrieve the appropriate system prompt based on the observation type and action space.

        Returns:
            str: The system prompt for the agent.

        Raises:
            ValueError: If the action space or observation type is invalid.
        """
        if self.observation_type == "screenshot":
            if self.action_space == "computer_13":
                return {"prompt": SYS_PROMPT_IN_SCREENSHOT_OUT_ACTION}
            elif self.action_space == "pyautogui":
                return {"prompt": SYS_PROMPT_IN_SCREENSHOT_OUT_CODE}
            else:
                raise ValueError("Invalid action space: " + self.action_space)
        elif self.observation_type == "a11y_tree":
            if self.action_space == "computer_13":
                return {"prompt": SYS_PROMPT_IN_A11Y_OUT_ACTION}
            elif self.action_space == "pyautogui":
                return {"prompt": SYS_PROMPT_IN_A11Y_OUT_CODE}
            else:
                raise ValueError("Invalid action space: " + self.action_space)
        elif self.observation_type == "screenshot_a11y_tree":
            if self.action_space == "computer_13":
                return {"prompt": SYS_PROMPT_IN_BOTH_OUT_ACTION}
            elif self.action_space == "pyautogui":
                return {"prompt": SYS_PROMPT_IN_BOTH_OUT_CODE}
            else:
                raise ValueError("Invalid action space: " + self.action_space)
        elif self.observation_type == "som":
            if self.action_space == "computer_13":
                raise ValueError("Invalid action space: " + self.action_space)
            elif self.action_space == "pyautogui":
                return {"prompt": SYS_PROMPT_IN_SOM_OUT_TAG}
            else:
                raise ValueError("Invalid action space: " + self.action_space)
        else:
            raise ValueError("Invalid experiment type: " + self.observation_type)

    @staticmethod
    def get_strategy(benchmark: str, **kwargs: Any) -> OSWorldBaseStrategy:
        """Returns the strategy corresponding to the benchmark.

        Args:
            benchmark (str): The benchmark name.
            **kwargs (Any): Additional arguments for the strategy.

        Returns:
            OSWorldBaseStrategy: The strategy instance.

        Raises:
            ValueError: If the benchmark is unsupported.
        """
        if benchmark not in OSWORLD_BASELINE_AGENT_STRATEGIES:
            raise ValueError(f"Unsupported benchmark: {benchmark} for agent ReAct")

        strategy = OSWORLD_BASELINE_AGENT_STRATEGIES[benchmark]
        return strategy(**kwargs)

    @staticmethod
    def get_fewshots(
        benchmark: str = "", fewshot_type: str = "", **kwargs: Any
    ) -> Dict[str, str]:
        """Retrieve few-shot examples based on the benchmark.

        Args:
            benchmark (str): The benchmark name.
            fewshot_type (str): The benchmark few-shot type.
            **kwargs (Any): Additional arguments.

        Returns:
            Dict[str, str]: A dictionary of few-shot examples.
        """
        return {"benchmark": benchmark, "fewshot_type": fewshot_type}

    def generate(
        self, instruction: str, obs: Dict[str, Any], prompt: str = ""
    ) -> OSWorldBaseOutput:
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
            prompt = self.get_prompts()["prompt"]

        osworld_base_output: OSWorldBaseOutput = self.strategy.generate(
            platform=self.platform,
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

        return osworld_base_output
