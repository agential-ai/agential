"""WebVoyagerBaseline Agent.

Original Paper: https://arxiv.org/abs/2401.13919
Paper Repository: https://github.com/MinorJerry/WebVoyager
"""

from typing import Any, Dict, List

from agential.agents.base.agent import BaseAgent
from agential.agents.computer_use.webvoyager_baseline.output import WebVoyagerBaseOutput
from agential.agents.computer_use.webvoyager_baseline.prompts import (
    SYSTEM_PROMPT,
    SYSTEM_PROMPT_TEXT_ONLY,
)
from agential.agents.computer_use.webvoyager_baseline.strategies.base import (
    WebVoyagerBaseStrategy,
)
from agential.agents.computer_use.webvoyager_baseline.strategies.general import (
    WebVoyagerGeneralStrategy,
)
from agential.core.llm import LLM, BaseLLM

WEBVOYAGER_BASELINE_AGENT_STRATEGIES = {"webvoyager": WebVoyagerGeneralStrategy}


class WebVoyagerBaseline(BaseAgent):
    """An agent designed for WebVoyager environments, capable of processing observations and generating actions.

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
        strategy (WebVoyagerBaseStrategy): The strategy used by the agent.
        thoughts (List): Accumulated thoughts during the agent's operation.
        actions (List): Actions taken by the agent.
        observations (List): Observations received by the agent.
    """

    def __init__(  ###### Clean Up Attributes ############
        self,
        output_dir: str,
        download_dir: str,
        test_file: str = "data/test.json",
        max_iter: int = 5,
        seed: int = None,
        max_attached_imgs: int = 1,
        temperature: float = 1.0,
        text_only: bool = False,
        headless: bool = False,
        save_accessibility_tree: bool = False,
        force_device_scale: bool = False,
        window_width: int = 1024,
        window_height: int = 768,
        fix_box_color: bool = False,
        llm: BaseLLM = LLM(model="gpt-4o"),
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

        self.strategy = WebVoyagerBaseline.get_strategy(
            benchmark=self.benchmark,
            llm=self.llm,
            testing=self.testing,
            **strategy_kwargs,
        )

    def get_prompts(self, textonly: bool, benchmark: str = "", **kwargs: Any) -> str:
        """Retrieve the appropriate system prompt based on the observation type and action space.

        Returns:
            str: The system prompt for the agent.

        Raises:
            ValueError: If the action space or observation type is invalid.
        """
        if textonly:
            return SYSTEM_PROMPT_TEXT_ONLY
        else:
            return SYSTEM_PROMPT

    @staticmethod
    def get_strategy(benchmark: str, **kwargs: Any) -> WebVoyagerBaseStrategy:
        """Returns the strategy corresponding to the benchmark.

        Args:
            benchmark (str): The benchmark name.
            **kwargs (Any): Additional arguments for the strategy.

        Returns:
            WebVoyagerBaseStrategy: The strategy instance.

        Raises:
            ValueError: If the benchmark is unsupported.
        """
        if benchmark not in WEBVOYAGER_BASELINE_AGENT_STRATEGIES:
            raise ValueError(f"Unsupported benchmark: {benchmark} for agent ReAct")

        strategy = WEBVOYAGER_BASELINE_AGENT_STRATEGIES[benchmark]
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
    ) -> WebVoyagerBaseOutput:
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
            prompt_text_only = self.get_prompts(textonly=True)["prompt"]
            prompt = self.get_prompts(textonly=False)["prompt"]

        webvoyager_base_output: WebVoyagerBaseOutput = self.strategy.generate(
            system_prompt,
            system_prompt_text_only,
            output_dir,
            download_dir,
            test_file="data/test.json",
            max_iter=5,
            seed=None,
            max_attached_imgs=1,
            temperature=1.0,
            text_only=False,
            headless=False,
            save_accessibility_tree=False,
            force_device_scale=False,
            window_width=1024,
            window_height=768,
            fix_box_color=False,
        )

        return webvoyager_base_output
