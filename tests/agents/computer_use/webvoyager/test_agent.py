"""Unit tests for the WebVoyager Baseline Agent."""

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
        seed: int = None,
        max_attached_imgs: int = 1,
        temperature: float = 1.0,
        text_only: bool = False,
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
        self.seed = seed
        self.max_attached_imgs = max_attached_imgs
        self.temperature = temperature
        self.text_only = text_only
        self.llm = llm
        self.testing = testing
        self.benchmark = benchmark

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
        self, obs: Dict[str, Any], task: Dict[str, Any], prompt: str = ""
    ) -> WebVoyagerBaseOutput:
        """Processes a given instruction and observations to generate a response.

        Args:
            instruction (str): Instruction for the agent.
            obs (Dict[str, Any]): Observations from the environment.
            task (Dict[str, Any]): Task to generate action for.
            prompt (str, optional): Predefined prompt for the agent. Defaults to "".

        Returns:
            Tuple[str, List, List]: A response from the agent, the list of actions,
                and additional messages.
        """
        if not prompt:
            system_prompt_text_only = self.get_prompts(textonly=True)["prompt"]
            system_prompt = self.get_prompts(textonly=False)["prompt"]

        webvoyager_base_output: WebVoyagerBaseOutput = self.strategy.generate(
            system_prompt=system_prompt,
            system_prompt_text_only=system_prompt_text_only,
            seed=self.seed,
            max_attached_imgs=self.max_attached_imgs,
            temperature=self.temperature,
            text_only=self.text_only,
            task=task,
            obs=obs
        )

        return webvoyager_base_output
