import json
import logging
import os
import time
import xml.etree.ElementTree as ET
from http import HTTPStatus
from typing import Dict, List, Any

from dotenv import load_dotenv
import backoff
import dashscope
import google.generativeai as genai
import openai
import requests
from google.api_core.exceptions import InvalidArgument, ResourceExhausted, InternalServerError, BadRequest
from groq import Groq
from requests.exceptions import SSLError

from agential.agents.mm_agents.accessibility_tree_wrap.heuristic_retrieve import filter_nodes, draw_bounding_boxes
from agential.agents.mm_agents.functional import (
    encode_image,
    encoded_img_to_pil_img,
    save_to_tmp_img_file,
    linearize_accessibility_tree,
    tag_screenshot,
    parse_actions_from_string,
    parse_code_from_string,
    parse_code_from_som_string,
    trim_accessibility_tree
)
from agential.agents.mm_agents.prompts import (
    SYS_PROMPT_IN_SCREENSHOT_OUT_CODE, 
    SYS_PROMPT_IN_SCREENSHOT_OUT_ACTION,
    SYS_PROMPT_IN_A11Y_OUT_CODE, 
    SYS_PROMPT_IN_A11Y_OUT_ACTION,
    SYS_PROMPT_IN_BOTH_OUT_CODE, 
    SYS_PROMPT_IN_BOTH_OUT_ACTION,
    SYS_PROMPT_IN_SOM_OUT_TAG
)
from agential.agents.mm_agents.strategies.base import MM_AgentBaseStrategy
from agential.agents.mm_agents.strategies.general import MM_AgentGeneralStrategy

MM_AGENT_STRATEGRIES = {
    "osworld": MM_AgentGeneralStrategy
}

logger = logging.getLogger("desktopenv.agent")

pure_text_settings = ['a11y_tree']

class MMAgent:
    def __init__(
            self,
            platform="ubuntu",
            model="gpt-4-vision-preview",
            max_tokens=1500,
            top_p=0.9,
            temperature=0.5,
            action_space="computer_13",
            observation_type="screenshot_a11y_tree",
            # observation_type can be in ["screenshot", "a11y_tree", "screenshot_a11y_tree", "som"]
            max_trajectory_length=3,
            a11y_tree_max_tokens=10000
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

        self.thoughts = []
        self.actions = []
        self.observations = []

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
            Dict[str, str]: A dictionary of prompt instructions.
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
    def get_strategy(benchmark: str, **kwargs: Any) -> MM_AgentBaseStrategy:
        """Returns an instance of the appropriate ReAct strategy based on the provided benchmark.

        Args:
            benchmark (str): The benchmark name.
            **kwargs (Any): Additional keyword arguments to pass to
                the strategy's constructor.

        Returns:
            ReActBaseStrategy: An instance of the appropriate ReAct strategy.
        """
        if benchmark not in MM_AGENT_STRATEGRIES:
            raise ValueError(f"Unsupported benchmark: {benchmark} for agent ReAct")

        strategy = MM_AGENT_STRATEGRIES[benchmark]
        return strategy(**kwargs)

    def generate(
        self,
        instruction: str, 
        obs: Dict
    ) -> Any:
        """Processes a given question through ReAct.

        Iteratively applies the think-act-observe cycle to generate an answer for the question.
        The process continues until the operation is halted based on certain conditions.

        Args:
            question (str): The question to be processed.
            examples (str, optional): Fewshot examples. Defaults to "".
            prompt (str, optional): Prompt template string. Defaults to "".
            additional_keys (Dict[str, str]): Additional keys to format the prompt. Defaults to {}.
            fewshot_type (str): The type of few-shot examples to use. Defaults to "".
            reset (bool, optional): Whether to reset the internal state before processing. Defaults to True.

        Returns:
            ReActOutput: The list of accumulated output from the ReAct process,
                each ReActOutput consists of a thought, action type/query, observation, answer, and external tool info.
        """
        if not prompt:
            prompt = MMAgent.get_prompts()
            # examples = fewshots["examples"]
            # prompt = prompts["prompt"]

        system_message = self.get_prompts()

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

        return response, actions
