"""Reflexion Agent strategies for Code."""

from typing import Any, Dict, List, Optional, Tuple

import tiktoken
import re

from langchain_core.language_models.chat_models import BaseChatModel
from tiktoken.core import Encoding

from agential.cog.modules.reflect.reflexion import (
    ReflexionCoTReflector,
    ReflexionReActReflector,
)
from agential.cog.strategies.reflexion.base import (
    ReflexionCoTBaseStrategy,
    ReflexionReActBaseStrategy,
)
from agential.cog.functional.reflexion import (
    _is_halted,
    _prompt_cot_agent,
    _prompt_react_agent,
    _truncate_scratchpad,
)
from agential.utils.parse import remove_newline
from agential.utils.general import safe_execute


def parse_code_action_cot(action: str) -> Tuple[str, str]:
    """Parses an action string to extract the action type and code content.

    Identifies action types (`Finish`) and extracts the
    corresponding code content enclosed within Markdown-style code blocks.
    The action type is case-insensitive and the code content is trimmed of
    leading and trailing whitespace.

    Args:
        action (str): The action string containing the action type and code content.

    Returns:
        Tuple[str, str]: A tuple containing the extracted action type (capitalized)
        and the extracted code content.
    """
    action_split = action.split("```python", maxsplit=1)
    match = re.search(r"\b(Finish)\b", action_split[0], re.IGNORECASE)

    action_type = match.group(0).lower().capitalize() if match else ""
    try:
        query = action_split[1].split("```")[0].strip() if action_type else ""
    except:
        action_type = ""
        query = ""

    return action_type, query


class ReflexionCoTCodeStrategy(ReflexionCoTBaseStrategy):
    def __init__(
        self,
        llm: BaseChatModel,
        reflector: Optional[ReflexionCoTReflector] = None,
        max_reflections: int = 3,
        max_trials: int = 1,
    ) -> None:
        """Initialization."""
        super().__init__(llm)
        self.llm = llm
        self.max_reflections = max_reflections
        self.max_trials = max_trials

        if not reflector:
            reflector = ReflexionCoTReflector(llm=llm, max_reflections=max_reflections)
        self.reflector = reflector

        self._scratchpad = ""
        self._finished = False
        self._answer = ""

    def generate(
        self,
        question: str,
        examples: str,
        reflections: str,
        prompt: str,
        additional_keys: Dict[str, str],
        **kwargs: Any,
    ) -> str:
        """Generates a thought based on the question, examples, and prompt.

        Args:
            question (str): The question to be answered.
            examples (str): Examples to guide the generation process.
            reflections (str): Reflections to consider during generation.
            prompt (str): The prompt used for generating the thought.
            additional_keys (Dict[str, str]): Additional keys for the generation process.
            **kwargs (Any): Additional arguments.

        Returns:
            str: The generated thought.
        """
        self._scratchpad += "\nThought:"
        thought = _prompt_cot_agent(
            llm=self.llm,
            examples=examples,
            reflections=reflections,
            question=question,
            scratchpad=self._scratchpad,
            prompt=prompt,
            additional_keys=additional_keys,
        )
        thought = remove_newline(thought).split("Action")[0].strip()
        self._scratchpad += " " + thought

        return thought
    
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
            reflections (str): Reflections to consider during generation.
            prompt (str): The prompt used for generating the action.
            additional_keys (Dict[str, str]): Additional keys for the generation process.
            **kwargs (Any): Additional arguments.

        Returns:
            Tuple[str, str]: The generated action type and query.
        """
        self._scratchpad += "\nAction:"
        action = _prompt_cot_agent(
            llm=self.llm,
            examples=examples,
            reflections=reflections,
            question=question,
            scratchpad=self._scratchpad,
            prompt=prompt,
            additional_keys=additional_keys,
        )
        action = action.split("Observation")[0].strip()

        action_type, query = parse_code_action_cot(action)
        self._scratchpad += f" {action_type}[\n```python\n{query}\n```\n]"

        return action_type, query
    
    def generate_observation(
        self, action_type: str, query: str, key: str, additional_keys: Dict[str, str]
    ) -> Tuple[bool, str]:
        _, execution_status = safe_execute(query)

        # self._scratchpad += f"\nObservation: "
        # if action_type.lower() == "finish":
        #     self._finished = True
        #     self._answer = query
        #     if EM(answer[0], key, normalize=False):
        #         obs = "Answer is CORRECT"
        #     else:
        #         obs = "Answer is INCORRECT"
        # else:
        #     obs = "Invalid action type, please try again."
        # self._scratchpad += obs

        # return EM(answer[0], key, normalize=False), obs
    
    def create_output_dict(
        self,
        thought: str,
        action_type: str,
        obs: str,
        is_correct: bool,
        reflections: List[str],
    ) -> Dict[str, Any]:
        pass

    def halting_condition(self, idx: int, key: str, **kwargs: Any) -> bool:
        pass

    def reset(self, **kwargs: Any) -> None:
        pass

    def reflect(
        self,
        reflect_strategy: str,
        question: str,
        examples: str,
        prompt: str,
        additional_keys: Dict[str, str],
    ) -> Tuple[List[str], str]:
        pass

    def reflect_condition(
        self, idx: int, reflect_strategy: Optional[str], key: str
    ) -> bool:
        pass


class ReflexionReActCodeStrategy(ReflexionReActBaseStrategy):
    def __init__(
        self,
        llm: BaseChatModel,
        reflector: Optional[ReflexionReActReflector] = None,
        max_reflections: int = 3,
        max_trials: int = 1,
        max_steps: int = 6,
        max_tokens: int = 5000,
        enc: Encoding = tiktoken.encoding_for_model("gpt-3.5-turbo"),
    ) -> None:
        pass

    def generate(
        self,
        question: str,
        examples: str,
        reflections: str,
        prompt: str,
        additional_keys: Dict[str, str],
        **kwargs: Any,
    ) -> str:
        pass

    def generate_action(
        self,
        question: str,
        examples: str,
        reflections: str,
        prompt: str,
        additional_keys: Dict[str, str],
        **kwargs: Any,
    ) -> Tuple[str, str]:
        pass

    def generate_observation(
        self, step_idx: int, action_type: str, query: str, key: str
    ) -> Tuple[bool, str, Dict[str, Any]]:
        pass

    def create_output_dict(
        self, react_out: List[Dict[str, Any]], reflections: List[str]
    ) -> Dict[str, Any]:
        pass

    def react_create_output_dict(
        self,
        thought: str,
        action_type: str,
        query: str,
        obs: str,
        external_tool_info: Dict[str, Any],
        is_correct: bool,
    ) -> Dict[str, Any]:
        pass

    def halting_condition(self, idx: int, key: str, **kwargs: Any) -> bool:
        pass

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
        pass

    def reset(self, **kwargs: Any) -> None:
        pass

    def reflect(
        self,
        reflect_strategy: str,
        question: str,
        examples: str,
        prompt: str,
        additional_keys: Dict[str, str],
    ) -> Tuple[List[str], str]:
        pass

    def reflect_condition(
        self,
        step_idx: int,
        reflect_strategy: Optional[str],
        question: str,
        examples: str,
        key: str,
        prompt: str,
        additional_keys: Dict[str, str],
        **kwargs: Dict[str, str],
    ) -> bool:
        pass


class ReflexionCoTHEvalStrategy(ReflexionCoTCodeStrategy):
    """A strategy class for the HumanEval benchmark using the ReflexionCoT agent."""

    pass


class ReflexionCoTMBPPStrategy(ReflexionCoTCodeStrategy):
    """A strategy class for the MBPP benchmark using the ReflexionCoT agent."""

    pass


class ReflexionReActHEvalStrategy(ReflexionReActCodeStrategy):
    """A strategy class for the HumanEval benchmark using the ReflexionReAct agent."""

    pass


class ReflexionReActMBPPStrategy(ReflexionReActCodeStrategy):
    """A strategy class for the MBPP benchmark using the ReflexionReAct agent."""

    pass
