"""Reflexion Agent strategies for Code."""

from typing import Any, Dict, List, Optional, Tuple

import tiktoken

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
        self, action_type: str, query: str, key: str
    ) -> Tuple[bool, str]:
        pass

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
