"""Handlers for Reflexion agent benchmarks (QA, Math, Code, Reflection).

Each handler encapsulates prompt building, action parsing, observation, and reflection handling.
"""

from typing import Any, Dict, Tuple, Optional
from abc import ABC, abstractmethod
import re
from agential.utils.general import safe_execute

from agential.core.llm import BaseLLM, Response

# Base handler class
class ReflexionHandler(ABC):
    def __init__(self, llm: BaseLLM, max_steps: int = 6, max_reflections: int = 3, testing: bool = False):
        self.llm = llm
        self.max_steps = max_steps
        self.max_reflections = max_reflections
        self.testing = testing

    @abstractmethod
    def get_prompt(self) -> str:
        pass

    @abstractmethod
    def parse_action(self, action: str) -> Tuple[str, str]:
        pass

    @abstractmethod
    def handle_observation(self, action_type: str, query: str, scratchpad: str) -> Tuple[str, str, bool, Dict[str, Any]]:
        pass

    @abstractmethod
    def handle_reflection(self, scratchpad: str, reflect_strategy: str, question: str, examples: str, prompt: str, additional_keys: Dict[str, str]) -> Tuple[list, str, Optional[Response]]:
        pass

    def reset(self) -> None:
        pass

# QA Handler
class QAHandler(ReflexionHandler):
    def get_prompt(self) -> str:
        return ""
    def parse_action(self, action: str) -> Tuple[str, str]:
        return "", ""
    def handle_observation(self, action_type: str, query: str, scratchpad: str) -> Tuple[str, str, bool, Dict[str, Any]]:
        return "", "", False, {}
    def handle_reflection(self, scratchpad: str, reflect_strategy: str, question: str, examples: str, prompt: str, additional_keys: Dict[str, str]) -> Tuple[list, str, Optional[Response]]:
        return [], "", None

# Math Handler
class MathHandler(ReflexionHandler):
    def get_prompt(self) -> str:
        return "Solve the following math problem step by step."
    def parse_action(self, action: str) -> Tuple[str, str]:
        # Parse actions like Calculate[code] or Finish[answer]
        pattern = r"^(\w+)\[(.+)\]$"
        match = re.match(pattern, action)
        return (match.group(1), match.group(2)) if match else ("", "")
    def handle_observation(self, action_type: str, query: str, scratchpad: str) -> Tuple[str, str, bool, Dict[str, Any]]:
        answer = ""
        finished = False
        external_tool_info = {}
        if action_type.lower() == "finish":
            answer = query
            finished = True
            obs = query
        elif action_type.lower() == "calculate":
            code_answer, execution_status = safe_execute(query)
            external_tool_info["code_answer"] = code_answer[0]
            external_tool_info["execution_status"] = execution_status
            answer = code_answer[0]
            obs = f"Execution Status: {execution_status}\nOutput: answer = {code_answer[0]}"
        else:
            obs = "Invalid Action. Valid Actions are Calculate[code] and Finish[answer]."
        return obs, answer, finished, external_tool_info
    def handle_reflection(self, scratchpad: str, reflect_strategy: str, question: str, examples: str, prompt: str, additional_keys: Dict[str, str]) -> Tuple[list, str, Optional[Response]]:
        # Dummy reflection for now
        return ["No reflection implemented."], "No reflection implemented.", None

# Code Handler
class CodeHandler(ReflexionHandler):
    def get_prompt(self) -> str:
        return ""
    def parse_action(self, action: str) -> Tuple[str, str]:
        return "", ""
    def handle_observation(self, action_type: str, query: str, scratchpad: str) -> Tuple[str, str, bool, Dict[str, Any]]:
        return "", "", False, {}
    def handle_reflection(self, scratchpad: str, reflect_strategy: str, question: str, examples: str, prompt: str, additional_keys: Dict[str, str]) -> Tuple[list, str, Optional[Response]]:
        return [], "", None

# Reflection Handler (for memory/reflection steps)
class ReflectionHandler(ReflexionHandler):
    def get_prompt(self) -> str:
        return ""
    def parse_action(self, action: str) -> Tuple[str, str]:
        return "", ""
    def handle_observation(self, action_type: str, query: str, scratchpad: str) -> Tuple[str, str, bool, Dict[str, Any]]:
        return "", "", False, {}
    def handle_reflection(self, scratchpad: str, reflect_strategy: str, question: str, examples: str, prompt: str, additional_keys: Dict[str, str]) -> Tuple[list, str, Optional[Response]]:
        return [], "", None

# Registry for handlers
BENCHMARK_HANDLERS = {
    "hotpotqa": QAHandler,
    "fever": QAHandler,
    "triviaqa": QAHandler,
    "ambignq": QAHandler,
    "gsm8k": MathHandler,
    "svamp": MathHandler,
    "tabmwp": MathHandler,
    "humaneval": CodeHandler,
    "mbpp": CodeHandler,
    # Add more as needed
} 