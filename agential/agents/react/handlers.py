"""Benchmark handlers for the ReAct agent.

This module contains the plugin-based handlers that make it easy to extend
ReAct to new benchmarks without modifying the core agent logic.
"""

import re
from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple

from agential.agents.react.prompts import (
    REACT_INSTRUCTION_AMBIGNQ,
    REACT_INSTRUCTION_FEVER,
    REACT_INSTRUCTION_GSM8K,
    REACT_INSTRUCTION_HOTPOTQA,
    REACT_INSTRUCTION_HUMANEVAL,
    REACT_INSTRUCTION_MBPP,
    REACT_INSTRUCTION_SVAMP,
    REACT_INSTRUCTION_TABMWP,
    REACT_INSTRUCTION_TRIVIAQA,
)
from agential.constants import Benchmarks
from agential.core.llm import BaseLLM
from agential.utils.docstore import DocstoreExplorer
from agential.utils.general import safe_execute
from agential.utils.parse import remove_newline
from langchain_community.docstore.wikipedia import Wikipedia


class BenchmarkHandler(ABC):
    """Abstract base class for benchmark-specific handlers."""
    
    def __init__(self, llm: BaseLLM, max_steps: int = 6, testing: bool = False):
        self.llm = llm
        self.max_steps = max_steps
        self.testing = testing
    
    @abstractmethod
    def get_prompt(self) -> str:
        """Return the prompt template for this benchmark."""
        pass
    
    @abstractmethod
    def parse_action(self, action: str) -> Tuple[str, str]:
        """Parse action string into action_type and query."""
        pass
    
    @abstractmethod
    def handle_observation(
        self, 
        action_type: str, 
        query: str, 
        scratchpad: str
    ) -> Tuple[str, str, bool, Dict[str, Any]]:
        """Handle observation based on action type and query."""
        pass
    
    def reset(self) -> None:
        """Reset internal state. Override if needed."""
        pass


class QAHandler(BenchmarkHandler):
    """Handler for QA benchmarks (HotpotQA, FEVER, TriviaQA, AmbigNQ)."""
    
    def __init__(self, llm: BaseLLM, max_steps: int = 6, testing: bool = False):
        super().__init__(llm, max_steps, testing)
        self.docstore = DocstoreExplorer(Wikipedia())
    
    def get_prompt(self) -> str:
        return ""  # Overridden by specific benchmarks
    
    def parse_action(self, action: str) -> Tuple[str, str]:
        """Parse QA action (Search, Lookup, Finish)."""
        pattern = r"^(\w+)\[(.+)\]$"
        match = re.match(pattern, action)
        return (match.group(1), match.group(2)) if match else ("", "")
    
    def handle_observation(
        self, 
        action_type: str, 
        query: str, 
        scratchpad: str
    ) -> Tuple[str, str, bool, Dict[str, Any]]:
        """Handle QA observation."""
        answer = ""
        finished = False
        external_tool_info = {}
        
        if action_type.lower() == "finish":
            answer = query
            finished = True
            obs = query
        elif action_type.lower() == "search":
            try:
                search_result = self.docstore.search(query)
                external_tool_info["search_result"] = search_result
                obs = remove_newline(search_result)
            except Exception:
                obs = "Could not find that page, please try again."
        elif action_type.lower() == "lookup":
            try:
                lookup_result = self.docstore.lookup(query)
                external_tool_info["lookup_result"] = lookup_result
                obs = remove_newline(lookup_result)
            except ValueError:
                obs = "The last page Searched was not found, so you cannot Lookup a keyword in it. Please try one of the similar pages given."
        else:
            obs = "Invalid Action. Valid Actions are Lookup[<topic>] Search[<topic>] and Finish[<answer>]."
        
        return obs, answer, finished, external_tool_info


class MathHandler(BenchmarkHandler):
    """Handler for Math benchmarks (GSM8K, SVAMP, TabMWP)."""
    
    def get_prompt(self) -> str:
        return ""  # Overridden by specific benchmarks
    
    def parse_action(self, action: str) -> Tuple[str, str]:
        """Parse math action (Calculate, Finish)."""
        action_split = action.split("```python", maxsplit=1)
        match = re.search(r"\b(Finish|Calculate)\b", action_split[0], re.IGNORECASE)
        action_type = match.group(0).lower().capitalize() if match else ""
        try:
            query = action_split[1].split("```")[0].strip() if action_type else ""
        except:
            action_type = ""
            query = ""
        return action_type, query
    
    def handle_observation(
        self, 
        action_type: str, 
        query: str, 
        scratchpad: str
    ) -> Tuple[str, str, bool, Dict[str, Any]]:
        """Handle math observation."""
        answer = ""
        finished = False
        external_tool_info = {}
        
        if action_type.lower() == "finish":
            answer = query
            finished = True
            obs = f"\n```python\n{answer}\n```"
        elif action_type.lower() == "calculate":
            code_answer, execution_status = safe_execute(query)
            external_tool_info["code_answer"] = code_answer[0]
            external_tool_info["execution_status"] = execution_status
            answer = query
            obs = f"\n```python\n{answer}\n```\nExecution Status: {execution_status}\nOutput: answer = {code_answer[0]}"
        else:
            obs = "Invalid Action. Valid Actions are Calculate[code] and Finish[answer]."
        
        return obs, answer, finished, external_tool_info


class CodeHandler(BenchmarkHandler):
    """Handler for Code benchmarks (HumanEval, MBPP)."""
    
    def __init__(self, llm: BaseLLM, max_steps: int = 6, testing: bool = False):
        super().__init__(llm, max_steps, testing)
        self._answer = ""
    
    def get_prompt(self) -> str:
        return ""  # Overridden by specific benchmarks
    
    def parse_action(self, action: str) -> Tuple[str, str]:
        """Parse code action (Implement, Test, Finish)."""
        action_split = action.split("```python", maxsplit=1)
        match = re.search(r"\b(Finish|Test|Implement)\b", action_split[0], re.IGNORECASE)
        action_type = match.group(0).lower().capitalize() if match else ""
        try:
            query = action_split[1].split("```")[0].strip() if action_type else ""
        except:
            action_type = ""
            query = ""
        return action_type, query
    
    def handle_observation(
        self, 
        action_type: str, 
        query: str, 
        scratchpad: str
    ) -> Tuple[str, str, bool, Dict[str, Any]]:
        """Handle code observation."""
        finished = False
        external_tool_info = {}
        
        if action_type.lower() == "finish":
            _, execution_status = safe_execute(query)
            external_tool_info["execution_status"] = execution_status
            self._answer = query
            finished = True
            obs = f"\n```python\n{self._answer}\n```"
        elif action_type.lower() == "implement":
            _, execution_status = safe_execute(query)
            external_tool_info["execution_status"] = execution_status
            self._answer = query
            obs = f"\n```python\n{self._answer}\n```\nExecution Status: {execution_status}"
        elif action_type.lower() == "test":
            obs = f"{self._answer}\n\n{query}"
            _, execution_status = safe_execute(obs)
            external_tool_info["execution_status"] = execution_status
            obs = f"\n```python\n{obs}\n```\nExecution Status: {execution_status}"
        else:
            obs = "Invalid Action. Valid Actions are Implement[code] Test[code] and Finish[answer]."
        
        return obs, f"\n```python\n{self._answer}\n```\n", finished, external_tool_info
    
    def reset(self) -> None:
        """Reset internal state."""
        self._answer = ""


# =============================================================================
# SPECIFIC BENCHMARK HANDLERS
# =============================================================================

class HotpotQAHandler(QAHandler):
    def get_prompt(self) -> str:
        return REACT_INSTRUCTION_HOTPOTQA


class FEVERHandler(QAHandler):
    def get_prompt(self) -> str:
        return REACT_INSTRUCTION_FEVER


class TriviaQAHandler(QAHandler):
    def get_prompt(self) -> str:
        return REACT_INSTRUCTION_TRIVIAQA


class AmbigNQHandler(QAHandler):
    def get_prompt(self) -> str:
        return REACT_INSTRUCTION_AMBIGNQ


class GSM8KHandler(MathHandler):
    def get_prompt(self) -> str:
        return REACT_INSTRUCTION_GSM8K


class SVAMPHandler(MathHandler):
    def get_prompt(self) -> str:
        return REACT_INSTRUCTION_SVAMP


class TabMWPHandler(MathHandler):
    def get_prompt(self) -> str:
        return REACT_INSTRUCTION_TABMWP


class HumanEvalHandler(CodeHandler):
    def get_prompt(self) -> str:
        return REACT_INSTRUCTION_HUMANEVAL


class MBPPHandler(CodeHandler):
    def get_prompt(self) -> str:
        return REACT_INSTRUCTION_MBPP


# Registry for benchmark handlers - this is the plugin registry
BENCHMARK_HANDLERS = {
    Benchmarks.HOTPOTQA: HotpotQAHandler,
    Benchmarks.FEVER: FEVERHandler,
    Benchmarks.TRIVIAQA: TriviaQAHandler,
    Benchmarks.AMBIGNQ: AmbigNQHandler,
    Benchmarks.GSM8K: GSM8KHandler,
    Benchmarks.SVAMP: SVAMPHandler,
    Benchmarks.TABMWP: TabMWPHandler,
    Benchmarks.HUMANEVAL: HumanEvalHandler,
    Benchmarks.MBPP: MBPPHandler,
} 