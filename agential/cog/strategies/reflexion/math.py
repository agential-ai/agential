"""Reflexion Agent strategies for Math."""

from typing import Any, Dict, List, Tuple
from langchain_core.language_models.chat_models import BaseChatModel
from agential.cog.strategies.reflexion.base import (
    ReflexionCoTBaseStrategy,
    ReflexionReActBaseStrategy,
)


class ReflexionCoTMathStrategy(ReflexionCoTBaseStrategy):
    def __init__(self, llm: BaseChatModel) -> None:
        super().__init__(llm)

    def generate(self, *args: Any, **kwargs: Any) -> str:
        return super().generate(*args, **kwargs)
    
    def generate_action(self, question: str, examples: str, reflections: str, prompt: str, additional_keys: Dict[str, str]) -> Tuple[str, str]:
        return super().generate_action(question, examples, reflections, prompt, additional_keys)
    
    def generate_observation(self, action_type: str, query: str, key: str) -> Tuple[bool | str]:
        return super().generate_observation(action_type, query, key)
    
    def create_output_dict(self, thought: str, action_type: str, obs: str, is_correct: bool, reflections: List[str]) -> Dict[str, Any]:
        return super().create_output_dict(thought, action_type, obs, is_correct, reflections)

    def halting_condition(self, idx: int, key: str, **kwargs: Any) -> bool:
        return super().halting_condition(idx, key, **kwargs)
    
    def reset(self, *args: Any, **kwargs: Any) -> None:
        return super().reset(*args, **kwargs)
    
    def reflect(self, reflect_strategy: str, question: str, examples: str, prompt: str, additional_keys: Dict[str, str]) -> Tuple[List[str] | str]:
        return super().reflect(reflect_strategy, question, examples, prompt, additional_keys)
    
    def reflect_condition(self, idx: int, reflect_strategy: str | None, key: str) -> bool:
        return super().reflect_condition(idx, reflect_strategy, key)


class ReflexionReActMathStrategy(ReflexionReActBaseStrategy):
    def __init__(self, llm: BaseChatModel) -> None:
        super().__init__(llm)

    def generate(self, *args: Any, **kwargs: Any) -> str:
        return super().generate(*args, **kwargs)
    
    def generate_action(self, question: str, examples: str, reflections: str, prompt: str, additional_keys: Dict[str, str], **kwargs: Any) -> Tuple[str]:
        return super().generate_action(question, examples, reflections, prompt, additional_keys, **kwargs)
    
    def generate_observation(self, step_idx: int, action_type: str, query: str, key: str) -> Tuple[bool | str]:
        return super().generate_observation(step_idx, action_type, query, key)
    
    def create_output_dict(self, react_out: List[Dict[str, Any]], reflections: List[str]) -> Dict[str, Any]:
        return super().create_output_dict(react_out, reflections)
    
    def react_create_output_dict(self, thought: str, action_type: str, query: str, obs: str, is_correct: bool) -> Dict[str, str]:
        return super().react_create_output_dict(thought, action_type, query, obs, is_correct)
    
    def halting_condition(self, idx: int, key: str, **kwargs: Any) -> bool:
        return super().halting_condition(idx, key, **kwargs)
    
    def react_halting_condition(self, step_idx: int, question: str, examples: str, reflections: str, prompt: str, additional_keys: Dict[str, str], **kwargs: Any) -> bool:
        return super().react_halting_condition(step_idx, question, examples, reflections, prompt, additional_keys, **kwargs)
    
    def reset(self, *args: Any, **kwargs: Any) -> None:
        return super().reset(*args, **kwargs)
    
    def reflect(self, reflect_strategy: str, question: str, examples: str, prompt: str, additional_keys: Dict[str, str]) -> Tuple[List[str] | str]:
        return super().reflect(reflect_strategy, question, examples, prompt, additional_keys)
    
    def reflect_condition(self, step_idx: int, reflect_strategy: str | None, question: str, examples: str, key: str, prompt: str, additional_keys: Dict[str, str], **kwargs: Dict[str, str]) -> bool:
        return super().reflect_condition(step_idx, reflect_strategy, question, examples, key, prompt, additional_keys, **kwargs)