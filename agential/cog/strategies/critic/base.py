from abc import abstractmethod
from typing import Dict
from agential.cog.strategies.base import BaseStrategy

class CriticBaseStrategy(BaseStrategy):
    @abstractmethod
    def generate_critique(self, llm, question: str, examples: str, answer: str, prompt: str, additional_keys: Dict[str, str], critique_additional_keys: Dict[str, str], tests: str, use_interpreter_tool: bool, use_search_tool: bool):
        pass

    @abstractmethod
    def create_output_dict(self, answer: str, critique: str, additional_keys_update: Dict[str, str]) -> Dict[str, str]:
        pass

    @abstractmethod
    def update_answer_based_on_critique(self, llm, question: str, answer: str, critique: str) -> str:
        pass
    
    @abstractmethod
    def halting_condition(self, critique: str) -> bool:
        pass
