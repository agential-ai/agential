from abc import ABC, abstractmethod
from typing import Dict
from langchain_core.language_models.chat_models import BaseChatModel

class BaseStrategy(ABC):
    @abstractmethod
    def generate(self, llm: BaseChatModel, question: str, examples: str, prompt: str, additional_keys: Dict[str, str]) -> str:
        pass
