"""Minimal BaseAgent class."""

from abc import ABC, abstractmethod
from typing import Dict, Any
from agential.core.llm import BaseLLM


class BaseAgent(ABC):
    """Minimal base agent class."""
    
    def __init__(
        self,
        llm: BaseLLM,
        benchmark: str,
        debug_mode: bool = False,
        **kwargs,
    ):
        self.llm = llm
        self.benchmark = benchmark
        self.debug_mode = debug_mode
    
    @abstractmethod
    def generate(self, question: str, **kwargs) -> Dict[str, Any]:
        """Generate answer for the given question."""
        pass 