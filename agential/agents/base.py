"""Minimal BaseAgent class."""

from abc import ABC, abstractmethod
from typing import Dict, Any
from agential.core.llm import BaseLLM


class BaseAgent(ABC):
    """Minimal base agent class with config support."""
    
    def __init__(
        self,
        llm: BaseLLM,
        benchmark: str,
        verbose: bool = False,
        config: dict = {},
        **kwargs,
    ):
        self.llm = llm
        self.benchmark = benchmark
        self.verbose = verbose
        self.config = config
    
    @abstractmethod
    def generate(self, question: str, **kwargs) -> Dict[str, Any]:
        """Generate answer for the given question."""
        pass 