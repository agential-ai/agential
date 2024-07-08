"""ReAct prompts and fewshot examples selector."""

from typing import Any, Dict

from agential.base.factory import BaseFactory
from agential.cog.react.prompts import (
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
from agential.cog.react.strategies.base import ReActBaseStrategy
from agential.cog.react.strategies.code import ReActHEvalStrategy, ReActMBPPStrategy
from agential.cog.react.strategies.math import (
    ReActGSM8KStrategy,
    ReActSVAMPStrategy,
    ReActTabMWPStrategy,
)
from agential.cog.react.strategies.qa import (
    ReActAmbigNQStrategy,
    ReActFEVERStrategy,
    ReActHotQAStrategy,
    ReActTriviaQAStrategy,
)
from agential.manager.constants import Benchmarks

REACT_PROMPTS = {
    Benchmarks.HOTPOTQA: {
        "prompt": REACT_INSTRUCTION_HOTPOTQA,
    },
    Benchmarks.FEVER: {
        "prompt": REACT_INSTRUCTION_FEVER,
    },
    Benchmarks.TRIVIAQA: {
        "prompt": REACT_INSTRUCTION_TRIVIAQA,
    },
    Benchmarks.AMBIGNQ: {
        "prompt": REACT_INSTRUCTION_AMBIGNQ,
    },
    Benchmarks.GSM8K: {
        "prompt": REACT_INSTRUCTION_GSM8K,
    },
    Benchmarks.SVAMP: {
        "prompt": REACT_INSTRUCTION_SVAMP,
    },
    Benchmarks.TABMWP: {
        "prompt": REACT_INSTRUCTION_TABMWP,
    },
    Benchmarks.HUMANEVAL: {
        "prompt": REACT_INSTRUCTION_HUMANEVAL,
    },
    Benchmarks.MBPP: {
        "prompt": REACT_INSTRUCTION_MBPP,
    },
}

REACT_STRATEGIES = {
    Benchmarks.HOTPOTQA: ReActHotQAStrategy,
    Benchmarks.FEVER: ReActFEVERStrategy,
    Benchmarks.TRIVIAQA: ReActTriviaQAStrategy,
    Benchmarks.AMBIGNQ: ReActAmbigNQStrategy,
    Benchmarks.GSM8K: ReActGSM8KStrategy,
    Benchmarks.SVAMP: ReActSVAMPStrategy,
    Benchmarks.TABMWP: ReActTabMWPStrategy,
    Benchmarks.HUMANEVAL: ReActHEvalStrategy,
    Benchmarks.MBPP: ReActMBPPStrategy,
}


class ReActSelector(BaseFactory):
    @staticmethod
    def get_fewshots(self, benchmark: str, **kwargs) -> Dict[str, str]:
        return {}

    @staticmethod
    def get_prompt(self, benchmark: str, **kwargs) -> Dict[str, str]:
        if benchmark not in REACT_PROMPTS:
            raise ValueError(f"Benchmark '{benchmark}' prompt not found for ReAct.")

        return REACT_PROMPTS[benchmark]


class ReactStrategyFactory:
    """A factory class for creating instances of ReAct strategies."""

    @staticmethod
    def get_strategy(benchmark: str, **strategy_kwargs: Any) -> ReActBaseStrategy:
        if benchmark not in REACT_STRATEGIES:
            raise ValueError(f"Unsupported benchmark: {benchmark} for agent ReAct")

        strategy = REACT_STRATEGIES[benchmark]
        return strategy(**strategy_kwargs)
