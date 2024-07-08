"""ReAct prompts and fewshot examples selector."""

from typing import Dict

from agential.base.selector import BaseSelector
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


class ReActSelector(BaseSelector):
    @staticmethod
    def get_fewshots(self, benchmark: str, **kwargs) -> Dict[str, str]:
        return {}

    @staticmethod
    def get_prompt(self, benchmark: str, **kwargs) -> Dict[str, str]:
        if benchmark not in REACT_PROMPTS:
            raise ValueError(f"Benchmark '{benchmark}' prompt not found for ReAct.")

        return REACT_PROMPTS[benchmark]
