"""Chain-of-Thought prompting for LLMs."""

from agential.methods.cot.qa_agent import CoTQA
from agential.methods.cot.math_agent import CoTMath
from agential.methods.cot.code_agent import CoTCode
from agential.methods.base import BaseMethod
from agential.methods.cot.prompts import (
    COT_INSTRUCTION_HOTPOTQA,
    COT_INSTRUCTION_FEVER,
    COT_INSTRUCTION_TRIVIAQA,
    COT_INSTRUCTION_AMBIGNQ,
    COT_INSTRUCTION_GSM8K,
    COT_INSTRUCTION_SVAMP,
    COT_INSTRUCTION_TABMWP,
    COT_INSTRUCTION_HUMANEVAL,
    COT_INSTRUCTION_MBPP,
    HOTPOTQA_FEWSHOT_EXAMPLES_COT,
    FEVER_FEWSHOT_EXAMPLES_COT,
    TRIVIAQA_FEWSHOT_EXAMPLES_COT,
    AMBIGNQ_FEWSHOT_EXAMPLES_COT,
    GSM8K_FEWSHOT_EXAMPLES_COT,
    SVAMP_FEWSHOT_EXAMPLES_COT,
    TABMWP_FEWSHOT_EXAMPLES_COT,
    HUMANEVAL_FEWSHOT_EXAMPLES_COT,
    MBPP_FEWSHOT_EXAMPLES_COT,
)

COT_BENCHMARK_CONFIG = {
    # QA
    "hotpotqa": {
        "prompt": COT_INSTRUCTION_HOTPOTQA,
        "examples": HOTPOTQA_FEWSHOT_EXAMPLES_COT,
        "agent": CoTQA,
    },
    "fever": {
        "prompt": COT_INSTRUCTION_FEVER,
        "examples": FEVER_FEWSHOT_EXAMPLES_COT,
        "agent": CoTQA,
    },
    "triviaqa": {
        "prompt": COT_INSTRUCTION_TRIVIAQA,
        "examples": TRIVIAQA_FEWSHOT_EXAMPLES_COT,
        "agent": CoTQA,
    },
    "ambignq": {
        "prompt": COT_INSTRUCTION_AMBIGNQ,
        "examples": AMBIGNQ_FEWSHOT_EXAMPLES_COT,
        "agent": CoTQA,
    },
    # Math
    "gsm8k": {
        "prompt": COT_INSTRUCTION_GSM8K,
        "examples": GSM8K_FEWSHOT_EXAMPLES_COT,
        "agent": CoTMath,
    },
    "svamp": {
        "prompt": COT_INSTRUCTION_SVAMP,
        "examples": SVAMP_FEWSHOT_EXAMPLES_COT,
        "agent": CoTMath,
    },
    "tabmwp": {
        "prompt": COT_INSTRUCTION_TABMWP,
        "examples": TABMWP_FEWSHOT_EXAMPLES_COT,
        "agent": CoTMath,
    },
    # Code
    "humaneval": {
        "prompt": COT_INSTRUCTION_HUMANEVAL,
        "examples": HUMANEVAL_FEWSHOT_EXAMPLES_COT,
        "agent": CoTCode,
    },
    "mbpp": {
        "prompt": COT_INSTRUCTION_MBPP,
        "examples": MBPP_FEWSHOT_EXAMPLES_COT,
        "agent": CoTCode,
    },
}

class CoT(BaseMethod):
    _agent: object

    def __new__(cls, llm, benchmark, *args, **kwargs):
        try:
            config = COT_BENCHMARK_CONFIG[benchmark]
            agent_cls = config["agent"]
        except KeyError:
            raise ValueError(f"Unknown benchmark: {benchmark}")
        agent = agent_cls(llm, benchmark, *args, **kwargs, config=config)
        instance = super().__new__(cls)
        instance._agent = agent
        instance.llm = agent.llm
        instance.benchmark = agent.benchmark
        instance.verbose = getattr(agent, "verbose", False)
        instance.config = getattr(agent, "config", {})
        return instance

    def generate(self, question: str, key: str = "", **kwargs):
        return self._agent.generate(question, key, **kwargs)  # type: ignore

__all__ = ["CoT"]
