"""Standard prompting."""

from .qa_agent import StandardQA
from .math_agent import StandardMath
from .code_agent import StandardCode
from .prompts import (
    STANDARD_INSTRUCTION_HOTPOTQA,
    STANDARD_INSTRUCTION_FEVER,
    STANDARD_INSTRUCTION_TRIVIAQA,
    STANDARD_INSTRUCTION_AMBIGNQ,
    STANDARD_INSTRUCTION_GSM8K,
    STANDARD_INSTRUCTION_SVAMP,
    STANDARD_INSTRUCTION_TABMWP,
    STANDARD_INSTRUCTION_HUMANEVAL,
    STANDARD_INSTRUCTION_MBPP,
    HOTPOTQA_FEWSHOT_EXAMPLES_DIRECT,
    FEVER_FEWSHOT_EXAMPLES_DIRECT,
    TRIVIAQA_FEWSHOT_EXAMPLES_DIRECT,
    AMBIGNQ_FEWSHOT_EXAMPLES_DIRECT,
    GSM8K_FEWSHOT_EXAMPLES_POT,
    SVAMP_FEWSHOT_EXAMPLES_POT,
    TABMWP_FEWSHOT_EXAMPLES_POT,
    HUMANEVAL_FEWSHOT_EXAMPLES_POT,
    MBPP_FEWSHOT_EXAMPLES_POT,
)
from agential.methods.base import BaseMethod

BENCHMARK_CONFIG = {
    # QA
    "hotpotqa": {
        "prompt_template": STANDARD_INSTRUCTION_HOTPOTQA,
        "fewshot": HOTPOTQA_FEWSHOT_EXAMPLES_DIRECT,
        "agent": StandardQA,
    },
    "fever": {
        "prompt_template": STANDARD_INSTRUCTION_FEVER,
        "fewshot": FEVER_FEWSHOT_EXAMPLES_DIRECT,
        "agent": StandardQA,
    },
    "triviaqa": {
        "prompt_template": STANDARD_INSTRUCTION_TRIVIAQA,
        "fewshot": TRIVIAQA_FEWSHOT_EXAMPLES_DIRECT,
        "agent": StandardQA,
    },
    "ambignq": {
        "prompt_template": STANDARD_INSTRUCTION_AMBIGNQ,
        "fewshot": AMBIGNQ_FEWSHOT_EXAMPLES_DIRECT,
        "agent": StandardQA,
    },
    # Math
    "gsm8k": {
        "prompt_template": STANDARD_INSTRUCTION_GSM8K,
        "fewshot": GSM8K_FEWSHOT_EXAMPLES_POT,
        "agent": StandardMath,
    },
    "svamp": {
        "prompt_template": STANDARD_INSTRUCTION_SVAMP,
        "fewshot": SVAMP_FEWSHOT_EXAMPLES_POT,
        "agent": StandardMath,
    },
    "tabmwp": {
        "prompt_template": STANDARD_INSTRUCTION_TABMWP,
        "fewshot": TABMWP_FEWSHOT_EXAMPLES_POT,
        "agent": StandardMath,
    },
    # Code
    "humaneval": {
        "prompt_template": STANDARD_INSTRUCTION_HUMANEVAL,
        "fewshot": HUMANEVAL_FEWSHOT_EXAMPLES_POT,
        "agent": StandardCode,
    },
    "mbpp": {
        "prompt_template": STANDARD_INSTRUCTION_MBPP,
        "fewshot": MBPP_FEWSHOT_EXAMPLES_POT,
        "agent": StandardCode,
    },
}


class Standard(BaseMethod):
    _agent: BaseMethod  # type: ignore

    def __new__(cls, llm, benchmark, *args, **kwargs):
        try:
            config = BENCHMARK_CONFIG[benchmark]
            agent_cls = config["agent"]
        except KeyError:
            raise ValueError(f"Unknown benchmark: {benchmark}")
        # Create the agent instance
        agent = agent_cls(llm, benchmark, *args, **kwargs, config=config)
        # Create a Standard instance and store the agent
        instance = super().__new__(cls)
        instance._agent = agent
        # Copy attributes for BaseMethod compliance
        instance.llm = agent.llm
        instance.benchmark = agent.benchmark
        instance.verbose = getattr(agent, "verbose", False)
        instance.config = getattr(agent, "config", {})
        return instance

    def generate(self, question: str, **kwargs):
        return self._agent.generate(question, **kwargs)


__all__ = ["Standard"]
