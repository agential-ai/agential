"""CLIN Agent.

Paper: https://arxiv.org/pdf/2310.10134
GitHub Repo: https://github.com/allenai/clin
"""

from .qa_agent import CLINQA
from .math_agent import CLINMath
from .code_agent import CLINCode
from agential.methods.base import BaseMethod
from agential.methods.clin.prompts import (
    CLIN_ADAPT_META_SUMMARY_SYSTEM,
    CLIN_ADAPT_SUMMARY_SYSTEM,
    CLIN_GEN_ENV_META_SUMMARY_SYSTEM,
    CLIN_GEN_ENV_SUMMARY_SYSTEM,
    CLIN_GEN_TASK_META_SUMMARY_SYSTEM,
    CLIN_GEN_TASK_SUMMARY_SYSTEM,
    CLIN_INSTRUCTION_AMBIGNQ,
    CLIN_INSTRUCTION_FEVER,
    CLIN_INSTRUCTION_GSM8K,
    CLIN_INSTRUCTION_HOTPOTQA,
    CLIN_INSTRUCTION_HUMANEVAL,
    CLIN_INSTRUCTION_MBPP,
    CLIN_INSTRUCTION_SVAMP,
    CLIN_INSTRUCTION_TABMWP,
    CLIN_INSTRUCTION_TRIVIAQA,
    CLIN_META_SUMMARY_INSTRUCTION_AMBIGNQ,
    CLIN_META_SUMMARY_INSTRUCTION_FEVER,
    CLIN_META_SUMMARY_INSTRUCTION_GSM8K,
    CLIN_META_SUMMARY_INSTRUCTION_HOTPOTQA,
    CLIN_META_SUMMARY_INSTRUCTION_HUMANEVAL,
    CLIN_META_SUMMARY_INSTRUCTION_MBPP,
    CLIN_META_SUMMARY_INSTRUCTION_SVAMP,
    CLIN_META_SUMMARY_INSTRUCTION_TABMWP,
    CLIN_META_SUMMARY_INSTRUCTION_TRIVIAQA,
    CLIN_SUMMARY_INSTRUCTION_AMBIGNQ,
    CLIN_SUMMARY_INSTRUCTION_FEVER,
    CLIN_SUMMARY_INSTRUCTION_GSM8K,
    CLIN_SUMMARY_INSTRUCTION_HOTPOTQA,
    CLIN_SUMMARY_INSTRUCTION_HUMANEVAL,
    CLIN_SUMMARY_INSTRUCTION_MBPP,
    CLIN_SUMMARY_INSTRUCTION_SVAMP,
    CLIN_SUMMARY_INSTRUCTION_TABMWP,
    CLIN_SUMMARY_INSTRUCTION_TRIVIAQA,
    AMBIGNQ_FEWSHOT_EXAMPLES_REACT,
    FEVER_FEWSHOT_EXAMPLES_REACT,
    GSM8K_FEWSHOT_EXAMPLES_REACT,
    HOTPOTQA_FEWSHOT_EXAMPLES_REACT,
    HUMANEVAL_FEWSHOT_EXAMPLES_REACT,
    MBPP_FEWSHOT_EXAMPLES_REACT,
    SVAMP_FEWSHOT_EXAMPLES_REACT,
    TABMWP_FEWSHOT_EXAMPLES_REACT,
    TRIVIAQA_FEWSHOT_EXAMPLES_REACT,
)

CLIN_BENCHMARK_CONFIG = {
    # QA benchmarks
    "hotpotqa": {
        "prompt": CLIN_INSTRUCTION_HOTPOTQA,
        "summary_prompt": CLIN_SUMMARY_INSTRUCTION_HOTPOTQA,
        "meta_summary_prompt": CLIN_META_SUMMARY_INSTRUCTION_HOTPOTQA,
        "examples": HOTPOTQA_FEWSHOT_EXAMPLES_REACT,
        "summary_system": CLIN_ADAPT_SUMMARY_SYSTEM,
        "meta_summary_system": CLIN_ADAPT_META_SUMMARY_SYSTEM,
        "agent": CLINQA,
    },
    "fever": {
        "prompt": CLIN_INSTRUCTION_FEVER,
        "summary_prompt": CLIN_SUMMARY_INSTRUCTION_FEVER,
        "meta_summary_prompt": CLIN_META_SUMMARY_INSTRUCTION_FEVER,
        "examples": FEVER_FEWSHOT_EXAMPLES_REACT,
        "summary_system": CLIN_ADAPT_SUMMARY_SYSTEM,
        "meta_summary_system": CLIN_ADAPT_META_SUMMARY_SYSTEM,
        "agent": CLINQA,
    },
    "triviaqa": {
        "prompt": CLIN_INSTRUCTION_TRIVIAQA,
        "summary_prompt": CLIN_SUMMARY_INSTRUCTION_TRIVIAQA,
        "meta_summary_prompt": CLIN_META_SUMMARY_INSTRUCTION_TRIVIAQA,
        "examples": TRIVIAQA_FEWSHOT_EXAMPLES_REACT,
        "summary_system": CLIN_ADAPT_SUMMARY_SYSTEM,
        "meta_summary_system": CLIN_ADAPT_META_SUMMARY_SYSTEM,
        "agent": CLINQA,
    },
    "ambignq": {
        "prompt": CLIN_INSTRUCTION_AMBIGNQ,
        "summary_prompt": CLIN_SUMMARY_INSTRUCTION_AMBIGNQ,
        "meta_summary_prompt": CLIN_META_SUMMARY_INSTRUCTION_AMBIGNQ,
        "examples": AMBIGNQ_FEWSHOT_EXAMPLES_REACT,
        "summary_system": CLIN_ADAPT_SUMMARY_SYSTEM,
        "meta_summary_system": CLIN_ADAPT_META_SUMMARY_SYSTEM,
        "agent": CLINQA,
    },
    # Math benchmarks
    "gsm8k": {
        "prompt": CLIN_INSTRUCTION_GSM8K,
        "summary_prompt": CLIN_SUMMARY_INSTRUCTION_GSM8K,
        "meta_summary_prompt": CLIN_META_SUMMARY_INSTRUCTION_GSM8K,
        "examples": GSM8K_FEWSHOT_EXAMPLES_REACT,
        "summary_system": CLIN_ADAPT_SUMMARY_SYSTEM,
        "meta_summary_system": CLIN_ADAPT_META_SUMMARY_SYSTEM,
        "agent": CLINMath,
    },
    "svamp": {
        "prompt": CLIN_INSTRUCTION_SVAMP,
        "summary_prompt": CLIN_SUMMARY_INSTRUCTION_SVAMP,
        "meta_summary_prompt": CLIN_META_SUMMARY_INSTRUCTION_SVAMP,
        "examples": SVAMP_FEWSHOT_EXAMPLES_REACT,
        "summary_system": CLIN_ADAPT_SUMMARY_SYSTEM,
        "meta_summary_system": CLIN_ADAPT_META_SUMMARY_SYSTEM,
        "agent": CLINMath,
    },
    "tabmwp": {
        "prompt": CLIN_INSTRUCTION_TABMWP,
        "summary_prompt": CLIN_SUMMARY_INSTRUCTION_TABMWP,
        "meta_summary_prompt": CLIN_META_SUMMARY_INSTRUCTION_TABMWP,
        "examples": TABMWP_FEWSHOT_EXAMPLES_REACT,
        "summary_system": CLIN_ADAPT_SUMMARY_SYSTEM,
        "meta_summary_system": CLIN_ADAPT_META_SUMMARY_SYSTEM,
        "agent": CLINMath,
    },
    # Code benchmarks
    "humaneval": {
        "prompt": CLIN_INSTRUCTION_HUMANEVAL,
        "summary_prompt": CLIN_SUMMARY_INSTRUCTION_HUMANEVAL,
        "meta_summary_prompt": CLIN_META_SUMMARY_INSTRUCTION_HUMANEVAL,
        "examples": HUMANEVAL_FEWSHOT_EXAMPLES_REACT,
        "summary_system": CLIN_ADAPT_SUMMARY_SYSTEM,
        "meta_summary_system": CLIN_ADAPT_META_SUMMARY_SYSTEM,
        "agent": CLINCode,
    },
    "mbpp": {
        "prompt": CLIN_INSTRUCTION_MBPP,
        "summary_prompt": CLIN_SUMMARY_INSTRUCTION_MBPP,
        "meta_summary_prompt": CLIN_META_SUMMARY_INSTRUCTION_MBPP,
        "examples": MBPP_FEWSHOT_EXAMPLES_REACT,
        "summary_system": CLIN_ADAPT_SUMMARY_SYSTEM,
        "meta_summary_system": CLIN_ADAPT_META_SUMMARY_SYSTEM,
        "agent": CLINCode,
    },
}


class CLIN(BaseMethod):
    """CLIN factory class that creates the appropriate agent based on benchmark."""

    def __init__(self, llm, benchmark, *args, **kwargs):
        try:
            config = CLIN_BENCHMARK_CONFIG[benchmark]
            agent_cls = config["agent"]
        except KeyError:
            raise ValueError(f"Unknown benchmark: {benchmark}")

        # Create the agent instance
        self._agent = agent_cls(llm, benchmark, *args, **kwargs, config=config)
        # Copy attributes for BaseMethod compliance
        super().__init__(
            llm=self._agent.llm,
            benchmark=self._agent.benchmark,
            verbose=getattr(self._agent, "verbose", False),
            config=getattr(self._agent, "config", {}),
        )

    def generate(self, question: str, **kwargs):
        return self._agent.generate(question, **kwargs)


__all__ = ["CLIN"]
