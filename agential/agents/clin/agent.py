"""CLIN Agent.

Paper: https://arxiv.org/pdf/2310.10134
GitHub Repo: https://github.com/allenai/clin
"""

from agential.agents.base.agent import BaseAgent
from agential.agents.clin.output import CLINOutput
from agential.agents.clin.prompts import (
    CLIN_ADAPT_SUMMARY_SYSTEM,
    CLIN_GEN_ENV_SUMMARY_SYSTEM,
    CLIN_GEN_TASK_SUMMARY_SYSTEM,
    CLIN_ADAPT_META_SUMMARY_SYSTEM,
    CLIN_GEN_ENV_META_SUMMARY_SYSTEM,
    CLIN_GEN_TASK_META_SUMMARY_SYSTEM,
    CLIN_INSTRUCTION_HOTPOTQA,
    CLIN_SUMMARY_INSTRUCTION_HOTPOTQA,
    CLIN_META_SUMMARY_INSTRUCTION_HOTPOTQA,
    CLIN_INSTRUCTION_AMBIGNQ,
    CLIN_SUMMARY_INSTRUCTION_AMBIGNQ,
    CLIN_META_SUMMARY_INSTRUCTION_AMBIGNQ,
    CLIN_INSTRUCTION_FEVER,
    CLIN_SUMMARY_INSTRUCTION_FEVER,
    CLIN_META_SUMMARY_INSTRUCTION_FEVER,
    CLIN_INSTRUCTION_TRIVIAQA,
    CLIN_SUMMARY_INSTRUCTION_TRIVIAQA,
    CLIN_META_SUMMARY_INSTRUCTION_TRIVIAQA,
    CLIN_INSTRUCTION_GSM8K,
    CLIN_SUMMARY_INSTRUCTION_GSM8K,
    CLIN_META_SUMMARY_INSTRUCTION_GSM8K,
    CLIN_INSTRUCTION_SVAMP,
    CLIN_SUMMARY_INSTRUCTION_SVAMP,
    CLIN_META_SUMMARY_INSTRUCTION_SVAMP,
    CLIN_INSTRUCTION_TABMWP,
    CLIN_SUMMARY_INSTRUCTION_TABMWP,
    CLIN_META_SUMMARY_INSTRUCTION_TABMWP,
    CLIN_INSTRUCTION_HUMANEVAL,
    CLIN_SUMMARY_INSTRUCTION_HUMANEVAL,
    CLIN_META_SUMMARY_INSTRUCTION_HUMANEVAL,
    CLIN_INSTRUCTION_MBPP,
    CLIN_SUMMARY_INSTRUCTION_MBPP,
    CLIN_META_SUMMARY_INSTRUCTION_MBPP,
)
from agential.agents.clin.strategies.base import CLINBaseStrategy
from agential.agents.clin.strategies.code import (
    CLINHumanEvalStrategy,
    CLINMBPPStrategy,
)
from agential.agents.clin.strategies.math import (
    CLINGSM8KStrategy,
    CLINSVAMPStrategy,
    CLINTabMWPStrategy,
)
from agential.agents.clin.strategies.qa import (
    CLINAmbigNQStrategy,
    CLINFEVERStrategy,
    CLINHotQAStrategy,
    CLINTriviaQAStrategy,
)
from agential.constants import BENCHMARK_FEWSHOTS, Benchmarks, FewShotType
from agential.core.llm import BaseLLM


CLIN_BENCHMARK_FEWSHOTS = {
    Benchmarks.HOTPOTQA: [FewShotType.REACT],
    Benchmarks.FEVER: [FewShotType.REACT],
    Benchmarks.TRIVIAQA: [FewShotType.REACT],
    Benchmarks.AMBIGNQ: [FewShotType.REACT],
    Benchmarks.GSM8K: [FewShotType.REACT],
    Benchmarks.SVAMP: [FewShotType.REACT],
    Benchmarks.TABMWP: [FewShotType.REACT],
    Benchmarks.HUMANEVAL: [FewShotType.REACT],
    Benchmarks.MBPP: [FewShotType.REACT],
}


CLIN_PROMPTS = {
    Benchmarks.HOTPOTQA: {
        "prompt": CLIN_INSTRUCTION_HOTPOTQA,
        "summary_prompt": CLIN_SUMMARY_INSTRUCTION_HOTPOTQA,
        "meta_summary_prompt": CLIN_META_SUMMARY_INSTRUCTION_HOTPOTQA,
    },
    Benchmarks.FEVER: {
        "prompt": CLIN_INSTRUCTION_FEVER,
        "summary_prompt": CLIN_SUMMARY_INSTRUCTION_FEVER,
        "meta_summary_prompt": CLIN_META_SUMMARY_INSTRUCTION_FEVER,
    },
    Benchmarks.TRIVIAQA: {
        "prompt": CLIN_INSTRUCTION_TRIVIAQA,
        "summary_prompt": CLIN_SUMMARY_INSTRUCTION_TRIVIAQA,
        "meta_summary_prompt": CLIN_META_SUMMARY_INSTRUCTION_TRIVIAQA,
    },
    Benchmarks.AMBIGNQ: {
        "prompt": CLIN_INSTRUCTION_AMBIGNQ,
        "summary_prompt": CLIN_SUMMARY_INSTRUCTION_AMBIGNQ,
        "meta_summary_prompt": CLIN_META_SUMMARY_INSTRUCTION_AMBIGNQ,
    },
    Benchmarks.GSM8K: {
        "prompt": CLIN_INSTRUCTION_GSM8K,
        "summary_prompt": CLIN_SUMMARY_INSTRUCTION_GSM8K,
        "meta_summary_prompt": CLIN_META_SUMMARY_INSTRUCTION_GSM8K,
    },
    Benchmarks.SVAMP: {
        "prompt": CLIN_INSTRUCTION_SVAMP,
        "summary_prompt": CLIN_SUMMARY_INSTRUCTION_SVAMP,
        "meta_summary_prompt": CLIN_META_SUMMARY_INSTRUCTION_SVAMP,
    },
    Benchmarks.TABMWP: {
        "prompt": CLIN_INSTRUCTION_TABMWP,
        "summary_prompt": CLIN_SUMMARY_INSTRUCTION_TABMWP,
        "meta_summary_prompt": CLIN_META_SUMMARY_INSTRUCTION_TABMWP,
    },
    Benchmarks.HUMANEVAL: {
        "prompt": CLIN_INSTRUCTION_HUMANEVAL,
        "summary_prompt": CLIN_SUMMARY_INSTRUCTION_HUMANEVAL,
        "meta_summary_prompt": CLIN_META_SUMMARY_INSTRUCTION_HUMANEVAL,
    },
    Benchmarks.MBPP: {
        "prompt": CLIN_INSTRUCTION_MBPP,
        "summary_prompt": CLIN_SUMMARY_INSTRUCTION_MBPP,
        "meta_summary_prompt": CLIN_META_SUMMARY_INSTRUCTION_MBPP,
    },
}


CLIN_FEWSHOTS = {
    Benchmarks.HOTPOTQA: {},
    Benchmarks.FEVER: {},
    Benchmarks.TRIVIAQA: {},
    Benchmarks.AMBIGNQ: {},
    Benchmarks.GSM8K: {},
    Benchmarks.SVAMP: {},
    Benchmarks.TABMWP: {},
    Benchmarks.HUMANEVAL: {},
    Benchmarks.MBPP: {},
}


CLIN_STRATEGIES = {
    Benchmarks.HOTPOTQA: CLINHotQAStrategy,
    Benchmarks.FEVER: CLINFEVERStrategy,
    Benchmarks.TRIVIAQA: CLINTriviaQAStrategy,
    Benchmarks.AMBIGNQ: CLINAmbigNQStrategy,
    Benchmarks.GSM8K: CLINGSM8KStrategy,
    Benchmarks.SVAMP: CLINSVAMPStrategy,
    Benchmarks.TABMWP: CLINTabMWPStrategy,
    Benchmarks.HUMANEVAL: CLINHumanEvalStrategy,
    Benchmarks.MBPP: CLINMBPPStrategy,
}


