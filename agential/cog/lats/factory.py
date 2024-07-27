"""LATS prompts and fewshot examples selector."""


from agential.base.factory import BaseFactory
from agential.cog.constants import BENCHMARK_FEWSHOTS, Benchmarks, FewShotType
from agential.cog.lats.prompts import (
    HOTPOTQA_FEWSHOT_EXAMPLES_LATS_REFLECT,
    HOTPOTQA_FEWSHOT_EXAMPLES_LATS_VALUE,
    LATS_INSTRUCTION_HOTPOTQA,
    LATS_VALUE_INSTRUCTION_HOTPOTQA,
    LATS_REFLECT_INSTRUCTION_HOTPOTQA,
    AMBIGNQ_FEWSHOT_EXAMPLES_LATS_REFLECT,
    AMBIGNQ_FEWSHOT_EXAMPLES_LATS_VALUE,
    LATS_INSTRUCTION_AMBIGNQ,
    LATS_VALUE_INSTRUCTION_AMBIGNQ,
    LATS_REFLECT_INSTRUCTION_AMBIGNQ,
    TRIVIAQA_FEWSHOT_EXAMPLES_LATS_REFLECT,
    TRIVIAQA_FEWSHOT_EXAMPLES_LATS_VALUE,
    LATS_INSTRUCTION_TRIVIAQA,
    LATS_VALUE_INSTRUCTION_TRIVIAQA,
    LATS_REFLECT_INSTRUCTION_TRIVIAQA,
    FEVER_FEWSHOT_EXAMPLES_LATS_REFLECT,
    FEVER_FEWSHOT_EXAMPLES_LATS_VALUE,
    LATS_INSTRUCTION_FEVER,
    LATS_VALUE_INSTRUCTION_FEVER,
    LATS_REFLECT_INSTRUCTION_FEVER,
)
from agential.cog.lats.strategies.base import LATSBaseStrategy
from agential.cog.lats.strategies.qa import (
    LATSFEVERStrategy,
    LATSHotQAStrategy,
    LATSTriviaQAStrategy,
    LATSAmbigNQStrategy
)

LATS_BENCHMARK_FEWSHOTS = {
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

LATS_PROMPTS = {
    Benchmarks.HOTPOTQA: {
        "prompt": LATS_INSTRUCTION_HOTPOTQA,
        "reflect_prompt": LATS_REFLECT_INSTRUCTION_HOTPOTQA,
        "value_prompt": LATS_VALUE_INSTRUCTION_HOTPOTQA,
    },
    Benchmarks.FEVER: {
        "prompt": LATS_INSTRUCTION_FEVER,
        "reflect_prompt": LATS_REFLECT_INSTRUCTION_FEVER,
        "value_prompt": LATS_VALUE_INSTRUCTION_FEVER,
    },
    Benchmarks.TRIVIAQA: {
        "prompt": LATS_INSTRUCTION_TRIVIAQA,
        "reflect_prompt": LATS_REFLECT_INSTRUCTION_TRIVIAQA,
        "value_prompt": LATS_VALUE_INSTRUCTION_TRIVIAQA,
    },
    Benchmarks.AMBIGNQ: {
        "prompt": LATS_INSTRUCTION_AMBIGNQ,
        "reflect_prompt": LATS_REFLECT_INSTRUCTION_AMBIGNQ,
        "value_prompt": LATS_VALUE_INSTRUCTION_AMBIGNQ,
    },
    Benchmarks.GSM8K: {
        "prompt": "",
        "reflect_prompt": "",
        "value_prompt": "",
    },
    Benchmarks.SVAMP: {
        "prompt": "",
        "reflect_prompt": "",
        "value_prompt": "",
    },
    Benchmarks.TABMWP: {
        "prompt": "",
        "reflect_prompt": "",
        "value_prompt": "",
    },
    Benchmarks.HUMANEVAL: {
        "prompt": "",
        "reflect_prompt": "",
        "value_prompt": "",
    },
    Benchmarks.MBPP: {
        "prompt": "",
        "reflect_prompt": "",
        "value_prompt": "",
    }
}

LATS_FEWSHOTS = {
    Benchmarks.HOTPOTQA: {
        "reflect_examples": HOTPOTQA_FEWSHOT_EXAMPLES_LATS_REFLECT,
        "value_examples": HOTPOTQA_FEWSHOT_EXAMPLES_LATS_VALUE
    },
    Benchmarks.FEVER: {
        "reflect_examples": FEVER_FEWSHOT_EXAMPLES_LATS_REFLECT,
        "value_examples": FEVER_FEWSHOT_EXAMPLES_LATS_VALUE
    },
    Benchmarks.TRIVIAQA: {
        "reflect_examples": TRIVIAQA_FEWSHOT_EXAMPLES_LATS_REFLECT,
        "value_examples": TRIVIAQA_FEWSHOT_EXAMPLES_LATS_VALUE
    },
    Benchmarks.AMBIGNQ: {
        "reflect_examples": AMBIGNQ_FEWSHOT_EXAMPLES_LATS_REFLECT,
        "value_examples": AMBIGNQ_FEWSHOT_EXAMPLES_LATS_VALUE
    },
    Benchmarks.GSM8K: {
        "reflect_examples": "",
        "value_examples": ""
    },
    Benchmarks.SVAMP: {
        "reflect_examples": "",
        "value_examples": ""
    },
    Benchmarks.TABMWP: {
        "reflect_examples": "",
        "value_examples": ""
    },
    Benchmarks.HUMANEVAL: {
        "reflect_examples": "",
        "value_examples": ""
    },
    Benchmarks.MBPP: {
        "reflect_examples": "",
        "value_examples": ""
    }
}

LATS_STRATEGIES = {
    Benchmarks.HOTPOTQA: LATSHotQAStrategy,
    Benchmarks.FEVER: LATSFEVERStrategy,
    Benchmarks.TRIVIAQA: LATSTriviaQAStrategy,
    Benchmarks.AMBIGNQ: LATSAmbigNQStrategy,
    Benchmarks.GSM8K: None,
    Benchmarks.SVAMP: None,
    Benchmarks.TABMWP: None,
    Benchmarks.HUMANEVAL: None,
    Benchmarks.MBPP: None,
}