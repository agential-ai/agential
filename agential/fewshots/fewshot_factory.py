"""Few-shot factory class."""

from agential.fewshots.ambignq import (
    AMBIGNQ_FEWSHOT_EXAMPLES_COT,
    AMBIGNQ_FEWSHOT_EXAMPLES_DIRECT,
    AMBIGNQ_FEWSHOT_EXAMPLES_REACT,
)
from agential.fewshots.fever import (
    FEVER_FEWSHOT_EXAMPLES_COT,
    FEVER_FEWSHOT_EXAMPLES_DIRECT,
    FEVER_FEWSHOT_EXAMPLES_REACT,
)
from agential.fewshots.gsm8k import (
    GSM8K_FEWSHOT_EXAMPLES_COT,
    GSM8K_FEWSHOT_EXAMPLES_POT,
    GSM8K_FEWSHOT_EXAMPLES_REACT,
)
from agential.fewshots.hotpotqa import (
    HOTPOTQA_FEWSHOT_EXAMPLES_COT,
    HOTPOTQA_FEWSHOT_EXAMPLES_DIRECT,
    HOTPOTQA_FEWSHOT_EXAMPLES_REACT,
)
from agential.fewshots.humaneval import (
    HUMANEVAL_FEWSHOT_EXAMPLES_COT,
    HUMANEVAL_FEWSHOT_EXAMPLES_POT,
    HUMANEVAL_FEWSHOT_EXAMPLES_REACT,
)
from agential.fewshots.mbpp import (
    MBPP_FEWSHOT_EXAMPLES_COT,
    MBPP_FEWSHOT_EXAMPLES_POT,
    MBPP_FEWSHOT_EXAMPLES_REACT,
)
from agential.fewshots.svamp import (
    SVAMP_FEWSHOT_EXAMPLES_COT,
    SVAMP_FEWSHOT_EXAMPLES_POT,
    SVAMP_FEWSHOT_EXAMPLES_REACT,
)
from agential.fewshots.tabmwp import (
    TABMWP_FEWSHOT_EXAMPLES_COT,
    TABMWP_FEWSHOT_EXAMPLES_POT,
    TABMWP_FEWSHOT_EXAMPLES_REACT,
)
from agential.fewshots.triviaqa import (
    TRIVIAQA_FEWSHOT_EXAMPLES_COT,
    TRIVIAQA_FEWSHOT_EXAMPLES_DIRECT,
    TRIVIAQA_FEWSHOT_EXAMPLES_REACT,
)
from agential.base.constants import Benchmarks, FewShotType

BENCHMARK_FEWSHOTS = {
    Benchmarks.HOTPOTQA: {
        FewShotType.COT: HOTPOTQA_FEWSHOT_EXAMPLES_COT,
        FewShotType.DIRECT: HOTPOTQA_FEWSHOT_EXAMPLES_DIRECT,
        FewShotType.REACT: HOTPOTQA_FEWSHOT_EXAMPLES_REACT,
    },
    Benchmarks.FEVER: {
        FewShotType.COT: FEVER_FEWSHOT_EXAMPLES_COT,
        FewShotType.DIRECT: FEVER_FEWSHOT_EXAMPLES_DIRECT,
        FewShotType.REACT: FEVER_FEWSHOT_EXAMPLES_REACT,
    },
    Benchmarks.TRIVIAQA: {
        FewShotType.COT: TRIVIAQA_FEWSHOT_EXAMPLES_COT,
        FewShotType.DIRECT: TRIVIAQA_FEWSHOT_EXAMPLES_DIRECT,
        FewShotType.REACT: TRIVIAQA_FEWSHOT_EXAMPLES_REACT,
    },
    Benchmarks.AMBIGNQ: {
        FewShotType.COT: AMBIGNQ_FEWSHOT_EXAMPLES_COT,
        FewShotType.DIRECT: AMBIGNQ_FEWSHOT_EXAMPLES_DIRECT,
        FewShotType.REACT: AMBIGNQ_FEWSHOT_EXAMPLES_REACT,
    },
    Benchmarks.GSM8K: {
        FewShotType.POT: GSM8K_FEWSHOT_EXAMPLES_POT,
        FewShotType.REACT: GSM8K_FEWSHOT_EXAMPLES_REACT,
        FewShotType.COT: GSM8K_FEWSHOT_EXAMPLES_COT,
    },
    Benchmarks.SVAMP: {
        FewShotType.POT: SVAMP_FEWSHOT_EXAMPLES_POT,
        FewShotType.REACT: SVAMP_FEWSHOT_EXAMPLES_REACT,
        FewShotType.COT: SVAMP_FEWSHOT_EXAMPLES_COT,
    },
    Benchmarks.TABMWP: {
        FewShotType.POT: TABMWP_FEWSHOT_EXAMPLES_POT,
        FewShotType.REACT: TABMWP_FEWSHOT_EXAMPLES_REACT,
        FewShotType.COT: TABMWP_FEWSHOT_EXAMPLES_COT,
    },
    Benchmarks.HUMANEVAL: {
        FewShotType.POT: HUMANEVAL_FEWSHOT_EXAMPLES_POT,
        FewShotType.REACT: HUMANEVAL_FEWSHOT_EXAMPLES_REACT,
        FewShotType.COT: HUMANEVAL_FEWSHOT_EXAMPLES_COT,
    },
    Benchmarks.MBPP: {
        FewShotType.POT: MBPP_FEWSHOT_EXAMPLES_POT,
        FewShotType.REACT: MBPP_FEWSHOT_EXAMPLES_REACT,
        FewShotType.COT: MBPP_FEWSHOT_EXAMPLES_COT,
    },
}


class FewShotFactory:
    """A factory class for retrieving few-shot examples for a given benchmark and few-shot type."""

    @staticmethod
    def get_benchmark_fewshots(benchmark: str, fewshot_type: str) -> str:
        """Retrieve few-shot examples for a given benchmark and few-shot type.

        Available Benchmarks:
            - hotpotqa: Supports "cot", "direct", "react"
            - fever: Supports "cot", "direct", "react"
            - triviaqa: Supports "cot", "direct", "react"
            - ambignq: Supports "cot", "direct", "react"
            - gsm8k: Supports "pot", "cot", "react"
            - svamp: Supports "pot", "cot", "react"
            - tabmwp: Supports "pot", "cot", "react"
            - humaneval: Supports "pot", "cot", "react"
            - mbpp: Supports "pot", "cot", "react"

        Available Few-Shot Types:
            - "cot"
            - "direct"
            - "react"
            - "pot"

        Args:
            benchmark (str): The benchmark name.
            fewshot_type (str): The type of few-shot examples. It should be one of the predefined types in the FewShotType class.

        Returns:
            str: The few-shot examples corresponding to the given benchmark and type.
            If the benchmark or few-shot type is not found, returns a detailed error message.
        """
        if benchmark not in BENCHMARK_FEWSHOTS:
            raise ValueError(f"Benchmark '{benchmark}' not found.")

        examples = BENCHMARK_FEWSHOTS[benchmark].get(fewshot_type)
        if examples is None:
            raise ValueError(
                f"Few-shot type '{fewshot_type}' not found for benchmark '{benchmark}'."
            )

        return examples
