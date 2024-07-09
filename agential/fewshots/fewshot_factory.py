"""Few-shot factory class."""

from typing import Dict

from agential.base.constants import Benchmarks, FewShotType
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

BENCHMARK_FEWSHOTS = {
    Benchmarks.HOTPOTQA: {
        FewShotType.COT: {"examples": HOTPOTQA_FEWSHOT_EXAMPLES_COT},
        FewShotType.DIRECT: {"examples": HOTPOTQA_FEWSHOT_EXAMPLES_DIRECT},
        FewShotType.REACT: {"examples": HOTPOTQA_FEWSHOT_EXAMPLES_REACT},
    },
    Benchmarks.FEVER: {
        FewShotType.COT: {"examples": FEVER_FEWSHOT_EXAMPLES_COT},
        FewShotType.DIRECT: {"examples": FEVER_FEWSHOT_EXAMPLES_DIRECT},
        FewShotType.REACT: {"examples": FEVER_FEWSHOT_EXAMPLES_REACT},
    },
    Benchmarks.TRIVIAQA: {
        FewShotType.COT: {"examples": TRIVIAQA_FEWSHOT_EXAMPLES_COT},
        FewShotType.DIRECT: {"examples": TRIVIAQA_FEWSHOT_EXAMPLES_DIRECT},
        FewShotType.REACT: {"examples": TRIVIAQA_FEWSHOT_EXAMPLES_REACT},
    },
    Benchmarks.AMBIGNQ: {
        FewShotType.COT: {"examples": AMBIGNQ_FEWSHOT_EXAMPLES_COT},
        FewShotType.DIRECT: {"examples": AMBIGNQ_FEWSHOT_EXAMPLES_DIRECT},
        FewShotType.REACT: {"examples": AMBIGNQ_FEWSHOT_EXAMPLES_REACT},
    },
    Benchmarks.GSM8K: {
        FewShotType.POT: {"examples": GSM8K_FEWSHOT_EXAMPLES_POT},
        FewShotType.REACT: {"examples": GSM8K_FEWSHOT_EXAMPLES_REACT},
        FewShotType.COT: {"examples": GSM8K_FEWSHOT_EXAMPLES_COT},
    },
    Benchmarks.SVAMP: {
        FewShotType.POT: {"examples": SVAMP_FEWSHOT_EXAMPLES_POT},
        FewShotType.REACT: {"examples": SVAMP_FEWSHOT_EXAMPLES_REACT},
        FewShotType.COT: {"examples": SVAMP_FEWSHOT_EXAMPLES_COT},
    },
    Benchmarks.TABMWP: {
        FewShotType.POT: {"examples": TABMWP_FEWSHOT_EXAMPLES_POT},
        FewShotType.REACT: {"examples": TABMWP_FEWSHOT_EXAMPLES_REACT},
        FewShotType.COT: {"examples": TABMWP_FEWSHOT_EXAMPLES_COT},
    },
    Benchmarks.HUMANEVAL: {
        FewShotType.POT: {"examples": HUMANEVAL_FEWSHOT_EXAMPLES_POT},
        FewShotType.REACT: {"examples": HUMANEVAL_FEWSHOT_EXAMPLES_REACT},
        FewShotType.COT: {"examples": HUMANEVAL_FEWSHOT_EXAMPLES_COT},
    },
    Benchmarks.MBPP: {
        FewShotType.POT: {"examples": MBPP_FEWSHOT_EXAMPLES_POT},
        FewShotType.REACT: {"examples": MBPP_FEWSHOT_EXAMPLES_REACT},
        FewShotType.COT: {"examples": MBPP_FEWSHOT_EXAMPLES_COT},
    },
}


class FewShotFactory:
    """A factory class for retrieving few-shot examples for a given benchmark and few-shot type."""

    @staticmethod
    def get_benchmark_fewshots(benchmark: str, fewshot_type: str) -> Dict[str, str]:
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
            Dict[str, str]: A dictionary containing the few-shot examples corresponding to the given benchmark and type.
            If the benchmark or few-shot type is not found, raises a ValueError with a detailed error message.
        """
        if benchmark not in BENCHMARK_FEWSHOTS:
            raise ValueError(f"Benchmark '{benchmark}' not found.")

        examples = BENCHMARK_FEWSHOTS[benchmark].get(fewshot_type)
        if examples is None:
            raise ValueError(
                f"Few-shot type '{fewshot_type}' not found for benchmark '{benchmark}'."
            )

        return examples
