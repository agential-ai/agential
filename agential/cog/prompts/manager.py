"""Main file for managing benchmark prompts/few-shot examples."""

from typing import Dict

from agential.cog.prompts.benchmarks.ambignq import (
    AMBIGNQ_FEWSHOT_EXAMPLES_COT,
    AMBIGNQ_FEWSHOT_EXAMPLES_DIRECT,
    AMBIGNQ_FEWSHOT_EXAMPLES_REACT,
)
from agential.cog.prompts.benchmarks.fever import (
    FEVER_FEWSHOT_EXAMPLES_COT,
    FEVER_FEWSHOT_EXAMPLES_DIRECT,
    FEVER_FEWSHOT_EXAMPLES_REACT,
)
from agential.cog.prompts.benchmarks.gsm8k import (
    GSM8K_FEWSHOT_EXAMPLES_POT,
)
from agential.cog.prompts.benchmarks.hotpotqa import (
    HOTPOTQA_FEWSHOT_EXAMPLES_COT,
    HOTPOTQA_FEWSHOT_EXAMPLES_DIRECT,
    HOTPOTQA_FEWSHOT_EXAMPLES_REACT,
)
from agential.cog.prompts.benchmarks.humaneval import (
    HUMANEVAL_FEWSHOT_EXAMPLES_POT,
)
from agential.cog.prompts.benchmarks.mbpp import (
    MBPP_FEWSHOT_EXAMPLES_POT,
)
from agential.cog.prompts.benchmarks.svamp import (
    SVAMP_FEWSHOT_EXAMPLES_POT,
)
from agential.cog.prompts.benchmarks.tabmwp import (
    TABMWP_FEWSHOT_EXAMPLES_POT,
)
from agential.cog.prompts.benchmarks.triviaqa import (
    TRIVIAQA_FEWSHOT_EXAMPLES_COT,
    TRIVIAQA_FEWSHOT_EXAMPLES_DIRECT,
    TRIVIAQA_FEWSHOT_EXAMPLES_REACT,
)


class Benchmarks:
    """Supported benchmarks."""

    class qa:
        """qa benchmarks."""

        HOTPOTQA = "hotpotqa"
        FEVER = "fever"
        TRIVIAQA = "triviaqa"
        AMBIGNQ = "ambignq"

    class math:
        """math benchmarks."""

        GSM8K = "gsm8k"
        SVAMP = "svamp"
        TABMWP = "tabmwp"

    class code:
        """code benchmarks."""

        HUMANEVAL = "humaneval"
        MBPP = "mbpp"


class FewShotType:
    """Few-shot types."""

    COT = "cot"
    DIRECT = "direct"
    REACT = "react"
    POT = "pot"


BENCHMARK_STRINGS = {
    Benchmarks.qa.HOTPOTQA: {
        FewShotType.COT: HOTPOTQA_FEWSHOT_EXAMPLES_COT,
        FewShotType.DIRECT: HOTPOTQA_FEWSHOT_EXAMPLES_DIRECT,
        FewShotType.REACT: HOTPOTQA_FEWSHOT_EXAMPLES_REACT,
    },
    Benchmarks.qa.FEVER: {
        FewShotType.COT: FEVER_FEWSHOT_EXAMPLES_COT,
        FewShotType.DIRECT: FEVER_FEWSHOT_EXAMPLES_DIRECT,
        FewShotType.REACT: FEVER_FEWSHOT_EXAMPLES_REACT,
    },
    Benchmarks.qa.TRIVIAQA: {
        FewShotType.COT: TRIVIAQA_FEWSHOT_EXAMPLES_COT,
        FewShotType.DIRECT: TRIVIAQA_FEWSHOT_EXAMPLES_DIRECT,
        FewShotType.REACT: TRIVIAQA_FEWSHOT_EXAMPLES_REACT,
    },
    Benchmarks.qa.AMBIGNQ: {
        FewShotType.COT: AMBIGNQ_FEWSHOT_EXAMPLES_COT,
        FewShotType.DIRECT: AMBIGNQ_FEWSHOT_EXAMPLES_DIRECT,
        FewShotType.REACT: AMBIGNQ_FEWSHOT_EXAMPLES_REACT,
    },
    Benchmarks.math.GSM8K: {
        FewShotType.POT: GSM8K_FEWSHOT_EXAMPLES_POT,
    },
    Benchmarks.math.SVAMP: {
        FewShotType.POT: SVAMP_FEWSHOT_EXAMPLES_POT,
    },
    Benchmarks.math.TABMWP: {
        FewShotType.POT: TABMWP_FEWSHOT_EXAMPLES_POT,
    },
    Benchmarks.code.HUMANEVAL: {
        FewShotType.POT: HUMANEVAL_FEWSHOT_EXAMPLES_POT,
    },
    Benchmarks.code.MBPP: {
        FewShotType.POT: MBPP_FEWSHOT_EXAMPLES_POT,
    },
}


def get_fewshot_examples(mode: Dict[str, str], fewshot_type: str) -> str:
    """Retrieve few-shot examples for a given benchmark type and benchmark name.

    Available Benchmark Types and Names:
        - qa:
            - Benchmarks.qa.HOTPOTQA: Supports FewShotType.COT, FewShotType.DIRECT, FewShotType.REACT
            - Benchmarks.qa.FEVER: Supports FewShotType.COT, FewShotType.DIRECT, FewShotType.REACT
            - Benchmarks.qa.TRIVIAQA: Supports FewShotType.COT, FewShotType.DIRECT, FewShotType.REACT
            - Benchmarks.qa.AMBIGNQ: Supports FewShotType.COT, FewShotType.DIRECT, FewShotType.REACT
        - math:
            - Benchmarks.math.GSM8K: Supports FewShotType.POT
            - Benchmarks.math.SVAMP: Supports FewShotType.POT
            - Benchmarks.math.TABMWP: Supports FewShotType.POT
        - code:
            - Benchmarks.code.HUMANEVAL: Supports FewShotType.POT
            - Benchmarks.code.MBPP: Supports FewShotType.POT

    Available Few-Shot Types:
        - FewShotType.COT: "cot"
        - FewShotType.DIRECT: "direct"
        - FewShotType.REACT: "react"
        - FewShotType.POT: "pot"

    Args:
        mode (dict): A dictionary with "benchmark type" as the key and "benchmark name" as the value.
        fewshot_type (str): The type of few-shot examples. It should be one of the predefined types in the FewShotType class.

    Returns:
        str: The few-shot examples corresponding to the given benchmark and type.
        If the benchmark or few-shot type is not found, returns a detailed error message.
    """
    benchmark_type, benchmark_name = list(mode.items())[0]
    if benchmark_type not in Benchmarks.__dict__:
        raise ValueError(f"Benchmark type '{benchmark_type}' not found.")

    if benchmark_name not in BENCHMARK_STRINGS:
        raise ValueError(
            f"Benchmark '{benchmark_name}' not found in benchmark type '{benchmark_type}'."
        )

    examples = BENCHMARK_STRINGS[benchmark_name].get(fewshot_type)
    if examples is None:
        raise ValueError(
            f"Few-shot type '{fewshot_type}' not found for benchmark '{benchmark_name}'."
        )

    return examples
