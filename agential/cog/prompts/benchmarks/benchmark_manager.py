"""Main file for managing benchmark prompts/few-shot examples."""

from agential.cog.prompts.benchmarks.hotpotqa import (
    HOTPOTQA_FEWSHOT_EXAMPLES_COT,
    HOTPOTQA_FEWSHOT_EXAMPLES_DIRECT,
    HOTPOTQA_FEWSHOT_EXAMPLES_REACT
)

class Benchmarks:
    HOTPOTQA = "hotpotqa"

class FewShotType:
    COT = "cot"
    DIRECT = "direct"
    REACT = "react"

BENCHMARK_STRINGS = {
    Benchmarks.HOTPOTQA: {
        FewShotType.COT: HOTPOTQA_FEWSHOT_EXAMPLES_COT,
        FewShotType.DIRECT: HOTPOTQA_FEWSHOT_EXAMPLES_DIRECT,
        FewShotType.REACT: HOTPOTQA_FEWSHOT_EXAMPLES_REACT,
    }
}

def get_fewshot_examples(benchmark: str, fewshot_type: str) -> str:
    """Retrieve few-shot examples for a given benchmark category and type.

    Available Benchmarks:
        - Benchmarks.HOTPOTQA: "hotpotqa"

    Available Few-Shot Types:
        - FewShotType.COT: "cot"
        - FewShotType.DIRECT: "direct"
        - FewShotType.REACT: "react"

    Args:
        benchmark (str): The benchmark category. It should be one of the predefined categories in the Benchmarks class.
        fewshot_type (str): The type of few-shot examples. It should be one of the predefined types in the FewShotType class.

    Returns:
        str: The few-shot examples corresponding to the given benchmark and type.
        If the benchmark or few-shot type is not found, returns a message indicating that the benchmark or few-shot type was not found.
    """
    return BENCHMARK_STRINGS.get(benchmark, {}).get(fewshot_type, "Benchmark or Few-shot type not found")