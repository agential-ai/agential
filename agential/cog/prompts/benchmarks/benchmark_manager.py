"""Main file for managing benchmark prompts/few-shot examples."""

from agential.cog.prompts.benchmarks.hotpotqa import (
    HOTPOTQA_FEWSHOT_EXAMPLES_COT,
    HOTPOTQA_FEWSHOT_EXAMPLES_DIRECT,
    HOTPOTQA_FEWSHOT_EXAMPLES_REACT,
)
from agential.cog.prompts.benchmarks.fever import (
    FEVER_FEWSHOT_EXAMPLES_COT,
    FEVER_FEWSHOT_EXAMPLES_DIRECT,
    FEVER_FEWSHOT_EXAMPLES_REACT,
)
from agential.cog.prompts.benchmarks.triviaqa import (
    TRIVIAQA_FEWSHOT_EXAMPLES_COT,
    TRIVIAQA_FEWSHOT_EXAMPLES_DIRECT,
    TRIVIAQA_FEWSHOT_EXAMPLES_REACT,
)
from agential.cog.prompts.benchmarks.ambignq import (
    AMBIGNQ_FEWSHOT_EXAMPLES_COT,
    AMBIGNQ_FEWSHOT_EXAMPLES_DIRECT,
    AMBIGNQ_FEWSHOT_EXAMPLES_REACT,
)
from agential.cog.prompts.benchmarks.gsm8k import (
    GSM8K_FEWSHOT_EXAMPLES_POT,
)
from agential.cog.prompts.benchmarks.svamp import (
    SVAMP_FEWSHOT_EXAMPLES_POT,
)
from agential.cog.prompts.benchmarks.tabmwp import (
    TABMWP_FEWSHOT_EXAMPLES_POT,
)
from agential.cog.prompts.benchmarks.humaneval import (
    HUMANEVAL_FEWSHOT_EXAMPLES_POT,
)
from agential.cog.prompts.benchmarks.mbpp import (
    MBPP_FEWSHOT_EXAMPLES_POT,
)

class Benchmarks:
    class QA:
        HOTPOTQA = "hotpotqa"
        FEVER = "fever"
        TRIVIAQA = "triviaqa"
        AMBIGNQ = "ambignq"

    class Math:
        GSM8K = "gsm8k"
        SVAMP = "svamp"
        TABMWP = "tabmwp"

    class Code:
        HUMANEVAL = "humaneval"
        MBPP = "mbpp"

class FewShotType:
    COT = "cot"
    DIRECT = "direct"
    REACT = "react"
    POT = "pot"

BENCHMARK_STRINGS = {
    Benchmarks.QA.HOTPOTQA: {
        FewShotType.COT: HOTPOTQA_FEWSHOT_EXAMPLES_COT,
        FewShotType.DIRECT: HOTPOTQA_FEWSHOT_EXAMPLES_DIRECT,
        FewShotType.REACT: HOTPOTQA_FEWSHOT_EXAMPLES_REACT,
    },
    Benchmarks.QA.FEVER: {
        FewShotType.COT: FEVER_FEWSHOT_EXAMPLES_COT,
        FewShotType.DIRECT: FEVER_FEWSHOT_EXAMPLES_DIRECT,
        FewShotType.REACT: FEVER_FEWSHOT_EXAMPLES_REACT,
    },
    Benchmarks.QA.TRIVIAQA: {
        FewShotType.COT: TRIVIAQA_FEWSHOT_EXAMPLES_COT,
        FewShotType.DIRECT: TRIVIAQA_FEWSHOT_EXAMPLES_DIRECT,
        FewShotType.REACT: TRIVIAQA_FEWSHOT_EXAMPLES_REACT,
    },
    Benchmarks.QA.AMBIGNQ: {
        FewShotType.COT: AMBIGNQ_FEWSHOT_EXAMPLES_COT,
        FewShotType.DIRECT: AMBIGNQ_FEWSHOT_EXAMPLES_DIRECT,
        FewShotType.REACT: AMBIGNQ_FEWSHOT_EXAMPLES_REACT,
    },
    Benchmarks.Math.GSM8K: {
        FewShotType.POT: GSM8K_FEWSHOT_EXAMPLES_POT,
    },
    Benchmarks.Math.SVAMP: {
        FewShotType.POT: SVAMP_FEWSHOT_EXAMPLES_POT,
    },
    Benchmarks.Math.TABMWP: {
        FewShotType.POT: TABMWP_FEWSHOT_EXAMPLES_POT,
    },
    Benchmarks.Code.HUMANEVAL: {
        FewShotType.POT: HUMANEVAL_FEWSHOT_EXAMPLES_POT,
    },
    Benchmarks.Code.MBPP: {
        FewShotType.POT: MBPP_FEWSHOT_EXAMPLES_POT,
    }
}

def get_fewshot_examples(mode: dict, fewshot_type: str) -> str:
    """Retrieve few-shot examples for a given benchmark type and benchmark name.

    Available Benchmark Types and Names:
        - QA: 
            - Benchmarks.QA.HOTPOTQA: Supports FewShotType.COT, FewShotType.DIRECT, FewShotType.REACT
            - Benchmarks.QA.FEVER: Supports FewShotType.COT, FewShotType.DIRECT, FewShotType.REACT
            - Benchmarks.QA.TRIVIAQA: Supports FewShotType.COT, FewShotType.DIRECT, FewShotType.REACT
            - Benchmarks.QA.AMBIGNQ: Supports FewShotType.COT, FewShotType.DIRECT, FewShotType.REACT
        - Math: 
            - Benchmarks.Math.GSM8K: Supports FewShotType.POT
            - Benchmarks.Math.SVAMP: Supports FewShotType.POT
            - Benchmarks.Math.TABMWP: Supports FewShotType.POT
        - Code: 
            - Benchmarks.Code.HUMANEVAL: Supports FewShotType.POT
            - Benchmarks.Code.MBPP: Supports FewShotType.POT

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
    try:
        benchmark_type, benchmark_name = list(mode.items())[0]
        if benchmark_type not in Benchmarks.__dict__:
            raise ValueError(f"Benchmark type '{benchmark_type}' not found.")
        
        if benchmark_name not in BENCHMARK_STRINGS:
            raise ValueError(f"Benchmark '{benchmark_name}' not found in benchmark type '{benchmark_type}'.")
        
        examples = BENCHMARK_STRINGS[benchmark_name].get(fewshot_type)
        if examples is None:
            raise ValueError(f"Few-shot type '{fewshot_type}' not found for benchmark '{benchmark_name}'.")
        
        return examples
    except Exception as e:
        return str(e)
