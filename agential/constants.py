"""Constants for supported benchmarks and few-shot types."""

from agential.core.fewshots.ambignq import (
    AMBIGNQ_FEWSHOT_EXAMPLES_COT,
    AMBIGNQ_FEWSHOT_EXAMPLES_DIRECT,
    AMBIGNQ_FEWSHOT_EXAMPLES_REACT,
)
from agential.core.fewshots.fever import (
    FEVER_FEWSHOT_EXAMPLES_COT,
    FEVER_FEWSHOT_EXAMPLES_DIRECT,
    FEVER_FEWSHOT_EXAMPLES_REACT,
)
from agential.core.fewshots.gsm8k import (
    GSM8K_FEWSHOT_EXAMPLES_COT,
    GSM8K_FEWSHOT_EXAMPLES_POT,
    GSM8K_FEWSHOT_EXAMPLES_REACT,
)
from agential.core.fewshots.hotpotqa import (
    HOTPOTQA_FEWSHOT_EXAMPLES_COT,
    HOTPOTQA_FEWSHOT_EXAMPLES_DIRECT,
    HOTPOTQA_FEWSHOT_EXAMPLES_REACT,
)
from agential.core.fewshots.humaneval import (
    HUMANEVAL_FEWSHOT_EXAMPLES_COT,
    HUMANEVAL_FEWSHOT_EXAMPLES_POT,
    HUMANEVAL_FEWSHOT_EXAMPLES_REACT,
)
from agential.core.fewshots.mbpp import (
    MBPP_FEWSHOT_EXAMPLES_COT,
    MBPP_FEWSHOT_EXAMPLES_POT,
    MBPP_FEWSHOT_EXAMPLES_REACT,
)
from agential.core.fewshots.svamp import (
    SVAMP_FEWSHOT_EXAMPLES_COT,
    SVAMP_FEWSHOT_EXAMPLES_POT,
    SVAMP_FEWSHOT_EXAMPLES_REACT,
)
from agential.core.fewshots.tabmwp import (
    TABMWP_FEWSHOT_EXAMPLES_COT,
    TABMWP_FEWSHOT_EXAMPLES_POT,
    TABMWP_FEWSHOT_EXAMPLES_REACT,
)
from agential.core.fewshots.triviaqa import (
    TRIVIAQA_FEWSHOT_EXAMPLES_COT,
    TRIVIAQA_FEWSHOT_EXAMPLES_DIRECT,
    TRIVIAQA_FEWSHOT_EXAMPLES_REACT,
)


class Benchmarks:
    """Supported benchmarks."""

    # QA.
    HOTPOTQA = "hotpotqa"
    FEVER = "fever"
    TRIVIAQA = "triviaqa"
    AMBIGNQ = "ambignq"

    # Math.
    GSM8K = "gsm8k"
    SVAMP = "svamp"
    TABMWP = "tabmwp"

    # Code.
    HUMANEVAL = "humaneval"
    MBPP = "mbpp"


class FewShotType:
    """Few-shot types."""

    COT = "cot"
    DIRECT = "direct"
    REACT = "react"
    POT = "pot"


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
        FewShotType.COT: GSM8K_FEWSHOT_EXAMPLES_COT,
        FewShotType.POT: GSM8K_FEWSHOT_EXAMPLES_POT,
        FewShotType.REACT: GSM8K_FEWSHOT_EXAMPLES_REACT,
    },
    Benchmarks.SVAMP: {
        FewShotType.COT: SVAMP_FEWSHOT_EXAMPLES_COT,
        FewShotType.POT: SVAMP_FEWSHOT_EXAMPLES_POT,
        FewShotType.REACT: SVAMP_FEWSHOT_EXAMPLES_REACT,
    },
    Benchmarks.TABMWP: {
        FewShotType.COT: TABMWP_FEWSHOT_EXAMPLES_COT,
        FewShotType.POT: TABMWP_FEWSHOT_EXAMPLES_POT,
        FewShotType.REACT: TABMWP_FEWSHOT_EXAMPLES_REACT,
    },
    Benchmarks.HUMANEVAL: {
        FewShotType.COT: HUMANEVAL_FEWSHOT_EXAMPLES_COT,
        FewShotType.POT: HUMANEVAL_FEWSHOT_EXAMPLES_POT,
        FewShotType.REACT: HUMANEVAL_FEWSHOT_EXAMPLES_REACT,
    },
    Benchmarks.MBPP: {
        FewShotType.COT: MBPP_FEWSHOT_EXAMPLES_COT,
        FewShotType.POT: MBPP_FEWSHOT_EXAMPLES_POT,
        FewShotType.REACT: MBPP_FEWSHOT_EXAMPLES_REACT,
    },
}
