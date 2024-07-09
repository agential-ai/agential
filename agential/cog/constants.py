"""Constants for supported benchmarks, few-shot types, and agents."""
from agential.cog.fewshots.ambignq import (
    AMBIGNQ_FEWSHOT_EXAMPLES_COT,
    AMBIGNQ_FEWSHOT_EXAMPLES_DIRECT,
    AMBIGNQ_FEWSHOT_EXAMPLES_REACT,
)
from agential.cog.fewshots.fever import (
    FEVER_FEWSHOT_EXAMPLES_COT,
    FEVER_FEWSHOT_EXAMPLES_DIRECT,
    FEVER_FEWSHOT_EXAMPLES_REACT,
)
from agential.cog.fewshots.gsm8k import (
    GSM8K_FEWSHOT_EXAMPLES_COT,
    GSM8K_FEWSHOT_EXAMPLES_POT,
    GSM8K_FEWSHOT_EXAMPLES_REACT,
)
from agential.cog.fewshots.hotpotqa import (
    HOTPOTQA_FEWSHOT_EXAMPLES_COT,
    HOTPOTQA_FEWSHOT_EXAMPLES_DIRECT,
    HOTPOTQA_FEWSHOT_EXAMPLES_REACT,
)
from agential.cog.fewshots.humaneval import (
    HUMANEVAL_FEWSHOT_EXAMPLES_COT,
    HUMANEVAL_FEWSHOT_EXAMPLES_POT,
    HUMANEVAL_FEWSHOT_EXAMPLES_REACT,
)
from agential.cog.fewshots.mbpp import (
    MBPP_FEWSHOT_EXAMPLES_COT,
    MBPP_FEWSHOT_EXAMPLES_POT,
    MBPP_FEWSHOT_EXAMPLES_REACT,
)
from agential.cog.fewshots.svamp import (
    SVAMP_FEWSHOT_EXAMPLES_COT,
    SVAMP_FEWSHOT_EXAMPLES_POT,
    SVAMP_FEWSHOT_EXAMPLES_REACT,
)
from agential.cog.fewshots.tabmwp import (
    TABMWP_FEWSHOT_EXAMPLES_COT,
    TABMWP_FEWSHOT_EXAMPLES_POT,
    TABMWP_FEWSHOT_EXAMPLES_REACT,
)
from agential.cog.fewshots.triviaqa import (
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
