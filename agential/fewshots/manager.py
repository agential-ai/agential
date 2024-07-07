"""Main file for managing benchmark prompts/few-shot examples."""

from typing import Dict

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
from agential.cog.react.prompts import (
    REACT_INSTRUCTION_AMBIGNQ,
    REACT_INSTRUCTION_FEVER,
    REACT_INSTRUCTION_GSM8K,
    REACT_INSTRUCTION_HOTPOTQA,
    REACT_INSTRUCTION_HUMANEVAL,
    REACT_INSTRUCTION_MBPP,
    REACT_INSTRUCTION_SVAMP,
    REACT_INSTRUCTION_TABMWP,
    REACT_INSTRUCTION_TRIVIAQA,
)
from agential.cog.reflexion.prompts import (
    # Few-shots.
    HOTPOTQA_FEWSHOT_EXAMPLES_REFLEXION_COT_REFLECT,
    TRIVIAQA_FEWSHOT_EXAMPLES_REFLEXION_COT_REFLECT,
    AMBIGNQ_FEWSHOT_EXAMPLES_REFLEXION_COT_REFLECT,
    FEVER_FEWSHOT_EXAMPLES_REFLEXION_COT_REFLECT,
    GSM8K_FEWSHOT_EXAMPLES_REFLEXION_COT_REFLECT,
    SVAMP_FEWSHOT_EXAMPLES_REFLEXION_COT_REFLECT,
    TABMWP_FEWSHOT_EXAMPLES_REFLEXION_COT_REFLECT,
    HUMANEVAL_FEWSHOT_EXAMPLES_REFLEXION_COT_REFLECT,
    MBPP_FEWSHOT_EXAMPLES_REFLEXION_COT_REFLECT,
    HOTPOTQA_FEWSHOT_EXAMPLES_REFLEXION_REACT_REFLECT,
    TRIVIAQA_FEWSHOT_EXAMPLES_REFLEXION_REACT_REFLECT,
    AMBIGNQ_FEWSHOT_EXAMPLES_REFLEXION_REACT_REFLECT,
    FEVER_FEWSHOT_EXAMPLES_REFLEXION_REACT_REFLECT,
    GSM8K_FEWSHOT_EXAMPLES_REFLEXION_REACT_REFLECT,
    SVAMP_FEWSHOT_EXAMPLES_REFLEXION_REACT_REFLECT,
    TABMWP_FEWSHOT_EXAMPLES_REFLEXION_REACT_REFLECT,
    HUMANEVAL_FEWSHOT_EXAMPLES_REFLEXION_REACT_REFLECT,
    MBPP_FEWSHOT_EXAMPLES_REFLEXION_REACT_REFLECT,
    # Instructions.
    REFLEXION_COT_REFLECT_INSTRUCTION_HOTPOTQA,
    REFLEXION_COT_INSTRUCTION_HOTPOTQA,
    REFLEXION_REACT_INSTRUCTION_HOTPOTQA,
    REFLEXION_REACT_REFLECT_INSTRUCTION_HOTPOTQA,
    REFLEXION_COT_REFLECT_INSTRUCTION_TRIVIAQA,
    REFLEXION_COT_INSTRUCTION_TRIVIAQA,
    REFLEXION_REACT_INSTRUCTION_TRIVIAQA,
    REFLEXION_REACT_REFLECT_INSTRUCTION_TRIVIAQA,
    REFLEXION_COT_REFLECT_INSTRUCTION_AMBIGNQ,
    REFLEXION_COT_INSTRUCTION_AMBIGNQ,
    REFLEXION_REACT_INSTRUCTION_AMBIGNQ,
    REFLEXION_REACT_REFLECT_INSTRUCTION_AMBIGNQ,
    REFLEXION_COT_REFLECT_INSTRUCTION_FEVER,
    REFLEXION_COT_INSTRUCTION_FEVER,
    REFLEXION_REACT_INSTRUCTION_FEVER,
    REFLEXION_REACT_REFLECT_INSTRUCTION_FEVER,
    REFLEXION_COT_REFLECT_INSTRUCTION_GSM8K,
    REFLEXION_COT_INSTRUCTION_GSM8K,
    REFLEXION_REACT_INSTRUCTION_GSM8K,
    REFLEXION_REACT_REFLECT_INSTRUCTION_GSM8K,
    REFLEXION_COT_REFLECT_INSTRUCTION_SVAMP,
    REFLEXION_COT_INSTRUCTION_SVAMP,
    REFLEXION_REACT_INSTRUCTION_SVAMP,
    REFLEXION_REACT_REFLECT_INSTRUCTION_SVAMP,
    REFLEXION_COT_REFLECT_INSTRUCTION_TABMWP,
    REFLEXION_COT_INSTRUCTION_TABMWP,
    REFLEXION_REACT_INSTRUCTION_TABMWP,
    REFLEXION_REACT_REFLECT_INSTRUCTION_TABMWP,
    REFLEXION_COT_REFLECT_INSTRUCTION_HUMANEVAL,
    REFLEXION_COT_INSTRUCTION_HUMANEVAL,
    REFLEXION_REACT_INSTRUCTION_HUMANEVAL,
    REFLEXION_REACT_REFLECT_INSTRUCTION_HUMANEVAL,
    REFLEXION_COT_REFLECT_INSTRUCTION_MBPP,
    REFLEXION_COT_INSTRUCTION_MBPP,
    REFLEXION_REACT_INSTRUCTION_MBPP,
    REFLEXION_REACT_REFLECT_INSTRUCTION_MBPP,
)
from agential.cog.critic.prompts import (
    # Few-shots.
    FEVER_FEWSHOT_EXAMPLES_CRITIC,
    AMBIGNQ_FEWSHOT_EXAMPLES_CRITIC,
    HOTPOTQA_FEWSHOT_EXAMPLES_CRITIC,
    TRIVIAQA_FEWSHOT_EXAMPLES_CRITIC,
    GSM8K_FEWSHOT_EXAMPLES_CRITIC,
    GSM8K_FEWSHOT_EXAMPLES_CRITIC_NO_TOOL,
    SVAMP_FEWSHOT_EXAMPLES_CRITIC,
    SVAMP_FEWSHOT_EXAMPLES_CRITIC_NO_TOOL,
    TABMWP_FEWSHOT_EXAMPLES_CRITIC,
    TABMWP_FEWSHOT_EXAMPLES_CRITIC_NO_TOOL,
    HUMANEVAL_FEWSHOT_EXAMPLES_CRITIC,
    HUMANEVAL_FEWSHOT_EXAMPLES_CRITIC_NO_TOOL,
    MBPP_FEWSHOT_EXAMPLES_CRITIC,
    MBPP_FEWSHOT_EXAMPLES_CRITIC_NO_TOOL,
    # Instructions.
    CRITIC_INSTRUCTION_FEVER,
    CRITIC_CRITIQUE_INSTRUCTION_FEVER,
    CRITIC_INSTRUCTION_HOTPOTQA,
    CRITIC_CRITIQUE_INSTRUCTION_HOTPOTQA,
    CRITIC_INSTRUCTION_AMBIGNQ,
    CRITIC_CRITIQUE_INSTRUCTION_AMBIGNQ,
    CRITIC_INSTRUCTION_TRIVIAQA,
    CRITIC_CRITIQUE_INSTRUCTION_TRIVIAQA,
    CRITIC_POT_INSTRUCTION_GSM8K,
    CRITIC_CRITIQUE_INSTRUCTION_GSM8K,
    CRITIC_CRITIQUE_NO_TOOL_INSTRUCTION_GSM8K,
    CRITIC_POT_INSTRUCTION_SVAMP,
    CRITIC_CRITIQUE_INSTRUCTION_SVAMP,
    CRITIC_CRITIQUE_NO_TOOL_INSTRUCTION_SVAMP,
    CRITIC_POT_INSTRUCTION_TABMWP,
    CRITIC_CRITIQUE_INSTRUCTION_TABMWP,
    CRITIC_CRITIQUE_NO_TOOL_INSTRUCTION_TABMWP,
    CRITIC_POT_INSTRUCTION_HUMANEVAL,
    CRITIC_CRITIQUE_INSTRUCTION_HUMANEVAL,
    CRITIC_CRITIQUE_NO_TOOL_INSTRUCTION_HUMANEVAL,
    CRITIC_POT_INSTRUCTION_MBPP,
    CRITIC_CRITIQUE_INSTRUCTION_MBPP,
    CRITIC_CRITIQUE_NO_TOOL_INSTRUCTION_MBPP,
)


class Benchmarks:
    """Supported benchmarks."""

    QA = "qa"
    MATH = "math"
    CODE = "code"

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


BENCHMARK_FEWSHOTS = {
    Benchmarks.QA: {
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
    },
    Benchmarks.MATH: {
        Benchmarks.math.GSM8K: {
            FewShotType.POT: GSM8K_FEWSHOT_EXAMPLES_POT,
            FewShotType.REACT: GSM8K_FEWSHOT_EXAMPLES_REACT,
            FewShotType.COT: GSM8K_FEWSHOT_EXAMPLES_COT,
        },
        Benchmarks.math.SVAMP: {
            FewShotType.POT: SVAMP_FEWSHOT_EXAMPLES_POT,
            FewShotType.REACT: SVAMP_FEWSHOT_EXAMPLES_REACT,
            FewShotType.COT: SVAMP_FEWSHOT_EXAMPLES_COT,
        },
        Benchmarks.math.TABMWP: {
            FewShotType.POT: TABMWP_FEWSHOT_EXAMPLES_POT,
            FewShotType.REACT: TABMWP_FEWSHOT_EXAMPLES_REACT,
            FewShotType.COT: TABMWP_FEWSHOT_EXAMPLES_COT,
        },
    },
    Benchmarks.CODE: {
        Benchmarks.code.HUMANEVAL: {
            FewShotType.POT: HUMANEVAL_FEWSHOT_EXAMPLES_POT,
            FewShotType.REACT: HUMANEVAL_FEWSHOT_EXAMPLES_REACT,
            FewShotType.COT: HUMANEVAL_FEWSHOT_EXAMPLES_COT,
        },
        Benchmarks.code.MBPP: {
            FewShotType.POT: MBPP_FEWSHOT_EXAMPLES_POT,
            FewShotType.REACT: MBPP_FEWSHOT_EXAMPLES_REACT,
            FewShotType.COT: MBPP_FEWSHOT_EXAMPLES_COT,
        },
    },
}


class Agents:
    """Supported agents."""

    REACT = "react"
    REFLEXION_COT = "reflexion_cot"
    REFLEXION_REACT = "reflexion_react"
    CRITIC = "critic"


AGENT_FEWSHOTS = {
    Agents.REACT: {},
    Agents.REFLEXION_COT: {
        Benchmarks.qa.HOTPOTQA: HOTPOTQA_FEWSHOT_EXAMPLES_REFLEXION_COT_REFLECT,
        Benchmarks.qa.TRIVIAQA: TRIVIAQA_FEWSHOT_EXAMPLES_REFLEXION_COT_REFLECT,
        Benchmarks.qa.AMBIGNQ: AMBIGNQ_FEWSHOT_EXAMPLES_REFLEXION_COT_REFLECT,
        Benchmarks.qa.FEVER: FEVER_FEWSHOT_EXAMPLES_REFLEXION_COT_REFLECT,
        Benchmarks.math.GSM8K: GSM8K_FEWSHOT_EXAMPLES_REFLEXION_COT_REFLECT,
        Benchmarks.math.SVAMP: SVAMP_FEWSHOT_EXAMPLES_REFLEXION_COT_REFLECT,
        Benchmarks.math.TABMWP: TABMWP_FEWSHOT_EXAMPLES_REFLEXION_COT_REFLECT,
        Benchmarks.code.HUMANEVAL: HUMANEVAL_FEWSHOT_EXAMPLES_REFLEXION_COT_REFLECT,
        Benchmarks.code.MBPP: MBPP_FEWSHOT_EXAMPLES_REFLEXION_COT_REFLECT,
    },
    Agents.REFLEXION_REACT: {
        Benchmarks.qa.HOTPOTQA: HOTPOTQA_FEWSHOT_EXAMPLES_REFLEXION_REACT_REFLECT,
        Benchmarks.qa.TRIVIAQA: TRIVIAQA_FEWSHOT_EXAMPLES_REFLEXION_REACT_REFLECT,
        Benchmarks.qa.AMBIGNQ: AMBIGNQ_FEWSHOT_EXAMPLES_REFLEXION_REACT_REFLECT,
        Benchmarks.qa.FEVER: FEVER_FEWSHOT_EXAMPLES_REFLEXION_REACT_REFLECT,
        Benchmarks.math.GSM8K: GSM8K_FEWSHOT_EXAMPLES_REFLEXION_REACT_REFLECT,
        Benchmarks.math.SVAMP: SVAMP_FEWSHOT_EXAMPLES_REFLEXION_REACT_REFLECT,
        Benchmarks.math.TABMWP: TABMWP_FEWSHOT_EXAMPLES_REFLEXION_REACT_REFLECT,
        Benchmarks.code.HUMANEVAL: HUMANEVAL_FEWSHOT_EXAMPLES_REFLEXION_REACT_REFLECT,
        Benchmarks.code.MBPP: MBPP_FEWSHOT_EXAMPLES_REFLEXION_REACT_REFLECT,
    },
    Agents.CRITIC: {
        Benchmarks.qa.HOTPOTQA: {
            "tool": HOTPOTQA_FEWSHOT_EXAMPLES_CRITIC,
            "no_tool": HOTPOTQA_FEWSHOT_EXAMPLES_CRITIC,
        },
        Benchmarks.qa.FEVER: {
            "tool": FEVER_FEWSHOT_EXAMPLES_CRITIC,
            "no_tool": FEVER_FEWSHOT_EXAMPLES_CRITIC,
        },
        Benchmarks.qa.TRIVIAQA: {
            "tool": TRIVIAQA_FEWSHOT_EXAMPLES_CRITIC,
            "no_tool": TRIVIAQA_FEWSHOT_EXAMPLES_CRITIC,
        },
        Benchmarks.qa.AMBIGNQ: {
            "tool": AMBIGNQ_FEWSHOT_EXAMPLES_CRITIC,
            "no_tool": AMBIGNQ_FEWSHOT_EXAMPLES_CRITIC,
        },
        Benchmarks.math.GSM8K: {
            "tool": GSM8K_FEWSHOT_EXAMPLES_CRITIC,
            "no_tool": GSM8K_FEWSHOT_EXAMPLES_CRITIC_NO_TOOL,
        },
        Benchmarks.math.SVAMP: {
            "tool": SVAMP_FEWSHOT_EXAMPLES_CRITIC,
            "no_tool": SVAMP_FEWSHOT_EXAMPLES_CRITIC_NO_TOOL,
        },
        Benchmarks.math.TABMWP: {
            "tool": TABMWP_FEWSHOT_EXAMPLES_CRITIC,
            "no_tool": TABMWP_FEWSHOT_EXAMPLES_CRITIC_NO_TOOL,
        },
        Benchmarks.code.HUMANEVAL: {
            "tool": HUMANEVAL_FEWSHOT_EXAMPLES_CRITIC,
            "no_tool": HUMANEVAL_FEWSHOT_EXAMPLES_CRITIC_NO_TOOL,
        },
        Benchmarks.code.MBPP: {
            "tool": MBPP_FEWSHOT_EXAMPLES_CRITIC,
            "no_tool": MBPP_FEWSHOT_EXAMPLES_CRITIC_NO_TOOL,
        },
    },
}


AGENT_PROMPTS = {
    Agents.REACT: {
        Benchmarks.qa.HOTPOTQA: {
            "prompt": REACT_INSTRUCTION_HOTPOTQA,
        },
        Benchmarks.qa.FEVER: {
            "prompt": REACT_INSTRUCTION_FEVER,
        },
        Benchmarks.qa.TRIVIAQA: {
            "prompt": REACT_INSTRUCTION_TRIVIAQA,
        },
        Benchmarks.qa.AMBIGNQ: {
            "prompt": REACT_INSTRUCTION_AMBIGNQ,
        },
        Benchmarks.math.GSM8K: {
            "prompt": REACT_INSTRUCTION_GSM8K,
        },
        Benchmarks.math.SVAMP: {
            "prompt": REACT_INSTRUCTION_SVAMP,
        },
        Benchmarks.math.TABMWP: {
            "prompt": REACT_INSTRUCTION_TABMWP,
        },
        Benchmarks.code.HUMANEVAL: {
            "prompt": REACT_INSTRUCTION_HUMANEVAL,
        },
        Benchmarks.code.MBPP: {
            "prompt": REACT_INSTRUCTION_MBPP,
        },
    },
    Agents.REFLEXION_COT: {
        Benchmarks.qa.HOTPOTQA: {
            "prompt": REFLEXION_COT_INSTRUCTION_HOTPOTQA,
            "reflect_prompt": REFLEXION_COT_REFLECT_INSTRUCTION_HOTPOTQA,
        },
        Benchmarks.qa.FEVER: {
            "prompt": REFLEXION_COT_INSTRUCTION_FEVER,
            "reflect_prompt": REFLEXION_COT_REFLECT_INSTRUCTION_FEVER,
        },
        Benchmarks.qa.TRIVIAQA: {
            "prompt": REFLEXION_COT_INSTRUCTION_TRIVIAQA,
            "reflect_prompt": REFLEXION_COT_REFLECT_INSTRUCTION_TRIVIAQA,
        },
        Benchmarks.qa.AMBIGNQ: {
            "prompt": REFLEXION_COT_INSTRUCTION_AMBIGNQ,
            "reflect_prompt": REFLEXION_COT_REFLECT_INSTRUCTION_AMBIGNQ,
        },
        Benchmarks.math.GSM8K: {
            "prompt": REFLEXION_COT_INSTRUCTION_GSM8K,
            "reflect_prompt": REFLEXION_COT_REFLECT_INSTRUCTION_GSM8K,
        },
        Benchmarks.math.SVAMP: {
            "prompt": REFLEXION_COT_INSTRUCTION_SVAMP,
            "reflect_prompt": REFLEXION_COT_REFLECT_INSTRUCTION_SVAMP,
        },
        Benchmarks.math.TABMWP: {
            "prompt": REFLEXION_COT_INSTRUCTION_TABMWP,
            "reflect_prompt": REFLEXION_COT_REFLECT_INSTRUCTION_TABMWP,
        },
        Benchmarks.code.HUMANEVAL: {
            "prompt": REFLEXION_COT_INSTRUCTION_HUMANEVAL,
            "reflect_prompt": REFLEXION_COT_REFLECT_INSTRUCTION_HUMANEVAL,
        },
        Benchmarks.code.MBPP: {
            "prompt": REFLEXION_COT_INSTRUCTION_MBPP,
            "reflect_prompt": REFLEXION_COT_REFLECT_INSTRUCTION_MBPP,
        },
    },
    Agents.REFLEXION_REACT: {
        Benchmarks.qa.HOTPOTQA: {
            "prompt": REFLEXION_REACT_INSTRUCTION_HOTPOTQA,
            "reflect_prompt": REFLEXION_REACT_REFLECT_INSTRUCTION_HOTPOTQA,
        },
        Benchmarks.qa.FEVER: {
            "prompt": REFLEXION_REACT_INSTRUCTION_FEVER,
            "reflect_prompt": REFLEXION_REACT_REFLECT_INSTRUCTION_FEVER,
        },
        Benchmarks.qa.TRIVIAQA: {
            "prompt": REFLEXION_REACT_INSTRUCTION_TRIVIAQA,
            "reflect_prompt": REFLEXION_REACT_REFLECT_INSTRUCTION_TRIVIAQA,
        },
        Benchmarks.qa.AMBIGNQ: {
            "prompt": REFLEXION_REACT_INSTRUCTION_AMBIGNQ,
            "reflect_prompt": REFLEXION_REACT_REFLECT_INSTRUCTION_AMBIGNQ,
        },
        Benchmarks.math.GSM8K: {
            "prompt": REFLEXION_REACT_INSTRUCTION_GSM8K,
            "reflect_prompt": REFLEXION_REACT_REFLECT_INSTRUCTION_GSM8K,
        },
        Benchmarks.math.SVAMP: {
            "prompt": REFLEXION_REACT_INSTRUCTION_SVAMP,
            "reflect_prompt": REFLEXION_REACT_REFLECT_INSTRUCTION_SVAMP,
        },
        Benchmarks.math.TABMWP: {
            "prompt": REFLEXION_REACT_INSTRUCTION_TABMWP,
            "reflect_prompt": REFLEXION_REACT_REFLECT_INSTRUCTION_TABMWP,
        },
        Benchmarks.code.HUMANEVAL: {
            "prompt": REFLEXION_REACT_INSTRUCTION_HUMANEVAL,
            "reflect_prompt": REFLEXION_REACT_REFLECT_INSTRUCTION_HUMANEVAL,
        },
        Benchmarks.code.MBPP: {
            "prompt": REFLEXION_REACT_INSTRUCTION_MBPP,
            "reflect_prompt": REFLEXION_REACT_REFLECT_INSTRUCTION_MBPP,
        },
    },
    Agents.CRITIC: {
        Benchmarks.qa.HOTPOTQA: {
            "prompt": CRITIC_INSTRUCTION_HOTPOTQA,
            "critique_prompt": CRITIC_CRITIQUE_INSTRUCTION_HOTPOTQA,
            "critique_prompt_no_tool": CRITIC_CRITIQUE_INSTRUCTION_HOTPOTQA,
        },
        Benchmarks.qa.FEVER: {
            "prompt": CRITIC_INSTRUCTION_FEVER,
            "critique_prompt": CRITIC_CRITIQUE_INSTRUCTION_FEVER,
            "critique_prompt_no_tool": CRITIC_CRITIQUE_INSTRUCTION_FEVER,
        },
        Benchmarks.qa.TRIVIAQA: {
            "prompt": CRITIC_INSTRUCTION_TRIVIAQA,
            "critique_prompt": CRITIC_CRITIQUE_INSTRUCTION_TRIVIAQA,
            "critique_prompt_no_tool": CRITIC_CRITIQUE_INSTRUCTION_TRIVIAQA,
        },
        Benchmarks.qa.AMBIGNQ: {
            "prompt": CRITIC_INSTRUCTION_AMBIGNQ,
            "critique_prompt": CRITIC_CRITIQUE_INSTRUCTION_AMBIGNQ,
            "critique_prompt_no_tool": CRITIC_CRITIQUE_INSTRUCTION_AMBIGNQ,
        },
        Benchmarks.math.GSM8K: {
            "prompt": CRITIC_POT_INSTRUCTION_GSM8K,
            "critique_prompt": CRITIC_CRITIQUE_INSTRUCTION_GSM8K,
            "critique_prompt_no_tool": CRITIC_CRITIQUE_NO_TOOL_INSTRUCTION_GSM8K,
        },
        Benchmarks.math.SVAMP: {
            "prompt": CRITIC_POT_INSTRUCTION_SVAMP,
            "critique_prompt": CRITIC_CRITIQUE_INSTRUCTION_SVAMP,
            "critique_prompt_no_tool": CRITIC_CRITIQUE_NO_TOOL_INSTRUCTION_SVAMP,
        },
        Benchmarks.math.TABMWP: {
            "prompt": CRITIC_POT_INSTRUCTION_TABMWP,
            "critique_prompt": CRITIC_CRITIQUE_INSTRUCTION_TABMWP,
            "critique_prompt_no_tool": CRITIC_CRITIQUE_NO_TOOL_INSTRUCTION_TABMWP,
        },
        Benchmarks.code.HUMANEVAL: {
            "prompt": CRITIC_POT_INSTRUCTION_HUMANEVAL,
            "critique_prompt": CRITIC_CRITIQUE_INSTRUCTION_HUMANEVAL,
            "critique_prompt_no_tool": CRITIC_CRITIQUE_NO_TOOL_INSTRUCTION_HUMANEVAL,
        },
        Benchmarks.code.MBPP: {
            "prompt": CRITIC_POT_INSTRUCTION_MBPP,
            "critique_prompt": CRITIC_CRITIQUE_INSTRUCTION_MBPP,
            "critique_prompt_no_tool": CRITIC_CRITIQUE_NO_TOOL_INSTRUCTION_MBPP,
        },
    },
}


def get_fewshot_examples(mode: Dict[str, str], fewshot_type: str) -> str:
    """Retrieve few-shot examples for a given benchmark type and benchmark name.

    Available Benchmark Types and Names:
        - qa:
            - hotpotqa: Supports "cot", "direct", "react"
            - fever: Supports "cot", "direct", "react"
            - triviaqa: Supports "cot", "direct", "react"
            - ambignq: Supports "cot", "direct", "react"
        - math:
            - gsm8k: Supports "pot", "cot", "react"
            - svamp: Supports "pot", "cot", "react"
            - tabmwp: Supports "pot", "cot", "react"
        - code:
            - humaneval: Supports "pot", "cot", "react"
            - mbpp: Supports "pot", "cot", "react"

    Available Few-Shot Types:
        - "cot"
        - "direct"
        - "react"
        - "pot"

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

    if benchmark_name not in BENCHMARK_FEWSHOTS[benchmark_type]:
        raise ValueError(
            f"Benchmark '{benchmark_name}' not found in benchmark type '{benchmark_type}'."
        )

    examples = BENCHMARK_FEWSHOTS[benchmark_type][benchmark_name].get(fewshot_type)
    if examples is None:
        raise ValueError(
            f"Few-shot type '{fewshot_type}' not found for benchmark '{benchmark_name}' of benchmark type '{benchmark_type}'."
        )

    return examples
