"""ExpeL prompts and fewshot examples selector."""

from typing import Any, Dict

from agential.base.factory import BaseFactory
from agential.cog.constants import BENCHMARK_FEWSHOTS, Benchmarks, FewShotType
from agential.cog.expel.prompts import (
    EXPEL_REFLEXION_REACT_INSTRUCTION_AMBIGNQ,
    EXPEL_REFLEXION_REACT_INSTRUCTION_FEVER,
    EXPEL_REFLEXION_REACT_INSTRUCTION_GSM8K,
    EXPEL_REFLEXION_REACT_INSTRUCTION_HOTPOTQA,
    EXPEL_REFLEXION_REACT_INSTRUCTION_HUMANEVAL,
    EXPEL_REFLEXION_REACT_INSTRUCTION_MBPP,
    EXPEL_REFLEXION_REACT_INSTRUCTION_SVAMP,
    EXPEL_REFLEXION_REACT_INSTRUCTION_TABMWP,
    EXPEL_REFLEXION_REACT_INSTRUCTION_TRIVIAQA,
)
from agential.cog.expel.strategies.base import ExpeLBaseStrategy
from agential.cog.expel.strategies.general import (
    ExpeLAmbigNQStrategy,
    ExpeLFEVERStrategy,
    ExpeLHotQAStrategy,
    ExpeLTriviaQAStrategy,
)
from agential.cog.reflexion.prompts import (
    AMBIGNQ_FEWSHOT_EXAMPLES_REFLEXION_REACT_REFLECT,
    FEVER_FEWSHOT_EXAMPLES_REFLEXION_REACT_REFLECT,
    GSM8K_FEWSHOT_EXAMPLES_REFLEXION_REACT_REFLECT,
    HOTPOTQA_FEWSHOT_EXAMPLES_REFLEXION_REACT_REFLECT,
    HUMANEVAL_FEWSHOT_EXAMPLES_REFLEXION_REACT_REFLECT,
    MBPP_FEWSHOT_EXAMPLES_REFLEXION_REACT_REFLECT,
    REFLEXION_REACT_REFLECT_INSTRUCTION_AMBIGNQ,
    REFLEXION_REACT_REFLECT_INSTRUCTION_FEVER,
    REFLEXION_REACT_REFLECT_INSTRUCTION_GSM8K,
    REFLEXION_REACT_REFLECT_INSTRUCTION_HOTPOTQA,
    REFLEXION_REACT_REFLECT_INSTRUCTION_HUMANEVAL,
    REFLEXION_REACT_REFLECT_INSTRUCTION_MBPP,
    REFLEXION_REACT_REFLECT_INSTRUCTION_SVAMP,
    REFLEXION_REACT_REFLECT_INSTRUCTION_TABMWP,
    REFLEXION_REACT_REFLECT_INSTRUCTION_TRIVIAQA,
    SVAMP_FEWSHOT_EXAMPLES_REFLEXION_REACT_REFLECT,
    TABMWP_FEWSHOT_EXAMPLES_REFLEXION_REACT_REFLECT,
    TRIVIAQA_FEWSHOT_EXAMPLES_REFLEXION_REACT_REFLECT,
)

EXPEL_BENCHMARK_FEWSHOTS = {
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

EXPEL_PROMPTS = {
    Benchmarks.HOTPOTQA: {
        "prompt": EXPEL_REFLEXION_REACT_INSTRUCTION_HOTPOTQA,
        "reflect_prompt": REFLEXION_REACT_REFLECT_INSTRUCTION_HOTPOTQA,
    },
    Benchmarks.FEVER: {
        "prompt": EXPEL_REFLEXION_REACT_INSTRUCTION_FEVER,
        "reflect_prompt": REFLEXION_REACT_REFLECT_INSTRUCTION_FEVER,
    },
    Benchmarks.TRIVIAQA: {
        "prompt": EXPEL_REFLEXION_REACT_INSTRUCTION_TRIVIAQA,
        "reflect_prompt": REFLEXION_REACT_REFLECT_INSTRUCTION_TRIVIAQA,
    },
    Benchmarks.AMBIGNQ: {
        "prompt": EXPEL_REFLEXION_REACT_INSTRUCTION_AMBIGNQ,
        "reflect_prompt": REFLEXION_REACT_REFLECT_INSTRUCTION_AMBIGNQ,
    },
    Benchmarks.GSM8K: {
        "prompt": EXPEL_REFLEXION_REACT_INSTRUCTION_GSM8K,
        "reflect_prompt": REFLEXION_REACT_REFLECT_INSTRUCTION_GSM8K,
    },
    Benchmarks.SVAMP: {
        "prompt": EXPEL_REFLEXION_REACT_INSTRUCTION_SVAMP,
        "reflect_prompt": REFLEXION_REACT_REFLECT_INSTRUCTION_SVAMP,
    },
    Benchmarks.TABMWP: {
        "prompt": EXPEL_REFLEXION_REACT_INSTRUCTION_TABMWP,
        "reflect_prompt": REFLEXION_REACT_REFLECT_INSTRUCTION_TABMWP,
    },
    Benchmarks.HUMANEVAL: {
        "prompt": EXPEL_REFLEXION_REACT_INSTRUCTION_HUMANEVAL,
        "reflect_prompt": REFLEXION_REACT_REFLECT_INSTRUCTION_HUMANEVAL,
    },
    Benchmarks.MBPP: {
        "prompt": EXPEL_REFLEXION_REACT_INSTRUCTION_MBPP,
        "reflect_prompt": REFLEXION_REACT_REFLECT_INSTRUCTION_MBPP,
    },
}

EXPEL_FEWSHOTS = {
    Benchmarks.HOTPOTQA: {
        "reflect_examples": HOTPOTQA_FEWSHOT_EXAMPLES_REFLEXION_REACT_REFLECT,
    },
    Benchmarks.TRIVIAQA: {
        "reflect_examples": TRIVIAQA_FEWSHOT_EXAMPLES_REFLEXION_REACT_REFLECT,
    },
    Benchmarks.AMBIGNQ: {
        "reflect_examples": AMBIGNQ_FEWSHOT_EXAMPLES_REFLEXION_REACT_REFLECT,
    },
    Benchmarks.FEVER: {
        "reflect_examples": FEVER_FEWSHOT_EXAMPLES_REFLEXION_REACT_REFLECT,
    },
    Benchmarks.GSM8K: {
        "reflect_examples": GSM8K_FEWSHOT_EXAMPLES_REFLEXION_REACT_REFLECT,
    },
    Benchmarks.SVAMP: {
        "reflect_examples": SVAMP_FEWSHOT_EXAMPLES_REFLEXION_REACT_REFLECT,
    },
    Benchmarks.TABMWP: {
        "reflect_examples": TABMWP_FEWSHOT_EXAMPLES_REFLEXION_REACT_REFLECT,
    },
    Benchmarks.HUMANEVAL: {
        "reflect_examples": HUMANEVAL_FEWSHOT_EXAMPLES_REFLEXION_REACT_REFLECT,
    },
    Benchmarks.MBPP: {
        "reflect_examples": MBPP_FEWSHOT_EXAMPLES_REFLEXION_REACT_REFLECT,
    },
}


EXPEL_STRATEGIES = {
    Benchmarks.HOTPOTQA: ExpeLHotQAStrategy,
    Benchmarks.FEVER: ExpeLFEVERStrategy,
    Benchmarks.TRIVIAQA: ExpeLTriviaQAStrategy,
    Benchmarks.AMBIGNQ: ExpeLAmbigNQStrategy,
    Benchmarks.GSM8K: None,
    Benchmarks.SVAMP: None,
    Benchmarks.TABMWP: None,
    Benchmarks.HUMANEVAL: None,
    Benchmarks.MBPP: None,
}


class ExpeLFactory(BaseFactory):
    """A factory class for creating instances of ExpeL strategies and selecting prompts and few-shot examples."""

    @staticmethod
    def get_fewshots(
        benchmark: str, fewshot_type: str, **kwargs: Any
    ) -> Dict[str, str]:
        """Retrieve few-shot examples based on the benchmark.

        Args:
            benchmark (str): The benchmark name.
            fewshot_type (str): The benchmark few-shot type.
            **kwargs (Any): Additional arguments.

        Returns:
            Dict[str, str]: A dictionary of few-shot examples.
        """
        if benchmark not in EXPEL_FEWSHOTS:
            raise ValueError(f"Benchmark '{benchmark}' few-shots not found for ExpeL.")

        if fewshot_type not in EXPEL_BENCHMARK_FEWSHOTS[benchmark]:
            raise ValueError(
                f"Benchmark '{benchmark}' few-shot type not supported for ExpeL."
            )

        benchmark_fewshots = BENCHMARK_FEWSHOTS[benchmark][fewshot_type]

        return {"examples": benchmark_fewshots, **EXPEL_FEWSHOTS[benchmark]}

    @staticmethod
    def get_prompts(benchmark: str, **kwargs: Any) -> Dict[str, str]:
        """Retrieve the prompt instruction based on the benchmark.

        Args:
            benchmark (str): The benchmark name.
            **kwargs (Any): Additional arguments.

        Returns:
            Dict[str, str]: The prompt instructions.
        """
        if benchmark not in EXPEL_PROMPTS:
            raise ValueError(f"Benchmark '{benchmark}' prompt not found for ExpeL.")

        return EXPEL_PROMPTS[benchmark]

    @staticmethod
    def get_strategy(benchmark: str, **kwargs: Any) -> ExpeLBaseStrategy:
        """Returns an instance of the appropriate ExpeL strategy based on the provided benchmark.

        Args:
            benchmark (str): The benchmark name.
            **kwargs (Any): Additional keyword arguments to pass to
                the strategy's constructor.

        Returns:
            ExpeLBaseStrategy: An instance of the appropriate ExpeL strategy.
        """
        if benchmark not in EXPEL_STRATEGIES:
            raise ValueError(f"Unsupported benchmark: {benchmark} for agent ExpeL")

        strategy = EXPEL_STRATEGIES[benchmark]
        return strategy(**kwargs)
