"""Reflexion prompts and fewshot examples selector."""

from typing import Any, Dict

from agential.cog.base.factory import BaseFactory
from agential.cog.constants import BENCHMARK_FEWSHOTS, Benchmarks, FewShotType
from agential.cog.reflexion.prompts import (
    AMBIGNQ_FEWSHOT_EXAMPLES_REFLEXION_COT_REFLECT,
    AMBIGNQ_FEWSHOT_EXAMPLES_REFLEXION_REACT_REFLECT,
    FEVER_FEWSHOT_EXAMPLES_REFLEXION_COT_REFLECT,
    FEVER_FEWSHOT_EXAMPLES_REFLEXION_REACT_REFLECT,
    GSM8K_FEWSHOT_EXAMPLES_REFLEXION_COT_REFLECT,
    GSM8K_FEWSHOT_EXAMPLES_REFLEXION_REACT_REFLECT,
    HOTPOTQA_FEWSHOT_EXAMPLES_REFLEXION_COT_REFLECT,
    HOTPOTQA_FEWSHOT_EXAMPLES_REFLEXION_REACT_REFLECT,
    HUMANEVAL_FEWSHOT_EXAMPLES_REFLEXION_COT_REFLECT,
    HUMANEVAL_FEWSHOT_EXAMPLES_REFLEXION_REACT_REFLECT,
    MBPP_FEWSHOT_EXAMPLES_REFLEXION_COT_REFLECT,
    MBPP_FEWSHOT_EXAMPLES_REFLEXION_REACT_REFLECT,
    REFLEXION_COT_INSTRUCTION_AMBIGNQ,
    REFLEXION_COT_INSTRUCTION_FEVER,
    REFLEXION_COT_INSTRUCTION_GSM8K,
    REFLEXION_COT_INSTRUCTION_HOTPOTQA,
    REFLEXION_COT_INSTRUCTION_HUMANEVAL,
    REFLEXION_COT_INSTRUCTION_MBPP,
    REFLEXION_COT_INSTRUCTION_SVAMP,
    REFLEXION_COT_INSTRUCTION_TABMWP,
    REFLEXION_COT_INSTRUCTION_TRIVIAQA,
    REFLEXION_COT_REFLECT_INSTRUCTION_AMBIGNQ,
    REFLEXION_COT_REFLECT_INSTRUCTION_FEVER,
    REFLEXION_COT_REFLECT_INSTRUCTION_GSM8K,
    REFLEXION_COT_REFLECT_INSTRUCTION_HOTPOTQA,
    REFLEXION_COT_REFLECT_INSTRUCTION_HUMANEVAL,
    REFLEXION_COT_REFLECT_INSTRUCTION_MBPP,
    REFLEXION_COT_REFLECT_INSTRUCTION_SVAMP,
    REFLEXION_COT_REFLECT_INSTRUCTION_TABMWP,
    REFLEXION_COT_REFLECT_INSTRUCTION_TRIVIAQA,
    REFLEXION_REACT_INSTRUCTION_AMBIGNQ,
    REFLEXION_REACT_INSTRUCTION_FEVER,
    REFLEXION_REACT_INSTRUCTION_GSM8K,
    REFLEXION_REACT_INSTRUCTION_HOTPOTQA,
    REFLEXION_REACT_INSTRUCTION_HUMANEVAL,
    REFLEXION_REACT_INSTRUCTION_MBPP,
    REFLEXION_REACT_INSTRUCTION_SVAMP,
    REFLEXION_REACT_INSTRUCTION_TABMWP,
    REFLEXION_REACT_INSTRUCTION_TRIVIAQA,
    REFLEXION_REACT_REFLECT_INSTRUCTION_AMBIGNQ,
    REFLEXION_REACT_REFLECT_INSTRUCTION_FEVER,
    REFLEXION_REACT_REFLECT_INSTRUCTION_GSM8K,
    REFLEXION_REACT_REFLECT_INSTRUCTION_HOTPOTQA,
    REFLEXION_REACT_REFLECT_INSTRUCTION_HUMANEVAL,
    REFLEXION_REACT_REFLECT_INSTRUCTION_MBPP,
    REFLEXION_REACT_REFLECT_INSTRUCTION_SVAMP,
    REFLEXION_REACT_REFLECT_INSTRUCTION_TABMWP,
    REFLEXION_REACT_REFLECT_INSTRUCTION_TRIVIAQA,
    SVAMP_FEWSHOT_EXAMPLES_REFLEXION_COT_REFLECT,
    SVAMP_FEWSHOT_EXAMPLES_REFLEXION_REACT_REFLECT,
    TABMWP_FEWSHOT_EXAMPLES_REFLEXION_COT_REFLECT,
    TABMWP_FEWSHOT_EXAMPLES_REFLEXION_REACT_REFLECT,
    TRIVIAQA_FEWSHOT_EXAMPLES_REFLEXION_COT_REFLECT,
    TRIVIAQA_FEWSHOT_EXAMPLES_REFLEXION_REACT_REFLECT,
)
from agential.cog.reflexion.strategies.base import (
    ReflexionCoTBaseStrategy,
    ReflexionReActBaseStrategy,
)
from agential.cog.reflexion.strategies.code import (
    ReflexionCoTHEvalStrategy,
    ReflexionCoTMBPPStrategy,
    ReflexionReActHEvalStrategy,
    ReflexionReActMBPPStrategy,
)
from agential.cog.reflexion.strategies.math import (
    ReflexionCoTGSM8KStrategy,
    ReflexionCoTSVAMPStrategy,
    ReflexionCoTTabMWPStrategy,
    ReflexionReActGSM8KStrategy,
    ReflexionReActSVAMPStrategy,
    ReflexionReActTabMWPStrategy,
)
from agential.cog.reflexion.strategies.qa import (
    ReflexionCoTAmbigNQStrategy,
    ReflexionCoTFEVERStrategy,
    ReflexionCoTHotQAStrategy,
    ReflexionCoTTriviaQAStrategy,
    ReflexionReActAmbigNQStrategy,
    ReflexionReActFEVERStrategy,
    ReflexionReActHotQAStrategy,
    ReflexionReActTriviaQAStrategy,
)

REFLEXION_REACT_BENCHMARK_FEWSHOTS = {
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


REFLEXION_REACT_PROMPTS = {
    Benchmarks.HOTPOTQA: {
        "prompt": REFLEXION_REACT_INSTRUCTION_HOTPOTQA,
        "reflect_prompt": REFLEXION_REACT_REFLECT_INSTRUCTION_HOTPOTQA,
    },
    Benchmarks.FEVER: {
        "prompt": REFLEXION_REACT_INSTRUCTION_FEVER,
        "reflect_prompt": REFLEXION_REACT_REFLECT_INSTRUCTION_FEVER,
    },
    Benchmarks.TRIVIAQA: {
        "prompt": REFLEXION_REACT_INSTRUCTION_TRIVIAQA,
        "reflect_prompt": REFLEXION_REACT_REFLECT_INSTRUCTION_TRIVIAQA,
    },
    Benchmarks.AMBIGNQ: {
        "prompt": REFLEXION_REACT_INSTRUCTION_AMBIGNQ,
        "reflect_prompt": REFLEXION_REACT_REFLECT_INSTRUCTION_AMBIGNQ,
    },
    Benchmarks.GSM8K: {
        "prompt": REFLEXION_REACT_INSTRUCTION_GSM8K,
        "reflect_prompt": REFLEXION_REACT_REFLECT_INSTRUCTION_GSM8K,
    },
    Benchmarks.SVAMP: {
        "prompt": REFLEXION_REACT_INSTRUCTION_SVAMP,
        "reflect_prompt": REFLEXION_REACT_REFLECT_INSTRUCTION_SVAMP,
    },
    Benchmarks.TABMWP: {
        "prompt": REFLEXION_REACT_INSTRUCTION_TABMWP,
        "reflect_prompt": REFLEXION_REACT_REFLECT_INSTRUCTION_TABMWP,
    },
    Benchmarks.HUMANEVAL: {
        "prompt": REFLEXION_REACT_INSTRUCTION_HUMANEVAL,
        "reflect_prompt": REFLEXION_REACT_REFLECT_INSTRUCTION_HUMANEVAL,
    },
    Benchmarks.MBPP: {
        "prompt": REFLEXION_REACT_INSTRUCTION_MBPP,
        "reflect_prompt": REFLEXION_REACT_REFLECT_INSTRUCTION_MBPP,
    },
}


REFLEXION_REACT_FEWSHOTS = {
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


REFLEXION_REACT_STRATEGIES = {
    Benchmarks.HOTPOTQA: ReflexionReActHotQAStrategy,
    Benchmarks.FEVER: ReflexionReActFEVERStrategy,
    Benchmarks.TRIVIAQA: ReflexionReActTriviaQAStrategy,
    Benchmarks.AMBIGNQ: ReflexionReActAmbigNQStrategy,
    Benchmarks.GSM8K: ReflexionReActGSM8KStrategy,
    Benchmarks.SVAMP: ReflexionReActSVAMPStrategy,
    Benchmarks.TABMWP: ReflexionReActTabMWPStrategy,
    Benchmarks.HUMANEVAL: ReflexionReActHEvalStrategy,
    Benchmarks.MBPP: ReflexionReActMBPPStrategy,
}


class ReflexionReActFactory(BaseFactory):
    """A factory class for creating instances of ReflexionReAct strategies and selecting prompts and few-shot examples."""

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
        if benchmark not in REFLEXION_REACT_FEWSHOTS:
            raise ValueError(
                f"Benchmark '{benchmark}' few-shots not found for ReflexionReAct."
            )

        if fewshot_type not in REFLEXION_REACT_BENCHMARK_FEWSHOTS[benchmark]:
            raise ValueError(
                f"Benchmark '{benchmark}' few-shot type not supported for ReflexionReAct."
            )

        benchmark_fewshots = BENCHMARK_FEWSHOTS[benchmark][fewshot_type]

        return {"examples": benchmark_fewshots, **REFLEXION_REACT_FEWSHOTS[benchmark]}

    @staticmethod
    def get_prompts(benchmark: str, **kwargs: Any) -> Dict[str, str]:
        """Retrieve the prompt instruction based on the benchmark.

        Args:
            benchmark (str): The benchmark name.
            **kwargs (Any): Additional arguments.

        Returns:
            Dict[str, str]: The prompt instructions.
        """
        if benchmark not in REFLEXION_REACT_PROMPTS:
            raise ValueError(
                f"Benchmark '{benchmark}' prompt not found for ReflexionReAct."
            )

        return REFLEXION_REACT_PROMPTS[benchmark]

    @staticmethod
    def get_strategy(benchmark: str, **kwargs: Any) -> ReflexionReActBaseStrategy:
        """Returns an instance of the appropriate ReflexionReAct strategy based on the provided benchmark.

        Args:
            benchmark (str): The benchmark name.
            **kwargs (Any): Additional keyword arguments to pass to
                the strategy's constructor.

        Returns:
            ReflexionReActBaseStrategy: An instance of the appropriate ReflexionReAct strategy.
        """
        if benchmark not in REFLEXION_REACT_STRATEGIES:
            raise ValueError(
                f"Unsupported benchmark: {benchmark} for agent ReflexionReAct"
            )

        strategy = REFLEXION_REACT_STRATEGIES[benchmark]
        return strategy(**kwargs)
