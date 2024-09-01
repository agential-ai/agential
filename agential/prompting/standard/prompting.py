"""Standard prompting module."""

from typing import Any, Dict, List, Optional

from agential.constants import BENCHMARK_FEWSHOTS, Benchmarks, FewShotType
from agential.core.base.prompting.prompting import BasePrompting
from agential.llm.llm import BaseLLM
from agential.prompting.standard.output import StandardOutput
from agential.prompting.standard.prompts import (
    STANDARD_INSTRUCTION_AMBIGNQ,
    STANDARD_INSTRUCTION_FEVER,
    STANDARD_INSTRUCTION_GSM8K,
    STANDARD_INSTRUCTION_HOTPOTQA,
    STANDARD_INSTRUCTION_HUMANEVAL,
    STANDARD_INSTRUCTION_MBPP,
    STANDARD_INSTRUCTION_SVAMP,
    STANDARD_INSTRUCTION_TABMWP,
    STANDARD_INSTRUCTION_TRIVIAQA,
)
from agential.prompting.standard.strategies.base import StandardBaseStrategy
from agential.prompting.standard.strategies.code import (
    StandardHEvalStrategy,
    StandardMBPPStrategy,
)
from agential.prompting.standard.strategies.math import (
    StandardGSM8KStrategy,
    StandardSVAMPStrategy,
    StandardTabMWPStrategy,
)
from agential.prompting.standard.strategies.qa import (
    StandardAmbigNQStrategy,
    StandardFEVERStrategy,
    StandardHotQAStrategy,
    StandardTriviaQAStrategy,
)

STANDARD_BENCHMARK_FEWSHOTS = {
    Benchmarks.HOTPOTQA: [FewShotType.DIRECT],
    Benchmarks.FEVER: [FewShotType.DIRECT],
    Benchmarks.TRIVIAQA: [FewShotType.DIRECT],
    Benchmarks.AMBIGNQ: [FewShotType.DIRECT],
    Benchmarks.GSM8K: [FewShotType.DIRECT],
    Benchmarks.SVAMP: [FewShotType.DIRECT],
    Benchmarks.TABMWP: [FewShotType.DIRECT],
    Benchmarks.HUMANEVAL: [FewShotType.DIRECT],
    Benchmarks.MBPP: [FewShotType.DIRECT],
}

STANDARD_PROMPTS = {
    Benchmarks.HOTPOTQA: {
        "prompt": STANDARD_INSTRUCTION_HOTPOTQA,
    },
    Benchmarks.FEVER: {
        "prompt": STANDARD_INSTRUCTION_FEVER,
    },
    Benchmarks.TRIVIAQA: {
        "prompt": STANDARD_INSTRUCTION_TRIVIAQA,
    },
    Benchmarks.AMBIGNQ: {
        "prompt": STANDARD_INSTRUCTION_AMBIGNQ,
    },
    Benchmarks.GSM8K: {
        "prompt": STANDARD_INSTRUCTION_GSM8K,
    },
    Benchmarks.SVAMP: {
        "prompt": STANDARD_INSTRUCTION_SVAMP,
    },
    Benchmarks.TABMWP: {
        "prompt": STANDARD_INSTRUCTION_TABMWP,
    },
    Benchmarks.HUMANEVAL: {
        "prompt": STANDARD_INSTRUCTION_HUMANEVAL,
    },
    Benchmarks.MBPP: {
        "prompt": STANDARD_INSTRUCTION_MBPP,
    },
}
STANDARD_FEWSHOTS: Dict[str, Dict] = {
    Benchmarks.HOTPOTQA: {},
    Benchmarks.FEVER: {},
    Benchmarks.TRIVIAQA: {},
    Benchmarks.AMBIGNQ: {},
    Benchmarks.GSM8K: {},
    Benchmarks.SVAMP: {},
    Benchmarks.TABMWP: {},
    Benchmarks.HUMANEVAL: {},
    Benchmarks.MBPP: {},
}
STANDARD_STRATEGIES = {
    Benchmarks.HOTPOTQA: StandardHotQAStrategy,
    Benchmarks.FEVER: StandardFEVERStrategy,
    Benchmarks.TRIVIAQA: StandardTriviaQAStrategy,
    Benchmarks.AMBIGNQ: StandardAmbigNQStrategy,
    Benchmarks.GSM8K: StandardGSM8KStrategy,
    Benchmarks.SVAMP: StandardSVAMPStrategy,
    Benchmarks.TABMWP: StandardTabMWPStrategy,
    Benchmarks.HUMANEVAL: StandardHEvalStrategy,
    Benchmarks.MBPP: StandardMBPPStrategy,
}


class Standard(BasePrompting):
    """Standard prompting method.

    Attributes:
        llm (BaseLLM): An instance of a language model used for generating initial answers
            and critiques.
        benchmark (str): The benchmark.
        testing (bool): Whether to run in testing mode. Defaults to False.
        **strategy_kwargs (Any): Additional strategy-specific arguments.
    """

    def __init__(
        self,
        llm: BaseLLM,
        benchmark: str,
        testing: bool = False,
        **strategy_kwargs: Any,
    ) -> None:
        """Initialization."""
        super().__init__(llm=llm, benchmark=benchmark, testing=testing)

        self.strategy = Standard.get_strategy(
            benchmark=self.benchmark, llm=self.llm, testing=testing, **strategy_kwargs
        )

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
        if benchmark not in STANDARD_FEWSHOTS:
            raise ValueError(
                f"Benchmark '{benchmark}' few-shots not found for Standard."
            )

        if fewshot_type not in STANDARD_BENCHMARK_FEWSHOTS[benchmark]:
            raise ValueError(
                f"Benchmark '{benchmark}' few-shot type not supported for Standard."
            )

        benchmark_fewshots = BENCHMARK_FEWSHOTS[benchmark][fewshot_type]

        return {"examples": benchmark_fewshots}

    @staticmethod
    def get_prompts(benchmark: str, **kwargs: Any) -> Dict[str, str]:
        """Retrieve the prompt instruction based on the benchmark.

        Args:
            benchmark (str): The benchmark name.
            **kwargs (Any): Additional arguments.

        Returns:
            Dict[str, str]: A dictionary of prompt instructions.
        """
        if benchmark not in STANDARD_PROMPTS:
            raise ValueError(f"Benchmark '{benchmark}' prompt not found for Standard.")

        return STANDARD_PROMPTS[benchmark]

    @staticmethod
    def get_strategy(benchmark: str, **kwargs: Any) -> StandardBaseStrategy:
        """Returns an instance of the appropriate Standard strategy based on the provided benchmark.

        Args:
            benchmark (str): The benchmark name.
            **kwargs (Any): Additional keyword arguments to pass to
                the strategy's constructor.

        Returns:
            StandardBaseStrategy: An instance of the appropriate Standard strategy.
        """
        if benchmark not in STANDARD_STRATEGIES:
            raise ValueError(f"Unsupported benchmark: {benchmark} for Standard")

        strategy = STANDARD_STRATEGIES[benchmark]
        return strategy(**kwargs)

    def generate(
        self,
        question: str,
        examples: str = "",
        prompt: str = "",
        additional_keys: Dict[str, str] = {},
        fewshot_type: str = "",
        num_retries: int = 1,
        warming: List[Optional[float]] = [None],
    ) -> StandardOutput:
        """Generates an answer and critique for the given question using the provided examples and prompts.

        Args:
            question (str): The question to be answered.
            examples (str): Few-shot examples to guide the language model in generating the answer. Defaults to "".
            prompt (str): The instruction template used to prompt the language model for the answer. Defaults to "".
            additional_keys (Dict[str, str]): Additional keys to format the answer prompt. Defaults to {}.
            fewshot_type (str): The type of few-shot examples to use. Defaults to "".
            num_retries (int): Number of retries. Defaults to 1.
            warming (List[Optional[float]]): List of warmup temperatures. Defaults to [None].

        Returns:
            StandardOutput: The output of the Standard strategy.
        """
        if not prompt or not examples:
            if not fewshot_type:
                fewshot_type = STANDARD_BENCHMARK_FEWSHOTS[self.benchmark][0]
            fewshots = Standard.get_fewshots(
                benchmark=self.benchmark, fewshot_type=fewshot_type
            )
            prompts = Standard.get_prompts(benchmark=self.benchmark)
            examples = fewshots["examples"]
            prompt = prompts["prompt"]

        if not warming:
            warming = [None]

        out = self.strategy.generate(
            question=question,
            examples=examples,
            prompt=prompt,
            additional_keys=additional_keys,
            num_retries=num_retries,
            warming=warming,
        )

        return out
