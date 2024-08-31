"""CoT prompting method."""

from typing import Any, Dict

from agential.constants import BENCHMARK_FEWSHOTS, Benchmarks, FewShotType
from agential.core.base.prompting.prompting import BasePrompting
from agential.llm.llm import BaseLLM
from agential.prompting.cot.output import CoTOutput
from agential.prompting.cot.prompts import (
    COT_INSTRUCTION_AMBIGNQ,
    COT_INSTRUCTION_FEVER,
    COT_INSTRUCTION_GSM8K,
    COT_INSTRUCTION_HOTPOTQA,
    COT_INSTRUCTION_HUMANEVAL,
    COT_INSTRUCTION_MBPP,
    COT_INSTRUCTION_SVAMP,
    COT_INSTRUCTION_TABMWP,
    COT_INSTRUCTION_TRIVIAQA,
)
from agential.prompting.cot.strategies.base import CoTBaseStrategy
from agential.prompting.cot.strategies.code import CoTHEvalStrategy, CoTMBPPStrategy
from agential.prompting.cot.strategies.math import (
    CoTGSM8KStrategy,
    CoTSVAMPStrategy,
    CoTTabMWPStrategy,
)
from agential.prompting.cot.strategies.qa import (
    CoTAmbigNQStrategy,
    CoTFEVERStrategy,
    CoTHotQAStrategy,
    CoTTriviaQAStrategy,
)

COT_BENCHMARK_FEWSHOTS = {
    Benchmarks.HOTPOTQA: [FewShotType.COT],
    Benchmarks.FEVER: [FewShotType.COT],
    Benchmarks.TRIVIAQA: [FewShotType.COT],
    Benchmarks.AMBIGNQ: [FewShotType.COT],
    Benchmarks.GSM8K: [FewShotType.COT],
    Benchmarks.SVAMP: [FewShotType.COT],
    Benchmarks.TABMWP: [FewShotType.COT],
    Benchmarks.HUMANEVAL: [FewShotType.COT],
    Benchmarks.MBPP: [FewShotType.COT],
}

COT_PROMPTS = {
    Benchmarks.HOTPOTQA: {
        "prompt": COT_INSTRUCTION_HOTPOTQA,
    },
    Benchmarks.FEVER: {
        "prompt": COT_INSTRUCTION_FEVER,
    },
    Benchmarks.TRIVIAQA: {
        "prompt": COT_INSTRUCTION_TRIVIAQA,
    },
    Benchmarks.AMBIGNQ: {
        "prompt": COT_INSTRUCTION_AMBIGNQ,
    },
    Benchmarks.GSM8K: {
        "prompt": COT_INSTRUCTION_GSM8K,
    },
    Benchmarks.SVAMP: {
        "prompt": COT_INSTRUCTION_SVAMP,
    },
    Benchmarks.TABMWP: {
        "prompt": COT_INSTRUCTION_TABMWP,
    },
    Benchmarks.HUMANEVAL: {
        "prompt": COT_INSTRUCTION_HUMANEVAL,
    },
    Benchmarks.MBPP: {
        "prompt": COT_INSTRUCTION_MBPP,
    },
}
COT_FEWSHOTS: Dict[str, Dict] = {
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
COT_STRATEGIES = {
    Benchmarks.HOTPOTQA: CoTHotQAStrategy,
    Benchmarks.FEVER: CoTFEVERStrategy,
    Benchmarks.TRIVIAQA: CoTTriviaQAStrategy,
    Benchmarks.AMBIGNQ: CoTAmbigNQStrategy,
    Benchmarks.GSM8K: CoTGSM8KStrategy,
    Benchmarks.SVAMP: CoTSVAMPStrategy,
    Benchmarks.TABMWP: CoTTabMWPStrategy,
    Benchmarks.HUMANEVAL: CoTHEvalStrategy,
    Benchmarks.MBPP: CoTMBPPStrategy,
}


class CoT(BasePrompting):
    """CoT prompting method.

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

        self.strategy = CoT.get_strategy(
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
        if benchmark not in COT_FEWSHOTS:
            raise ValueError(f"Benchmark '{benchmark}' few-shots not found for CoT.")

        if fewshot_type not in COT_BENCHMARK_FEWSHOTS[benchmark]:
            raise ValueError(
                f"Benchmark '{benchmark}' few-shot type not supported for CoT."
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
        if benchmark not in COT_PROMPTS:
            raise ValueError(f"Benchmark '{benchmark}' prompt not found for CoT.")

        return COT_PROMPTS[benchmark]

    @staticmethod
    def get_strategy(benchmark: str, **kwargs: Any) -> CoTBaseStrategy:
        """Returns an instance of the appropriate CoT strategy based on the provided benchmark.

        Args:
            benchmark (str): The benchmark name.
            **kwargs (Any): Additional keyword arguments to pass to
                the strategy's constructor.

        Returns:
            CoTBaseStrategy: An instance of the appropriate CoT strategy.
        """
        if benchmark not in COT_STRATEGIES:
            raise ValueError(f"Unsupported benchmark: {benchmark} for CoT")

        strategy = COT_STRATEGIES[benchmark]
        return strategy(**kwargs)

    def generate(
        self,
        question: str,
        examples: str,
        prompt: str,
        additional_keys: Dict[str, str],
        fewshot_type: str = "",
        num_retries: int = 1,
    ) -> CoTOutput:
        """Generates an answer and critique for the given question using the provided examples and prompts.

        Args:
            question (str): The question to be answered.
            examples (str): Few-shot examples to guide the language model in generating the answer.
            prompt (str): The instruction template used to prompt the language model for the answer.
            additional_keys (Dict[str, str]): Additional keys to format the answer prompt.
            fewshot_type (str): The type of few-shot examples to use. Defaults to "".
            num_retries (int): Number of retries. Defaults to 1.

        Returns:
            CoTOutput: The output of the CoT strategy.
        """
        if not prompt or not examples:
            if not fewshot_type:
                fewshot_type = COT_BENCHMARK_FEWSHOTS[self.benchmark][0]
            fewshots = CoT.get_fewshots(
                benchmark=self.benchmark, fewshot_type=fewshot_type
            )
            prompts = CoT.get_prompts(benchmark=self.benchmark)
            examples = fewshots["examples"]
            prompt = prompts["prompt"]

        out = self.strategy.generate(
            question=question,
            examples=examples,
            prompt=prompt,
            additional_keys=additional_keys,
            num_retries=num_retries,
        )

        return out
