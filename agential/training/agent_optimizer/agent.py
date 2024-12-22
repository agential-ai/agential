"""ReAct Agent.

Original Paper: https://arxiv.org/abs/2210.03629
Paper Repository: https://github.com/ysymyth/ReAct
"""

from typing import Any, Dict, Optional

from agential.training.agent_optimizer.agent import BaseAgent
from agential.training.agent_optimizer.output import PromptOptimizerOutput
from agential.training.agent_optimizer.prompts import (
    PROMPT_OPTIMIZER_INSTRUCTION_AMBIGNQ,
    PROMPT_OPTIMIZER_INSTRUCTION_FEVER,
    PROMPT_OPTIMIZER_INSTRUCTION_GSM8K,
    PROMPT_OPTIMIZER_INSTRUCTION_HOTPOTQA,
    PROMPT_OPTIMIZER_INSTRUCTION_HUMANEVAL,
    PROMPT_OPTIMIZER_INSTRUCTION_MBPP,
    PROMPT_OPTIMIZER_INSTRUCTION_SVAMP,
    PROMPT_OPTIMIZER_INSTRUCTION_TABMWP,
    PROMPT_OPTIMIZER_INSTRUCTION_TRIVIAQA,
)
from agential.training.agent_optimizer.strategies.base import PromptOptimizerBaseStrategy, ReActBaseStrategy
from agential.training.agent_optimizer.strategies.code import PromptOptimizerHEvalStrategy, PromptOptimizerMBPPStrategy
from agential.training.agent_optimizer.strategies.math import (
    PromptOptimizerGSM8KStrategy,
    PromptOptimizerSVAMPStrategy,
    PromptOptimizerTabMWPStrategy,
)
from agential.training.agent_optimizer.strategies.qa import (
    PromptOptimizerAmbigNQStrategy,
    PromptOptimizerFEVERStrategy,
    PromptOptimizerHotQAStrategy,
    PromptOptimizerTriviaQAStrategy,
)
from agential.constants import BENCHMARK_FEWSHOTS, Benchmarks, FewShotType
from agential.core.llm import BaseLLM

PROMPT_OPTIMIZER_BENCHMARK_FEWSHOTS = {
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

REACT_PROMPTS = {
    Benchmarks.HOTPOTQA: {
        "prompt": PROMPT_OPTIMIZER_INSTRUCTION_HOTPOTQA,
    },
    Benchmarks.FEVER: {
        "prompt": PROMPT_OPTIMIZER_INSTRUCTION_FEVER,
    },
    Benchmarks.TRIVIAQA: {
        "prompt": PROMPT_OPTIMIZER_INSTRUCTION_TRIVIAQA,
    },
    Benchmarks.AMBIGNQ: {
        "prompt": PROMPT_OPTIMIZER_INSTRUCTION_AMBIGNQ,
    },
    Benchmarks.GSM8K: {
        "prompt": PROMPT_OPTIMIZER_INSTRUCTION_GSM8K,
    },
    Benchmarks.SVAMP: {
        "prompt": PROMPT_OPTIMIZER_INSTRUCTION_SVAMP,
    },
    Benchmarks.TABMWP: {
        "prompt": PROMPT_OPTIMIZER_INSTRUCTION_TABMWP,
    },
    Benchmarks.HUMANEVAL: {
        "prompt": PROMPT_OPTIMIZER_INSTRUCTION_HUMANEVAL,
    },
    Benchmarks.MBPP: {
        "prompt": PROMPT_OPTIMIZER_INSTRUCTION_MBPP,
    },
}
REACT_FEWSHOTS: Dict[str, Dict] = {
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
REACT_STRATEGIES = {
    Benchmarks.HOTPOTQA: PromptOptimizerHotQAStrategy,
    Benchmarks.FEVER: PromptOptimizerFEVERStrategy,
    Benchmarks.TRIVIAQA: PromptOptimizerTriviaQAStrategy,
    Benchmarks.AMBIGNQ: PromptOptimizerAmbigNQStrategy,
    Benchmarks.GSM8K: PromptOptimizerGSM8KStrategy,
    Benchmarks.SVAMP: PromptOptimizerSVAMPStrategy,
    Benchmarks.TABMWP: PromptOptimizerTabMWPStrategy,
    Benchmarks.HUMANEVAL: PromptOptimizerHEvalStrategy,
    Benchmarks.MBPP: PromptOptimizerMBPPStrategy,
}


class PromptOptimizer(BaseAgent):
    """PromptOptimizer class for optimizing the agent's performance."""

    def __init__(
        self,
        llm: BaseLLM,
        benchmark: str,
        testing: bool = False,
        max_steps: int = 3,
        optimizer_model: Optional[str] = "gpt-4-1106-preview",
        **strategy_kwargs: Any,
    ) -> None:
        """Initialization."""
        super().__init__(llm=llm, benchmark=benchmark, testing=testing)


class PromptOptimizer(BaseAgent):
    """PromptOptimizer agent.

    Attributes:
        llm (BaseLLM): An instance of a language model used for generating initial answers
            and critiques.
        benchmark (str): The benchmark.
        testing (bool, optional): Whether to run in testing mode. Defaults to False.
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

        self.strategy = PromptOptimizer.get_strategy(
            benchmark=self.benchmark,
            llm=self.llm,
            testing=self.testing,
            **strategy_kwargs,
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
        if benchmark not in REACT_FEWSHOTS:
            raise ValueError(f"Benchmark '{benchmark}' few-shots not found for ReAct.")

        if fewshot_type not in PROMPT_OPTIMIZER_BENCHMARK_FEWSHOTS[benchmark]:
            raise ValueError(
                f"Benchmark '{benchmark}' few-shot type not supported for ReAct."
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
        if benchmark not in REACT_PROMPTS:
            raise ValueError(f"Benchmark '{benchmark}' prompt not found for ReAct.")

        return REACT_PROMPTS[benchmark]

    @staticmethod
    def get_strategy(benchmark: str, **kwargs: Any) -> PromptOptimizerBaseStrategy:
        """Returns an instance of the appropriate ReAct strategy based on the provided benchmark.

        Args:
            benchmark (str): The benchmark name.
            **kwargs (Any): Additional keyword arguments to pass to
                the strategy's constructor.

        Returns:
            ReActBaseStrategy: An instance of the appropriate ReAct strategy.
        """
        if benchmark not in REACT_STRATEGIES:
            raise ValueError(f"Unsupported benchmark: {benchmark} for agent ReAct")

        strategy = REACT_STRATEGIES[benchmark]
        return strategy(**kwargs)

    def generate(
        self,
        question: str,
        examples: str = "",
        prompt: str = "",
        additional_keys: Dict[str, str] = {},
        fewshot_type: str = "",
        reset: bool = True,
    ) -> PromptOptimizerOutput:
        """Processes a given question through ReAct.

        Iteratively applies the think-act-observe cycle to generate an answer for the question.
        The process continues until the operation is halted based on certain conditions.

        Args:
            question (str): The question to be processed.
            examples (str, optional): Fewshot examples. Defaults to "".
            prompt (str, optional): Prompt template string. Defaults to "".
            additional_keys (Dict[str, str]): Additional keys to format the prompt. Defaults to {}.
            fewshot_type (str): The type of few-shot examples to use. Defaults to "".
            reset (bool, optional): Whether to reset the internal state before processing. Defaults to True.

        Returns:
            ReActOutput: The list of accumulated output from the ReAct process,
                each ReActOutput consists of a thought, action type/query, observation, answer, and external tool info.
        """
        if not prompt or not examples:
            if not fewshot_type:
                fewshot_type = PROMPT_OPTIMIZER_BENCHMARK_FEWSHOTS[self.benchmark][0]
            fewshots = PromptOptimizer.get_fewshots(
                benchmark=self.benchmark, fewshot_type=fewshot_type
            )
            prompts = PromptOptimizer.get_prompts(benchmark=self.benchmark)
            examples = fewshots["examples"]
            prompt = prompts["prompt"]

        out = self.strategy.generate(
            question=question,
            examples=examples,
            prompt=prompt,
            additional_keys=additional_keys,
            reset=reset,
        )

        return out
