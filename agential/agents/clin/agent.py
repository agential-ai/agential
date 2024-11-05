"""CLIN Agent.

Paper: https://arxiv.org/pdf/2310.10134
GitHub Repo: https://github.com/allenai/clin
"""

from typing import Any, Dict, Optional

from agential.agents.base.agent import BaseAgent
from agential.agents.clin.memory import CLINMemory
from agential.agents.clin.output import CLINOutput
from agential.agents.clin.prompts import (
    CLIN_ADAPT_META_SUMMARY_SYSTEM,
    CLIN_ADAPT_SUMMARY_SYSTEM,
    CLIN_GEN_ENV_META_SUMMARY_SYSTEM,
    CLIN_GEN_ENV_SUMMARY_SYSTEM,
    CLIN_GEN_TASK_META_SUMMARY_SYSTEM,
    CLIN_GEN_TASK_SUMMARY_SYSTEM,
    CLIN_INSTRUCTION_AMBIGNQ,
    CLIN_INSTRUCTION_FEVER,
    CLIN_INSTRUCTION_GSM8K,
    CLIN_INSTRUCTION_HOTPOTQA,
    CLIN_INSTRUCTION_HUMANEVAL,
    CLIN_INSTRUCTION_MBPP,
    CLIN_INSTRUCTION_SVAMP,
    CLIN_INSTRUCTION_TABMWP,
    CLIN_INSTRUCTION_TRIVIAQA,
    CLIN_META_SUMMARY_INSTRUCTION_AMBIGNQ,
    CLIN_META_SUMMARY_INSTRUCTION_FEVER,
    CLIN_META_SUMMARY_INSTRUCTION_GSM8K,
    CLIN_META_SUMMARY_INSTRUCTION_HOTPOTQA,
    CLIN_META_SUMMARY_INSTRUCTION_HUMANEVAL,
    CLIN_META_SUMMARY_INSTRUCTION_MBPP,
    CLIN_META_SUMMARY_INSTRUCTION_SVAMP,
    CLIN_META_SUMMARY_INSTRUCTION_TABMWP,
    CLIN_META_SUMMARY_INSTRUCTION_TRIVIAQA,
    CLIN_SUMMARY_INSTRUCTION_AMBIGNQ,
    CLIN_SUMMARY_INSTRUCTION_FEVER,
    CLIN_SUMMARY_INSTRUCTION_GSM8K,
    CLIN_SUMMARY_INSTRUCTION_HOTPOTQA,
    CLIN_SUMMARY_INSTRUCTION_HUMANEVAL,
    CLIN_SUMMARY_INSTRUCTION_MBPP,
    CLIN_SUMMARY_INSTRUCTION_SVAMP,
    CLIN_SUMMARY_INSTRUCTION_TABMWP,
    CLIN_SUMMARY_INSTRUCTION_TRIVIAQA,
)
from agential.agents.clin.strategies.base import CLINBaseStrategy
from agential.agents.clin.strategies.code import (
    CLINHumanEvalStrategy,
    CLINMBPPStrategy,
)
from agential.agents.clin.strategies.math import (
    CLINGSM8KStrategy,
    CLINSVAMPStrategy,
    CLINTabMWPStrategy,
)
from agential.agents.clin.strategies.qa import (
    CLINAmbigNQStrategy,
    CLINFEVERStrategy,
    CLINHotQAStrategy,
    CLINTriviaQAStrategy,
)
from agential.constants import BENCHMARK_FEWSHOTS, Benchmarks, FewShotType
from agential.core.llm import BaseLLM

CLIN_BENCHMARK_FEWSHOTS = {
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


CLIN_PROMPTS = {
    Benchmarks.HOTPOTQA: {
        "prompt": CLIN_INSTRUCTION_HOTPOTQA,
        "summary_prompt": CLIN_SUMMARY_INSTRUCTION_HOTPOTQA,
        "meta_summary_prompt": CLIN_META_SUMMARY_INSTRUCTION_HOTPOTQA,
    },
    Benchmarks.FEVER: {
        "prompt": CLIN_INSTRUCTION_FEVER,
        "summary_prompt": CLIN_SUMMARY_INSTRUCTION_FEVER,
        "meta_summary_prompt": CLIN_META_SUMMARY_INSTRUCTION_FEVER,
    },
    Benchmarks.TRIVIAQA: {
        "prompt": CLIN_INSTRUCTION_TRIVIAQA,
        "summary_prompt": CLIN_SUMMARY_INSTRUCTION_TRIVIAQA,
        "meta_summary_prompt": CLIN_META_SUMMARY_INSTRUCTION_TRIVIAQA,
    },
    Benchmarks.AMBIGNQ: {
        "prompt": CLIN_INSTRUCTION_AMBIGNQ,
        "summary_prompt": CLIN_SUMMARY_INSTRUCTION_AMBIGNQ,
        "meta_summary_prompt": CLIN_META_SUMMARY_INSTRUCTION_AMBIGNQ,
    },
    Benchmarks.GSM8K: {
        "prompt": CLIN_INSTRUCTION_GSM8K,
        "summary_prompt": CLIN_SUMMARY_INSTRUCTION_GSM8K,
        "meta_summary_prompt": CLIN_META_SUMMARY_INSTRUCTION_GSM8K,
    },
    Benchmarks.SVAMP: {
        "prompt": CLIN_INSTRUCTION_SVAMP,
        "summary_prompt": CLIN_SUMMARY_INSTRUCTION_SVAMP,
        "meta_summary_prompt": CLIN_META_SUMMARY_INSTRUCTION_SVAMP,
    },
    Benchmarks.TABMWP: {
        "prompt": CLIN_INSTRUCTION_TABMWP,
        "summary_prompt": CLIN_SUMMARY_INSTRUCTION_TABMWP,
        "meta_summary_prompt": CLIN_META_SUMMARY_INSTRUCTION_TABMWP,
    },
    Benchmarks.HUMANEVAL: {
        "prompt": CLIN_INSTRUCTION_HUMANEVAL,
        "summary_prompt": CLIN_SUMMARY_INSTRUCTION_HUMANEVAL,
        "meta_summary_prompt": CLIN_META_SUMMARY_INSTRUCTION_HUMANEVAL,
    },
    Benchmarks.MBPP: {
        "prompt": CLIN_INSTRUCTION_MBPP,
        "summary_prompt": CLIN_SUMMARY_INSTRUCTION_MBPP,
        "meta_summary_prompt": CLIN_META_SUMMARY_INSTRUCTION_MBPP,
    },
}


CLIN_SUMMARY_SYSTEM = {
    "adapt": CLIN_ADAPT_SUMMARY_SYSTEM,
    "gen_env": CLIN_GEN_ENV_SUMMARY_SYSTEM,
    "gen_task": CLIN_GEN_TASK_SUMMARY_SYSTEM,
}


CLIN_META_SUMMARY_SYSTEM = {
    "adapt": CLIN_ADAPT_META_SUMMARY_SYSTEM,
    "gen_env": CLIN_GEN_ENV_META_SUMMARY_SYSTEM,
    "gen_task": CLIN_GEN_TASK_META_SUMMARY_SYSTEM,
}


CLIN_FEWSHOTS = {
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


CLIN_STRATEGIES = {
    Benchmarks.HOTPOTQA: CLINHotQAStrategy,
    Benchmarks.FEVER: CLINFEVERStrategy,
    Benchmarks.TRIVIAQA: CLINTriviaQAStrategy,
    Benchmarks.AMBIGNQ: CLINAmbigNQStrategy,
    Benchmarks.GSM8K: CLINGSM8KStrategy,
    Benchmarks.SVAMP: CLINSVAMPStrategy,
    Benchmarks.TABMWP: CLINTabMWPStrategy,
    Benchmarks.HUMANEVAL: CLINHumanEvalStrategy,
    Benchmarks.MBPP: CLINMBPPStrategy,
}


class CLIN(BaseAgent):
    """CLIN agent.

    Attributes:
        llm (BaseLLM): The language model used to generate responses.
        benchmark (str): The benchmark.
        memory (CLINMemory): The memory used to store and retrieve information. Defaults to None.
        testing (bool): Whether the agent is in testing mode. Defaults to False.
    """

    def __init__(
        self,
        llm: BaseLLM,
        benchmark: str,
        memory: Optional[CLINMemory] = None,
        testing: bool = False,
        **strategy_kwargs: Any,
    ) -> None:
        """Initialization."""
        super().__init__(llm=llm, benchmark=benchmark, testing=testing)

        self.strategy = CLIN.get_strategy(
            benchmark=self.benchmark,
            llm=self.llm,
            memory=memory,
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
        if fewshot_type not in CLIN_BENCHMARK_FEWSHOTS[benchmark]:
            raise ValueError(
                f"Benchmark '{benchmark}' few-shot type not supported for CLIN."
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
            Dict[str, str]: The prompt instructions.
        """
        if benchmark not in CLIN_PROMPTS:
            raise ValueError(f"Benchmark '{benchmark}' prompt not found for CLIN.")

        return CLIN_PROMPTS[benchmark]

    @staticmethod
    def get_strategy(benchmark: str, **kwargs: Any) -> CLINBaseStrategy:
        """Returns an instance of the appropriate CLIN strategy based on the provided benchmark.

        Args:
            benchmark (str): The benchmark name.
            **kwargs (Any): Additional keyword arguments to pass to
                the strategy's constructor.

        Returns:
            CLINBaseStrategy: An instance of the appropriate CLIN strategy.
        """
        if benchmark not in CLIN_STRATEGIES:
            raise ValueError(f"Unsupported benchmark: {benchmark} for agent CLIN")

        strategy = CLIN_STRATEGIES[benchmark]
        return strategy(**kwargs)

    def generate(
        self,
        question: str,
        key: str,
        examples: str = "",
        prompt: str = "",
        summary_prompt: str = "",
        meta_summary_prompt: str = "",
        additional_keys: Dict[str, str] = {},
        summary_additional_keys: Dict[str, str] = {},
        meta_summary_additional_keys: Dict[str, str] = {},
        fewshot_type: str = "",
        summary_system: str = "",
        meta_summary_system: str = "",
        quadrant: str = "adapt",
        patience: int = 3,
        reset: bool = False,
    ) -> CLINOutput:
        """Generate a response to a given question.

        Args:
            question (str): The question to be answered.
            key (str): The key for the question.
            examples (str): The examples for the question. Defaults to "".
            prompt (str): The prompt for the question. Defaults to "".
            summary_prompt (str): The summary prompt for the question. Defaults to "".
            meta_summary_prompt (str): The meta-summary prompt for the question. Defaults to "".
            additional_keys (Dict[str, str]): Additional keys for the question. Defaults to {}.
            summary_additional_keys (Dict[str, str]): Additional keys for the summary. Defaults to {}.
            meta_summary_additional_keys (Dict[str, str]): Additional keys for the meta-summary. Defaults to {}.
            fewshot_type (str): The type of few-shot examples to use. Defaults to "".
            summary_system (str): The system for the summary. Defaults to "".
            meta_summary_system (str): The system for the meta-summary. Defaults to "".
            quadrant (str): The quadrant for the question. Defaults to "adapt".
            patience (int): The patience for the question. Defaults to 3.
            reset (bool): Whether to reset the agent. Defaults to False.

        Returns:
                CLINOutput: The output of the agent.
        """
        if quadrant not in ["adapt", "gen_env", "gen_task"]:
            raise ValueError(f"Quadrant '{quadrant}' not supported for CLIN.")

        if not prompt or not summary_prompt or not meta_summary_prompt or not examples:
            if not fewshot_type:
                fewshot_type = CLIN_BENCHMARK_FEWSHOTS[self.benchmark][0]  # type: ignore
            fewshots = CLIN.get_fewshots(
                benchmark=self.benchmark, fewshot_type=fewshot_type
            )
            prompts = CLIN.get_prompts(benchmark=self.benchmark)
            examples = fewshots["examples"]
            prompt = prompts["prompt"]
            summary_prompt = prompts["summary_prompt"]
            meta_summary_prompt = prompts["meta_summary_prompt"]

        if not summary_system:
            summary_system = CLIN_SUMMARY_SYSTEM[quadrant]

        if not meta_summary_system:
            meta_summary_system = CLIN_META_SUMMARY_SYSTEM[quadrant]

        out = self.strategy.generate(
            question=question,
            key=key,
            examples=examples,
            prompt=prompt,
            summary_prompt=summary_prompt,
            meta_summary_prompt=meta_summary_prompt,
            additional_keys=additional_keys,
            summary_additional_keys=summary_additional_keys,
            meta_summary_additional_keys=meta_summary_additional_keys,
            summary_system=summary_system,
            meta_summary_system=meta_summary_system,
            quadrant=quadrant,
            patience=patience,
            reset=reset,
        )

        return out
