"""LATS agent.

Original Paper: https://arxiv.org/pdf/2310.04406
Paper Repository: https://github.com/lapisrocks/LanguageAgentTreeSearch
"""

from typing import Any, Dict, List, Tuple

from agential.constants import BENCHMARK_FEWSHOTS, Benchmarks, FewShotType
from agential.agents.lats.node import Node
from agential.agents.lats.output import LATSOutput
from agential.agents.lats.prompts import (
    AMBIGNQ_FEWSHOT_EXAMPLES_LATS_REFLECT,
    AMBIGNQ_FEWSHOT_EXAMPLES_LATS_VALUE,
    FEVER_FEWSHOT_EXAMPLES_LATS_REFLECT,
    FEVER_FEWSHOT_EXAMPLES_LATS_VALUE,
    GSM8K_FEWSHOT_EXAMPLES_LATS_REFLECT,
    GSM8K_FEWSHOT_EXAMPLES_LATS_VALUE,
    HOTPOTQA_FEWSHOT_EXAMPLES_LATS_REFLECT,
    HOTPOTQA_FEWSHOT_EXAMPLES_LATS_VALUE,
    HUMANEVAL_FEWSHOT_EXAMPLES_LATS_REFLECT,
    HUMANEVAL_FEWSHOT_EXAMPLES_LATS_VALUE,
    LATS_INSTRUCTION_AMBIGNQ,
    LATS_INSTRUCTION_FEVER,
    LATS_INSTRUCTION_GSM8K,
    LATS_INSTRUCTION_HOTPOTQA,
    LATS_INSTRUCTION_HUMANEVAL,
    LATS_INSTRUCTION_MBPP,
    LATS_INSTRUCTION_SVAMP,
    LATS_INSTRUCTION_TABMWP,
    LATS_INSTRUCTION_TRIVIAQA,
    LATS_REFLECT_INSTRUCTION_AMBIGNQ,
    LATS_REFLECT_INSTRUCTION_FEVER,
    LATS_REFLECT_INSTRUCTION_GSM8K,
    LATS_REFLECT_INSTRUCTION_HOTPOTQA,
    LATS_REFLECT_INSTRUCTION_HUMANEVAL,
    LATS_REFLECT_INSTRUCTION_MBPP,
    LATS_REFLECT_INSTRUCTION_SVAMP,
    LATS_REFLECT_INSTRUCTION_TABMWP,
    LATS_REFLECT_INSTRUCTION_TRIVIAQA,
    LATS_VALUE_INSTRUCTION_AMBIGNQ,
    LATS_VALUE_INSTRUCTION_FEVER,
    LATS_VALUE_INSTRUCTION_GSM8K,
    LATS_VALUE_INSTRUCTION_HOTPOTQA,
    LATS_VALUE_INSTRUCTION_HUMANEVAL,
    LATS_VALUE_INSTRUCTION_MBPP,
    LATS_VALUE_INSTRUCTION_SVAMP,
    LATS_VALUE_INSTRUCTION_TABMWP,
    LATS_VALUE_INSTRUCTION_TRIVIAQA,
    MBPP_FEWSHOT_EXAMPLES_LATS_REFLECT,
    MBPP_FEWSHOT_EXAMPLES_LATS_VALUE,
    SVAMP_FEWSHOT_EXAMPLES_LATS_REFLECT,
    SVAMP_FEWSHOT_EXAMPLES_LATS_VALUE,
    TABMWP_FEWSHOT_EXAMPLES_LATS_REFLECT,
    TABMWP_FEWSHOT_EXAMPLES_LATS_VALUE,
    TRIVIAQA_FEWSHOT_EXAMPLES_LATS_REFLECT,
    TRIVIAQA_FEWSHOT_EXAMPLES_LATS_VALUE,
)
from agential.agents.lats.strategies.base import LATSBaseStrategy
from agential.agents.lats.strategies.code import (
    LATSHEvalStrategy,
    LATSMBPPStrategy,
)
from agential.agents.lats.strategies.math import (
    LATSGSM8KStrategy,
    LATSSVAMPStrategy,
    LATSTabMWPStrategy,
)
from agential.agents.lats.strategies.qa import (
    LATSAmbigNQStrategy,
    LATSFEVERStrategy,
    LATSHotQAStrategy,
    LATSTriviaQAStrategy,
)
from agential.core.base.agents.agent import BaseAgent
from agential.llm.llm import BaseLLM

LATS_BENCHMARK_FEWSHOTS = {
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

LATS_PROMPTS = {
    Benchmarks.HOTPOTQA: {
        "prompt": LATS_INSTRUCTION_HOTPOTQA,
        "reflect_prompt": LATS_REFLECT_INSTRUCTION_HOTPOTQA,
        "value_prompt": LATS_VALUE_INSTRUCTION_HOTPOTQA,
    },
    Benchmarks.FEVER: {
        "prompt": LATS_INSTRUCTION_FEVER,
        "reflect_prompt": LATS_REFLECT_INSTRUCTION_FEVER,
        "value_prompt": LATS_VALUE_INSTRUCTION_FEVER,
    },
    Benchmarks.TRIVIAQA: {
        "prompt": LATS_INSTRUCTION_TRIVIAQA,
        "reflect_prompt": LATS_REFLECT_INSTRUCTION_TRIVIAQA,
        "value_prompt": LATS_VALUE_INSTRUCTION_TRIVIAQA,
    },
    Benchmarks.AMBIGNQ: {
        "prompt": LATS_INSTRUCTION_AMBIGNQ,
        "reflect_prompt": LATS_REFLECT_INSTRUCTION_AMBIGNQ,
        "value_prompt": LATS_VALUE_INSTRUCTION_AMBIGNQ,
    },
    Benchmarks.GSM8K: {
        "prompt": LATS_INSTRUCTION_GSM8K,
        "reflect_prompt": LATS_REFLECT_INSTRUCTION_GSM8K,
        "value_prompt": LATS_VALUE_INSTRUCTION_GSM8K,
    },
    Benchmarks.SVAMP: {
        "prompt": LATS_INSTRUCTION_SVAMP,
        "reflect_prompt": LATS_REFLECT_INSTRUCTION_SVAMP,
        "value_prompt": LATS_VALUE_INSTRUCTION_SVAMP,
    },
    Benchmarks.TABMWP: {
        "prompt": LATS_INSTRUCTION_TABMWP,
        "reflect_prompt": LATS_REFLECT_INSTRUCTION_TABMWP,
        "value_prompt": LATS_VALUE_INSTRUCTION_TABMWP,
    },
    Benchmarks.HUMANEVAL: {
        "prompt": LATS_INSTRUCTION_HUMANEVAL,
        "reflect_prompt": LATS_REFLECT_INSTRUCTION_HUMANEVAL,
        "value_prompt": LATS_VALUE_INSTRUCTION_HUMANEVAL,
    },
    Benchmarks.MBPP: {
        "prompt": LATS_INSTRUCTION_MBPP,
        "reflect_prompt": LATS_REFLECT_INSTRUCTION_MBPP,
        "value_prompt": LATS_VALUE_INSTRUCTION_MBPP,
    },
}

LATS_FEWSHOTS = {
    Benchmarks.HOTPOTQA: {
        "reflect_examples": HOTPOTQA_FEWSHOT_EXAMPLES_LATS_REFLECT,
        "value_examples": HOTPOTQA_FEWSHOT_EXAMPLES_LATS_VALUE,
    },
    Benchmarks.FEVER: {
        "reflect_examples": FEVER_FEWSHOT_EXAMPLES_LATS_REFLECT,
        "value_examples": FEVER_FEWSHOT_EXAMPLES_LATS_VALUE,
    },
    Benchmarks.TRIVIAQA: {
        "reflect_examples": TRIVIAQA_FEWSHOT_EXAMPLES_LATS_REFLECT,
        "value_examples": TRIVIAQA_FEWSHOT_EXAMPLES_LATS_VALUE,
    },
    Benchmarks.AMBIGNQ: {
        "reflect_examples": AMBIGNQ_FEWSHOT_EXAMPLES_LATS_REFLECT,
        "value_examples": AMBIGNQ_FEWSHOT_EXAMPLES_LATS_VALUE,
    },
    Benchmarks.GSM8K: {
        "reflect_examples": GSM8K_FEWSHOT_EXAMPLES_LATS_REFLECT,
        "value_examples": GSM8K_FEWSHOT_EXAMPLES_LATS_VALUE,
    },
    Benchmarks.SVAMP: {
        "reflect_examples": SVAMP_FEWSHOT_EXAMPLES_LATS_REFLECT,
        "value_examples": SVAMP_FEWSHOT_EXAMPLES_LATS_VALUE,
    },
    Benchmarks.TABMWP: {
        "reflect_examples": TABMWP_FEWSHOT_EXAMPLES_LATS_REFLECT,
        "value_examples": TABMWP_FEWSHOT_EXAMPLES_LATS_VALUE,
    },
    Benchmarks.HUMANEVAL: {
        "reflect_examples": HUMANEVAL_FEWSHOT_EXAMPLES_LATS_REFLECT,
        "value_examples": HUMANEVAL_FEWSHOT_EXAMPLES_LATS_VALUE,
    },
    Benchmarks.MBPP: {
        "reflect_examples": MBPP_FEWSHOT_EXAMPLES_LATS_REFLECT,
        "value_examples": MBPP_FEWSHOT_EXAMPLES_LATS_VALUE,
    },
}

LATS_STRATEGIES = {
    Benchmarks.HOTPOTQA: LATSHotQAStrategy,
    Benchmarks.FEVER: LATSFEVERStrategy,
    Benchmarks.TRIVIAQA: LATSTriviaQAStrategy,
    Benchmarks.AMBIGNQ: LATSAmbigNQStrategy,
    Benchmarks.GSM8K: LATSGSM8KStrategy,
    Benchmarks.SVAMP: LATSSVAMPStrategy,
    Benchmarks.TABMWP: LATSTabMWPStrategy,
    Benchmarks.HUMANEVAL: LATSHEvalStrategy,
    Benchmarks.MBPP: LATSMBPPStrategy,
}


class LATSAgent(BaseAgent):
    """LATS (Language Agent Tree Search) agent.

    Attributes:
        llm (BaseLLM): The language model used by the LATS agent.
        benchmark (str): The benchmark or task the agent is designed to solve.
        testing (bool): A flag indicating whether the agent is in testing mode. Defaults to False.
        **strategy_kwargs (Any): Additional keyword arguments for the strategy.
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

        self.strategy = LATSAgent.get_strategy(
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
        if benchmark not in LATS_FEWSHOTS:
            raise ValueError(f"Benchmark '{benchmark}' few-shots not found for LATS.")

        if fewshot_type not in LATS_BENCHMARK_FEWSHOTS[benchmark]:
            raise ValueError(
                f"Benchmark '{benchmark}' few-shot type not supported for LATS."
            )

        benchmark_fewshots = BENCHMARK_FEWSHOTS[benchmark][fewshot_type]

        return {"examples": benchmark_fewshots, **LATS_FEWSHOTS[benchmark]}

    @staticmethod
    def get_prompts(benchmark: str, **kwargs: Any) -> Dict[str, str]:
        """Retrieve the prompt instruction based on the benchmark.

        Args:
            benchmark (str): The benchmark name.
            **kwargs (Any): Additional arguments.

        Returns:
            Dict[str, str]: A dictionary of prompt instructions.
        """
        if benchmark not in LATS_PROMPTS:
            raise ValueError(f"Benchmark '{benchmark}' prompt not found for LATS.")

        return LATS_PROMPTS[benchmark]

    @staticmethod
    def get_strategy(benchmark: str, **kwargs: Any) -> LATSBaseStrategy:
        """Returns an instance of the appropriate ReAct strategy based on the provided benchmark.

        Args:
            benchmark (str): The benchmark name.
            **kwargs (Any): Additional keyword arguments to pass to
                the strategy's constructor.

        Returns:
            LATSBaseStrategy: An instance of the appropriate ReAct strategy.
        """
        if benchmark not in LATS_STRATEGIES:
            raise ValueError(f"Unsupported benchmark: {benchmark} for agent LATS")

        strategy = LATS_STRATEGIES[benchmark]
        return strategy(**kwargs)  # type: ignore

    def generate(
        self,
        question: str,
        key: str,
        examples: str = "",
        reflect_examples: str = "",
        value_examples: str = "",
        prompt: str = "",
        reflect_prompt: str = "",
        value_prompt: str = "",
        additional_keys: Dict[str, str] = {},
        reflect_additional_keys: Dict[str, str] = {},
        value_additional_keys: Dict[str, str] = {},
        fewshot_type: str = "",
        max_iterations: int = 30,
        reset: bool = True,
    ) -> LATSOutput:
        """Generate an output for the given question.

        Args:
            question (str): The question or task to be solved.
            key (str): The key associated with the question.
            examples (str): Examples to guide the agent's reasoning. Defaults to "".
            reflect_examples (str): Examples to guide the agent's reflections. Defaults to "".
            value_examples (str): Examples to guide the agent's value estimation. Defaults to "".
            prompt (str): The prompt to guide the agent's reasoning. Defaults to "".
            reflect_prompt (str): The prompt to guide the agent's reflections. Defaults to "".
            value_prompt (str): The prompt to guide the agent's value estimation. Defaults to "".
            additional_keys (Dict[str, str]): Additional keys for formatting the prompts. Defaults to {}.
            reflect_additional_keys (Dict[str, str]): Additional keys for formatting the reflection prompts. Defaults to {}.
            value_additional_keys (Dict[str, str]): Additional keys for formatting the value prompts. Defaults to {}.
            fewshot_type (str): The type of few-shot examples to use. Defaults to "".
            max_iterations (int): The maximum number of iterations to run the agent. Defaults to 30.
            reset (bool): Whether to reset the agent before generating the output. Defaults to True.

        Returns:
            LATSOutput: The generated output.
        """
        if not prompt or not examples:
            if not fewshot_type:
                fewshot_type = LATS_BENCHMARK_FEWSHOTS[self.benchmark][0]
            fewshots = LATSAgent.get_fewshots(
                benchmark=self.benchmark, fewshot_type=fewshot_type
            )
            prompts = LATSAgent.get_prompts(benchmark=self.benchmark)
            examples = fewshots["examples"]
            reflect_examples = fewshots["reflect_examples"]
            value_examples = fewshots["value_examples"]
            prompt = prompts["prompt"]
            reflect_prompt = prompts["reflect_prompt"]
            value_prompt = prompts["value_prompt"]

        out = self.strategy.generate(
            question=question,
            key=key,
            examples=examples,
            reflect_examples=reflect_examples,
            value_examples=value_examples,
            prompt=prompt,
            reflect_prompt=reflect_prompt,
            value_prompt=value_prompt,
            additional_keys=additional_keys,
            reflect_additional_keys=reflect_additional_keys,
            value_additional_keys=value_additional_keys,
            max_iterations=max_iterations,
            reset=reset,
        )

        return out

    def reset(self) -> None:
        """Reset the agent."""
        self.strategy.reset()
