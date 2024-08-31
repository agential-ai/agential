"""Self-Refine Agent.

Original Webpage: https://selfrefine.info/
Paper Repository: https://github.com/madaan/self-refine
"""

from typing import Any, Dict, List

from agential.agents.self_refine.output import SelfRefineOutput
from agential.agents.self_refine.prompts import (
    AMBIGNQ_CRITIQUE_FEWSHOT_EXAMPLES,
    AMBIGNQ_REFINE_FEWSHOT_EXAMPLES,
    FEVER_CRITIQUE_FEWSHOT_EXAMPLES,
    FEVER_REFINE_FEWSHOT_EXAMPLES,
    GSM8K_CRITIQUE_FEWSHOT_EXAMPLES,
    GSM8K_REFINE_FEWSHOT_EXAMPLES,
    HOTPOTQA_CRITIQUE_FEWSHOT_EXAMPLES,
    HOTPOTQA_REFINE_FEWSHOT_EXAMPLES,
    HUMANEVAL_CRITIQUE_FEWSHOT_EXAMPLES,
    HUMANEVAL_REFINE_FEWSHOT_EXAMPLES,
    MBPP_CRITIQUE_FEWSHOT_EXAMPLES,
    MBPP_REFINE_FEWSHOT_EXAMPLES,
    SELF_REFINE_CRITIQUE_INSTRUCTION_AMBIGNQ,
    SELF_REFINE_CRITIQUE_INSTRUCTION_FEVER,
    SELF_REFINE_CRITIQUE_INSTRUCTION_GSM8K,
    SELF_REFINE_CRITIQUE_INSTRUCTION_HOTPOTQA,
    SELF_REFINE_CRITIQUE_INSTRUCTION_HUMANEVAL,
    SELF_REFINE_CRITIQUE_INSTRUCTION_MBPP,
    SELF_REFINE_CRITIQUE_INSTRUCTION_SVAMP,
    SELF_REFINE_CRITIQUE_INSTRUCTION_TABMWP,
    SELF_REFINE_CRITIQUE_INSTRUCTION_TRIVIAQA,
    SELF_REFINE_INSTRUCTION_AMBIGNQ,
    SELF_REFINE_INSTRUCTION_FEVER,
    SELF_REFINE_INSTRUCTION_GSM8K,
    SELF_REFINE_INSTRUCTION_HOTPOTQA,
    SELF_REFINE_INSTRUCTION_HUMANEVAL,
    SELF_REFINE_INSTRUCTION_MBPP,
    SELF_REFINE_INSTRUCTION_SVAMP,
    SELF_REFINE_INSTRUCTION_TABMWP,
    SELF_REFINE_INSTRUCTION_TRIVIAQA,
    SELF_REFINE_REFINE_INSTRUCTION_AMBIGNQ,
    SELF_REFINE_REFINE_INSTRUCTION_FEVER,
    SELF_REFINE_REFINE_INSTRUCTION_GSM8K,
    SELF_REFINE_REFINE_INSTRUCTION_HOTPOTQA,
    SELF_REFINE_REFINE_INSTRUCTION_HUMANEVAL,
    SELF_REFINE_REFINE_INSTRUCTION_MBPP,
    SELF_REFINE_REFINE_INSTRUCTION_SVAMP,
    SELF_REFINE_REFINE_INSTRUCTION_TABMWP,
    SELF_REFINE_REFINE_INSTRUCTION_TRIVIAQA,
    SVAMP_CRITIQUE_FEWSHOT_EXAMPLES,
    SVAMP_REFINE_FEWSHOT_EXAMPLES,
    TABMWP_CRITIQUE_FEWSHOT_EXAMPLES,
    TABMWP_REFINE_FEWSHOT_EXAMPLES,
    TRIVIAQA_CRITIQUE_FEWSHOT_EXAMPLES,
    TRIVIAQA_REFINE_FEWSHOT_EXAMPLES,
)
from agential.agents.self_refine.strategies.base import SelfRefineBaseStrategy
from agential.agents.self_refine.strategies.code import (
    SelfRefineHEvalStrategy,
    SelfRefineMBPPStrategy,
)
from agential.agents.self_refine.strategies.math import (
    SelfRefineGSM8KStrategy,
    SelfRefineSVAMPStrategy,
    SelfRefineTabMWPStrategy,
)
from agential.agents.self_refine.strategies.qa import (
    SelfRefineAmbigNQStrategy,
    SelfRefineFEVERStrategy,
    SelfRefineHotQAStrategy,
    SelfRefineTriviaQAStrategy,
)
from agential.constants import BENCHMARK_FEWSHOTS, Benchmarks, FewShotType
from agential.core.base.agents.agent import BaseAgent
from agential.llm.llm import BaseLLM

SELF_REFINE_BENCHMARK_FEWSHOTS = {
    Benchmarks.HOTPOTQA: [FewShotType.COT, FewShotType.DIRECT, FewShotType.REACT],
    Benchmarks.FEVER: [FewShotType.COT, FewShotType.DIRECT, FewShotType.REACT],
    Benchmarks.TRIVIAQA: [FewShotType.COT, FewShotType.DIRECT, FewShotType.REACT],
    Benchmarks.AMBIGNQ: [FewShotType.COT, FewShotType.DIRECT, FewShotType.REACT],
    Benchmarks.GSM8K: [FewShotType.POT],
    Benchmarks.SVAMP: [FewShotType.POT],
    Benchmarks.TABMWP: [FewShotType.POT],
    Benchmarks.HUMANEVAL: [FewShotType.POT],
    Benchmarks.MBPP: [FewShotType.POT],
}

SELF_REFINE_PROMPTS = {
    Benchmarks.HOTPOTQA: {
        "prompt": SELF_REFINE_INSTRUCTION_HOTPOTQA,
        "critique_prompt": SELF_REFINE_CRITIQUE_INSTRUCTION_HOTPOTQA,
        "refine_prompt": SELF_REFINE_REFINE_INSTRUCTION_HOTPOTQA,
    },
    Benchmarks.FEVER: {
        "prompt": SELF_REFINE_INSTRUCTION_FEVER,
        "critique_prompt": SELF_REFINE_CRITIQUE_INSTRUCTION_FEVER,
        "refine_prompt": SELF_REFINE_REFINE_INSTRUCTION_FEVER,
    },
    Benchmarks.TRIVIAQA: {
        "prompt": SELF_REFINE_INSTRUCTION_TRIVIAQA,
        "critique_prompt": SELF_REFINE_CRITIQUE_INSTRUCTION_TRIVIAQA,
        "refine_prompt": SELF_REFINE_REFINE_INSTRUCTION_TRIVIAQA,
    },
    Benchmarks.AMBIGNQ: {
        "prompt": SELF_REFINE_INSTRUCTION_AMBIGNQ,
        "critique_prompt": SELF_REFINE_CRITIQUE_INSTRUCTION_AMBIGNQ,
        "refine_prompt": SELF_REFINE_REFINE_INSTRUCTION_AMBIGNQ,
    },
    Benchmarks.GSM8K: {
        "prompt": SELF_REFINE_INSTRUCTION_GSM8K,
        "critique_prompt": SELF_REFINE_CRITIQUE_INSTRUCTION_GSM8K,
        "refine_prompt": SELF_REFINE_REFINE_INSTRUCTION_GSM8K,
    },
    Benchmarks.SVAMP: {
        "prompt": SELF_REFINE_INSTRUCTION_SVAMP,
        "critique_prompt": SELF_REFINE_CRITIQUE_INSTRUCTION_SVAMP,
        "refine_prompt": SELF_REFINE_REFINE_INSTRUCTION_SVAMP,
    },
    Benchmarks.TABMWP: {
        "prompt": SELF_REFINE_INSTRUCTION_TABMWP,
        "critique_prompt": SELF_REFINE_CRITIQUE_INSTRUCTION_TABMWP,
        "refine_prompt": SELF_REFINE_REFINE_INSTRUCTION_TABMWP,
    },
    Benchmarks.HUMANEVAL: {
        "prompt": SELF_REFINE_INSTRUCTION_HUMANEVAL,
        "critique_prompt": SELF_REFINE_CRITIQUE_INSTRUCTION_HUMANEVAL,
        "refine_prompt": SELF_REFINE_REFINE_INSTRUCTION_HUMANEVAL,
    },
    Benchmarks.MBPP: {
        "prompt": SELF_REFINE_INSTRUCTION_MBPP,
        "critique_prompt": SELF_REFINE_CRITIQUE_INSTRUCTION_MBPP,
        "refine_prompt": SELF_REFINE_REFINE_INSTRUCTION_MBPP,
    },
}

SELF_REFINE_FEWSHOTS: Dict[str, Dict] = {
    Benchmarks.HOTPOTQA: {
        "critique_examples": HOTPOTQA_CRITIQUE_FEWSHOT_EXAMPLES,
        "refine_examples": HOTPOTQA_REFINE_FEWSHOT_EXAMPLES,
    },
    Benchmarks.FEVER: {
        "critique_examples": FEVER_CRITIQUE_FEWSHOT_EXAMPLES,
        "refine_examples": FEVER_REFINE_FEWSHOT_EXAMPLES,
    },
    Benchmarks.TRIVIAQA: {
        "critique_examples": TRIVIAQA_CRITIQUE_FEWSHOT_EXAMPLES,
        "refine_examples": TRIVIAQA_REFINE_FEWSHOT_EXAMPLES,
    },
    Benchmarks.AMBIGNQ: {
        "critique_examples": AMBIGNQ_CRITIQUE_FEWSHOT_EXAMPLES,
        "refine_examples": AMBIGNQ_REFINE_FEWSHOT_EXAMPLES,
    },
    Benchmarks.GSM8K: {
        "critique_examples": GSM8K_CRITIQUE_FEWSHOT_EXAMPLES,
        "refine_examples": GSM8K_REFINE_FEWSHOT_EXAMPLES,
    },
    Benchmarks.SVAMP: {
        "critique_examples": SVAMP_CRITIQUE_FEWSHOT_EXAMPLES,
        "refine_examples": SVAMP_REFINE_FEWSHOT_EXAMPLES,
    },
    Benchmarks.TABMWP: {
        "critique_examples": TABMWP_CRITIQUE_FEWSHOT_EXAMPLES,
        "refine_examples": TABMWP_REFINE_FEWSHOT_EXAMPLES,
    },
    Benchmarks.HUMANEVAL: {
        "critique_examples": HUMANEVAL_CRITIQUE_FEWSHOT_EXAMPLES,
        "refine_examples": HUMANEVAL_REFINE_FEWSHOT_EXAMPLES,
    },
    Benchmarks.MBPP: {
        "critique_examples": MBPP_CRITIQUE_FEWSHOT_EXAMPLES,
        "refine_examples": MBPP_REFINE_FEWSHOT_EXAMPLES,
    },
}

SELF_REFINE_STRATEGIES = {
    Benchmarks.HOTPOTQA: SelfRefineHotQAStrategy,
    Benchmarks.FEVER: SelfRefineFEVERStrategy,
    Benchmarks.TRIVIAQA: SelfRefineTriviaQAStrategy,
    Benchmarks.AMBIGNQ: SelfRefineAmbigNQStrategy,
    Benchmarks.GSM8K: SelfRefineGSM8KStrategy,
    Benchmarks.SVAMP: SelfRefineSVAMPStrategy,
    Benchmarks.TABMWP: SelfRefineTabMWPStrategy,
    Benchmarks.HUMANEVAL: SelfRefineHEvalStrategy,
    Benchmarks.MBPP: SelfRefineMBPPStrategy,
}


class SelfRefineAgent(BaseAgent):
    """The Self-Refine agent that utilizes the self-refinement process to iteratively improve solutions based on critique.

    The agent prompts a language model to generate solutions to a given problem, obtains critique on the generated
    solutions, and then refines the solutions based on this critique. This process can be repeated a specified number
    of times or until the critique indicates that no further improvements are needed.

    Attributes:
        llm (BaseLLM): An instance of a language model used for generating initial answers
            and critiques.
        benchmark (str): The benchmark name.
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

        self.strategy = SelfRefineAgent.get_strategy(
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
        if benchmark not in SELF_REFINE_FEWSHOTS:
            raise ValueError(
                f"Benchmark '{benchmark}' few-shots not found for Self-Refine."
            )

        if fewshot_type not in SELF_REFINE_BENCHMARK_FEWSHOTS[benchmark]:
            raise ValueError(
                f"Benchmark '{benchmark}' few-shot type not supported for Self-Refine."
            )

        benchmark_fewshots = BENCHMARK_FEWSHOTS[benchmark][fewshot_type]

        return {"examples": benchmark_fewshots, **SELF_REFINE_FEWSHOTS[benchmark]}  # type: ignore

    @staticmethod
    def get_prompts(benchmark: str, **kwargs: Any) -> Dict[str, str]:
        """Retrieve the prompt instruction based on the benchmark.

        Args:
            benchmark (str): The benchmark name.
            **kwargs (Any): Additional arguments.

        Returns:
            Dict[str, str]: A dictionary of prompt instructions.
        """
        if benchmark not in SELF_REFINE_PROMPTS:
            raise ValueError(
                f"Benchmark '{benchmark}' prompt not found for Self-Refine."
            )

        return SELF_REFINE_PROMPTS[benchmark]

    @staticmethod
    def get_strategy(benchmark: str, **kwargs: Any) -> SelfRefineBaseStrategy:
        """Returns an instance of the appropriate Self-Refine strategy based on the provided benchmark.

        Args:
            benchmark (str): The benchmark name.
            **kwargs (Any): Additional keyword arguments to pass to
                the strategy's constructor.

        Returns:
            SelfRefineBaseStrategy: An instance of the appropriate Self-Refine strategy.
        """
        if benchmark not in SELF_REFINE_STRATEGIES:
            raise ValueError(
                f"Unsupported benchmark: {benchmark} for agent Self-Refine"
            )

        strategy = SELF_REFINE_STRATEGIES[benchmark]
        if strategy is None:
            raise ValueError(f"No strategy defined for benchmark: {benchmark}")

        return strategy(**kwargs)  # type: ignore

    def generate(
        self,
        question: str,
        examples: str = "",
        prompt: str = "",
        critique_examples: str = "",
        critique_prompt: str = "",
        refine_examples: str = "",
        refine_prompt: str = "",
        additional_keys: Dict[str, str] = {},
        critique_additional_keys: Dict[str, str] = {},
        refine_additional_keys: Dict[str, str] = {},
        fewshot_type: str = "",
        max_interactions: int = 3,
        reset: bool = True,
    ) -> SelfRefineOutput:
        """Generates a refined solution for a given question through an iterative self-refinement process.

        The process includes generating initial solutions, soliciting critique, and refining the solution
        based on critique, repeated for a maximum number of attempts or until critique indicates satisfaction.

        Args:
            question (str): The question or problem to solve.
            examples (str, optional): Precedent examples to guide initial solution generation. Defaults to "".
            prompt (str, optional): Instructional prompt for initial solution generation. Defaults to "".
            critique_examples (str, optional): Precedent examples to guide critique generation. Defaults to "".
            critique_prompt (str, optional): Instructional prompt for critique generation. Defaults to "".
            refine_examples (str, optional): Precedent examples to guide solution refinement. Defaults to "".
            refine_prompt (str, optional): Instructional prompt for refining the solution. Defaults to "".
            additional_keys (Dict[str, str]): Additional keys to format the prompt. Defaults to {}.
            critique_additional_keys (Dict[str, str]): Additional keys to format the critique_prompt. Defaults to {}.
            refine_additional_keys (Dict[str, str]): Additional keys to format the refine_prompt. Defaults to {}.
            fewshot_type (str): The type of few-shot examples to use. Defaults to "".
            max_interactions (int): Maximum number of refinement iterations.
            reset (bool): Resets the agent's state. Defaults to True.

        Returns:
            SelfRefineOutput:The agent's output.
        """
        if (
            not prompt
            or not critique_prompt
            or not examples
            or not critique_examples
            or not refine_examples
            or not refine_prompt
        ):
            if not fewshot_type:
                fewshot_type = SELF_REFINE_BENCHMARK_FEWSHOTS[self.benchmark][0]  # type: ignore
            fewshots = SelfRefineAgent.get_fewshots(
                benchmark=self.benchmark, fewshot_type=fewshot_type
            )
            prompts = SelfRefineAgent.get_prompts(benchmark=self.benchmark)
            examples = fewshots["examples"]
            critique_examples = fewshots["critique_examples"]
            refine_examples = fewshots["refine_examples"]
            prompt = prompts["prompt"]
            critique_prompt = prompts["critique_prompt"]
            refine_prompt = prompts["refine_prompt"]

        out = self.strategy.generate(
            question=question,
            examples=examples,
            prompt=prompt,
            critique_examples=critique_examples,
            critique_prompt=critique_prompt,
            refine_examples=refine_examples,
            refine_prompt=refine_prompt,
            additional_keys=additional_keys,
            critique_additional_keys=critique_additional_keys,
            refine_additional_keys=refine_additional_keys,
            max_interactions=max_interactions,
            reset=reset,
        )

        return out
