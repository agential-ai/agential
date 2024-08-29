"""CRITIC Agent.

Original Paper: https://arxiv.org/pdf/2305.11738
Paper Repository: https://github.com/microsoft/ProphetNet/tree/master/CRITIC
"""

from typing import Any, Dict

from agential.cog.base.agent import BaseAgent
from agential.cog.constants import BENCHMARK_FEWSHOTS, Benchmarks, FewShotType
from agential.cog.critic.output import CriticOutput
from agential.cog.critic.prompts import (
    AMBIGNQ_FEWSHOT_EXAMPLES_CRITIC,
    CRITIC_CRITIQUE_INSTRUCTION_AMBIGNQ,
    CRITIC_CRITIQUE_INSTRUCTION_FEVER,
    CRITIC_CRITIQUE_INSTRUCTION_GSM8K,
    CRITIC_CRITIQUE_INSTRUCTION_HOTPOTQA,
    CRITIC_CRITIQUE_INSTRUCTION_HUMANEVAL,
    CRITIC_CRITIQUE_INSTRUCTION_MBPP,
    CRITIC_CRITIQUE_INSTRUCTION_SVAMP,
    CRITIC_CRITIQUE_INSTRUCTION_TABMWP,
    CRITIC_CRITIQUE_INSTRUCTION_TRIVIAQA,
    CRITIC_CRITIQUE_NO_TOOL_INSTRUCTION_GSM8K,
    CRITIC_CRITIQUE_NO_TOOL_INSTRUCTION_HUMANEVAL,
    CRITIC_CRITIQUE_NO_TOOL_INSTRUCTION_MBPP,
    CRITIC_CRITIQUE_NO_TOOL_INSTRUCTION_SVAMP,
    CRITIC_CRITIQUE_NO_TOOL_INSTRUCTION_TABMWP,
    CRITIC_INSTRUCTION_AMBIGNQ,
    CRITIC_INSTRUCTION_FEVER,
    CRITIC_INSTRUCTION_HOTPOTQA,
    CRITIC_INSTRUCTION_TRIVIAQA,
    CRITIC_POT_INSTRUCTION_GSM8K,
    CRITIC_POT_INSTRUCTION_HUMANEVAL,
    CRITIC_POT_INSTRUCTION_MBPP,
    CRITIC_POT_INSTRUCTION_SVAMP,
    CRITIC_POT_INSTRUCTION_TABMWP,
    FEVER_FEWSHOT_EXAMPLES_CRITIC,
    GSM8K_FEWSHOT_EXAMPLES_CRITIC,
    GSM8K_FEWSHOT_EXAMPLES_CRITIC_NO_TOOL,
    HOTPOTQA_FEWSHOT_EXAMPLES_CRITIC,
    HUMANEVAL_FEWSHOT_EXAMPLES_CRITIC,
    HUMANEVAL_FEWSHOT_EXAMPLES_CRITIC_NO_TOOL,
    MBPP_FEWSHOT_EXAMPLES_CRITIC,
    MBPP_FEWSHOT_EXAMPLES_CRITIC_NO_TOOL,
    SVAMP_FEWSHOT_EXAMPLES_CRITIC,
    SVAMP_FEWSHOT_EXAMPLES_CRITIC_NO_TOOL,
    TABMWP_FEWSHOT_EXAMPLES_CRITIC,
    TABMWP_FEWSHOT_EXAMPLES_CRITIC_NO_TOOL,
    TRIVIAQA_FEWSHOT_EXAMPLES_CRITIC,
)
from agential.cog.critic.strategies.base import CriticBaseStrategy
from agential.cog.critic.strategies.code import (
    CriticHEvalStrategy,
    CriticMBPPStrategy,
)
from agential.cog.critic.strategies.math import (
    CriticGSM8KStrategy,
    CriticSVAMPStrategy,
    CriticTabMWPStrategy,
)
from agential.cog.critic.strategies.qa import (
    CriticAmbigNQStrategy,
    CriticFEVERStrategy,
    CriticHotQAStrategy,
    CriticTriviaQAStrategy,
)
from agential.llm.llm import BaseLLM

CRITIC_BENCHMARK_FEWSHOTS = {
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


CRITIC_PROMPTS = {
    Benchmarks.HOTPOTQA: {
        "prompt": CRITIC_INSTRUCTION_HOTPOTQA,
        "critique_prompt": CRITIC_CRITIQUE_INSTRUCTION_HOTPOTQA,
        "critique_prompt_no_tool": CRITIC_CRITIQUE_INSTRUCTION_HOTPOTQA,
    },
    Benchmarks.FEVER: {
        "prompt": CRITIC_INSTRUCTION_FEVER,
        "critique_prompt": CRITIC_CRITIQUE_INSTRUCTION_FEVER,
        "critique_prompt_no_tool": CRITIC_CRITIQUE_INSTRUCTION_FEVER,
    },
    Benchmarks.TRIVIAQA: {
        "prompt": CRITIC_INSTRUCTION_TRIVIAQA,
        "critique_prompt": CRITIC_CRITIQUE_INSTRUCTION_TRIVIAQA,
        "critique_prompt_no_tool": CRITIC_CRITIQUE_INSTRUCTION_TRIVIAQA,
    },
    Benchmarks.AMBIGNQ: {
        "prompt": CRITIC_INSTRUCTION_AMBIGNQ,
        "critique_prompt": CRITIC_CRITIQUE_INSTRUCTION_AMBIGNQ,
        "critique_prompt_no_tool": CRITIC_CRITIQUE_INSTRUCTION_AMBIGNQ,
    },
    Benchmarks.GSM8K: {
        "prompt": CRITIC_POT_INSTRUCTION_GSM8K,
        "critique_prompt": CRITIC_CRITIQUE_INSTRUCTION_GSM8K,
        "critique_prompt_no_tool": CRITIC_CRITIQUE_NO_TOOL_INSTRUCTION_GSM8K,
    },
    Benchmarks.SVAMP: {
        "prompt": CRITIC_POT_INSTRUCTION_SVAMP,
        "critique_prompt": CRITIC_CRITIQUE_INSTRUCTION_SVAMP,
        "critique_prompt_no_tool": CRITIC_CRITIQUE_NO_TOOL_INSTRUCTION_SVAMP,
    },
    Benchmarks.TABMWP: {
        "prompt": CRITIC_POT_INSTRUCTION_TABMWP,
        "critique_prompt": CRITIC_CRITIQUE_INSTRUCTION_TABMWP,
        "critique_prompt_no_tool": CRITIC_CRITIQUE_NO_TOOL_INSTRUCTION_TABMWP,
    },
    Benchmarks.HUMANEVAL: {
        "prompt": CRITIC_POT_INSTRUCTION_HUMANEVAL,
        "critique_prompt": CRITIC_CRITIQUE_INSTRUCTION_HUMANEVAL,
        "critique_prompt_no_tool": CRITIC_CRITIQUE_NO_TOOL_INSTRUCTION_HUMANEVAL,
    },
    Benchmarks.MBPP: {
        "prompt": CRITIC_POT_INSTRUCTION_MBPP,
        "critique_prompt": CRITIC_CRITIQUE_INSTRUCTION_MBPP,
        "critique_prompt_no_tool": CRITIC_CRITIQUE_NO_TOOL_INSTRUCTION_MBPP,
    },
}

CRITIC_FEWSHOTS = {
    Benchmarks.HOTPOTQA: {
        "critique_examples": HOTPOTQA_FEWSHOT_EXAMPLES_CRITIC,
        "critique_examples_no_tool": HOTPOTQA_FEWSHOT_EXAMPLES_CRITIC,
    },
    Benchmarks.FEVER: {
        "critique_examples": FEVER_FEWSHOT_EXAMPLES_CRITIC,
        "critique_examples_no_tool": FEVER_FEWSHOT_EXAMPLES_CRITIC,
    },
    Benchmarks.TRIVIAQA: {
        "critique_examples": TRIVIAQA_FEWSHOT_EXAMPLES_CRITIC,
        "critique_examples_no_tool": TRIVIAQA_FEWSHOT_EXAMPLES_CRITIC,
    },
    Benchmarks.AMBIGNQ: {
        "critique_examples": AMBIGNQ_FEWSHOT_EXAMPLES_CRITIC,
        "critique_examples_no_tool": AMBIGNQ_FEWSHOT_EXAMPLES_CRITIC,
    },
    Benchmarks.GSM8K: {
        "critique_examples": GSM8K_FEWSHOT_EXAMPLES_CRITIC,
        "critique_examples_no_tool": GSM8K_FEWSHOT_EXAMPLES_CRITIC_NO_TOOL,
    },
    Benchmarks.SVAMP: {
        "critique_examples": SVAMP_FEWSHOT_EXAMPLES_CRITIC,
        "critique_examples_no_tool": SVAMP_FEWSHOT_EXAMPLES_CRITIC_NO_TOOL,
    },
    Benchmarks.TABMWP: {
        "critique_examples": TABMWP_FEWSHOT_EXAMPLES_CRITIC,
        "critique_examples_no_tool": TABMWP_FEWSHOT_EXAMPLES_CRITIC_NO_TOOL,
    },
    Benchmarks.HUMANEVAL: {
        "critique_examples": HUMANEVAL_FEWSHOT_EXAMPLES_CRITIC,
        "critique_examples_no_tool": HUMANEVAL_FEWSHOT_EXAMPLES_CRITIC_NO_TOOL,
    },
    Benchmarks.MBPP: {
        "critique_examples": MBPP_FEWSHOT_EXAMPLES_CRITIC,
        "critique_examples_no_tool": MBPP_FEWSHOT_EXAMPLES_CRITIC_NO_TOOL,
    },
}

CRITIC_STRATEGIES = {
    Benchmarks.HOTPOTQA: CriticHotQAStrategy,
    Benchmarks.FEVER: CriticFEVERStrategy,
    Benchmarks.TRIVIAQA: CriticTriviaQAStrategy,
    Benchmarks.AMBIGNQ: CriticAmbigNQStrategy,
    Benchmarks.GSM8K: CriticGSM8KStrategy,
    Benchmarks.SVAMP: CriticSVAMPStrategy,
    Benchmarks.TABMWP: CriticTabMWPStrategy,
    Benchmarks.HUMANEVAL: CriticHEvalStrategy,
    Benchmarks.MBPP: CriticMBPPStrategy,
}


class CriticAgent(BaseAgent):
    """CRITIC Agent.

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

        self.strategy = CriticAgent.get_strategy(
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
        if (
            benchmark not in CRITIC_FEWSHOTS
            or benchmark not in CRITIC_BENCHMARK_FEWSHOTS
        ):
            raise ValueError(f"Benchmark '{benchmark}' few-shots not found for Critic.")

        if fewshot_type not in CRITIC_BENCHMARK_FEWSHOTS[benchmark]:
            raise ValueError(
                f"Benchmark '{benchmark}' few-shot type not supported for Critic."
            )

        benchmark_fewshots = BENCHMARK_FEWSHOTS[benchmark][fewshot_type]

        use_tool = kwargs.get("use_tool")
        if use_tool is None:
            raise ValueError("`use_tool` not specified.")

        if use_tool:
            return {
                "examples": benchmark_fewshots,
                "critique_examples": CRITIC_FEWSHOTS[benchmark]["critique_examples"],
            }
        return {
            "examples": benchmark_fewshots,
            "critique_examples": CRITIC_FEWSHOTS[benchmark][
                "critique_examples_no_tool"
            ],
        }

    @staticmethod
    def get_prompts(benchmark: str, **kwargs: Any) -> Dict[str, str]:
        """Retrieve the prompt instruction based on the benchmark.

        Args:
            benchmark (str): The benchmark name.
            **kwargs (Any): Additional arguments.

        Returns:
            Dict[str, str]: The prompt instructions.
        """
        if benchmark not in CRITIC_PROMPTS:
            raise ValueError(f"Benchmark '{benchmark}' prompt not found for Critic.")

        use_tool = kwargs.get("use_tool")
        if use_tool is None:
            raise ValueError("`use_tool` not specified.")

        if use_tool:
            return {
                "prompt": CRITIC_PROMPTS[benchmark]["prompt"],
                "critique_prompt": CRITIC_PROMPTS[benchmark]["critique_prompt"],
            }
        return {
            "prompt": CRITIC_PROMPTS[benchmark]["prompt"],
            "critique_prompt": CRITIC_PROMPTS[benchmark]["critique_prompt_no_tool"],
        }

    @staticmethod
    def get_strategy(benchmark: str, **kwargs: Any) -> CriticBaseStrategy:
        """Returns an instance of the appropriate Critic strategy based on the provided benchmark.

        Args:
            benchmark (str): The benchmark name.
            **kwargs (Any): Additional keyword arguments to pass to
                the strategy's constructor.

        Returns:
            CriticBaseStrategy: An instance of the appropriate Critic strategy.
        """
        if benchmark not in CRITIC_STRATEGIES:
            raise ValueError(f"Unsupported benchmark: {benchmark} for agent Critic")

        strategy = CRITIC_STRATEGIES[benchmark]
        return strategy(**kwargs)

    def generate(
        self,
        question: str,
        examples: str = "",
        prompt: str = "",
        critique_examples: str = "",
        critique_prompt: str = "",
        additional_keys: Dict[str, str] = {},
        critique_additional_keys: Dict[str, str] = {},
        fewshot_type: str = "",
        max_interactions: int = 7,
        use_tool: bool = True,
        reset: bool = True,
    ) -> CriticOutput:
        """Generates an answer that is refined with search results.

        Args:
            question (str): The question to be answered.
            examples (str, optional): Few-shot examples to guide the language model in generating the initial answer. Defaults to "".
            prompt (str, optional): The instruction template used to prompt the language model for the initial answer. Defaults to "".
            critique_examples (str, optional): Few-shot examples to guide the language model in generating critiques. Defaults to "".
            critique_prompt (str, optional): The instruction template for generating critiques. Defaults to "".
            additional_keys (Dict[str, str]): Additional keys to format the prompt. Defaults to {}.
            critique_additional_keys (Dict[str, str]): Additional keys to format the critique_prompt. Defaults to {}.
            fewshot_type (str): The type of few-shot examples to use. Defaults to "".
            max_interactions (int): The maximum number of critique cycles. Defaults to 7.
            use_tool (bool): Use the external tool. Flag to decide whether to use the interpreter tool for math/code execution, or search tool for QA. Defaults to True.
            reset (bool): Resets the agent's state. Defaults to True.

        Returns:
            CriticOutput: The output of the CRITIC agent.
        """
        if not prompt or not critique_prompt or not examples or not critique_examples:
            if not fewshot_type:
                fewshot_type = CRITIC_BENCHMARK_FEWSHOTS[self.benchmark][0]
            fewshots = CriticAgent.get_fewshots(
                benchmark=self.benchmark, fewshot_type=fewshot_type, use_tool=use_tool
            )
            prompts = CriticAgent.get_prompts(
                benchmark=self.benchmark, use_tool=use_tool
            )
            examples = fewshots["examples"]
            prompt = prompts["prompt"]
            critique_examples = fewshots["critique_examples"]
            critique_prompt = prompts["critique_prompt"]

        out = self.strategy.generate(
            question=question,
            examples=examples,
            critique_examples=critique_examples,
            prompt=prompt,
            critique_prompt=critique_prompt,
            additional_keys=additional_keys,
            critique_additional_keys=critique_additional_keys,
            max_interactions=max_interactions,
            use_tool=use_tool,
            reset=reset,
        )

        return out
