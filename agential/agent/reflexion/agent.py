"""Reflexion Agent.

Original Paper: https://arxiv.org/abs/2303.11366
Paper Repositories:
    - https://github.com/noahshinn/reflexion-draft
    - https://github.com/noahshinn/reflexion
"""

from typing import Any, Dict, Optional

from agential.core.base.agent import BaseAgent
from agential.agent.constants import BENCHMARK_FEWSHOTS, Benchmarks, FewShotType
from agential.agent.reflexion.output import (
    ReflexionCoTOutput,
    ReflexionReActOutput,
)
from agential.agent.reflexion.prompts import (
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
from agential.agent.reflexion.reflect import (
    ReflexionCoTReflector,
    ReflexionReActReflector,
)
from agential.agent.reflexion.strategies.base import (
    ReflexionCoTBaseStrategy,
    ReflexionReActBaseStrategy,
)
from agential.agent.reflexion.strategies.code import (
    ReflexionCoTHEvalStrategy,
    ReflexionCoTMBPPStrategy,
    ReflexionReActHEvalStrategy,
    ReflexionReActMBPPStrategy,
)
from agential.agent.reflexion.strategies.math import (
    ReflexionCoTGSM8KStrategy,
    ReflexionCoTSVAMPStrategy,
    ReflexionCoTTabMWPStrategy,
    ReflexionReActGSM8KStrategy,
    ReflexionReActSVAMPStrategy,
    ReflexionReActTabMWPStrategy,
)
from agential.agent.reflexion.strategies.qa import (
    ReflexionCoTAmbigNQStrategy,
    ReflexionCoTFEVERStrategy,
    ReflexionCoTHotQAStrategy,
    ReflexionCoTTriviaQAStrategy,
    ReflexionReActAmbigNQStrategy,
    ReflexionReActFEVERStrategy,
    ReflexionReActHotQAStrategy,
    ReflexionReActTriviaQAStrategy,
)
from agential.llm.llm import BaseLLM

REFLEXION_COT_BENCHMARK_FEWSHOTS = {
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

REFLEXION_COT_PROMPTS = {
    Benchmarks.HOTPOTQA: {
        "prompt": REFLEXION_COT_INSTRUCTION_HOTPOTQA,
        "reflect_prompt": REFLEXION_COT_REFLECT_INSTRUCTION_HOTPOTQA,
    },
    Benchmarks.FEVER: {
        "prompt": REFLEXION_COT_INSTRUCTION_FEVER,
        "reflect_prompt": REFLEXION_COT_REFLECT_INSTRUCTION_FEVER,
    },
    Benchmarks.TRIVIAQA: {
        "prompt": REFLEXION_COT_INSTRUCTION_TRIVIAQA,
        "reflect_prompt": REFLEXION_COT_REFLECT_INSTRUCTION_TRIVIAQA,
    },
    Benchmarks.AMBIGNQ: {
        "prompt": REFLEXION_COT_INSTRUCTION_AMBIGNQ,
        "reflect_prompt": REFLEXION_COT_REFLECT_INSTRUCTION_AMBIGNQ,
    },
    Benchmarks.GSM8K: {
        "prompt": REFLEXION_COT_INSTRUCTION_GSM8K,
        "reflect_prompt": REFLEXION_COT_REFLECT_INSTRUCTION_GSM8K,
    },
    Benchmarks.SVAMP: {
        "prompt": REFLEXION_COT_INSTRUCTION_SVAMP,
        "reflect_prompt": REFLEXION_COT_REFLECT_INSTRUCTION_SVAMP,
    },
    Benchmarks.TABMWP: {
        "prompt": REFLEXION_COT_INSTRUCTION_TABMWP,
        "reflect_prompt": REFLEXION_COT_REFLECT_INSTRUCTION_TABMWP,
    },
    Benchmarks.HUMANEVAL: {
        "prompt": REFLEXION_COT_INSTRUCTION_HUMANEVAL,
        "reflect_prompt": REFLEXION_COT_REFLECT_INSTRUCTION_HUMANEVAL,
    },
    Benchmarks.MBPP: {
        "prompt": REFLEXION_COT_INSTRUCTION_MBPP,
        "reflect_prompt": REFLEXION_COT_REFLECT_INSTRUCTION_MBPP,
    },
}

REFLEXION_COT_FEWSHOTS = {
    Benchmarks.HOTPOTQA: {
        "reflect_examples": HOTPOTQA_FEWSHOT_EXAMPLES_REFLEXION_COT_REFLECT,
    },
    Benchmarks.TRIVIAQA: {
        "reflect_examples": TRIVIAQA_FEWSHOT_EXAMPLES_REFLEXION_COT_REFLECT,
    },
    Benchmarks.AMBIGNQ: {
        "reflect_examples": AMBIGNQ_FEWSHOT_EXAMPLES_REFLEXION_COT_REFLECT,
    },
    Benchmarks.FEVER: {
        "reflect_examples": FEVER_FEWSHOT_EXAMPLES_REFLEXION_COT_REFLECT,
    },
    Benchmarks.GSM8K: {
        "reflect_examples": GSM8K_FEWSHOT_EXAMPLES_REFLEXION_COT_REFLECT,
    },
    Benchmarks.SVAMP: {
        "reflect_examples": SVAMP_FEWSHOT_EXAMPLES_REFLEXION_COT_REFLECT,
    },
    Benchmarks.TABMWP: {
        "reflect_examples": TABMWP_FEWSHOT_EXAMPLES_REFLEXION_COT_REFLECT,
    },
    Benchmarks.HUMANEVAL: {
        "reflect_examples": HUMANEVAL_FEWSHOT_EXAMPLES_REFLEXION_COT_REFLECT,
    },
    Benchmarks.MBPP: {
        "reflect_examples": MBPP_FEWSHOT_EXAMPLES_REFLEXION_COT_REFLECT,
    },
}


REFLEXION_COT_STRATEGIES = {
    Benchmarks.HOTPOTQA: ReflexionCoTHotQAStrategy,
    Benchmarks.FEVER: ReflexionCoTFEVERStrategy,
    Benchmarks.TRIVIAQA: ReflexionCoTTriviaQAStrategy,
    Benchmarks.AMBIGNQ: ReflexionCoTAmbigNQStrategy,
    Benchmarks.GSM8K: ReflexionCoTGSM8KStrategy,
    Benchmarks.SVAMP: ReflexionCoTSVAMPStrategy,
    Benchmarks.TABMWP: ReflexionCoTTabMWPStrategy,
    Benchmarks.HUMANEVAL: ReflexionCoTHEvalStrategy,
    Benchmarks.MBPP: ReflexionCoTMBPPStrategy,
}


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


class ReflexionCoTAgent(BaseAgent):
    """Reflexion with Chain-of-Thought actor.

    Attributes:
        llm (BaseLLM): The language model used to generate responses.
        benchmark (str): The benchmark.
        reflector (Optional[ReflexionCoTReflector]): An optional reflector module for guided self-reflection.
        testing (bool, optional): Whether to run in testing mode. Defaults to False.
        **strategy_kwargs (Any): Additional keyword arguments for the strategy.

    Methods:
        generate(): Generates a response.
    """

    def __init__(
        self,
        llm: BaseLLM,
        benchmark: str,
        reflector: Optional[ReflexionCoTReflector] = None,
        testing: bool = False,
        **strategy_kwargs: Any,
    ) -> None:
        """Initialization."""
        super().__init__(llm=llm, benchmark=benchmark, testing=testing)

        self.strategy = ReflexionCoTAgent.get_strategy(
            benchmark=self.benchmark,
            llm=self.llm,
            reflector=reflector,
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
        if benchmark not in REFLEXION_COT_FEWSHOTS:
            raise ValueError(
                f"Benchmark '{benchmark}' few-shots not found for ReflexionCoT."
            )

        if fewshot_type not in REFLEXION_COT_BENCHMARK_FEWSHOTS[benchmark]:
            raise ValueError(
                f"Benchmark '{benchmark}' few-shot type not supported for ReflexionCoT."
            )

        benchmark_fewshots = BENCHMARK_FEWSHOTS[benchmark][fewshot_type]

        return {"examples": benchmark_fewshots, **REFLEXION_COT_FEWSHOTS[benchmark]}

    @staticmethod
    def get_prompts(benchmark: str, **kwargs: Any) -> Dict[str, str]:
        """Retrieve the prompt instruction based on the benchmark.

        Args:
            benchmark (str): The benchmark name.
            **kwargs (Any): Additional arguments.

        Returns:
            Dict[str, str]: The prompt instructions.
        """
        if benchmark not in REFLEXION_COT_PROMPTS:
            raise ValueError(
                f"Benchmark '{benchmark}' prompt not found for ReflexionCoT."
            )

        return REFLEXION_COT_PROMPTS[benchmark]

    @staticmethod
    def get_strategy(benchmark: str, **kwargs: Any) -> ReflexionCoTBaseStrategy:
        """Returns an instance of the appropriate ReflexionCoT strategy based on the provided benchmark.

        Args:
            benchmark (str): The benchmark name.
            **kwargs (Any): Additional keyword arguments to pass to
                the strategy's constructor.

        Returns:
            ReflexionCoTBaseStrategy: An instance of the appropriate ReflexionCoT strategy.
        """
        if benchmark not in REFLEXION_COT_STRATEGIES:
            raise ValueError(
                f"Unsupported benchmark: {benchmark} for agent ReflexionCoT"
            )

        strategy = REFLEXION_COT_STRATEGIES[benchmark]
        return strategy(**kwargs)  # type: ignore

    def generate(
        self,
        question: str,
        key: str,
        examples: str = "",
        prompt: str = "",
        reflect_examples: str = "",
        reflect_prompt: str = "",
        reflect_strategy: str = "reflexion",
        additional_keys: Dict[str, str] = {},
        reflect_additional_keys: Dict[str, str] = {},
        fewshot_type: str = "",
        patience: int = 3,
        reset: bool = True,
    ) -> ReflexionCoTOutput:
        """Generates a response based on the provided context, question, and key.

        The `generate` method internally calls reflect (if possible), resets the memory,
        and generates a thought, action, and the observation (Finish).

        Args:
            question (str): The question to answer.
            key (str): The key to evaluate the correctness of the answer.
            examples (str, optional): Fewshot examples. Defaults to "".
            prompt (str, optional): Prompt template string. Defaults to "".
            reflect_examples (str, optional): Reflection fewshot examples. Defaults to "".
            reflect_prompt (str, optional): Reflect prompt template string. Defaults to "".
            reflect_strategy (str): The strategy to use for reflection. Can be one of "last_attempt",
                "reflexion", or "last_attempt_and_reflexion". Defaults to "reflexion".
            additional_keys (Dict[str, str], optional): Additional keys for the prompt. Defaults to {}.
            reflect_additional_keys (Dict[str, str], optional): Additional keys for the reflect prompt. Defaults to {}.
            fewshot_type (str): The type of few-shot examples to use. Defaults to "".
            patience (int, optional): The patience for the agent. Defaults to 3.
            reset (bool, optional): Whether to reset the agent's memory. Defaults to True.

        Returns:
            ReflexionCoTOutput: The output of the agent's response.
        """
        if not prompt or not reflect_prompt or not examples or not reflect_examples:
            if not fewshot_type:
                fewshot_type = REFLEXION_COT_BENCHMARK_FEWSHOTS[self.benchmark][0]  # type: ignore
            fewshots = ReflexionCoTAgent.get_fewshots(
                benchmark=self.benchmark, fewshot_type=fewshot_type
            )
            prompts = ReflexionCoTAgent.get_prompts(benchmark=self.benchmark)
            examples = fewshots["examples"]
            prompt = prompts["prompt"]
            reflect_examples = fewshots["reflect_examples"]
            reflect_prompt = prompts["reflect_prompt"]

        out = self.strategy.generate(
            question=question,
            key=key,
            examples=examples,
            reflect_examples=reflect_examples,
            prompt=prompt,
            reflect_prompt=reflect_prompt,
            reflect_strategy=reflect_strategy,
            additional_keys=additional_keys,
            reflect_additional_keys=reflect_additional_keys,
            patience=patience,
            reset=reset,
        )

        return out


class ReflexionReActAgent(BaseAgent):
    """Reflexion with ReAct actor.

    Attributes:
        llm (BaseLLM): The language model used to generate responses.
        benchmark (str): The benchmark.
        reflector (Optional[ReflexionReActReflector]): An optional reflector module for guided self-reflection. Defaults to None.
        testing (bool, optional): Whether to run in testing mode. Defaults to False.
        **strategy_kwargs (Any): Additional keyword arguments for the strategy.

    Methods:
        generate(): Generates a response.
    """

    def __init__(
        self,
        llm: BaseLLM,
        benchmark: str,
        reflector: Optional[ReflexionReActReflector] = None,
        testing: bool = False,
        **strategy_kwargs: Any,
    ) -> None:
        """Initialization."""
        super().__init__(llm=llm, benchmark=benchmark, testing=testing)

        self.strategy = ReflexionReActAgent.get_strategy(
            benchmark=self.benchmark,
            llm=self.llm,
            reflector=reflector,
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

    def generate(
        self,
        question: str,
        key: str,
        examples: str = "",
        prompt: str = "",
        reflect_examples: str = "",
        reflect_prompt: str = "",
        reflect_strategy: str = "reflexion",
        additional_keys: Dict[str, str] = {},
        reflect_additional_keys: Dict[str, str] = {},
        fewshot_type: str = "",
        patience: int = 3,
        reset: bool = True,
    ) -> ReflexionReActOutput:
        """Processes a given question through ReAct and reflects using Reflexion strategies when possible.

        Iteratively applies the think-act-observe cycle to generate an answer for the question.
        The process continues until the operation is halted based on certain conditions.

        Args:
            question (str): The question to be processed.
            key (str): The answer to the question.
            examples (str, optional): Fewshot examples. Defaults to "".
            prompt (str, optional): Prompt template string. Defaults to "".
            reflect_examples (str, optional): Reflection fewshot examples. Defaults to "".
            reflect_prompt (str, optional): Reflect prompt template string. Defaults to "".
            reflect_strategy (Optional[str]): The reflection strategy. Can be of 3 types. Defaults to "reflexion".
                - "last_attempt": This strategy uses only 'question' and 'scratchpad'. The 'reflections' list is updated with the current scratchpad.
                - "reflexion": This strategy uses all the parameters. It adds a new reflexion generated by the language model to the 'reflections' list.
                - "last_attempt_and_reflexion": This strategy combines the 'last_attempt' and 'reflexion' strategies.
            additional_keys (Dict[str, str], optional): Additional keys for the prompt. Defaults to {}.
            reflect_additional_keys (Dict[str, str], optional): Additional keys for the reflect prompt. Defaults to {}.
            fewshot_type (str): The type of few-shot examples to use. Defaults to "".
            patience (int, optional): The patience for the agent. Defaults to 3.
            reset (bool): Whether to reset the internal state before processing. Defaults to True.

        Returns:
            ReflexionReActOutput: The agent's output.
        """
        if not prompt or not reflect_prompt or not examples or not reflect_examples:
            if not fewshot_type:
                fewshot_type = REFLEXION_REACT_BENCHMARK_FEWSHOTS[self.benchmark][0]  # type: ignore
            fewshots = ReflexionReActAgent.get_fewshots(
                benchmark=self.benchmark, fewshot_type=fewshot_type
            )
            prompts = ReflexionReActAgent.get_prompts(benchmark=self.benchmark)
            examples = fewshots["examples"]
            prompt = prompts["prompt"]
            reflect_examples = fewshots["reflect_examples"]
            reflect_prompt = prompts["reflect_prompt"]

        out = self.strategy.generate(
            question=question,
            key=key,
            examples=examples,
            reflect_examples=reflect_examples,
            prompt=prompt,
            reflect_prompt=reflect_prompt,
            reflect_strategy=reflect_strategy,
            additional_keys=additional_keys,
            reflect_additional_keys=reflect_additional_keys,
            patience=patience,
            reset=reset,
        )

        return out
