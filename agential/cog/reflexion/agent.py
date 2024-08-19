"""Reflexion Agent.

Original Paper: https://arxiv.org/abs/2303.11366
Paper Repositories:
    - https://github.com/noahshinn/reflexion-draft
    - https://github.com/noahshinn/reflexion
"""

import re

from typing import Any, Dict, List, Optional, Tuple

from agential.cog.base.agent import BaseAgent
from agential.cog.constants import BENCHMARK_FEWSHOTS, Benchmarks, FewShotType
from agential.cog.reflexion.factory import (
    REFLEXION_REACT_BENCHMARK_FEWSHOTS,
    ReflexionReActFactory,
)
from agential.cog.reflexion.output import (
    ReflexionCoTOutput,
    ReflexionReActOutput,
    ReflexionReActStepOutput,
)
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
from agential.cog.reflexion.reflect import (
    ReflexionCoTReflector,
    ReflexionReActReflector,
)
from agential.cog.reflexion.strategies.base import (
    ReflexionCoTBaseStrategy,
)
from agential.cog.reflexion.strategies.code import (
    ReflexionCoTHEvalStrategy,
    ReflexionCoTMBPPStrategy,
)
from agential.cog.reflexion.strategies.math import (
    ReflexionCoTGSM8KStrategy,
    ReflexionCoTSVAMPStrategy,
    ReflexionCoTTabMWPStrategy,
)
from agential.cog.reflexion.strategies.qa import (
    ReflexionCoTAmbigNQStrategy,
    ReflexionCoTFEVERStrategy,
    ReflexionCoTHotQAStrategy,
    ReflexionCoTTriviaQAStrategy,
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


def parse_action(string: str) -> Tuple[str, str]:
    """Parses an action string into an action type and its argument.

    This method is used in ReAct and Reflexion.

    Args:
        string (str): The action string to be parsed.

    Returns:
        Tuple[str, str]: A tuple containing the action type and argument.
    """
    pattern = r"^(\w+)\[(.+)\]$"
    match = re.match(pattern, string)

    if match:
        action_type = match.group(1)
        argument = match.group(2)
    else:
        action_type = ""
        argument = ""
    return action_type, argument


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
        reset(): Resets the agent's state for a new problem-solving session.
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
        **strategy_kwargs (Any): Additional keyword arguments for the strategy.

    Methods:
        generate(): Generates a response.
        reset(): Resets the agent's state for a new problem-solving session.
    """

    def __init__(
        self,
        llm: BaseLLM,
        benchmark: str,
        reflector: Optional[ReflexionReActReflector] = None,
        **strategy_kwargs: Any,
    ) -> None:
        """Initialization."""
        super().__init__()

        self.llm = llm
        self.benchmark = benchmark

        self.strategy = ReflexionReActFactory().get_strategy(
            benchmark=self.benchmark,
            llm=self.llm,
            reflector=reflector,
            **strategy_kwargs,
        )

    def _generate_react(
        self,
        question: str,
        key: str,
        examples: str,
        reflections: str,
        prompt: str,
        additional_keys: Dict[str, str] = {},
        **kwargs: Any,
    ) -> Tuple[int, bool, List[ReflexionReActStepOutput]]:
        out = []
        step_idx = 1
        self.strategy.reset(no_reflector=True)
        while not self.strategy.react_halting_condition(
            step_idx=step_idx,
            question=question,
            examples=examples,
            reflections=reflections,
            prompt=prompt,
            additional_keys=additional_keys,
            **kwargs,
        ):
            # Think.
            thought = self.strategy.generate(
                question=question,
                examples=examples,
                reflections=reflections,
                prompt=prompt,
                additional_keys=additional_keys,
                **kwargs,
            )

            # Act.
            action_type, query = self.strategy.generate_action(
                question=question,
                examples=examples,
                reflections=reflections,
                prompt=prompt,
                additional_keys=additional_keys,
                **kwargs,
            )

            # Observe.
            is_correct, obs, external_tool_info = self.strategy.generate_observation(
                step_idx=step_idx,
                action_type=action_type,
                query=query,
                key=key,
            )

            out.append(
                ReflexionReActStepOutput(
                    **self.strategy.react_create_output_dict(
                        thought=thought,
                        action_type=action_type,
                        query=query,
                        obs=obs,
                        external_tool_info=external_tool_info,
                        is_correct=is_correct,
                    )
                )
            )

            step_idx += 1

        return step_idx, is_correct, out

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
        **kwargs: Any,
    ) -> List[ReflexionReActOutput]:
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
            **kwargs (Any): Additional keyword arguments for the strategy.

        Returns:
            List[ReflexionReActOutput]: List of ReflexionReActOutput where each ReflexionReActOutput contains the ReAct output and
                the reflections at the end of the trial.
        """
        if not prompt or not reflect_prompt or not examples or not reflect_examples:
            if not fewshot_type:
                fewshot_type = REFLEXION_REACT_BENCHMARK_FEWSHOTS[self.benchmark][0]  # type: ignore
            fewshots = ReflexionReActFactory.get_fewshots(
                benchmark=self.benchmark, fewshot_type=fewshot_type
            )
            prompts = ReflexionReActFactory.get_prompts(benchmark=self.benchmark)
            examples = fewshots["examples"]
            prompt = prompts["prompt"]
            reflect_examples = fewshots["reflect_examples"]
            reflect_prompt = prompts["reflect_prompt"]

        # Reset.
        if reset:
            self.reset()

        idx, step_idx, patience_cnt = 1, 1, 0
        out = []
        while not self.strategy.halting_condition(idx=idx, key=key, **kwargs):
            # Reflect if possible.
            reflections: List[str] = []
            reflections_str = ""
            if self.strategy.reflect_condition(
                step_idx=step_idx,
                reflect_strategy=reflect_strategy,
                question=question,
                examples=examples,
                key=key,
                prompt=prompt,
                additional_keys=additional_keys,
                **kwargs,
            ):
                assert isinstance(reflect_strategy, str)
                reflections, reflections_str = self.strategy.reflect(
                    reflect_strategy=reflect_strategy,
                    question=question,
                    examples=reflect_examples,
                    prompt=reflect_prompt,
                    additional_keys=reflect_additional_keys,
                )

            step_idx, is_correct, react_out = self._generate_react(
                question=question,
                key=key,
                examples=examples,
                reflections=reflections_str,
                prompt=prompt,
                additional_keys=additional_keys,
                **kwargs,
            )

            out.append(
                ReflexionReActOutput(
                    **self.strategy.create_output_dict(
                        react_out=react_out,
                        reflections=reflections,
                    )
                )
            )

            # Increment patience counter.
            if not is_correct:
                patience_cnt += 1
            if patience_cnt == patience:
                break

            idx += 1

        return out

    def reset(self) -> None:
        """Resets the internal state of the ReflexionReAct agent.

        Sets the step number, finished flag, and scratchpad to their initial values.
        """
        self.strategy.reset()
