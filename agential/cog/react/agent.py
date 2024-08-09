"""ReAct Agent.

Original Paper: https://arxiv.org/abs/2210.03629
Paper Repository: https://github.com/ysymyth/ReAct
"""

from typing import Any, Dict, List

from agential.cog.base.agent import BaseAgent
from agential.cog.react.factory import REACT_BENCHMARK_FEWSHOTS, ReActFactory
from agential.cog.react.output import ReActOutput
from agential.llm.llm import BaseLLM


class ReActAgent(BaseAgent):
    """ReAct agent.

    Attributes:
        llm (BaseLLM): An instance of a language model used for generating initial answers
            and critiques.
        benchmark (str): The benchmark.
        **strategy_kwargs (Any): Additional strategy-specific arguments.
    """

    def __init__(
        self,
        llm: BaseLLM,
        benchmark: str,
        **strategy_kwargs: Any,
    ) -> None:
        """Initialization."""
        super().__init__()
        self.llm = llm
        self.benchmark = benchmark

        self.strategy = ReActFactory().get_strategy(
            benchmark=self.benchmark, llm=self.llm, **strategy_kwargs
        )

    def generate(
        self,
        question: str,
        examples: str = "",
        prompt: str = "",
        additional_keys: Dict[str, str] = {},
        fewshot_type: str = "",
        reset: bool = True,
        **kwargs: Any,
    ) -> List[ReActOutput]:
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
            **kwargs (Any): Additional parameters for flexibility.

        Returns:
            List[ReActOutput]: The list of accumulated output from the ReAct process,
                each ReActOutput consists of a thought, action type/query, observation, answer, and external tool info.
        """
        if not prompt or not examples:
            if not fewshot_type:
                fewshot_type = REACT_BENCHMARK_FEWSHOTS[self.benchmark][0]
            fewshots = ReActFactory.get_fewshots(
                benchmark=self.benchmark, fewshot_type=fewshot_type
            )
            prompts = ReActFactory.get_prompts(benchmark=self.benchmark)
            examples = fewshots["examples"]
            prompt = prompts["prompt"]

        if reset:
            self.reset()

        idx = 1
        out = []
        while not self.strategy.halting_condition(
            idx=idx,
            question=question,
            examples=examples,
            prompt=prompt,
            additional_keys=additional_keys,
            **kwargs,
        ):
            # Think.
            thought = self.strategy.generate(
                question=question,
                examples=examples,
                prompt=prompt,
                additional_keys=additional_keys,
                **kwargs,
            )

            # Act.
            action_type, query = self.strategy.generate_action(
                question=question,
                examples=examples,
                prompt=prompt,
                additional_keys=additional_keys,
                **kwargs,
            )

            # Observe.
            obs, external_tool_info = self.strategy.generate_observation(
                idx=idx, action_type=action_type, query=query
            )

            out.append(
                ReActOutput(
                    **self.strategy.create_output_dict(
                        thought=thought,
                        action_type=action_type,
                        query=query,
                        obs=obs,
                        external_tool_info=external_tool_info,
                    )
                )
            )

            idx += 1

        return out

    def reset(self) -> None:
        """Resets the internal state of the ReAct agent.

        Sets the step number, finished flag, and scratchpad to their initial values.
        """
        self.strategy.reset()
