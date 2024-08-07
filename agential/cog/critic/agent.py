"""CRITIC Agent.

Original Paper: https://arxiv.org/pdf/2305.11738
Paper Repository: https://github.com/microsoft/ProphetNet/tree/master/CRITIC
"""

from typing import Any, Dict, List

from langchain_core.language_models.chat_models import BaseChatModel

from agential.base.agent import BaseAgent
from agential.cog.constants import FewShotType
from agential.cog.critic.factory import CRITIC_BENCHMARK_FEWSHOTS, CriticFactory
from agential.cog.critic.output import CriticOutput


class CriticAgent(BaseAgent):
    """CRITIC Agent.

    Attributes:
        llm (BaseChatModel): An instance of a language model used for generating initial answers
            and critiques.
        benchmark (str): The benchmark.
        **strategy_kwargs (Any): Additional strategy-specific arguments.
    """

    def __init__(
        self,
        llm: BaseChatModel,
        benchmark: str,
        **strategy_kwargs: Any,
    ) -> None:
        """Initialization."""
        super().__init__()

        self.llm = llm
        self.benchmark = benchmark

        self.strategy = CriticFactory().get_strategy(
            benchmark=self.benchmark, llm=self.llm, **strategy_kwargs
        )

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
        **kwargs: Any,
    ) -> List[CriticOutput]:
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
            **kwargs (Any): Additional parameters for flexibility.

        Returns:
            List[CriticOutput]: A list of CriticOutput instances where each CriticOutput instance contains the "answer", "critique", and "external_tool_info".
        """
        if not prompt or not critique_prompt or not examples or not critique_examples:
            if not fewshot_type:
                fewshot_type = CRITIC_BENCHMARK_FEWSHOTS[self.benchmark][0]
            fewshots = CriticFactory.get_fewshots(
                benchmark=self.benchmark, fewshot_type=fewshot_type, use_tool=use_tool
            )
            prompts = CriticFactory.get_prompts(
                benchmark=self.benchmark, use_tool=use_tool
            )
            examples = fewshots["examples"]
            prompt = prompts["prompt"]
            critique_examples = fewshots["critique_examples"]
            critique_prompt = prompts["critique_prompt"]

        if reset:
            self.reset()

        out = []

        # Initial answer generation.
        answer = self.strategy.generate(question, examples, prompt, additional_keys)

        critique = ""
        for idx in range(max_interactions):
            critique, external_tool_info = self.strategy.generate_critique(
                idx=idx,
                question=question,
                examples=critique_examples,
                answer=answer,
                critique=critique,
                prompt=critique_prompt,
                additional_keys=critique_additional_keys,
                use_tool=use_tool,
                max_interactions=max_interactions,
                **kwargs,
            )

            out.append(
                CriticOutput(
                    **self.strategy.create_output_dict(
                        answer, critique, external_tool_info
                    )
                )
            )

            if self.strategy.halting_condition():
                break

            # Update answer for the next iteration.
            answer = self.strategy.update_answer_based_on_critique(
                question=question,
                examples=critique_examples,
                answer=answer,
                critique=critique,
                prompt=critique_prompt,
                additional_keys=critique_additional_keys,
                external_tool_info=external_tool_info,
                **kwargs,
            )

        return out

    def reset(self) -> None:
        """Resets the CRITIC Agent's internal state."""
        self.strategy.reset()
