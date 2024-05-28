"""ReAct Agent.

Original Paper: https://arxiv.org/abs/2210.03629
Paper Repository: https://github.com/ysymyth/ReAct
"""

from typing import Any, Dict, List

from langchain_core.language_models.chat_models import BaseChatModel

from agential.cog.agent.base import BaseAgent
from agential.cog.prompts.agents.react import (
    REACT_INSTRUCTION_HOTPOTQA,
)
from agential.cog.prompts.benchmarks.hotpotqa import HOTPOTQA_FEWSHOT_EXAMPLES_REACT
from agential.cog.strategies.strategy_factory import ReActStrategyFactory


class ReActAgent(BaseAgent):
    """ReAct agent from the original paper.

    Implements the ReAct algorithm as described in the original paper.
    This agent uses a language model to iteratively process a question
    through a sequence of think-act-observe steps, utilizing a document
    store for information retrieval.

    Attributes:
        llm (BaseChatModel): The language model used by the agent.
        max_steps (int): Maximum number of steps to process the question.
        max_tokens (int): Maximum token limit for the language model.
        docstore (DocstoreExplorer): Document store for information retrieval.
        enc (Encoding): Encoder for calculating token lengths.
    """

    def __init__(
        self,
        llm: BaseChatModel,
        mode: Dict[str, str],
        **strategy_kwargs: Dict[str, Any],
    ) -> None:
        """Initialization."""
        super().__init__()
        self.llm = llm
        self.mode = mode

        self.strategy = ReActStrategyFactory().get_strategy(
            mode=self.mode, llm=self.llm, **strategy_kwargs
        )

    def generate(
        self,
        question: str,
        examples: str = HOTPOTQA_FEWSHOT_EXAMPLES_REACT,
        prompt: str = REACT_INSTRUCTION_HOTPOTQA,
        additional_keys: Dict[str, str] = {},
        reset: bool = True,
        **kwargs: Dict[str, Any],
    ) -> List[Dict[str, str]]:
        """Processes a given question through ReAct.

        Iteratively applies the think-act-observe cycle to generate an answer for the question.
        The process continues until the operation is halted based on certain conditions.

        Args:
            question (str): The question to be processed.
            examples (str, optional): Fewshot examples. Defaults to HOTPOTQA_FEWSHOT_EXAMPLES_REACT.
            prompt (str, optional): Prompt template string. Defaults to REACT_INSTRUCTION_HOTPOTQA. Must include question,
                scratchpad, examples, and max_steps.
            additional_keys (Dict[str, str]): Additional keys to format the prompt. Defaults to {}.
            reset (bool, optional): Whether to reset the internal state before processing. Defaults to True.
            **kwargs (Dict[str, Any]): Additional parameters for flexibility.

        Returns:
            List[Dict[str, str]]: The list of accumulated output from the ReAct process,
                each dictionary consists of a thought, action type/query, and observation.
        """
        if reset:
            self.reset()

        idx = 1
        out = []
        while not self.strategy.halting_condition(
            idx=idx, question=question, examples=examples, prompt=prompt, **kwargs
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
            obs = self.strategy.generate_observation(
                idx=idx, action_type=action_type, query=query
            )

            out.append(
                self.strategy.create_output_dict(
                    thought=thought, action_type=action_type, query=query, obs=obs
                )
            )

            idx += 1

        return out

    def reset(self) -> None:
        """Resets the internal state of the ReAct agent.

        Sets the step number, finished flag, and scratchpad to their initial values.
        """
        self.strategy.reset()
