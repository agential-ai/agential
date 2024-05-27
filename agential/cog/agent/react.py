"""ReAct Agent.

Original Paper: https://arxiv.org/abs/2210.03629
Paper Repository: https://github.com/ysymyth/ReAct
"""

from typing import Any, Dict, List, Optional

import tiktoken

from langchain.agents.react.base import DocstoreExplorer
from langchain_community.docstore.wikipedia import Wikipedia
from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import BaseModel, Field
from tiktoken.core import Encoding

from agential.cog.agent.base import BaseAgent
from agential.cog.functional.react import _is_halted, _prompt_agent
from agential.cog.modules.memory.react import ReActMemory
from agential.cog.prompts.benchmarks.hotpotqa import HOTPOTQA_FEWSHOT_EXAMPLES_REACT
from agential.cog.prompts.agents.react import (
    REACT_INSTRUCTION_HOTPOTQA,
)
from agential.utils.parse import parse_action, remove_newline


class ReActOutput(BaseModel):
    """The output of the ReAct agent.

    Attributes:
        thought (str): The thought generated by the agent.
        action (str): The action taken by the agent.
        observation (str): The observation made by the agent.
    """

    thought: str = Field(..., description="The thought generated by the agent.")
    action: str = Field(..., description="The action taken by the agent.")
    observation: str = Field(..., description="The observation made by the agent.")


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

    See: https://github.com/ysymyth/ReAct
    """

    def __init__(
        self,
        llm: BaseChatModel,
        memory: Optional[ReActMemory] = None,
        max_steps: int = 6,
        max_tokens: int = 3896,
        docstore: DocstoreExplorer = DocstoreExplorer(Wikipedia()),
        enc: Encoding = tiktoken.encoding_for_model("gpt-3.5-turbo"),
    ) -> None:
        """Initialization."""
        super().__init__()
        self.llm = llm

        if not memory:
            self.memory = ReActMemory()
        else:
            self.memory = memory

        self.max_steps = max_steps
        self.max_tokens = max_tokens
        self.docstore = docstore
        self.enc = enc

        # Internal variables.
        self._step_n = 1  #: :meta private:
        self._finished = False  #: :meta private:

    def generate(
        self,
        question: str,
        reset: bool = True,
        examples: str = HOTPOTQA_FEWSHOT_EXAMPLES_REACT,
        prompt: str = REACT_INSTRUCTION_HOTPOTQA,
    ) -> List[ReActOutput]:
        """Processes a given question through ReAct.

        Iteratively applies the think-act-observe cycle to generate an answer for the question.
        The process continues until the operation is halted based on certain conditions.

        Args:
            question (str): The question to be processed.
            reset (bool, optional): Whether to reset the internal state before processing. Defaults to True.
            examples (str, optional): Fewshot examples. Defaults to HOTPOTQA_FEWSHOT_EXAMPLES_REACT.
            prompt (str, optional): Prompt template string. Defaults to REACT_INSTRUCTION_HOTPOTQA. Must include question,
                scratchpad, examples, and max_steps.

        Returns:
            List[Tuple[str, str, str]]: The list of accumulated output from the ReAct process,
                each tuple consists of a thought-action-observation triplet.
        """
        if reset:
            self.reset()

        out = []
        while not _is_halted(
            finished=self._finished,
            step_n=self._step_n,
            question=question,
            scratchpad=self.memory.load_memories()["scratchpad"],
            examples=examples,
            max_steps=self.max_steps,
            max_tokens=self.max_tokens,
            enc=self.enc,
            prompt=prompt,
        ):
            # Think.
            self.memory.add_memories("\nThought:")
            thought = _prompt_agent(
                llm=self.llm,
                question=question,
                scratchpad=self.memory.load_memories()["scratchpad"],
                examples=examples,
                max_steps=self.max_steps,
                prompt=prompt,
            ).split("Action")[0]
            self.memory.add_memories(" " + thought)

            # Act.
            self.memory.add_memories("\nAction:")
            action = _prompt_agent(
                llm=self.llm,
                question=question,
                scratchpad=self.memory.load_memories()["scratchpad"],
                examples=examples,
                max_steps=self.max_steps,
                prompt=prompt,
            ).split("Observation")[0]
            self.memory.add_memories(" " + action)
            action_type, query = parse_action(action)

            # Observe.
            self.memory.add_memories(f"\nObservation {self._step_n}: ")
            if action_type.lower() == "finish":
                self._answer = query
                self._finished = True
                obs = query
            elif action_type.lower() == "search":
                try:
                    obs = remove_newline(self.docstore.search(query))
                except Exception:
                    obs = "Could not find that page, please try again."
            elif action_type.lower() == "lookup":
                try:
                    obs = remove_newline(self.docstore.lookup(query))
                except ValueError:
                    obs = "The last page Searched was not found, so you cannot Lookup a keyword in it. Please try one of the similar pages given."
            else:
                obs = "Invalid Action. Valid Actions are Lookup[<topic>] Search[<topic>] and Finish[<answer>]."
            self.memory.add_memories(obs)

            out.append(
                ReActOutput(
                    thought=f"Thought: {thought}",
                    action=f"Action: {action}",
                    observation=f"Observation {self._step_n}: {obs}",
                )
            )

            self._step_n += 1

        return out

    def retrieve(self) -> Dict[str, Any]:
        """Retrieves the current state of the agent's memory.

        Returns:
            Dict[str, Any]: The current state of the agent's memory.
        """
        return self.memory.load_memories()

    def reset(self) -> None:
        """Resets the internal state of the ReAct agent.

        Sets the step number, finished flag, and scratchpad to their initial values.
        """
        self._step_n = 1
        self._finished = False
        self.memory.clear()
