"""ReAct Agent strategies for QA."""

from typing import Dict, Tuple
import tiktoken
from tiktoken.core import Encoding

from langchain.agents.react.base import DocstoreExplorer
from langchain_community.docstore.wikipedia import Wikipedia
from langchain_core.language_models.chat_models import BaseChatModel
from agential.cog.strategies.react.base import ReActBaseStrategy
from agential.cog.functional.react import _is_halted, _prompt_agent
from agential.utils.parse import parse_action, remove_newline

# max_steps: int = 6,

class ReActQAStrategy(ReActBaseStrategy):
    """A strategy class for QA benchmarks using the ReAct agent.

    Attributes:
        llm (BaseChatModel): The language model used for generating answers and critiques.
    """

    def __init__(
        self,
        llm: BaseChatModel,
        max_steps: int = 6,
        max_tokens: int = 3896,
        docstore: DocstoreExplorer = DocstoreExplorer(Wikipedia()),
        enc: Encoding = tiktoken.encoding_for_model("gpt-3.5-turbo"),
    ) -> None:
        """Initialization."""
        super().__init__(llm)
        self.max_steps = max_steps
        self.max_tokens = max_tokens
        self.docstore = docstore
        self.enc = enc

        self._scratchpad = ""
        self._finished = False

    def generate(self, question: str, examples: str, prompt: str, additional_keys: Dict[str, str]) -> str:
        self._scratchpad += "\nThought:"
        thought = _prompt_agent(
                llm=self.llm,
                question=question,
                scratchpad=self._scratchpad,
                examples=examples,
                max_steps=self.max_steps,
                additional_keys=additional_keys,
                prompt=prompt,
        ).split("Action")[0]
        self._scratchpad += " " + thought

        return thought

    def generate_action(self, question: str, examples: str, prompt: str, additional_keys: Dict[str, str]) -> Tuple[str, str]:
        self._scratchpad += "\nAction:"
        action = _prompt_agent(
            llm=self.llm,
            question=question,
            scratchpad=self._scratchpad,
            examples=examples,
            max_steps=self.max_steps,
            additional_keys=additional_keys,
            prompt=prompt,
        ).split("Observation")[0]
        self._scratchpad += " " + action
        action_type, query = parse_action(action)

        return action_type, query

    def generate_observation(self, idx: int, action_type: str, query: str) -> str:
        self._scratchpad += f"\nObservation {idx}: "
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
        self._scratchpad += obs

        return obs

    def create_output_dict(self, thought: str, action: str, obs: str) -> Dict[str, str]:
        return {}

    def halting_condition(self, action_type: str) -> bool:
        return self._finished

    def reset(self) -> None:
        self._scratchpad = ""
        self._finished = False


class ReActHotQAStrategy(ReActQAStrategy):
    """A strategy class for the HotpotQA benchmark using the ReAct agent."""

    pass


class ReActTriviaQAStrategy(ReActQAStrategy):
    """A strategy class for the TriviaQA benchmark using the ReAct agent."""

    pass


class ReActAmbigNQStrategy(ReActQAStrategy):
    """A strategy class for the AmbigNQ benchmark using the ReAct agent."""

    pass


class ReActFEVERStrategy(ReActQAStrategy):
    """A strategy class for the FEVER benchmark using the ReAct agent."""

    pass
