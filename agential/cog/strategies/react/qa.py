"""ReAct Agent strategies for QA."""

from typing import Dict
import tiktoken
from tiktoken.core import Encoding

from langchain.agents.react.base import DocstoreExplorer
from langchain_community.docstore.wikipedia import Wikipedia
from langchain_core.language_models.chat_models import BaseChatModel
from agential.cog.strategies.react.base import ReActBaseStrategy

# max_steps: int = 6,

class ReActQAStrategy(ReActBaseStrategy):
    """A strategy class for QA benchmarks using the ReAct agent.

    Attributes:
        llm (BaseChatModel): The language model used for generating answers and critiques.
    """

    def __init__(
        self,
        llm: BaseChatModel,
        max_tokens: int = 3896,
        docstore: DocstoreExplorer = DocstoreExplorer(Wikipedia()),
        enc: Encoding = tiktoken.encoding_for_model("gpt-3.5-turbo"),
    ) -> None:
        """Initialization."""
        super().__init__(llm)
        self.max_tokens = max_tokens
        self.docstore = docstore
        self.enc = enc

        self._step_n = 1
        self._finished = False

    def generate(self, question: str, examples: str, prompt: str, additional_keys: Dict[str, str]) -> str:
        return super().generate(question, examples, prompt, additional_keys)
    
    def generate_action(self, question: str, examples: str, answer: str, prompt: str, additional_keys: Dict[str, str]) -> str:
        return super().generate_action(question, examples, answer, prompt, additional_keys)
    
    def generate_observation(self, question: str, examples: str, answer: str, prompt: str, additional_keys: Dict[str, str]) -> str:
        return super().generate_observation(question, examples, answer, prompt, additional_keys)
    
    def create_output_dict(self, answer: str, critique: str) -> Dict[str, str]:
        return super().create_output_dict(answer, critique)
    
    def halting_condition(self) -> bool:
        return super().halting_condition()

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
