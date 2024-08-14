"""ReAct Agent strategies for QA."""

from typing import Any, Dict, Tuple

import tiktoken

from langchain_community.docstore.wikipedia import Wikipedia
from tiktoken.core import Encoding

from agential.cog.react.functional import _prompt_agent, parse_qa_action
from agential.cog.react.strategies.general import ReActGeneralStrategy
from agential.llm.llm import BaseLLM, ModelResponse
from agential.utils.docstore import DocstoreExplorer
from agential.utils.parse import remove_newline


class ReActQAStrategy(ReActGeneralStrategy):
    """A strategy class for QA benchmarks using the ReAct agent.

    Attributes:
        llm (BaseLLM): The language model used for generating answers and critiques.
        max_steps (int): The maximum number of steps the agent can take.
        max_tokens (int): The maximum number of tokens allowed for a response.
        enc (Encoding): The encoding used for the language model.
        docstore (DocstoreExplorer): The document store used for searching and looking up information.
        testing (bool): Whether the strategy is in testing mode. Defaults to False.
    """

    def __init__(
        self,
        llm: BaseLLM,
        max_steps: int = 6,
        max_tokens: int = 5000,
        enc: Encoding = tiktoken.encoding_for_model("gpt-3.5-turbo"),
        docstore: DocstoreExplorer = DocstoreExplorer(Wikipedia()),
        testing: bool = False,
    ) -> None:
        """Initialization."""
        super().__init__(llm=llm, max_steps=max_steps, max_tokens=max_tokens, enc=enc, testing=testing)
        self.docstore = docstore

    def generate_action(
        self,
        idx: int,
        scratchpad: str,
        question: str,
        examples: str,
        prompt: str,
        additional_keys: Dict[str, str],
    ) -> Tuple[str, str, str, ModelResponse]:
        """Generates an action based on the provided input parameters.

        Args:
            idx (int): The index of the current action.
            scratchpad (str): The current state of the scratchpad.
            question (str): The question to be answered.
            examples (str): Examples of previous actions and observations.
            prompt (str): The prompt for the language model.
            additional_keys (Dict[str, str]): Additional key-value pairs to be passed to the language model.

        Returns:
            Tuple[str, str, str, ModelResponse]: The updated scratchpad, the action type, the query, and the language model response.
        """
        scratchpad += f"\nAction {idx}: "

        out = _prompt_agent(
            llm=self.llm,
            question=question,
            scratchpad=scratchpad,
            examples=examples,
            max_steps=self.max_steps,
            prompt=prompt,
            additional_keys=additional_keys,
        )
        action = out.choices[0].message.content
        action = remove_newline(action).split("Observation")[0]
        action_type, query = parse_qa_action(action)
        scratchpad += f"{action_type}[{query}]"

        return scratchpad, action_type, query, out

    def generate_observation(
        self, idx: int, scratchpad: str, action_type: str, query: str
    ) -> Tuple[str, str, str, bool, Dict[str, Any]]:
        """Generates an observation based on the provided action type and query.

        Args:
            idx (int): The index of the current observation.
            scratchpad (str): The current state of the scratchpad.
            action_type (str): The type of action performed (e.g. "search", "lookup", "finish").
            query (str): The query for the action.

        Returns:
            Tuple[str, str, str, bool, Dict[str, Any]]: The updated scratchpad, the answer, the observation, a flag indicating if the task is finished, and a dictionary containing external tool information.
        """
        answer = ""
        finished = False
        external_tool_info = {"search_result": "", "lookup_result": ""}

        scratchpad += f"\nObservation {idx}: "
        if action_type.lower() == "finish":
            answer = query
            finished = True
            obs = query
        elif action_type.lower() == "search":
            try:
                search_result = self.docstore.search(query)
                external_tool_info["search_result"] = search_result
                obs = remove_newline(search_result)
            except Exception:
                obs = "Could not find that page, please try again."
        elif action_type.lower() == "lookup":
            try:
                lookup_result = self.docstore.lookup(query)
                external_tool_info["lookup_result"] = lookup_result
                obs = remove_newline(lookup_result)

            except ValueError:
                obs = "The last page Searched was not found, so you cannot Lookup a keyword in it. Please try one of the similar pages given."
        else:
            obs = "Invalid Action. Valid Actions are Lookup[<topic>] Search[<topic>] and Finish[<answer>]."
        scratchpad += obs

        return scratchpad, answer, obs, finished, external_tool_info


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
