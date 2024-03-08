"""ReAct Agent implementation and LangChain's zero-shot ReAct.

This includes the original ReAct agent implementation and the LangChain-adapted
Zero-shot ReAct, with a wikipedia searcher default tool.

Original Paper: https://arxiv.org/abs/2210.03629
Paper Repository: https://github.com/ysymyth/ReAct
LangChain: https://github.com/langchain-ai/langchain
LangChain ReAct: https://python.langchain.com/docs/modules/agents/agent_types/react
"""
# Standard library imports
from typing import Any, Dict, List, Optional

# Third-party library imports
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain.agents.react.base import DocstoreExplorer
from langchain_community.docstore.wikipedia import Wikipedia
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import BaseTool, tool
from tiktoken.core import Encoding
import tiktoken

# Local imports
from discussion_agents.cog.agent.base import BaseAgent
from discussion_agents.cog.functional.react import (
    _check_keyword,
    _is_halted,
    _process_ob,
    _prompt_agent,
)
from discussion_agents.cog.modules.memory.react import ReActMemory
from discussion_agents.utils.parse import parse_action, remove_newline
from discussion_agents.cog.prompts.react import (
    REACT_ALFWORLD_INSTRUCTION,
    REACT_INSTRUCTION_FEVER,
    REACT_INSTRUCTION_HOTPOTQA,
)


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
        benchmark_type (str): Specifies the benchmark type used for selecting the appropriate examples. Acceptable values are limited to 'HotpotQA' or 'FEVER' or 'Alfworld'.

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
        """Initialize the ReActAgent.

        Args:
            llm (BaseChatModel): The language model used by the agent.
            memory (Optional[ReActMemory], optional): The memory module. Defaults to None.
            max_steps (int, optional): Maximum number of steps to process the question. Defaults to 6.
            max_tokens (int, optional): Maximum token limit for the language model. Defaults to 3896.
            docstore (DocstoreExplorer, optional): Document store for information retrieval. Defaults to DocstoreExplorer(Wikipedia()).
            enc (Encoding, optional): Encoder for calculating token lengths. Defaults to tiktoken.encoding_for_model("gpt-3.5-turbo").
        """
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
        examples: str = "",
        env: Any = None,
        prompt_template: str = "",
    ) -> str:
        """Processes a given question through ReAct.

        Iteratively applies the think-act-observe cycle to generate an answer for the question.
        The process continues until the operation is halted based on certain conditions.

        Args:
            question (str): The question to be processed.
            reset (bool, optional): Whether to reset the internal state before processing. Defaults to True.
            prompt_template (str): The template for benchmark.
            examples (str): The example of text generated.
            env (Alfworld_environment): The variable to interact with Alfworld.

        Returns:
            str: The accumulated output from the ReAct process.
        """

        if reset:
            self.reset()

        output = ""
        scratchpad = self.memory.load_memories()["scratchpad"]

        while not _is_halted(
            finished=self._finished,
            step_n=self._step_n,
            max_steps=self.max_steps,
            question=question,
            scratchpad=scratchpad,
            max_tokens=self.max_tokens,
            enc=self.enc,
            examples=examples,
            prompt_template=prompt_template,
        ):
            output += self.execute_step(question, examples, prompt_template, env)
            scratchpad = self.memory.load_memories()["scratchpad"]

        return output

    def execute_step(
        self, question: str, examples: str, prompt_template: str, env: Any
    ) -> str:
        """Execute a single step of the ReAct process.

        Args:
            question (str): The question to be processed.
            examples (str): The example of text generated.
            prompt_template (str): The template for benchmark.
            env (Any): The variable to interact with Alfworld.

        Returns:
            str: The output from the current step.
        """
        step_boolean = _check_keyword(example=examples)
        scratchpad = self.memory.load_memories()["scratchpad"]
        if step_boolean[0]:
            thought = self.think_step(question, examples, prompt_template, scratchpad)
            out = f"\n{thought}"
        else:
            return ""

        if step_boolean[1]:
            action = self.act_step(question, examples, prompt_template, scratchpad)
            out += f"\n{action}"
            action = action.replace(">", "").strip()
            if "think" not in action:
                action = action.replace(" in ", " in/on ")
        else:
            return out

        if step_boolean[2]:
            if step_boolean[3]:
                self.observe_step_env(action, env)
            else:
                self.observe_step_action(action)
                out += "\n" + self.memory.load_memories()["scratchpad"].split("\n")[-1]
                self._step_n += 1
        else:
            return out

        return out

    def think_step(
        self, question: str, examples: str, prompt_template: str, scratchpad: Any
    ) -> str:
        """Perform the "think" step of the ReAct process.

        Args:
            question (str): The question to be processed.
            examples (str): The example of text generated.
            prompt_template (str): The template for benchmark.
            scratchpad (Any): The current state of the agent's memory.

        Returns:
            str: The output from the "think" step.
        """
        self.memory.add_memories("\nThought:")
        thought = _prompt_agent(
            llm=self.llm,
            question=question,
            scratchpad=scratchpad,
            examples=examples,
            prompt_template=prompt_template,
        ).strip()
        self.memory.add_memories(" " + thought)
        return thought

    def act_step(
        self, question: str, examples: str, prompt_template: str, scratchpad: Any
    ) -> str:
        """Perform the "act" step of the ReAct process.

        Args:
            question (str): The question to be processed.
            examples (str): The example of text generated.
            prompt_template (str): The template for benchmark.
            scratchpad (Any): The current state of the agent's memory.

        Returns:
            str: The output from the "act" step.
        """
        self.memory.add_memories(f"\nAction {self._step_n}:")
        action = _prompt_agent(
            llm=self.llm,
            question=question,
            scratchpad=scratchpad,
            examples=examples,
            prompt_template=prompt_template,
        ).strip()
        # Prepare the action
        action = action.replace(">", "").strip()
        if "think" not in action:
            action = action.replace(" in ", " in/on ")
        self.memory.add_memories(" " + action)
        return action

    def observe_step_env(self, action: Any, env: Any) -> None:
        """Perform the "observe" step for an environment interaction.

        Args:
            action (Any): The action to be performed in the environment.
            env (Any): The environment to interact with.
        """
        self.memory.add_memories(f"\nObservation {self._step_n}: ")
        observation, _, done, info = env.step([action])
        observation, done = _process_ob(observation[0]), done[0]
        if done:
            self._finished = True
        if "think:" in action:
            observation = "OK."
        self.memory.add_memories(" " + observation)

    def observe_step_action(self, action: Any) -> None:
        """Perform the "observe" step for a non-environment action.

        Args:
            action (Any): The action to be performed.
        """
        self.memory.add_memories(f"\nObservation {self._step_n}: ")
        action_type, query = parse_action(action)

        if action_type.lower() == "finish":
            self.handle_finish_action(query)
        elif action_type.lower() == "search":
            self.handle_search_action(query)
        elif action_type.lower() == "lookup":
            self.handle_lookup_action(query)
        else:
            self.memory.add_memories("Your default action here")

    def handle_finish_action(self, query: str) -> None:
        """Handle the "finish" action.

        Args:
            query (str): The answer to the question.
        """
        self._answer = query
        self._finished = True
        self.memory.add_memories(query)

    def handle_search_action(self, query: str) -> None:
        """Handle the "search" action.

        Args:
            query (str): The search query.
        """
        try:
            self.memory.add_memories(remove_newline(self.docstore.search(query)))
        except Exception:
            self.memory.add_memories("Could not find that page, please try again.")

    def handle_lookup_action(self, query: str) -> None:
        """Handle the "lookup" action.

        Args:
            query (str): The query for looking up information.
        """
        try:
            self.memory.add_memories(remove_newline(self.docstore.lookup(query)))
        except ValueError:
            self.memory.add_memories(
                "The last page Searched was not found, so you cannot Lookup a keyword in it. Please try one of the similar pages given."
            )

    def retrieve(self) -> Dict[str, Any]:
        """Retrieve the current state of the agent's memory.

        Returns:
            Dict[str, Any]: The current state of the agent's memory.
        """
        return self.memory.load_memories()

    def reset(self) -> None:
        """Reset the internal state of the ReAct agent.

        Sets the step number, finished flag, and scratchpad to their initial values.
        """
        self._step_n = 1
        self._finished = False
        self.memory.clear()


@tool
def search(query: str) -> str:
    """Search Wikipedia given a query.

    Args:
        query (str): The search query.

    Returns:
        str: The search results from Wikipedia.
    """
    docstore = DocstoreExplorer(Wikipedia())
    return docstore.search(query)


class ZeroShotReActAgent(BaseAgent):
    """The Zero-Shot ReAct Agent class adapted from LangChain.

    Attributes:
        llm (Any): An attribute for a language model or a similar interface. The exact type is to be determined.
        tools (List[BaseTool]): A list of tools that the agent can use to interact or perform tasks.
        prompt (str, optional): An initial prompt for the agent. If not provided, a default prompt is fetched from a specified hub.

    See: https://github.com/langchain-ai/langchain/tree/master/libs/langchain/langchain/agents/react
    """

    def __init__(
        self,
        llm: Any,
        tools: List[BaseTool] = [],
        # tools: Optional[List[BaseTool]] = None,
        prompt: Optional[str] = None,
    ) -> None:
        """Initialization."""
        super().__init__()
        # if tools is None:
        # tools = []
        self.llm = llm  # TODO: Why is `LLM` not usable here?
        self.tools = tools
        self.tools.append(search)  # type: ignore
        prompt = hub.pull("hwchase17/react") if not prompt else prompt
        self.prompt = prompt
        if self.llm and self.tools and self.prompt:
            agent = create_react_agent(llm, tools, prompt)  # type: ignore
            agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)  # type: ignore
            self.agent = agent_executor

    def generate(self, observation_dict: Dict[str, str]) -> str:
        """Generates a response based on the provided observation dictionary.

        This method wraps around the `AgentExecutor`'s `invoke` method.

        Args:
            observation_dict (Dict[str, str]): A dictionary containing observation data.

        Returns:
            str: The generated response.
        """
        return self.agent.invoke(observation_dict)  # type: ignore