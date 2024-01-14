"""ReAct Agent implementation and LangChain's zero-shot ReAct.

This includes the original ReAct agent implementation and the LangChain-adapted
Zero-shot ReAct, with a wikipedia searcher default tool.

Original Paper: https://arxiv.org/abs/2210.03629
Paper Repository: https://github.com/ysymyth/ReAct
LangChain: https://github.com/langchain-ai/langchain
LangChain ReAct: https://python.langchain.com/docs/modules/agents/agent_types/react
"""
from typing import Any, Dict, List, Optional

import tiktoken
from tiktoken.core import Encoding

from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain_community.docstore.wikipedia import Wikipedia
from langchain.agents.react.base import DocstoreExplorer
from langchain_core.tools import BaseTool, tool

from discussion_agents.cog.agent.base import BaseAgent
from discussion_agents.utils.parse import parse_action
from discussion_agents.cog.functional.react import _is_halted, react_think, react_act, react_observe

class ReActAgent(BaseAgent):
    """ReAct agent from the original paper.

    This agent has 2 methods: `search` and `generate`. It does not
    have any memory, planning, reflecting, or scoring capabilities.
    Given a question, this agent, equipped with Wikipedia search,
    attempts to answer the question in, a maximum of, 7 steps. Each step
    is a thought-action-observation sequence.

    Available actions are:
        - Search[], search for relevant info on Wikipedia (5 sentences)
        - Lookup[], lookup keywords in Wikipedia search
        - Finish[], finish task

    Note:
        By default, HOTPOTQA_FEWSHOT_EXAMPLES are used as fewshot context examples.
        You have the option to provide your own fewshot examples in the `generate` method.

    Attributes:
        llm (LLM): An instance of a language model used for processing and generating content.

    See: https://github.com/ysymyth/ReAct
    """
    def __init__(
        self, 
        llm: Any, 
        max_steps: int = 6, 
        max_tokens: int = 3896,
        docstore: Optional[DocstoreExplorer] = DocstoreExplorer(Wikipedia()),
        enc: Optional[Encoding] = tiktoken.encoding_for_model("gpt-3.5-turbo")
    ) -> None:
        """Initialization."""
        super().__init__()
        self.llm = llm
        self.max_steps = max_steps
        self.max_tokens = max_tokens
        self.docstore = docstore
        self.enc = enc

        # Internal variables.
        self.__step_n = 1  #: :meta private:
        self.__finished = False  #: :meta private:
        self.__scratchpad: str = ""  #: :meta private:

    def generate(self, question: str, reset: bool = True) -> str:
        if reset:
            self.reset()
        
        out = ""
        while not _is_halted(
            finished=self.__finished,
            step_n=self.__step_n, 
            max_steps=self.max_steps, 
            question=question, 
            scratchpad=self.__scratchpad, 
            max_tokens=self.max_tokens, 
            enc=self.enc
        ):
            # Think.
            self.__scratchpad = react_think(
                llm=self.llm, 
                question=question, 
                scratchpad=self.__scratchpad
            )
            out += "\n" + self.__scratchpad.split('\n')[-1]
            
            # Act.
            self.__scratchpad, action = react_act(
                llm=self.llm, 
                question=question, 
                scratchpad=self.__scratchpad
            )
            action_type, query = parse_action(action)
            out += "\n" + self.__scratchpad.split('\n')[-1]

            # Observe.
            observation = react_observe(
                action_type=action_type, 
                query=query, 
                scratchpad=self.__scratchpad, 
                step_n=self.__step_n, 
                docstore=self.docstore
            )
            self.__scratchpad = observation["scratchpad"]
            self.__step_n = observation["step_n"]
            self.__finished = observation["finished"]
            out += "\n" + self.__scratchpad.split('\n')[-1]

        return out


    def reset(self) -> None:
        self.__step_n = 1
        self.__finished = False
        self.__scratchpad: str = ""


@tool
def search(query: str) -> str:
    """Searches Wikipedia given query."""
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
    def __init__(self, llm: Any, tools: Optional[List[BaseTool]] = [], prompt: Optional[str] = None) -> None:
        """Initialization."""
        super().__init__()
        self.llm = llm  # TODO: Why is `LLM` not usable here?
        self.tools = tools
        self.tools.append(search)
        prompt = hub.pull("hwchase17/react") if not prompt else prompt
        self.prompt = prompt
        if self.llm and self.tools and self.prompt:
            agent = create_react_agent(llm, tools, prompt)
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
