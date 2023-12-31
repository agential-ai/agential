"""ReAct Agent implementation adapted from LangChain's zero-shot ReAct.

LangChain-adapted Zero-shot ReAct, except the default tool is the wikipedia searcher.
This implementation uses parts of the zero-shot ReAct prompt from langchain-hub, but it's 
structured to match the original paper's implementation. It is open to other tools.

Original Paper: https://arxiv.org/abs/2210.03629
Paper Repository: https://github.com/ysymyth/ReAct
LangChain: https://github.com/langchain-ai/langchain
LangChain ReAct: https://python.langchain.com/docs/modules/agents/agent_types/react
"""
from typing import Any

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders.wikipedia import WikipediaLoader

from discussion_agents.cog.agent.base import BaseAgent
from discussion_agents.cog.prompts.react import (
    INSTRUCTION,
    HOTPOTQA_FEWSHOT_EXAMPLES
)

# TODO: We should also have zero-shot ReAct.
class ReActAgent(BaseAgent):

    llm: Any  # TODO: Why is `LLM` not usable here? 
    i: int = 0  # Count.

    def search(self, query: str):
        docs = WikipediaLoader(
            query=query, 
            load_max_docs=1, 
            doc_content_chars_max=-1
        ).load()
        return docs


    def generate(self, observation: str) -> str:
        """Main method for interacting with zero-shot ReAct agent."""
        prompt_template = (
            INSTRUCTION + 
            HOTPOTQA_FEWSHOT_EXAMPLES + 
            "\n" + 
            "Question: " + 
            "{observation}" +
            "\n" + 
            "Thought {i}: "
        )
        chain = LLMChain(llm=self.llm, prompt=PromptTemplate.from_template(prompt_template))
        thought_action = chain.run(observation=observation, i=self.i).split(f"\nObservation {self.i}:")[0]

        # TODO: Find a way to enforce llm outputs.
        out = prompt_template
        try:
            thought, action = thought_action.strip().split(f"\nAction {self.i}: ")
        except:
            thought = thought_action.strip().split('\n')[0]
            revised_prompt = prompt_template + f"{thought}\n" + "Action {i}: "
            chain = LLMChain(llm=self.llm, prompt=PromptTemplate.from_template(revised_prompt))
            action = chain.run(observation=observation, i=self.i).strip().split("\n")[0]
        out += f"Thought {self.i}: {thought}\n" + f"Action {self.i}: {action}\n"

        if action.lower().startswith("search[") and action.endswith("]"):
            query = action[len("search["):-1]
            docs = self.search(query)
            if not docs:
                obs = f"Could not find {query}."
            else:
                obs = " ".join(docs[0].page_content.replace("\n", "").split(". ")[:5])
            out += f"Observation {self.i}: {obs}\n"

        return out