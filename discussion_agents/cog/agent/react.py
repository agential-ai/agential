"""ReAct Agent implementation adapted from LangChain's zero-shot ReAct.

LangChain-adapted Zero-shot ReAct, except the default tool is the wikipedia searcher.
This implementation uses parts of the zero-shot ReAct prompt from langchain-hub, but it's 
structured to match the original paper's implementation. It is open to other tools.

Original Paper: https://arxiv.org/abs/2210.03629
Paper Repository: https://github.com/ysymyth/ReAct
LangChain: https://github.com/langchain-ai/langchain
LangChain ReAct: https://python.langchain.com/docs/modules/agents/agent_types/react
"""
from typing import Any, Optional

import requests
from bs4 import BeautifulSoup

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders.wikipedia import WikipediaLoader

from discussion_agents.utils.parse import clean_str, get_page_obs, construct_lookup_list
from discussion_agents.cog.agent.base import BaseAgent
from discussion_agents.cog.prompts.react import (
    INSTRUCTION,
    HOTPOTQA_FEWSHOT_EXAMPLES
)

# TODO: We should also have zero-shot ReAct.
class ReActAgent(BaseAgent):

    llm: Any  # TODO: Why is `LLM` not usable here? 

    i: int = 0  # Count.
    page: str = ""
    result_titles: list = []
    lookup_keyword: str = ""
    lookup_list: list = []
    lookup_cnt: int = 0

    def search(self, query: str):
        docs = WikipediaLoader(
            query=query, 
            load_max_docs=1, 
            doc_content_chars_max=-1
        ).load()
        return docs

    def search_step(self, entity: str, k: Optional[int] = 5):
        entity_ = entity.replace(" ", "+")
        search_url = f"https://en.wikipedia.org/w/index.php?search={entity_}"
        response_text = requests.get(search_url).text
        soup = BeautifulSoup(response_text, features="html.parser")
        result_divs = soup.find_all("div", {"class": "mw-search-result-heading"})
        if result_divs:  # Mismatch.
            self.result_titles = [clean_str(div.get_text().strip()) for div in result_divs]
            obs = f"Could not find {entity}. Similar: {self.result_titles[:5]}."
        else:
            page = [p.get_text().strip() for p in soup.find_all("p") + soup.find_all("ul")]
            if any("may refer to:" in p for p in page):
                obs = self.search_step("[" + entity + "]")
            else:
                self.page = ""
                for p in page:
                    if len(p.split(" ")) > 2:
                        self.page += clean_str(p)
                        if not p.endswith("\n"):
                            self.page += "\n"
                obs = get_page_obs(self.page, k=k)
                self.lookup_keyword = self.lookup_list = self.lookup_cnt = None
        return obs

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
        # TODO: Find a way to enforce llm outputs.

        # Generate thought and action.
        chain = LLMChain(llm=self.llm, prompt=PromptTemplate.from_template(prompt_template))
        thought_action = chain.run(observation=observation, i=self.i).split(f"\nObservation {self.i}:")[0]

        out = prompt_template
        try:
            thought, action = thought_action.strip().split(f"\nAction {self.i}: ")
        except:
            thought = thought_action.strip().split('\n')[0]
            revised_prompt = prompt_template + f"{thought}\n" + "Action {i}: "
            chain = LLMChain(llm=self.llm, prompt=PromptTemplate.from_template(revised_prompt))
            action = chain.run(observation=observation, i=self.i).strip().split("\n")[0]
        out += f"Thought {self.i}: {thought}\n" + f"Action {self.i}: {action}\n"

        # Execute action and get observation.
        if action.lower().startswith("search[") and action.endswith("]"):
            query = action[len("search["):-1]
            obs = self.search_step(query)
            out += f"Observation {self.i}: {obs}\n"
        elif action.lower().startswith("lookup[") and action.endswith("]"):
            keyword = action[len("lookup["):-1]

            # Reset lookup.
            if self.lookup_keyword != keyword:
                self.lookup_keyword = keyword
                self.lookup_list = construct_lookup_list(keyword)
                self.lookup_cnt = 0

            # All lookups used.
            if self.lookup_cnt >= len(self.lookup_list):
                obs = "No more results.\n"
            else:
                obs = f"(Result {self.lookup_cnt + 1} / {len(self.lookup_list)}) " + self.lookup_list[self.lookup_cnt]
                self.lookup_cnt += 1
        return out