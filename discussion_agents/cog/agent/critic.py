"""CRITIC Agent.

GitHub Repository: https://github.com/microsoft/ProphetNet/tree/master/CRITIC
Original Paper: http://arxiv.org/abs/2305.11738
"""

from typing import Any
from discussion_agents.cog.agent.base import BaseAgent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_community.utilities.google_search import GoogleSearchAPIWrapper
from discussion_agents.cog.functional.critic import (
    _prompt_agent, 
    _prompt_critique, 
)
from discussion_agents.cog.prompts.critic import (
    HOTPOTQA_FEWSHOT_EXAMPLES_COT, 
    CRITIC_INSTRUCTION_HOTPOTQA,
    HOTPOTQA_FEWSHOT_EXAMPLES_CRITIC,
    CRITIC_CRITIQUE_INSTRUCTION_HOTPOTQA,
)

class CriticAgent(BaseAgent):
    def __init__(
        self,
        llm: BaseChatModel,
        search: GoogleSearchAPIWrapper
    ) -> None:
        super().__init__()

        self.llm = llm
        self.search = search

    def generate(
        self, 
        question: str,
        examples: str = HOTPOTQA_FEWSHOT_EXAMPLES_COT,
        prompt: str = CRITIC_INSTRUCTION_HOTPOTQA,
        critique_examples: str = HOTPOTQA_FEWSHOT_EXAMPLES_CRITIC,
        critique_prompt: str = CRITIC_CRITIQUE_INSTRUCTION_HOTPOTQA,
        max_interactions: int = 7,
        use_tool: bool = True,
        evidence_length: int = 400
    ) -> Any:
        answer = _prompt_agent(
            llm=self.llm,
            question=question,
            examples=examples,
            prompt=prompt
        )

        out = ""
        exist_query = []
        exist_evidence = set()
        for idx in range(max_interactions):
            critique = _prompt_critique(
                llm=self.llm,
                question=question,
                examples=critique_examples,
                answer=answer,
                critique="" if not idx else out,
                prompt=critique_prompt
            ).split("> Evidence: ")[0]
            out += critique

            if "> Search Query: " in critique:
                _, search_query = critique.split("> Search Query:")[:2]
                search_query = search_query.split("\n")[0].strip()

                if use_tool:
                    exist_query.append(search_query)
                    for k in range(exist_query.count(search_query), 8):
                        search_result = self.search.results(search_query, num_results=k)[-1]
                        if search_result['snippet'] not in exist_evidence:
                            exist_evidence.add(search_result['snippet'])
                            break

                    context = f"""> Evidence: [{search_result['title']}] {search_result['snippet'][:evidence_length]}\n\n"""
                    if idx == max_interactions - 2:
                        context += f"Let's give the most possible answer.\n\nQuestion: {question}\nHere's "
                else:
                    context = """> Evidence: """

                out += context
            elif "most possible answer: " in critique:
                _, revised_answer = critique.split("most possible answer: ")
                revised_answer = revised_answer.strip()
                break
            else:
                if not critique:
                    break
                out += f"Let's give the most possible answer.\n\nQuestion: {question}\nHere's "
