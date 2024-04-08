"""CRITIC Agent.

GitHub Repository: https://github.com/microsoft/ProphetNet/tree/master/CRITIC
Original Paper: http://arxiv.org/abs/2305.11738
"""


from typing import Any

from langchain_community.utilities.google_search import GoogleSearchAPIWrapper
from langchain_core.language_models.chat_models import BaseChatModel

from discussion_agents.cog.agent.base import BaseAgent
from discussion_agents.cog.functional.critic import _prompt_agent, _prompt_critique
from discussion_agents.cog.prompts.critic import (
    HOTPOTQA_FEWSHOT_EXAMPLES_COT,
    HOTPOTQA_FEWSHOT_EXAMPLES_CRITIC,
    TRIVIAQA_FEWSHOT_EXAMPLES_COT,
    TRIVIAQA_FEWSHOT_EXAMPLES_CRITIC,
)


class CriticAgent(BaseAgent):
    """CRITIC Agent.

    Attributes:
        llm (BaseChatModel): An instance of a language model used for generating initial answers
            and critiques.
        search (GoogleSearchAPIWrapper): A search API wrapper used for obtaining evidence to
            support or refute generated answers and critiques.
    """

    def __init__(self, llm: BaseChatModel, search: GoogleSearchAPIWrapper) -> None:
        """Initialize the class with a language model and a search API wrapper.

        :param llm: An instance of BaseChatModel for language model operations.
        :param search: An instance of GoogleSearchAPIWrapper for search operations.
        """
        super().__init__()
        self.llm = llm
        self.search = search

    def generate(
        self,
        question: str,
        benchmark_prompt: str,
        max_interactions: int = 7,
        use_tool: bool = True,
        evidence_length: int = 400,
    ) -> Any:
        """Generates an answer that is refined with search results.

        Args:
            question (str): The question to be answered.
            benchmark_prompt (str) : The user input what benchmark want run :  hotpotqa , triviaqa
            max_interactions (int): The maximum number of critique cycles. Defaults to 7.
            use_tool (bool): Flag to decide whether to use the search tool for evidence gathering. Defaults to True.
            evidence_length (int): The maximum length of the evidence snippet to be included in the context. Defaults to 400.

        Returns:
            str: The most refined answer after the specified number of critique iterations, or until
            a satisfactory answer is reached.
        """
        if benchmark_prompt == "hotpotqa":
            examples = HOTPOTQA_FEWSHOT_EXAMPLES_COT
            critique_examples = HOTPOTQA_FEWSHOT_EXAMPLES_CRITIC

        elif benchmark_prompt == "triviaqa":
            examples = TRIVIAQA_FEWSHOT_EXAMPLES_COT
            critique_examples = TRIVIAQA_FEWSHOT_EXAMPLES_CRITIC

        else:
            raise ValueError("Unsupported benchmark")

        answer = _prompt_agent(
            llm=self.llm,
            question=question,
            examples=examples,
            benchmark=benchmark_prompt,
        )

        out, revised_answer = "", ""
        exist_query = []
        exist_evidence = set()
        for idx in range(max_interactions):
            critique = _prompt_critique(
                llm=self.llm,
                question=question,
                examples=critique_examples,
                answer=answer,
                critique="" if not idx else out,
                benchmark=benchmark_prompt,
            ).split("> Evidence: ")[0]

            out += critique

            print(critique)

            if "> Search Query: " in critique:
                _, search_query = critique.split("> Search Query:")[:2]
                search_query = search_query.split("\n")[0].strip()

                if use_tool:
                    exist_query.append(search_query)
                    for k in range(exist_query.count(search_query), 8):
                        search_result = self.search.results(
                            search_query, num_results=k
                        )[-1]
                        if search_result["snippet"] not in exist_evidence:
                            exist_evidence.add(search_result["snippet"])
                            break

                    context = f"""> Evidence: [{search_result['title']}] {search_result['snippet'][:evidence_length]}\n\n"""
                    print("context : ", context)
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
                out += f"\nLet's give the most possible answer.\n\nQuestion: {question}\nHere's "

        return revised_answer
