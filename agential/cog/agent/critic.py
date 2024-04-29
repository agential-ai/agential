"""CRITIC Agent.

GitHub Repository: https://github.com/microsoft/ProphetNet/tree/master/CRITIC
Original Paper: http://arxiv.org/abs/2305.11738
"""

from langchain_community.utilities.google_search import GoogleSearchAPIWrapper
from langchain_core.language_models.chat_models import BaseChatModel

from agential.cog.agent.base import BaseAgent
from agential.cog.functional.critic import (
    _prompt_agent,
    _prompt_critique,
)
from agential.cog.prompts.critic import (
    CRITIC_CRITIQUE_INSTRUCTION_HOTPOTQA,
    CRITIC_INSTRUCTION_HOTPOTQA,
    HOTPOTQA_FEWSHOT_EXAMPLES_COT,
    HOTPOTQA_FEWSHOT_EXAMPLES_CRITIC, 
    TabMWP_MULTI_CHOICE, 
    TabMWP_FREE_TEXT
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
        """Initialization."""
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
        evidence_length: int = 400,
    ) -> str:
        """Generates an answer that is refined with search results.

        Args:
            question (str): The question to be answered.
            examples (str): Few-shot examples to guide the language model in generating the initial answer. Defaults to HOTPOTQA_FEWSHOT_EXAMPLES_COT.
            prompt (str): The instruction template used to prompt the language model for the initial answer. Defaults to CRITIC_INSTRUCTION_HOTPOTQA.
            critique_examples (str): Few-shot examples to guide the language model in generating critiques. Defaults to HOTPOTQA_FEWSHOT_EXAMPLES_CRITIC.
            critique_prompt (str): The instruction template for generating critiques. Defaults to CRITIC_CRITIQUE_INSTRUCTION_HOTPOTQA.
            max_interactions (int): The maximum number of critique cycles. Defaults to 7.
            use_tool (bool): Flag to decide whether to use the search tool for evidence gathering. Defaults to True.
            evidence_length (int): The maximum length of the evidence snippet to be included in the context. Defaults to 400.

        Returns:
            str: The most refined answer after the specified number of critique iterations, or until
            a satisfactory answer is reached.
        """
        answer = _prompt_agent(
            llm=self.llm, question=question, examples=examples, prompt=prompt
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
                prompt=critique_prompt,
            ).split("> Evidence: ")[0]
            out += critique

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
