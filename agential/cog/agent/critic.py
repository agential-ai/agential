from typing import Dict, List, Optional

from langchain_community.utilities.google_serper import GoogleSerperAPIWrapper
from langchain_core.language_models.chat_models import BaseChatModel

from agential.cog.agent.base import BaseAgent
from agential.cog.strategies.strategy_factory import CriticStrategyFactory

from agential.cog.prompts.critic import (
    CRITIC_CRITIQUE_INSTRUCTION_HOTPOTQA,
    CRITIC_INSTRUCTION_HOTPOTQA,
    CRITIC_POT_INSTRUCTION_TEST_HUMANEVAL,
    HOTPOTQA_FEWSHOT_EXAMPLES_COT,
    HOTPOTQA_FEWSHOT_EXAMPLES_CRITIC,
    HUMANEVAL_FEWSHOT_EXAMPLES_POT_TEST,
)

class CriticAgent(BaseAgent):
    """CRITIC Agent.

    Attributes:
        llm (BaseChatModel): An instance of a language model used for generating initial answers
            and critiques.
        mode (str): The CRITIC agent's mode. Can be "qa", "math", or "code".
        search (Optional[GoogleSerperAPIWrapper]): A search API wrapper used for obtaining evidence to
            support or refute generated answers and critiques. Defaults to None. Required if mode = "qa".
    """

    def __init__(
        self,
        llm: BaseChatModel,
        mode: str,
        search: Optional[GoogleSerperAPIWrapper] = None,
    ) -> None:
        """Initialization."""
        super().__init__()

        self.llm = llm
        self.mode = mode
        self.search = search
        if self.mode == "qa" and not self.search:
            raise ValueError("`GoogleSerperAPIWrapper` is required when mode is 'qa'.")

        self.strategy = CriticStrategyFactory.get_strategy(self.mode)

    def generate(
        self,
        question: str,
        examples: str = HOTPOTQA_FEWSHOT_EXAMPLES_COT,
        prompt: str = CRITIC_INSTRUCTION_HOTPOTQA,
        critique_examples: str = HOTPOTQA_FEWSHOT_EXAMPLES_CRITIC,
        critique_prompt: str = CRITIC_CRITIQUE_INSTRUCTION_HOTPOTQA,
        additional_keys: Dict[str, str] = {},
        critique_additional_keys: Dict[str, str] = {},
        max_interactions: int = 7,
        use_search_tool: bool = True,
        use_interpreter_tool: bool = True,
        evidence_length: int = 400,
        tests: str = "",
        test_prompt: str = CRITIC_POT_INSTRUCTION_TEST_HUMANEVAL,
        test_examples: str = HUMANEVAL_FEWSHOT_EXAMPLES_POT_TEST,
    ) -> List[Dict[str, str]]:
        """Generates an answer that is refined with search results.

        Args:
            question (str): The question to be answered.
            examples (str): Few-shot examples to guide the language model in generating the initial answer. Defaults to HOTPOTQA_FEWSHOT_EXAMPLES_COT.
            prompt (str): The instruction template used to prompt the language model for the initial answer. Defaults to CRITIC_INSTRUCTION_HOTPOTQA.
            critique_examples (str): Few-shot examples to guide the language model in generating critiques. Defaults to HOTPOTQA_FEWSHOT_EXAMPLES_CRITIC.
            critique_prompt (str): The instruction template for generating critiques. Defaults to CRITIC_CRITIQUE_INSTRUCTION_HOTPOTQA.
            additional_keys (Dict[str, str]): Additional keys to format the prompt. Defaults to {}.
            critique_additional_keys (Dict[str, str]): Additional keys to format the critique_prompt. Defaults to {}.
            max_interactions (int): The maximum number of critique cycles. Defaults to 7.
            use_search_tool (bool): Only for "qa" mode. Flag to decide whether to use the search tool for evidence gathering. Defaults to True.
            use_interpreter_tool (bool): Only for "math" or "code" mode. Flag to decide whether to use the interpreter tool for code execution. Defaults to True.
            evidence_length (int): The maximum length of the evidence snippet to be included in the context. Used only in "qa" mode. Defaults to 400.
            tests (str): The unit tests. Used in "code" mode. Defaults to "".
            test_prompt (str): The instruction template for generating unit tests. Used only in "code" mode. Defaults to CRITIC_POT_INSTRUCTION_TEST_HUMANEVAL.
            test_examples (str): Few-shot examples to guide model in generating unit tests. Used only in "code" mode. Defaults to HUMANEVAL_FEWSHOT_EXAMPLES_POT_TEST.

        Returns:
            List[Dict[str, str]]: A list of dictionaries.
                "qa" mode:
                    - Each dictionary contains an "answer" and "critique". Optionally, a
                    dictionary may include the search "query" and "search_result", and the final dictionary includes the final "revised_answer".
                "math" mode:
                    - Each dictionary contains "code" and "critique". Optionally, a dictionary may include
                    the "execution_status" and "code_answer" if use_interpreter_tool is True. If the critic
                    improves the solution, then the dictionary will have an "improved_code" key.
        """
        out = []

        # Initial answer generation
        answer = self.strategy.generate(self.llm, question, examples, prompt, additional_keys)

        for idx in range(max_interactions):
            critique, additional_keys_update = self.strategy.generate_critique(
                self.llm, question, critique_examples, answer, critique_prompt, additional_keys, critique_additional_keys, tests, use_interpreter_tool
            )

            out.append(self.strategy.create_output_dict(answer, critique, additional_keys_update))

            if "query" in additional_keys_update:
                search_result, evidence_context = self.handle_search_query(
                    additional_keys_update["query"], evidence_length, use_search_tool
                )
                critique += evidence_context
                out[idx]["search_result"] = search_result

            if "revised_answer" in additional_keys_update:
                out[idx]["revised_answer"] = additional_keys_update["revised_answer"]
                break

            if not critique:
                break

            # Update answer for the next iteration
            answer = self.strategy.update_answer_based_on_critique(self.llm, question, answer, critique)

        return out

    def handle_search_query(self, search_query, evidence_length, use_search_tool):
        if use_search_tool and self.search:
            search_result = self.search.results(search_query)["organic"][0]
            context = f"""> Evidence: [{search_result['title']}] {search_result['snippet'][:evidence_length]}\n\n"""
        else:
            search_result = ""
            context = """> Evidence: """
        return search_result, context
