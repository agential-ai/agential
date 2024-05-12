"""CRITIC Agent.

GitHub Repository: https://github.com/microsoft/ProphetNet/tree/master/CRITIC
Original Paper: http://arxiv.org/abs/2305.11738
"""

from typing import Dict, List, Optional

from langchain_community.utilities.google_serper import GoogleSerperAPIWrapper
from langchain_core.language_models.chat_models import BaseChatModel

from agential.cog.agent.base import BaseAgent
from agential.cog.functional.critic import (
    _prompt_agent,
    _prompt_critique,
    safe_execute,
)
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
        if self.mode == "qa":
            out = []

            answer = _prompt_agent(
                llm=self.llm,
                question=question,
                examples=examples,
                additional_keys=additional_keys,
                prompt=prompt,
            )

            criticism, revised_answer = "", ""
            for idx in range(max_interactions):
                critique = _prompt_critique(
                    llm=self.llm,
                    question=question,
                    examples=critique_examples,
                    answer=answer,
                    critique=criticism,
                    additional_keys=critique_additional_keys,
                    prompt=critique_prompt,
                ).split("> Evidence: ")[
                    0
                ]  # Stop at ""> Evidence: ".
                criticism += critique

                out.append({"answer": answer, "critique": critique})

                if "> Search Query: " in critique:
                    _, search_query = critique.split("> Search Query:")[:2]
                    search_query = search_query.split("\n")[0].strip()

                    if use_search_tool and self.search:
                        search_result = self.search.results(search_query)["organic"][0]
                        context = f"""> Evidence: [{search_result['title']}] {search_result['snippet'][:evidence_length]}\n\n"""
                        if idx == max_interactions - 2:
                            context += f"Let's give the most possible answer.\n\nQuestion: {question}\nHere's "
                    else:
                        context = """> Evidence: """

                    criticism += context
                    out[idx]["query"] = search_query if use_search_tool else ""
                    out[idx]["search_result"] = (
                        search_result["snippet"][:evidence_length]
                        if use_search_tool
                        else ""
                    )

                elif "most possible answer: " in critique:
                    _, revised_answer = critique.split("most possible answer: ")
                    revised_answer = revised_answer.strip()
                    out[idx]["revised_answer"] = revised_answer
                    break
                else:
                    if not critique:
                        break
                    criticism += f"\nLet's give the most possible answer.\n\nQuestion: {question}\nHere's "

            return out
        elif self.mode == "math":
            out = []
            code = _prompt_agent(
                llm=self.llm,
                question=question,
                examples=examples,
                additional_keys=additional_keys,
                prompt=prompt,
            )

            for idx in range(max_interactions):
                # Get additional code execution information.
                if use_interpreter_tool:
                    code_answer, execution_status = safe_execute(
                        code
                    )  # Can be None, "Exception".
                    critique_additional_keys = {
                        "execution_status": execution_status,
                        "code_answer": code_answer if code_answer else "",
                    }

                # Generate code critique.
                critique = _prompt_critique(
                    llm=self.llm,
                    question=question,
                    examples=critique_examples,
                    answer=code,
                    critique="",
                    additional_keys=critique_additional_keys,  # type: ignore
                    prompt=critique_prompt,
                ).split("Here's")[
                    0
                ]  # Stop at Here's.
                out.append({"code": code, "critique": critique})
                if use_interpreter_tool:
                    out[idx]["execution_status"] = execution_status
                    out[idx]["code_answer"] = code_answer  # type: ignore

                # Halting condition.
                if "it is correct." in critique.lower():
                    break

                # Generate the new solution from the critique.
                code = _prompt_critique(
                    llm=self.llm,
                    question=question,
                    examples=critique_examples,
                    answer=code,
                    critique=critique
                    + "\n\n"
                    + "Here's a better solution:\n```python\n",
                    additional_keys=critique_additional_keys,  # type: ignore
                    prompt=critique_prompt,
                ).split("```")[
                    0
                ]  # Stop at ```.
                out[idx]["improved_code"] = code

            return out
        elif self.mode == "code":
            out = []
            code = _prompt_agent(
                llm=self.llm,
                question=question,
                examples=examples,
                additional_keys=additional_keys,
                prompt=prompt,
            )

            for idx in range(max_interactions):
                # Generate unit tests like in Reflexion and execute unit tests.
                if use_interpreter_tool:
                    if not tests:
                        tests = _prompt_agent(
                            llm=self.llm,
                            question=question,
                            examples=test_examples,
                            prompt=test_prompt,
                        )
                    code_answer, execution_status = safe_execute(
                        code + "\n\n" + tests
                    )  # Can be None, "Exception".
                    critique_additional_keys = {
                        "execution_status": execution_status,
                        "code_answer": code_answer if code_answer else "",
                    }

                # Generate code critique.
                critique = _prompt_critique(
                    llm=self.llm,
                    question=code,
                    examples=critique_examples,
                    answer=tests,
                    critique="",
                    additional_keys=critique_additional_keys,
                    prompt=critique_prompt,
                ).split("Here's")[
                    0
                ]  # Stop at Here's.
                out.append({"code": code, "critique": critique})
                if use_interpreter_tool:
                    out[idx]["execution_status"] = execution_status
                    out[idx]["code_answer"] = code_answer  # type: ignore

                # Halting condition.
                if "it is correct." in critique.lower():
                    break

                # Generate the new solution from the critique.
                code = _prompt_critique(
                    llm=self.llm,
                    question=code,
                    examples=critique_examples,
                    answer=tests,
                    critique=critique
                    + "\n\n"
                    + "Here's a better solution:\n```python\n",
                    additional_keys=critique_additional_keys,  # type: ignore
                    prompt=critique_prompt,
                ).split("```")[
                    0
                ]  # Stop at ```.
                out[idx]["improved_code"] = code

            return out
        else:
            raise ValueError("mode must be set to either 'qa', 'math', or 'code'.")
