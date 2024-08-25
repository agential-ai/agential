"""CRITIC Agent strategies for QA."""

from typing import Any, Dict, List, Optional, Set, Tuple

from langchain_community.utilities.google_serper import GoogleSerperAPIWrapper

from agential.cog.critic.functional import _prompt_agent, _prompt_critique
from agential.cog.critic.strategies.general import CriticGeneralStrategy
from agential.llm.llm import BaseLLM, Response


class CriticQAStrategy(CriticGeneralStrategy):
    """A strategy class for QA benchmarks using the CRITIC agent.

    Attributes:
        llm (BaseLLM): The language model used for generating answers and critiques.
        search (Optional[GoogleSerperAPIWrapper]): An optional search API wrapper for obtaining evidence. Required if use_tool is True.
        evidence_length (int): The maximum length of the evidence snippet to be included in the context. Defaults to 400.
        num_results (int): The number of search results to retrieve. Defaults to 8.
        testing (bool): Whether the strategy is in test mode. Defaults to False.
    """

    def __init__(
        self,
        llm: BaseLLM,
        search: Optional[GoogleSerperAPIWrapper] = None,
        evidence_length: int = 400,
        num_results: int = 8,
        testing: bool = False,
    ) -> None:
        """Initialization."""
        super().__init__(llm=llm, testing=testing)
        self.search = search
        self.evidence_length = evidence_length
        self.num_results = num_results

        self._query_history: List[str] = []
        self._evidence_history: Set[str] = set()

    def generate_answer(
        self,
        question: str,
        examples: str,
        prompt: str,
        additional_keys: Dict[str, str],
    ) -> Tuple[str, List[Response]]:
        """Generates an answer using the provided language model, question, examples, and prompt.

        Args:
            question (str): The question to be answered by the language model.
            examples (str): Few-shot examples to guide the language model in generating the answer.
            prompt (str): The instruction template used to prompt the language model.
            additional_keys (Dict[str, str]): Additional keys to format the prompt.

        Returns:
            Tuple[str, List[Response]]: The generated answer and model responses.
        """
        out = _prompt_agent(
            llm=self.llm,
            question=question,
            examples=examples,
            prompt=prompt,
            additional_keys=additional_keys,
        )

        return out.output_text, [out]

    def generate_critique(
        self,
        idx: int,
        question: str,
        examples: str,
        answer: str,
        critique: str,
        prompt: str,
        additional_keys: Dict[str, str],
        use_tool: bool,
        max_interactions: int,
    ) -> Tuple[str, Dict[str, Any], bool, List[Response]]:
        """Generates a critique of the provided answer using the given language model, question, examples, and prompt.

        This method does the following:
            1. Use the language model to generate an initial critique based on the provided question, examples, answer, and prompt.
            2. Check if the generated critique suggests a search query:
                - If yes, execute the search query using the search tool if `use_tool` is True.
                - Append the search result and context to the critique.
                - If `use_tool` is False, re-prompt the language model to generate a critique including the search result.
            3. If no search query is suggested:
                - Add a prompt for providing the most possible answer to the critique.
                - Use the language model to generate the final critique based on this new prompt.
                - Set the halt flag to True.
            4. Return the final critique and any external tool information.

        Args:
            idx (int): The index of the current interaction.
            question (str): The question that was answered by the language model.
            examples (str): Few-shot examples to guide the language model in generating the critique.
            answer (str): The answer to be critiqued.
            critique (str): The previous critique, if any.
            prompt (str): The instruction template used to prompt the language model for the critique.
            additional_keys (Dict[str, str]): Additional keys to format the critique prompt.
            use_tool (bool): Whether to use an external tool (e.g., interpreter, search tool) during critique.
            max_interactions (int): The maximum number of critique interactions.

        Returns:
            Tuple[str, Dict[str, Any], bool, List[Response]]: The generated critique, any external tool information, a boolean for if it finished, and the responses.
        """
        external_tool_info = {"search_query": "", "search_result": ""}
        responses = []

        critique_response = _prompt_critique(
            llm=self.llm,
            question=question,
            examples=examples,
            answer=answer,
            critique=critique,
            prompt=prompt,
            additional_keys=additional_keys,
        )
        responses.append(critique_response)
        new_critique = critique_response.output_text
        new_critique = new_critique.split("> Evidence: ")[0]

        finished = False
        if "> Search Query: " in new_critique:
            _, search_query = new_critique.split("> Search Query:")[:2]
            search_query = search_query.split("\n")[0].strip()

            search_result, context = self.handle_search_query(
                idx=idx,
                question=question,
                search_query=search_query,
                use_tool=use_tool,
                max_interactions=max_interactions,
            )
            new_critique = f"{critique}\n{new_critique}{context}"
            if not use_tool:
                search_result_response = _prompt_critique(
                    llm=self.llm,
                    question=question,
                    examples=examples,
                    answer=answer,
                    critique=new_critique,
                    prompt=prompt,
                    additional_keys=additional_keys,
                )
                responses.append(search_result_response)
                search_result_no_tool = search_result_response.output_text
                search_result_no_tool = search_result_no_tool.split("> Evidence: ")[0]

                new_critique = (
                    f"{critique}\n{new_critique}{search_result_no_tool.strip()}"
                )
            external_tool_info["search_query"] = search_query
            external_tool_info["search_result"] = search_result if use_tool else search_result_no_tool  # type: ignore
        else:
            if "most possible answer: " not in new_critique:
                new_critique = f"{critique}\n{new_critique}\nLet's give the most possible answer.\n\nQuestion: {question}\nHere's "
                answer_response = _prompt_critique(
                    llm=self.llm,
                    question=question,
                    examples=examples,
                    answer=answer,
                    critique=new_critique,
                    prompt=prompt,
                    additional_keys=additional_keys,
                )
                responses.append(answer_response)
                new_critique = answer_response.output_text
                new_critique = new_critique.split("> Evidence: ")[0]

            new_critique = new_critique.split("most possible answer: ")[-1].strip()
            finished = True

        return new_critique, external_tool_info, finished, responses

    def create_output_dict(
        self,
        finished: bool,
        answer: str,
        critique: str,
        external_tool_info: Dict[str, Any],
        answer_response: List[Response],
        critique_response: List[Response],
    ) -> Dict[str, Any]:
        """Creates a dictionary containing the answer and critique, along with any additional key updates.

        This function compiles the final output dictionary which includes the original answer,
        the generated critique, and any information gathered from external tools. If the halting
        condition is met, the critique is used in place of the answer.

        Args:
            finished (bool): Whether the critique process has finished.
            answer (str): The original answer.
            critique (str): The generated critique.
            external_tool_info (Dict[str, Any]): Information from any external tools used during the critique.
            answer_response (List[Response]): The responses from the answer.
            critique_response (List[Response]): The responses from the critique.

        Returns:
            Dict[str, Any]: A dictionary containing the answer, critique, and additional key updates.
        """
        output_dict = {
            "answer": answer if not finished else critique,
            "critique": critique,
            "external_tool_info": external_tool_info,
            "critique_response": critique_response,
            "answer_response": answer_response,
        }
        return output_dict

    def update_answer_based_on_critique(
        self,
        question: str,
        examples: str,
        answer: str,
        critique: str,
        prompt: str,
        additional_keys: Dict[str, str],
        external_tool_info: Dict[str, str],
    ) -> Tuple[str, List[Response]]:
        """Updates the answer based on the provided critique using the given language model and question.

        Args:
            question (str): The question that was answered by the language model.
            examples (str): Few-shot examples to guide the language model in generating the updated answer.
            answer (str): The original answer to be updated.
            critique (str): The critique of the original answer.
            prompt (str): The instruction template used to prompt the language model for the update.
            additional_keys (Dict[str, str]): Additional keys to format the update prompt.
            external_tool_info (Dict[str, str]): Information from any external tools used during the critique.

        Returns:
            str: The updated answer.
            List[Response]: The responses from the critique.
        """
        return answer, []

    def halting_condition(self, finished: bool) -> bool:
        """Checks if the halting condition is met.

        Args:
            finished (bool): Whether the interaction

        Returns:
            bool: True if the halting condition is met, False otherwise.
        """
        return finished

    def reset(self) -> None:
        """Resets the strategy's internal state."""
        self._query_history = []
        self._evidence_history = set()

    def handle_search_query(
        self,
        idx: int,
        question: str,
        search_query: str,
        use_tool: bool,
        max_interactions: int,
    ) -> Tuple[Dict[str, str], str]:
        """Handles a search query and returns the search result and context.

        This function processes a search query to gather evidence. If the use_tool flag is set,
        it performs the search using the provided search tool and compiles the search result
        and context to be used in the critique process. Attempts up to num_results if using search tool.
        If search tool is not used, a string is returned.

        Args:
            idx (int): The index of the current interaction.
            question (str): The question that was answered by the language model.
            search_query (str): The search query to be executed.
            use_tool (bool): Whether to use an external tool (e.g., search tool) during critique.
            max_interactions (int): The maximum number of critique interactions.

        Returns:
            Tuple[Dict[str, str], str]: The search result and context.
        """
        if use_tool:
            if not self.search:
                raise ValueError("Search tool is required but not provided.")

            self._query_history.append(search_query)
            count = self._query_history.count(search_query)
            start = count if count < self.num_results else self.num_results - 1  # type: ignore

            for k in range(start, self.num_results):  # type: ignore
                search_result = self.search.results(search_query, num_results=k)[-1]
                if (
                    "snippet" in search_result
                    and search_result["snippet"] not in self._evidence_history
                ):
                    self._evidence_history.add(search_result["snippet"])
                    break

            if "title" not in search_result and "snippet" not in search_result:
                context = f"""> Evidence: [] No results found\n\n"""
            else:
                context = f"""> Evidence: [{search_result['title']}] {search_result['snippet'][:self.evidence_length]}\n\n"""  # type: ignore
            if idx == max_interactions - 2:
                context += f"Let's give the most possible answer.\n\nQuestion: {question}\nHere's "
        else:
            search_result = {}
            context = """> Evidence: """

        return search_result, context


class CritHotQAStrategy(CriticQAStrategy):
    """A strategy class for the HotpotQA benchmark using the CRITIC agent."""

    pass


class CritTriviaQAStrategy(CriticQAStrategy):
    """A strategy class for the TriviaQA benchmark using the CRITIC agent."""

    pass


class CritAmbigNQStrategy(CriticQAStrategy):
    """A strategy class for the AmbigNQ benchmark using the CRITIC agent."""

    pass


class CritFEVERStrategy(CriticQAStrategy):
    """A strategy class for the FEVER benchmark using the CRITIC agent."""

    pass
