"""CRITIC QA Agent for question-answering benchmarks."""

from typing import Any, Dict, List, Optional, Set, Tuple
import time
import os

from agential.agents.base import BaseAgent
from agential.agents.critic.utils import (
    log_llm_io,
)
from agential.core.llm import BaseLLM, Response


class CriticQA(BaseAgent):
    """CRITIC QA Agent for question-answering benchmarks.

    This agent implements the CRITIC methodology for QA tasks, which involves:
    1. Generating an initial answer
    2. Critiquing the answer with potential search queries
    3. Using search tools to gather evidence
    4. Refining the answer based on the critique and evidence
    5. Repeating the process until convergence or max interactions reached
    """

    def __init__(
        self,
        llm: BaseLLM,
        benchmark: str,
        use_search: bool = True,
        evidence_length: int = 400,
        num_results: int = 8,
        max_interactions: int = 7,
        verbose: bool = False,
        config: dict = {},
    ):
        """Initialize the CRITIC QA Agent.

        Args:
            llm: The language model to use for generation
            benchmark: The benchmark name (hotpotqa, fever, triviaqa, ambignq)
            use_search: Whether to use search functionality for evidence gathering
            evidence_length: Maximum length of evidence snippets
            num_results: Number of search results to retrieve
            max_interactions: Maximum number of critique cycles
            verbose: Whether to enable verbose logging
            config: Additional configuration
        """
        super().__init__(llm=llm, benchmark=benchmark, verbose=verbose, config=config)
        self.use_search = use_search
        self.evidence_length = evidence_length
        self.num_results = num_results
        self.max_interactions = max_interactions

        # Select appropriate critique prompt and examples based on use_search
        if not use_search:
            if "critique_prompt_no_tool" in self.config:
                self.config["critique_prompt"] = self.config["critique_prompt_no_tool"]
            if "critique_examples_no_tool" in self.config:
                self.config["critique_examples"] = self.config[
                    "critique_examples_no_tool"
                ]

        # Initialize search client if needed
        self.search = None
        if self.use_search:
            try:
                from tavily import TavilyClient

                api_key = os.getenv("TAVILY_API_KEY")
                if not api_key:
                    raise ValueError("TAVILY_API_KEY environment variable not set")
                self.search = TavilyClient(api_key=api_key)
            except ImportError:
                print("Warning: Tavily not available. Search functionality disabled.")
                self.use_search = False

    def generate(
        self,
        question: str,
        examples: str = "",
        prompt: str = "",
        critique_examples: str = "",
        critique_prompt: str = "",
        additional_keys: Dict[str, str] = {},
        critique_additional_keys: Dict[str, str] = {},
        fewshot_type: str = "",
        max_interactions: Optional[int] = None,
        use_search: Optional[bool] = None,
        reset: bool = True,
    ) -> dict:
        """Generate an answer using the CRITIC methodology.

        Args:
            question: The question to answer
            examples: Few-shot examples for answer generation
            prompt: Prompt template for answer generation
            critique_examples: Few-shot examples for critique generation
            critique_prompt: Prompt template for critique generation
            additional_keys: Additional keys to format answer prompts
            critique_additional_keys: Additional keys to format critique prompts
            fewshot_type: Type of few-shot examples to use
            max_interactions: Override max_interactions from init
            use_search: Override use_search from init

        Returns:
            dict: The final answer and critique information
        """
        max_interactions = max_interactions or self.max_interactions
        use_search = use_search if use_search is not None else self.use_search

        # Use provided parameters or fall back to config
        if not prompt or not critique_prompt or not examples or not critique_examples:
            prompt = self.config["prompt"]
            critique_prompt = self.config["critique_prompt"]
            examples = self.config["examples"]
            critique_examples = self.config["critique_examples"]

        # Initialize search history tracking
        query_history: List[str] = []
        evidence_history: Set[str] = set()

        start_time = time.time()
        steps = []
        total_tokens = total_cost = 0
        step_metrics = []

        # Generate initial answer
        answer, answer_responses = self._generate_answer(
            question=question,
            examples=examples,
            prompt=prompt,
            additional_keys=additional_keys,
        )

        # Track metrics for initial answer
        for response in answer_responses:
            total_tokens += response.total_tokens
            total_cost += response.total_cost

        critique = ""
        finished = False

        # Iterative critique and refinement
        for idx in range(max_interactions):
            step_start = time.time()

            # Generate critique
            new_critique, external_tool_info, finished, critique_responses = (
                self._generate_critique(
                    idx=idx,
                    question=question,
                    examples=critique_examples,
                    answer=answer,
                    critique=critique,
                    prompt=critique_prompt,
                    additional_keys=critique_additional_keys,
                    use_search=use_search,
                    max_interactions=max_interactions,
                    query_history=query_history,
                    evidence_history=evidence_history,
                )
            )

            # Track metrics for critique
            step_tokens = step_cost = 0
            for response in critique_responses:
                step_tokens += response.total_tokens
                step_cost += response.total_cost
                total_tokens += response.total_tokens
                total_cost += response.total_cost

            step_time = time.time() - step_start
            step_metrics.append(
                {
                    "step": idx + 1,
                    "total_step_time": step_time,
                    "total_step_tokens": step_tokens,
                    "total_step_cost": step_cost,
                }
            )

            # Create step output
            step_output = {
                "answer": answer,
                "critique": new_critique,
                "external_tool_info": external_tool_info,
                "answer_response": answer_responses,
                "critique_response": critique_responses,
            }
            steps.append(step_output)

            # Update critique
            critique = new_critique

            # Check if finished
            if finished:
                break

        total_time = time.time() - start_time

        # Calculate metrics
        metrics = {
            "total_time": total_time,
            "total_tokens": total_tokens,
            "total_cost": total_cost,
            "step_metrics": step_metrics,
        }

        # Determine final answer
        final_answer = critique if finished else answer

        return {
            "answer": final_answer,
            "steps": steps,
            "metrics": metrics,
        }

    def _generate_answer(
        self,
        question: str,
        examples: str,
        prompt: str,
        additional_keys: Dict[str, str],
    ) -> Tuple[str, List[Response]]:
        """Generate an initial answer.

        Args:
            question: The question to answer
            examples: Few-shot examples
            prompt: The prompt template
            additional_keys: Additional formatting keys

        Returns:
            Tuple of (answer, responses)
        """
        # Build prompt
        formatted_prompt = prompt.format(
            question=question, examples=examples, **additional_keys
        )

        # Generate response
        response = self.llm(formatted_prompt)

        if self.verbose:
            log_llm_io(response, "Generate Answer", self.verbose)

        return response.output_text, [response]

    def _generate_critique(
        self,
        idx: int,
        question: str,
        examples: str,
        answer: str,
        critique: str,
        prompt: str,
        additional_keys: Dict[str, str],
        use_search: bool,
        max_interactions: int,
        query_history: List[str],
        evidence_history: Set[str],
    ) -> Tuple[str, Dict[str, Any], bool, List[Response]]:
        """Generate a critique of the answer.

        Args:
            idx: Current interaction index
            question: The original question
            examples: Few-shot examples
            answer: The answer to critique
            critique: Previous critique
            prompt: Critique prompt template
            additional_keys: Additional formatting keys
            use_search: Whether to use search tools
            max_interactions: Maximum number of interactions
            query_history: List of previous search queries
            evidence_history: Set of previously retrieved evidence

        Returns:
            Tuple of (critique, external_tool_info, finished, responses)
        """
        external_tool_info = {"search_query": "", "search_result": ""}
        responses = []

        # Build critique prompt
        formatted_prompt = prompt.format(
            question=question,
            examples=examples,
            answer=answer,
            critique=critique,
            **additional_keys,
        )

        # Generate initial critique
        critique_response = self.llm(formatted_prompt)
        responses.append(critique_response)

        if self.verbose:
            log_llm_io(critique_response, f"Generate Critique {idx + 1}", self.verbose)

        new_critique = critique_response.output_text
        new_critique = new_critique.split("> Evidence: ")[0]

        finished = False

        # Check if critique suggests a search query
        if "> Search Query: " in new_critique:
            _, search_query = new_critique.split("> Search Query:")[:2]
            search_query = search_query.split("\n")[0].strip()

            # Handle search query
            search_result, context = self._handle_search_query(
                idx=idx,
                question=question,
                search_query=search_query,
                use_search=use_search,
                max_interactions=max_interactions,
                query_history=query_history,
                evidence_history=evidence_history,
            )

            new_critique = f"{critique}\n{new_critique}{context}"

            # If not using search, generate critique with search result
            if not use_search:
                search_result_response = self.llm(formatted_prompt)
                responses.append(search_result_response)

                if self.verbose:
                    log_llm_io(
                        search_result_response,
                        f"Search Result Critique {idx + 1}",
                        self.verbose,
                    )

                search_result_no_tool = search_result_response.output_text
                search_result_no_tool = search_result_no_tool.split("> Evidence: ")[0]
                new_critique = (
                    f"{critique}\n{new_critique}{search_result_no_tool.strip()}"
                )

            external_tool_info["search_query"] = search_query
            if use_search:
                # Convert dict to string for logging/record
                external_tool_info["search_result"] = str(search_result)
            else:
                external_tool_info["search_result"] = search_result_no_tool
        else:
            # No search query suggested, generate final answer
            if "Answer: " not in new_critique:
                new_critique = f"{critique}\n{new_critique}\nLet's give the most possible answer.\n\nQuestion: {question}\nProvide a concise response to the question.\n "
                formatted_prompt = prompt.format(
                    question=question,
                    examples=examples,
                    answer=answer,
                    critique=new_critique,
                    **additional_keys,
                )
                answer_response = self.llm(formatted_prompt)
                responses.append(answer_response)

                if self.verbose:
                    log_llm_io(answer_response, f"Final Answer {idx + 1}", self.verbose)

                new_critique = answer_response.output_text
                new_critique = new_critique.split("> Evidence: ")[0]

            new_critique = new_critique.split("Answer: ")[-1].strip()
            finished = True

        return new_critique, external_tool_info, finished, responses

    def _handle_search_query(
        self,
        idx: int,
        question: str,
        search_query: str,
        use_search: bool,
        max_interactions: int,
        query_history: List[str],
        evidence_history: Set[str],
    ) -> Tuple[Dict[str, str], str]:
        """Handle a search query and return results.

        Args:
            idx: Current interaction index
            question: The original question
            search_query: The search query to execute
            use_search: Whether to use search tools
            max_interactions: Maximum number of interactions
            query_history: List of previous search queries
            evidence_history: Set of previously retrieved evidence

        Returns:
            Tuple of (search_result, context)
        """
        if use_search and self.search:
            query_history.append(search_query)
            count = query_history.count(search_query)
            start = count if count < self.num_results else self.num_results - 1

            search_result = {}
            for k in range(start, self.num_results):
                try:
                    search_result = self.search.search(search_query, max_results=k)[
                        "results"
                    ][-1]
                except:
                    search_result = {}

                if (
                    "content" in search_result
                    and search_result["content"] not in evidence_history
                ):
                    evidence_history.add(search_result["content"])
                    break

            if "title" not in search_result and "content" not in search_result:
                context = f"""> Evidence: [] No results found\n\n"""
            else:
                context = f"""> Evidence: [{search_result["title"]}] {search_result["content"][: self.evidence_length]}\n\n"""

            if idx == max_interactions - 2:
                context += f"Let's give the most possible answer.\n\nQuestion: {question}\nProvide a concise response to the question.\n "
        else:
            search_result = {}
            context = """> Evidence: """

        return search_result, context
