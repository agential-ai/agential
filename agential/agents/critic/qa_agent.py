"""CRITIC QA Agent for question-answering benchmarks."""

from typing import Any, Dict, List, Optional, Set, Tuple
import time
from rich.console import Console

from tavily import TavilyClient

from agential.agents.base import BaseAgent
from agential.agents.critic.output import CriticOutput, CriticStepOutput

from agential.agents.critic.utils import (
    log_llm_io,
)
from agential.core.llm import BaseLLM, Response

console = Console()


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
        search: Optional[TavilyClient] = None,
        evidence_length: int = 400,
        num_results: int = 8,
        max_interactions: int = 7,
        use_tool: bool = True,
        verbose: bool = False,
        config: dict = {},
    ):
        """Initialize the CRITIC QA Agent.
        
        Args:
            llm: The language model to use for generation
            benchmark: The benchmark name (hotpotqa, fever, triviaqa, ambignq)
            search: Optional search API wrapper for obtaining evidence
            evidence_length: Maximum length of evidence snippets
            num_results: Number of search results to retrieve
            max_interactions: Maximum number of critique cycles
            use_tool: Whether to use external search tools
            verbose: Whether to enable verbose logging
            config: Additional configuration
        """
        super().__init__(llm=llm, benchmark=benchmark, verbose=verbose, config=config)
        self.search = search
        self.evidence_length = evidence_length
        self.num_results = num_results
        self.max_interactions = max_interactions
        self.use_tool = use_tool
        
        # Initialize search history tracking
        self._query_history: List[str] = []
        self._evidence_history: Set[str] = set()

    def _build_agent_prompt(
        self,
        question: str,
        examples: str,
        prompt: str,
        additional_keys: Dict[str, str] = {},
    ) -> str:
        """Builds a prompt for questioning the agent using a template.

        Args:
            question: The question to be answered by the agent.
            examples: Contextual examples related to the question.
            prompt: Prompt template string.
            additional_keys: Additional keys to format the prompt.

        Returns:
            A formatted prompt ready for use with the language model.
        """
        prompt = prompt.format(question=question, examples=examples, **additional_keys)
        return prompt

    def _prompt_agent(
        self,
        question: str,
        examples: str,
        prompt: str,
        additional_keys: Dict[str, str] = {},
    ) -> Response:
        """Prompts the agent to answer a question using the language model.

        Args:
            question: The question to be answered.
            examples: Contextual examples relevant to the question.
            prompt: Prompt template string.
            additional_keys: Additional keys to format the prompt.

        Returns:
            The answer from the language model, with no leading or trailing whitespace.
        """
        prompt = self._build_agent_prompt(
            question=question,
            examples=examples,
            prompt=prompt,
            additional_keys=additional_keys,
        )

        out = self.llm(prompt)
        return out

    def _build_critique_prompt(
        self,
        question: str,
        examples: str,
        answer: str,
        critique: str,
        prompt: str,
        additional_keys: Dict[str, str] = {},
    ) -> str:
        """Builds a critique prompt for the agent using a template.

        Args:
            question: The original question related to the answer.
            examples: Contextual examples used in the question.
            answer: The agent's answer to the question.
            critique: Additional critique information.
            prompt: Prompt template string.
            additional_keys: Additional keys to format the prompt.

        Returns:
            A formatted critique prompt ready for use with the language model.
        """
        prompt = prompt.format(
            question=question,
            examples=examples,
            answer=answer,
            critique=critique,
            **additional_keys,
        )
        return prompt

    def _prompt_critique(
        self,
        question: str,
        examples: str,
        answer: str,
        critique: str,
        prompt: str,
        additional_keys: Dict[str, str] = {},
    ) -> Response:
        """Prompts the agent for a critique of an answer using the language model.

        Args:
            question: The question related to the answer.
            examples: Contextual examples related to the question.
            answer: The answer to critique.
            critique: Initial critique to refine the response.
            prompt: Prompt template string.
            additional_keys: Additional keys to format the prompt.

        Returns:
            The critique from the language model, with no leading or trailing whitespace.
        """
        prompt = self._build_critique_prompt(
            question=question,
            examples=examples,
            answer=answer,
            critique=critique,
            prompt=prompt,
            additional_keys=additional_keys,
        )
        out = self.llm(prompt)
        return out

    def generate(
        self,
        question: str,
        additional_keys: Dict[str, str] = {},
        max_interactions: Optional[int] = None,
        use_tool: Optional[bool] = None,
        reset: bool = True,
    ) -> CriticOutput:
        """Generate an answer using the CRITIC methodology.
        
        Args:
            question: The question to answer
            additional_keys: Additional keys to format prompts
            max_interactions: Override max_interactions from init
            use_tool: Override use_tool from init
            reset: Whether to reset the agent's state
            
        Returns:
            CriticOutput: The final answer and critique information
        """
        if reset:
            self.reset()
        
        max_interactions = max_interactions or self.max_interactions
        use_tool = use_tool if use_tool is not None else self.use_tool
        
        start_time = time.time()
        steps = []
        total_tokens = total_cost = 0
        step_metrics = []
        
        # Get prompts and examples from config
        prompt = self.config["prompt"]
        critique_prompt = self.config["critique_prompt"]
        examples = self.config["examples"]
        
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
            new_critique, external_tool_info, finished, critique_responses = self._generate_critique(
                idx=idx,
                question=question,
                examples=examples,
                answer=answer,
                critique=critique,
                prompt=critique_prompt,
                additional_keys=additional_keys,
                use_tool=use_tool,
                max_interactions=max_interactions,
            )
            
            # Track metrics for critique
            step_tokens = step_cost = 0
            for response in critique_responses:
                step_tokens += response.total_tokens
                step_cost += response.total_cost
                total_tokens += response.total_tokens
                total_cost += response.total_cost
            
            step_time = time.time() - step_start
            step_metrics.append({
                "step": idx + 1,
                "total_step_time": step_time,
                "total_step_tokens": step_tokens,
                "total_step_cost": step_cost,
            })
            
            # Create step output
            step_output = CriticStepOutput(
                answer=answer,
                critique=new_critique,
                external_tool_info=external_tool_info,
                answer_response=answer_responses,
                critique_response=critique_responses,
            )
            steps.append(step_output)
            
            # Update critique
            critique = new_critique
            
            # Check if finished
            if finished:
                break
                
            # Update answer based on critique (for QA, we keep the original answer)
            answer, answer_responses = self._update_answer_based_on_critique(
                question=question,
                examples=examples,
                answer=answer,
                critique=critique,
                prompt=prompt,
                additional_keys=additional_keys,
                external_tool_info=external_tool_info,
            )
        
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
        
        return CriticOutput(
            answer=final_answer,
            additional_info=steps,
            metrics=metrics,
        )

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
        response = self._prompt_agent(
            question=question,
            examples=examples,
            prompt=prompt,
            additional_keys=additional_keys,
        )
        
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
        use_tool: bool,
        max_interactions: int,
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
            use_tool: Whether to use search tools
            max_interactions: Maximum number of interactions
            
        Returns:
            Tuple of (critique, external_tool_info, finished, responses)
        """
        external_tool_info = {"search_query": "", "search_result": ""}
        responses = []

        # Generate initial critique
        critique_response = self._prompt_critique(
            question=question,
            examples=examples,
            answer=answer,
            critique=critique,
            prompt=prompt,
            additional_keys=additional_keys,
        )
        responses.append(critique_response)
        
        if self.verbose:
            log_llm_io(critique_response, f"Generate Critique {idx+1}", self.verbose)
        
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
                use_tool=use_tool,
                max_interactions=max_interactions,
            )
            
            new_critique = f"{critique}\n{new_critique}{context}"
            
            # If not using tool, generate critique with search result
            if not use_tool:
                search_result_response = self._prompt_critique(
                    question=question,
                    examples=examples,
                    answer=answer,
                    critique=new_critique,
                    prompt=prompt,
                    additional_keys=additional_keys,
                )
                responses.append(search_result_response)
                
                if self.verbose:
                    log_llm_io(search_result_response, f"Search Result Critique {idx+1}", self.verbose)
                
                search_result_no_tool = search_result_response.output_text
                search_result_no_tool = search_result_no_tool.split("> Evidence: ")[0]
                new_critique = f"{critique}\n{new_critique}{search_result_no_tool.strip()}"
            
            external_tool_info["search_query"] = search_query
            # Fix: always assign a string to external_tool_info["search_result"]
            if use_tool:
                # Convert dict to string for logging/record
                external_tool_info["search_result"] = str(search_result)
            else:
                external_tool_info["search_result"] = search_result_no_tool
        else:
            # No search query suggested, generate final answer
            if "Answer: " not in new_critique:
                new_critique = f"{critique}\n{new_critique}\nLet's give the most possible answer.\n\nQuestion: {question}\nProvide a concise response to the question.\n "
                answer_response = self._prompt_critique(
                    question=question,
                    examples=examples,
                    answer=answer,
                    critique=new_critique,
                    prompt=prompt,
                    additional_keys=additional_keys,
                )
                responses.append(answer_response)
                
                if self.verbose:
                    log_llm_io(answer_response, f"Final Answer {idx+1}", self.verbose)
                
                new_critique = answer_response.output_text
                new_critique = new_critique.split("> Evidence: ")[0]

            new_critique = new_critique.split("Answer: ")[-1].strip()
            finished = True

        return new_critique, external_tool_info, finished, responses

    def _update_answer_based_on_critique(
        self,
        question: str,
        examples: str,
        answer: str,
        critique: str,
        prompt: str,
        additional_keys: Dict[str, str],
        external_tool_info: Dict[str, str],
    ) -> Tuple[str, List[Response]]:
        """Update answer based on critique (for QA, we keep the original answer).
        
        Args:
            question: The original question
            examples: Few-shot examples
            answer: The current answer
            critique: The critique
            prompt: The prompt template
            additional_keys: Additional formatting keys
            external_tool_info: External tool information
            
        Returns:
            Tuple of (updated_answer, responses)
        """
        # For QA tasks, we typically keep the original answer
        # The critique process is more about validation than refinement
        return answer, []

    def _handle_search_query(
        self,
        idx: int,
        question: str,
        search_query: str,
        use_tool: bool,
        max_interactions: int,
    ) -> Tuple[Dict[str, str], str]:
        """Handle a search query and return results.
        
        Args:
            idx: Current interaction index
            question: The original question
            search_query: The search query to execute
            use_tool: Whether to use search tools
            max_interactions: Maximum number of interactions
            
        Returns:
            Tuple of (search_result, context)
        """
        if use_tool:
            if not self.search:
                raise ValueError("Search tool is required but not provided.")

            self._query_history.append(search_query)
            count = self._query_history.count(search_query)
            start = count if count < self.num_results else self.num_results - 1

            search_result = {}
            for k in range(start, self.num_results):
                try:
                    search_result = self.search.search(search_query, max_results=k)["results"][-1]
                except:
                    search_result = {}

                if (
                    "content" in search_result
                    and search_result["content"] not in self._evidence_history
                ):
                    self._evidence_history.add(search_result["content"])
                    break

            if "title" not in search_result and "content" not in search_result:
                context = f"""> Evidence: [] No results found\n\n"""
            else:
                context = f"""> Evidence: [{search_result['title']}] {search_result['content'][:self.evidence_length]}\n\n"""
                
            if idx == max_interactions - 2:
                context += f"Let's give the most possible answer.\n\nQuestion: {question}\nProvide a concise response to the question.\n "
        else:
            search_result = {}
            context = """> Evidence: """

        return search_result, context

    def reset(self) -> None:
        """Reset the agent's internal state."""
        self._query_history = []
        self._evidence_history = set() 