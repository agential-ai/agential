"""CRITIC Code Agent.

Original Paper: https://arxiv.org/pdf/2305.11738
Paper Repository: https://github.com/microsoft/ProphetNet/tree/master/CRITIC
"""

import time
from typing import Any, Dict, List, Optional, Tuple

from agential.agents.base import BaseMethod
from agential.core.llm import BaseLLM, Response
from agential.utils.general import safe_execute
from agential.agents.critic.utils import (
    log_llm_io,
)
from agential.utils.validation import validate_overlapping_keys


class CriticCode(BaseMethod):
    """CRITIC Code Agent for code benchmarks (HumanEval, MBPP).

    This agent implements the CRITIC methodology for code problems, using code execution
    to validate answers and iterative critique to improve solutions.

    Attributes:
        llm (BaseLLM): The language model to use for generation
        benchmark (str): The benchmark name (humaneval, mbpp)
        use_execution (bool): Whether to use code execution for validation
        max_interactions (int): Maximum number of critique cycles
        verbose (bool): Whether to enable verbose logging
        config (dict): Additional configuration
    """

    def __init__(
        self,
        llm: BaseLLM,
        benchmark: str,
        use_execution: bool = True,
        max_interactions: int = 7,
        verbose: bool = False,
        config: dict = {},
    ):
        """Initialize the CRITIC Code Agent.

        Args:
            llm: The language model to use for generation
            benchmark: The benchmark name (humaneval, mbpp)
            use_execution: Whether to use code execution for validation
            max_interactions: Maximum number of critique cycles
            verbose: Whether to enable verbose logging
            config: Additional configuration
        """
        super().__init__(llm=llm, benchmark=benchmark, verbose=verbose, config=config)
        self.use_execution = use_execution
        self.max_interactions = max_interactions

        # Select appropriate critique prompt and examples based on use_execution
        if not use_execution:
            if "critique_prompt_no_tool" in self.config:
                self.config["critique_prompt"] = self.config["critique_prompt_no_tool"]
            if "critique_examples_no_tool" in self.config:
                self.config["critique_examples"] = self.config[
                    "critique_examples_no_tool"
                ]

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
        use_execution: Optional[bool] = None,
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
            use_execution: Override use_execution from init

        Returns:
            dict: The final answer and critique information
        """
        max_interactions = max_interactions or self.max_interactions
        use_execution = (
            use_execution if use_execution is not None else self.use_execution
        )

        # Use provided parameters or fall back to config
        if not prompt or not critique_prompt or not examples or not critique_examples:
            prompt = self.config["prompt"]
            critique_prompt = self.config["critique_prompt"]
            examples = self.config["examples"]
            critique_examples = self.config["critique_examples"]

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
        final_answer = answer

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
                    use_execution=use_execution,
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

            # Update answer based on critique if not finished
            if not finished:
                answer, update_responses = self._update_answer_based_on_critique(
                    question=question,
                    examples=examples,
                    answer=answer,
                    critique=new_critique,
                    prompt=prompt,
                    additional_keys=additional_keys,
                    external_tool_info=external_tool_info,
                )

                # Track metrics for answer update
                for response in update_responses:
                    total_tokens += response.total_tokens
                    total_cost += response.total_cost

                final_answer = answer
            else:
                # If finished, use the current answer
                final_answer = answer
                break

        total_time = time.time() - start_time

        # Calculate metrics
        metrics = {
            "total_time": total_time,
            "total_tokens": total_tokens,
            "total_cost": total_cost,
            "step_metrics": step_metrics,
        }

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

        # Extract code from response
        answer = response.output_text
        answer = answer.split("```python")[-1].split("```")[0].strip("\n")

        return f"\n```python\n{answer}\n```\n", [response]

    def _generate_critique(
        self,
        idx: int,
        question: str,
        examples: str,
        answer: str,
        critique: str,
        prompt: str,
        additional_keys: Dict[str, str],
        use_execution: bool,
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
            use_execution: Whether to use code execution

        Returns:
            Tuple of (critique, external_tool_info, finished, responses)
        """
        external_tool_info = {"execution_status": ""}
        answer_code = answer.split("```python")[-1].split("```")[0].strip()

        finished = False

        if use_execution:
            # Check if tests are provided
            if "tests" not in additional_keys:
                raise ValueError(
                    "The 'tests' parameter must be specified in `critique_additional_keys`."
                )
            tests = additional_keys["tests"]

            # Execute the code with tests - different for HumanEval vs MBPP
            if self.benchmark == "humaneval":
                # HumanEval doesn't include "from typing import *"
                _, execution_status = safe_execute(f"{answer_code}\n\n{tests}")
            else:  # mbpp
                _, execution_status = safe_execute(
                    f"from typing import *\n\n{answer_code}\n\n{tests}"
                )

            if execution_status == "Done":
                finished = True

            external_tool_info = {
                "execution_status": execution_status,
            }

            # Validate overlapping keys
            validate_overlapping_keys(additional_keys, external_tool_info)

        # Update additional keys with external tool info
        additional_keys = additional_keys.copy()
        additional_keys.update(external_tool_info if use_execution else {})

        # Build critique prompt
        formatted_prompt = prompt.format(
            question=question,
            examples=examples,
            answer=answer_code,
            critique="",
            **additional_keys,
        )

        # Generate critique
        critique_response = self.llm(formatted_prompt)

        if self.verbose:
            log_llm_io(critique_response, f"Generate Critique {idx + 1}", self.verbose)

        new_critique = critique_response.output_text

        # Handle different benchmark-specific critique parsing
        if self.benchmark == "humaneval":
            new_critique = new_critique.split("```python")[0].strip("\n")
        else:  # mbpp
            new_critique = new_critique.split("Here's")[0]

        return new_critique, external_tool_info, finished, [critique_response]

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
        """Update the answer based on the critique.

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
        # Validate overlapping keys
        validate_overlapping_keys(additional_keys, external_tool_info)

        # Update additional keys with external tool info
        additional_keys = additional_keys.copy()
        additional_keys.update(external_tool_info)

        answer_code = answer.split("```python")[-1].split("```")[0].strip()

        # Handle different benchmark-specific update prompts
        if self.benchmark == "humaneval":
            critique_prefix = f"{critique}\n\nIf no changes are needed, return the same code.\n```python\n"
        else:  # mbpp
            critique_prefix = f"{critique}\n\nHere's a better solution:\n```python\n"

        # Build update prompt
        formatted_prompt = prompt.format(
            question=question,
            examples=examples,
            answer=answer_code,
            critique=critique_prefix,
            **additional_keys,
        )

        # Generate updated answer
        response = self.llm(formatted_prompt)

        if self.verbose:
            log_llm_io(response, "Update Answer", self.verbose)

        new_answer = response.output_text
        if "```python" in new_answer:
            new_answer = new_answer.split("```python")[-1].split("```")[0].strip()
        else:
            # If no code blocks, try to extract the answer directly
            new_answer = new_answer.strip()

        return f"\n```python\n{new_answer}\n```\n", [response]
