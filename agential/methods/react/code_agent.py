"""
ReAct Code Agent for code generation and testing benchmarks.
"""

from typing import Dict, Any, Optional
import time
from agential.core.llm import BaseLLM
from agential.utils.general import safe_execute
from agential.methods.base import BaseMethod
from agential.methods.react.prompts import *
from agential.methods.react.utils import (
    parse_action,
    log_llm_io,
)
import re


class ReActCode(BaseMethod):
    def __init__(
        self,
        llm: BaseLLM,
        benchmark: str,
        max_steps: int = 6,
        truncate_length: int = -1,
        verbose: bool = False,
        config: dict = {},
    ):
        super().__init__(llm=llm, benchmark=benchmark, verbose=verbose, config=config)
        self.max_steps = max_steps
        self.truncate_length = truncate_length
        self.verbose = verbose
        self._answer = ""

    def generate(
        self,
        question: str,
        key: str = "",
        additional_keys: dict = {},
        prompt: Optional[str] = None,
        fewshot: Optional[str] = None,
    ) -> Dict[str, Any]:
        start_time = time.time()
        total_tokens = total_cost = 0
        scratchpad, answer, steps, step_metrics = "", "", [], []
        finished = False

        # Use provided parameters or fall back to config
        prompt = prompt or self.config["prompt"]
        fewshot = fewshot or self.config["fewshot"]

        for idx in range(1, self.max_steps + 1):
            step_start = time.time()
            # 1. Generate Thought (no retry)
            thought_prompt_kwargs = dict(
                examples=fewshot,
                question=question,
                scratchpad=scratchpad,
                max_steps=self.max_steps,
            )
            thought_prompt_kwargs.update(additional_keys)
            thought_prompt = (
                prompt.format(**thought_prompt_kwargs) + f"\nThought {idx}:"
            )
            thought_response = self.llm(thought_prompt)
            log_llm_io(
                thought_response,
                f"Step {idx} - Thought",
                self.verbose,
                self.truncate_length,
            )
            thought = thought_response.output_text.strip().split("\n")[0]
            # Remove 'Thought <int>:' prefix if present
            thought = re.sub(r"^Thought \d+:\s*", "", thought)
            scratchpad += f"\nThought {idx}: {thought}"

            # 2. Generate Action (no retry)
            action_prompt_kwargs = dict(
                examples=fewshot,
                question=question,
                scratchpad=scratchpad,
                max_steps=self.max_steps,
            )
            action_prompt_kwargs.update(additional_keys)
            action_prompt = prompt.format(**action_prompt_kwargs) + f"\nAction {idx}:"
            action_response = self.llm(action_prompt)
            log_llm_io(
                action_response,
                f"Step {idx} - Action",
                self.verbose,
                self.truncate_length,
            )
            action_block = action_response.output_text.strip()
            # Remove 'Action <int>:' prefix if present
            action_block = re.sub(r"^Action \d+:\s*", "", action_block)
            action_type, query = parse_action(action_block, "code")
            scratchpad += f"\nAction {idx}: {action_type}[{query}]"

            # Continue as before
            scratchpad += f"\nObservation {idx}: "
            if action_type.lower() == "finish":
                self._answer = query
                obs, finished = query, True
            elif action_type.lower() == "implement":
                code = query
                if "```python" in code:
                    code = code.split("```python")[-1].split("```", 1)[0].strip()
                code_with_imports = f"from typing import *\n{code}"
                _, execution_status = safe_execute(code_with_imports)
                self._answer = code
                obs = f"\n```python\n{self._answer}\n```\nExecution Status: {execution_status}"
                finished = False
            elif action_type.lower() == "test":
                if not self._answer:
                    obs = "No code to test. Please implement code first."
                    finished = False
                else:
                    # Extract test code, removing any existing markdown delimiters
                    test_code = query
                    if "```python" in test_code:
                        test_code = (
                            test_code.split("```python")[-1].split("```", 1)[0].strip()
                        )

                    # Combine the implemented code with the test code
                    combined_code = (
                        f"from typing import *\n{self._answer}\n\n{test_code}"
                    )
                    _, execution_status = safe_execute(combined_code)
                    obs = f"\n```python\n{combined_code}\n```\nExecution Status: {execution_status}"
                    finished = False
            else:
                obs, finished = (
                    "Invalid Action. Valid Actions are Implement[code], Test[code], and Finish[answer].",
                    False,
                )
            scratchpad += obs
            if finished:
                answer = query
            step_tokens = thought_response.total_tokens + action_response.total_tokens
            step_cost = thought_response.total_cost + action_response.total_cost
            step_time = time.time() - step_start
            total_tokens += step_tokens
            total_cost += step_cost
            step_metrics.append(
                {
                    "step": idx,
                    "total_step_time": step_time,
                    "total_step_tokens": step_tokens,
                    "total_step_cost": step_cost,
                }
            )
            steps.append(
                {
                    "thought": thought,
                    "action_type": action_type,
                    "query": query,
                    "observation": obs,
                    "answer": answer,
                }
            )
            if finished:
                break

        total_time = time.time() - start_time

        return {
            "answer": answer,
            "steps": steps,
            "scratchpad": scratchpad,
            "metrics": {
                "total_time": total_time,
                "total_tokens": total_tokens,
                "total_cost": total_cost,
                "step_metrics": step_metrics,
            },
        }
