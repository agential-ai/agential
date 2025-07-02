"""
ReAct Math Agent for mathematical problem-solving benchmarks.
"""

from typing import Dict, Any, Optional
import time
from rich.console import Console
from agential.core.llm import BaseLLM
from agential.utils.general import safe_execute
from agential.methods.base import BaseMethod
from agential.methods.react.prompts import *
from agential.methods.react.utils import (
    parse_thought,
    parse_action,
    log_llm_io,
)
import re

console = Console()


class ReActMath(BaseMethod):
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

    def generate(
        self,
        question: str,
        key: str = "",
        additional_keys: dict = {},
        max_llm_retries: int = 3,
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
            thought = parse_thought(thought_response.output_text)
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
            action_type, query = parse_action(action_response.output_text, "math")
            scratchpad += f"\nAction {idx}: {action_type}[{query}]"

            # Compute observation for this step only
            if action_type.lower() == "finish":
                obs, finished = query, True
            elif action_type.lower() == "calculate":
                code = query
                if "```python" in code:
                    code = code.split("```python")[-1].split("```", 1)[0].strip()
                code_with_imports = f"from typing import *\n{code}"
                code_answer, execution_status = safe_execute(code_with_imports)
                answer = code
                obs = f"\n```python\n{answer}\n```\nExecution Status: {execution_status}\nOutput: answer = {code_answer[0]}"
                finished = False
            else:
                obs, finished = (
                    "Invalid Action. Valid Actions are Calculate[code] and Finish[answer].",
                    False,
                )
            # Only append the current step's observation
            scratchpad += f"\nObservation {idx}: {obs}"
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
