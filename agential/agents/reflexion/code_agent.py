from typing import Dict, Any, Optional, Tuple
import time
import re
from rich.console import Console
from rich.panel import Panel
from rich.markup import escape
from agential.core.llm import BaseLLM
from agential.utils.general import safe_execute
from agential.agents.base import BaseAgent
from agential.agents.reflexion.prompts import *

console = Console()


def parse_llm_response(response_text: str) -> Tuple[str, str, str]:
    """
    Parse LLM response to extract thought, action_type, and query.

    Args:
        response_text: The raw text response from the LLM

    Returns:
        Tuple of (thought, action_type, query) where each can be empty string if parsing fails
    """
    # Primary parsing with strict regex
    thought_match = re.search(r"Thought.*?:\s*(.*?)(?:\n|$)", response_text, re.DOTALL)
    action_match = re.search(
        r"Action.*?:\s*([\w]+)\[(.*?)]\s*(?:\n|$)", response_text, re.DOTALL
    )

    thought = thought_match.group(1).strip() if thought_match else ""
    action_type = action_match.group(1) if action_match else ""
    query = action_match.group(2).strip() if action_match else ""

    # Fallback parsing if primary parsing failed
    if not thought:
        thought_match = re.search(r"Thought.*?:\s*(.*)", response_text, re.DOTALL)
        if thought_match:
            thought = thought_match.group(1).strip()

    if not action_type:
        action_fallback = re.search(r"Action.*?:\s*(.*)", response_text, re.DOTALL)
        action_raw = action_fallback.group(1).strip() if action_fallback else ""

        # Try to parse action with various patterns
        match = re.match(r"^(\w+)\[(.*)\]$", action_raw.strip(), re.DOTALL)
        if match:
            action_type = match.group(1)
            query = match.group(2).strip()
        else:
            match = re.match(r"^(\w+)\[(.*)", action_raw.strip(), re.DOTALL)
            if match:
                action_type = match.group(1)
                query = match.group(2).strip()
            else:
                # Last resort: split on whitespace
                action_type = (
                    action_raw.strip().split()[0] if action_raw.strip() else ""
                )
                query = action_raw.strip()[len(action_type) :].strip()

    return thought, action_type, query


def parse_action_string(action_string: str) -> Tuple[str, str]:
    """
    Parse action string to extract action_type and query.

    Args:
        action_string: String in format "action_type[query]" or similar

    Returns:
        Tuple of (action_type, query)
    """
    # Try to parse with various patterns
    match = re.match(r"^(\w+)\[(.*)\]$", action_string.strip(), re.DOTALL)
    if match:
        return match.group(1), match.group(2).strip()

    match = re.match(r"^(\w+)\[(.*)", action_string.strip(), re.DOTALL)
    if match:
        return match.group(1), match.group(2).strip()

    # Last resort: split on whitespace
    action_type = action_string.strip().split()[0] if action_string.strip() else ""
    query = action_string.strip()[len(action_type) :].strip()
    return action_type, query


class ReflexionCode(BaseAgent):
    def __init__(
        self,
        llm: BaseLLM,
        benchmark: str,
        max_steps: int = 3,
        max_trials: int = 1,
        max_reflections: int = 2,
        reflect_strategy: Optional[str] = "last_attempt_and_reflexion",
        truncate_length: Optional[int] = None,
        verbose: bool = False,
        config: dict = {},
    ):
        super().__init__(llm=llm, benchmark=benchmark, verbose=verbose, config=config)
        self.max_steps = max_steps
        self.max_trials = max_trials
        self.max_reflections = max_reflections
        self.reflect_strategy = reflect_strategy
        self.truncate_length = truncate_length
        self.verbose = verbose
        self._answer = ""

    def log_llm_io(self, response, context: str = ""):
        if not self.verbose:
            return
        input_text = str(response.input_text)
        output_text = response.output_text
        if self.truncate_length is not None:
            if len(input_text) > self.truncate_length:
                input_text = input_text[: self.truncate_length] + "..."
            if len(output_text) > self.truncate_length:
                output_text = output_text[: self.truncate_length] + "..."
        # Escape context for rich markup
        context_escaped = escape(context)
        content = f"[bold blue]LLM {context_escaped}[/bold blue]\n\n[bold green]INPUT:[/bold green]\n{input_text}\n\n[bold yellow]OUTPUT:[/bold yellow]\n{output_text}"
        console.print(Panel(content, title="🤖 LLM Call", border_style="blue"))

    def generate(
        self,
        question: str,
        key: str = "",
        additional_keys: dict = {},
        reflect_additional_keys: dict = {},
        max_llm_retries: int = 3,
    ) -> Dict[str, Any]:
        start_time = time.time()
        total_tokens = total_cost = 0
        reflections = ""
        all_trials = []
        for trial in range(1, self.max_trials + 1):
            trial_start = time.time()
            trial_tokens = trial_cost = 0
            scratchpad, answer, steps, step_metrics = "", "", [], []
            finished = False
            for idx in range(1, self.max_steps + 1):
                step_start = time.time()
                prompt_kwargs = dict(
                    examples=self.config["fewshot"],
                    reflections=reflections,
                    question=question,
                    scratchpad=scratchpad,
                    max_steps=self.max_steps,
                )
                prompt_kwargs.update(additional_keys)
                full_prompt = self.config["prompt"].format(**prompt_kwargs)
                for _ in range(max_llm_retries):
                    response = self.llm(full_prompt)
                    self.log_llm_io(response, f"Trial {trial}, Step {idx}")
                    response_text = response.output_text
                    thought, action_type, query = parse_llm_response(response_text)
                    if thought and action_type:
                        break
                scratchpad += f"\nThought {idx}: {thought}"
                scratchpad += f"\nAction {idx}: {action_type}[{query}]"
                scratchpad += f"\nObservation {idx}: "
                code = query
                if "```python" in code:
                    code = code.split("```python")[-1].split("```", 1)[0].strip()
                if action_type.lower() == "finish":
                    self._answer = code
                    obs, finished = code, True
                    answer = code
                elif action_type.lower() == "implement":
                    _, execution_status = safe_execute(f"from typing import *\n{code}")
                    self._answer = code
                    obs, finished = (
                        f"```python\n{code}\n```\nExecution Status: {execution_status}",
                        False,
                    )
                    if code:
                        answer = code
                elif action_type.lower() == "test":
                    if not self._answer:
                        obs, finished = (
                            "No code implemented yet. Please use Implement action first.",
                            False,
                        )
                    else:
                        test_code = f"from typing import *\n{self._answer}\n\n{code}"
                        _, execution_status = safe_execute(test_code)
                        obs, finished = (
                            f"```python\n{test_code}\n```\nExecution Status: {execution_status}",
                            False,
                        )
                else:
                    obs, finished = (
                        "Invalid Action. Valid Actions are Implement[code], Test[code], and Finish[answer].",
                        False,
                    )
                scratchpad += obs
                if finished:
                    answer = code
                step_tokens = response.total_tokens
                step_cost = response.total_cost
                step_time = time.time() - step_start
                total_tokens += step_tokens
                total_cost += step_cost
                step_tokens += step_tokens
                step_cost += step_cost
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
            correct = False
            if key and answer:
                try:
                    # For code tasks, the key contains test cases
                    test_code = f"from typing import *\n{answer}\n\n{key}"
                    _, execution_status = safe_execute(test_code)
                    # If execution succeeds without errors, consider it correct
                    correct = execution_status == "Done"
                except Exception:
                    correct = False
            else:
                correct = False
            should_reflect = (
                self.reflect_strategy and not correct and trial < self.max_trials
            )
            trial_time = time.time() - trial_start
            trial_data = {
                "trial": trial,
                "answer": answer,
                "correct": correct,
                "steps": steps,
                "scratchpad": scratchpad,
                "trial_time": trial_time,
                "trial_tokens": trial_tokens,
                "trial_cost": trial_cost,
                "step_metrics": step_metrics,
            }
            all_trials.append(trial_data)
            if should_reflect:
                reflect_kwargs = dict(
                    examples=self.config["reflect_examples"],
                    question=question,
                    scratchpad=scratchpad,
                )
                reflect_kwargs.update(reflect_additional_keys)
                reflection_prompt = self.config["reflect_prompt"].format(
                    **reflect_kwargs
                )
                reflection = self.llm(reflection_prompt)
                self.log_llm_io(reflection, "Reflection Generation")
                reflections += (
                    f"\n\nReflection {trial}: {reflection.output_text.strip()}"
                )
            if correct:
                break
        total_time = time.time() - start_time
        return {
            "answer": f"\n```python\n{answer}\n```\n",
            "correct": correct,
            "trials": all_trials,
            "reflections": reflections,
            "metrics": {
                "total_time": total_time,
                "total_tokens": total_tokens,
                "total_cost": total_cost,
                "trials_taken": len(all_trials),
            },
        }
