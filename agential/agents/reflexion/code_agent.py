from typing import Dict, Any, Optional
import time
import re
from rich.console import Console
from rich.panel import Panel
from agential.core.llm import BaseLLM
from agential.eval.metrics.classification import EM
from agential.utils.general import safe_execute
from agential.agents.base import BaseAgent
from agential.agents.reflexion.prompts import *

console = Console()

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
        content = f"[bold blue]LLM {context}[/bold blue]\n\n[bold green]INPUT:[/bold green]\n{input_text}\n\n[bold yellow]OUTPUT:[/bold yellow]\n{output_text}"
        console.print(Panel(content, title="🤖 LLM Call", border_style="blue"))

    def generate(self, question: str, key: str = "") -> Dict[str, Any]:
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
                full_prompt = self.config["prompt"].format(
                    examples=self.config["fewshot"],
                    reflections=reflections,
                    question=question,
                    scratchpad=scratchpad,
                    max_steps=self.max_steps,
                )
                response = self.llm(full_prompt)
                self.log_llm_io(response, f"Trial {trial}, Step {idx}")
                response_text = response.output_text
                # Parse
                lines = response_text.split("\n")
                thought = ""
                action_raw = ""
                for line in lines:
                    if line.strip().startswith("Thought"):
                        thought = (
                            line.split(":", 1)[1].strip()
                            if ":" in line
                            else line.split(" ", 1)[1].strip()
                        )
                    elif line.strip().startswith("Action"):
                        action_raw = (
                            line.split(":", 1)[1].strip()
                            if ":" in line
                            else line.split(" ", 1)[1].strip()
                        )
                        break
                scratchpad += f"\nThought {idx}: {thought}"
                scratchpad += f"\nAction {idx}: {action_raw}"
                # Parse action
                lines_action = action_raw.strip().split("\n")
                first_line = lines_action[0].strip()
                bracket_match = re.match(r"^(\w+)\[", first_line)
                if bracket_match:
                    action_type = bracket_match.group(1)
                    query_lines = [
                        line for line in lines_action[1:] if line.strip() != "]"
                    ]
                    query = "\n".join(query_lines).strip()
                else:
                    action_type = first_line.split()[0] if first_line else ""
                    query = "\n".join(lines_action[1:]).strip()
                # Handle observation
                scratchpad += f"\nObservation {idx}: "
                code = query
                if "```python" in code:
                    code = code.split("```python")[-1].split("```", 1)[0].strip()
                if action_type.lower() == "finish":
                    self._answer = code
                    obs, finished = code, True
                elif action_type.lower() == "implement":
                    _, execution_status = safe_execute(code)
                    self._answer = code
                    obs, finished = (
                        f"```python\n{code}\n```\nExecution Status: {execution_status}",
                        False,
                    )
                elif action_type.lower() == "test":
                    if not self._answer:
                        obs, finished = (
                            "No code implemented yet. Please use Implement action first.",
                            False,
                        )
                    else:
                        test_code = f"{self._answer}\n\n{code}"
                        _, execution_status = safe_execute(test_code)
                        obs, finished = (
                            f"```python\n{test_code}\n```\nExecution Status: {execution_status}",
                            False,
                        )
                else:
                    obs, finished = (
                        "Invalid Action. Valid Actions are Implement[[code]], Test[[code]], and Finish[[answer]].",
                        False,
                    )
                scratchpad += obs
                if finished:
                    answer = code
                # Metrics
                step_tokens = response.total_tokens
                step_cost = response.total_cost
                step_time = time.time() - step_start
                total_tokens += step_tokens
                total_cost += step_cost
                trial_tokens += step_tokens
                trial_cost += step_cost
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
            correct = EM(answer, key, normalize=False) if key else False
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
                reflection_prompt = self.config["reflect_prompt"].format(
                    examples=self.config["reflect_examples"],
                    question=question,
                    scratchpad=scratchpad,
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
            "answer": answer,
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
