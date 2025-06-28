from typing import Dict, Any, Optional
import time
from agential.core.llm import BaseLLM
from agential.utils.general import safe_execute
from agential.eval.classification import EM
from agential.agents.base import BaseAgent
from agential.agents.reflexion.prompts import *
from agential.agents.react.utils import (
    parse_thought,
    parse_action,
    log_llm_io,
)
import re


class ReflexionCode(BaseAgent):
    def __init__(
        self,
        llm: BaseLLM,
        benchmark: str,
        max_steps: int = 6,
        truncate_length: int = -1,
        verbose: bool = False,
        config: dict = {},
        max_parse_retries: int = 3,
        reflect_strategy: str = "reflexion",
        max_reflections: int = 3,
        max_trials: int = 3,
    ):
        super().__init__(llm=llm, benchmark=benchmark, verbose=verbose, config=config)
        self.max_steps = max_steps
        self.truncate_length = truncate_length
        self.max_parse_retries = max_parse_retries
        self.reflect_strategy = reflect_strategy
        self.max_reflections = max_reflections
        self.max_trials = max_trials
        self.reflections = []
        self.reflections_str = ""
        self._answer = ""

    def _format_last_attempt(self, question, scratchpad):
        return f"Last Attempt:\nQuestion: {question}\n{scratchpad}"

    def _format_reflections(self, reflections, header="Reflections:"):
        if not reflections:
            return ""
        return header + "\n" + "\n".join(reflections)

    def _react_reflect_last_attempt(self, scratchpad):
        return [scratchpad], None

    def _react_reflect_reflexion(
        self, question, examples, scratchpad, prompt, additional_keys
    ):
        reflect_prompt = prompt.format(
            question=question,
            examples=examples,
            scratchpad=scratchpad,
            **additional_keys,
        )
        out = self.llm(reflect_prompt)
        new_reflection = out.output_text.strip().replace("\n", " ")
        reflections = self.reflections + [new_reflection]
        return reflections, out

    def _react_reflect_last_attempt_and_reflexion(
        self, question, examples, scratchpad, prompt, additional_keys
    ):
        reflect_prompt = prompt.format(
            question=question,
            examples=examples,
            scratchpad=scratchpad,
            **additional_keys,
        )
        out = self.llm(reflect_prompt)
        new_reflection = out.output_text.strip().replace("\n", " ")
        reflections = [new_reflection]
        return reflections, out

    def generate(
        self,
        question: str,
        key: str = "",
        additional_keys: dict = {},
        max_llm_retries: int = 3,
        prompt: Optional[str] = None,
        fewshot: Optional[str] = None,
        reflect_fewshot: Optional[str] = None,
        reflect_prompt: Optional[str] = None,
        reflect_additional_keys: dict = {},
    ) -> Dict[str, Any]:
        start_time = time.time()
        total_tokens = total_cost = 0
        all_trials = []
        self.reflections = []
        self.reflections_str = ""
        prompt = prompt or self.config["prompt"]
        fewshot = fewshot or self.config["fewshot"]
        reflect_fewshot = reflect_fewshot or self.config.get("reflect_fewshot", "")
        reflect_prompt = reflect_prompt or self.config.get("reflect_prompt", "")
        answer = ""
        correct = False
        test_passed = False
        for trial in range(1, self.max_trials + 1):
            scratchpad, steps, step_metrics = "", [], []
            finished = False
            for idx in range(1, self.max_steps + 1):
                step_start = time.time()
                # 1. Generate Thought
                thought_prompt_kwargs = dict(
                    examples=fewshot,
                    question=question,
                    scratchpad=scratchpad,
                    max_steps=self.max_steps,
                    reflections=self.reflections_str,
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
                if not thought:
                    # Check if the LLM went straight to action
                    if "Action" in thought_response.output_text:
                        thought = "Continuing with action..."
                    else:
                        # Fallback: try to extract any meaningful content
                        raw_text = thought_response.output_text.strip()
                        if raw_text:
                            # Take the first line or first sentence as thought
                            thought = raw_text.split("\n")[0].split(".")[0].strip()
                            if not thought:
                                thought = "Thinking about the problem..."
                        else:
                            thought = "Thinking about the problem..."
                scratchpad += f"\nThought {idx}: {thought}"

                # 2. Generate Action
                action_prompt_kwargs = dict(
                    examples=fewshot,
                    question=question,
                    scratchpad=scratchpad,
                    max_steps=self.max_steps,
                    reflections=self.reflections_str,
                )
                action_prompt_kwargs.update(additional_keys)
                action_prompt = (
                    prompt.format(**action_prompt_kwargs) + f"\nAction {idx}:"
                )
                action_response = self.llm(action_prompt)
                log_llm_io(
                    action_response,
                    f"Step {idx} - Action",
                    self.verbose,
                    self.truncate_length,
                )
                action_text = action_response.output_text.strip()
                action_type, query = parse_action(action_text, benchmark_type="code")
                if not action_type:
                    # Fallback: try to extract action from the text
                    if "Action" in action_text:
                        # Try to find action in the text
                        action_match = re.search(
                            r"Action\s*\d*:\s*(\w+)\[?(.*?)\]?", action_text, re.DOTALL
                        )
                        if action_match:
                            action_type = action_match.group(1)
                            query = action_match.group(2).strip()
                        else:
                            # Last resort: try to find any action-like pattern
                            action_type = "implement"
                            query = action_text
                    else:
                        action_type = "implement"
                        query = action_text
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
                    code_answer, execution_status = safe_execute(code_with_imports)
                    obs = (
                        f"```python\n{code}\n``" + "`\n"
                        f"Execution Status: {execution_status}"
                    )
                    finished = False
                    self._answer = code
                elif action_type.lower() == "test":
                    if not self._answer:
                        obs = "No code to test. Please implement code first."
                        finished = False
                    else:
                        # Extract test code, removing any existing markdown delimiters
                        test_code = query
                        if "```python" in test_code:
                            test_code = (
                                test_code.split("```python")[-1]
                                .split("```", 1)[0]
                                .strip()
                            )

                        # Combine the implemented code with the test code
                        combined_code = (
                            f"from typing import *\n{self._answer}\n\n{test_code}"
                        )
                        _, execution_status = safe_execute(combined_code)
                        obs = f"\n```python\n{combined_code}\n```\nExecution Status: {execution_status}"
                        if execution_status == "Done":
                            test_passed = True
                            finished = True
                            answer = self._answer
                        else:
                            finished = False
                else:
                    obs, finished = (
                        "Invalid Action. Valid Actions are Code[code], Test[code], and Finish[answer].",
                        False,
                    )
                scratchpad += obs
                if finished:
                    answer = query
                step_tokens = (
                    thought_response.total_tokens + action_response.total_tokens
                )
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
                        "thought_parse_retries": 0,
                        "action_parse_retries": 0,
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
            # Reflection logic (after trial, if not correct)
            if self.reflect_strategy == "last_attempt":
                self.reflections, _ = self._react_reflect_last_attempt(scratchpad)
                self.reflections_str = self._format_last_attempt(question, scratchpad)
            elif self.reflect_strategy == "reflexion":
                self.reflections, _ = self._react_reflect_reflexion(
                    question,
                    reflect_fewshot,
                    scratchpad,
                    reflect_prompt,
                    reflect_additional_keys,
                )
                self.reflections = self.reflections[-self.max_reflections :]
                self.reflections_str = self._format_reflections(self.reflections)
            elif self.reflect_strategy == "last_attempt_and_reflexion":
                self.reflections, _ = self._react_reflect_last_attempt_and_reflexion(
                    question,
                    reflect_fewshot,
                    scratchpad,
                    reflect_prompt,
                    reflect_additional_keys,
                )
                self.reflections = self.reflections[-self.max_reflections :]
                self.reflections_str = self._format_last_attempt(question, scratchpad)
                self.reflections_str += "\n" + self._format_reflections(
                    self.reflections, header="Reflections after last trial:"
                )
            else:
                raise NotImplementedError(
                    f"Unknown reflection strategy: {self.reflect_strategy}."
                )

            # Check if answer is correct and halt if so
            if finished and test_passed:
                correct = True
                break

            all_trials.append(
                {
                    "answer": answer,
                    "steps": steps,
                    "scratchpad": scratchpad,
                    "step_metrics": step_metrics,
                }
            )
        total_time = time.time() - start_time
        return {
            "answer": answer,
            "correct": correct,
            "steps": steps,
            "scratchpad": scratchpad,
            "trials": all_trials,
            "metrics": {
                "total_time": total_time,
                "total_tokens": total_tokens,
                "total_cost": total_cost,
                "step_metrics": step_metrics,
                "trials_taken": len(all_trials),
            },
            "reflections": self.reflections,
            "reflections_str": self.reflections_str,
        }
