from typing import Dict, Any, Optional, Tuple
import time
import re
from agential.core.llm import BaseLLM
from agential.utils.docstore import DocstoreExplorer
from langchain_community.docstore.wikipedia import Wikipedia
from agential.eval.classification import EM, fuzzy_EM
from agential.agents.base import BaseAgent
from agential.agents.reflexion.prompts import *
from agential.agents.reflexion.utils import (
    parse_llm_response,
    parse_action_string,
    log_llm_io,
)


class ReflexionQA(BaseAgent):
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
        self.docstore = DocstoreExplorer(Wikipedia())

    def generate(
        self,
        question: str,
        key: str = "",
        additional_keys: dict = {},
        reflect_additional_keys: dict = {},
        max_llm_retries: int = 3,
        prompt: Optional[str] = None,
        fewshot: Optional[str] = None,
        reflect_prompt: Optional[str] = None,
        reflect_fewshot: Optional[str] = None,
    ) -> Dict[str, Any]:
        start_time = time.time()
        total_tokens = total_cost = 0
        reflections = ""
        all_trials = []
        
        # Use provided parameters or fall back to config
        prompt = prompt or self.config["prompt"]
        fewshot = fewshot or self.config["fewshot"]
        reflect_prompt = reflect_prompt or self.config["reflect_prompt"]
        reflect_fewshot = reflect_fewshot or self.config["reflect_examples"]
        
        for trial in range(1, self.max_trials + 1):
            trial_start = time.time()
            trial_tokens = trial_cost = 0
            scratchpad, answer, steps, step_metrics = "", "", [], []
            finished = False
            for idx in range(1, self.max_steps + 1):
                step_start = time.time()
                prompt_kwargs = dict(
                    examples=fewshot,
                    reflections=reflections,
                    question=question,
                    scratchpad=scratchpad,
                    max_steps=self.max_steps,
                )
                prompt_kwargs.update(additional_keys)
                full_prompt = prompt.format(**prompt_kwargs)
                for llm_attempt in range(max_llm_retries):
                    response = self.llm(full_prompt)
                    log_llm_io(
                        response,
                        f"Trial {trial}, Step {idx}",
                        self.verbose,
                        self.truncate_length,
                    )
                    response_text = response.output_text
                    thought, action_type, query = parse_llm_response(response_text)
                    if thought and action_type:
                        break
                scratchpad += f"\nThought {idx}: {thought}"
                scratchpad += f"\nAction {idx}: {action_type}[{query}]"
                # Parse action using the new function
                action_type, query = parse_action_string(f"{action_type}[{query}]")
                scratchpad += f"\nObservation {idx}: "
                if action_type.lower() == "finish":
                    obs, finished = query, True
                elif action_type.lower() == "search":
                    try:
                        obs = self.docstore.search(query).replace("\n", " ")
                    except Exception:
                        obs = "Could not find that page, please try again."
                    finished = False
                elif action_type.lower() == "lookup":
                    try:
                        obs = self.docstore.lookup(query).replace("\n", " ")
                    except ValueError:
                        obs = "The last page Searched was not found, so you cannot Lookup a keyword in it. Please try one of the similar pages given."
                    finished = False
                else:
                    obs, finished = (
                        "Invalid Action. Valid Actions are Search[entity], Lookup[keyword], and Finish[answer].",
                        False,
                    )
                scratchpad += obs
                if finished:
                    answer = query
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
                correct = EM(answer, key)
                if not correct:
                    correct = fuzzy_EM(answer, key)
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
                    examples=reflect_fewshot,
                    question=question,
                    scratchpad=scratchpad,
                )
                reflect_kwargs.update(reflect_additional_keys)
                reflection_prompt = reflect_prompt.format(
                    **reflect_kwargs
                )
                reflection = self.llm(reflection_prompt)
                log_llm_io(
                    reflection,
                    "Reflection Generation",
                    self.verbose,
                    self.truncate_length,
                )
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
