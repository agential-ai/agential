"""
LATS QA Agent for question-answering benchmarks (simplified version).
"""

from typing import Dict, Any, List, Optional, Tuple
import time
from rich.console import Console
from agential.core.llm import BaseLLM, Response
from agential.utils.docstore import DocstoreExplorer
from langchain_community.docstore.wikipedia import Wikipedia
from agential.agents.base import BaseAgent
from agential.agents.lats.prompts import *
from agential.agents.lats.utils import (
    parse_llm_response,
    parse_action_string,
    parse_qa_action,
    parse_value,
    log_llm_io,
)
from agential.eval.classification import EM
from agential.utils.parse import remove_newline

console = Console()

class LATSQA(BaseAgent):
    """Simplified LATS QA Agent that implements tree search directly."""
    
    def __init__(
        self,
        llm: BaseLLM,
        benchmark: str,
        max_steps: int = 6,
        n_samples: int = 3,
        max_reflections: int = 2,
        depth_limit: int = 4,
        cache_values: bool = True,
        truncate_length: int = -1,
        verbose: bool = False,
        config: dict = {},
    ):
        super().__init__(llm=llm, benchmark=benchmark, verbose=verbose, config=config)
        self.max_steps = max_steps
        self.n_samples = n_samples
        self.max_reflections = max_reflections
        self.depth_limit = depth_limit
        self.cache_values = cache_values
        self.truncate_length = truncate_length
        self.verbose = verbose
        self.docstore = DocstoreExplorer(Wikipedia())
        
        # State for tree search
        self.failed_trajectories: List[Dict[str, str]] = []
        self.value_cache: Dict[str, float] = {}

    def generate(
        self,
        question: str,
        key: str = "",
        additional_keys: dict = {},
        max_llm_retries: int = 3,
    ) -> Dict[str, Any]:
        start_time = time.time()
        total_tokens = total_cost = 0
        scratchpad, answer, steps, step_metrics = "", "", [], []
        finished = False
        
        # Get prompts and examples
        examples = self.config.get("fewshot", "")
        prompt = self.config.get("prompt", "")
        reflect_prompt = self.config.get("reflect_prompt", "")
        value_prompt = self.config.get("value_prompt", "")
        
        # Initialize tree search state
        current_trajectory = ""
        reflections = ""
        
        for step_idx in range(1, self.max_steps + 1):
            step_start = time.time()
            
            # Generate multiple samples for tree search
            samples = []
            for sample_idx in range(self.n_samples):
                # Build prompt with current trajectory and reflections
                prompt_kwargs = dict(
                    examples=examples,
                    question=question,
                    trajectory=current_trajectory,
                    reflections=reflections,
                    max_steps=self.max_steps,
                )
                prompt_kwargs.update(additional_keys)
                full_prompt = prompt.format(**prompt_kwargs)
                
                # Generate thought and action
                for _ in range(max_llm_retries):
                    response = self.llm(full_prompt)
                    log_llm_io(response, f"Step {step_idx} Sample {sample_idx + 1}", self.verbose, self.truncate_length)
                    response_text = response.output_text
                    thought, action_type, query = parse_llm_response(response_text)
                    if thought and action_type:
                        break
                
                # Generate observation
                obs, done, reward = self._generate_observation(action_type, query, key)
                
                # Calculate value for this sample
                value = self._evaluate_sample(
                    question, current_trajectory, thought, action_type, query, obs, 
                    value_prompt, examples, additional_keys
                )
                
                samples.append({
                    "thought": thought,
                    "action_type": action_type,
                    "query": query,
                    "observation": obs,
                    "done": done,
                    "reward": reward,
                    "value": value,
                    "response": response,
                })
                
                total_tokens += getattr(response, "total_tokens", 0)
                total_cost += getattr(response, "total_cost", 0)
            
            # Select best sample based on value
            best_sample = max(samples, key=lambda x: x["value"])
            
            # Update trajectory
            current_trajectory += f"\nThought {step_idx}: {best_sample['thought']}"
            current_trajectory += f"\nAction {step_idx}: {best_sample['action_type']}[{best_sample['query']}]"
            current_trajectory += f"\nObservation {step_idx}: {best_sample['observation']}"
            
            # Add to steps
            steps.append({
                "thought": best_sample["thought"],
                "action_type": best_sample["action_type"],
                "query": best_sample["query"],
                "observation": best_sample["observation"],
                "answer": best_sample["query"] if best_sample["done"] else "",
            })
            
            # Update metrics
            step_time = time.time() - step_start
            step_metrics.append({
                "step": step_idx,
                "total_step_time": step_time,
                "total_step_tokens": getattr(best_sample["response"], "total_tokens", 0),
                "total_step_cost": getattr(best_sample["response"], "total_cost", 0),
                "value": best_sample["value"],
            })
            
            # Check if finished
            if best_sample["done"]:
                answer = best_sample["query"]
                finished = True
                break
            
            # Generate reflection if needed and not finished
            if step_idx <= self.max_reflections and not finished:
                reflection = self._generate_reflection(
                    question, current_trajectory, reflect_prompt, examples, additional_keys
                )
                reflections += f"\n\nReflection {step_idx}: {reflection}"
                
                # Add failed trajectory for future reference
                self.failed_trajectories.append({
                    "trajectory": current_trajectory,
                    "final_answer": best_sample["query"],
                })

        total_time = time.time() - start_time
        scratchpad = current_trajectory

        return {
            "answer": answer,
            "steps": steps,
            "scratchpad": scratchpad,
            "reflections": reflections,
            "metrics": {
                "total_time": total_time,
                "total_tokens": total_tokens,
                "total_cost": total_cost,
                "step_metrics": step_metrics,
            },
        }

    def _generate_observation(
        self, action_type: str, query: str, key: str
    ) -> Tuple[str, bool, int]:
        """Generate observation for an action."""
        if action_type.lower() == "finish":
            correct = EM(query, key) if key else False
            obs = "Answer is CORRECT" if correct else "Answer is INCORRECT"
            return obs, True, 1 if correct else 0
        elif action_type.lower() == "search":
            try:
                obs = self.docstore.search(query).replace("\n", " ")
            except Exception:
                obs = "Could not find that page, please try again."
            return obs, False, 0
        elif action_type.lower() == "lookup":
            try:
                obs = self.docstore.lookup(query).replace("\n", " ")
            except ValueError:
                obs = "The last page Searched was not found, so you cannot Lookup a keyword in it. Please try one of the similar pages given."
            return obs, False, 0
        else:
            obs = "Invalid Action. Valid Actions are Search[entity], Lookup[keyword], and Finish[answer]."
            return obs, False, 0

    def _evaluate_sample(
        self, question: str, trajectory: str, thought: str, action_type: str, 
        query: str, obs: str, value_prompt: str, examples: str, additional_keys: dict
    ) -> float:
        """Evaluate the value of a sample using the value function."""
        # Create unique key for caching
        sample_key = f"{trajectory}::{thought}::{action_type}::{query}::{obs}"
        
        if self.cache_values and sample_key in self.value_cache:
            return self.value_cache[sample_key]
        
        # Build value prompt
        failed_trajectories = ""
        if self.failed_trajectories:
            for failed in self.failed_trajectories[-2:]:  # Last 2 failed trajectories
                failed_trajectories += f"\n\nFailed Trajectory:\n{failed['trajectory']}"
        
        value_prompt_kwargs = dict(
            question=question,
            examples=examples,
            trajectory=trajectory,
            failed_trajectories=failed_trajectories,
        )
        value_prompt_kwargs.update(additional_keys)
        full_value_prompt = value_prompt.format(**value_prompt_kwargs)
        
        # Get value from LLM
        response = self.llm(full_value_prompt)
        log_llm_io(response, "Value Evaluation", self.verbose, self.truncate_length)
        
        # Parse value
        explanation, value = parse_value(response.output_text)
        value = value / 10.0  # Normalize to 0-1
        
        # Cache the value
        if self.cache_values:
            self.value_cache[sample_key] = value
        
        return value

    def _generate_reflection(
        self, question: str, trajectory: str, reflect_prompt: str, 
        examples: str, additional_keys: dict
    ) -> str:
        """Generate reflection on the current trajectory."""
        reflect_kwargs = dict(
            question=question,
            examples=examples,
            trajectory=trajectory,
        )
        reflect_kwargs.update(additional_keys)
        full_reflect_prompt = reflect_prompt.format(**reflect_kwargs)
        
        response = self.llm(full_reflect_prompt)
        log_llm_io(response, "Reflection Generation", self.verbose, self.truncate_length)
        
        return response.output_text.strip() 