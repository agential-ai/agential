"""CLIN QA Agent.

Paper: https://arxiv.org/pdf/2310.10134
GitHub Repo: https://github.com/allenai/clin
"""

import re
import time
from typing import Any, Dict, List, Optional, Tuple

from langchain_community.docstore.wikipedia import Wikipedia

from agential.agents.base import BaseMethod
from agential.agents.clin.memory import CLINMemory
from agential.core.llm import BaseLLM, Response
from agential.utils.docstore import DocstoreExplorer
from agential.utils.parse import remove_newline
from agential.eval.classification import EM, fuzzy_EM
from agential.agents.react.utils import log_llm_io
from agential.agents.clin.prompts import (
    CLIN_ADAPT_SUMMARY_SYSTEM,
    CLIN_GEN_ENV_SUMMARY_SYSTEM,
    CLIN_GEN_TASK_SUMMARY_SYSTEM,
    CLIN_ADAPT_META_SUMMARY_SYSTEM,
    CLIN_GEN_ENV_META_SUMMARY_SYSTEM,
    CLIN_GEN_TASK_META_SUMMARY_SYSTEM,
)

# Mapping for summary and meta-summary system prompts
CLIN_SUMMARY_SYSTEM = {
    "adapt": CLIN_ADAPT_SUMMARY_SYSTEM,
    "gen_env": CLIN_GEN_ENV_SUMMARY_SYSTEM,
    "gen_task": CLIN_GEN_TASK_SUMMARY_SYSTEM,
}
CLIN_META_SUMMARY_SYSTEM = {
    "adapt": CLIN_ADAPT_META_SUMMARY_SYSTEM,
    "gen_env": CLIN_GEN_ENV_META_SUMMARY_SYSTEM,
    "gen_task": CLIN_GEN_TASK_META_SUMMARY_SYSTEM,
}


def parse_qa_action(string: str) -> Tuple[str, str]:
    """Parses an action string into an action type and its argument.

    Args:
        string (str): The action string to be parsed.

    Returns:
        Tuple[str, str]: A tuple containing the action type and argument.
    """
    pattern = r"^(\w+)\[(.+)\]$"
    match = re.match(pattern, string)

    if match:
        action_type = match.group(1)
        argument = match.group(2)
    else:
        action_type = ""
        argument = ""
    return action_type, argument


class CLINQA(BaseMethod):
    """CLIN QA agent for question answering tasks.

    Attributes:
        llm (BaseLLM): The language model used to generate responses.
        benchmark (str): The benchmark name.
        config (Dict[str, Any]): Configuration dictionary containing prompts and examples.
        memory (CLINMemory): The memory used to store and retrieve information.
        max_trials (int): The maximum number of trials allowed.
        max_steps (int): The maximum number of steps allowed.
        docstore (DocstoreExplorer): The document store for search and lookup operations.
        verbose (bool): Whether to print verbose output.
    """

    def __init__(
        self,
        llm: BaseLLM,
        benchmark: str,
        memory: Optional[CLINMemory] = None,
        max_trials: int = 3,
        max_steps: int = 6,
        docstore: Optional[DocstoreExplorer] = None,
        verbose: bool = False,
        config: dict = {},
        **kwargs,
    ) -> None:
        """Initialize CLIN QA agent."""
        super().__init__(
            llm=llm, benchmark=benchmark, verbose=verbose, config=config, **kwargs
        )

        self.memory = memory or CLINMemory()
        self.max_trials = max_trials
        self.max_steps = max_steps
        self.docstore = docstore or DocstoreExplorer(Wikipedia())

    def generate(
        self,
        question: str,
        key: str,
        examples: str = "",
        prompt: str = "",
        summary_prompt: str = "",
        meta_summary_prompt: str = "",
        additional_keys: Dict[str, str] = {},
        summary_additional_keys: Dict[str, str] = {},
        meta_summary_additional_keys: Dict[str, str] = {},
        summary_system: str = "",
        meta_summary_system: str = "",
        quadrant: str = "adapt",
        patience: int = 3,
        reset: bool = False,
    ) -> Dict[str, Any]:
        """Generate a response to a given question.

        Args:
            question (str): The question to be answered.
            key (str): The key for the question.
            examples (str): The examples for the question. Defaults to "".
            prompt (str): The prompt for the question. Defaults to "".
            summary_prompt (str): The summary prompt for the question. Defaults to "".
            meta_summary_prompt (str): The meta-summary prompt for the question. Defaults to "".
            additional_keys (Dict[str, str]): Additional keys for the question. Defaults to {}.
            summary_additional_keys (Dict[str, str]): Additional keys for the summary. Defaults to {}.
            meta_summary_additional_keys (Dict[str, str]): Additional keys for the meta-summary. Defaults to {}.
            summary_system (str): The system for the summary. Defaults to "".
            meta_summary_system (str): The system for the meta-summary. Defaults to "".
            quadrant (str): The quadrant for the question. Defaults to "adapt".
            patience (int): The patience for the question. Defaults to 3.
            reset (bool): Whether to reset the agent. Defaults to False.

        Returns:
            Dict[str, Any]: The output of the agent.
        """
        start_time = time.time()

        if quadrant not in ["adapt", "gen_env", "gen_task"]:
            raise ValueError(f"Quadrant '{quadrant}' not supported for CLIN.")

        # Use provided parameters or fall back to config
        if not prompt:
            prompt = self.config["prompt"]
        if not examples:
            examples = self.config["examples"]
        if not summary_prompt:
            summary_prompt = self.config["summary_prompt"]
        if not meta_summary_prompt:
            meta_summary_prompt = self.config["meta_summary_prompt"]

        # Set summary_system and meta_summary_system based on quadrant if not provided
        if not summary_system:
            summary_system = CLIN_SUMMARY_SYSTEM.get(
                quadrant, CLIN_SUMMARY_SYSTEM["adapt"]
            )
        if not meta_summary_system:
            meta_summary_system = CLIN_META_SUMMARY_SYSTEM.get(
                quadrant, CLIN_META_SUMMARY_SYSTEM["adapt"]
            )

        # Reset if requested
        if reset:
            self.memory.clear()

        scratchpad = ""
        answer = ""
        finished = False
        idx, step_idx, patience_cnt = 1, 1, 0
        steps = []

        # Load meta-summaries if applicable
        if quadrant == "gen_env" or quadrant == "gen_task":
            meta_summaries = self.memory.load_meta_summaries()["meta_summaries"]
        else:
            meta_summaries = ""

        while not self._halting_condition(idx=idx, key=key, answer=answer):
            # Load previous memories
            previous_memories = self.memory.load_memories(question=question)
            summaries = previous_memories["latest_summaries"]
            previous_trials = previous_memories["previous_trials"]

            # Generate ReAct trial
            step_idx, is_correct, scratchpad, finished, answer, react_steps = (
                self._generate_react(
                    question=question,
                    key=key,
                    examples=examples,
                    summaries=summaries,
                    summary_system=summary_system,
                    meta_summaries=meta_summaries,
                    meta_summary_system=meta_summary_system,
                    prompt=prompt,
                    additional_keys=additional_keys,
                )
            )

            # Generate summaries
            summaries, summaries_response = self._generate_summary(
                question=question,
                previous_trials=previous_trials,
                scratchpad=scratchpad,
                is_correct=is_correct,
                prompt=summary_prompt,
                additional_keys=summary_additional_keys,
            )

            steps.append(
                {
                    "steps": react_steps,
                    "summaries": summaries,
                    "summaries_response": summaries_response,
                    "meta_summaries": meta_summaries,
                    "previous_trials": previous_trials,
                }
            )

            # Increment patience counter
            if not is_correct:
                patience_cnt += 1
            if patience_cnt == patience:
                break

            idx += 1

        # Generate meta-summary
        meta_summaries_response = None
        if quadrant == "gen_env" or quadrant == "gen_task":
            meta_summaries, meta_summaries_response = self._generate_meta_summary(
                question=question,
                meta_summaries=meta_summaries,
                meta_summary_system=meta_summary_system,
                previous_trials=previous_trials,
                scratchpad=scratchpad,
                prompt=meta_summary_prompt,
                additional_keys=meta_summary_additional_keys,
            )

        total_time = time.time() - start_time

        # Calculate metrics directly (inlined from _accumulate_metrics, now incrementally)
        total_prompt_tokens = 0
        total_completion_tokens = 0
        total_tokens = 0
        total_prompt_cost = 0.0
        total_completion_cost = 0.0
        total_cost = 0.0
        total_prompt_time = 0.0

        for step in steps:
            for react_step in step["steps"]:
                total_prompt_tokens += react_step["thought_response"].prompt_tokens
                total_completion_tokens += react_step[
                    "thought_response"
                ].completion_tokens
                total_tokens += react_step["thought_response"].total_tokens
                total_prompt_cost += react_step["thought_response"].prompt_cost
                total_completion_cost += react_step["thought_response"].completion_cost
                total_cost += react_step["thought_response"].total_cost
                total_prompt_time += react_step["thought_response"].prompt_time

                total_prompt_tokens += react_step["action_response"].prompt_tokens
                total_completion_tokens += react_step[
                    "action_response"
                ].completion_tokens
                total_tokens += react_step["action_response"].total_tokens
                total_prompt_cost += react_step["action_response"].prompt_cost
                total_completion_cost += react_step["action_response"].completion_cost
                total_cost += react_step["action_response"].total_cost
                total_prompt_time += react_step["action_response"].prompt_time

            total_prompt_tokens += step["summaries_response"].prompt_tokens
            total_completion_tokens += step["summaries_response"].completion_tokens
            total_tokens += step["summaries_response"].total_tokens
            total_prompt_cost += step["summaries_response"].prompt_cost
            total_completion_cost += step["summaries_response"].completion_cost
            total_cost += step["summaries_response"].total_cost
            total_prompt_time += step["summaries_response"].prompt_time

        if meta_summaries_response is not None:
            total_prompt_tokens += meta_summaries_response.prompt_tokens
            total_completion_tokens += meta_summaries_response.completion_tokens
            total_tokens += meta_summaries_response.total_tokens
            total_prompt_cost += meta_summaries_response.prompt_cost
            total_completion_cost += meta_summaries_response.completion_cost
            total_cost += meta_summaries_response.total_cost
            total_prompt_time += meta_summaries_response.prompt_time

        return {
            "answer": answer,
            "total_prompt_tokens": total_prompt_tokens,
            "total_completion_tokens": total_completion_tokens,
            "total_tokens": total_tokens,
            "total_prompt_cost": total_prompt_cost,
            "total_completion_cost": total_completion_cost,
            "total_cost": total_cost,
            "total_prompt_time": total_prompt_time,
            "total_time": total_time,
            "additional_info": steps,
        }

    def _generate_react(
        self,
        question: str,
        key: str,
        examples: str,
        summaries: str,
        summary_system: str,
        meta_summaries: str,
        meta_summary_system: str,
        prompt: str,
        additional_keys: Dict[str, str],
    ) -> Tuple[int, bool, str, bool, str, List[Dict[str, Any]]]:
        """Generate a ReAct trial."""
        react_steps = []
        step_idx = 1
        scratchpad = ""
        finished = False
        answer = ""

        while not self._react_halting_condition(finished=finished, idx=step_idx):
            # Think
            scratchpad, thought, thought_response = self._generate_thought(
                idx=step_idx,
                scratchpad=scratchpad,
                question=question,
                examples=examples,
                summaries=summaries,
                summary_system=summary_system,
                meta_summaries=meta_summaries,
                meta_summary_system=meta_summary_system,
                prompt=prompt,
                additional_keys=additional_keys,
            )

            # Act
            scratchpad, action_type, query, action_response = self._generate_action(
                idx=step_idx,
                scratchpad=scratchpad,
                question=question,
                examples=examples,
                summaries=summaries,
                summary_system=summary_system,
                meta_summaries=meta_summaries,
                meta_summary_system=meta_summary_system,
                prompt=prompt,
                additional_keys=additional_keys,
            )

            # Observe
            scratchpad, answer, finished, is_correct, obs, external_tool_info = (
                self._generate_observation(
                    idx=step_idx,
                    scratchpad=scratchpad,
                    action_type=action_type,
                    query=query,
                    key=key,
                )
            )

            react_steps.append(
                {
                    "thought": thought,
                    "action_type": action_type,
                    "query": query,
                    "observation": obs,
                    "answer": answer,
                    "external_tool_info": external_tool_info,
                    "is_correct": is_correct,
                    "thought_response": thought_response,
                    "action_response": action_response,
                }
            )

            step_idx += 1

        return step_idx, is_correct, scratchpad, finished, answer, react_steps

    def _generate_thought(
        self,
        idx: int,
        scratchpad: str,
        question: str,
        examples: str,
        summaries: str,
        summary_system: str,
        meta_summaries: str,
        meta_summary_system: str,
        prompt: str,
        additional_keys: Dict[str, str],
    ) -> Tuple[str, str, Response]:
        """Generate a thought."""
        scratchpad += f"\nThought {idx}: "

        # Build prompt directly (inlined from _build_react_agent_prompt)
        formatted_prompt = prompt.format(
            question=question,
            examples=examples,
            summaries=summaries,
            scratchpad=scratchpad,
            max_steps=self.max_steps,
            summary_system=summary_system,
            meta_summaries=meta_summaries,
            meta_summary_system=meta_summary_system,
            **additional_keys,
        )

        out = self.llm(formatted_prompt)
        log_llm_io(out, f"Thought {idx}", self.verbose)
        thought = remove_newline(out.output_text).split("Action")[0].strip()
        scratchpad += thought

        return scratchpad, thought, out

    def _generate_action(
        self,
        idx: int,
        scratchpad: str,
        question: str,
        examples: str,
        summaries: str,
        summary_system: str,
        meta_summaries: str,
        meta_summary_system: str,
        prompt: str,
        additional_keys: Dict[str, str],
    ) -> Tuple[str, str, str, Response]:
        """Generate an action."""
        scratchpad += f"\nAction {idx}: "

        # Build prompt directly (inlined from _build_react_agent_prompt)
        formatted_prompt = prompt.format(
            question=question,
            examples=examples,
            summaries=summaries,
            scratchpad=scratchpad,
            max_steps=self.max_steps,
            summary_system=summary_system,
            meta_summaries=meta_summaries,
            meta_summary_system=meta_summary_system,
            **additional_keys,
        )

        out = self.llm(formatted_prompt)
        log_llm_io(out, f"Action {idx}", self.verbose)
        action = out.output_text
        action = remove_newline(action).split("Observation")[0]
        scratchpad += action
        action_type, query = parse_qa_action(action)

        return scratchpad, action_type, query, out

    def _generate_observation(
        self, idx: int, scratchpad: str, action_type: str, query: str, key: str
    ) -> Tuple[str, str, bool, bool, str, Dict[str, Any]]:
        """Generate an observation."""
        external_tool_info = {"search_result": "", "lookup_result": ""}

        answer = ""
        finished = False
        scratchpad += f"\nObservation {idx}: "

        if action_type.lower() == "finish":
            answer = query
            finished = True
            if self._evaluate_answer(answer, key):
                obs = "Answer is CORRECT"
            else:
                obs = "Answer is INCORRECT"
        elif action_type.lower() == "search":
            try:
                search_result = self.docstore.search(query)
                external_tool_info["search_result"] = search_result
                obs = remove_newline(search_result)
            except Exception:
                obs = "Could not find that page, please try again."
        elif action_type.lower() == "lookup":
            try:
                lookup_result = self.docstore.lookup(query)
                external_tool_info["lookup_result"] = lookup_result
                obs = remove_newline(lookup_result)
            except ValueError:
                obs = "The last page Searched was not found, so you cannot Lookup a keyword in it. Please try one of the similar pages given."
        else:
            obs = "Invalid Action. Valid Actions are Lookup[<topic>] Search[<topic>] and Finish[<answer>]."

        scratchpad += obs

        return (
            scratchpad,
            answer,
            finished,
            self._evaluate_answer(answer, key),
            obs,
            external_tool_info,
        )

    def _generate_summary(
        self,
        question: str,
        previous_trials: str,
        scratchpad: str,
        is_correct: bool,
        prompt: str,
        additional_keys: Dict[str, str],
    ) -> Tuple[str, Response]:
        """Generate a summary."""
        # Build prompt directly (inlined from _build_summary_prompt)
        formatted_prompt = prompt.format(
            question=question,
            previous_trials=previous_trials,
            scratchpad=scratchpad,
            **additional_keys,
        )

        out = self.llm(formatted_prompt)
        log_llm_io(out, "Summary", self.verbose)

        # Add summaries to memory
        eval_report = "Answer is CORRECT" if is_correct else "Answer is INCORRECT"
        self.memory.add_memories(
            question=question,
            summaries=out.output_text,
            trial=f"Question: {question}\n{out.output_text}\nEVALUATION REPORT: {eval_report}",
            is_correct=is_correct,
        )

        return out.output_text, out

    def _generate_meta_summary(
        self,
        question: str,
        meta_summaries: str,
        meta_summary_system: str,
        previous_trials: str,
        scratchpad: str,
        prompt: str,
        additional_keys: Dict[str, str],
    ) -> Tuple[str, Response]:
        """Generate a meta-summary."""
        # Build prompt directly (inlined from _build_meta_summary_prompt)
        formatted_prompt = prompt.format(
            question=question,
            meta_summary_system=meta_summary_system,
            meta_summaries=meta_summaries,
            previous_trials=previous_trials,
            scratchpad=scratchpad,
            **additional_keys,
        )

        out = self.llm(formatted_prompt)
        log_llm_io(out, "Meta-Summary", self.verbose)

        # Add meta-summaries to memory
        self.memory.add_meta_summaries(
            question=question,
            meta_summaries=out.output_text,
        )

        return out.output_text, out

    def _halting_condition(self, idx: int, key: str, answer: str) -> bool:
        """Determine whether the halting condition has been met."""
        return self._evaluate_answer(answer, key) or idx >= self.max_trials + 1

    def _react_halting_condition(self, finished: bool, idx: int) -> bool:
        """Determine whether the ReAct halting condition has been met."""
        return finished or idx > self.max_steps

    def _evaluate_answer(self, answer: str, key: str) -> bool:
        """Evaluate if the answer is correct using EM metric."""
        return EM(answer, key) or fuzzy_EM(answer, key)
