"""CLIN general strategy."""

import time

from typing import Any, Dict, List, Tuple

import tiktoken

from langchain_community.docstore.wikipedia import Wikipedia
from tiktoken import Encoding

from agential.agents.clin.functional import (
    _is_halted,
    _prompt_react_agent,
    _prompt_summaries,
    parse_qa_action,
)
from agential.agents.clin.output import CLINOutput, CLINReActStepOutput, CLINStepOutput
from agential.agents.clin.strategies.base import CLINBaseStrategy
from agential.core.llm import BaseLLM, Response
from agential.eval.metrics.classification import EM
from agential.utils.docstore import DocstoreExplorer
from agential.utils.parse import remove_newline


class CLINGeneralStrategy(CLINBaseStrategy):
    def __init__(
        self,
        llm: BaseLLM,
        max_trials: int = 3,
        max_steps: int = 6,
        max_tokens: int = 5000,
        enc: Encoding = tiktoken.encoding_for_model("gpt-3.5-turbo"),
        docstore: DocstoreExplorer = DocstoreExplorer(Wikipedia()),
        testing: bool = False,
    ) -> None:
        """Initialization."""
        super().__init__(
            llm=llm,
            max_trials=max_trials,
            max_steps=max_steps,
            max_tokens=max_tokens,
            enc=enc,
            testing=testing,
        )
        self.docstore = docstore

    def generate(
        self,
        question: str,
        key: str,
        examples: str,
        prompt: str,
        additional_keys: Dict[str, str],
        summary_system: str,
        patience: int,
        reset: bool,
    ) -> CLINOutput:
        start = time.time()

        # Reset.
        if reset:
            self.reset()

        scratchpad = ""
        summaries = []
        answer = ""
        finished = False
        idx, step_idx, patience_cnt = 1, 1, 0
        steps: List[CLINStepOutput] = []
        while not self.halting_condition(idx=idx, key=key, answer=answer):

            step_idx, is_correct, scratchpad, finished, answer, react_steps = (
                self.generate_react(
                    question=question,
                    key=key,
                    examples=examples,
                    summaries=summaries,
                    summary_system=summary_system,
                    prompt=prompt,
                    additional_keys=additional_keys,
                )
            )

            # Update summaries.
            self.generate_summaries(
                scratchpad=scratchpad,
                summaries=summaries,
                answer=answer,
                step_idx=step_idx,
                key=key,
                examples=examples,
                prompt=prompt,
                additional_keys=additional_keys,
            )

            steps.append(
                CLINStepOutput(
                    steps=react_steps,
                )
            )

            # Increment patience counter.
            if not is_correct:
                patience_cnt += 1
            if patience_cnt == patience:
                break

            idx += 1

    def generate_react(
        self,
        question: str,
        key: str,
        examples: str,
        summaries: List[str],
        summary_system: str,
        prompt: str,
        additional_keys: Dict[str, str],
    ) -> Tuple[int, bool, str, bool, str, List[CLINReActStepOutput]]:
        """Generates a reaction based on the given question, key, examples, reflections, prompt, and additional keys.

        Args:
            question (str): The question to be answered.
            key (str): The key for the observation.
            examples (str): Examples to guide the reaction process.
            summaries (List[str]): The summaries of the previous steps.
            summary_system (str): The system prompt for the summaries.
            prompt (str): The prompt or instruction to guide the reaction.
            additional_keys (Dict[str, str]): Additional keys for the reaction process.

        Returns:
            Tuple[int, bool, str, bool, str, List[CLINReActStepOutput]]: The reaction, whether the reaction is finished, the answer, whether the reaction is valid, the scratchpad, and the steps.
        """
        react_steps = []
        step_idx = 1
        scratchpad = ""
        finished = False
        answer = ""
        while not self.react_halting_condition(
            finished=finished,
            idx=step_idx,
            scratchpad=scratchpad,
            question=question,
            examples=examples,
            summaries=summaries,
            summary_system=summary_system,
            prompt=prompt,
            additional_keys=additional_keys,
        ):
            # Think.
            scratchpad, thought, thought_response = self.generate_thought(
                idx=step_idx,
                scratchpad=scratchpad,
                question=question,
                examples=examples,
                summaries=summaries,
                summary_system=summary_system,
                prompt=prompt,
                additional_keys=additional_keys,
            )

            # Act.
            scratchpad, action_type, query, action_response = self.generate_action(
                idx=step_idx,
                scratchpad=scratchpad,
                question=question,
                examples=examples,
                summaries=summaries,
                summary_system=summary_system,
                prompt=prompt,
                additional_keys=additional_keys,
            )

            # Observe.
            scratchpad, answer, finished, is_correct, obs, external_tool_info = (
                self.generate_observation(
                    idx=step_idx,
                    scratchpad=scratchpad,
                    action_type=action_type,
                    query=query,
                    key=key,
                )
            )

            react_steps.append(
                CLINReActStepOutput(
                    thought=thought,
                    action_type=action_type,
                    query=query,
                    observation=obs,
                    answer=answer,
                    external_tool_info=external_tool_info,
                    is_correct=is_correct,
                    thought_response=thought_response,
                    action_response=action_response,
                )
            )

            step_idx += 1

        return step_idx, is_correct, scratchpad, finished, answer, react_steps

    def generate_thought(
        self,
        idx: int,
        scratchpad: str,
        question: str,
        examples: str,
        summaries: str,
        summary_system: str,
        prompt: str,
        additional_keys: Dict[str, str],
    ) -> Tuple[str, str, Response]:
        """Generates a thought based on the given question, examples, summaries, prompt, and additional keys.

        Args:
            idx (int): The current step.
            scratchpad (str): The scratchpad containing previous thoughts.
            question (str): The question to generate a thought for.
            examples (str): Examples to guide the thought generation process.
            summaries (str): Summaries of previous steps.
            summary_system (str): The system prompt for the summaries.
            prompt (str): The prompt or instruction to guide the thought generation.
            additional_keys (Dict[str, str]): Additional keys for the thought generation process.

        Returns:
            Tuple[str, str, Response]: The updated scratchpad, the generated thought, and the thought responses.
        """
        scratchpad += f"\nThought {idx}: "
        out = _prompt_react_agent(
            llm=self.llm,
            question=question,
            examples=examples,
            summaries=summaries,
            scratchpad=scratchpad,
            max_steps=self.max_steps,
            summary_system=summary_system,
            prompt=prompt,
            additional_keys=additional_keys,
        )
        thought = remove_newline(out.output_text).split("Action")[0].strip()
        scratchpad += thought

        return scratchpad, thought, out

    def generate_action(
        self,
        idx: int,
        scratchpad: str,
        question: str,
        examples: str,
        summaries: str,
        summary_system: str,
        prompt: str,
        additional_keys: Dict[str, str],
    ) -> Tuple[str, str, str, Response]:
        """Generate an action for the current step in the reasoning process.

        Args:
            idx (int): The current step index.
            scratchpad (str): The scratchpad containing previous thoughts and actions.
            question (str): The main question or task to be addressed.
            examples (str): Relevant examples to provide context for action generation.
            trajectory (str): The current trajectory or history of thoughts and actions.
            summaries (str): Summaries of previous steps.
            summary_system (str): The system prompt for the summaries.
            depth (int): The current depth in the search tree.
            prompt (str): The prompt template for action generation.
            additional_keys (Dict[str, str]): Additional keys for prompt formatting.

        Returns:
            Tuple[str, str, str, Response]: A tuple containing the updated trajectory, action type, query, and the metrics.
        """
        scratchpad += f"\nAction {idx}: "
        out = _prompt_react_agent(
            llm=self.llm,
            question=question,
            examples=examples,
            summaries=summaries,
            scratchpad=scratchpad,
            max_steps=self.max_steps,
            summary_system=summary_system,
            prompt=prompt,
            additional_keys=additional_keys,
        )
        action = out.output_text
        action = remove_newline(action).split("Observation")[0]
        scratchpad += action
        action_type, query = parse_qa_action(action)

        return scratchpad, action_type, query, out

    def generate_observation(
        self, idx: int, scratchpad: str, action_type: str, query: str, key: str
    ) -> Tuple[str, str, bool, bool, str, Dict[str, Any]]:
        """Generate an observation based on the given inputs.

        Args:
            idx (int): The current index of the observation.
            scratchpad (str): The current state of the scratchpad.
            action_type (str): The type of action performed.
            query (str): The query or action to observe.
            key (str): The key for the observation.

        Returns:
            Tuple[str, str, str, bool, Dict[str, Any]]: A tuple containing:
                - The updated scratchpad.
                - The answer.
                - A boolean indicating if finished.
                - A boolean indicating if the task is finished.
                - The generated observation.
                - The observation.
                - A dictionary with additional information.
        """
        external_tool_info = {"search_result": "", "lookup_result": ""}

        answer = ""
        finished = False
        scratchpad += f"\nObservation {idx}: "
        if action_type.lower() == "finish":
            answer = query
            finished = True
            if EM(answer, key):
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

        return scratchpad, answer, finished, EM(answer, key), obs, external_tool_info

    def generate_summaries(
        self,
        question: str,
        meta_summaries: str,
        meta_summary_system: str,
        previous_trials: str,
        scratchpad: str,
        prompt: str,
        additional_keys: Dict[str, str],
    ) -> Tuple[str | Response]:
        out = _prompt_summaries(
            llm=self.llm,
            question=question,
            meta_summaries=meta_summaries,
            meta_summary_system=meta_summary_system,
            previous_trials=previous_trials,
            scratchpad=scratchpad,
            prompt=prompt,
            additional_keys=additional_keys,
        )



        return out



    def meta_summarize(self) -> Tuple[str | Response]:
        return super().meta_summarize()

    def halting_condition(self, finished: bool) -> bool:
        return super().halting_condition(finished)

    def react_halting_condition(
        self,
        finished: bool,
        idx: int,
        scratchpad: str,
        question: str,
        examples: str,
        summaries: str,
        summary_system: str,
        prompt: str,
        additional_keys: Dict[str, str],
    ) -> bool:
        """Determine whether the halting condition has been met in the ReflexionReAct agent.

        Args:
            finished (bool): A boolean indicating whether the task is finished.
            idx (int): The index of the current step.
            scratchpad (str): The scratchpad containing previous thoughts and actions.
            question (str): The question to generate an action for.
            examples (str): Examples to guide the action generation process.
            summaries (str): Summaries of previous steps.
            summary_system (str): The system prompt for summarization.
            prompt (str): The prompt or instruction to guide the action generation.
            additional_keys (Dict[str, str]): Additional keys for the action generation process.

        Returns:
            bool: True if the halting condition is met, False otherwise. The halting condition is met when the answer is not correct and the current step index is less than the maximum number of steps plus one.
        """
        return _is_halted(
            finished=finished,
            step_idx=idx,
            question=question,
            scratchpad=scratchpad,
            examples=examples,
            summaries=summaries,
            summary_system=summary_system,
            max_steps=self.max_steps,
            max_tokens=self.max_tokens,
            enc=self.enc,
            prompt=prompt,
            additional_keys=additional_keys,
        )

    def reset(self) -> None:
        return super().reset()
