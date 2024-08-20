"""Reflexion Agent strategies for QA."""

from typing import Any, Dict, List, Optional, Tuple

import tiktoken

from langchain_community.docstore.wikipedia import Wikipedia
from tiktoken import Encoding

from agential.cog.reflexion.functional import (
    _is_halted,
    _prompt_cot_agent,
    _prompt_react_agent,
    _truncate_scratchpad,
    parse_qa_action,
)
from agential.cog.reflexion.output import ReflexionReActStepOutput
from agential.cog.reflexion.reflect import (
    ReflexionCoTReflector,
    ReflexionReActReflector,
)
from agential.cog.reflexion.strategies.base import (
    ReflexionReActBaseStrategy,
)
from agential.cog.reflexion.strategies.general import ReflexionCoTGeneralStrategy, ReflexionReActGeneralStrategy
from agential.eval.em import EM
from agential.llm.llm import BaseLLM
from agential.utils.docstore import DocstoreExplorer
from agential.utils.metrics import PromptMetrics, get_token_cost_time
from agential.utils.parse import remove_newline


class ReflexionCoTQAStrategy(ReflexionCoTGeneralStrategy):
    """A strategy class for QA benchmarks using the ReflexionCoT agent.

    Attributes:
        llm (BaseLLM): The language model used for generating answers and critiques.
        reflector (Optional[ReflexionCoTReflector]): The reflector used for generating reflections. Defaults to None.
        max_reflections (int): The maximum number of reflections allowed. Defaults to 3.
        max_trials (int): The maximum number of trials allowed. Defaults to 3.
        testing (bool): Whether to run in testing mode. Defaults to False.
    """

    def __init__(
        self,
        llm: BaseLLM,
        reflector: Optional[ReflexionCoTReflector] = None,
        max_reflections: int = 3,
        max_trials: int = 3,
        testing: bool = False,
    ) -> None:
        """Initialization."""
        if reflector is None:
            reflector = ReflexionCoTReflector(llm=llm, max_reflections=max_reflections)
        super().__init__(
            llm=llm,
            reflector=reflector,
            max_reflections=max_reflections,
            max_trials=max_trials,
            testing=testing,
        )

    def generate_action(
        self,
        scratchpad: str,
        question: str,
        examples: str,
        reflections: str,
        prompt: str,
        additional_keys: Dict[str, str],
    ) -> Tuple[str, str, str, PromptMetrics]:
        """Generates an action based on the question, examples, and prompt.

        Args:
            scratchpad (str): The current state of the scratchpad.
            question (str): The question to be answered.
            examples (str): Examples to guide the generation process.
            reflections (str): Reflections to consider during generation.
            prompt (str): The prompt used for generating the action.
            additional_keys (Dict[str, str]): Additional keys for the generation process.
            **kwargs (Any): Additional arguments.

        Returns:
            Tuple[str, str, str, PromptMetrics]: The updated scratchpad, the generated action, the action type, and the metrics for the action.
        """
        scratchpad += f"\nAction: "
        out = _prompt_cot_agent(
            llm=self.llm,
            examples=examples,
            reflections=reflections,
            question=question,
            scratchpad=scratchpad,
            prompt=prompt,
            additional_keys=additional_keys,
        )
        action = out.choices[0].message.content
        action = remove_newline(action).strip()
        scratchpad += action
        action_type, query = parse_qa_action(action)

        return scratchpad, action_type, query, get_token_cost_time(out)

    def generate_observation(
        self, scratchpad: str, action_type: str, query: str, key: str
    ) -> Tuple[str, str, bool, str]:
        """Generates an observation based on the action type and query.

        Args:
            scratchpad (str): The current state of the scratchpad.
            action_type (str): The type of action to be performed.
            query (str): The query for the action.
            key (str): The key for the observation.

        Returns:
            Tuple[str, str, bool, str, bool]: The updated scratchpad, the answer, a boolean indicating if the observation is correct, and the observation itself.
        """
        answer = ""
        scratchpad += f"\nObservation: "
        if action_type.lower() == "finish":
            answer = query
            if EM(answer, key):
                obs = "Answer is CORRECT"
            else:
                obs = "Answer is INCORRECT"
        else:
            obs = "Invalid action type, please try again."
        scratchpad += obs

        return scratchpad, answer, EM(answer, key), obs

    def halting_condition(
        self,
        idx: int,
        key: str,
        answer: str,
    ) -> bool:
        """Determines whether the halting condition has been met.

        Args:
            idx (int): The current step index.
            key (str): The key for the observation.
            answer (str): The answer generated.

        Returns:
            bool: True if the halting condition is met, False otherwise.
        """
        return EM(answer, key) or idx >= self.max_trials

    def reflect_condition(
        self,
        idx: int,
        reflect_strategy: Optional[str],
        key: str,
        answer: str,
    ) -> bool:
        """Determines whether the reflection condition has been met.

        Args:
            idx (int): The current step.
            reflect_strategy (Optional[str]): The strategy to use for reflection.
            key (str): The key for the observation.
            answer (str): The answer generated.

        Returns:
            bool: True if the reflection condition is met, False otherwise.
        """
        return idx > 0 and not EM(answer, key) and reflect_strategy is not None


class ReflexionReActQAStrategy(ReflexionReActGeneralStrategy):
    """A strategy class for QA benchmarks using the ReflexionReAct agent.

    Attributes:
        llm (BaseLLM): The language model used for generating answers and critiques.
        reflector (Optional[ReflexionReActReflector]): The reflector used for generating reflections. Defaults to None.
        max_reflections (int): The maximum number of reflections allowed. Defaults to 3.
        max_trials (int): The maximum number of trials allowed. Defaults to 3.
        max_steps (int): The maximum number of steps allowed. Defaults to 6.
        max_tokens (int): The maximum number of tokens allowed. Defaults to 5000.
        enc (Encoding): The encoding for tokenization. Defaults to gpt-3.5-turbo.
        docstore (DocstoreExplorer): The document store explorer for retrieving relevant documents. Defaults to Wikipedia.
        testing (bool): Whether the strategy is in testing mode. Defaults to False.
    """

    def __init__(
        self,
        llm: BaseLLM,
        reflector: Optional[ReflexionReActReflector] = None,
        max_reflections: int = 3,
        max_trials: int = 3,
        max_steps: int = 6,
        max_tokens: int = 5000,
        enc: Encoding = tiktoken.encoding_for_model("gpt-3.5-turbo"),
        docstore: DocstoreExplorer = DocstoreExplorer(Wikipedia()),
        testing: bool = False,
    ) -> None:
        """Initialization."""
        if reflector is None:
            reflector = ReflexionReActReflector(
                llm=llm, max_reflections=max_reflections
            )
        super().__init__(
            llm=llm, 
            reflector=reflector, 
            max_reflections=max_reflections, 
            max_trials=max_trials, 
            max_steps=max_steps, 
            max_tokens=max_tokens, 
            enc=enc,
            testing=testing
        )
        self.docstore = docstore

        self._finished = False
        self._answer = ""
        self._scratchpad = ""

    def generate_action(
        self,
        idx: int,
        scratchpad: str,
        question: str,
        examples: str,
        reflections: str,
        prompt: str,
        additional_keys: Dict[str, str],
    ) -> Tuple[str, str, str, PromptMetrics]:
        """Generate an action for the current step in the reasoning process.

        Args:
            idx (int): The current step index.
            scratchpad (str): The scratchpad containing previous thoughts and actions.
            question (str): The main question or task to be addressed.
            examples (str): Relevant examples to provide context for action generation.
            trajectory (str): The current trajectory or history of thoughts and actions.
            reflections (str): Previous reflections to guide the action generation.
            depth (int): The current depth in the search tree.
            prompt (str): The prompt template for action generation.
            additional_keys (Dict[str, str]): Additional keys for prompt formatting.

        Returns:
            Tuple[str, str, str, PromptMetrics]: A tuple containing the updated trajectory, action type, query, and the metrics.
        """
        scratchpad += f"\nAction {idx}: "
        out = _prompt_react_agent(
            llm=self.llm,
            question=question,
            examples=examples,
            reflections=reflections,
            scratchpad=scratchpad,
            max_steps=self.max_steps,
            prompt=prompt,
            additional_keys=additional_keys,
        )
        action = out.choices[0].message.content
        action = remove_newline(action).split("Observation")[0]
        scratchpad += action
        action_type, query = parse_qa_action(action)

        return scratchpad, action_type, query, get_token_cost_time(out)

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
                - The generated observation.
                - A boolean indicating if the task is finished.
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

    def create_output_dict(
        self,
        react_out: List[ReflexionReActStepOutput],
        reflections: List[str],
    ) -> Dict[str, Any]:
        """Create a dictionary containing the output of the ReflexionReAct agent.

        Args:
            react_out (List[ReflexionReActStepOutput]): The output of the ReflexionReAct agent, containing the thought, action type, query, observation, and whether the answer is correct for each step.
            reflections (List[str]): The reflections generated by the ReflexionReAct agent.

        Returns:
            Dict[str, str]: A dictionary containing the 'react_output' and 'reflections'.
        """
        return {
            "react_output": react_out,
            "reflections": reflections,
            "prompt_metrics": self._prompt_metrics,
        }

    def react_create_output_dict(
        self,
        thought: str,
        action_type: str,
        query: str,
        obs: str,
        external_tool_info: Dict[str, Any],
        is_correct: bool,
    ) -> Dict[str, Any]:
        """Create a dictionary containing the output of a single step in the ReflexionReAct agent.

        Args:
            thought (str): The thought generated in the current step.
            action_type (str): The type of action performed in the current step.
            query (str): The query or information related to the action performed in the current step.
            obs (str): The observation generated in the current step.
            external_tool_info (Dict[str, Any]): The external tool outputs.
            is_correct (bool): A boolean indicating whether the answer generated in the current step is correct.

        Returns:
            Dict[str, Any]: A dictionary containing the 'thought', 'action_type', 'query', 'observation', 'answer', 'external_tool_info', and 'is_correct' of the current step.
        """
        return {
            "thought": thought,
            "action_type": action_type,
            "query": query,
            "observation": obs,
            "answer": self._answer,
            "external_tool_info": external_tool_info,
            "is_correct": is_correct,
            "prompt_metrics": self._prompt_metrics_react,
        }

    def halting_condition(
        self,
        idx: int,
        key: str,
        answer: str,
    ) -> bool:
        """Determines whether the halting condition has been met.

        Args:
            idx (int): The current step index.
            key (str): The key for the observation.
            answer (str): The answer generated.

        Returns:
            bool: True if the halting condition is met, False otherwise.
        """
        return EM(answer, key) or idx >= self.max_trials + 1

    def react_halting_condition(
        self,
        finished: bool,
        idx: int,
        scratchpad: str,
        question: str,
        examples: str,
        reflections: str,
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
            reflections (str): Reflections to consider during the action generation process.
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
            reflections=reflections,
            max_steps=self.max_steps,
            max_tokens=self.max_tokens,
            enc=self.enc,
            prompt=prompt,
            additional_keys=additional_keys,
        )

    def reset(self, **kwargs: Any) -> None:
        """Resets the internal state of the strategy.

        Resets the scratchpad and the finished flag.
        Resets only the scratchpad if specified with 'only_scratchpad'.

        Args:
            **kwargs (Any): Additional keyword arguments.
        """
        no_reflector = kwargs.get("no_reflector", False)
        if not no_reflector:
            self.reflector.reset()
        self._scratchpad = ""
        self._finished = False
        self._answer = ""
        self._prompt_metrics_react = {"thought": None, "action": None}
        self._prompt_metrics = {"reflection": None}

    def reflect(
        self,
        reflect_strategy: str,
        question: str,
        examples: str,
        prompt: str,
        additional_keys: Dict[str, str],
    ) -> Tuple[List[str], str]:
        """Reflects on a given question, context, examples, prompt, and additional keys using the specified reflection strategy.

        Args:
            reflect_strategy (str): The strategy to use for reflection.
            question (str): The question to be reflected upon.
            examples (str): Examples to guide the reflection process.
            prompt (str): The prompt or instruction to guide the reflection.
            additional_keys (Dict[str, str]): Additional keys for the reflection process.

        Returns:
            Tuple[List[str], str]: The reflections and reflection string.
        """
        reflections, reflections_str, reflections_out = self.reflector.reflect(
            reflect_strategy=reflect_strategy,
            question=question,
            examples=examples,
            scratchpad=_truncate_scratchpad(
                scratchpad=self._scratchpad, tokenizer=self.enc
            ),
            prompt=prompt,
            additional_keys=additional_keys,
        )
        self._prompt_metrics["reflection"] = (
            get_token_cost_time(reflections_out) if reflections_out else None
        )

        return reflections, reflections_str

    def reflect_condition(
        self,
        answer: str,
        finished: bool,
        idx: int,
        scratchpad: str,
        reflect_strategy: Optional[str],
        question: str,
        examples: str,
        key: str,
        prompt: str,
        additional_keys: Dict[str, str],
    ) -> bool:
        """Determine whether the reflection condition has been met in the ReflexionReAct agent.

        Args:
            answer (str): The answer generated.
            finished (bool): A boolean indicating whether the task is finished.
            idx (int): The index of the current step.
            scratchpad (str): The scratchpad containing previous thoughts and actions.
            reflect_strategy (Optional[str]): The strategy to use for reflection.
            question (str): The question to be reflected upon.
            examples (str): Examples to guide the reflection process.
            key (str): The key for the observation.
            prompt (str): The prompt or instruction to guide the reflection.
            additional_keys (Dict[str, str]): Additional keys for the reflection process.

        Returns:
            bool: True if the reflection condition is met, False otherwise. The reflection condition is met when the agent is halted, the answer is not correct, and the reflection strategy is provided.
        """

        halted = _is_halted(
            finished=finished,
            step_idx=idx,
            question=question,
            scratchpad=scratchpad,
            examples=examples,
            reflections=self.reflector.reflections_str,
            max_steps=self.max_steps,
            max_tokens=self.max_tokens,
            enc=self.enc,
            prompt=prompt,
            additional_keys=additional_keys,
        )

        return halted and not EM(answer, key) and reflect_strategy is not None


class ReflexionCoTHotQAStrategy(ReflexionCoTQAStrategy):
    """A strategy class for the HotpotQA benchmark using the ReflexionCoT agent."""

    pass


class ReflexionCoTTriviaQAStrategy(ReflexionCoTQAStrategy):
    """A strategy class for the TriviaQA benchmark using the ReflexionCoT agent."""

    pass


class ReflexionCoTAmbigNQStrategy(ReflexionCoTQAStrategy):
    """A strategy class for the AmbigNQ benchmark using the ReflexionCoT agent."""

    pass


class ReflexionCoTFEVERStrategy(ReflexionCoTQAStrategy):
    """A strategy class for the FEVER benchmark using the ReflexionCoT agent."""

    pass


class ReflexionReActHotQAStrategy(ReflexionReActQAStrategy):
    """A strategy class for the HotpotQA benchmark using the ReflexionReAct agent."""

    pass


class ReflexionReActTriviaQAStrategy(ReflexionReActQAStrategy):
    """A strategy class for the TriviaQA benchmark using the ReflexionReAct agent."""

    pass


class ReflexionReActAmbigNQStrategy(ReflexionReActQAStrategy):
    """A strategy class for the AmbigNQ benchmark using the ReflexionReAct agent."""

    pass


class ReflexionReActFEVERStrategy(ReflexionReActQAStrategy):
    """A strategy class for the FEVER benchmark using the ReflexionReAct agent."""

    pass
