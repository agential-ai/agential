"""Reflexion Agent.

Original Paper: https://arxiv.org/abs/2303.11366
Paper Repositories:
    - https://github.com/noahshinn/reflexion-draft
    - https://github.com/noahshinn/reflexion
"""

import re

from typing import Any, Dict, List, Optional, Tuple

import tiktoken

from langchain_community.docstore.wikipedia import Wikipedia
from langchain_core.language_models.chat_models import BaseChatModel
from tiktoken import Encoding

from agential.cog.agent.base import BaseAgent
from agential.cog.eval.reflexion import EM
from agential.cog.functional.reflexion import (
    _is_halted,
    _prompt_react_agent,
    _truncate_scratchpad,
)
from agential.cog.modules.reflect.reflexion import (
    ReflexionCoTReflector,
    ReflexionReActReflector,
)
from agential.cog.prompts.agent.reflexion import (
    HOTPOTQA_FEWSHOT_EXAMPLES_REFLEXION_REACT_REFLECT,
    REFLEXION_REACT_INSTRUCTION_HOTPOTQA,
    REFLEXION_REACT_REFLECT_INSTRUCTION_HOTPOTQA,
)
from agential.cog.prompts.benchmark.hotpotqa import HOTPOTQA_FEWSHOT_EXAMPLES_REACT
from agential.cog.strategies.strategy_factory import ReflexionCoTStrategyFactory
from agential.utils.docstore import DocstoreExplorer
from agential.utils.parse import remove_newline


def parse_action(string: str) -> Tuple[str, str]:
    """Parses an action string into an action type and its argument.

    This method is used in ReAct and Reflexion.

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


class ReflexionCoTAgent(BaseAgent):
    """Reflexion with Chain-of-Thought actor.

    Attributes:
        self_reflect_llm (BaseChatModel): The language model used for self-reflection.
        action_llm (BaseChatModel): The language model used for generating thoughts/actions.
        reflector (Optional[ReflexionReflector]): An optional reflector module for guided self-reflection.
        max_reflections (int): An int specifying the max number of reflections to use in a subsequent run. Defaults to 3.
        max_trials (int): Max number of answering attempts before stopping generation. Must be greater than 1 for reflection to occur. Defaults to 1.
        patience (int): The number of incorrect retries before stopping. Must be >= 1 and <= max_trials. Defaults to max_trials.

    Methods:
        generate(context, question, key, strategy): Generates a response based on the given context, question, and strategy.
        reflect(context, question, strategy): Reflects on the previous response and modifies the strategy accordingly.
        retrieve(): Retrieves the current memory state of the agent.
        reset(): Resets the agent's state for a new problem-solving session.
    """

    def __init__(
        self,
        llm: BaseChatModel,
        mode: Dict[str, str],
        reflector: Optional[ReflexionCoTReflector] = None,
        **strategy_kwargs: Dict[str, Any],
    ) -> None:
        """Initialization with default or provided values."""
        super().__init__()

        self.llm = llm
        self.mode = mode

        self.strategy = ReflexionCoTStrategyFactory().get_strategy(
            mode=self.mode, llm=self.llm, reflector=reflector, **strategy_kwargs
        )

    def generate(
        self,
        question: str,
        key: str,
        examples: str,
        reflection_strategy: str,
        prompt: str,
        reflect_examples: str,
        reflect_prompt: str,
        additional_keys: Dict[str, str] = {},
        reflection_additional_keys: Dict[str, str] = {},
        patience: int = 1,
        reset: bool = True,
        **kwargs: Dict[str, Any],
    ) -> List[Tuple[bool, str, Tuple[str, str, str]]]:
        """Generates a response based on the provided context, question, and key.

        The `generate` method internally calls reflect (if possible), resets the memory,
        and generates a thought, action, and the observation (Finish).

        Args:
            question (str): The question to answer.
            key (str): The key to evaluate the correctness of the answer.
            examples (str, optional): Fewshot examples.
            reflection_strategy (str): The strategy to use for reflection. Can be one of "last_attempt",
                "reflexion", or "last_attempt_and_reflexion".
            prompt (str, optional): Prompt template string.
            reflect_examples (str, optional): Reflection fewshot examples.
            reflect_prompt (str, optional): Reflect prompt template string.
            reset (bool): Resets the agent's memory. Defaults to True.

        Returns:
            result (List[Tuple[bool, str, List[str, str, str]]]): A list of tuples containing (is_correct, answer, output)
                where output is a thought-action-observation 3-tuple.
        """
        # Reset.
        if reset:
            self.reset()

        idx, patience_cnt = 0, 0
        out = []
        while self.strategy.halting_condition(idx=idx, key=key, **kwargs):

            # Reflect if possible.
            reflections = ""
            if self.strategy.should_reflect(
                idx=idx,
                reflection_strategy=reflection_strategy,
                key=key,
            ):
                reflections = self.strategy.reflect(
                    reflection_strategy=reflection_strategy,
                    question=question,
                    examples=reflect_examples,
                    prompt=reflect_prompt,
                    additional_keys=reflection_additional_keys,
                )

            self.strategy.reset(only_scratchpad=True)

            # Think.
            thought = self.strategy.generate(
                question=question,
                examples=examples,
                reflections=reflections,
                prompt=prompt,
                additional_keys=additional_keys,
                **kwargs,
            )

            # Act.
            action_type, query = self.strategy.generate_action(
                question=question,
                examples=examples,
                reflections=reflections,
                prompt=prompt,
                additional_keys=additional_keys,
                **kwargs,
            )

            # Observe.
            is_correct, obs = self.strategy.generate_observation(
                action_type=action_type, query=query, key=key
            )

            out.append(
                self.strategy.create_output_dict(
                    thought=thought,
                    action_type=action_type,
                    obs=obs,
                    is_correct=is_correct,
                )
            )

            # Increment patience counter.
            if not is_correct:
                patience_cnt += 1
            if patience_cnt == patience:
                break

            idx += 1

        return out

    def reset(self) -> None:
        """Resets the agent's memory and state."""
        self.strategy.reset()


class ReflexionReActAgent(BaseAgent):
    """Reflexion with ReAct actor.

    Attributes:


    Methods:
        generate(question, key, strategy): Generates a response based on the given question and strategy.
        reflect(question, strategy): Reflects on the previous response and modifies the strategy accordingly.
        retrieve(): Retrieves the current memory state of the agent.
        reset(): Resets the agent's state for a new problem-solving session.
    """

    def __init__(
        self,
        llm: BaseChatModel,
        mode: Dict[str, str],
        reflector: Optional[ReflexionReActReflector] = None,
        **strategy_kwargs: Dict[str, Any],

        # self_reflect_llm: BaseChatModel,
        # action_llm: BaseChatModel,
        # memory: Any,
        # reflector: Optional[ReflexionReActReflector] = None,
        # max_reflections: int = 3,
        # max_steps: int = 6,
        # max_tokens: int = 3896,
        # max_trials: int = 1,
        # patience: Optional[int] = None,
        # docstore: DocstoreExplorer = DocstoreExplorer(Wikipedia()),
        # enc: Encoding = tiktoken.encoding_for_model("gpt-3.5-turbo"),
    ) -> None:
        """Initialization."""
        super().__init__()
        self.self_reflect_llm = self_reflect_llm
        self.action_llm = action_llm

        # if not memory:
        #     self.memory = ReflexionMemory()
        # else:
        self.memory = memory

        self.max_reflections = max_reflections
        if not reflector:
            self.reflector = ReflexionReActReflector(
                llm=self_reflect_llm, max_reflections=max_reflections
            )
        else:
            self.reflector = reflector

        self.max_steps = max_steps
        self.max_tokens = max_tokens
        self.max_trials = max_trials

        if not patience:
            self.patience = max_trials
        else:
            self.patience = patience
        assert self.patience >= 0 and self.patience <= max_trials

        self.docstore = docstore
        self.enc = enc

        # Private variables.
        self._trial_n = 1
        self._step_n = 1
        self._finished = False
        self._answer = ""

    def generate(
        self,
        question: str,
        key: str,
        examples: str = HOTPOTQA_FEWSHOT_EXAMPLES_REACT,
        strategy: Optional[str] = None,
        reset: bool = True,
        prompt: str = REFLEXION_REACT_INSTRUCTION_HOTPOTQA,
        reflect_examples: str = HOTPOTQA_FEWSHOT_EXAMPLES_REFLEXION_REACT_REFLECT,
        reflect_prompt: str = REFLEXION_REACT_REFLECT_INSTRUCTION_HOTPOTQA,
    ) -> List[Tuple[bool, str, List[Tuple[str, str, str]]]]:
        """Processes a given question through ReAct and reflects using Reflexion strategies when possible.

        Iteratively applies the think-act-observe cycle to generate an answer for the question.
        The process continues until the operation is halted based on certain conditions.

        Args:
            question (str): The question to be processed.
            key (str): The answer to the question.
            examples (str, optional): Fewshot examples. Defaults to HOTPOTQA_FEWSHOT_EXAMPLES_REACT.
            strategy (Optional[str]): The reflection strategy. Can be of 3 types. Defaults to None.
                - "last_attempt": This strategy uses only 'question' and 'scratchpad'. The 'reflections' list is updated with the current scratchpad.
                - "reflexion": This strategy uses all the parameters. It adds a new reflexion generated by the language model to the 'reflections' list.
                - "last_attempt_and_reflexion": This strategy combines the 'last_attempt' and 'reflexion' strategies.
            reset (bool): Whether to reset the internal state before processing. Defaults to True.
            prompt (str, optional): Prompt template string. Defaults to REFLEXION_REACT_INSTRUCTION_HOTPOTQA.
            reflect_examples (str, optional): Reflection fewshot examples. Defaults to HOTPOTQA_FEWSHOT_EXAMPLES_REFLEXION_REACT_REFLECT.
            reflect_prompt (str, optional): Reflect prompt template string. Defaults to REFLEXION_REACT_REFLECT_INSTRUCTION_HOTPOTQA.

        Returns:
            result (List[Tuple[bool, str, List[Tuple[str, str, str]]]]): List of trials where each trial is
                in the format (is_correct, answer, output) and output is in a thought-action-observation 3-tuple.
        """
        # Reset.
        if reset:
            self.reset()

        patience_cnt = 0
        result = []
        while not EM(self._answer, key) and self._trial_n < self.max_trials + 1:
            # Reflect if possible.
            if (
                _is_halted(
                    finished=self._finished,
                    step_n=self._step_n,
                    question=question,
                    scratchpad=self.memory.load_memories()["scratchpad"],
                    examples=examples,
                    reflections=self.reflector.reflections_str,
                    max_steps=self.max_steps,
                    max_tokens=self.max_tokens,
                    enc=self.enc,
                    prompt=prompt,
                )
                and not EM(self._answer, key)
                and strategy
            ):
                self.reflect(
                    strategy, question, examples=reflect_examples, prompt=reflect_prompt
                )

            out = []
            self._step_n = 1
            self._finished = False
            self._answer = ""
            self.memory.clear()
            while not _is_halted(
                finished=self._finished,
                step_n=self._step_n,
                question=question,
                scratchpad=self.memory.load_memories()["scratchpad"],
                examples=examples,
                reflections=self.reflector.reflections_str,
                max_steps=self.max_steps,
                max_tokens=self.max_tokens,
                enc=self.enc,
                prompt=prompt,
            ):
                # Think.
                self.memory.add_memories("\nThought:")
                thought = _prompt_react_agent(
                    llm=self.action_llm,
                    examples=examples,
                    reflections=self.reflector.reflections_str,
                    question=question,
                    scratchpad=self.memory.load_memories()["scratchpad"],
                    max_steps=self.max_steps,
                    prompt=prompt,
                ).split("Action")[0]
                self.memory.add_memories(" " + thought)

                # Act.
                self.memory.add_memories("\nAction:")
                action = _prompt_react_agent(
                    llm=self.action_llm,
                    examples=examples,
                    reflections=self.reflector.reflections_str,
                    question=question,
                    scratchpad=self.memory.load_memories()["scratchpad"],
                    max_steps=self.max_steps,
                    prompt=prompt,
                ).split("Observation")[0]
                self.memory.add_memories(" " + action)
                action_type, query = parse_action(action)

                # Observe.
                self.memory.add_memories(f"\nObservation {self._step_n}: ")
                if action_type.lower() == "finish":
                    self._answer = query
                    self._finished = True
                    if EM(self._answer, key):
                        obs = "Answer is CORRECT"
                    else:
                        obs = "Answer is INCORRECT"
                elif action_type.lower() == "search":
                    try:
                        obs = remove_newline(self.docstore.search(query))
                    except Exception:
                        obs = "Could not find that page, please try again."
                elif action_type.lower() == "lookup":
                    try:
                        obs = remove_newline(self.docstore.lookup(query))
                    except ValueError:
                        obs = "The last page Searched was not found, so you cannot Lookup a keyword in it. Please try one of the similar pages given."
                else:
                    obs = "Invalid Action. Valid Actions are Lookup[<topic>] Search[<topic>] and Finish[<answer>]."
                self.memory.add_memories(obs)

                out.append(
                    (
                        f"Thought: {thought}",
                        f"Action: {action}",
                        f"Observation {self._step_n}: {obs}",
                    )
                )

                self._step_n += 1

            is_correct = EM(self._answer, key)
            result.append((is_correct, self._answer, out))

            # Increment patience counter.
            if not is_correct:
                patience_cnt += 1
            if patience_cnt == self.patience:
                break

            self._trial_n += 1

        return result

    def reflect(
        self,
        strategy: str,
        question: str,
        examples: str = HOTPOTQA_FEWSHOT_EXAMPLES_REFLEXION_REACT_REFLECT,
        prompt: str = REFLEXION_REACT_REFLECT_INSTRUCTION_HOTPOTQA,
    ) -> str:
        """Reflects on the previous steps to improve the response.

        Given the agent can reflect (strategy is not `None`), the strategy
        can be of 3 types:
        - "last_attempt": This strategy uses only 'question' and 'scratchpad'. The 'reflections' list is updated with the current scratchpad.
        - "reflexion": This strategy uses all the parameters. It adds a new reflexion generated by the language model to the 'reflections' list.
        - "last_attempt_and_reflexion": This strategy combines the 'last_attempt' and 'reflexion' strategies.
          It first formats the last attempt using 'question' and 'scratchpad', then adds a new reflexion using all the parameters.

        Args:
            strategy (str): The strategy to use for reflection.
            question (str): The question to answer.
            examples (str, optional): Fewshot examples. Defaults to HOTPOTQA_FEWSHOT_EXAMPLES_REFLEXION_REACT_REFLECT.
            prompt (str, optional): Reflect prompt template string. Defaults to REFLEXION_REACT_REFLECT_INSTRUCTION_HOTPOTQA.

        Returns:
            str: Generated reflections based on the strategy.
        """
        _, reflections_str = self.reflector.reflect(
            strategy=strategy,
            examples=examples,
            question=question,
            scratchpad=_truncate_scratchpad(
                scratchpad=self.memory.load_memories()["scratchpad"], tokenizer=self.enc
            ),
            prompt=prompt,
        )

        return reflections_str

    def retrieve(self) -> Dict[str, Any]:
        """Retrieves the current state of the agent's memory.

        Returns:
            Dict[str, Any]: The current state of the agent's memory.
        """
        return self.memory.load_memories()

    def reset(self) -> None:
        """Resets the internal state of the ReflexionReAct agent.

        Sets the step number, finished flag, and scratchpad to their initial values.
        """
        self._trial_n = 1
        self._step_n = 1
        self._finished = False
        self._answer = ""
        self.memory.clear()
        self.reflector.clear()
