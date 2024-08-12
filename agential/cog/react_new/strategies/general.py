"""General strategy for the ReAct Agent."""

from typing import Any, Dict, Tuple, List

import tiktoken

from tiktoken.core import Encoding

from agential.cog.react_new.functional import _is_halted, _prompt_agent
from agential.cog.react_new.strategies.base import ReActBaseStrategy
from agential.cog.react_new.output import ReActStepOutput
from agential.llm.llm import BaseLLM
from agential.utils.general import get_token_cost_time
from agential.utils.parse import remove_newline


class ReActGeneralStrategy(ReActBaseStrategy):
    """A general strategy class using the ReAct agent.

    Attributes:
        llm (BaseLLM): The language model used for generating answers and critiques.
        max_steps (int): The maximum number of steps the agent can take.
        max_tokens (int): The maximum number of tokens allowed for a response.
        enc (Encoding): The encoding used for the language model.
    """

    def __init__(
        self,
        llm: BaseLLM,
        max_steps: int = 6,
        max_tokens: int = 5000,
        enc: Encoding = tiktoken.encoding_for_model("gpt-3.5-turbo"),
    ) -> None:
        """Initialization."""
        super().__init__(llm, max_steps, max_tokens, enc)

        self._scratchpad = ""
        self._answer = ""
        self._finished = False
        self._prompt_metrics: Dict[str, Any] = {"thought": None, "action": None}

    def generate(
        self,
        question: str,
        examples: str,
        prompt: str,
        additional_keys: Dict[str, str],
        reset: bool,
        **kwargs: Any,
    ) -> List[ReActOutput]:
        if reset:
            self.reset()

        idx = 1
        out = []
        while not self.halting_condition(
            idx=idx,
            question=question,
            examples=examples,
            prompt=prompt,
            additional_keys=additional_keys,
            **kwargs,
        ):
            # Think.
            thought = self.generate(
                question=question,
                examples=examples,
                prompt=prompt,
                additional_keys=additional_keys,
                **kwargs,
            )

            # Act.
            action_type, query = self.generate_action(
                question=question,
                examples=examples,
                prompt=prompt,
                additional_keys=additional_keys,
                **kwargs,
            )

            # Observe.
            obs, external_tool_info = self.generate_observation(
                idx=idx, action_type=action_type, query=query
            )

            out.append(
                ReActOutput(
                    **self.create_output_dict(
                        thought=thought,
                        action_type=action_type,
                        query=query,
                        obs=obs,
                        external_tool_info=external_tool_info,
                    )
                )
            )

            idx += 1

        return out
    
    def generate_thought(
        self,
        question: str,
        examples: str,
        prompt: str,
        additional_keys: Dict[str, str],
    ) -> str:
        """Generates a thought based on the question, examples, and prompt.

        Args:
            question (str): The question to be answered.
            examples (str): Examples to guide the generation process.
            prompt (str): The prompt used for generating the thought.
            additional_keys (Dict[str, str]): Additional keys for the generation process.

        Returns:
            str: The generated thought.
        """
        self._scratchpad += "\nThought:"
        out = _prompt_agent(
            llm=self.llm,
            question=question,
            scratchpad=self._scratchpad,
            examples=examples,
            max_steps=self.max_steps,
            prompt=prompt,
            additional_keys=additional_keys,
        )
        self._prompt_metrics["thought"] = get_token_cost_time(out)
        thought = out.choices[0].message.content

        thought = remove_newline(thought).split("Action")[0].strip()
        self._scratchpad += " " + thought

        return thought

    def generate_action(
        self,
        question: str,
        examples: str,
        prompt: str,
        additional_keys: Dict[str, str],
    ) -> Tuple[str, str]:
        """Generates an action based on the question, examples, and prompt.

        Args:
            question (str): The question to be answered.
            examples (str): Examples to guide the generation process.
            prompt (str): The prompt used for generating the action.
            additional_keys (Dict[str, str]): Additional keys for the generation process.

        Returns:
            Tuple[str, str]: The generated action type and query.
        """
        raise NotImplementedError("Subclasses must implement this method")

    def generate_observation(
        self, idx: int, action_type: str, query: str
    ) -> Tuple[str, Dict[str, Any]]:
        """Generates an observation based on the action type and query.

        Args:
            idx (int): The index of the observation.
            action_type (str): The type of action to be performed.
            query (str): The query for the action.

        Returns:
            Tuple[str, Dict[str, Any]]: The generated observation and external tool outputs.
        """
        raise NotImplementedError("Subclasses must implement this method")


    def create_output_dict(
        self,
        thought: str,
        action_type: str,
        query: str,
        obs: str,
        external_tool_info: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Creates a dictionary of the output components.

        Args:
            thought (str): The generated thought.
            action_type (str): The type of action performed.
            query (str): The query for the action.
            obs (str): The generated observation.
            external_tool_info (Dict[str, Any]): The external tool outputs.

        Returns:
            Dict[str, Any]: A dictionary containing the thought, action type, query, observation, answer, external tool output, and prompt metrics.
        """
        return {
            "thought": thought,
            "action_type": action_type,
            "query": query,
            "observation": obs,
            "answer": self._answer,
            "external_tool_info": external_tool_info,
            "prompt_metrics": self._prompt_metrics,
        }

    def halting_condition(
        self,
        idx: int,
        question: str,
        examples: str,
        prompt: str,
        additional_keys: Dict[str, str],
    ) -> bool:
        """Determines whether the halting condition has been met.

        Args:
            idx (int): The current step index.
            question (str): The question being answered.
            examples (str): Examples to guide the generation process.
            prompt (str): The prompt used for generating the thought and action.
            additional_keys (Dict[str, str]): Additional keys for the generation process.

        Returns:
            bool: True if the halting condition is met, False otherwise.
        """

        return _is_halted(
            finished=self._finished,
            idx=idx,
            question=question,
            scratchpad=self._scratchpad,
            examples=examples,
            max_steps=self.max_steps,
            max_tokens=self.max_tokens,
            enc=self.enc,
            prompt=prompt,
            additional_keys=additional_keys,
        )

    def reset(self, **kwargs: Any) -> None:
        """Resets the internal state of the strategy.

        Resets the scratchpad and the finished flag.

        Args:
            **kwargs (Any): Additional arguments.

        Returns:
            None
        """
        self._scratchpad = ""
        self._answer = ""
        self._finished = False
        self._prompt_metrics = {"thought": None, "action": None}