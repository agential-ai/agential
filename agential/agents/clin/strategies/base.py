"""CLIN base strategy."""

from abc import abstractmethod
from typing import Any, Dict, List, Tuple

from tiktoken import Encoding

from agential.agents.base.strategies import BaseAgentStrategy
from agential.agents.clin.memory import CLINMemory
from agential.agents.clin.output import CLINOutput, CLINReActStepOutput
from agential.core.llm import BaseLLM, Response


class CLINBaseStrategy(BaseAgentStrategy):
    """An abstract base class for defining strategies for the CLIN Agent.

    Attributes:
        llm (BaseLLM): An instance of a language model used for generating responses.
        memory (CLINMemory): An instance of a memory used for storing and retrieving information.
        max_trials (int): The maximum number of trials allowed.
        max_steps (int): The maximum number of steps allowed.
        max_tokens (int): The maximum number of tokens allowed.
        enc (Encoding): The encoding for tokenization.
        testing (bool): Whether the generation is for testing purposes. Defaults to False.
    """

    def __init__(
        self,
        llm: BaseLLM,
        memory: CLINMemory,
        max_trials: int,
        max_steps: int,
        max_tokens: int,
        enc: Encoding,
        testing: bool = False,
    ) -> None:
        """Initialization."""
        super().__init__(llm=llm, testing=testing)
        self.memory = memory
        self.max_trials = max_trials
        self.max_steps = max_steps
        self.max_tokens = max_tokens
        self.enc = enc

    @abstractmethod
    def generate(
        self,
        question: str,
        examples: str,
        critique_examples: str,
        prompt: str,
        critique_prompt: str,
        additional_keys: Dict[str, str],
        critique_additional_keys: Dict[str, str],
        max_interactions: int,
        use_tool: bool,
        reset: bool,
    ) -> CLINOutput:
        """Generates an answer.

        Args:
            question (str): The question to be answered.
            examples (str): Few-shot examples to guide the language model in generating the answer.
            critique_examples (str): Few-shot examples to guide the language model in generating the critique.
            prompt (str): The instruction template used to prompt the language model for the answer.
            critique_prompt (str): The instruction template used to prompt the language model for the critique.
            additional_keys (Dict[str, str]): Additional keys to format the answer and critique prompts.
            critique_additional_keys (Dict[str, str]): Additional keys to format the critique prompt.
            max_interactions (int): The maximum number of interactions to perform.
            use_tool (bool): Whether to use a tool for generating the critique.
            reset (bool): Whether to reset the strategy.

        Returns:
            CLINOutput: The generated answer and critique.
        """
        raise NotImplementedError

    @abstractmethod
    def generate_react(
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
    ) -> Tuple[int, bool, str, bool, str, List[CLINReActStepOutput]]:
        """Generates a reaction based on the given question, key, examples, reflections, prompt, and additional keys.

        Args:
            question (str): The question to be answered.
            key (str): The key for the observation.
            examples (str): Examples to guide the reaction process.
            summaries (str): The summaries of the previous steps.
            summary_system (str): The system prompt for the summaries.
            meta_summaries (str): The meta-summaries of the previous steps.
            meta_summary_system (str): The system prompt for the meta-summaries.
            prompt (str): The prompt or instruction to guide the reaction.
            additional_keys (Dict[str, str]): Additional keys for the reaction process.

        Returns:
            Tuple[int, bool, str, bool, str, List[CLINReActStepOutput]]: The reaction, whether the reaction is finished, the answer, whether the reaction is valid, the scratchpad, and the steps.
        """
        raise NotImplementedError

    @abstractmethod
    def generate_thought(
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
        """Generates a thought based on the given question, examples, summaries, prompt, and additional keys.

        Args:
            idx (int): The current step.
            scratchpad (str): The scratchpad containing previous thoughts.
            question (str): The question to generate a thought for.
            examples (str): Examples to guide the thought generation process.
            summaries (str): Summaries of previous steps.
            summary_system (str): The system prompt for the summaries.
            meta_summaries (str): Meta-summaries of previous steps.
            meta_summary_system (str): The system prompt for the meta-summaries.
            prompt (str): The prompt or instruction to guide the thought generation.
            additional_keys (Dict[str, str]): Additional keys for the thought generation process.

        Returns:
            Tuple[str, str, Response]: The updated scratchpad, the generated thought, and the thought responses.
        """
        raise NotImplementedError

    @abstractmethod
    def generate_action(
        self,
        question: str,
        examples: str,
        prompt: str,
        additional_keys: Dict[str, str],
    ) -> Tuple[str, List[Response]]:
        """Generates an action.

        Args:
            question (str): The question to be answered.
            examples (str): Few-shot examples to guide the language model in generating the answer.
            prompt (str): The instruction template used to prompt the language model for the answer.
            additional_keys (Dict[str, str]): Additional keys to format the answer prompt.

        Returns:
            Tuple[str, List[Response]]: The generated answer and model responses.
        """
        raise NotImplementedError

    @abstractmethod
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
        raise NotImplementedError

    @abstractmethod
    def generate_summary(
        self,
        question: str,
        previous_trials: str,
        scratchpad: str,
        prompt: str,
        additional_keys: Dict[str, str],
    ) -> Tuple[str | Response]:
        """Generates a summary based on the given inputs.

        Args:
            question (str): The question to be answered.
            previous_trials (str): The previous trials.
            scratchpad (str): The scratchpad containing previous thoughts.
            prompt (str): The prompt or instruction to guide the summary generation.
            additional_keys (Dict[str, str]): Additional keys for the summary generation.

        Returns:
            Tuple[str | Response]: The generated summary or response.

        """
        raise NotImplementedError

    @abstractmethod
    def generate_meta_summary(
        self,
        question: str,
        meta_summaries: str,
        meta_summary_system: str,
        previous_trials: str,
        scratchpad: str,
        prompt: str,
        additional_keys: Dict[str, str],
    ) -> Tuple[str | Response]:
        """Generates a meta-summary based on the given inputs.

        Args:
            question (str): The question to be answered.
            meta_summaries (str): The meta-summaries of the previous steps.
            meta_summary_system (str): The system prompt for the meta-summaries.
            previous_trials (str): The previous trials.
            scratchpad (str): The scratchpad containing previous thoughts.
            prompt (str): The prompt or instruction to guide the meta-summary generation.
            additional_keys (Dict[str, str]): Additional keys for the meta-summary generation.

        Returns:
            Tuple[str | Response]: The generated meta-summary.

        """
        raise NotImplementedError

    @abstractmethod
    def halting_condition(self, finished: bool) -> bool:
        """Checks if the halting condition is met.

        Args:
            finished (bool): Whether the interaction

        Returns:
            bool: True if the halting condition is met, False otherwise.
        """
        raise NotImplementedError

    @abstractmethod
    def react_halting_condition(
        self,
        finished: bool,
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
            meta_summaries (str): Meta-summaries of previous steps.
            meta_summary_system (str): The system prompt for meta-summarization.
            prompt (str): The prompt or instruction to guide the action generation.
            additional_keys (Dict[str, str]): Additional keys for the action generation process.

        Returns:
            bool: True if the halting condition is met, False otherwise. The halting condition is met when the answer is not correct and the current step index is less than the maximum number of steps plus one.
        """
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> None:
        """Resets the strategy's internal state."""
        raise NotImplementedError
