"""ReAct Agent strategies for QA."""

from agential.cog.strategies.react.base import ReActBaseStrategy


class ReActQAStrategy(ReActBaseStrategy):
    """A strategy class for QA benchmarks using the ReAct agent.

    Attributes:
        llm (BaseChatModel): The language model used for generating answers and critiques.
    """




class ReActHotQAStrategy(ReActQAStrategy):
    """A strategy class for the HotpotQA benchmark using the ReAct agent."""

    pass


class ReActTriviaQAStrategy(ReActQAStrategy):
    """A strategy class for the TriviaQA benchmark using the ReAct agent."""

    pass


class ReActAmbigNQStrategy(ReActQAStrategy):
    """A strategy class for the AmbigNQ benchmark using the ReAct agent."""

    pass


class ReActFEVERStrategy(ReActQAStrategy):
    """A strategy class for the FEVER benchmark using the ReAct agent."""

    pass
