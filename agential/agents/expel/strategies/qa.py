"""ExpeL Agent strategies for QA."""

from agential.agents.expel.strategies.general import ExpeLGeneralStrategy


class ExpeLQAStrategy(ExpeLGeneralStrategy):
    """A strategy class for QA benchmarks using the ExpeL agent."""

    pass


class ExpeLHotQAStrategy(ExpeLQAStrategy):
    """A strategy class for the HotpotQA benchmark using the ExpeL agent."""

    pass


class ExpeLTriviaQAStrategy(ExpeLQAStrategy):
    """A strategy class for the TriviaQA benchmark using the ExpeL agent."""

    pass


class ExpeLAmbigNQStrategy(ExpeLQAStrategy):
    """A strategy class for the AmbigNQ benchmark using the ExpeL agent."""

    pass


class ExpeLFEVERStrategy(ExpeLQAStrategy):
    """A strategy class for the FEVER benchmark using the ExpeL agent."""

    pass
