"""ExpeL Agent strategies for Code."""

from agential.agents.expel.strategies.general import ExpeLGeneralStrategy


class ExpeLCodeStrategy(ExpeLGeneralStrategy):
    """A strategy class for Code benchmarks using the ExpeL agent."""

    pass


class ExpeLHEvalStrategy(ExpeLCodeStrategy):
    """A strategy class for the HumanEval benchmark using the ExpeL agent."""

    pass


class ExpeLMBPPStrategy(ExpeLCodeStrategy):
    """A strategy class for the MBPP benchmark using the ExpeL agent."""

    pass
