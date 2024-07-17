"""ExpeL Agent strategies for Code."""

from agential.cog.expel.strategies.general import ExpeLStrategy


class ExpeLCodeStrategy(ExpeLStrategy):
    """A strategy class for Code benchmarks using the ExpeL agent."""

    pass


class ExpeLHEvalStrategy(ExpeLCodeStrategy):
    """A strategy class for the HumanEval benchmark using the ExpeL agent."""

    pass


class ExpeLMBPPStrategy(ExpeLCodeStrategy):
    """A strategy class for the MBPP benchmark using the ExpeL agent."""

    pass
