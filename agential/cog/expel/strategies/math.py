"""ExpeL Agent strategies for Math."""

from agential.cog.expel.strategies.general import ExpeLStrategy

class ExpeLMathStrategy(ExpeLStrategy):
    """A strategy class for Math benchmarks using the ExpeL agent."""
    pass

class ExpeLSVAMPStrategy(ExpeLMathStrategy):
    """A strategy class for the SVAMP benchmark using the ExpeL agent."""
    pass

class ExpeLTabMWPStrategy(ExpeLMathStrategy):
    """A strategy class for the TabMWP benchmark using the ExpeL agent."""
    pass

class ExpeLGSM8KStrategy(ExpeLMathStrategy):
    """A strategy class for the GSM8K benchmark using the ExpeL agent."""
    pass