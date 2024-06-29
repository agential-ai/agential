"""Reflexion Agent strategies for Code."""

from agential.cog.strategies.reflexion.base import (
    ReflexionCoTBaseStrategy,
    ReflexionReActBaseStrategy,
)


class ReflexionCoTCodeStrategy(ReflexionCoTBaseStrategy):
    pass

class ReflexionReActCodeStrategy(ReflexionReActBaseStrategy):
    pass

class ReflexionCoTHEvalStrategy(ReflexionCoTCodeStrategy):
    """A strategy class for the HumanEval benchmark using the ReflexionCoT agent."""

    pass


class ReflexionCoTMBPPStrategy(ReflexionCoTCodeStrategy):
    """A strategy class for the MBPP benchmark using the ReflexionCoT agent."""

    pass


class ReflexionReActHEvalStrategy(ReflexionReActCodeStrategy):
    """A strategy class for the HumanEval benchmark using the ReflexionReAct agent."""

    pass


class ReflexionReActMBPPStrategy(ReflexionReActCodeStrategy):
    """A strategy class for the MBPP benchmark using the ReflexionReAct agent."""

    pass
