"""Code strategies for CoT."""


from agential.cog.cot.strategies.general import CoTGeneralStrategy

class CoTMBPPStrategy(CoTGeneralStrategy):
    """A strategy class for the MBPP benchmark using CoT."""

    pass


class CoTHEvalStrategy(CoTGeneralStrategy):
    """A strategy class for the HumanEval benchmark using CoT."""

    pass