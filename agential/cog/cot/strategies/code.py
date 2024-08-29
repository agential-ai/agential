"""Code strategies for CoT."""


from agential.cog.cot.strategies.general import CoTGeneralStrategy

class CoTMBPPCodeStrategy(CoTGeneralStrategy):
    """A strategy class for the MBPP benchmark using CoT."""

    pass


class CoTHEvalCodeStrategy(CoTGeneralStrategy):
    """A strategy class for the HumanEval benchmark using CoT."""

    pass