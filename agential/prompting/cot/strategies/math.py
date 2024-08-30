"""Math strategies for CoT."""

from agential.prompting.cot.strategies.general import CoTGeneralStrategy


class CoTGSM8KStrategy(CoTGeneralStrategy):
    """A strategy class for the GSM8K benchmark using the CoT."""

    pass


class CoTSVAMPStrategy(CoTGeneralStrategy):
    """A strategy class for the SVAMP benchmark using the CoT."""

    pass


class CoTTabMWPStrategy(CoTGeneralStrategy):
    """A strategy class for the TabMWP benchmark using the CoT."""

    pass
