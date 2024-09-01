"""Math strategies for standard prompting."""

from agential.prompting.standard.strategies.general import StandardGeneralStrategy


class StandardGSM8KStrategy(StandardGeneralStrategy):
    """A strategy class for the GSM8K benchmark using the standard vanilla prompting."""

    pass


class StandardSVAMPStrategy(StandardGeneralStrategy):
    """A strategy class for the SVAMP benchmark using the standard vanilla prompting."""

    pass


class StandardTabMWPStrategy(StandardGeneralStrategy):
    """A strategy class for the TabMWP benchmark using the standard vanilla prompting."""

    pass
