"""QA strategies for standard prompting."""

from agential.prompting.standard.strategies.general import StandardGeneralStrategy


class StandardHotQAStrategy(StandardGeneralStrategy):
    """A strategy class for the HotpotQA benchmark using standard vanilla prompting."""

    pass


class StandardTriviaQAStrategy(StandardGeneralStrategy):
    """A strategy class for the TriviaQA benchmark using standard vanilla prompting."""

    pass


class StandardAmbigNQStrategy(StandardGeneralStrategy):
    """A strategy class for the AmbigNQ benchmark using standard vanilla prompting."""

    pass


class StandardFEVERStrategy(StandardGeneralStrategy):
    """A strategy class for the FEVER benchmark using standard vanilla prompting."""

    pass
