"""QA strategies for CoT."""

from agential.prompting.cot.strategies.general import CoTGeneralStrategy


class CoTHotQAStrategy(CoTGeneralStrategy):
    """A strategy class for the HotpotQA benchmark using Chain of Thought."""

    pass


class CoTTriviaQAStrategy(CoTGeneralStrategy):
    """A strategy class for the TriviaQA benchmark using Chain of Thought."""

    pass


class CoTAmbigNQStrategy(CoTGeneralStrategy):
    """A strategy class for the AmbigNQ benchmark using Chain of Thought."""

    pass


class CoTFEVERStrategy(CoTGeneralStrategy):
    """A strategy class for the FEVER benchmark using Chain of Thought."""

    pass
