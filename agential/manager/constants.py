"""Constants for supported benchmarks, few-shot types, and agents."""

class Benchmarks:
    """Supported benchmarks."""

    # QA.
    HOTPOTQA = "hotpotqa"
    FEVER = "fever"
    TRIVIAQA = "triviaqa"
    AMBIGNQ = "ambignq"

    # Math.
    GSM8K = "gsm8k"
    SVAMP = "svamp"
    TABMWP = "tabmwp"

    # Code.
    HUMANEVAL = "humaneval"
    MBPP = "mbpp"

class FewShotType:
    """Few-shot types."""

    COT = "cot"
    DIRECT = "direct"
    REACT = "react"
    POT = "pot"

class Agents:
    """Supported agents."""

    REACT = "react"
    REFLEXION_COT = "reflexion_cot"
    REFLEXION_REACT = "reflexion_react"
    CRITIC = "critic"