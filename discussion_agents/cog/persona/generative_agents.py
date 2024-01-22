"""GenerativeAgent persona classes."""

from discussion_agents.cog.persona.base import BasePersona


class GenerativeAgentPersona(BasePersona):
    """A general GenerativeAgent persona class with default values.

    This class represents a persona for a Generative Agent, providing specific
    attributes like name, age, traits, status, and lifestyle. These attributes
    are set to default values but can be overridden during instantiation.

    Attributes:
        name (str): Name of the persona.
        age (int): Age of the persona.
        traits (str): Traits describing the persona.
        status (str): Current status of the persona.
        lifestyle (str): Lifestyle description of the persona.
    """

    def __init__(
        self,
        name: str = "Klaus Mueller",
        age: int = 20,
        traits: str = "kind, inquisitive, passionate",
        status: str = "Klaus Mueller is writing a research paper on the effects of gentrification in low-income communities.",
        lifestyle: str = "Klaus Mueller goes to bed around 11pm, wakes up around 7am, eats dinner around 5pm.",
    ) -> None:
        """Initialization."""
        super().__init__(
            name=name, age=age, traits=traits, status=status, lifestyle=lifestyle
        )
