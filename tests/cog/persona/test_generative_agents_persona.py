"""Unit tests for Generative Agents Persona."""

from discussion_agents.cog.persona.generative_agents import GenerativeAgentPersona

def test_generative_agent_persona() -> None:
    """Tests GenerativeAgentPersona."""
    persona = GenerativeAgentPersona()
    assert persona.name == "Klaus Mueller"
    assert persona.age == 20
    assert persona.traits == "kind, inquisitive, passionate"
    assert persona.status == "Klaus Mueller is writing a research paper on the effects of gentrification in low-income communities."
    assert persona.lifestyle == "Klaus Mueller goes to bed around 11pm, awakes up around 7am, eats dinner around 5pm."
    