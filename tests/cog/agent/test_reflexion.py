"""Unit tests for Reflexion."""
from langchain_community.chat_models.fake import FakeListChatModel

from discussion_agents.cog.agent.reflexion import ReflexionCoTAgent

def test_reflexion_cot_init() -> None:
    """Test initialization."""
    agent = ReflexionCoTAgent(
        self_reflect_llm=FakeListChatModel(responses=["1"]),
        action_llm=FakeListChatModel(responses=["1"])
    )
    assert agent
    assert agent.self_reflect_llm
    assert agent.action_llm
    assert agent.memory
    assert agent.reflector


def test_reflexion_cot_is_finished(reflexion_cot_agent: ReflexionCoTAgent) -> None:
    """Test is_finished method."""
    assert not reflexion_cot_agent.is_finished()
    reflexion_cot_agent.finished = True
    assert reflexion_cot_agent.is_finished()


def test_reflexion_cot_reset(reflexion_cot_agent: ReflexionCoTAgent) -> None:
    """Test reset method."""
    reflexion_cot_agent.finished = True
    reflexion_cot_agent.reset()
    assert not reflexion_cot_agent.is_finished()
    assert reflexion_cot_agent.memory.scratchpad == ""

def test_reflexion_cot_retrieve(reflexion_cot_agent: ReflexionCoTAgent) -> None:
    """Test retrieve method."""
    out = reflexion_cot_agent.retrieve()
    assert isinstance(out, dict)
    assert "scratchpad" in out
    assert out["scratchpad"] == ""

def test_reflexion_cot_reflect(reflexion_cot_agent: ReflexionCoTAgent) -> None:
    """Test reflect method."""
    pass


def test_reflexion_cot_generate(reflexion_cot_agent: ReflexionCoTAgent) -> None:
    """Test generate method."""
    pass
