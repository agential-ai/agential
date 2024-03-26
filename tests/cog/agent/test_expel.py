"""Unit tests for ExpeL."""
import joblib

from langchain_community.chat_models.fake import FakeListChatModel
from langchain_core.language_models.chat_models import BaseChatModel
from discussion_agents.cog.agent.reflexion import ReflexionReActAgent

from discussion_agents.cog.agent.expel import ExpeLAgent
from discussion_agents.cog.modules.memory.expel import ExpeLExperienceMemory, ExpeLInsightMemory

def test_init(expel_experiences_10_fake_path: str) -> None:
    """Test initialization."""
    llm = FakeListChatModel(responses=["1"])

    agent = ExpeLAgent(
        llm=llm,
    )
    assert isinstance(agent.llm, BaseChatModel)
    assert isinstance(agent.reflexion_react_agent, ReflexionReActAgent)
    assert isinstance(agent.experience_memory, ExpeLExperienceMemory)
    assert isinstance(agent.insight_memory, ExpeLInsightMemory)
    assert agent.success_batch_size == 8
    assert agent.experience_memory.experiences == {'idxs': [], 'questions': [], 'keys': [], 'trajectories': [], 'reflections': []}
    assert not agent.experience_memory.success_traj_docs
    assert not agent.experience_memory.vectorstore
    assert not agent.insight_memory.insights

    # Test with all parameters specified except experience memory and reflexion_react_agent.
    agent = ExpeLAgent(
        llm=llm,
        self_reflect_llm=FakeListChatModel(responses=["2"]),
        action_llm=FakeListChatModel(responses=["3"]),
        reflexion_react_kwargs={"max_steps": 3},
        insight_memory=ExpeLInsightMemory(insights=[{"insight": "blah blah", "score": 10}]),
        success_batch_size=10
    )
    assert isinstance(agent.llm, BaseChatModel)
    assert isinstance(agent.reflexion_react_agent, ReflexionReActAgent)
    assert isinstance(agent.experience_memory, ExpeLExperienceMemory)
    assert isinstance(agent.insight_memory, ExpeLInsightMemory)
    assert agent.success_batch_size == 10
    assert agent.reflexion_react_agent.self_reflect_llm.responses == ["2"]
    assert agent.reflexion_react_agent.action_llm.responses == ["3"]
    assert agent.reflexion_react_agent.max_steps == 3
    assert agent.experience_memory.experiences == {'idxs': [], 'questions': [], 'keys': [], 'trajectories': [], 'reflections': []}
    assert not agent.experience_memory.success_traj_docs
    assert not agent.experience_memory.vectorstore
    assert agent.insight_memory.insights == [{"insight": "blah blah", "score": 10}]

    # Test with custom reflexion_react_agent (verify it overrides reflexion_react_kwargs)
    agent = ExpeLAgent(
        llm=llm,
        reflexion_react_kwargs={"max_steps": 100},
        reflexion_react_agent=ReflexionReActAgent(self_reflect_llm=llm, action_llm=llm),
    )
    assert agent.reflexion_react_agent.max_steps == 6

    # Test with custom experience memory (verify correct initialization).
    experiences = joblib.load(expel_experiences_10_fake_path)
    experiences = {key: value[:1] for key, value in experiences.items()}

    agent = ExpeLAgent(
        llm=llm,
        experience_memory=ExpeLExperienceMemory(experiences)
    )
    assert agent.experience_memory.experiences == experiences
    assert agent.insight_memory.insights == []


def test_generate() -> None:
    """Test generate."""

def test_reset(expel_agent: ExpeLAgent) -> None:
    """Test reset."""
    pass

def test_retrieve() -> None:
    """Test retrieve."""

def test_gather_experience() -> None:
    """Test gather_experience."""

def test_extract_insights() -> None:
    """Test extract_insights."""

def test_update_insights() -> None:
    """"Test update_insights."""
