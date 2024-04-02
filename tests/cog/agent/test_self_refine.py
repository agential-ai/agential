"""Unit tests for Self-Refine."""

from discussion_agents.cog.agent.self_refine import SelfRefineAgent
from discussion_agents.cog.modules.memory.self_refine import SelfRefineMemory
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_community.chat_models.fake import FakeListChatModel


def test_init() -> None:
    """Test initialization."""
    agent = SelfRefineAgent(llm=FakeListChatModel(responses=['1']))
    assert isinstance(agent.llm, BaseChatModel)
    assert isinstance(agent.memory, SelfRefineMemory)

    agent = SelfRefineAgent(llm=FakeListChatModel(responses=['1']), memory=SelfRefineMemory(solution=["solution #1"]))
    assert isinstance(agent.llm, BaseChatModel)
    assert isinstance(agent.memory, SelfRefineMemory)
    assert agent.memory.solution[0] == "solution #1"


def test_reset() -> None:
    """Test reset."""
    agent = SelfRefineAgent(llm=FakeListChatModel(responses=['response']))
    agent.memory.add_memories('solution1', 'feedback1')
    assert agent.memory.solution != []
    assert agent.memory.feedback != []
    agent.reset()
    assert agent.memory.solution == []
    assert agent.memory.feedback == []


def test_retrieve() -> None:
    """Test retrieve."""
    agent = SelfRefineAgent(llm=FakeListChatModel(responses=['response']))
    agent.memory.add_memories('solution1', 'feedback1')
    retrieved_memory = agent.retrieve()
    assert retrieved_memory['solution'] == ['solution1']
    assert retrieved_memory['feedback'] == ['feedback1']


def test_generate() -> None:
    """Test generate."""
    question = "A robe takes 2 bolts of blue fiber and half that much white fiber.  How many bolts in total does it take?"
    
    gt_out = ""
    responses = [

    ]
    agent = SelfRefineAgent(llm=FakeListChatModel(responses=responses))
    out = agent.generate(question=question)
    assert out == gt_out