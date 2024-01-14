"""Unit tests for ReAct."""
from tiktoken import Encoding
from langchain.llms.fake import FakeListLLM
from langchain.agents.react.base import DocstoreExplorer

from discussion_agents.cog.agent.react import ReActAgent, ZeroShotReActAgent


def test_init() -> None:
    llm = FakeListLLM(responses=["1"])
    agent = ReActAgent(llm=llm)
    assert agent.llm
    assert agent.max_steps == 6
    assert agent.max_tokens == 3896
    assert isinstance(agent.docstore, DocstoreExplorer)
    assert isinstance(agent.enc, Encoding)

    assert agent.step_n == 1
    assert agent.finished == False
    assert agent.scratchpad == ""


def test_generate() -> None:
    llm = FakeListLLM(responses=["1"])
    agent = ReActAgent(llm=llm)


def test_zeroshot_react_init() -> None:
    """Tests ZeroShotReActAgent's initialization."""
    agent = ZeroShotReActAgent(llm=FakeListLLM(responses=["1"]))
    assert agent is not None
    assert agent.llm is not None
    assert len(agent.tools) >= 1
    assert agent.agent is not None
