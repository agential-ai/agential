"""Unit tests for ReAct."""
from langchain.agents.react.base import DocstoreExplorer
from langchain.llms.fake import FakeListLLM
from langchain_community.chat_models.fake import FakeListChatModel
from langchain_core.language_models.chat_models import BaseChatModel
from tiktoken import Encoding

from discussion_agents.cog.prompts.react import (
    HOTPOTQA_FEWSHOT_EXAMPLES,
    REACT_INSTRUCTION_HOTPOTQA,
)
from discussion_agents.cog.agent.react import ReActAgent, ZeroShotReActAgent
from discussion_agents.cog.modules.memory.react import ReActMemory


def test_init() -> None:
    """Test initialization."""
    llm = FakeListChatModel(responses=["1"])
    agent = ReActAgent(llm=llm)
    assert isinstance(agent, ReActAgent)
    assert isinstance(agent.llm, BaseChatModel)
    assert isinstance(agent.memory, ReActMemory)
    assert agent.max_steps == 6
    assert agent.max_tokens == 3896
    assert isinstance(agent.docstore, DocstoreExplorer)
    assert isinstance(agent.enc, Encoding)

    assert agent._step_n == 1
    assert agent._finished == False
    assert agent.memory.scratchpad == ""


def test_generate() -> None:
    """Test generate."""
    q = 'Who was once considered the best kick boxer in the world, however he has been involved in a number of controversies relating to his "unsportsmanlike conducts" in the sport and crimes of violence outside of the ring'

    # Test a single max step and verify no more steps can be taken.
    responses = [
        ' I need to search for the best kick boxer in the world, and then find any controversies or crimes they have been involved in.\nAction: Search[best kick boxer in the world]\nObservation: (Result 1/1) Ramon Dekkers is considered by many to be the best kickboxer in the world.\nThought: It mentions "unsportsmanlike conducts" and crimes of violence. I need to find more information about Ramon Dekkers.\nAction: Lookup[crimes]\nObservation: (Result 1/1) Dekkers was involved in a number of controversies relating to his "unsportsmanlike conducts" in the sport and crimes of violence outside of the ring.\nThought: Ramon Dekkers has been involved in controversies and crimes. I need to find more information about them.\nAction: Lookup[controversies]\nObservation: (Result 1/1) Dekkers was known for his aggressive style and has been involved in a number of controversies, including a bar brawl and an altercation with a bouncer.\nThought: It mentions "unsportsmanlike conducts" and crimes of violence. I need to find more information about the controversies and crimes.\nAction: Lookup[unsportsmanlike conducts]\nObservation: (Result',
        ' Search[best kick boxer]\nObservation: The best kick boxer in the world is often a highly debated topic, but some notable names include Semmy Schilt, Peter Aerts, Ernesto Hoost, and Ramon Dekkers.\nThought: Since the question mentions controversies and crimes, I should focus on more recent kick boxers. I will look up the controversies and crimes of Semmy Schilt.\nAction: Lookup[controversies and crimes]\nObservation: (Result 1/1) Semmy Schilt has been involved in several controversies, including accusations of using performance-enhancing drugs and unsportsmanlike conducts such as eye-gouging and low blows.\nThought: The question mentions "unsportsmanlike conducts" specifically, so I will look up more information on those incidents.\nAction: Lookup[unsportsmanlike conducts]\nObservation: (Result 1/1) Semmy Schilt has been known for his aggressive and sometimes controversial fighting style, with incidents such as eye-gouging and low blows being reported by his opponents.\nThought: The question also mentions crimes outside of the ring, so I will search for any criminal record or charges against Semmy Schilt.\nAction: Search[Semmy Schilt criminal record]\nObservation',
    ]
    llm = FakeListChatModel(responses=responses)
    agent = ReActAgent(llm=llm)
    out = agent.generate(question=q, examples=HOTPOTQA_FEWSHOT_EXAMPLES, prompt_template=REACT_INSTRUCTION_HOTPOTQA)
    assert isinstance(out, str)
    assert agent._step_n <= agent.max_steps
    assert not agent._finished

    # Verify no more steps can be taken.
    out = agent.generate(question=q, reset=False)
    assert not out
    assert isinstance(out, list)
    for triplet in out:
        assert isinstance(triplet, tuple)
    assert agent._step_n == agent.max_steps + 1
    assert not agent._finished

    scratchpad = "\n".join(agent.retrieve()["scratchpad"].split("\n")[:-1])
    assert scratchpad.strip() == gt_out

    # Test agent runs out of tokens (must ensure that max_steps is not reached and task is not finished).
    responses = [
        ' I need to search for the best kick boxer in the world, and then find any controversies or crimes they have been involved in.\nAction: Search[best kick boxer in the world]\nObservation: (Result 1/1) Ramon Dekkers is considered by many to be the best kickboxer in the world.\nThought: It mentions "unsportsmanlike conducts" and crimes of violence. I need to find more information about Ramon Dekkers.\nAction: Lookup[crimes]\nObservation: (Result 1/1) Dekkers was involved in a number of controversies relating to his "unsportsmanlike conducts" in the sport and crimes of violence outside of the ring.\nThought: Ramon Dekkers has been involved in controversies and crimes. I need to find more information about them.\nAction: Lookup[controversies]\nObservation: (Result 1/1) Dekkers was known for his aggressive style and has been involved in a number of controversies, including a bar brawl and an altercation with a bouncer.\nThought: It mentions "unsportsmanlike conducts" and crimes of violence. I need to find more information about the controversies and crimes.\nAction: Lookup[unsportsmanlike conducts]\nObservation: (Result',
        " INVALID[best kick boxer]\n",
    ]
    llm = FakeListChatModel(responses=responses)
    agent = ReActAgent(
        llm=llm, max_steps=3, max_tokens=1750
    )  # 3 steps leads to 1774 tokens.
    out = agent.generate(question=q)

    gt_out = (
        "Thought: I need to search for the best kick boxer in the world, and then find any controversies or crimes they have been involved in.\n"
        "Action: INVALID[best kick boxer]\n"
        "Observation 1: Invalid Action. Valid Actions are Lookup[<topic>] Search[<topic>] and Finish[<answer>].\n"
        "Thought: I need to search for the best kick boxer in the world, and then find any controversies or crimes they have been involved in.\n"
        "Action: INVALID[best kick boxer]\n"
        "Observation 2: Invalid Action. Valid Actions are Lookup[<topic>] Search[<topic>] and Finish[<answer>]."
    )

    assert isinstance(out, list)
    for triplet in out:
        assert isinstance(triplet, tuple)
    assert "\n".join(["\n".join(triplet) for triplet in out]) == gt_out
    assert agent.memory.load_memories()["scratchpad"].strip() == gt_out

    # Test full trajectoy/trial till finish.
    responses = [
        ' I need to search for the best kick boxer in the world, and then find any controversies or crimes they have been involved in.\nAction: Search[best kick boxer in the world]\nObservation: (Result 1/1) Ramon Dekkers is considered by many to be the best kickboxer in the world.\nThought: It mentions "unsportsmanlike conducts" and crimes of violence. I need to find more information about Ramon Dekkers.\nAction: Lookup[crimes]\nObservation: (Result 1/1) Dekkers was involved in a number of controversies relating to his "unsportsmanlike conducts" in the sport and crimes of violence outside of the ring.\nThought: Ramon Dekkers has been involved in controversies and crimes. I need to find more information about them.\nAction: Lookup[controversies]\nObservation: (Result 1/1) Dekkers was known for his aggressive style and has been involved in a number of controversies, including a bar brawl and an altercation with a bouncer.\nThought: It mentions "unsportsmanlike conducts" and crimes of violence. I need to find more information about the controversies and crimes.\nAction: Lookup[unsportsmanlike conducts]\nObservation: (Result',
        " Finish[Badr Hari]\n",
    ]
    llm = FakeListChatModel(responses=responses)
    agent = ReActAgent(llm=llm, max_steps=5)
    out = agent.generate(question=q)
    gt_out = (
        "\n"
        "Thought: I need to search for the best kick boxer in the world, and then find any controversies or crimes they have been involved in.\n"
        "Action: Finish[Badr Hari]\n"
        "Observation 1: Badr Hari"
    )
    assert isinstance(out, list)
    for triplet in out:
        assert isinstance(triplet, tuple)
    assert gt_out == gt_out
    assert agent.memory.load_memories()["scratchpad"] == gt_out


def test_reset(react_agent: ReActAgent) -> None:
    """Test reset."""
    assert react_agent.memory.scratchpad == ""
    react_agent.memory.scratchpad = "abc"
    assert not react_agent._finished
    react_agent._finished = True
    assert react_agent._step_n == 1
    react_agent._step_n = 10
    react_agent.reset()
    assert react_agent.memory.scratchpad == ""
    assert not react_agent._finished
    assert react_agent._step_n == 1


def test_retrieve(react_agent: ReActAgent) -> None:
    """Test retrieve."""
    out = react_agent.retrieve()
    assert isinstance(out, dict)
    assert "scratchpad" in out
    assert not out["scratchpad"]


def test_zeroshot_react_init() -> None:
    """Tests ZeroShotReActAgent's initialization."""
    agent = ZeroShotReActAgent(llm=FakeListLLM(responses=["1"]))
    assert agent is not None
    assert agent.llm is not None
    assert len(agent.tools) >= 1
    assert agent.agent is not None