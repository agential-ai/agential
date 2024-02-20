"""Unit tests for ReAct."""
from langchain.agents.react.base import DocstoreExplorer
from langchain.llms.fake import FakeListLLM
from langchain_community.chat_models.fake import FakeListChatModel
from tiktoken import Encoding

from discussion_agents.cog.agent.react import ReActAgent, ZeroShotReActAgent


def test_init() -> None:
    """Test initialization."""
    llm = FakeListChatModel(responses=["1"])
    agent = ReActAgent(llm=llm)
    assert agent.llm
    assert agent.memory
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

    # Test simple run.
    responses = [
        ' I need to search for the best kick boxer in the world, and then find any controversies or crimes they have been involved in.\nAction: Search[best kick boxer in the world]\nObservation: (Result 1/1) Ramon Dekkers is considered by many to be the best kickboxer in the world.\nThought: It mentions "unsportsmanlike conducts" and crimes of violence. I need to find more information about Ramon Dekkers.\nAction: Lookup[crimes]\nObservation: (Result 1/1) Dekkers was involved in a number of controversies relating to his "unsportsmanlike conducts" in the sport and crimes of violence outside of the ring.\nThought: Ramon Dekkers has been involved in controversies and crimes. I need to find more information about them.\nAction: Lookup[controversies]\nObservation: (Result 1/1) Dekkers was known for his aggressive style and has been involved in a number of controversies, including a bar brawl and an altercation with a bouncer.\nThought: It mentions "unsportsmanlike conducts" and crimes of violence. I need to find more information about the controversies and crimes.\nAction: Lookup[unsportsmanlike conducts]\nObservation: (Result',
        ' Search[best kick boxer]\nObservation: The best kick boxer in the world is often a highly debated topic, but some notable names include Semmy Schilt, Peter Aerts, Ernesto Hoost, and Ramon Dekkers.\nThought: Since the question mentions controversies and crimes, I should focus on more recent kick boxers. I will look up the controversies and crimes of Semmy Schilt.\nAction: Lookup[controversies and crimes]\nObservation: (Result 1/1) Semmy Schilt has been involved in several controversies, including accusations of using performance-enhancing drugs and unsportsmanlike conducts such as eye-gouging and low blows.\nThought: The question mentions "unsportsmanlike conducts" specifically, so I will look up more information on those incidents.\nAction: Lookup[unsportsmanlike conducts]\nObservation: (Result 1/1) Semmy Schilt has been known for his aggressive and sometimes controversial fighting style, with incidents such as eye-gouging and low blows being reported by his opponents.\nThought: The question also mentions crimes outside of the ring, so I will search for any criminal record or charges against Semmy Schilt.\nAction: Search[Semmy Schilt criminal record]\nObservation',
    ]
    llm = FakeListChatModel(responses=responses)
    agent = ReActAgent(llm=llm)
    out = agent.generate(question=q)
    assert isinstance(out, str)
    assert agent._step_n == agent.max_steps + 1
    assert not agent._finished


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


def test_FEVER_react_generate() -> None:

    """Test generate."""
    q = 'Brad Wilk died before being a drummer for Greta.'

    # Test simple run.
    responses = [
        'Thought: I need to search for Brad Wilk and see if he died before being a drummer for Greta.\nAction: Search[Brad Wilk]\\nObservation 1: Could not find [Brad Wilk]. Similar: [\'Brad Wilk\', \'Rage Against the Machine\', \'Audioslave\', \'Prophets of Rage\', \'Wilk\', \'Tim Commerford\', \'Tom Morello\', \'Show Me How to Live (song)\', \'Greta (band)\', \'Zack de la Rocha\']\nThought: The search did not return Brad Wilk directly, but it mentioned Greta (band). I should search for Greta (band) to see if Brad Wilk was a drummer for them.\nAction: Search[Greta (band)]\nObservation 2: Greta was an American hard rock band formed in 1990 by Paul Plagens, Kyle Baer, Josh Gordon, and Brad Wilk.Wilk left the band in 1991 to join Rage Against the Machine, and was replaced on the drums by Scott Carneghi who went on to co-found the band Buffalocomotive in 2012.The band signed a two-record deal with Mercury Records in 1993 and released their debut album, No Biting, on September 21 of that year. Their second and final album, This Is Greta, was released in 1995, after which the band was dropped from the label. Greta disbanded in 1995. Vocalist Paul Plagens died in 2015\nThought: The observation clearly states that Brad Wilk left Greta in 1991 to join Rage Against the Machine, so he did not die before being a drummer for Greta.\nAction: Finish[REFUTES]\nObservation 3: REFUTES' 
    ]
    llm = FakeListChatModel(responses=responses)
    agent = ReActAgent(llm=llm)
    out = agent.generate(question=q)
    assert isinstance(out, str)
    assert agent._step_n == agent.max_steps + 1
    assert not agent._finished
    return