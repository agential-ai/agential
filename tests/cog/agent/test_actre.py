"""Unit tests for ActreReActAgent."""
from langchain.agents.react.base import DocstoreExplorer
from langchain.llms.fake import FakeListLLM
from langchain_community.chat_models.fake import FakeListChatModel
from langchain_core.language_models.chat_models import BaseChatModel
from tiktoken import Encoding

from discussion_agents.cog.agent.actre import (
    ReActOutput,
    ActReAgent,
    ActreReActAgent
)
from discussion_agents.cog.modules.memory.react import ReActMemory
from discussion_agents.cog.prompts.react import (
    HOTPOTQA_FEWSHOT_EXAMPLES,
    REACT_INSTRUCTION_HOTPOTQA,
)


def test_init() -> None:
    """Test initialization."""
    llm = FakeListChatModel(responses=["1"])
    agent = ActreReActAgent(llm=llm)
    assert isinstance(agent, ActreReActAgent)
    assert isinstance(agent.llm, BaseChatModel)
    assert isinstance(agent.memory, ReActMemory)
    assert agent.max_steps == 6
    assert agent.max_tokens == 3896
    assert isinstance(agent.docstore, DocstoreExplorer)
    assert isinstance(agent.enc, Encoding)
    assert agent.epsilon == 0.1

    assert agent._step_n == 1
    assert agent._finished == False
    assert agent.memory.scratchpad == ""


def test_generate() -> None:
    """Test generate."""
    q = 'Who was once considered the best kick boxer in the world, however he has been involved in a number of controversies relating to his "unsportsmanlike conducts" in the sport and crimes of violence outside of the ring'

    # Test a single max step and verify no more steps can be taken.
    gt_out = [
        ReActOutput(
            thought="Thought: I need to search for the best kick boxer in the world, and then find any controversies or crimes they have been involved in.",
            action="Action: Search[best kick boxer]",
            observation="Observation 1: ",
        )
    ]
    responses = [
        ' I need to search for the best kick boxer in the world, and then find any controversies or crimes they have been involved in.\nAction: Search[best kick boxer]\nObservation: (Result 1/1) Ramon Dekkers is considered by many to be the best kickboxer in the world.\nThought: It mentions "unsportsmanlike conducts" and crimes of violence. I need to find more information about Ramon Dekkers.\nAction: Lookup[crimes]\nObservation: (Result 1/1) Dekkers was involved in a number of controversies relating to his "unsportsmanlike conducts" in the sport and crimes of violence outside of the ring.\nSynthesized Reasoning: Based on the observation that Ramon Dekkers is considered the best kickboxer but also involved in controversies and crimes, I decided to look up more details about his specific crimes to better answer the question.\nAction: Lookup[controversies]\nObservation: (Result 1/1) Dekkers was known for his aggressive style and has been involved in a number of controversies, including a bar brawl and an altercation with a bouncer.\nSynthesized Reasoning: The previous observation provided some examples of Dekkers\' controversies, like bar fights. To fully answer the question, I still need more information on his "unsportsmanlike conduct" in the sport.\nAction: Lookup[unsportsmanlike conducts]\nObservation: (Result',
        ' Search[best kick boxer]\nObservation: The best kick boxer in the world is often a highly debated topic, but some notable names include Semmy Schilt, Peter Aerts, Ernesto Hoost, and Ramon Dekkers.\nSynthesized Reasoning: The search revealed several top kickboxers, but to answer the question about controversies and crimes, I should focus on a more recent one. Semmy Schilt is a good candidate to look into further.\nAction: Lookup[controversies and crimes]\nObservation: (Result 1/1) Semmy Schilt has been involved in several controversies, including accusations of using performance-enhancing drugs and unsportsmanlike conducts such as eye-gouging and low blows.\nSynthesized Reasoning: The lookup provided useful information about Schilt\'s controversies, specifically mentioning the "unsportsmanlike conducts" the question asks about. I will look up more details on those incidents.\nAction: Lookup[unsportsmanlike conducts]\nObservation: (Result 1/1) Semmy Schilt has been known for his aggressive and sometimes controversial fighting style, with incidents such as eye-gouging and low blows being reported by his opponents.\nSynthesized Reasoning: I now have good details on Schilt\'s unsportsmanlike conduct in his fights. The question also asks about crimes outside the ring, so I will search for any criminal charges against him to cover that aspect.\nAction: Search[Semmy Schilt criminal record]\nObservation',
    ]
    llm = FakeListChatModel(responses=responses)
    agent = ActreReActAgent(llm=llm, max_steps=1)
    out = agent.generate(
        question=q,
        examples=HOTPOTQA_FEWSHOT_EXAMPLES,
        prompt=REACT_INSTRUCTION_HOTPOTQA,
    )
    assert out
    assert isinstance(out, list)
    for triplet in out:
        assert isinstance(triplet, ReActOutput)
    assert agent._step_n == 2  # Increased by 1 after the step
    assert not agent._finished
    assert out[0].thought == gt_out[0].thought
    assert out[0].action == gt_out[0].action

    # Verify no more steps can be taken.
    out = agent.generate(question=q, reset=False)
    assert not out
    assert isinstance(out, list)
    assert agent._step_n == 2
    assert not agent._finished

    # Test full trajectory/trial till finish.
    gt_out = [
        ReActOutput(
            thought="Thought: I need to search for the best kick boxer in the world, and then find any controversies or crimes they have been involved in.",
            action="Action: Finish[Badr Hari]",
            observation="Observation 1: Badr Hari",
        )
    ]
    responses = [
        ' I need to search for the best kick boxer in the world, and then find any controversies or crimes they have been involved in.\nAction: Search[best kick boxer in the world]\nObservation: (Result 1/1) Ramon Dekkers is considered by many to be the best kickboxer in the world.\nThought: It mentions "unsportsmanlike conducts" and crimes of violence. I need to find more information about Ramon Dekkers.\nAction: Lookup[crimes]\nObservation: (Result 1/1) Dekkers was involved in a number of controversies relating to his "unsportsmanlike conducts" in the sport and crimes of violence outside of the ring.\nSynthesized Reasoning: The observation confirms that Dekkers, considered the best, was involved in controversies and crimes both in and out of the ring. I have enough information to provide a final answer to the question.\nAction: Finish[Ramon Dekkers, considered by many the best kickboxer in the world, was involved in a number of controversies relating to "unsportsmanlike conducts" in the sport and crimes of violence outside the ring.]\nObservation: Ramon Dekkers, considered by many the best kickboxer in the world, was involved in a number of controversies relating to "unsportsmanlike conducts" in the sport and crimes of violence outside the ring.',
    ]
    llm = FakeListChatModel(responses=responses)
    agent = ActreReActAgent(llm=llm, max_steps=5)
    out = agent.generate(question=q)
    assert isinstance(out, list)
    for triplet in out:
        assert isinstance(triplet, ReActOutput)
    assert out[-1].action.startswith("Action: Finish[")
    assert agent._finished


def test_actre_reasoning() -> None:
    """Test ActRe reasoning step."""
    observation = "(Result 1/1) Ramon Dekkers is considered by many to be the best kickboxer in the world."
    action = "Search[controversies and crimes related to Ramon Dekkers]"

    llm = FakeListChatModel(responses=[
        "\nBased on the observation that Ramon Dekkers is considered the best kickboxer, I decided to search for any controversies and crimes related to him to answer the question about his unsportsmanlike conduct and violence outside the ring."
    ])
    actre_agent = ActReAgent(llm)
    reason = actre_agent.generate(observation, action)

    assert reason.strip() == "Based on the observation that Ramon Dekkers is considered the best kickboxer, I decided to search for any controversies and crimes related to him to answer the question about his unsportsmanlike conduct and violence outside the ring."


def test_sample_alternative_action() -> None:
    """Test sampling alternative actions."""
    action1 = "Search[Ramon Dekkers controversies]"
    alt_actions1 = [
        "Search[information about Ramon Dekkers controversies]",
        "Search[details about Ramon Dekkers controversies]",
        "Search[facts about Ramon Dekkers controversies]",
        "Search[Ramon Dekkers controversies overview]",
        "Search[Ramon Dekkers controversies summary]",
        "Search[Ramon Dekkers controversies introduction]",
        "Search[related to Ramon Dekkers controversies]",
        "Search[similar to Ramon Dekkers controversies]",
        "Search[connected with Ramon Dekkers controversies]",
    ]
    assert ActreReActAgent.sample_alternative_action(action1) in alt_actions1

    action2 = "Lookup[unsportsmanlike conduct]"
    alt_actions2 = [
        "Lookup[definition of unsportsmanlike conduct]",
        "Lookup[meaning of unsportsmanlike conduct]",
        "Lookup[explanation of unsportsmanlike conduct]",
        "Lookup[unsportsmanlike conduct in the context of]",
        "Lookup[unsportsmanlike conduct with respect to]",  
        "Lookup[unsportsmanlike conduct in relation to]",
        "Lookup[examples of unsportsmanlike conduct]",
        "Lookup[instances of unsportsmanlike conduct]",
        "Lookup[illustrations of unsportsmanlike conduct]",
    ]
    assert ActreReActAgent.sample_alternative_action(action2) in alt_actions2

    action3 = "Finish[Badr Hari]"
    assert ActreReActAgent.sample_alternative_action(action3) == action3


def test_reset(actre_react_agent: ActreReActAgent) -> None:
    """Test reset."""
    assert actre_react_agent.memory.scratchpad == ""
    actre_react_agent.memory.scratchpad = "abc" 
    assert not actre_react_agent._finished
    actre_react_agent._finished = True
    assert actre_react_agent._step_n == 1
    actre_react_agent._step_n = 10
    actre_react_agent.reset()
    assert actre_react_agent.memory.scratchpad == ""
    assert not actre_react_agent._finished  
    assert actre_react_agent._step_n == 1


def test_retrieve(actre_react_agent: ActreReActAgent) -> None:
    """Test retrieve."""
    out = actre_react_agent.retrieve()
    assert isinstance(out, dict)
    assert "scratchpad" in out
    assert not out["scratchpad"]