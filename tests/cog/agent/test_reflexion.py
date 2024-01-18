"""Unit tests for Reflexion."""
from langchain_community.chat_models.fake import FakeListChatModel
from langchain_core.language_models.chat_models import BaseChatModel
from langchain.agents.react.base import DocstoreExplorer

from tiktoken.core import Encoding

from discussion_agents.cog.modules.memory.reflexion import ReflexionMemory
from discussion_agents.cog.modules.reflect.reflexion import ReflexionReActReflector
from discussion_agents.cog.agent.reflexion import ReflexionCoTAgent, ReflexionReActAgent


def test_reflexion_cot_init() -> None:
    """Test initialization."""
    agent = ReflexionCoTAgent(
        self_reflect_llm=FakeListChatModel(responses=["1"]),
        action_llm=FakeListChatModel(responses=["1"]),
    )
    assert agent
    assert agent.self_reflect_llm
    assert agent.action_llm
    assert agent.memory
    assert agent.reflector


def test_reflexion_cot_reset(reflexion_cot_agent: ReflexionCoTAgent) -> None:
    """Test reset method."""
    reflexion_cot_agent._finished = True
    reflexion_cot_agent.reset()
    assert not reflexion_cot_agent._finished
    assert reflexion_cot_agent.memory.scratchpad == ""


def test_reflexion_cot_retrieve(reflexion_cot_agent: ReflexionCoTAgent) -> None:
    """Test retrieve method."""
    out = reflexion_cot_agent.retrieve()
    assert isinstance(out, dict)
    assert "scratchpad" in out
    assert out["scratchpad"] == ""


def test_reflexion_cot_reflect(reflexion_cot_agent: ReflexionCoTAgent) -> None:
    """Test reflect method."""
    gt_reflections_str = "You have attempted to answer the following question before and failed. Below is the last trial you attempted to answer the question.\nQuestion: \n\n(END PREVIOUS TRIAL)\n"
    reflections_str = reflexion_cot_agent.reflect(
        strategy="last_attempt",
        question="",
        context="",
    )
    assert reflections_str == gt_reflections_str

    # Test with no context.
    gt_reflections_str = "You have attempted to answer the following question before and failed. Below is the last trial you attempted to answer the question.\nQuestion: \n\n(END PREVIOUS TRIAL)\n"
    reflections_str = reflexion_cot_agent.reflect(
        strategy="last_attempt",
        question="",
        context=None,
    )
    assert reflections_str == gt_reflections_str


def test_reflexion_cot_generate() -> None:
    """Test generate method."""
    question = "VIVA Media AG changed it's name in 2004. What does their new acronym stand for?"
    key = "Gesellschaft mit beschränkter Haftung"
    context = 'VIVA Media GmbH (until 2004 "VIVA Media AG") is a music television network originating from Germany. It was founded for broadcast of VIVA Germany as VIVA Media AG in 1993 and has been owned by their original concurrent Viacom, the parent company of MTV, since 2004. Viva channels exist in some European countries; the first spin-offs were launched in Poland and Switzerland in 2000.\n\nA Gesellschaft mit beschränkter Haftung (] , abbreviated GmbH ] and also GesmbH in Austria) is a type of legal entity very common in Germany, Austria, Switzerland (where it is equivalent to a S.à r.l.) and Liechtenstein. In the United States, the equivalent type of entity is the limited liability company (LLC). The name of the GmbH form emphasizes the fact that the owners ("Gesellschafter", also known as members) of the entity are not personally liable for the company\'s debts. "GmbH"s are considered legal persons under German and Austrian law. Other variations include mbH (used when the term "Gesellschaft" is part of the company name itself), and gGmbH ("gemeinnützige" GmbH) for non-profit companies.'

    # Incorrect.
    action_llm = FakeListChatModel(
        responses=[
            'The question is asking for the acronym that VIVA Media AG changed its name to in 2004. Based on the context, I know that VIVA Media AG is now known as VIVA Media GmbH. Therefore, the acronym "GmbH" stands for "Gesellschaft mit beschränkter Haftung" in German, which translates to "company with limited liability" in English.',
            "Finish[Company with Limited Liability]",
        ]
    )
    reflexion_cot_agent = ReflexionCoTAgent(
        self_reflect_llm=FakeListChatModel(responses=["1"]), action_llm=action_llm
    )
    out = reflexion_cot_agent.generate(
        question=question, key=key, context=context, strategy=None
    )
    gt_out_str = 'Thought: The question is asking for the acronym that VIVA Media AG changed its name to in 2004. Based on the context, I know that VIVA Media AG is now known as VIVA Media GmbH. Therefore, the acronym "GmbH" stands for "Gesellschaft mit beschränkter Haftung" in German, which translates to "company with limited liability" in English.\nAction: Finish[Company with Limited Liability]\n\nAnswer is INCORRECT'
    assert isinstance(out, str)
    assert out == gt_out_str

    # Correct.
    action_llm = FakeListChatModel(
        responses=[
            'The question is asking for the acronym that VIVA Media AG changed its name to in 2004. Based on the context, I know that VIVA Media AG is now known as VIVA Media GmbH. Therefore, the acronym "GmbH" stands for "Gesellschaft mit beschränkter Haftung" in German, which translates to "company with limited liability" in English.',
            "Finish[Gesellschaft mit beschränkter Haftung]",
        ]
    )
    reflexion_cot_agent = ReflexionCoTAgent(
        self_reflect_llm=FakeListChatModel(responses=["1"]), action_llm=action_llm
    )
    out = reflexion_cot_agent.generate(
        question=question, key=key, context=context, strategy=None
    )
    gt_out_str = 'Thought: The question is asking for the acronym that VIVA Media AG changed its name to in 2004. Based on the context, I know that VIVA Media AG is now known as VIVA Media GmbH. Therefore, the acronym "GmbH" stands for "Gesellschaft mit beschränkter Haftung" in German, which translates to "company with limited liability" in English.\nAction: Finish[Gesellschaft mit beschränkter Haftung]\n\nAnswer is CORRECT'
    assert isinstance(out, str)
    assert out == gt_out_str

    # Invalid.
    action_llm = FakeListChatModel(
        responses=[
            'The question is asking for the acronym that VIVA Media AG changed its name to in 2004. Based on the context, I know that VIVA Media AG is now known as VIVA Media GmbH. Therefore, the acronym "GmbH" stands for "Gesellschaft mit beschränkter Haftung" in German, which translates to "company with limited liability" in English.',
            "INVALID[Gesellschaft mit beschränkter Haftung]",
        ]
    )
    reflexion_cot_agent = ReflexionCoTAgent(
        self_reflect_llm=FakeListChatModel(responses=["1"]), action_llm=action_llm
    )
    out = reflexion_cot_agent.generate(
        question=question, key=key, context=context, strategy=None
    )
    gt_out_str = 'Thought: The question is asking for the acronym that VIVA Media AG changed its name to in 2004. Based on the context, I know that VIVA Media AG is now known as VIVA Media GmbH. Therefore, the acronym "GmbH" stands for "Gesellschaft mit beschränkter Haftung" in German, which translates to "company with limited liability" in English.\nAction: INVALID[Gesellschaft mit beschränkter Haftung]\n\nInvalid action type, please try again.'
    assert isinstance(out, str)
    assert out == gt_out_str

    # With reflection strategy on (last attempt).
    action_llm = FakeListChatModel(
        responses=[
            'The question is asking for the acronym that VIVA Media AG changed its name to in 2004. Based on the context, I know that VIVA Media AG is now known as VIVA Media GmbH. Therefore, the acronym "GmbH" stands for "Gesellschaft mit beschränkter Haftung" in German, which translates to "company with limited liability" in English.',
            "Finish[Company with Limited Liability]",
        ]
    )
    reflexion_cot_agent = ReflexionCoTAgent(
        self_reflect_llm=FakeListChatModel(responses=["1"]), action_llm=action_llm
    )
    out = reflexion_cot_agent.generate(
        question=question, key=key, context=context, strategy="last_attempt"
    )
    gt_out_str = 'Thought: The question is asking for the acronym that VIVA Media AG changed its name to in 2004. Based on the context, I know that VIVA Media AG is now known as VIVA Media GmbH. Therefore, the acronym "GmbH" stands for "Gesellschaft mit beschränkter Haftung" in German, which translates to "company with limited liability" in English.\nAction: Finish[Company with Limited Liability]\n\nAnswer is INCORRECT'
    assert isinstance(out, str)
    assert out == gt_out_str

    # With no reflection strategy and no context.
    action_llm = FakeListChatModel(
        responses=[
            "Let's think step by step. VIVA Media AG changed its name in 2004. The new acronym must stand for the new name of the company. Unfortunately, without further information, it is not possible to determine what the new acronym stands for.",
            "Finish[Unknown]",
        ]
    )
    reflexion_cot_agent = ReflexionCoTAgent(
        self_reflect_llm=FakeListChatModel(responses=["1"]), action_llm=action_llm
    )
    out = reflexion_cot_agent.generate(
        question=question, key=key, context=None, strategy=None
    )
    gt_out_str = "Thought: Let's think step by step. VIVA Media AG changed its name in 2004. The new acronym must stand for the new name of the company. Unfortunately, without further information, it is not possible to determine what the new acronym stands for.\nAction: Finish[Unknown]\n\nAnswer is INCORRECT"
    assert isinstance(out, str)
    assert out == gt_out_str


def test_reflexion_react_init() -> None:
    """Test ReflexionReActAgent initialization."""
    llm = FakeListChatModel(responses=["1"])
    agent = ReflexionReActAgent(
        self_reflect_llm=llm,
        action_llm=llm,
    )
    assert isinstance(agent.self_reflect_llm, BaseChatModel)
    assert isinstance(agent.action_llm, BaseChatModel)
    assert isinstance(agent.memory, ReflexionMemory)
    assert isinstance(agent.reflector, ReflexionReActReflector)
    assert agent.max_steps == 6
    assert agent.max_tokens == 3896
    assert isinstance(agent.enc, Encoding)
    assert isinstance(agent.docstore, DocstoreExplorer)


def test_reflexion_react_reset(reflexion_react_agent: ReflexionReActAgent) -> None:
    """Test reset method."""
    reflexion_react_agent._finished = True
    reflexion_react_agent.reset()
    assert not reflexion_react_agent._finished
    assert reflexion_react_agent.memory.scratchpad == ""


def test_reflexion_react_retrieve(reflexion_react_agent: ReflexionReActAgent) -> None:
    """Test retrieve method."""
    out = reflexion_react_agent.retrieve()
    assert isinstance(out, dict)
    assert "scratchpad" in out
    assert out["scratchpad"] == ""


def test_reflexion_react_reflect(reflexion_react_agent: ReflexionReActAgent) -> None:
    """Test reflect method."""
    gt_reflections_str = "You have attempted to answer the following question before and failed. Below is the last trial you attempted to answer the question.\nQuestion: \n\n(END PREVIOUS TRIAL)\n"
    reflections_str = reflexion_react_agent.reflect(
        strategy="last_attempt",
        question="",
    )
    assert reflections_str == gt_reflections_str


def test_reflexion_react_generate() -> None:
    """Test generate method."""
    question = "VIVA Media AG changed it's name in 2004. What does their new acronym stand for?"
    key = "Gesellschaft mit beschränkter Haftung"

    responses = [
        "I need to search for VIVA Media AG and find out what their new acronym stands for.",
        "Search[VIVA Media AG]",
        "The search for VIVA Media AG did not yield any results. I should try searching for their new acronym instead.",
        "Search[new acronym for VIVA Media AG]",
        "The search for the new acronym for VIVA Media AG also did not yield any results. I should try looking for any information about the name change in 2004 and see if it mentions the new acronym.",
        "Lookup[name change of VIVA Media AG in 2004]",
        "The lookup for the name change of VIVA Media AG in 2004 did not yield any results either. I should try searching for any news articles or press releases about the company's name change in 2004.",
        "Search[VIVA Media AG name change 2004 news articles]",
        "The search for information about VIVA Media AG's name change in 2004 did not yield any results. It seems that there is limited information available on this topic. Without further information, I am unable to determine what their new acronym stands for.",
        "Finish[unable to determine]"
    ]
    action_llm = FakeListChatModel(responses=responses)
    agent = ReflexionReActAgent(
        self_reflect_llm=FakeListChatModel(responses=["1"]),
        action_llm=action_llm
    )
    out = agent.generate(
        question=question, 
        key=key, 
        strategy=None
    )
    assert isinstance(out, str)