"""Unit tests for Reflexion."""
from langchain.agents.react.base import DocstoreExplorer
from langchain_community.chat_models.fake import FakeListChatModel
from langchain_core.language_models.chat_models import BaseChatModel
from tiktoken.core import Encoding

from discussion_agents.cog.agent.reflexion import ReflexionCoTAgent, ReflexionReActAgent
from discussion_agents.cog.modules.memory.reflexion import ReflexionMemory
from discussion_agents.cog.modules.reflect.reflexion import ReflexionCoTReflector, ReflexionReActReflector


def test_reflexion_cot_init() -> None:
    """Test initialization."""
    agent = ReflexionCoTAgent(
        self_reflect_llm=FakeListChatModel(responses=["1"]),
        action_llm=FakeListChatModel(responses=["1"]),
    )
    assert isinstance(agent, ReflexionCoTAgent)
    assert isinstance(agent.self_reflect_llm, BaseChatModel)
    assert isinstance(agent.action_llm, BaseChatModel)
    assert isinstance(agent.memory, ReflexionMemory)
    assert isinstance(agent.reflector, ReflexionCoTReflector)
    assert agent.max_reflections == 3
    assert agent.max_trials == 1
    assert agent.patience == agent.max_trials 


def test_reflexion_cot_reset(reflexion_cot_agent: ReflexionCoTAgent) -> None:
    """Test reset method."""
    reflexion_cot_agent._finished = True
    reflexion_cot_agent._step_n = 143
    reflexion_cot_agent._answer = "cat"
    reflexion_cot_agent.memory.scratchpad = "dog"
    reflexion_cot_agent.reflector.reflections = ["puppy"]
    reflexion_cot_agent.reflector.reflections_str = "puppy"
    reflexion_cot_agent.reset()
    assert not reflexion_cot_agent._finished
    assert not reflexion_cot_agent.memory.scratchpad
    assert not reflexion_cot_agent.reflector.reflections
    assert not reflexion_cot_agent.reflector.reflections_str
    assert not reflexion_cot_agent._step_n
    assert not reflexion_cot_agent._answer


def test_reflexion_cot_retrieve(reflexion_cot_agent: ReflexionCoTAgent) -> None:
    """Test retrieve method."""
    out = reflexion_cot_agent.retrieve()
    assert isinstance(out, dict)
    assert "scratchpad" in out
    assert out["scratchpad"] == ""


def test_reflexion_cot_reflect() -> None:
    """Test reflect method."""
    reflexion_cot_agent = ReflexionCoTAgent(
        self_reflect_llm=FakeListChatModel(responses=["1"]),
        action_llm=FakeListChatModel(responses=["1"]),
    )

    reflexion_cot_agent.reset()

    # Test last attempt with context.
    gt_reflections_str = "You have attempted to answer the following question before and failed. Below is the last trial you attempted to answer the question.\nQuestion: \n\n(END PREVIOUS TRIAL)\n"
    reflections_str = reflexion_cot_agent.reflect(
        strategy="last_attempt",
        question="",
        context="",
    )
    assert reflections_str == gt_reflections_str

    reflexion_cot_agent.reset()

    # Test last attempt with no context.
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

    assert reflexion_cot_agent.patience >= 1
    assert reflexion_cot_agent.max_trials >= 1
    assert reflexion_cot_agent.patience <= reflexion_cot_agent.max_trials
    assert reflexion_cot_agent._step_n == 0

    out = reflexion_cot_agent.generate(
        question=question, key=key, context=context, strategy=None
    )

    gt_out_str = 'Thought: The question is asking for the acronym that VIVA Media AG changed its name to in 2004. Based on the context, I know that VIVA Media AG is now known as VIVA Media GmbH. Therefore, the acronym "GmbH" stands for "Gesellschaft mit beschränkter Haftung" in German, which translates to "company with limited liability" in English.\nAction: Finish[Company with Limited Liability]\n\nAnswer is INCORRECT'
    assert isinstance(out, list)
    assert len(out) == 1
    assert out[0] == gt_out_str

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

    assert reflexion_cot_agent.patience >= 1
    assert reflexion_cot_agent.max_trials >= 1
    assert reflexion_cot_agent.patience <= reflexion_cot_agent.max_trials
    assert reflexion_cot_agent._step_n == 0

    out = reflexion_cot_agent.generate(
        question=question, key=key, context=context, strategy=None
    )
    gt_out_str = 'Thought: The question is asking for the acronym that VIVA Media AG changed its name to in 2004. Based on the context, I know that VIVA Media AG is now known as VIVA Media GmbH. Therefore, the acronym "GmbH" stands for "Gesellschaft mit beschränkter Haftung" in German, which translates to "company with limited liability" in English.\nAction: Finish[Gesellschaft mit beschränkter Haftung]\n\nAnswer is CORRECT'
    assert isinstance(out, list)
    assert len(out) == 1
    assert out[0] == gt_out_str

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

    assert reflexion_cot_agent.patience >= 1
    assert reflexion_cot_agent.max_trials >= 1
    assert reflexion_cot_agent.patience <= reflexion_cot_agent.max_trials
    assert reflexion_cot_agent._step_n == 0
    out = reflexion_cot_agent.generate(
        question=question, key=key, context=context, strategy=None
    )
    gt_out_str = 'Thought: The question is asking for the acronym that VIVA Media AG changed its name to in 2004. Based on the context, I know that VIVA Media AG is now known as VIVA Media GmbH. Therefore, the acronym "GmbH" stands for "Gesellschaft mit beschränkter Haftung" in German, which translates to "company with limited liability" in English.\nAction: INVALID[Gesellschaft mit beschränkter Haftung]\n\nInvalid action type, please try again.'
    assert isinstance(out, list)
    assert len(out) == 1
    assert out[0] == gt_out_str

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
    assert reflexion_cot_agent.patience >= 1
    assert reflexion_cot_agent.max_trials >= 1
    assert reflexion_cot_agent.patience <= reflexion_cot_agent.max_trials
    assert reflexion_cot_agent._step_n == 0
    out = reflexion_cot_agent.generate(
        question=question, key=key, context=context, strategy="last_attempt"
    )
    gt_out_str = 'Thought: The question is asking for the acronym that VIVA Media AG changed its name to in 2004. Based on the context, I know that VIVA Media AG is now known as VIVA Media GmbH. Therefore, the acronym "GmbH" stands for "Gesellschaft mit beschränkter Haftung" in German, which translates to "company with limited liability" in English.\nAction: Finish[Company with Limited Liability]\n\nAnswer is INCORRECT'
    assert isinstance(out, list)
    assert len(out) == 1
    assert out[0] == gt_out_str

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
    assert reflexion_cot_agent.patience >= 1
    assert reflexion_cot_agent.max_trials >= 1
    assert reflexion_cot_agent.patience <= reflexion_cot_agent.max_trials
    assert reflexion_cot_agent._step_n == 0
    out = reflexion_cot_agent.generate(
        question=question, key=key, context=None, strategy=None
    )
    gt_out_str = "Thought: Let's think step by step. VIVA Media AG changed its name in 2004. The new acronym must stand for the new name of the company. Unfortunately, without further information, it is not possible to determine what the new acronym stands for.\nAction: Finish[Unknown]\n\nAnswer is INCORRECT"
    assert isinstance(out, list)
    assert len(out) == 1
    assert out[0] == gt_out_str


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

    # General generate.
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
        "Finish[unable to determine]",
    ]
    action_llm = FakeListChatModel(responses=responses)
    agent = ReflexionReActAgent(
        self_reflect_llm=FakeListChatModel(responses=["1"]), action_llm=action_llm
    )
    assert agent.patience >= 1
    assert agent.max_trials >= 1
    assert agent.patience <= agent.max_trials
    assert agent._step_n == 1
    out = agent.generate(question=question, key=key, strategy=None)
    assert isinstance(out, list)
    assert len(out) == 1

    # Test generate with reflection (last_attempt_and_reflexion).
    action_responses = [
        "I need to search for VIVA Media AG and find out what their new acronym stands for.",
        "Search[VIVA Media AG]",
        "The search for VIVA Media AG did not yield any results. I should try searching for their new acronym instead.",
        "Search[new acronym for VIVA Media AG]",
        "The search for the new acronym for VIVA Media AG also did not yield any results. I should try searching for any information about the name change in 2004.",
        "Search[VIVA Media AG name change 2004]",
        "The search for information about the name change in 2004 also did not yield any results. It seems that I am unable to find the answer using the available search options. I should consider other sources or methods to find the acronym for VIVA Media AG after their name change.",
        "Finish[unable to find answer]",
        "I need to search for VIVA Media AG and find out what their new acronym stands for.",
        "Search[VIVA Media AG name change]",
        'The search for "VIVA Media AG name change" did not yield any results. I should try searching for the company\'s name change in 2004 using different search terms or variations of the company name.',
        "Search[VIVA Media AG rebranding 2004]",
        "The search for \"VIVA Media AG rebranding 2004\" also did not yield any results. I should try searching for news articles or press releases about the company's name change in 2004. Additionally, I can try searching for information about the company's history or any announcements they made around that time. It's possible that the new acronym may be mentioned in those sources.",
        "Search[VIVA Media AG news articles 2004]",
        "The search for news articles about VIVA Media AG's name change in 2004 also did not yield any results. It seems that there is limited information available about this specific topic. To further investigate, I can try searching for general information about the company's history or any announcements they made during that time period. Additionally, reaching out to industry experts or contacting the company directly may provide more accurate and specific information about their new acronym.",
        "Search[VIVA Media AG history]",
        "The search for general information about VIVA Media AG's history also did not yield any results. It seems that there is limited information available about this company. To further investigate, I can try searching for any announcements or press releases made by the company during the time of their name change in 2004. Additionally, reaching out to industry experts or contacting the company directly may provide more accurate and specific information about their new acronym.",
        "Search[VIVA Media AG press release 2004]",
        "The search for a press release from VIVA Media AG in 2004 also did not yield any results. It seems that there is limited information available about their name change and the new acronym. To further investigate, I can try searching for any interviews or articles featuring the company's executives or employees during that time period. Additionally, reaching out to industry experts or contacting the company directly may provide more accurate and specific information about their new acronym.",
        "Search[VIVA Media AG interview 2004]",
    ]
    self_reflect_responses = [
        "The failure in this reasoning trial was due to the inability to find the necessary information through the available search options. To mitigate this failure, a new plan could involve expanding the search to different sources such as news articles, company websites, or industry publications. Additionally, utilizing different search terms or variations of the company name could help in finding the desired information.",
    ]
    action_llm = FakeListChatModel(responses=action_responses)
    self_reflect_llm = FakeListChatModel(responses=self_reflect_responses)
    agent = ReflexionReActAgent(
        self_reflect_llm=self_reflect_llm, action_llm=action_llm
    )
    assert agent.patience >= 1
    assert agent.max_trials >= 1
    assert agent.patience <= agent.max_trials
    assert agent._step_n == 1
    out = agent.generate(question=question, key=key, strategy=None)

    assert isinstance(out, list)
    assert len(out) == 1

    out = agent.generate(
        question=question, key=key, strategy="last_attempt_and_reflexion"
    )
    assert isinstance(out, list)
    assert len(out) == 1
