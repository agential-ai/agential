"""Unit tests for ReActAgent."""
from langchain.llms.fake import FakeListLLM

from discussion_agents.cog.agent.react import ReActAgent

def test_init() -> None:
    """Tests initialization."""
    llm = FakeListLLM(responses=["1"])
    agent = ReActAgent(llm=llm)
    assert agent.llm
    assert not agent.page
    assert not agent.result_titles
    assert not agent.lookup_cnt
    assert not agent.lookup_keyword
    assert not agent.lookup_list

def test_search() -> None:
    """Tests search method."""
    llm = FakeListLLM(responses=["1"])
    agent = ReActAgent(llm=llm)
    obs = agent.search(entity="best kick boxer in the world")
    assert isinstance(obs, str)

def test_generate() -> None:
    """Tests generate method."""
    responses = [
        'I need to search for the best kick boxer in the world and find information about his controversies and crimes.\nAction 1: Search[best kick boxer in the world]',
        'Thought 2: Since the question mentions "he", I need to search for male kick boxers.\nAction 2: Search[male kick boxers]',
        'Thought 3: The kick boxer in question must have a Wikipedia page. I should try searching for the name mentioned in the question, possibly with keywords like "controversies" and "crimes". \nAction 3: Search[best kick boxer controversies crimes]',
        'Thought 4: The question mentions that the kick boxer was once considered the best, so I should try searching for "former" best kick boxer.\nAction 4: Search[former best kick boxer controversies crimes]',
        'Thought 5: The question mentions "he" and "crimes of violence outside of the ring", so I should try searching for male kick boxers who have been involved in crimes.\nAction 5: Search[male kick boxers crimes of violence]',
        'Thought 6: The question mentions "he" and "crimes of violence outside of the ring", so I should try searching for male kick boxers who have been involved in crimes of violence.\nAction 6: Search[male kick boxers crimes of violence outside of the ring]',
        'Thought 7: I should try searching for the name mentioned in the question, possibly with keywords like "kick boxer" and "controversies". \nAction 7: Search[kick boxer controversies]'
    ]
    
    q = "Who was once considered the best kick boxer in the world, however he has been involved in a number of controversies relating to his \"unsportsmanlike conducts\" in the sport and crimes of violence outside of the ring"
    llm = FakeListLLM(responses=responses)
    agent = ReActAgent(llm=llm)
    out = agent.generate(observation=q)
    assert isinstance(out, str)