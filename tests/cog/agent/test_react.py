"""Unit tests for ReAct."""

from langchain_community.chat_models.fake import FakeListChatModel
from langchain_core.language_models.chat_models import BaseChatModel

from agential.cog.agent.react import (
    ReActAgent,
)
from agential.cog.prompts.agents.react import (
    REACT_INSTRUCTION_HOTPOTQA,
)
from agential.cog.prompts.benchmarks.hotpotqa import HOTPOTQA_FEWSHOT_EXAMPLES_REACT
from agential.cog.strategies.react.base import ReActBaseStrategy


def test_init() -> None:
    """Test initialization."""
    llm = FakeListChatModel(responses=[])
    agent = ReActAgent(llm=llm, mode={"qa": "hotpotqa"})
    assert isinstance(agent, ReActAgent)
    assert isinstance(agent.llm, BaseChatModel)
    assert agent.mode == {"qa": "hotpotqa"}
    assert isinstance(agent.strategy, ReActBaseStrategy) 


def test_generate() -> None:
    """Test generate."""
    question = 'Who was once considered the best kick boxer in the world, however he has been involved in a number of controversies relating to his "unsportsmanlike conducts" in the sport and crimes of violence outside of the ring'

    responses = [
        "I need to search for the best kickboxer in the world who has been involved in controversies and crimes.\nAction: Search[best kickboxer in the world]\nObservation: Could not find exact match. Similar: ['List of male kickboxers', 'List of female kickboxers', 'Kickboxing in Japan', 'Kickboxing in the United Kingdom', 'Kickboxing in the United States', 'Kickboxing in Thailand', 'Kickboxing in the Netherlands'].\nThought: I couldn't find the exact information, so I will try to find a famous kickboxer known for controversies.\nAction: Search[kickboxer controversies]\nObservation: Could not find exact match. Similar: ['List of male kickboxers', 'List of female kickboxers', 'Kickboxing in Japan', 'Kickboxing in the United Kingdom', 'Kickboxing in the United States', 'Kickboxing in Thailand', 'Kickboxing in the Netherlands'].\nThought: I need to search for famous kickboxers involved in controversies.\nAction: Search[famous kickboxers controversies]\nObservation: Could not find exact match. Similar: ['List of male kickboxers', 'List of female kickboxers', 'Kickboxing in Japan', 'Kickboxing in the United Kingdom', 'Kickboxing in the United States', 'Kickboxing in Thailand', 'Kickboxing in the Netherlands'].\nThought: I need to try a different approach to find the answer.\nAction: Search[kickboxer controversies crimes]\nObservation: Could not find exact match. Similar: ['List of male kickboxers', 'List of female kickboxers', 'Kickboxing in Japan', 'Kickboxing in the United Kingdom', 'Kickboxing in the United States', 'Kickboxing in Thailand', 'Kickboxing in the Netherlands'].\nThought: I need to refine my search terms to find the specific kickboxer mentioned in the question.\nAction: Search[kickboxer crimes violence unsportsmanlike conduct]\nObservation: Could not find exact match. Similar: ['List of male kickboxers', 'List of female kickboxers', 'Kickboxing in Japan', 'Kickboxing in the United Kingdom', 'Kickboxing in the United States', 'Kickboxing in Thailand', 'Kickboxing in the Netherlands'].\nThought: I'm unable to find the specific kickboxer mentioned in the question. \nAction: Finish[unable to find specific kickboxer]",
        "Search[best kickboxer in the world controversies crimes]\nObservation: Could not find exact match. Similar: ['List of male kickboxers', 'List of female kickboxers', 'List of kickboxing organizations', 'List of kickboxing events', 'List of kickboxers', 'List of female kickboxers', 'List of male kickboxers', 'Kickboxing', 'Muay Thai', 'Mixed martial arts']\nThought: I need to search for a list of kickboxers and look for the one who fits the description.\nAction: Search[List of kickboxers]\nObservation: The list includes many kickboxers. I need to look up each one to find the one involved in controversies and crimes.\nThought: I should start with the most famous kickboxers on the list and look for information on controversies and crimes.\nAction: Lookup[controversies crimes]\nObservation: (Result 1 / 1) The kickboxer known as Badr Hari was once considered the best kickboxer in the world but has been involved in controversies and crimes.\nThought: Badr Hari fits the description, so the answer is Badr Hari.\nAction: Finish[Badr Hari]",
        'This passage does not mention the name of the kickboxer. I need to search for another source or keyword that could lead me to the answer.\nAction: Search[Emory Andrew Tate III kickboxing controversies]\nObservation 2: Could not find [Emory Andrew Tate III kickboxing controversies]. Similar: [\'Emory Andrew Tate III\']\nThought: I should search for Emory Andrew Tate III\'s kickboxing career to see if that provides more information about his reputation in the sport.\nAction: Search[Emory Andrew Tate III kickboxing career]\nObservation 3: Emory Andrew Tate III first began to kickbox in 2005, winning several kickboxing titles in the late 2000s and early 2010s.\nThought: Since the previous searches did not provide clear information about the best kickboxer in the world involved in controversies and crimes, I need to reevaluate my approach and consider searching for other well-known kickboxers who fit the description given in the question.\nAction: Search[most controversial kickboxers crimes]\nObservation 4: Could not find [most controversial kickboxers crimes]. Similar: [\'most controversial kickboxers of all time\', \'most controversial boxers\', \'most controversial UFC fighters\']\nThought: I should narrow down my search to specific well-known kickboxers who have been involved in controversies and crimes to find the answer.\nAction: Search[well-known kickboxers controversies crimes]\nObservation 5: Could not find [well-known kickboxers controversies crimes]. Similar: [\'well-known kickboxers 2024\', \'well-known kickboxers 2023\', \'well-known kickboxers 2022\']\nThought: I need to find a more specific keyword that can lead me to the answer. Maybe I can search for "kickboxer controversies crimes" to find a list of kickboxers known for such behavior.\nAction: Search[kickboxer controversies crimes]\nObservation 6: Could not find [kickboxer controversies crimes]. Similar: [\'kickboxer crimes\', \'kickboxer controversies\', \'boxer controversies crimes\']\nThought: The searches are not yielding the desired results. I may have to broaden my search to include more general information about kickboxers known for controversies and crimes to identify the individual in question.\nAction: Search[kickboxers with criminal records controversies]\nObservation 7: Could not find [kickboxers with criminal records controversies]. Similar: [\'famous kickboxers\', \'kickboxers\', \'kickboxers 2023\']\nThought: The information available is not specific enough to pinpoint the kickboxer mentioned in the question. I may need to reconsider the keywords I am using in my searches or seek additional sources to find the answer.\nAction: Finish[unable to determine the answer based on the available information]',
        'Search[former best kickboxer controversies crimes]',
        'This passage still does not mention the name of the kickboxer. I need to search for a different keyword or source that could lead me to the answer.\nAction: Search[kickboxer controversies crimes]\nObservation 3: Sorry, I couldn’t find any information on the kickboxer involved in controversies and crimes. Would you like me to suggest some similar entities to search for?\nThought: Since the search did not yield the answer, I need to try a different approach. I will search for famous kickboxers and look for the one involved in controversies and crimes.\nAction: Search[famous kickboxers]\nObservation 4: Famous kickboxers include: Joe Lewis, Benny Urquidez, Bill Wallace, Peter Aerts, Badr Hari, Ernesto Hoost, and more. \nThought: None of these famous kickboxers are matching the description of being involved in controversies and crimes. I need to try a different search strategy or keyword.\nAction: Search[kickboxers involved in controversies crimes]\nObservation 5: Sorry, I couldn’t find any information on kickboxers specifically involved in controversies and crimes. Would you like me to suggest some similar entities to search for?\nThought: Since the direct search is not yielding results, I need to think of another way to approach this question. I will try to search for kickboxers known for violent behavior outside the ring.\nAction: Search[kickboxers violent behavior outside ring]\nObservation 6: Sorry, I couldn’t find any information on kickboxers known for violent behavior outside the ring. Would you like me to suggest some similar entities to search for?\nThought: This question seems difficult to answer based on the available information. I am unable to find the specific kickboxer mentioned in the question involved in controversies and crimes. \nAction: Finish[unable to find answer]',
        'Search[kickboxing controversies crimes famous]'
    ]
    llm = FakeListChatModel(responses=responses)
    agent = ReActAgent(llm=llm, mode={"qa": "hotpotqa"})

    out = agent.generate(
        question=question,
        examples=HOTPOTQA_FEWSHOT_EXAMPLES_REACT,
        prompt=REACT_INSTRUCTION_HOTPOTQA,
        additional_keys={},
        reset=True,
        max_steps=3
    )
    assert isinstance(out, list)
    assert len(out) == 3