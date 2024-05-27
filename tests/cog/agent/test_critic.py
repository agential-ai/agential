"""Unit tests for CRITIC."""

from unittest.mock import MagicMock

from langchain_community.chat_models.fake import FakeListChatModel
from langchain_community.utilities.google_serper import GoogleSerperAPIWrapper
from langchain_core.language_models.chat_models import BaseChatModel

from agential.cog.agent.critic import CriticAgent
from agential.cog.prompts.critic import (
    CRITIC_CRITIQUE_INSTRUCTION_GSM8K,
    CRITIC_CRITIQUE_INSTRUCTION_HUMANEVAL,
    CRITIC_CRITIQUE_INSTRUCTION_MBPP,
    CRITIC_CRITIQUE_NO_TOOL_INSTRUCTION_GSM8K,
    CRITIC_CRITIQUE_NO_TOOL_INSTRUCTION_HUMANEVAL,
    CRITIC_CRITIQUE_NO_TOOL_INSTRUCTION_MBPP,
    CRITIC_POT_INSTRUCTION_GSM8K,
    CRITIC_POT_INSTRUCTION_HUMANEVAL,
    GSM8K_FEWSHOT_EXAMPLES_CRITIC,
    GSM8K_FEWSHOT_EXAMPLES_CRITIC_NO_TOOL,
    HUMANEVAL_FEWSHOT_EXAMPLES_CRITIC,
    HUMANEVAL_FEWSHOT_EXAMPLES_CRITIC_NO_TOOL,
    MBPP_FEWSHOT_EXAMPLES_CRITIC,
    MBPP_FEWSHOT_EXAMPLES_CRITIC_NO_TOOL,
)
from agential.cog.prompts.benchmarks.gsm8k import GSM8K_FEWSHOT_EXAMPLES_POT
from agential.cog.prompts.benchmarks.humaneval import HUMANEVAL_FEWSHOT_EXAMPLES_POT


def test_init() -> None:
    """Test initialization."""
    llm = FakeListChatModel(responses=["1"])
    search = MagicMock(spec=GoogleSerperAPIWrapper)
    agent = CriticAgent(llm=llm, mode={"qa": "hotpotqa"}, search=search)
    assert isinstance(agent.llm, BaseChatModel)
    assert isinstance(search, GoogleSerperAPIWrapper)


def test_generate() -> None:
    """Test generate method."""
    question = 'Who was once considered the best kick boxer in the world, however he has been involved in a number of controversies relating to his "unsportsmanlike conducts" in the sport and crimes of violence outside of the ring'

    # Test "qa" mode without search tool.
    responses = [
        "Let's think step by step. The person described in the question is Mike Tyson, who was once considered the best kickboxer in the world but has been involved in controversies and crimes of violence. So the answer is: Mike Tyson.",
        'The question specifies that the individual was once considered the best kickboxer in the world, however, Mike Tyson is not a kickboxer, he is a former professional boxer. So the answer is not plausible.\n\n2. Truthfulness:\n\nLet\'s search the question in google:\n\n> Search Query: Who was once considered the best kick boxer in the world, however he has been involved in a number of controversies relating to his "unsportsmanlike conducts" in the sport and crimes of violence outside of the ring\n> Evidence: There is no relevant information available in the search results to confirm the accuracy of the proposed answer.\n\nGiven that the proposed answer incorrectly identifies Mike Tyson as a kickboxer, it is not truthful.\n\nQuestion: Who was once considered the best kick boxer in the world, however he has been involved in a number of controversies relating to his "unsportsmanlike conducts" in the sport and crimes of violence outside of the ring\nHere\'s the most possible answer: The individual described in the question is not Mike Tyson, as he is a former professional boxer, not a kickboxer. Without further information, it is not possible to determine the correct answer.',
        'There is no specific evidence found for this question.\n\nLet\'s search the proposed answer in google:\n\n> Search Query: Mike Tyson kickboxing\n> Evidence: [Mike Tyson - Wikipedia] Tyson made his professional kickboxing debut on May 20, 2006, against Jonathan Paredes in a three-round exhibition. Tyson knocked Paredes down in the first round with a right hook.\n\nThe evidence shows that Mike Tyson did participate in kickboxing, but he is primarily known as a boxer, not a kickboxer. \n\nTherefore, the proposed answer is not accurate in terms of the sport, and it should be revised to reflect the correct sport. \n\nQuestion: Who was once considered the best kick boxer in the world, however he has been involved in a number of controversies relating to his "unsportsmanlike conducts" in the sport and crimes of violence outside of the ring?\nHere\'s the revised answer: The person described in the question is Mike Tyson, who was once considered one of the best professional boxers in the world but has been involved in controversies and crimes of violence. So the answer is: Mike Tyson.',
        '\n> Evidence: [Mike Tyson - Wikipedia] Mike Tyson is not primarily known for kickboxing. He is a former professional boxer who competed from 1985 to 2005.\n\nThe evidence confirms that Mike Tyson is not a kickboxer but a former professional boxer.\n\nTherefore, the proposed answer is incorrect and needs revision.\n\nQuestion: Who was once considered the best kick boxer in the world, however he has been involved in a number of controversies relating to his "unsportsmanlike conducts" in the sport and crimes of violence outside of the ring?\nHere\'s the most possible answer: Let\'s think step by step. The individual described in the question is not Mike Tyson, who is a former professional boxer, not a kickboxer. The correct answer needs further research to identify the kickboxer matching the description provided.',
        'the most possible answer: The individual described in the question is not Mike Tyson, as he is a former professional boxer, not a kickboxer. Unfortunately, without further information or evidence, it is not possible to determine the correct answer to this question.',
    ]
    agent = CriticAgent(
        llm=FakeListChatModel(responses=responses), mode={"qa": "hotpotqa"}
    )
    out = agent.generate(question=question, use_tool=False)
    assert isinstance(out, list)
    assert len(out) == 2

    # Test "qa" mode with search tool.
    search = MagicMock(spec=GoogleSerperAPIWrapper)
    responses = [
        "Let's break it down step by step. The kickboxer who fits this description is Badr Hari. So the answer is: Badr Hari.",
        'The question asks for a kickboxer who fits the description provided, and the answer "Badr Hari" is a plausible response.\n\n2. Truthfulness:\n\nLet\'s search the question in Google:\n\n> Search Query: Who was once considered the best kick boxer in the world, however he has been involved in a number of controversies relating to his "unsportsmanlike conducts" in the sport and crimes of violence outside of the ring\n> Evidence: [Badr Hari - Wikipedia] Badr Hari is a Moroccan-Dutch super heavyweight kickboxer from the Netherlands, fighting out of Mike\'s Gym in Oostzaan. He is a former K-1 Heavyweight Champion (2007-2008) and It\'s Showtime Heavyweight Champion (2009-2010).\n\nThe evidence confirms that Badr Hari fits the description provided in the question.\n\nOverall, the proposed answer is both plausible and truthful.\n\nQuestion: Who was once considered the best kick boxer in the world, however he has been involved in a number of controversies relating to his "unsportsmanlike conducts" in the sport and crimes of violence outside of the ring?\nHere\'s the most possible answer: Let\'s break it down step by step. The kickboxer who fits this description is Badr Hari. So the answer is: Badr Hari.',
        'The evidence does not provide any relevant information about the question.\n\nLet\'s search the proposed answer in Google:\n\n> Search Query: Badr Hari kickboxer controversies\n> Evidence: [Badr Hari - Wikipedia] Badr Hari is a Moroccan-Dutch kickboxer. He is a former K-1 Heavyweight champion, It\'s worth noting that Hari has been involved in several controversies, including unsportsmanlike conduct and criminal charges.\n\nThe evidence supports the claim that Badr Hari fits the description provided in the question.\n\nOverall, the proposed answer is both plausible and truthful.\n\nQuestion: Who was once considered the best kick boxer in the world, however he has been involved in a number of controversies relating to his "unsportsmanlike conducts" in the sport and crimes of violence outside of the ring?\nHere\'s the most possible answer: Let\'s break it down step by step. The kickboxer who fits this description is Badr Hari. So the answer is: Badr Hari.',
        'the most possible answer: The kickboxer who fits this description is Badr Hari. So the answer is: Badr Hari.',
    ]
    agent = CriticAgent(
        llm=FakeListChatModel(responses=responses), 
        mode={"qa": "hotpotqa"},
        search=search
    )
    out = agent.generate(question=question, use_tool=True)
    assert isinstance(out, list)
    assert len(out) == 3

    # Test "math" mode without code interpreter tool.

    # Test "math" mode with code interpreter tool.

    # Test "code" mode without code interpreter tool.

    # Test "code" mode with code interpreter tool.

    # Test "code" mode with code interpreter tool and unit tests.