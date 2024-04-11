"""Unit tests for CRITIC."""

from langchain_community.chat_models.fake import FakeListChatModel
from langchain_community.utilities.google_search import GoogleSearchAPIWrapper
from langchain_core.language_models.chat_models import BaseChatModel

from discussion_agents.cog.agent.critic import CriticAgent


def test_init(google_api_key: str, google_cse_id: str) -> None:
    """Test initialization."""
    llm = FakeListChatModel(responses=["1"])
    search = GoogleSearchAPIWrapper(
        google_api_key=google_api_key, google_cse_id=google_cse_id
    )
    agent = CriticAgent(llm=llm, search=search)
    assert isinstance(agent.llm, BaseChatModel)
    assert isinstance(search, GoogleSearchAPIWrapper)


def test_generate(google_api_key: str, google_cse_id: str) -> None:
    """Test generate method."""
    question = 'Who was once considered the best kick boxer in the world, however he has been involved in a number of controversies relating to his "unsportsmanlike conducts" in the sport and crimes of violence outside of the ring'

    responses = [
        "Let's think step by step. The kick boxer who fits this description is Badr Hari. So the answer is: Badr Hari.",
        'The question asks for a kick boxer who was once considered the best in the world but has been involved in controversies and crimes. The answer "Badr Hari" is a plausible response.\n\n2. Truthfulness:\n\nLet\'s search the question in google:\n\n> Search Query: Who was once considered the best kick boxer in the world, however he has been involved in a number of controversies relating to his "unsportsmanlike conducts" in the sport and crimes of violence outside of the ring\n> Evidence: [Badr Hari - Wikipedia] Badr Hari is a Moroccan-Dutch super heavyweight kickboxer from Amsterdam, fighting out of Mike\'s Gym in Oostzaan. Hari has been a prominent figure in the world of kickboxing and considered one of the best kickboxers in the world.\n\nThe evidence supports that Badr Hari was once considered one of the best kickboxers in the world but has been involved in controversies and crimes.\n\nTherefore, the proposed answer is both plausible and truthful.\n\nQuestion: Who was once considered the best kick boxer in the world, however he has been involved in a number of controversies relating to his "unsportsmanlike conducts" in the sport and crimes of violence outside of the ring\nHere\'s the most possible answer: Let\'s think step by step. The kick boxer who was once considered the best in the world, involved in controversies and crimes is Badr Hari. So the answer is: Badr Hari.',
        "The evidence does not provide specific information about the kick boxer in question.\n\nLet's search the proposed answer in google:\n\n> Search Query: Badr Hari kickboxing controversies\n> Evidence: [Badr Hari - Wikipedia] Badr Hari is a Moroccan-Dutch super heavyweight kickboxer from Amsterdam, fighting out of Mike's Gym in Oostzaan.\n\nThe evidence confirms that Badr Hari is a kick boxer known for controversies and unsportsmanlike conduct, supporting the proposed answer.\n\nOverall, the proposed answer is plausible and truthful based on the evidence found.\n\nQuestion: Who was once considered the best kick boxer in the world, however he has been involved in a number of controversies relating to his \"unsportsmanlike conducts\" in the sport and crimes of violence outside of the ring\nHere's the most possible answer: Let's think step by step. The kick boxer who fits this description is Badr Hari. So the answer is: Badr Hari.",
        'The evidence supports the fact that Badr Hari has been involved in controversies and crimes related to kickboxing and outside of the ring, making him a suitable answer to the question. \n\nTherefore, the proposed answer "Badr Hari" is correct and aligns with the information available.',
        "the most possible answer: Let's think step by step. The kick boxer who fits this description is Badr Hari. So the answer is: Badr Hari.",
    ]
    search = GoogleSearchAPIWrapper(
        google_api_key=google_api_key, google_cse_id=google_cse_id
    )
    agent = CriticAgent(llm=FakeListChatModel(responses=responses), search=search)
    out = agent.generate(question=question)
    assert isinstance(out, str)