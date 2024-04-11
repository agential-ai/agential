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
    """Test generate method with different benchmarks."""
    question = 'Who was once considered the best kick boxer in the world, however he has been involved in a number of controversies relating to his "unsportsmanlike conducts" in the sport and crimes of violence outside of the ring'

    responses = [
        "Let's think step by step. The kick boxer who fits this description is Badr Hari. So the answer is: Badr Hari.",
        # Other responses omitted for brevity.
    ]
    search = GoogleSearchAPIWrapper(
        google_api_key=google_api_key, google_cse_id=google_cse_id
    )
    agent = CriticAgent(llm=FakeListChatModel(responses=responses), search=search)

    out = agent.generate(question=question)

    assert isinstance(out, str)
