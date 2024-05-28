"""Unit tests for ReAct QA strategies."""
from tiktoken import Encoding
from langchain_core.language_models.chat_models import BaseChatModel
from langchain.agents.react.base import DocstoreExplorer

from langchain_community.chat_models.fake import FakeListChatModel

from agential.cog.strategies.react.qa import ReActQAStrategy


def test_init() -> None:
    """Test ReActQAStrategy initialization."""
    llm = FakeListChatModel(responses=["1"])
    strategy = ReActQAStrategy(llm=llm)
    assert isinstance(strategy, ReActQAStrategy)
    assert isinstance(strategy.llm, BaseChatModel)
    assert strategy.max_steps == 6
    assert strategy.max_tokens == 3896
    assert isinstance(strategy.docstore, DocstoreExplorer)
    assert isinstance(strategy.enc, Encoding)

    assert strategy._step_n == 1
    assert strategy._finished == False


def test_generate() -> None:
    """Tests ReActQAStrategy generate."""

def test_generate_action() -> None:
    """Tests ReActQAStrategy generate_action."""

def test_generate_observation() -> None:
    """Tests ReActQAStrategy generate_observation."""

def test_create_output_dict() -> None:
    """Tests ReActQAStrategy create_output_dict."""

def test_halting_condition() -> None:
    """Tests ReActQAStrategy halting_condition."""

def test_reset() -> None:
    """Tests ReActQAStrategy reset."""

def test_instantiate_strategies() -> None:
    """Test instantiate all QA strategies."""