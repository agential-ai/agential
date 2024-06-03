"""Unit tests for Reflexion QA strategies."""
from langchain_community.chat_models.fake import FakeListChatModel
from langchain_core.language_models.chat_models import BaseChatModel

from agential.cog.modules.reflect.reflexion import ReflexionCoTReflector
from agential.cog.strategies.reflexion.qa import (
    parse_qa_action,
    ReflexionCoTQAStrategy,
)


def test_parse_qa_action() -> None:
    """Test the parse_qa_action function."""
    assert parse_qa_action("QA[question]") == ("QA", "question")
    assert parse_qa_action("QA[]") == ("", "")
    assert parse_qa_action("QA") == ("", "")


def test_reflexion_cot_init() -> None:
    """Test ReflexionCoTQAStrategy initialization."""
    llm = FakeListChatModel(responses=[])
    strategy = ReflexionCoTQAStrategy(llm=llm)
    assert isinstance(strategy.llm, BaseChatModel)
    assert isinstance(strategy.reflector, ReflexionCoTReflector)
    assert strategy.max_reflections == 3
    assert strategy.max_trials == 1
    assert strategy._scratchpad == ""
    assert strategy._finished == False
    assert strategy._answer == ""


def test_reflexion_cot_generate() -> None:
    """Tests ReflexionCoTQAStrategy generate."""


def test_reflexion_cot_generate_action() -> None:
    """Tests ReflexionCoTQAStrategy generate_action."""


def test_reflexion_cot_generate_observation() -> None:
    """Tests ReflexionCoTQAStrategy generate_observation."""


def test_reflexion_cot_create_output_dict() -> None:
    """Tests ReflexionCoTQAStrategy create_output_dict."""


def test_reflexion_cot_halting_condition() -> None:
    """Tests ReflexionCoTQAStrategy halting_condition."""


def test_reflexion_cot_reset() -> None:
    """Tests ReflexionCoTQAStrategy reset."""


def test_reflexion_cot_reflect() -> None:
    """Tests ReflexionCoTQAStrategy reflect."""


def test_reflexion_cot_should_reflect() -> None:
    """Tests ReflexionCoTQAStrategy should_reflect."""
