"""Unit tests for Reflexion QA strategies."""

from agential.cog.strategies.reflexion.qa import parse_qa_action


def test_parse_qa_action() -> None:
    """Test the parse_qa_action function."""

    assert parse_qa_action("QA[question]") == ("QA", "question")
    assert parse_qa_action("QA[]") == ("", "")
    assert parse_qa_action("QA") == ("", "")


def test_reflexion_cot_init() -> None:
    """Test ReflexionCoTQAStrategy initialization."""


