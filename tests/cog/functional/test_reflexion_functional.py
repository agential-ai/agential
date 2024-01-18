"""Unit tests for Reflexion functional methods."""
import pytest

from langchain_community.chat_models.fake import FakeListChatModel

from discussion_agents.cog.functional.reflexion import (
    _format_last_attempt,
    _format_reflections,
    _prompt_cot_agent,
    _prompt_cot_reflection,
    _prompt_react_agent,
    _prompt_react_reflection,
    _truncate_scratchpad,
    cot_reflect,
    cot_reflect_last_attempt,
    cot_reflect_last_attempt_and_reflexion,
    cot_reflect_reflexion,
    react_reflect,
    react_reflect_last_attempt,
    react_reflect_last_attempt_and_reflexion,
    react_reflect_reflexion,
)


def test__truncate_scratchpad() -> None:
    """Test _truncate_scratchpad function."""
    scratchpad = "Observation: This is a test.\nAnother line."
    truncated = _truncate_scratchpad(scratchpad, 1600)
    assert truncated == scratchpad

    long_observation = "Observation: " + "long text " * 100
    scratchpad = long_observation + "\nAnother line."
    truncated = _truncate_scratchpad(scratchpad, 100)
    assert long_observation not in truncated
    assert "truncated wikipedia excerpt" in truncated

    scratchpad = "Regular line 1.\nRegular line 2."
    truncated = _truncate_scratchpad(scratchpad, 1600)
    assert truncated == scratchpad

    observation1 = "Observation: short text"
    observation2 = "Observation: " + "long text " * 100
    scratchpad = observation1 + "\n" + observation2
    truncated = _truncate_scratchpad(scratchpad, 100)
    assert observation1 in truncated, "Shorter observation should remain"
    assert observation2 not in truncated, "Longer observation should be truncated"


def test__format_reflections() -> None:
    """Test _format_reflections function."""
    reflections = []
    assert _format_reflections(reflections) == ""

    # Test: Non-empty reflections
    reflections = ["Reflection 1", "Reflection 2"]
    expected_result = "You have attempted to answer following question before and failed. The following reflection(s) give a plan to avoid failing to answer the question in the same way you did previously. Use them to improve your strategy of correctly answering the given question.\nReflections:\n- Reflection 1\n- Reflection 2"
    assert _format_reflections(reflections) == expected_result

    # Test: Reflections with spaces
    reflections = ["  Reflection 1  ", "  Reflection 2"]
    expected_result = "You have attempted to answer following question before and failed. The following reflection(s) give a plan to avoid failing to answer the question in the same way you did previously. Use them to improve your strategy of correctly answering the given question.\nReflections:\n- Reflection 1\n- Reflection 2"
    assert _format_reflections(reflections) == expected_result

    # Test: Custom header
    reflections = ["Reflection"]
    custom_header = "Custom Header: "
    expected_result = "Custom Header: Reflections:\n- Reflection"
    assert _format_reflections(reflections, custom_header) == expected_result


def test__format_last_attempt() -> None:
    """Test _format_last_attempt function."""
    question = "What is the capital of France?"
    scratchpad = "The capital of France is Paris."
    expected_format = "You have attempted to answer the following question before and failed. Below is the last trial you attempted to answer the question.\nQuestion: What is the capital of France?\nThe capital of France is Paris.\n(END PREVIOUS TRIAL)\n"
    result = _format_last_attempt(question, scratchpad)
    assert result == expected_format


def test__prompt_cot_agent() -> None:
    """Test _prompt_cot_agent function."""
    out = _prompt_cot_agent(
        llm=FakeListChatModel(responses=["1"]),
        examples="",
        reflections="",
        question="",
        scratchpad="",
        context="",
    )
    assert isinstance(out, str)
    assert out == "1"

    # Test with no context.
    out = _prompt_cot_agent(
        llm=FakeListChatModel(responses=["1"]),
        examples="",
        reflections="",
        question="",
        scratchpad="",
        context=None,
    )
    assert isinstance(out, str)
    assert out == "1"


def test__prompt_cot_reflection() -> None:
    """Test _prompt_cot_reflection function."""
    out = _prompt_cot_reflection(
        llm=FakeListChatModel(responses=["1"]),
        examples="",
        question="",
        scratchpad="",
        context="",
    )
    assert isinstance(out, str)
    assert out == "1"

    # Test with no context.
    out = _prompt_cot_reflection(
        llm=FakeListChatModel(responses=["1"]),
        examples="",
        question="",
        scratchpad="",
        context=None,
    )
    assert isinstance(out, str)
    assert out == "1"


def test_react_reflect_last_attempt() -> None:
    """Test cot_reflect_last_attempt function."""
    scratchpad = ""
    out = cot_reflect_last_attempt(scratchpad)
    assert out == [""]


def test_cot_reflect_reflexion() -> None:
    """Test cot_reflect_reflexion function."""
    out = cot_reflect_reflexion(
        llm=FakeListChatModel(responses=["1"]),
        reflections=[""],
        examples="",
        question="",
        scratchpad="",
        context="",
    )
    assert isinstance(out, list)
    assert out == ["", "1"]


def test_cot_reflect_last_attempt_and_reflexion() -> None:
    """Test cot_reflect_last_attempt_and_reflexion function."""
    out = cot_reflect_last_attempt_and_reflexion(
        llm=FakeListChatModel(responses=["1"]),
        examples="",
        question="",
        scratchpad="",
        context="",
    )
    assert isinstance(out, list)
    assert out == ["1"]


def test_cot_reflect() -> None:
    """Test cot_reflect function."""
    # Invalid strategy.
    with pytest.raises(NotImplementedError):
        out = cot_reflect(
            strategy="invalid input",
            llm=FakeListChatModel(responses=["1"]),
            reflections=[""],
            examples="",
            question="",
            scratchpad="",
            context="",
        )

    # Last attempt.
    out = cot_reflect(
        strategy="last_attempt",
        llm=FakeListChatModel(responses=["1"]),
        reflections=[""],
        examples="",
        question="",
        scratchpad="",
        context="",
    )
    assert out == [""]

    # Reflexion.
    out = cot_reflect(
        strategy="reflexion",
        llm=FakeListChatModel(responses=["1"]),
        reflections=[""],
        examples="",
        question="",
        scratchpad="",
        context="",
    )
    assert isinstance(out, list)
    assert out == ["", "1"]

    # Last attempt and Reflexion.
    out = cot_reflect(
        strategy="last_attempt_and_reflexion",
        llm=FakeListChatModel(responses=["1"]),
        reflections=[""],
        examples="",
        question="",
        scratchpad="",
        context="",
    )
    assert isinstance(out, list)
    assert out == ["1"]


def test__prompt_react_agent() -> None:
    """Test _prompt_react_agent function."""
    out = _prompt_react_agent(
        llm=FakeListChatModel(responses=["1"]),
        examples="",
        reflections="",
        question="",
        scratchpad="",
    )
    assert isinstance(out, str)
    assert out == "1"


def test__prompt_react_reflection() -> None:
    """Test _prompt_react_reflection function."""
    out = _prompt_react_reflection(
        llm=FakeListChatModel(responses=["1"]),
        examples="",
        question="",
        scratchpad="",
    )
    assert isinstance(out, str)
    assert out == "1"


def test_react_reflect_last_attempt() -> None:
    """Test react_reflect_last_attempt function."""
    scratchpad = ""
    out = react_reflect_last_attempt(scratchpad)
    assert out == [""]


def test_react_reflect_reflexion() -> None:
    """Test react_reflect_reflexion function."""
    out = react_reflect_reflexion(
        llm=FakeListChatModel(responses=["1"]),
        reflections=[""],
        examples="",
        question="",
        scratchpad="",
    )
    assert isinstance(out, list)
    assert out == ["", "1"]


def test_react_reflect_last_attempt_and_reflexion() -> None:
    """Test react_reflect_last_attempt_and_reflexion function."""
    out = react_reflect_last_attempt_and_reflexion(
        llm=FakeListChatModel(responses=["1"]),
        examples="",
        question="",
        scratchpad="",
    )
    assert isinstance(out, list)
    assert out == ["1"]


def test_react_reflect() -> None:
    """Test react_reflect function."""
    # Invalid strategy.
    with pytest.raises(NotImplementedError):
        out = react_reflect(
            strategy="invalid input",
            llm=FakeListChatModel(responses=["1"]),
            reflections=[""],
            examples="",
            question="",
            scratchpad="",
        )

    # Last attempt.
    out = react_reflect(
        strategy="last_attempt",
        llm=FakeListChatModel(responses=["1"]),
        reflections=[""],
        examples="",
        question="",
        scratchpad="",
    )
    assert out == [""]

    # Reflexion.
    out = react_reflect(
        strategy="reflexion",
        llm=FakeListChatModel(responses=["1"]),
        reflections=[""],
        examples="",
        question="",
        scratchpad="",
    )
    assert isinstance(out, list)
    assert out == ["", "1"]

    # Last attempt and Reflexion.
    out = react_reflect(
        strategy="last_attempt_and_reflexion",
        llm=FakeListChatModel(responses=["1"]),
        reflections=[""],
        examples="",
        question="",
        scratchpad="",
    )
    assert isinstance(out, list)
    assert out == ["1"]
