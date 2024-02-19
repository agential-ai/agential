"""Unit tests for Reflexion functional methods."""
import pytest

from langchain_community.chat_models.fake import FakeListChatModel
from discussion_agents.cog.prompts.reflexion import (
    REFLEXION_COT_FEWSHOT_EXAMPLES_NO_CONTEXT, 
    REFLEXION_COT_FEWSHOT_EXAMPLES
)

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

    # Test non-truncated case.
    scratchpad = "Observation: This is a test.\nAnother line."
    truncated = _truncate_scratchpad(scratchpad, 1600)
    assert truncated == scratchpad

    # Test truncated case.
    gt_out = "Observation: [truncated wikipedia excerpt]\nAnother line."
    long_observation = "Observation: " + "long text " * 100
    scratchpad = long_observation + "\nAnother line."
    truncated = _truncate_scratchpad(scratchpad, 100)
    assert long_observation not in truncated
    assert "truncated wikipedia excerpt" in truncated
    assert gt_out == truncated

    # Test non-truncated case with random format.
    scratchpad = "Regular line 1.\nRegular line 2."
    truncated = _truncate_scratchpad(scratchpad, 1600)
    assert truncated == scratchpad

    # Test truncated case with long text and multiple observations.
    gt_out = "Observation: short text\nObservation: [truncated wikipedia excerpt]"
    observation1 = "Observation: short text"
    observation2 = "Observation: " + "long text " * 100
    scratchpad = observation1 + "\n" + observation2
    truncated = _truncate_scratchpad(scratchpad, 100)
    assert observation1 in truncated, "Shorter observation should remain"
    assert observation2 not in truncated, "Longer observation should be truncated"
    assert gt_out == truncated

def test__format_reflections() -> None:
    """Test _format_reflections function."""

    # Test empty.
    reflections = []
    assert _format_reflections(reflections) == ""

    # Test non-empty reflections.
    reflections = ["Reflection 1", "Reflection 2"]
    expected_result = "You have attempted to answer following question before and failed. The following reflection(s) give a plan to avoid failing to answer the question in the same way you did previously. Use them to improve your strategy of correctly answering the given question.\nReflections:\n- Reflection 1\n- Reflection 2"
    assert _format_reflections(reflections) == expected_result

    # Test reflections with spaces.
    reflections = ["  Reflection 1  ", "  Reflection 2"]
    expected_result = "You have attempted to answer following question before and failed. The following reflection(s) give a plan to avoid failing to answer the question in the same way you did previously. Use them to improve your strategy of correctly answering the given question.\nReflections:\n- Reflection 1\n- Reflection 2"
    assert _format_reflections(reflections) == expected_result

    # Test custom header.
    reflections = ["Reflection"]
    custom_header = "Custom Header: "
    expected_result = "Custom Header: Reflections:\n- Reflection"
    assert _format_reflections(reflections, custom_header) == expected_result


def test__format_last_attempt() -> None:
    """Test _format_last_attempt function."""

    # Test simple case.
    question = "What is the capital of France?"
    scratchpad = "The capital of France is Paris."
    expected_format = "You have attempted to answer the following question before and failed. Below is the last trial you attempted to answer the question.\nQuestion: What is the capital of France?\nThe capital of France is Paris.\n(END PREVIOUS TRIAL)\n"
    result = _format_last_attempt(question, scratchpad)
    assert result == expected_format


def test__prompt_cot_agent() -> None:
    """Test _prompt_cot_agent function."""

    q = "VIVA Media AG changed it's name in 2004. What does their new acronym stand for?"
    context = 'VIVA Media GmbH (until 2004 "VIVA Media AG") is a music television network originating from Germany. It was founded for broadcast of VIVA Germany as VIVA Media AG in 1993 and has been owned by their original concurrent Viacom, the parent company of MTV, since 2004. Viva channels exist in some European countries; the first spin-offs were launched in Poland and Switzerland in 2000.\n\nA Gesellschaft mit beschränkter Haftung (] , abbreviated GmbH ] and also GesmbH in Austria) is a type of legal entity very common in Germany, Austria, Switzerland (where it is equivalent to a S.à r.l.) and Liechtenstein. In the United States, the equivalent type of entity is the limited liability company (LLC). The name of the GmbH form emphasizes the fact that the owners ("Gesellschafter", also known as members) of the entity are not personally liable for the company\'s debts. "GmbH"s are considered legal persons under German and Austrian law. Other variations include mbH (used when the term "Gesellschaft" is part of the company name itself), and gGmbH ("gemeinnützige" GmbH) for non-profit companies.'

    # Test with context.
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

    # Test simple case (no reflection) with context.
    gt_out = (
        "Let\'s think step by step. VIVA Media AG changed its name to VIVA Media GmbH in 2004. "
        "GmbH stands for \"Gesellschaft mit beschränkter Haftung\" which translates to \"company with "
        "limited liability\" in English.Action: Finish[company with limited liability]"
    )
    responses = [
        (
            "Let\'s think step by step. VIVA Media AG changed its name to VIVA Media GmbH in 2004. "
            "GmbH stands for \"Gesellschaft mit beschränkter Haftung\" which translates to \"company with limited liability\" "
            "in English.\nAction: Finish[company with limited liability]"
        )
    ]
    out = _prompt_cot_agent(
        llm=FakeListChatModel(responses=responses), 
        examples=REFLEXION_COT_FEWSHOT_EXAMPLES,
        reflections="",
        question=q,
        scratchpad='\nThought:',
        context=context
    )
    assert out == gt_out

    # Test simple case (no reflection) with no context.
    gt_out = (
        'Thought: Let\'s think step by step. The new acronym for VIVA Media AG after changing its name in '
        '2004 is "Vivendi Visual and Interactive." Action: Finish[Vivendi Visual and Interactive]'
    )
    responses = [
        (
            "Thought: Let's think step by step. The new acronym for VIVA Media AG after changing its name in 2004 "
            "is \"Vivendi Visual and Interactive.\" \nAction: Finish[Vivendi Visual and Interactive]"
        )
    ]
    out = _prompt_cot_agent(
        llm=FakeListChatModel(responses=responses), 
        examples=REFLEXION_COT_FEWSHOT_EXAMPLES_NO_CONTEXT,
        reflections="",
        question=q,
        scratchpad="\nThought:",
        context=None
    )
    assert out == gt_out

    # Test simple case (reflection) with context.


    # Test simple case (reflection) with no context.


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
