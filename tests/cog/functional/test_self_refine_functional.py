"""Unit tests for Self-Refine functional methods."""

from langchain_community.chat_models.fake import FakeListChatModel

from agential.cog.functional.self_refine import (
    _build_agent_prompt,
    _build_feedback_prompt,
    _build_refine_prompt,
    _prompt_agent,
    _prompt_feedback,
    _prompt_refine,
)


def test__build_agent_prompt() -> None:
    """Test _build_agent_prompt."""
    gt_out = "\n\n\n# Q: \n# solution using Python:"
    out = _build_agent_prompt(
        question="",
        examples="",
    )
    assert out == gt_out

    # Test custom prompt.
    out = _build_agent_prompt(question="", examples="", prompt="{question}{examples}")
    assert out == ""


def test__prompt_agent() -> None:
    """Test _prompt_agent."""
    out = _prompt_agent(
        llm=FakeListChatModel(responses=["1"]), question="", examples=""
    )
    assert out == "1"

    # Test custom prompt.
    out = _prompt_agent(
        llm=FakeListChatModel(responses=["1"]),
        question="",
        examples="",
        prompt="{question}{examples}",
    )
    assert out == "1"


def test__build_feedback_prompt() -> None:
    """Test _build_feedback_prompt."""
    gt_out = "\n\n\n\n# There is an error in the code above because of lack of understanding of the question. What is the error? To find the error, go through semantically complete blocks of the code, and check if everything looks good. If there is no error, simply output 'It is correct.'"
    out = _build_feedback_prompt(
        examples="",
        solution="",
    )
    assert out == gt_out

    # Test custom prompt.
    out = _build_feedback_prompt(
        examples="", solution="", prompt="{examples}{solution}"
    )
    assert out == ""


def test__prompt_feedback() -> None:
    """Test _prompt_feedback."""
    out = _prompt_feedback(
        llm=FakeListChatModel(responses=["1"]), examples="", solution=""
    )
    assert out == "1"

    # Test custom prompt.
    out = _prompt_feedback(
        llm=FakeListChatModel(responses=["1"]),
        examples="",
        solution="",
        prompt="{examples}{solution}",
    )
    assert out == "1"


def test__build_refine_prompt() -> None:
    """Test _build_refine_prompt."""
    gt_out = "\n\n\n\n\n\n# There is an error in the code above because of lack of understanding of the question. What is the error? To find the error, go through semantically complete blocks of the code, and check if everything looks good. Provide the improved solution. If there is no error, write out the entire solution again."
    out = _build_refine_prompt(examples="", solution="", feedback="")
    assert out == gt_out

    # Test custom prompt.
    out = _build_refine_prompt(
        examples="", solution="", feedback="", prompt="{examples}{solution}{feedback}"
    )
    assert out == ""


def test__prompt_refine() -> None:
    """Test _prompt_refine."""
    out = _prompt_refine(
        llm=FakeListChatModel(responses=["1"]), examples="", solution="", feedback=""
    )
    assert out == "1"

    # Test custom prompt.
    out = _prompt_refine(
        llm=FakeListChatModel(responses=["1"]),
        examples="",
        solution="",
        feedback="",
        prompt="{examples}{solution}{feedback}",
    )
    assert out == "1"
