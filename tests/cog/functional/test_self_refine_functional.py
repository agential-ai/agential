"""Unit tests for Self-Refine functional methods."""

from langchain_community.chat_models.fake import FakeListChatModel

from agential.cog.functional.self_refine import (
    _build_agent_prompt,
    _build_critique_prompt,
    _build_refine_prompt,
    _prompt_agent,
    _prompt_critique,
    _prompt_refine,
)
from agential.cog.prompts.agent.self_refine import (
    SELF_REFINE_CRITIQUE_INSTRUCTION_GSM8K,
    SELF_REFINE_INSTRUCTION_GSM8K,
    SELF_REFINE_REFINE_INSTRUCTION_GSM8K,
)


def test__build_agent_prompt() -> None:
    """Test _build_agent_prompt."""
    gt_out = "\n(END OF EXAMPLES)\n\nQuestion: \n# Python code, return answer"
    out = _build_agent_prompt(
        question="", examples="", prompt=SELF_REFINE_INSTRUCTION_GSM8K
    )
    assert out == gt_out

    # Test custom prompt.
    out = _build_agent_prompt(question="", examples="", prompt="{question}{examples}")
    assert out == ""


def test__prompt_agent() -> None:
    """Test _prompt_agent."""
    out = _prompt_agent(
        llm=FakeListChatModel(responses=["1"]),
        question="",
        examples="",
        prompt=SELF_REFINE_INSTRUCTION_GSM8K,
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


def test__build_critique_prompt() -> None:
    """Test _build_critique_prompt."""
    gt_out = "\n(END OF EXAMPLES)\n\nQuestion: \n```python\n\n```\n\n# There is an error in the code above because of lack of understanding of the question. What is the error? To find the error, go through semantically complete blocks of the code, and check if everything looks good."
    out = _build_critique_prompt(
        question="",
        examples="",
        answer="",
        prompt=SELF_REFINE_CRITIQUE_INSTRUCTION_GSM8K,
    )
    assert out == gt_out

    # Test custom prompt.
    out = _build_critique_prompt(
        question="", examples="", answer="", prompt="{examples}{answer}"
    )
    assert out == ""


def test__prompt_critique() -> None:
    """Test _prompt_critique."""
    out = _prompt_critique(
        llm=FakeListChatModel(responses=["1"]),
        question="",
        examples="",
        answer="",
        prompt=SELF_REFINE_CRITIQUE_INSTRUCTION_GSM8K,
    )
    assert out == "1"

    # Test custom prompt.
    out = _prompt_critique(
        llm=FakeListChatModel(responses=["1"]),
        question="",
        examples="",
        answer="",
        prompt="{examples}{answer}",
    )
    assert out == "1"


def test__build_refine_prompt() -> None:
    """Test _build_refine_prompt."""
    gt_out = "\n(END OF EXAMPLES)\n\nQuestion: \n```python\n\n```\n\n# There is an error in the code above because of lack of understanding of the question. What is the error? To find the error, go through semantically complete blocks of the code, and check if everything looks good. Provide the improved solution. If there is no error, write out the entire solution again.\n\n\n\nOkay! Here is the rewrite:"
    out = _build_refine_prompt(
        question="",
        examples="",
        answer="",
        critique="",
        prompt=SELF_REFINE_REFINE_INSTRUCTION_GSM8K,
    )
    assert out == gt_out

    # Test custom prompt.
    out = _build_refine_prompt(
        question="",
        examples="",
        answer="",
        critique="",
        prompt="{examples}{answer}{critique}",
    )
    assert out == ""


def test__prompt_refine() -> None:
    """Test _prompt_refine."""
    out = _prompt_refine(
        llm=FakeListChatModel(responses=["1"]),
        question="",
        examples="",
        answer="",
        critique="",
        prompt="",
    )
    assert out == "1"

    # Test custom prompt.
    out = _prompt_refine(
        llm=FakeListChatModel(responses=["1"]),
        question="",
        examples="",
        answer="",
        critique="",
        prompt="{examples}{answer}{critique}",
    )
    assert out == "1"
