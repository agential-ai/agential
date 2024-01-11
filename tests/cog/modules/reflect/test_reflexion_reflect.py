"""Unit tests for Reflexion reflect module."""
import pytest
from langchain_community.chat_models.fake import FakeListChatModel

from discussion_agents.cog.modules.reflect.reflexion import ReflexionReflector

def test_reflexion_reflector() -> None:
    """Unit tests for Reflexion Reflector."""
    reflector = ReflexionReflector(
        llm=FakeListChatModel(responses=["1"]),
    )

    # Test with invalid input.
    with pytest.raises(NotImplementedError):
        out = reflector.reflect(
            strategy="invalid input",
            examples="",
            context="",
            question="",
            scratchpad=""
        )

    # Test with last attempt.
    out = reflector.reflect(
        strategy="last_attempt",
        examples="",
        context="",
        question="",
        scratchpad=""
    )
    assert isinstance(out, tuple)
    assert len(out) == 2
    assert isinstance(out[0], list)
    assert isinstance(out[1], str)
    assert out[0] == [""]
    assert out[1] ==  'You have attempted to answer the following question before and failed. Below is the last trial you attempted to answer the question.\nQuestion: \n\n(END PREVIOUS TRIAL)\n'

    # Test with Reflexion.
    reflector = ReflexionReflector(
        llm=FakeListChatModel(responses=["1"]),
    )
    out = reflector.reflect(
        strategy="reflexion",
        examples="",
        context="",
        question="",
        scratchpad=""
    )

    assert isinstance(out, tuple)
    assert len(out) == 2
    assert isinstance(out[0], list)
    assert isinstance(out[1], str)
    assert out[0] == ["1"]
    assert out[1] == 'You have attempted to answer following question before and failed. The following reflection(s) give a plan to avoid failing to answer the question in the same way you did previously. Use them to improve your strategy of correctly answering the given question.\nReflections:\n- 1'

    # Test with last attempt and Reflexion.
    reflector = ReflexionReflector(
        llm=FakeListChatModel(responses=["1"]),
    )
    out = reflector.reflect(
        strategy="last_attempt_and_reflexion",
        examples="",
        context="",
        question="",
        scratchpad=""
    )
    assert isinstance(out, tuple)
    assert len(out) == 2
    assert isinstance(out[0], list)
    assert isinstance(out[1], str)
    assert out[0] == ["1"]
    assert out[1] ==  'You have attempted to answer the following question before and failed. Below is the last trial you attempted to answer the question.\nQuestion: \n\n(END PREVIOUS TRIAL)\n\nThe following reflection(s) give a plan to avoid failing to answer the question in the same way you did previously. Use them to improve your strategy of correctly answering the given question.\nReflections:\n- 1'