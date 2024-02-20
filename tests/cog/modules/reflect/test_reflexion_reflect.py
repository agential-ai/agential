"""Unit tests for Reflexion reflect module."""
import pytest

from langchain_community.chat_models.fake import FakeListChatModel
from langchain_core.language_models.chat_models import BaseChatModel
from discussion_agents.cog.functional.reflexion import (
    _format_reflections,
)
from discussion_agents.cog.modules.reflect.reflexion import (
    ReflexionCoTReflector,
    ReflexionReActReflector,
)


def test_reflexion_cot_init() -> None:
    """Unit test for ReflexionCoT Reflector initialization."""
    reflector = ReflexionCoTReflector(
        llm=FakeListChatModel(responses=["1"]),
    )
    assert isinstance(reflector.llm, BaseChatModel)
    assert not reflector.reflections
    assert not reflector.reflections_str
    assert reflector.max_reflections == 3


def test_reflexion_cot_reflector() -> None:
    """Unit tests for ReflexionCoT Reflector."""
    reflector = ReflexionCoTReflector(
        llm=FakeListChatModel(responses=["1"]),
    )

    # Test with invalid input.
    with pytest.raises(NotImplementedError):
        out = reflector.reflect(
            strategy="invalid input",
            examples="",
            question="",
            scratchpad="",
            context="",
        )

    # Test with last attempt.
    out = reflector.reflect(
        strategy="last_attempt", examples="", question="", scratchpad="", context=""
    )
    assert isinstance(out, tuple)
    assert len(out) == 2
    assert isinstance(out[0], list)
    assert isinstance(out[1], str)
    assert out[0] == [""]
    assert (
        out[1]
        == "You have attempted to answer the following question before and failed. Below is the last trial you attempted to answer the question.\nQuestion: \n\n(END PREVIOUS TRIAL)\n"
    )

    # Test with Reflexion.
    reflector = ReflexionCoTReflector(
        llm=FakeListChatModel(responses=["1"]),
    )
    out = reflector.reflect(
        strategy="reflexion", examples="", question="", scratchpad="", context=""
    )

    assert isinstance(out, tuple)
    assert len(out) == 2
    assert isinstance(out[0], list)
    assert isinstance(out[1], str)
    assert out[0] == ["1"]
    assert (
        out[1]
        == "You have attempted to answer following question before and failed. The following reflection(s) give a plan to avoid failing to answer the question in the same way you did previously. Use them to improve your strategy of correctly answering the given question.\nReflections:\n- 1"
    )

    # Test with last attempt and Reflexion.
    reflector = ReflexionCoTReflector(
        llm=FakeListChatModel(responses=["1"]),
    )
    out = reflector.reflect(
        strategy="last_attempt_and_reflexion",
        examples="",
        question="",
        scratchpad="",
        context="",
    )
    assert isinstance(out, tuple)
    assert len(out) == 2
    assert isinstance(out[0], list)
    assert isinstance(out[1], str)
    assert out[0] == ["1"]
    assert (
        out[1]
        == "You have attempted to answer the following question before and failed. Below is the last trial you attempted to answer the question.\nQuestion: \n\n(END PREVIOUS TRIAL)\n\nThe following reflection(s) give a plan to avoid failing to answer the question in the same way you did previously. Use them to improve your strategy of correctly answering the given question.\nReflections:\n- 1"
    )

    # Test len(self.reflections) > max_reflections.
    reflections = [
        'The failure in this reasoning trial was due to not being able to find specific information on VIVA Media AG and its name change in 2004. To mitigate this failure, the agent should consider broadening the search terms to include related keywords such as "corporate rebranding" or "corporate name change" in addition to the specific company name. This will help in obtaining more relevant and specific results that may provide the necessary information to answer the question accurately.'
    ] * 3
    reflections_str = _format_reflections(reflections)
    reflector = ReflexionCoTReflector(
        llm=FakeListChatModel(responses=["1"]),
        max_reflections=2
    )
    reflector.reflections = reflections
    reflector.reflections_str = reflections_str
    _ = reflector.reflect(
        strategy="reflexion",  # Only applicable to reflexion strategy.
        examples="",
        question="",
        scratchpad="",
        context="",
    )
    assert len(reflector.reflections) == 2
    assert reflector.reflections_str == _format_reflections(reflections[-2:])


def test_reflexion_cot_clear() -> None:
    """Unit tests for ReflexionCoT Reflector clear method."""
    reflector = ReflexionCoTReflector(
        llm=FakeListChatModel(responses=["1"]),
    )
    reflector.reflections = ["c", "a", "t"]
    reflector.reflections_str = "cat"
    reflector.clear()
    assert reflector.reflections == []
    assert reflector.reflections_str == ""


def test_reflexion_react_reflector() -> None:
    """Unit tests for ReflexionReAct Reflector."""
    reflector = ReflexionReActReflector(
        llm=FakeListChatModel(responses=["1"]),
    )

    # Test with invalid input.
    with pytest.raises(NotImplementedError):
        out = reflector.reflect(
            strategy="invalid input",
            examples="",
            question="",
            scratchpad="",
        )

    # Test with last attempt.
    out = reflector.reflect(
        strategy="last_attempt", examples="", question="", scratchpad=""
    )
    assert isinstance(out, tuple)
    assert len(out) == 2
    assert isinstance(out[0], list)
    assert isinstance(out[1], str)
    assert out[0] == [""]
    assert (
        out[1]
        == "You have attempted to answer the following question before and failed. Below is the last trial you attempted to answer the question.\nQuestion: \n\n(END PREVIOUS TRIAL)\n"
    )

    # Test with Reflexion.
    reflector = ReflexionReActReflector(
        llm=FakeListChatModel(responses=["1"]),
    )
    out = reflector.reflect(
        strategy="reflexion", examples="", question="", scratchpad=""
    )

    assert isinstance(out, tuple)
    assert len(out) == 2
    assert isinstance(out[0], list)
    assert isinstance(out[1], str)
    assert out[0] == ["1"]
    assert (
        out[1]
        == "You have attempted to answer following question before and failed. The following reflection(s) give a plan to avoid failing to answer the question in the same way you did previously. Use them to improve your strategy of correctly answering the given question.\nReflections:\n- 1"
    )

    # Test with last attempt and Reflexion.
    reflector = ReflexionReActReflector(
        llm=FakeListChatModel(responses=["1"]),
    )
    out = reflector.reflect(
        strategy="last_attempt_and_reflexion",
        examples="",
        question="",
        scratchpad="",
    )
    assert isinstance(out, tuple)
    assert len(out) == 2
    assert isinstance(out[0], list)
    assert isinstance(out[1], str)
    assert out[0] == ["1"]
    assert (
        out[1]
        == "You have attempted to answer the following question before and failed. Below is the last trial you attempted to answer the question.\nQuestion: \n\n(END PREVIOUS TRIAL)\n\nThe following reflection(s) give a plan to avoid failing to answer the question in the same way you did previously. Use them to improve your strategy of correctly answering the given question.\nReflections:\n- 1"
    )


def test_reflexion_react_clear() -> None:
    """Unit tests for ReflexionReAct Reflector clear method."""
    reflector = ReflexionReActReflector(
        llm=FakeListChatModel(responses=["1"]),
    )
    reflector.reflections = ["c", "a", "t"]
    reflector.reflections_str = "cat"
    reflector.clear()
    assert reflector.reflections == []
    assert reflector.reflections_str == ""
