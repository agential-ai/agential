"""Unit tests for Reflexion reflect module."""

import pytest

from agential.agents.reflexion.functional import (
    _format_reflections,
)
from agential.agents.reflexion.prompts import (
    REFLEXION_REACT_REFLECT_INSTRUCTION_HOTPOTQA,
)
from agential.agents.reflexion.reflect import (
    ReflexionCoTReflector,
    ReflexionReActReflector,
)
from agential.llm.llm import BaseLLM, MockLLM, Response


def test_reflexion_cot_init() -> None:
    """Unit test for ReflexionCoT Reflector initialization."""
    reflector = ReflexionCoTReflector(
        llm=MockLLM("gpt-3.5-turbo", responses=["1"]),
    )
    assert isinstance(reflector.llm, BaseLLM)
    assert not reflector.reflections
    assert not reflector.reflections_str
    assert reflector.max_reflections == 3


def test_reflexion_cot_reflector() -> None:
    """Unit tests for ReflexionCoT Reflector."""
    reflector = ReflexionCoTReflector(
        llm=MockLLM("gpt-3.5-turbo", responses=["1"]),
    )

    # Test with invalid input.
    with pytest.raises(NotImplementedError):
        out = reflector.reflect(
            reflect_strategy="invalid input",
            question="",
            examples="",
            scratchpad="",
            prompt="",
        )

    # Test with last attempt.
    out = reflector.reflect(
        reflect_strategy="last_attempt",
        question="",
        examples="",
        scratchpad="",
        prompt="",
    )
    assert isinstance(out, tuple)
    assert len(out) == 3
    assert isinstance(out[0], list)
    assert isinstance(out[1], str)
    assert out[0] == [""]
    assert (
        out[1]
        == "You have attempted to answer the following question before and failed. Below is the last trial you attempted to answer the question.\nQuestion: \n\n(END PREVIOUS TRIAL)\n"
    )
    assert out[2] == None

    # Test with Reflexion.
    reflector = ReflexionCoTReflector(
        llm=MockLLM("gpt-3.5-turbo", responses=["1"]),
    )
    out = reflector.reflect(
        reflect_strategy="reflexion",
        question="",
        examples="",
        scratchpad="",
        prompt="",
    )

    assert isinstance(out, tuple)
    assert len(out) == 3
    assert isinstance(out[0], list)
    assert isinstance(out[1], str)
    assert out[0] == ["1"]
    assert (
        out[1]
        == "You have attempted to answer following question before and failed. The following reflection(s) give a plan to avoid failing to answer the question in the same way you did previously. Use them to improve your strategy of correctly answering the given question.\nReflections:\n- 1"
    )
    assert out[2] == Response(
        input_text="",
        output_text="1",
        prompt_tokens=10,
        completion_tokens=20,
        total_tokens=30,
        prompt_cost=1.5e-05,
        completion_cost=3.9999999999999996e-05,
        total_cost=5.4999999999999995e-05,
        prompt_time=0.5,
    )

    # Test with last attempt and Reflexion.
    reflector = ReflexionCoTReflector(
        llm=MockLLM("gpt-3.5-turbo", responses=["1"]),
    )
    out = reflector.reflect(
        reflect_strategy="last_attempt_and_reflexion",
        question="",
        examples="",
        scratchpad="",
        prompt="",
    )
    assert isinstance(out, tuple)
    assert len(out) == 3
    assert isinstance(out[0], list)
    assert isinstance(out[1], str)
    assert out[0] == ["1"]
    assert (
        out[1]
        == "You have attempted to answer the following question before and failed. Below is the last trial you attempted to answer the question.\nQuestion: \n\n(END PREVIOUS TRIAL)\n\nThe following reflection(s) give a plan to avoid failing to answer the question in the same way you did previously. Use them to improve your strategy of correctly answering the given question.\nReflections:\n- 1"
    )
    assert out[2] == Response(
        input_text="",
        output_text="1",
        prompt_tokens=10,
        completion_tokens=20,
        total_tokens=30,
        prompt_cost=1.5e-05,
        completion_cost=3.9999999999999996e-05,
        total_cost=5.4999999999999995e-05,
        prompt_time=0.5,
    )

    # Test len(self.reflections) > max_reflections.
    reflections = [
        'The failure in this reasoning trial was due to not being able to find specific information on VIVA Media AG and its name change in 2004. To mitigate this failure, the agent should consider broadening the search terms to include related keywords such as "corporate rebranding" or "corporate name change" in addition to the specific company name. This will help in obtaining more relevant and specific results that may provide the necessary information to answer the question accurately.'
    ] * 3
    reflections_str = _format_reflections(reflections)
    reflector = ReflexionCoTReflector(
        llm=MockLLM("gpt-3.5-turbo", responses=["1"]), max_reflections=2
    )
    reflector.reflections = reflections
    reflector.reflections_str = reflections_str
    _ = reflector.reflect(
        reflect_strategy="reflexion",  # Only applicable to reflexion strategy.
        question="",
        examples="",
        scratchpad="",
        prompt="",
    )
    assert len(reflector.reflections) == 2
    assert reflector.reflections_str == _format_reflections(reflections[-2:])


def test_reflexion_cot_reflect_strat() -> None:
    """Unit tests for ReflexionCoT Reflector reflect method with different strategies."""
    examples = "Example content"
    question = "What is the capital of France?"
    scratchpad = "Initial scratchpad content"

    # Test last attempt followed by reflexion.
    gt_out_reflections = ["Initial scratchpad content"]
    gt_out_reflections_str = "You have attempted to answer the following question before and failed. Below is the last trial you attempted to answer the question.\nQuestion: What is the capital of France?\nInitial scratchpad content\n(END PREVIOUS TRIAL)\n"
    reflector = ReflexionCoTReflector(
        llm=MockLLM("gpt-3.5-turbo", responses=["1"]),
    )
    reflections, reflections_str, reflections_out = reflector.reflect(
        reflect_strategy="last_attempt",
        question=question,
        examples=examples,
        scratchpad=scratchpad,
        prompt="",
    )
    assert reflections == gt_out_reflections
    assert reflections_str == gt_out_reflections_str
    gt_out_reflections_str = "You have attempted to answer following question before and failed. The following reflection(s) give a plan to avoid failing to answer the question in the same way you did previously. Use them to improve your strategy of correctly answering the given question.\nReflections:\n- Initial scratchpad content\n- 1"
    gt_out_reflections = ["Initial scratchpad content", "1"]
    reflections, reflections_str, reflections_out = reflector.reflect(
        reflect_strategy="reflexion",
        question=question,
        examples=examples,
        scratchpad=scratchpad,
        prompt="",
    )
    assert reflections == gt_out_reflections
    assert reflections_str == gt_out_reflections_str

    # Test reflexion followed by last attempt.
    gt_out_reflections = ["1"]
    gt_out_reflections_str = "You have attempted to answer following question before and failed. The following reflection(s) give a plan to avoid failing to answer the question in the same way you did previously. Use them to improve your strategy of correctly answering the given question.\nReflections:\n- 1"
    reflector = ReflexionCoTReflector(
        llm=MockLLM("gpt-3.5-turbo", responses=["1"]),
    )
    reflections, reflections_str, reflections_out = reflector.reflect(
        reflect_strategy="reflexion",
        question=question,
        examples=examples,
        scratchpad=scratchpad,
        prompt="",
    )
    assert reflections == gt_out_reflections
    assert reflections_str == gt_out_reflections_str
    gt_out_reflections = ["Initial scratchpad content"]
    gt_out_reflections_str = "You have attempted to answer the following question before and failed. Below is the last trial you attempted to answer the question.\nQuestion: What is the capital of France?\nInitial scratchpad content\n(END PREVIOUS TRIAL)\n"
    reflections, reflections_str, reflections_out = reflector.reflect(
        reflect_strategy="last_attempt",
        question=question,
        examples=examples,
        scratchpad=scratchpad,
        prompt="",
    )
    assert reflections == gt_out_reflections
    assert reflections_str == gt_out_reflections_str

    # Test last attempt followed by last attempt and reflexion.
    gt_out_reflections = ["Initial scratchpad content"]
    gt_out_reflections_str = "You have attempted to answer the following question before and failed. Below is the last trial you attempted to answer the question.\nQuestion: What is the capital of France?\nInitial scratchpad content\n(END PREVIOUS TRIAL)\n"
    reflector = ReflexionCoTReflector(
        llm=MockLLM("gpt-3.5-turbo", responses=["1"]),
    )
    reflections, reflections_str, reflections_out = reflector.reflect(
        reflect_strategy="last_attempt",
        question=question,
        examples=examples,
        scratchpad=scratchpad,
        prompt="",
    )
    assert reflections == gt_out_reflections
    assert reflections_str == gt_out_reflections_str
    gt_out_reflections = ["1"]
    gt_out_reflections_str = "You have attempted to answer the following question before and failed. Below is the last trial you attempted to answer the question.\nQuestion: What is the capital of France?\nInitial scratchpad content\n(END PREVIOUS TRIAL)\n\nThe following reflection(s) give a plan to avoid failing to answer the question in the same way you did previously. Use them to improve your strategy of correctly answering the given question.\nReflections:\n- 1"
    reflections, reflections_str, reflections_out = reflector.reflect(
        reflect_strategy="last_attempt_and_reflexion",
        question=question,
        examples=examples,
        scratchpad=scratchpad,
        prompt="",
    )
    assert reflections_out
    assert reflections == gt_out_reflections
    assert reflections_str == gt_out_reflections_str

    # Test last attempt and reflexion followed by last attempt.
    gt_out_reflections = ["1"]
    gt_out_reflections_str = "You have attempted to answer the following question before and failed. Below is the last trial you attempted to answer the question.\nQuestion: What is the capital of France?\nInitial scratchpad content\n(END PREVIOUS TRIAL)\n\nThe following reflection(s) give a plan to avoid failing to answer the question in the same way you did previously. Use them to improve your strategy of correctly answering the given question.\nReflections:\n- 1"
    reflector = ReflexionCoTReflector(
        llm=MockLLM("gpt-3.5-turbo", responses=["1"]),
    )
    reflections, reflections_str, reflections_out = reflector.reflect(
        reflect_strategy="last_attempt_and_reflexion",
        question=question,
        examples=examples,
        scratchpad=scratchpad,
        prompt="",
    )
    assert reflections_out
    assert reflections == gt_out_reflections
    assert reflections_str == gt_out_reflections_str
    gt_out_reflections = ["Initial scratchpad content"]
    gt_out_reflections_str = "You have attempted to answer the following question before and failed. Below is the last trial you attempted to answer the question.\nQuestion: What is the capital of France?\nInitial scratchpad content\n(END PREVIOUS TRIAL)\n"
    reflections, reflections_str, reflections_out = reflector.reflect(
        reflect_strategy="last_attempt",
        question=question,
        examples=examples,
        scratchpad=scratchpad,
        prompt="",
    )
    assert not reflections_out
    assert reflections == gt_out_reflections
    assert reflections_str == gt_out_reflections_str

    # Test reflexion followed by last attempt and reflexion.
    gt_out_reflections = ["1"]
    gt_out_reflections_str = "You have attempted to answer following question before and failed. The following reflection(s) give a plan to avoid failing to answer the question in the same way you did previously. Use them to improve your strategy of correctly answering the given question.\nReflections:\n- 1"
    reflector = ReflexionCoTReflector(
        llm=MockLLM("gpt-3.5-turbo", responses=["1"]),
    )
    reflections, reflections_str, reflections_out = reflector.reflect(
        reflect_strategy="reflexion",
        question=question,
        examples=examples,
        scratchpad=scratchpad,
        prompt="",
    )
    assert reflections_out
    assert reflections == gt_out_reflections
    assert reflections_str == gt_out_reflections_str
    gt_out_reflections = ["1"]
    gt_out_reflections_str = "You have attempted to answer the following question before and failed. Below is the last trial you attempted to answer the question.\nQuestion: What is the capital of France?\nInitial scratchpad content\n(END PREVIOUS TRIAL)\n\nThe following reflection(s) give a plan to avoid failing to answer the question in the same way you did previously. Use them to improve your strategy of correctly answering the given question.\nReflections:\n- 1"
    reflections, reflections_str, reflections_out = reflector.reflect(
        reflect_strategy="last_attempt_and_reflexion",
        question=question,
        examples=examples,
        scratchpad=scratchpad,
        prompt="",
    )
    assert reflections_out
    assert reflections == gt_out_reflections
    assert reflections_str == gt_out_reflections_str

    # Test last attempt and reflexion followed by reflexion.
    gt_out_reflections = ["1"]
    gt_out_reflections_str = "You have attempted to answer the following question before and failed. Below is the last trial you attempted to answer the question.\nQuestion: What is the capital of France?\nInitial scratchpad content\n(END PREVIOUS TRIAL)\n\nThe following reflection(s) give a plan to avoid failing to answer the question in the same way you did previously. Use them to improve your strategy of correctly answering the given question.\nReflections:\n- 1"
    reflector = ReflexionCoTReflector(
        llm=MockLLM("gpt-3.5-turbo", responses=["1"]),
    )
    reflections, reflections_str, reflections_out = reflector.reflect(
        reflect_strategy="last_attempt_and_reflexion",
        question=question,
        examples=examples,
        scratchpad=scratchpad,
        prompt="",
    )
    assert reflections_out
    assert reflections == gt_out_reflections
    assert reflections_str == gt_out_reflections_str
    gt_out_reflections = ["1", "1"]
    gt_out_reflections_str = "You have attempted to answer following question before and failed. The following reflection(s) give a plan to avoid failing to answer the question in the same way you did previously. Use them to improve your strategy of correctly answering the given question.\nReflections:\n- 1\n- 1"
    reflections, reflections_str, reflections_out = reflector.reflect(
        reflect_strategy="reflexion",
        question=question,
        examples=examples,
        scratchpad=scratchpad,
        prompt="",
    )
    assert reflections_out
    assert reflections == gt_out_reflections
    assert reflections_str == gt_out_reflections_str


def test_reflexion_cot_reset() -> None:
    """Unit tests for ReflexionCoT Reflector reset method."""
    reflector = ReflexionCoTReflector(
        llm=MockLLM("gpt-3.5-turbo", responses=["1"]),
    )
    reflector.reflections = ["c", "a", "t"]
    reflector.reflections_str = "cat"
    reflector.reset()
    assert reflector.reflections == []
    assert reflector.reflections_str == ""


def test_reflexion_react_reflector() -> None:
    """Unit tests for ReflexionReAct Reflector."""
    reflector = ReflexionReActReflector(
        llm=MockLLM("gpt-3.5-turbo", responses=["1"]),
    )

    # Test with invalid input.
    with pytest.raises(NotImplementedError):
        out = reflector.reflect(
            reflect_strategy="invalid input",
            examples="",
            question="",
            scratchpad="",
            prompt=REFLEXION_REACT_REFLECT_INSTRUCTION_HOTPOTQA,
        )

    # Test with last attempt.
    out = reflector.reflect(
        reflect_strategy="last_attempt",
        examples="",
        question="",
        scratchpad="",
        prompt=REFLEXION_REACT_REFLECT_INSTRUCTION_HOTPOTQA,
    )
    assert isinstance(out, tuple)
    assert len(out) == 3
    assert isinstance(out[0], list)
    assert isinstance(out[1], str)
    assert out[0] == [""]
    assert (
        out[1]
        == "You have attempted to answer the following question before and failed. Below is the last trial you attempted to answer the question.\nQuestion: \n\n(END PREVIOUS TRIAL)\n"
    )
    assert out[2] == None

    # Test with Reflexion.
    reflector = ReflexionReActReflector(
        llm=MockLLM("gpt-3.5-turbo", responses=["1"]),
    )
    out = reflector.reflect(
        reflect_strategy="reflexion",
        examples="",
        question="",
        scratchpad="",
        prompt=REFLEXION_REACT_REFLECT_INSTRUCTION_HOTPOTQA,
    )

    assert isinstance(out, tuple)
    assert len(out) == 3
    assert isinstance(out[0], list)
    assert isinstance(out[1], str)
    assert out[0] == ["1"]
    assert (
        out[1]
        == "You have attempted to answer following question before and failed. The following reflection(s) give a plan to avoid failing to answer the question in the same way you did previously. Use them to improve your strategy of correctly answering the given question.\nReflections:\n- 1"
    )
    assert out[2] == Response(
        input_text="",
        output_text="1",
        prompt_tokens=10,
        completion_tokens=20,
        total_tokens=30,
        prompt_cost=1.5e-05,
        completion_cost=3.9999999999999996e-05,
        total_cost=5.4999999999999995e-05,
        prompt_time=0.5,
    )

    # Test with last attempt and Reflexion.
    reflector = ReflexionReActReflector(
        llm=MockLLM("gpt-3.5-turbo", responses=["1"]),
    )
    out = reflector.reflect(
        reflect_strategy="last_attempt_and_reflexion",
        examples="",
        question="",
        scratchpad="",
        prompt=REFLEXION_REACT_REFLECT_INSTRUCTION_HOTPOTQA,
    )
    assert isinstance(out, tuple)
    assert len(out) == 3
    assert isinstance(out[0], list)
    assert isinstance(out[1], str)
    assert out[0] == ["1"]
    assert (
        out[1]
        == "You have attempted to answer the following question before and failed. Below is the last trial you attempted to answer the question.\nQuestion: \n\n(END PREVIOUS TRIAL)\n\nThe following reflection(s) give a plan to avoid failing to answer the question in the same way you did previously. Use them to improve your strategy of correctly answering the given question.\nReflections:\n- 1"
    )
    assert out[2] == Response(
        input_text="",
        output_text="1",
        prompt_tokens=10,
        completion_tokens=20,
        total_tokens=30,
        prompt_cost=1.5e-05,
        completion_cost=3.9999999999999996e-05,
        total_cost=5.4999999999999995e-05,
        prompt_time=0.5,
    )

    # Test len(self.reflections) > max_reflections.
    reflections = [
        'The failure in this reasoning trial was due to not being able to find specific information on VIVA Media AG and its name change in 2004. To mitigate this failure, the agent should consider broadening the search terms to include related keywords such as "corporate rebranding" or "corporate name change" in addition to the specific company name. This will help in obtaining more relevant and specific results that may provide the necessary information to answer the question accurately.'
    ] * 3
    reflections_str = _format_reflections(reflections)
    reflector = ReflexionReActReflector(
        llm=MockLLM("gpt-3.5-turbo", responses=["1"]), max_reflections=2
    )
    reflector.reflections = reflections
    reflector.reflections_str = reflections_str
    _ = reflector.reflect(
        reflect_strategy="reflexion",  # Only applicable to reflexion strategy.
        examples="",
        question="",
        scratchpad="",
        prompt=REFLEXION_REACT_REFLECT_INSTRUCTION_HOTPOTQA,
    )
    assert len(reflector.reflections) == 2
    assert reflector.reflections_str == _format_reflections(reflections[-2:])


def test_reflexion_react_reflect_strat() -> None:
    """Unit tests for ReflexionReAct Reflector reflect method with different strategies."""
    examples = "Example content"
    question = "What is the capital of France?"
    scratchpad = "Initial scratchpad content"

    # Test last attempt followed by reflexion.
    gt_out_reflections = ["Initial scratchpad content"]
    gt_out_reflections_str = "You have attempted to answer the following question before and failed. Below is the last trial you attempted to answer the question.\nQuestion: What is the capital of France?\nInitial scratchpad content\n(END PREVIOUS TRIAL)\n"
    reflector = ReflexionReActReflector(
        llm=MockLLM("gpt-3.5-turbo", responses=["1"]),
    )
    reflections, reflections_str, reflections_out = reflector.reflect(
        reflect_strategy="last_attempt",
        examples=examples,
        question=question,
        scratchpad=scratchpad,
        prompt=REFLEXION_REACT_REFLECT_INSTRUCTION_HOTPOTQA,
    )
    assert not reflections_out
    assert reflections == gt_out_reflections
    assert reflections_str == gt_out_reflections_str
    gt_out_reflections_str = "You have attempted to answer following question before and failed. The following reflection(s) give a plan to avoid failing to answer the question in the same way you did previously. Use them to improve your strategy of correctly answering the given question.\nReflections:\n- Initial scratchpad content\n- 1"
    gt_out_reflections = ["Initial scratchpad content", "1"]
    reflections, reflections_str, reflections_out = reflector.reflect(
        reflect_strategy="reflexion",
        examples=examples,
        question=question,
        scratchpad=scratchpad,
        prompt=REFLEXION_REACT_REFLECT_INSTRUCTION_HOTPOTQA,
    )
    assert reflections_out
    assert reflections == gt_out_reflections
    assert reflections_str == gt_out_reflections_str

    # Test reflexion followed by last attempt.
    gt_out_reflections = ["1"]
    gt_out_reflections_str = "You have attempted to answer following question before and failed. The following reflection(s) give a plan to avoid failing to answer the question in the same way you did previously. Use them to improve your strategy of correctly answering the given question.\nReflections:\n- 1"
    reflector = ReflexionReActReflector(
        llm=MockLLM("gpt-3.5-turbo", responses=["1"]),
    )
    reflections, reflections_str, reflections_out = reflector.reflect(
        reflect_strategy="reflexion",
        examples=examples,
        question=question,
        scratchpad=scratchpad,
        prompt=REFLEXION_REACT_REFLECT_INSTRUCTION_HOTPOTQA,
    )
    assert reflections_out
    assert reflections == gt_out_reflections
    assert reflections_str == gt_out_reflections_str

    gt_out_reflections = ["Initial scratchpad content"]
    gt_out_reflections_str = "You have attempted to answer the following question before and failed. Below is the last trial you attempted to answer the question.\nQuestion: What is the capital of France?\nInitial scratchpad content\n(END PREVIOUS TRIAL)\n"
    reflections, reflections_str, reflections_out = reflector.reflect(
        reflect_strategy="last_attempt",
        examples=examples,
        question=question,
        scratchpad=scratchpad,
        prompt=REFLEXION_REACT_REFLECT_INSTRUCTION_HOTPOTQA,
    )
    assert reflections == gt_out_reflections
    assert reflections_str == gt_out_reflections_str

    # Test last attempt followed by last attempt and reflexion.
    gt_out_reflections = ["Initial scratchpad content"]
    gt_out_reflections_str = "You have attempted to answer the following question before and failed. Below is the last trial you attempted to answer the question.\nQuestion: What is the capital of France?\nInitial scratchpad content\n(END PREVIOUS TRIAL)\n"
    reflector = ReflexionReActReflector(
        llm=MockLLM("gpt-3.5-turbo", responses=["1"]),
    )
    reflections, reflections_str, reflections_out = reflector.reflect(
        reflect_strategy="last_attempt",
        examples=examples,
        question=question,
        scratchpad=scratchpad,
        prompt=REFLEXION_REACT_REFLECT_INSTRUCTION_HOTPOTQA,
    )
    assert not reflections_out
    assert reflections == gt_out_reflections
    assert reflections_str == gt_out_reflections_str
    gt_out_reflections = ["1"]
    gt_out_reflections_str = "You have attempted to answer the following question before and failed. Below is the last trial you attempted to answer the question.\nQuestion: What is the capital of France?\nInitial scratchpad content\n(END PREVIOUS TRIAL)\n\nThe following reflection(s) give a plan to avoid failing to answer the question in the same way you did previously. Use them to improve your strategy of correctly answering the given question.\nReflections:\n- 1"
    reflections, reflections_str, reflections_out = reflector.reflect(
        reflect_strategy="last_attempt_and_reflexion",
        examples=examples,
        question=question,
        scratchpad=scratchpad,
        prompt=REFLEXION_REACT_REFLECT_INSTRUCTION_HOTPOTQA,
    )
    assert reflections == gt_out_reflections
    assert reflections_str == gt_out_reflections_str

    # Test last attempt and reflexion followed by last attempt.
    gt_out_reflections = ["1"]
    gt_out_reflections_str = "You have attempted to answer the following question before and failed. Below is the last trial you attempted to answer the question.\nQuestion: What is the capital of France?\nInitial scratchpad content\n(END PREVIOUS TRIAL)\n\nThe following reflection(s) give a plan to avoid failing to answer the question in the same way you did previously. Use them to improve your strategy of correctly answering the given question.\nReflections:\n- 1"
    reflector = ReflexionReActReflector(
        llm=MockLLM("gpt-3.5-turbo", responses=["1"]),
    )
    reflections, reflections_str, reflections_out = reflector.reflect(
        reflect_strategy="last_attempt_and_reflexion",
        examples=examples,
        question=question,
        scratchpad=scratchpad,
        prompt=REFLEXION_REACT_REFLECT_INSTRUCTION_HOTPOTQA,
    )
    assert reflections_out
    assert reflections == gt_out_reflections
    assert reflections_str == gt_out_reflections_str
    gt_out_reflections = ["Initial scratchpad content"]
    gt_out_reflections_str = "You have attempted to answer the following question before and failed. Below is the last trial you attempted to answer the question.\nQuestion: What is the capital of France?\nInitial scratchpad content\n(END PREVIOUS TRIAL)\n"
    reflections, reflections_str, reflections_out = reflector.reflect(
        reflect_strategy="last_attempt",
        examples=examples,
        question=question,
        scratchpad=scratchpad,
        prompt=REFLEXION_REACT_REFLECT_INSTRUCTION_HOTPOTQA,
    )
    assert reflections == gt_out_reflections
    assert reflections_str == gt_out_reflections_str

    # Test reflexion followed by last attempt and reflexion.
    gt_out_reflections = ["1"]
    gt_out_reflections_str = "You have attempted to answer following question before and failed. The following reflection(s) give a plan to avoid failing to answer the question in the same way you did previously. Use them to improve your strategy of correctly answering the given question.\nReflections:\n- 1"
    reflector = ReflexionReActReflector(
        llm=MockLLM("gpt-3.5-turbo", responses=["1"]),
    )
    reflections, reflections_str, reflections_out = reflector.reflect(
        reflect_strategy="reflexion",
        examples=examples,
        question=question,
        scratchpad=scratchpad,
        prompt=REFLEXION_REACT_REFLECT_INSTRUCTION_HOTPOTQA,
    )
    assert reflections_out
    assert reflections == gt_out_reflections
    assert reflections_str == gt_out_reflections_str
    gt_out_reflections = ["1"]
    gt_out_reflections_str = "You have attempted to answer the following question before and failed. Below is the last trial you attempted to answer the question.\nQuestion: What is the capital of France?\nInitial scratchpad content\n(END PREVIOUS TRIAL)\n\nThe following reflection(s) give a plan to avoid failing to answer the question in the same way you did previously. Use them to improve your strategy of correctly answering the given question.\nReflections:\n- 1"
    reflections, reflections_str, reflections_out = reflector.reflect(
        reflect_strategy="last_attempt_and_reflexion",
        examples=examples,
        question=question,
        scratchpad=scratchpad,
        prompt=REFLEXION_REACT_REFLECT_INSTRUCTION_HOTPOTQA,
    )
    assert reflections == gt_out_reflections
    assert reflections_str == gt_out_reflections_str

    # Test last attempt and reflexion followed by reflexion.
    gt_out_reflections = ["1"]
    gt_out_reflections_str = "You have attempted to answer the following question before and failed. Below is the last trial you attempted to answer the question.\nQuestion: What is the capital of France?\nInitial scratchpad content\n(END PREVIOUS TRIAL)\n\nThe following reflection(s) give a plan to avoid failing to answer the question in the same way you did previously. Use them to improve your strategy of correctly answering the given question.\nReflections:\n- 1"
    reflector = ReflexionReActReflector(
        llm=MockLLM("gpt-3.5-turbo", responses=["1"]),
    )
    reflections, reflections_str, reflections_out = reflector.reflect(
        reflect_strategy="last_attempt_and_reflexion",
        examples=examples,
        question=question,
        scratchpad=scratchpad,
        prompt=REFLEXION_REACT_REFLECT_INSTRUCTION_HOTPOTQA,
    )
    assert reflections_out
    assert reflections == gt_out_reflections
    assert reflections_str == gt_out_reflections_str
    gt_out_reflections = ["1", "1"]
    gt_out_reflections_str = "You have attempted to answer following question before and failed. The following reflection(s) give a plan to avoid failing to answer the question in the same way you did previously. Use them to improve your strategy of correctly answering the given question.\nReflections:\n- 1\n- 1"
    reflections, reflections_str, reflections_out = reflector.reflect(
        reflect_strategy="reflexion",
        examples=examples,
        question=question,
        scratchpad=scratchpad,
        prompt=REFLEXION_REACT_REFLECT_INSTRUCTION_HOTPOTQA,
    )
    assert reflections_out
    assert reflections == gt_out_reflections
    assert reflections_str == gt_out_reflections_str


def test_reflexion_react_reset() -> None:
    """Unit tests for ReflexionReAct Reflector reset method."""
    reflector = ReflexionReActReflector(
        llm=MockLLM("gpt-3.5-turbo", responses=["1"]),
    )
    reflector.reflections = ["c", "a", "t"]
    reflector.reflections_str = "cat"
    reflector.reset()
    assert reflector.reflections == []
    assert reflector.reflections_str == ""
