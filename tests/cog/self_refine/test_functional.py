"""Unit tests for Self-Refine functional."""

from agential.cog.self_refine.functional import (
    _build_agent_prompt,
    _build_critique_prompt,
    _build_refine_prompt,
    _prompt_agent,
    _prompt_critique,
    _prompt_refine,
    accumulate_metrics,
)
from agential.cog.self_refine.output import SelfRefineStepOutput
from agential.llm.llm import MockLLM, Response


def test__build_agent_prompt() -> None:
    """Tests _build_agent_prompt."""
    question = "What is the capital of France?"
    examples = "Example 1: What is the capital of Germany? Berlin.\nExample 2: What is the capital of Italy? Rome."
    prompt = "Question: {question}\nExamples:\n{examples}\nAnswer:"
    additional_keys = {"additional_info": "This is some additional info."}

    expected_output = (
        "Question: What is the capital of France?\n"
        "Examples:\nExample 1: What is the capital of Germany? Berlin.\n"
        "Example 2: What is the capital of Italy? Rome.\n"
        "Answer:"
    )

    result = _build_agent_prompt(question, examples, prompt, additional_keys)
    assert result == expected_output


def test__prompt_agent() -> None:
    """Tests _prompt_agent."""
    question = "What is the capital of France?"
    examples = "Example 1: What is the capital of Germany? Berlin.\nExample 2: What is the capital of Italy? Rome."
    prompt = "Question: {question}\nExamples:\n{examples}\nAnswer:"
    additional_keys = {"additional_info": "This is some additional info."}

    llm = MockLLM("gpt-3.5-turbo", responses=["1"])

    result = _prompt_agent(llm, question, examples, prompt, additional_keys)
    assert result == Response(
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


def test__build_critique_prompt() -> None:
    """Tests _build_critique_prompt."""
    question = "What is the capital of France?"
    examples = "Example 1: What is the capital of Germany? Berlin.\nExample 2: What is the capital of Italy? Rome."
    answer = "Paris"
    prompt = "Question: {question}\nExamples:\n{examples}\nAnswer: {answer}\nCritique:"
    additional_keys = {"additional_info": "This is some additional info."}

    expected_output = (
        "Question: What is the capital of France?\n"
        "Examples:\nExample 1: What is the capital of Germany? Berlin.\n"
        "Example 2: What is the capital of Italy? Rome.\n"
        "Answer: Paris\n"
        "Critique:"
    )

    result = _build_critique_prompt(question, examples, answer, prompt, additional_keys)
    assert result == expected_output


def test__prompt_critique() -> None:
    """Tests _prompt_critique."""
    question = "What is the capital of France?"
    examples = "Example 1: What is the capital of Germany? Berlin.\nExample 2: What is the capital of Italy? Rome."
    answer = "Paris"
    prompt = "Question: {question}\nExamples:\n{examples}\nAnswer: {answer}\nCritique:"
    additional_keys = {"additional_info": "This is some additional info."}

    llm = MockLLM("gpt-3.5-turbo", responses=["The answer is correct."])

    result = _prompt_critique(llm, question, examples, answer, prompt, additional_keys)
    assert result == Response(
        input_text="",
        output_text="The answer is correct.",
        prompt_tokens=10,
        completion_tokens=20,
        total_tokens=30,
        prompt_cost=1.5e-05,
        completion_cost=3.9999999999999996e-05,
        total_cost=5.4999999999999995e-05,
        prompt_time=0.5,
    )


def test__build_refine_prompt() -> None:
    """Tests _build_refine_prompt."""
    question = "What is the capital of France?"
    examples = "Example 1: What is the capital of Germany? Berlin.\nExample 2: What is the capital of Italy? Rome."
    answer = "Paris"
    critique = "The answer is correct but lacks detail."
    prompt = "Question: {question}\nExamples:\n{examples}\nAnswer: {answer}\nCritique: {critique}\nRefined Answer:"
    additional_keys = {"additional_info": "This is some additional info."}

    expected_output = (
        "Question: What is the capital of France?\n"
        "Examples:\nExample 1: What is the capital of Germany? Berlin.\n"
        "Example 2: What is the capital of Italy? Rome.\n"
        "Answer: Paris\n"
        "Critique: The answer is correct but lacks detail.\n"
        "Refined Answer:"
    )

    result = _build_refine_prompt(
        question, examples, answer, critique, prompt, additional_keys
    )
    assert result == expected_output


def test__prompt_refine() -> None:
    """Tests _prompt_refine."""
    question = "What is the capital of France?"
    examples = "Example 1: What is the capital of Germany? Berlin.\nExample 2: What is the capital of Italy? Rome."
    answer = "Paris"
    critique = "The answer is correct but lacks detail."
    prompt = "Question: {question}\nExamples:\n{examples}\nAnswer: {answer}\nCritique: {critique}\nRefined Answer:"
    additional_keys = {"additional_info": "This is some additional info."}

    llm = MockLLM(
        "gpt-3.5-turbo",
        responses=["The capital of France, Paris, is known for its rich history."],
    )

    result = _prompt_refine(
        llm, question, examples, answer, critique, prompt, additional_keys
    )
    assert result == Response(
        input_text="",
        output_text="The capital of France, Paris, is known for its rich history.",
        prompt_tokens=10,
        completion_tokens=20,
        total_tokens=30,
        prompt_cost=1.5e-05,
        completion_cost=3.9999999999999996e-05,
        total_cost=5.4999999999999995e-05,
        prompt_time=0.5,
    )


def test_accumulate_metrics() -> None:
    """Tests accumulate_metrics function."""
    answer_response_1 = Response(
        input_text="",
        output_text="",
        prompt_tokens=10,
        completion_tokens=20,
        total_tokens=30,
        prompt_cost=0.001,
        completion_cost=0.002,
        total_cost=0.003,
        prompt_time=0.5,
    )

    critique_response_1 = Response(
        input_text="",
        output_text="",
        prompt_tokens=5,
        completion_tokens=10,
        total_tokens=15,
        prompt_cost=0.0005,
        completion_cost=0.001,
        total_cost=0.0015,
        prompt_time=0.25,
    )

    answer_response_2 = Response(
        input_text="",
        output_text="",
        prompt_tokens=8,
        completion_tokens=16,
        total_tokens=24,
        prompt_cost=0.0008,
        completion_cost=0.0016,
        total_cost=0.0024,
        prompt_time=0.4,
    )

    critique_response_2 = Response(
        input_text="",
        output_text="",
        prompt_tokens=4,
        completion_tokens=8,
        total_tokens=12,
        prompt_cost=0.0004,
        completion_cost=0.0008,
        total_cost=0.0012,
        prompt_time=0.2,
    )

    step_output_1 = SelfRefineStepOutput(
        answer="Paris",
        critique="Correct, but you might mention it's the capital of France.",
        answer_response=answer_response_1,
        critique_response=critique_response_1,
    )

    step_output_2 = SelfRefineStepOutput(
        answer="Berlin",
        critique="Correct, but you might mention it's the capital of Germany.",
        answer_response=answer_response_2,
        critique_response=critique_response_2,
    )

    steps = [step_output_1, step_output_2]

    expected_metrics = {
        "total_prompt_tokens": 27,  # 10 + 5 + 8 + 4
        "total_completion_tokens": 54,  # 20 + 10 + 16 + 8
        "total_tokens": 81,  # 30 + 15 + 24 + 12
        "total_prompt_cost": 0.0027,  # 0.001 + 0.0005 + 0.0008 + 0.0004
        "total_completion_cost": 0.0054,  # 0.002 + 0.001 + 0.0016 + 0.0008
        "total_cost": 0.0081,  # 0.003 + 0.0015 + 0.0024 + 0.0012
        "total_prompt_time": 1.35,  # 0.5 + 0.25 + 0.4 + 0.2
    }

    result_metrics = accumulate_metrics(steps)

    assert result_metrics == expected_metrics
