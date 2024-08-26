"""Unit tests for Self-Refine functional."""

from agential.cog.self_refine.functional import _build_agent_prompt, _build_critique_prompt, _build_refine_prompt, _prompt_agent, _prompt_critique, _prompt_refine
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
    assert result == Response(input_text='', output_text='1', prompt_tokens=10, completion_tokens=20, total_tokens=30, prompt_cost=1.5e-05, completion_cost=3.9999999999999996e-05, total_cost=5.4999999999999995e-05, prompt_time=0.5)


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
    assert result == Response(input_text='', output_text='The answer is correct.', prompt_tokens=10, completion_tokens=20, total_tokens=30, prompt_cost=1.5e-05, completion_cost=3.9999999999999996e-05, total_cost=5.4999999999999995e-05, prompt_time=0.5)


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

    result = _build_refine_prompt(question, examples, answer, critique, prompt, additional_keys)
    assert result == expected_output


def test__prompt_refine() -> None:
    """Tests _prompt_refine."""
    question = "What is the capital of France?"
    examples = "Example 1: What is the capital of Germany? Berlin.\nExample 2: What is the capital of Italy? Rome."
    answer = "Paris"
    critique = "The answer is correct but lacks detail."
    prompt = "Question: {question}\nExamples:\n{examples}\nAnswer: {answer}\nCritique: {critique}\nRefined Answer:"
    additional_keys = {"additional_info": "This is some additional info."}

    llm = MockLLM("gpt-3.5-turbo", responses=["The capital of France, Paris, is known for its rich history."])

    result = _prompt_refine(llm, question, examples, answer, critique, prompt, additional_keys)
    assert result == Response(input_text='', output_text='The capital of France, Paris, is known for its rich history.', prompt_tokens=10, completion_tokens=20, total_tokens=30, prompt_cost=1.5e-05, completion_cost=3.9999999999999996e-05, total_cost=5.4999999999999995e-05, prompt_time=0.5)
