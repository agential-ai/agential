"""Unit tests the CoT functional module."""

from agential.agents.cot.functional import _build_agent_prompt, _prompt_agent
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

    llm = MockLLM("gpt-3.5-turbo", responses=["Paris"])

    result = _prompt_agent(llm, question, examples, prompt, additional_keys)
    assert result == Response(
        input_text="",
        output_text="Paris",
        prompt_tokens=10,
        completion_tokens=20,
        total_tokens=30,
        prompt_cost=1.5e-05,
        completion_cost=3.9999999999999996e-05,
        total_cost=5.4999999999999995e-05,
        prompt_time=0.5,
    )
