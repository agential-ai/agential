"""Unit tests the CoT functional module."""

from agential.llm.llm import MockLLM, Response
from agential.prompting.cot.functional import _build_prompt, _prompt_llm, accumulate_metrics
from agential.prompting.cot.output import CoTStepOutput


def test__build_prompt() -> None:
    """Tests _build_prompt."""
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

    result = _build_prompt(question, examples, prompt, additional_keys)
    assert result == expected_output


def test__prompt_llm() -> None:
    """Tests _prompt_llm."""
    question = "What is the capital of France?"
    examples = "Example 1: What is the capital of Germany? Berlin.\nExample 2: What is the capital of Italy? Rome."
    prompt = "Question: {question}\nExamples:\n{examples}\nAnswer:"
    additional_keys = {"additional_info": "This is some additional info."}

    llm = MockLLM("gpt-3.5-turbo", responses=["Paris"])

    result = _prompt_llm(llm, question, examples, prompt, additional_keys)
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


def test_accumulate_metrics():
    """Accumulate total metrics from a list of CoTStepOutput objects.

    This function calculates and aggregates various metrics across all steps in the input list.
    It sums up token counts, costs, and time measurements for both thought and action components.

    Args:
        steps (List[CoTStepOutput]): A list of CoTStepOutput objects representing individual steps.

    Returns:
        Dict[str, Any]: A dictionary containing the following accumulated metrics:
            - total_prompt_tokens (int): Total number of prompt tokens used.
            - total_completion_tokens (int): Total number of completion tokens generated.
            - total_tokens (int): Total number of tokens (prompt + completion).
            - total_prompt_cost (float): Total cost associated with prompts.
            - total_completion_cost (float): Total cost associated with completions.
            - total_cost (float): Total overall cost (prompt + completion).
            - total_prompt_time (float): Total time spent on prompts.
    """
    steps = [
        CoTStepOutput(
            thought="Let's think step by step. Given the information provided, the person described is likely to be Badr Hari, a Moroccan-Dutch kickboxer known for his skills in the ring as well as his controversial behavior both inside and outside of the sport.",
            answer="Badr Hari",
            thought_response=Response(
                input_text="",
                output_text="Let's think step by step. Given the information provided, the person described is likely to be Badr Hari, a Moroccan-Dutch kickboxer known for his skills in the ring as well as his controversial behavior both inside and outside of the sport.\nAction: Finish[Badr Hari]",
                prompt_tokens=10,
                completion_tokens=20,
                total_tokens=30,
                prompt_cost=1.5e-05,
                completion_cost=3.9999999999999996e-05,
                total_cost=5.4999999999999995e-05,
                prompt_time=0.5,
            ),
            answer_response=Response(
                input_text="",
                output_text="Finish[Badr Hari]",
                prompt_tokens=10,
                completion_tokens=20,
                total_tokens=30,
                prompt_cost=1.5e-05,
                completion_cost=3.9999999999999996e-05,
                total_cost=5.4999999999999995e-05,
                prompt_time=0.5,
            ),
        )
    ]

    result = accumulate_metrics(steps)

    assert result == {'total_prompt_tokens': 20, 'total_completion_tokens': 40, 'total_tokens': 0, 'total_prompt_cost': 3e-05, 'total_completion_cost': 7.999999999999999e-05, 'total_cost': 0.0, 'total_prompt_time': 1.0}