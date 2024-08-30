"""Unit tests for CoT Code strategies."""

from agential.cog.cot.output import CoTOutput, CoTStepOutput
from agential.cog.cot.prompts import COT_INSTRUCTION_HUMANEVAL
from agential.cog.cot.strategies.code import (
    CoTHEvalStrategy,
    CoTMBPPStrategy,
)
from agential.cog.fewshots.humaneval import HUMANEVAL_FEWSHOT_EXAMPLES_COT
from agential.llm.llm import MockLLM, Response


def test_heval_generate() -> None:
    """Test CoTHEvalStrategy generate."""
    gt_out = CoTOutput(
        answer="```python\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    for i in range(len(numbers)):\n        for j in range(i + 1, len(numbers)):\n            if abs(numbers[i] - numbers[j]) < threshold:\n                return True\n    return False\n\n```",
        total_prompt_tokens=20,
        total_completion_tokens=40,
        total_tokens=60,
        total_prompt_cost=3e-05,
        total_completion_cost=7.999999999999999e-05,
        total_cost=0.00010999999999999999,
        total_prompt_time=1.0,
        total_time=0.5,
        additional_info=CoTStepOutput(
            thought="We need to iterate through the list of numbers and check if any two numbers are closer than the threshold.",
            answer="```python\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    for i in range(len(numbers)):\n        for j in range(i + 1, len(numbers)):\n            if abs(numbers[i] - numbers[j]) < threshold:\n                return True\n    return False\n\n```",
            thought_response=Response(
                input_text="",
                output_text="We need to iterate through the list of numbers and check if any two numbers are closer than the threshold.\n\nFinish\n```python\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    for i in range(len(numbers)):\n        for j in range(i + 1, len(numbers)):\n            if abs(numbers[i] - numbers[j]) < threshold:\n                return True\n    return False\n```",
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
                output_text="Finish\n```python\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    for i in range(len(numbers)):\n        for j in range(i + 1, len(numbers)):\n            if abs(numbers[i] - numbers[j]) < threshold:\n                return True\n    return False\n```\n```",
                prompt_tokens=10,
                completion_tokens=20,
                total_tokens=30,
                prompt_cost=1.5e-05,
                completion_cost=3.9999999999999996e-05,
                total_cost=5.4999999999999995e-05,
                prompt_time=0.5,
            ),
        ),
    )
    responses = [
        "We need to iterate through the list of numbers and check if any two numbers are closer than the threshold.\n\nFinish\n```python\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    for i in range(len(numbers)):\n        for j in range(i + 1, len(numbers)):\n            if abs(numbers[i] - numbers[j]) < threshold:\n                return True\n    return False\n```",
        "Finish\n```python\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    for i in range(len(numbers)):\n        for j in range(i + 1, len(numbers)):\n            if abs(numbers[i] - numbers[j]) < threshold:\n                return True\n    return False\n```\n```",
    ]
    llm = MockLLM("gpt-3.5-turbo", responses=responses)
    strategy = CoTHEvalStrategy(llm=llm, testing=True)

    inst = {
        "task_id": "HumanEval/0",
        "prompt": 'from typing import List\n\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    """ Check if in given list of numbers, are any two numbers closer to each other than\n    given threshold.\n    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n    False\n    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n    True\n    """\n',
        "entry_point": "has_close_elements",
        "canonical_solution": "    for idx, elem in enumerate(numbers):\n        for idx2, elem2 in enumerate(numbers):\n            if idx != idx2:\n                distance = abs(elem - elem2)\n                if distance < threshold:\n                    return True\n\n    return False\n",
        "test": "\n\nMETADATA = {\n    'author': 'jt',\n    'dataset': 'test'\n}\n\n\ndef check(candidate):\n    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.3) == True\n    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.05) == False\n    assert candidate([1.0, 2.0, 5.9, 4.0, 5.0], 0.95) == True\n    assert candidate([1.0, 2.0, 5.9, 4.0, 5.0], 0.8) == False\n    assert candidate([1.0, 2.0, 3.0, 4.0, 5.0, 2.0], 0.1) == True\n    assert candidate([1.1, 2.2, 3.1, 4.1, 5.1], 1.0) == True\n    assert candidate([1.1, 2.2, 3.1, 4.1, 5.1], 0.5) == False\n\n",
    }
    question = inst["prompt"]

    out = strategy.generate(
        question=question,
        examples=HUMANEVAL_FEWSHOT_EXAMPLES_COT,
        prompt=COT_INSTRUCTION_HUMANEVAL,
        additional_keys={},
    )
    assert out == gt_out


def test_instantiate_code_strategies() -> None:
    """Test instantiate all Code strategies."""
    llm = MockLLM("gpt-3.5-turbo", responses=[])
    mbpp_strategy = CoTMBPPStrategy(llm=llm)
    heval_strategy = CoTHEvalStrategy(llm=llm)

    assert isinstance(mbpp_strategy, CoTMBPPStrategy)
    assert isinstance(heval_strategy, CoTHEvalStrategy)
