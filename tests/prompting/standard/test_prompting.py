"""Unit tests for Standard prompting."""

import pytest

from agential.constants import Benchmarks
from agential.core.fewshots.gsm8k import GSM8K_FEWSHOT_EXAMPLES_DIRECT
from agential.core.fewshots.hotpotqa import HOTPOTQA_FEWSHOT_EXAMPLES_DIRECT
from agential.core.fewshots.humaneval import HUMANEVAL_FEWSHOT_EXAMPLES_DIRECT
from agential.llm.llm import BaseLLM, MockLLM, Response
from agential.prompting.standard.output import StandardOutput, StandardStepOutput
from agential.prompting.standard.prompting import Standard
from agential.prompting.standard.prompts import (
    STANDARD_INSTRUCTION_GSM8K,
    STANDARD_INSTRUCTION_HOTPOTQA,
    STANDARD_INSTRUCTION_HUMANEVAL,
)
from agential.prompting.standard.strategies.base import StandardBaseStrategy
from agential.prompting.standard.strategies.code import (
    StandardHEvalStrategy,
    StandardMBPPStrategy,
)
from agential.prompting.standard.strategies.math import (
    StandardGSM8KStrategy,
    StandardSVAMPStrategy,
    StandardTabMWPStrategy,
)
from agential.prompting.standard.strategies.qa import (
    StandardAmbigNQStrategy,
    StandardFEVERStrategy,
    StandardHotQAStrategy,
    StandardTriviaQAStrategy,
)


def test_init() -> None:
    """Test initialization."""
    llm = MockLLM("gpt-3.5-turbo", responses=[])
    method = Standard(llm=llm, benchmark="hotpotqa", testing=True)
    assert isinstance(method, Standard)
    assert isinstance(method.llm, BaseLLM)
    assert method.benchmark == "hotpotqa"
    assert isinstance(method.strategy, StandardBaseStrategy)


def test_get_strategy() -> None:
    """Tests Standard get_strategy method."""
    llm = MockLLM("gpt-3.5-turbo", responses=[])

    # QA benchmarks.
    assert isinstance(
        Standard.get_strategy(Benchmarks.HOTPOTQA, llm=llm),
        StandardHotQAStrategy,
    )
    assert isinstance(
        Standard.get_strategy(Benchmarks.TRIVIAQA, llm=llm),
        StandardTriviaQAStrategy,
    )
    assert isinstance(
        Standard.get_strategy(Benchmarks.AMBIGNQ, llm=llm),
        StandardAmbigNQStrategy,
    )
    assert isinstance(
        Standard.get_strategy(Benchmarks.FEVER, llm=llm),
        StandardFEVERStrategy,
    )

    # Math benchmarks.
    assert isinstance(
        Standard.get_strategy(Benchmarks.GSM8K, llm=llm),
        StandardGSM8KStrategy,
    )
    assert isinstance(
        Standard.get_strategy(Benchmarks.SVAMP, llm=llm),
        StandardSVAMPStrategy,
    )
    assert isinstance(
        Standard.get_strategy(Benchmarks.TABMWP, llm=llm),
        StandardTabMWPStrategy,
    )

    # Code benchmarks.
    assert isinstance(
        Standard.get_strategy(Benchmarks.HUMANEVAL, llm=llm),
        StandardHEvalStrategy,
    )
    assert isinstance(
        Standard.get_strategy(Benchmarks.MBPP, llm=llm),
        StandardMBPPStrategy,
    )

    # Unsupported benchmark.
    with pytest.raises(ValueError, match="Unsupported benchmark: unknown for Standard"):
        Standard.get_strategy("unknown", llm=llm)


def test_get_fewshots() -> None:
    """Tests Standard get_fewshots method."""
    # Test valid input.
    benchmark = Benchmarks.HOTPOTQA
    result = Standard.get_fewshots(benchmark, fewshot_type="direct")
    assert isinstance(result, dict)
    assert result == {"examples": HOTPOTQA_FEWSHOT_EXAMPLES_DIRECT}

    # Test unsupported benchmark.
    with pytest.raises(
        ValueError, match="Benchmark 'unknown' few-shots not found for Standard."
    ):
        Standard.get_fewshots("unknown", fewshot_type="react")

    # Test unsupported fewshot_type.
    with pytest.raises(
        ValueError,
        match="Benchmark 'hotpotqa' few-shot type not supported for Standard.",
    ):
        Standard.get_fewshots("hotpotqa", fewshot_type="pot")


def test_get_prompts() -> None:
    """Tests Standard get_prompts method."""
    # Test valid input.
    benchmark = Benchmarks.HOTPOTQA
    result = Standard.get_prompts(benchmark)
    assert result == {"prompt": STANDARD_INSTRUCTION_HOTPOTQA}

    # Test unsupported benchmark.
    with pytest.raises(
        ValueError, match="Benchmark 'unknown' prompt not found for Standard."
    ):
        Standard.get_prompts("unknown")


def test_generate() -> None:
    """Test generate."""
    # Test QA.
    question = 'Who was once considered the best kick boxer in the world, however he has been involved in a number of controversies relating to his "unsportsmanlike conducts" in the sport and crimes of violence outside of the ring'

    gt_out = StandardOutput(
        answer=[["Badr Hari"]],
        total_prompt_tokens=10,
        total_completion_tokens=20,
        total_tokens=30,
        total_prompt_cost=1.5e-05,
        total_completion_cost=3.9999999999999996e-05,
        total_cost=5.4999999999999995e-05,
        total_prompt_time=0.5,
        total_time=0.5,
        additional_info=[
            [
                StandardStepOutput(
                    answer="Badr Hari",
                    answer_response=Response(
                        input_text="",
                        output_text="Badr Hari",
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
        ],
    )
    responses = ["Badr Hari"]
    llm = MockLLM("gpt-3.5-turbo", responses=responses)
    method = Standard(llm=llm, benchmark="hotpotqa", testing=True)

    out = method.generate(
        question=question,
        examples=HOTPOTQA_FEWSHOT_EXAMPLES_DIRECT,
        prompt=STANDARD_INSTRUCTION_HOTPOTQA,
        additional_keys={},
    )
    assert out == gt_out

    # Test Math.
    question = "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with 4933828. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?"

    gt_out = StandardOutput(
        answer=[["96"]],
        total_prompt_tokens=10,
        total_completion_tokens=20,
        total_tokens=30,
        total_prompt_cost=1.5e-05,
        total_completion_cost=3.9999999999999996e-05,
        total_cost=5.4999999999999995e-05,
        total_prompt_time=0.5,
        total_time=0.5,
        additional_info=[
            [
                StandardStepOutput(
                    answer="96",
                    answer_response=Response(
                        input_text="",
                        output_text="96",
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
        ],
    )
    responses = ["96"]
    llm = MockLLM("gpt-3.5-turbo", responses=responses)
    method = Standard(llm=llm, benchmark="gsm8k", testing=True)

    out = method.generate(
        question=question,
        examples=GSM8K_FEWSHOT_EXAMPLES_DIRECT,
        prompt=STANDARD_INSTRUCTION_GSM8K,
        additional_keys={},
        num_retries=1,
        warming=[None],
    )
    assert out == gt_out

    # Test Code.
    inst = {
        "task_id": "HumanEval/0",
        "prompt": 'from typing import List\n\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    """ Check if in given list of numbers, are any two numbers closer to each other than\n    given threshold.\n    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n    False\n    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n    True\n    """\n',
        "entry_point": "has_close_elements",
        "canonical_solution": "    for idx, elem in enumerate(numbers):\n        for idx2, elem2 in enumerate(numbers):\n            if idx != idx2:\n                distance = abs(elem - elem2)\n                if distance < threshold:\n                    return True\n\n    return False\n",
        "test": "\n\nMETADATA = {\n    'author': 'jt',\n    'dataset': 'test'\n}\n\n\ndef check(candidate):\n    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.3) == True\n    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.05) == False\n    assert candidate([1.0, 2.0, 5.9, 4.0, 5.0], 0.95) == True\n    assert candidate([1.0, 2.0, 5.9, 4.0, 5.0], 0.8) == False\n    assert candidate([1.0, 2.0, 3.0, 4.0, 5.0, 2.0], 0.1) == True\n    assert candidate([1.1, 2.2, 3.1, 4.1, 5.1], 1.0) == True\n    assert candidate([1.1, 2.2, 3.1, 4.1, 5.1], 0.5) == False\n\n",
    }
    question = inst["prompt"]

    gt_out = StandardOutput(
        answer=[
            [
                "from typing import List\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    for i in range(len(numbers)):\n        for j in range(i+1, len(numbers)):\n            if abs(numbers[i] - numbers[j]) < threshold:\n                return True\n    return False\n\n# Testing the function\nprint(has_close_elements([1.0, 2.0, 3.0], 0.5))  # False\nprint(has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3))  # True"
            ]
        ],
        total_prompt_tokens=10,
        total_completion_tokens=20,
        total_tokens=30,
        total_prompt_cost=1.5e-05,
        total_completion_cost=3.9999999999999996e-05,
        total_cost=5.4999999999999995e-05,
        total_prompt_time=0.5,
        total_time=0.5,
        additional_info=[
            [
                StandardStepOutput(
                    answer="from typing import List\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    for i in range(len(numbers)):\n        for j in range(i+1, len(numbers)):\n            if abs(numbers[i] - numbers[j]) < threshold:\n                return True\n    return False\n\n# Testing the function\nprint(has_close_elements([1.0, 2.0, 3.0], 0.5))  # False\nprint(has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3))  # True",
                    answer_response=Response(
                        input_text="",
                        output_text="from typing import List\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    for i in range(len(numbers)):\n        for j in range(i+1, len(numbers)):\n            if abs(numbers[i] - numbers[j]) < threshold:\n                return True\n    return False\n\n# Testing the function\nprint(has_close_elements([1.0, 2.0, 3.0], 0.5))  # False\nprint(has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3))  # True",
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
        ],
    )
    responses = [
        "from typing import List\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    for i in range(len(numbers)):\n        for j in range(i+1, len(numbers)):\n            if abs(numbers[i] - numbers[j]) < threshold:\n                return True\n    return False\n\n# Testing the function\nprint(has_close_elements([1.0, 2.0, 3.0], 0.5))  # False\nprint(has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3))  # True"
    ]
    llm = MockLLM("gpt-3.5-turbo", responses=responses)
    method = Standard(llm=llm, benchmark="humaneval", testing=True)

    out = method.generate(
        question=question,
        examples=HUMANEVAL_FEWSHOT_EXAMPLES_DIRECT,
        prompt=STANDARD_INSTRUCTION_HUMANEVAL,
        additional_keys={},
        num_retries=1,
        warming=[None],
    )
    assert out == gt_out
