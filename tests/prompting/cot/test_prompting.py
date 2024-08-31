"""Unit tests for CoT."""

import pytest

from agential.constants import Benchmarks
from agential.core.fewshots.gsm8k import GSM8K_FEWSHOT_EXAMPLES_COT
from agential.core.fewshots.hotpotqa import HOTPOTQA_FEWSHOT_EXAMPLES_COT
from agential.core.fewshots.humaneval import HUMANEVAL_FEWSHOT_EXAMPLES_COT
from agential.llm.llm import BaseLLM, MockLLM, Response
from agential.prompting.cot.output import CoTOutput, CoTStepOutput
from agential.prompting.cot.prompting import CoT
from agential.prompting.cot.prompts import (
    COT_INSTRUCTION_GSM8K,
    COT_INSTRUCTION_HOTPOTQA,
    COT_INSTRUCTION_HUMANEVAL,
)
from agential.prompting.cot.strategies.base import CoTBaseStrategy
from agential.prompting.cot.strategies.code import CoTHEvalStrategy, CoTMBPPStrategy
from agential.prompting.cot.strategies.math import (
    CoTGSM8KStrategy,
    CoTSVAMPStrategy,
    CoTTabMWPStrategy,
)
from agential.prompting.cot.strategies.qa import (
    CoTAmbigNQStrategy,
    CoTFEVERStrategy,
    CoTHotQAStrategy,
    CoTTriviaQAStrategy,
)


def test_init() -> None:
    """Test initialization."""
    llm = MockLLM("gpt-3.5-turbo", responses=[])
    method = CoT(llm=llm, benchmark="hotpotqa", testing=True)
    assert isinstance(method, CoT)
    assert isinstance(method.llm, BaseLLM)
    assert method.benchmark == "hotpotqa"
    assert isinstance(method.strategy, CoTBaseStrategy)


def test_get_strategy() -> None:
    """Tests CoT get_strategy method."""
    llm = MockLLM("gpt-3.5-turbo", responses=[])

    # QA benchmarks.
    assert isinstance(
        CoT.get_strategy(Benchmarks.HOTPOTQA, llm=llm),
        CoTHotQAStrategy,
    )
    assert isinstance(
        CoT.get_strategy(Benchmarks.TRIVIAQA, llm=llm),
        CoTTriviaQAStrategy,
    )
    assert isinstance(
        CoT.get_strategy(Benchmarks.AMBIGNQ, llm=llm),
        CoTAmbigNQStrategy,
    )
    assert isinstance(
        CoT.get_strategy(Benchmarks.FEVER, llm=llm),
        CoTFEVERStrategy,
    )

    # Math benchmarks.
    assert isinstance(
        CoT.get_strategy(Benchmarks.GSM8K, llm=llm),
        CoTGSM8KStrategy,
    )
    assert isinstance(
        CoT.get_strategy(Benchmarks.SVAMP, llm=llm),
        CoTSVAMPStrategy,
    )
    assert isinstance(
        CoT.get_strategy(Benchmarks.TABMWP, llm=llm),
        CoTTabMWPStrategy,
    )

    # Code benchmarks.
    assert isinstance(
        CoT.get_strategy(Benchmarks.HUMANEVAL, llm=llm),
        CoTHEvalStrategy,
    )
    assert isinstance(
        CoT.get_strategy(Benchmarks.MBPP, llm=llm),
        CoTMBPPStrategy,
    )

    # Unsupported benchmark.
    with pytest.raises(ValueError, match="Unsupported benchmark: unknown for CoT"):
        CoT.get_strategy("unknown", llm=llm)


def test_get_fewshots() -> None:
    """Tests CoT get_fewshots method."""
    # Test valid input.
    benchmark = Benchmarks.HOTPOTQA
    result = CoT.get_fewshots(benchmark, fewshot_type="cot")
    assert isinstance(result, dict)
    assert result == {"examples": HOTPOTQA_FEWSHOT_EXAMPLES_COT}

    # Test unsupported benchmark.
    with pytest.raises(
        ValueError, match="Benchmark 'unknown' few-shots not found for CoT."
    ):
        CoT.get_fewshots("unknown", fewshot_type="react")

    # Test unsupported fewshot_type.
    with pytest.raises(
        ValueError, match="Benchmark 'hotpotqa' few-shot type not supported for CoT."
    ):
        CoT.get_fewshots("hotpotqa", fewshot_type="pot")


def test_get_prompts() -> None:
    """Tests CoT get_prompts method."""
    # Test valid input.
    benchmark = Benchmarks.HOTPOTQA
    result = CoT.get_prompts(benchmark)
    assert result == {"prompt": COT_INSTRUCTION_HOTPOTQA}

    # Test unsupported benchmark.
    with pytest.raises(
        ValueError, match="Benchmark 'unknown' prompt not found for CoT."
    ):
        CoT.get_prompts("unknown")


def test_generate() -> None:
    """Test generate."""
    # Test QA.
    question = 'Who was once considered the best kick boxer in the world, however he has been involved in a number of controversies relating to his "unsportsmanlike conducts" in the sport and crimes of violence outside of the ring'

    gt_out = CoTOutput(
        answer=["Badr Hari"],
        total_prompt_tokens=20,
        total_completion_tokens=40,
        total_tokens=0,
        total_prompt_cost=3e-05,
        total_completion_cost=7.999999999999999e-05,
        total_cost=0.0,
        total_prompt_time=1.0,
        total_time=0.5,
        additional_info=[
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
        ],
    )
    responses = [
        "Let's think step by step. Given the information provided, the person described is likely to be Badr Hari, a Moroccan-Dutch kickboxer known for his skills in the ring as well as his controversial behavior both inside and outside of the sport.\nAction: Finish[Badr Hari]",
        "Finish[Badr Hari]",
    ]
    method = CoT(
        llm=MockLLM("gpt-3.5-turbo", responses=responses),
        benchmark="hotpotqa",
        testing=True,
    )
    out = method.generate(
        question=question,
        examples=HOTPOTQA_FEWSHOT_EXAMPLES_COT,
        prompt=COT_INSTRUCTION_HOTPOTQA,
        additional_keys={},
    )
    assert out == gt_out

    # Test Math.
    question = "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with 4933828. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?"

    gt_out = CoTOutput(
        answer=[
            "```python\neggs_laid_per_day = 16\neggs_eaten_for_breakfast = 3\neggs_for_muffins = 4933828\neggs_remaining = eggs_laid_per_day - eggs_eaten_for_breakfast - eggs_for_muffins\neggs_sold = max(eggs_remaining, 0)\nmoney_per_egg = 2\nmoney_made_daily = eggs_sold * money_per_egg\nanswer = money_made_daily\n```"
        ],
        total_prompt_tokens=20,
        total_completion_tokens=40,
        total_tokens=0,
        total_prompt_cost=3e-05,
        total_completion_cost=7.999999999999999e-05,
        total_cost=0.0,
        total_prompt_time=1.0,
        total_time=0.5,
        additional_info=[
            CoTStepOutput(
                thought="Let's break this down step by step. Janet's ducks lay 16 eggs per day. She eats 3 for breakfast, so the remaining eggs are 16 - 3 = 13. She bakes muffins with 4933828 eggs, so the number of eggs available for sale is 13 - 4933828 = -4933815, which doesn't make sense. There seems to be a mistake in the calculation of the available eggs. Let's correct this and calculate how much Janet makes at the farmers' market daily.",
                answer="```python\neggs_laid_per_day = 16\neggs_eaten_for_breakfast = 3\neggs_for_muffins = 4933828\neggs_remaining = eggs_laid_per_day - eggs_eaten_for_breakfast - eggs_for_muffins\neggs_sold = max(eggs_remaining, 0)\nmoney_per_egg = 2\nmoney_made_daily = eggs_sold * money_per_egg\nanswer = money_made_daily\n```",
                thought_response=Response(
                    input_text="",
                    output_text="Let's break this down step by step. Janet's ducks lay 16 eggs per day. She eats 3 for breakfast, so the remaining eggs are 16 - 3 = 13. She bakes muffins with 4933828 eggs, so the number of eggs available for sale is 13 - 4933828 = -4933815, which doesn't make sense. There seems to be a mistake in the calculation of the available eggs. Let's correct this and calculate how much Janet makes at the farmers' market daily.\n\nAction: Finish[\n```python\neggs_per_day = 16\neggs_for_breakfast = 3\neggs_remaining = eggs_per_day - eggs_for_breakfast\neggs_for_muffins = 4933828\neggs_available_for_sale = eggs_remaining - eggs_for_muffins\negg_price = 2\ndaily_earnings = eggs_available_for_sale * egg_price\nanswer = daily_earnings\n```\n]",
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
                    output_text="```python\neggs_laid_per_day = 16\neggs_eaten_for_breakfast = 3\neggs_for_muffins = 4933828\neggs_remaining = eggs_laid_per_day - eggs_eaten_for_breakfast - eggs_for_muffins\neggs_sold = max(eggs_remaining, 0)\nmoney_per_egg = 2\nmoney_made_daily = eggs_sold * money_per_egg\nanswer = money_made_daily\n```",
                    prompt_tokens=10,
                    completion_tokens=20,
                    total_tokens=30,
                    prompt_cost=1.5e-05,
                    completion_cost=3.9999999999999996e-05,
                    total_cost=5.4999999999999995e-05,
                    prompt_time=0.5,
                ),
            )
        ],
    )
    responses = [
        "Let's break this down step by step. Janet's ducks lay 16 eggs per day. She eats 3 for breakfast, so the remaining eggs are 16 - 3 = 13. She bakes muffins with 4933828 eggs, so the number of eggs available for sale is 13 - 4933828 = -4933815, which doesn't make sense. There seems to be a mistake in the calculation of the available eggs. Let's correct this and calculate how much Janet makes at the farmers' market daily.\n\nAction: Finish[\n```python\neggs_per_day = 16\neggs_for_breakfast = 3\neggs_remaining = eggs_per_day - eggs_for_breakfast\neggs_for_muffins = 4933828\neggs_available_for_sale = eggs_remaining - eggs_for_muffins\negg_price = 2\ndaily_earnings = eggs_available_for_sale * egg_price\nanswer = daily_earnings\n```\n]",
        "```python\neggs_laid_per_day = 16\neggs_eaten_for_breakfast = 3\neggs_for_muffins = 4933828\neggs_remaining = eggs_laid_per_day - eggs_eaten_for_breakfast - eggs_for_muffins\neggs_sold = max(eggs_remaining, 0)\nmoney_per_egg = 2\nmoney_made_daily = eggs_sold * money_per_egg\nanswer = money_made_daily\n```",
    ]
    method = CoT(
        llm=MockLLM("gpt-3.5-turbo", responses=responses),
        benchmark="gsm8k",
        testing=True,
    )
    out = method.generate(
        question=question,
        examples=GSM8K_FEWSHOT_EXAMPLES_COT,
        prompt=COT_INSTRUCTION_GSM8K,
        additional_keys={},
    )
    assert out == gt_out

    # Test code.
    gt_out = CoTOutput(
        answer=[
            "```python\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    for i in range(len(numbers)):\n        for j in range(i + 1, len(numbers)):\n            if abs(numbers[i] - numbers[j]) < threshold:\n                return True\n    return False\n\n```"
        ],
        total_prompt_tokens=20,
        total_completion_tokens=40,
        total_tokens=0,
        total_prompt_cost=3e-05,
        total_completion_cost=7.999999999999999e-05,
        total_cost=0.0,
        total_prompt_time=1.0,
        total_time=0.5,
        additional_info=[
            CoTStepOutput(
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
            )
        ],
    )
    responses = [
        "We need to iterate through the list of numbers and check if any two numbers are closer than the threshold.\n\nFinish\n```python\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    for i in range(len(numbers)):\n        for j in range(i + 1, len(numbers)):\n            if abs(numbers[i] - numbers[j]) < threshold:\n                return True\n    return False\n```",
        "Finish\n```python\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    for i in range(len(numbers)):\n        for j in range(i + 1, len(numbers)):\n            if abs(numbers[i] - numbers[j]) < threshold:\n                return True\n    return False\n```\n```",
    ]
    llm = MockLLM("gpt-3.5-turbo", responses=responses)
    method = CoT(llm=llm, benchmark="humaneval", testing=True)

    inst = {
        "task_id": "HumanEval/0",
        "prompt": 'from typing import List\n\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    """ Check if in given list of numbers, are any two numbers closer to each other than\n    given threshold.\n    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n    False\n    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n    True\n    """\n',
        "entry_point": "has_close_elements",
        "canonical_solution": "    for idx, elem in enumerate(numbers):\n        for idx2, elem2 in enumerate(numbers):\n            if idx != idx2:\n                distance = abs(elem - elem2)\n                if distance < threshold:\n                    return True\n\n    return False\n",
        "test": "\n\nMETADATA = {\n    'author': 'jt',\n    'dataset': 'test'\n}\n\n\ndef check(candidate):\n    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.3) == True\n    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.05) == False\n    assert candidate([1.0, 2.0, 5.9, 4.0, 5.0], 0.95) == True\n    assert candidate([1.0, 2.0, 5.9, 4.0, 5.0], 0.8) == False\n    assert candidate([1.0, 2.0, 3.0, 4.0, 5.0, 2.0], 0.1) == True\n    assert candidate([1.1, 2.2, 3.1, 4.1, 5.1], 1.0) == True\n    assert candidate([1.1, 2.2, 3.1, 4.1, 5.1], 0.5) == False\n\n",
    }
    question = inst["prompt"]

    out = method.generate(
        question=question,
        examples=HUMANEVAL_FEWSHOT_EXAMPLES_COT,
        prompt=COT_INSTRUCTION_HUMANEVAL,
        additional_keys={},
    )
    assert out == gt_out
