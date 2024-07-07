"""Unit tests for prompt manager logic."""

import pytest

from agential.fewshots.hotpotqa import (
    HOTPOTQA_FEWSHOT_EXAMPLES_COT,
)
from agential.fewshots.manager import FewShotType, get_fewshot_examples


def test_get_fewshot_examples() -> None:
    """Test get_fewshot_examples."""
    # Test valid input.
    benchmark = "hotpotqa"
    fewshot_type = FewShotType.COT
    result = get_fewshot_examples(benchmark, fewshot_type)
    assert result == HOTPOTQA_FEWSHOT_EXAMPLES_COT

    # Test invalid benchmark.
    benchmark = "invalid_benchmark"
    fewshot_type = FewShotType.COT
    with pytest.raises(ValueError):
        result = get_fewshot_examples(benchmark, fewshot_type)

    # Test invalid few-shot type.
    benchmark = "hotpotqa"
    fewshot_type = "invalid_fewshot"
    with pytest.raises(ValueError):
        result = get_fewshot_examples(benchmark, fewshot_type)

    # Test invalid few-shot type for the given benchmark.
    benchmark = "hotpotqa"
    fewshot_type = FewShotType.POT
    with pytest.raises(ValueError):
        result = get_fewshot_examples(benchmark, fewshot_type)
