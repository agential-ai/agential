"""Unit tests for prompt manager logic."""

import pytest

from agential.fewshots.hotpotqa import (
    HOTPOTQA_FEWSHOT_EXAMPLES_COT,
)
from agential.fewshots.manager import FewShotType, get_benchmark_fewshots


def test_get_benchmark_fewshots() -> None:
    """Test get_benchmark_fewshots."""
    # Test valid input.
    benchmark = "hotpotqa"
    fewshot_type = FewShotType.COT
    result = get_benchmark_fewshots(benchmark, fewshot_type)
    assert result == HOTPOTQA_FEWSHOT_EXAMPLES_COT

    # Test invalid benchmark.
    benchmark = "invalid_benchmark"
    fewshot_type = FewShotType.COT
    with pytest.raises(ValueError):
        result = get_benchmark_fewshots(benchmark, fewshot_type)

    # Test invalid few-shot type.
    benchmark = "hotpotqa"
    fewshot_type = "invalid_fewshot"
    with pytest.raises(ValueError):
        result = get_benchmark_fewshots(benchmark, fewshot_type)

    # Test invalid few-shot type for the given benchmark.
    benchmark = "hotpotqa"
    fewshot_type = FewShotType.POT
    with pytest.raises(ValueError):
        result = get_benchmark_fewshots(benchmark, fewshot_type)
