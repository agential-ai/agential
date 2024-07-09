"""Unit tests for few-shot factory."""

import pytest

from agential.fewshots.hotpotqa import HOTPOTQA_FEWSHOT_EXAMPLES_COT
from agential.base.constants import FewShotType
from agential.fewshots.fewshot_factory import FewShotFactory


def test_fewshot_factory_get_strategy() -> None:
    """Test FewShotFactory get_benchmark_fewshots."""
    # Test valid input.
    benchmark = "hotpotqa"
    fewshot_type = FewShotType.COT
    result = FewShotFactory.get_benchmark_fewshots(benchmark, fewshot_type)
    assert result == HOTPOTQA_FEWSHOT_EXAMPLES_COT

    # Test invalid benchmark.
    benchmark = "invalid_benchmark"
    fewshot_type = FewShotType.COT
    with pytest.raises(ValueError, match="Benchmark 'invalid_benchmark' not found."):
        FewShotFactory.get_benchmark_fewshots(benchmark, fewshot_type)

    # Test invalid few-shot type.
    benchmark = "hotpotqa"
    fewshot_type = "invalid_fewshot"
    with pytest.raises(
        ValueError,
        match="Few-shot type 'invalid_fewshot' not found for benchmark 'hotpotqa'.",
    ):
        FewShotFactory.get_benchmark_fewshots(benchmark, fewshot_type)

    # Test invalid few-shot type for the given benchmark.
    benchmark = "hotpotqa"
    fewshot_type = FewShotType.POT
    with pytest.raises(
        ValueError, match="Few-shot type 'pot' not found for benchmark 'hotpotqa'."
    ):
        FewShotFactory.get_benchmark_fewshots(benchmark, fewshot_type)
