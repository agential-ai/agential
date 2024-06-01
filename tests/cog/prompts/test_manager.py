"""Unit tests for prompt manager logic."""

import pytest

from agential.cog.prompts.benchmark.hotpotqa import (
    HOTPOTQA_FEWSHOT_EXAMPLES_COT,
)
from agential.cog.prompts.manager import FewShotType, get_fewshot_examples


def test_get_fewshot_examples() -> None:
    """Test get_fewshot_examples."""
    # Test valid input.
    mode = {"qa": "hotpotqa"}
    fewshot_type = FewShotType.COT
    result = get_fewshot_examples(mode, fewshot_type)
    assert result == HOTPOTQA_FEWSHOT_EXAMPLES_COT

    # Test invalid benchmark type.
    mode = {"invalid_type": "hotpotqa"}
    fewshot_type = FewShotType.COT
    with pytest.raises(ValueError):
        result = get_fewshot_examples(mode, fewshot_type)

    # Test invalid benchmark name.
    mode = {"qa": "invalid_benchmark"}
    fewshot_type = FewShotType.COT
    with pytest.raises(ValueError):
        result = get_fewshot_examples(mode, fewshot_type)

    # Test invalid few-shot type.
    mode = {"qa": "hotpotqa"}
    fewshot_type = "invalid_fewshot"
    with pytest.raises(ValueError):
        result = get_fewshot_examples(mode, fewshot_type)

    # Test invalid few-shot type for the given benchmark.
    mode = {"qa": "hotpotqa"}
    fewshot_type = FewShotType.POT
    with pytest.raises(ValueError):
        result = get_fewshot_examples(mode, fewshot_type)
