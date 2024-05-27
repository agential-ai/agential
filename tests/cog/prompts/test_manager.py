"""Unit tests for prompt manager logic."""


from agential.cog.prompts.manager import get_fewshot_examples, Benchmarks, FewShotType


def test_get_fewshot_examples() -> None:
    """Test get_fewshot_examples."""

    # Test valid input.
    mode = {"qa": "hotpotqa"}
    fewshot_type = FewShotType.COT
    result = get_fewshot_examples(mode, fewshot_type)
    assert result == "HOTPOTQA_FEWSHOT_EXAMPLES_COT", "Valid input test failed."

    # Test invalid benchmark type.
    mode = {"invalid_type": "hotpotqa"}
    fewshot_type = FewShotType.COT
    result = get_fewshot_examples(mode, fewshot_type)
    assert "Benchmark type 'invalid_type' not found." in result, "Invalid benchmark type test failed."

    # Test invalid benchmark name.
    mode = {"qa": "invalid_benchmark"}
    fewshot_type = FewShotType.COT
    result = get_fewshot_examples(mode, fewshot_type)
    assert "Benchmark 'invalid_benchmark' not found in benchmark type 'hotpotqa'." in result, "Invalid benchmark name test failed."

    # Test invalid few-shot type.
    mode = {"qa": "hotpotqa"}
    fewshot_type = "invalid_fewshot"
    result = get_fewshot_examples(mode, fewshot_type)
    assert "Few-shot type 'invalid_fewshot' not found for benchmark 'hotpotqa'." in result, "Invalid few-shot type test failed."

    # Test invalid few-shot type for the given benchmark.
    mode = {"qa": "hotpotqa"}
    fewshot_type = FewShotType.POT
    result = get_fewshot_examples(mode, fewshot_type)
    assert "Few-shot type 'pot' not found for benchmark 'hotpotqa'." in result, "Invalid few-shot type for benchmark test failed."
