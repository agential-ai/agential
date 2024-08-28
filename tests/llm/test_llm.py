"""Unit tests for simple LLM wrapper for LiteLLM's completion function."""

from agential.llm.llm import LLM, MockLLM, Response


def test_llm_init() -> None:
    """Test LLM initialization."""
    llm = LLM("gpt-3.5-turbo")
    assert llm.model == "gpt-3.5-turbo"


def test_llm_call() -> None:
    """Test LLM call."""
    llm = LLM("gpt-3.5-turbo")

    response = llm("Test prompt", mock_response="Test response")

    assert response.input_text == "Test prompt"
    assert response.output_text == "Test response"
    assert response.prompt_tokens == 10
    assert response.completion_tokens == 20
    assert response.total_tokens == 30
    assert response.prompt_cost == 1.5e-05
    assert response.completion_cost == 3.9999999999999996e-05
    assert response.total_cost == 5.4999999999999995e-05


def test_mock_llm_init() -> None:
    """Test MockLLM initialization."""
    mock_llm = MockLLM("mock-model", ["Response 1", "Response 2"])
    assert mock_llm.model == "mock-model"
    assert mock_llm.responses == ["Response 1", "Response 2"]
    assert mock_llm.current_index == 0


def test_mock_llm_call() -> None:
    """Test MockLLM call."""
    mock_llm = MockLLM("gpt-3.5-turbo", ["Response 1", "Response 2"])

    gt_response = Response(input_text='', output_text='Response 1', prompt_tokens=10, completion_tokens=20, total_tokens=30, prompt_cost=1.5e-05, completion_cost=3.9999999999999996e-05, total_cost=5.4999999999999995e-05, prompt_time=0.5)

    response1 = mock_llm("Prompt 1")
    assert mock_llm.current_index == 1
    assert response1 == gt_response

    gt_response = Response(input_text='', output_text='Response 2', prompt_tokens=10, completion_tokens=20, total_tokens=30, prompt_cost=1.5e-05, completion_cost=3.9999999999999996e-05, total_cost=5.4999999999999995e-05, prompt_time=0.5)

    response2 = mock_llm("Prompt 2")
    assert mock_llm.current_index == 0
    assert response2 == gt_response

    gt_response = Response(input_text='', output_text='Response 1', prompt_tokens=10, completion_tokens=20, total_tokens=30, prompt_cost=1.5e-05, completion_cost=3.9999999999999996e-05, total_cost=5.4999999999999995e-05, prompt_time=0.5)

    response3 = mock_llm("Prompt 3")
    assert mock_llm.current_index == 1
    assert response3 == gt_response

    # Test call with kwargs.
    gt_response = Response(input_text='', output_text='Response 2', prompt_tokens=10, completion_tokens=20, total_tokens=30, prompt_cost=1.5e-05, completion_cost=3.9999999999999996e-05, total_cost=5.4999999999999995e-05, prompt_time=0.5)

    response4 = mock_llm("Prompt 4", temperature=0.7, max_tokens=100)
    assert response4 == gt_response
