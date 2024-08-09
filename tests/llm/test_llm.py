"""Unit tests for simple LLM wrapper for LiteLLM's completion function."""

from agential.llm.llm import LLM, MockLLM


def test_llm_init() -> None:
    """Test LLM initialization."""
    llm = LLM("gpt-3.5-turbo")
    assert llm.model == "gpt-3.5-turbo"


def test_llm_call() -> None:
    """Test LLM call."""
    llm = LLM("gpt-3.5-turbo")

    response = llm("Test prompt", mock_response="Test response")

    assert response.choices[0]["message"]["content"] == "Test response"


def test_mock_llm_init() -> None:
    """Test MockLLM initialization."""
    mock_llm = MockLLM("mock-model", ["Response 1", "Response 2"])
    assert mock_llm.model == "mock-model"
    assert mock_llm.responses == ["Response 1", "Response 2"]
    assert mock_llm.current_index == 0


def test_mock_llm_call() -> None:
    """Test MockLLM call."""
    mock_llm = MockLLM("gpt-3.5-turbo", ["Response 1", "Response 2"])

    response1 = mock_llm("Prompt 1")
    assert response1.choices[0]["message"]["content"] == "Response 1"
    assert mock_llm.current_index == 1

    response2 = mock_llm("Prompt 2")
    assert response2.choices[0]["message"]["content"] == "Response 2"
    assert mock_llm.current_index == 0

    response3 = mock_llm("Prompt 3")
    assert response3.choices[0]["message"]["content"] == "Response 1"
    assert mock_llm.current_index == 1

    # Test call with kwargs.
    response4 = mock_llm("Prompt 4", temperature=0.7, max_tokens=100)
    assert response4.choices[0]["message"]["content"] == "Response 2"
