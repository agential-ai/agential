"""Test Base (WebVoyager) Agent strategy class."""

import pytest

from typing import Any, Dict
from agential.agents.computer_use.webvoyager_baseline.output import WebVoyagerBaseOutput
from agential.agents.computer_use.webvoyager_baseline.strategies.general import WebVoyagerGeneralStrategy
from agential.agents.computer_use.webvoyager_baseline.strategies.base import WebVoyagerBaseStrategy

from agential.core.llm import BaseLLM, Response, MockLLM

def test__init__() -> None:
    """Test WebVoyagerGeneralStrategy initialization."""
    responses = [
        '```json\n{\n  "action_type": "CLICK",\n  "x": 300,\n  "y": 200\n}\n```'
    ]
    llm = MockLLM("gpt-4o", responses=responses)
    strategy = WebVoyagerGeneralStrategy(llm=llm, testing=True)
    assert strategy.testing == True
    assert isinstance(strategy.llm, BaseLLM)
    assert isinstance(strategy, WebVoyagerBaseStrategy)
    
@pytest.fixture
def sample_data_format_msg():
    """Test WebVoyagerGeneralStrategy format_msg Data Sample."""
    return {
        "it": 1,
        "init_msg": "Initial message.",
        "pdf_obs": "Sample PDF observation.",
        "warn_obs": "Sample warning observation.",
        "web_img_b64": "base64image==",
        "web_text": "Sample web text."
    }

def test_format_msg(sample_data_format_msg: Dict[str, Any]) -> None:
    """Test WebVoyagerGeneralStrategy format_msg."""
    responses = [
        '```json\n{\n  "action_type": "CLICK",\n  "x": 300,\n  "y": 200\n}\n```'
    ]
    llm = MockLLM("gpt-4o", responses=responses)
    strategy = WebVoyagerGeneralStrategy(llm=llm, testing=True)
    
    result = strategy.format_msg(
        sample_data_format_msg['it'],
        sample_data_format_msg["init_msg"],
        sample_data_format_msg["pdf_obs"],
        sample_data_format_msg["warn_obs"],
        sample_data_format_msg["web_img_b64"],
        sample_data_format_msg["web_text"]
    )

    assert result["role"] == "user"
    assert len(result["content"]) == 2
    assert result["content"][0]["type"] == "text"
    assert sample_data_format_msg["web_text"] in result["content"][0]["text"]
    assert result["content"][1]["type"] == "image_url"
    assert result["content"][1]["image_url"]["url"] == f"data:image/png;base64,{sample_data_format_msg['web_img_b64']}"

    sample_data_format_msg["it"] = 2
    sample_data_format_msg["pdf_obs"] = ""

    result = strategy.format_msg(
        it=sample_data_format_msg["it"],
        init_msg=sample_data_format_msg["init_msg"],
        pdf_obs=sample_data_format_msg["pdf_obs"],
        warn_obs=sample_data_format_msg["warn_obs"],
        web_img_b64=sample_data_format_msg["web_img_b64"],
        web_text=sample_data_format_msg["web_text"]
    )

    assert result["role"] == "user"
    assert "Observation:" in result["content"][0]["text"]
    assert sample_data_format_msg["warn_obs"] in result["content"][0]["text"]
    assert result["content"][1]["image_url"]["url"] == f"data:image/png;base64,{sample_data_format_msg['web_img_b64']}"

    sample_data_format_msg["it"] = 2

    result = strategy.format_msg(
        it=sample_data_format_msg["it"],
        init_msg=sample_data_format_msg["init_msg"],
        pdf_obs=sample_data_format_msg["pdf_obs"],
        warn_obs=sample_data_format_msg["warn_obs"],
        web_img_b64=sample_data_format_msg["web_img_b64"],
        web_text=sample_data_format_msg["web_text"]
    )

    assert result["role"] == "user"
    assert "Observation:" in result["content"][0]["text"]
    assert sample_data_format_msg["pdf_obs"] in result["content"][0]["text"]
    assert result["content"][1]["image_url"]["url"] == f"data:image/png;base64,{sample_data_format_msg['web_img_b64']}"
    
@pytest.fixture
def sample_data_format_msg_text_only() -> None:
    """Test WebVoyagerGeneralStrategy format_msg_text_only Data Sample."""
    return {
        "it": 1,
        "init_msg": "Initial message.",
        "pdf_obs": "Sample PDF observation.",
        "warn_obs": "Sample warning observation.",
        "ac_tree": "Accessibility tree data."
    }

def test_format_msg_text_only(sample_data_format_msg_text_only: Dict[str, Any]) -> None:
    """Test WebVoyagerGeneralStrategy format_msg_text_only."""
    responses = [
        '```json\n{\n  "action_type": "CLICK",\n  "x": 300,\n  "y": 200\n}\n```'
    ]
    llm = MockLLM("gpt-4o", responses=responses)
    strategy = WebVoyagerGeneralStrategy(llm=llm, testing=True)

    result = strategy.format_msg_text_only(
        it=sample_data_format_msg_text_only["it"],
        init_msg=sample_data_format_msg_text_only["init_msg"],
        pdf_obs=sample_data_format_msg_text_only["pdf_obs"],
        warn_obs=sample_data_format_msg_text_only["warn_obs"],
        ac_tree=sample_data_format_msg_text_only["ac_tree"]
    )
    
    assert result["role"] == "user"
    assert result["content"] == sample_data_format_msg_text_only["init_msg"] + "\n" + sample_data_format_msg_text_only["ac_tree"]

    sample_data_format_msg_text_only["it"] = 2
    sample_data_format_msg_text_only["pdf_obs"] = ""  # No PDF observation, just warning observation
    result = strategy.format_msg_text_only(
        it=sample_data_format_msg_text_only["it"],
        init_msg=sample_data_format_msg_text_only["init_msg"],
        pdf_obs=sample_data_format_msg_text_only["pdf_obs"],
        warn_obs=sample_data_format_msg_text_only["warn_obs"],
        ac_tree=sample_data_format_msg_text_only["ac_tree"]
    )
    
    assert result["role"] == "user"
    assert result["content"] == f"Observation:{sample_data_format_msg_text_only['warn_obs']} please analyze the accessibility tree and give the Thought and Action.\n{sample_data_format_msg_text_only['ac_tree']}"

    sample_data_format_msg_text_only["it"] = 2
    result = strategy.format_msg_text_only(
        it=sample_data_format_msg_text_only["it"],
        init_msg=sample_data_format_msg_text_only["init_msg"],
        pdf_obs=sample_data_format_msg_text_only["pdf_obs"],
        warn_obs=sample_data_format_msg_text_only["warn_obs"],
        ac_tree=sample_data_format_msg_text_only["ac_tree"]
    )
    
    assert result["role"] == "user"
    assert result["content"] == f"Observation:{sample_data_format_msg_text_only['warn_obs']} please analyze the accessibility tree and give the Thought and Action.\n{sample_data_format_msg_text_only['ac_tree']}"
    
@pytest.fixture
def sample_data_generate_thought() -> None:
    """Test WebVoyagerGeneralStrategy generate_thought Data Sample."""
    return {
        "messages": [
            {"role": "user", "content": "Tell me about the accessibility tree."},
            {"role": "assistant", "content": "The accessibility tree represents the DOM structure for screen readers."}
        ],
        "seed": 42,
        "max_tokens": 1000,
        "timeout": 30
    }

def test_generate_thought(sample_data_generate_thought: Dict[str, Any]) -> None:
    """Test WebVoyagerGeneralStrategy generate_thought."""

    responses = [
        '```json\n{\n  "action_type": "CLICK",\n  "x": 300,\n  "y": 200\n}\n```'
    ]

    llm = MockLLM("gpt-4o", responses=responses)
    strategy = WebVoyagerGeneralStrategy(llm=llm, testing=True)

    result = strategy.generate_thought(
        messages=sample_data_generate_thought["messages"],
        seed=sample_data_generate_thought["seed"],
        max_tokens=sample_data_generate_thought["max_tokens"],
        timeout=sample_data_generate_thought["timeout"]
    )
    
    # Check if the response is as expected
    assert isinstance(result, Response)  # Assuming the response should be a Response object
    assert result.output_text == responses[0]  # Adjust based on actual response structure


@pytest.fixture
def sample_task_and_obs():
    """Test WebVoyagerGeneralStrategy generate Data Sample."""
    return {
        "task": {"ques": "What is the main task?", "web": "https://www.example.com"},
        "obs": {
            "pdf_obs": "Some PDF observation",
            "warn_obs": "Some warning",
            "encoded_image_som": "image_data",
            "web_eles_text": "Element text",
            "ac_tree": "Accessibility tree data"
        }
    }

def test_generate(sample_task_and_obs: Dict[str, Any]) -> None:
    """Test WebVoyagerGeneralStrategy generate."""
    responses = [
        '```Thought: The user likely wants to interact with the interface at the specified coordinates. Action: CLICK. Observation: No additional observation provided.```'
    ]

    llm = MockLLM("gpt-4o", responses=responses)
    strategy = WebVoyagerGeneralStrategy(llm=llm, testing=True)

    # Define input arguments for the generate method
    system_prompt = "System prompt content"
    system_prompt_text_only = "System prompt content for text-only"
    seed = 42
    max_attached_imgs = 2
    temperature = 0.7
    text_only = False
    task = sample_task_and_obs["task"]
    obs = sample_task_and_obs["obs"]

    # Call the method
    result = strategy.generate(
        system_prompt=system_prompt,
        system_prompt_text_only=system_prompt_text_only,
        seed=seed,
        max_attached_imgs=max_attached_imgs,
        temperature=temperature,
        text_only=text_only,
        task=task,
        obs=obs
    )

    # Assertions to verify the output
    assert isinstance(result, WebVoyagerBaseOutput)
    assert result.answer == responses[0]
    assert result.total_prompt_tokens == 10
    assert result.total_completion_tokens == 20
    assert result.total_tokens == 30
    assert result.total_cost == 0.00022500000000000002
    assert "actions" in result.additional_info
    assert "thoughts" in result.additional_info
    assert "observations" in result.additional_info
    assert "messages" in result.additional_info


def test_reset() -> None:
    """Test WebVoyagerGeneralStrategy reset."""
    # Create a mock object with the reset method
    responses = [
        '```Thought: The user likely wants to interact with the interface at the specified coordinates. Action: CLICK. Observation: No additional observation provided.```'
    ]

    llm = MockLLM("gpt-4o", responses=responses)
    strategy = WebVoyagerGeneralStrategy(llm=llm, testing=True)
    
    # Call the reset method
    result = strategy.reset()

    # Assertions
    assert result is None
