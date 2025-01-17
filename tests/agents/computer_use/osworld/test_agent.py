"""Unit tests for the OSWorld Baseline Agent."""

import pytest

from agential.agents.computer_use.osworld_baseline.agent import OSWorldBaseline
from agential.agents.computer_use.osworld_baseline.functional import encode_image
from agential.agents.computer_use.osworld_baseline.output import OSWorldBaseOutput
from agential.agents.computer_use.osworld_baseline.prompts import (
    SYS_PROMPT_IN_A11Y_OUT_ACTION,
    SYS_PROMPT_IN_A11Y_OUT_CODE,
    SYS_PROMPT_IN_BOTH_OUT_ACTION,
    SYS_PROMPT_IN_BOTH_OUT_CODE,
    SYS_PROMPT_IN_SCREENSHOT_OUT_ACTION,
    SYS_PROMPT_IN_SCREENSHOT_OUT_CODE,
    SYS_PROMPT_IN_SOM_OUT_TAG,
)
from agential.agents.computer_use.osworld_baseline.strategies.general import (
    OSWorldBaseGeneralStrategy,
)
from agential.core.llm import BaseLLM, MockLLM


def test_init() -> None:
    """Test OSWorldBaseline initialization."""
    responses = """
            ```json
            {
            "action_type": "CLICK",
            "x": 300,
            "y": 200
            }
            ```
            """

    strategy = OSWorldBaseline(
        llm=MockLLM(model="gpt-4o", responses=[responses]),
        observation_type="screenshot",
    )

    assert strategy.platform == "ubuntu"
    assert isinstance(strategy.llm, BaseLLM)
    assert strategy.observation_type == "screenshot"
    assert strategy.action_space == "computer_13"


def test_get_prompts() -> None:
    """Test OSWorldBaseline get_promts."""
    responses = """
        ```json
        {
        "action_type": "CLICK",
        "x": 300,
        "y": 200
        }
        ```
        """

    # Test 1: Valid `screenshot` observation type and `pyautogui` action space
    strategy = OSWorldBaseline(
        llm=MockLLM(model="gpt-4o", responses=[responses]),
        observation_type="screenshot",
        action_space="pyautogui",
    )

    assert strategy.get_prompts()["prompt"] == SYS_PROMPT_IN_SCREENSHOT_OUT_CODE

    # Test 2: Valid `a11y_tree` observation type and `computer_13` action space
    strategy = OSWorldBaseline(
        llm=MockLLM(model="gpt-4o", responses=[responses]),
        observation_type="a11y_tree",
        action_space="computer_13",
    )

    assert strategy.get_prompts()["prompt"] == SYS_PROMPT_IN_A11Y_OUT_ACTION

    # Test 3: Valid `a11y_tree` observation type and `pyautogui` action space
    strategy = OSWorldBaseline(
        llm=MockLLM(model="gpt-4o", responses=[responses]),
        observation_type="a11y_tree",
        action_space="pyautogui",
    )

    assert strategy.get_prompts()["prompt"] == SYS_PROMPT_IN_A11Y_OUT_CODE

    # Test 4: Valid `screenshot_a11y_tree` observation type and `computer_13` action space
    strategy = OSWorldBaseline(
        llm=MockLLM(model="gpt-4o", responses=[responses]),
        observation_type="screenshot_a11y_tree",
        action_space="computer_13",
    )

    assert strategy.get_prompts()["prompt"] == SYS_PROMPT_IN_BOTH_OUT_ACTION

    # Test 5: Valid `screenshot_a11y_tree` observation type and `pyautogui` action space
    strategy = OSWorldBaseline(
        llm=MockLLM(model="gpt-4o", responses=[responses]),
        observation_type="screenshot_a11y_tree",
        action_space="pyautogui",
    )

    assert strategy.get_prompts()["prompt"] == SYS_PROMPT_IN_BOTH_OUT_CODE

    # Test 6: Valid `som` observation type and `computer_13` action space
    strategy = OSWorldBaseline(
        llm=MockLLM(model="gpt-4o", responses=[responses]),
        observation_type="som",
        action_space="computer_13",
    )

    with pytest.raises(ValueError, match="Invalid action space: computer_13"):
        strategy.get_prompts()

    # Test 7: Valid `som` observation type and `pyautogui` action space
    strategy = OSWorldBaseline(
        llm=MockLLM(model="gpt-4o", responses=[responses]),
        observation_type="som",
        action_space="pyautogui",
    )

    assert strategy.get_prompts()["prompt"] == SYS_PROMPT_IN_SOM_OUT_TAG

    # Test 8: Valid `screenshot` observation type and invalid action space
    strategy = OSWorldBaseline(
        llm=MockLLM(model="gpt-4o", responses=[responses]),
        observation_type="screenshot",
        action_space="blah",
    )

    with pytest.raises(ValueError, match="Invalid action space: blah"):
        strategy.get_prompts()

    # Test 9: Valid `a11y_tree` observation type and invalid action space
    strategy = OSWorldBaseline(
        llm=MockLLM(model="gpt-4o", responses=[responses]),
        observation_type="a11y_tree",
        action_space="blah",
    )

    with pytest.raises(ValueError, match="Invalid action space: blah"):
        strategy.get_prompts()

    # Test 10: Valid `som` observation type and invalid action space
    strategy = OSWorldBaseline(
        llm=MockLLM(model="gpt-4o", responses=[responses]),
        observation_type="som",
        action_space="blah",
    )

    with pytest.raises(ValueError, match="Invalid action space: blah"):
        strategy.get_prompts()

    # Test 11: Valid `screenshot_a11y_tree` observation type and invalid action space
    strategy = OSWorldBaseline(
        llm=MockLLM(model="gpt-4o", responses=[responses]),
        observation_type="screenshot_a11y_tree",
        action_space="blah",
    )

    with pytest.raises(ValueError, match="Invalid action space: blah"):
        strategy.get_prompts()

    # Test 12: Invalid observation type and action space
    strategy = OSWorldBaseline(
        llm=MockLLM(model="gpt-4o", responses=[responses]),
        observation_type="blah",
        action_space="blah",
    )

    with pytest.raises(ValueError, match="Invalid experiment type: blah"):
        strategy.get_prompts()


def test_get_strategy() -> None:
    """Test OSWorldBaseline get_strategy."""
    # Test 1: Valid Benchmark
    responses = """
            ```json
            {
            "action_type": "CLICK",
            "x": 300,
            "y": 200
            }
            ```
            """

    llm = MockLLM(model="gpt-4o", responses=[responses])
    strategy = OSWorldBaseline(
        llm=llm,
        observation_type="screenshot",
    )

    assert isinstance(
        strategy.get_strategy(benchmark="osworld", llm=llm, testing=True),
        OSWorldBaseGeneralStrategy,
    )

    # Test 2: Invalid Benchmark
    strategy = OSWorldBaseline(
        llm=MockLLM(model="gpt-4o", responses=[responses]),
        observation_type="screenshot",
    )

    with pytest.raises(ValueError, match="Unsupported benchmark: blah for agent ReAct"):
        strategy.get_strategy(benchmark="blah", llm=llm, testing=True)


def test_generate(osworld_screenshot_path: str) -> None:
    """Test OSWorldBaseline generate."""
    observation_type = "screenshot"
    _system_message = SYS_PROMPT_IN_SCREENSHOT_OUT_ACTION
    instruction = "Please help me to find the nearest restaurant."
    obs = {"screenshot": open(osworld_screenshot_path, "rb").read()}

    base64_image = encode_image(obs["screenshot"])
    system_message = (
        _system_message
        + "\nYou are asked to complete the following task: {}".format(instruction)
    )

    message = [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": system_message},
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "Given the screenshot as below. What's the next step that you will do to help with the task?"
                        if observation_type == "screenshot"
                        else "Given the screenshot and info from accessibility tree as below:\n{}\nWhat's the next step that you will do to help with the task?".format(
                            None
                        )
                    ),
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{base64_image}",
                        "detail": "high",
                    },
                },
            ],
        },
    ]

    action = [{"action_type": "CLICK", "x": 300, "y": 200}]

    responses = """
            ```json
            {
            "action_type": "CLICK",
            "x": 300,
            "y": 200
            }
            ```
            """

    strategy = OSWorldBaseline(
        llm=MockLLM(model="gpt-4o", responses=[responses]),
        observation_type=observation_type,
    )

    osworld_base_output: OSWorldBaseOutput = strategy.generate(
        instruction=instruction, obs=obs
    )

    assert responses == osworld_base_output.additional_info["response"]
    assert action == osworld_base_output.additional_info["actions"]
    assert message == osworld_base_output.additional_info["messages"]


def test_get_fewshots():
    """Test OSWorldBaseline get_fewshots."""
    responses = """
            ```json
            {
            "action_type": "CLICK",
            "x": 300,
            "y": 200
            }
            ```
            """

    strategy = OSWorldBaseline(
        llm=MockLLM(model="gpt-4o", responses=[responses]),
        observation_type="screenshot",
    )
    result = strategy.get_fewshots()

    assert result["benchmark"] == ""
    assert result["fewshot_type"] == ""
