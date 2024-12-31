"""Unit tests for the OSWorld Baseline general strategy."""

import pytest
from litellm import completion, cost_per_token

from agential.agents.OSWorldBaseline.strategies.general import OSWorldBaselineAgentGeneralStrategy
from agential.agents.OSWorldBaseline.strategies.base import OSWorldBaselineAgentBaseStrategy

from agential.agents.OSWorldBaseline.functional import (
    encode_image,
)

from agential.agents.OSWorldBaseline.prompts import (
    SYS_PROMPT_IN_SCREENSHOT_OUT_CODE, 
    SYS_PROMPT_IN_SCREENSHOT_OUT_ACTION,
    SYS_PROMPT_IN_A11Y_OUT_CODE, 
    SYS_PROMPT_IN_A11Y_OUT_ACTION,
    SYS_PROMPT_IN_BOTH_OUT_CODE, 
    SYS_PROMPT_IN_BOTH_OUT_ACTION,
    SYS_PROMPT_IN_SOM_OUT_TAG
)

def test_init() -> None:
    """Test ReActGeneralStrategy initialization."""
    strategy = OSWorldBaselineAgentGeneralStrategy(testing=True)
    assert strategy.testing == True
    assert isinstance(strategy, OSWorldBaselineAgentBaseStrategy)

def test_generate_thought() -> None:
    """Tests OSWorldBaselineAgentGeneralStrategy generate_thought."""
    payload = {
        "model": "blah-blah",
        "messages": "messages",
        "max_tokens": 1500,
        "top_p": 0.9,
        "temperature": 0
    }
    with pytest.raises(ValueError, match="Invalid model: .*"):
        strategy = OSWorldBaselineAgentGeneralStrategy()
        strategy.generate_thought(
            payload,
            "blah-blah",
            "screenshot"
        )

def test_generate_observation(osworld_screenshot_path: str) -> None:
    """Tests OSWorldBaselineAgentGeneralStrategy generate_observation."""
    _platform = "ubuntu"
    observation_type = "screenshot"
    max_trajectory_length = 3
    a11y_tree_max_tokens = 10000
    observations = []
    actions = [] 
    thoughts = []
    _system_message = SYS_PROMPT_IN_SCREENSHOT_OUT_ACTION
    instruction = "Please help me to find the nearest restaurant."
    obs = {"screenshot": open(osworld_screenshot_path, 'rb').read()}

    base64_image = encode_image(obs["screenshot"])
    system_message = _system_message + "\nYou are asked to complete the following task: {}".format(instruction)

    observation = [
        {
            "screenshot": base64_image,
            "accessibility_tree": None
        }
    ]
    message = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": system_message
                },
            ]
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Given the screenshot as below. What's the next step that you will do to help with the task?"
                    if observation_type == "screenshot"
                    else "Given the screenshot and info from accessibility tree as below:\n{}\nWhat's the next step that you will do to help with the task?".format(None)
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{base64_image}",
                        "detail": "high"
                    }
                }
            ]
        }
    ]

    strategy = OSWorldBaselineAgentGeneralStrategy()
    masks, thoughts, actions, observations = strategy.generate_observation(
        _platform,
        observation_type,
        max_trajectory_length,
        a11y_tree_max_tokens,
        observations,
        actions,
        thoughts,
        _system_message,
        instruction,
        obs
    )

    assert masks == []
    assert thoughts == []
    assert actions == []
    assert observations == observation
    assert strategy.messages == message

def test_generate_action() -> None:
    """Tests OSWorldBaselineAgentGeneralStrategy generate_action."""
    response = """
    ```json
    {
    "action_type": "CLICK",
    "x": 1000,
    "y": 400
    }
    ```
    """
    action_space = "computer_13"
    observation_type = "screenshot"
    actions_list = []
    masks = None

    action = [{'action_type': 'CLICK', 'x': 1000, 'y': 400}]

    strategy = OSWorldBaselineAgentGeneralStrategy()

    actions, actions_list = strategy.generate_action(
        action_space,
        observation_type,
        actions_list,
        response,
        masks
    )

    assert actions == action
    assert actions_list == [action]

def test_generate(osworld_screenshot_path: str) -> None:
    """Tests OSWorldBaselineAgentGeneralStrategy generate."""
    _platform = "ubuntu"
    observation_type = "screenshot"
    _system_message = SYS_PROMPT_IN_SCREENSHOT_OUT_ACTION
    instruction = "Please help me to find the nearest restaurant."
    obs = {"screenshot": open(osworld_screenshot_path, 'rb').read()}

    base64_image = encode_image(obs["screenshot"])
    system_message = _system_message + "\nYou are asked to complete the following task: {}".format(instruction)

    observation = [
        {
            "screenshot": base64_image,
            "accessibility_tree": None
        }
    ]
    message = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": system_message
                },
            ]
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Given the screenshot as below. What's the next step that you will do to help with the task?"
                    if observation_type == "screenshot"
                    else "Given the screenshot and info from accessibility tree as below:\n{}\nWhat's the next step that you will do to help with the task?".format(None)
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{base64_image}",
                        "detail": "high"
                    }
                }
            ]
        }
    ]

    action = [{'action_type': 'CLICK', 'x': 300, 'y': 200}]

    responses = """
            ```json
            {
            "action_type": "CLICK",
            "x": 300,
            "y": 200
            }
            ```
            """

    strategy = OSWorldBaselineAgentGeneralStrategy(testing=True)

    response, actions, actions_list, thoughts_list, observations_list, messages = strategy.generate(
        platform = _platform,
        model = "gpt-4o",
        max_tokens = 1500,
        top_p = 0.9,
        temperature = 0,
        action_space = "computer_13",
        observation_type = "screenshot",
        max_trajectory_length = 3,
        a11y_tree_max_tokens = 10000,
        observations = [],
        actions = [],
        thoughts = [],
        _system_message = SYS_PROMPT_IN_SCREENSHOT_OUT_ACTION,
        instruction = "Please help me to find the nearest restaurant.",
        obs = obs
    )

    assert actions == action
    assert responses == response
    assert actions_list == [action]
    assert thoughts_list == [response]
    assert observations_list == observation
    assert messages == message

def test_reset(osworld_screenshot_path: str) -> None:
    """Tests OSWorldBaselineAgentGeneralStrategy reset."""
    observation_type = "screenshot"
    _system_message = SYS_PROMPT_IN_SCREENSHOT_OUT_ACTION
    instruction = "Please help me to find the nearest restaurant."
    obs = {"screenshot": open(osworld_screenshot_path, 'rb').read()}


    base64_image = encode_image(obs["screenshot"])
    system_message = _system_message + "\nYou are asked to complete the following task: {}".format(instruction)
    
    observation = [
        {
            "screenshot": base64_image,
            "accessibility_tree": None
        }
    ]
    message = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": system_message
                },
            ]
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Given the screenshot as below. What's the next step that you will do to help with the task?"
                    if observation_type == "screenshot"
                    else "Given the screenshot and info from accessibility tree as below:\n{}\nWhat's the next step that you will do to help with the task?".format(None)
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{base64_image}",
                        "detail": "high"
                    }
                }
            ]
        }
    ]

    action = [{'action_type': 'CLICK', 'x': 300, 'y': 200}]

    thoughts = ["""
            ```json
            {
            "action_type": "CLICK",
            "x": 300,
            "y": 200
            }
            ```
            """
    ]

    strategy = OSWorldBaselineAgentGeneralStrategy(testing=True)

    strategy.messages = message

    thoughts, actions, observations = strategy.reset(
        action,
        thoughts,
        observation
    )

    assert thoughts == []
    assert actions == []
    assert observations == []
    assert strategy.messages == []
    