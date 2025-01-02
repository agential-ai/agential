"""Unit tests for the OSWorld Baseline general strategy."""

import pytest

from litellm import completion, cost_per_token

from agential.agents.osworld_baseline.functional import (
    encode_image,
    encoded_img_to_pil_img,
    linearize_accessibility_tree,
    parse_actions_from_string,
    parse_code_from_som_string,
    parse_code_from_string,
    save_to_tmp_img_file,
    tag_screenshot,
    trim_accessibility_tree,
)
from agential.agents.osworld_baseline.output import OSWorldBaseOutput
from agential.agents.osworld_baseline.prompts import (
    SYS_PROMPT_IN_A11Y_OUT_ACTION,
    SYS_PROMPT_IN_A11Y_OUT_CODE,
    SYS_PROMPT_IN_BOTH_OUT_ACTION,
    SYS_PROMPT_IN_BOTH_OUT_CODE,
    SYS_PROMPT_IN_SCREENSHOT_OUT_ACTION,
    SYS_PROMPT_IN_SCREENSHOT_OUT_CODE,
    SYS_PROMPT_IN_SOM_OUT_TAG,
)
from agential.agents.osworld_baseline.strategies.base import (
    OSWorldBaseStrategy,
)
from agential.agents.osworld_baseline.strategies.general import (
    OSWorldBaseGeneralStrategy,
)
from agential.core.llm import BaseLLM, MockLLM


def test_init() -> None:
    """Test ReActGeneralStrategy initialization."""
    strategy = OSWorldBaseGeneralStrategy(testing=True)
    assert strategy.testing == True
    assert isinstance(strategy, OSWorldBaseStrategy)


def test_generate_thought(osworld_screenshot_path: str) -> None:
    """Tests OSWorldBaseGeneralStrategy generate_thought."""
    _system_message = SYS_PROMPT_IN_SCREENSHOT_OUT_ACTION
    instruction = "Please help me to find the nearest restaurant."
    obs = {"screenshot": open(osworld_screenshot_path, "rb").read()}

    base64_image = encode_image(obs["screenshot"])
    system_message = (
        _system_message
        + "\nYou are asked to complete the following task: {}".format(instruction)
    )

    responses = [
        '```json\n{\n  "action_type": "CLICK",\n  "x": 300,\n  "y": 200\n}\n```'
    ]

    llm = MockLLM("gpt-4o", responses=responses)
    observation_type = "screenshot"

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

    payload = {
        "model": llm,
        "messages": message,
        "max_tokens": 1500,
        "top_p": 0.9,
        "temperature": 0,
    }

    strategy = OSWorldBaseGeneralStrategy()

    response = strategy.generate_thought(payload=payload, model=llm)

    assert response.output_text == responses[0]


def test_generate_observation(
    osworld_screenshot_path: str, osworld_access_tree: str
) -> None:
    """Tests OSWorldBaseGeneralStrategy generate_observation."""
    _platform = "ubuntu"
    observation_type = "screenshot"
    max_trajectory_length = 3
    a11y_tree_max_tokens = 10000
    observations = []
    actions = []
    thoughts = []
    _system_message = SYS_PROMPT_IN_SCREENSHOT_OUT_ACTION
    instruction = "Please help me to find the nearest restaurant."
    obs = {"screenshot": open(osworld_screenshot_path, "rb").read()}

    base64_image = encode_image(obs["screenshot"])
    system_message = (
        _system_message
        + "\nYou are asked to complete the following task: {}".format(instruction)
    )

    observation = [{"screenshot": base64_image, "accessibility_tree": None}]
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
    strategy = OSWorldBaseGeneralStrategy()

    # Test 1: Everything Valid
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
        obs,
    )

    assert masks == []
    assert thoughts == []
    assert actions == []
    assert observations == observation
    assert strategy.messages == message

    # Test 2: Invalid Observation
    action = [{"action_type": "CLICK", "x": 1000, "y": 400}]
    thought = ['```\n{\n  "action_type": "CLICK",\n  "x": 300,\n  "y": 200\n}\n```']

    with pytest.raises(ValueError, match="Invalid observation_type type: blah"):
        strategy.generate_observation(
            _platform=_platform,
            observation_type="blah",
            max_trajectory_length=max_trajectory_length,
            a11y_tree_max_tokens=a11y_tree_max_tokens,
            observations=observation,
            actions=action,
            thoughts=thought,
            _system_message=_system_message,
            instruction=instruction,
            obs=obs,
        )

    # Test 3: Valid `som` Observation Type
    with open(osworld_access_tree, "r", encoding="utf-8") as file:
        accessibility_tree = file.read()

    obs = {
        "screenshot": open(osworld_screenshot_path, "rb").read(),
        "accessibility_tree": accessibility_tree,
    }

    strategy = OSWorldBaseGeneralStrategy()

    masks, thoughts, actions, observations = strategy.generate_observation(
        _platform=_platform,
        observation_type="som",
        max_trajectory_length=max_trajectory_length,
        a11y_tree_max_tokens=a11y_tree_max_tokens,
        observations=observation,
        actions=action,
        thoughts=thought,
        _system_message=_system_message,
        instruction=instruction,
        obs=obs,
    )

    _screenshot = observation[0]["screenshot"]
    previous_thought = thoughts[-1]
    masks, drew_nodes, tagged_screenshot, linearized_accessibility_tree = (
        tag_screenshot(obs["screenshot"], obs["accessibility_tree"], _platform)
    )
    base64_image_test3 = encode_image(tagged_screenshot)
    if linearized_accessibility_tree:
        linearized_accessibility_tree_test3 = trim_accessibility_tree(
            linearized_accessibility_tree, a11y_tree_max_tokens
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
                    "text": "Given the tagged screenshot as below. What's the next step that you will do to help with the task?",
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{_screenshot}",
                        "detail": "high",
                    },
                },
            ],
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": (
                        previous_thought.strip()
                        if len(previous_thought) > 0
                        else "No valid action"
                    ),
                },
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Given the tagged screenshot and info from accessibility tree as below:\n{}\nWhat's the next step that you will do to help with the task?".format(
                        linearized_accessibility_tree_test3
                    ),
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{base64_image_test3}",
                        "detail": "high",
                    },
                },
            ],
        },
    ]
    observation_test_3 = [
        {"screenshot": base64_image, "accessibility_tree": None},
        {
            "screenshot": base64_image_test3,
            "accessibility_tree": linearized_accessibility_tree_test3,
        },
    ]

    mask = [
        [1324, 649, 64, 64],
        [1336, 717, 40, 17],
        [1191, 184, 223, 31],
        [1191, 260, 223, 31],
        [1191, 298, 223, 31],
        [1191, 329, 223, 31],
        [1191, 367, 223, 31],
        [1191, 398, 223, 31],
        [1191, 436, 223, 31],
        [1191, 474, 223, 31],
        [1191, 505, 223, 31],
        [207, 64, 28, 41],
        [443, 70, 28, 28],
        [479, 64, 28, 41],
        [206, 110, 34, 34],
        [278, 110, 34, 34],
        [326, 115, 24, 24],
        [358, 115, 531, 24],
        [923, 115, 24, 24],
        [955, 115, 24, 24],
        [1000, 110, 34, 34],
        [1036, 110, 34, 34],
        [1072, 110, 34, 34],
        [201, 151, 910, 616],
        [201, 151, 910, 616],
        [22, 46, 1280, 1024],
        [217, 163, 57, 52],
        [282, 175, 166, 24],
        [449, 164, 49, 52],
        [549, 175, 29, 24],
        [216, 223, 76, 44],
        [216, 229, 76, 32],
        [229, 236, 34, 18],
        [297, 223, 86, 44],
        [297, 229, 86, 32],
        [310, 236, 44, 18],
        [388, 223, 91, 44],
        [388, 229, 91, 32],
        [401, 236, 49, 18],
        [484, 223, 82, 44],
        [484, 229, 82, 32],
        [497, 236, 40, 18],
        [571, 223, 103, 44],
        [571, 229, 103, 32],
        [571, 229, 32, 32],
        [225, 277, 59, 24],
        [225, 278, 59, 22],
        [275, 269, 41, 40],
        [201, 311, 402, 214],
        [225, 327, 83, 20],
        [225, 327, 83, 20],
        [225, 349, 209, 20],
        [225, 349, 142, 20],
        [373, 351, 4, 16],
        [376, 351, 58, 16],
        [225, 369, 262, 20],
        [225, 371, 39, 16],
        [263, 371, 4, 16],
        [273, 369, 16, 17],
        [295, 371, 5, 16],
        [299, 371, 104, 16],
        [225, 389, 262, 20],
        [225, 391, 155, 16],
        [225, 409, 262, 20],
        [225, 411, 33, 16],
        [257, 411, 97, 16],
        [225, 437, 43, 20],
        [225, 439, 43, 16],
        [279, 437, 101, 20],
        [279, 439, 101, 16],
        [391, 437, 50, 20],
        [391, 439, 50, 16],
        [225, 461, 173, 48],
        [225, 468, 173, 34],
        [252, 476, 119, 18],
        [406, 461, 173, 48],
        [406, 468, 173, 34],
        [441, 476, 103, 18],
        [201, 526, 402, 214],
        [225, 542, 182, 20],
        [225, 542, 182, 20],
        [225, 564, 209, 20],
        [225, 564, 142, 20],
        [373, 566, 4, 16],
        [376, 566, 58, 16],
        [225, 584, 262, 20],
        [225, 586, 43, 16],
        [267, 586, 4, 16],
        [277, 584, 16, 17],
        [300, 586, 4, 16],
        [303, 586, 94, 16],
        [225, 604, 262, 20],
        [225, 606, 241, 16],
        [225, 624, 262, 20],
        [225, 626, 33, 16],
        [257, 626, 97, 16],
        [225, 652, 43, 20],
        [225, 654, 43, 16],
        [279, 652, 101, 20],
        [279, 654, 101, 16],
        [391, 652, 121, 20],
        [391, 654, 121, 16],
        [225, 676, 173, 48],
        [225, 683, 173, 34],
        [252, 691, 119, 18],
        [406, 676, 173, 48],
        [406, 683, 173, 34],
        [441, 691, 103, 18],
        [219, 738, 238, 24],
        [253, 742, 204, 16],
        [609, 435, 23, 48],
        [966, 165, 31, 31],
        [1011, 165, 70, 31],
        [1024, 172, 44, 16],
        [727, 633, 266, 101],
        [727, 633, 266, 101],
        [735, 641, 97, 72],
        [845, 641, 54, 36],
        [845, 641, 54, 20],
        [845, 642, 51, 18],
        [845, 661, 54, 16],
        [845, 662, 48, 14],
        [845, 685, 61, 41],
        [910, 640, 37, 37],
        [910, 690, 37, 37],
        [972, 635, 19, 99],
        [1062, 644, 29, 29],
        [1062, 672, 29, 29],
        [1062, 704, 29, 30],
        [631, 657, 75, 75],
        [657, 711, 33, 14],
        [657, 711, 33, 14],
        [631, 657, 75, 75],
        [612, 742, 109, 11],
        [612, 742, 109, 11],
        [735, 741, 61, 13],
        [810, 741, 29, 13],
        [853, 741, 33, 13],
        [612, 754, 106, 13],
        [995, 742, 95, 11],
        [995, 742, 20, 11],
        [995, 742, 20, 11],
        [1319, 0, 106, 27],
        [0, 33, 70, 64],
        [0, 101, 70, 64],
        [0, 169, 70, 64],
        [0, 237, 70, 64],
        [0, 305, 70, 64],
        [0, 373, 70, 64],
        [0, 441, 70, 64],
        [0, 509, 70, 64],
        [0, 577, 70, 64],
        [0, 645, 70, 64],
        [0, 713, 70, 64],
        [0, 697, 70, 70],
    ]

    assert masks == mask
    assert thoughts == thought
    assert actions == action
    assert observations == observation_test_3
    assert strategy.messages == message

    # Test 4: valid `a11y_tree` Observation Type
    obs = {
        "screenshot": open(osworld_screenshot_path, "rb").read(),
        "accessibility_tree": accessibility_tree,
    }

    observation = [{"screenshot": base64_image, "accessibility_tree": None}]
    action = [{"action_type": "CLICK", "x": 1000, "y": 400}]
    thought = ['```\n{\n  "action_type": "CLICK",\n  "x": 300,\n  "y": 200\n}\n```']

    strategy = OSWorldBaseGeneralStrategy()

    masks, thoughts, actions, observations = strategy.generate_observation(
        _platform=_platform,
        observation_type="a11y_tree",
        max_trajectory_length=max_trajectory_length,
        a11y_tree_max_tokens=a11y_tree_max_tokens,
        observations=observation,
        actions=action,
        thoughts=thought,
        _system_message=_system_message,
        instruction=instruction,
        obs=obs,
    )

    _screenshot = observation[0]["screenshot"]
    previous_thought = thoughts[-1]
    linearized_accessibility_tree = linearize_accessibility_tree(
        accessibility_tree=obs["accessibility_tree"], platform=_platform
    )
    base64_image_test3 = encode_image(tagged_screenshot)
    if linearized_accessibility_tree:
        linearized_accessibility_tree_test3 = trim_accessibility_tree(
            linearized_accessibility_tree, a11y_tree_max_tokens
        )
    _linearized_accessibility_tree = observation[0]["accessibility_tree"]

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
                    "text": "Given the info from accessibility tree as below:\n{}\nWhat's the next step that you will do to help with the task?".format(
                        _linearized_accessibility_tree
                    ),
                }
            ],
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": (
                        previous_thought.strip()
                        if len(previous_thought) > 0
                        else "No valid action"
                    ),
                },
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Given the info from accessibility tree as below:\n{}\nWhat's the next step that you will do to help with the task?".format(
                        linearized_accessibility_tree
                    ),
                }
            ],
        },
    ]
    observation = [
        {"screenshot": base64_image, "accessibility_tree": None},
        {
            "screenshot": None,
            "accessibility_tree": linearized_accessibility_tree,
        },
    ]

    assert masks == []
    assert thoughts == thought
    assert actions == action
    assert observations == observation
    assert strategy.messages == message

    # Test 5: valid `screenshot_a11y_tree` Observation Type
    obs = {
        "screenshot": open(osworld_screenshot_path, "rb").read(),
        "accessibility_tree": accessibility_tree,
    }

    observation = [{"screenshot": base64_image, "accessibility_tree": None}]
    action = [{"action_type": "CLICK", "x": 1000, "y": 400}]
    thought = ['```\n{\n  "action_type": "CLICK",\n  "x": 300,\n  "y": 200\n}\n```']

    strategy = OSWorldBaseGeneralStrategy()

    masks, thoughts, actions, observations = strategy.generate_observation(
        _platform=_platform,
        observation_type="screenshot_a11y_tree",
        max_trajectory_length=max_trajectory_length,
        a11y_tree_max_tokens=a11y_tree_max_tokens,
        observations=observation,
        actions=action,
        thoughts=thought,
        _system_message=_system_message,
        instruction=instruction,
        obs=obs,
    )
    observation_type_test_5 = "screenshot_a11y_tree"
    _screenshot = observation[0]["screenshot"]
    previous_thought = thoughts[-1]
    linearized_accessibility_tree = (
        linearize_accessibility_tree(
            accessibility_tree=obs["accessibility_tree"], platform=_platform
        )
        if observation_type_test_5 == "screenshot_a11y_tree"
        else None
    )
    base64_image_test5 = encode_image(obs["screenshot"])
    if linearized_accessibility_tree:
        linearized_accessibility_tree = trim_accessibility_tree(
            linearized_accessibility_tree, a11y_tree_max_tokens
        )
    _linearized_accessibility_tree = observation[0]["accessibility_tree"]

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
                    "text": "Given the screenshot and info from accessibility tree as below:\n{}\nWhat's the next step that you will do to help with the task?".format(
                        _linearized_accessibility_tree
                    ),
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{_screenshot}",
                        "detail": "high",
                    },
                },
            ],
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": (
                        previous_thought.strip()
                        if len(previous_thought) > 0
                        else "No valid action"
                    ),
                },
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "Given the screenshot as below. What's the next step that you will do to help with the task?"
                        if observation_type_test_5 == "screenshot"
                        else "Given the screenshot and info from accessibility tree as below:\n{}\nWhat's the next step that you will do to help with the task?".format(
                            linearized_accessibility_tree
                        )
                    ),
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{base64_image_test5}",
                        "detail": "high",
                    },
                },
            ],
        },
    ]
    observation = [
        {"screenshot": base64_image, "accessibility_tree": None},
        {
            "screenshot": base64_image_test5,
            "accessibility_tree": linearized_accessibility_tree,
        },
    ]

    assert masks == []
    assert thoughts == thought
    assert actions == action
    assert observations == observation
    assert strategy.messages == message

    # Test 6: valid `screenshot` Observation Type
    obs = {
        "screenshot": open(osworld_screenshot_path, "rb").read(),
        "accessibility_tree": accessibility_tree,
    }

    observation = [{"screenshot": base64_image, "accessibility_tree": None}]
    action = [{"action_type": "CLICK", "x": 1000, "y": 400}]
    thought = ['```\n{\n  "action_type": "CLICK",\n  "x": 300,\n  "y": 200\n}\n```']

    strategy = OSWorldBaseGeneralStrategy()

    masks, thoughts, actions, observations = strategy.generate_observation(
        _platform=_platform,
        observation_type="screenshot",
        max_trajectory_length=max_trajectory_length,
        a11y_tree_max_tokens=a11y_tree_max_tokens,
        observations=observation,
        actions=action,
        thoughts=thought,
        _system_message=_system_message,
        instruction=instruction,
        obs=obs,
    )
    observation_type_test_6 = "screenshot"
    _screenshot = observation[0]["screenshot"]
    previous_thought = thoughts[-1]
    linearized_accessibility_tree = (
        linearize_accessibility_tree(
            accessibility_tree=obs["accessibility_tree"], platform=_platform
        )
        if observation_type_test_6 == "screenshot"
        else None
    )
    base64_image_test6 = encode_image(obs["screenshot"])
    if linearized_accessibility_tree:
        linearized_accessibility_tree = trim_accessibility_tree(
            linearized_accessibility_tree, a11y_tree_max_tokens
        )
    _linearized_accessibility_tree = observation[0]["accessibility_tree"]

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
                    "text": "Given the screenshot as below. What's the next step that you will do to help with the task?",
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{_screenshot}",
                        "detail": "high",
                    },
                },
            ],
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": (
                        previous_thought.strip()
                        if len(previous_thought) > 0
                        else "No valid action"
                    ),
                },
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
                            linearized_accessibility_tree
                        )
                    ),
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{base64_image_test6}",
                        "detail": "high",
                    },
                },
            ],
        },
    ]
    observation = [
        {"screenshot": base64_image, "accessibility_tree": None},
        {"screenshot": base64_image_test6, "accessibility_tree": None},
    ]

    assert masks == []
    assert thoughts == thought
    assert actions == action
    assert observations == observation
    assert strategy.messages == message

    # Test 7: Max Trajecotry is less than observation
    observation = [{"screenshot": base64_image, "accessibility_tree": None}]
    action = [{"action_type": "CLICK", "x": 1000, "y": 400}]
    thought = ['```\n{\n  "action_type": "CLICK",\n  "x": 300,\n  "y": 200\n}\n```']

    obs = {
        "screenshot": open(osworld_screenshot_path, "rb").read(),
        "accessibility_tree": accessibility_tree,
    }

    strategy = OSWorldBaseGeneralStrategy()
    masks, thoughts, actions, observations = strategy.generate_observation(
        _platform=_platform,
        observation_type="screenshot",
        max_trajectory_length=0,
        a11y_tree_max_tokens=a11y_tree_max_tokens,
        observations=observation,
        actions=action,
        thoughts=thought,
        _system_message=_system_message,
        instruction=instruction,
        obs=obs,
    )

    observation_type_test_7 = "screenshot"
    base64_image_test_7 = encode_image(obs["screenshot"])
    linearized_accessibility_tree = (
        linearize_accessibility_tree(
            accessibility_tree=obs["accessibility_tree"], platform=_platform
        )
        if observation_type_test_7 == "screenshot_a11y_tree"
        else None
    )

    if linearized_accessibility_tree:
        linearized_accessibility_tree = trim_accessibility_tree(
            linearized_accessibility_tree, a11y_tree_max_tokens
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
                            linearized_accessibility_tree
                        )
                    ),
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{base64_image_test_7}",
                        "detail": "high",
                    },
                },
            ],
        },
    ]
    observation = [
        {"screenshot": base64_image, "accessibility_tree": None},
        {
            "screenshot": base64_image_test_7,
            "accessibility_tree": linearized_accessibility_tree,
        },
    ]

    assert masks == []
    assert thoughts == thought
    assert actions == action
    assert observations == observation
    assert strategy.messages == message


def test_generate_action() -> None:
    """Tests OSWorldBaseGeneralStrategy generate_action."""
    response = """
    ```json
    {
    "action_type": "CLICK",
    "x": 1000,
    "y": 400
    }
    ```
    """

    # Test 1: Valid `computer_13` Action Space with `screenshot` Observation Type
    action_space = "computer_13"
    observation_type = "screenshot"
    actions_list = []
    masks = None

    action = [{"action_type": "CLICK", "x": 1000, "y": 400}]

    strategy = OSWorldBaseGeneralStrategy()

    actions, actions_list = strategy.generate_action(
        action_space, observation_type, actions_list, response, masks
    )

    assert actions == action
    assert actions_list == [action]

    # Test 2: Valid `pyautogui` Action Space with `som` Observation Type
    action_space = "pyautogui"
    observation_type = "som"
    actions_list = []
    masks = []

    value = """{
    "action_type": "CLICK",
    "x": 1000,
    "y": 400
    }"""
    action = [value]

    actions, actions_list = strategy.generate_action(
        action_space, observation_type, actions_list, response, masks
    )

    assert actions == action
    assert actions_list == [action]

    # Test 3: Invalid `computer_13` Action Space with `som` Observation Type (Expect ValueError)
    action_space = "computer_13"
    observation_type = "som"
    actions_list = []
    masks = []

    strategy = OSWorldBaseGeneralStrategy()

    with pytest.raises(ValueError, match="Invalid action space: computer_13"):
        strategy.generate_action(
            action_space, observation_type, actions_list, response, masks
        )

    # Test 4: Valid `computer_13` Action Space with `a11y_tree` Observation Type
    action_space = "computer_13"
    observation_type = "a11y_tree"
    actions_list = []
    masks = None

    action = [{"action_type": "CLICK", "x": 1000, "y": 400}]

    strategy = OSWorldBaseGeneralStrategy()

    actions, actions_list = strategy.generate_action(
        action_space, observation_type, actions_list, response, masks
    )

    assert actions == action
    assert actions_list == [action]

    # Test 5: Valid `pyautogui` Action Space with `screenshot` Observation Type
    action_space = "computer_13"
    observation_type = "a11y_tree"
    actions_list = []
    masks = None

    action = [{"action_type": "CLICK", "x": 1000, "y": 400}]

    strategy = OSWorldBaseGeneralStrategy()

    actions, actions_list = strategy.generate_action(
        action_space, observation_type, actions_list, response, masks
    )

    assert actions == action
    assert actions_list == [action]

    # Test 6: Invalid Action Space with `som` Observation Type
    action_space = "blah"
    observation_type = "som"
    actions_list = []
    masks = None

    strategy = OSWorldBaseGeneralStrategy()

    with pytest.raises(ValueError, match="Invalid action space: blah"):
        strategy.generate_action(
            action_space, observation_type, actions_list, response, masks
        )

    # Test 7: Invalid Observation Type
    action_space = "blah"
    observation_type = "blah"
    actions_list = []
    masks = None

    strategy = OSWorldBaseGeneralStrategy()

    actions, actions_list = strategy.generate_action(
        action_space, observation_type, actions_list, response, masks
    )

    assert actions == []
    assert actions_list == []

    # Test 8: Invalid Action Space with `screenshot` Observation Type
    action_space = "blah"
    observation_type = "screenshot"
    actions_list = []
    masks = None

    strategy = OSWorldBaseGeneralStrategy()

    with pytest.raises(ValueError, match="Invalid action space: blah"):
        strategy.generate_action(
            action_space, observation_type, actions_list, response, masks
        )

    # Test 9: Valid `som` pyautogui Space with `screenshot` Observation Type
    action_space = "pyautogui"
    observation_type = "screenshot"
    actions_list = []
    masks = None

    value = """{
    "action_type": "CLICK",
    "x": 1000,
    "y": 400
    }"""
    action = [value]
    strategy = OSWorldBaseGeneralStrategy()

    actions, actions_list = strategy.generate_action(
        action_space, observation_type, actions_list, response, masks
    )

    assert actions == action
    assert actions_list == [action]


def test_generate(osworld_screenshot_path: str) -> None:
    """Tests OSWorldBaseGeneralStrategy generate."""
    # Test 1: Valid
    _platform = "ubuntu"
    observation_type = "screenshot"
    _system_message = SYS_PROMPT_IN_SCREENSHOT_OUT_ACTION
    instruction = "Please help me to find the nearest restaurant."
    obs = {"screenshot": open(osworld_screenshot_path, "rb").read()}

    base64_image = encode_image(obs["screenshot"])
    system_message = (
        _system_message
        + "\nYou are asked to complete the following task: {}".format(instruction)
    )

    observation = [{"screenshot": base64_image, "accessibility_tree": None}]
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

    llm_model: BaseLLM = MockLLM("gpt-4o", responses=[responses])
    strategy = OSWorldBaseGeneralStrategy()

    osworldbaseoutput: OSWorldBaseOutput = strategy.generate(
        platform=_platform,
        model=llm_model,
        max_tokens=1500,
        top_p=0.9,
        temperature=0,
        action_space="computer_13",
        observation_type="screenshot",
        max_trajectory_length=3,
        a11y_tree_max_tokens=10000,
        observations=[],
        actions=[],
        thoughts=[],
        _system_message=SYS_PROMPT_IN_SCREENSHOT_OUT_ACTION,
        instruction="Please help me to find the nearest restaurant.",
        obs=obs,
    )

    assert osworldbaseoutput.additional_info["actions"] == action
    assert osworldbaseoutput.additional_info["response"] == responses
    assert osworldbaseoutput.additional_info["actions_list"] == [action]
    assert osworldbaseoutput.additional_info["thoughts_list"] == [responses]
    assert osworldbaseoutput.additional_info["observations_list"] == observation
    assert osworldbaseoutput.additional_info["messages"] == message

    # Test 2: Invalid action space
    strategy = OSWorldBaseGeneralStrategy()

    osworldbaseoutput: OSWorldBaseOutput = strategy.generate(
        platform=_platform,
        model=llm_model,
        max_tokens=1500,
        top_p=0.9,
        temperature=0,
        action_space="blah",
        observation_type="screenshot",
        max_trajectory_length=3,
        a11y_tree_max_tokens=10000,
        observations=[],
        actions=[],
        thoughts=[],
        _system_message=SYS_PROMPT_IN_SCREENSHOT_OUT_ACTION,
        instruction="Please help me to find the nearest restaurant.",
        obs=obs,
    )

    assert osworldbaseoutput.additional_info["actions"] == []
    assert osworldbaseoutput.additional_info["response"] == responses
    assert osworldbaseoutput.additional_info["actions_list"] == []
    assert osworldbaseoutput.additional_info["thoughts_list"] == [""]
    assert osworldbaseoutput.additional_info["observations_list"] == observation
    assert osworldbaseoutput.additional_info["messages"] == message


def test_reset(osworld_screenshot_path: str) -> None:
    """Tests OSWorldBaseGeneralStrategy reset."""
    observation_type = "screenshot"
    _system_message = SYS_PROMPT_IN_SCREENSHOT_OUT_ACTION
    instruction = "Please help me to find the nearest restaurant."
    obs = {"screenshot": open(osworld_screenshot_path, "rb").read()}

    base64_image = encode_image(obs["screenshot"])
    system_message = (
        _system_message
        + "\nYou are asked to complete the following task: {}".format(instruction)
    )

    observation = [{"screenshot": base64_image, "accessibility_tree": None}]
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

    thoughts = [
        """
            ```json
            {
            "action_type": "CLICK",
            "x": 300,
            "y": 200
            }
            ```
            """
    ]

    strategy = OSWorldBaseGeneralStrategy(testing=True)

    strategy.messages = message

    thoughts, actions, observations = strategy.reset(action, thoughts, observation)

    assert thoughts == []
    assert actions == []
    assert observations == []
    assert strategy.messages == []
