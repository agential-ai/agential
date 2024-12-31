import logging

from typing import Any, Dict, List, Tuple

from agential.agents.OSWorldBaseline.functional import (
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
from agential.agents.OSWorldBaseline.strategies.base import (
    OSWorldBaselineAgentBaseStrategy,
)
from agential.core.llm import LLM, BaseLLM

logger = logging.getLogger("desktopenv.agent")
pure_text_settings = ["a11y_tree"]


class OSWorldBaselineAgentGeneralStrategy(OSWorldBaselineAgentBaseStrategy):

    def __init__(self, testing: bool = False) -> None:
        super().__init__(testing=testing)
        self.messages: List = []

    def generate(
        self,
        platform: str,
        model: BaseLLM,
        max_tokens: int,
        top_p: float,
        temperature: float,
        action_space: str,
        observation_type: str,
        max_trajectory_length: int,
        a11y_tree_max_tokens: int,
        observations: List,
        actions: List,
        thoughts: List,
        _system_message: str,
        instruction: str,
        obs: Dict,
    ) -> Tuple[str, List, List, List, List, List]:

        masks, thoughts_list, actions_list, observations_list = (
            self.generate_observation(
                platform,
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
        )

        try:
            response = self.generate_thought(
                {
                    "model": model,
                    "messages": self.messages,
                    "max_tokens": max_tokens,
                    "top_p": top_p,
                    "temperature": temperature,
                },
                model,
            )
        except Exception as e:
            logger.error("Failed to call" + model + ", Error: " + str(e))
            response = ""

        logger.info("RESPONSE: %s", response)

        try:
            actions, actions_list = self.generate_action(
                action_space, observation_type, actions_list, response, masks
            )
            thoughts_list.append(response)
        except ValueError as e:
            print("Failed to parse action from response", e)
            actions, actions_list = [], []
            thoughts_list.append("")

        return (
            response,
            actions,
            actions_list,
            thoughts_list,
            observations_list,
            self.messages,
        )

    def generate_observation(
        self,
        _platform: str,
        observation_type: str,
        max_trajectory_length: int,
        a11y_tree_max_tokens: int,
        observations: List,
        actions: List,
        thoughts: List,
        _system_message: str,
        instruction: str,
        obs: Dict,
    ) -> Tuple[List, List, List, List]:

        system_message = (
            _system_message
            + "\nYou are asked to complete the following task: {}".format(instruction)
        )

        # Prepare the payload for the API call
        masks: List = []

        self.messages.append(
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": system_message},
                ],
            }
        )

        assert len(observations) == len(actions) and len(actions) == len(
            thoughts
        ), "The number of observations and actions should be the same."

        if len(observations) > max_trajectory_length:
            if max_trajectory_length == 0:
                _observations = []
                _actions = []
                _thoughts = []
            else:
                _observations = observations[-max_trajectory_length:]
                _actions = actions[-max_trajectory_length:]
                _thoughts = thoughts[-max_trajectory_length:]
        else:
            _observations = observations
            _actions = actions
            _thoughts = thoughts

        for previous_obs, previous_action, previous_thought in zip(
            _observations, _actions, _thoughts
        ):

            if observation_type == "screenshot_a11y_tree":
                _screenshot = previous_obs["screenshot"]
                _linearized_accessibility_tree = previous_obs["accessibility_tree"]

                self.messages.append(
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
                    }
                )
            elif observation_type in ["som"]:
                _screenshot = previous_obs["screenshot"]

                self.messages.append(
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
                    }
                )
            elif observation_type == "screenshot":
                _screenshot = previous_obs["screenshot"]

                self.messages.append(
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
                    }
                )
            elif observation_type == "a11y_tree":
                _linearized_accessibility_tree = previous_obs["accessibility_tree"]

                self.messages.append(
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
                    }
                )
            else:
                raise ValueError("Invalid observation_type type: " + observation_type)

            self.messages.append(
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
                }
            )

        if observation_type in ["screenshot", "screenshot_a11y_tree"]:
            base64_image = encode_image(obs["screenshot"])
            linearized_accessibility_tree = (
                linearize_accessibility_tree(
                    accessibility_tree=obs["accessibility_tree"], platform=_platform
                )
                if observation_type == "screenshot_a11y_tree"
                else None
            )
            logger.debug("LINEAR AT: %s", linearized_accessibility_tree)

            if linearized_accessibility_tree:
                linearized_accessibility_tree = trim_accessibility_tree(
                    linearized_accessibility_tree, a11y_tree_max_tokens
                )

            if observation_type == "screenshot_a11y_tree":
                observations.append(
                    {
                        "screenshot": base64_image,
                        "accessibility_tree": linearized_accessibility_tree,
                    }
                )
            else:
                observations.append(
                    {"screenshot": base64_image, "accessibility_tree": None}
                )

            self.messages.append(
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
                                "url": f"data:image/png;base64,{base64_image}",
                                "detail": "high",
                            },
                        },
                    ],
                }
            )
        elif observation_type == "a11y_tree":
            linearized_accessibility_tree = linearize_accessibility_tree(
                accessibility_tree=obs["accessibility_tree"], platform=_platform
            )
            logger.debug("LINEAR AT: %s", linearized_accessibility_tree)

            if linearized_accessibility_tree:
                linearized_accessibility_tree = trim_accessibility_tree(
                    linearized_accessibility_tree, a11y_tree_max_tokens
                )

            observations.append(
                {
                    "screenshot": None,
                    "accessibility_tree": linearized_accessibility_tree,
                }
            )

            self.messages.append(
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
                }
            )
        elif observation_type == "som":
            # Add som to the screenshot
            masks, drew_nodes, tagged_screenshot, linearized_accessibility_tree = (
                tag_screenshot(obs["screenshot"], obs["accessibility_tree"], _platform)
            )
            base64_image = encode_image(tagged_screenshot)
            logger.debug("LINEAR AT: %s", linearized_accessibility_tree)

            if linearized_accessibility_tree:
                linearized_accessibility_tree = trim_accessibility_tree(
                    linearized_accessibility_tree, a11y_tree_max_tokens
                )

            observations.append(
                {
                    "screenshot": base64_image,
                    "accessibility_tree": linearized_accessibility_tree,
                }
            )

            self.messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Given the tagged screenshot and info from accessibility tree as below:\n{}\nWhat's the next step that you will do to help with the task?".format(
                                linearized_accessibility_tree
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
                }
            )
        else:
            raise ValueError("Invalid observation_type type: " + observation_type)

        return masks, thoughts, actions, observations

    def generate_thought(
        self,
        payload: Dict,
        model: str,
    ) -> str:

        response = model(
            payload["messages"],
            max_tokens=payload["max_tokens"],
            temperature=payload["temperature"],
            top_p=payload["top_p"],
        )

        return response.output_text

    def generate_action(
        self,
        action_space: str,
        observation_type: str,
        actions_list: List,
        response: str,
        masks: List,
    ) -> Tuple[List, List]:

        if observation_type in ["screenshot", "a11y_tree", "screenshot_a11y_tree"]:
            # parse from the response
            if action_space == "computer_13":
                actions = parse_actions_from_string(response)
            elif action_space == "pyautogui":
                actions = parse_code_from_string(response)
            else:
                raise ValueError("Invalid action space: " + action_space)

            actions_list.append(actions)

            return actions, actions_list

        elif observation_type in ["som"]:
            # parse from the response
            if action_space == "computer_13":
                raise ValueError("Invalid action space: " + action_space)
            elif action_space == "pyautogui":
                actions = parse_code_from_som_string(response, masks)
            else:
                raise ValueError("Invalid action space: " + action_space)

            actions_list.append(actions)

            return actions, actions_list

        return actions, actions_list

    def reset(
        self,
        actions: List,
        thought: List,
        observations: List,
    ) -> Tuple[List, List, List]:
        thought.clear()
        actions.clear()
        observations.clear()
        self.messages.clear()

        return thought, actions, observations
