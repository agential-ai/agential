"""General strategy for the OSWorldBaseline Agent."""

import time

from typing import Any, Dict, List, Tuple

from agential.agents.computer_use.osworld_baseline.functional import (
    encode_image,
    linearize_accessibility_tree,
    parse_actions_from_string,
    parse_code_from_som_string,
    parse_code_from_string,
    tag_screenshot,
    trim_accessibility_tree,
)
from agential.agents.computer_use.osworld_baseline.output import OSWorldBaseOutput
from agential.agents.computer_use.osworld_baseline.strategies.base import (
    OSWorldBaseStrategy,
)
from agential.core.llm import BaseLLM, Response

pure_text_settings = ["a11y_tree"]


class OSWorldBaseGeneralStrategy(OSWorldBaseStrategy):
    """A strategy class for the OS World Baseline Agent.

    This class defines methods for generating actions, thoughts, and observations
    in an agent-based environment.

    Attributes:
        messages (List): A list of messages exchanged during the agent's operation.
    """

    def __init__(self, llm: BaseLLM, testing: bool = False) -> None:
        """Initializes the OSWorldBaseGeneralStrategy.

        Args:
            llm (BaseLLM): The language model instance.
            testing (bool): If True, the agent operates in testing mode. Defaults to False.
        """
        super().__init__(llm=llm, testing=testing)
        self.messages: List = []

    def generate(
        self,
        platform: str,
        max_tokens: int,
        top_p: float,
        temperature: float,
        action_space: str,
        observation_type: str,
        max_trajectory_length: int,
        a11y_tree_max_tokens: int,
        observations: List[Any],
        actions: List[Dict[str, Any]],
        thoughts: List[str],
        _system_message: str,
        instruction: str,
        obs: Dict[str, Any],
    ) -> OSWorldBaseOutput:
        """Generates responses, actions, and updated agent states.

        Args:
            platform (str): The platform on which the agent operates.
            model (BaseLLM): The language model used for generating thoughts and actions.
            max_tokens (int): The maximum number of tokens for the model's response.
            top_p (float): Sampling hyperparameter for nucleus sampling.
            temperature (float): Sampling temperature for generating responses.
            action_space (str): The defined action space for the agent.
            observation_type (str): Type of observations the agent should generate.
            max_trajectory_length (int): Maximum length of the trajectory.
            a11y_tree_max_tokens (int): Maximum tokens allowed for accessibility tree observations.
            observations (List[Any]): A list of past observations.
            actions (List[Dict[str, Any]]): A list of past actions.
            thoughts (List[str]): A list of past thoughts.
            _system_message (str): System message for guiding the agent.
            instruction (str): Instructions for the agent.
            obs (Dict): Current observation context.

        Returns:
            OSWorldBaseOutput: The output of the generation process.
        """
        start_time = time.time()

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

        response = self.generate_thought(
            {
                "model": self.llm,
                "messages": self.messages,
                "max_tokens": max_tokens,
                "top_p": top_p,
                "temperature": temperature,
            }
        )

        try:
            action, actions_lists = self.generate_action(
                action_space,
                observation_type,
                actions_list,
                response.output_text,
                masks,
            )
            thoughts_list.append(response.output_text)
        except ValueError as e:
            action, actions_lists = [], []
            thoughts_list.append("")

        end_time = time.time()
        total_time = end_time - start_time

        return OSWorldBaseOutput(
            answer=response.output_text,
            total_prompt_tokens=response.prompt_tokens,
            total_completion_tokens=response.completion_tokens,
            total_tokens=response.total_tokens,
            total_prompt_cost=response.prompt_cost,
            total_completion_cost=response.completion_cost,
            total_cost=response.total_cost,
            total_prompt_time=response.prompt_time,
            total_time=total_time if not self.testing else 0.5,
            additional_info={
                "response": response.output_text,
                "actions": action,
                "actions_list": actions_lists,
                "thoughts_list": thoughts_list,
                "observations_list": observations_list,
                "messages": self.messages,
            },
        )

    def generate_observation(
        self,
        _platform: str,
        observation_type: str,
        max_trajectory_length: int,
        a11y_tree_max_tokens: int,
        observations: List[Any],
        actions: List[Dict[str, Any]],
        thoughts: List[str],
        _system_message: str,
        instruction: str,
        obs: Dict[str, Any],
    ) -> Tuple[List[int], List[Any], List[Dict[str, Any]], List[str]]:
        """Generate observations and prepare the input for the next step of the task.

        This method processes observations, actions, and thoughts, constructs messages for the system, and updates
        the internal state of observations based on the provided `observation_type`.

        Args:
            _platform (str): The platform for which the accessibility tree is linearized.
            observation_type (str): The type of observation to process. Can be one of:
                - "screenshot_a11y_tree": Includes both screenshots and accessibility trees.
                - "som": Screenshot with tagged nodes.
                - "screenshot": Only the screenshot.
                - "a11y_tree": Only the accessibility tree.
            max_trajectory_length (int): Maximum number of trajectory steps to keep in memory.
            a11y_tree_max_tokens (int): Maximum number of tokens allowed for the accessibility tree.
            observations (List[Any]): A list of prior observations.
            actions (List[Dict[str, Any]]): A list of prior actions corresponding to the observations.
            thoughts (List[str]): A list of thoughts corresponding to the observations and actions.
            _system_message (str): The base system message provided for the task.
            instruction (str): Specific instructions for the task.
            obs (Dict): The latest observation, including "screenshot" and/or "accessibility_tree".

        Returns:
            Tuple[List[int], List[Any], List[Dict[str, Any]], List[str]]: A tuple containing:
                - masks (List[int]): Visual masks for annotated screenshots (if applicable).
                - thoughts (List[str]): Updated list of thoughts.
                - actions (List[Dict[str, Any]]): Updated list of actions.
                - observations (List[Any]): Updated list of observations.

        Raises:
            ValueError: If an invalid `observation_type` is provided.
        """
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

        return masks, thoughts, actions, observations

    def generate_thought(
        self,
        payload: Dict[str, Any],
    ) -> Response:
        """Generates a thought response using the specified model and input payload.

        Args:
            payload (Dict): A dictionary containing the input parameters for the model, including:
                - "messages" (list): The input messages for the model.
                - "max_tokens" (int): The maximum number of tokens for the response.
                - "temperature" (float): The sampling temperature for response generation.
                - "top_p" (float): The nucleus sampling parameter.
            model (str): The model used to generate the response.

        Returns:
            Response: The generated output text from the model.
        """
        response = self.llm(
            payload["messages"],
            max_tokens=payload["max_tokens"],
            temperature=payload["temperature"],
            top_p=payload["top_p"],
        )

        return response

    def generate_action(
        self,
        action_space: str,
        observation_type: str,
        actions_list: List[Any],
        response: str,
        masks: List[int],
    ) -> Tuple[List[str], List[Any]]:
        """Generates actions based on the given action space, observation type, and response.

        Args:
            action_space (str): The type of action space being used (e.g., "computer_13", "pyautogui").
            observation_type (str): The type of observation data (e.g., "screenshot", "a11y_tree", "som").
            actions_list (List[Any]): A list to store the generated actions.
            response (str): The response from which actions will be parsed.
            masks (List[int]): Optional masks used for parsing in specific cases.

        Returns:
            Tuple[List[str], List[Any]]: A tuple containing the parsed actions and the updated actions list.

        Raises:
            ValueError: If an invalid action space or observation type is provided.
        """
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

        return [], actions_list

    def reset(
        self,
        actions: List[Dict[str, Any]],
        thought: List[str],
        observations: List[Any],
    ) -> Tuple[List[str], List[Dict[str, Any]], List[Any]]:
        """Resets the actions, thoughts, and observations to an empty state.

        Args:
            actions (List[str]): A list of actions to be cleared.
            thought (List[str]): A list of thoughts to be cleared.
            observations (List[str]): A list of observations to be cleared.

        Returns:
            Tuple[List[str], List[Dict[str, Any]], List[Any]]: A tuple containing the cleared thought, actions, and observations lists.
        """
        thought.clear()
        actions.clear()
        observations.clear()
        self.messages.clear()

        return thought, actions, observations
