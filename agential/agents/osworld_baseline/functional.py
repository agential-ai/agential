"""Functional module for OSWorld."""

import base64
import json
import os
import re
import tempfile
import xml.etree.ElementTree as ET

from io import BytesIO
from typing import Any, List, Tuple

import tiktoken

from PIL import Image

attributes_ns_ubuntu = "https://accessibility.windows.example.org/ns/attributes"
attributes_ns_windows = "https://accessibility.windows.example.org/ns/attributes"
state_ns_ubuntu = "https://accessibility.ubuntu.example.org/ns/state"
state_ns_windows = "https://accessibility.windows.example.org/ns/state"
component_ns_ubuntu = "https://accessibility.ubuntu.example.org/ns/component"
component_ns_windows = "https://accessibility.windows.example.org/ns/component"
value_ns_ubuntu = "https://accessibility.ubuntu.example.org/ns/value"
value_ns_windows = "https://accessibility.windows.example.org/ns/value"
class_ns_windows = "https://accessibility.windows.example.org/ns/class"


from agential.agents.osworld_baseline.accessibility_tree_wrap.heuristic_retrieve import (
    draw_bounding_boxes,
    filter_nodes,
)


def encode_image(image_content: bytes) -> str:
    """Encodes image content into a base64 string.

    Args:
        image_content (bytes): The binary content of the image to be encoded.

    Returns:
        str: A base64-encoded string representation of the image.
    """
    return base64.b64encode(image_content).decode("utf-8")


def encoded_img_to_pil_img(data_str: str) -> Image.Image:
    """Decodes a base64-encoded image string into a PIL Image object.

    Args:
        data_str (str): The base64-encoded string representation of the image.

    Returns:
        Image.Image: A PIL Image object.
    """
    base64_str = data_str.replace("data:image/png;base64,", "")
    image_data = base64.b64decode(base64_str)
    image = Image.open(BytesIO(image_data))

    return image


def save_to_tmp_img_file(data_str: str) -> str:
    """Decodes a base64-encoded image string and saves it to a temporary file.

    Args:
        data_str (str): The base64-encoded string representation of the image.

    Returns:
        str: The file path to the saved temporary image file.
    """
    base64_str = data_str.replace("data:image/png;base64,", "")
    image_data = base64.b64decode(base64_str)
    image = Image.open(BytesIO(image_data))

    tmp_img_path = os.path.join(tempfile.mkdtemp(), "tmp_img.png")
    image.save(tmp_img_path)

    return tmp_img_path


def linearize_accessibility_tree(
    accessibility_tree: str, platform: str = "ubuntu"
) -> str:
    """Converts an accessibility tree XML into a linearized tabular format.

    Args:
        accessibility_tree (str): The XML representation of the accessibility tree.
        platform (str): The platform for which the tree is processed ("ubuntu" or "windows").

    Returns:
        str: A tab-delimited string representing the linearized accessibility tree.

    Raises:
        ValueError: If the platform is not "ubuntu" or "windows".
    """
    if platform == "ubuntu":
        _attributes_ns = attributes_ns_ubuntu
        _state_ns = state_ns_ubuntu
        _component_ns = component_ns_ubuntu
        _value_ns = value_ns_ubuntu
    elif platform == "windows":
        _attributes_ns = attributes_ns_windows
        _state_ns = state_ns_windows
        _component_ns = component_ns_windows
        _value_ns = value_ns_windows
    else:
        raise ValueError("Invalid platform, must be 'ubuntu' or 'windows'")

    filtered_nodes = filter_nodes(ET.fromstring(accessibility_tree), platform)
    linearized_accessibility_tree = [
        "tag\tname\ttext\tclass\tdescription\tposition (top-left x&y)\tsize (w&h)"
    ]

    # Linearize the accessibility tree nodes into a table format
    for node in filtered_nodes:
        if node.text:
            text = (
                node.text
                if '"' not in node.text
                else '"{:}"'.format(node.text.replace('"', '""'))
            )

        elif node.get("{{{:}}}class".format(class_ns_windows), "").endswith(
            "EditWrapper"
        ) and node.get("{{{:}}}value".format(_value_ns)):
            node_text = node.get("{{{:}}}value".format(_value_ns), "")
            text = (
                node_text
                if '"' not in node_text
                else '"{:}"'.format(node_text.replace('"', '""'))
            )
        else:
            text = '""'

        linearized_accessibility_tree.append(
            "{:}\t{:}\t{:}\t{:}\t{:}\t{:}\t{:}".format(
                node.tag,
                node.get("name", ""),
                text,
                (
                    node.get("{{{:}}}class".format(_attributes_ns), "")
                    if platform == "ubuntu"
                    else node.get("{{{:}}}class".format(class_ns_windows), "")
                ),
                node.get("{{{:}}}description".format(_attributes_ns), ""),
                node.get("{{{:}}}screencoord".format(_component_ns), ""),
                node.get("{{{:}}}size".format(_component_ns), ""),
            )
        )

    return "\n".join(linearized_accessibility_tree)


def tag_screenshot(
    screenshot: bytes, accessibility_tree: str, platform: str = "ubuntu"
) -> Tuple[List, List, bytes, str]:
    """Tags a screenshot with bounding boxes based on an accessibility tree.

    Args:
        screenshot (bytes): The binary content of the screenshot.
        accessibility_tree (str): The XML representation of the accessibility tree.
        platform (str): The platform for which the tagging is performed ("ubuntu" or "windows").

    Returns:
        Tuple[List, List, bytes, str]: A tuple containing marks, drawn nodes, the tagged screenshot, and a list of elements.
    """
    nodes = filter_nodes(
        ET.fromstring(accessibility_tree), platform=platform, check_image=True
    )
    # Make tag screenshot
    marks, drew_nodes, element_list, tagged_screenshot = draw_bounding_boxes(
        nodes, screenshot
    )

    return marks, drew_nodes, tagged_screenshot, element_list


def parse_actions_from_string(input_string: str) -> Any:
    """Parses actions from an input string, including JSON-formatted or text-based actions.

    Args:
        input_string (str): The string containing the actions.

    Returns:
        Any: A list of parsed actions or a failure message.

    Raises:
        ValueError: If the input string format is invalid.
    """
    if input_string.strip() in ["WAIT", "DONE", "FAIL"]:
        return [input_string.strip()]
    # Search for a JSON string within the input string
    actions = []
    matches = re.findall(r"```json\s+(.*?)\s+```", input_string, re.DOTALL)
    if matches:
        # Assuming there's only one match, parse the JSON string into a dictionary
        try:
            for match in matches:
                action_dict = json.loads(match)
                actions.append(action_dict)
            return actions
        except json.JSONDecodeError as e:
            return f"Failed to parse JSON: {e}"
    else:
        matches = re.findall(r"```\s+(.*?)\s+```", input_string, re.DOTALL)
        if matches:
            # Assuming there's only one match, parse the JSON string into a dictionary
            try:
                for match in matches:
                    action_dict = json.loads(match)
                    actions.append(action_dict)
                return actions
            except json.JSONDecodeError as e:
                return f"Failed to parse JSON: {e}"
        else:
            try:
                action_dict = json.loads(input_string)
                return [action_dict]
            except json.JSONDecodeError:
                raise ValueError("Invalid response format: " + input_string)


def parse_code_from_string(input_string: str) -> List:
    """Extracts and parses code snippets from a string enclosed in triple backticks.

    Args:
        input_string (str): The string containing code snippets.

    Returns:
        List: A list of parsed code snippets.
    """
    input_string = "\n".join(
        [line.strip() for line in input_string.split(";") if line.strip()]
    )
    if input_string.strip() in ["WAIT", "DONE", "FAIL"]:
        return [input_string.strip()]

    # This regular expression will match both ```code``` and ```python code```
    # and capture the `code` part. It uses a non-greedy match for the content inside.
    pattern = r"```(?:\w+\s+)?(.*?)```"
    # Find all non-overlapping matches in the string
    matches = re.findall(pattern, input_string, re.DOTALL)

    # The regex above captures the content inside the triple backticks.
    # The `re.DOTALL` flag allows the dot `.` to match newline characters as well,
    # so the code inside backticks can span multiple lines.

    # matches now contains all the captured code snippets

    codes = []
    print("hi")

    for match in matches:
        match = match.strip()
        commands = [
            "WAIT",
            "DONE",
            "FAIL",
        ]  # fixme: updates this part when we have more commands

        if match.split("\n")[-1] in commands:
            if len(match.split("\n")) > 1:
                codes.append("\n".join(match.split("\n")[:-1]))
            codes.append(match.split("\n")[-1])
        else:
            codes.append(match)

    return codes


def parse_code_from_som_string(input_string: str, masks: List) -> List:
    """Parses code actions from a string and augments them with positional tag variables based on masks.

    Args:
        input_string (str): The string containing code actions.
        masks (List): A list of masks defining positional data.

    Returns:
        List: A list of parsed and augmented code actions.
    """
    # parse the output string by masks
    tag_vars = ""
    for i, mask in enumerate(masks):
        x, y, w, h = mask
        tag_vars += (
            "tag_"
            + str(i + 1)
            + "="
            + "({}, {})".format(int(x + w // 2), int(y + h // 2))
        )
        tag_vars += "\n"

    actions = parse_code_from_string(input_string)

    for i, action in enumerate(actions):
        if action.strip() in ["WAIT", "DONE", "FAIL"]:
            pass
        else:
            action = tag_vars + action
            actions[i] = action

    return actions


def trim_accessibility_tree(linearized_accessibility_tree: str, max_tokens: int) -> str:
    """Trims a linearized accessibility tree string to fit within a specified token limit.

    Args:
        linearized_accessibility_tree (str): The linearized accessibility tree as a string.
        max_tokens (int): The maximum number of tokens allowed.

    Returns:
        str: The trimmed accessibility tree string.
    """
    enc = tiktoken.encoding_for_model("gpt-4")
    tokens = enc.encode(linearized_accessibility_tree)
    if len(tokens) > max_tokens:
        linearized_accessibility_tree = enc.decode(tokens[:max_tokens])
        linearized_accessibility_tree += "[...]\n"
    return linearized_accessibility_tree
