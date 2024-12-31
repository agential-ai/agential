"""Unit tests for the OSWorld Baseline Agent Helper Functions."""

import base64
import os

from io import BytesIO

import pytest

from PIL import Image

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


def load_accessibility_tree(file_path):
    """Helper function to load accessibility tree."""
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()


def test_encode_image() -> None:
    """Test encode_image function."""
    # Simulate image binary content
    dummy_image_content = b"test_image_content"

    # Expected base64-encoded string (calculated beforehand)
    expected_encoded_string = base64.b64encode(dummy_image_content).decode("utf-8")

    # Call the function
    encoded_result = encode_image(dummy_image_content)

    # Assert the result matches the expected output
    assert (
        encoded_result == expected_encoded_string
    ), f"Expected {expected_encoded_string} but got {encoded_result}"


def test_encoded_img_to_pil_img() -> None:
    """Test encoded_img_to_pil_img function."""
    # Create a sample image for testing
    image = Image.new("RGB", (100, 100), color="red")
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)

    # Create a base64 string from the image
    image_data = buffer.getvalue()
    base64_str = base64.b64encode(image_data).decode("utf-8")
    data_str = f"data:image/png;base64,{base64_str}"

    # Decode the base64 string
    encoded_image = encoded_img_to_pil_img(data_str)

    # Assertions
    assert encoded_image.size == (
        100,
        100,
    ), f"Expected size (100, 100), got {encoded_image.size}"
    assert encoded_image.mode == "RGB", f"Expected mode RGB, got {encoded_image.mode}"


def test_save_to_tmp_img_file() -> None:
    # Create a sample image
    """Test save_to_tmp_img_file function."""
    image = Image.new("RGB", (100, 100), color="red")
    buffer = BytesIO()
    image.save(buffer, format="PNG")

    # Encode the image to Base64
    image_data = buffer.getvalue()
    base64_str = base64.b64encode(image_data).decode("utf-8")
    data_str = f"data:image/png;base64,{base64_str}"

    # Save to image file
    saved_image_path = save_to_tmp_img_file(data_str)

    # Assertions
    assert os.path.exists(saved_image_path), "Image file was not saved."
    saved_image = Image.open(saved_image_path)
    assert saved_image.size == (100, 100), "Image size does not match."
    assert saved_image.format == "PNG", "Image format is not PNG."


def test_linearize_accessibility_tree() -> None:
    """Test linearize_accessibility_tree function."""
    platform = "ubuntu"
    accessibility_tree = """
    <root>
        <button name="Button1" class="button-class" description="First Button"/>
        <input name="Input1" class="input-class" description="Input Field"/>
    </root>
    """
    expected_output = (
        "tag\tname\ttext\tclass\tdescription\tposition (top-left x&y)\tsize (w&h)"
    )
    # Call the function with the mocked filter_nodes
    result = linearize_accessibility_tree(accessibility_tree, platform)

    # Assert the result matches the expected output
    assert result == expected_output


def test_tag_screenshot(osworld_screenshot_path: str, osworld_access_tree: str) -> None:
    """Test tag_screenshot function."""
    # screenshot = open(osworld_screenshot_path, 'rb').read()
    screenshot = Image.new("RGB", (400, 400), color="white")
    img_byte_arr = BytesIO()
    screenshot.save(img_byte_arr, format="PNG")
    img_byte_arr.seek(0)
    screenshot_bytes = img_byte_arr.getvalue()
    accessibility_tree = load_accessibility_tree(
        osworld_access_tree
    )  # Load from the file

    marks, drew_nodes, tagged_screenshot, element_list = tag_screenshot(
        screenshot_bytes, accessibility_tree
    )
    print(type(screenshot_bytes))

    # Assert that the expected results are returned
    assert marks == [
        [358, 115, 531, 24],
        [201, 151, 910, 616],
        [201, 151, 910, 616],
        [22, 46, 1280, 1024],
        [282, 175, 166, 24],
        [388, 223, 91, 44],
        [388, 229, 91, 32],
        [201, 311, 402, 214],
        [225, 349, 209, 20],
        [225, 349, 142, 20],
        [376, 351, 58, 16],
        [225, 369, 262, 20],
        [225, 371, 39, 16],
        [263, 371, 4, 16],
        [273, 369, 16, 17],
        [299, 371, 104, 16],
        [225, 389, 262, 20],
        [225, 391, 155, 16],
        [0, 33, 70, 64],
        [0, 101, 70, 64],
        [0, 169, 70, 64],
        [0, 237, 70, 64],
        [0, 305, 70, 64],
        [0, 373, 70, 64],
    ]
    assert isinstance(tagged_screenshot, bytes)
    assert len(element_list) == 901


def test_parse_actions_from_string() -> None:
    """Test parse_actions_from_string function."""
    # Test special value
    assert parse_actions_from_string("WAIT") == ["WAIT"]
    assert parse_actions_from_string("DONE") == ["DONE"]
    assert parse_actions_from_string("FAIL") == ["FAIL"]

    # Test json action block
    input_string = '```json\n{"action": "click", "x": 100, "y": 200}\n```'
    expected_output = [{"action": "click", "x": 100, "y": 200}]
    assert parse_actions_from_string(input_string) == expected_output

    # Test non-json action block
    input_string = '```\n{"action": "click", "x": 100, "y": 200}\n```'
    expected_output = [{"action": "click", "x": 100, "y": 200}]
    assert parse_actions_from_string(input_string) == expected_output

    # Test valid json string
    input_string = '{"action": "click", "x": 100, "y": 200}'
    expected_output = [{"action": "click", "x": 100, "y": 200}]
    assert parse_actions_from_string(input_string) == expected_output

    # Test invalid json
    input_string = '{"action": "click", "x": 100, "y":}'
    with pytest.raises(ValueError):
        parse_actions_from_string(input_string)

    # Test failed json parsing
    input_string = '```json\n{"action": "click", "x": 100, "y":}\n```'
    assert (
        parse_actions_from_string(input_string)
        == "Failed to parse JSON: Expecting value: line 1 column 35 (char 34)"
    )


def test_parse_code_from_string() -> None:
    """Test parse_code_from_string function."""
    # Test single command
    input_string = "WAIT"
    result = parse_code_from_string(input_string)
    assert result == ["WAIT"]

    # Test code with command
    input_string = "Here is some text; ```python\ncode snippet here\nWAIT```"
    result = parse_code_from_string(input_string)
    assert result == ["code snippet here", "WAIT"]

    # Test code with command at the end
    input_string = "```python\ncode snippet here\nWAIT```"
    result = parse_code_from_string(input_string)
    assert result == ["code snippet here", "WAIT"]

    # Test code without command
    input_string = "```python\ncode snippet here```"
    result = parse_code_from_string(input_string)
    assert result == ["code snippet here"]

    # Test multiple code snippets
    input_string = "```python\ncode 1```; ```python\ncode 2```"
    result = parse_code_from_string(input_string)
    assert result == ["code 1", "code 2"]

    # Test code with multiline command
    input_string = "```python\ncode snippet here\nWAIT```"
    result = parse_code_from_string(input_string)
    assert result == ["code snippet here", "WAIT"]

    # Test empty input
    input_string = ""
    result = parse_code_from_string(input_string)
    assert result == []

    # Test no matching code
    input_string = "Here is some text without code or commands"
    result = parse_code_from_string(input_string)
    assert result == []

    # Test commands in code
    input_string = "```python\ncode snippet here\nWAIT```; DONE"
    result = parse_code_from_string(input_string)
    assert result == ["code snippet here", "WAIT"]


def test_get_parse_code_from_som_string() -> None:
    """Test get_parse_code_from_som_string function."""
    # Test input
    input_string = "```python\ncode snippet here\nWAIT```"
    masks = [
        (100, 100, 50, 50),  # Example mask with x, y, width, height
    ]

    # Expected output after adding tag variables to action lines
    expected_actions = ["tag_1=(125, 125)\ncode snippet here", "WAIT"]

    # Call the function
    result = parse_code_from_som_string(input_string, masks)

    assert result == expected_actions


def test_trim_accessibility_tree() -> None:
    """Test trim_accessibility_tree function."""
    input_string = (
        "tag\tname\ttext\tclass\tdescription\tposition (top-left x&y)\tsize (w&h)"
    )
    max_tokens = 20  # Set a token limit that should not be exceeded

    trimmed_string = trim_accessibility_tree(input_string, max_tokens)

    # The output should be the same since the input string doesn't exceed the max tokens
    assert trimmed_string == input_string

    input_string = 'button\tButton1\t"First Button"\tbutton-class\tFirst Button\n' * 10
    max_tokens = 10  # Set a token limit that the string will exceed

    trimmed_string = trim_accessibility_tree(input_string, max_tokens)

    # The string should be truncated and end with "[...]"
    assert trimmed_string.endswith("[...]\n")
    assert len(trimmed_string) < len(input_string)
