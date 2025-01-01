"""Unit tests for the OSWorld Baseline Agent Heurisitic Retreive Functions."""

import xml.etree.ElementTree as ET

from io import BytesIO
from xml.etree.ElementTree import Element, ElementTree

import pytest

from PIL import Image

from agential.agents.OSWorldBaseline.accessibility_tree_wrap.heuristic_retrieve import (
    draw_bounding_boxes,
    filter_nodes,
    judge_node,
)

state_ns_ubuntu = "https://accessibility.ubuntu.example.org/ns/state"
state_ns_windows = "https://accessibility.windows.example.org/ns/state"
component_ns_ubuntu = "https://accessibility.ubuntu.example.org/ns/component"
component_ns_windows = "https://accessibility.windows.example.org/ns/component"
value_ns_ubuntu = "https://accessibility.ubuntu.example.org/ns/value"
value_ns_windows = "https://accessibility.windows.example.org/ns/value"
class_ns_windows = "https://accessibility.windows.example.org/ns/class"


def test_judge_node() -> None:
    """Test judge_node function."""
    # Create a sample valid XML node
    node = Element("button")
    node.set(f"{{{state_ns_ubuntu}}}showing", "true")
    node.set(f"{{{state_ns_ubuntu}}}visible", "true")
    node.set(f"{{{state_ns_ubuntu}}}enabled", "true")
    node.set(f"{{{component_ns_ubuntu}}}screencoord", "(10, 10)")
    node.set(f"{{{component_ns_ubuntu}}}size", "(50, 50)")
    node.set("name", "Submit")

    assert judge_node(node, platform="ubuntu") == True

    # Create a sample node
    node = Element("button")
    with pytest.raises(
        ValueError, match="Invalid platform, must be 'ubuntu' or 'windows'"
    ):
        judge_node(node, platform="macOS")

    # Create a node with the image property
    node = Element("image")
    node.set(f"{{{state_ns_ubuntu}}}showing", "true")
    node.set(f"{{{state_ns_ubuntu}}}visible", "true")
    node.set(f"{{{state_ns_ubuntu}}}enabled", "true")
    node.set(f"{{{component_ns_ubuntu}}}screencoord", "(10, 10)")
    node.set(f"{{{component_ns_ubuntu}}}size", "(50, 50)")
    node.set("image", "true")

    assert judge_node(node, platform="ubuntu", check_image=True) == True

    # Create a node missing required attributes
    node = Element("button")
    node.set(f"{{{state_ns_ubuntu}}}showing", "false")
    node.set(f"{{{state_ns_ubuntu}}}visible", "false")
    node.set(f"{{{state_ns_ubuntu}}}enabled", "false")

    assert judge_node(node, platform="ubuntu") == False

    # Create a node with invalid coordinates
    node = Element("button")
    node.set(f"{{{state_ns_ubuntu}}}showing", "true")
    node.set(f"{{{state_ns_ubuntu}}}visible", "true")
    node.set(f"{{{state_ns_ubuntu}}}enabled", "true")
    node.set(f"{{{component_ns_ubuntu}}}screencoord", "(-1, -1)")
    node.set(f"{{{component_ns_ubuntu}}}size", "(50, 50)")
    node.set("name", "Submit")

    assert judge_node(node, platform="ubuntu") == False

    # Create a sample valid XML node
    node = Element("button")
    node.set(f"{{{state_ns_ubuntu}}}showing", "true")
    node.set(f"{{{state_ns_ubuntu}}}visible", "true")
    node.set(f"{{{state_ns_ubuntu}}}enabled", "true")
    node.set(f"{{{component_ns_ubuntu}}}screencoord", "(10, 10)")
    node.set(f"{{{component_ns_ubuntu}}}size", "(50, 50)")
    node.set("name", "Submit")

    assert judge_node(node, platform="windows") == False


def test_filter_nodes() -> None:
    """Test filter_nodes function."""
    root = Element("root")

    # Create some example nodes
    button_node = Element("button")
    button_node.set(f"{{{state_ns_ubuntu}}}showing", "true")
    button_node.set(f"{{{state_ns_ubuntu}}}visible", "true")
    button_node.set(f"{{{state_ns_ubuntu}}}enabled", "true")
    button_node.set(f"{{{component_ns_ubuntu}}}screencoord", "(10, 10)")
    button_node.set(f"{{{component_ns_ubuntu}}}size", "(50, 50)")
    button_node.set("name", "Submit")

    image_node = Element("image")
    image_node.set(f"{{{state_ns_ubuntu}}}showing", "true")
    image_node.set(f"{{{state_ns_ubuntu}}}visible", "true")
    image_node.set(f"{{{state_ns_ubuntu}}}enabled", "true")
    image_node.set(f"{{{component_ns_ubuntu}}}screencoord", "(20, 20)")
    image_node.set(f"{{{component_ns_ubuntu}}}size", "(100, 100)")
    image_node.set("image", "true")

    # Add nodes to the root
    root.append(button_node)
    root.append(image_node)

    tree = ElementTree(root)
    root = tree.getroot()

    # Test with check_image = False for Ubuntu
    filtered_nodes = filter_nodes(root, platform="ubuntu", check_image=False)
    assert (
        len(filtered_nodes) == 1
    )  # Only the button should pass the filter without image check
    assert filtered_nodes[0].tag == "button"

    # Test with check_image = True for Ubuntu
    filtered_nodes = filter_nodes(root, platform="ubuntu", check_image=True)
    assert (
        len(filtered_nodes) == 2
    )  # Both button and image should pass the filter with image check
    assert {node.tag for node in filtered_nodes} == {"button", "image"}

    with pytest.raises(
        ValueError, match="Invalid platform, must be 'ubuntu' or 'windows'"
    ):
        filter_nodes(root, platform="invalid_platform", check_image=False)


def test_draw_bounding_boxes(
    osworld_screenshot_path: str, osworld_access_tree: str
) -> None:
    """Test draw_bounding_boxes function."""
    # Test 1: Ubuntu
    with open(osworld_access_tree, "r", encoding="utf-8") as file:
        accessibility_tree = file.read()

    screenshot = open(osworld_screenshot_path, "rb").read()

    nodes = filter_nodes(
        ET.fromstring(accessibility_tree), platform="ubuntu", check_image=True
    )
    # Make tag screenshot
    marks, drew_nodes, element_list, tagged_screenshot = draw_bounding_boxes(
        nodes, screenshot
    )

    assert len(marks) == 155  # 2 nodes, so 2 bounding boxes
    assert len(drew_nodes) == 155  # Both nodes should have bounding boxes
    assert "button" in element_list  # Check that button information is in the text info
    assert "image" in element_list  # Check that image information is in the text info
    assert isinstance(tagged_screenshot, bytes)

    # Test 2: Windows
    with open(osworld_access_tree, "r", encoding="utf-8") as file:
        accessibility_tree = file.read()

    screenshot = open(osworld_screenshot_path, "rb").read()

    nodes = filter_nodes(
        ET.fromstring(accessibility_tree), platform="windows", check_image=True
    )
    # Make tag screenshot
    marks, drew_nodes, element_list, tagged_screenshot = draw_bounding_boxes(
        nodes, screenshot
    )

    assert len(marks) == 0  # 2 nodes, so 2 bounding boxes
    assert len(drew_nodes) == 0  # Both nodes should have bounding boxes
    assert isinstance(tagged_screenshot, bytes)

    # Test 3: Ubuntu and down_sampling_ratio != 1.0
    down_sampling_ratio_test_3 = 0.5
    with open(osworld_access_tree, "r", encoding="utf-8") as file:
        accessibility_tree = file.read()

    screenshot = open(osworld_screenshot_path, "rb").read()

    nodes = filter_nodes(
        ET.fromstring(accessibility_tree), platform="ubuntu", check_image=True
    )
    # Make tag screenshot
    marks, drew_nodes, element_list, tagged_screenshot = draw_bounding_boxes(
        nodes, screenshot, down_sampling_ratio_test_3
    )

    assert len(marks) == 154  # 2 nodes, so 2 bounding boxes
    assert len(drew_nodes) == 154  # Both nodes should have bounding boxes
    assert "button" in element_list  # Check that button information is in the text info
    assert "image" in element_list  # Check that image information is in the text info
    assert isinstance(tagged_screenshot, bytes)

    # Test 4: Invalid Platform
    platform_test_4 = "blah"
    with pytest.raises(
        ValueError, match="Invalid platform, must be 'ubuntu' or 'windows'"
    ):
        draw_bounding_boxes(
            nodes, screenshot, down_sampling_ratio_test_3, platform_test_4
        )
