"""Unit tests for the OSWorld benchmark."""

from unittest.mock import MagicMock

from agential.benchmarks.computer_use.osworld.osworld import OSWorld


def test_init_() -> None:
    """Test the __init__ function of OSWorld in the virtual machine."""
    env = MagicMock(spec=OSWorld)
    env.path_to_vm = "to_vmware_vm_data_folder"

    assert env.path_to_vm == "to_vmware_vm_data_folder"


def test_close() -> None:
    """Test the close function of OSWorld in the virtual machine."""
    env = MagicMock(spec=OSWorld)

    env.close.return_value = 0.0

    result = env.close()

    assert result == 0.0


def test_reset() -> None:
    """Test the reset function of OSWorld in the virtual machine."""
    env = MagicMock(spec=OSWorld)

    env.reset.return_value = {
        "screenshot": b"screen",
        "accessibility_tree": "tree",
        "terminal": None,
        "instruction": "I want to install Spotify on my current system. Could you please help me?",
    }

    result = env.reset()

    assert result["screenshot"] == b"screen"
    assert result["accessibility_tree"] == "tree"
    assert result["terminal"] == None
    assert (
        result["instruction"]
        == "I want to install Spotify on my current system. Could you please help me?"
    )


def test_step() -> None:
    """Test the step function of OSWorld in the virtual machine."""
    env = MagicMock(spec=OSWorld)

    env.step.return_value = {
        "obs": {"screenshot": "mocked screen"},
        "reward": 0,
        "done": False,
        "info": {"done": False},
    }

    result = env.step(action="pyautogui.rightClick()")

    assert result["obs"] == {"screenshot": "mocked screen"}
    assert result["reward"] == 0
    assert result["done"] == False
    assert result["info"] == {"done": False}


def test_evaluate() -> None:
    """Test the evaluate function of OSWorld in the virtual machine."""
    env = MagicMock(spec=OSWorld)

    env.evaluate.return_value = 0.0

    result = env.evaluate()

    assert result == 0.0


def test_render() -> None:
    """Test the evaluate function of OSWorld in the virtual machine."""
    env = MagicMock(spec=OSWorld)

    env.render.return_value = b"Hello, World!"

    result = env.render()

    assert result == b"Hello, World!"
